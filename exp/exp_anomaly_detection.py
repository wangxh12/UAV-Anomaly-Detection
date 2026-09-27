import csv
import os
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from data_provider.data_factory import data_provider
from exp.exp_basic_ddp import Exp_Basic
from utils.distributed import global_average, on_main, sync_buffers
from utils.tools import adjust_learning_rate, adjustment


class Exp_Anomaly_Detection(Exp_Basic):
    def _build_model(self):
        return self.model_dict[self.args.model](self.args).float()

    def _get_data(self, flag, **kwargs):
        return data_provider(self.args, flag, **kwargs)

    def _select_optimizer(self):
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        if not parameters:
            raise ValueError("The model has no trainable parameters")
        return torch.optim.Adam(parameters, lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.MSELoss()

    def _reconstruct(self, batch_x, model=None):
        """Adapt model I/O here without changing the stage or DDP lifecycle."""
        model = self.model if model is None else model
        if self.args.forward_api == "tslib":
            output = model(batch_x, None, None, None)
        elif self.args.forward_api == "x":
            output = model(batch_x)
        else:
            output = model(batch_x, norm=self.args.norm)
        if isinstance(output, (tuple, list)):
            if self.args.reconstruction_index is None:
                raise ValueError("Tuple model output requires --reconstruction_index")
            output = output[self.args.reconstruction_index]
        if isinstance(output, dict):
            output = output["reconstruction"]
        if not isinstance(output, torch.Tensor) or output.ndim != 3:
            raise ValueError(
                "A reconstruction model must return a [batch, time, channels] tensor"
            )
        target = batch_x
        if self.args.features == "MS":
            output, target = output[:, :, -1:], target[:, :, -1:]
        if output.shape != target.shape:
            raise ValueError(
                f"Reconstruction shape {tuple(output.shape)} != target {tuple(target.shape)}"
            )
        return output, target

    @torch.no_grad()
    def vali(self, vali_data, vali_loader, criterion):
        was_training = self.model.training
        self.model.eval()
        sync_buffers(self.raw_model)
        total, count = 0.0, 0
        # No DDP forward collectives: shards may have unequal sizes, even zero.
        for batch_x, _ in vali_loader:
            batch_x = batch_x.float().to(self.device)
            outputs, target = self._reconstruct(batch_x, self.raw_model)
            total += criterion(outputs, target).item() * target.numel()
            count += target.numel()
        loss = global_average(total, count, self.device)
        self.model.train(was_training)
        return loss

    def train(self, setting):
        return self._fit(setting, self.args.train_epochs)

    def pretrain(self, setting):
        # Labels are deliberately not consumed: plain unsupervised reconstruction.
        return self._fit(setting, self.args.pretrain_epochs or self.args.train_epochs)

    def finetuning(self, setting, train=1):
        # The base class loads weights and freezes requested modules BEFORE DDP.
        if not self.args.checkpoint:
            raise ValueError("Fine-tuning requires --checkpoint")
        return self._fit(setting, self.args.finetune_epochs)

    def _fit(self, setting, epochs):
        _, train_loader = self._get_data("train")
        vali_data, vali_loader = self._get_data("val")
        test_pair = self._get_data("test") if self.args.train_test else None
        checkpoint = os.path.join(self.args.checkpoints, setting, "checkpoint.pth")
        if self.args.checkpoint and os.path.realpath(checkpoint) == os.path.realpath(
            self.args.checkpoint
        ):
            raise ValueError("Output checkpoint must differ from the input checkpoint")

        optimizer = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp and self.device.type != "cuda":
            raise ValueError("--use_amp currently requires CUDA")
        scaler = torch.amp.GradScaler("cuda", enabled=self.args.use_amp)
        scheduler = None
        if self.args.lradj == "constant_with_warmup":
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lambda step: min((step + 1) / max(1, self.args.warmup_steps), 1.0),
            )

        history = []
        best_loss, bad_epochs = float("inf"), 0
        self.print_main(
            "Trainable parameters:",
            sum(p.numel() for p in self.model.parameters() if p.requires_grad),
        )
        for epoch in range(epochs):
            if self.distributed:
                train_loader.sampler.set_epoch(epoch)

            self.model.train()
            total, count = 0.0, 0
            started = time.time()
            for index, (batch_x, _) in enumerate(train_loader):
                optimizer.zero_grad(set_to_none=True)
                batch_x = batch_x.float().to(self.device)
                with torch.autocast(
                    device_type=self.device.type, enabled=self.args.use_amp
                ):
                    outputs, target = self._reconstruct(batch_x)
                    loss = criterion(outputs, target)
                if not torch.isfinite(loss):
                    raise ValueError(
                        f"Non-finite training loss at epoch {epoch + 1}, batch {index}"
                    )

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None:
                    scheduler.step()
                total += loss.item() * target.numel()
                count += target.numel()

                if (index + 1) % 100 == 0:
                    self.print_main(f"Epoch {epoch + 1}, batch {index + 1}: loss={loss.item():.7f}")

            train_loss = global_average(total, count, self.device)
            val_loss = self.vali(vali_data, vali_loader, criterion)
            if not np.isfinite(val_loss):
                raise ValueError("Non-finite validation loss")
            history.append(dict(epoch=epoch + 1, train_loss=train_loss, val_loss=val_loss,
                                learning_rate=optimizer.param_groups[0]['lr']))
            def save_history():
                from pathlib import Path
                directory = Path(checkpoint).parent
                directory.mkdir(parents=True, exist_ok=True)
                with (directory / 'history.csv').open('w', newline='') as output:
                    writer = csv.DictWriter(output, fieldnames=list(history[0]))
                    writer.writeheader()
                    writer.writerows(history)
            on_main(save_history)
            message = f"Epoch {epoch + 1}: train={train_loss:.7f}, val={val_loss:.7f}"

            if test_pair is not None:
                message += f", test={self.vali(*test_pair, criterion):.7f}"
            self.print_main(message + f", elapsed={time.time() - started:.1f}s")
            decision = torch.zeros(2, dtype=torch.int64, device=self.device)
            if self.is_main_process:
                improved = val_loss < best_loss
                if improved:
                    best_loss, bad_epochs = val_loss, 0
                else:
                    bad_epochs += 1
                decision[0] = int(improved)
                decision[1] = int(bad_epochs >= self.args.patience)
            if self.distributed:
                dist.broadcast(decision, src=0)
            if decision[0].item():
                self.save_checkpoint(checkpoint, epoch + 1, val_loss)
            if decision[1].item():
                self.print_main("Early stopping")
                break
            if self.args.lradj not in ("constant", "constant_with_warmup"):
                adjust_learning_rate(optimizer, epoch + 1, self.args)
        if getattr(self.args, 'visualize', False):
            from pathlib import Path
            from utils.anomaly_visualization import plot_history
            on_main(lambda: plot_history(history, Path(checkpoint).parent))
        # save_checkpoint's status broadcast completes only after rank zero saves.
        self.load_checkpoint(checkpoint)
        self.print_main(f"Best checkpoint: {checkpoint}")
        return self.model

    def test(self, setting, test=0):
        if test:
            path = self.args.checkpoint or os.path.join(
                self.args.checkpoints, setting, "checkpoint.pth"
            )
            self.load_checkpoint(path)
        sync_buffers(self.raw_model)
        return on_main(lambda: self._test_main(setting))

    def zeroshot(self, setting):
        if not self.args.checkpoint:
            raise ValueError("Zero-shot evaluation requires --checkpoint")
        # No optimizer, backward, or target-domain weight updates.
        return self.test(setting)

    @torch.no_grad()
    def _test_main(self, setting):
        test_data, test_loader = self._get_data(
            flag="test", distributed=False, shuffle=False
        )
        train_data, train_loader = self._get_data(
            flag="train", distributed=False, shuffle=False
        )
        attens_energy = []
        folder_path = os.path.join(self.args.results, setting)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        self.anomaly_criterion = nn.MSELoss(reduction="none")

        # (1) stastic on the train set
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(train_loader):
                batch_x = batch_x.float().to(self.device)
                # reconstruction
                outputs, target = self._reconstruct(batch_x, self.raw_model)
                # criterion
                score = torch.mean(self.anomaly_criterion(target, outputs), dim=-1)
                score = score.detach().cpu().numpy()
                attens_energy.append(score)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        visualizer, visual_channels = None, None
        if getattr(self.args, 'visualize', False):
            from utils.anomaly_visualization import prepare_visualization
            visualizer, visual_channels = prepare_visualization(self.args, test_data)

        # (2) find the threshold
        attens_energy = []
        test_labels = []
        for i, (batch_x, batch_y) in enumerate(test_loader):
            batch_x = batch_x.float().to(self.device)
            # reconstruction
            outputs, target = self._reconstruct(batch_x, self.raw_model)
            # criterion
            score = torch.mean(self.anomaly_criterion(target, outputs), dim=-1)
            score = score.detach().cpu().numpy()
            attens_energy.append(score)
            test_labels.append(batch_y)
            if visualizer is not None:
                visualizer.update(score, outputs.detach().cpu().numpy())

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        threshold = np.percentile(combined_energy, 100 - self.args.anomaly_ratio)
        print("Threshold :", threshold)

        if visualizer is not None:
            from utils.anomaly_visualization import export_visualization
            export_visualization(self.args, test_data, visualizer, visual_channels, threshold, folder_path)

        # (3) evaluation on the test set
        pred = (test_energy > threshold).astype(int)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_labels = np.array(test_labels)
        gt = test_labels.astype(int)

        print("pred:   ", pred.shape)
        print("gt:     ", gt.shape)

        # (4) detection adjustment
        gt, pred = adjustment(gt, pred)

        pred = np.array(pred)
        gt = np.array(gt)
        print("pred: ", pred.shape)
        print("gt:   ", gt.shape)

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(
            gt, pred, average="binary"
        )
        print(
            "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
                accuracy, precision, recall, f_score
            )
        )

        f = open(os.path.join(folder_path, "metrics.txt"), "a", encoding="utf-8")
        f.write(setting + "  \n")
        f.write(
            "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
                accuracy, precision, recall, f_score
            )
        )
        f.write("\n")
        f.write("\n")
        f.close()
        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f_score": float(f_score),
            "threshold": float(threshold),
        }
