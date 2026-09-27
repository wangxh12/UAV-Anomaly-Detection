"""Device, checkpoint initialization and DDP ownership for experiments."""

import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from exp.exp_basic import LazyModelDict
from utils.distributed import active, on_main


class Exp_Basic:
    def __init__(self, args):
        self.args = args
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.distributed = self.world_size > 1
        self.device = self._acquire_device()
        if self.distributed and not active():
            backend = "nccl" if self.device.type == "cuda" else "gloo"
            dist.init_process_group(backend=backend, init_method="env://")
        if active():
            self.rank, self.world_size = dist.get_rank(), dist.get_world_size()
            self.distributed = self.world_size > 1
        args.device = self.device
        args.rank, args.local_rank, args.world_size = (
            self.rank,
            self.local_rank,
            self.world_size,
        )
        args.distributed = args.use_multi_gpu = self.distributed
        model_dir = Path(__file__).resolve().parents[1] / "models"
        self.model_dict = LazyModelDict(
            {
                file.stem: f"models.{file.stem}"
                for file in model_dir.glob("*.py")
                if file.stem != "__init__"
            }
        )
        self.model = self._build_model().to(self.device)
        if args.checkpoint:
            self.load_checkpoint(args.checkpoint)
        self._configure_trainable_parameters()
        if self.distributed and args.stage in ("train", "pretrain", "finetune"):
            options = {"find_unused_parameters": args.ddp_find_unused_parameters}
            if self.device.type == "cuda":
                options.update(
                    device_ids=[self.local_rank], output_device=self.local_rank
                )
            self.model = DDP(self.model, **options)
        self.print_main(f"Device: {self.device}; world_size: {self.world_size}")

    def _acquire_device(self):
        if not self.args.use_gpu:
            return torch.device("cpu")
        if self.args.gpu_type == "cuda":
            if torch.cuda.is_available():
                index = self.local_rank if self.distributed else self.args.gpu
                torch.cuda.set_device(index)
                return torch.device("cuda", index)
            if self.distributed:
                raise RuntimeError(
                    "CUDA DDP requested but CUDA is unavailable; use --no_use_gpu for Gloo"
                )
            return torch.device("cpu")
        if self.args.gpu_type == "mps":
            if self.distributed:
                raise ValueError("MPS distributed training is not supported")
            if torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        raise ValueError(f"Unsupported gpu_type: {self.args.gpu_type}")

    @property
    def is_main_process(self):
        return self.rank == 0

    @property
    def raw_model(self):
        return self.model.module if isinstance(self.model, DDP) else self.model

    def print_main(self, *args, **kwargs):
        if self.is_main_process:
            print(*args, **kwargs)

    def _build_model(self):
        raise NotImplementedError

    def _configure_trainable_parameters(self):
        prefixes = self.args.finetune_modules
        if self.args.stage != "finetune" or not prefixes:
            return
        names = list(self.raw_model.named_parameters())
        for prefix in prefixes:
            if not any(
                name == prefix or name.startswith(prefix + ".") for name, _ in names
            ):
                raise ValueError(f"No parameters match finetune module: {prefix}")
        for name, parameter in names:
            parameter.requires_grad_(
                any(name == p or name.startswith(p + ".") for p in prefixes)
            )

    def load_checkpoint(self, path):
        """Accept new checkpoints and legacy plain DP/DDP state dictionaries."""
        payload = torch.load(path, map_location=self.device, weights_only=True)
        state = payload.get("model_state_dict", payload.get("state_dict", payload))
        while state and all(key.startswith("module.") for key in state):
            state = {key[7:]: value for key, value in state.items()}
        self.raw_model.load_state_dict(state, strict=True)
        self.print_main(f"Loaded checkpoint: {path}")
        return payload

    def save_checkpoint(self, path, epoch, val_loss):
        def save():
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            config = {
                key: str(value) if isinstance(value, (torch.device, Path)) else value
                for key, value in vars(self.args).items()
            }
            payload = {
                "model_state_dict": self.raw_model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "stage": self.args.stage,
                "config": config,
            }
            temporary = target.with_suffix(target.suffix + ".tmp")
            torch.save(payload, temporary)
            os.replace(temporary, target)

        on_main(save)
