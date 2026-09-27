"""End-to-end stage and distributed regression tests; no real flight data."""
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run import build_parser, resolve_stage
from utils.distributed import DistributedEvalSampler


@pytest.fixture
def workspace(tmp_path):
    rng = np.random.default_rng(7)
    data = tmp_path / "data"
    data.mkdir()
    np.save(data / "ALFA_train.npy", rng.normal(size=(26, 2)).astype("float32"))
    np.save(data / "ALFA_test.npy", rng.normal(size=(20, 2)).astype("float32"))
    labels = np.zeros(20, dtype="float32")
    labels[7:10] = 1
    np.save(data / "ALFA_test_label.npy", labels)
    return tmp_path


def invoke(workspace, stage, setting, *extra, distributed=False, gpu=False):
    command = [sys.executable]
    if distributed:
        command += ["-m", "torch.distributed.run", "--standalone", "--nproc-per-node=2"]
    command += [str(ROOT / "run.py"), "--stage", stage, "--setting", setting,
                "--model", "LSTM_AE", "--data", "ALFA", "--root_path", str(workspace / "data"),
                "--checkpoints", str(workspace / "checkpoints"), "--results", str(workspace / "results"),
                "--seq_len", "4", "--enc_in", "2", "--hidden_dim", "3", "--depth", "1",
                "--dropout", "0", "--batch_size", "4", "--train_epochs", "3", "--finetune_epochs", "1",
                "--num_workers", "0", "--train_test", "0", "--patience", "1",
                "--learning_rate", "0", "--lradj", "constant"]
    if not gpu:
        command += ["--no_use_gpu"]
    command += list(extra)
    environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    completed = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True, timeout=150)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return completed.stdout


def load(workspace, setting):
    path = workspace / "checkpoints" / setting / "checkpoint.pth"
    return path, torch.load(path, map_location="cpu", weights_only=True)


def test_stage_validation_and_eval_sampler():
    parser = build_parser()
    args = parser.parse_args(["--model", "LSTM_AE", "--data", "ALFA", "--is_training", "1"])
    assert resolve_stage(args, parser) == "pretrain"
    for flags in (["--stage", "zeroshot"], ["--is_training", "1", "--is_finetuning", "1"]):
        args = parser.parse_args(["--model", "LSTM_AE", "--data", "ALFA", *flags])
        with pytest.raises(SystemExit):
            resolve_stage(args, parser)
    for length in (1, 3, 7):
        shards = [list(DistributedEvalSampler(range(length), rank, 2)) for rank in range(2)]
        assert sorted(shards[0] + shards[1]) == list(range(length))
        assert not set(shards[0]).intersection(shards[1])


def test_pretrain_finetune_zeroshot_and_legacy_checkpoint(workspace):
    output = invoke(workspace, "pretrain", "source")
    assert "Early stopping" in output and "Epoch 3:" not in output
    source, before = load(workspace, "source")
    source_bytes = source.read_bytes()
    assert before["stage"] == "pretrain"
    assert all(not key.startswith("module.") for key in before["model_state_dict"])
    # Old DataParallel checkpoints remain usable.
    legacy = workspace / "legacy.pth"
    torch.save({"module." + key: value for key, value in before["model_state_dict"].items()}, legacy)
    invoke(workspace, "zeroshot", "zero", "--checkpoint", str(legacy))
    assert not (workspace / "checkpoints" / "zero").exists()
    assert (workspace / "results" / "zero" / "metrics.txt").is_file()
    invoke(workspace, "finetune", "target", "--checkpoint", str(source),
           "--finetune_modules", "projection", "--learning_rate", "0.01", "--eval_after_train")
    _, after = load(workspace, "target")
    assert source.read_bytes() == source_bytes
    for key, value in before["model_state_dict"].items():
        if not key.startswith("projection."):
            assert torch.equal(value, after["model_state_dict"][key]), key
    assert any(not torch.equal(value, after["model_state_dict"][key])
               for key, value in before["model_state_dict"].items() if key.startswith("projection."))
    assert (workspace / "results" / "target" / "metrics.txt").is_file()


def test_two_process_training_and_exact_validation(workspace):
    output = invoke(workspace, "pretrain", "ddp", "--eval_after_train", distributed=True)
    assert output.count("Early stopping") == 1
    assert "Epoch 3:" not in output
    checkpoint, payload = load(workspace, "ddp")
    assert payload["config"]["world_size"] == 2
    metrics = (workspace / "results" / "ddp" / "metrics.txt").read_text()
    assert metrics.count("Accuracy") == 1
    assert not list((workspace / "checkpoints").rglob("*.tmp"))
    # Independently calculate full-dataset validation loss (3 windows: unequal shards).
    from types import SimpleNamespace
    from models.LSTM_AE import Model
    from sklearn.preprocessing import StandardScaler
    train = np.load(workspace / "data" / "ALFA_train.npy")
    normalized = StandardScaler().fit_transform(train).astype("float32")
    val = normalized[int(len(normalized) * 0.8):]
    inputs = torch.from_numpy(np.stack([val[i:i + 4] for i in range(len(val) - 3)]))
    model = Model(SimpleNamespace(**payload["config"]))
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    with torch.no_grad():
        expected = torch.nn.functional.mse_loss(model(inputs), inputs).item()
    assert payload["val_loss"] == pytest.approx(expected, rel=1e-6)
    # DDP-produced weights must load for a standalone single-process evaluation.
    invoke(workspace, "test", "ddp_loaded", "--checkpoint", str(checkpoint))
    assert (workspace / "results" / "ddp_loaded" / "metrics.txt").is_file()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_reconstruction_amp(workspace):
    invoke(workspace, "pretrain", "cuda", "--train_epochs", "1", "--use_amp", gpu=True)
    _, payload = load(workspace, "cuda")
    assert payload["config"]["device"].startswith("cuda")
    assert np.isfinite(payload["val_loss"])


def test_ddp_gradient_matches_global_batch_and_empty_validation_rank(workspace):
    # 16 training points -> 13 windows, padded to 14 across ranks.
    # Validation contains exactly one window, leaving rank 1's shard empty.
    path = workspace / "data" / "ALFA_train.npy"
    np.save(path, np.load(path)[:16])
    invoke(workspace, "pretrain", "gradient", "--train_epochs", "1",
           "--batch_size", "32", "--learning_rate", "0.01", distributed=True)
    _, payload = load(workspace, "gradient")
    from types import SimpleNamespace
    from models.LSTM_AE import Model
    from sklearn.preprocessing import StandardScaler
    from torch.utils.data.distributed import DistributedSampler
    normalized = StandardScaler().fit_transform(np.load(path)).astype("float32")
    windows = torch.from_numpy(np.stack([normalized[i:i + 4] for i in range(len(normalized) - 3)]))
    indices = []
    for rank in range(2):
        indices += list(DistributedSampler(windows, num_replicas=2, rank=rank, seed=2021))
    torch.manual_seed(2021)
    reference = Model(SimpleNamespace(**payload["config"]))
    optimizer = torch.optim.Adam(reference.parameters(), lr=0.01)
    batch = windows[indices]
    torch.nn.functional.mse_loss(reference(batch), batch).backward()
    optimizer.step()
    for key, value in reference.state_dict().items():
        torch.testing.assert_close(value, payload["model_state_dict"][key], rtol=1e-5, atol=1e-6)
    reference.eval()
    with torch.no_grad():
        val = torch.from_numpy(normalized[-4:]).unsqueeze(0)
        expected = torch.nn.functional.mse_loss(reference(val), val).item()
    assert payload["val_loss"] == pytest.approx(expected, rel=1e-6)


def test_visualization_keeps_metric_protocol(workspace):
    invoke(workspace, "pretrain", "visual", "--train_epochs", "1", "--eval_after_train",
           "--visualize", "--vis_channels", "0", "--vis_max_events", "1")
    directory = workspace / "results" / "visual" / "visualization"
    data = np.load(directory / "timeline.npz")
    assert data['score'].shape == (20,)
    assert data['series'].shape == (20, 1)
    assert data['coverage'][0] == 1 and data['coverage'][5] == 4
    np.testing.assert_array_equal(data['label'], np.load(workspace / 'data' / 'ALFA_test_label.npy'))
    assert (directory / 'overview.pdf').is_file()
    assert (directory / 'overview.png').is_file()
    assert len(list(directory.glob('event_*.png'))) == 1
    assert (workspace / 'checkpoints' / 'visual' / 'history.csv').is_file()
    assert (workspace / 'checkpoints' / 'visual' / 'loss_curve.png').is_file()
    checkpoint, _ = load(workspace, 'visual')
    invoke(workspace, 'test', 'no_visual', '--checkpoint', str(checkpoint))
    visual_metrics = (workspace / 'results' / 'visual' / 'metrics.txt').read_text().splitlines()[1]
    plain_metrics = (workspace / 'results' / 'no_visual' / 'metrics.txt').read_text().splitlines()[1]
    assert visual_metrics == plain_metrics


def test_timeline_overlap_aggregation_and_gaps():
    from utils.anomaly_visualization import TimelineAccumulator, intervals
    scores = np.arange(9).reshape(3, 3)
    outputs = np.stack([scores * 2, scores * 3], axis=-1)
    accumulator = TimelineAccumulator(5, 1, [1])
    accumulator.update(scores[:2], outputs[:2])
    accumulator.update(scores[2:], outputs[2:])
    mean, reconstruction, count = accumulator.result()
    np.testing.assert_allclose(mean, [0, 2, 4, 6, 8])
    np.testing.assert_allclose(reconstruction[:, 0], mean * 3)
    np.testing.assert_array_equal(count, [1, 2, 3, 2, 1])
    gaps = TimelineAccumulator(7, 4, [0])
    gaps.update(np.ones((2, 2)), np.ones((2, 2, 1)))
    mean, _, count = gaps.result()
    assert np.isnan(mean[[2, 3, 6]]).all()
    assert intervals([1, 1, 0, 1]) == [(0, 2), (3, 4)]
    assert intervals([0, 0]) == []
