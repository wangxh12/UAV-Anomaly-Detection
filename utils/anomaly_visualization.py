"""Diagnostic plots on the original timeline; independent of metric protocol."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def intervals(labels):
    edges = np.diff(np.r_[False, np.asarray(labels, dtype=bool), False].astype(int))
    return list(zip(map(int, np.flatnonzero(edges == 1)), map(int, np.flatnonzero(edges == -1))))


class TimelineAccumulator:
    """Overlap-average scores/reconstructions; uncovered points remain NaN."""
    def __init__(self, length, step, channels):
        if length <= 0 or step <= 0:
            raise ValueError("Timeline length and step must be positive")
        self.length, self.step, self.channels = length, step, list(channels)
        self.score_sum = np.zeros(length, dtype=np.float64)
        self.recon_sum = np.zeros((length, len(channels)), dtype=np.float64)
        self.count = np.zeros(length, dtype=np.int64)
        self.windows = 0

    def update(self, scores, reconstruction):
        scores, reconstruction = np.asarray(scores), np.asarray(reconstruction)
        if scores.ndim != 2 or reconstruction.shape[:2] != scores.shape:
            raise ValueError("Expected scores [B,T] and reconstruction [B,T,C]")
        starts = (self.windows + np.arange(len(scores))) * self.step
        if len(starts) and starts[-1] + scores.shape[1] > self.length:
            raise ValueError("Window positions exceed the original timeline")
        for offset in range(scores.shape[1]):
            index = starts + offset
            self.score_sum[index] += scores[:, offset]
            self.recon_sum[index] += reconstruction[:, offset, self.channels]
            self.count[index] += 1
        self.windows += len(scores)

    def result(self):
        scores = np.full(self.length, np.nan)
        reconstruction = np.full_like(self.recon_sum, np.nan)
        covered = self.count > 0
        scores[covered] = self.score_sum[covered] / self.count[covered]
        reconstruction[covered] = self.recon_sum[covered] / self.count[covered, None]
        return scores, reconstruction, self.count.copy()


def prepare_visualization(args, dataset):
    if not hasattr(dataset, "test") or not hasattr(dataset, "step"):
        raise ValueError("Visualization requires a sequential test dataset with test and step attributes")
    series = np.asarray(dataset.test)
    if series.ndim != 2:
        raise ValueError("Visualization expects test data [time, channels]")
    available = [series.shape[1] - 1] if args.features == "MS" else list(range(series.shape[1]))
    channels = args.vis_channels if args.vis_channels is not None else available[:3]
    if not channels or len(set(channels)) != len(channels) or any(c not in available for c in channels):
        raise ValueError(f"vis_channels must be unique channel indices from {available}")
    # MS reconstruction contains only the final input channel.
    output_channels = [0] if args.features == "MS" else channels
    accumulator = TimelineAccumulator(len(series), dataset.step, output_channels)
    return accumulator, channels


def _save(fig, stem):
    fig.canvas.draw()
    # Persist plot rectangles for shared-axis layout review.
    bounds = [list(ax.get_position().bounds) for ax in fig.axes]
    if len(bounds) > 1 and not all(abs(b[0] - bounds[0][0]) < 1e-4 and abs(b[2] - bounds[0][2]) < 1e-4 for b in bounds):
        raise ValueError("Shared timeline plot areas are not aligned")
    Path(str(stem) + '.layout.json').write_text(json.dumps(bounds), encoding='utf-8')
    fig.savefig(str(stem) + ".png", dpi=160)
    fig.savefig(str(stem) + ".pdf")
    plt.close(fig)


def export_visualization(args, dataset, accumulator, channels, threshold, folder):
    folder = Path(folder) / "visualization"
    folder.mkdir(parents=True, exist_ok=True)
    score, recon, count = accumulator.result()
    series = np.asarray(dataset.test)[:, channels]
    labels = np.asarray(getattr(dataset, "test_labels", None))
    if labels.size != len(score):
        raise ValueError("Dataset test_labels must have one label per original time point")
    labels = labels.reshape(-1).astype(int)
    names = [f"Channel {c}" for c in channels]
    flights = []
    manifest = Path(args.root_path) / "benchmark_manifest.json"
    if manifest.is_file():
        metadata = json.loads(manifest.read_text())
        feature_names = metadata.get("feature_columns", [])
        if len(feature_names) == dataset.test.shape[1]:
            names = [feature_names[c] for c in channels]
        flights = metadata.get("test", {}).get("flights", [])
        if any(not 0 <= f["start"] < f["end"] <= len(score) for f in flights):
            raise ValueError("Flight boundaries do not match the test timeline")
    start = args.vis_start
    end = len(score) if args.vis_end is None else args.vis_end
    if not 0 <= start < end <= len(score):
        raise ValueError(f"Visualization range must satisfy 0 <= start < end <= {len(score)}")
    predicted = np.full(len(score), -1, dtype=np.int8)
    predicted[count > 0] = (score[count > 0] > threshold).astype(np.int8)
    np.savez_compressed(folder / "timeline.npz", index=np.arange(len(score)),
                        series=series, reconstruction=recon, score=score, label=labels,
                        coverage=count, prediction=predicted, threshold=threshold, channels=channels)
    # Split GT intervals at flight boundaries as well as normal/abnormal transitions.
    segments = flights or [{"start": 0, "end": len(score), "file": "test sequence"}]
    events = []
    for flight in segments:
        a, b = flight["start"], flight["end"]
        for left, right in intervals(labels[a:b]):
            events.append((a + left, a + right, flight))
    def draw(left, right, title, filename):
        with plt.rc_context({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                             "pdf.fonttype": 42, "axes.grid": False}):
            fig, axes = plt.subplots(len(channels) + 1, 1, figsize=(12, 2.0 * (len(channels) + 1)),
                                     sharex=True, layout="constrained", squeeze=False)
            axes = axes[:, 0]
            x = np.arange(left, right)
            for j, name in enumerate(names):
                axes[j].plot(x, series[left:right, j], color="#35618F", lw=0.7, label="Input")
                axes[j].plot(x, recon[left:right, j], color="#C87920", lw=0.7, alpha=0.85, label="Mean reconstruction")
                axes[j].set_ylabel(name + "\n(standardized)")
            axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2, frameon=False)
            axes[-1].plot(x, score[left:right], color="#257F70", lw=0.8, label="Mean anomaly score")
            axes[-1].axhline(threshold, color="#8B4056", ls="--", lw=1, label="Evaluation threshold")
            mask = predicted[left:right] == 1
            axes[-1].scatter(x[mask], score[left:right][mask], s=5, color="#C87920", label="Score > threshold", zorder=3)
            axes[-1].set_ylabel("Reconstruction MSE")
            axes[-1].set_xlabel("Original test sample index")
            axes[-1].legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=3, frameon=False)
            for ax in axes:
                for a, b in intervals(labels[left:right]):
                    ax.axvspan(left + a, left + b, color="#B34755", alpha=0.12, lw=0)
                for flight in flights[1:]:
                    if left < flight['start'] < right:
                        ax.axvline(flight['start'], color='0.5', lw=0.5, ls=':')
                ax.set_xlim(left, max(left + 1, right - 1))
            fig.suptitle(title + "\nRed shading: ground-truth anomaly | Overlap-averaged display; no point adjustment", fontsize=10)
            _save(fig, folder / filename)
    draw(start, end, "Test overview", "overview")
    chosen = [event for event in events if event[0] < end and event[1] > start][:args.vis_max_events]
    for number, (a, b, flight) in enumerate(chosen, 1):
        left = max(start, flight['start'], a - args.vis_context)
        right = min(end, flight['end'], b + args.vis_context)
        draw(left, right, f"Event {number}: {flight['file']}", f"event_{number:02d}_{a}_{b}")
    summary = dict(channels=channels, channel_names=names, threshold=float(threshold),
                   original_points=len(score), covered_points=int((count > 0).sum()),
                   windows=accumulator.windows, step=accumulator.step, plotted_range=[start, end],
                   events=[dict(start=a, end=b, flight=f['file']) for a,b,f in chosen],
                   aggregation="Mean of all window scores/reconstructions covering each time point. Uncovered values are NaN.",
                   protocol="Plots use original timeline; existing metrics still use flattened windows and point adjustment. Threshold is reused, not refitted. Dotted lines indicate flight boundaries; cross-flight training/test windows remain unchanged.")
    (folder / "metadata.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Visualization saved: {folder}")


def plot_history(rows, folder):
    folder = Path(folder)
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    ax.plot([r['epoch'] for r in rows], [r['train_loss'] for r in rows], label="Train")
    ax.plot([r['epoch'] for r in rows], [r['val_loss'] for r in rows], label="Validation")
    ax.set(xlabel="Epoch", ylabel="Reconstruction MSE", title="Training history")
    ax.legend(frameon=False)
    _save(fig, folder / "loss_curve")
