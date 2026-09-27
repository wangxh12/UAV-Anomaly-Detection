"""Small distributed helpers shared by all reconstruction stages."""
import torch
import torch.distributed as dist
from torch.utils.data import Sampler


def active():
    return dist.is_available() and dist.is_initialized()


def main_process():
    return not active() or dist.get_rank() == 0


def on_main(function):
    """All ranks call this; propagate rank-zero I/O/evaluation failures."""
    result, status = None, [None]
    if main_process():
        try:
            result = function()
        except Exception as error:
            status[0] = f"{type(error).__name__}: {error}"
    if active():
        dist.broadcast_object_list(status, src=0)
    if status[0] is not None:
        raise RuntimeError(status[0])
    return result


class DistributedEvalSampler(Sampler):
    """Partition evaluation indices without padding or dropping any samples."""
    def __init__(self, dataset, rank=None, world_size=None):
        self.dataset = dataset
        self.rank = dist.get_rank() if rank is None else rank
        self.world_size = dist.get_world_size() if world_size is None else world_size

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self):
        return len(range(self.rank, len(self.dataset), self.world_size))


def sync_buffers(model):
    # Evaluation uses the unwrapped model to support unequal shard lengths.
    if active():
        for buffer in model.buffers():
            dist.broadcast(buffer, src=0)


def global_average(total, count, device):
    values = torch.tensor([total, count], dtype=torch.float32 if device.type == "mps" else torch.float64, device=device)
    if active():
        dist.all_reduce(values)
    if values[1].item() == 0:
        raise ValueError("Cannot calculate loss on an empty dataset")
    return (values[0] / values[1]).item()
