import os
import importlib

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args

        # -------------------------------------------------------
        #  Distributed environment
        # -------------------------------------------------------
        self._init_distributed()

        # -------------------------------------------------------
        #  Automatically generate model map
        # -------------------------------------------------------
        model_map = self._scan_models_directory()

        # Use smart dictionary
        self.model_dict = LazyModelDict(model_map)

        # -------------------------------------------------------
        #  Device
        # -------------------------------------------------------
        self.device = self._acquire_device()

        # -------------------------------------------------------
        #  Build model
        # -------------------------------------------------------
        self.model = self._build_model().to(self.device)

        # -------------------------------------------------------
        #  DistributedDataParallel
        # -------------------------------------------------------
        if self.distributed:
            self.model = DDP(
                self.model, device_ids=[self.local_rank], output_device=self.local_rank
            )

    def _init_distributed(self):
        """
        Initialize distributed training environment.

        torchrun automatically provides:
            LOCAL_RANK
            RANK
            WORLD_SIZE
        """

        self.distributed = (
            "RANK" in os.environ
            and "WORLD_SIZE" in os.environ
            and int(os.environ["WORLD_SIZE"]) > 1
        )

        if self.distributed:
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.rank = int(os.environ["RANK"])
            self.world_size = int(os.environ["WORLD_SIZE"])

            torch.cuda.set_device(self.local_rank)

            if not dist.is_initialized():
                dist.init_process_group(backend="nccl", init_method="env://")

        else:
            self.local_rank = 0
            self.rank = 0
            self.world_size = 1

    @property
    def is_main_process(self):
        """
        Only global rank 0 is responsible for logging,
        checkpoint saving, etc.
        """
        return self.rank == 0

    def print_main(self, *args, **kwargs):
        """
        Print only on global rank 0.
        """
        if self.is_main_process:
            print(*args, **kwargs)

    def _scan_models_directory(self):
        """
        Automatically scan all .py files in the models folder.
        """
        model_map = {}
        models_dir = "models"

        if os.path.exists(models_dir):
            for filename in os.listdir(models_dir):
                if filename.endswith(".py") and filename != "__init__.py":
                    module_name = filename[:-3]
                    full_path = f"{models_dir}.{module_name}"
                    model_map[module_name] = full_path

        return model_map

    def _build_model(self):
        raise NotImplementedError

    def _acquire_device(self):
        """
        Acquire computing device.

        In DDP:
            each process uses exactly one GPU specified by LOCAL_RANK.

        In non-DDP:
            use the original single-GPU / MPS / CPU behavior.
        """

        # -------------------------------------------------------
        # DDP
        # -------------------------------------------------------
        if self.distributed:
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "DDP with NCCL requires CUDA, but CUDA is unavailable."
                )
            device = torch.device(f"cuda:{self.local_rank}")
            self.print_main(f"Use Distributed GPU: " f"world_size={self.world_size}")
            return device

        # -------------------------------------------------------
        # Single GPU
        # -------------------------------------------------------
        if self.args.use_gpu and self.args.gpu_type == "cuda":
            device = torch.device(f"cuda:{self.args.gpu}")
            print(f"Use GPU: cuda:{self.args.gpu}")

        # -------------------------------------------------------
        # MPS
        # -------------------------------------------------------
        elif self.args.use_gpu and self.args.gpu_type == "mps":
            device = torch.device("mps")
            print("Use GPU: mps")

        # -------------------------------------------------------
        # CPU
        # -------------------------------------------------------
        else:
            device = torch.device("cpu")
            print("Use CPU")
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass


class LazyModelDict(dict):
    """
    Smart Lazy-Loading Dictionary
    """

    def __init__(self, model_map):
        self.model_map = model_map
        super().__init__()

    def __getitem__(self, key):
        if key in self:
            return super().__getitem__(key)

        if key not in self.model_map:
            raise NotImplementedError(f"Model [{key}] not found in 'models' directory.")

        module_path = self.model_map[key]
        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            raise ImportError(
                f"Failed to import model [{key}] "
                f"from [{module_path}]. "
                f"Dependencies missing?"
            ) from e

        # Try to find the model class
        if hasattr(module, "Model"):
            model_class = module.Model
        elif hasattr(module, key):
            model_class = getattr(module, key)
        else:
            raise AttributeError(
                f"Module {module_path} has no class 'Model' or '{key}'"
            )

        self[key] = model_class
        return model_class
