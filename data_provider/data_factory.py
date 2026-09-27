from data_provider.data_loader import (
    Dataset_ETT_hour,
    Dataset_ETT_minute,
    Dataset_Custom,
    PSMSegLoader,
    MSLSegLoader,
    SMAPSegLoader,
    SMDSegLoader,
    SWATSegLoader,
)

from data_provider.data_loader import ALFASegLoader
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from utils.distributed import active, main_process, DistributedEvalSampler

data_dict = {
    "ETTh1": Dataset_ETT_hour,
    "ETTh2": Dataset_ETT_hour,
    "ETTm1": Dataset_ETT_minute,
    "ETTm2": Dataset_ETT_minute,
    "custom": Dataset_Custom,
    "PSM": PSMSegLoader,
    "MSL": MSLSegLoader,
    "SMAP": SMAPSegLoader,
    "SMD": SMDSegLoader,
    "SWAT": SWATSegLoader,
    "ALFA": ALFASegLoader,
}


def data_provider(args, flag, distributed=None, shuffle=None):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != "timeF" else 1

    shuffle_flag = (flag == "train") if shuffle is None else shuffle
    use_distributed = active() if distributed is None else distributed
    drop_last = False
    batch_size = args.batch_size
    freq = args.freq

    if args.task_name == "anomaly_detection":
        drop_last = False
        data_set = Data(
            args=args,
            root_path=args.root_path,
            win_size=args.seq_len,
            flag=flag,
        )
        if len(data_set) == 0:
            raise ValueError(f"Empty {flag} dataset; check sequence length and data files")
        if main_process():
            print(flag, len(data_set))
        sampler = None
        if use_distributed:
            sampler = (DistributedSampler(data_set, shuffle=shuffle_flag, seed=getattr(args, "random_seed", 2021))
                       if flag == "train" else DistributedEvalSampler(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag if sampler is None else False,
            sampler=sampler,
            num_workers=args.num_workers,
            drop_last=drop_last,
        )
        return data_set, data_loader
    else:
        if args.data == "m4":
            drop_last = False
        data_set = Data(
            args=args,
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns,
        )
        if len(data_set) == 0:
            raise ValueError(f"Empty {flag} dataset; check sequence length and data files")
        if main_process():
            print(flag, len(data_set))
        sampler = None
        if use_distributed:
            sampler = (DistributedSampler(data_set, shuffle=shuffle_flag, seed=getattr(args, "random_seed", 2021))
                       if flag == "train" else DistributedEvalSampler(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag if sampler is None else False,
            sampler=sampler,
            num_workers=args.num_workers,
            drop_last=drop_last,
        )
        return data_set, data_loader
