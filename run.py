import argparse
import os
import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist


def build_parser():
    parser = argparse.ArgumentParser(description='UAV reconstruction anomaly detection')

    # basic config
    parser.add_argument('--task_name', type=str, default='anomaly_detection',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, choices=[0, 1], default=None, help='legacy pretraining flag')
    parser.add_argument('--model_id', type=str, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')
    parser.add_argument('--is_finetuning', type=int, choices=[0, 1], default=None, help='status')
    parser.add_argument('--is_zeroshot', type=int, choices=[0, 1], default=None, help='status')
    parser.add_argument('--train_test', type=int, default=1, help='train_test')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--stride', type=int, default=1, help='stride')
    parser.add_argument('--step', type=int, default=1, help='step')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%%)')

    # model define
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    parser.add_argument('--tv_dt', type=int, default=0, help='whether to use time variant dt for MambaSL')
    parser.add_argument('--tv_B', type=int, default=0, help='whether to use time variant B for MambaSL')
    parser.add_argument('--tv_C', type=int, default=0, help='whether to use time variant C for MambaSL')
    parser.add_argument('--use_D', type=int, default=0, help='whether to use D for MambaSL')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default=None,
                        help='down sampling method, only support avg, max, conv')
    parser.add_argument('--seg_len', type=int, default=96,
                        help='the length of segmen-wise iteration of SegRNN')
    parser.add_argument('--norm', type=int, default=0, help='True 1 False 0')
    parser.add_argument('--hidden_dim', type=int, default=64, help='embedding dimenison')
    parser.add_argument('--depth', type=int, default=10, help='number of layers')

    # evaluation
    parser.add_argument('--metric', type=str, nargs="+", default="affiliation", help="metric")
    parser.add_argument('--q', type=float, nargs="+", default=[0.03], help="for SPOT")
    parser.add_argument('--t', type=float, nargs="+", default=[0.06], help="threshold found by SPOT")

    # optimization
    parser.add_argument("--percentage", type=float, default=1, help="the percentage(*100) of train data")
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', choices=['MSE'], default='MSE', help='loss function')
    parser.add_argument('--lradj', choices=['type1', 'type2', 'type3', 'cosine', 'constant', 'constant_with_warmup'], default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--finetune_epochs', type=int, default=10, help='finetuning epochs')

    # GPU
    parser.add_argument('--use_gpu', action='store_true', default=True, help='use gpu (default: on)')
    parser.add_argument('--no_use_gpu', action='store_false', dest='use_gpu', help='disable gpu (force cpu)')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type')  # cuda or mps
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default=None, help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # metrics (dtw)
    parser.add_argument('--use_dtw', action='store_true', default=False,
                        help='enable dtw metric (time consuming; default: off)')

    # Augmentation
    parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
    parser.add_argument('--seed', type=int, default=2, help="Randomization seed")
    parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")
    parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
    parser.add_argument('--permutation', default=False, action="store_true",
                        help="Equal Length Permutation preset augmentation")
    parser.add_argument('--randompermutation', default=False, action="store_true",
                        help="Random Length Permutation preset augmentation")
    parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
    parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
    parser.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")
    parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
    parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
    parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
    parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
    parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")
    parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
    parser.add_argument('--discdtw', default=False, action="store_true",
                        help="Discrimitive DTW warp preset augmentation")
    parser.add_argument('--discsdtw', default=False, action="store_true",
                        help="Discrimitive shapeDTW warp preset augmentation")
    parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")

    # TimeXer
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')

    # GCN
    parser.add_argument('--node_dim', type=int, default=10, help='each node embbed to dim dimentions')
    parser.add_argument('--gcn_depth', type=int, default=2, help='')
    parser.add_argument('--gcn_dropout', type=float, default=0.3, help='')
    parser.add_argument('--propalpha', type=float, default=0.3, help='')
    parser.add_argument('--conv_channel', type=int, default=32, help='')
    parser.add_argument('--skip_channel', type=int, default=32, help='')

    parser.add_argument('--individual', action='store_true', default=False,
                        help='DLinear: a linear layer for each variate(channel) individually')

    # TimeFilter
    parser.add_argument('--alpha', type=float, default=0.1, help='KNN for Graph Construction')
    parser.add_argument('--top_p', type=float, default=0.5, help='Dynamic Routing in MoE')
    parser.add_argument('--pos', type=int, choices=[0, 1], default=1, help='Positional Embedding. Set pos to 0 or 1')

    # Stage is independent of the model's task_name (anomaly_detection).
    parser.add_argument('--stage', choices=['train', 'pretrain', 'finetune', 'zeroshot', 'test'])
    parser.add_argument('--checkpoint', help='input weights for fine-tuning/evaluation; never inferred from a dataset name')
    parser.add_argument('--setting', help='optional output run name (single directory component)')
    parser.add_argument('--results', default='./test_results', help='evaluation output root')
    parser.add_argument('--eval_after_train', action='store_true', help='evaluate best weights after train/pretrain/finetune')
    parser.add_argument('--pretrain_epochs', type=int, default=None, help='defaults to train_epochs')
    parser.add_argument('--finetune_modules', nargs='+', help='parameter/module prefixes to update; default: all parameters')
    parser.add_argument('--forward_api', choices=['tslib', 'x', 'norm'], default='tslib')
    parser.add_argument('--reconstruction_index', type=int, help='explicit reconstruction index for tuple/list outputs')
    parser.add_argument('--ddp_find_unused_parameters', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument('--random_seed', type=int, default=2021)
    parser.add_argument('--visualize', action='store_true', help='save original-timeline diagnostic plots after evaluation')
    parser.add_argument('--vis_channels', type=int, nargs='+', help='zero-based channel indices; defaults to first three')
    parser.add_argument('--vis_start', type=int, default=0)
    parser.add_argument('--vis_end', type=int, help='exclusive original sample index')
    parser.add_argument('--vis_context', type=int, default=100, help='samples around each anomaly event')
    parser.add_argument('--vis_max_events', type=int, default=3)

    return parser


def resolve_stage(args, parser):
    legacy = []
    if args.is_training == 1:
        legacy.append('pretrain')
    if args.is_finetuning == 1:
        legacy.append('finetune')
    if args.is_zeroshot == 1:
        legacy.append('zeroshot')
    if len(legacy) > 1:
        parser.error('Select one stage per invocation; connect stages with --checkpoint')
    if args.stage and legacy and args.stage != legacy[0]:
        if not (args.stage == 'train' and legacy[0] == 'pretrain'):
            parser.error('--stage conflicts with legacy stage flags')
    args.stage = args.stage or (legacy[0] if legacy else ('test' if args.is_training == 0 else 'train'))
    if args.stage in ('finetune', 'zeroshot', 'test') and not args.checkpoint:
        parser.error(f'{args.stage} requires --checkpoint')
    if args.checkpoint and not Path(args.checkpoint).is_file():
        parser.error(f'Checkpoint not found: {args.checkpoint}')
    if args.percentage != 1:
        parser.error('Current loaders do not implement percentage sampling; provide data prepared under your protocol')
    if args.finetune_modules and args.stage != 'finetune':
        parser.error('--finetune_modules applies only to finetune')
    if args.task_name != 'anomaly_detection':
        parser.error('Use --task_name anomaly_detection; choose the lifecycle with --stage')
    if args.setting and (args.setting in ('.', '..') or '/' in args.setting or chr(92) in args.setting):
        parser.error('--setting must be a single directory name')
    for name in ('train_epochs', 'finetune_epochs', 'batch_size', 'itr', 'patience'):
        if getattr(args, name) <= 0:
            parser.error(f'--{name} must be positive')
    if args.pretrain_epochs is not None and args.pretrain_epochs <= 0:
        parser.error('--pretrain_epochs must be positive')
    if args.num_workers < 0 or args.warmup_steps < 0:
        parser.error('num_workers and warmup_steps must be nonnegative')
    if not 0 <= args.anomaly_ratio <= 100:
        parser.error('--anomaly_ratio must be in [0, 100]')
    if args.use_multi_gpu and int(os.environ.get('WORLD_SIZE', '1')) <= 1 and 'RANK' not in os.environ:
        parser.error('--use_multi_gpu requires torchrun; it does not spawn processes')
    if args.vis_start < 0 or args.vis_context < 0 or args.vis_max_events < 0:
        parser.error('Visualization start/context/event count must be nonnegative')
    if args.vis_end is not None and args.vis_end <= args.vis_start:
        parser.error('--vis_end must be greater than --vis_start')
    return args.stage


def make_setting(args, repeat):
    if args.setting:
        return args.setting if args.itr == 1 else f'{args.setting}_{repeat}'
    name = (f'{args.stage}_{args.model_id}_{args.model}_{args.data}_sl{args.seq_len}'
            f'_dm{args.d_model}_hd{args.hidden_dim}_dp{args.depth}_{repeat}')
    return re.sub(r'[^a-zA-Z0-9_.-]', '_', name)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    resolve_stage(args, parser)
    # Visibility must be set before CUDA is initialized. GPU ids inside Python
    # are logical ids in this visible set; torchrun selects LOCAL_RANK.
    if args.devices:
        devices = args.devices.replace(' ', '')
        visible = os.environ.get('CUDA_VISIBLE_DEVICES')
        if visible is not None and visible != devices:
            parser.error('--devices conflicts with CUDA_VISIBLE_DEVICES')
        os.environ['CUDA_VISIBLE_DEVICES'] = devices
    from exp.exp_anomaly_detection import Exp_Anomaly_Detection
    try:
        for repeat in range(args.itr):
            seed_everything(args.random_seed + repeat)
            setting = make_setting(args, repeat)
            args.setting_name = setting
            exp = Exp_Anomaly_Detection(args)
            seed_everything(args.random_seed + repeat + exp.rank)
            exp.print_main(f'>>>>>>> {args.stage}: {setting}')
            if args.stage == 'pretrain':
                exp.pretrain(setting)
            elif args.stage == 'finetune':
                exp.finetuning(setting)
            elif args.stage == 'train':
                exp.train(setting)
            elif args.stage == 'zeroshot':
                exp.zeroshot(setting)
            else:
                exp.test(setting)
            if args.eval_after_train and args.stage in ('train', 'pretrain', 'finetune'):
                exp.test(setting)
            del exp
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
