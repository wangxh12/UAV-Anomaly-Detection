#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=0

python -u run.py \
    --task_name anomaly_detection \
    --stage test \
    --model LSTM_AE \
    --data ALFA \
    --root_path ./dataset/finetune/alfa/benchmark927 \
    --checkpoint ./checkpoints/alfa_lstm_h64_l2_sl100_cosine50/checkpoint.pth \
    --setting alfa_lstm_visualization \
    --features M \
    --seq_len 100 \
    --enc_in 16 \
    --hidden_dim 64 \
    --depth 2 \
    --batch_size 64 \
    --num_workers 2 \
    --anomaly_ratio 1 \
    --visualize \
    --vis_channels 0 3 6 \
    --vis_start 0 \
    --vis_context 100 \
    --vis_max_events 3
