#!/bin/bash

set -e

# ============================================================
# GPU configuration
# ============================================================

# Two-GPU DDP smoke test
export CUDA_VISIBLE_DEVICES=0

NPROC_PER_NODE=1


# ============================================================
# Model configuration
# ============================================================

MODEL=LSTM_AE

SEQ_LEN=100
ENC_IN=25

HIDDEN_DIM=64
DEPTH=2

BATCH_SIZE=64
LEARNING_RATE=0.001
TRAIN_EPOCHS=1


# ============================================================
# Run
# ============================================================

torchrun --standalone --nproc_per_node=${NPROC_PER_NODE} run.py \
    --task_name anomaly_detection \
    --is_training 1 \
    --is_finetuning 0 \
    --model_id LSTM_AE_PSM_DDP \
    --model ${MODEL} \
    --data PSM \
    --root_path ./dataset/PSM \
    --features M \
    --seq_len ${SEQ_LEN} \
    --pred_len 0 \
    --enc_in ${ENC_IN} \
    --c_out ${ENC_IN} \
    --hidden_dim ${HIDDEN_DIM} \
    --depth ${DEPTH} \
    --d_model ${HIDDEN_DIM} \
    --patch_len 4 \
    --stride 4 \
    --dropout 0.1 \
    --batch_size ${BATCH_SIZE} \
    --learning_rate ${LEARNING_RATE} \
    --train_epochs ${TRAIN_EPOCHS} \
    --patience 3 \
    --anomaly_ratio 1 \
    --num_workers 2 \
    --itr 1