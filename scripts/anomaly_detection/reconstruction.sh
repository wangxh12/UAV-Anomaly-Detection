#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
stage=${1:-pretrain}
if [[ $# -gt 0 ]]; then shift; fi
: "${DATA_ROOT:?Set DATA_ROOT to the prepared dataset directory}"
: "${CHANNELS:?Set CHANNELS to the actual number of input channels}"
model=${MODEL:-LSTM_AE}
data=${DATA:-ALFA}
runner=("${PYTHON:-python}")
if [[ ${NPROC:-1} -gt 1 ]]; then
	runner+=(-m torch.distributed.run --standalone --nproc_per_node="${NPROC}")
fi
"${runner[@]}" run.py \
	--task_name anomaly_detection \
	--stage "$stage" \
	--model "$model" \
	--data "$data" \
	--root_path "$DATA_ROOT" \
	--enc_in "$CHANNELS" \
	--seq_len "${SEQ_LEN:-96}" \
	--hidden_dim "${HIDDEN_DIM:-64}" \
	--depth "${DEPTH:-2}" \
	--batch_size "${BATCH_SIZE:-64}" \
	--num_workers "${NUM_WORKERS:-4}" \
	--train_test 0 \
	--setting "${RUN_NAME:-${stage}_${model}_${data}}" "$@"
