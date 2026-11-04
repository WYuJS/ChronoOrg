#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_NAME="chronoorg"
DATA_TYPE="exchange"
SEQ_LEN=336

LOG_DIR="${SCRIPT_DIR}/logs/logs_${MODEL_NAME}_${DATA_TYPE}_seq${SEQ_LEN}"
mkdir -p "${LOG_DIR}"

RESULT_DIR="${SCRIPT_DIR}/TXT"
mkdir -p "${RESULT_DIR}"

RESULT_FILE="${RESULT_DIR}/results_${MODEL_NAME}_${DATA_TYPE}_seq${SEQ_LEN}.txt"
> "${RESULT_FILE}"

run_task () {
  local pred_len=$1
  shift

  local log_file="${LOG_DIR}/${MODEL_NAME}_${DATA_TYPE}_seq${SEQ_LEN}_pred${pred_len}.log"

  echo "Running ${MODEL_NAME} | ${DATA_TYPE} | seq_len=${SEQ_LEN} | pred_len=${pred_len}"
  echo "Running ${MODEL_NAME} | ${DATA_TYPE} | seq_len=${SEQ_LEN} | pred_len=${pred_len}" >> "${RESULT_FILE}"

  (
    cd "${PROJECT_ROOT}" || exit 1

    python3 train.py \
      --model_name "${MODEL_NAME}" \
      --data_type "${DATA_TYPE}" \
      --seq_len "${SEQ_LEN}" \
      --pred_len "${pred_len}" \
      "$@"
  ) 2>&1 | tee "${log_file}"

  grep "\[TEST\]" "${log_file}" | tail -n 1 >> "${RESULT_FILE}"
  echo "" >> "${RESULT_FILE}"
}


# pred_len=96
run_task 96 \
  --patch_len 24 \
  --patch_stride 8 \
  --d_model 128 \
  --dropout 0.005 \
  --pooling last \
  --stage_dims 128 192 256 \
  --depths 2 2 2 \
  --stage_use_global 0,1 0,0 0,1 \
  --num_heads 4 4 4 \
  --window_size 4 4 7 \
  --affine 1 \
  --mlp_ratio 2 \
  --subtract_last 1 \
  --period 30 \
  --patience 5


# pred_len=192
run_task 192 \
  --patch_len 24 \
  --patch_stride 6 \
  --d_model 128 \
  --dropout 0.1 \
  --pooling last \
  --stage_dims 128 256 256 \
  --depths 2 2 2 \
  --stage_use_global 0,0 0,1 0,0 \
  --num_heads 4 4 4 \
  --window_size 5 4 4 \
  --affine 1 \
  --mlp_ratio 2 \
  --subtract_last 1 \
  --period 30 \
  --patience 5


# pred_len=336
run_task 336 \
  --patch_len 24 \
  --patch_stride 12 \
  --d_model 128 \
  --dropout 0.1 \
  --pooling last \
  --stage_dims 128 256 256 \
  --depths 2 2 4 \
  --stage_use_global 0,0 0,0 0,1,0,1 \
  --num_heads 4 4 4 \
  --window_size 4 4 4 \
  --affine 0 \
  --mlp_ratio 2 \
  --period 30 \
  --patience 3


# pred_len=720
run_task 720 \
  --patch_len 24 \
  --patch_stride 12 \
  --d_model 128 \
  --dropout 0.15 \
  --pooling last \
  --stage_dims 128 192 256 \
  --depths 2 2 2 \
  --num_heads 4 4 4 \
  --window_size 4 4 4 \
  --affine 0 \
  --mlp_ratio 2 \
  --period 30 \
  --patience 5


echo "Results saved to ${RESULT_FILE}"