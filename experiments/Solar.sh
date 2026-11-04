#!/bin/bash

set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TRAIN_PY="${PROJECT_ROOT}/train.py"

DATA_TYPE="solar"
MODEL_NAME="chronoorg"
SEQ_LEN=336

PRED_LENS=(96 192 336 720)

PATCH_LEN=12
PATCH_STRIDE=6
D_MODEL=128
POOLING="last"
DEPTHS=(2 2 2)
NUM_HEADS=(8 4 4)
PERIOD=144
PATIENCE=5

LOG_DIR="${SCRIPT_DIR}/logs/logs_${MODEL_NAME}_${DATA_TYPE}_seq${SEQ_LEN}"
mkdir -p "${LOG_DIR}"

RESULT_DIR="${SCRIPT_DIR}/TXT"
mkdir -p "${RESULT_DIR}"

RESULT_FILE="${RESULT_DIR}/results_${MODEL_NAME}_${DATA_TYPE}_seq${SEQ_LEN}.txt"
> "${RESULT_FILE}"

if [ ! -f "${TRAIN_PY}" ]; then
    echo "ERROR: train.py not found at ${TRAIN_PY}"
    exit 1
fi

get_config () {
    local pred_len=$1

    case "${pred_len}" in
        96|720)
            DROPOUT=0.2
            STAGE_DIMS=(168 192 256)
            WINDOW_SIZE=(24 24 24)
            AFFINE=0
            MLP_RATIO=2
            ;;

        192)
            DROPOUT=0.1
            STAGE_DIMS=(168 192 256)
            WINDOW_SIZE=(24 12 12)
            AFFINE=0
            MLP_RATIO=4
            ;;

        336)
            DROPOUT=0.1
            STAGE_DIMS=(128 192 256)
            WINDOW_SIZE=(12 12 6)
            AFFINE=1
            MLP_RATIO=4
            ;;

        *)
            echo "ERROR: Unsupported pred_len=${pred_len}" >&2
            return 1
            ;;
    esac
}

run_task () {
    local pred_len=$1

    get_config "${pred_len}" || return 1

    local log_file="${LOG_DIR}/${MODEL_NAME}_${DATA_TYPE}_seq${SEQ_LEN}_pred${pred_len}.log"

    echo "============================================================"
    echo "Running configuration:"
    echo "  model_name   = ${MODEL_NAME}"
    echo "  data_type    = ${DATA_TYPE}"
    echo "  seq_len      = ${SEQ_LEN}"
    echo "  pred_len     = ${pred_len}"
    echo "  patch_len    = ${PATCH_LEN}"
    echo "  patch_stride = ${PATCH_STRIDE}"
    echo "  d_model      = ${D_MODEL}"
    echo "  dropout      = ${DROPOUT}"
    echo "  pooling      = ${POOLING}"
    echo "  stage_dims   = ${STAGE_DIMS[*]}"
    echo "  depths       = ${DEPTHS[*]}"
    echo "  num_heads    = ${NUM_HEADS[*]}"
    echo "  window_size  = ${WINDOW_SIZE[*]}"
    echo "  affine       = ${AFFINE}"
    echo "  period       = ${PERIOD}"
    echo "  mlp_ratio    = ${MLP_RATIO}"
    echo "  patience     = ${PATIENCE}"
    echo "  log_file     = ${log_file}"
    echo "============================================================"

    : > "${log_file}"

    (
        cd "${PROJECT_ROOT}" || exit 1

        python3 "${TRAIN_PY}" \
            --model_name "${MODEL_NAME}" \
            --data_type "${DATA_TYPE}" \
            --seq_len "${SEQ_LEN}" \
            --pred_len "${pred_len}" \
            --patch_len "${PATCH_LEN}" \
            --patch_stride "${PATCH_STRIDE}" \
            --d_model "${D_MODEL}" \
            --dropout "${DROPOUT}" \
            --pooling "${POOLING}" \
            --stage_dims "${STAGE_DIMS[@]}" \
            --depths "${DEPTHS[@]}" \
            --num_heads "${NUM_HEADS[@]}" \
            --window_size "${WINDOW_SIZE[@]}" \
            --affine "${AFFINE}" \
            --period "${PERIOD}" \
            --mlp_ratio "${MLP_RATIO}" \
            --patience "${PATIENCE}"
    ) 2>&1 | tee "${log_file}"

    local status=${PIPESTATUS[0]}
    local test_line

    test_line=$(grep -F "[TEST]" "${log_file}" | tail -n 1)

    if [ -z "${test_line}" ]; then
        test_line=$(tail -n 1 "${log_file}")
    fi

    if [ "${status}" -eq 0 ]; then
        echo "${MODEL_NAME} | data_type=${DATA_TYPE} | seq_len=${SEQ_LEN} | pred_len=${pred_len} | patch_len=${PATCH_LEN} | patch_stride=${PATCH_STRIDE} | d_model=${D_MODEL} | dropout=${DROPOUT} | stage_dims=${STAGE_DIMS[*]} | depths=${DEPTHS[*]} | num_heads=${NUM_HEADS[*]} | window_size=${WINDOW_SIZE[*]} | affine=${AFFINE} | period=${PERIOD} | mlp_ratio=${MLP_RATIO} | ${test_line}" \
            | tee -a "${RESULT_FILE}"

        echo "Finished: ${MODEL_NAME} | pred_len=${pred_len}"
    else
        echo "${MODEL_NAME} | data_type=${DATA_TYPE} | seq_len=${SEQ_LEN} | pred_len=${pred_len} | patch_len=${PATCH_LEN} | patch_stride=${PATCH_STRIDE} | d_model=${D_MODEL} | dropout=${DROPOUT} | stage_dims=${STAGE_DIMS[*]} | depths=${DEPTHS[*]} | num_heads=${NUM_HEADS[*]} | window_size=${WINDOW_SIZE[*]} | affine=${AFFINE} | period=${PERIOD} | mlp_ratio=${MLP_RATIO} | FAILED | ${test_line}" \
            | tee -a "${RESULT_FILE}"

        echo "FAILED: ${MODEL_NAME} | pred_len=${pred_len}"
    fi

    echo "Log saved to: ${log_file}"
    echo "------------------------------------------------------------"
}

for pred_len in "${PRED_LENS[@]}"; do
    run_task "${pred_len}"
done

echo "============================================================"
echo "All tasks finished."
echo "Results saved to: ${RESULT_FILE}"
echo "============================================================"