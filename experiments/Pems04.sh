#!/bin/bash 


# chronoorg | data_type=pems04 | seq_len=336 | pred_len=12 | [TEST] Model=chronoorg | Data=pems04 | Prediction=12 | MSE=0.0704 | MAE=0.1702 | RMSE=0.2653
# chronoorg | data_type=pems04 | seq_len=336 | pred_len=24 | [TEST] Model=chronoorg | Data=pems04 | Prediction=24 | MSE=0.0831 | MAE=0.1850 | RMSE=0.2882
# chronoorg | data_type=pems04 | seq_len=336 | pred_len=48 | [TEST] Model=chronoorg | Data=pems04 | Prediction=48 | MSE=0.1038 | MAE=0.2062 | RMSE=0.3222
# chronoorg | data_type=pems04 | seq_len=336 | pred_len=96 | [TEST] Model=chronoorg | Data=pems04 | Prediction=96 | MSE=0.1191 | MAE=0.2157 | RMSE=0.3451



SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SEQ_LEN=336
DATA_TYPE="pems04"
MODEL_NAME="chronoorg"

PRED_LENS=(12 24 48 96)
# PRED_LENS=(12)

LOG_DIR="${SCRIPT_DIR}/logs/logs_${MODEL_NAME}_${DATA_TYPE}_seq${SEQ_LEN}"
mkdir -p "${LOG_DIR}"

RESULT_DIR="${SCRIPT_DIR}/TXT"
mkdir -p "${RESULT_DIR}"

RESULT_FILE="${RESULT_DIR}/results_${MODEL_NAME}_${DATA_TYPE}_seq${SEQ_LEN}.txt"
> "${RESULT_FILE}"

run_task () {
    local pred_len=$1
    local log_file="${LOG_DIR}/${MODEL_NAME}_${DATA_TYPE}_seq${SEQ_LEN}_pred${pred_len}.log"

    echo "Running: ${MODEL_NAME} | data_type=${DATA_TYPE} | seq_len=${SEQ_LEN} | pred_len=${pred_len}"

    : > "${log_file}"

    cd "${PROJECT_ROOT}" || exit 1

    python3 train.py \
        --model_name "${MODEL_NAME}" \
        --data_type "${DATA_TYPE}" \
        --seq_len "${SEQ_LEN}" \
        --pred_len "${pred_len}" \
        --patch_len 12 \
        --patch_stride 6 \
        --d_model 128 \
        --dropout 0.1 \
        --pooling last \
        --stage_dims 128 192 256 \
        --depths 2 2 2 \
        --num_heads 8 4 4\
        --window_size 24 24 24  \
        --affine 1 \
        --period 288 \
        --mlp_ratio 4 \
        2>&1 | tee "${log_file}"

    STATUS=${PIPESTATUS[0]}
    LAST_LINE=$(tail -n 1 "${log_file}")

    if [ ${STATUS} -eq 0 ]; then
        echo "${MODEL_NAME} | data_type=${DATA_TYPE} | seq_len=${SEQ_LEN} | pred_len=${pred_len} | ${LAST_LINE}" >> "${RESULT_FILE}"
        echo "Finished: ${MODEL_NAME} | data_type=${DATA_TYPE} | seq_len=${SEQ_LEN} | pred_len=${pred_len}"
    else
        echo "${MODEL_NAME} | data_type=${DATA_TYPE} | seq_len=${SEQ_LEN} | pred_len=${pred_len} | FAILED | ${LAST_LINE}" >> "${RESULT_FILE}"
        echo "FAILED: ${MODEL_NAME} | data_type=${DATA_TYPE} | seq_len=${SEQ_LEN} | pred_len=${pred_len}"
    fi

    echo "----------------------------------------"
}

for pred_len in "${PRED_LENS[@]}"; do
    run_task "${pred_len}"
done

echo "All tasks finished."
echo "Results saved to: ${RESULT_FILE}"