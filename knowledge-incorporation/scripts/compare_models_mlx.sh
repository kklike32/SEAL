#!/usr/bin/env bash
set -euo pipefail

###############################################################################
#  Compare two MLX models with the SEAL inner-loop.
#  - Starts a TTT server for each model on a private port
#  - Runs the same query-client
#  - Scrapes mean_gain from the JSON summary
#  - Prints a side-by-side table
###############################################################################

######################  USER SETTINGS  ########################################
DATASET="knowledge-incorporation/mlx_experiments/data/synthetic_data/train/squad_train_mlx_generated.json"

N_ARTICLES=10         
K_COMPLETIONS=3
EVAL_TIMES=1

BASE_MODEL="mlx-community/Meta-Llama-3-8B-Instruct"
NEW_MODEL="knowledge-incorporation/mlx_experiments/models/SEAL-Llama3-8B-final-fused"

RESULTS_ROOT="knowledge-incorporation/mlx_experiments/results/compare_$(date +%m%d_%H%M)"
###############################################################################

mkdir -p "${RESULTS_ROOT}"

# --- GLOBAL CLEAN-UP ----------------------------------------------------------
PIDS_TO_KILL=()
cleanup () {
  for pid in "${PIDS_TO_KILL[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" && wait "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT
# -----------------------------------------------------------------------------


# ---------- helper: find a random free port ----------------------------------
find_free_port () {
  local port
  while :; do
    port=$(( ( RANDOM % 20000 ) + 40000 ))   # 40000-60000
    if ! lsof -Pi :"$port" -sTCP:LISTEN -t >/dev/null; then
      echo "$port"; return
    fi
  done
}

# ---------- helper: wait until ZMQ socket listens ----------------------------
wait_port_ready () {
  local PORT=$1
  for _ in {1..20}; do
    if lsof -Pi :"$PORT" -sTCP:LISTEN -t >/dev/null; then
      return
    fi
    sleep 1
  done
  echo "Server on port ${PORT} did not come up in time" >&2
  return 1
}

run_one () {
  local MODEL=$1
  local NAME=$2
  local PORT
  PORT=$(find_free_port)

  local OUT_DIR="${RESULTS_ROOT}/${NAME}"
  mkdir -p "${OUT_DIR}"

  echo "------------------------------------------------------------"
  echo ">> Starting TTT server for ${NAME} on port ${PORT}"
  echo "------------------------------------------------------------"

  python -u -m knowledge-incorporation.src.inner.TTT_server_mlx \
        --model_id "${MODEL}" \
        --zmq_port "${PORT}"  \
        --max_seq_length 2048 \
        --eval_max_tokens 64  \
        --eval_temperature 0.0 \
        --instruct_model       \
        > "${OUT_DIR}/server.log" 2>&1 &
  SERVER_PID=$!
  PIDS_TO_KILL+=("${SERVER_PID}")

  wait_port_ready "${PORT}"

  echo ">> Running client for ${NAME}"
  python -u -m knowledge-incorporation.src.query.query_server_mlx \
        --exp_name        "${NAME}" \
        --dataset         "${DATASET}" \
        --output_dir      "${OUT_DIR}" \
        --server_host     "localhost" \
        --zmq_port        "${PORT}" \
        --n_articles      "${N_ARTICLES}" \
        --k_completions   "${K_COMPLETIONS}" \
        --eval_times      "${EVAL_TIMES}"   \
        --lora_rank 32 --lora_alpha 64 --lora_layers 2 \
        --finetune_epochs 3 --finetune_lr 1e-3 \
        --batch_size 1 --gradient_accumulation_steps 1 \
        > "${OUT_DIR}/client.log" 2>&1 || {
          echo "Client failed for ${NAME}. See ${OUT_DIR}/client.log" >&2
        }

  echo ">> Shutting down server ${NAME}"
  kill "${SERVER_PID}" && wait "${SERVER_PID}" 2>/dev/null || true
}

# ---------- baseline then new model ---------
run_one "${BASE_MODEL}" "baseline" 
run_one "${NEW_MODEL}"  "fused"

echo
echo "================  RESULTS  ================"
printf "%-10s %10s\n" "Model" "mean_gain"
for DIR in baseline fused; do
  if [ -f "${RESULTS_ROOT}/${DIR}/run_summary.json" ]; then
    MG=$(jq '.overall.mean_gain' "${RESULTS_ROOT}/${DIR}/run_summary.json")
  else
    MG="NaN"
  fi
  printf "%-10s %10s\n" "${DIR}" "${MG}"
done | column -t
echo "==========================================="
echo "Logs & JSON summaries are in ${RESULTS_ROOT}"
