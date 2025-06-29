#!/bin/bash

echo "Launching MLX Query Server Client..."

# -------- Environment ------------------------------------------------ #
# source ~/.bashrc
# conda activate seal_mlx
# cd ~/SEAL

# -------- User-editable Configurations --------------------------------- #
# This section defines the parameters for a single experimental run.
# These values will be sent to the TTT server.

# --- Paths and Names ---
EXP_NAME="mlx_run_iter1"
DATASET="knowledge-incorporation/mlx_experiments/data/synthetic_data/train/squad_train_mlx_generated.json"
OUTPUT_DIR="knowledge-incorporation/mlx_experiments/results/query_server_6"
SERVER_HOST="localhost"
ZMQ_PORT=5555

# --- Data Handling ---
# Limit the number of articles to process from the dataset for a quicker test run.
# Set to -1 to use all articles.
N_ARTICLES=3
# Use the top K completions from each article for training.
K_COMPLETIONS=5
# Run fine-tuning N times for each completion to check stability.
EVAL_TIMES=1

# --- LoRA / Optimisation Hyper-params (matching original script's defaults) ---
LORA_RANK=32
LORA_ALPHA=64
LORA_DROPOUT=0.0
LORA_LAYERS=4
FINETUNE_EPOCHS=10
FINETUNE_LR=1e-3
BATCH_SIZE=1
GRAD_ACC=1 # Our python script will multiply this with BATCH_SIZE

# ------------------------------------------------------------------------- #
mkdir -p "${OUTPUT_DIR}"

echo "Starting run: ${EXP_NAME}"
python3 -u -m knowledge-incorporation.src.query.query_server_mlx \
    --exp_name "${EXP_NAME}" \
    --dataset "${DATASET}" \
    --output_dir "${OUTPUT_DIR}" \
    --server_host "${SERVER_HOST}" \
    --zmq_port "${ZMQ_PORT}" \
    --n_articles "${N_ARTICLES}" \
    --k_completions "${K_COMPLETIONS}" \
    --eval_times "${EVAL_TIMES}" \
    --lora_rank "${LORA_RANK}" \
    --lora_alpha "${LORA_ALPHA}" \
    --lora_dropout "${LORA_DROPOUT}" \
    --lora_layers "${LORA_LAYERS}" \
    --finetune_epochs "${FINETUNE_EPOCHS}" \
    --finetune_lr "${FINETUNE_LR}" \
    --batch_size "${BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRAD_ACC}"

echo "Job finished. Results are in ${OUTPUT_DIR}"
