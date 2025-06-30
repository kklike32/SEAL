#!/bin/bash

echo "Launching MLX Supervised Fine-Tuning..."

# -------- User-editable Configurations --------------------------------- #
# The base model we are fine-tuning
MODEL_ID="mlx-community/Meta-Llama-3-8B-Instruct-MLX"

# The SFT dataset we just created from the 30-article run
TRAIN_DATASET="knowledge-incorporation/mlx_experiments/results/query_server1_1/sft_best1of5_0629_192932.jsonl"

# Where to save the new, fully merged model
OUTPUT_DIR="knowledge-incorporation/results/SFT/run1_merged"

# --- Training Hyper-params ---
EPOCHS=3
LEARNING_RATE=2e-5
LORA_LAYERS=16
LORA_RANK=64
LORA_ALPHA=128
BATCH_SIZE=2
# ------------------------------------------------------------------------- #

mkdir -p "${OUTPUT_DIR}"

python3 -u -m knowledge-incorporation.src.EM.train_SFT_mlx \
    --model "${MODEL_ID}" \
    --train_file "${TRAIN_DATASET}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs "${EPOCHS}" \
    --learning_rate "${LEARNING_RATE}" \
    --lora_layers "${LORA_LAYERS}" \
    --lora_rank "${LORA_RANK}" \
    --lora_alpha "${LORA_ALPHA}" \
    --batch_size "${BATCH_SIZE}"

echo "Job finished. New merged model is in ${OUTPUT_DIR}"