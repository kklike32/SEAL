#!/bin/bash

echo "Launching MLX Supervised Fine-Tuning..."

# -------- User-editable Configurations --------------------------------- #
# The base model we are fine-tuning
MODEL_ID="mlx-community/Meta-Llama-3-8B-Instruct"

# The SFT dataset we just created
TRAIN_DATASET="knowledge-incorporation/mlx_experiments/results/query_server1_1/sft_best1of5_0629_192932.jsonl"

# Where to save the new LoRA adapter
OUTPUT_DIR="knowledge-incorporation/results/SFT/run1"

# --- Training Hyper-params ---
EPOCHS=3
LEARNING_RATE=1e-5
LORA_LAYERS=16
# ------------------------------------------------------------------------- #

mkdir -p "${OUTPUT_DIR}"

python3 -u -m knowledge-incorporation.src.EM.train_SFT_mlx \
    --model "${MODEL_ID}" \
    --train-dataset "${TRAIN_DATASET}" \
    --output_dir "${OUTPUT_DIR}" \
    --epochs "${EPOCHS}" \
    --lr "${LEARNING_RATE}" \
    --lora-layers "${LORA_LAYERS}"

echo "Job finished. New LoRA adapter is in ${OUTPUT_DIR}"
