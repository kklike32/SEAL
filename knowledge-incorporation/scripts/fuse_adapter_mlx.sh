#!/bin/bash

# =================================================================================== #
#   Fuses a trained LoRA adapter with the base model to create a final,
#   standalone, merged model ready for inference using the official mlx-lm utility.
# =================================================================================== #

echo "Launching LoRA Adapter Fusion Script..."

# -------- Environment ------------------------------------------------ #
# source ~/.bashrc
# conda activate seal_mlx
# cd ~/SEAL

# -------- User-editable Configurations --------------------------------- #
# This section defines the paths for the fusion process.
# Ensure these paths match the outputs from your SFT training step.

# --- Paths and Model ---
# The original base model that the adapter was trained on.
BASE_MODEL="mlx-community/Meta-Llama-3-8B-Instruct"

# The directory where your SFT training script saved the adapter.
# This directory should contain the 'adapters.safetensors' and 'adapter_config.json' files.
ADAPTER_DIR="knowledge-incorporation/mlx_experiments/results/SFT/run1_merged"

# The final destination for the new, merged, standalone model.
OUTPUT_DIR="knowledge-incorporation/mlx_experiments/models/SEAL-Llama3-8B-final-fused"
# ------------------------------------------------------------------------- #

# --- Safety Check ---
if [ ! -d "$ADAPTER_DIR" ]; then
    echo "Error: Adapter directory not found at ${ADAPTER_DIR}"
    echo "Please ensure the --output_dir from your SFT training matches the ADAPTER_DIR here."
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "Starting model fusion using 'mlx_lm.fuse'..."
echo "| Base Model:    ${BASE_MODEL}"
echo "| Adapter Dir:   ${ADAPTER_DIR}"
echo "| Output Dir:    ${OUTPUT_DIR}"

# Use the official, built-in mlx-lm utility for fusing adapters.
mlx_lm.fuse --model "$BASE_MODEL" --adapter-path "$ADAPTER_DIR" --save-path "$OUTPUT_DIR"

echo "Fusion complete! Your new standalone model is ready at: ${OUTPUT_DIR}"
