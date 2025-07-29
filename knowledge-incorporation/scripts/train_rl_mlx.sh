#!/bin/bash

# This script launches the PPO reinforcement learning training for the SEAL MLX project.

# It assumes that the TTT_server_mlx.py is already running in a separate terminal.
# You can start it with: bash knowledge-incorporation/scripts/TTT_server_mlx.sh

# --- Configuration ---
MODEL_ID="mlx-community/Meta-Llama-3-8B-Instruct"
OUTPUT_DIR="logs/rl_training_run_3"
TOTAL_STEPS=8
SAVE_EVERY=2

# --- Argument Array ---
# Using an array to safely handle arguments
ARGS=(
    --model_id "$MODEL_ID"
    --output_dir "$OUTPUT_DIR"
    --total_ppo_steps "$TOTAL_STEPS"
    --save_every "$SAVE_EVERY"
    --batch_size3 32           # Much smaller for testing
    --mini_batch_size 4      # Smaller mini-batch
    --ppo_epochs 2           # Fewer epochs
    # You can override other parameters here, for example:
    # --learning_rate 2e-5
)

# --- Execution ---
echo "Starting RL training..."

# MLX Performance optimizations for Apple Silicon
export MLX_GPU_MEMORY_LIMIT=0.9
export PYTORCH_ENABLE_MPS_FALLBACK=1
export MLX_ENABLE_UNIFIED_MEMORY=1

python knowledge-incorporation/src/rl/train_rl_mlx.py "${ARGS[@]}"

echo "RL training script finished."
