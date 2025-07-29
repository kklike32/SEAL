#!/bin/bash

# This script launches the PPO reinforcement learning training for the SEAL MLX project.

# It assumes that the TTT_server_mlx.py is already running in a separate terminal.
# You can start it with: bash knowledge-incorporation/scripts/TTT_server_mlx.sh

# --- Configuration ---
MODEL_ID="mlx-community/Qwen1.5-7B-Chat-MLX-4bit"
OUTPUT_DIR="mlx_experiments/rl_training_run_1"
TOTAL_STEPS=20
SAVE_EVERY=5

# --- Argument Array ---
# Using an array to safely handle arguments
ARGS=(
    --model_id "$MODEL_ID"
    --output_dir "$OUTPUT_DIR"
    --total_ppo_steps "$TOTAL_STEPS"
    --save_every "$SAVE_EVERY"
    # You can override other parameters here, for example:
    # --learning_rate 2e-5
    # --batch_size 128
)

# --- Execution ---
echo "Starting RL training..."

python knowledge-incorporation/src/rl/train_rl_mlx.py "${ARGS[@]}"

echo "RL training script finished."
