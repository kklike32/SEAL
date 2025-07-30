#!/bin/bash

# This script launches the PPO reinforcement learning training for the SEAL MLX project.

# It assumes that the TTT_server_mlx.py is already running in a separate terminal.
# You can start it with: bash knowledge-incorporation/scripts/TTT_server_mlx.sh

# --- Configuration ---
MODEL_ID="mlx-community/Meta-Llama-3-8B-Instruct"
OUTPUT_DIR="logs/rl_training_run_5"
TOTAL_STEPS=8
SAVE_EVERY=2

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# --- Argument Array ---
# Using an array to safely handle arguments
ARGS=(
    --model_id "$MODEL_ID"
    --output_dir "$OUTPUT_DIR"
    --total_ppo_steps "$TOTAL_STEPS"
    --save_every "$SAVE_EVERY"
    --batch_size 8            # Reduced for better memory usage
    --mini_batch_size 4       # Smaller mini-batch
    --ppo_epochs 2            # Fewer epochs
    --learning_rate 1.41e-5   # Default SEAL learning rate
)

# --- Execution ---
echo "========================================"
echo "SEAL RL Training Starting..."
echo "========================================"
echo "Model: $MODEL_ID"
echo "Output Directory: $OUTPUT_DIR"
echo "Total PPO Steps: $TOTAL_STEPS"
echo "Batch Size: 8"
echo "Mini Batch Size: 4"
echo "Learning Rate: 1.41e-5"
echo "========================================"

# Check if TTT server is running
echo "Checking if TTT server is running on port 5555..."
if ! nc -z localhost 5555; then
    echo "ERROR: TTT server is not running on port 5555!"
    echo "Please start the TTT server first with:"
    echo "bash knowledge-incorporation/scripts/TTT_server_mlx.sh"
    exit 1
fi
echo "TTT server is running"

# MLX Performance optimizations for Apple Silicon
export MLX_GPU_MEMORY_LIMIT=0.9
export PYTORCH_ENABLE_MPS_FALLBACK=1
export MLX_ENABLE_UNIFIED_MEMORY=1

echo "Starting RL training..."
echo "Logs will be saved to: $OUTPUT_DIR/training_log.txt"
echo "========================================"

python knowledge-incorporation/src/rl/train_rl_mlx.py "${ARGS[@]}"

echo "========================================"
echo "RL training script finished."
echo "Check $OUTPUT_DIR for results and logs."
echo "========================================"
