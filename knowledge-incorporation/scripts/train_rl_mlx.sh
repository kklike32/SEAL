#!/bin/bash

# This script launches the PPO reinforcement learning training for the SEAL MLX project.

# It assumes that the TTT_server_mlx.py is already running in a separate terminal.
# You can start it with: bash knowledge-incorporation/scripts/TTT_server_mlx.sh

# --- Configuration ---
MODEL_ID="mlx-community/Meta-Llama-3-8B-Instruct"
OUTPUT_DIR="logs/rl_training_run_8"  
TOTAL_STEPS=16  
SAVE_EVERY=4  
BATCH_SIZE=16  
MINI_BATCH_SIZE=8  
PPO_EPOCHS=4  

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# --- Argument Array ---
# Using an array to safely handle arguments
ARGS=(
    --model_id "$MODEL_ID"
    --output_dir "$OUTPUT_DIR"
    --total_ppo_steps "$TOTAL_STEPS"
    --save_every "$SAVE_EVERY"
    --batch_size "$BATCH_SIZE"
    --mini_batch_size "$MINI_BATCH_SIZE"
    --ppo_epochs "$PPO_EPOCHS"
    --learning_rate 3e-5
)

# --- Execution ---
echo "========================================"
echo "SEAL RL Training Starting..."
echo "========================================"
echo "Model: $MODEL_ID"
echo "Output Directory: $OUTPUT_DIR"
echo "Total PPO Steps: $TOTAL_STEPS"
echo "Batch Size: $BATCH_SIZE"
echo "Mini Batch Size: $MINI_BATCH_SIZE"
echo "Learning Rate: 3e-5"
echo "========================================"

# Check if TTT server is running
echo "Checking if TTT server is running on port 5555..."
if ! nc -z localhost 5555; then
    echo "WARNING: TTT server is not currently running on port 5555!"
    echo "The RL training will wait for the server to become available."
    echo "Recommended: Use the monitor script - bash monitor_ttt_server.sh"
    echo "Continuing in 5 seconds..."
    sleep 5
fi
echo "TTT server connection available"

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
