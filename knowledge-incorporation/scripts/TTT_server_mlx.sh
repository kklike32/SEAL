#!/bin/bash

# =================================================================================== #
#   Launches the unified MLX-based Test-Time Training (TTT) Server for SEAL.
#   This script starts a single process that listens for ZMQ requests and performs
#   both LoRA fine-tuning and evaluation on the Mac's GPU.
# =================================================================================== #

echo "Launching MLX TTT server on $(hostname)..."

# -------- Environment ------------------------------------------------ #
# source ~/.bashrc
# conda activate seal_mlx
# # Navigate to the root of the SEAL project directory
# cd ~/SEAL

# -------- User-editable Configurations ---------------------------------------------- #
# The Hugging Face repo ID of the MLX model to use for training and evaluation.
MODEL_ID="Qwen/Qwen3-8B-MLX-4bit"

# The ZMQ port the server will listen on for requests from the outer-loop driver.
ZMQ_PORT=5555

# Set if the model is an instruction-tuned variant.
INSTRUCT_MODEL_FLAG="--instruct_model"

# Maximum sequence length for the training sequences.
MAX_SEQ_LENGTH=2048

# Parameters for the evaluation generation step.
EVAL_MAX_TOKENS=64
EVAL_TEMPERATURE=0.0
# ---------------------------------------------------------------------------------- #

# Ensure the OPENAI_API_KEY is available for the grading step.
# Make sure you have a .env file in the SEAL root directory with OPENAI_API_KEY=...
if [ -f .env ]; then
    set -a
    source .env
    set +a
    echo "Sourced .env file."
fi

if [ -z "$OPENAI_API_KEY" ]; then
    echo "Warning: OPENAI_API_KEY is not set. The grading step will fail."
fi

# Construct the arguments for the python script.
ARGS=(
    --model_id "$MODEL_ID"
    --zmq_port ${ZMQ_PORT}
    --max_seq_length ${MAX_SEQ_LENGTH}
    --eval_max_tokens ${EVAL_MAX_TOKENS}
    --eval_temperature ${EVAL_TEMPERATURE}
    ${INSTRUCT_MODEL_FLAG}
)

echo "Starting Inner Loop server..."
echo "Model: ${MODEL_ID}"
echo "Listening on ZMQ port: ${ZMQ_PORT}"

# Launch the single, unified server process.
# The -u flag ensures unbuffered python output, which is better for logging.
python3 -u -m knowledge-incorporation.src.inner.TTT_server_mlx "${ARGS[@]}"

echo "Job finished."
