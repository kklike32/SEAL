#!/bin/bash

# ------------------------------------------------------------------------------------ #
# This script runs the MLX-based data generation for the SEAL knowledge incorporation task.
# It automatically downloads a specified model from Hugging Face and uses it to 
# generate synthetic data from the SQuAD dataset.
# ------------------------------------------------------------------------------------ #

# -------- Environment ------------------------------------------------ #
# Activate the conda environment
# conda activate seal

# -------- User-editable Configurations ---------------------------------------------- #

# Set the Hugging Face repository ID of the model you want to use.
# mlx-lm will automatically download it if it's not in your cache.
MODEL_ID="Qwen/Qwen3-8B-MLX-4bit"

# Set to "instruct" or "base" to choose the appropriate prompt template.
# Qwen1.5-Chat is an instruction-tuned model.
MODEL_TYPE="instruct"

# --- Input and Output Paths ---
DATASET_IN="knowledge-incorporation/data/squad_train.json"
DATASET_OUT_DIR="knowledge-incorporation/mlx_experiments/data/synthetic_data/train"

# Uncomment the following lines to generate validation data instead.
# DATASET_IN="knowledge-incorporation/data/squad_val.json"
# DATASET_OUT_DIR="knowledge-incorporation/mlx_experiments/data/synthetic_data/eval"


# --- Generation Parameters ---
NUM_ARTICLES=50 
START_ARTICLE=0
K=5
TEMPERATURE=0.7
TOP_P=0.95
MAX_TOKENS=1024

# ---------------------------------------------------------------------------------- #

# Prepare arguments in an array. This is the most robust way to handle optional flags.
ARGS=(
    --model_path "$MODEL_ID"
    --dataset_in "$DATASET_IN"
    --dataset_out_dir "$DATASET_OUT_DIR"
    --n "$NUM_ARTICLES"
    --start "$START_ARTICLE"
    --k "$K"
    --temperature "$TEMPERATURE"
    --top_p "$TOP_P"
    --max_tokens "$MAX_TOKENS"
)

# Conditionally add the --instruct_model flag to the arguments array.
if [ "$MODEL_TYPE" == "instruct" ]; then
    ARGS+=(--instruct_model)
fi

# Run the Python script for data generation, passing the arguments array.
python3 knowledge-incorporation/src/data_generation/make_squad_data_mlx.py "${ARGS[@]}"

echo "Data generation finished. Output saved in ${DATASET_OUT_DIR}"
