#!/bin/bash

# ARC RestEM Pipeline Runner
# Convenient script to run the SEAL RestEM implementation with configurable parameters

# ================================
# CONFIGURATION VARIABLES
# ================================

# Model Configuration
MODEL_NAME="mlx-community/Meta-Llama-3-8B-Instruct"

# Experiment Configuration
MAX_TASKS=3              # Number of ARC tasks to process
N_CONFIGS=15             # Number of configurations per task
RESTEM_EPOCHS=8          # Number of epochs for RestEM training

# Logging Configuration
LOG_LEVEL="INFO"         # DEBUG, INFO, WARNING, ERROR

# Directory Configuration (relative to SEAL root)
DATA_DIR="few-shot/data"                       # ARC data location
OUTPUT_DIR="logs/arc_restem_experiments"       # Results output

# ================================
# QUICK PRESET CONFIGURATIONS
# ================================

# Uncomment one of these presets to use predefined configurations:

# QUICK TEST (30 minutes)
# MAX_TASKS=2
# N_CONFIGS=5
# RESTEM_EPOCHS=3

# MEDIUM TEST (1 hour)
# MAX_TASKS=3
# N_CONFIGS=10
# RESTEM_EPOCHS=5

# FULL EXPERIMENT (2-3 hours)
# MAX_TASKS=5
# N_CONFIGS=15
# RESTEM_EPOCHS=8

# HIGH PERFORMANCE MODE (for M4 Pro/Max with 32GB+)
# MAX_TASKS=8
# N_CONFIGS=20
# RESTEM_EPOCHS=10

# DEBUG MODE
# MAX_TASKS=1
# N_CONFIGS=3
# RESTEM_EPOCHS=2
# LOG_LEVEL="DEBUG"

# LOW MEMORY MODE (for systems with limited GPU memory)
# MAX_TASKS=1
# N_CONFIGS=2
# RESTEM_EPOCHS=1
# LOG_LEVEL="INFO"

# ================================
# SCRIPT EXECUTION
# ================================

# Get the script directory and navigate to SEAL root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEAL_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESTEM_DIR="$SEAL_ROOT/knowledge-incorporation/src/restem"

echo "=================================="
echo "ARC RestEM Pipeline Runner"
echo "=================================="
echo "SEAL Root: $SEAL_ROOT"
echo "Model: $MODEL_NAME"
echo "Tasks: $MAX_TASKS"
echo "Configs per task: $N_CONFIGS"
echo "RestEM epochs: $RESTEM_EPOCHS"
echo "Log level: $LOG_LEVEL"
echo "=================================="

# Check if we're in the right directory
if [ ! -f "$RESTEM_DIR/run_arc_restem.py" ]; then
    echo "ERROR: Cannot find run_arc_restem.py at $RESTEM_DIR"
    echo "Make sure you're running this script from the SEAL repository"
    exit 1
fi

# Stay in SEAL root directory (don't change to restem)
cd "$SEAL_ROOT" || exit 1

# Check Python dependencies
echo "Checking Python environment..."
if ! python3 -c "import mlx" 2>/dev/null; then
    echo "WARNING: MLX not found. Installing requirements..."
    pip install -r requirements_mlx.txt
fi

# Configure MLX memory settings for Apple Silicon
export MLX_GPU_MEMORY_LIMIT=0.8
export PYTORCH_ENABLE_MPS_FALLBACK=1
export MLX_ENABLE_UNIFIED_MEMORY=1

echo "MLX memory configuration applied (80% GPU memory limit for high-memory system)"

# Create timestamp for this run
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
echo "Starting RestEM run at $TIMESTAMP"

# Run the pipeline from SEAL root
echo "Executing ARC RestEM Pipeline..."
python3 knowledge-incorporation/src/restem/run_arc_restem.py \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --model-name "$MODEL_NAME" \
    --max-tasks "$MAX_TASKS" \
    --n-configs "$N_CONFIGS" \
    --restem-epochs "$RESTEM_EPOCHS" \
    --log-level "$LOG_LEVEL"

EXIT_CODE=$?

echo "=================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ RestEM run completed successfully!"
    echo "Results saved to: $SEAL_ROOT/$OUTPUT_DIR"
    echo "Check the logs for detailed information."
elif [ $EXIT_CODE -eq 130 ]; then
    echo "⚠️  Run interrupted by user (Ctrl+C)"
else
    echo "❌ RestEM run failed with exit code $EXIT_CODE"
    echo "Check the logs for error details."
fi
echo "=================================="

exit $EXIT_CODE
