#!/bin/bash

# Quick Test Script for ARC RestEM
# This runs a minimal test to verify the implementation works

# ================================
# TEST CONFIGURATION
# ================================

# Model Configuration  
MODEL_NAME="mlx-community/Meta-Llama-3-8B-Instruct"

# Minimal test configuration (should complete in ~10-15 minutes)
MAX_TASKS=1              # Just 1 task for quick test
N_CONFIGS=1              # Just 1 configuration for extreme minimal test 
RESTEM_EPOCHS=1          # Minimal epochs
LOG_LEVEL="INFO"

# Directory Configuration
DATA_DIR="few-shot/data"
OUTPUT_DIR="logs/arc_restem_test"

# ================================
# SCRIPT EXECUTION
# ================================

# Get script directory and navigate to SEAL root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEAL_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESTEM_DIR="$SEAL_ROOT/knowledge-incorporation/src/restem"

echo "=================================="
echo "ARC RestEM Quick Test"
echo "=================================="
echo "SEAL Root: $SEAL_ROOT"
echo "Model: $MODEL_NAME"
echo "Tasks: $MAX_TASKS (minimal test)"
echo "Configs per task: $N_CONFIGS"
echo "RestEM epochs: $RESTEM_EPOCHS"
echo "Expected duration: ~10-15 minutes"
echo "=================================="

# Check if we're in the right directory
if [ ! -f "$RESTEM_DIR/run_arc_restem.py" ]; then
    echo "ERROR: Cannot find run_arc_restem.py at $RESTEM_DIR"
    echo "Make sure you're running this script from the SEAL repository"
    exit 1
fi

# Stay in SEAL root directory
cd "$SEAL_ROOT" || exit 1

# Check Python dependencies
echo "Checking Python environment..."
if ! python3 -c "import mlx" 2>/dev/null; then
    echo "WARNING: MLX not found. Installing requirements..."
    pip install -r requirements_mlx.txt
fi

# Configure MLX memory settings for Apple Silicon - ULTRA CONSERVATIVE
export MLX_GPU_MEMORY_LIMIT=0.3
export PYTORCH_ENABLE_MPS_FALLBACK=1
export MLX_ENABLE_UNIFIED_MEMORY=1
export MLX_METAL_BUFFER_CACHE_LIMIT=50
export MLX_MEMORY_POOL_LIMIT=2048

echo "MLX memory configuration applied (30% GPU memory limit for conservative test)"

# Create timestamp for this run
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
echo "Starting quick test at $TIMESTAMP"

# Run the pipeline from SEAL root
echo "Executing ARC RestEM Quick Test..."
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
    echo "✅ Quick test completed successfully!"
    echo "Results saved to: $SEAL_ROOT/$OUTPUT_DIR"
    echo ""
    echo "Next steps:"
    echo "1. Check the logs for detailed output"
    echo "2. Verify the 4-phase pipeline executed correctly"
    echo "3. Run the full experiment with: ./knowledge-incorporation/scripts/run_arc_restem.sh"
elif [ $EXIT_CODE -eq 130 ]; then
    echo "⚠️  Test interrupted by user (Ctrl+C)"
else
    echo "❌ Quick test failed with exit code $EXIT_CODE"
    echo "Check the logs for error details."
    echo ""
    echo "Common issues:"
    echo "- Make sure MLX is properly installed"
    echo "- Verify you're running from the SEAL root directory"
    echo "- Check that all dependencies are available"
fi
echo "=================================="

exit $EXIT_CODE
