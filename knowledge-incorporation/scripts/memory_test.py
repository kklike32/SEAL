#!/usr/bin/env python3
"""
Simple memory test to isolate the MLX memory issue
"""

import os
import sys
import gc

# Configure MLX memory settings
os.environ['MLX_GPU_MEMORY_LIMIT'] = '0.3'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['MLX_ENABLE_UNIFIED_MEMORY'] = '1'
os.environ['MLX_METAL_BUFFER_CACHE_LIMIT'] = '50'

print("Testing MLX memory allocation...")
print(f"MLX_GPU_MEMORY_LIMIT: {os.environ.get('MLX_GPU_MEMORY_LIMIT', 'not set')}")

try:
    print("1. Importing MLX...")
    import mlx.core as mx
    from mlx_lm import load
    print("   ✓ MLX imported successfully")
    
    print("2. Loading single model...")
    model, tokenizer = load("mlx-community/Meta-Llama-3-8B-Instruct")
    print("   ✓ Model loaded successfully")
    
    print("3. Testing generation...")
    from mlx_lm import generate
    response = generate(
        model, 
        tokenizer, 
        prompt="Hello", 
        max_tokens=10
    )
    print(f"   ✓ Generation successful: {response[:50]}...")
    
    print("4. Cleaning up...")
    del model, tokenizer
    gc.collect()
    print("   ✓ Cleanup complete")
    
    print("✅ Memory test passed!")
    
except Exception as e:
    print(f"❌ Memory test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
