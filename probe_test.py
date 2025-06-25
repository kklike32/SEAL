# probe_test.py using mlx_lm
from mlx_lm import load, generate
import torch

MODEL_NAME = "Qwen/Qwen3-8B-MLX-4bit"

print("✅ MPS device is available." if torch.backends.mps.is_available() else "MPS not available")

# Load model using MLX
print(f"Loading MLX model: {MODEL_NAME}")
model, tokenizer = load(MODEL_NAME)

# Simple test prompt
prompt = "Hello, how can I help you today?"

# Format as chat if needed
if tokenizer.chat_template is not None:
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False
    )

# Generate output
response = generate(model, tokenizer, prompt=prompt, verbose=True)
print("\n=== Response ===")
print(response)



#!/usr/bin/env python3

# =================================================================================== #
#   Launches the unified MLX-based Test-Time Training (TTT) Server for SEAL.
#   This script starts a single process that listens for ZMQ requests and performs
#   both LoRA fine-tuning and evaluation on the Mac's GPU using MLX.
#   VERSION: Aligned with the modern mlx_lm.tuner API.
# =================================================================================== #

