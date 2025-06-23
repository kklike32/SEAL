# probe_test
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

# --- Configuration ---
MODEL_NAME = "Qwen/Qwen2.5-7B" # The model used in the scripts

# --- Device Check ---
if not torch.backends.mps.is_available():
    print("MPS not available. Exiting.")
    exit()

device = torch.device("mps")
print(f"✅ MPS device is available. Attempting to load model to: {device}")

# --- Model Loading Test ---
try:
    print(f"\n1. Loading Tokenizer for '{MODEL_NAME}'...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    print("   Tokenizer loaded successfully.")

    print(f"\n2. Loading Model '{MODEL_NAME}' to MPS device...")
    print("   This may take a few minutes and download several gigabytes...")

    start_time = time.time()
    # We load the model directly to the MPS device.
    # torch_dtype=torch.bfloat16 is not well supported on MPS, so we use float16
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16, 
        device_map=device,
        trust_remote_code=True,
    )
    end_time = time.time()

    print(f"\nSUCCESS! Model loaded to MPS in {end_time - start_time:.2f} seconds.")
    print(f"   Model is on device: {model.device}")

except Exception as e:
    print(f"\nFAILED to load model.")
    print(f"   Error: {e}")