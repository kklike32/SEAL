# Project SEAL on MLX: Phase 2 - Test-Time Training (TTT) Server

**Date:** June 24, 2025  
**Objective:** To adapt the core inner-loop fine-tuning mechanism of the SEAL paper—the Test-Time Training (TTT) server—to run efficiently on a macOS environment with Apple Silicon using the MLX framework.

---

## 1. Project Goal

Following the successful generation of synthetic data in Phase 1, the goal of Phase 2 was to build the engine that uses this data. The original SEAL repository implements a TTT server using a complex, two-process architecture involving a PyTorch/PEFT training script and a separate VLLM inference server. Our objective was to re-engineer this system into a single, unified server that performs all necessary tasks (evaluation, on-demand LoRA fine-tuning, and adapted evaluation) locally on a Mac.

---

## 2. Architectural Adaptation: From Two-Process to a Unified Server

A fundamental change was required to adapt the SEAL TTT framework to our environment.

### Original Architecture (for GPU Clusters):
- **VLLM Inference Server:** A dedicated process running on one or more GPUs, serving as a fast engine for text generation. It could dynamically load LoRA adapters via API calls.
- **PyTorch Training Script:** A separate process on another GPU that handled the LoRA fine-tuning using `transformers.Trainer` and `peft`. It would save the adapter to disk and then make API calls to the VLLM server to load, use, and unload it.

### Our New MLX Architecture (for Apple Silicon):
This two-process model is inefficient and incompatible with a single-GPU Mac setup. We re-designed it as a single, unified Python script (`TTT_server_mlx.py`) that:
- Loads the base MLX model once at startup.
- Listens for network requests via a ZMQ socket.
- For each request, it performs the entire TTT cycle in memory: baseline evaluation, LoRA fine-tuning, and adapted evaluation.
- Cleverly manages model state to ensure each request is stateless and independent.

This unified architecture is significantly cleaner, more efficient for our hardware, and removes the dependency on VLLM and complex inter-process communication.

---

## 3. Methodology & Implementation

The implementation was split into a reusable utility module and the core server script.

### 3.1. The Utility Layer (`utils_mlx.py`)

To avoid code duplication and promote modularity, we created a new `utils_mlx.py` file. Its purpose is to provide MLX-native replacements for VLLM-dependent functions while importing all other platform-agnostic helpers from the original `utils.py`.

**Core Functions:**
- `generate_mlx()`: An MLX-native replacement for the VLLM-based text generation function, using `mlx_lm.generate`.
- `accuracy_and_texts_mlx()`: An orchestrator for the evaluation pipeline. It uses the imported `format_answer_prompts` and `grade_with_gpt4` from the original `utils.py` but calls our new `generate_mlx` for inference.

**Final Code:** The full code is available in `knowledge-incorporation/src/utils_mlx.py`.

### 3.2. The Core Server Logic (`TTT_server_mlx.py`)

This is the main server file, built from the ground up to be robust and efficient on MLX.

- **`ServerState` Class:** A key feature of our implementation is this class, which handles model state. It loads the base model from Hugging Face once at startup and immediately creates a deep copy of its "pristine" weights. After each fine-tuning request, it restores the model from this pristine copy, ensuring perfect statelessness without the high cost of reloading the model from disk every time.
- **LoRA Fine-Tuning with `mlx_lm.tuner`:** We use the modern, high-level `mlx_lm.tuner.train` API. This abstracts away the complexities of writing a manual training loop and is the officially recommended method for fine-tuning in `mlx-lm`. It correctly handles batching, optimization, and applying LoRA layers.
- **Methodological Fidelity (Prompt Loss Masking):** To faithfully replicate the paper's methodology, our data preparation pipeline explicitly masks the prompt tokens in the training labels by setting them to -100. This ensures the model only learns from the new information in the completion, which is a critical detail for efficient training.

**Final Code:** The complete, final code for the server is documented in `knowledge-incorporation/src/inner/TTT_server_mlx.py`.

---

## 4. Parameter & Hyperparameter Parity

A thorough comparison was made to ensure the hyperparameters in our MLX server could match the original script. The server is designed to accept all training parameters from the client via the ZMQ message, but the defaults were also aligned for clarity.

| Feature / Parameter     | Original `TTT_server.py` | Our `TTT_server_mlx.py` | Analysis |
|-------------------------|---------------------------|--------------------------|----------|
| **LoRA Rank**           | Default: 32               | Default: 32              | ✅ Exact Match: The default matches, and the final value is controlled by the client. |
| **LoRA Alpha**          | Default: 64               | Default: 64              | ✅ Exact Match: The default matches, and the final value is controlled by the client. |
| **Fine-Tuning Epochs**  | Default: 10               | Default: 10              | ✅ Exact Match: The default matches, and the final value is controlled by the client. |
| **Learning Rate**       | Default: 1e-3             | Default: 1e-3            | ✅ Exact Match: The default matches, and the final value is controlled by the client. |
| **Gradient Accumulation** | Has `gradient_accumulation_steps` parameter. | Has `gradient_accumulation_steps` parameter. | ✅ Exact Match: The server now accepts this argument directly, just like the original implementation. |
| **Adapter Saving**      | Saves adapter to disk, controlled by `--keep_adapter_dir`. | Operates fully in-memory (`adapter_file=None`). `--keep_adapter_dir` has no effect. | ⚠️ Deliberate Difference: Our in-memory approach is faster and simpler. The ability to inspect saved adapters is sacrificed for efficiency, which is a reasonable trade-off. |
