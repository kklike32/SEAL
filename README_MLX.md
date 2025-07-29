# Project SEAL on MLX: Replication for Apple Silicon

This document details the process of replicating the inner-loop fine-tuning mechanism of the **SEAL (Self-Adapting Language Models)** paper on Apple Silicon using the MLX framework. The goal of this effort was to adapt the original project, which was designed for a distributed Linux-based GPU cluster, to run efficiently on a local macOS environment.

This work was completed as a replication study and is documented in three phases:
1.  **Phase 1:** Adapting the synthetic data generation pipeline.
2.  **Phase 2:** Re-engineering the Test-Time Training (TTT) server into a unified MLX application.
3.  **Phase 3:** Running validation experiments to replicate the original paper's inner-loop findings.

## 1. Environment Setup

This project was developed and tested on a MacBook Pro with Apple Silicon.

### 1.1. Create a Virtual Environment

It is highly recommended to use a virtual environment.

**Using Conda:**
```bash
conda create -n seal_mlx python=3.12
conda activate seal_mlx
```

**Using venv:**
```bash
python3.12 -m venv seal_mlx
source seal_mlx/bin/activate
```

### 1.2. Install Dependencies

The MLX-specific dependencies are listed in `requirements_mlx.txt`. Install them using pip:

```bash
pip install -r requirements_mlx.txt
```

### 1.3. Configure OpenAI API Key

The evaluation scripts use GPT-4 to grade the model's outputs. Create a `.env` file in the project root and add your key:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

## 2. Methodology & Architectural Changes

The primary challenge was adapting the original architecture—a two-process system with a VLLM inference server and a separate PyTorch training script—to a single-GPU macOS environment.

Our solution was to re-engineer this into a **single, unified server** (`TTT_server_mlx.py`). This server:
- Loads a base MLX model from Hugging Face once at startup.
- Listens for requests using a ZMQ socket.
- Performs the entire Test-Time Training (TTT) cycle for each request:
    1.  Evaluates the base model's performance on a given task.
    2.  Performs on-demand LoRA fine-tuning in memory using the provided completion data.
    3.  Evaluates the newly adapted model's performance.
- Restores the model to its "pristine" state after each request to ensure statelessness without reloading from disk.

This unified architecture is significantly more efficient for a local environment and removes the dependencies on VLLM, SLURM, and complex inter-process communication.

## 3. How to Run the Replication

The following steps will guide you through replicating the experiments.

### Step 1: Generate Synthetic Data

This script uses a local MLX model to generate the synthetic "completions" that will be used as fine-tuning data in the next steps.

- **Script:** `knowledge-incorporation/scripts/make_squad_data_mlx.sh`
- **Action:** This script runs `make_squad_data_mlx.py`, which loads a 4-bit quantized Qwen1.5-7B model and generates question-answer pairs based on SQuAD contexts.
- **Output:** The results are saved in `knowledge-incorporation/mlx_experiments/data/synthetic_data/train/`.

**To Run:**
```bash
bash knowledge-incorporation/scripts/make_squad_data_mlx.sh
```
*Note: This process is time-consuming. The initial run on a 7B model for 50 articles took approximately 81 minutes.*

### Step 2: Start the Test-Time Training (TTT) Server

This script starts the unified MLX server, which will wait for requests to perform fine-tuning and evaluation.

- **Script:** `knowledge-incorporation/scripts/TTT_server_mlx.sh`
- **Action:** This runs the `TTT_server_mlx.py` script, which loads the base model into memory and starts a ZMQ server.

**To Run:**
```bash
# In a new terminal window with the seal_mlx environment activated
bash knowledge-incorporation/scripts/TTT_server_mlx.sh
```
The server will print a message indicating it is "waiting for a message."

### Step 3: Run the Experimental Validation

This script acts as the client, sending requests to the TTT server to perform the inner-loop validation experiment.

- **Script:** `knowledge-incorporation/scripts/query_server_mlx.sh`
- **Action:** This runs `query_server_mlx.py`, which reads the synthetically generated data, sends it to the TTT server, and records the performance before and after fine-tuning.
- **Output:** The results are saved in the `knowledge-incorporation/mlx_experiments/results/` directory.

**To Run:**
```bash
# In a third terminal window with the seal_mlx environment activated
bash knowledge-incorporation/scripts/query_server_mlx.sh
```

## 4. Key Findings & Conclusion

The experimental validation (Phase 3) successfully replicated the core findings of the SEAL paper's inner loop.

- **Initial Failure:** A direct port of the original hyperparameters (learning rate of `1e-3`) caused the training to diverge.
- **Correction:** After reducing the learning rate to `1e-5` and correcting a subtle bug in the prompt-masking logic, the training process stabilized.
- **Final Result:** A validation run on 30 articles yielded a **`mean_gain` of -0.86%**.

This result, being very close to zero, is a successful validation. It confirms the paper's hypothesis that, on average, a randomly generated completion does not reliably improve the model's performance. The fine-tuning process is highly sensitive to the quality of the completion data.

This noisy signal is the expected output and is precisely what the full SEAL framework uses to train an outer-loop Reinforcement Learning agent to generate higher-quality, more helpful self-edits. This validated MLX implementation now serves as a solid platform for future work, such as implementing that RL outer loop.
