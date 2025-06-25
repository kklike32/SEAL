# knowledge-incorporation/src/inner/TTT_server_mlx.py

# =================================================================================== #
#   Launches the unified MLX-based Test-Time Training (TTT) Server for SEAL.
#   This script starts a single process that listens for ZMQ requests and performs
#   both LoRA fine-tuning and evaluation on the Mac's GPU using MLX.
# =================================================================================== #

import argparse
import gc
import json
import logging
import os
import time
from typing import Dict

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils as mx_utils
import zmq
from datasets import Dataset
from mlx_lm.utils import load

# Import the primary evaluation function from our MLX utils file.
from ..utils_mlx import accuracy_and_texts_mlx

# --- Use the modern 'tuner' API ---
from mlx_lm.tuner import train, linear_to_lora_layers
from mlx_lm.tuner.trainer import TrainingArgs

# ---------------------------  CONFIG & LOGGING  --------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] TTT_server_mlx - %(message)s",
    datefmt="%H:%M:%S",
)
LOG = logging.getLogger(__name__)

# ---------------------------  SERVER STATE MANAGEMENT  -------------------- #

class ServerState:
    """Manages the MLX model and tokenizer to ensure a clean state for each request."""

    def __init__(self, model_id: str):
        LOG.info("Loading base model '%s'...", model_id)
        self._model_id = model_id
        self.model, self.tokenizer = load(self._model_id)
        
        # Use mx.array(p) to perform a deep copy a tensor.
        self._pristine_weights = mx_utils.tree_map(lambda p: mx.array(p), self.model.parameters())
        
        LOG.info("Base model loaded and pristine weights copied.")

    def restore_base_model(self):
        """Restores the model to its original, pre-fine-tuning state."""
        LOG.debug("Restoring model to pristine state.")
        flat_pristine_weights = mx_utils.tree_flatten(self._pristine_weights)
        self.model.load_weights(flat_pristine_weights)
        gc.collect()

# ---------------------------  MAIN SERVER LOGIC  -------------------------- #

def main():
    p = argparse.ArgumentParser(description="MLX-based Test-Time Training Server (Tuner API)")
    p.add_argument("--zmq_port", type=int, default=5555, help="ZMQ port to listen on.")
    p.add_argument("--model_id", type=str, required=True, help="Hugging Face repo ID of the MLX model.")
    p.add_argument("--instruct_model", action="store_true", help="Set if the model is an instruction-tuned variant.")
    p.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length for the training sequences.")
    p.add_argument("--eval_temperature", type=float, default=0.0, help="Sampling temperature for evaluation.")
    p.add_argument("--eval_top_p", type=float, default=1.0, help="Eval nucleus sampling (top-p)")
    p.add_argument("--eval_max_tokens", type=int, default=64, help="Maximum tokens for evaluation generation.")
    p.add_argument("--keep_adapter_dir",  action="store_true",
                   help="Skip tmp-dir deletion so outer driver can reuse the LoRA. This causes high disk usage and is only used in continual_self_edits.py or for debugging.")
    
    args = p.parse_args()

    if "OPENAI_API_KEY" not in os.environ:
        LOG.warning("OPENAI_API_KEY environment variable not set. Grading may fail.")

    state = ServerState(args.model_id)
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://*:{args.zmq_port}")
    LOG.info("ZMQ listening for requests at tcp://*:%d", args.zmq_port)

    step = 0
    while True:
        try:
            LOG.info("Waiting for new request...")
            msg = sock.recv_json()
            LOG.info("Received request with keys: %s", list(msg.keys()))

            if msg.get("cmd") == "shutdown":
                sock.send_json({"status": "ok", "message": "Shutting down."})
                break

            recv_start = time.time()
            
            # --- Extract parameters from the request ---
            train_sequences = msg.get("train_sequences", [])
            questions = msg.get("eval_questions", [])
            lora_rank = msg.get("lora_rank", 32)
            lora_layers = msg.get("lora_layers", 16)
            lora_alpha = msg.get("lora_alpha", 64)
            lora_dropout = msg.get("lora_dropout", 0.0)
            finetune_epochs = msg.get("finetune_epochs", 10)
            finetune_lr = msg.get("finetune_lr", 1e-3)
            batch_size = msg.get("batch_size", 1)
            end_mask_substring = msg.get("end_mask_substring", "")
            skip_training = bool(msg.get("skip_training", False))

            sampling_cfg = {"max_tokens": args.eval_max_tokens, 
                            "temperature": args.eval_temperature, 
                            "top_p": args.eval_top_p}

            # --- 1. Baseline Evaluation ---
            LOG.info("Step %d: Performing baseline evaluation...", step)
            base_acc, base_texts, base_ok = accuracy_and_texts_mlx(
                state.model, 
                state.tokenizer, 
                questions, 
                sampling_cfg, 
                args.instruct_model
            )
            LOG.info("Step %d: Baseline accuracy: %.3f", step, base_acc)

            if skip_training or not train_sequences:
                reply = { 
                    "baseline_accuracy": base_acc, 
                    "adapter_accuracy": base_acc, 
                    "adapter_gain": 0.0, 
                    "baseline_texts": base_texts, 
                    "adapter_texts": base_texts, 
                    "baseline_correct": base_ok, 
                    "adapter_correct": base_ok, 
                    "gains": [0] * len(base_ok)
                }
                sock.send_json(reply)
                LOG.info("Step %d: Finished (training skipped). Took %.2fs", step, time.time() - recv_start)
                step += 1
                continue

            # --- 2. LoRA Fine-Tuning (using the 'tuner' API) ---
            LOG.info("Step %d: Starting LoRA fine-tuning with the 'tuner' API...", step)
            
            state.model.freeze()
            lora_config = {
                "rank": lora_rank, 
                "alpha": lora_alpha, 
                "dropout": lora_dropout, 
                "scale": 10.0}
            linear_to_lora_layers(state.model, lora_layers, lora_config)

            if not end_mask_substring:
                raise ValueError("'end_mask_substring' is required for training.")
            sub_ids = state.tokenizer.encode(end_mask_substring)
            
            processed_data = []
            for seq in train_sequences:
                tokens = state.tokenizer.encode(seq)
                labels = list(tokens)
                
                # Find the mask position
                M = len(sub_ids)
                mask_pos = -1
                for i in range(len(labels) - M + 1):
                    if labels[i : i + M] == sub_ids:
                        mask_pos = i + M
                        break
                
                # Apply mask
                if mask_pos != -1:
                    labels[:mask_pos] = [-100] * mask_pos
                
                processed_data.append({"input_ids": tokens, "labels": labels})

            train_dataset = Dataset.from_list(processed_data)

            total_iters = (len(train_dataset) // batch_size) * finetune_epochs
            
            training_args = TrainingArgs(
                iters=total_iters,
                batch_size=batch_size,
                steps_per_report=min(10, total_iters if total_iters > 0 else 1),
                adapter_file=None, # Do not save adapters to disk
            )
            
            # The model is trained in-place
            train(
                model=state.model,
                tokenizer=state.tokenizer,
                args=training_args,
                optimizer=optim.Adam(learning_rate=finetune_lr),
                train_dataset=train_dataset,
                val_dataset=None,
            )

            # --- 3. Adapter Evaluation ---
            LOG.info("Step %d: Performing evaluation with adapter...", step)
            adapter_acc, adapter_texts, adapter_ok = accuracy_and_texts_mlx(
                state.model, state.tokenizer, questions, sampling_cfg, args.instruct_model
            )
            LOG.info("Step %d: Adapter accuracy: %.3f", step, adapter_acc)
            
            gains = [
                1 if a and not b else 
                -1 if b and not a else 
                0 
                for b, a in zip(base_ok, adapter_ok)]
            reply = { 
                "baseline_accuracy": round(base_acc, 4), 
                "adapter_accuracy": round(adapter_acc, 4), 
                "adapter_gain": round(adapter_acc - base_acc, 4), 
                "baseline_texts": base_texts, 
                "adapter_texts": adapter_texts, 
                "baseline_correct": base_ok, 
                "adapter_correct": adapter_ok, 
                "gains": gains
            }
            sock.send_json(reply)

        except (KeyboardInterrupt, SystemExit):
            LOG.info("Shutdown signal received.")
            break

        except Exception as e:
            LOG.exception("An error occurred while processing the request.")
            reply = {"error": f"{type(e).__name__}: {str(e)}"}
            if sock.getsockopt(zmq.EVENTS) & zmq.POLLOUT:
                sock.send_json(reply)
        finally:
            # --- 4. Restore Model State for Next Request ---
            state.restore_base_model()
            LOG.info("Step %d complete. Model restored to base state.", step)
            step += 1

    # --- Cleanup ---
    sock.close()
    ctx.term()
    LOG.info("Server shut down.")

if __name__ == "__main__":
    main()
