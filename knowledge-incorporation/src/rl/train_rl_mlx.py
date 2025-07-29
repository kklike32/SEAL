# knowledge-incorporation/src/rl/train_rl_mlx.py
import sys
import os
import argparse
import logging
import time

# Add knowledge-incorporation directory to path for cleaner imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load as mlx_lm_load, generate as mlx_lm_generate
from transformers import AutoTokenizer
from trl import PPOConfig

# Import using relative imports from the src directory
from src.rl.dataset import build_dataset, get_squad_prompts
from src.rl.reward import initialize_reward_client, get_reward, close_reward_client
from src.rl.ppo_mlx import MLXPPO
from src.rl.ppo_buffer import PPOBuffer

def parse_args():
    """
    Parses command-line arguments for the RL training script.
    """
    parser = argparse.ArgumentParser(description="Train a PPO agent to generate self-edits.")
    
    # Model and tokenizer arguments
    parser.add_argument("--model_id", type=str, default="mlx-community/Qwen1.5-7B-Chat-MLX-4bit", help="The base model ID for the Actor and Critic.")
    
    # Output and saving arguments
    parser.add_argument("--output_dir", type=str, default="logs/rl_training", help="Directory to save models and logs.")
    parser.add_argument("--save_every", type=int, default=5, help="Save a checkpoint every N PPO steps.")

    # PPO configuration arguments
    parser.add_argument("--learning_rate", type=float, default=1.41e-5, help="Learning rate for the PPO agent.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for PPO training (rollout buffer size).")
    parser.add_argument("--mini_batch_size", type=int, default=64, help="Mini-batch size for PPO training.")
    parser.add_argument("--ppo_epochs", type=int, default=4, help="Number of PPO epochs per rollout.")
    parser.add_argument("--clip_param", type=float, default=0.2, help="PPO clipping parameter.")
    parser.add_argument("--vf_coeff", type=float, default=0.5, help="Value function coefficient in the PPO loss.")
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--reward_port", type=int, default=5555, help="Port for the reward server.")
    parser.add_argument("--total_ppo_steps", type=int, default=20, help="Total number of PPO steps (rollouts).")

    return parser.parse_args()

def setup_logging(output_dir):
    """
    Sets up logging to both console and a file.
    """
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] - %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # File handler
    file_handler = logging.FileHandler(os.path.join(output_dir, "training_log.txt"))
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)

def save_models(actor, critic, path):
    """
    Saves the actor and critic model weights.
    """
    logging.info(f"Saving models to {path}...")
    os.makedirs(path, exist_ok=True)
    actor.model.save_weights(os.path.join(path, "actor.safetensors"))
    critic.model.save_weights(os.path.join(path, "critic.safetensors"))

class RLActor(nn.Module):
    """
    A LoRA-finetuned model that will be our Actor in the PPO setup.
    """
    def __init__(self, model_id: str):
        super().__init__()
        self.model, self.tokenizer = mlx_lm_load(model_id)

    def generate(self, prompt: str, max_tokens: int = 1000):
        """
        Generate a completion from a prompt using mlx_lm's generate function.
        """
        return mlx_lm_generate(self.model, self.tokenizer, prompt, max_tokens=max_tokens)
    
    def compute_log_probs(self, input_ids, response_ids):
        """
        Compute log probabilities for the response tokens given the input.
        """
        # Ensure inputs are MLX arrays
        if not isinstance(input_ids, mx.array):
            input_ids = mx.array(input_ids)
        if not isinstance(response_ids, mx.array):
            response_ids = mx.array(response_ids)
            
        # Ensure proper shape (batch_size, sequence_length)
        if input_ids.ndim == 1:
            input_ids = input_ids.reshape(1, -1)
        if response_ids.ndim == 1:
            response_ids = response_ids.reshape(1, -1)
        
        # Concatenate prompt and response
        full_sequence = mx.concatenate([input_ids, response_ids], axis=-1)
        
        # Get logits for the full sequence
        logits = self.model(full_sequence)
        
        # Convert to log probabilities
        log_probs = mx.log_softmax(logits, axis=-1)
        
        # Extract log probabilities for the response tokens
        # We want log_probs for positions [len(input_ids):len(input_ids)+len(response_ids)]
        response_start = input_ids.shape[-1]
        response_end = response_start + response_ids.shape[-1]
        
        # Get the log probs for response tokens (shifted by 1 for next token prediction)
        response_log_probs = log_probs[:, response_start-1:response_end-1, :]
        
        # Gather the log probabilities for the actual response tokens
        response_token_log_probs = mx.take_along_axis(
            response_log_probs, 
            response_ids.reshape(1, -1, 1), 
            axis=-1
        ).squeeze(-1)
        
        # Return mean log probability
        return mx.mean(response_token_log_probs)

class RLCritic(nn.Module):
    """
    A model with a value head to act as our Critic.
    """
    def __init__(self, model_id: str):
        super().__init__()
        self.model, _ = mlx_lm_load(model_id)
        
        # Determine the hidden_size from the model's architecture
        # For Llama models, we need to find the correct hidden dimension
        hidden_size = None
        
        # Try multiple ways to get the hidden size
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            # For models like LlamaForCausalLM
            if len(self.model.model.layers) > 0:
                last_layer = self.model.model.layers[-1]
                if hasattr(last_layer, 'mlp') and hasattr(last_layer.mlp, 'down_proj'):
                    hidden_size = last_layer.mlp.down_proj.weight.shape[0]
                elif hasattr(last_layer, 'self_attn') and hasattr(last_layer.self_attn, 'o_proj'):
                    hidden_size = last_layer.self_attn.o_proj.weight.shape[0]
        
        # Fallback: try to get from embedding layer
        if hidden_size is None:
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'embed_tokens'):
                hidden_size = self.model.model.embed_tokens.weight.shape[-1]
        
        # Last resort: use common Llama-3-8B size
        if hidden_size is None:
            hidden_size = 4096
            print(f"Warning: Could not determine hidden size, using default: {hidden_size}")
        else:
            print(f"Determined hidden_size: {hidden_size}")
        
        self.hidden_size = hidden_size
        self.value_head = nn.Linear(hidden_size, 1)

    def __call__(self, x):
        """
        Forward pass to get the value estimate.
        """
        # The key issue: we need hidden states, not logits
        # The model(x) call typically returns logits for language models
        
        if hasattr(self.model, 'model'):
            # For models with .model attribute (like LlamaForCausalLM)
            # Get the base model outputs (hidden states)
            hidden_states = self.model.model(x)
            # hidden_states might be a tuple, extract the actual states
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]
        else:
            # Fallback: direct model call (this might give logits)
            outputs = self.model(x)
            # If this is logits, we need to handle it differently
            if outputs.shape[-1] > self.hidden_size:
                # This is likely logits (vocab_size dimension), not hidden states
                # We need to get the hidden states instead
                raise ValueError(f"Model output shape {outputs.shape} suggests logits, not hidden states. "
                               f"Need to access model.model instead of model directly.")
            hidden_states = outputs
        
        # Take the last token's hidden state
        last_hidden = hidden_states[:, -1, :]
        
        # Verify dimensions match
        if last_hidden.shape[-1] != self.hidden_size:
            raise ValueError(f"Hidden state dimension {last_hidden.shape[-1]} doesn't match "
                           f"expected dimension {self.hidden_size}")
        
        value = self.value_head(last_hidden)
        return value

def main():
    """
    The main function for the RL training loop.
    """
    
    print(f"MLX is using default device: {mx.default_device()}")
    
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(args.output_dir)
    
    initialize_reward_client(args.reward_port)
    
    ppo_config = PPOConfig(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        num_ppo_epochs=args.ppo_epochs,
        seed=args.seed,
        cliprange=args.clip_param,
        vf_coef=args.vf_coeff
    )

    actor_model = RLActor(args.model_id)
    critic_model = RLCritic(args.model_id)
    
    dataset = build_dataset()
    prompts = get_squad_prompts(dataset, num_samples=args.batch_size * args.total_ppo_steps)
    ppo_buffer = PPOBuffer(buffer_size=args.batch_size)

    ppo_trainer = MLXPPO(actor_model, critic_model, actor_model.tokenizer, ppo_config)

    logging.info("RL Training Script - Production Setup")
    logging.info("------------------------------------")

    for step in range(args.total_ppo_steps):
        logging.info(f"--- PPO Step {step + 1}/{args.total_ppo_steps} ---")
        
        # --- Rollout Phase ---
        rollout_start_time = time.time()
        for i in range(args.batch_size):
            sample_start_time = time.time()
            prompt = prompts[step * args.batch_size + i]
            
            response = actor_model.generate(prompt)
            
            # Encode the text to tensors - handle different tokenizer return formats
            prompt_encoded = actor_model.tokenizer.encode(prompt)
            response_encoded = actor_model.tokenizer.encode(response)
            
            # Convert to MLX arrays with proper shape handling
            if isinstance(prompt_encoded, list):
                prompt_tensor = mx.array(prompt_encoded).reshape(1, -1)
            elif isinstance(prompt_encoded, mx.array):
                prompt_tensor = prompt_encoded.reshape(1, -1) if prompt_encoded.ndim == 1 else prompt_encoded
            else:
                # Handle other formats (e.g., torch tensors, numpy arrays)
                prompt_tensor = mx.array(prompt_encoded).reshape(1, -1)
                
            if isinstance(response_encoded, list):
                response_tensor = mx.array(response_encoded).reshape(1, -1)
            elif isinstance(response_encoded, mx.array):
                response_tensor = response_encoded.reshape(1, -1) if response_encoded.ndim == 1 else response_encoded
            else:
                # Handle other formats
                response_tensor = mx.array(response_encoded).reshape(1, -1)
            
            # Get value from critic
            value = critic_model(prompt_tensor)
            
            # Compute log probabilities properly using the actor's method
            log_prob = actor_model.compute_log_probs(prompt_tensor, response_tensor)

            reward = get_reward(prompt, response)
            
            # Convert scalar values properly
            value_scalar = float(value.item()) if hasattr(value, 'item') else float(value)
            log_prob_scalar = float(log_prob.item()) if hasattr(log_prob, 'item') else float(log_prob)
            
            ppo_buffer.add(prompt, response, reward, value_scalar, log_prob_scalar)
            
            sample_time = time.time() - sample_start_time
            if (i + 1) % 10 == 0:  # Log every 10 samples
                logging.info(f"    Sample {i+1}/{args.batch_size} complete. Time: {sample_time:.1f}s")
        
        rollout_time = time.time() - rollout_start_time
        logging.info(f"  Rollout complete in {rollout_time:.1f}s. Starting learning phase...")

        ppo_buffer.finish_path()

        # --- Learning Phase ---
        buffer_data = ppo_buffer.get()
        stats = ppo_trainer.learn(buffer_data)
        logging.info(f"  Learning complete. Stats: {stats}")

        # --- Checkpointing ---
        if (step + 1) % args.save_every == 0:
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint_step_{step + 1}")
            save_models(actor_model, critic_model, checkpoint_path)

    final_model_path = os.path.join(args.output_dir, "final_model")
    save_models(actor_model, critic_model, final_model_path)

    close_reward_client()
    logging.info("Training complete.")

if __name__ == "__main__":
    main()
