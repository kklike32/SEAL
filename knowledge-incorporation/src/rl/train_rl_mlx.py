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
    parser.add_argument("--model_id", type=str, default="mlx-community/Meta-Llama-3-8B-Instruct", help="The base model ID for the Actor and Critic.")
    
    # Output and saving arguments
    parser.add_argument("--output_dir", type=str, default="knowledge-incorporation/logs/rl_training", help="Directory to save models and logs.")
    parser.add_argument("--save_every", type=int, default=2, help="Save a checkpoint every N PPO steps.")

    # PPO configuration arguments
    parser.add_argument("--learning_rate", type=float, default=1.41e-5, help="Learning rate for the PPO agent.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for PPO training (rollout buffer size).")
    parser.add_argument("--mini_batch_size", type=int, default=4, help="Mini-batch size for PPO training.")
    parser.add_argument("--ppo_epochs", type=int, default=2, help="Number of PPO epochs per rollout.")
    parser.add_argument("--clip_param", type=float, default=0.2, help="PPO clipping parameter.")
    parser.add_argument("--vf_coeff", type=float, default=0.5, help="Value function coefficient in the PPO loss.")
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--reward_port", type=int, default=5555, help="Port for the reward server.")
    parser.add_argument("--total_ppo_steps", type=int, default=3, help="Total number of PPO steps (rollouts).")

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

    def generate(self, prompt: str, max_tokens: int = 125):
        """
        Generate a completion from a prompt using mlx_lm's generate function.
        """
        return mlx_lm_generate(self.model, self.tokenizer, prompt, max_tokens=max_tokens)
    
    def compute_log_probs(self, input_ids, response_ids):
        """
        Compute log probabilities for the response tokens given the input.
        """
        # Ensure inputs are MLX arrays
        if input_ids.ndim == 1: input_ids = mx.expand_dims(input_ids, 0)
        if response_ids.ndim == 1: response_ids = mx.expand_dims(response_ids, 0)

        full_sequence = mx.concatenate([input_ids, response_ids], axis=-1)
        logits = self.model(full_sequence)
        log_probs = nn.log_softmax(logits, axis=-1)

        # We want the log probs for the response part, which are predicted from
        # the sequence up to that point.
        response_start_idx = input_ids.shape[-1]
        
        # The logits for the first response token are at index (response_start_idx - 1)
        response_logits_indices = slice(response_start_idx - 1, -1)
        response_log_probs = log_probs[:, response_logits_indices, :]

        # Gather the log probabilities for the actual response tokens
        gathered = mx.take_along_axis(
            response_log_probs,
            mx.expand_dims(response_ids, -1),
            axis=-1
        ).squeeze(-1)

        # Return the SUM of log probabilities for the sequence
        return mx.sum(gathered, axis=-1)

class RLCritic(nn.Module):
    """
    A model with a value head to act as our Critic.
    """
    def __init__(self, shared_model, shared_tokenizer):
        super().__init__()
        self.model = shared_model  # Reuse the actor's model
        self.tokenizer = shared_tokenizer
        
        # Determine the hidden_size from the shared model
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
        # For the critic, we only need a scalar value estimate from the prompt
        # We'll use the model's output logits and project to a value
        
        # Get the model's output (logits)
        outputs = self.model(x)
        
        # Take the last token's output and project to hidden size if needed
        last_token_output = outputs[:, -1, :]  # Shape: [batch_size, vocab_size]
        
        # If the output is logits (vocab_size), we need to project to hidden_size
        if last_token_output.shape[-1] != self.hidden_size:
            # Create a simple projection layer if it doesn't exist
            if not hasattr(self, 'projection'):
                self.projection = nn.Linear(last_token_output.shape[-1], self.hidden_size)
            last_hidden = self.projection(last_token_output)
        else:
            last_hidden = last_token_output
        
        # Project to value
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

    # Set it to the end-of-sentence token for padding consistency.
    if actor_model.tokenizer.pad_token_id is None:
        logging.info("Tokenizer missing pad_token_id. Setting to eos_token_id.")
        actor_model.tokenizer.pad_token_id = actor_model.tokenizer.eos_token_id

    critic_model = RLCritic(actor_model.model, actor_model.tokenizer)
    
    # Set random seed for reproducible dataset sampling
    import random
    random.seed(args.seed)
    
    dataset = build_dataset()
    prompts = get_squad_prompts(dataset, num_samples=args.batch_size * args.total_ppo_steps)
    ppo_buffer = PPOBuffer(buffer_size=args.batch_size)

    ppo_trainer = MLXPPO(actor_model, critic_model, actor_model.tokenizer, ppo_config)

    logging.info("RL Training Script - Production Setup")
    logging.info("------------------------------------")
    logging.info(f"Model ID: {args.model_id}")
    logging.info(f"Output Directory: {args.output_dir}")
    logging.info(f"Batch Size: {args.batch_size}")
    logging.info(f"Mini Batch Size: {args.mini_batch_size}")
    logging.info(f"Learning Rate: {args.learning_rate}")
    logging.info(f"Total PPO Steps: {args.total_ppo_steps}")
    logging.info(f"PPO Epochs per Step: {args.ppo_epochs}")
    logging.info(f"Max Tokens per Generation: 125")
    logging.info(f"Reward Server Port: {args.reward_port}")
    logging.info("------------------------------------")

    for step in range(args.total_ppo_steps):
        logging.info(f"--- PPO Step {step + 1}/{args.total_ppo_steps} ---")
        
        # --- Rollout Phase ---
        rollout_start_time = time.time()
        logging.info(f"  Starting rollout phase with {args.batch_size} samples...")
        
        rollout_rewards = []
        rollout_values = []
        rollout_log_probs = []
        
        for i in range(args.batch_size):
            sample_start_time = time.time()
            prompt = prompts[step * args.batch_size + i]
            
            logging.info(f"    Sample {i+1}: Generating response...")
            logging.info(f"    Sample {i+1}: Prompt: {prompt[:100]}...")  # Log first 100 chars of prompt
            generation_start = time.time()
            full_response = actor_model.generate(prompt)
            
            # Extract just the completion (action) by removing the prompt
            # mlx_lm_generate returns the full sequence including the prompt
            if full_response.startswith(prompt):
                completion = full_response[len(prompt):]
            else:
                # Fallback: assume the generated text is the completion
                completion = full_response
            
            generation_time = time.time() - generation_start
            logging.info(f"    Sample {i+1}: Generation complete in {generation_time:.2f}s (completion length: {len(completion)} chars)")
            logging.info(f"    Sample {i+1}: Generated completion: {completion[:200]}...")  # Log first 200 chars
            
            # The action is the completion, not the full response
            action = completion
            
            # Encode the text to tensors - handle different tokenizer return formats
            prompt_encoded = actor_model.tokenizer.encode(prompt)
            action_encoded = actor_model.tokenizer.encode(action)
            
            # Convert to MLX arrays with proper shape handling
            if isinstance(prompt_encoded, list):
                prompt_tensor = mx.array(prompt_encoded).reshape(1, -1)
            elif isinstance(prompt_encoded, mx.array):
                prompt_tensor = prompt_encoded.reshape(1, -1) if prompt_encoded.ndim == 1 else prompt_encoded
            else:
                # Handle other formats (e.g., torch tensors, numpy arrays)
                prompt_tensor = mx.array(prompt_encoded).reshape(1, -1)
                
            if isinstance(action_encoded, list):
                action_tensor = mx.array(action_encoded).reshape(1, -1)
            elif isinstance(action_encoded, mx.array):
                action_tensor = action_encoded.reshape(1, -1) if action_encoded.ndim == 1 else action_encoded
            else:
                # Handle other formats
                action_tensor = mx.array(action_encoded).reshape(1, -1)
            
            # Get value from critic
            critic_start = time.time()
            value = critic_model(prompt_tensor)
            critic_time = time.time() - critic_start
            
            # Compute log probabilities properly using the actor's method
            logprob_start = time.time()
            log_prob = actor_model.compute_log_probs(prompt_tensor, action_tensor)
            logprob_time = time.time() - logprob_start
            logging.info(f"    Sample {i+1}: Computing reward via TTT server...")
            reward_start = time.time()
            reward = get_reward(prompt, action)
            reward_time = time.time() - reward_start
            logging.info(f"    Sample {i+1}: Reward computation complete in {reward_time:.2f}s (reward: {reward:.4f})")
            
            # Convert scalar values properly
            value_scalar = float(value.item()) if hasattr(value, 'item') else float(value)
            log_prob_scalar = float(log_prob.item()) if hasattr(log_prob, 'item') else float(log_prob)
            
            rollout_rewards.append(reward)
            rollout_values.append(value_scalar)
            rollout_log_probs.append(log_prob_scalar)
            
            ppo_buffer.add(prompt, action, reward, value_scalar, log_prob_scalar)
            
            sample_time = time.time() - sample_start_time
            logging.info(f"    Sample {i+1}: Complete in {sample_time:.2f}s total "
                        f"(gen: {generation_time:.2f}s, critic: {critic_time:.3f}s, "
                        f"logprob: {logprob_time:.3f}s, reward: {reward_time:.2f}s)")
        
        rollout_time = time.time() - rollout_start_time
        avg_reward = sum(rollout_rewards) / len(rollout_rewards)
        avg_value = sum(rollout_values) / len(rollout_values)
        avg_log_prob = sum(rollout_log_probs) / len(rollout_log_probs)
        
        logging.info(f"  Rollout complete in {rollout_time:.1f}s")
        logging.info(f"  Rollout stats - Avg Reward: {avg_reward:.4f}, Avg Value: {avg_value:.4f}, Avg LogProb: {avg_log_prob:.4f}")
        logging.info(f"  Starting learning phase...")

        ppo_buffer.finish_path()

        # --- Learning Phase ---
        learning_start_time = time.time()
        buffer_data = ppo_buffer.get()
        logging.info(f"  Learning phase: Processing {len(buffer_data['rewards'])} samples...")
        stats = ppo_trainer.learn(buffer_data)
        learning_time = time.time() - learning_start_time
        logging.info(f"  Learning complete in {learning_time:.2f}s")
        logging.info(f"  Learning stats: {stats}")

        # --- Checkpointing ---
        if (step + 1) % args.save_every == 0:
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint_step_{step + 1}")
            logging.info(f"  Saving checkpoint to {checkpoint_path}")
            save_models(actor_model, critic_model, checkpoint_path)
            logging.info(f"  Checkpoint saved successfully")
        
        step_total_time = time.time() - rollout_start_time
        logging.info(f"--- PPO Step {step + 1} Complete in {step_total_time:.1f}s ---")
        logging.info("")

    final_model_path = os.path.join(args.output_dir, "final_model")
    logging.info(f"Saving final model to {final_model_path}")
    save_models(actor_model, critic_model, final_model_path)
    logging.info("Final model saved successfully")

    close_reward_client()
    logging.info("Reward client connection closed")
    logging.info("=== RL Training Complete ===")
    logging.info(f"Total training steps: {args.total_ppo_steps}")
    logging.info(f"Final model saved at: {final_model_path}")
    logging.info("=============================")

if __name__ == "__main__":
    main()
