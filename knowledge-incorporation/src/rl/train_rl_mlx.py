import argparse
import mlx.core as mx
import mlx.nn as nn
import mlx_lm.core as lm_core
from transformers import AutoTokenizer
from trl import PPOConfig
import os
import logging

from knowledge-incorporation.src.rl.dataset import build_dataset, get_squad_prompts
from knowledge-incorporation.src.rl.reward import initialize_reward_client, get_reward, close_reward_client
from knowledge-incorporation.src.rl.ppo_mlx import MLXPPO
from knowledge-incorporation.src.rl.ppo_buffer import PPOBuffer

def parse_args():
    """
    Parses command-line arguments for the RL training script.
    """
    parser = argparse.ArgumentParser(description="Train a PPO agent to generate self-edits.")
    
    # Model and tokenizer arguments
    parser.add_argument("--model_id", type=str, default="mlx-community/Qwen1.5-7B-Chat-MLX-4bit", help="The base model ID for the Actor and Critic.")
    
    # Output and saving arguments
    parser.add_argument("--output_dir", type=str, default="mlx_experiments/rl_training", help="Directory to save models and logs.")
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
        self.model, self.tokenizer = lm_core.load(model_id)

    def generate(self, prompt: str, max_tokens: int = 100):
        """
        Generate a completion from a prompt.
        """
        return lm_core.generate(self.model, self.tokenizer, prompt, max_tokens=max_tokens)

class RLCritic(nn.Module):
    """
    A model with a value head to act as our Critic.
    """
    def __init__(self, model_id: str):
        super().__init__()
        self.model, _ = lm_core.load(model_id)
        
        # Add a value head
        self.value_head = nn.Linear(self.model.config.hidden_size, 1)

    def __call__(self, x):
        """
        Forward pass to get the value estimate.
        """
        hidden_states = self.model(x)
        value = self.value_head(hidden_states[:, -1, :])
        return value

def main():
    """
    The main function for the RL training loop.
    """
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(args.output_dir)
    
    initialize_reward_client(args.reward_port)
    
    ppo_config = PPOConfig(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        ppo_epochs=args.ppo_epochs,
        seed=args.seed,
        clip_param=args.clip_param,
        vf_coeff=args.vf_coeff
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
        for i in range(args.batch_size):
            prompt = prompts[step * args.batch_size + i]
            
            response = actor_model.generate(prompt)
            
            prompt_tensor = actor_model.tokenizer.encode(prompt, return_tensors="mx")
            response_tensor = actor_model.tokenizer.encode(response, return_tensors="mx")
            value = critic_model(prompt_tensor)
            log_prob = actor_model.model(response_tensor).log_softmax(axis=-1)

            reward = get_reward(prompt, response)
            
            ppo_buffer.add(prompt, response, reward, value.item(), log_prob.item())

        ppo_buffer.finish_path()

        # --- Learning Phase ---
        logging.info("  Rollout complete. Starting learning phase...")
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