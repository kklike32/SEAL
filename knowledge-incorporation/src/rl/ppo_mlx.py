# knowledge-incorporation/src/rl/ppo_mlx.py
import mlx.core as mlx
import mlx.nn as nn
from mlx.optimizers import Adam

import mlx.core as mlx
import mlx.nn as nn
from mlx.optimizers import Adam
import numpy as np

import mlx.core as mx
import mlx.nn as nn
from mlx.optimizers import Adam
import numpy as np

class MLXPPO:
    """
    A production-level PPO trainer for MLX.
    """
    def __init__(self, actor, critic, tokenizer, ppo_config):
        self.actor = actor
        self.critic = critic
        self.tokenizer = tokenizer
        self.config = ppo_config

        # Initialize optimizers
        self.actor_optimizer = Adam(learning_rate=self.config.learning_rate)
        self.critic_optimizer = Adam(learning_rate=self.config.learning_rate)

        # Compile the loss and gradient function for performance
        self.grad_fn = mx.compile(self._loss_and_grad_fn)

    def _loss_and_grad_fn(self, actor, critic, prompt_tokens, response_tokens, advantages, returns, old_log_probs):
        """
        A single function to compute the loss and gradients for both models.
        This will be JIT-compiled by MLX.
        """
        # Get new log probabilities and values
        log_probs = actor.model(response_tokens).log_softmax(axis=-1)
        values = critic(prompt_tokens)

        # Policy loss (actor)
        ratio = mx.exp(log_probs - old_log_probs)
        policy_loss_1 = advantages * ratio
        policy_loss_2 = advantages * mx.clip(ratio, 1.0 - self.config.clip_param, 1.0 + self.config.clip_param)
        policy_loss = -mx.mean(mx.minimum(policy_loss_1, policy_loss_2))

        # Value loss (critic)
        value_loss = mx.mean((returns - values) ** 2)

        # Total loss
        total_loss = policy_loss + self.config.vf_coeff * value_loss

        return total_loss, (policy_loss, value_loss)

    def learn(self, buffer_data):
        """
        The main learning loop. Trains the models on the data in the buffer.
        """
        # Extract data from the buffer
        states = buffer_data["states"]
        actions = buffer_data["actions"]
        old_log_probs = buffer_data["log_probs"]
        advantages = buffer_data["advantages"]
        returns = buffer_data["returns"]

        # Training loop
        for _ in range(self.config.num_ppo_epochs):
            # Create mini-batches
            indices = np.arange(len(states))
            np.random.shuffle(indices)
            for i in range(0, len(states), self.config.mini_batch_size):
                mini_batch_indices = indices[i : i + self.config.mini_batch_size]

                # Get mini-batch data
                batch_prompts = [states[j] for j in mini_batch_indices]
                batch_responses = [actions[j] for j in mini_batch_indices]
                batch_advantages = advantages[mini_batch_indices]
                batch_returns = returns[mini_batch_indices]
                batch_old_log_probs = old_log_probs[mini_batch_indices]

                # Tokenize the mini-batch
                prompt_tokens = [self.tokenizer.encode(p, return_tensors="mx") for p in batch_prompts]
                response_tokens = [self.tokenizer.encode(r, return_tensors="mx") for r in batch_responses]
                
                # For simplicity, we process one sample at a time in the mini-batch
                # A more advanced implementation would pad and batch these tensors.
                prompt_tensor = prompt_tokens[0]
                response_tensor = response_tokens[0]

                # Compute loss and gradients
                (loss, (p_loss, v_loss)), grads = self.grad_fn(
                    self.actor, self.critic, prompt_tensor, response_tensor, 
                    batch_advantages[0], batch_returns[0], batch_old_log_probs[0]
                )

                # Update models
                self.actor_optimizer.update(self.actor, grads[0])
                self.critic_optimizer.update(self.critic, grads[1])

                mx.eval(self.actor.parameters(), self.critic.parameters(), self.actor_optimizer.state, self.critic_optimizer.state)

        return {
            "ppo/loss": loss.item(),
            "ppo/policy_loss": p_loss.item(),
            "ppo/value_loss": v_loss.item()
        }
