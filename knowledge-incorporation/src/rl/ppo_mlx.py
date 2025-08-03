# knowledge-incorporation/src/rl/ppo_mlx.py
import logging
import mlx.core as mx
import mlx.nn as nn
from mlx.optimizers import Adam
import numpy as np
from trl import PPOConfig

class MLXPPO:
    """
    A production-level PPO trainer for MLX.
    """
    def __init__(self, actor, critic, tokenizer, ppo_config):
        # Store the RLActor container for generation, but the raw model for training
        self.actor_container = actor
        self.actor_model = actor.model
        self.critic_model = critic
        
        self.tokenizer = tokenizer
        self.config = ppo_config
        
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Initialize optimizers for the actual trainable models
        self.actor_optimizer = Adam(learning_rate=self.config.learning_rate)
        self.critic_optimizer = Adam(learning_rate=self.config.learning_rate)

        # We'll compute gradients in the learn method instead of using compiled functions
        # This is simpler and more reliable for MLX

    def _pad_and_stack_tensors(self, tensors, pad_value):
        """Pads a list of tensors to the same length and stacks them into a batch."""
        if not tensors:
            return mx.array([])
            
        max_len = max(t.shape[-1] for t in tensors)
        padded_list = []
        for t in tensors:
            padding_size = max_len - t.shape[-1]
            # Handle both 1D and 2D tensors
            if len(t.shape) == 1:
                padded_t = mx.pad(t, [(0, padding_size)], constant_values=pad_value)
            else:
                padded_t = mx.pad(t, [(0, 0), (0, padding_size)], constant_values=pad_value)
            padded_list.append(padded_t)
        # Stack to create batch dimension
        return mx.stack(padded_list, axis=0)

    def learn(self, buffer_data):
        """
        The main learning loop. Trains the models on the data in the buffer.
        """
        # Extract data from the dictionary returned by PPOBuffer.get()
        prompts = buffer_data['states']  # states are prompts in our case
        responses = buffer_data['actions']  # actions are responses in our case
        rewards = buffer_data['rewards']
        values = buffer_data['values']
        log_probs = buffer_data['log_probs']
        returns = buffer_data['returns']
        advantages = buffer_data['advantages']

        logging.info(f"Learning phase: Processing {len(prompts)} samples...")

        # Add validation
        if len(prompts) == 0:
            logging.warning("No prompts to process!")
            return {"policy_loss": 0.0, "value_loss": 0.0, "total_loss": 0.0}
        
        # Log advantage statistics for debugging
        adv_mean = float(mx.mean(advantages))
        adv_std = float(mx.std(advantages))
        logging.info(f"Advantage stats - Mean: {adv_mean:.4f}, Std: {adv_std:.4f}")
        
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "total_loss": 0.0, "kl_div": 0.0}

        for epoch in range(self.config.num_ppo_epochs):
            # Convert prompts and responses to token lists
            prompt_tokens_list = [self.tokenizer.encode(p) for p in prompts]
            response_tokens_list = [self.tokenizer.encode(r) for r in responses]
            
            # Create batches
            for i in range(0, len(prompts), self.config.mini_batch_size):
                end_idx = min(i + self.config.mini_batch_size, len(prompts))
                
                # Get batch data
                batch_prompt_tokens = prompt_tokens_list[i:end_idx]
                batch_response_tokens = response_tokens_list[i:end_idx]
                batch_advantages = advantages[i:end_idx]  # advantages is already an MLX array
                batch_returns = returns[i:end_idx]  # returns is already an MLX array
                batch_old_log_probs = log_probs[i:end_idx]  # log_probs is already an MLX array

                # Pad and stack tensors
                batch_prompt = self._pad_and_stack_tensors([mx.array(t) for t in batch_prompt_tokens], self.tokenizer.pad_token_id)
                batch_response = self._pad_and_stack_tensors([mx.array(t) for t in batch_response_tokens], self.tokenizer.pad_token_id)
                attention_mask = (batch_response != self.tokenizer.pad_token_id).astype(mx.float32)

                # Compute losses and update models using the correct MLX pattern
                
                # Actor update - create loss function that takes model and data
                def actor_loss_fn(model, prompt_batch, response_batch, advantages_batch, old_log_probs_batch, attention_mask):
                    # Compute new log probabilities
                    full_sequences = mx.concatenate([prompt_batch, response_batch], axis=-1)
                    logits = model(full_sequences)
                    log_probs_all = nn.log_softmax(logits, axis=-1)
                    
                    # Extract log probs for response tokens
                    # For causal LM, logits[i] predicts token[i+1], so:
                    # - logits[prompt_length-1] predicts response[0]
                    # - logits[prompt_length+j-1] predicts response[j]
                    prompt_length = prompt_batch.shape[-1]
                    response_length = response_batch.shape[-1]
                    response_log_probs = log_probs_all[:, prompt_length-1:prompt_length-1+response_length, :]
                    
                    # Gather log probs for actual response tokens
                    gathered_log_probs = mx.take_along_axis(
                        response_log_probs,
                        mx.expand_dims(response_batch, -1),
                        axis=-1
                    ).squeeze(-1)
                    
                    new_log_probs = mx.sum(gathered_log_probs * attention_mask, axis=-1)
                    
                    # Calculate KL divergence for monitoring
                    kl_div = mx.mean(old_log_probs_batch - new_log_probs)
                    
                    # PPO loss
                    ratio = mx.exp(new_log_probs - old_log_probs_batch)
                    policy_loss_1 = advantages_batch * ratio
                    policy_loss_2 = advantages_batch * mx.clip(ratio, 1.0 - self.config.cliprange, 1.0 + self.config.cliprange)
                    policy_loss = -mx.mean(mx.minimum(policy_loss_1, policy_loss_2))
                    
                    return policy_loss, kl_div

                # Critic update - create loss function that takes model and data  
                def critic_loss_fn(model, prompt_batch, returns_batch):
                    values = model(prompt_batch).squeeze(-1)
                    value_loss = mx.mean((returns_batch - values) ** 2)
                    return value_loss

                # Create gradient functions using MLX pattern
                actor_loss_and_grad_fn = nn.value_and_grad(self.actor_model, actor_loss_fn)
                critic_loss_and_grad_fn = nn.value_and_grad(self.critic_model, critic_loss_fn)
                
                # Compute gradients and update
                (actor_loss, kl_div), actor_grads = actor_loss_and_grad_fn(self.actor_model, batch_prompt, batch_response, batch_advantages, batch_old_log_probs, attention_mask)
                critic_loss, critic_grads = critic_loss_and_grad_fn(self.critic_model, batch_prompt, batch_returns)
                
                # Ensure computations are evaluated (MLX lazy evaluation)
                mx.eval(actor_loss, kl_div, actor_grads, critic_loss, critic_grads)
                
                # Early stopping if KL divergence is too high (prevents catastrophic forgetting)
                kl_threshold = 0.01  # Conservative threshold
                if float(kl_div) > kl_threshold:
                    logging.warning(f"High KL divergence detected: {float(kl_div):.6f} > {kl_threshold}. Skipping update.")
                    continue
                
                # Gradient clipping for stability
                actor_grads = mx.clip(actor_grads, -1.0, 1.0)
                critic_grads = mx.clip(critic_grads, -1.0, 1.0)
                
                # Update models
                self.actor_optimizer.update(self.actor_model, actor_grads)
                self.critic_optimizer.update(self.critic_model, critic_grads)
                
                # Evaluate updated models
                mx.eval(self.actor_model, self.critic_model)

                # Track stats
                stats["policy_loss"] += float(actor_loss)
                stats["value_loss"] += float(critic_loss)
                stats["total_loss"] += float(actor_loss + critic_loss)
                if "kl_div" not in stats:
                    stats["kl_div"] = 0.0
                stats["kl_div"] += float(kl_div)

        # Average the stats
        num_updates = self.config.num_ppo_epochs * ((len(prompts) + self.config.mini_batch_size - 1) // self.config.mini_batch_size)
        for key in stats:
            stats[key] /= num_updates

        return stats

    def _compute_returns(self, rewards, values, gamma=0.99, lam=0.95):
        """
        Computes the returns using Generalized Advantage Estimation (GAE).
        """
        returns = []
        gae = 0
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[i + 1]
            delta = rewards[i] + gamma * next_value - values[i]
            gae = delta + gamma * lam * gae
            returns.insert(0, gae + values[i])
        return returns
