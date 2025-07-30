# knowledge-incorporation/src/rl/ppo_mlx.py
import mlx.core as mx
import mlx.nn as nn
from mlx.optimizers import Adam
import numpy as np

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

        # Correctly define the value and gradient function.
        # It computes gradients of _loss_fn w.r.t. argnums 0 and 1 (actor_model, critic_model).
        self.grad_fn = mx.value_and_grad(self._loss_fn, argnums=(0, 1))
        self.grad_fn = mx.compile(self.grad_fn)

    def _pad_and_stack_tensors(self, tensors, pad_value):
        """Pads a list of tensors to the same length and concatenates them."""
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
        return mx.concatenate(padded_list, axis=0)

    def _loss_fn(self, actor_model, critic_model, prompt_batch, response_batch, advantages_batch, returns_batch, old_log_probs_batch, attention_mask):
        """
        A pure function that computes the loss.
        This is the function that will be differentiated.
        """
        # --- Actor Loss (Policy Gradient) ---
        # Compute new log probabilities for the response tokens
        # We need to pass the full sequence (prompt + response) to get proper log probs
        full_sequences = mx.concatenate([prompt_batch, response_batch], axis=-1)
        logits = actor_model(full_sequences)
        log_probs_all = mx.log_softmax(logits, axis=-1)
        
        # Extract log probs for response tokens only
        prompt_length = prompt_batch.shape[-1]
        response_length = response_batch.shape[-1]
        
        # The logits for response tokens start at prompt_length - 1 (because of causal modeling)
        response_logits = log_probs_all[:, prompt_length-1:prompt_length-1+response_length, :]
        
        # Gather log probabilities for actual response tokens
        response_batch_expanded = mx.expand_dims(response_batch, -1)
        gathered_log_probs = mx.take_along_axis(response_logits, response_batch_expanded, axis=-1).squeeze(-1)
        
        # Sum log probabilities for the entire response sequence (with attention mask)
        new_log_probs = mx.sum(gathered_log_probs * attention_mask, axis=-1)

        # --- Critic Loss (Value Function) ---
        values = critic_model(prompt_batch).squeeze(-1)

        # --- PPO Objective ---
        ratio = mx.exp(new_log_probs - old_log_probs_batch)
        policy_loss_1 = advantages_batch * ratio
        policy_loss_2 = advantages_batch * mx.clip(ratio, 1.0 - self.config.cliprange, 1.0 + self.config.cliprange)
        policy_loss = -mx.mean(mx.minimum(policy_loss_1, policy_loss_2))
        
        value_loss = mx.mean((returns_batch - values) ** 2)
        total_loss = policy_loss + self.config.vf_coef * value_loss

        # value_and_grad expects (loss, auxiliary_data)
        return total_loss, (policy_loss, value_loss)

    def learn(self, buffer_data):
        """
        The main learning loop. Trains the models on the data in the buffer.
        """
        states = buffer_data["states"]
        actions = buffer_data["actions"]
        old_log_probs = mx.array(buffer_data["log_probs"])
        advantages = mx.array(buffer_data["advantages"])
        returns = mx.array(buffer_data["returns"])

        indices = np.arange(len(states))

        for _ in range(self.config.num_ppo_epochs):
            np.random.shuffle(indices)
            for i in range(0, len(states), self.config.mini_batch_size):
                mini_batch_indices = indices[i : i + self.config.mini_batch_size]

                batch_prompts = [states[j] for j in mini_batch_indices]
                batch_responses = [actions[j] for j in mini_batch_indices]
                
                mini_batch_indices_mx = mx.array(mini_batch_indices)
                batch_advantages = advantages[mini_batch_indices_mx]
                batch_returns = returns[mini_batch_indices_mx]
                batch_old_log_probs = old_log_probs[mini_batch_indices_mx]

                prompt_tokens_list = [mx.array(self.tokenizer.encode(p)) for p in batch_prompts]
                response_tokens_list = [mx.array(self.tokenizer.encode(r)) for r in batch_responses]

                batch_prompt = self._pad_and_stack_tensors(prompt_tokens_list, self.tokenizer.pad_token_id)
                batch_response = self._pad_and_stack_tensors(response_tokens_list, self.tokenizer.pad_token_id)
                attention_mask = (batch_response != self.tokenizer.pad_token_id).astype(mx.float32)

                # Call the compiled gradient function
                ((loss, (p_loss, v_loss)),
                 (actor_grads, critic_grads)) = self.grad_fn(
                    self.actor_model, 
                    self.critic_model, 
                    batch_prompt, batch_response, 
                    batch_advantages, batch_returns, batch_old_log_probs,
                    attention_mask
                )

                # Update models using the computed gradients
                self.actor_optimizer.update(self.actor_model, actor_grads)
                self.critic_optimizer.update(self.critic_model, critic_grads)

                mx.eval(self.actor_model.parameters(), self.critic_model.parameters(), self.actor_optimizer.state, self.critic_optimizer.state)
        
        return {
            "ppo/loss": loss.item(),
            "ppo/policy_loss": p_loss.item(),
            "ppo/value_loss": v_loss.item()
        }
