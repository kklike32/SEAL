# knowledge-incorporation/src/rl/ppo_buffer.py
import numpy as np
import mlx.core as mx

class PPOBuffer:
    """
    A buffer for storing trajectories and calculating GAE for PPO.
    """
    def __init__(self, buffer_size, gamma=0.99, gae_lambda=0.95):
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        # Initialize buffers
        self.states = [None] * buffer_size
        self.actions = [None] * buffer_size
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)

        self.ptr = 0
        self.path_start_idx = 0

    def add(self, state, action, reward, value, log_prob):
        """
        Add a new experience to the buffer.
        """
        if self.ptr >= self.buffer_size:
            raise ValueError("Buffer is full. Call finish_path() and get() before adding more.")

        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Calculate advantages and returns for the latest trajectory.
        This should be called at the end of a rollout or when the buffer is full.
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rewards = np.append(self.rewards[path_slice], last_val)
        values = np.append(self.values[path_slice], last_val)

        # GAE-Lambda advantage calculation
        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
        self.advantages[path_slice] = self._discount_cumsum(deltas, self.gamma * self.gae_lambda)

        # Rewards-to-go calculation
        self.returns[path_slice] = self._discount_cumsum(rewards, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        """
        Get all data from the buffer and reset it.
        """
        if self.ptr != self.buffer_size:
            raise ValueError(f"Buffer is not yet full. Current size: {self.ptr}/{self.buffer_size}")
        
        # Normalize advantages
        adv_mean, adv_std = np.mean(self.advantages), np.std(self.advantages)
        self.advantages = (self.advantages - adv_mean) / (adv_std + 1e-8)

        data = dict(
            states=np.array(self.states[:self.ptr]),
            actions=np.array(self.actions[:self.ptr]),
            rewards=np.array(self.rewards[:self.ptr]),
            returns=mx.array(self.returns[:self.ptr]),
            advantages=mx.array(self.advantages[:self.ptr]),
            log_probs=mx.array(self.log_probs[:self.ptr]),
            values=mx.array(self.values[:self.ptr])
        )

        # Reset buffer
        self.ptr, self.path_start_idx = 0, 0
        return data

    def _discount_cumsum(self, x, discount):
        """
        Computes discounted cumulative sums of vectors.
        input: [x0, x1, x2]
        output: [x0 + discount * x1 + discount^2 * x2, x1 + discount * x2, x2]
        """
        # This is a standard implementation of discounted cumulative sum
        # using scipy's lfilter, which is highly efficient.
        import scipy.signal
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]
