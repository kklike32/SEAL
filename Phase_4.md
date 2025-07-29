# Project SEAL on MLX: Phase 4 - Outer-Loop Reinforcement Learning

**Objective:** To implement the outer-loop Reinforcement Learning (RL) agent of the SEAL framework. This agent will be trained to generate high-quality "completions" (fine-tuning data) that improve the base model's performance on downstream tasks.

---

## 1. Project Goal

Having successfully replicated the *inner-loop* Test-Time Training (TTT) mechanism in Phases 1-3, the next logical step is to build the *outer-loop* learning system. The inner loop provides a critical signal: a `mean_gain` score that measures how "helpful" a given completion is. The goal of Phase 4 is to use this signal to train an RL agent to generate better completions automatically.

This will involve creating a new training script that orchestrates the interaction between a policy model (the "Actor"), an optional value model (the "Critic"), and our existing TTT server, which will serve as the environment's reward function.

## 2. The Reinforcement Learning Framework

We will frame this as a standard RL problem:

- **State:** The current input text (e.g., a context paragraph from the SQuAD dataset).
- **Action:** The RL agent (a language model) generates a completion string. This completion is the self-edit data that will be used for fine-tuning.
- **Reward:** The `mean_gain` score produced by our existing `TTT_server_mlx.py`. A positive `mean_gain` constitutes a positive reward, while a negative `mean_gain` is a penalty.
- **Objective:** To train the agent to learn a policy (a strategy for generating completions) that maximizes the cumulative expected reward.

## 3. Final Implementation

We have implemented a robust, production-level PPO training system tailored for MLX. The implementation follows best practices to ensure stability and performance, adhering closely to the methodologies used in modern RL research.

### 3.1. The PPO Buffer (`ppo_buffer.py`)

Instead of training after every step, we first collect a large batch of experiences into a dedicated `PPOBuffer`. This is a critical component for stable training.

- **Role:** Stores trajectories (states, actions, rewards, values, and log probabilities) from the rollout phase.
- **Key Feature (GAE):** The buffer is responsible for calculating the **Generalized Advantage Estimation (GAE)**. GAE provides a more stable and lower-variance estimate of the advantage function compared to simpler methods, which is crucial for effective training.
- **Normalization:** The buffer also normalizes the calculated advantages, a standard technique that prevents scaling issues and further stabilizes the learning process.

### 3.2. The MLX PPO Trainer (`ppo_mlx.py`)

The `MLXPPO` class was significantly upgraded to be a high-performance, reusable trainer.

- **JIT Compilation:** The core function that calculates the PPO loss and gradients for both the Actor and Critic models is JIT-compiled using `@mx.compile`. This dramatically speeds up the most computationally intensive part of the training loop.
- **Learning Phase:** The trainer's `learn` method orchestrates the learning phase. It takes the data collected in the `PPOBuffer` and iterates over it for multiple epochs, updating the model weights on mini-batches. This ensures that the agent learns efficiently from the collected experiences.

### 3.3. The Main Training Loop (`train_rl_mlx.py`)

The main script was restructured to follow the standard PPO algorithm structure, which separates experience collection from learning.

- **Rollout Phase:** The script first enters a "rollout" phase, where it uses the current policy (the Actor) to interact with the environment (generating completions and getting rewards from the TTT server). These experiences are collected in the `PPOBuffer`.
- **Learning Phase:** Once the buffer is full, the script kicks off the "learning" phase by calling `ppo_trainer.learn()`. The trainer then uses the collected data to update the Actor and Critic models.
- **Iteration:** This entire process (rollout -> learn) is repeated for a set number of steps, allowing the agent to progressively improve its policy for generating helpful self-edits.

This phase represents the culmination of the SEAL project, bringing together the data generation and inner-loop validation into a complete, self-improving system.
