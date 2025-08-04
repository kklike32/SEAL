# knowledge-incorporation/src/restem/__init__.py

"""
RestEM (Reward-based Self-Training with EM) Implementation

This module implements the original SEAL methodology:
1. Self-Edit Generation: Generate multiple solution attempts
2. Test-Time Training: Fine-tune LoRA adapters per solution
3. Evaluation: Test adapters to identify correct solutions
4. RestEM Training: Behavioral cloning on correct solutions only

This follows the original SEAL paper methodology, not traditional RL.
"""
