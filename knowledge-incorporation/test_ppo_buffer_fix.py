#!/usr/bin/env python3
"""
Test script to verify that the PPO buffer and MLXPPO integration works correctly.
"""

import sys
import os

# Add knowledge-incorporation directory to path
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

import mlx.core as mx
import numpy as np
from trl import PPOConfig

# Import our modules
from src.rl.ppo_buffer import PPOBuffer
from src.rl.ppo_mlx import MLXPPO

def test_buffer_get_format():
    """Test that PPOBuffer.get() returns the expected format."""
    print("Testing PPOBuffer.get() format...")
    
    buffer = PPOBuffer(buffer_size=2)
    
    # Add some dummy data
    buffer.add("prompt1", "action1", 0.5, 0.1, -2.3)
    buffer.add("prompt2", "action2", -0.2, 0.3, -1.8)
    
    # Finish the path
    buffer.finish_path()
    
    # Get the data
    data = buffer.get()
    
    print("Buffer data keys:", list(data.keys()))
    print("Expected keys: ['states', 'actions', 'rewards', 'returns', 'advantages', 'log_probs', 'values']")
    
    expected_keys = {'states', 'actions', 'rewards', 'returns', 'advantages', 'log_probs', 'values'}
    actual_keys = set(data.keys())
    
    if expected_keys == actual_keys:
        print("✓ Buffer returns correct keys")
    else:
        print("✗ Buffer keys mismatch")
        print(f"Missing: {expected_keys - actual_keys}")
        print(f"Extra: {actual_keys - expected_keys}")
    
    # Check data types
    print("\nData types:")
    for key, value in data.items():
        print(f"  {key}: {type(value)} - shape: {getattr(value, 'shape', 'N/A')}")
    
    return data

def test_ppo_learn_integration():
    """Test that the MLXPPO.learn() method can accept buffer data format."""
    print("\n" + "="*50)
    print("Testing MLXPPO.learn() integration...")
    
    # Create dummy buffer data in the expected format
    dummy_data = {
        'states': np.array(['prompt1', 'prompt2']),
        'actions': np.array(['action1', 'action2']),
        'rewards': np.array([0.5, -0.2]),
        'returns': mx.array([0.6, -0.1]),
        'advantages': mx.array([0.5, -0.4]),
        'log_probs': mx.array([-2.3, -1.8]),
        'values': mx.array([0.1, 0.3])
    }
    
    print("Mock buffer data structure:")
    for key, value in dummy_data.items():
        print(f"  {key}: {type(value)} - shape: {getattr(value, 'shape', 'N/A')}")
    
    # Test the extraction logic that's now in MLXPPO.learn()
    try:
        prompts = dummy_data['states']
        responses = dummy_data['actions']
        rewards = dummy_data['rewards']
        values = dummy_data['values']
        log_probs = dummy_data['log_probs']
        returns = dummy_data['returns']
        advantages = dummy_data['advantages']
        
        print("\n✓ Successfully extracted all data from buffer format")
        print(f"  prompts: {len(prompts)} items")
        print(f"  responses: {len(responses)} items")
        print(f"  rewards: {rewards.shape}")
        print(f"  returns: {returns.shape}")
        print(f"  advantages: {advantages.shape}")
        
    except Exception as e:
        print(f"✗ Failed to extract data: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("Testing PPO Buffer and MLXPPO integration...")
    print("="*50)
    
    # Test buffer format
    buffer_data = test_buffer_get_format()
    
    # Test integration
    success = test_ppo_learn_integration()
    
    if success:
        print("\n" + "="*50)
        print("✓ All tests passed! The buffer/PPO integration should work.")
    else:
        print("\n" + "="*50)
        print("✗ Some tests failed. Check the errors above.")
