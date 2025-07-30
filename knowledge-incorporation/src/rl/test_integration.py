#!/usr/bin/env python3
# knowledge-incorporation/src/rl/test_integration.py
"""
Test script to verify the RL-TTT integration is working properly.
This script tests the reward function with the TTT server.
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

from src.rl.reward import initialize_reward_client, get_reward, close_reward_client
from src.rl.dataset import build_dataset, get_squad_prompts

def test_reward_integration():
    """Test that the reward function works with the TTT server."""
    print("Testing RL-TTT integration...")
    
    try:
        # Initialize reward client
        print("1. Initializing reward client...")
        initialize_reward_client(port=5555)
        print("✓ Reward client initialized")
        
        # Get a sample prompt
        print("2. Loading sample data...")
        dataset = build_dataset()
        prompts = get_squad_prompts(dataset, num_samples=1)
        prompt = prompts[0]
        print(f"✓ Sample prompt loaded: {prompt[:100]}...")
        
        # Test with a simple completion
        print("3. Testing reward computation...")
        test_completion = " The answer is that this is a test completion for SEAL."
        
        print(f"Prompt: {prompt}")
        print(f"Completion: {test_completion}")
        print("Sending to TTT server...")
        
        reward = get_reward(prompt, test_completion)
        print(f"✓ Received reward: {reward}")
        
        # Close client
        print("4. Closing reward client...")
        close_reward_client()
        print("✓ Reward client closed")
        
        print(f"\n✅ Integration test successful! Reward: {reward}")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_reward_integration()
    sys.exit(0 if success else 1)
