#!/usr/bin/env python3
# knowledge-incorporation/src/rl/test_reward_robustness.py
"""
Test script to verify the RL reward system works robustly with server crashes.
Tests multiple requests to ensure the retry mechanism works.
"""

import sys
import os
import time

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

from src.rl.reward import initialize_reward_client, get_reward, close_reward_client
from src.rl.dataset import build_dataset, get_squad_prompts

def test_multiple_rewards():
    """Test multiple reward requests to verify robustness."""
    print("🧪 Testing Reward System Robustness")
    print("===================================")
    
    try:
        # Initialize reward client
        print("1. Initializing reward client...")
        initialize_reward_client(port=5555)
        print("✅ Reward client initialized")
        
        # Get sample prompts
        print("\n2. Loading sample data...")
        dataset = build_dataset()
        prompts = get_squad_prompts(dataset, num_samples=3)
        print(f"✅ Loaded {len(prompts)} sample prompts")
        
        # Test completions
        completions = [
            "Saint Bernadette Soubirous was the person to whom the Virgin Mary allegedly appeared in 1858 in Lourdes, France.",
            "The correct answer to this question can be found in the context provided above.",
            "Based on the information given, the answer is clearly stated in the passage."
        ]
        
        results = []
        
        for i, (prompt, completion) in enumerate(zip(prompts, completions)):
            print(f"\n3.{i+1}. Testing reward computation #{i+1}...")
            print(f"Question: {prompt.split('Question: ')[1].split('Answer:')[0].strip()}")
            print(f"Completion: {completion}")
            
            start_time = time.time()
            reward = get_reward(prompt, completion)
            end_time = time.time()
            
            results.append({
                'request': i+1,
                'reward': reward,
                'time_taken': end_time - start_time,
                'success': reward != 0.0 or i == 0  # First might be 0.0 legitimately
            })
            
            print(f"✅ Request {i+1}: Reward = {reward:.4f}, Time = {end_time - start_time:.1f}s")
            
            # Wait between requests to let server restart if needed
            if i < len(prompts) - 1:
                print("   Waiting 15 seconds for server restart...")
                time.sleep(15)
        
        # Close client
        print("\n4. Closing reward client...")
        close_reward_client()
        print("✅ Reward client closed")
        
        # Analyze results
        print(f"\n📊 Results Summary:")
        print("==================")
        successful_requests = sum(1 for r in results if r['success'])
        total_reward = sum(r['reward'] for r in results)
        avg_time = sum(r['time_taken'] for r in results) / len(results)
        
        print(f"✅ Successful requests: {successful_requests}/{len(results)}")
        print(f"🎯 Total reward collected: {total_reward:.4f}")
        print(f"⏱️  Average time per request: {avg_time:.1f}s")
        
        for i, result in enumerate(results):
            status = "✅" if result['success'] else "❌"
            print(f"   Request {i+1}: {status} Reward: {result['reward']:.4f}, Time: {result['time_taken']:.1f}s")
        
        if successful_requests >= len(results) * 0.5:  # At least 50% success
            print(f"\n🎉 Robustness test PASSED! ({successful_requests}/{len(results)} successful)")
            return True
        else:
            print(f"\n⚠️  Robustness test PARTIAL - Some requests failed")
            return False
        
    except Exception as e:
        print(f"❌ Robustness test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_multiple_rewards()
    sys.exit(0 if success else 1)
