#!/usr/bin/env python3
"""
Test pure NCAP performance without environment adaptations
"""

import torch
import numpy as np
import time
import os
from swimmer.models.ncap_swimmer import NCAPSwimmer, NCAPSwimmerActor
from swimmer.environments.mixed_environment import ImprovedMixedSwimmerEnv

def test_pure_ncap_performance():
    """Test pure NCAP without environment adaptations."""
    print("=== TESTING PURE NCAP PERFORMANCE ===")
    
    # Test 1: Pure NCAP with correct joint count (5 joints for 6-link swimmer)
    print("1. Testing pure NCAP (5 joints, no environment adaptation)...")
    pure_ncap = NCAPSwimmer(
        n_joints=5,  # 6-link swimmer has 5 joints
        oscillator_period=60,
        include_environment_adaptation=False  # Disable our adaptations
    )
    
    # Test 2: NCAP with environment adaptations  
    print("2. Testing NCAP with environment adaptations...")
    adapted_ncap = NCAPSwimmer(
        n_joints=5,
        oscillator_period=60, 
        include_environment_adaptation=True  # Enable our adaptations
    )
    
    # Test both in our mixed environment
    print("\n=== TESTING IN MIXED ENVIRONMENT ===")
    
    # Create environment with 6 links (5 joints)
    env = ImprovedMixedSwimmerEnv(n_links=6)
    env.reset()
    
    def test_ncap_model(ncap_model, name):
        print(f"\nTesting {name}...")
        
        actor = NCAPSwimmerActor(ncap_model)
        
        # Reset environment and model
        obs = env.reset()
        ncap_model.reset()
        
        total_distance = 0
        velocities = []
        rewards = []
        env_transitions = 0
        current_env = None
        
        initial_pos = env.physics.named.data.xpos['head'].copy()
        
        for step in range(1000):  # Test for 1000 steps
            # Get action
            action = actor(obs)
            
            # Take step
            obs, reward, done, info = env.step(action)
            
            # Track performance
            current_pos = env.physics.named.data.xpos['head']
            distance = np.linalg.norm(current_pos[:2] - initial_pos[:2])
            total_distance = distance
            
            velocity = np.linalg.norm(env.physics.named.data.sensordata['head_vel'][:2])
            velocities.append(velocity)
            rewards.append(reward)
            
            # Track environment changes  
            try:
                new_env = env.current_environment
                if new_env != current_env and current_env is not None:
                    env_transitions += 1
                    print(f"  Environment transition at step {step}: {current_env} -> {new_env}")
                current_env = new_env
            except AttributeError:
                # Environment doesn't track current environment
                pass
            
            if step % 200 == 0:
                print(f"  Step {step}: Distance={distance:.3f}, Velocity={velocity:.3f}, Reward={reward:.3f}")
            
            if done:
                break
        
        avg_velocity = np.mean(velocities) if velocities else 0
        avg_reward = np.mean(rewards) if rewards else 0
        
        print(f"  {name} Results:")
        print(f"    Total distance: {total_distance:.3f}")
        print(f"    Average velocity: {avg_velocity:.3f}")
        print(f"    Average reward: {avg_reward:.3f}")
        print(f"    Environment transitions: {env_transitions}")
        
        return {
            'distance': total_distance,
            'avg_velocity': avg_velocity,
            'avg_reward': avg_reward,
            'env_transitions': env_transitions
        }
    
    # Test both models
    pure_results = test_ncap_model(pure_ncap, "Pure NCAP (no adaptations)")
    adapted_results = test_ncap_model(adapted_ncap, "NCAP with adaptations")
    
    print(f"\n=== COMPARISON ===")
    print(f"Pure NCAP distance: {pure_results['distance']:.3f}")
    print(f"Adapted NCAP distance: {adapted_results['distance']:.3f}")
    
    if pure_results['distance'] > 1.0:
        print("✅ SUCCESS: Pure NCAP shows good swimming performance!")
    elif adapted_results['distance'] > pure_results['distance'] * 1.2:
        print("✅ Environment adaptations help performance")
    else:
        print("❌ Still low performance - may need different approach")
        print("   Possible issues:")
        print("   - Environment physics scaling")
        print("   - Time step mismatches")
        print("   - Different oscillator parameters needed")
    
    return pure_results, adapted_results

if __name__ == "__main__":
    pure_results, adapted_results = test_pure_ncap_performance() 