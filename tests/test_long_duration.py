#!/usr/bin/env python3
"""
Test NCAP with longer duration and different parameters to match notebook performance
"""

import torch
import numpy as np
import time
import os
from swimmer.models.ncap_swimmer import NCAPSwimmer, NCAPSwimmerActor
from swimmer.environments.mixed_environment import ImprovedMixedSwimmerEnv

def test_long_duration_performance():
    """Test NCAP with longer duration and different parameters."""
    print("=== TESTING LONG DURATION NCAP PERFORMANCE ===")
    
    # Test different configurations
    configs = [
        {'period': 60, 'steps': 10000, 'name': 'Standard (60 period, 10k steps)'},
        {'period': 30, 'steps': 10000, 'name': 'Faster oscillation (30 period, 10k steps)'},
        {'period': 120, 'steps': 10000, 'name': 'Slower oscillation (120 period, 10k steps)'},
        {'period': 60, 'steps': 20000, 'name': 'Very long (60 period, 20k steps)'},
    ]
    
    env = ImprovedMixedSwimmerEnv(n_links=6)
    
    results = []
    
    for config in configs:
        print(f"\n=== {config['name']} ===")
        
        # Create NCAP with current config
        ncap = NCAPSwimmer(
            n_joints=5,
            oscillator_period=config['period'],
            include_environment_adaptation=True
        )
        
        actor = NCAPSwimmerActor(ncap)
        
        # Reset everything
        obs = env.reset()
        ncap.reset()
        
        total_distance = 0
        max_distance = 0
        velocities = []
        rewards = []
        positions = []
        
        initial_pos = env.physics.named.data.xpos['head'].copy()
        
        print(f"Starting position: {initial_pos[:2]}")
        
        for step in range(config['steps']):
            # Get action
            action = actor(obs)
            
            # Take step
            obs, reward, done, info = env.step(action)
            
            # Track performance
            current_pos = env.physics.named.data.xpos['head']
            distance = np.linalg.norm(current_pos[:2] - initial_pos[:2])
            total_distance = distance
            max_distance = max(max_distance, distance)
            
            velocity = np.linalg.norm(env.physics.named.data.sensordata['head_vel'][:2])
            velocities.append(velocity)
            rewards.append(reward)
            positions.append(current_pos[:2].copy())
            
            # Print progress
            if step % 2000 == 0:
                print(f"  Step {step}: Distance={distance:.3f}, Max Distance={max_distance:.3f}, Velocity={velocity:.3f}")
            
            if done:
                print(f"  Episode ended at step {step}")
                break
        
        # Calculate final metrics
        avg_velocity = np.mean(velocities) if velocities else 0
        max_velocity = np.max(velocities) if velocities else 0
        avg_reward = np.mean(rewards) if rewards else 0
        
        # Calculate total path length
        positions = np.array(positions)
        if len(positions) > 1:
            path_diffs = np.diff(positions, axis=0)
            path_lengths = np.linalg.norm(path_diffs, axis=1)
            total_path_length = np.sum(path_lengths)
        else:
            total_path_length = 0
        
        result = {
            'config': config,
            'final_distance': total_distance,
            'max_distance': max_distance,
            'total_path_length': total_path_length,
            'avg_velocity': avg_velocity,
            'max_velocity': max_velocity,
            'avg_reward': avg_reward,
            'steps_completed': step + 1
        }
        
        results.append(result)
        
        print(f"  Final Results:")
        print(f"    Final distance from start: {total_distance:.3f}")
        print(f"    Maximum distance reached: {max_distance:.3f}")
        print(f"    Total path length: {total_path_length:.3f}")
        print(f"    Average velocity: {avg_velocity:.3f}")
        print(f"    Maximum velocity: {max_velocity:.3f}")
        print(f"    Average reward: {avg_reward:.3f}")
        print(f"    Steps completed: {step + 1}")
    
    print(f"\n=== SUMMARY COMPARISON ===")
    for result in results:
        config = result['config']
        print(f"{config['name']}:")
        print(f"  Distance: {result['final_distance']:.3f}")
        print(f"  Max Distance: {result['max_distance']:.3f}")
        print(f"  Path Length: {result['total_path_length']:.3f}")
        print(f"  Avg Velocity: {result['avg_velocity']:.3f}")
    
    # Find best performing config
    best_result = max(results, key=lambda x: x['max_distance'])
    print(f"\n🏆 Best performing configuration:")
    print(f"   {best_result['config']['name']}")
    print(f"   Max distance: {best_result['max_distance']:.3f}")
    print(f"   Path length: {best_result['total_path_length']:.3f}")
    
    if best_result['max_distance'] > 5.0:
        print("✅ SUCCESS: Found configuration with good swimming performance!")
    elif best_result['max_distance'] > 1.0:
        print("🔄 PARTIAL SUCCESS: Some improvement but still not optimal")
    else:
        print("❌ Still low performance across all configurations")
        print("   This suggests the issue might be:")
        print("   - Environment physics parameters")
        print("   - Incorrect NCAP implementation details")
        print("   - Missing critical components from notebook")
    
    return results

if __name__ == "__main__":
    results = test_long_duration_performance() 