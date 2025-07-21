#!/usr/bin/env python3
"""
Quick test NCAP with longer duration 
"""

import torch
import numpy as np
import time
import os
from swimmer.models.ncap_swimmer import NCAPSwimmer, NCAPSwimmerActor
from swimmer.environments.mixed_environment import ImprovedMixedSwimmerEnv

def test_quick_duration():
    """Quick test with longer duration."""
    print("=== QUICK LONG DURATION TEST ===")
    
    # Test one configuration with more steps
    config = {'period': 60, 'steps': 5000, 'name': 'Long duration test'}
    
    env = ImprovedMixedSwimmerEnv(n_links=6)
    
    print(f"Testing {config['name']}...")
    
    # Create NCAP
    ncap = NCAPSwimmer(
        n_joints=5,
        oscillator_period=config['period'],
        include_environment_adaptation=True
    )
    
    actor = NCAPSwimmerActor(ncap)
    
    # Reset everything
    obs = env.reset()
    ncap.reset()
    
    max_distance = 0
    velocities = []
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
        max_distance = max(max_distance, distance)
        
        velocity = np.linalg.norm(env.physics.named.data.sensordata['head_vel'][:2])
        velocities.append(velocity)
        positions.append(current_pos[:2].copy())
        
        # Print progress more frequently
        if step % 500 == 0:
            print(f"  Step {step}: Current Distance={distance:.3f}, Max Distance={max_distance:.3f}, Velocity={velocity:.3f}")
        
        if done:
            print(f"  Episode ended at step {step}")
            break
    
    # Calculate total path length
    positions = np.array(positions)
    if len(positions) > 1:
        path_diffs = np.diff(positions, axis=0)
        path_lengths = np.linalg.norm(path_diffs, axis=1)
        total_path_length = np.sum(path_lengths)
    else:
        total_path_length = 0
    
    avg_velocity = np.mean(velocities) if velocities else 0
    max_velocity = np.max(velocities) if velocities else 0
    
    print(f"\n  FINAL RESULTS:")
    print(f"    Final distance from start: {distance:.3f}")
    print(f"    Maximum distance reached: {max_distance:.3f}")
    print(f"    Total path traveled: {total_path_length:.3f}")
    print(f"    Average velocity: {avg_velocity:.3f}")
    print(f"    Maximum velocity: {max_velocity:.3f}")
    print(f"    Steps completed: {step + 1}")
    
    # Analysis
    if max_distance > 5.0:
        print("✅ SUCCESS: Good swimming performance with longer duration!")
    elif max_distance > 1.0:
        print("🔄 IMPROVEMENT: Better than short tests but still not optimal")
    elif max_distance > 0.5:
        print("📈 PROGRESS: Some improvement from longer duration")
    else:
        print("❌ No significant improvement from longer duration")
    
    # Compare with expected performance
    expected_distance = 20  # From your observation of notebook
    ratio = max_distance / expected_distance
    print(f"\nComparison with expected baseline:")
    print(f"  Expected: ~{expected_distance} distance units")
    print(f"  Actual: {max_distance:.3f} distance units")
    print(f"  Ratio: {ratio:.1%} of expected performance")
    
    return {
        'max_distance': max_distance,
        'total_path': total_path_length,
        'avg_velocity': avg_velocity,
        'steps': step + 1
    }

if __name__ == "__main__":
    results = test_quick_duration() 