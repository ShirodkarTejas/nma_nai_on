#!/usr/bin/env python3
"""
Test if action scaling or longer duration can achieve expected performance
"""

import torch
import numpy as np
from swimmer.models.ncap_swimmer import NCAPSwimmer, NCAPSwimmerActor
from dm_control import suite
import time

def test_action_scaling():
    """Test different action scaling factors and durations."""
    print("=== TESTING ACTION SCALING AND DURATION ===")
    
    # Test configurations
    configs = [
        {'scale': 1.0, 'steps': 3000, 'name': 'Baseline (scale=1.0, 3k steps)'},
        {'scale': 2.0, 'steps': 3000, 'name': 'Double actions (scale=2.0, 3k steps)'},
        {'scale': 3.0, 'steps': 3000, 'name': 'Triple actions (scale=3.0, 3k steps)'},
        {'scale': 1.0, 'steps': 10000, 'name': 'Long duration (scale=1.0, 10k steps)'},
        {'scale': 2.0, 'steps': 10000, 'name': 'Double + long (scale=2.0, 10k steps)'},
    ]
    
    env = suite.load('swimmer', 'swimmer6')
    
    results = []
    
    for config in configs:
        print(f"\n=== {config['name']} ===")
        
        # Create NCAP
        ncap = NCAPSwimmer(
            n_joints=5,
            oscillator_period=60,
            include_environment_adaptation=False
        )
        
        # Create actor with scaling
        class ScaledActor:
            def __init__(self, base_actor, scale):
                self.base_actor = base_actor
                self.scale = scale
            
            def __call__(self, obs):
                action = self.base_actor(obs)
                return action * self.scale
        
        base_actor = NCAPSwimmerActor(ncap)
        actor = ScaledActor(base_actor, config['scale'])
        
        # Test performance
        timestep = env.reset()
        ncap.reset()
        
        total_distance = 0
        max_distance = 0
        velocities = []
        positions = []
        energies = []  # Track energy/power
        
        initial_pos = env.physics.named.data.xpos['head'].copy()
        start_time = time.time()
        
        for step in range(config['steps']):
            # Get action
            obs = timestep.observation
            action = actor(obs['joints'])
            
            # Track action energy
            action_energy = np.sum(action**2)
            energies.append(action_energy)
            
            # Take step
            timestep = env.step(action)
            
            # Track performance
            current_pos = env.physics.named.data.xpos['head']
            distance = np.linalg.norm(current_pos[:2] - initial_pos[:2])
            total_distance = distance
            max_distance = max(max_distance, distance)
            
            velocity = np.linalg.norm(env.physics.named.data.sensordata['head_vel'][:2])
            velocities.append(velocity)
            positions.append(current_pos[:2].copy())
            
            # Progress reporting
            if step % 1000 == 0:
                elapsed = time.time() - start_time
                print(f"  Step {step}: Distance={distance:.3f}, Max={max_distance:.3f}, Vel={velocity:.3f}, Energy={action_energy:.3f}, Time={elapsed:.1f}s")
            
            if timestep.last():
                break
        
        # Calculate metrics
        avg_velocity = np.mean(velocities) if velocities else 0
        max_velocity = np.max(velocities) if velocities else 0
        avg_energy = np.mean(energies) if energies else 0
        
        # Calculate total path traveled
        positions = np.array(positions)
        if len(positions) > 1:
            path_diffs = np.diff(positions, axis=0)
            path_lengths = np.linalg.norm(path_diffs, axis=1)
            total_path_length = np.sum(path_lengths)
        else:
            total_path_length = 0
        
        elapsed_time = time.time() - start_time
        
        result = {
            'config': config,
            'final_distance': total_distance,
            'max_distance': max_distance,
            'total_path_length': total_path_length,
            'avg_velocity': avg_velocity,
            'max_velocity': max_velocity,
            'avg_energy': avg_energy,
            'elapsed_time': elapsed_time,
            'steps_completed': step + 1
        }
        
        results.append(result)
        
        print(f"  Results:")
        print(f"    Final distance: {total_distance:.3f}")
        print(f"    Max distance: {max_distance:.3f}")
        print(f"    Total path length: {total_path_length:.3f}")
        print(f"    Avg velocity: {avg_velocity:.3f}")
        print(f"    Avg action energy: {avg_energy:.3f}")
        print(f"    Time: {elapsed_time:.1f}s")
        print(f"    Steps: {step + 1}")
    
    # Analysis
    print(f"\n=== COMPARISON RESULTS ===")
    print(f"{'Configuration':<35} {'Max Dist':<10} {'Path Len':<10} {'Avg Vel':<10} {'Energy':<10}")
    print("-" * 80)
    
    for result in results:
        config = result['config']
        print(f"{config['name']:<35} {result['max_distance']:<10.3f} {result['total_path_length']:<10.3f} {result['avg_velocity']:<10.3f} {result['avg_energy']:<10.3f}")
    
    # Find best performer
    best_result = max(results, key=lambda x: x['max_distance'])
    print(f"\n🏆 Best performing configuration:")
    print(f"   {best_result['config']['name']}")
    print(f"   Max distance: {best_result['max_distance']:.3f}")
    print(f"   Path length: {best_result['total_path_length']:.3f}")
    
    # Compare with expected
    expected_distance = 20
    ratio = best_result['max_distance'] / expected_distance
    print(f"\nComparison with expected notebook performance:")
    print(f"  Expected: ~{expected_distance} distance units")
    print(f"  Best achieved: {best_result['max_distance']:.3f} ({ratio:.1%})")
    
    if best_result['max_distance'] > 10.0:
        print("✅ SUCCESS: Achieved good swimming performance!")
    elif best_result['max_distance'] > 5.0:
        print("🔄 IMPROVEMENT: Significant improvement, but still below notebook")
    elif best_result['max_distance'] > best_result['max_distance'] * 1.5:  # Improvement over baseline
        print("📈 PROGRESS: Some improvement from scaling")
    else:
        print("❌ No significant improvement from scaling/duration")
        print("   The issue may be:")
        print("   - Fundamental difference in environment setup")
        print("   - Different time scales between notebook and dm_control")
        print("   - Notebook performance was from trained model, not untrained")
        print("   - Need different NCAP parameters (period, etc)")
    
    # Recommendations
    if best_result['avg_energy'] > 5.0:
        print(f"\n💡 High action energy detected ({best_result['avg_energy']:.1f})")
        print("   Consider that the swimmer might be fighting against physics")
    
    if best_result['total_path_length'] > best_result['max_distance'] * 3:
        print(f"\n💡 High path efficiency: {best_result['total_path_length']:.1f} vs {best_result['max_distance']:.1f}")
        print("   Swimmer is moving effectively, but may be going in circles")
    
    return results

if __name__ == "__main__":
    results = test_action_scaling() 