#!/usr/bin/env python3
"""
Quick test of action scaling to see if it helps performance
"""

import torch
import numpy as np
from swimmer.models.ncap_swimmer import NCAPSwimmer, NCAPSwimmerActor
from dm_control import suite
import time

def test_quick_scaling():
    """Quick test of action scaling."""
    print("=== QUICK ACTION SCALING TEST ===")
    
    # Test two configurations
    configs = [
        {'scale': 1.0, 'steps': 5000, 'name': 'Baseline (scale=1.0)'},
        {'scale': 3.0, 'steps': 5000, 'name': 'Triple actions (scale=3.0)'},
    ]
    
    env = suite.load('swimmer', 'swimmer6')
    
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
        
        max_distance = 0
        velocities = []
        positions = []
        energies = []
        
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
            max_distance = max(max_distance, distance)
            
            velocity = np.linalg.norm(env.physics.named.data.sensordata['head_vel'][:2])
            velocities.append(velocity)
            positions.append(current_pos[:2].copy())
            
            # Progress reporting
            if step % 1000 == 0:
                elapsed = time.time() - start_time
                print(f"  Step {step}: Current={distance:.3f}, Max={max_distance:.3f}, Vel={velocity:.3f}, Energy={action_energy:.3f}")
            
            if timestep.last():
                break
        
        # Calculate metrics
        avg_velocity = np.mean(velocities) if velocities else 0
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
        
        print(f"  Final Results:")
        print(f"    Max distance: {max_distance:.3f}")
        print(f"    Total path length: {total_path_length:.3f}")
        print(f"    Avg velocity: {avg_velocity:.3f}")
        print(f"    Avg action energy: {avg_energy:.3f}")
        print(f"    Time: {elapsed_time:.1f}s")
        
        # Store for comparison
        if config['scale'] == 1.0:
            baseline_distance = max_distance
        else:
            improvement = max_distance / baseline_distance if 'baseline_distance' in locals() else 1.0
            print(f"    Improvement over baseline: {improvement:.1f}x")
            
            if max_distance > 10.0:
                print("    ✅ SUCCESS: Achieved good performance with scaling!")
            elif max_distance > 5.0:
                print("    🔄 SIGNIFICANT IMPROVEMENT: Much better than baseline")
            elif improvement > 1.5:
                print("    📈 GOOD IMPROVEMENT: Scaling helps performance")
            else:
                print("    ❌ No significant improvement from scaling")

if __name__ == "__main__":
    test_quick_scaling() 