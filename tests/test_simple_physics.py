#!/usr/bin/env python3
"""
Test NCAP with simplified physics - only viscosity changes
"""

import torch
import numpy as np
import time
import os
from swimmer.models.ncap_swimmer import NCAPSwimmer, NCAPSwimmerActor
from dm_control import suite
from dm_control.rl import control
from dm_control.suite import swimmer
from dm_control import mujoco
import collections

class SimplifiedMixedSwimmer(swimmer.Swimmer):
    """Simplified mixed environment that only changes viscosity."""
    
    def __init__(self, desired_speed=0.1, **kwargs):
        super().__init__(**kwargs)
        self._desired_speed = desired_speed
        self.time_step = 0
        
    def initialize_episode(self, physics):
        super().initialize_episode(physics)
        
        # Use standard dm_control defaults - don't change anything initially
        print(f"Default physics: Viscosity={physics.model.opt.viscosity}, Density={physics.model.opt.density}")
        
        # Store initial position
        self._initial_position = physics.named.data.xpos['head', :2].copy()
        self.time_step = 0
        
    def get_current_environment_simple(self, physics):
        """Simple environment switching based on time/position."""
        # Simple rule: switch between environments based on time or position
        head_pos = physics.named.data.xpos['head', :2]
        
        # Switch every 1000 steps or based on Y position
        if (self.time_step // 1000) % 2 == 0 or head_pos[1] < 0:
            return "water"  # Low viscosity
        else:
            return "land"   # Standard viscosity
    
    def apply_simple_physics(self, physics, env_type):
        """Apply ONLY viscosity changes - keep everything else standard."""
        if env_type == "water":
            # Lower viscosity for easier movement
            physics.model.opt.viscosity = 0.0  # Same as standard
        else:
            # Higher viscosity for harder movement  
            physics.model.opt.viscosity = 0.1  # Slight increase only
        
        # Keep density and friction at standard dm_control values
        # Don't modify anything else!
    
    def get_observation(self, physics):
        """Standard observation with minimal environment info."""
        obs = collections.OrderedDict()
        obs['joints'] = physics.joints()
        obs['body_velocities'] = physics.body_velocities()
        
        # Simple environment detection
        current_env = self.get_current_environment_simple(physics)
        self.apply_simple_physics(physics, current_env)
        
        # Minimal environment encoding
        obs['environment_type'] = np.array([1.0, 0.0] if current_env == "water" else [0.0, 1.0])
        obs['head_position'] = physics.named.data.xpos['head', :2]
        obs['in_water_zone'] = np.array([1.0 if current_env == "water" else 0.0])
        obs['in_land_zone'] = np.array([1.0 if current_env == "land" else 0.0])
        
        self.time_step += 1
        return obs
    
    def get_reward(self, physics):
        """Standard swimmer reward - simple forward movement."""
        forward_velocity = -physics.named.data.sensordata['head_vel'][1]  # Note: negative for forward
        return max(0, forward_velocity)  # Simple: just reward forward movement

def test_simple_physics():
    """Test NCAP with simplified physics (only viscosity changes)."""
    print("=== TESTING SIMPLIFIED PHYSICS (VISCOSITY ONLY) ===")
    
    # Test 1: Standard dm_control swimmer (baseline)
    print("\n1. Testing standard dm_control swimmer...")
    standard_env = suite.load('swimmer', 'swimmer6')
    
    # Create NCAP for standard environment
    ncap_standard = NCAPSwimmer(
        n_joints=5,  
        oscillator_period=60,
        include_environment_adaptation=False  # No adaptations for baseline
    )
    actor_standard = NCAPSwimmerActor(ncap_standard)
    
    def test_env_performance(env, actor, name, steps=3000):
        timestep = env.reset()
        actor.swimmer.reset()
        
        total_distance = 0
        max_distance = 0
        velocities = []
        rewards = []
        
        initial_pos = env.physics.named.data.xpos['head'].copy()
        
        for step in range(steps):
            # Get observation
            if hasattr(timestep, 'observation'):
                obs = timestep.observation
                joint_pos = obs['joints'] if 'joints' in obs else obs[:5]
            else:
                joint_pos = env.physics.joints()
            
            # Get action
            action = actor(joint_pos)
            
            # Take step
            timestep = env.step(action)
            
            # Track performance
            current_pos = env.physics.named.data.xpos['head']
            distance = np.linalg.norm(current_pos[:2] - initial_pos[:2])
            total_distance = distance
            max_distance = max(max_distance, distance)
            
            velocity = np.linalg.norm(env.physics.named.data.sensordata['head_vel'][:2])
            velocities.append(velocity)
            
            if hasattr(timestep, 'reward'):
                rewards.append(timestep.reward)
            
            if step % 500 == 0:
                print(f"  {name} Step {step}: Distance={distance:.3f}, Max={max_distance:.3f}, Vel={velocity:.3f}")
            
            if hasattr(timestep, 'last') and timestep.last():
                break
        
        avg_velocity = np.mean(velocities) if velocities else 0
        avg_reward = np.mean(rewards) if rewards else 0
        
        print(f"  {name} Results:")
        print(f"    Final distance: {total_distance:.3f}")
        print(f"    Max distance: {max_distance:.3f}")
        print(f"    Avg velocity: {avg_velocity:.3f}")
        print(f"    Avg reward: {avg_reward:.3f}")
        
        return {
            'final_distance': total_distance,
            'max_distance': max_distance,
            'avg_velocity': avg_velocity,
            'avg_reward': avg_reward
        }
    
    # Test standard environment
    standard_results = test_env_performance(standard_env, actor_standard, "Standard dm_control")
    
    # Test 2: Our simplified mixed environment
    print("\n2. Testing simplified mixed environment (viscosity only)...")
    
    # Create simplified environment
    model_string, assets = swimmer.get_model_and_assets(6)
    physics = swimmer.Physics.from_xml_string(model_string, assets=assets)
    task = SimplifiedMixedSwimmer(desired_speed=0.1)
    simplified_env = control.Environment(
        physics, task, time_limit=30, control_timestep=swimmer._CONTROL_TIMESTEP
    )
    
    # Create NCAP for mixed environment
    ncap_mixed = NCAPSwimmer(
        n_joints=5,
        oscillator_period=60,
        include_environment_adaptation=True  # Enable adaptations
    )
    actor_mixed = NCAPSwimmerActor(ncap_mixed)
    
    # Test simplified environment
    simplified_results = test_env_performance(simplified_env, actor_mixed, "Simplified mixed")
    
    print(f"\n=== COMPARISON ===")
    print(f"Standard dm_control distance: {standard_results['max_distance']:.3f}")
    print(f"Simplified mixed distance: {simplified_results['max_distance']:.3f}")
    
    improvement_ratio = simplified_results['max_distance'] / standard_results['max_distance'] if standard_results['max_distance'] > 0 else 0
    
    if standard_results['max_distance'] > 5.0:
        print("✅ SUCCESS: Standard dm_control shows good baseline performance!")
        if simplified_results['max_distance'] > standard_results['max_distance'] * 0.8:
            print("✅ Simplified mixed environment maintains good performance!")
        else:
            print("⚠️  Simplified mixed environment reduces performance")
    else:
        print("❌ Even standard dm_control shows low performance")
        print("   This suggests the issue is with NCAP implementation or test setup")
    
    expected_distance = 20  # Your observation from notebook
    standard_ratio = standard_results['max_distance'] / expected_distance
    print(f"\nComparison with expected notebook performance:")
    print(f"  Expected: ~{expected_distance} distance units")
    print(f"  Standard dm_control: {standard_results['max_distance']:.3f} ({standard_ratio:.1%})")
    print(f"  Simplified mixed: {simplified_results['max_distance']:.3f}")
    
    return standard_results, simplified_results

if __name__ == "__main__":
    standard_results, simplified_results = test_simple_physics() 