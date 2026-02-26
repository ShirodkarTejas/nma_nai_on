#!/usr/bin/env python3
"""
Diagnostic Land Boundary Testing
Tests the swimmer's ability to cross land zone boundaries by tracking exact distances.
Shows whether the issue is movement capability or boundary detection logic.
"""

import os
import sys
import numpy as np
import math
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from swimmer.environments.progressive_mixed_env import ProgressiveMixedSwimmerEnv


class DiagnosticAgent:
    """Agent that generates strong directional movement to test boundary crossing."""
    
    def __init__(self, action_space):
        self.action_space = action_space
        self.step_count = 0
        
    def test_step(self, obs):
        """Generate strong directional actions designed to escape land zones."""
        # Create coordinated forward thrust
        action = np.zeros(self.action_space.shape[0])
        
        # Strategy: All joints work together for maximum forward movement
        base_amplitude = 0.9  # Near maximum
        
        # Coordinated forward thrust pattern
        action[0] = base_amplitude  # Strong primary thrust
        if len(action) > 1:
            action[1] = base_amplitude * 0.8  # Supporting thrust
        if len(action) > 2:
            action[2] = base_amplitude * 0.6  # Additional support
        if len(action) > 3:
            action[3] = base_amplitude * 0.4  # Fine control
        
        # Add oscillatory component for swimming-like motion
        time_factor = self.step_count * 0.15
        oscillation = 0.3 * np.sin(time_factor)
        action[0] += oscillation
        if len(action) > 1:
            action[1] += oscillation * 0.5
        
        # Clamp to action space
        action = np.clip(action, -1.0, 1.0)
        
        self.step_count += 1
        return action


def run_diagnostic_test():
    """Run diagnostic test to determine exact boundary crossing capability."""
    
    print("🔍 Diagnostic Land Boundary Testing")
    print("=" * 60)
    print("Testing swimmer's ability to cross land zone boundaries")
    print("with strong directional movement and exact distance tracking")
    print()
    
    # Create environment
    env = ProgressiveMixedSwimmerEnv(n_links=5, time_limit=3000)
    
    # Test with progressively smaller zones to find the breaking point
    test_zones = [
        {"radius": 0.05, "steps": 150, "name": "Ultra Micro Zone"},
        {"radius": 0.08, "steps": 200, "name": "Super Tiny Zone"}, 
        {"radius": 0.12, "steps": 250, "name": "Tiny Zone"},
        {"radius": 0.2, "steps": 300, "name": "Small Zone"},
        {"radius": 0.3, "steps": 400, "name": "Medium Zone"}
    ]
    
    agent = DiagnosticAgent(env.action_spec)
    
    for zone_config in test_zones:
        print(f"🔬 Testing {zone_config['name']} (radius: {zone_config['radius']}m)")
        
        # Set up environment
        env.set_manual_progress(0.5, force_land_start=True)
        
        # Override land zones
        land_zones = [{'center': [0.0, 0.0], 'radius': zone_config['radius']}]
        if hasattr(env.env, '_task'):
            task = env.env._task
            task._current_land_zones = land_zones
        
        # Reset and get initial state
        timestep = env.reset()
        agent.step_count = 0
        
        # Track position and distances
        positions = []
        distances_from_center = []
        boundary_crossings = []
        max_distance = 0
        
        print(f"   Zone boundary at: {zone_config['radius']:.3f}m from origin")
        
        for step in range(zone_config['steps']):
            # Get action and step
            observation = timestep.observation if hasattr(timestep, 'observation') else timestep
            action = agent.test_step(observation)
            timestep = env.step(action)
            
            # Get swimmer position
            if hasattr(env.env, '_physics'):
                physics = env.env._physics
                # Try different body part names used in swimmer
                try:
                    swimmer_pos = physics.named.data.xpos['head'][:2]  # Try head first
                except KeyError:
                    try:
                        swimmer_pos = physics.named.data.geom_xpos['head'][:2]  # Try geom_xpos
                    except KeyError:
                        # Fallback: use the first body part
                        first_body = list(physics.named.data.xpos.keys())[0] if hasattr(physics.named.data, 'xpos') else None
                        if first_body:
                            swimmer_pos = physics.named.data.xpos[first_body][:2]
                        else:
                            swimmer_pos = np.array([0.0, 0.0])  # Final fallback
                
                # Calculate distance from land zone center
                distance = math.sqrt(swimmer_pos[0]**2 + swimmer_pos[1]**2)
                
                positions.append(swimmer_pos.copy())
                distances_from_center.append(distance)
                max_distance = max(max_distance, distance)
                
                # Check for boundary crossing
                if distance > zone_config['radius']:
                    boundary_crossings.append((step, distance, swimmer_pos))
                
                # Progress updates
                if step % 50 == 0 and step > 0:
                    boundary_status = "OUTSIDE" if distance > zone_config['radius'] else "inside"
                    print(f"   Step {step:3d}: pos=({swimmer_pos[0]:+.3f}, {swimmer_pos[1]:+.3f}), " +
                          f"distance={distance:.3f}m, status={boundary_status}")
        
        # Final results
        final_pos = positions[-1] if positions else [0, 0]
        final_distance = distances_from_center[-1] if distances_from_center else 0
        
        print(f"   📊 Results:")
        print(f"      Final position: ({final_pos[0]:+.3f}, {final_pos[1]:+.3f})")
        print(f"      Final distance: {final_distance:.3f}m")
        print(f"      Max distance reached: {max_distance:.3f}m")
        print(f"      Zone radius: {zone_config['radius']:.3f}m")
        
        if boundary_crossings:
            print(f"      ✅ BOUNDARY CROSSED! {len(boundary_crossings)} times")
            first_crossing = boundary_crossings[0]
            print(f"      First escape at step {first_crossing[0]}: distance {first_crossing[1]:.3f}m")
        else:
            deficit = zone_config['radius'] - max_distance
            percentage = (max_distance / zone_config['radius']) * 100
            print(f"      ❌ Never escaped zone")
            print(f"      Closest approach: {percentage:.1f}% of boundary distance")
            print(f"      Deficit: {deficit:.3f}m short of escape")
        
        print()
    
    print("🔍 Diagnostic Summary:")
    print("   This test shows the swimmer's actual movement capability")
    print("   vs. the land zone sizes to determine if boundary crossing")
    print("   is physically possible with the current agent.")


if __name__ == "__main__":
    run_diagnostic_test() 