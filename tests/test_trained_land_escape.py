#!/usr/bin/env python3
"""
Trained Model Land Escape Test
Tests the trained enhanced_ncap model's ability to escape from land zones
using the properly sized zones discovered in diagnostic testing.
"""

import os
import sys
import numpy as np
import torch
import math
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from swimmer.environments.progressive_mixed_env import ProgressiveMixedSwimmerEnv
from swimmer.models.enhanced_biological_ncap import EnhancedBiologicalNCAPSwimmer


class TrainedModelAgent:
    """Agent wrapper for the trained enhanced_ncap model."""
    
    def __init__(self, model_path, n_links=5):
        self.n_links = n_links
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create the model with correct parameters
        self.model = EnhancedBiologicalNCAPSwimmer(
            n_joints=n_links-1,  # n_joints = n_links - 1 for swimmer
            oscillator_period=60,
            use_weight_sharing=True,
            use_weight_constraints=True,
            include_proprioception=True,
            include_head_oscillators=True,
            include_environment_adaptation=True,
            include_goal_direction=True
        )
        
        # Load trained weights
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                elif 'model' in checkpoint:
                    self.model.load_state_dict(checkpoint['model'])
                elif 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'])
                else:
                    # Try loading the checkpoint directly as state dict
                    self.model.load_state_dict(checkpoint)
            else:
                # Checkpoint is likely the model itself
                self.model = checkpoint
                
            print(f"   ✅ Loaded trained model from: {model_path}")
        except Exception as e:
            print(f"   ❌ Error loading model: {e}")
            print(f"   Available keys in checkpoint: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'Not a dict'}")
            raise
        
        self.model.eval()  # Set to evaluation mode
        
    def test_step(self, observation):
        """Get action from trained model."""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            action = self.model(obs_tensor)
            return action.cpu().numpy().flatten()


def test_trained_land_escape():
    """Test trained model's land escape capabilities."""
    
    print("🤖 Trained Model Land Escape Test")
    print("=" * 60)
    print("Testing trained enhanced_ncap model's ability to escape land zones")
    print("using optimally sized zones discovered in diagnostic testing")
    print()
    
    # Create environment
    env = ProgressiveMixedSwimmerEnv(n_links=5, time_limit=3000)
    
    # Model path - use the checkpoint specified by user
    model_path = "outputs/curriculum_training/checkpoints/enhanced_ncap/40000.pt"
    
    # Test zones - using sizes where random agent succeeded
    test_zones = [
        {"radius": 0.08, "steps": 200, "name": "Super Tiny Zone (Random Agent: ✅)"},
        {"radius": 0.12, "steps": 250, "name": "Tiny Zone (Random Agent: ✅)"}, 
        {"radius": 0.2, "steps": 300, "name": "Small Zone (Random Agent: ✅)"},
        {"radius": 0.3, "steps": 400, "name": "Medium Zone (Random Agent: ✅)"},
        {"radius": 0.5, "steps": 500, "name": "Large Zone (Testing Trained Model)"},
        {"radius": 0.75, "steps": 600, "name": "Extra Large Zone (Training Challenge)"}
    ]
    
    try:
        agent = TrainedModelAgent(model_path)
    except Exception as e:
        print(f"❌ Failed to load trained model: {e}")
        print("   This suggests the model file may not exist or be corrupted.")
        print("   You may need to re-train the model or check the file path.")
        return
    
    print(f"🎯 Testing {len(test_zones)} different land zone sizes...")
    print()
    
    results = []
    
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
        
        # Track position and distances
        positions = []
        distances_from_center = []
        boundary_crossings = []
        time_outside = 0
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
                try:
                    swimmer_pos = physics.named.data.xpos['head'][:2]
                except KeyError:
                    try:
                        swimmer_pos = physics.named.data.geom_xpos['head'][:2]
                    except KeyError:
                        first_body = list(physics.named.data.xpos.keys())[0]
                        swimmer_pos = physics.named.data.xpos[first_body][:2]
                
                # Calculate distance from land zone center
                distance = math.sqrt(swimmer_pos[0]**2 + swimmer_pos[1]**2)
                
                positions.append(swimmer_pos.copy())
                distances_from_center.append(distance)
                max_distance = max(max_distance, distance)
                
                # Check for boundary crossing
                if distance > zone_config['radius']:
                    boundary_crossings.append((step, distance, swimmer_pos))
                    time_outside += 1
                
                # Progress updates
                if step % 100 == 0 and step > 0:
                    boundary_status = "OUTSIDE" if distance > zone_config['radius'] else "inside"
                    print(f"   Step {step:3d}: pos=({swimmer_pos[0]:+.3f}, {swimmer_pos[1]:+.3f}), " +
                          f"distance={distance:.3f}m, status={boundary_status}")
        
        # Calculate results
        final_pos = positions[-1] if positions else [0, 0]
        final_distance = distances_from_center[-1] if distances_from_center else 0
        escape_percentage = (time_outside / zone_config['steps']) * 100
        
        result = {
            'zone_name': zone_config['name'],
            'radius': zone_config['radius'],
            'max_distance': max_distance,
            'final_distance': final_distance,
            'boundary_crossings': len(boundary_crossings),
            'time_outside_percentage': escape_percentage,
            'successful_escape': len(boundary_crossings) > 0
        }
        results.append(result)
        
        print(f"   📊 Results:")
        print(f"      Final position: ({final_pos[0]:+.3f}, {final_pos[1]:+.3f})")
        print(f"      Final distance: {final_distance:.3f}m")
        print(f"      Max distance reached: {max_distance:.3f}m")
        print(f"      Zone radius: {zone_config['radius']:.3f}m")
        
        if result['successful_escape']:
            first_crossing = boundary_crossings[0]
            print(f"      ✅ ESCAPED! {len(boundary_crossings)} boundary crossings")
            print(f"      First escape at step {first_crossing[0]}: distance {first_crossing[1]:.3f}m")
            print(f"      Time outside zone: {escape_percentage:.1f}% of episode")
        else:
            deficit = zone_config['radius'] - max_distance
            percentage = (max_distance / zone_config['radius']) * 100
            print(f"      ❌ Never escaped zone")
            print(f"      Closest approach: {percentage:.1f}% of boundary distance")
            print(f"      Deficit: {deficit:.3f}m short of escape")
        
        print()
    
    # Summary
    print("🤖 Trained Model vs. Random Agent Comparison:")
    print("=" * 60)
    
    successful_escapes = sum(1 for r in results if r['successful_escape'])
    print(f"   Zones escaped: {successful_escapes}/{len(results)}")
    print()
    
    for result in results:
        status = "✅ ESCAPED" if result['successful_escape'] else "❌ TRAPPED"
        print(f"   {result['zone_name']}: {status}")
        print(f"      Max distance: {result['max_distance']:.3f}m (zone: {result['radius']:.3f}m)")
        if result['successful_escape']:
            print(f"      Escape success: {result['time_outside_percentage']:.1f}% of time outside")
    
    # Determine capabilities
    max_escaped_radius = max([r['radius'] for r in results if r['successful_escape']], default=0)
    print(f"\n   🎯 Trained model can escape zones up to: {max_escaped_radius:.3f}m radius")
    
    if max_escaped_radius >= 0.5:
        print("   🏆 EXCELLENT: Trained model significantly outperforms random agent!")
    elif max_escaped_radius >= 0.3:
        print("   ✅ GOOD: Trained model performs well on medium-sized zones")
    elif max_escaped_radius >= 0.2:
        print("   ⚠️ LIMITED: Trained model similar to random agent performance")
    else:
        print("   ❌ POOR: Trained model may need more training or debugging")
    
    return results


if __name__ == "__main__":
    test_trained_land_escape() 