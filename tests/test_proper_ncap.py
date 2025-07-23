#!/usr/bin/env python3
"""
Test script to compare proper NCAP implementation with current one
"""

import torch
import numpy as np
from swimmer.models.proper_ncap import ProperSwimmerModule, ProperSwimmerActor
from swimmer.environments.mixed_environment import test_improved_mixed_environment
from swimmer.models.ncap_swimmer import NCAPSwimmer
import time

def test_proper_ncap():
    """Test the proper NCAP implementation to establish correct baseline."""
    print("=== TESTING PROPER NCAP IMPLEMENTATION ===")
    
    # Test with proper NCAP from notebook
    print("Testing proper NCAP (from notebook)...")
    
    # Create proper swimmer with notebook architecture
    proper_swimmer = ProperSwimmerModule(
        n_joints=5,  # dm_control swimmer has 5 joints
        oscillator_period=60,
        use_weight_sharing=True,
        use_weight_constraints=True,
        include_proprioception=True,
        include_head_oscillators=True
    )
    
    # Test just the module output
    joint_pos = torch.randn(5) * 0.1  # Small joint positions
    output = proper_swimmer(joint_pos)
    print(f"Proper NCAP output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    print(f"Proper NCAP learnable parameters:")
    for name, param in proper_swimmer.params.items():
        print(f"  {name}: {param.item():.3f}")
    
    # Now test a simple motion test
    print("\n=== SIMPLE MOTION TEST ===")
    print("Testing 100 timesteps of oscillatory motion...")
    
    outputs = []
    for t in range(100):
        joint_pos = torch.randn(5) * 0.1
        output = proper_swimmer(joint_pos)
        outputs.append(output.detach().numpy())
    
    outputs = np.array(outputs)
    print(f"Motion range over 100 steps: [{outputs.min():.3f}, {outputs.max():.3f}]")
    print(f"Motion std: {outputs.std():.3f}")
    print(f"Output shape: {outputs.shape}")
    
    # Test our current implementation for comparison
    print("\n=== COMPARING WITH CURRENT IMPLEMENTATION ===")
    current_swimmer = NCAPSwimmer(n_joints=5, oscillator_period=60)
    
    current_outputs = []
    for t in range(100):
        joint_pos = torch.randn(5) * 0.1
        # Convert to batch format for our implementation
        obs = {
            'joints': joint_pos.numpy(),
            'environment_type': np.array([1.0, 0.0]),  # land environment
            'head_position': np.array([0.0, 0.0]),
            'in_water_zone': np.array([0.0]),
            'in_land_zone': np.array([1.0])
        }
        output = current_swimmer(joint_pos.numpy(), environment_type=obs['environment_type'])
        if torch.is_tensor(output):
            output = output.detach().cpu().numpy()  # Move to CPU before converting to numpy
        current_outputs.append(output)
    
    current_outputs = np.array(current_outputs)
    print(f"Current implementation range: [{current_outputs.min():.3f}, {current_outputs.max():.3f}]")
    print(f"Current implementation std: {current_outputs.std():.3f}")
    
    print("\n=== COMPARISON SUMMARY ===")
    print(f"Proper NCAP - Range: [{outputs.min():.3f}, {outputs.max():.3f}], Std: {outputs.std():.3f}")
    print(f"Current NCAP - Range: [{current_outputs.min():.3f}, {current_outputs.max():.3f}], Std: {current_outputs.std():.3f}")
    
    # The proper NCAP should have much more dynamic range if implemented correctly
    if outputs.std() > current_outputs.std() * 2:
        print("✅ Proper NCAP shows more dynamic behavior!")
    else:
        print("❌ Both implementations show similar behavior - issue may be elsewhere")

if __name__ == "__main__":
    test_proper_ncap() 