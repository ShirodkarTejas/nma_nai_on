#!/usr/bin/env python3
"""
Simplified test to establish proper NCAP baseline without dm_control complications
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from swimmer.models.proper_ncap import ProperSwimmerModule
from swimmer.models.ncap_swimmer import NCAPSwimmer

def test_simple_baseline():
    """Test proper NCAP with known good parameters."""
    print("=== SIMPLE BASELINE TEST ===")
    
    # Test 1: Proper NCAP from notebook
    print("1. Testing Proper NCAP (from notebook)...")
    proper_swimmer = ProperSwimmerModule(
        n_joints=5,
        oscillator_period=60,
        use_weight_sharing=True,
        use_weight_constraints=True,
        include_proprioception=True,
        include_head_oscillators=True
    )
    
    # Test oscillatory behavior over time
    proper_outputs = []
    joint_positions = torch.zeros(5)  # Start with zero joint positions
    
    for t in range(200):  # Test for 200 timesteps to see oscillations
        output = proper_swimmer(joint_positions)
        proper_outputs.append(output.detach().numpy())
        # Simulate some joint movement based on previous output
        joint_positions = joint_positions + 0.1 * output.detach()
        joint_positions = torch.clamp(joint_positions, -0.5, 0.5)  # Keep reasonable range
    
    proper_outputs = np.array(proper_outputs)
    
    print(f"Proper NCAP - Range: [{proper_outputs.min():.3f}, {proper_outputs.max():.3f}]")
    print(f"Proper NCAP - Std: {proper_outputs.std():.3f}")
    print(f"Proper NCAP - Final joint 0 output: {proper_outputs[-1, 0]:.3f}")
    
    # Test 2: Our current implementation
    print("\n2. Testing Our Current NCAP...")
    current_swimmer = NCAPSwimmer(n_joints=5, oscillator_period=60)
    
    current_outputs = []
    joint_positions_np = np.zeros(5)
    
    for t in range(200):
        # Create observation dict
        obs = {
            'joints': joint_positions_np,
            'environment_type': np.array([1.0, 0.0]),  # land environment
            'head_position': np.array([0.0, 0.0]),
            'in_water_zone': np.array([0.0]),
            'in_land_zone': np.array([1.0])
        }
        
        output = current_swimmer(joint_positions_np, environment_type=obs['environment_type'])
        if torch.is_tensor(output):
            output = output.detach().cpu().numpy()
        
        current_outputs.append(output)
        # Simulate joint movement
        joint_positions_np = joint_positions_np + 0.1 * output
        joint_positions_np = np.clip(joint_positions_np, -0.5, 0.5)
    
    current_outputs = np.array(current_outputs)
    
    print(f"Current NCAP - Range: [{current_outputs.min():.3f}, {current_outputs.max():.3f}]")
    print(f"Current NCAP - Std: {current_outputs.std():.3f}")
    print(f"Current NCAP - Final joint 0 output: {current_outputs[-1][0]:.3f}")
    
    # Test 3: Analyze oscillatory patterns
    print("\n3. Oscillatory Pattern Analysis...")
    
    # Check if proper NCAP shows clear oscillations
    proper_joint0 = proper_outputs[:, 0]
    proper_oscillations = np.abs(np.diff(proper_joint0)).mean()
    
    current_joint0 = current_outputs[:, 0]
    current_oscillations = np.abs(np.diff(current_joint0)).mean()
    
    print(f"Proper NCAP - Average oscillation magnitude: {proper_oscillations:.3f}")
    print(f"Current NCAP - Average oscillation magnitude: {current_oscillations:.3f}")
    
    # Test 4: Check parameter values
    print("\n4. Parameter Comparison...")
    print("Proper NCAP parameters:")
    for name, param in proper_swimmer.params.items():
        print(f"  {name}: {param.item():.3f}")
    
    print("Current NCAP parameters:")
    for name, param in current_swimmer.params.items():
        print(f"  {name}: {param.item():.3f}")
    print(f"Current NCAP oscillator period: {current_swimmer.oscillator_period}")
    print(f"Current NCAP n_joints: {current_swimmer.n_joints}")
    
    # Conclusion
    print(f"\n=== ANALYSIS ===")
    if proper_outputs.std() > 0.3 and np.abs(proper_outputs).max() > 0.5:
        print("✅ Proper NCAP shows good dynamic range and oscillations")
        print(f"   Expected performance: HIGH (range ~[-1, 1], std ~0.4)")
    else:
        print("❌ Proper NCAP also shows limited behavior")
    
    if current_outputs.std() > proper_outputs.std() * 3:
        print("⚠️  Current NCAP shows excessive/unbounded behavior")
        print("   This explains why swimming performance is poor")
    elif current_outputs.std() < proper_outputs.std() * 0.3:
        print("⚠️  Current NCAP shows insufficient oscillatory behavior") 
        print("   This explains why swimming performance is poor")
    else:
        print("❓ Current NCAP behavior is similar to proper NCAP")
        print("   The issue may be elsewhere in the system")
    
    return {
        'proper': {
            'range': [proper_outputs.min(), proper_outputs.max()],
            'std': proper_outputs.std(),
            'oscillations': proper_oscillations
        },
        'current': {
            'range': [current_outputs.min(), current_outputs.max()],
            'std': current_outputs.std(),
            'oscillations': current_oscillations
        }
    }

if __name__ == "__main__":
    results = test_simple_baseline()
    print(f"\nFinal comparison: Proper std={results['proper']['std']:.3f}, Current std={results['current']['std']:.3f}") 