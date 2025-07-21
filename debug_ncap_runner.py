#!/usr/bin/env python3
"""
NCAP Debug Runner
Standalone script to debug NCAP issues.
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from swimmer.models.ncap_swimmer import NCAPSwimmer
import torch
import torch.nn as nn
import numpy as np

def debug_ncap_forward():
    """Debug NCAP forward pass step by step to find NaN source."""
    print("=== DEBUGGING NCAP FORWARD PASS ===")
    
    # Create NCAP model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    ncap = NCAPSwimmer(n_joints=5, oscillator_period=60, memory_size=10)
    ncap.to(device)
    
    # Test input (typical joint positions)
    joint_pos = torch.tensor([0.1, -0.2, 0.3, -0.1, 0.2], dtype=torch.float32, device=device)
    print(f"Input joint_pos: {joint_pos}")
    print(f"Input device: {joint_pos.device}")
    print(f"Input contains NaN: {torch.isnan(joint_pos).any()}")
    
    # Step 1: Check input normalization
    print("\n--- Step 1: Input Normalization ---")
    joint_limit = 2 * np.pi / (ncap.n_joints + 1)
    print(f"Joint limit: {joint_limit}")
    joint_pos_norm = torch.clamp(joint_pos / joint_limit, min=-1, max=1)
    print(f"Normalized joint_pos: {joint_pos_norm}")
    print(f"Normalized contains NaN: {torch.isnan(joint_pos_norm).any()}")
    
    # Step 2: Check graded activation (dorsal/ventral separation)
    print("\n--- Step 2: Graded Activation ---")
    joint_pos_d = joint_pos_norm.clamp(min=0, max=1)  # Dorsal
    joint_pos_v = (-joint_pos_norm).clamp(min=0, max=1)  # Ventral
    print(f"Dorsal (d): {joint_pos_d}")
    print(f"Ventral (v): {joint_pos_v}")
    print(f"Dorsal contains NaN: {torch.isnan(joint_pos_d).any()}")
    print(f"Ventral contains NaN: {torch.isnan(joint_pos_v).any()}")
    
    # Step 3: Check oscillator computation
    print("\n--- Step 3: Oscillator Computation ---")
    timestep = 30  # Test with specific timestep
    oscillator_val = 1.0 if (timestep % ncap.oscillator_period) < (ncap.oscillator_period // 2) else -1.0
    print(f"Timestep: {timestep}, Period: {ncap.oscillator_period}")
    print(f"Oscillator value: {oscillator_val}")
    oscillator_tensor = torch.tensor([oscillator_val], dtype=torch.float32, device=device)
    print(f"Oscillator tensor: {oscillator_tensor}")
    print(f"Oscillator contains NaN: {torch.isnan(oscillator_tensor).any()}")
    
    # Step 4: Check NCAP parameter values
    print("\n--- Step 4: NCAP Parameters ---")
    for name, param in ncap.params.items():
        print(f"{name}: {param.data}")
        print(f"{name} contains NaN: {torch.isnan(param.data).any()}")
        print(f"{name} contains Inf: {torch.isinf(param.data).any()}")
        if param.data.abs().max() > 1000:
            print(f"WARNING: {name} has very large values: {param.data.abs().max()}")
    
    # Step 7: Test full forward pass
    print("\n--- Step 7: Full Forward Pass ---")
    try:
        with torch.no_grad():
            full_output = ncap(joint_pos)
            print(f"Full NCAP output: {full_output}")
            print(f"Full output contains NaN: {torch.isnan(full_output).any()}")
            print(f"Output range: [{full_output.min():.3f}, {full_output.max():.3f}]")
    except Exception as e:
        print(f"ERROR in full forward pass: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== DEBUG COMPLETE ===")

def test_simple_fix():
    """Test a simple fix for NCAP stability."""
    print("\n=== TESTING SIMPLE NCAP FIX ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create NCAP with stable initialization
    ncap = NCAPSwimmer(n_joints=5, oscillator_period=60, memory_size=10, 
                       include_environment_adaptation=False)  # Disable complex features first
    
    # Apply very conservative initialization
    with torch.no_grad():
        for name, param in ncap.params.items():
            if 'muscle' in name or 'bneuron' in name:
                nn.init.constant_(param, 0.5)  # Safe constant value
    
    ncap.to(device)
    
    # Test with simple input
    for i in range(10):
        test_input = torch.randn(5, device=device) * 0.01  # Very small random input
        
        try:
            with torch.no_grad():
                output = ncap(test_input)
                has_nan = torch.isnan(output).any()
                print(f"Test {i+1}: Input max={test_input.abs().max():.4f}, "
                      f"Output max={output.abs().max():.4f}, Has NaN: {has_nan}")
                
                if has_nan:
                    print(f"  NaN locations: {torch.isnan(output)}")
                    break
        except Exception as e:
            print(f"Test {i+1} failed: {e}")
            break
    
    print("=== SIMPLE FIX TEST COMPLETE ===")

if __name__ == "__main__":
    debug_ncap_forward()
    test_simple_fix() 