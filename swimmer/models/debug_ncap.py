#!/usr/bin/env python3
"""
Debug NCAP Forward Pass
Isolate the exact source of NaN values in NCAP computation.
"""

import torch
import torch.nn as nn
import numpy as np
from .ncap_swimmer import NCAPSwimmer

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
    
    # Step 5: Test individual B-neuron computations
    print("\n--- Step 5: B-neuron Computations ---")
    
    # Head oscillator contribution (if enabled)
    if ncap.include_head_oscillators:
        osc_contrib_d = ncap.exc(ncap.params['bneuron_osc']) * oscillator_tensor
        osc_contrib_v = ncap.exc(ncap.params['bneuron_osc']) * oscillator_tensor
        print(f"Head oscillator contribution (d): {osc_contrib_d}")
        print(f"Head oscillator contribution (v): {osc_contrib_v}")
        print(f"Osc contrib contains NaN: {torch.isnan(osc_contrib_d).any() or torch.isnan(osc_contrib_v).any()}")
    
    # Proprioception contribution (for joints > 0)
    if ncap.include_proprioception and ncap.n_joints > 1:
        prop_contrib_d = ncap.exc(ncap.params['bneuron_prop']) * joint_pos_d[:-1]  # Previous joint
        prop_contrib_v = ncap.exc(ncap.params['bneuron_prop']) * joint_pos_v[:-1]
        print(f"Proprioception contribution (d): {prop_contrib_d}")
        print(f"Proprioception contribution (v): {prop_contrib_v}")
        print(f"Prop contrib contains NaN: {torch.isnan(prop_contrib_d).any() or torch.isnan(prop_contrib_v).any()}")
    
    # Step 6: Test muscle computation
    print("\n--- Step 6: Muscle Computation ---")
    
    # Compute B-neurons for first joint (simplified)
    if ncap.include_head_oscillators:
        bneuron_d_0 = osc_contrib_d
        bneuron_v_0 = osc_contrib_v
    else:
        bneuron_d_0 = torch.zeros(1, device=device)
        bneuron_v_0 = torch.zeros(1, device=device)
    
    print(f"B-neuron d[0]: {bneuron_d_0}")
    print(f"B-neuron v[0]: {bneuron_v_0}")
    print(f"B-neurons contain NaN: {torch.isnan(bneuron_d_0).any() or torch.isnan(bneuron_v_0).any()}")
    
    # Apply graded activation to B-neurons
    bneuron_d_0_graded = bneuron_d_0.clamp(min=0, max=1)
    bneuron_v_0_graded = bneuron_v_0.clamp(min=0, max=1)
    print(f"B-neuron d[0] graded: {bneuron_d_0_graded}")
    print(f"B-neuron v[0] graded: {bneuron_v_0_graded}")
    print(f"Graded B-neurons contain NaN: {torch.isnan(bneuron_d_0_graded).any() or torch.isnan(bneuron_v_0_graded).any()}")
    
    # Compute muscles
    muscle_d_0 = ncap.exc(ncap.params['muscle_ipsi']) * bneuron_d_0_graded + ncap.inh(ncap.params['muscle_contra']) * bneuron_v_0_graded
    muscle_v_0 = ncap.exc(ncap.params['muscle_ipsi']) * bneuron_v_0_graded + ncap.inh(ncap.params['muscle_contra']) * bneuron_d_0_graded
    
    print(f"Raw muscle d[0]: {muscle_d_0}")
    print(f"Raw muscle v[0]: {muscle_v_0}")
    print(f"Raw muscles contain NaN: {torch.isnan(muscle_d_0).any() or torch.isnan(muscle_v_0).any()}")
    
    # Apply graded activation to muscles
    muscle_d_0_graded = muscle_d_0.clamp(min=0, max=1)
    muscle_v_0_graded = muscle_v_0.clamp(min=0, max=1)
    print(f"Graded muscle d[0]: {muscle_d_0_graded}")
    print(f"Graded muscle v[0]: {muscle_v_0_graded}")
    print(f"Graded muscles contain NaN: {torch.isnan(muscle_d_0_graded).any() or torch.isnan(muscle_v_0_graded).any()}")
    
    # Final output computation
    output_0 = muscle_d_0_graded - muscle_v_0_graded
    print(f"Final output[0]: {output_0}")
    print(f"Final output contains NaN: {torch.isnan(output_0).any()}")
    
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
    
    # Step 8: Test with environment adaptation
    print("\n--- Step 8: Environment Adaptation ---")
    if ncap.include_environment_adaptation:
        try:
            environment_type = np.array([1.0, 0.0])  # Water environment
            with torch.no_grad():
                env_output = ncap(joint_pos, environment_type=environment_type)
                print(f"NCAP output with environment: {env_output}")
                print(f"Environment output contains NaN: {torch.isnan(env_output).any()}")
        except Exception as e:
            print(f"ERROR with environment adaptation: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n=== DEBUG COMPLETE ===")

def test_parameter_initialization():
    """Test different parameter initialization strategies."""
    print("\n=== TESTING PARAMETER INITIALIZATION ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test 1: Default initialization
    print("\n--- Test 1: Default Initialization ---")
    ncap1 = NCAPSwimmer(n_joints=5, oscillator_period=60, memory_size=10)
    ncap1.to(device)
    
    test_input = torch.randn(5, device=device) * 0.1  # Small random input
    try:
        output1 = ncap1(test_input)
        print(f"Default init output: {output1}")
        print(f"Contains NaN: {torch.isnan(output1).any()}")
    except Exception as e:
        print(f"Default init failed: {e}")
    
    # Test 2: Conservative initialization
    print("\n--- Test 2: Conservative Initialization ---")
    ncap2 = NCAPSwimmer(n_joints=5, oscillator_period=60, memory_size=10)
    
    # Apply conservative initialization
    with torch.no_grad():
        for name, param in ncap2.params.items():
            if 'muscle' in name or 'bneuron' in name:
                nn.init.constant_(param, 0.5)  # Safe constant value
    
    ncap2.to(device)
    
    try:
        output2 = ncap2(test_input)
        print(f"Conservative init output: {output2}")
        print(f"Contains NaN: {torch.isnan(output2).any()}")
    except Exception as e:
        print(f"Conservative init failed: {e}")
    
    print("\n=== INITIALIZATION TEST COMPLETE ===")

if __name__ == "__main__":
    debug_ncap_forward()
    test_parameter_initialization() 