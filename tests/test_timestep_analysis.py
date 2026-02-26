#!/usr/bin/env python3
"""
Analyze timestep and oscillator synchronization issues
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from swimmer.models.ncap_swimmer import NCAPSwimmer, NCAPSwimmerActor
from swimmer.models.proper_ncap import ProperSwimmerModule
from dm_control import suite
from dm_control.suite import swimmer

def analyze_timestep_issues():
    """Analyze timestep and oscillator behavior."""
    print("=== ANALYZING TIMESTEP AND OSCILLATOR ISSUES ===")
    
    # Test 1: Check environment timestep
    print("\n1. Checking environment timestep...")
    env = suite.load('swimmer', 'swimmer6')
    timestep = env.reset()
    
    control_timestep = swimmer._CONTROL_TIMESTEP
    time_limit = 30  # seconds
    max_steps = int(time_limit / control_timestep)
    
    print(f"Control timestep: {control_timestep}")
    print(f"Time limit: {time_limit}s")
    print(f"Max steps: {max_steps}")
    print(f"Expected steps per second: {1/control_timestep}")
    
    # Test 2: Compare oscillator behavior
    print("\n2. Comparing oscillator behavior...")
    
    # Our NCAP
    our_ncap = NCAPSwimmer(
        n_joints=5,
        oscillator_period=60,
        include_environment_adaptation=False
    )
    
    # Proper NCAP
    proper_ncap = ProperSwimmerModule(
        n_joints=5,
        oscillator_period=60,
        include_head_oscillators=True,
        include_proprioception=True
    )
    
    # Test oscillator outputs over time
    joint_positions = torch.zeros(5)
    our_outputs = []
    proper_outputs = []
    timesteps = []
    
    print("Testing oscillator outputs over 200 timesteps...")
    
    for t in range(200):
        # Our NCAP
        our_output = our_ncap(joint_positions.numpy())
        if torch.is_tensor(our_output):
            our_output = our_output.detach().cpu().numpy()
        our_outputs.append(our_output[0])  # First joint
        
        # Proper NCAP
        proper_output = proper_ncap(joint_positions)
        proper_outputs.append(proper_output[0].item())  # First joint
        
        timesteps.append(t)
        
        # Slight position update to simulate feedback
        our_val = our_output[0] if not torch.is_tensor(our_output) else our_output[0]
        joint_positions[0] += 0.01 * our_val
        joint_positions = torch.clamp(joint_positions, -0.5, 0.5)
    
    our_outputs = np.array(our_outputs)
    proper_outputs = np.array(proper_outputs)
    
    print(f"Our NCAP oscillator - Range: [{our_outputs.min():.3f}, {our_outputs.max():.3f}], Std: {our_outputs.std():.3f}")
    print(f"Proper NCAP oscillator - Range: [{proper_outputs.min():.3f}, {proper_outputs.max():.3f}], Std: {proper_outputs.std():.3f}")
    
    # Test 3: Check oscillator frequency
    print("\n3. Analyzing oscillator frequency...")
    
    # Look for oscillation period in outputs
    def find_oscillation_period(signal):
        """Find the period of oscillation in a signal."""
        # Simple peak detection
        peaks = []
        for i in range(1, len(signal)-1):
            if signal[i] > signal[i-1] and signal[i] > signal[i+1] and signal[i] > 0:
                peaks.append(i)
        
        if len(peaks) > 1:
            periods = np.diff(peaks)
            avg_period = np.mean(periods) if len(periods) > 0 else 0
            return avg_period, peaks
        return 0, peaks
    
    our_period, our_peaks = find_oscillation_period(our_outputs)
    proper_period, proper_peaks = find_oscillation_period(proper_outputs)
    
    print(f"Our NCAP oscillation period: {our_period:.1f} timesteps (peaks at: {our_peaks[:5]}...)")
    print(f"Proper NCAP oscillation period: {proper_period:.1f} timesteps (peaks at: {proper_peaks[:5]}...)")
    print(f"Expected period: 60 timesteps")
    
    # Test 4: Check parameter values
    print("\n4. Comparing parameter values...")
    print("Our NCAP parameters:")
    for name, param in our_ncap.params.items():
        print(f"  {name}: {param.item():.3f}")
    
    print("Proper NCAP parameters:")
    for name, param in proper_ncap.params.items():
        print(f"  {name}: {param.item():.3f}")
    
    # Test 5: Test with different joint normalization
    print("\n5. Testing joint position normalization...")
    
    test_joint_pos = np.array([0.1, -0.1, 0.2, -0.2, 0.1])  # Example joint positions
    
    # Our normalization
    joint_limit_ours = 2 * np.pi / (5 + 1)  # n_joints + 1
    normalized_ours = np.clip(test_joint_pos / joint_limit_ours, -1, 1)
    
    # Notebook normalization (what we think it should be)
    joint_limit_notebook = 2 * np.pi / (5 + 1)  
    normalized_notebook = np.clip(test_joint_pos / joint_limit_notebook, -1, 1)
    
    print(f"Test joint positions: {test_joint_pos}")
    print(f"Joint limit (ours): {joint_limit_ours:.3f}")
    print(f"Normalized (ours): {normalized_ours}")
    print(f"Joint limit (notebook): {joint_limit_notebook:.3f}")
    print(f"Normalized (notebook): {normalized_notebook}")
    
    # Test 6: Environment scaling analysis
    print("\n6. Analyzing environment scaling...")
    
    # Run short test in standard environment
    actor = NCAPSwimmerActor(our_ncap)
    timestep = env.reset()
    our_ncap.reset()
    
    positions = []
    actions = []
    joint_positions = []
    
    for step in range(100):
        obs = timestep.observation
        joint_pos = obs['joints']
        
        action = actor(joint_pos)
        timestep = env.step(action)
        
        # Track data
        current_pos = env.physics.named.data.xpos['head']
        positions.append(current_pos[:2].copy())
        actions.append(action.copy())
        joint_positions.append(joint_pos.copy())
        
        if step < 10:  # Print first few steps
            print(f"  Step {step}: Joint_pos_range=[{joint_pos.min():.3f}, {joint_pos.max():.3f}], Action_range=[{action.min():.3f}, {action.max():.3f}], Pos={current_pos[:2]}")
    
    positions = np.array(positions)
    actions = np.array(actions)
    joint_positions = np.array(joint_positions)
    
    print(f"Joint position range over 100 steps: [{joint_positions.min():.3f}, {joint_positions.max():.3f}]")
    print(f"Action range over 100 steps: [{actions.min():.3f}, {actions.max():.3f}]")
    print(f"Position change over 100 steps: {np.linalg.norm(positions[-1] - positions[0]):.3f}")
    
    # Analysis and recommendations
    print(f"\n=== ANALYSIS ===")
    
    issues_found = []
    
    if our_period == 0 or abs(our_period - 60) > 10:
        issues_found.append(f"Oscillation period wrong: {our_period:.1f} vs expected 60")
    
    if our_outputs.std() < 0.3:
        issues_found.append(f"Low oscillation amplitude: std={our_outputs.std():.3f}")
    
    if np.max(np.abs(actions)) < 0.5:
        issues_found.append(f"Low action magnitude: max={np.max(np.abs(actions)):.3f}")
    
    if len(issues_found) > 0:
        print("Issues found:")
        for issue in issues_found:
            print(f"  ❌ {issue}")
    else:
        print("✅ No obvious issues found - problem may be elsewhere")
    
    return {
        'our_oscillator_std': our_outputs.std(),
        'proper_oscillator_std': proper_outputs.std(),
        'our_period': our_period,
        'proper_period': proper_period,
        'action_range': [actions.min(), actions.max()],
        'position_change': np.linalg.norm(positions[-1] - positions[0])
    }

if __name__ == "__main__":
    results = analyze_timestep_issues() 