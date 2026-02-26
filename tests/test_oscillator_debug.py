#!/usr/bin/env python3
"""
Debug oscillator behavior specifically
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from swimmer.models.ncap_swimmer import NCAPSwimmer
from swimmer.models.proper_ncap import ProperSwimmerModule

def debug_oscillator():
    """Debug oscillator behavior step by step."""
    print("=== DEBUGGING OSCILLATOR BEHAVIOR ===")
    
    # Test both implementations
    our_ncap = NCAPSwimmer(n_joints=5, oscillator_period=60, include_environment_adaptation=False)
    proper_ncap = ProperSwimmerModule(n_joints=5, oscillator_period=60)
    
    # Test with fixed joint positions (no feedback)
    joint_pos = torch.zeros(5)
    
    print("Testing oscillator with zero joint positions...")
    
    our_outputs = []
    proper_outputs = []
    our_internal_timesteps = []
    proper_internal_timesteps = []
    
    for t in range(120):  # Test 2 full cycles (60*2)
        # Our NCAP
        our_output = our_ncap(joint_pos.numpy())
        if torch.is_tensor(our_output):
            our_output = our_output.detach().cpu().numpy()
        our_outputs.append(our_output[0])
        our_internal_timesteps.append(our_ncap.timestep - 1)  # -1 because timestep increments after forward
        
        # Proper NCAP
        proper_output = proper_ncap(joint_pos)
        proper_outputs.append(proper_output[0].item())
        proper_internal_timesteps.append(proper_ncap.timestep - 1)
        
        # Print some debug info
        if t < 10 or t % 10 == 0:
            print(f"  t={t}: Our_timestep={our_ncap.timestep-1}, Our_output={our_output[0]:.3f}, Proper_timestep={proper_ncap.timestep-1}, Proper_output={proper_output[0].item():.3f}")
    
    our_outputs = np.array(our_outputs)
    proper_outputs = np.array(proper_outputs)
    
    print(f"\nOur NCAP: Range=[{our_outputs.min():.3f}, {our_outputs.max():.3f}], Std={our_outputs.std():.3f}")
    print(f"Proper NCAP: Range=[{proper_outputs.min():.3f}, {proper_outputs.max():.3f}], Std={proper_outputs.std():.3f}")
    
    # Test explicit oscillator logic
    print(f"\n=== TESTING EXPLICIT OSCILLATOR LOGIC ===")
    
    def test_square_wave_logic(timestep, period):
        """Test square wave logic manually."""
        phase = timestep % period
        if phase < period // 2:
            return 1.0, 0.0  # dorsal, ventral
        else:
            return 0.0, 1.0
    
    print("Manual square wave test (period=60):")
    for t in [0, 10, 20, 29, 30, 40, 50, 59, 60, 70]:
        dorsal, ventral = test_square_wave_logic(t, 60)
        print(f"  t={t}: phase={t%60}, dorsal={dorsal}, ventral={ventral}")
    
    # Test our NCAP's internal oscillator logic
    print(f"\n=== TESTING OUR NCAP INTERNAL LOGIC ===")
    
    # Reset and test first joint specifically
    our_ncap.reset()
    
    joint_pos_zero = torch.zeros(5)
    
    print("Testing first joint (head oscillator) over time:")
    for t in range(10):
        # Call forward to advance timestep
        output = our_ncap(joint_pos_zero.numpy())
        
        # Check internal state
        internal_t = our_ncap.timestep - 1  # Already incremented
        phase = internal_t % 60
        expected_dorsal = 1.0 if phase < 30 else 0.0
        expected_ventral = 0.0 if phase < 30 else 1.0
        
        print(f"  t={t}, internal_t={internal_t}, phase={phase}, expected_d={expected_dorsal}, expected_v={expected_ventral}, actual_output={output[0]:.3f}")
    
    # Test proper NCAP's internal logic
    print(f"\n=== TESTING PROPER NCAP INTERNAL LOGIC ===")
    
    proper_ncap.reset()
    
    print("Testing proper NCAP first joint:")
    for t in range(10):
        output = proper_ncap(joint_pos_zero)
        
        internal_t = proper_ncap.timestep - 1
        phase = internal_t % 60
        
        print(f"  t={t}, internal_t={internal_t}, phase={phase}, actual_output={output[0].item():.3f}")
    
    # Test with explicit timesteps parameter
    print(f"\n=== TESTING WITH EXPLICIT TIMESTEPS ===")
    
    our_ncap.reset()
    proper_ncap.reset()
    
    print("Testing with explicit timesteps (bypassing internal counter):")
    for t in range(10):
        timesteps_tensor = torch.tensor([t], dtype=torch.float32)
        
        # Our NCAP with explicit timesteps
        our_output = our_ncap(joint_pos_zero.numpy(), timesteps=timesteps_tensor)
        if torch.is_tensor(our_output):
            our_output = our_output.detach().cpu().numpy()
        
        # Proper NCAP with explicit timesteps  
        proper_output = proper_ncap(joint_pos_zero, timesteps=timesteps_tensor)
        
        print(f"  explicit_t={t}: Our_output={our_output[0]:.3f}, Proper_output={proper_output[0].item():.3f}")
    
    # Analyze patterns
    def analyze_pattern(outputs, name):
        print(f"\n{name} Pattern Analysis:")
        
        # Check for constant values
        if np.std(outputs) < 0.01:
            print(f"  ❌ Constant output (std={np.std(outputs):.4f})")
            return False
        
        # Check for alternating pattern
        changes = 0
        for i in range(1, len(outputs)):
            if abs(outputs[i] - outputs[i-1]) > 0.5:
                changes += 1
        
        print(f"  Changes detected: {changes}")
        
        # Look for periodicity
        if len(outputs) >= 60:
            first_half = outputs[:30]
            second_half = outputs[30:60] 
            correlation = np.corrcoef(first_half, second_half)[0, 1] if len(first_half) == len(second_half) else 0
            print(f"  Half-period correlation: {correlation:.3f}")
        
        return changes > 0
    
    our_working = analyze_pattern(our_outputs, "Our NCAP")
    proper_working = analyze_pattern(proper_outputs, "Proper NCAP")
    
    print(f"\n=== SUMMARY ===")
    print(f"Our NCAP oscillating: {'✅' if our_working else '❌'}")
    print(f"Proper NCAP oscillating: {'✅' if proper_working else '❌'}")
    
    if not our_working and not proper_working:
        print("❌ Neither implementation is oscillating!")
        print("   Possible causes:")
        print("   - Oscillator logic not being triggered")
        print("   - Head oscillator condition not met")
        print("   - Muscle computation canceling out oscillations")
        print("   - Internal timestep not advancing correctly")
    
    return {
        'our_oscillating': our_working,
        'proper_oscillating': proper_working,
        'our_std': np.std(our_outputs),
        'proper_std': np.std(proper_outputs)
    }

if __name__ == "__main__":
    results = debug_oscillator() 