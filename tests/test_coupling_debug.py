#!/usr/bin/env python3
"""
Debug proprioceptive coupling in NCAP models.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from swimmer.models.simple_ncap import SimpleNCAPSwimmer


def debug_coupling():
    """Debug the proprioceptive coupling mechanism."""
    print("🔍 Debugging NCAP Proprioceptive Coupling...")
    
    # Create simplified NCAP model
    ncap = SimpleNCAPSwimmer(
        n_joints=5,
        oscillator_period=60,
        use_weight_sharing=True,
        use_weight_constraints=True,
        include_proprioception=True,
        include_head_oscillators=True
    )
    
    print(f"📊 NCAP parameters:")
    for name, param in ncap.params.items():
        print(f"  {name}: {param.item():.3f}")
    
    # Test with different joint positions to see coupling
    test_cases = [
        torch.zeros(5),  # All joints at zero
        torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0]),  # First joint active
        torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0]),  # Second joint active
        torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0]),  # First two joints active
    ]
    
    print("\n🧪 Testing different joint configurations:")
    
    for i, joint_pos in enumerate(test_cases):
        print(f"\nTest {i+1}: Joint positions = {joint_pos.numpy()}")
        
        # Test at step 0 (oscillator_d = 1.0)
        with torch.no_grad():
            action = ncap(joint_pos, timesteps=torch.tensor([0], dtype=torch.float32))
            print(f"  Actions at step 0: {action.numpy()}")
        
        # Test at step 30 (oscillator_v = 1.0)
        ncap.reset()  # Reset timestep
        with torch.no_grad():
            action = ncap(joint_pos, timesteps=torch.tensor([30], dtype=torch.float32))
            print(f"  Actions at step 30: {action.numpy()}")
    
    # Test oscillator propagation over time
    print("\n🔄 Testing oscillator propagation over time...")
    
    ncap.reset()
    joint_pos = torch.zeros(5)  # Start with zero positions
    actions_over_time = []
    joint_positions_over_time = []
    
    for step in range(120):  # 2 oscillator periods
        with torch.no_grad():
            action = ncap(joint_pos, timesteps=torch.tensor([step], dtype=torch.float32))
            actions_over_time.append(action.numpy().copy())
            
            # Simulate that joint position changes based on action (simple integration)
            joint_pos = joint_pos + 0.1 * action  # Simple position update
            joint_positions_over_time.append(joint_pos.numpy().copy())
    
    actions_over_time = np.array(actions_over_time)
    joint_positions_over_time = np.array(joint_positions_over_time)
    
    # Analyze propagation
    print(f"\nPropagation Analysis:")
    for joint in range(5):
        action_range = actions_over_time[:, joint].max() - actions_over_time[:, joint].min()
        print(f"  Joint {joint} action range: {action_range:.3f}")
    
    # Plot detailed analysis
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot actions over time
    for joint in range(5):
        axes[0].plot(actions_over_time[:, joint], label=f'Joint {joint}')
    axes[0].set_title('Actions Over Time (With Position Updates)')
    axes[0].set_ylabel('Action Value')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot joint positions over time
    for joint in range(5):
        axes[1].plot(joint_positions_over_time[:, joint], label=f'Joint {joint}')
    axes[1].set_title('Joint Positions Over Time')
    axes[1].set_ylabel('Joint Position')
    axes[1].legend()
    axes[1].grid(True)
    
    # Plot phase relationships
    if actions_over_time[:, 0].max() - actions_over_time[:, 0].min() > 0.1:
        for joint in range(1, 5):
            if actions_over_time[:, joint].max() - actions_over_time[:, joint].min() > 0.1:
                # Calculate phase lag
                correlation = np.correlate(actions_over_time[:, 0], actions_over_time[:, joint], mode='full')
                lag = np.argmax(correlation) - len(actions_over_time[:, 0]) + 1
                axes[2].bar(joint, lag, alpha=0.7)
        axes[2].set_title('Phase Lag Relative to Joint 0')
        axes[2].set_xlabel('Joint')
        axes[2].set_ylabel('Phase Lag (steps)')
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/coupling_debug_analysis.png', dpi=150, bbox_inches='tight')
    print("📊 Coupling analysis saved to 'outputs/coupling_debug_analysis.png'")


def test_manual_coupling():
    """Test coupling with manually set strong parameters."""
    print("\n🔧 Testing with manually strengthened coupling...")
    
    ncap = SimpleNCAPSwimmer(n_joints=5)
    
    # Manually strengthen coupling
    with torch.no_grad():
        ncap.params['bneuron_prop'].data = torch.tensor(3.0)  # Strengthen proprioception
        ncap.params['muscle_ipsi'].data = torch.tensor(2.0)   # Strengthen muscles
        ncap.params['muscle_contra'].data = torch.tensor(-2.0)
    
    print(f"📊 Strengthened parameters:")
    for name, param in ncap.params.items():
        print(f"  {name}: {param.item():.3f}")
    
    # Test oscillation
    joint_pos = torch.zeros(5)
    actions = []
    
    for step in range(120):
        with torch.no_grad():
            action = ncap(joint_pos, timesteps=torch.tensor([step], dtype=torch.float32))
            actions.append(action.numpy().copy())
            # Update joint positions based on actions
            joint_pos = joint_pos + 0.1 * action
    
    actions = np.array(actions)
    
    print(f"\nStrengthened coupling results:")
    for joint in range(5):
        action_range = actions[:, joint].max() - actions[:, joint].min()
        print(f"  Joint {joint} range: {action_range:.3f}")
    
    return actions


if __name__ == "__main__":
    # Debug standard coupling
    debug_coupling()
    
    # Test with strengthened coupling
    strengthened_actions = test_manual_coupling()
    
    print("\n🎯 Diagnosis:")
    print("If strengthened coupling works → need stronger initialization")
    print("If it still doesn't work → algorithmic coupling issue") 