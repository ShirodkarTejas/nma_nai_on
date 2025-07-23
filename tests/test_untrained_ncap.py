#!/usr/bin/env python3
"""
Test script to evaluate untrained NCAP architecture.
This helps isolate architectural issues from training issues.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from swimmer.models.ncap_swimmer import NCAPSwimmer
from swimmer.environments.mixed_environment import ImprovedMixedSwimmerEnv


def test_untrained_ncap():
    """Test the untrained NCAP model to see if it produces oscillatory behavior."""
    print("🧪 Testing untrained NCAP model...")
    
    # Create NCAP model
    ncap = NCAPSwimmer(
        n_joints=5,  # Match environment
        oscillator_period=60,
        memory_size=10,
        use_weight_sharing=True,
        use_weight_constraints=True,
        include_proprioception=True,
        include_head_oscillators=True,
        include_environment_adaptation=False  # Disable adaptation to test pure biology
    )
    
    # Create environment
    env = ImprovedMixedSwimmerEnv(n_links=6)
    obs = env.reset()
    
    print(f"📊 Initial NCAP parameters:")
    for name, param in ncap.params.items():
        print(f"  {name}: {param.item():.3f}")
    
    # Run simulation
    joint_positions = []
    actions = []
    timesteps = []
    
    print("🏊 Running untrained NCAP simulation...")
    
    for step in range(300):  # 5 seconds at 60 FPS
        # Extract joint positions
        joint_pos = torch.tensor(obs['joints'], dtype=torch.float32)
        joint_positions.append(joint_pos.numpy().copy())
        
        # Get NCAP action
        with torch.no_grad():
            action = ncap(joint_pos, timesteps=torch.tensor([step], dtype=torch.float32))
            actions.append(action.cpu().numpy().copy())
        
        timesteps.append(step)
        
        # Step environment
        obs, reward, done, info = env.step(action.cpu().numpy())
        
        if step % 60 == 0:  # Every second
            print(f"  Step {step}: Action range [{action.min():.3f}, {action.max():.3f}], "
                  f"Head velocity: {obs.get('head_velocity', 0):.3f}")
    
    # Analyze results
    joint_positions = np.array(joint_positions)
    actions = np.array(actions)
    
    print("\n📈 Analysis:")
    print(f"Action statistics:")
    print(f"  Mean: {actions.mean(axis=0)}")
    print(f"  Std:  {actions.std(axis=0)}")
    print(f"  Range: [{actions.min():.3f}, {actions.max():.3f}]")
    
    # Check for oscillatory behavior
    action_range = actions.max(axis=0) - actions.min(axis=0)
    avg_action_range = action_range.mean()
    print(f"  Average action range: {avg_action_range:.3f}")
    
    if avg_action_range > 0.1:
        print("✅ Model shows oscillatory behavior!")
    else:
        print("❌ Model shows minimal oscillatory behavior")
    
    # Plot results
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot joint positions
    axes[0].plot(timesteps, joint_positions)
    axes[0].set_title('Joint Positions Over Time')
    axes[0].set_ylabel('Joint Angle (rad)')
    axes[0].legend([f'Joint {i}' for i in range(5)])
    axes[0].grid(True)
    
    # Plot actions
    axes[1].plot(timesteps, actions)
    axes[1].set_title('NCAP Actions Over Time')
    axes[1].set_ylabel('Action Value')
    axes[1].legend([f'Joint {i}' for i in range(5)])
    axes[1].grid(True)
    
    # Plot action differences (coupling)
    action_diffs = np.diff(actions, axis=1)
    axes[2].plot(timesteps, action_diffs)
    axes[2].set_title('Action Differences (Coupling Between Joints)')
    axes[2].set_xlabel('Timestep')
    axes[2].set_ylabel('Action Difference')
    axes[2].legend([f'Joint {i}-{i+1}' for i in range(4)])
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('outputs/untrained_ncap_analysis.png', dpi=150, bbox_inches='tight')
    print("📊 Analysis plots saved to 'outputs/untrained_ncap_analysis.png'")
    
    env.close()
    return actions, joint_positions


def test_oscillator_only():
    """Test just the oscillator component."""
    print("\n🔄 Testing pure oscillator behavior...")
    
    ncap = NCAPSwimmer(
        n_joints=5,
        oscillator_period=60,
        use_weight_sharing=True,
        use_weight_constraints=True,
        include_proprioception=False,  # Disable to test pure oscillator
        include_head_oscillators=True,
        include_environment_adaptation=False
    )
    
    # Test oscillator with zero joint positions
    joint_pos = torch.zeros(5)
    actions = []
    
    for step in range(120):  # 2 oscillator periods
        with torch.no_grad():
            action = ncap(joint_pos, timesteps=torch.tensor([step], dtype=torch.float32))
            actions.append(action[0].item())  # First joint only
    
    actions = np.array(actions)
    
    # Check if it oscillates
    if actions.max() - actions.min() > 0.1:
        print(f"✅ Pure oscillator works! Range: {actions.max() - actions.min():.3f}")
    else:
        print(f"❌ Pure oscillator weak! Range: {actions.max() - actions.min():.3f}")
    
    # Plot oscillator
    plt.figure(figsize=(10, 4))
    plt.plot(actions)
    plt.title('Pure Oscillator Output (First Joint)')
    plt.xlabel('Timestep')
    plt.ylabel('Action Value')
    plt.grid(True)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.savefig('outputs/pure_oscillator_test.png', dpi=150, bbox_inches='tight')
    print("📊 Oscillator plot saved to 'outputs/pure_oscillator_test.png'")
    
    return actions


if __name__ == "__main__":
    # Test pure oscillator first
    test_oscillator_only()
    
    # Test full untrained model
    test_untrained_ncap()
    
    print("\n🎯 Recommendations:")
    print("1. If oscillator works but full model doesn't → proprioceptive coupling issue")
    print("2. If both fail → weight initialization/constraint issue")
    print("3. If both work → training algorithm is corrupting the biology") 