#!/usr/bin/env python3
"""
Test the simplified NCAP model to ensure it produces strong oscillatory behavior.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from swimmer.models.simple_ncap import SimpleNCAPSwimmer


def test_simple_ncap():
    """Test the simplified NCAP model."""
    print("🔬 Testing Simplified NCAP Model...")
    
    # Create simplified NCAP model
    ncap = SimpleNCAPSwimmer(
        n_joints=5,
        oscillator_period=60,
        use_weight_sharing=True,
        use_weight_constraints=True,
        include_proprioception=True,
        include_head_oscillators=True
    )
    
    print(f"📊 Simplified NCAP parameters:")
    for name, param in ncap.params.items():
        print(f"  {name}: {param.item():.3f}")
    
    # Test oscillatory behavior
    joint_pos = torch.zeros(5)  # Start with all joints at zero
    actions = []
    timesteps = []
    
    print("🔄 Testing oscillatory behavior...")
    
    for step in range(180):  # 3 seconds worth
        with torch.no_grad():
            action = ncap(joint_pos, timesteps=torch.tensor([step], dtype=torch.float32))
            actions.append(action.numpy().copy())
        timesteps.append(step)
    
    actions = np.array(actions)
    
    # Analyze oscillatory strength
    action_ranges = actions.max(axis=0) - actions.min(axis=0)
    avg_range = action_ranges.mean()
    
    print(f"\n📈 Oscillatory Analysis:")
    print(f"  Action ranges per joint: {action_ranges}")
    print(f"  Average range: {avg_range:.3f}")
    print(f"  Overall range: [{actions.min():.3f}, {actions.max():.3f}]")
    
    if avg_range > 1.0:
        print("✅ Strong oscillatory behavior detected!")
    elif avg_range > 0.5:
        print("⚠️ Moderate oscillatory behavior")
    else:
        print("❌ Weak oscillatory behavior")
    
    # Plot results
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot all joint actions
    axes[0].plot(timesteps, actions)
    axes[0].set_title('Simplified NCAP Actions Over Time')
    axes[0].set_ylabel('Action Value')
    axes[0].legend([f'Joint {i}' for i in range(5)])
    axes[0].grid(True)
    axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # Plot coupling between joints
    coupling_strength = []
    for i in range(4):  # Compare adjacent joints
        correlation = np.corrcoef(actions[:, i], actions[:, i+1])[0, 1]
        coupling_strength.append(correlation)
    
    axes[1].bar(range(4), coupling_strength)
    axes[1].set_title('Coupling Strength Between Adjacent Joints')
    axes[1].set_xlabel('Joint Pair')
    axes[1].set_ylabel('Correlation')
    axes[1].set_xticks(range(4))
    axes[1].set_xticklabels([f'{i}-{i+1}' for i in range(4)])
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/simple_ncap_analysis.png', dpi=150, bbox_inches='tight')
    print("📊 Analysis plots saved to 'outputs/simple_ncap_analysis.png'")
    
    return actions, coupling_strength


def compare_models():
    """Compare the simple vs complex NCAP models."""
    print("\n🆚 Comparing Simple vs Complex NCAP...")
    
    from swimmer.models.ncap_swimmer import NCAPSwimmer
    
    # Test simple model
    simple_ncap = SimpleNCAPSwimmer(n_joints=5)
    
    # Test complex model without adaptation
    complex_ncap = NCAPSwimmer(
        n_joints=5,
        include_environment_adaptation=False,  # Disable complex features
        include_proprioception=True,
        include_head_oscillators=True
    )
    
    joint_pos = torch.zeros(5)
    
    # Get actions from both models
    simple_action = simple_ncap(joint_pos)
    complex_action = complex_ncap(joint_pos, environment_type=None)
    
    print(f"Simple NCAP action: {simple_action.numpy()}")
    print(f"Complex NCAP action: {complex_action.cpu().numpy()}")
    
    # Check if they're similar
    diff = torch.abs(simple_action - complex_action.cpu()).mean().item()
    print(f"Mean difference: {diff:.4f}")
    
    if diff < 0.01:
        print("✅ Models produce similar outputs!")
    else:
        print("⚠️ Models differ significantly")


if __name__ == "__main__":
    # Test simplified model
    actions, coupling = test_simple_ncap()
    
    # Compare models
    compare_models()
    
    print("\n🎯 Simple NCAP Results:")
    avg_coupling = np.mean(coupling)
    print(f"  Average coupling strength: {avg_coupling:.3f}")
    if avg_coupling > 0.5:
        print("  ✅ Good proprioceptive coupling!")
    else:
        print("  ⚠️ Weak proprioceptive coupling") 