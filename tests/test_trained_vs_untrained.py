#!/usr/bin/env python3
"""
Compare trained vs untrained NCAP behavior to diagnose training issues.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from swimmer.models.simple_ncap import SimpleNCAPSwimmer


def test_trained_vs_untrained():
    """Compare trained and untrained NCAP models."""
    print("🔍 Comparing Trained vs Untrained NCAP Models...")
    
    # Create untrained model
    untrained = SimpleNCAPSwimmer(n_joints=5)
    
    print("📊 Untrained NCAP parameters:")
    for name, param in untrained.params.items():
        print(f"  {name}: {param.item():.3f}")
    
    # Load trained model  
    try:
        trained = SimpleNCAPSwimmer(n_joints=5)
        checkpoint = torch.load('outputs/training/improved_ncap_5links.pt', map_location='cpu')
        
        # Extract NCAP parameters from checkpoint
        ncap_params = {}
        for key, value in checkpoint.items():
            if 'ncap.params' in key:
                param_name = key.split('ncap.params.')[-1]
                ncap_params[param_name] = value
        
        # Load parameters
        if ncap_params:
            for name, param in ncap_params.items():
                if name in trained.params:
                    trained.params[name].data = param
            
            print("\n📊 Trained NCAP parameters:")
            for name, param in trained.params.items():
                print(f"  {name}: {param.item():.3f}")
        else:
            print("\n⚠️ Could not extract NCAP parameters from checkpoint")
            return
            
    except Exception as e:
        print(f"\n❌ Could not load trained model: {e}")
        return
    
    # Test both models
    joint_pos = torch.zeros(5)
    
    print("\n🧪 Testing oscillatory behavior...")
    
    untrained_actions = []
    trained_actions = []
    
    for step in range(120):  # 2 seconds
        # Untrained model
        with torch.no_grad():
            untrained_action = untrained(joint_pos, timesteps=torch.tensor([step], dtype=torch.float32))
            untrained_actions.append(untrained_action.numpy().copy())
        
        # Trained model
        with torch.no_grad():
            trained_action = trained(joint_pos, timesteps=torch.tensor([step], dtype=torch.float32))
            trained_actions.append(trained_action.numpy().copy())
    
    untrained_actions = np.array(untrained_actions)
    trained_actions = np.array(trained_actions)
    
    # Analyze differences
    print("\n📈 Action Range Comparison:")
    for joint in range(5):
        untrained_range = untrained_actions[:, joint].max() - untrained_actions[:, joint].min()
        trained_range = trained_actions[:, joint].max() - trained_actions[:, joint].min()
        print(f"  Joint {joint}: Untrained={untrained_range:.3f}, Trained={trained_range:.3f}")
    
    # Overall comparison
    untrained_avg_range = (untrained_actions.max(axis=0) - untrained_actions.min(axis=0)).mean()
    trained_avg_range = (trained_actions.max(axis=0) - trained_actions.min(axis=0)).mean()
    
    print(f"\nOverall Action Range:")
    print(f"  Untrained: {untrained_avg_range:.3f}")
    print(f"  Trained: {trained_avg_range:.3f}")
    print(f"  Ratio (trained/untrained): {trained_avg_range/untrained_avg_range:.3f}")
    
    if trained_avg_range / untrained_avg_range < 0.5:
        print("❌ Training significantly weakened oscillatory behavior!")
    elif trained_avg_range / untrained_avg_range < 0.8:
        print("⚠️ Training moderately weakened oscillatory behavior")
    else:
        print("✅ Oscillatory behavior reasonably preserved")
    
    # Plot comparison
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot untrained
    axes[0].plot(untrained_actions)
    axes[0].set_title('Untrained NCAP Actions')
    axes[0].set_ylabel('Action Value')
    axes[0].legend([f'Joint {i}' for i in range(5)])
    axes[0].grid(True)
    
    # Plot trained
    axes[1].plot(trained_actions)
    axes[1].set_title('Trained NCAP Actions')
    axes[1].set_ylabel('Action Value')
    axes[1].set_xlabel('Timestep')
    axes[1].legend([f'Joint {i}' for i in range(5)])
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('outputs/trained_vs_untrained_comparison.png', dpi=150, bbox_inches='tight')
    print("📊 Comparison plot saved to 'outputs/trained_vs_untrained_comparison.png'")


if __name__ == "__main__":
    test_trained_vs_untrained() 