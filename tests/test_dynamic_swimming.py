#!/usr/bin/env python3
"""
Test NCAP with dynamic joint position updates to simulate realistic swimming.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from swimmer.models.simple_ncap import SimpleNCAPSwimmer


def test_dynamic_swimming():
    """Test NCAP with dynamic joint position updates."""
    print("🏊 Testing NCAP with Dynamic Joint Position Updates...")
    
    # Create NCAP model
    ncap = SimpleNCAPSwimmer(n_joints=5)
    
    # Start with zero joint positions
    joint_pos = torch.zeros(5)
    actions_over_time = []
    positions_over_time = []
    
    print("🔄 Running dynamic swimming simulation...")
    
    for step in range(180):  # 3 seconds
        # Get action from NCAP
        with torch.no_grad():
            action = ncap(joint_pos, timesteps=torch.tensor([step], dtype=torch.float32))
            actions_over_time.append(action.numpy().copy())
            positions_over_time.append(joint_pos.numpy().copy())
        
        # Update joint positions based on actions (simplified physics)
        # This simulates how joint positions would change in a real environment
        joint_pos = joint_pos + 0.1 * action  # Simple integration
        
        # Apply some damping to prevent runaway
        joint_pos = joint_pos * 0.99
    
    actions_over_time = np.array(actions_over_time)
    positions_over_time = np.array(positions_over_time)
    
    # Analyze results
    print("\n📈 Dynamic Swimming Analysis:")
    for joint in range(5):
        action_range = actions_over_time[:, joint].max() - actions_over_time[:, joint].min()
        pos_range = positions_over_time[:, joint].max() - positions_over_time[:, joint].min()
        print(f"  Joint {joint}: Action range={action_range:.3f}, Position range={pos_range:.3f}")
    
    avg_action_range = (actions_over_time.max(axis=0) - actions_over_time.min(axis=0)).mean()
    
    print(f"\nOverall Results:")
    print(f"  Average action range: {avg_action_range:.3f}")
    
    if avg_action_range > 1.0:
        print("✅ Strong undulation behavior achieved!")
    elif avg_action_range > 0.5:
        print("⚠️ Moderate undulation behavior")
    else:
        print("❌ Weak undulation behavior")
    
    # Check for coupling between joints
    coupling_detected = False
    for joint in range(1, 5):
        action_range = actions_over_time[:, joint].max() - actions_over_time[:, joint].min()
        if action_range > 0.1:
            coupling_detected = True
            break
    
    if coupling_detected:
        print("✅ Proprioceptive coupling working!")
    else:
        print("❌ No proprioceptive coupling detected")
    
    # Plot results
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot joint positions
    axes[0].plot(positions_over_time)
    axes[0].set_title('Joint Positions Over Time (Dynamic)')
    axes[0].set_ylabel('Position (rad)')
    axes[0].legend([f'Joint {i}' for i in range(5)])
    axes[0].grid(True)
    
    # Plot actions
    axes[1].plot(actions_over_time)
    axes[1].set_title('Actions Over Time (Dynamic)')
    axes[1].set_ylabel('Action Value')
    axes[1].legend([f'Joint {i}' for i in range(5)])
    axes[1].grid(True)
    
    # Plot phase relationships
    if actions_over_time[:, 0].max() - actions_over_time[:, 0].min() > 0.1:
        for joint in range(1, 5):
            if actions_over_time[:, joint].max() - actions_over_time[:, joint].min() > 0.1:
                # Phase analysis
                joint0_signal = actions_over_time[:, 0]
                joint_signal = actions_over_time[:, joint]
                
                # Simple phase lag calculation
                correlation = np.correlate(joint0_signal, joint_signal, mode='full')
                lag = np.argmax(correlation) - len(joint0_signal) + 1
                axes[2].bar(joint, lag, alpha=0.7, label=f'Joint {joint}')
        
        axes[2].set_title('Phase Lag Relative to Joint 0')
        axes[2].set_xlabel('Joint')
        axes[2].set_ylabel('Phase Lag (steps)')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/dynamic_swimming_test.png', dpi=150, bbox_inches='tight')
    print("📊 Dynamic swimming plot saved to 'outputs/dynamic_swimming_test.png'")
    
    return actions_over_time, avg_action_range, coupling_detected


def test_trained_model_dynamic():
    """Test the trained model with dynamic positions."""
    print("\n🔬 Testing Trained Model with Dynamic Positions...")
    
    try:
        # Load trained model
        trained = SimpleNCAPSwimmer(n_joints=5)
        checkpoint = torch.load('outputs/training/improved_ncap_5links.pt', map_location='cpu')
        
        # Extract and load NCAP parameters
        ncap_params = {}
        for key, value in checkpoint.items():
            if 'ncap.params' in key:
                param_name = key.split('ncap.params.')[-1]
                ncap_params[param_name] = value
        
        if ncap_params:
            for name, param in ncap_params.items():
                if name in trained.params:
                    trained.params[name].data = param
            
            print("📊 Trained model parameters:")
            for name, param in trained.params.items():
                print(f"  {name}: {param.item():.3f}")
        else:
            print("⚠️ Could not extract parameters")
            return None, None, None
            
        # Run dynamic test
        joint_pos = torch.zeros(5)
        actions = []
        
        for step in range(180):
            with torch.no_grad():
                action = trained(joint_pos, timesteps=torch.tensor([step], dtype=torch.float32))
                actions.append(action.numpy().copy())
            joint_pos = joint_pos + 0.1 * action * 0.99
        
        actions = np.array(actions)
        avg_range = (actions.max(axis=0) - actions.min(axis=0)).mean()
        
        coupling = any(actions[:, j].max() - actions[:, j].min() > 0.1 for j in range(1, 5))
        
        print(f"  Average action range: {avg_range:.3f}")
        print(f"  Coupling detected: {coupling}")
        
        return actions, avg_range, coupling
        
    except Exception as e:
        print(f"❌ Error loading trained model: {e}")
        return None, None, None


if __name__ == "__main__":
    # Test untrained model with dynamic positions
    untrained_actions, untrained_range, untrained_coupling = test_dynamic_swimming()
    
    # Test trained model with dynamic positions
    trained_actions, trained_range, trained_coupling = test_trained_model_dynamic()
    
    if trained_actions is not None:
        print(f"\n🎯 Summary:")
        print(f"  Untrained: Range={untrained_range:.3f}, Coupling={untrained_coupling}")
        print(f"  Trained: Range={trained_range:.3f}, Coupling={trained_coupling}")
        
        if trained_range > 0.8 and trained_coupling:
            print("✅ Trained model maintains strong swimming behavior!")
        else:
            print("⚠️ Trained model has weakened swimming behavior")
    
    print("\n💡 This test simulates realistic swimming with joint position feedback") 