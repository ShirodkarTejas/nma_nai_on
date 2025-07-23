#!/usr/bin/env python3
"""
Diagnostic test for trained model coupling behavior.
Compare trained vs untrained model to see if training weakens coupling.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from swimmer.models.simple_ncap import SimpleNCAPSwimmer
from swimmer.environments.physics_fix import create_fixed_swimmer_env


def test_trained_model_coupling():
    """Test if trained model maintains proper coupling behavior."""
    print("🔍 Analyzing Trained Model Coupling Behavior...")
    
    # Load trained model
    try:
        trained_model = SimpleNCAPSwimmer(n_joints=4)
        checkpoint = torch.load('outputs/training/improved_ncap_5links.pt', map_location='cpu')
        
        # Extract NCAP parameters from checkpoint
        ncap_params = {}
        for key, value in checkpoint.items():
            if 'ncap.params.' in key:
                param_name = key.replace('ncap.params.', '')
                ncap_params[param_name] = value
        
        # Load parameters into trained model
        for name, param in trained_model.params.items():
            if name in ncap_params:
                param.data = ncap_params[name]
        
        print("✅ Loaded trained model parameters")
        
    except Exception as e:
        print(f"❌ Could not load trained model: {e}")
        return
    
    # Create untrained model for comparison
    untrained_model = SimpleNCAPSwimmer(n_joints=4)
    
    print(f"\n📊 Parameter Comparison:")
    print(f"{'Parameter':<15} {'Untrained':<10} {'Trained':<10} {'Change':<10}")
    print("-" * 50)
    
    param_changes = {}
    for name in trained_model.params.keys():
        untrained_val = untrained_model.params[name].item()
        trained_val = trained_model.params[name].item()
        change = trained_val - untrained_val
        param_changes[name] = change
        
        print(f"{name:<15} {untrained_val:<10.3f} {trained_val:<10.3f} {change:<10.3f}")
    
    # Test coupling strength in dynamic conditions
    print(f"\n🧪 Testing Coupling Strength in Dynamic Conditions...")
    
    untrained_coupling = test_dynamic_coupling(untrained_model, "Untrained")
    trained_coupling = test_dynamic_coupling(trained_model, "Trained")
    
    # Analysis
    print(f"\n📈 Coupling Analysis:")
    print(f"  Untrained model coupling strength: {untrained_coupling:.3f}")
    print(f"  Trained model coupling strength: {trained_coupling:.3f}")
    
    coupling_degradation = (untrained_coupling - trained_coupling) / untrained_coupling * 100
    print(f"  Coupling degradation: {coupling_degradation:.1f}%")
    
    # Test with physics environment
    print(f"\n🏊 Testing Swimming Performance with Physics Fix...")
    
    untrained_distance = test_swimming_performance(untrained_model, "Untrained")
    trained_distance = test_swimming_performance(trained_model, "Trained")
    
    print(f"\n🎯 Swimming Performance:")
    print(f"  Untrained distance: {untrained_distance:.4f}m")
    print(f"  Trained distance: {trained_distance:.4f}m")
    
    performance_ratio = trained_distance / untrained_distance if untrained_distance > 0 else 0
    print(f"  Performance ratio: {performance_ratio:.2f}x")
    
    # Recommendations
    print(f"\n💡 Diagnosis and Recommendations:")
    
    if coupling_degradation > 20:
        print("❌ TRAINING IS WEAKENING COUPLING")
        print("   - Biological constraints are not strong enough")
        print("   - Longer training will likely make it worse")
        print("   - Recommend: Stronger coupling constraints or architecture changes")
    elif coupling_degradation > 10:
        print("⚠️ MODERATE COUPLING DEGRADATION")  
        print("   - Training is somewhat weakening coupling")
        print("   - Longer training might help if early stopping prevents further degradation")
        print("   - Recommend: Monitor coupling strength during training")
    else:
        print("✅ COUPLING PRESERVED")
        print("   - Training maintains biological behavior")
        print("   - Longer training should improve performance")
    
    if performance_ratio < 0.5:
        print("❌ TRAINING HURTS PERFORMANCE")
        print("   - Model performs worse after training")
        print("   - Architecture or training algorithm issue")
    elif performance_ratio < 1.0:
        print("⚠️ TRAINING PROVIDES MODEST IMPROVEMENT")
        print("   - Some improvement but could be better")
        print("   - Longer training with proper constraints might help")
    else:
        print("✅ TRAINING IMPROVES PERFORMANCE")
        print("   - Training is working as expected")
    
    return {
        'param_changes': param_changes,
        'coupling_degradation': coupling_degradation,
        'performance_ratio': performance_ratio,
        'untrained_distance': untrained_distance,
        'trained_distance': trained_distance
    }


def test_dynamic_coupling(model, model_name):
    """Test coupling strength with dynamic joint positions."""
    print(f"  Testing {model_name} model...")
    
    # Start with zero positions, apply initial action, see how it propagates
    joint_pos = torch.zeros(4)
    actions_over_time = []
    
    for step in range(60):  # One oscillation cycle
        with torch.no_grad():
            action = model(joint_pos, timesteps=torch.tensor([step], dtype=torch.float32))
            actions_over_time.append(action.cpu().numpy().copy())
            
            # Update joint positions based on previous action (simple integration)
            joint_pos += action * 0.1  # Simple physics approximation
    
    actions = np.array(actions_over_time)
    
    # Calculate coupling strength as correlation between adjacent joints
    coupling_strength = 0
    for i in range(actions.shape[1] - 1):
        correlation = np.corrcoef(actions[:, i], actions[:, i+1])[0, 1]
        if not np.isnan(correlation):
            coupling_strength += abs(correlation)
    
    coupling_strength /= (actions.shape[1] - 1)  # Average correlation
    
    # Also calculate range to see if oscillations are strong
    action_range = actions.max() - actions.min()
    print(f"    Action range: {action_range:.3f}, Coupling: {coupling_strength:.3f}")
    
    return coupling_strength


def test_swimming_performance(model, model_name):
    """Test swimming performance with physics fix."""
    print(f"  Testing {model_name} swimming...")
    
    # Create environment with physics fix
    env = create_fixed_swimmer_env(n_links=5, gear_ratio=0.1)
    
    obs = env.reset()
    initial_pos = env.physics.named.data.xpos['head', :2].copy()
    
    # Run swimming test
    for step in range(120):  # 2 seconds
        joint_pos = obs['joints'] if isinstance(obs, dict) else obs[:4]
        
        with torch.no_grad():
            action = model(torch.tensor(joint_pos, dtype=torch.float32), 
                         timesteps=torch.tensor([step], dtype=torch.float32))
            action = action.cpu().numpy()
        
        obs, reward, done, info = env.step(action)
    
    final_pos = env.physics.named.data.xpos['head', :2].copy()
    distance = np.linalg.norm(final_pos - initial_pos)
    
    env.close()
    return distance


if __name__ == "__main__":
    test_trained_model_coupling() 