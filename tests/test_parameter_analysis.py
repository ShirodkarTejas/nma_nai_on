#!/usr/bin/env python3
"""
Parameter Analysis Script
Diagnose why the model learns parameters that don't produce effective swimming.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from swimmer.models.enhanced_biological_ncap import EnhancedBiologicalNCAPSwimmer
from swimmer.environments.progressive_mixed_env import ProgressiveMixedSwimmerEnv

def analyze_model_parameters():
    """Analyze what parameters the model has learned and their effects."""
    
    print("🔍 ENHANCED NCAP PARAMETER ANALYSIS")
    print("=" * 50)
    
    # Create model
    model = EnhancedBiologicalNCAPSwimmer(
        n_joints=4,  # 5-link swimmer = 4 joints
        oscillator_period=60,
        include_environment_adaptation=True,
        include_goal_direction=True
    )
    
    print(f"\n📊 TOTAL PARAMETERS: {sum(p.numel() for p in model.parameters())}")
    
    # Analyze core NCAP parameters
    print("\n🧠 CORE BIOLOGICAL PARAMETERS:")
    for name, param in model.params.items():
        value = param.item() if param.numel() == 1 else param.detach().cpu().numpy()
        print(f"  {name}: {value:.4f}")
    
    # Analyze adaptation parameters
    print("\n🌊 ENVIRONMENT ADAPTATION PARAMETERS:")
    if model.include_environment_adaptation:
        print(f"  water_frequency_scale: {model.water_frequency_scale.item():.4f}")
        print(f"  land_frequency_scale: {model.land_frequency_scale.item():.4f}")
        print(f"  water_amplitude_scale: {model.water_amplitude_scale.item():.4f}")
        print(f"  land_amplitude_scale: {model.land_amplitude_scale.item():.4f}")
    
    # Analyze goal-directed parameters
    print("\n🎯 GOAL-DIRECTED PARAMETERS:")
    if model.include_goal_direction:
        print(f"  goal_sensitivity: {model.goal_sensitivity.item():.4f}")
        print(f"  goal_persistence: {model.goal_persistence.item():.4f}")
    
    return model

def test_action_generation(model):
    """Test what actions the model generates for different inputs."""
    
    print("\n⚡ ACTION GENERATION TEST")
    print("=" * 30)
    
    model.eval()
    
    # Test different joint positions
    test_positions = [
        torch.zeros(4),  # Straight swimmer
        torch.tensor([0.5, -0.5, 0.5, -0.5]),  # S-curve
        torch.tensor([1.0, 0.5, 0.0, -0.5]),  # Curved swimmer
    ]
    
    for i, joint_pos in enumerate(test_positions):
        print(f"\n  Test {i+1}: Joint positions = {joint_pos.numpy()}")
        
        # Generate actions over time
        actions_over_time = []
        for timestep in range(0, 180, 30):  # 3 seconds, every 1 second
            with torch.no_grad():
                action = model(joint_pos, timesteps=torch.tensor([timestep], dtype=torch.float32))
                actions_over_time.append(action.cpu().numpy())
                print(f"    t={timestep/30:.1f}s: actions = {action.cpu().numpy()}")
        
        # Check action range and variation
        actions_array = np.array(actions_over_time)
        action_range = actions_array.max() - actions_array.min()
        action_std = actions_array.std()
        
        print(f"    Action range: {action_range:.4f}")
        print(f"    Action std: {action_std:.4f}")
        
        if action_range < 0.1:
            print("    ⚠️  WARNING: Very small action range - may not produce movement!")
        if action_std < 0.05:
            print("    ⚠️  WARNING: Very low action variation - may be stuck!")

def test_swimming_simulation(model):
    """Simulate swimming behavior and measure distance traveled."""
    
    print("\n🏊 SWIMMING SIMULATION TEST")
    print("=" * 30)
    
    # Create environment
    env = ProgressiveMixedSwimmerEnv(n_links=5)
    obs = env.reset()
    
    model.eval()
    
    # Track swimming performance
    initial_pos = env.physics.named.data.xpos['head'][:2].copy()
    positions = [initial_pos.copy()]
    actions_taken = []
    rewards = []
    
    print(f"  Initial position: {initial_pos}")
    
    # Simulate 180 steps (6 seconds)
    for step in range(180):
        # Get joint positions from observation (handle different formats)
        try:
            if hasattr(obs, 'observation'):
                joint_pos = torch.tensor(obs.observation['joints'], dtype=torch.float32)
            elif isinstance(obs, dict):
                joint_pos = torch.tensor(obs['joints'], dtype=torch.float32)
            else:
                # dm_control TimeStep format
                joint_pos = torch.tensor(obs.observation['joints'], dtype=torch.float32)
        except:
            print(f"  Debug: obs type = {type(obs)}")
            if hasattr(obs, 'observation'):
                print(f"  Debug: obs.observation type = {type(obs.observation)}")
            elif isinstance(obs, (tuple, list)):
                print(f"  Debug: obs tuple/list length = {len(obs)}")
                if len(obs) > 0:
                    print(f"  Debug: obs[0] type = {type(obs[0])}")
            break
        
        # Generate action
        with torch.no_grad():
            action = model(joint_pos, timesteps=torch.tensor([step], dtype=torch.float32))
            action_np = action.cpu().numpy()
        
        actions_taken.append(action_np.copy())
        
        # Step environment
        time_step = env.step(action_np)
        
        # Handle different return formats
        if hasattr(time_step, 'observation'):
            obs = time_step
            reward = time_step.reward if hasattr(time_step, 'reward') else 0
        elif isinstance(time_step, (tuple, list)) and len(time_step) >= 3:
            obs, reward, done = time_step[:3]
        else:
            obs = time_step
            reward = 0
            
        rewards.append(reward)
        
        # Track position
        current_pos = env.physics.named.data.xpos['head'][:2].copy()
        positions.append(current_pos.copy())
        
        # Print periodic updates
        if step % 60 == 0:  # Every 2 seconds
            distance_so_far = np.linalg.norm(current_pos - initial_pos)
            print(f"    t={step/30:.1f}s: pos={current_pos}, distance={distance_so_far:.4f}m")
    
    # Final analysis
    final_pos = positions[-1]
    total_distance = np.linalg.norm(final_pos - initial_pos)
    total_reward = sum(rewards)
    
    actions_array = np.array(actions_taken)
    action_magnitude = np.mean(np.abs(actions_array))
    action_variation = np.std(actions_array)
    
    print(f"\n  📏 SIMULATION RESULTS:")
    print(f"    Total distance traveled: {total_distance:.4f}m")
    print(f"    Average speed: {total_distance/6:.4f}m/s")
    print(f"    Total reward: {total_reward:.2f}")
    print(f"    Average action magnitude: {action_magnitude:.4f}")
    print(f"    Action variation: {action_variation:.4f}")
    
    if total_distance < 0.1:
        print("    ❌ PROBLEM: No significant movement!")
        print("    💡 Possible causes:")
        print("       - Parameters too small to generate effective torques")
        print("       - Oscillator frequency too low")
        print("       - Biological constraints too restrictive")
        print("       - Goal-directed interference with locomotion")
    elif total_distance < 0.5:
        print("    ⚠️  WARNING: Very slow movement")
    else:
        print("    ✅ GOOD: Reasonable movement detected")
    
    return total_distance, actions_array, positions

def plot_swimming_analysis(positions, actions):
    """Plot swimming trajectory and action patterns."""
    
    positions = np.array(positions)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot trajectory
    ax1.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2)
    ax1.scatter(positions[0, 0], positions[0, 1], color='green', s=100, label='Start')
    ax1.scatter(positions[-1, 0], positions[-1, 1], color='red', s=100, label='End')
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.set_title('Swimming Trajectory')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Plot action patterns
    time_steps = np.arange(len(actions))
    for i in range(actions.shape[1]):
        ax2.plot(time_steps, actions[:, i], label=f'Joint {i+1}')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Action Value')
    ax2.set_title('Action Patterns Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('parameter_analysis_results.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Analysis plots saved to: parameter_analysis_results.png")
    
    return fig

def main():
    """Run complete parameter analysis."""
    
    print("🚀 Starting Enhanced NCAP Parameter Analysis...")
    
    # Analyze parameters
    model = analyze_model_parameters()
    
    # Test action generation
    test_action_generation(model)
    
    # Test swimming simulation
    distance, actions, positions = test_swimming_simulation(model)
    
    # Create visualizations
    plot_swimming_analysis(positions, actions)
    
    print("\n" + "="*50)
    print("🎯 ANALYSIS COMPLETE")
    
    if distance < 0.1:
        print("❌ DIAGNOSIS: Model parameters are not producing effective swimming")
        print("💊 RECOMMENDED FIXES:")
        print("   1. Increase parameter constraint ranges")
        print("   2. Remove or reduce goal-directed interference")
        print("   3. Check action scaling and torque generation")
        print("   4. Eliminate timeout reward exploitation")
    
    return model, distance, actions, positions

if __name__ == "__main__":
    model, distance, actions, positions = main() 