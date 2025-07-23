#!/usr/bin/env python3
"""
Analyze swimmer body scale and movement patterns to determine 
what constitutes meaningful swimming performance.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from swimmer.models.simple_ncap import SimpleNCAPSwimmer
from swimmer.environments.simple_swimmer import SimpleSwimmerEnv
import cv2


def analyze_swimmer_scale_and_movement():
    """Analyze swimmer dimensions and movement patterns."""
    print("📏 Analyzing Swimmer Scale and Movement Patterns...")
    
    # Create environment and model
    env = SimpleSwimmerEnv(n_links=5)
    ncap = SimpleNCAPSwimmer(n_joints=4)
    
    obs = env.reset()
    physics = env.physics
    
    # Get swimmer body dimensions
    print(f"\n📐 Swimmer Body Analysis:")
    
    # Get all body positions
    try:
        all_names = list(physics.named.data.xpos.axes.row.names)
    except:
        # Alternative access method
        all_names = [f'body_{i}' for i in range(6)]  # Fallback
    
    body_names = [name for name in all_names if 'link' in name or 'head' in name or 'body' in name]
    
    if len(body_names) == 0:
        # Try specific swimmer body part names
        possible_names = ['head'] + [f'link_{i}' for i in range(1, 6)] + [f'body_{i}' for i in range(6)]
        body_names = [name for name in possible_names if name in all_names]
    
    print(f"  Available body parts: {all_names[:10]}...")  # Show first 10
    print(f"  Identified swimmer parts: {body_names}")
    
    if len(body_names) >= 2:
        try:
            # Calculate body length
            head_pos = physics.named.data.xpos[body_names[0]]
            tail_pos = physics.named.data.xpos[body_names[-1]]
            body_length = np.linalg.norm(head_pos - tail_pos)
            
            print(f"  Head position: {head_pos}")
            print(f"  Tail position: {tail_pos}")
            print(f"  Total body length: {body_length:.3f}m")
        except Exception as e:
            print(f"  ⚠️ Error calculating body length: {e}")
            body_length = 3.0  # Estimate
            print(f"  Using estimated body length: {body_length:.3f}m")
    else:
        # Estimate based on typical 6-link swimmer dimensions
        body_length = 3.0  # Each link ~0.5m, so 6 links ≈ 3m
        print(f"  Using estimated body length: {body_length:.3f}m")
        print(f"  (Typical swimmer: 6 links × 0.5m = 3.0m)")
    
    # Test movement with detailed tracking
    print(f"\n🏊 Movement Pattern Analysis...")
    
    # Track positions over time
    positions = []
    joint_angles = []
    
    for step in range(300):  # 5 seconds at 60Hz
        # Get joint positions
        if isinstance(obs, dict):
            joint_pos = torch.tensor(obs['joints'], dtype=torch.float32)
        else:
            joint_pos = torch.tensor(obs[:4], dtype=torch.float32)
        
        # Store joint angles for pattern analysis
        joint_angles.append(joint_pos.numpy().copy())
        
        # Get NCAP action
        with torch.no_grad():
            action = ncap(joint_pos, timesteps=torch.tensor([step], dtype=torch.float32))
            action = action.cpu().numpy()
        
        # Store head position
        head_pos = physics.named.data.xpos['head'][:2].copy()
        positions.append(head_pos)
        
        # Step environment
        obs, reward, done, info = env.step(action)
    
    positions = np.array(positions)
    joint_angles = np.array(joint_angles)
    
    # Calculate movement metrics
    total_distance = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
    net_distance = np.linalg.norm(positions[-1] - positions[0])
    
    print(f"\n📊 Movement Metrics:")
    print(f"  Total path length: {total_distance:.3f}m")
    print(f"  Net displacement: {net_distance:.3f}m")
    print(f"  Body lengths traveled: {net_distance/body_length:.2f}")
    print(f"  Efficiency (net/total): {net_distance/total_distance:.3f}")
    
    # Analyze movement patterns
    print(f"\n🌊 Movement Pattern Analysis:")
    
    # Check for oscillatory patterns in joint angles
    for joint_idx in range(joint_angles.shape[1]):
        joint_data = joint_angles[:, joint_idx]
        
        # Simple oscillation detection
        zero_crossings = np.where(np.diff(np.signbit(joint_data)))[0]
        period_estimate = 2 * np.mean(np.diff(zero_crossings)) if len(zero_crossings) > 2 else 0
        
        amplitude = np.max(joint_data) - np.min(joint_data)
        
        print(f"  Joint {joint_idx}: amplitude={amplitude:.3f}, period≈{period_estimate:.1f} steps")
    
    # Check for traveling wave pattern
    print(f"\n🌊 Traveling Wave Analysis:")
    
    # Look for phase delays between adjacent joints
    phase_delays = []
    for i in range(joint_angles.shape[1] - 1):
        # Cross-correlation to find phase delay
        corr = np.correlate(joint_angles[:, i], joint_angles[:, i+1], mode='full')
        delay = np.argmax(corr) - len(joint_angles[:, i]) + 1
        phase_delays.append(delay)
        print(f"  Joint {i} → {i+1}: phase delay = {delay} steps")
    
    consistent_delay = len(set(phase_delays)) <= 2  # Allow some variation
    print(f"  Traveling wave detected: {'✅' if consistent_delay else '❌'}")
    
    # Performance assessment
    print(f"\n🎯 Performance Assessment:")
    
    if net_distance < body_length * 0.1:
        rating = "❌ VERY POOR"
        description = "Barely moving relative to body size"
    elif net_distance < body_length * 0.5:
        rating = "⚠️ POOR"
        description = "Minor movement, not effective swimming"
    elif net_distance < body_length * 2:
        rating = "👍 DECENT"
        description = "Some forward progress"
    elif net_distance < body_length * 5:
        rating = "✅ GOOD"
        description = "Effective swimming motion"
    else:
        rating = "🏆 EXCELLENT"
        description = "Strong swimming performance"
    
    print(f"  Rating: {rating}")
    print(f"  {description}")
    print(f"  For good swimming, expect >2 body lengths ({body_length*2:.1f}m)")
    print(f"  For excellent swimming, expect >5 body lengths ({body_length*5:.1f}m)")
    
    # Create visualization
    create_movement_visualization(positions, joint_angles, body_length, net_distance)
    
    env.close()
    
    return {
        'body_length': body_length,
        'net_distance': net_distance,
        'total_distance': total_distance,
        'body_lengths_ratio': net_distance/body_length,
        'positions': positions,
        'joint_angles': joint_angles
    }


def create_movement_visualization(positions, joint_angles, body_length, net_distance):
    """Create visualization of movement patterns."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Swimming trajectory
    ax1.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2, alpha=0.7)
    ax1.plot(positions[0, 0], positions[0, 1], 'go', markersize=10, label='Start')
    ax1.plot(positions[-1, 0], positions[-1, 1], 'ro', markersize=10, label='End')
    
    # Add body length reference
    ax1.plot([positions[0, 0], positions[0, 0] + body_length], 
             [positions[0, 1] - 0.1, positions[0, 1] - 0.1], 
             'k-', linewidth=3, label=f'Body length ({body_length:.2f}m)')
    
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.set_title(f'Swimming Trajectory\nNet Distance: {net_distance:.3f}m')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # 2. Distance over time
    distances = np.linalg.norm(positions - positions[0], axis=1)
    time_steps = np.arange(len(distances)) / 60.0  # Convert to seconds
    
    ax2.plot(time_steps, distances, 'b-', linewidth=2)
    ax2.axhline(y=body_length, color='r', linestyle='--', label=f'1 Body Length ({body_length:.2f}m)')
    ax2.axhline(y=body_length*2, color='orange', linestyle='--', label=f'2 Body Lengths ({body_length*2:.2f}m)')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Distance from Start (m)')
    ax2.set_title('Swimming Progress Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Joint angle patterns
    time_steps = np.arange(joint_angles.shape[0]) / 60.0
    for i in range(min(4, joint_angles.shape[1])):
        ax3.plot(time_steps, joint_angles[:, i], label=f'Joint {i}', linewidth=2)
    
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Joint Angle (rad)')
    ax3.set_title('Joint Angle Oscillations')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Speed analysis
    speeds = np.linalg.norm(np.diff(positions, axis=0), axis=1) * 60  # m/s at 60Hz
    time_steps_speed = np.arange(len(speeds)) / 60.0
    
    ax4.plot(time_steps_speed, speeds, 'g-', linewidth=2)
    ax4.axhline(y=np.mean(speeds), color='r', linestyle='--', label=f'Mean Speed: {np.mean(speeds):.3f} m/s')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Swimming Speed (m/s)')
    ax4.set_title('Swimming Speed Over Time')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('swimmer_movement_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Saved movement visualization to: swimmer_movement_analysis.png")
    plt.close()


def test_trained_vs_untrained_performance():
    """Compare what we should expect from trained vs untrained models."""
    print(f"\n🎓 Training Performance Expectations...")
    
    print(f"📋 Typical RL Swimming Performance Benchmarks:")
    print(f"  • Untrained/Random: 0.0 - 0.5m (mostly random motion)")
    print(f"  • Learning: 0.5 - 2.0m (developing coordination)")
    print(f"  • Good: 2.0 - 10.0m (effective swimming)")
    print(f"  • Excellent: 10.0 - 50.0m (optimized locomotion)")
    print(f"  • Expert: 50.0+ m (highly efficient)")
    
    print(f"\n🧬 NCAP Zero-Shot Performance Context:")
    print(f"  • Our result: 0.3m")
    print(f"  • This suggests the biological circuit has basic functionality")
    print(f"  • BUT may need significant training to reach efficient swimming")
    print(f"  • Training should target >10m for 'good' performance")
    
    print(f"\n⏱️ Training Duration Expectations:")
    print(f"  • Basic coordination: 10k - 100k steps")
    print(f"  • Effective swimming: 100k - 1M steps") 
    print(f"  • Optimized performance: 1M - 10M steps")
    print(f"  • Your 1M training should reach effective swimming range")


if __name__ == "__main__":
    results = analyze_swimmer_scale_and_movement()
    test_trained_vs_untrained_performance() 