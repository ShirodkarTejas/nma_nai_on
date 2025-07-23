#!/usr/bin/env python3
"""
Test script to establish correct baseline performance using proper NCAP
"""

import torch
import numpy as np
import time
import os
from swimmer.models.proper_ncap import ProperSwimmerModule, ProperSwimmerActor
from swimmer.environments.mixed_environment import ImprovedMixedSwimmerEnv
from dm_control import suite
import imageio

class ProperNCAPSwimmerActor:
    """Actor wrapper for testing proper NCAP in our environment."""
    
    def __init__(self, swimmer_module):
        self.swimmer = swimmer_module
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.swimmer = self.swimmer.to(self.device)
    
    def __call__(self, obs):
        # Extract joint positions from observation
        if isinstance(obs, dict):
            joint_pos = obs['joints']
        else:
            # Assume first 5 values are joint positions for 6-link swimmer (5 joints)
            joint_pos = obs[:5]
        
        # Convert to tensor and move to device
        if not isinstance(joint_pos, torch.Tensor):
            joint_pos = torch.tensor(joint_pos, dtype=torch.float32, device=self.device)
        else:
            joint_pos = joint_pos.to(self.device)
        
        # Normalize joint positions to [-1, 1] range (assume they're in radians)
        # Max joint angle is about 2*pi/(n_joints+1) = 2*pi/6 ≈ 1.05 radians
        joint_limit = np.pi / 3  # Conservative limit
        joint_pos = torch.clamp(joint_pos / joint_limit, min=-1, max=1)
        
        # Get action from NCAP
        with torch.no_grad():
            actions = self.swimmer(joint_pos)
            
        # Convert back to numpy for environment
        if torch.is_tensor(actions):
            actions = actions.cpu().numpy()
            
        return actions

def test_proper_ncap_baseline():
    """Test proper NCAP implementation to establish correct baseline."""
    print("=== TESTING PROPER NCAP BASELINE PERFORMANCE ===")
    
    # Create proper NCAP swimmer from notebook
    proper_swimmer = ProperSwimmerModule(
        n_joints=5,  # 6-link swimmer has 5 joints
        oscillator_period=60,
        use_weight_sharing=True,
        use_weight_constraints=True,
        include_proprioception=True,
        include_head_oscillators=True
    )
    
    # Create actor wrapper
    actor = ProperNCAPSwimmerActor(proper_swimmer)
    
    # Test in regular dm_control environment first
    print("Testing in standard dm_control swimmer environment...")
    try:
        env = suite.load('swimmer', 'swimmer6')  # Try swimmer6 task
    except:
        try:
            env = suite.load('swimmer', 'swimmer15')  # Try swimmer15 task  
        except:
            # Create basic swimming environment manually
            from dm_control import suite
            print("Using default swimmer task...")
            env = suite.load(domain_name='swimmer', task_name='swimmer6')
    
    timestep = env.reset()
    total_reward = 0
    total_distance = 0
    frame_count = 0
    frames = []
    
    initial_pos = env.physics.named.data.xpos['head'].copy()
    
    for step in range(1000):  # Test for 1000 steps
        # Get observation
        obs = timestep.observation
        joint_positions = obs['joints']
        
        # Get action from proper NCAP
        action = actor(joint_positions)
        
        # Take step
        timestep = env.step(action)
        total_reward += timestep.reward
        
        # Calculate distance
        current_pos = env.physics.named.data.xpos['head']
        distance = np.linalg.norm(current_pos[:2] - initial_pos[:2])
        total_distance = distance
        
        # Record frame occasionally
        if step % 50 == 0:
            frame = env.physics.render(camera_id=0, height=240, width=320)
            frames.append(frame)
            print(f"Step {step}: Distance = {distance:.3f}, Reward = {timestep.reward:.3f}")
        
        frame_count += 1
        
        if timestep.last():
            break
    
    # Save video
    os.makedirs("outputs/baseline_test", exist_ok=True)
    video_path = "outputs/baseline_test/proper_ncap_baseline.mp4"
    imageio.mimsave(video_path, frames, fps=10)
    
    print(f"\n=== PROPER NCAP BASELINE RESULTS ===")
    print(f"Total steps: {frame_count}")
    print(f"Total distance traveled: {total_distance:.3f}")
    print(f"Average reward per step: {total_reward/frame_count:.3f}")
    print(f"Final reward: {timestep.reward:.3f}")
    print(f"Video saved to: {video_path}")
    
    # Compare with our previous results
    print(f"\n=== COMPARISON WITH PREVIOUS RESULTS ===")
    print(f"Proper NCAP baseline distance: {total_distance:.3f}")
    print(f"Our implementation distance: ~0.20")
    
    if total_distance > 1.0:
        print("✅ SUCCESS: Proper NCAP shows much better performance!")
        print("This confirms the issue is in our NCAP implementation.")
    else:
        print("❌ Still low performance - may need different environment or parameters")
    
    return {
        'distance': total_distance,
        'avg_reward': total_reward/frame_count,
        'steps': frame_count
    }

if __name__ == "__main__":
    results = test_proper_ncap_baseline() 