#!/usr/bin/env python3
"""
Test optimal gear ratio for effective swimming.
"""

import numpy as np
from swimmer.environments.physics_fix import test_optimal_gear_ratio, create_fixed_swimmer_env
from swimmer.environments.mixed_environment import ImprovedMixedSwimmerEnv


def main():
    print("🔍 Finding Optimal Gear Ratio for NCAP Swimming...")
    
    # Test different gear ratios
    test_ratios = [0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0]
    
    optimal_gear, results = test_optimal_gear_ratio(ImprovedMixedSwimmerEnv, test_ratios)
    
    if optimal_gear:
        print(f"\n🎯 Recommended gear ratio: {optimal_gear}")
        
        # Test the fixed environment with optimal gear
        print(f"\n🧪 Testing fixed environment with gear ratio {optimal_gear}...")
        
        fixed_env = create_fixed_swimmer_env(n_links=5, gear_ratio=optimal_gear)
        
        # Test with NCAP-style actions (5 links = 4 actuators)
        from swimmer.models.simple_ncap import SimpleNCAPSwimmer
        
        ncap = SimpleNCAPSwimmer(n_joints=4)  # Match actuator count
        
        obs = fixed_env.reset()
        initial_pos = fixed_env.physics.named.data.xpos['head', :2].copy()
        
        # Run NCAP swimming test
        print("🏊 Testing with NCAP model...")
        import torch
        
        for step in range(180):  # 3 seconds
            joint_pos = obs['joints'] if isinstance(obs, dict) else obs[:4]
            
            with torch.no_grad():
                action = ncap(torch.tensor(joint_pos, dtype=torch.float32), 
                            timesteps=torch.tensor([step], dtype=torch.float32))
                action = action.cpu().numpy()
            
            obs, reward, done, info = fixed_env.step(action)
        
        final_pos = fixed_env.physics.named.data.xpos['head', :2].copy()
        ncap_distance = np.linalg.norm(final_pos - initial_pos)
        
        print(f"🎯 NCAP with fixed gear ({optimal_gear}): {ncap_distance:.4f}m")
        
        if ncap_distance > 0.2:
            print("✅ Excellent! Physics fix enables effective NCAP swimming!")
        elif ncap_distance > 0.1:
            print("⚠️ Good improvement, but could be better")
        else:
            print("❌ Still limited movement - may need additional fixes")
        
        fixed_env.close()
        
        # Print summary
        print(f"\n📊 Gear Ratio Test Results:")
        for gear, result in results.items():
            status = "✅" if result['stable'] else "❌"
            print(f"  {gear:4.1f}: {result['distance']:6.4f}m {status}")


if __name__ == "__main__":
    main() 