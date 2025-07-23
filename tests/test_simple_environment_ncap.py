#!/usr/bin/env python3
"""
Direct test of NCAP performance in simple vs complex environments.
Bypasses training framework to focus on environment physics.
"""

import torch
import numpy as np
from swimmer.models.simple_ncap import SimpleNCAPSwimmer
from swimmer.environments.simple_swimmer import SimpleSwimmerEnv
from swimmer.environments.mixed_environment import ImprovedMixedSwimmerEnv
from swimmer.environments.physics_fix import create_fixed_swimmer_env


def test_ncap_in_different_environments():
    """Test NCAP performance across different environments."""
    print("🔬 Testing NCAP Performance Across Environments...")
    
    # Create NCAP model
    ncap = SimpleNCAPSwimmer(n_joints=4)  # 4 actuators for 5-link swimmer
    
    print(f"📊 NCAP parameters:")
    for name, param in ncap.params.items():
        print(f"  {name}: {param.item():.3f}")
    
    # Test environments
    environments = {
        "Simple Environment": SimpleSwimmerEnv(n_links=5),
        "Complex Environment": ImprovedMixedSwimmerEnv(n_links=5), 
        "Complex + Gear Fix": create_fixed_swimmer_env(n_links=5, gear_ratio=0.1)
    }
    
    results = {}
    
    for env_name, env in environments.items():
        print(f"\n🧪 Testing: {env_name}")
        distance = test_ncap_swimming(ncap, env, env_name)
        results[env_name] = distance
    
    # Analysis
    print(f"\n📈 Results Summary:")
    for env_name, distance in results.items():
        print(f"  {env_name:<20}: {distance:.4f}m")
    
    simple_distance = results["Simple Environment"]
    complex_distance = results["Complex Environment"]
    gear_distance = results["Complex + Gear Fix"]
    
    print(f"\n💡 Performance Analysis:")
    
    if simple_distance > complex_distance * 3:
        print("✅ CONFIRMED: Environment is the primary issue!")
        print(f"   Simple environment: {simple_distance:.4f}m")
        print(f"   Complex environment: {complex_distance:.4f}m") 
        print(f"   Improvement ratio: {simple_distance/complex_distance:.1f}x")
        print("   📋 Recommendation: Use simple environment for NCAP training")
        
        if gear_distance > complex_distance * 1.5:
            print(f"   ⚙️ Gear fix also helps: {gear_distance:.4f}m")
        
    elif gear_distance > complex_distance * 2:
        print("⚙️ CONFIRMED: Gear ratio is the primary issue!")
        print("   📋 Recommendation: Always apply gear ratio fix")
        
    else:
        print("🤔 Mixed results - may need multiple fixes")
    
    # Test if NCAP has good zero-shot performance
    if simple_distance > 0.15:
        print(f"\n🎯 EXCELLENT: NCAP shows good zero-shot swimming ({simple_distance:.4f}m)")
        print("   This confirms the biological architecture works as intended")
    elif simple_distance > 0.05:
        print(f"\n👍 DECENT: NCAP shows some zero-shot capability ({simple_distance:.4f}m)")
    else:
        print(f"\n😐 LIMITED: NCAP zero-shot performance is weak ({simple_distance:.4f}m)")
    
    return results


def test_ncap_swimming(ncap, env, env_name):
    """Test NCAP swimming in a specific environment."""
    
    try:
        obs = env.reset()
        if hasattr(env, 'physics'):
            physics = env.physics
        elif hasattr(env, 'env') and hasattr(env.env, 'physics'):
            physics = env.env.physics
        else:
            print(f"  ⚠️ Cannot access physics for {env_name}")
            return 0.0
            
        initial_pos = physics.named.data.xpos['head', :2].copy()
        
        # Test swimming for 3 seconds
        for step in range(180):
            # Get joint positions from observation
            if isinstance(obs, dict):
                joint_pos = torch.tensor(obs['joints'], dtype=torch.float32)
            else:
                joint_pos = torch.tensor(obs[:4], dtype=torch.float32)  # First 4 joints
            
            # Get NCAP action
            with torch.no_grad():
                action = ncap(joint_pos, timesteps=torch.tensor([step], dtype=torch.float32))
                action = action.cpu().numpy()
            
            # Step environment
            try:
                obs, reward, done, info = env.step(action)
            except Exception as e:
                print(f"  ⚠️ Step error in {env_name}: {e}")
                break
        
        # Calculate distance
        final_pos = physics.named.data.xpos['head', :2].copy()
        distance = np.linalg.norm(final_pos - initial_pos)
        
        print(f"  Distance: {distance:.4f}m")
        
        env.close()
        return distance
        
    except Exception as e:
        print(f"  ❌ Error testing {env_name}: {e}")
        return 0.0


if __name__ == "__main__":
    test_ncap_in_different_environments() 