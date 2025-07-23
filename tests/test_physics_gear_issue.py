#!/usr/bin/env python3
"""
Diagnostic test for physics gear ratio issues.
"""

import numpy as np
from swimmer.environments.mixed_environment import ImprovedMixedSwimmerEnv


def test_gear_ratios():
    """Test different gear ratios to see their effect on movement."""
    print("🔧 Testing Physics Gear Ratios...")
    
    # Create environment
    env = ImprovedMixedSwimmerEnv(n_links=5)
    physics = env.physics
    
    print(f"📊 Current Physics Parameters:")
    print(f"  Number of actuators: {physics.model.nu}")
    print(f"  Actuator names: {[physics.model.id2name(i, 'actuator') for i in range(physics.model.nu)]}")
    
    # Check current gear ratios
    current_gears = []
    for i in range(physics.model.nu):
        gear = physics.model.actuator_gear[i, 0]  # First component of gear vector
        current_gears.append(gear)
        actuator_name = physics.model.id2name(i, 'actuator') or f"actuator_{i}"
        print(f"  {actuator_name}: gear={gear}")
    
    # Test with current gear ratios
    print(f"\n🧪 Testing with CURRENT gear ratios:")
    distance_current = test_swimming_with_gears(env, current_gears, "current")
    
    # Test with increased gear ratios
    increased_gears = [g * 1000 for g in current_gears]  # 1000x increase
    print(f"\n🧪 Testing with INCREASED gear ratios (1000x):")
    distance_increased = test_swimming_with_gears(env, increased_gears, "increased")
    
    # Test with gym-style gear ratios
    gym_gears = [150.0] * len(current_gears)
    print(f"\n🧪 Testing with GYM-STYLE gear ratios (150.0):")
    distance_gym = test_swimming_with_gears(env, gym_gears, "gym_style")
    
    print(f"\n📈 Results Summary:")
    print(f"  Current gears ({current_gears[0]:.2e}): {distance_current:.4f}m")
    print(f"  Increased gears ({increased_gears[0]:.2e}): {distance_increased:.4f}m")  
    print(f"  Gym gears ({gym_gears[0]:.1f}): {distance_gym:.4f}m")
    
    ratio_increased = distance_increased / distance_current if distance_current > 0 else float('inf')
    ratio_gym = distance_gym / distance_current if distance_current > 0 else float('inf')
    
    print(f"\n🎯 Improvement ratios:")
    print(f"  Increased vs Current: {ratio_increased:.1f}x")
    print(f"  Gym vs Current: {ratio_gym:.1f}x")
    
    if ratio_gym > 10:
        print("✅ Gear ratio fix will dramatically improve swimming!")
    elif ratio_gym > 2:
        print("⚠️ Gear ratio fix will moderately improve swimming")
    else:
        print("❌ Gear ratio may not be the primary issue")


def test_swimming_with_gears(env, gear_ratios, test_name):
    """Test swimming performance with specific gear ratios."""
    physics = env.physics
    
    # Apply gear ratios
    for i, gear in enumerate(gear_ratios):
        if i < physics.model.nu:
            physics.model.actuator_gear[i, 0] = gear
    
    # Reset environment
    obs = env.reset()
    initial_pos = physics.named.data.xpos['head', :2].copy()
    
    # Run swimming test with strong oscillatory actions
    actions = []
    for step in range(180):  # 3 seconds
        # Use simple sinusoidal actions for testing (5 links = 4 actuators)
        phase = step * 2 * np.pi / 60  # 60-step period
        action = np.array([
            np.sin(phase),                # Joint 0: primary oscillator
            np.sin(phase - np.pi/4),      # Joint 1: phase lag
            np.sin(phase - np.pi/2),      # Joint 2: more phase lag  
            np.sin(phase - 3*np.pi/4),    # Joint 3: more phase lag
        ])[:physics.model.nu]  # Match actual number of actuators
        
        actions.append(action.copy())
        obs, reward, done, info = env.step(action)
    
    # Calculate distance traveled
    final_pos = physics.named.data.xpos['head', :2].copy()
    distance = np.linalg.norm(final_pos - initial_pos)
    
    actions = np.array(actions)
    action_range = actions.max() - actions.min()
    
    print(f"  {test_name}: Distance={distance:.4f}m, Action range={action_range:.3f}")
    
    return distance


if __name__ == "__main__":
    test_gear_ratios() 