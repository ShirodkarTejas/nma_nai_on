#!/usr/bin/env python3
"""
Physics Fix Module
Fixes gear ratios and other physics parameters for effective NCAP swimming.
"""

import numpy as np


def apply_swimming_physics_fix(physics, gear_ratio=1.0):
    """
    Apply physics fixes for effective NCAP swimming.
    
    Args:
        physics: MuJoCo physics object
        gear_ratio: Gear ratio for actuators (default: 1.0)
    """
    print(f"🔧 Applying swimming physics fix with gear ratio: {gear_ratio}")
    
    # Fix actuator gear ratios
    original_gears = []
    for i in range(physics.model.nu):
        original_gear = physics.model.actuator_gear[i, 0]
        original_gears.append(original_gear)
        
        # Set new gear ratio
        physics.model.actuator_gear[i, 0] = gear_ratio
        
        actuator_name = physics.model.id2name(i, 'actuator') or f"actuator_{i}"
        print(f"  {actuator_name}: {original_gear:.2e} → {gear_ratio:.2f}")
    
    print(f"✅ Applied gear ratio fix to {physics.model.nu} actuators")
    return original_gears


def test_optimal_gear_ratio(env_class, test_ratios=[0.1, 0.5, 1.0, 2.0, 5.0]):
    """
    Test different gear ratios to find the optimal one.
    
    Args:
        env_class: Environment class to test
        test_ratios: List of gear ratios to test
    
    Returns:
        Optimal gear ratio and performance results
    """
    print("🧪 Testing optimal gear ratios...")
    
    results = {}
    
    for gear_ratio in test_ratios:
        print(f"\n📊 Testing gear ratio: {gear_ratio}")
        
        try:
            # Create fresh environment
            env = env_class(n_links=5)
            physics = env.physics
            
            # Apply gear ratio fix
            apply_swimming_physics_fix(physics, gear_ratio)
            
            # Test swimming performance
            distance = test_swimming_performance(env)
            results[gear_ratio] = {'distance': distance, 'stable': True}
            
            print(f"  ✅ Gear {gear_ratio}: Distance = {distance:.4f}m")
            
        except Exception as e:
            print(f"  ❌ Gear {gear_ratio}: Failed - {str(e)[:50]}...")
            results[gear_ratio] = {'distance': 0.0, 'stable': False}
        
        finally:
            if 'env' in locals():
                env.close()
    
    # Find optimal gear ratio
    stable_results = {k: v for k, v in results.items() if v['stable']}
    if stable_results:
        optimal_gear = max(stable_results.keys(), key=lambda k: stable_results[k]['distance'])
        optimal_distance = stable_results[optimal_gear]['distance']
        
        print(f"\n🎯 Optimal gear ratio: {optimal_gear} (Distance: {optimal_distance:.4f}m)")
        
        return optimal_gear, results
    else:
        print("\n❌ No stable gear ratios found!")
        return None, results


def test_swimming_performance(env):
    """Test swimming performance with sinusoidal actions."""
    physics = env.physics
    
    # Reset environment
    obs = env.reset()
    initial_pos = physics.named.data.xpos['head', :2].copy()
    
    # Run swimming test
    for step in range(120):  # 2 seconds
        # Sinusoidal swimming pattern
        phase = step * 2 * np.pi / 60  # 60-step period
        action = np.array([
            np.sin(phase),                # Joint 0
            np.sin(phase - np.pi/4),      # Joint 1: phase lag
            np.sin(phase - np.pi/2),      # Joint 2: more lag  
            np.sin(phase - 3*np.pi/4),    # Joint 3: more lag
        ])[:physics.model.nu]  # Match number of actuators
        
        obs, reward, done, info = env.step(action)
    
    # Calculate distance
    final_pos = physics.named.data.xpos['head', :2].copy()
    distance = np.linalg.norm(final_pos - initial_pos)
    
    return distance


class FixedGearSwimmerEnv:
    """Wrapper that automatically applies gear ratio fix."""
    
    def __init__(self, base_env_class, gear_ratio=1.0, **kwargs):
        self.base_env = base_env_class(**kwargs)
        self.gear_ratio = gear_ratio
        self._fix_applied = False
        
    def reset(self):
        obs = self.base_env.reset()
        if not self._fix_applied:
            apply_swimming_physics_fix(self.base_env.physics, self.gear_ratio)
            self._fix_applied = True
        return obs
    
    def step(self, action):
        return self.base_env.step(action)
    
    def close(self):
        return self.base_env.close()
    
    @property
    def physics(self):
        return self.base_env.physics
    
    @property
    def action_spec(self):
        return self.base_env.action_spec
    
    @property
    def observation_spec(self):
        return self.base_env.observation_spec


def create_fixed_swimmer_env(n_links=5, gear_ratio=1.0):
    """Create swimmer environment with gear ratio fix applied."""
    from .mixed_environment import ImprovedMixedSwimmerEnv
    return FixedGearSwimmerEnv(ImprovedMixedSwimmerEnv, gear_ratio=gear_ratio, n_links=n_links) 