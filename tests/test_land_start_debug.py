#!/usr/bin/env python3
"""
Debug Land Starting Positions
Diagnostic test to understand why forced land starting isn't working correctly.
"""

import os
import sys
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from swimmer.environments.progressive_mixed_env import ProgressiveMixedSwimmerEnv


def debug_land_starting():
    """Debug the land starting mechanism to understand why it's not working."""
    
    print("🔍 Debugging Land Starting Mechanism")
    print("=" * 60)
    
    # Create environment
    env = ProgressiveMixedSwimmerEnv(n_links=5, time_limit=3000)
    
    # Test different phases
    test_phases = [
        {"progress": 0.5, "name": "Phase 2: Single Land Zone"},
        {"progress": 0.7, "name": "Phase 3: Two Land Zones"}, 
        {"progress": 0.9, "name": "Phase 4: Island Complex"}
    ]
    
    for phase in test_phases:
        print(f"\n🏝️ Testing {phase['name']} (Progress: {phase['progress']:.1%})")
        print("-" * 50)
        
        # Set manual progress with forced land start
        env.set_manual_progress(phase['progress'], force_land_start=True)
        
        # Check if the task has the force_land_start_evaluation flag set
        if hasattr(env.env, '_task'):
            task = env.env._task
            print(f"📊 Task State:")
            print(f"   Training Progress: {task._training_progress:.3f}")
            print(f"   Force Land Start Evaluation: {getattr(task, '_force_land_start_evaluation', 'NOT SET')}")
            
            # Check current land zones
            land_zones = getattr(task, '_current_land_zones', [])
            print(f"   Current Land Zones: {len(land_zones)}")
            for i, zone in enumerate(land_zones):
                print(f"     Zone {i+1}: center={zone['center']}, radius={zone['radius']}")
            
            # Check if we're in the correct training progress range for land starting
            if task._training_progress >= 0.3:
                print(f"   ✅ Training progress >= 0.3 - land starting should be active")
            else:
                print(f"   ❌ Training progress < 0.3 - land starting not active")
        
        # Test multiple resets to see starting positions
        print(f"\n🎯 Testing 5 resets to observe starting positions:")
        
        for reset_idx in range(5):
            obs = env.reset()
            
            # Get actual starting position
            try:
                start_pos = env.physics.named.data.xpos['head'][:2]
                print(f"   Reset {reset_idx + 1}: Position ({start_pos[0]:.3f}, {start_pos[1]:.3f})", end="")
                
                # Check if position is in any land zone
                in_land = False
                if hasattr(env.env, '_task') and hasattr(env.env._task, '_current_land_zones'):
                    for zone in env.env._task._current_land_zones:
                        distance_to_zone = np.linalg.norm(start_pos - zone['center'])
                        if distance_to_zone < zone['radius']:
                            in_land = True
                            print(f" -> 🏝️ IN LAND ZONE (distance from zone center: {distance_to_zone:.3f}m)")
                            break
                
                if not in_land:
                    print(f" -> 🌊 IN WATER")
                    # Show distance to nearest land zone
                    if hasattr(env.env, '_task') and hasattr(env.env._task, '_current_land_zones'):
                        min_distance = float('inf')
                        nearest_zone = None
                        for j, zone in enumerate(env.env._task._current_land_zones):
                            distance = np.linalg.norm(start_pos - zone['center'])
                            if distance < min_distance:
                                min_distance = distance
                                nearest_zone = j
                        if nearest_zone is not None:
                            print(f"     Nearest land zone: Zone {nearest_zone + 1}, distance {min_distance:.3f}m")
                
            except Exception as e:
                print(f"   Reset {reset_idx + 1}: Error getting position - {e}")
    
    print(f"\n🔍 Debug Analysis Complete!")
    print("\n📋 Summary:")
    print("If swimmers are starting in water despite force_land_start=True:")
    print("1. Check if _force_land_start_evaluation flag is properly set")
    print("2. Verify land zones are configured for the current phase")
    print("3. Check if _set_progressive_starting_position is being called")
    print("4. Investigate if position is being overridden after setting")
    
    # Let me also directly test the land zone configuration logic
    print(f"\n🧪 Direct Land Zone Configuration Test:")
    
    # Create a direct task instance to test the logic
    from swimmer.environments.progressive_mixed_env import ProgressiveSwimCrawl
    
    for phase in test_phases:
        task = ProgressiveSwimCrawl(training_progress=phase['progress'])
        task._force_land_start_evaluation = True  # Manually set the flag
        
        print(f"\n   {phase['name']} (Progress: {phase['progress']:.1%}):")
        land_zones = task._get_progressive_land_zones()
        print(f"     Configured Land Zones: {len(land_zones)}")
        for i, zone in enumerate(land_zones):
            print(f"       Zone {i+1}: center={zone['center']}, radius={zone['radius']}")
        
        # Test the starting position logic directly
        print(f"     Testing starting position logic:")
        if phase['progress'] < 0.3:
            print(f"       Phase 1: Water start (progress < 0.3)")
        elif phase['progress'] < 0.6:
            print(f"       Phase 2: Should force land start with 100% probability")
            print(f"       Target zone: center=[3.0, 0], radius=1.8")
        elif phase['progress'] < 0.8:
            print(f"       Phase 3: Should force land start with 100% probability")
            print(f"       Target zones: LEFT [-2.0, 0] or RIGHT [3.5, 0]")
        else:
            print(f"       Phase 4: Should force land start with 100% probability")
            print(f"       Target islands: 4 different islands to choose from")


if __name__ == "__main__":
    debug_land_starting() 