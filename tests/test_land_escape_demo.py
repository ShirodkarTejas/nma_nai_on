#!/usr/bin/env python3
"""
Land Escape Demonstration Test
Forces swimmer to start inside land zones to demonstrate crawling and land-to-water transition behavior.
"""

import os
import sys
import numpy as np
import torch
import imageio
from tqdm import tqdm
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from swimmer.environments.progressive_mixed_env import TonicProgressiveMixedWrapper
from swimmer.models.enhanced_biological_ncap import EnhancedBiologicalNCAPSwimmer
from swimmer.utils.curriculum_visualization import add_enhanced_zone_disks


def create_land_escape_video(agent, env, save_path, test_scenarios=None, show_minimap=True):
    """Create a specialized video showing land escape behavior."""
    
    if test_scenarios is None:
        test_scenarios = [
            {
                "name": "Phase 2 Land Escape",
                "progress": 0.5,  # Phase 2: Single land zone
                "steps": 800,
                "force_land_start": True,
                "description": "Swimmer starts deep in land zone and must escape to water"
            },
            {
                "name": "Phase 3 Cross-Land",
                "progress": 0.7,  # Phase 3: Two land zones
                "steps": 1000,
                "force_land_start": True,
                "description": "Swimmer starts in one land zone and must traverse to other zones"
            },
            {
                "name": "Phase 4 Island Navigation", 
                "progress": 0.9,  # Phase 4: Complex islands
                "steps": 1200,
                "force_land_start": True,
                "description": "Swimmer starts on an island and must navigate complex terrain"
            }
        ]
    
    print(f"🏝️ Creating land escape demonstration video...")
    print(f"   Testing {len(test_scenarios)} scenarios")
    
    all_frames = []
    
    # Calculate total steps for progress bar
    total_steps = sum(scenario['steps'] for scenario in test_scenarios)
    total_steps += (len(test_scenarios) - 1) * 60  # Transition frames
    
    with tqdm(total=total_steps, desc="🎬 Recording Land Escape Demo", unit="frame",
             bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
        
        for scenario_idx, scenario in enumerate(test_scenarios):
            pbar.set_description(f"🎬 {scenario['name']}")
            
            # Set environment to specific phase with forced land start
            env.env.set_manual_progress(scenario['progress'], force_land_start=scenario['force_land_start'])
            
            print(f"🏝️ Testing Scenario: {scenario['name']}")
            print(f"   Progress: {scenario['progress']:.1%}")
            print(f"   Description: {scenario['description']}")
            print(f"   Recording {scenario['steps']} steps...")
            
            # Record this scenario
            scenario_frames = []
            obs = env.reset()
            
            # Track environment transitions for this scenario
            transitions_logged = 0
            last_environment = None
            
            for step in range(scenario['steps']):
                try:
                    # Monitor environment transitions
                    try:
                        if hasattr(env, 'env') and hasattr(env.env, 'env'):
                            physics = env.env.env.physics
                            head_pos = physics.named.data.xpos['head'][:2]
                            
                            # Check current environment
                            current_environment = "water"
                            if hasattr(env.env.env, '_task') and hasattr(env.env.env._task, '_current_land_zones'):
                                land_zones = env.env.env._task._current_land_zones or []
                                for zone in land_zones:
                                    distance_to_zone = np.linalg.norm(head_pos - zone['center'])
                                    if distance_to_zone < zone['radius']:
                                        current_environment = "land"
                                        break
                            
                            # Log transitions (first few only to avoid spam)
                            if last_environment is not None and last_environment != current_environment and transitions_logged < 3:
                                print(f"   🔄 Step {step}: {last_environment} → {current_environment} transition detected!")
                                transitions_logged += 1
                            last_environment = current_environment
                            
                    except Exception as e:
                        pass  # Silent if position tracking fails
                    
                    # Render frame
                    frame = env.render(mode='rgb_array')
                    if frame is not None:
                        # Add enhanced zone indicators
                        frame_with_zones = add_enhanced_zone_disks(
                            frame, env, step, minimap=show_minimap
                        )
                        scenario_frames.append(frame_with_zones)
                    
                    # Take action
                    action = agent.test_step(obs)
                    obs, reward, done, _ = env.step(action)
                    
                    if done:
                        obs = env.reset()
                    
                    # Update progress every 10 frames
                    if step % 10 == 0:
                        pbar.update(10)
                        
                except Exception as e:
                    print(f"⚠️ Error in scenario {scenario_idx}, step {step}: {e}")
                    break
            
            # Update for any remaining frames
            remaining_frames = scenario['steps'] - (step // 10) * 10
            if remaining_frames > 0:
                pbar.update(remaining_frames)
            
            all_frames.extend(scenario_frames)
            print(f"   ✅ {scenario['name']}: {len(scenario_frames)} frames recorded")
            
            # Add transition frames between scenarios (title card)
            if scenario_idx < len(test_scenarios) - 1:
                pbar.set_description(f"🎬 Adding transition")
                
                # Create title card for next scenario
                next_scenario = test_scenarios[scenario_idx + 1]
                transition_frames = create_title_card_frames(
                    f"Next: {next_scenario['name']}", 
                    next_scenario['description'],
                    duration_frames=60
                )
                all_frames.extend(transition_frames)
                pbar.update(60)
    
    # Save the complete land escape demo video
    pbar.set_description(f"🎬 Saving land escape demo")
    if len(all_frames) > 100:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            imageio.mimsave(save_path, all_frames, fps=30)
            print(f"🎬 Land escape demo saved to: {save_path} ({len(all_frames)} frames)")
            return True
        except Exception as e:
            print(f"❌ Video save error: {e}")
            return False
    else:
        print(f"⚠️ Not enough frames for demo video ({len(all_frames)})")
        return False


def create_title_card_frames(title, description, duration_frames=60):
    """Create title card frames for scenario transitions."""
    import cv2
    
    frames = []
    
    for frame_idx in range(duration_frames):
        # Create black background
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add title text
        title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        title_x = (640 - title_size[0]) // 2
        title_y = 200
        cv2.putText(frame, title, (title_x, title_y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        # Add description text (wrapped if needed)
        desc_lines = wrap_text(description, 50)  # 50 chars per line
        for i, line in enumerate(desc_lines):
            desc_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            desc_x = (640 - desc_size[0]) // 2
            desc_y = 250 + i * 30
            cv2.putText(frame, line, (desc_x, desc_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        # Add progress indicator (fading in)
        alpha = min(1.0, frame_idx / 30.0)  # Fade in over first 30 frames
        progress_text = f"Frame {frame_idx + 1}/{duration_frames}"
        cv2.putText(frame, progress_text, (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                   (int(150 * alpha), int(150 * alpha), int(150 * alpha)), 1)
        
        frames.append(frame)
    
    return frames


def wrap_text(text, width):
    """Simple text wrapping for title cards."""
    words = text.split()
    lines = []
    current_line = []
    
    for word in words:
        if len(' '.join(current_line + [word])) <= width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]
    
    if current_line:
        lines.append(' '.join(current_line))
    
    return lines


def test_land_escape_with_model(model_path=None, n_links=5):
    """Test land escape behavior with a trained model or untrained model."""
    
    print("🏝️ Testing Land Escape Behavior")
    print("=" * 50)
    
    # Create environment
    env = TonicProgressiveMixedWrapper(n_links=n_links, time_feature=True)
    
    # Create NCAP model
    n_joints = n_links - 1  # 4 joints for 5-link swimmer
    model = EnhancedBiologicalNCAPSwimmer(
        n_joints=n_joints,
        oscillator_period=60,
        include_environment_adaptation=True,
        include_goal_direction=True,
        locomotion_only_mode=False,  # Enable all features for testing
        action_scaling_factor=1.8
    )
    
    # Create simple agent wrapper
    class SimpleTestAgent:
        def __init__(self, model):
            self.model = model
            self.step_count = 0
            
        def test_step(self, obs):
            """Get action from model."""
            device = next(self.model.parameters()).device
            
            # Extract joint positions
            if isinstance(obs, dict):
                joint_pos = torch.tensor(obs['joints'], dtype=torch.float32, device=device)
                
                # Extract environment information
                environment_type = None
                if 'environment_type' in obs and 'fluid_viscosity' in obs:
                    env_flags = obs['environment_type']
                    viscosity = obs['fluid_viscosity'][0] if hasattr(obs['fluid_viscosity'], '__len__') else obs['fluid_viscosity']
                    vis_norm = np.clip((np.log10(viscosity) - np.log10(1e-4)) / (np.log10(1.5) - np.log10(1e-4)), 0.0, 1.0)
                    environment_type = torch.tensor([env_flags[0], env_flags[1], vis_norm], dtype=torch.float32, device=device)
                
                # Extract target information
                target_direction = None
                if 'target_direction' in obs:
                    target_direction = torch.tensor(obs['target_direction'], dtype=torch.float32, device=device)
                elif 'target_position' in obs:
                    target_pos = obs['target_position']
                    target_norm = np.linalg.norm(target_pos)
                    if target_norm > 0.1:
                        target_direction = torch.tensor(target_pos / target_norm, dtype=torch.float32, device=device)
            else:
                joint_pos = torch.tensor(obs[:4], dtype=torch.float32, device=device)
                environment_type = None
                target_direction = None
            
            # Get action from model
            with torch.no_grad():
                action = self.model(
                    joint_pos, 
                    environment_type=environment_type,
                    target_direction=target_direction,
                    timesteps=torch.tensor([self.step_count], device=device)
                )
                self.step_count += 1
            
            return action.cpu().numpy()
    
    # Load weights if provided
    if model_path and os.path.exists(model_path):
        print(f"📂 Loading trained model from: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model_status = "trained"
    else:
        print("🧬 Using untrained Enhanced Biological NCAP model for testing")
        model_status = "untrained"
    
    agent = SimpleTestAgent(model)
    
    # Generate unique artifact name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_name = f"enhanced_ncap_ppo_{n_links}links_land_escape_test_{model_status}_{timestamp}"
    
    # Create output directory
    output_dir = os.path.join("outputs", "land_escape_tests")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create land escape demo video
    video_path = os.path.join(output_dir, f"{artifact_name}_land_escape_demo.mp4")
    
    print(f"🎬 Creating land escape demonstration video...")
    print(f"   Model: {model_status} Enhanced Biological NCAP")
    print(f"   Links: {n_links}")
    print(f"   Output: {video_path}")
    
    success = create_land_escape_video(agent, env, video_path)
    
    if success:
        print("\n✅ Land Escape Test Complete!")
        print(f"🎬 Video saved to: {video_path}")
        print("\n🔍 Expected Behavior:")
        print("   - Swimmer should start deep inside land zones")
        print("   - Should attempt to crawl/wiggle within land (high viscosity)")
        print("   - Should transition to swimming when reaching water zones")
        print("   - Environment transitions should be clearly visible")
        print("\n📊 Analysis Notes:")
        print("   - Watch for different movement patterns in land vs water")
        print("   - Look for viscosity changes affecting swimming gait")
        print("   - Observe navigation attempts towards targets")
        
        # Create summary report
        summary_path = os.path.join(output_dir, f"{artifact_name}_land_escape_summary.md")
        create_land_escape_summary(summary_path, artifact_name, model_status, video_path)
        
        return video_path
    else:
        print("❌ Land escape test failed - video creation unsuccessful")
        return None


def create_land_escape_summary(summary_path, artifact_name, model_status, video_path):
    """Create a summary report for the land escape test."""
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    summary_content = f"""# Land Escape Test Summary

**Test ID:** {artifact_name}  
**Timestamp:** {timestamp}  
**Model Status:** {model_status}  

## Test Overview

This test demonstrates the swimmer's ability to escape from land zones and transition between different environment types (land/water) with different physics properties.

## Test Scenarios

### Scenario 1: Phase 2 Land Escape (Progress: 50%)
- **Environment:** Single large land zone (center=[3.0, 0], radius=1.8)
- **Starting Condition:** Swimmer forced to start deep within land zone
- **Expected Behavior:** Crawling movement in high-viscosity land, escape to low-viscosity water
- **Duration:** 800 steps (~26.7 seconds at 30fps)

### Scenario 2: Phase 3 Cross-Land (Progress: 70%) 
- **Environment:** Two land zones (left and right)
- **Starting Condition:** Swimmer starts in one of the land zones
- **Expected Behavior:** Navigate between land zones, cross water gaps
- **Duration:** 1000 steps (~33.3 seconds at 30fps)

### Scenario 3: Phase 4 Island Navigation (Progress: 90%)
- **Environment:** Four land islands in complex arrangement
- **Starting Condition:** Swimmer starts on random island
- **Expected Behavior:** Complex navigation requiring mixed locomotion
- **Duration:** 1200 steps (~40 seconds at 30fps)

## Physics Analysis

### Land Zone Physics
- **Viscosity:** 0.15 (150x higher than water)
- **Expected Movement:** Slower, crawling-like gait
- **Visual Indicators:** Brown/green zone overlays, "LAND" zone indicator

### Water Zone Physics  
- **Viscosity:** 0.001 (standard swimming)
- **Expected Movement:** Fluid swimming motions
- **Visual Indicators:** Blue water areas, "WATER" zone indicator

## Video Output

**Location:** `{video_path}`

### Visual Features
- Enhanced zone indicators showing land/water boundaries
- Real-time swimmer position tracking
- Environment transition notifications
- Phase and progress indicators
- Minimap showing overall navigation

## Evaluation Criteria

### Success Indicators
1. **Zone Recognition:** Swimmer behavior changes between land/water
2. **Transition Capability:** Clear movement patterns when crossing boundaries  
3. **Escape Behavior:** Ability to move from land zones toward water
4. **Navigation Intent:** Movement toward targets (even if not reached)

### Expected Challenges for Untrained Models
- Random movement patterns
- No clear navigation strategy
- May get stuck in land zones due to high viscosity
- Circular swimming behaviors

### Expected Improvements for Trained Models
- Directed movement toward targets
- Efficient land-to-water transitions
- Adaptive gait changes based on environment
- Goal-oriented navigation

## Technical Notes

### Environment Configuration
- **Training Progress Override:** Manual phase setting for testing
- **Force Land Start:** 100% probability of starting in land zones
- **Target System:** Land targets require traversing land zones
- **Reward Structure:** Bonuses for land target completion and environment transitions

### Model Architecture
- **Type:** Enhanced Biological NCAP
- **Inspiration:** C. elegans neural circuits
- **Key Features:** Oscillatory patterns, biological constraints, frequency adaptation

## Next Steps

1. Compare trained vs untrained model performance
2. Analyze specific gait changes between environments
3. Measure transition success rates
4. Evaluate target-reaching efficiency across phases

---
*Generated by Land Escape Test Suite - {timestamp}*
"""
    
    with open(summary_path, 'w') as f:
        f.write(summary_content)
    
    print(f"📄 Test summary saved to: {summary_path}")


if __name__ == "__main__":
    import argparse
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description="Test land escape behavior")
    parser.add_argument("--model_path", type=str, help="Path to trained model (optional)")
    parser.add_argument("--n_links", type=int, default=5, help="Number of swimmer links")
    parser.add_argument("--output_dir", type=str, default="outputs/land_escape_tests", 
                       help="Output directory for test results")
    
    args = parser.parse_args()
    
    # Run the land escape test
    result = test_land_escape_with_model(
        model_path=args.model_path,
        n_links=args.n_links
    )
    
    if result:
        print(f"\n🎯 Land escape test completed successfully!")
        print(f"🎬 Video: {result}")
    else:
        print(f"\n❌ Land escape test failed!")
        sys.exit(1) 