#!/usr/bin/env python3
"""
Curriculum Training Visualization Utilities
Specialized plotting and video generation for progressive training.
"""

import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
import warnings
from datetime import datetime
from tqdm import tqdm

# Suppress the matplotlib tight_layout warning
warnings.filterwarnings("ignore", message=".*This figure includes Axes that are not compatible with tight_layout.*")


def add_minimap(frame, land_zones, swimmer_pos, frame_width, frame_height):
    """Add a minimap showing environment zones and swimmer position."""
    import cv2
    import numpy as np
    
    # Minimap parameters
    minimap_size = 120  # 120x120 pixels
    minimap_margin = 10
    
    # Position minimap in top-right corner
    minimap_x = frame_width - minimap_size - minimap_margin
    minimap_y = minimap_margin
    
    # Create minimap background (dark blue for water)
    minimap = np.ones((minimap_size, minimap_size, 3), dtype=np.uint8) * 20  # Dark background
    minimap[:, :, 0] = 40  # Dark blue water
    
    # Environment bounds for mapping
    env_bounds = 8.0  # Environment extends roughly -4 to +4 in both X and Y
    
    def world_to_minimap(pos):
        """Convert world coordinates to minimap pixel coordinates."""
        # Normalize to 0-1 range
        norm_x = (pos[0] + env_bounds/2) / env_bounds
        norm_y = (pos[1] + env_bounds/2) / env_bounds
        
        # Convert to minimap coordinates
        map_x = int(norm_x * minimap_size)
        map_y = int(norm_y * minimap_size)  # No Y flip - maintain physics orientation in minimap
        
        # Clamp to bounds
        map_x = max(0, min(minimap_size - 1, map_x))
        map_y = max(0, min(minimap_size - 1, map_y))
        
        return (map_x, map_y)
    
    # Draw land zones on minimap
    for i, zone in enumerate(land_zones):
        center = world_to_minimap(zone['center'])
        radius = int(zone['radius'] / env_bounds * minimap_size)
        radius = max(3, radius)  # Minimum visible radius
        
        # Draw land zone (brown)
        cv2.circle(minimap, center, radius, (101, 67, 33), -1)  # Brown fill
        cv2.circle(minimap, center, radius, (139, 69, 19), 1)   # Brown border
    
    # Draw swimmer position (red dot)
    swimmer_minimap_pos = world_to_minimap(swimmer_pos)
    cv2.circle(minimap, swimmer_minimap_pos, 3, (0, 0, 255), -1)  # Red swimmer dot
    cv2.circle(minimap, swimmer_minimap_pos, 3, (255, 255, 255), 1)  # White border
    
    # Add minimap border
    cv2.rectangle(minimap, (0, 0), (minimap_size-1, minimap_size-1), (255, 255, 255), 2)
    
    # Overlay minimap on main frame
    frame[minimap_y:minimap_y + minimap_size, minimap_x:minimap_x + minimap_size] = minimap
    
    # Add minimap title
    cv2.rectangle(frame, (minimap_x, minimap_y - 20), (minimap_x + minimap_size, minimap_y), (0, 0, 0), -1)
    cv2.putText(frame, "Environment Map", (minimap_x + 5, minimap_y - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return frame


def add_minimap_with_trail(frame, land_zones, position_history, frame_width, frame_height):
    """Add a minimap showing environment zones and swimmer position trail."""
    import cv2
    import numpy as np
    
    # Minimap parameters
    minimap_size = 120  # 120x120 pixels
    minimap_margin = 10
    
    # Position minimap in top-right corner
    minimap_x = frame_width - minimap_size - minimap_margin
    minimap_y = minimap_margin
    
    # Create minimap background (dark blue for water)
    minimap = np.ones((minimap_size, minimap_size, 3), dtype=np.uint8) * 20  # Dark background
    minimap[:, :, 0] = 40  # Dark blue water
    
    # Environment bounds for mapping
    env_bounds = 8.0  # Environment extends roughly -4 to +4 in both X and Y
    
    def world_to_minimap(pos):
        """Convert world coordinates to minimap pixel coordinates."""
        # Normalize to 0-1 range
        norm_x = (pos[0] + env_bounds/2) / env_bounds
        norm_y = (pos[1] + env_bounds/2) / env_bounds
        
        # Convert to minimap coordinates
        map_x = int(norm_x * minimap_size)
        map_y = int(norm_y * minimap_size)  # No Y flip - maintain physics orientation in minimap
        
        # Clamp to bounds
        map_x = max(0, min(minimap_size - 1, map_x))
        map_y = max(0, min(minimap_size - 1, map_y))
        
        return (map_x, map_y)
    
    # Draw land zones on minimap
    for i, zone in enumerate(land_zones):
        center = world_to_minimap(zone['center'])
        radius = int(zone['radius'] / env_bounds * minimap_size)
        radius = max(3, radius)  # Minimum visible radius
        
        # Draw land zone (brown)
        cv2.circle(minimap, center, radius, (101, 67, 33), -1)  # Brown fill
        cv2.circle(minimap, center, radius, (139, 69, 19), 1)   # Brown border
    
    # Draw swimmer trail (green line)
    if len(position_history) > 1:
        trail_points = []
        for pos in position_history:
            trail_points.append(world_to_minimap(pos))
        
        # Draw trail as connected lines with fading effect
        for i in range(1, len(trail_points)):
            # Fade older trail points
            alpha = i / len(trail_points)  # Newer points are more opaque
            intensity = int(255 * alpha)
            
            cv2.line(minimap, trail_points[i-1], trail_points[i], 
                    (0, intensity, 0), 1)  # Green trail
    
    # Draw current swimmer position (red dot)
    if position_history:
        swimmer_minimap_pos = world_to_minimap(position_history[-1])
        cv2.circle(minimap, swimmer_minimap_pos, 3, (0, 0, 255), -1)  # Red swimmer dot
        cv2.circle(minimap, swimmer_minimap_pos, 3, (255, 255, 255), 1)  # White border
    
    # Add minimap border
    cv2.rectangle(minimap, (0, 0), (minimap_size-1, minimap_size-1), (255, 255, 255), 2)
    
    # Overlay minimap on main frame
    frame[minimap_y:minimap_y + minimap_size, minimap_x:minimap_x + minimap_size] = minimap
    
    # Add minimap title
    cv2.rectangle(frame, (minimap_x, minimap_y - 20), (minimap_x + minimap_size, minimap_y), (0, 0, 0), -1)
    cv2.putText(frame, "Environment Map", (minimap_x + 5, minimap_y - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return frame


def add_zone_indicators_with_trail(frame, env, step_count, position_history, show_minimap=True):
    """Add zone indicators with minimap showing swimmer trail and navigation targets."""
    try:
        import cv2
        import numpy as np
        
        frame_with_zones = frame.copy()
        height, width = frame.shape[:2]
        
        # Get current zones and targets from environment
        land_zones = []
        current_targets = []
        current_target_index = 0
        targets_reached = 0
        phase_name = ""
        progress = 0.0
        current_zone_type = "Water"  # Default
        swimmer_pos = None
        
        if hasattr(env, 'env') and hasattr(env.env, 'env') and hasattr(env.env.env, '_task'):
            task = env.env.env._task
            progress = getattr(task, '_training_progress', 0.0)
            
            # Get current land zones
            if hasattr(task, '_current_land_zones'):
                land_zones = task._current_land_zones or []
            
            # Get current targets
            if hasattr(task, '_current_targets'):
                current_targets = task._current_targets or []
            if hasattr(task, '_current_target_index'):
                current_target_index = task._current_target_index
            if hasattr(task, '_targets_reached'):
                targets_reached = task._targets_reached
                
                # Phase detection
                if progress < 0.3:
                    phase_name = "Phase 1: Pure Swimming"
                elif progress < 0.6:
                    phase_name = "Phase 2: Single Land Zone"
                elif progress < 0.8:
                    phase_name = "Phase 3: Two Land Zones" 
                else:
                    phase_name = "Phase 4: Full Complexity"
            
            # **NEW: Detect current zone type for prominent display**
            try:
                if hasattr(env, 'env') and hasattr(env.env, 'env'):
                    physics = env.env.env.physics
                    swimmer_pos = physics.named.data.xpos['head'][:2]
                    
                    # Check if swimmer is in any land zone
                    in_land = False
                    for zone in land_zones:
                        distance = np.linalg.norm(swimmer_pos - zone['center'])
                        if distance < zone['radius']:
                            in_land = True
                            break
                    
                    current_zone_type = "Land" if in_land else "Water"
            except Exception as e:
                current_zone_type = "Water"  # Default if detection fails
        
        # Add phase indicator with enhanced info
        if phase_name:
            # Main phase indicator
            cv2.rectangle(frame_with_zones, (5, 5), (400, 35), (0, 0, 0), -1)
            cv2.putText(frame_with_zones, phase_name, (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Debug info: Show progress and zone count  
            debug_text = f"Progress: {progress:.1%} | Zones: {len(land_zones)}"
            cv2.rectangle(frame_with_zones, (5, 40), (350, 65), (0, 0, 0), -1)
            cv2.putText(frame_with_zones, debug_text, (10, 57), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # **NEW: Prominent Current Zone Indicator**
        # Large, bright indicator showing swimmer's current environment
        zone_color = (0, 255, 255) if current_zone_type == "Water" else (0, 165, 255)  # Cyan for water, orange for land
        zone_text_color = (0, 0, 0) if current_zone_type == "Water" else (255, 255, 255)  # Black text on cyan, white on orange
        
        # Large zone indicator (top-right corner, below minimap)
        zone_box_width = 180
        zone_box_height = 45
        zone_x = width - zone_box_width - 15
        zone_y = 140  # Below minimap (120px + 10px margin + 10px gap = 140px)
        
        # Draw zone indicator box with bright border
        cv2.rectangle(frame_with_zones, (zone_x, zone_y), (zone_x + zone_box_width, zone_y + zone_box_height), zone_color, -1)
        cv2.rectangle(frame_with_zones, (zone_x-3, zone_y-3), (zone_x + zone_box_width+3, zone_y + zone_box_height+3), (255, 255, 255), 3)
        
        # Zone type text (large and prominent)
        zone_display_text = f"ZONE: {current_zone_type.upper()}"
        text_size = cv2.getTextSize(zone_display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = zone_x + (zone_box_width - text_size[0]) // 2
        text_y = zone_y + (zone_box_height + text_size[1]) // 2
        cv2.putText(frame_with_zones, zone_display_text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, zone_text_color, 2)
        
        # Add step counter (positioned to avoid collisions)
        step_text = f"Step: {step_count}"
        if show_minimap and len(land_zones) > 0:
            # Place step counter below zone indicator when both minimap and zones are shown
            cv2.putText(frame_with_zones, step_text, (width-150, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            # Original position when no minimap
            cv2.putText(frame_with_zones, step_text, (width-150, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add swimmer position indicator (bright circle at swimmer location)
        try:
            if hasattr(env, 'env') and hasattr(env.env, 'env'):
                physics = env.env.env.physics
                swimmer_pos = physics.named.data.xpos['head'][:2]
                
                # Convert swimmer position to screen coordinates
                x_range = 12.0  # -6 to 6 
                y_range = 8.0   # -4 to 4
                
                swimmer_screen_x = int((swimmer_pos[0] + 6.0) / x_range * width)
                swimmer_screen_y = int(height - (swimmer_pos[1] + 4.0) / y_range * height)  # Flip Y for screen coords
                
                if 0 <= swimmer_screen_x < width and 0 <= swimmer_screen_y < height:
                    # Draw bright indicator around swimmer - MUCH more visible
                    cv2.circle(frame_with_zones, (swimmer_screen_x, swimmer_screen_y), 25, (0, 255, 255), 4)  # Larger cyan circle
                    cv2.circle(frame_with_zones, (swimmer_screen_x, swimmer_screen_y), 12, (255, 255, 255), -1)  # Larger white center
                    cv2.circle(frame_with_zones, (swimmer_screen_x, swimmer_screen_y), 12, (0, 0, 0), 2)  # Black border for contrast
                    cv2.putText(frame_with_zones, "SWIMMER", (swimmer_screen_x-35, swimmer_screen_y-30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    # Add arrow pointing to swimmer for extra visibility
                    cv2.arrowedLine(frame_with_zones, (swimmer_screen_x-50, swimmer_screen_y-50), 
                                   (swimmer_screen_x-15, swimmer_screen_y-15), (255, 255, 0), 3, tipLength=0.3)
        except Exception as e:
            print(f"⚠️ Swimmer position indicator error: {e}")
        
        # Add training status indicator (dynamic based on training progress)
        if progress < 0.1:  # Less than 10% training progress
            status_text = "UNTRAINED MODEL"
            status_color = (0, 0, 139)  # Dark red
            text_color = (255, 255, 255)  # White
        elif progress < 0.5:  # 10-50% training progress
            status_text = "TRAINING IN PROGRESS"
            status_color = (0, 139, 139)  # Dark yellow/orange
            text_color = (255, 255, 255)  # White
        else:  # 50%+ training progress
            status_text = "TRAINED MODEL"
            status_color = (0, 139, 0)  # Dark green
            text_color = (255, 255, 255)  # White
        
        cv2.rectangle(frame_with_zones, (width-220, height-40), (width-5, height-10), status_color, -1)
        cv2.putText(frame_with_zones, status_text, (width-215, height-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
        
        # Add navigation targets (if available)
        if current_targets and len(current_targets) > 0:
            # Convert physics coordinates to screen coordinates
            x_range = 12.0  # -6 to 6 
            y_range = 8.0   # -4 to 4
            
            for i, target in enumerate(current_targets):
                target_pos = target['position']
                target_type = target['type']
                
                # Convert target position to screen coordinates
                target_screen_x = int((target_pos[0] + 6.0) / x_range * width)
                target_screen_y = int(height - (target_pos[1] + 4.0) / y_range * height)  # Flip Y for screen coords
                
                if 0 <= target_screen_x < width and 0 <= target_screen_y < height:
                    if i == current_target_index:
                        # Current target - bright and pulsing
                        pulse_intensity = int(127 + 128 * abs(np.sin(step_count * 0.3)))
                        if target_type == 'swim':
                            color = (0, pulse_intensity, 255)  # Bright blue for swim targets
                        else:
                            color = (0, pulse_intensity, 0)    # Bright green for land targets
                        
                        # Draw target with border
                        cv2.circle(frame_with_zones, (target_screen_x, target_screen_y), 15, color, -1)
                        cv2.circle(frame_with_zones, (target_screen_x, target_screen_y), 15, (255, 255, 255), 2)
                        
                        # Add target number
                        cv2.putText(frame_with_zones, str(i+1), 
                                  (target_screen_x-6, target_screen_y+6), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    elif i < current_target_index:
                        # Completed target - dimmed
                        if target_type == 'swim':
                            color = (0, 64, 128)  # Dim blue
                        else:
                            color = (0, 64, 0)    # Dim green
                        
                        cv2.circle(frame_with_zones, (target_screen_x, target_screen_y), 10, color, -1)
                        cv2.circle(frame_with_zones, (target_screen_x, target_screen_y), 10, (128, 128, 128), 1)
                        
                        # Checkmark for completed
                        cv2.putText(frame_with_zones, "✓", 
                                  (target_screen_x-6, target_screen_y+6), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    else:
                        # Future target - faded
                        if target_type == 'swim':
                            color = (0, 100, 200)  # Faded blue
                        else:
                            color = (0, 100, 0)    # Faded green
                        
                        cv2.circle(frame_with_zones, (target_screen_x, target_screen_y), 8, color, 1)
                        cv2.putText(frame_with_zones, str(i+1), 
                                  (target_screen_x-4, target_screen_y+4), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Add minimap with trail if requested and zones exist
        if show_minimap and len(land_zones) > 0 and len(position_history) > 0:
            try:
                frame_with_zones = add_minimap_with_trail(
                    frame_with_zones, land_zones, position_history, width, height
                )
            except Exception as e:
                print(f"⚠️ Minimap error: {e}")
        
        return frame_with_zones
        
    except Exception as e:
        print(f"⚠️ Zone overlay error: {e}")
        return frame


def create_trajectory_analysis(agent, env, save_path, num_steps=500, phase_name=""):
    """Create detailed trajectory analysis similar to trained_model_analysis style."""
    
    # Create organized output directory
    plots_dir = os.path.join("outputs", "curriculum_training", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Update save path to use organized structure
    if not save_path.startswith("outputs/curriculum_training/"):
        filename = os.path.basename(save_path)
        save_path = os.path.join(plots_dir, filename)
    
    print(f"📊 Creating trajectory analysis for {phase_name}")
    
    # Record trajectory data
    positions = []
    distances = []
    velocities = []
    env_types = []
    joint_torques = {i: [] for i in range(4)}  # Assuming 4 joints
    
    obs = env.reset()
    total_distance = 0.0
    prev_pos = None
    
    for step in range(num_steps):
        # Get current position if possible
        try:
            if hasattr(env, 'env') and hasattr(env.env, 'env'):
                physics = env.env.env.physics
                head_pos = physics.named.data.xpos['head'][:2]  # X, Y only
                positions.append(head_pos.copy())
                
                # Calculate distance traveled
                if prev_pos is not None:
                    step_distance = np.linalg.norm(head_pos - prev_pos)
                    total_distance += step_distance
                distances.append(total_distance)
                prev_pos = head_pos.copy()
                
                # Get velocity
                head_vel = physics.named.data.sensordata['head_vel'][:2]
                velocity_magnitude = np.linalg.norm(head_vel)
                velocities.append(velocity_magnitude)
                
                # Get joint torques
                joint_data = physics.data.actuator_force[:4]  # First 4 joints
                for i, torque in enumerate(joint_data):
                    joint_torques[i].append(torque)
                
                # Determine environment type
                if hasattr(env.env.env, '_task') and hasattr(env.env.env._task, '_current_land_zones'):
                    land_zones = env.env.env._task._current_land_zones or []
                    in_land = False
                    for zone in land_zones:
                        distance_to_zone = np.linalg.norm(head_pos - zone['center'])
                        if distance_to_zone < zone['radius']:
                            in_land = True
                            break
                    env_types.append('Land' if in_land else 'Water')
                else:
                    env_types.append('Water')
                    
        except Exception as e:
            # Fallback if position tracking fails
            positions.append([0.0, 0.0])
            distances.append(total_distance)
            velocities.append(0.0)
            env_types.append('Water')
            for i in range(4):
                joint_torques[i].append(0.0)
        
        # Take action
        try:
            action = agent.test_step(obs)
            obs, reward, done, _ = env.step(action)
            
            if done:
                obs = env.reset()
        except Exception as e:
            print(f"⚠️ Step error: {e}")
            break
    
    # Create the analysis plot
    fig = plt.figure(figsize=(16, 12))
    
    # Create a 3x2 grid
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], hspace=0.3, wspace=0.3)
    
    # 1. Environment Zones and Swimmer Trajectory (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title('Environment Zones and Swimmer Trajectory', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.grid(True, alpha=0.3)
    
    # Get land zones and targets from environment
    land_zones = []
    current_targets = []
    if hasattr(env, 'env') and hasattr(env.env, 'env') and hasattr(env.env.env, '_task'):
        task = env.env.env._task
        if hasattr(task, '_current_land_zones'):
            land_zones = task._current_land_zones or []
        if hasattr(task, '_current_targets'):
            current_targets = task._current_targets or []
    
    # Draw land zones
    for i, zone in enumerate(land_zones):
        circle = plt.Circle(zone['center'], zone['radius'], 
                          facecolor='lightgreen', edgecolor='green', 
                          alpha=0.6, linewidth=2, label=f'Land Zone {i+1}')
        ax1.add_patch(circle)
    
    # Draw navigation targets
    for i, target in enumerate(current_targets):
        target_pos = target['position']
        target_type = target['type']
        
        if target_type == 'swim':
            color = 'blue'
            marker = 'o'
            label_prefix = 'Swim'
        else:
            color = 'red'
            marker = 's'
            label_prefix = 'Land'
        
        ax1.plot(target_pos[0], target_pos[1], marker=marker, color=color, 
                markersize=10, markeredgecolor='white', markeredgewidth=2,
                label=f'{label_prefix} Target {i+1}')
        
        # Add target number annotation
        ax1.annotate(f'{i+1}', (target_pos[0], target_pos[1]), 
                    textcoords="offset points", xytext=(0,0), ha='center', va='center',
                    fontsize=8, fontweight='bold', color='white')
    
    # Plot trajectory
    if len(positions) > 1:
        positions = np.array(positions)
        
        # Color trajectory by environment type
        water_indices = [i for i, env_type in enumerate(env_types) if env_type == 'Water']
        land_indices = [i for i, env_type in enumerate(env_types) if env_type == 'Land']
        
        if water_indices:
            ax1.plot(positions[water_indices, 0], positions[water_indices, 1], 
                    'b-', linewidth=2, alpha=0.7, label='Water Movement')
        if land_indices:
            ax1.plot(positions[land_indices, 0], positions[land_indices, 1], 
                    'r-', linewidth=2, alpha=0.7, label='Land Movement')
        
        # Mark start and end points with enhanced visibility
        ax1.plot(positions[0, 0], positions[0, 1], 'ko', markersize=12, 
                markeredgecolor='white', markeredgewidth=2, label='Start')
        ax1.plot(positions[-1, 0], positions[-1, 1], 'ro', markersize=12, 
                markeredgecolor='white', markeredgewidth=2, label='End')
        
        # Add trajectory points (every 25th point for better visibility)
        sample_indices = range(0, len(positions), 25)
        ax1.plot(positions[sample_indices, 0], positions[sample_indices, 1], 
                'ko', markersize=4, alpha=0.7, markeredgecolor='yellow', 
                markeredgewidth=0.5, label='Trajectory Points')
        
        # Add swimmer direction arrows (every 100th point)
        arrow_indices = range(50, len(positions)-50, 100)
        for i in arrow_indices:
            if i+10 < len(positions):
                start_pos = positions[i]
                direction = positions[i+10] - positions[i]
                if np.linalg.norm(direction) > 0.01:  # Only if moving
                    direction = direction / np.linalg.norm(direction) * 0.3  # Normalize and scale
                    ax1.arrow(start_pos[0], start_pos[1], direction[0], direction[1],
                            head_width=0.1, head_length=0.05, fc='purple', ec='purple', alpha=0.7)
    
    ax1.legend(loc='upper right')
    ax1.set_aspect('equal')
    
    # 2. Distance Traveled Over Time (top-right)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title('Distance Traveled Over Time', fontsize=14, fontweight='bold')
    ax2.plot(distances, 'b-', linewidth=2)
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Distance From Start')
    ax2.grid(True, alpha=0.3)
    
    # Add final distance annotation
    if distances:
        final_distance = distances[-1]
        ax2.text(0.02, 0.98, f'Final Distance: {final_distance:.4f}', 
                transform=ax2.transAxes, fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                verticalalignment='top')
    
    # 3. Velocity Over Time (middle-left)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_title('Velocity Over Time', fontsize=14, fontweight='bold')
    ax3.plot(velocities, 'r-', linewidth=1, alpha=0.7)
    
    # Add moving average
    if len(velocities) > 10:
        window = min(50, len(velocities) // 4)
        moving_avg = np.convolve(velocities, np.ones(window)/window, mode='valid')
        ax3.plot(range(window-1, len(velocities)), moving_avg, 'gold', linewidth=2, label='Moving Avg')
        ax3.legend()
    
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Velocity Magnitude')
    ax3.grid(True, alpha=0.3)
    
    # 4. Environment Type Over Time (middle-right)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_title('Environment Type Over Time', fontsize=14, fontweight='bold')
    
    # Convert environment types to numeric for plotting
    env_numeric = [0 if env_type == 'Water' else 1 for env_type in env_types]
    ax4.plot(env_numeric, 'g-', linewidth=2)
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Environment Type')
    ax4.set_yticks([0, 1])
    ax4.set_yticklabels(['Water', 'Land'])
    ax4.grid(True, alpha=0.3)
    
    # Count transitions
    transitions = sum(1 for i in range(1, len(env_types)) if env_types[i] != env_types[i-1])
    ax4.text(0.02, 0.98, f'Transitions: {transitions}', 
            transform=ax4.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
            verticalalignment='top')
    
    # 5. Joint Torques Over Time (bottom, spanning both columns)
    ax5 = fig.add_subplot(gs[2, :])
    ax5.set_title('Joint Torques Over Time', fontsize=14, fontweight='bold')
    
    colors = ['blue', 'orange', 'green', 'red']
    for i in range(4):
        if joint_torques[i]:
            ax5.plot(joint_torques[i], color=colors[i], linewidth=1, 
                    alpha=0.8, label=f'Joint {i}')
    
    ax5.set_xlabel('Time Step')
    ax5.set_ylabel('Torque')
    ax5.legend(loc='upper right')
    ax5.grid(True, alpha=0.3)
    
    # Add overall title with phase info
    if phase_name:
        plt.suptitle(f'Trajectory Analysis - {phase_name}', fontsize=16, fontweight='bold')
    else:
        plt.suptitle(f'Trajectory Analysis - {datetime.now().strftime("%Y-%m-%d %H:%M")}', 
                    fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Trajectory analysis saved to: {save_path}")
    
    return {
        'final_distance': distances[-1] if distances else 0.0,
        'max_velocity': max(velocities) if velocities else 0.0,
        'transitions': transitions,
        'water_time': env_types.count('Water'),
        'land_time': env_types.count('Land')
    }


def create_curriculum_plots(phase_rewards, phase_distances, eval_results, save_path):
    """Create comprehensive plots for curriculum training progress."""
    
    # Create organized output directory
    plots_dir = os.path.join("outputs", "curriculum_training", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Update save path to use organized structure
    if not save_path.startswith("outputs/curriculum_training/"):
        filename = os.path.basename(save_path)
        save_path = os.path.join(plots_dir, filename)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Phase names for plotting
    phase_names = ["Pure Swimming", "Single Land Zone", "Two Land Zones", "Full Complexity"]
    colors = ['blue', 'green', 'orange', 'red']
    
    # 1. Reward progression by phase
    ax1.set_title('Reward Progress by Training Phase', fontsize=14, fontweight='bold')
    legend_added = False
    for phase in range(4):
        if phase in phase_rewards and len(phase_rewards[phase]) > 0:
            rewards = phase_rewards[phase]
            episodes = range(len(rewards))
            # Plot raw data with markers
            ax1.plot(episodes, rewards, color=colors[phase], alpha=0.4, linewidth=1, 
                    marker='o', markersize=2, label=f'Phase {phase}: {phase_names[phase]} (raw)')
            legend_added = True
            
            # Add moving average if enough data
            if len(rewards) > 10:
                window = min(50, len(rewards) // 4)
                moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
                ax1.plot(episodes[window-1:], moving_avg, color=colors[phase], 
                        linewidth=3, label=f'Phase {phase}: {phase_names[phase]} (avg)')
            elif len(rewards) > 1:
                # For short runs, just show a thicker line
                ax1.plot(episodes, rewards, color=colors[phase], linewidth=3, 
                        marker='o', markersize=4, label=f'Phase {phase}: {phase_names[phase]}')
    
    ax1.set_xlabel('Episodes in Phase')
    ax1.set_ylabel('Reward')
    if legend_added:
        ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # 2. Distance progression by phase
    ax2.set_title('Distance Progress by Training Phase', fontsize=14, fontweight='bold')
    legend_added_dist = False
    for phase in range(4):
        if phase in phase_distances and len(phase_distances[phase]) > 0:
            distances = phase_distances[phase]
            episodes = range(len(distances))
            # Plot raw data with markers
            ax2.plot(episodes, distances, color=colors[phase], alpha=0.4, linewidth=1, 
                    marker='s', markersize=2, label=f'Phase {phase}: {phase_names[phase]} (raw)')
            legend_added_dist = True
            
            # Add moving average if enough data
            if len(distances) > 10:
                window = min(50, len(distances) // 4)
                moving_avg = np.convolve(distances, np.ones(window)/window, mode='valid')
                ax2.plot(episodes[window-1:], moving_avg, color=colors[phase], 
                        linewidth=3, label=f'Phase {phase}: {phase_names[phase]} (avg)')
            elif len(distances) > 1:
                # For short runs, just show a thicker line
                ax2.plot(episodes, distances, color=colors[phase], linewidth=3, 
                        marker='s', markersize=4, label=f'Phase {phase}: {phase_names[phase]}')
    
    ax2.set_xlabel('Episodes in Phase')
    ax2.set_ylabel('Distance (m)')
    if legend_added_dist:
        ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 3. Performance comparison across phases
    ax3.set_title('Final Performance by Phase', fontsize=14, fontweight='bold')
    phases = list(eval_results.keys())
    distances = [eval_results[p]['mean_distance'] for p in phases]
    errors = [eval_results[p]['std_distance'] for p in phases]
    
    bars = ax3.bar([phase_names[p] for p in phases], distances, 
                   color=[colors[p] for p in phases], alpha=0.7,
                   yerr=errors, capsize=5)
    
    ax3.set_ylabel('Distance (m)')
    ax3.set_xlabel('Training Phase')
    
    # Add value labels on bars
    for i, (bar, dist) in enumerate(zip(bars, distances)):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + errors[i] + 0.1,
                f'{dist:.2f}m', ha='center', va='bottom', fontweight='bold')
    
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Training progression summary
    ax4.set_title('Curriculum Training Summary', fontsize=14, fontweight='bold')
    
    # Calculate total episodes per phase
    phase_episode_counts = []
    phase_avg_rewards = []
    phase_avg_distances = []
    
    for phase in range(4):
        episode_count = len(phase_rewards.get(phase, []))
        avg_reward = np.mean(phase_rewards.get(phase, [0])) if phase_rewards.get(phase) else 0
        avg_distance = np.mean(phase_distances.get(phase, [0])) if phase_distances.get(phase) else 0
        
        phase_episode_counts.append(episode_count)
        phase_avg_rewards.append(avg_reward)
        phase_avg_distances.append(avg_distance)
    
    # Create summary table
    ax4.axis('tight')
    ax4.axis('off')
    
    table_data = []
    for i, phase_name in enumerate(phase_names):
        if i < len(phase_episode_counts):
            table_data.append([
                phase_name,
                f"{phase_episode_counts[i]}",
                f"{phase_avg_rewards[i]:.2f}",
                f"{phase_avg_distances[i]:.3f}m"
            ])
    
    table = ax4.table(cellText=table_data,
                     colLabels=['Phase', 'Episodes', 'Avg Reward', 'Avg Distance'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    plt.suptitle(f'Curriculum Training Analysis - {datetime.now().strftime("%Y-%m-%d %H:%M")}', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Create directory if needed
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    try:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📊 Curriculum training plots saved to: {save_path}")
        
        # Verify file was actually created
        if os.path.exists(save_path):
            file_size = os.path.getsize(save_path)
            print(f"✅ Plot file verified: {file_size} bytes")
        else:
            print(f"❌ Plot file was not created at: {save_path}")
            
    except Exception as e:
        print(f"❌ Error saving curriculum plots: {e}")
        plt.close()


def add_zone_indicators(frame, env, step_count, minimap=True):
    """Add visual indicators for environment zones to video frame."""
    try:
        import cv2
        import numpy as np
        
        frame_with_zones = frame.copy()
        height, width = frame.shape[:2]
        
        # Get current zones from environment
        land_zones = []
        phase_name = ""
        progress = 0.0
        
        if hasattr(env, 'env') and hasattr(env.env, 'env') and hasattr(env.env.env, '_task'):
            task = env.env.env._task
            progress = getattr(task, '_training_progress', 0.0)
            
            # Get current land zones
            if hasattr(task, '_current_land_zones'):
                land_zones = task._current_land_zones or []
                
                # Phase detection
                if progress < 0.3:
                    phase_name = "Phase 1: Pure Swimming"
                elif progress < 0.6:
                    phase_name = "Phase 2: Single Land Zone"
                elif progress < 0.8:
                    phase_name = "Phase 3: Two Land Zones" 
                else:
                    phase_name = "Phase 4: Full Complexity"
                
                # Debug: Print zone info (first few frames only to avoid spam) - reduced for video creation
                if step_count < 2:  # Reduced from 5 to 2
                    print(f"🔍 Debug zone info: Progress={progress:.3f}, Phase='{phase_name}', Land zones={len(land_zones)}")
                    if land_zones:
                        for i, zone in enumerate(land_zones):
                            print(f"   Zone {i}: center={zone['center']}, radius={zone['radius']}")
                
                # Convert physics coordinates to screen coordinates
                # Use wider range to ensure zones are visible
                x_range = 12.0  # -6 to 6 (wider than previous -5 to 5)
                y_range = 8.0   # -4 to 4 (wider than previous -3 to 3)
                
                # Add land zone overlays
                for i, zone in enumerate(land_zones):
                    center_x, center_y = zone['center']
                    radius = zone['radius']
                    
                    # Convert to screen coordinates with wider view
                    screen_x = int((center_x + 6.0) / x_range * width)
                    screen_y = int(height - (center_y + 4.0) / y_range * height)  # Flip Y for screen coords
                    screen_radius = max(10, int(radius / x_range * width))  # Minimum visible radius
                    
                    # Debug: Print coordinate conversion (reduced for video creation)
                    if step_count < 1:  # Reduced from 3 to 1
                        print(f"   Zone {i} coords: physics=({center_x}, {center_y}, r={radius}) → screen=({screen_x}, {screen_y}, r={screen_radius})")
                    
                    # Ensure coordinates are within frame bounds
                    if 0 <= screen_x < width and 0 <= screen_y < height:
                        # Draw land zone (brown circle with transparency)
                        overlay = frame_with_zones.copy()
                        cv2.circle(overlay, (screen_x, screen_y), screen_radius, (101, 67, 33), -1)  # Brown fill
                        cv2.circle(overlay, (screen_x, screen_y), screen_radius, (139, 69, 19), 3)   # Brown border
                        
                        # Apply transparency
                        alpha = 0.4  # Slightly more opaque
                        frame_with_zones = cv2.addWeighted(frame_with_zones, 1-alpha, overlay, alpha, 0)
                        
                        # Add zone label
                        label = f"Land {i+1}"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
                        label_x = max(5, screen_x - label_size[0] // 2)
                        label_y = max(20, screen_y - screen_radius - 10)
                        
                        cv2.rectangle(frame_with_zones, (label_x-3, label_y-18), 
                                    (label_x + label_size[0]+3, label_y+3), (255, 255, 255), -1)
                        cv2.putText(frame_with_zones, label, (label_x, label_y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (139, 69, 19), 2)
        
        # Note: Target visualization removed from simple zone indicators
        # Use add_zone_indicators_with_trail for full target visualization
        
        # Add water zone indicator (if not pure land)
        if len(land_zones) > 0:
            cv2.putText(frame_with_zones, "Water Zone", (10, height-60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 144, 255), 2)  # Blue text
        
        # Add phase indicator with enhanced info
        if phase_name:
            # Main phase indicator
            cv2.rectangle(frame_with_zones, (5, 5), (400, 35), (0, 0, 0), -1)
            cv2.putText(frame_with_zones, phase_name, (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Debug info: Show progress and zone count
            debug_text = f"Progress: {progress:.1%} | Zones: {len(land_zones)}"
            cv2.rectangle(frame_with_zones, (5, 40), (350, 65), (0, 0, 0), -1)
            cv2.putText(frame_with_zones, debug_text, (10, 57), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Add step counter
        step_text = f"Step: {step_count}"
        cv2.putText(frame_with_zones, step_text, (width-150, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add training status indicator
        status_text = "CURRICULUM TRAINING"
        status_color = (0, 139, 139)  # Dark yellow/orange
        text_color = (255, 255, 255)  # White
        cv2.rectangle(frame_with_zones, (width-220, height-40), (width-5, height-10), status_color, -1)
        cv2.putText(frame_with_zones, status_text, (width-215, height-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
        
        # Add minimap if requested and zones exist
        if minimap and len(land_zones) > 0:
            try:
                # Get swimmer position
                swimmer_pos = None
                if hasattr(env, 'env') and hasattr(env.env, 'env'):
                    physics = env.env.env.physics
                    swimmer_pos = physics.named.data.xpos['head'][:2]  # X, Y only
                
                if swimmer_pos is not None:
                    frame_with_zones = add_minimap(frame_with_zones, land_zones, swimmer_pos, width, height)
            except Exception as e:
                print(f"⚠️ Minimap error: {e}")
        
        return frame_with_zones
        
    except Exception as e:
        print(f"⚠️ Zone overlay error: {e}")
        return frame


def create_test_video(agent, env, save_path, num_steps=300, episode_name="Test Episode", show_minimap=True):
    """Create a video of the agent performing in the environment with zone indicators."""
    
    # Create organized output directory
    videos_dir = os.path.join("outputs", "curriculum_training", "videos")
    os.makedirs(videos_dir, exist_ok=True)
    
    # Update save path to use organized structure
    if not save_path.startswith("outputs/curriculum_training/"):
        filename = os.path.basename(save_path)
        save_path = os.path.join(videos_dir, filename)
    
    print(f"🎬 Creating test video: {episode_name}")
    
    frames = []
    position_history = []  # Track swimmer position for trail
    obs = env.reset()
    
    for step in range(num_steps):
        # Get swimmer position for tracking
        try:
            if hasattr(env, 'env') and hasattr(env.env, 'env'):
                physics = env.env.env.physics
                swimmer_pos = physics.named.data.xpos['head'][:2].copy()
                position_history.append(swimmer_pos)
                
                # Keep only recent positions for trail (last 50 steps)
                if len(position_history) > 50:
                    position_history = position_history[-50:]
        except:
            pass
        
        # Render frame
        try:
            frame = env.render(mode='rgb_array')
            if frame is not None:
                # Add zone indicators with minimap
                frame_with_zones = add_zone_indicators_with_trail(
                    frame, env, step, position_history, show_minimap
                )
                frames.append(frame_with_zones)
        except Exception as e:
            print(f"⚠️ Render error at step {step}: {e}")
            break
        
        # Get action and step
        try:
            action = agent.test_step(obs)
            obs, reward, done, _ = env.step(action)
            
            if done:
                obs = env.reset()
        except Exception as e:
            print(f"⚠️ Step error at step {step}: {e}")
            break
    
    # Save video if we have frames
    if len(frames) > 10:  # At least 10 frames for a meaningful video
        try:
            # Create directory if needed
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Save video
            imageio.mimsave(save_path, frames, fps=30)
            print(f"🎬 Test video saved to: {save_path} ({len(frames)} frames)")
            
        except Exception as e:
            print(f"❌ Video save error: {e}")
    else:
        print(f"⚠️ Not enough frames ({len(frames)}) to create video")


def create_phase_comparison_video(agent, env, save_path, phases_to_test=None):
    """Create a video showing performance across different phases."""
    
    # Create organized output directory
    videos_dir = os.path.join("outputs", "curriculum_training", "videos")
    os.makedirs(videos_dir, exist_ok=True)
    
    # Update save path to use organized structure
    if not save_path.startswith("outputs/curriculum_training/"):
        filename = os.path.basename(save_path)
        save_path = os.path.join(videos_dir, filename)
    
    if phases_to_test is None:
        phases_to_test = [0, 1, 2, 3]  # All phases
    
    phase_names = ["Pure Swimming", "Single Land Zone", "Two Land Zones", "Full Complexity"]
    all_frames = []
    
    # Create progress bar for video recording (500 steps per phase + transitions)
    total_steps = len(phases_to_test) * 500 + (len(phases_to_test) - 1) * 30  # 30 transition frames
    
    with tqdm(total=total_steps, desc="🎬 Recording Video", unit="frame",
             bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as video_pbar:
        
        for phase in phases_to_test:
            video_pbar.set_description(f"🎬 Recording {phase_names[phase]}")
            
            # Set environment to specific phase using manual override
            temp_progress = (phase + 0.5) * 0.25  # Middle of each phase
            env.env.set_manual_progress(temp_progress)
            
            # Record this phase
            phase_frames = []
            obs = env.reset()
            
            # Track swimmer position for trail in this phase
            phase_position_history = []
            
            for step in range(500):  # 500 steps per phase (16+ seconds each)
                try:
                    # Track swimmer position
                    try:
                        if hasattr(env, 'env') and hasattr(env.env, 'env'):
                            physics = env.env.env.physics
                            swimmer_pos = physics.named.data.xpos['head'][:2].copy()
                            phase_position_history.append(swimmer_pos)
                            
                            # Keep trail length reasonable (last 30 steps)
                            if len(phase_position_history) > 30:
                                phase_position_history = phase_position_history[-30:]
                    except:
                        pass
                    
                    frame = env.render(mode='rgb_array')
                    if frame is not None:
                        # Use enhanced zone indicators with targets and minimap for phase comparison
                        frame_with_zones = add_zone_indicators_with_trail(
                            frame, env, step, phase_position_history, show_minimap=True
                        )
                        phase_frames.append(frame_with_zones)
                    
                    action = agent.test_step(obs)
                    obs, reward, done, _ = env.step(action)
                    
                    if done:
                        obs = env.reset()
                    
                    # Update progress every 10 frames to avoid spam
                    if step % 10 == 0:
                        video_pbar.update(10)
                        
                except Exception as e:
                    print(f"⚠️ Error in phase {phase}, step {step}: {e}")
                    break
            
            # Update for any remaining frames
            remaining_frames = 500 - (step // 10) * 10
            if remaining_frames > 0:
                video_pbar.update(remaining_frames)
            
            all_frames.extend(phase_frames)
            
            # Add transition frames (black screen with text)
            if phase < max(phases_to_test):
                video_pbar.set_description(f"🎬 Adding transition")
                transition_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                for _ in range(30):  # 1 second transition at 30fps
                    all_frames.append(transition_frame)
                video_pbar.update(30)
    
        # Save combined video
        video_pbar.set_description(f"🎬 Saving video file")
        if len(all_frames) > 50:
            try:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                imageio.mimsave(save_path, all_frames, fps=30)
                print(f"🎬 Phase comparison video saved to: {save_path} ({len(all_frames)} frames)")
            except Exception as e:
                print(f"❌ Video save error: {e}")
        else:
            print(f"⚠️ Not enough frames for phase comparison video")


def save_training_summary(eval_results, training_history, save_path):
    """Save a text summary of training results."""
    
    # Create organized output directory
    summaries_dir = os.path.join("outputs", "curriculum_training", "summaries")
    os.makedirs(summaries_dir, exist_ok=True)
    
    # Update save path to use organized structure
    if not save_path.startswith("outputs/curriculum_training/"):
        filename = os.path.basename(save_path)
        save_path = os.path.join(summaries_dir, filename)
    
    phase_names = ["Pure Swimming", "Single Land Zone", "Two Land Zones", "Full Complexity"]
    
    with open(save_path, 'w') as f:
        f.write("# Curriculum Training Summary\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Final Performance by Phase\n")
        for phase, results in eval_results.items():
            f.write(f"**{phase_names[phase]}**:\n")
            f.write(f"  - Mean Distance: {results['mean_distance']:.3f}m ± {results['std_distance']:.3f}\n")
            f.write(f"  - Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}\n\n")
        
        # Add trajectory analysis if available
        if 'trajectory_stats' in training_history:
            f.write("## Trajectory Analysis\n")
            for phase, stats in training_history['trajectory_stats'].items():
                f.write(f"**{phase_names[phase]}**:\n")
                f.write(f"  - Final Distance: {stats['final_distance']:.3f}m\n")
                f.write(f"  - Max Velocity: {stats['max_velocity']:.3f}\n")
                f.write(f"  - Environment Transitions: {stats['transitions']}\n")
                f.write(f"  - Time in Water: {stats['water_time']} steps\n")
                f.write(f"  - Time on Land: {stats['land_time']} steps\n\n")
        
        f.write("## Training Progress\n")
        for phase in range(4):
            if phase in training_history['phase_rewards']:
                episodes = len(training_history['phase_rewards'][phase])
                avg_reward = np.mean(training_history['phase_rewards'][phase]) if episodes > 0 else 0
                avg_distance = np.mean(training_history['phase_distances'][phase]) if episodes > 0 else 0
                
                f.write(f"**Phase {phase} - {phase_names[phase]}**:\n")
                f.write(f"  - Episodes: {episodes}\n")
                f.write(f"  - Average Reward: {avg_reward:.2f}\n")
                f.write(f"  - Average Distance: {avg_distance:.3f}m\n\n")
        
        f.write("## Generated Files\n")
        f.write("### Plots\n")
        f.write("- `curriculum_final_plots.png` - Training progress summary\n")
        f.write("- `final_trajectory_phase_0.png` - Pure Swimming trajectory analysis\n")
        f.write("- `final_trajectory_phase_1.png` - Single Land Zone trajectory analysis\n")
        f.write("- `final_trajectory_phase_2.png` - Two Land Zones trajectory analysis\n")
        f.write("- `final_trajectory_phase_3.png` - Full Complexity trajectory analysis\n")
        f.write("- `trajectory_analysis_step_*.png` - Periodic trajectory analyses\n\n")
        
        f.write("### Videos\n")
        f.write("- `curriculum_final_video.mp4` - Phase comparison demonstration\n")
        f.write("- `curriculum_video_step_*.mp4` - Periodic training videos with zone indicators\n\n")
        
        f.write("### Models\n")
        f.write("- `curriculum_final_model_*links.pt` - Final trained model\n\n")
    
    print(f"📄 Training summary saved to: {save_path}") 