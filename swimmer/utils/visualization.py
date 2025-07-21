#!/usr/bin/env python3
"""
Visualization and logging utilities for swimmer training.
Contains functions for creating plots, videos, and parameter logs.
"""

import numpy as np
import matplotlib.pyplot as plt
import imageio
from datetime import datetime
from ..environments.environment_types import EnvironmentType

def create_comprehensive_visualization(task, results, save_path):
    """Create comprehensive visualization of environment zones, swimmer trajectory, and performance."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Debug: Print actual trajectory data
    if hasattr(task, 'position_history') and task.position_history:
        positions = np.array(task.position_history)
        print(f"Trajectory bounds: X=[{positions[:, 0].min():.3f}, {positions[:, 0].max():.3f}], Y=[{positions[:, 1].min():.3f}, {positions[:, 1].max():.3f}]")
        print(f"Zone centers: Water={task.water_zones}, Land={task.land_zones}")
        
        # Check which positions are actually in zones
        water_positions = []
        land_positions = []
        for i, pos in enumerate(positions):
            in_water = any(np.linalg.norm(pos - np.array(zone['center'])) <= zone['radius'] for zone in task.water_zones)
            in_land = any(np.linalg.norm(pos - np.array(zone['center'])) <= zone['radius'] for zone in task.land_zones)
            if in_water:
                water_positions.append(pos)
            elif in_land:
                land_positions.append(pos)
        
        print(f"Positions in water zones: {len(water_positions)}")
        print(f"Positions in land zones: {len(land_positions)}")
        print(f"Total positions: {len(positions)}")
    
    # Plot 1: Environment zones and swimmer trajectory
    ax1.set_title('Environment Zones and Swimmer Trajectory', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.grid(True, alpha=0.3)
    
    # Draw water zones
    for i, zone in enumerate(task.water_zones):
        circle = plt.Circle(zone['center'], zone['radius'], 
                          facecolor='lightblue', edgecolor='blue', 
                          alpha=0.6, linewidth=2, label=f'Water Zone {i+1}' if i == 0 else "")
        ax1.add_patch(circle)
    
    # Draw land zones
    for i, zone in enumerate(task.land_zones):
        circle = plt.Circle(zone['center'], zone['radius'], 
                          facecolor='lightgreen', edgecolor='green', 
                          alpha=0.6, linewidth=2, label=f'Land Zone {i+1}' if i == 0 else "")
        ax1.add_patch(circle)
    
    # Plot trajectory with environment colors
    if hasattr(task, 'position_history') and task.position_history:
        positions = np.array(task.position_history)
        env_history = task.env_history
        
        # Ensure positions and environment history have matching dimensions
        min_length = min(len(positions), len(env_history))
        positions = positions[:min_length]
        env_history = env_history[:min_length]
        
        # Create masks for different environments
        water_mask = [env == EnvironmentType.WATER for env in env_history]
        land_mask = [env == EnvironmentType.LAND for env in env_history]
        
        if any(water_mask):
            ax1.plot(positions[water_mask, 0], positions[water_mask, 1], 
                    'b-', linewidth=3, label='Water Movement')
        if any(land_mask):
            ax1.plot(positions[land_mask, 0], positions[land_mask, 1], 
                    'g-', linewidth=3, label='Land Movement')
        
        # Mark start and end points
        ax1.plot(positions[0, 0], positions[0, 1], 'ko', markersize=12, label='Start')
        ax1.plot(positions[-1, 0], positions[-1, 1], 'ro', markersize=12, label='End')
        
        # Add trajectory points every 100 steps
        step_indices = range(0, len(positions), 100)
        if len(step_indices) > 1:
            ax1.plot(positions[step_indices, 0], positions[step_indices, 1], 
                    'k.', markersize=4, alpha=0.5, label='Trajectory Points')
    
    ax1.legend()
    ax1.set_aspect('equal')
    ax1.set_xlim(-4, 4)
    ax1.set_ylim(-2, 2)
    
    # Plot 2: Distance over time
    ax2.set_title('Distance Traveled Over Time', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Distance from Start')
    ax2.grid(True, alpha=0.3)
    
    if results and 'distances' in results:
        ax2.plot(results['distances'], 'b-', linewidth=2)
        ax2.set_ylim(bottom=0)
        
        # Add final distance annotation
        final_distance = results['distances'][-1] if results['distances'] else 0
        ax2.text(0.02, 0.98, f'Final Distance: {final_distance:.4f}', 
                transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot 3: Velocity over time
    ax3.set_title('Velocity Over Time', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Velocity Magnitude')
    ax3.grid(True, alpha=0.3)
    
    if results and 'velocities' in results:
        ax3.plot(results['velocities'], 'r-', linewidth=2)
        ax3.set_ylim(bottom=0)
        
        # Add moving average
        if len(results['velocities']) > 50:
            window = 50
            moving_avg = [np.mean(results['velocities'][max(0, i-window):i+1]) 
                         for i in range(len(results['velocities']))]
            ax3.plot(moving_avg, 'orange', linewidth=2, alpha=0.7, label='Moving Avg')
            ax3.legend()
    
    # Plot 4: Environment transitions
    ax4.set_title('Environment Type Over Time', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Environment Type')
    ax4.grid(True, alpha=0.3)
    
    if hasattr(task, 'env_history') and task.env_history:
        env_numeric = [1 if env == EnvironmentType.WATER else 0 for env in task.env_history]
        ax4.plot(env_numeric, 'g-', linewidth=2)
        ax4.set_yticks([0, 1])
        ax4.set_yticklabels(['Land', 'Water'])
        ax4.set_ylim(-0.1, 1.1)
        
        # Count transitions
        transitions = sum(1 for i in range(1, len(env_numeric)) if env_numeric[i] != env_numeric[i-1])
        ax4.text(0.02, 0.98, f'Transitions: {transitions}', 
                transform=ax4.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Comprehensive visualization saved as {save_path}")

def create_parameter_log(task, results, n_links, oscillator_period, amplitude, save_path):
    """Create a comprehensive parameter log file with all environment and performance information."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    log_content = f"""
=== SWIMMER EXPERIMENT PARAMETER LOG ===
Timestamp: {timestamp}
Experiment: Improved Mixed Environment Swimmer

=== ENVIRONMENT PARAMETERS ===
Swimmer Links: {n_links}
Oscillator Period: {oscillator_period}
Base Amplitude: {amplitude}

=== ZONE CONFIGURATION ===
Water Zones:
"""
    
    for i, zone in enumerate(task.water_zones):
        log_content += f"  Zone {i+1}: center={zone['center']}, radius={zone['radius']}\n"
    
    log_content += "\nLand Zones:\n"
    for i, zone in enumerate(task.land_zones):
        log_content += f"  Zone {i+1}: center={zone['center']}, radius={zone['radius']}\n"
    
    log_content += f"""
=== PHYSICS PROPERTIES ===
Water Environment:
  - Viscosity: 0.01
  - Density: 100.0
  - Friction: [0.01, 0.001, 0.001]

Land Environment:
  - Viscosity: 0.0
  - Density: 1.0
  - Friction: [1.5, 0.3, 0.3]

=== SWIMMER MODEL PARAMETERS ===
NCAP Architecture:
  - Joints: {task.n_joints if hasattr(task, 'n_joints') else 'Unknown'}
  - Memory Size: 10
  - Environment Modulation: Linear layers
  - Oscillator Type: Square wave
  - Phase Offsets: Linear spacing

=== PERFORMANCE METRICS ===
Total Distance Traveled: {results['total_distance']:.4f}
Average Velocity: {results['avg_velocity']:.4f}
Maximum Velocity: {results['max_velocity']:.4f}
Average Reward: {results['avg_reward']:.4f}
Environment Transitions: {results['env_transitions']}

=== TRAJECTORY ANALYSIS ===
"""
    
    if hasattr(task, 'position_history') and task.position_history:
        positions = np.array(task.position_history)
        log_content += f"""
Trajectory Bounds:
  - X Range: [{positions[:, 0].min():.4f}, {positions[:, 0].max():.4f}]
  - Y Range: [{positions[:, 1].min():.4f}, {positions[:, 1].max():.4f}]
  - Total Steps: {len(positions)}

Zone Analysis:
"""
        
        # Count positions in each zone
        water_count = 0
        land_count = 0
        neither_count = 0
        
        for pos in positions:
            in_water = any(np.linalg.norm(pos - np.array(zone['center'])) <= zone['radius'] for zone in task.water_zones)
            in_land = any(np.linalg.norm(pos - np.array(zone['center'])) <= zone['radius'] for zone in task.land_zones)
            
            if in_water:
                water_count += 1
            elif in_land:
                land_count += 1
            else:
                neither_count += 1
        
        log_content += f"""
  - Positions in Water Zones: {water_count} ({water_count/len(positions)*100:.1f}%)
  - Positions in Land Zones: {land_count} ({land_count/len(positions)*100:.1f}%)
  - Positions in Neither: {neither_count} ({neither_count/len(positions)*100:.1f}%)
"""
    
    log_content += f"""
=== ENVIRONMENT TRANSITIONS ===
Transition History:
"""
    
    if hasattr(task, 'env_history') and task.env_history:
        transitions = []
        for i in range(1, len(task.env_history)):
            if task.env_history[i] != task.env_history[i-1]:
                transitions.append(f"Step {i}: {task.env_history[i-1]} -> {task.env_history[i]}")
        
        for transition in transitions[:10]:  # Show first 10 transitions
            log_content += f"  {transition}\n"
        
        if len(transitions) > 10:
            log_content += f"  ... and {len(transitions) - 10} more transitions\n"
    
    log_content += f"""
=== FILES GENERATED ===
Video: outputs/improved_mixed_env/improved_adaptation_{n_links}links.mp4
Plots: outputs/improved_mixed_env/improved_environment_analysis_{n_links}links.png
Parameter Log: {save_path}

=== NOTES ===
- Environment transitions are detected based on swimmer head position
- Physics properties change dynamically based on current environment
- NCAP model adapts oscillator parameters based on environment type
- Square wave oscillators provide more aggressive movement than sine waves
"""
    
    # Write to file
    with open(save_path, 'w') as f:
        f.write(log_content)
    
    print(f"Parameter log saved as {save_path}")

def add_zone_overlay(frame, task, current_env):
    """Add visual zone overlay to the rendered frame."""
    # For now, just return the original frame to avoid overlay issues
    # The zone visualization will be shown in the plots instead
    return frame 