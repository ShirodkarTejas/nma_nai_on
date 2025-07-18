#!/usr/bin/env python3
"""
Training Logger for Swimmer Training
Handles logging training metrics, creating charts, and saving training progress.
"""

import os
import json
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict

class TrainingLogger:
    """
    Comprehensive training logger for tracking and visualizing training progress.
    """
    def __init__(self, log_dir='outputs/training_logs', experiment_name=None):
        self.log_dir = log_dir
        self.experiment_name = experiment_name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.log_path = os.path.join(log_dir, self.experiment_name)
        
        # Create log directory
        os.makedirs(self.log_path, exist_ok=True)
        
        # Initialize metrics storage
        self.metrics = defaultdict(list)
        self.episode_data = []
        self.training_config = {}
        
        # Training state
        self.start_time = None
        self.current_step = 0
        self.current_episode = 0
        
        # Log file paths
        self.metrics_file = os.path.join(self.log_path, 'metrics.json')
        self.config_file = os.path.join(self.log_path, 'config.json')
        self.episode_file = os.path.join(self.log_path, 'episodes.json')
        
        print(f"Training logger initialized. Logs will be saved to: {self.log_path}")
    
    def log_config(self, config_dict):
        """Log training configuration."""
        self.training_config = config_dict
        self.training_config['experiment_name'] = self.experiment_name
        self.training_config['start_time'] = datetime.now().isoformat()
        
        with open(self.config_file, 'w') as f:
            json.dump(self.training_config, f, indent=2)
        
        print(f"Configuration logged to {self.config_file}")
    
    def start_training(self):
        """Start training session."""
        self.start_time = time.time()
        print(f"Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    def log_metric(self, metric_name, value, step=None):
        """Log a single metric."""
        if step is None:
            step = self.current_step
        
        self.metrics[metric_name].append({
            'step': step,
            'value': value,
            'timestamp': time.time()
        })
    
    def log_episode(self, episode_reward, episode_length, episode_distance, 
                   env_transitions=None, avg_velocity=None, max_velocity=None):
        """Log episode results."""
        episode_data = {
            'episode': self.current_episode,
            'reward': episode_reward,
            'length': episode_length,
            'distance': episode_distance,
            'env_transitions': env_transitions,
            'avg_velocity': avg_velocity,
            'max_velocity': max_velocity,
            'timestamp': time.time()
        }
        
        self.episode_data.append(episode_data)
        self.current_episode += 1
        
        # Log episode metrics
        self.log_metric('episode_reward', episode_reward)
        self.log_metric('episode_length', episode_length)
        self.log_metric('episode_distance', episode_distance)
        if env_transitions is not None:
            self.log_metric('env_transitions', env_transitions)
        if avg_velocity is not None:
            self.log_metric('avg_velocity', avg_velocity)
        if max_velocity is not None:
            self.log_metric('max_velocity', max_velocity)
    
    def log_training_step(self, loss=None, policy_loss=None, value_loss=None, 
                         entropy=None, learning_rate=None):
        """Log training step metrics."""
        if loss is not None:
            self.log_metric('loss', loss)
        if policy_loss is not None:
            self.log_metric('policy_loss', policy_loss)
        if value_loss is not None:
            self.log_metric('value_loss', value_loss)
        if entropy is not None:
            self.log_metric('entropy', entropy)
        if learning_rate is not None:
            self.log_metric('learning_rate', learning_rate)
        
        self.current_step += 1
    
    def save_metrics(self):
        """Save all metrics to file."""
        with open(self.metrics_file, 'w') as f:
            json.dump(dict(self.metrics), f, indent=2)
        
        with open(self.episode_file, 'w') as f:
            json.dump(self.episode_data, f, indent=2)
    
    def create_training_plots(self, save_plots=True):
        """Create comprehensive training visualization plots."""
        if not self.metrics:
            print("No metrics to plot")
            return
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Training Progress - {self.experiment_name}', fontsize=16, fontweight='bold')
        
        # Plot 1: Episode Rewards
        if 'episode_reward' in self.metrics:
            rewards = [m['value'] for m in self.metrics['episode_reward']]
            episodes = [m['step'] for m in self.metrics['episode_reward']]
            
            ax1.plot(episodes, rewards, 'b-', alpha=0.7, label='Episode Reward')
            
            # Add moving average
            if len(rewards) > 10:
                window = min(10, len(rewards) // 10)
                moving_avg = pd.Series(rewards).rolling(window=window).mean()
                ax1.plot(episodes, moving_avg, 'r-', linewidth=2, label=f'Moving Avg ({window})')
            
            ax1.set_title('Episode Rewards Over Time')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Reward')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Plot 2: Episode Distances
        if 'episode_distance' in self.metrics:
            distances = [m['value'] for m in self.metrics['episode_distance']]
            episodes = [m['step'] for m in self.metrics['episode_distance']]
            
            ax2.plot(episodes, distances, 'g-', alpha=0.7, label='Episode Distance')
            
            # Add moving average
            if len(distances) > 10:
                window = min(10, len(distances) // 10)
                moving_avg = pd.Series(distances).rolling(window=window).mean()
                ax2.plot(episodes, moving_avg, 'orange', linewidth=2, label=f'Moving Avg ({window})')
            
            ax2.set_title('Episode Distances Over Time')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Distance')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Training Losses
        if 'loss' in self.metrics or 'policy_loss' in self.metrics or 'value_loss' in self.metrics:
            if 'loss' in self.metrics:
                losses = [m['value'] for m in self.metrics['loss']]
                steps = [m['step'] for m in self.metrics['loss']]
                ax3.plot(steps, losses, 'b-', alpha=0.7, label='Total Loss')
            
            if 'policy_loss' in self.metrics:
                policy_losses = [m['value'] for m in self.metrics['policy_loss']]
                steps = [m['step'] for m in self.metrics['policy_loss']]
                ax3.plot(steps, policy_losses, 'r-', alpha=0.7, label='Policy Loss')
            
            if 'value_loss' in self.metrics:
                value_losses = [m['value'] for m in self.metrics['value_loss']]
                steps = [m['step'] for m in self.metrics['value_loss']]
                ax3.plot(steps, value_losses, 'g-', alpha=0.7, label='Value Loss')
            
            ax3.set_title('Training Losses')
            ax3.set_xlabel('Training Step')
            ax3.set_ylabel('Loss')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Environment Transitions and Velocity
        if 'env_transitions' in self.metrics:
            transitions = [m['value'] for m in self.metrics['env_transitions']]
            episodes = [m['step'] for m in self.metrics['env_transitions']]
            
            ax4_twin = ax4.twinx()
            ax4.plot(episodes, transitions, 'purple', alpha=0.7, label='Env Transitions')
            ax4.set_xlabel('Episode')
            ax4.set_ylabel('Environment Transitions', color='purple')
            ax4.tick_params(axis='y', labelcolor='purple')
            
            if 'avg_velocity' in self.metrics:
                velocities = [m['value'] for m in self.metrics['avg_velocity']]
                episodes = [m['step'] for m in self.metrics['avg_velocity']]
                ax4_twin.plot(episodes, velocities, 'orange', alpha=0.7, label='Avg Velocity')
                ax4_twin.set_ylabel('Average Velocity', color='orange')
                ax4_twin.tick_params(axis='y', labelcolor='orange')
            
            ax4.set_title('Environment Transitions and Velocity')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = os.path.join(self.log_path, 'training_plots.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Training plots saved to {plot_path}")
        
        plt.show()
    
    def create_summary_report(self):
        """Create a comprehensive training summary report."""
        if not self.metrics:
            return "No training data available."
        
        # Calculate summary statistics
        if 'episode_reward' in self.metrics:
            rewards = [m['value'] for m in self.metrics['episode_reward']]
            avg_reward = np.mean(rewards)
            max_reward = np.max(rewards)
            min_reward = np.min(rewards)
            final_reward = rewards[-1] if rewards else 0
        else:
            avg_reward = max_reward = min_reward = final_reward = 0
        
        if 'episode_distance' in self.metrics:
            distances = [m['value'] for m in self.metrics['episode_distance']]
            avg_distance = np.mean(distances)
            max_distance = np.max(distances)
            final_distance = distances[-1] if distances else 0
        else:
            avg_distance = max_distance = final_distance = 0
        
        # Training duration
        duration = time.time() - self.start_time if self.start_time else 0
        duration_str = f"{duration/3600:.2f} hours" if duration > 3600 else f"{duration/60:.2f} minutes"
        
        # Create summary
        summary = f"""
=== TRAINING SUMMARY REPORT ===
Experiment: {self.experiment_name}
Duration: {duration_str}
Total Episodes: {self.current_episode}
Total Training Steps: {self.current_step}

=== PERFORMANCE METRICS ===
Rewards:
  - Average: {avg_reward:.4f}
  - Maximum: {max_reward:.4f}
  - Minimum: {min_reward:.4f}
  - Final: {final_reward:.4f}

Distances:
  - Average: {avg_distance:.4f}
  - Maximum: {max_distance:.4f}
  - Final: {final_distance:.4f}

=== CONFIGURATION ===
"""
        
        for key, value in self.training_config.items():
            if key not in ['experiment_name', 'start_time']:
                summary += f"  {key}: {value}\n"
        
        summary += f"""
=== FILES GENERATED ===
- Configuration: {self.config_file}
- Metrics: {self.metrics_file}
- Episodes: {self.episode_file}
- Plots: {os.path.join(self.log_path, 'training_plots.png')}
"""
        
        # Save summary
        summary_path = os.path.join(self.log_path, 'summary_report.txt')
        with open(summary_path, 'w') as f:
            f.write(summary)
        
        print(summary)
        return summary
    
    def get_latest_metrics(self):
        """Get the latest values for all metrics."""
        latest = {}
        for metric_name, metric_list in self.metrics.items():
            if metric_list:
                latest[metric_name] = metric_list[-1]['value']
        return latest
    
    def print_progress(self, episode=None, step=None):
        """Print current training progress."""
        if episode is not None:
            self.current_episode = episode
        if step is not None:
            self.current_step = step
        
        latest = self.get_latest_metrics()
        
        progress_str = f"Episode {self.current_episode}, Step {self.current_step}"
        if 'episode_reward' in latest:
            progress_str += f", Reward: {latest['episode_reward']:.4f}"
        if 'episode_distance' in latest:
            progress_str += f", Distance: {latest['episode_distance']:.4f}"
        if 'env_transitions' in latest:
            progress_str += f", Transitions: {latest['env_transitions']}"
        
        print(progress_str) 