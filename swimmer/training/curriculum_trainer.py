#!/usr/bin/env python3
"""
Curriculum Trainer for Swimming and Crawling
Manages progressive training from simple swimming to complex mixed environments.
"""

import torch
import torch.nn as nn
import numpy as np
import os
import time
import tonic
import warnings
from tqdm import tqdm

# Suppress the harmless gym Box precision warning
warnings.filterwarnings("ignore", message=".*Box bound precision lowered by casting to.*")
from ..models.simple_ncap import SimpleNCAPSwimmer
from ..environments.progressive_mixed_env import TonicProgressiveMixedWrapper
from ..utils.training_logger import TrainingLogger
from ..utils.curriculum_visualization import create_curriculum_plots, create_test_video, create_phase_comparison_video, save_training_summary, create_trajectory_analysis

try:
    from ..utils.advanced_logger import AdvancedTrainingLogger
    ADVANCED_LOGGING_AVAILABLE = True
except ImportError:
    ADVANCED_LOGGING_AVAILABLE = False
    print("⚠️ Advanced logging not available (missing psutil). Using basic logging.")


class CurriculumNCAPTrainer:
    """
    Curriculum trainer for NCAP swimmer with progressive complexity.
    
    Designed for 1M episode training with curriculum learning:
    - Phase 1 (0-30%): Pure swimming in simple environment
    - Phase 2 (30-60%): Introduction of single land zone
    - Phase 3 (60-80%): Two land zones for complex navigation
    - Phase 4 (80-100%): Full mixed environment complexity
    """
    
    def __init__(self, 
                 n_links=5,
                 learning_rate=3e-5,
                 training_steps=1000000,
                 save_steps=50000,
                 log_episodes=50,
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 oscillator_period=60,
                 min_oscillator_strength=1.2,
                 min_coupling_strength=0.8,
                 biological_constraint_frequency=5000):  # Every 5k steps
        
        self.n_links = n_links
        self.learning_rate = learning_rate
        self.training_steps = training_steps
        self.save_steps = save_steps
        self.log_episodes = log_episodes
        self.device = device
        self.oscillator_period = oscillator_period
        self.min_oscillator_strength = min_oscillator_strength
        self.min_coupling_strength = min_coupling_strength
        self.biological_constraint_frequency = biological_constraint_frequency
        
        # Training state
        self.current_step = 0
        self.current_episode = 0
        self.phase_rewards = {0: [], 1: [], 2: [], 3: []}
        self.phase_distances = {0: [], 1: [], 2: [], 3: []}
        
        # Initialize components with advanced logging if available
        if ADVANCED_LOGGING_AVAILABLE:
            self.logger = AdvancedTrainingLogger(experiment_name=f"curriculum_ncap_{n_links}links")
            print("🔬 Using advanced logging with hardware monitoring")
        else:
            self.logger = TrainingLogger(f"curriculum_ncap_{n_links}links")
            print("📊 Using standard logging")
        
        print(f"🎓 Initialized Curriculum NCAP Trainer")
        print(f"   Device: {device}")
        print(f"   Total training: {training_steps:,} steps")
        print(f"   Phase progression:")
        print(f"     Phase 1 (0-30%): Pure swimming")
        print(f"     Phase 2 (30-60%): Single land zone")
        print(f"     Phase 3 (60-80%): Two land zones")
        print(f"     Phase 4 (80-100%): Full complexity")
        
    def create_environment(self):
        """Create progressive mixed environment."""
        env = TonicProgressiveMixedWrapper(
            n_links=self.n_links,
            time_feature=True,
            desired_speed=0.15
        )
        
        print(f"🌊 Created progressive mixed environment")
        print(f"   Environment: {env.name}")
        print(f"   Observation space: {env.observation_space.shape}")
        print(f"   Action space: {env.action_space.shape}")
        
        return env
    
    def create_model(self):
        """Create NCAP model optimized for curriculum learning."""
        model = SimpleNCAPSwimmer(
            n_joints=self.n_links - 1,  # 4 joints for 5-link swimmer
            oscillator_period=self.oscillator_period
        ).to(self.device)
        
        print(f"🧬 Created NCAP model with {sum(p.numel() for p in model.parameters())} parameters")
        return model
    
    def create_agent(self, model, env):
        """Create simplified agent for curriculum training."""
        
        # For testing purposes, create a minimal agent wrapper
        class SimpleNCAPAgent:
            def __init__(self, ncap_model):
                self.ncap_model = ncap_model
                self.step_count = 0
                self.use_stable_init = True  # Reduce erratic motion for untrained models
                
            def step(self, obs):
                """Training step - returns action."""
                return self.test_step(obs)
            
            def test_step(self, obs):
                """Test step - returns action without training."""
                # Get device from model parameters
                device = next(self.ncap_model.parameters()).device
                
                # Extract joint positions from observation
                if isinstance(obs, dict):
                    joint_pos = torch.tensor(obs['joints'], dtype=torch.float32, device=device)
                else:
                    joint_pos = torch.tensor(obs[:4], dtype=torch.float32, device=device)
                
                # Get NCAP action
                with torch.no_grad():
                    action = self.ncap_model(
                        joint_pos, 
                        timesteps=torch.tensor([self.step_count], device=device)
                    )
                    self.step_count += 1
                    
                    # For untrained models, reduce action magnitude to prevent erratic motion
                    if self.use_stable_init:
                        action = torch.clamp(action, -0.3, 0.3)  # Reduced from default range
                
                return action.cpu().numpy()
        
        agent = SimpleNCAPAgent(model)
        
        print(f"🤖 Created simplified NCAP agent for curriculum learning")
        
        return agent, model
    
    def apply_biological_constraints(self, model):
        """Apply biological constraints to maintain realism."""
        constraints_applied = []
        
        with torch.no_grad():
            # Ensure oscillator strength minimum
            if model.params['bneuron_osc'].item() < self.min_oscillator_strength:
                old_val = model.params['bneuron_osc'].item()
                model.params['bneuron_osc'].data.fill_(self.min_oscillator_strength)
                constraints_applied.append(f"oscillator {old_val:.3f} → {self.min_oscillator_strength}")
            
            # Ensure coupling strength minimum
            if model.params['bneuron_prop'].item() < self.min_coupling_strength:
                old_val = model.params['bneuron_prop'].item()
                model.params['bneuron_prop'].data.fill_(self.min_coupling_strength)
                constraints_applied.append(f"coupling {old_val:.3f} → {self.min_coupling_strength}")
            
            # Ensure ipsilateral muscle is positive
            if model.params['muscle_ipsi'].item() < 0.8:
                old_val = model.params['muscle_ipsi'].item()
                model.params['muscle_ipsi'].data.fill_(0.8)
                constraints_applied.append(f"ipsi {old_val:.3f} → 0.8")
            
            # Ensure contralateral muscle is negative
            if model.params['muscle_contra'].item() > -0.8:
                old_val = model.params['muscle_contra'].item()
                model.params['muscle_contra'].data.fill_(-0.8)
                constraints_applied.append(f"contra {old_val:.3f} → -0.8")
        
        if constraints_applied:
            print(f"🧬 Applied biological constraints: {', '.join(constraints_applied)}")
        
        return len(constraints_applied) > 0
    
    def get_current_phase(self, progress):
        """Get current training phase based on progress."""
        if progress < 0.3:
            return 0  # Pure swimming
        elif progress < 0.6:
            return 1  # Single land zone
        elif progress < 0.8:
            return 2  # Two land zones
        else:
            return 3  # Full complexity
    
    def evaluate_performance(self, agent, env, num_episodes=5, progress_bar=None):
        """Evaluate current performance across different phases."""
        evaluation_results = {}
        
        for phase in range(4):
            # Create temporary environment for this phase
            temp_progress = (phase + 0.5) * 0.25  # Middle of each phase
            
            distances = []
            rewards = []
            
            for episode in range(num_episodes):
                # Set environment to specific phase
                env.env.training_progress = temp_progress
                env.env._create_environment()
                
                obs = env.reset()
                episode_reward = 0
                initial_pos = env.env.physics.named.data.xpos['head'][:2].copy()
                
                for _ in range(200):  # 200 steps per episode
                    action = agent.test_step(obs)
                    obs, reward, done, _ = env.step(action)
                    episode_reward += reward
                    
                    if done:
                        break
                
                final_pos = env.env.physics.named.data.xpos['head'][:2].copy()
                distance = np.linalg.norm(final_pos - initial_pos)
                
                distances.append(distance)
                rewards.append(episode_reward)
                
                # Update progress bar if provided
                if progress_bar is not None:
                    phase_names = ["Pure Swimming", "Single Land Zone", "Two Land Zones", "Full Complexity"]
                    progress_bar.set_description(f"🔬 Evaluating {phase_names[phase]} ({episode+1}/{num_episodes})")
                    progress_bar.update(1)
            
            evaluation_results[phase] = {
                'mean_distance': np.mean(distances),
                'mean_reward': np.mean(rewards),
                'std_distance': np.std(distances),
                'std_reward': np.std(rewards)
            }
        
        return evaluation_results
    
    def train(self):
        """Run curriculum training for 1M episodes."""
        print(f"\n🎓 Starting Curriculum NCAP Training...")
        print(f"   Target: {self.training_steps:,} steps")
        print(f"   Biological constraints every {self.biological_constraint_frequency:,} steps")
        
        # Create environment and model
        env = self.create_environment()
        model = self.create_model()
        agent, tonic_model = self.create_agent(model, env)
        
        # Training loop with advanced monitoring
        start_time = time.time()
        self.logger.start_time = start_time
        last_phase = -1
        
        # Start hardware monitoring if available
        if ADVANCED_LOGGING_AVAILABLE:
            self.logger.start_monitoring()
        
        # Initialize progress bars
        main_pbar = tqdm(
            total=self.training_steps,
            desc="🎓 Curriculum Training",
            unit="steps",
            unit_scale=True,
            position=0,
            leave=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )
        
        # Phase progress tracking
        phase_names = ["🏊 Pure Swimming", "🏝️ Single Land Zone", "🏝️🏝️ Two Land Zones", "🌍 Full Complexity"]
        
        while self.current_step < self.training_steps:
            # Get current training progress
            progress = self.current_step / self.training_steps
            current_phase = self.get_current_phase(progress)
            
            # Check for phase transitions
            if current_phase != last_phase:
                # Update progress bar description with new phase
                main_pbar.set_description(f"🎓 Curriculum Training - {phase_names[current_phase]}")
                
                tqdm.write(f"\n🎓 PHASE TRANSITION: {last_phase} → {current_phase}")
                tqdm.write(f"   Progress: {progress:.2%}")
                tqdm.write(f"   Step: {self.current_step:,}/{self.training_steps:,}")
                
                # Evaluate performance at phase transition
                if last_phase >= 0:  # Skip initial evaluation
                    eval_results = self.evaluate_performance(agent, env)
                    tqdm.write(f"   Phase {last_phase} final performance:")
                    for phase, results in eval_results.items():
                        if phase <= last_phase:
                            tqdm.write(f"     Phase {phase}: {results['mean_distance']:.3f}m ± {results['std_distance']:.3f}")
                
                last_phase = current_phase
            
            # Apply biological constraints periodically
            if self.current_step % self.biological_constraint_frequency == 0:
                self.apply_biological_constraints(model)
            
            # Training step
            obs = env.reset()
            episode_reward = 0
            episode_steps = 0
            initial_pos = env.env.physics.named.data.xpos['head'][:2].copy()
            
            # Run episode
            episode_start_step = self.current_step
            for _ in range(1000):  # Max episode length
                action = agent.step(obs)
                obs, reward, done, _ = env.step(action)
                episode_reward += reward
                episode_steps += 1
                self.current_step += 1
                
                if done or self.current_step >= self.training_steps:
                    break
            
            # Update progress bar for steps taken this episode
            steps_this_episode = self.current_step - episode_start_step
            main_pbar.update(steps_this_episode)
            
            # Calculate episode distance
            final_pos = env.env.physics.named.data.xpos['head'][:2].copy()
            episode_distance = np.linalg.norm(final_pos - initial_pos)
            
            # Log episode results
            self.current_episode += 1
            self.phase_rewards[current_phase].append(episode_reward)
            self.phase_distances[current_phase].append(episode_distance)
            
            # Periodic logging with ETA
            if self.current_episode % self.log_episodes == 0:
                elapsed_time = time.time() - start_time
                steps_per_sec = self.current_step / elapsed_time if elapsed_time > 0 else 0
                
                recent_rewards = self.phase_rewards[current_phase][-10:] if self.phase_rewards[current_phase] else [0]
                recent_distances = self.phase_distances[current_phase][-10:] if self.phase_distances[current_phase] else [0]
                
                # Update progress bar postfix with current stats
                recent_reward = np.mean(recent_rewards)
                recent_distance = np.mean(recent_distances)
                
                main_pbar.set_postfix({
                    'Phase': current_phase,
                    'Episode': f"{self.current_episode:,}",
                    'Reward': f"{recent_reward:.1f}",
                    'Distance': f"{recent_distance:.3f}m",
                    'Steps/s': f"{steps_per_sec:.1f}"
                })
                
                # Calculate ETA if advanced logging is available
                eta_str = ""
                if ADVANCED_LOGGING_AVAILABLE:
                    eta = self.logger.calculate_eta(self.current_step, self.training_steps)
                    eta_str = f" | ETA: {eta}"
                
                # Detailed logging (less frequent to avoid clutter)
                if self.current_episode % (self.log_episodes * 4) == 0:  # Every 200 episodes instead of 50
                    tqdm.write(f"[{self.current_step:7d}/{self.training_steps:7d}] "
                              f"Phase {current_phase} | "
                              f"Episode {self.current_episode:6d} | "
                              f"Reward: {recent_reward:6.2f} | "
                              f"Distance: {recent_distance:6.3f}m | "
                              f"Steps/s: {steps_per_sec:.1f}{eta_str}")
                
                # Log to file
                self.logger.log_training_step({
                    'step': self.current_step,
                    'episode': self.current_episode,
                    'phase': current_phase,
                    'progress': progress,
                    'reward': episode_reward,
                    'distance': episode_distance,
                    'mean_reward_10': np.mean(recent_rewards),
                    'mean_distance_10': np.mean(recent_distances),
                })
            
            # Periodic saves and evaluation
            if self.current_step % self.save_steps == 0:
                tqdm.write(f"\n💾 Checkpoint at step {self.current_step:,}")
                
                # Save model
                checkpoint_path = f"outputs/curriculum_checkpoints/step_{self.current_step}.pt"
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'step': self.current_step,
                    'episode': self.current_episode,
                    'phase': current_phase,
                    'progress': progress,
                }, checkpoint_path)
                
                # Comprehensive evaluation
                eval_results = self.evaluate_performance(agent, env, num_episodes=10)
                tqdm.write(f"📊 Performance across all phases:")
                for phase, results in eval_results.items():
                    tqdm.write(f"   Phase {phase}: {results['mean_distance']:.3f}m ± {results['std_distance']:.3f} "
                              f"(reward: {results['mean_reward']:.2f})")
                
                # Advanced checkpoint logging
                if ADVANCED_LOGGING_AVAILABLE:
                    checkpoint_data = self.logger.log_checkpoint(
                        step=self.current_step,
                        model=tonic_model,
                        performance_metrics=eval_results
                    )
                    
                    # Show training dashboard
                    dashboard = self.logger.create_training_dashboard()
                    tqdm.write(dashboard)
                
                # Create visualizations
                if self.current_step >= 50000:  # After some training
                    plot_path = f"outputs/curriculum_training/plots/curriculum_plots_step_{self.current_step}.png"
                    create_curriculum_plots(
                        phase_rewards=self.phase_rewards,
                        phase_distances=self.phase_distances,
                        eval_results=eval_results,
                        save_path=plot_path
                    )
                
                # Create trajectory analysis
                trajectory_path = f"outputs/curriculum_training/plots/trajectory_analysis_step_{self.current_step}.png"
                current_phase = min(int(self.current_step / (self.training_steps / 4)), 3)
                phase_names = ["Pure Swimming", "Single Land Zone", "Two Land Zones", "Full Complexity"]
                
                trajectory_stats = create_trajectory_analysis(
                    agent=agent,
                    env=env,
                    save_path=trajectory_path,
                    num_steps=500,
                    phase_name=f"Step {self.current_step} - {phase_names[current_phase]}"
                )
                
                tqdm.write(f"📊 Trajectory stats: distance={trajectory_stats['final_distance']:.3f}m, "
                          f"transitions={trajectory_stats['transitions']}")
                
                # Create test video
                video_path = f"outputs/curriculum_training/videos/curriculum_video_step_{self.current_step}.mp4"
                create_test_video(
                    agent=agent,
                    env=env,
                    save_path=video_path,
                    num_steps=300,
                    episode_name=f"Curriculum Step {self.current_step}"
                )
        
        # Close progress bar
        main_pbar.close()
        
        # Final evaluation and save
        tqdm.write(f"\n🏁 Training Complete!")
        total_time_hours = (time.time() - start_time) / 3600
        tqdm.write(f"   Total time: {total_time_hours:.2f} hours")
        
        # Stop hardware monitoring with indicator
        if ADVANCED_LOGGING_AVAILABLE:
            self.logger.stop_monitoring()  # Advanced logger handles its own progress messages
        else:
            tqdm.write(f"🖥️ Hardware monitoring stopped")
        
        # Final evaluation with progress indicator
        tqdm.write(f"\n🔬 Running final evaluation across all phases...")
        with tqdm(total=80, desc="🔬 Final Evaluation", unit="episode",
                 bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as eval_pbar:
            final_eval = self.evaluate_performance(agent, env, num_episodes=20, progress_bar=eval_pbar)
        
        tqdm.write(f"\n📊 Final Performance Summary:")
        for phase, results in final_eval.items():
            phase_names_final = ["Pure Swimming", "Single Land Zone", "Two Land Zones", "Full Complexity"]
            tqdm.write(f"   {phase_names_final[phase]}: {results['mean_distance']:.3f}m ± {results['std_distance']:.3f}")
        
        # Save final model
        models_dir = f"outputs/curriculum_training/models"
        os.makedirs(models_dir, exist_ok=True)
        final_path = f"{models_dir}/curriculum_final_model_{self.n_links}links.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'final_evaluation': final_eval,
            'training_history': {
                'phase_rewards': self.phase_rewards,
                'phase_distances': self.phase_distances,
            }
        }, final_path)
        
        tqdm.write(f"💾 Final model saved to: {final_path}")
        
        # Create final visualizations
        tqdm.write(f"\n🎨 Creating final training visualizations...")
        
        # Create progress bar for final visualizations
        final_tasks = [
            "Creating final training plots",
            "Trajectory analysis: Pure Swimming", 
            "Trajectory analysis: Single Land Zone",
            "Trajectory analysis: Two Land Zones", 
            "Trajectory analysis: Full Complexity",
            "Creating phase comparison video",
            "Generating training summary",
            "Creating comprehensive report"
        ]
        
        with tqdm(total=len(final_tasks), desc="🎬 Final Analysis", unit="task", 
                 bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
            
            # Final training plots
            pbar.set_description("📊 Creating training plots")
            final_plot_path = f"outputs/curriculum_training/plots/curriculum_final_plots.png"
            create_curriculum_plots(
                phase_rewards=self.phase_rewards,
                phase_distances=self.phase_distances,
                eval_results=final_eval,
                save_path=final_plot_path
            )
            pbar.update(1)
            tqdm.write(f"✅ Training plots saved to: {final_plot_path}")
            
            # Final trajectory analysis for each phase
            phase_names = ["Pure Swimming", "Single Land Zone", "Two Land Zones", "Full Complexity"]
            final_trajectory_stats = {}
            
            for phase in range(4):
                pbar.set_description(f"📊 Analyzing {phase_names[phase]}")
                
                # Set environment to specific phase using manual override
                temp_progress = (phase + 0.5) * 0.25  # Middle of each phase
                env.env.set_manual_progress(temp_progress)
                
                trajectory_path = f"outputs/curriculum_training/plots/final_trajectory_phase_{phase}.png"
                stats = create_trajectory_analysis(
                    agent=agent,
                    env=env,
                    save_path=trajectory_path,
                    num_steps=1000,  # Longer analysis for final evaluation
                    phase_name=f"Final - {phase_names[phase]}"
                )
                
                final_trajectory_stats[phase] = stats
                pbar.update(1)
                tqdm.write(f"   ✅ {phase_names[phase]}: {stats['final_distance']:.3f}m, {stats['transitions']} transitions")
            
            # Final test video with phase comparisons
            pbar.set_description("🎬 Creating phase comparison video")
            final_video_path = f"outputs/curriculum_training/videos/curriculum_final_video.mp4"
            create_phase_comparison_video(
                agent=agent,
                env=env,
                save_path=final_video_path,
                phases_to_test=[0, 1, 2, 3]
            )
            pbar.update(1)
            tqdm.write(f"✅ Phase comparison video: {final_video_path}")
            
            # Training summary
            pbar.set_description("📄 Generating training summary")
            summary_path = f"outputs/curriculum_training/summaries/curriculum_training_summary.md"
            save_training_summary(
                eval_results=final_eval,
                training_history={
                    'phase_rewards': self.phase_rewards,
                    'phase_distances': self.phase_distances,
                    'trajectory_stats': final_trajectory_stats,
                },
                save_path=summary_path
            )
            pbar.update(1)
            tqdm.write(f"✅ Training summary: {summary_path}")
            
            # Generate comprehensive report with advanced metrics
            if ADVANCED_LOGGING_AVAILABLE:
                pbar.set_description("📊 Creating comprehensive report")
                comprehensive_report = self.logger.save_comprehensive_report()
                pbar.update(1)
                tqdm.write(f"✅ Advanced training analysis complete")
            else:
                pbar.update(1)  # Skip if not available
        
        env.close()
        return model, final_eval 