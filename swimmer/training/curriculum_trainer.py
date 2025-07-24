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
from ..models.biological_ncap import BiologicalNCAPSwimmer, BiologicalNCAPActor
from ..models.enhanced_biological_ncap import EnhancedBiologicalNCAPSwimmer
from ..environments.progressive_mixed_env import TonicProgressiveMixedWrapper
from ..utils.training_logger import TrainingLogger
from ..utils.curriculum_visualization import create_curriculum_plots, create_test_video, create_phase_comparison_video, save_training_summary, create_trajectory_analysis
from ..utils.artifact_naming import ArtifactNamer, detect_model_type

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
    
    # Phase duration configuration (easily modifiable)
    PHASE_DURATION_CONFIG = {
        'evaluation_steps': [400, 600, 800, 1200],     # **INCREASED** Steps per episode for each phase (was 200,200,200,400)
        'video_steps': [800, 1000, 1200, 1500],         # **INCREASED** Steps per video for each phase (was 500,500,500,1000)
        'trajectory_multiplier': [1.5, 2.0, 2.5, 3.0] # **INCREASED** Multiplier for trajectory analysis (was 1.0,1.0,1.0,2.0)
    }
    
    def __init__(self, 
                 n_links=5,
                 learning_rate=3e-5,
                 training_steps=1000000,
                 save_steps=50000,
                 log_episodes=50,
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 oscillator_period=60,
                 min_oscillator_strength=0.8,  # **REDUCED** from 1.2 to 0.8 for speed flexibility
                 min_coupling_strength=0.5,  # **REDUCED** from 0.8 to 0.5 for speed flexibility  
                 biological_constraint_frequency=25000,  # **REDUCED** frequency: every 25k steps
                 resume_from_checkpoint=None,  # Path to checkpoint to resume from
                 model_type='enhanced_ncap',  # Model type: biological_ncap, enhanced_ncap
                 algorithm='ppo',  # Algorithm for naming
                 use_locomotion_only_early_training=True):  # **NEW**: Use pure locomotion for first 30% of training
        
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
        self.resume_from_checkpoint = resume_from_checkpoint
        self.model_type = model_type
        self.algorithm = algorithm
        self.use_locomotion_only_early_training = use_locomotion_only_early_training
        
        # Initialize artifact namer for consistent naming across all outputs
        self.artifact_namer = ArtifactNamer(
            model_type=model_type,
            n_links=n_links,
            algorithm=algorithm,
            additional_config={
                'oscillator_period': oscillator_period,
                'training_mode': 'curriculum'
            }
        )
        
        # Training state
        self.current_step = 0
        self.current_episode = 0
        self.phase_rewards = {0: [], 1: [], 2: [], 3: []}
        self.phase_distances = {0: [], 1: [], 2: [], 3: []}
        
        # Initialize components with advanced logging if available
        log_dir = os.path.dirname(self.artifact_namer.training_log_dir())
        experiment_name = self.artifact_namer.base_id
        
        if ADVANCED_LOGGING_AVAILABLE:
            self.logger = AdvancedTrainingLogger(
                log_dir=log_dir, 
                experiment_name=experiment_name
            )
            print("🔬 Using advanced logging with hardware monitoring")
        else:
            self.logger = TrainingLogger(
                log_dir=log_dir,
                experiment_name=experiment_name
            )
            print("📊 Using standard logging")
        
        print(f"🎓 Initialized Curriculum {model_type.upper()} Trainer")
        print(f"   Model: {model_type} with {n_links} links")
        print(f"   Algorithm: {algorithm}")
        print(f"   Device: {device}")
        print(f"   Total training: {training_steps:,} steps")
        print(f"   Artifact ID: {self.artifact_namer.base_id}")
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
        """Create NCAP model optimized for curriculum learning based on model_type."""
        n_joints = self.n_links - 1  # 4 joints for 5-link swimmer
        
        # **NEW**: Determine if we should use locomotion-only mode for early training
        training_progress = self.current_step / self.training_steps
        use_locomotion_only = (self.use_locomotion_only_early_training and 
                             training_progress < 0.3)  # First 30% of training
        
        if self.model_type == 'enhanced_ncap':
            model = EnhancedBiologicalNCAPSwimmer(
                n_joints=n_joints,
                oscillator_period=self.oscillator_period,
                include_environment_adaptation=True,  # Dramatic frequency adaptation
                include_goal_direction=not use_locomotion_only,  # **DISABLED** for early training
                locomotion_only_mode=use_locomotion_only,  # **NEW**: Pure swimming mode
                action_scaling_factor=1.8  # **NEW**: Increased scaling for stronger swimming
            ).to(self.device)
            
            print(f"🚀 Created ENHANCED Biological NCAP model with {sum(p.numel() for p in model.parameters())} parameters")
            print(f"🔬 Relaxation oscillator: Asymmetric (60/40 phase) with 5x frequency adaptation")
            if use_locomotion_only:
                print(f"🏊 TRAINING MODE: Pure locomotion (first 30% of training)")
            else:
                print(f"🎯 Goal-directed navigation: Target-seeking with anti-tail-chasing fixes")
            print(f"📄 Based on C. elegans research: https://elifesciences.org/articles/69905")
            
        elif self.model_type == 'biological_ncap':
            model = BiologicalNCAPSwimmer(
                n_joints=n_joints,
                oscillator_period=self.oscillator_period,
                include_environment_adaptation=True  # Enable biological adaptation
            ).to(self.device)
            
            print(f"🧬 Created standard Biological NCAP model with {sum(p.numel() for p in model.parameters())} parameters")
            print(f"🔬 Biological adaptation: ENABLED (no LSTM - pure neuromodulation)")
            
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}. "
                           f"Supported: 'biological_ncap', 'enhanced_ncap'")
        
        return model
    
    def create_agent(self, model, env):
        """Create simplified agent for curriculum training."""
        
        # Create biological NCAP agent wrapper with environment adaptation
        class BiologicalNCAPAgent:
            def __init__(self, ncap_model, environment):
                self.ncap_model = ncap_model
                self.step_count = 0
                self.use_stable_init = False  # **FIXED**: Allow full action range for speed development
                
                # Initialize RL training components
                # Get learning rate from parent trainer
                learning_rate = 3e-5  # Default learning rate for NCAP training
                self.optimizer = torch.optim.Adam(ncap_model.parameters(), lr=learning_rate)
                self.episode_buffer = {'obs': [], 'actions': [], 'rewards': []}
                self.training_enabled = True
                
            def step(self, obs):
                """Training step - returns action AND updates model."""
                action = self.test_step(obs)
                
                # Store experience for training (if training enabled)
                if self.training_enabled and len(self.episode_buffer['obs']) > 0:
                    # Store previous transition
                    self.episode_buffer['obs'].append(obs)
                    self.episode_buffer['actions'].append(action)
                
                return action
            
            def add_reward(self, reward):
                """Add reward for the last action."""
                if self.training_enabled:
                    self.episode_buffer['rewards'].append(reward)
            
            def end_episode(self):
                """End episode and train on collected experience."""
                if not self.training_enabled or len(self.episode_buffer['rewards']) < 5:
                    self._reset_buffer()
                    return
                
                # Simple policy gradient training
                self._train_on_episode()
                self._reset_buffer()
            
            def _train_on_episode(self):
                """Train model on episode buffer using policy gradient."""
                if len(self.episode_buffer['rewards']) == 0:
                    return
                
                try:
                    device = next(self.ncap_model.parameters()).device
                    
                    # Calculate returns (discounted rewards)
                    returns = []
                    running_return = 0
                    for reward in reversed(self.episode_buffer['rewards']):
                        running_return = reward + 0.99 * running_return
                        returns.insert(0, running_return)
                    
                    if len(returns) == 0:
                        return
                    
                    # Normalize returns
                    returns = torch.FloatTensor(returns).to(device)
                    if returns.std() > 1e-6:
                        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
                    
                    # Convert observations and actions to tensors
                    obs_batch = []
                    action_batch = []
                    
                    for i in range(min(len(self.episode_buffer['obs']), len(self.episode_buffer['actions']))):
                        obs_batch.append(self.episode_buffer['obs'][i])
                        action_batch.append(self.episode_buffer['actions'][i])
                    
                    if len(obs_batch) == 0:
                        return
                    
                    # Train on mini-batches
                    batch_size = min(32, len(obs_batch))
                    for start_idx in range(0, len(obs_batch), batch_size):
                        end_idx = min(start_idx + batch_size, len(obs_batch))
                        
                        batch_obs = obs_batch[start_idx:end_idx]
                        batch_actions = action_batch[start_idx:end_idx]
                        batch_returns = returns[start_idx:end_idx]
                        
                        if len(batch_obs) < 2:
                            continue
                        
                        # Get model predictions
                        predicted_actions = []
                        for obs in batch_obs:
                            action = self._get_model_action(obs)
                            predicted_actions.append(action)
                        
                        if len(predicted_actions) == 0:
                            continue
                        
                        predicted_actions = torch.stack(predicted_actions)
                        actual_actions = torch.FloatTensor(batch_actions).to(device)
                        
                        # Policy gradient loss
                        loss = torch.nn.functional.mse_loss(predicted_actions, actual_actions, reduction='none')
                        policy_loss = (loss.mean(dim=1) * batch_returns[:len(loss)]).mean()
                        
                        # Update model
                        self.optimizer.zero_grad()
                        policy_loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.ncap_model.parameters(), 0.5)
                        self.optimizer.step()
                        
                except Exception as e:
                    print(f"⚠️ Training step failed: {e}")
            
            def _reset_buffer(self):
                """Reset episode buffer."""
                self.episode_buffer = {'obs': [], 'actions': [], 'rewards': []}
            
            def _get_model_action(self, obs):
                """Get action from NCAP model as tensor for training."""
                device = next(self.ncap_model.parameters()).device
                
                if not isinstance(obs, torch.Tensor):
                    obs = torch.tensor(obs, dtype=torch.float32, device=device)
                elif obs.device != device:
                    obs = obs.to(device)
                
                # Extract joint positions and other inputs (same logic as test_step)
                joint_pos = obs[:4] if len(obs) >= 4 else torch.zeros(4, device=device)
                
                # Environment adaptation
                environment_type = None
                if len(obs) >= 9:
                    water_flag = obs[8:9]
                    land_flag = 1.0 - water_flag
                    viscosity_norm = torch.zeros_like(water_flag) + 0.1
                    environment_type = torch.cat([water_flag, land_flag, viscosity_norm])
                
                # Goal direction (for enhanced NCAP)
                target_direction = None
                if len(obs) >= 32 and hasattr(self.ncap_model, 'include_goal_direction') and self.ncap_model.include_goal_direction:
                    target_x, target_y = obs[-4:-3], obs[-3:-2]
                    swimmer_x, swimmer_y = obs[-2:-1], obs[-1:]
                    target_direction = torch.cat([target_x - swimmer_x, target_y - swimmer_y])
                
                # Get action from model
                with torch.no_grad():
                    if hasattr(self.ncap_model, 'include_goal_direction') and self.ncap_model.include_goal_direction:
                        action = self.ncap_model(
                            joint_pos, 
                            environment_type=environment_type,
                            target_direction=target_direction,
                            timesteps=torch.tensor([self.step_count], device=device)
                        )
                    else:
                        action = self.ncap_model(
                            joint_pos, 
                            environment_type=environment_type,
                            timesteps=torch.tensor([self.step_count], device=device)
                        )
                
                return action
            
            def test_step(self, obs):
                """Test step - returns action without training."""
                # Get device from model parameters
                device = next(self.ncap_model.parameters()).device
                
                # Extract joint positions, environment info, and target info from observation
                if isinstance(obs, dict):
                    joint_pos = torch.tensor(obs['joints'], dtype=torch.float32, device=device)
                    
                    # Extract environment information for biological adaptation
                    environment_type = None
                    if 'environment_type' in obs and 'fluid_viscosity' in obs:
                        env_flags = obs['environment_type']  # [water_flag, land_flag]
                        viscosity = obs['fluid_viscosity'][0] if hasattr(obs['fluid_viscosity'], '__len__') else obs['fluid_viscosity']
                        # Normalize viscosity for biological model
                        vis_norm = np.clip((np.log10(viscosity) - np.log10(1e-4)) / (np.log10(1.5) - np.log10(1e-4)), 0.0, 1.0)
                        environment_type = np.array([env_flags[0], env_flags[1], vis_norm], dtype=np.float32)
                    
                    # **NEW**: Extract target information for goal-directed navigation
                    target_direction = None
                    if hasattr(self.ncap_model, 'include_goal_direction') and self.ncap_model.include_goal_direction:
                        if 'target_direction' in obs:
                            target_direction = obs['target_direction']
                        elif 'target_position' in obs:
                            # Use target position as direction (simplified)
                            target_pos = obs['target_position']
                            target_norm = np.linalg.norm(target_pos)
                            if target_norm > 0.1:  # Valid target
                                target_direction = target_pos / target_norm
                else:
                    joint_pos = torch.tensor(obs[:4], dtype=torch.float32, device=device)
                    environment_type = None
                    target_direction = None
                
                # Get NCAP action with biological adaptation and goal-directed navigation
                with torch.no_grad():
                    if hasattr(self.ncap_model, 'include_goal_direction') and self.ncap_model.include_goal_direction:
                        # Enhanced NCAP with goal-directed navigation
                        action = self.ncap_model(
                            joint_pos, 
                            environment_type=environment_type,
                            target_direction=target_direction,
                            timesteps=torch.tensor([self.step_count], device=device)
                        )
                    else:
                        # Standard biological NCAP
                        action = self.ncap_model(
                            joint_pos, 
                            environment_type=environment_type,
                            timesteps=torch.tensor([self.step_count], device=device)
                        )
                    self.step_count += 1
                    
                    # For untrained models, reduce action magnitude to prevent erratic motion
                    if self.use_stable_init:
                        action = torch.clamp(action, -0.3, 0.3)  # Reduced from default range
                
                return action.cpu().numpy()
        
        agent = BiologicalNCAPAgent(model, env)
        
        print(f"🧬 Created biological NCAP agent with environment adaptation for curriculum learning")
        
        return agent, model
    
    def save_checkpoint(self, model, step, eval_results=None):
        """Save training checkpoint with model-specific naming."""
        checkpoint_path = self.artifact_namer.checkpoint_name(
            step=step, 
            base_dir="outputs/curriculum_training/checkpoints"
        )
        
        checkpoint_data = {
            'model_state_dict': model.state_dict(),
            'current_step': self.current_step,
            'current_episode': self.current_episode,
            'phase_rewards': self.phase_rewards,
            'phase_distances': self.phase_distances,
            'training_config': {
                'n_links': self.n_links,
                'learning_rate': self.learning_rate,
                'training_steps': self.training_steps,
                'oscillator_period': self.oscillator_period,
                'min_oscillator_strength': self.min_oscillator_strength,
                'min_coupling_strength': self.min_coupling_strength,
                'biological_constraint_frequency': self.biological_constraint_frequency,
            },
            'eval_results': eval_results,
        }
        
        torch.save(checkpoint_data, checkpoint_path)
        print(f"💾 Checkpoint saved: {checkpoint_path}")
        return checkpoint_path
    
    def load_checkpoint(self, model, checkpoint_path):
        """Load training checkpoint with backward compatibility."""
        print(f"📂 Loading checkpoint: {checkpoint_path}")
        
        checkpoint_data = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        model.load_state_dict(checkpoint_data['model_state_dict'])
        
        # Load training state with backward compatibility
        if 'current_step' in checkpoint_data:
            # New checkpoint format
            self.current_step = checkpoint_data['current_step']
            self.current_episode = checkpoint_data['current_episode']
            self.phase_rewards = checkpoint_data.get('phase_rewards', {0: [], 1: [], 2: [], 3: []})
            self.phase_distances = checkpoint_data.get('phase_distances', {0: [], 1: [], 2: [], 3: []})
        else:
            # Old checkpoint format (legacy compatibility)
            self.current_step = checkpoint_data.get('step', 0)
            self.current_episode = checkpoint_data.get('episode', 0)
            self.phase_rewards = {0: [], 1: [], 2: [], 3: []}  # Reset for old checkpoints
            self.phase_distances = {0: [], 1: [], 2: [], 3: []}
            print("⚠️ Legacy checkpoint format detected - phase history reset")
        
        print(f"✅ Checkpoint loaded successfully!")
        print(f"   Resuming from step: {self.current_step:,}")
        print(f"   Episode: {self.current_episode:,}")
        
        return checkpoint_data.get('eval_results', {})
    
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
            
            # **RELAXED**: Ensure ipsilateral muscle is positive (less restrictive)
            if model.params['muscle_ipsi'].item() < 0.5:  # **REDUCED** from 0.8 to 0.5
                old_val = model.params['muscle_ipsi'].item()
                model.params['muscle_ipsi'].data.fill_(0.5)
                constraints_applied.append(f"ipsi {old_val:.3f} → 0.5")
            
            # **RELAXED**: Ensure contralateral muscle is negative (less restrictive)
            if model.params['muscle_contra'].item() > -0.5:  # **REDUCED** from -0.8 to -0.5
                old_val = model.params['muscle_contra'].item()
                model.params['muscle_contra'].data.fill_(-0.5)
                constraints_applied.append(f"contra {old_val:.3f} → -0.5")
        
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
            
            # Get phase-specific episode duration from configuration
            steps_per_episode = self.PHASE_DURATION_CONFIG['evaluation_steps'][phase]
            phase_names = ["Pure Swimming", "Single Land Zone", "Two Land Zones", "Full Complexity"]
            
            if steps_per_episode != 200:  # Log when using non-standard duration
                print(f"🎯 {phase_names[phase]}: Using {steps_per_episode} steps per episode")
            
            distances = []
            rewards = []
            
            for episode in range(num_episodes):
                # Set environment to specific phase
                env.env.training_progress = temp_progress
                env.env._create_environment()
                
                obs = env.reset()
                episode_reward = 0
                initial_pos = env.env.physics.named.data.xpos['head'][:2].copy()
                
                for _ in range(steps_per_episode):  # Data-driven steps per episode
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
        
        # Create environment and model based on model_type
        env = self.create_environment()
        model = self.create_model()
        agent, tonic_model = self.create_agent(model, env)
        
        # Load checkpoint if resuming
        if self.resume_from_checkpoint:
            self.load_checkpoint(tonic_model, self.resume_from_checkpoint)
        
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
                
                # CRITICAL: Add reward to agent for training (was missing!)
                agent.add_reward(reward)
                
                episode_reward += reward
                episode_steps += 1
                self.current_step += 1
                
                if done or self.current_step >= self.training_steps:
                    break
            
            # CRITICAL: Train on episode experience (was missing!)
            agent.end_episode()
            
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
                
                # Comprehensive evaluation first
                eval_results = self.evaluate_performance(agent, env, num_episodes=10)
                
                # Save comprehensive checkpoint with eval results
                checkpoint_path = self.save_checkpoint(tonic_model, self.current_step, eval_results)
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
                    plot_path = self.artifact_namer.analysis_plot_name(
                        "curriculum_progress", 
                        step=self.current_step,
                        base_dir="outputs/curriculum_training/plots"
                    )
                    create_curriculum_plots(
                        phase_rewards=self.phase_rewards,
                        phase_distances=self.phase_distances,
                        eval_results=eval_results,
                        save_path=plot_path
                    )
                
                # Create trajectory analysis
                current_phase = min(int(self.current_step / (self.training_steps / 4)), 3)
                phase_names = ["Pure Swimming", "Single Land Zone", "Two Land Zones", "Full Complexity"]
                trajectory_path = self.artifact_namer.analysis_plot_name(
                    "trajectory_analysis", 
                    step=self.current_step,
                    phase=f"phase{current_phase}",
                    base_dir="outputs/curriculum_training/plots"
                )
                
                trajectory_stats = create_trajectory_analysis(
                    agent=agent,
                    env=env,
                    save_path=trajectory_path,
                    num_steps=500,
                    phase_name=f"Step {self.current_step} - {phase_names[current_phase]}",
                    trajectory_multiplier=self.PHASE_DURATION_CONFIG['trajectory_multiplier'][current_phase]
                )
                
                tqdm.write(f"📊 Trajectory stats: distance={trajectory_stats['final_distance']:.3f}m, "
                          f"transitions={trajectory_stats['transitions']}")
                
                # Create test video
                video_path = self.artifact_namer.training_video_name(
                    step=self.current_step,
                    phase=f"phase{current_phase}",
                    base_dir="outputs/curriculum_training/videos"
                )
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
        
        # Generate final performance summary from training data
        tqdm.write(f"\n📊 Generating final performance summary from training data...")
        
        # Convert training data to evaluation format for model saving and artifacts
        final_eval = {}
        for phase in range(4):
            if phase in self.phase_rewards and len(self.phase_rewards[phase]) > 0:
                # Use actual training data from this phase
                phase_rewards = self.phase_rewards[phase]
                phase_distances = self.phase_distances[phase]
                
                final_eval[phase] = {
                    'mean_distance': np.mean(phase_distances),
                    'std_distance': np.std(phase_distances) if len(phase_distances) > 1 else 0.0,
                    'mean_reward': np.mean(phase_rewards),
                    'std_reward': np.std(phase_rewards) if len(phase_rewards) > 1 else 0.0
                }
            else:
                # Fallback for phases not trained yet (shouldn't happen in normal training)
                final_eval[phase] = {
                    'mean_distance': 0.0,
                    'std_distance': 0.0,
                    'mean_reward': 0.0,
                    'std_reward': 0.0
                }
        
        total_episodes = sum(len(self.phase_rewards[p]) for p in range(4) if p in self.phase_rewards)
        active_phases = len([p for p in range(4) if p in self.phase_rewards and len(self.phase_rewards[p]) > 0])
        tqdm.write(f"✅ Training performance summary: {total_episodes} episodes across {active_phases} phases")
        
        tqdm.write(f"\n📊 Final Performance Summary:")
        for phase, results in final_eval.items():
            phase_names_final = ["Pure Swimming", "Single Land Zone", "Two Land Zones", "Full Complexity"]
            tqdm.write(f"   {phase_names_final[phase]}: {results['mean_distance']:.3f}m ± {results['std_distance']:.3f}")
        
        # Save final model
        final_path = self.artifact_namer.final_model_name(
            base_dir="outputs/curriculum_training/models"
        )
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
            final_plot_path = self.artifact_namer.analysis_plot_name(
                "curriculum_final", 
                base_dir="outputs/curriculum_training/plots"
            )
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
                
                trajectory_path = self.artifact_namer.analysis_plot_name(
                    "final_trajectory", 
                    phase=f"phase{phase}",
                    base_dir="outputs/curriculum_training/plots"
                )
                stats = create_trajectory_analysis(
                    agent=agent,
                    env=env,
                    save_path=trajectory_path,
                    num_steps=1000,  # Longer analysis for final evaluation
                    phase_name=f"Final - {phase_names[phase]}",
                    trajectory_multiplier=self.PHASE_DURATION_CONFIG['trajectory_multiplier'][phase]
                )
                
                final_trajectory_stats[phase] = stats
                pbar.update(1)
                tqdm.write(f"   ✅ {phase_names[phase]}: {stats['final_distance']:.3f}m, {stats['transitions']} transitions")
            
            # Final test video with phase comparisons
            pbar.set_description("🎬 Creating phase comparison video")
            final_video_path = self.artifact_namer.evaluation_video_name(
                evaluation_type="phase_comparison_final",
                base_dir="outputs/curriculum_training/videos"
            )
            create_phase_comparison_video(
                agent=agent,
                env=env,
                save_path=final_video_path,
                phases_to_test=[0, 1, 2, 3],
                phase_video_steps=self.PHASE_DURATION_CONFIG['video_steps']
            )
            pbar.update(1)
            tqdm.write(f"✅ Phase comparison video: {final_video_path}")
            
            # Training summary
            pbar.set_description("📄 Generating training summary")
            summary_path = self.artifact_namer.experiment_summary_name(
                base_dir="outputs/curriculum_training/summaries"
            )
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
    
    def evaluate_only(self, eval_episodes=20, video_steps=400):
        """Run evaluation and visualization only (no training) from a checkpoint."""
        print(f"\n📊 Starting Curriculum Evaluation (No Training)")
        print(f"   Checkpoint: {self.resume_from_checkpoint}")
        print(f"   Links: {self.n_links}")
        print(f"   Episodes per phase: {eval_episodes}")
        print(f"   Video length: {video_steps} steps")
        
        # Create environment and model based on model_type
        env = self.create_environment()
        model = self.create_model()
        agent, tonic_model = self.create_agent(model, env)
        
        # Load checkpoint
        if self.resume_from_checkpoint:
            checkpoint_results = self.load_checkpoint(tonic_model, self.resume_from_checkpoint)
        else:
            print("⛔ No checkpoint provided for evaluation!")
            return
        
        start_time = time.time()
        
        print(f"\n📊 Generating performance summary from checkpoint training data...")
        
        # Convert training data to evaluation format for visualization artifacts
        final_eval = {}
        for phase in range(4):
            if phase in self.phase_rewards and len(self.phase_rewards[phase]) > 0:
                # Use actual training data from this phase
                phase_rewards = self.phase_rewards[phase]
                phase_distances = self.phase_distances[phase]
                
                final_eval[phase] = {
                    'mean_distance': np.mean(phase_distances),
                    'std_distance': np.std(phase_distances) if len(phase_distances) > 1 else 0.0,
                    'mean_reward': np.mean(phase_rewards),
                    'std_reward': np.std(phase_rewards) if len(phase_rewards) > 1 else 0.0
                }
            else:
                # Fallback for phases not trained yet (shouldn't happen in normal training)
                final_eval[phase] = {
                    'mean_distance': 0.0,
                    'std_distance': 0.0,
                    'mean_reward': 0.0,
                    'std_reward': 0.0
                }
        
        total_episodes = sum(len(self.phase_rewards[p]) for p in range(4) if p in self.phase_rewards)
        active_phases = len([p for p in range(4) if p in self.phase_rewards and len(self.phase_rewards[p]) > 0])
        print(f"✅ Checkpoint performance summary: {total_episodes} episodes across {active_phases} phases")
        

        print(f"\n📊 Performance Summary:")
        phase_names_final = ["Pure Swimming", "Single Land Zone", "Two Land Zones", "Full Complexity"]
        for phase, results in final_eval.items():
            print(f"   {phase_names_final[phase]}: {results['mean_distance']:.3f}m ± {results['std_distance']:.3f}")
        
        print(f"\n🎨 Creating comprehensive visualizations...")
        with tqdm(total=8, desc="📊 Creating Visualizations", unit="task",
                 bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as vis_pbar:
            
            # Training plots
            vis_pbar.set_description("📊 Creating final training plots")
            eval_plot_path = self.artifact_namer.analysis_plot_name(
                "evaluation_final", 
                base_dir="outputs/curriculum_training/plots"
            )
            create_curriculum_plots(
                phase_rewards=self.phase_rewards,
                phase_distances=self.phase_distances,
                eval_results=final_eval,
                save_path=eval_plot_path
            )
            vis_pbar.update(1)
            print(f"✅ Training plots saved to: {eval_plot_path}")
            
            # Trajectory analysis for each phase
            phase_names = ["Pure Swimming", "Single Land Zone", "Two Land Zones", "Full Complexity"]
            final_trajectory_stats = {}
            
            for phase in range(4):
                vis_pbar.set_description(f"📊 Analyzing {phase_names[phase]}")
                
                # Set environment to specific phase
                env.env.set_manual_progress((phase + 0.5) * 0.25)  # Middle of each phase
                
                eval_trajectory_path = self.artifact_namer.analysis_plot_name(
                    "evaluation_trajectory", 
                    phase=f"phase{phase}",
                    base_dir="outputs/curriculum_training/plots"
                )
                trajectory_stats = create_trajectory_analysis(
                    agent=agent,
                    env=env,
                    save_path=eval_trajectory_path,
                    num_steps=video_steps,
                    phase_name=f"Evaluation - {phase_names[phase]}",
                    trajectory_multiplier=self.PHASE_DURATION_CONFIG['trajectory_multiplier'][phase]
                )
                
                final_trajectory_stats[phase] = trajectory_stats
                vis_pbar.update(1)
                print(f"   ✅ {phase_names[phase]}: {trajectory_stats['final_distance']:.3f}m, {trajectory_stats['transitions']} transitions")
            
            # Phase comparison video
            vis_pbar.set_description("🎬 Creating phase comparison video")
            eval_comparison_video_path = self.artifact_namer.evaluation_video_name(
                evaluation_type="phase_comparison",
                base_dir="outputs/curriculum_training/videos"
            )
            create_phase_comparison_video(
                agent=agent,
                env=env,
                save_path=eval_comparison_video_path,
                phases_to_test=[0, 1, 2, 3],
                phase_video_steps=self.PHASE_DURATION_CONFIG['video_steps']
            )
            vis_pbar.update(1)
            print(f"✅ Phase comparison video: {eval_comparison_video_path}")
            
            # Individual test videos for each phase
            for phase in range(4):
                vis_pbar.set_description(f"🎬 Creating {phase_names[phase]} video")
                
                # Set environment to specific phase
                env.env.set_manual_progress((phase + 0.5) * 0.25)
                
                phase_video_path = self.artifact_namer.evaluation_video_name(
                    evaluation_type=f"phase{phase}_{phase_names[phase].lower().replace(' ', '_')}",
                    base_dir="outputs/curriculum_training/videos"
                )
                create_test_video(
                    agent=agent,
                    env=env,
                    save_path=phase_video_path,
                    num_steps=video_steps,
                    episode_name=f"Evaluation - {phase_names[phase]}"
                )
                print(f"   ✅ {phase_names[phase]} video: {phase_video_path}")
            vis_pbar.update(1)
            
            # Training summary
            vis_pbar.set_description("📄 Generating evaluation summary")
            eval_summary_path = self.artifact_namer.experiment_summary_name(
                base_dir="outputs/curriculum_training/summaries"
            ).replace("_experiment_summary.md", "_evaluation_summary.md")
            save_training_summary(
                eval_results=final_eval,
                training_history={
                    'phase_rewards': self.phase_rewards,
                    'phase_distances': self.phase_distances,
                    'trajectory_stats': final_trajectory_stats,
                },
                save_path=eval_summary_path
            )
            vis_pbar.update(1)
            print(f"✅ Evaluation summary: {eval_summary_path}")
        
        total_time = time.time() - start_time
        print(f"\n🏁 Evaluation Complete!")
        print(f"   Total time: {total_time/60:.1f} minutes")
        print(f"   Step: {self.current_step:,}")
        print(f"   Episode: {self.current_episode:,}")
        
        env.close()
        return final_eval 