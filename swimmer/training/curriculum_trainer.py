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

# Force basic logging to avoid background monitoring overhead
ADVANCED_LOGGING_AVAILABLE = False


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
                 num_workers=8,    # NEW: Number of parallel environments
                 use_multi_gpu=True, # NEW: Use all available GPUs
                 use_locomotion_only_early_training=True,
                 expose_environment_observation=True,
                 expose_viscosity_observation=True,
                 anisotropic_drag_mode='off',
                 anisotropic_drag_ratio=10.0,
                 anisotropic_drag_gain=0.02,
                 anisotropic_drag_land_only=True):  # **NEW**: Use pure locomotion for first 30% of training
        
        self.n_links = n_links
        self.learning_rate = learning_rate
        self.training_steps = training_steps
        self.save_steps = save_steps
        self.log_episodes = log_episodes
        self.device = device
        self.num_workers = num_workers
        self.use_multi_gpu = use_multi_gpu
        self.oscillator_period = oscillator_period
        self.min_oscillator_strength = min_oscillator_strength
        self.min_coupling_strength = min_coupling_strength
        self.biological_constraint_frequency = biological_constraint_frequency
        self.resume_from_checkpoint = resume_from_checkpoint
        self.model_type = model_type
        self.algorithm = algorithm
        self.use_locomotion_only_early_training = use_locomotion_only_early_training
        self.expose_environment_observation = expose_environment_observation
        self.expose_viscosity_observation = expose_viscosity_observation
        self.anisotropic_drag_mode = anisotropic_drag_mode
        self.anisotropic_drag_ratio = anisotropic_drag_ratio
        self.anisotropic_drag_gain = anisotropic_drag_gain
        self.anisotropic_drag_land_only = anisotropic_drag_land_only
        
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
            desired_speed=0.15,
            expose_environment_observation=self.expose_environment_observation,
            expose_viscosity_observation=self.expose_viscosity_observation,
            anisotropic_drag_mode=self.anisotropic_drag_mode,
            anisotropic_drag_ratio=self.anisotropic_drag_ratio,
            anisotropic_drag_gain=self.anisotropic_drag_gain,
            anisotropic_drag_land_only=self.anisotropic_drag_land_only
        )
        return env
    
    def create_vectorized_environment(self):
        """Create multiple environments running in parallel."""
        from ..environments.vectorized_env import SubprocVecEnv
        print(f"🌊 Creating {self.num_workers} parallel environments...")
        env_fns = [lambda: self.create_environment() for _ in range(self.num_workers)]
        return SubprocVecEnv(env_fns)
    
    def create_model(self):
        """Create NCAP model optimized for curriculum learning based on model_type."""
        n_joints = self.n_links - 1  # 4 joints for 5-link swimmer
        
        # **NEW**: Determine if we should use locomotion-only mode for early training
        # Handle evaluation mode where training_steps=0
        if self.training_steps > 0:
            training_progress = self.current_step / self.training_steps
        else:
            # Evaluation mode - use full progress (1.0) to enable all features
            training_progress = 1.0
            
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
        
        # Enable Multi-GPU if requested and available
        if self.use_multi_gpu and torch.cuda.device_count() > 1:
            print(f"⚡ Using {torch.cuda.device_count()} GPUs with DataParallel")
            model = torch.nn.DataParallel(model)
        
        return model
    
    def create_agent(self, model, env):
        """Create simplified agent for curriculum training."""
        trainer_lr = self.learning_rate
        
        # Create biological NCAP agent wrapper with environment adaptation
        class BiologicalNCAPAgent:
            def __init__(self, ncap_model, environment, n_links):
                self.ncap_model = ncap_model
                self.n_links = n_links
                self.n_joints = n_links - 1
                self.step_count = 0  # Global step count for evaluation
                self.use_stable_init = False
                
                # Initialize RL training components
                self.optimizer = torch.optim.Adam(ncap_model.parameters(), lr=trainer_lr)
                self.num_workers = len(environment.remotes) if hasattr(environment, 'remotes') else 1
                self.episode_buffers = [{'obs': [], 'actions': [], 'rewards': [], 'timesteps': []} for _ in range(self.num_workers)]
                self.training_enabled = True
                self.step_counts = np.zeros(self.num_workers, dtype=int)
                
                # Precompute offsets for observation processing
                self.body_vel_size = self.n_links * 3 + 3
                self.env_features_start = self.n_joints + self.body_vel_size
                self.goal_features_start = self.env_features_start + 5
                
            def step(self, obs):
                """Batch training step - returns actions for all workers."""
                # obs shape: (num_workers, obs_dim)
                actions = self.test_step(obs)
                
                # Store experience for each worker
                if self.training_enabled:
                    for i in range(self.num_workers):
                        self.episode_buffers[i]['obs'].append(obs[i])
                        # Store detached action snapshots so replay does not retain autograd graphs.
                        self.episode_buffers[i]['actions'].append(np.asarray(actions[i], dtype=np.float32).copy())
                        self.episode_buffers[i]['timesteps'].append(self.step_counts[i])
                
                # step_counts increment moved inside test_step for parallelism consistency
                return actions
            
            def add_rewards(self, rewards):
                """Add rewards for all workers."""
                if self.training_enabled:
                    for i, reward in enumerate(rewards):
                        self.episode_buffers[i]['rewards'].append(reward)
            
            def end_episodes(self, indices=None):
                """End episodes for specified workers and train."""
                if not self.training_enabled:
                    return
                    
                if indices is None:
                    indices = range(self.num_workers)
                
                for i in indices:
                    if len(self.episode_buffers[i]['rewards']) >= 5:
                        self._train_on_episode(i)
                    self._reset_buffer(i)
            
            def _train_on_episode(self, worker_idx):
                """Train model on specific worker's episode buffer."""
                buffer = self.episode_buffers[worker_idx]
                if len(buffer['rewards']) == 0:
                    return
                
                try:
                    # Access the actual model if wrapped in DataParallel
                    actual_model = self.ncap_model.module if isinstance(self.ncap_model, torch.nn.DataParallel) else self.ncap_model
                    device = next(actual_model.parameters()).device
                    
                    # Calculate returns
                    returns = []
                    running_return = 0
                    for reward in reversed(buffer['rewards']):
                        running_return = reward + 0.99 * running_return
                        returns.insert(0, running_return)
                    
                    if len(returns) == 0:
                        return
                    
                    # Normalize returns
                    returns = torch.FloatTensor(returns).to(device)
                    if returns.std() > 1e-6:
                        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
                    
                    # Convert observations, actions, and timesteps to tensors
                    obs_batch = []
                    action_batch = []
                    timestep_batch = []
                    
                    for i in range(min(len(buffer['obs']), len(buffer['actions']), len(buffer['timesteps']))):
                        obs_batch.append(buffer['obs'][i])
                        action_batch.append(buffer['actions'][i])
                        timestep_batch.append(buffer['timesteps'][i])
                    
                    if len(obs_batch) == 0:
                        return
                    
                    # Train on mini-batches
                    batch_size = min(64, len(obs_batch)) # Increased batch size for multi-GPU
                    for start_idx in range(0, len(obs_batch), batch_size):
                        end_idx = min(start_idx + batch_size, len(obs_batch))
                        
                        batch_obs = np.array(obs_batch[start_idx:end_idx])
                        batch_actions = torch.FloatTensor(np.array(action_batch[start_idx:end_idx])).to(device)
                        batch_timesteps = torch.FloatTensor(np.array(timestep_batch[start_idx:end_idx])).to(device)
                        batch_returns = returns[start_idx:end_idx]
                        
                        # Batched forward pass (Uses DataParallel if wrapped)
                        # Extract components for whole batch
                        joint_pos = torch.FloatTensor(batch_obs[:, :self.n_joints]).to(device)
                        
                        environment_type = None
                        if batch_obs.shape[1] >= self.env_features_start + 3:
                            water_flag = torch.FloatTensor(batch_obs[:, self.env_features_start + 1:self.env_features_start + 2]).to(device)
                            land_flag = torch.FloatTensor(batch_obs[:, self.env_features_start + 2:self.env_features_start + 3]).to(device)
                            vis_norm = torch.FloatTensor(batch_obs[:, self.env_features_start:self.env_features_start + 1]).to(device)
                            environment_type = torch.cat([water_flag, land_flag, vis_norm], dim=1)
                        
                        target_direction = None
                        if batch_obs.shape[1] >= self.goal_features_start + 3:
                            target_direction = torch.FloatTensor(batch_obs[:, self.goal_features_start + 1:self.goal_features_start + 3]).to(device)
                        
                        # Get model predictions for the whole mini-batch at once
                        if hasattr(actual_model, 'include_goal_direction') and actual_model.include_goal_direction:
                            predicted_actions = self.ncap_model(
                                joint_pos, 
                                environment_type=environment_type,
                                target_direction=target_direction,
                                timesteps=batch_timesteps
                            )
                        else:
                            predicted_actions = self.ncap_model(
                                joint_pos, 
                                environment_type=environment_type,
                                timesteps=batch_timesteps
                            )
                        
                        # Policy gradient loss
                        loss = torch.nn.functional.mse_loss(predicted_actions, batch_actions, reduction='none')
                        policy_loss = (loss.mean(dim=1) * batch_returns[:len(loss)]).mean()
                        
                        # Update model
                        self.optimizer.zero_grad()
                        policy_loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.ncap_model.parameters(), 0.5)
                        has_bad_grad = False
                        for param in self.ncap_model.parameters():
                            if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                                has_bad_grad = True
                                break
                        if has_bad_grad:
                            self.optimizer.zero_grad()
                            continue
                        self.optimizer.step()
                        with torch.no_grad():
                            for param in self.ncap_model.parameters():
                                if torch.isnan(param).any() or torch.isinf(param).any():
                                    param.data = torch.nan_to_num(param.data, nan=0.0, posinf=1.0, neginf=-1.0)
                        
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"⚠️ Training step failed: {e}")
            
            def _reset_buffer(self, worker_idx):
                """Reset specific worker's episode buffer."""
                self.episode_buffers[worker_idx] = {'obs': [], 'actions': [], 'rewards': [], 'timesteps': []}
                self.step_counts[worker_idx] = 0
            
            def _get_model_action(self, obs):
                """Get action from NCAP model for a single observation (e.g. during evaluation)."""
                # CRITICAL: Always use the underlying module for single samples to avoid DataParallel scattering errors
                actual_model = self.ncap_model.module if isinstance(self.ncap_model, torch.nn.DataParallel) else self.ncap_model
                device = next(actual_model.parameters()).device
                
                if not isinstance(obs, torch.Tensor):
                    obs = torch.tensor(obs, dtype=torch.float32, device=device)
                elif obs.device != device:
                    obs = obs.to(device)
                
                # Single observation handling
                if obs.dim() == 0:
                    return torch.zeros(self.n_joints, device=device)
                
                # Use dynamic offsets
                joint_pos = obs[:self.n_joints]
                
                # Environment adaptation [water, land, viscosity_norm]
                environment_type = None
                if len(obs) >= self.env_features_start + 3:
                    water_flag = obs[self.env_features_start + 1]
                    land_flag = obs[self.env_features_start + 2]
                    vis_norm = obs[self.env_features_start]
                    environment_type = torch.tensor([water_flag, land_flag, vis_norm], dtype=torch.float32, device=device)
                
                # Goal direction
                target_direction = None
                if len(obs) >= self.goal_features_start + 3 and hasattr(actual_model, 'include_goal_direction') and actual_model.include_goal_direction:
                    target_direction = obs[self.goal_features_start + 1:self.goal_features_start + 3]
                
                # Get action from the ALREADY UNWRAPPED model
                with torch.no_grad():
                    # We MUST increment evaluation step count separately
                    t = torch.tensor([self.step_count], dtype=torch.float32, device=device)
                    if hasattr(actual_model, 'include_goal_direction') and actual_model.include_goal_direction:
                        action = actual_model(
                            joint_pos, 
                            environment_type=environment_type,
                            target_direction=target_direction,
                            timesteps=t
                        )
                    else:
                        action = actual_model(
                            joint_pos, 
                            environment_type=environment_type,
                            timesteps=t
                        )
                    self.step_count += 1 # Increment for next call
                
                return action
            
            def test_step(self, obs):
                """Test step - returns action without training."""
                # Get device from model parameters
                device = next(self.ncap_model.parameters()).device
                
                # NCAP processing
                with torch.no_grad():
                    # Handle batch observations from SubprocVecEnv
                    if obs.ndim == 2:
                        batch_size = obs.shape[0]
                        joint_pos = torch.tensor(obs[:, :self.n_joints], dtype=torch.float32, device=device)
                        
                        environment_type = None
                        target_direction = None
                        
                        # Use precomputed offsets for batch extraction
                        if obs.shape[1] >= self.env_features_start + 3:
                            # viscosity = obs[:, start], water = obs[:, start+1], land = obs[:, start+2]
                            vis_norm = torch.tensor(obs[:, self.env_features_start:self.env_features_start+1], dtype=torch.float32, device=device)
                            water_flag = torch.tensor(obs[:, self.env_features_start+1:self.env_features_start+2], dtype=torch.float32, device=device)
                            land_flag = torch.tensor(obs[:, self.env_features_start+2:self.env_features_start+3], dtype=torch.float32, device=device)
                            environment_type = torch.cat([water_flag, land_flag, vis_norm], dim=1)
                        
                        if obs.shape[1] >= self.goal_features_start + 3:
                            # target_direction is at goal_features_start + 1, + 2
                            target_direction = torch.tensor(obs[:, self.goal_features_start+1:self.goal_features_start+3], dtype=torch.float32, device=device)
                        
                        timesteps = torch.tensor(self.step_counts, dtype=torch.float32, device=device)
                        
                        # Forward pass - self.ncap_model might be DataParallel
                        if hasattr(self.ncap_model, 'module') and hasattr(self.ncap_model.module, 'include_goal_direction') and self.ncap_model.module.include_goal_direction:
                            action = self.ncap_model(
                                joint_pos, 
                                environment_type=environment_type,
                                target_direction=target_direction,
                                timesteps=timesteps
                            )
                        elif not hasattr(self.ncap_model, 'module') and hasattr(self.ncap_model, 'include_goal_direction') and self.ncap_model.include_goal_direction:
                            action = self.ncap_model(
                                joint_pos, 
                                environment_type=environment_type,
                                target_direction=target_direction,
                                timesteps=timesteps
                            )
                        else:
                            action = self.ncap_model(
                                joint_pos, 
                                environment_type=environment_type,
                                timesteps=timesteps
                            )
                        self.step_counts += 1
                        return action.cpu().numpy()
                    else:
                        # Single observation (fallback to _get_model_action which handles unwrap)
                        return self._get_model_action(obs).cpu().numpy()
        
        agent = BiologicalNCAPAgent(model, env, self.n_links)
        
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
                'expose_environment_observation': self.expose_environment_observation,
                'expose_viscosity_observation': self.expose_viscosity_observation,
                'anisotropic_drag_mode': self.anisotropic_drag_mode,
                'anisotropic_drag_ratio': self.anisotropic_drag_ratio,
                'anisotropic_drag_gain': self.anisotropic_drag_gain,
                'anisotropic_drag_land_only': self.anisotropic_drag_land_only,
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
        
        # Load model state with compatibility for missing parameters
        try:
            model.load_state_dict(checkpoint_data['model_state_dict'])
        except RuntimeError as e:
            if "Missing key(s)" in str(e):
                print(f"⚠️ Checkpoint compatibility issue: {e}")
                print("🔧 Attempting to load compatible parameters only...")
                
                # Load only the parameters that exist in both model and checkpoint
                model_state = model.state_dict()
                checkpoint_state = checkpoint_data['model_state_dict']
                
                # Filter out missing parameters and load the rest
                compatible_state = {}
                missing_params = []
                extra_params = []
                
                for key, value in checkpoint_state.items():
                    if key in model_state:
                        compatible_state[key] = value
                    else:
                        extra_params.append(key)
                
                for key in model_state.keys():
                    if key not in checkpoint_state:
                        missing_params.append(key)
                
                # Load compatible parameters
                model.load_state_dict(compatible_state, strict=False)
                
                print(f"✅ Loaded {len(compatible_state)} compatible parameters")
                if missing_params:
                    print(f"⚠️ Missing parameters (will use defaults): {missing_params}")
                if extra_params:
                    print(f"ℹ️ Extra parameters in checkpoint (ignored): {extra_params}")
            else:
                raise
        
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
    
    def evaluate_performance(self, agent, env_ignored, num_episodes=5, progress_bar=None):
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
                from ..environments.progressive_mixed_env import TonicProgressiveMixedWrapper
                eval_env = TonicProgressiveMixedWrapper(n_links=self.n_links)
                eval_env.env.training_progress = temp_progress
                eval_env.env._create_environment()
                
                obs = eval_env.reset()
                episode_reward = 0
                initial_pos = eval_env.head_position
                
                for _ in range(steps_per_episode):  # Data-driven steps per episode
                    action = agent.test_step(obs)
                    obs, reward, done, _ = eval_env.step(action)
                    episode_reward += reward
                    
                    if done:
                        break
                
                final_pos = eval_env.head_position
                distance = np.linalg.norm(final_pos - initial_pos)
                
                distances.append(distance)
                rewards.append(episode_reward)
                
                # Update progress bar if provided
                if progress_bar is not None:
                    phase_names = ["Pure Swimming", "Single Land Zone", "Two Land Zones", "Full Complexity"]
                    progress_bar.set_description(f"🔬 Evaluating {phase_names[phase]} ({episode+1}/{num_episodes})")
                    progress_bar.update(1)
                eval_env.close()
            
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
        
        # Create vectorized environment
        env = self.create_vectorized_environment()
        model = self.create_model()
        agent, tonic_model = self.create_agent(model, env)
        
        # Load checkpoint if resuming
        if self.resume_from_checkpoint:
            self.load_checkpoint(model, self.resume_from_checkpoint)
        
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
        
        # Reset vectorized environment
        obs = env.reset()
        episode_rewards = np.zeros(self.num_workers)
        episode_distances = np.zeros(self.num_workers)
        initial_positions = np.array(env.get_attr('head_position')).copy()
        
        while self.current_step < self.training_steps:
            # Get current training progress
            progress = self.current_step / self.training_steps
            current_phase = self.get_current_phase(progress)
            
            # Update environment progress across all workers (Reduced frequency to optimize speed)
            if self.current_step % (self.num_workers * 100) < self.num_workers:
                env.set_attr('env.training_progress', progress)
            
            # Check for phase transitions
            if current_phase != last_phase:
                main_pbar.set_description(f"🎓 Curriculum Training - {phase_names[current_phase]}")
                tqdm.write(f"\n🎓 PHASE TRANSITION: {last_phase} → {current_phase}")
                last_phase = current_phase
            
            # Apply biological constraints periodically
            if self.current_step % self.biological_constraint_frequency < self.num_workers:
                actual_model = model.module if isinstance(model, torch.nn.DataParallel) else model
                self.apply_biological_constraints(actual_model)
            
            # Step all environments in parallel
            actions = agent.step(obs)
            next_obs, rewards, dones, infos = env.step(actions)
            rewards = np.asarray(rewards, dtype=np.float32)
            rewards = np.nan_to_num(rewards, nan=0.0, posinf=0.0, neginf=0.0)
            dones = np.asarray(dones, dtype=bool)
            agent.add_rewards(rewards)
            
            episode_rewards += rewards
            self.current_step += self.num_workers
            
            # Update progress bar periodically to reduce overhead
            if self.current_step % (self.num_workers * 10) < self.num_workers:
                main_pbar.update(self.num_workers * 10)
            
            # Handle episode completions
            if np.any(dones):
                done_indices = np.where(dones)[0]
                
                # Calculate distances for completed episodes
                current_head_positions = np.array(env.get_attr('head_position')).copy()
                
                for idx in done_indices:
                    episode_distance = np.linalg.norm(current_head_positions[idx] - initial_positions[idx])
                    episode_reward_value = float(np.nan_to_num(episode_rewards[idx], nan=0.0, posinf=0.0, neginf=0.0))
                    episode_distance_value = float(np.nan_to_num(episode_distance, nan=0.0, posinf=0.0, neginf=0.0))
                    
                    self.current_episode += 1
                    self.phase_rewards[current_phase].append(episode_reward_value)
                    self.phase_distances[current_phase].append(episode_distance_value)
                    
                    # Log to file periodically
                    if self.current_episode % self.log_episodes == 0:
                        self.logger.log_training_step({
                            'step': self.current_step,
                            'episode': self.current_episode,
                            'phase': current_phase,
                            'reward': episode_reward_value,
                            'distance': episode_distance_value,
                        })
                    
                    # Reset worker state
                    episode_rewards[idx] = 0
                    episode_distances[idx] = 0
                    initial_positions[idx] = current_head_positions[idx]
                
                # Train on completed episodes
                agent.end_episodes(done_indices)
            
            obs = next_obs
            
            # Periodic logging
            if self.current_episode % self.log_episodes == 0 and np.any(dones):
                elapsed_time = time.time() - start_time
                steps_per_sec = self.current_step / elapsed_time if elapsed_time > 0 else 0
                
                recent_reward = np.mean(self.phase_rewards[current_phase][-10:]) if self.phase_rewards[current_phase] else 0
                recent_distance = np.mean(self.phase_distances[current_phase][-10:]) if self.phase_distances[current_phase] else 0
                
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
                    'reward': recent_reward,
                    'distance': recent_distance,
                    'mean_reward_10': recent_reward,   # recent_reward is already a mean of 10
                    'mean_distance_10': recent_distance,
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
                
                # Need a single environment for visualizations
                eval_env = self.create_environment()
                
                trajectory_stats = create_trajectory_analysis(
                    agent=agent,
                    env=eval_env,
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
                    env=eval_env,
                    save_path=video_path,
                    num_steps=300,
                    episode_name=f"Curriculum Step {self.current_step}"
                )
                eval_env.close()
        
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
                eval_env = self.create_environment()
                temp_progress = (phase + 0.5) * 0.25  # Middle of each phase
                force_land_for_evaluation = phase >= 1 and not getattr(eval_env.env, 'prefer_transition_evaluation', False)
                eval_env.env.set_manual_progress(temp_progress, force_land_start=force_land_for_evaluation)
                
                trajectory_path = self.artifact_namer.analysis_plot_name(
                    "final_trajectory", 
                    phase=f"phase{phase}",
                    base_dir="outputs/curriculum_training/plots"
                )
                stats = create_trajectory_analysis(
                    agent=agent,
                    env=eval_env,
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
            eval_env = self.create_environment()
            create_phase_comparison_video(
                agent=agent,
                env=eval_env,
                save_path=final_video_path,
                phases_to_test=[0, 1, 2, 3],
                phase_video_steps=self.PHASE_DURATION_CONFIG['video_steps']
            )
            eval_env.close()
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
                eval_env = self.create_environment()
                temp_progress = (phase + 0.5) * 0.25  # Middle of each phase
                force_land_for_evaluation = phase >= 1 and not getattr(eval_env.env, 'prefer_transition_evaluation', False)
                eval_env.env.set_manual_progress(temp_progress, force_land_start=force_land_for_evaluation)
                
                eval_trajectory_path = self.artifact_namer.analysis_plot_name(
                    "evaluation_trajectory", 
                    phase=f"phase{phase}",
                    base_dir="outputs/curriculum_training/plots"
                )
                trajectory_stats = create_trajectory_analysis(
                    agent=agent,
                    env=eval_env,
                    save_path=eval_trajectory_path,
                    num_steps=video_steps,
                    phase_name=f"Evaluation - {phase_names[phase]}",
                    trajectory_multiplier=self.PHASE_DURATION_CONFIG['trajectory_multiplier'][phase]
                )
                
                final_trajectory_stats[phase] = trajectory_stats
                eval_env.close()
                vis_pbar.update(1)
                print(f"   ✅ {phase_names[phase]}: {trajectory_stats['final_distance']:.3f}m, {trajectory_stats['transitions']} transitions")
            
            # Phase comparison video
            vis_pbar.set_description("🎬 Creating phase comparison video")
            eval_comparison_video_path = self.artifact_namer.evaluation_video_name(
                evaluation_type="phase_comparison",
                base_dir="outputs/curriculum_training/videos"
            )
            eval_env = self.create_environment()
            create_phase_comparison_video(
                agent=agent,
                env=eval_env,
                save_path=eval_comparison_video_path,
                phases_to_test=[0, 1, 2, 3],
                phase_video_steps=self.PHASE_DURATION_CONFIG['video_steps']
            )
            eval_env.close()
            vis_pbar.update(1)
            print(f"✅ Phase comparison video: {eval_comparison_video_path}")
            
            # Individual test videos for each phase
            for phase in range(4):
                vis_pbar.set_description(f"🎬 Creating {phase_names[phase]} video")
                
                eval_env = self.create_environment()
                temp_progress = (phase + 0.5) * 0.25
                force_land_for_evaluation = phase >= 1 and not getattr(eval_env.env, 'prefer_transition_evaluation', False)
                eval_env.env.set_manual_progress(temp_progress, force_land_start=force_land_for_evaluation)
                
                phase_video_path = self.artifact_namer.evaluation_video_name(
                    evaluation_type=f"phase{phase}_{phase_names[phase].lower().replace(' ', '_')}",
                    base_dir="outputs/curriculum_training/videos"
                )
                create_test_video(
                    agent=agent,
                    env=eval_env,
                    save_path=phase_video_path,
                    num_steps=video_steps,
                    episode_name=f"Evaluation - {phase_names[phase]}"
                )
                eval_env.close()
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
