#!/usr/bin/env python3
"""
Curriculum Trainer for Swimming and Crawling
Manages progressive training from simple swimming to complex mixed environments.
"""

import random
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
from ..environments.micro_publication_env import TonicMicroPublicationWrapper
from ..utils.training_logger import TrainingLogger
from ..utils.curriculum_visualization import create_curriculum_plots, create_test_video, create_phase_comparison_video, save_training_summary, create_trajectory_analysis
from ..utils.artifact_naming import ArtifactNamer, detect_model_type

try:
    from NMAP.connectome_priors.swimmer_priors import generate_ncap_segment_priors, refresh_inventory_files
except Exception:
    generate_ncap_segment_priors = None
    refresh_inventory_files = None

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
    
    # Single source of truth for curriculum phase names
    PHASE_NAMES = ["Pure Swimming", "Single Land Zone", "Two Land Zones", "Full Complexity"]

    # Phase duration configuration (easily modifiable)
    PHASE_DURATION_CONFIG = {
        'evaluation_steps': [400, 600, 800, 1200],
        'video_steps': [800, 1000, 1200, 1500],
        'trajectory_multiplier': [1.5, 2.0, 2.5, 3.0]
    }
    _SPARSE_PATHWAYS = (
        "ipsi_db",
        "ipsi_vb",
        "contra_db",
        "contra_vb",
        "next_db",
        "next_vb",
    )
    
    def __init__(self, 
                 n_links=5,
                 learning_rate=3e-5,
                 training_steps=1000000,
                 save_steps=50000,
                 log_episodes=50,
                 log_dir='results/manual_run',
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 oscillator_period=60,
                 min_oscillator_strength=0.8,  # **REDUCED** from 1.2 to 0.8 for speed flexibility
                 min_coupling_strength=0.5,  # **REDUCED** from 0.8 to 0.5 for speed flexibility  
                 biological_constraint_frequency=25000,  # **REDUCED** frequency: every 25k steps
                 resume_from_checkpoint=None,  # Path to checkpoint to resume from
                 model_type='enhanced_ncap',  # Model type: biological_ncap, enhanced_ncap
                 algorithm='ppo',  # Algorithm for naming
                 use_locomotion_only_early_training=True,
                 sparse_init: bool = False,
                 sparse_reg_lambda: float = 0.0,
                 force_oscillation: bool = False,
                 num_workers=8,
                 use_multi_gpu=True,
                 expose_environment_observation=True,
                 expose_viscosity_observation=True,
                 anisotropic_drag_mode='off',
                 anisotropic_drag_ratio=10.0,
                 anisotropic_drag_gain=0.02,
                 anisotropic_drag_land_only=True):
        
        self.n_links = n_links
        self.learning_rate = learning_rate
        self.training_steps = training_steps
        self.save_steps = save_steps
        self.log_episodes = log_episodes
        self.log_dir = os.path.abspath(log_dir)
        self.device = device
        self.oscillator_period = oscillator_period
        self.min_oscillator_strength = min_oscillator_strength
        self.min_coupling_strength = min_coupling_strength
        self.biological_constraint_frequency = biological_constraint_frequency
        self.resume_from_checkpoint = resume_from_checkpoint
        self.model_type = model_type
        self.algorithm = algorithm
        self.use_locomotion_only_early_training = use_locomotion_only_early_training
        self.force_oscillation = bool(force_oscillation)
        self.sparse_init = bool(sparse_init)
        self.sparse_reg_lambda = float(sparse_reg_lambda)
        self.prior_modulation_scale = 0.15 if self.sparse_init else 0.0
        self.effective_prior_lambda = self.sparse_reg_lambda
        # Backward-compatible aliases.
        self.use_sparse_priors = self.sparse_init
        self.prior_lambda = self.sparse_reg_lambda
        self.sparse_prior_scalars = {}
        self.sparse_prior_metadata = {}
        self._prepare_sparse_priors()
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

        self.curriculum_output_root = os.path.join(self.log_dir, "curriculum_training")
        self.curriculum_checkpoints_dir = os.path.join(self.curriculum_output_root, "checkpoints")
        self.curriculum_plots_dir = os.path.join(self.curriculum_output_root, "plots")
        self.curriculum_videos_dir = os.path.join(self.curriculum_output_root, "videos")
        self.curriculum_models_dir = os.path.join(self.curriculum_output_root, "models")
        self.curriculum_summaries_dir = os.path.join(self.curriculum_output_root, "summaries")
        self.curriculum_logs_dir = os.path.join(self.curriculum_output_root, "logs")
        
        # Training state
        self.current_step = 0
        self.current_episode = 0
        self.phase_rewards = {0: [], 1: [], 2: [], 3: []}
        self.phase_distances = {0: [], 1: [], 2: [], 3: []}
        
        # Initialize components with advanced logging if available
        log_dir = os.path.dirname(self.artifact_namer.training_log_dir(base_dir=self.curriculum_logs_dir))
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
        if self.sparse_init:
            print(f"   Sparse init: enabled (modulation={self.prior_modulation_scale:.2f})")
        else:
            print("   Sparse init: disabled (tabula rasa initialization)")
        if self.effective_prior_lambda > 0.0:
            print(f"   Sparse regularization: enabled (lambda={self.effective_prior_lambda:.4f})")
        else:
            print("   Sparse regularization: disabled")
        if self.force_oscillation:
            print("   Forced oscillation: enabled (min variance=0.1)")
        else:
            print("   Forced oscillation: disabled")

    def _default_sparse_priors(self):
        return {
            "dist_ipsi_db": 1.0,
            "dist_ipsi_vb": 1.0,
            "dist_contra_db": 1.0,
            "dist_contra_vb": 1.0,
            "dist_next_db": 1.0,
            "dist_next_vb": 1.0,
            "syn_ipsi_db": 0.0,
            "syn_ipsi_vb": 0.0,
            "syn_contra_db": 0.0,
            "syn_contra_vb": 0.0,
            "syn_next_db": 0.0,
            "syn_next_vb": 0.0,
        }

    def _prepare_sparse_priors(self):
        if not self.sparse_init and self.effective_prior_lambda <= 0.0:
            self.sparse_prior_scalars = self._default_sparse_priors()
            self.sparse_prior_metadata = {"status": "disabled"}
            return

        priors_fn = generate_ncap_segment_priors
        if priors_fn is None:
            self.sparse_init = False
            self.use_sparse_priors = False
            self.effective_prior_lambda = 0.0
            self.prior_modulation_scale = 0.0
            self.sparse_prior_scalars = self._default_sparse_priors()
            self.sparse_prior_metadata = {"status": "import_error", "reason": "sparse prior module unavailable"}
            print("⚠️ Could not import sparse priors. Falling back to tabula rasa curriculum mode.")
            return

        try:
            inventory_refresh = {"status": "skipped"}
            if refresh_inventory_files is not None:
                try:
                    inventory_refresh = refresh_inventory_files()
                    inventory_refresh["status"] = "ok"
                except Exception as refresh_exc:
                    inventory_refresh = {"status": "error", "reason": str(refresh_exc)}

            priors = priors_fn(num_segments=max(1, int(self.n_links - 1)))
            defaults = self._default_sparse_priors()
            self.sparse_prior_scalars = {
                key: float(priors.get(key, defaults[key]))
                for key in defaults
            }
            metadata = priors.get("metadata", {}) if isinstance(priors, dict) else {}
            if isinstance(metadata, dict):
                metadata["inventory_refresh"] = inventory_refresh
            self.sparse_prior_metadata = {
                "status": "ok",
                "sources": priors.get("sources", {}),
                "metadata": metadata,
            }
        except Exception as exc:
            self.sparse_init = False
            self.use_sparse_priors = False
            self.effective_prior_lambda = 0.0
            self.prior_modulation_scale = 0.0
            self.sparse_prior_scalars = self._default_sparse_priors()
            self.sparse_prior_metadata = {"status": "load_error", "reason": str(exc)}
            print(f"⚠️ Sparse prior generation failed. Falling back to tabula rasa curriculum mode. ({exc})")

    def _iter_sparse_pathway_params(self, model, pathway: str):
        if not hasattr(model, "params"):
            return
        for name, param in model.params.items():
            if pathway == "ipsi_db":
                if name.startswith("muscle_d_d_") or name == "muscle_ipsi":
                    yield param
            elif pathway == "ipsi_vb":
                if name.startswith("muscle_v_v_") or name == "muscle_ipsi":
                    yield param
            elif pathway == "contra_db":
                if name.startswith("muscle_v_d_") or name == "muscle_contra":
                    yield param
            elif pathway == "contra_vb":
                if name.startswith("muscle_d_v_") or name == "muscle_contra":
                    yield param
            elif pathway == "next_db":
                if name.startswith("bneuron_d_prop_") or name == "bneuron_prop":
                    yield param
            elif pathway == "next_vb":
                if name.startswith("bneuron_v_prop_") or name == "bneuron_prop":
                    yield param

    def _apply_sparse_prior_initialization(self, model):
        if not self.sparse_init or not hasattr(model, "params"):
            return

        syn_values = []
        for pathway in self._SPARSE_PATHWAYS:
            syn = float(self.sparse_prior_scalars.get(f"syn_{pathway}", 0.0))
            if np.isfinite(syn) and syn > 0:
                syn_values.append(syn)
        syn_scale = max(syn_values) if syn_values else 1.0
        jitter_fraction = float(max(0.0, min(self.prior_modulation_scale, 0.45)))

        with torch.no_grad():
            for pathway in self._SPARSE_PATHWAYS:
                syn = float(self.sparse_prior_scalars.get(f"syn_{pathway}", 0.0))
                norm_strength = (syn / syn_scale) if syn_scale > 0 else 0.0
                norm_strength = float(min(max(norm_strength, 0.0), 1.0))
                base_strength = float(max(0.05, norm_strength))
                jitter = base_strength * jitter_fraction
                low = max(0.0, base_strength - jitter)
                high = min(1.0, base_strength + jitter)
                if low > high:
                    low, high = high, low

                inhibitory = pathway.startswith("contra_")
                for param in self._iter_sparse_pathway_params(model, pathway):
                    if inhibitory:
                        param.uniform_(-high, -low)
                    else:
                        param.uniform_(low, high)

    def compute_sparse_prior_loss(self, model, device=None):
        if self.effective_prior_lambda <= 0.0:
            if device is None:
                return torch.tensor(0.0)
            return torch.zeros((), device=device, dtype=torch.float32)

        if not hasattr(model, "params"):
            if device is None:
                return torch.tensor(0.0)
            return torch.zeros((), device=device, dtype=torch.float32)

        if device is None:
            device = next(model.parameters()).device

        total = torch.zeros((), device=device, dtype=torch.float32)
        for pathway in self._SPARSE_PATHWAYS:
            dist = float(self.sparse_prior_scalars.get(f"dist_{pathway}", 1.0))
            terms = [(p ** 2).sum() for p in self._iter_sparse_pathway_params(model, pathway)]
            if terms:
                total = total + dist * torch.stack(terms).sum()
        return self.effective_prior_lambda * total
        
    def create_environment(self):
        """Create micro-publication mixed environment with clean reward decomposition."""
        env = TonicMicroPublicationWrapper(
            n_links=self.n_links,
            time_feature=True,
            desired_speed=0.15,
            land_start_probability=0.35,
        )
        # Step-based curriculum: disable auto episode-count update so trainer
        # can set training_progress directly as a fraction of training_steps.
        env.env.manual_progress_override = True

        print(f"🌊 Created micro-publication mixed environment")
        print(f"   Environment: {env.name}")
        print(f"   Observation space: {env.observation_space.shape}")
        print(f"   Action space: {env.action_space.shape}")

        return env
    
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
                use_weight_sharing=not self.sparse_init,
                include_environment_adaptation=True,  # Dramatic frequency adaptation
                include_goal_direction=not use_locomotion_only,  # **DISABLED** for early training
                locomotion_only_mode=use_locomotion_only,  # **NEW**: Pure swimming mode
                action_scaling_factor=1.8  
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
                use_weight_sharing=not self.sparse_init,
                include_environment_adaptation=True  # Enable biological adaptation
            ).to(self.device)
            
            print(f"🧬 Created standard Biological NCAP model with {sum(p.numel() for p in model.parameters())} parameters")
            print(f"🔬 Biological adaptation: ENABLED (no LSTM - pure neuromodulation)")
            
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}. "
                           f"Supported: 'biological_ncap', 'enhanced_ncap'")

        self._apply_sparse_prior_initialization(model)

        # Wire the prior scalars onto the model so its own
        # compute_topological_prior_loss() method can be called directly
        # (e.g. by custom_tonic_agent if this model is ever used in that path).
        if hasattr(model, "configure_sparse_priors"):
            model.configure_sparse_priors(self.sparse_prior_scalars)

        return model
    
    def create_agent(self, model, env):
        """Create simplified agent for curriculum training."""
        trainer_ref = self
        
        # Create biological NCAP agent wrapper with environment adaptation
        class BiologicalNCAPAgent:
            def __init__(self, ncap_model, environment):
                self.ncap_model = ncap_model
                self.step_count = 0
                self.use_stable_init = False

                # Precompute observation layout offsets from n_links.
                # Layout: [joints(n_j), body_vel(n_links*3+3), env(5), goal(7), time(1)]
                _n_links = trainer_ref.n_links
                self._n_joints = _n_links - 1
                self._env_features_start = self._n_joints + (_n_links * 3 + 3)
                self._goal_features_start = self._env_features_start + 5

                # Initialize RL training components
                learning_rate = trainer_ref.learning_rate
                self.optimizer = torch.optim.Adam(ncap_model.parameters(), lr=learning_rate)
                self.episode_buffer = {'obs': [], 'actions': [], 'rewards': []}
                self.training_enabled = True
                
            def step(self, obs):
                """Training step - returns action and buffers (obs, action)."""
                action = self.test_step(obs)
                # Store every (obs, action) so it aligns 1-to-1 with rewards
                # added via add_reward().  Removing the len>0 guard fixes the
                # off-by-one that caused obs[i] to be paired with reward[i-1].
                if self.training_enabled:
                    self.episode_buffer['obs'].append(obs)
                    self.episode_buffer['actions'].append(action)
                return action

            def add_reward(self, reward):
                """Add reward for the action taken in the current step."""
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
                        batch_actions =  np.array(batch_actions, dtype=np.float32)
                        actual_actions = torch.FloatTensor(batch_actions).to(device)
                        
                        # Policy gradient loss
                        loss = torch.nn.functional.mse_loss(predicted_actions, actual_actions, reduction='none')
                        policy_loss = (loss.mean(dim=1) * batch_returns[:len(loss)]).mean()
                        prior_loss = trainer_ref.compute_sparse_prior_loss(self.ncap_model, device=device)
                        if trainer_ref.force_oscillation:
                            action_variance = predicted_actions.var(dim=0, unbiased=False).mean()
                            min_variance = torch.tensor(0.1, device=device, dtype=predicted_actions.dtype)
                            oscillation_penalty = torch.relu(min_variance - action_variance)
                        else:
                            oscillation_penalty = torch.zeros((), device=device, dtype=predicted_actions.dtype)
                        total_loss = policy_loss + prior_loss + oscillation_penalty
                        
                        # Update model
                        self.optimizer.zero_grad()
                        total_loss.backward()
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

                # Joint positions
                nj = self._n_joints
                joint_pos = obs[:nj] if len(obs) >= nj else torch.zeros(nj, device=device)

                # Environment type: [water_flag, land_flag, viscosity_norm]
                environment_type = None
                ef_start = self._env_features_start
                if len(obs) >= ef_start + 3:
                    water_flag = obs[ef_start + 1:ef_start + 2]
                    land_flag  = obs[ef_start + 2:ef_start + 3]
                    vis_norm   = obs[ef_start:ef_start + 1]
                    environment_type = torch.cat([water_flag, land_flag, vis_norm])

                # Goal direction: target_direction at goal_features_start + 1 .. +3
                target_direction = None
                gf_start = self._goal_features_start
                if (len(obs) >= gf_start + 3
                        and hasattr(self.ncap_model, 'include_goal_direction')
                        and self.ncap_model.include_goal_direction):
                    target_direction = obs[gf_start + 1:gf_start + 3]
                
                # Keep the action path differentiable for policy updates.
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
                    nj = self._n_joints
                    joint_pos = torch.tensor(obs[:nj], dtype=torch.float32, device=device)

                    environment_type = None
                    ef_start = self._env_features_start
                    if len(obs) >= ef_start + 3:
                        water_flag = float(obs[ef_start + 1])
                        land_flag  = float(obs[ef_start + 2])
                        vis_norm   = float(obs[ef_start])
                        environment_type = np.array([water_flag, land_flag, vis_norm], dtype=np.float32)

                    target_direction = None
                    gf_start = self._goal_features_start
                    if (len(obs) >= gf_start + 3
                            and hasattr(self.ncap_model, 'include_goal_direction')
                            and self.ncap_model.include_goal_direction):
                        target_direction = obs[gf_start + 1:gf_start + 3].copy()
                
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
    
    def save_checkpoint(self, model, step, eval_results=None, optimizer=None):
        """Save training checkpoint with model-specific naming.

        Saves model weights, full training state, optimizer momentum/velocity
        (so Adam resumes correctly), and all RNG states (so sampling is
        reproducible after a resume).  Flushes in-memory logger metrics to
        disk so no data is lost if the process crashes before training ends.
        """
        checkpoint_path = self.artifact_namer.checkpoint_name(
            step=step,
            base_dir=self.curriculum_checkpoints_dir
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
            # Optimizer adaptive state (Adam m/v vectors) for seamless resume
            'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
            # RNG states for reproducibility across resume boundaries
            'torch_rng_state': torch.get_rng_state(),
            'numpy_rng_state': np.random.get_state(),
            'python_rng_state': random.getstate(),
        }
        if torch.cuda.is_available():
            checkpoint_data['cuda_rng_state'] = torch.cuda.get_rng_state()

        torch.save(checkpoint_data, checkpoint_path)
        print(f"💾 Checkpoint saved: {checkpoint_path}")

        # Flush in-memory metrics to disk (guards against crash data loss)
        self.logger.save_metrics()

        return checkpoint_path
    
    def load_checkpoint(self, model, checkpoint_path, optimizer=None):
        """Load training checkpoint with backward compatibility.

        Restores model weights, training counters, phase history, optimizer
        adaptive state (Adam m/v vectors), and all RNG states so that a
        resumed run is statistically identical to an uninterrupted one.
        Syncs the logger's internal step/episode counters so metrics are
        aligned with the restored training state.
        """
        print(f"📂 Loading checkpoint: {checkpoint_path}")

        checkpoint_data = torch.load(checkpoint_path, map_location=self.device,
                                     weights_only=False)

        # ── Model weights ────────────────────────────────────────────────────
        try:
            model.load_state_dict(checkpoint_data['model_state_dict'])
        except RuntimeError as e:
            if "Missing key(s)" in str(e):
                print(f"⚠️ Checkpoint compatibility issue: {e}")
                print("🔧 Attempting to load compatible parameters only...")

                model_state = model.state_dict()
                checkpoint_state = checkpoint_data['model_state_dict']

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

                model.load_state_dict(compatible_state, strict=False)

                print(f"✅ Loaded {len(compatible_state)} compatible parameters")
                if missing_params:
                    print(f"⚠️ Missing parameters (will use defaults): {missing_params}")
                if extra_params:
                    print(f"ℹ️ Extra parameters in checkpoint (ignored): {extra_params}")
            else:
                raise

        # ── Optimizer adaptive state ─────────────────────────────────────────
        # Restoring Adam's m/v accumulators means the effective learning rate
        # is immediately correct for each parameter (no warm-up artefact).
        if (optimizer is not None
                and 'optimizer_state_dict' in checkpoint_data
                and checkpoint_data['optimizer_state_dict'] is not None):
            try:
                optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
                print("✅ Optimizer state restored")
            except Exception as e:
                print(f"⚠️ Could not restore optimizer state (fresh optimizer used): {e}")

        # ── RNG states ───────────────────────────────────────────────────────
        # Restoring all RNG states ensures the resumed run draws the same
        # random sequences as it would have without interruption.
        if 'torch_rng_state' in checkpoint_data:
            torch.set_rng_state(checkpoint_data['torch_rng_state'])
        if 'numpy_rng_state' in checkpoint_data:
            np.random.set_state(checkpoint_data['numpy_rng_state'])
        if 'python_rng_state' in checkpoint_data:
            random.setstate(checkpoint_data['python_rng_state'])
        if 'cuda_rng_state' in checkpoint_data and torch.cuda.is_available():
            torch.cuda.set_rng_state(checkpoint_data['cuda_rng_state'])

        # ── Training counters and phase history ──────────────────────────────
        if 'current_step' in checkpoint_data:
            self.current_step = checkpoint_data['current_step']
            self.current_episode = checkpoint_data['current_episode']
            self.phase_rewards = checkpoint_data.get('phase_rewards', {0: [], 1: [], 2: [], 3: []})
            self.phase_distances = checkpoint_data.get('phase_distances', {0: [], 1: [], 2: [], 3: []})
        else:
            # Legacy checkpoint format
            self.current_step = checkpoint_data.get('step', 0)
            self.current_episode = checkpoint_data.get('episode', 0)
            self.phase_rewards = {0: [], 1: [], 2: [], 3: []}
            self.phase_distances = {0: [], 1: [], 2: [], 3: []}
            print("⚠️ Legacy checkpoint format detected — phase history reset")

        # ── Sync logger state ────────────────────────────────────────────────
        # Without this, the logger's internal step counter would restart at 0
        # and write metrics with wrong step indices.
        self.logger.current_step = self.current_step
        self.logger.current_episode = self.current_episode

        print(f"✅ Checkpoint loaded successfully!")
        print(f"   Resuming from step: {self.current_step:,}")
        print(f"   Episode: {self.current_episode:,}")

        return checkpoint_data.get('eval_results', {})
    
    def apply_biological_constraints(self, model):
        """Apply biological constraints to maintain realism."""
        constraints_applied = []

        # When sparse_init is active, use lower floor/ceiling so Cook-2019-derived
        # weights (some as small as 0.05 for low-synapse pathways) are not immediately
        # overridden at step 25 k before the optimizer can move them.  Without this,
        # sparse init has no lasting effect on experiments 03/04.
        if self.sparse_init:
            muscle_exc_min  = 0.1
            muscle_inh_max  = -0.1
            coupling_min    = 0.2
        else:
            muscle_exc_min  = 0.5
            muscle_inh_max  = -0.5
            coupling_min    = self.min_coupling_strength

        with torch.no_grad():
            for name, param in model.params.items():
                # Oscillator strength minimum — matches global 'bneuron_osc' and
                # segment-specific 'bneuron_d_osc_N' / 'bneuron_v_osc_N'
                if 'bneuron_d_osc' in name or 'bneuron_v_osc' in name or name == 'bneuron_osc':
                    if param.item() < self.min_oscillator_strength:
                        old_val = param.item()
                        param.data.fill_(self.min_oscillator_strength)
                        constraints_applied.append(f"{name} {old_val:.3f} → {self.min_oscillator_strength}")

                # Coupling strength minimum — matches global 'bneuron_prop' and
                # segment-specific 'bneuron_d_prop_N' / 'bneuron_v_prop_N'
                elif 'bneuron_d_prop' in name or 'bneuron_v_prop' in name or name == 'bneuron_prop':
                    if param.item() < coupling_min:
                        old_val = param.item()
                        param.data.fill_(coupling_min)
                        constraints_applied.append(f"{name} {old_val:.3f} → {coupling_min}")

                # Ipsilateral muscle minimum — matches global 'muscle_ipsi'
                # and segment-specific 'muscle_d_d_N' / 'muscle_v_v_N'
                elif name == 'muscle_ipsi' or name.startswith('muscle_d_d_') or name.startswith('muscle_v_v_'):
                    if param.item() < muscle_exc_min:
                        old_val = param.item()
                        param.data.fill_(muscle_exc_min)
                        constraints_applied.append(f"{name} {old_val:.3f} → {muscle_exc_min}")

                # Contralateral muscle maximum — matches global 'muscle_contra'
                # and segment-specific 'muscle_d_v_N' / 'muscle_v_d_N'
                elif name == 'muscle_contra' or name.startswith('muscle_d_v_') or name.startswith('muscle_v_d_'):
                    if param.item() > muscle_inh_max:
                        old_val = param.item()
                        param.data.fill_(muscle_inh_max)
                        constraints_applied.append(f"{name} {old_val:.3f} → {muscle_inh_max}")
        
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
            phase_names = self.PHASE_NAMES

            if steps_per_episode != self.PHASE_DURATION_CONFIG['evaluation_steps'][0]:  # Log when using non-standard duration
                print(f"🎯 {phase_names[phase]}: Using {steps_per_episode} steps per episode")
            
            distances = []
            rewards = []
            
            for episode in range(num_episodes):
                # Set environment to specific phase (property handles recreation)
                env.env.training_progress = temp_progress

                obs = env.reset()
                episode_reward = 0
                initial_pos = env.head_position.copy()

                for _ in range(steps_per_episode):
                    action = agent.test_step(obs)
                    obs, reward, done, _ = env.step(action)
                    episode_reward += reward

                    if done:
                        break

                final_pos = env.head_position.copy()
                distance = np.linalg.norm(final_pos - initial_pos)
                
                distances.append(distance)
                rewards.append(episode_reward)
                
                # Update progress bar if provided
                if progress_bar is not None:
                    progress_bar.set_description(
                        f"🔬 Evaluating {self.PHASE_NAMES[phase]} ({episode+1}/{num_episodes})"
                    )
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
        
        # Load checkpoint if resuming (pass optimizer so its adaptive state is restored)
        if self.resume_from_checkpoint:
            self.load_checkpoint(tonic_model, self.resume_from_checkpoint,
                                 optimizer=agent.optimizer)
        
        # Training loop with advanced monitoring
        start_time = time.time()
        self.logger.start_time = start_time
        last_phase = -1
        # Rolling throughput: track the step count and wall-clock time at the
        # last log event so steps/sec reflects recent rate, not total average.
        self._last_log_step = self.current_step
        self._last_log_time = start_time
        
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

                # ── Curriculum gate: unlock goal-directed navigation ──────────
                # The model is created at step-0 in locomotion_only_mode so that
                # the agent first masters pure forward propulsion.  Once we leave
                # Phase 0 (steps > 30 % of total) we flip the runtime flags so
                # the same model starts receiving and acting on goal signals.
                # No weight re-initialisation is needed: the locomotion sub-net
                # is already trained and the new goal pathway starts from its
                # random initial weights.
                if (last_phase == 0 and current_phase > 0
                        and self.use_locomotion_only_early_training):
                    if hasattr(model, "locomotion_only_mode"):
                        model.locomotion_only_mode = False
                        tqdm.write("   🔓 locomotion_only_mode → False")
                    if hasattr(model, "include_goal_direction"):
                        model.include_goal_direction = True
                        tqdm.write("   🎯 include_goal_direction → True")
                    tqdm.write("   Locomotion phase complete — goal-directed "
                               "navigation now active.")

                last_phase = current_phase
            
            # Apply biological constraints periodically.
            # Skip step 0 so that connectome-based sparse initialisation is not
            # immediately overridden by the minimum-value clamps.
            if (self.current_step > 0
                    and self.current_step % self.biological_constraint_frequency == 0):
                self.apply_biological_constraints(model)
            
            # Training step — update curriculum progress each episode
            env.env.training_progress = progress
            obs = env.reset()
            episode_reward = 0
            episode_steps = 0
            initial_pos = env.head_position.copy()
            
            # Run episode
            episode_start_step = self.current_step
            env_type_samples = []  # Track environment-type signal for neuromodulation diagnostics
            for _ in range(1000):  # Max episode length
                action = agent.step(obs)
                obs, reward, done, _ = env.step(action)

                # Track neuromodulation signal: land_flag is at env_features_start + 2
                _ef = agent._env_features_start
                if len(obs) > _ef + 2:
                    env_type_samples.append(float(obs[_ef + 2]))

                # CRITICAL: Add reward to agent for training (was missing!)
                agent.add_reward(reward)

                episode_reward += reward
                episode_steps += 1
                self.current_step += 1

                if done or self.current_step >= self.training_steps:
                    break

            # CRITICAL: Train on episode experience (was missing!)
            agent.end_episode()

            # Neuromodulation diagnostic: log variance of land_flag across episode.
            # Near-zero variance in Phase 0 is expected (always water). Non-zero in
            # Phase 2+ means the agent is actually triggering environment transitions
            # and the frequency-adaptation pathway is receiving varied input.
            neuromod_variance = float(np.var(env_type_samples)) if env_type_samples else 0.0
            
            # Update progress bar for steps taken this episode
            steps_this_episode = self.current_step - episode_start_step
            main_pbar.update(steps_this_episode)
            
            # Calculate episode distance
            final_pos = env.head_position.copy()
            episode_distance = np.linalg.norm(final_pos - initial_pos)
            
            # Log episode results
            self.current_episode += 1
            self.phase_rewards[current_phase].append(episode_reward)
            self.phase_distances[current_phase].append(episode_distance)
            
            # Periodic logging with ETA
            if self.current_episode % self.log_episodes == 0:
                elapsed_time = time.time() - start_time
                # Rolling window rate (steps since last log / time since last log)
                _now = time.time()
                _window_steps = self.current_step - self._last_log_step
                _window_time = _now - self._last_log_time
                steps_per_sec = _window_steps / _window_time if _window_time > 1e-6 else 0.0
                self._last_log_step = self.current_step
                self._last_log_time = _now
                
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
                
                # Log to file — keys match plot_ablation.py's _series() expectations
                self.logger.log_training_step({
                    'step': self.current_step,
                    'episode': self.current_episode,
                    'phase': current_phase,
                    'progress': progress,
                    'episode_reward': episode_reward,
                    'episode_distance': episode_distance,
                    'mean_reward_10': np.mean(recent_rewards),
                    'mean_distance_10': np.mean(recent_distances),
                    'neuromod_variance': neuromod_variance,
                })
            
            # Periodic saves and evaluation
            if self.current_step % self.save_steps == 0:
                tqdm.write(f"\n💾 Checkpoint at step {self.current_step:,}")
                
                # Comprehensive evaluation first
                eval_results = self.evaluate_performance(agent, env, num_episodes=10)
                
                # Save comprehensive checkpoint with eval results
                checkpoint_path = self.save_checkpoint(tonic_model, self.current_step, eval_results,
                                                        optimizer=agent.optimizer)
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
                        base_dir=self.curriculum_plots_dir
                    )
                    create_curriculum_plots(
                        phase_rewards=self.phase_rewards,
                        phase_distances=self.phase_distances,
                        eval_results=eval_results,
                        save_path=plot_path
                    )
                
                # Create trajectory analysis
                current_phase = min(int(self.current_step / (self.training_steps / 4)), 3)
                phase_names = self.PHASE_NAMES
                trajectory_path = self.artifact_namer.analysis_plot_name(
                    "trajectory_analysis", 
                    step=self.current_step,
                    phase=f"phase{current_phase}",
                    base_dir=self.curriculum_plots_dir
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
                    base_dir=self.curriculum_videos_dir
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
            tqdm.write(f"   {self.PHASE_NAMES[phase]}: {results['mean_distance']:.3f}m ± {results['std_distance']:.3f}")
        
        # Save final model
        final_path = self.artifact_namer.final_model_name(
            base_dir=self.curriculum_models_dir
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
                base_dir=self.curriculum_plots_dir
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
            phase_names = self.PHASE_NAMES
            final_trajectory_stats = {}
            
            for phase in range(4):
                pbar.set_description(f"📊 Analyzing {phase_names[phase]}")
                
                # Set environment to specific phase using manual override
                temp_progress = (phase + 0.5) * 0.25  # Middle of each phase
                force_land_for_evaluation = (phase >= 1 and
                    not getattr(env.env, 'prefer_transition_evaluation', False))
                env.env.set_manual_progress(temp_progress, force_land_start=force_land_for_evaluation)
                
                trajectory_path = self.artifact_namer.analysis_plot_name(
                    "final_trajectory", 
                    phase=f"phase{phase}",
                    base_dir=self.curriculum_plots_dir
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
                base_dir=self.curriculum_videos_dir
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
                base_dir=self.curriculum_summaries_dir
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
        for phase, results in final_eval.items():
            print(f"   {self.PHASE_NAMES[phase]}: {results['mean_distance']:.3f}m ± {results['std_distance']:.3f}")
        
        print(f"\n🎨 Creating comprehensive visualizations...")
        with tqdm(total=8, desc="📊 Creating Visualizations", unit="task",
                 bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as vis_pbar:
            
            # Training plots
            vis_pbar.set_description("📊 Creating final training plots")
            eval_plot_path = self.artifact_namer.analysis_plot_name(
                "evaluation_final", 
                base_dir=self.curriculum_plots_dir
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
            phase_names = self.PHASE_NAMES
            final_trajectory_stats = {}
            
            for phase in range(4):
                vis_pbar.set_description(f"📊 Analyzing {phase_names[phase]}")
                
                # Set environment to specific phase
                temp_progress = (phase + 0.5) * 0.25  # Middle of each phase
                force_land_for_evaluation = (phase >= 1 and
                    not getattr(env.env, 'prefer_transition_evaluation', False))
                env.env.set_manual_progress(temp_progress, force_land_start=force_land_for_evaluation)
                
                eval_trajectory_path = self.artifact_namer.analysis_plot_name(
                    "evaluation_trajectory", 
                    phase=f"phase{phase}",
                    base_dir=self.curriculum_plots_dir
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
                base_dir=self.curriculum_videos_dir
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
                temp_progress = (phase + 0.5) * 0.25
                force_land_for_evaluation = (phase >= 1 and
                    not getattr(env.env, 'prefer_transition_evaluation', False))
                env.env.set_manual_progress(temp_progress, force_land_start=force_land_for_evaluation)
                
                phase_video_path = self.artifact_namer.evaluation_video_name(
                    evaluation_type=f"phase{phase}_{phase_names[phase].lower().replace(' ', '_')}",
                    base_dir=self.curriculum_videos_dir
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
                base_dir=self.curriculum_summaries_dir
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
