#!/usr/bin/env python3
"""
Swimmer Trainer Implementation
Contains trainer classes for RL training of swimmer models.
"""

import torch
import numpy as np
import os
from pathlib import Path
import tonic
import tonic.torch
from ..environments.mixed_environment import MixedSwimmerEnv
from ..environments.tonic_wrapper import TonicSwimmerWrapper
from ..models.ncap_swimmer import NCAPSwimmer
from ..models.tonic_ncap import create_tonic_ncap_model
from ..utils.training_logger import TrainingLogger

class SwimmerTrainer:
    """
    Trainer class for swimmer models using RL algorithms.
    Supports NCAP and MLP models with PPO and other algorithms.
    """
    def __init__(
        self,
        model_type='ncap',
        algorithm='ppo',
        n_links=6,
        training_steps=2000000,
        save_steps=100000,
        output_dir=None,
        log_dir='results/manual_run',
        log_episodes=10,
        action_scale=1.0,
        sparse_init: bool = False,
        sparse_reg_lambda: float = 0.0,
        force_oscillation: bool = False,
    ):
        self.model_type = model_type.lower()
        self.algorithm = algorithm.lower()
        self.n_links = n_links
        self.training_steps = training_steps
        self.save_steps = save_steps
        self.log_episodes = log_episodes
        self.sparse_init = bool(sparse_init)
        self.sparse_reg_lambda = float(sparse_reg_lambda)
        self.force_oscillation = bool(force_oscillation)
        self.prior_modulation_scale = 0.15 if self.sparse_init else 0.0
        self.effective_prior_lambda = self.sparse_reg_lambda

        base_artifact_dir = Path(log_dir or output_dir or 'results/manual_run').resolve()
        self.log_dir = str(base_artifact_dir)
        self.output_dir = str(base_artifact_dir / "models")
        self.training_log_dir = str(base_artifact_dir / "logs")
        self.tonic_log_dir = str(base_artifact_dir / "tonic")
        self.eval_output_dir = str(base_artifact_dir / "mixed_env")

        # Backward-compatible aliases for legacy helper methods.
        self.use_sparse_priors = self.sparse_init
        self.prior_lambda = self.sparse_reg_lambda
        self._sparse_prior_cache = None

        # Store the action scaling factor so that environments and wrappers
        # that depend on it (e.g.
        # `TonicSwimmerWrapper` in `create_tonic_environment`) can access it.
        self.action_scale = action_scale
        
        # Create artifact directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.training_log_dir, exist_ok=True)
        os.makedirs(self.tonic_log_dir, exist_ok=True)
        os.makedirs(self.eval_output_dir, exist_ok=True)
        
        # Check for GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize environment and model
        self.env = None
        self.model = None
        self.agent = None
        self.trainer = None
        
        # Initialize training logger
        experiment_name = f"{self.model_type}_{self.algorithm}_{self.n_links}links"
        self.logger = TrainingLogger(
            log_dir=self.training_log_dir,
            experiment_name=experiment_name
        )

    def _prepare_sparse_priors(self, num_segments: int):
        """Load sparse priors once when sparse init or sparse reg is enabled."""
        if not self.sparse_init and self.effective_prior_lambda <= 0.0:
            return None
        if self._sparse_prior_cache is not None:
            return self._sparse_prior_cache
        from NMAP.connectome_priors.swimmer_priors import generate_ncap_segment_priors, refresh_inventory_files

        inventory_refresh = {"status": "skipped"}
        try:
            inventory_refresh = refresh_inventory_files()
            inventory_refresh["status"] = "ok"
        except Exception as exc:
            inventory_refresh = {"status": "error", "reason": str(exc)}

        priors = generate_ncap_segment_priors(num_segments=int(max(1, num_segments)))
        metadata = priors.get("metadata", {}) if isinstance(priors, dict) else {}
        if isinstance(metadata, dict):
            metadata["inventory_refresh"] = inventory_refresh
            if isinstance(priors, dict):
                priors["metadata"] = metadata
        self._sparse_prior_cache = priors
        return self._sparse_prior_cache
        
    def create_environment(self):
        """Create the training environment."""
        return MixedSwimmerEnv(n_links=self.n_links)
    
    def create_tonic_environment(self):
        """Create Tonic-compatible environment (no action scaling needed)."""
        return TonicSwimmerWrapper(n_links=self.n_links, time_feature=True)
    
    def create_ncap_model(self, n_joints):
        """Create NCAP model."""
        model = NCAPSwimmer(n_joints=n_joints, oscillator_period=60, memory_size=10)
        model.to(self.device)
        return model
    
    def create_tonic_ncap_model(self, n_joints):
        """Create Tonic-compatible NCAP model."""
        self._prepare_sparse_priors(num_segments=n_joints)
        model = create_tonic_ncap_model(
            n_joints=n_joints,
            oscillator_period=60,
            memory_size=10,
            num_segments=max(1, int(n_joints)),
            prior_modulation_scale=self.prior_modulation_scale,
        )
        model.to(self.device)
        return model
    
    def create_mlp_model(self, n_joints):
        """Create MLP model (placeholder for future implementation)."""
        # This would be implemented similar to the notebook's ppo_mlp_model
        raise NotImplementedError("MLP model not yet implemented")
    
    def create_model(self, n_joints):
        """Create model based on model_type."""
        if self.model_type == 'ncap':
            return self.create_ncap_model(n_joints)
        elif self.model_type == 'mlp':
            return self.create_mlp_model(n_joints)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def create_tonic_agent(self, model):
        """Create the functional Tonic agent according to the algorithm toggle."""
        from .custom_tonic_agent import CustomA2C, CustomPPO

        if self.algorithm == 'ppo':
            agent_cls = CustomPPO
        elif self.algorithm == 'a2c':
            agent_cls = CustomA2C
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        return agent_cls(
            model=model,
            prior_reg_lambda=self.effective_prior_lambda,
            force_oscillation=self.force_oscillation,
        )
    
    def create_agent(self, model):
        """Return the single supported training agent implementation."""
        return self.create_tonic_agent(model)
    
    def train_with_tonic(self):
        """Train using Tonic framework."""
        from datetime import datetime
        start_ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{start_ts}] ▶ Starting Tonic training with {self.model_type.upper()} model and {self.algorithm.upper()} algorithm")
        
        # Set up Tonic logger with proper directory
        import tonic
        tonic.logger.initialize(path=self.tonic_log_dir)
        
        # Log training configuration
        config = {
            'model_type': self.model_type,
            'algorithm': self.algorithm,
            'n_links': self.n_links,
            'training_steps': self.training_steps,
            'save_steps': self.save_steps,
            'device': str(self.device),
            'log_episodes': self.log_episodes,
            'framework': 'tonic',
            'sparse_init': self.sparse_init,
            'sparse_reg_lambda': self.effective_prior_lambda,
            'prior_modulation_scale': self.prior_modulation_scale,
            'force_oscillation': self.force_oscillation,
        }
        self.logger.log_config(config)
        self.logger.start_training()
        
        # Create environment
        env = self.create_tonic_environment()
        
        # Get action space info
        n_joints = env.action_space.shape[0]
        
        # Create model
        if self.model_type == 'ncap':
            model = self.create_tonic_ncap_model(n_joints)
        else:
            model = self.create_mlp_model(n_joints)
        
        # Move model to device
        model = model.to(self.device)
        print(f"Model moved to device: {self.device}")
        
        # Create agent
        agent = self.create_tonic_agent(model)
        
        # Create trainer with custom callbacks for logging
        trainer = tonic.Trainer(
            steps=self.training_steps,
            save_steps=self.save_steps,
            test_episodes=5
        )
        
        # Initialize agent and trainer
        agent.initialize(
            observation_space=env.observation_space,
            action_space=env.action_space,
            seed=42
        )
        
        trainer.initialize(
            agent=agent,
            environment=env,
            test_environment=env
        )
        
        # Run training
        print(f"Training for {self.training_steps} steps...")
        trainer.run()
        
        # Save final model
        self.save_tonic_model(agent)
        
        # Store the trained agent and environment for evaluation
        self.agent = agent
        self.env = env
        self.model = model
        
        end_ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{end_ts}] ✔ Tonic training completed!")
    
    def train(self):
        """Train the model."""
        # Use Tonic training
        self.train_with_tonic()
    
    def _simple_training_loop(self):
        """Simple training loop (placeholder for Tonic integration)."""
        print("Using simple training loop (Tonic integration pending)")
        
        # Create environment
        self.env = self.create_environment()
        
        # Get action space info
        action_spec = self.env.action_spec
        n_joints = action_spec.shape[0]
        
        # Create model
        self.model = self.create_model(n_joints)
        
        # Move model to GPU and verify
        if self.device.type == 'cuda':
            self.model = self.model.to(self.device)
            print(f"Model moved to GPU: {next(self.model.parameters()).device}")
            print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        else:
            print(f"Model on CPU: {next(self.model.parameters()).device}")
        
        # Create agent
        self.agent = self.create_agent(self.model)
        
        # Log training configuration
        config = {
            'model_type': self.model_type,
            'algorithm': self.algorithm,
            'n_links': self.n_links,
            'training_steps': self.training_steps,
            'save_steps': self.save_steps,
            'device': str(self.device),
            'log_episodes': self.log_episodes,
            'framework': 'simple'
        }
        self.logger.log_config(config)
        self.logger.start_training()
        
        # Simulate some training episodes for demonstration
        for episode in range(min(20, self.training_steps // 1000)):  # Simulate 20 episodes
            obs = self.env.reset()
            episode_reward = 0
            episode_length = 0
            episode_distance = 0
            env_transitions = 0
            velocities = []
            initial_pos = None
            current_env = None
            
            while not self.env.done:
                # Get action from agent
                action = self.agent.act(obs)
                
                # Take step
                obs, reward, done, info = self.env.step(action)
                episode_reward += reward
                episode_length += 1
                
                # Track distance
                if initial_pos is None:
                    initial_pos = self.env.physics.named.data.xpos['head'][:2].copy()
                current_pos = self.env.physics.named.data.xpos['head'][:2]
                episode_distance = np.linalg.norm(current_pos - initial_pos)
                
                # Track velocity
                current_velocity = self.env.physics.named.data.sensordata['head_vel']
                velocity_mag = np.linalg.norm(current_velocity[:2])
                velocities.append(velocity_mag)
                
                # Track environment transitions
                new_env = self.env.env.task.get_current_environment(self.env.physics)
                if current_env != new_env and current_env is not None:
                    env_transitions += 1
                current_env = new_env
            
            # Calculate episode metrics
            avg_velocity = np.mean(velocities) if velocities else 0.0
            max_velocity = np.max(velocities) if velocities else 0.0
            
            # Log episode
            self.logger.log_episode(
                episode_reward=episode_reward,
                episode_length=episode_length,
                episode_distance=episode_distance,
                env_transitions=env_transitions,
                avg_velocity=avg_velocity,
                max_velocity=max_velocity
            )
            
            # Log some fake training metrics (will be replaced by real PPO metrics)
            if episode % 5 == 0:  # Log every 5 episodes
                fake_loss = 1.0 / (1.0 + episode)  # Decreasing loss
                fake_policy_loss = fake_loss * 0.7
                fake_value_loss = fake_loss * 0.3
                
                self.logger.log_training_step(
                    loss=fake_loss,
                    policy_loss=fake_policy_loss,
                    value_loss=fake_value_loss,
                    entropy=0.1 + 0.05 * np.random.random(),
                    learning_rate=0.001
                )
            
            # Print progress
            if episode % self.log_episodes == 0:
                self.logger.print_progress(episode=episode)
        
        # Save final model
        self.save_model()
        
        # Create training plots and summary
        self.logger.save_metrics()
        self.logger.create_training_plots()
        self.logger.create_summary_report()
    
    def evaluate(self, num_episodes=10):
        """Evaluate the trained model."""
        if self.model is None:
            raise ValueError("No model loaded. Train or load a model first.")
        
        print(f"Evaluating model over {num_episodes} episodes...")
        
        total_rewards = []
        total_lengths = []
        
        for episode in range(num_episodes):
            obs = self.env.reset()
            episode_reward = 0
            step_count = 0
            max_steps = 1000  # Maximum episode length
            
            while step_count < max_steps:
                # Get action from agent
                if hasattr(self.agent, 'test_step'):
                    # Use Tonic agent interface
                    action = self.agent.test_step(obs, steps=step_count)
                else:
                    # Use simple agent interface
                    action = self.agent.act(obs)
                
                # Take step
                if hasattr(self.env, 'step') and hasattr(self.env, 'env'):
                    # Tonic environment wrapper
                    obs, infos = self.env.step(action)
                    reward = infos['rewards'][0]
                    done = infos['resets'][0]
                else:
                    # Standard environment
                    obs, reward, done, info = self.env.step(action)
                
                episode_reward += reward
                step_count += 1
                
                if done:
                    break
            
            total_rewards.append(episode_reward)
            total_lengths.append(step_count)
            
            # Print progress
            if (episode + 1) % 2 == 0:
                print(f"  Episode {episode + 1}/{num_episodes}: Reward = {episode_reward:.2f}, Length = {step_count}")
        
        avg_reward = np.mean(total_rewards)
        avg_length = np.mean(total_lengths)
        
        print(f"\nEvaluation Results:")
        print(f"  Average Reward: {avg_reward:.4f}")
        print(f"  Average Episode Length: {avg_length:.1f}")
        print(f"  Reward Range: {min(total_rewards):.2f} - {max(total_rewards):.2f}")
        
        return {
            'avg_reward': avg_reward,
            'avg_length': avg_length,
            'rewards': total_rewards,
            'lengths': total_lengths
        }
    
    def evaluate_mixed_environment(self, max_frames=5000, speed_factor=1.0):
        """Evaluate the trained model in the mixed environment using existing infrastructure."""
        if self.model is None:
            raise ValueError("No model loaded. Train or load a model first.")
        
        print(f"Evaluating model in mixed environment for {max_frames} frames...")
        
        # Import the existing test function and modify it to use our trained model
        from ..environments.mixed_environment import MixedSwimmerEnv
        from ..utils.visualization import create_comprehensive_visualization, create_parameter_log
        import imageio
        import time
        
        # Create mixed environment
        env = MixedSwimmerEnv(n_links=self.n_links, speed_factor=speed_factor)
        physics = env.physics
        action_spec = env.action_spec
        n_joints = action_spec.shape[0]
        
        # Performance tracking
        start_time = time.time()
        initial_head_pos = physics.named.data.xpos['head'].copy()
        velocities = []
        rewards_list = []
        distances = []
        environment_history = []
        
        # Video generation
        os.makedirs(self.eval_output_dir, exist_ok=True)
        video_filename = os.path.join(self.eval_output_dir, f"trained_model_evaluation_{self.n_links}links.mp4")
        plot_filename = os.path.join(self.eval_output_dir, f"trained_model_analysis_{self.n_links}links.png")
        log_filename = os.path.join(self.eval_output_dir, f"trained_model_log_{self.n_links}links.txt")
        
        frame_count = 0
        frames = []
        reset_rng = np.random.default_rng(42)

        def _reset_with_randomized_start():
            """Reset env and jitter initial pose so evaluation requires navigation."""
            reset_obs = env.reset()
            try:
                with physics.reset_context():
                    qpos = physics.data.qpos.copy()
                    qvel = physics.data.qvel.copy()
                    base_xy = qpos[:2].copy()
                    target_xy = None
                    try:
                        target_xy = np.asarray(physics.named.model.geom_pos['target'][:2], dtype=np.float64)
                    except Exception:
                        target_xy = None

                    chosen_xy = base_xy
                    for _ in range(24):
                        offset = reset_rng.uniform(-0.55, 0.55, size=2)
                        if np.linalg.norm(offset) < 0.2:
                            continue
                        candidate = base_xy + offset
                        if target_xy is not None and np.linalg.norm(candidate - target_xy) < 0.5:
                            continue
                        chosen_xy = candidate
                        break

                    qpos[0] = float(chosen_xy[0])
                    qpos[1] = float(chosen_xy[1])
                    if qpos.size > 2:
                        qpos[2:] += reset_rng.uniform(-0.03, 0.03, size=qpos.size - 2)
                    qvel[:] = 0.0
                    physics.data.qpos[:] = qpos
                    physics.data.qvel[:] = qvel
            except Exception:
                # Fallback: inject a few random actions to move away from immediate spawn.
                warmup_steps = int(reset_rng.integers(3, 8))
                for _ in range(warmup_steps):
                    warm_action = reset_rng.uniform(action_spec.minimum, action_spec.maximum)
                    reset_obs, _, warm_done, _ = env.step(warm_action)
                    if warm_done:
                        reset_obs = env.reset()
                        break
            return reset_obs
        
        # Reset environment
        obs = _reset_with_randomized_start()
        
        try:
            camera = physics.render(camera_id=0, height=480, width=640)
            if camera.dtype != np.uint8:
                camera = (camera * 255).astype(np.uint8)
            frames.append(camera)
        except Exception as e:
            print(f"Initial frame error: {e}")
            return None
        
        current_env = None
        env_transitions = 0
        
        while frame_count < max_frames:
            # Convert observation to Tonic format
            tonic_obs = self.env._process_observation(obs)
            
            # Get action from trained model
            with torch.no_grad():
                if hasattr(self.agent, 'test_step'):
                    # Use Tonic agent interface
                    action = self.agent.test_step(tonic_obs, steps=frame_count)
                    if torch.is_tensor(action):
                        action = action.cpu().numpy()
                else:
                    # Use simple agent interface
                    action = self.agent.act(tonic_obs)
            
            # Clip action to environment bounds
            action = np.clip(action, action_spec.minimum, action_spec.maximum)
            
            # Take step in mixed environment
            obs, reward, done, info = env.step(action)
            
            # Track environment changes
            new_env = env.env.task.get_current_environment(physics)
            if current_env != new_env and current_env is not None:
                env_transitions += 1
                print(f"Environment transition: {current_env} -> {new_env} at frame {frame_count}")
            current_env = new_env
            environment_history.append(current_env)
            
            # Track performance metrics
            current_head_pos = physics.named.data.xpos['head']
            current_velocity = physics.named.data.sensordata['head_vel']
            
            # Calculate metrics
            distance = np.linalg.norm(current_head_pos[:2] - initial_head_pos[:2])
            distances.append(distance)
            
            velocity_mag = np.linalg.norm(current_velocity[:2])
            velocities.append(velocity_mag)
            
            rewards_list.append(reward)
            
            # Capture frame
            try:
                camera = physics.render(camera_id=0, height=480, width=640)
                if camera.dtype != np.uint8:
                    camera = (camera * 255).astype(np.uint8)
                frames.append(camera)
            except Exception as e:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                frame[:, :, 0] = 255
                frames.append(frame)
            
            frame_count += 1
            if frame_count % 60 == 0:
                print(f"Captured {frame_count} frames, Environment: {current_env}, Transitions: {env_transitions}")
            
            if done:
                obs = _reset_with_randomized_start()
        
        # Calculate final metrics
        total_time = time.time() - start_time
        total_distance = distances[-1] if distances else 0.0
        avg_velocity = np.mean(velocities) if velocities else 0.0
        max_velocity = np.max(velocities) if velocities else 0.0
        avg_reward = np.mean(rewards_list) if rewards_list else 0.0
        
        print(f"\n=== TRAINED MODEL MIXED ENVIRONMENT PERFORMANCE ===")
        print(f"Total distance traveled: {total_distance:.4f}")
        print(f"Average velocity: {avg_velocity:.4f}")
        print(f"Maximum velocity: {max_velocity:.4f}")
        print(f"Average reward: {avg_reward:.4f}")
        print(f"Environment transitions: {env_transitions}")
        
        # Save video
        if frames:
            imageio.mimsave(video_filename, frames, fps=30, quality=8)
            print(f"Video saved as {video_filename}")
        
        # Create and save comprehensive visualization using existing utilities
        results = {
            'total_distance': total_distance,
            'avg_velocity': avg_velocity,
            'max_velocity': max_velocity,
            'avg_reward': avg_reward,
            'env_transitions': env_transitions,
            'environment_history': environment_history,
            'distances': distances,
            'velocities': velocities
        }
        
        # Store position history in task for visualization
        env.env.task.position_history = [initial_head_pos[:2]] + [physics.named.data.xpos['head'][:2] for _ in range(len(distances))]
        env.env.task.env_history = environment_history
        
        create_comprehensive_visualization(env.env.task, results, plot_filename)
        create_parameter_log(env.env.task, results, self.n_links, self.model_type, self.algorithm, log_filename)
        
        return results
    
    def save_model(self, filename=None):
        """Save the trained model."""
        if filename is None:
            filename = f"{self.model_type}_{self.algorithm}_{self.n_links}links.pth"
        
        save_path = os.path.join(self.output_dir, filename)
        
        if self.model is not None:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'model_type': self.model_type,
                'algorithm': self.algorithm,
                'n_links': self.n_links,
                'training_steps': self.training_steps
            }, save_path)
            print(f"Model saved to {save_path}")
        else:
            print("No model to save")
    
    def save_tonic_model(self, agent, filename=None):
        """Save Tonic-trained model."""
        if filename is None:
            filename = f"{self.model_type}_{self.algorithm}_{self.n_links}links_tonic"
        
        save_path = os.path.join(self.output_dir, filename)
        agent.save(save_path)
        print(f"Tonic model saved to {save_path}")
    
    def load_tonic_model(self, model_name):
        """Load a trained Tonic model."""
        # Accept absolute path OR already-resolved relative path
        if os.path.isabs(model_name) or os.path.exists(model_name):
            load_path = model_name
        else:
            load_path = os.path.join(self.output_dir, model_name)

        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Model file not found: {load_path}")
        
        print(f"Loading Tonic model from: {load_path}")
        
        # Create Tonic environment
        self.env = self.create_tonic_environment()
        n_joints = self.env.action_space.shape[0]
        
        # Create Tonic model
        if self.model_type == 'ncap':
            self.model = self.create_tonic_ncap_model(n_joints)
        else:
            self.model = self.create_mlp_model(n_joints)
        
        # Load model weights (Tonic saves as state dict directly)
        checkpoint = torch.load(load_path, map_location=self.device)
        
        # Load the full state dict (no more circular references)
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        
        # Create Tonic agent
        self.agent = self.create_tonic_agent(self.model)
        
        # Initialize agent
        self.agent.initialize(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            seed=42
        )
        
        print(f"Tonic model loaded from {load_path}")
        print(f"Model type: {self.model_type}")
        print(f"Algorithm: {self.algorithm}")
        print(f"Action space: {n_joints} joints")
    
    def load_model(self, filename):
        """Load a trained model."""
        # Handle both relative and absolute paths
        if os.path.isabs(filename):
            load_path = filename
        else:
            load_path = os.path.join(self.output_dir, filename)
        
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Model file not found: {load_path}")
        
        checkpoint = torch.load(load_path, map_location=self.device)
        
        # Create environment to get action space
        self.env = self.create_environment()
        action_spec = self.env.action_spec
        n_joints = action_spec.shape[0]
        
        # Create model
        self.model = self.create_model(n_joints)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        
        # Create agent
        self.agent = self.create_agent(self.model)
        
        print(f"Model loaded from {load_path}")
        print(f"Model type: {checkpoint['model_type']}")
        print(f"Algorithm: {checkpoint['algorithm']}")
        print(f"Training steps: {checkpoint['training_steps']}")
