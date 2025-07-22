#!/usr/bin/env python3
"""
Swimmer Trainer Implementation
Contains trainer classes for RL training of swimmer models.
"""

import torch
import numpy as np
import os
import tonic
import tonic.torch
from ..environments.mixed_environment import ImprovedMixedSwimmerEnv
from ..environments.tonic_wrapper import TonicSwimmerWrapper
from ..models.ncap_swimmer import NCAPSwimmer
from ..models.tonic_ncap import create_tonic_ncap_model
from ..utils.training_logger import TrainingLogger

class SwimmerTrainer:
    """
    Trainer class for swimmer models using RL algorithms.
    Supports NCAP and MLP models with PPO and other algorithms.
    """
    def __init__(self, model_type='ncap', algorithm='ppo', n_links=6, 
                 training_steps=500000, save_steps=100000, 
                 output_dir='outputs/training', log_episodes=10, action_scale=1.0):
        self.model_type = model_type.lower()
        self.algorithm = algorithm.lower()
        self.n_links = n_links
        self.training_steps = training_steps
        self.save_steps = save_steps
        self.output_dir = output_dir
        self.log_episodes = log_episodes

        # Store the action scaling factor so that environments and wrappers
        # that depend on it (e.g.
        # `TonicSwimmerWrapper` in `create_tonic_environment`) can access it.
        self.action_scale = action_scale
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
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
            log_dir='outputs/training_logs',
            experiment_name=experiment_name
        )
        
    def create_environment(self):
        """Create the training environment."""
        return ImprovedMixedSwimmerEnv(n_links=self.n_links)
    
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
        model = create_tonic_ncap_model(n_joints=n_joints, oscillator_period=60, memory_size=10)
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
    
    def create_ppo_agent(self, model):
        """Create PPO agent with the given model."""
        # This would use Tonic's PPO implementation
        # For now, we'll create a simple wrapper
        return PPOAgent(model)
    
    def create_tonic_ppo_agent(self, model):
        """Create Tonic-compatible agent for evaluation.

        Training currently uses A2C via CustomA2C; for evaluation we can
        reuse the same lightweight wrapper instead of the removed CustomPPO.
        """
        from .custom_tonic_agent import CustomA2C
        return CustomA2C(model=model)
    
    def create_agent(self, model):
        """Create agent based on algorithm."""
        if self.algorithm == 'ppo':
            return self.create_ppo_agent(model)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
    
    def train_with_tonic(self):
        """Train using Tonic framework."""
        from datetime import datetime
        start_ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{start_ts}] ▶ Starting Tonic training with {self.model_type.upper()} model and {self.algorithm.upper()} algorithm")
        
        # Set up Tonic logger with proper directory
        import tonic
        log_dir = os.path.join('outputs', 'training_logs', f'{self.model_type}_{self.algorithm}_{self.n_links}links_tonic')
        tonic.logger.initialize(path=log_dir)
        
        # Log training configuration
        config = {
            'model_type': self.model_type,
            'algorithm': self.algorithm,
            'n_links': self.n_links,
            'training_steps': self.training_steps,
            'save_steps': self.save_steps,
            'device': str(self.device),
            'log_episodes': self.log_episodes,
            'framework': 'tonic'
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
        agent = self.create_tonic_ppo_agent(model)
        
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
    
    def evaluate_mixed_environment(self, max_frames=1800, speed_factor=1.0):
        """Evaluate the trained model in the mixed environment using existing infrastructure."""
        if self.model is None:
            raise ValueError("No model loaded. Train or load a model first.")
        
        print(f"Evaluating model in mixed environment for {max_frames} frames...")
        
        # Import the existing test function and modify it to use our trained model
        from ..environments.mixed_environment import ImprovedMixedSwimmerEnv
        from ..utils.visualization import create_comprehensive_visualization, create_parameter_log
        import imageio
        import time
        
        # Create mixed environment
        env = ImprovedMixedSwimmerEnv(n_links=self.n_links, speed_factor=speed_factor)
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
        os.makedirs("outputs/improved_mixed_env", exist_ok=True)
        video_filename = f"outputs/improved_mixed_env/trained_model_evaluation_{self.n_links}links.mp4"
        plot_filename = f"outputs/improved_mixed_env/trained_model_analysis_{self.n_links}links.png"
        log_filename = f"outputs/improved_mixed_env/trained_model_log_{self.n_links}links.txt"
        
        frame_count = 0
        frames = []
        
        # Reset environment
        obs = env.reset()
        
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
                obs = env.reset()
        
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
        self.agent = self.create_tonic_ppo_agent(self.model)
        
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
