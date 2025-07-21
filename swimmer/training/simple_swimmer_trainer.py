#!/usr/bin/env python3
"""
Simple Swimmer Trainer
Trains NCAP models on basic forward swimming (matches notebook approach)
"""

import torch
import torch.nn as nn
import numpy as np
import os
import tonic
import tonic.torch
from ..environments.simple_swimmer import TonicSimpleSwimmerWrapper
from ..models.ncap_swimmer import NCAPSwimmer
from ..models.tonic_ncap import create_tonic_ncap_model
from ..utils.training_logger import TrainingLogger
from .custom_tonic_agent import CustomA2C


class SimpleSwimmerTrainer:
    """
    Simple trainer for basic swimming using notebook's approach.
    Focuses on forward velocity reward only.
    """
    
    def __init__(self, n_links=6, training_steps=100000, save_steps=20000,
                 output_dir='outputs/training', log_episodes=10):
        self.n_links = n_links
        self.training_steps = training_steps
        self.save_steps = save_steps
        self.output_dir = output_dir
        self.log_episodes = log_episodes
        
        # Training parameters optimized for simple swimming
        self.learning_rate = 3e-4  # Standard RL learning rate
        self.gradient_clip = 0.5   # Standard gradient clipping
        self.desired_speed = 0.1   # Match notebook target speed
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Check for GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize training logger
        experiment_name = f"simple_ncap_a2c_{self.n_links}links"
        self.logger = TrainingLogger(
            log_dir='outputs/training_logs',
            experiment_name=experiment_name
        )
        
        # Initialize components
        self.env = None
        self.model = None
        self.agent = None
        
    def create_simple_environment(self):
        """Create simple swimmer environment for basic training."""
        return TonicSimpleSwimmerWrapper(
            n_links=self.n_links, 
            time_feature=True,
            desired_speed=self.desired_speed
        )
    
    def create_simple_ncap_model(self, n_joints):
        """Create NCAP model optimized for simple swimming."""
        model = create_tonic_ncap_model(
            n_joints=n_joints, 
            oscillator_period=60, 
            memory_size=10,
            action_noise=0.1  # Standard noise for exploration
        )
        
        # Apply good initialization for simple swimming
        self._apply_simple_initialization(model.ncap)
        
        model.to(self.device)
        return model
    
    def _apply_simple_initialization(self, model):
        """Apply initialization optimized for simple swimming."""
        with torch.no_grad():
            # Initialize NCAP biological parameters with good defaults
            for name, param in model.params.items():
                if 'muscle' in name or 'bneuron' in name:
                    # Initialize with good biological values for swimming
                    nn.init.normal_(param, mean=1.0, std=0.2)  # Higher mean for good swimming
                    param.data.clamp_(0.3, 3.0)  # Good range for swimming
            
            # Initialize environment adaptation modules (less important for simple swimming)
            if hasattr(model, 'env_modulation'):
                nn.init.xavier_uniform_(model.env_modulation.weight, gain=0.1)
                nn.init.constant_(model.env_modulation.bias, 0.0)
            
            if hasattr(model, 'amplitude_modulation'):
                nn.init.xavier_uniform_(model.amplitude_modulation.weight, gain=0.1)
                nn.init.constant_(model.amplitude_modulation.bias, 0.0)
            
            if hasattr(model, 'memory_decoder'):
                nn.init.xavier_uniform_(model.memory_decoder.weight, gain=0.1)
                nn.init.constant_(model.memory_decoder.bias, 0.0)
            
            # Initialize LSTM
            if hasattr(model, 'memory_encoder'):
                for name, param in model.memory_encoder.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param, gain=0.5)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0.0)
    
    def create_simple_a2c_agent(self, model):
        """Create A2C agent optimized for simple swimming."""
        from tonic.torch import updaters
        
        # Create updaters with good hyperparameters for simple swimming
        actor_updater = updaters.StochasticPolicyGradient(
            optimizer=lambda params: torch.optim.Adam(params, lr=self.learning_rate),
            entropy_coeff=0.01,  # Small entropy for stable exploration
            gradient_clip=self.gradient_clip
        )
        
        critic_updater = updaters.VRegression(
            optimizer=lambda params: torch.optim.Adam(params, lr=self.learning_rate * 2),
            gradient_clip=self.gradient_clip
        )
        
        # Create replay buffer optimized for simple swimming
        from tonic import replays
        replay = replays.Segment(size=2048, batch_iterations=32)  # Standard settings
        
        return CustomA2C(
            model=model,
            replay=replay,
            actor_updater=actor_updater,
            critic_updater=critic_updater
        )
    
    def train(self):
        """Train the NCAP model on simple swimming."""
        print("=== STARTING SIMPLE SWIMMER TRAINING ===")
        print(f"Training for {self.training_steps} steps with simple forward velocity reward")
        print(f"Target speed: {self.desired_speed} m/s (matches notebook)")
        print(f"Learning rate: {self.learning_rate}")
        
        # Set up Tonic logger
        log_dir = os.path.join(
            'outputs', 'training_logs', 
            f'simple_ncap_a2c_{self.n_links}links'
        )
        tonic.logger.initialize(path=log_dir)
        
        # Create simple environment
        env = self.create_simple_environment()
        n_joints = env.action_space.shape[0]
        
        print(f"Environment: {env.name}")
        print(f"Observation space: {env.observation_space.shape}")
        print(f"Action space: {env.action_space.shape}")
        
        # Create NCAP model for simple swimming
        model = self.create_simple_ncap_model(n_joints)
        
        # Create A2C agent
        agent = self.create_simple_a2c_agent(model)
        
        # Initialize agent
        agent.initialize(
            observation_space=env.observation_space,
            action_space=env.action_space,
            seed=42
        )
        
        # Create Tonic trainer
        trainer = tonic.Trainer(
            steps=self.training_steps,
            save_steps=self.save_steps,
            test_episodes=5  # Regular testing
        )
        
        trainer.initialize(
            agent=agent,
            environment=env,
            test_environment=env
        )
        
        # Run training
        print(f"Starting training for {self.training_steps} steps...")
        trainer.run()
        
        # Save final model
        self.save_model(agent, f"simple_ncap_{self.n_links}links")
        
        # Store for evaluation
        self.agent = agent
        self.env = env
        self.model = model
        
        print("✅ Simple swimmer training completed!")
        
        return agent, env, model
    
    def save_model(self, agent, filename):
        """Save the trained model."""
        save_path = os.path.join(self.output_dir, f"{filename}.pt")
        
        # Save model state dict
        torch.save(agent.model.state_dict(), save_path)
        
        print(f"Model saved to: {save_path}")
        
        # Also save to Tonic format (for consistency)
        tonic_save_path = os.path.join(self.output_dir, filename)
        agent.save(tonic_save_path)
        
        print(f"Tonic model saved to: {tonic_save_path}")
    
    def evaluate_simple_swimming(self, agent=None, env=None, num_episodes=5):
        """Evaluate the trained model on simple swimming."""
        if agent is None:
            agent = self.agent
        if env is None:
            env = self.env
            
        if agent is None or env is None:
            raise ValueError("No trained model available. Train first.")
        
        print(f"\n=== EVALUATING SIMPLE SWIMMING PERFORMANCE ===")
        print(f"Running {num_episodes} episodes...")
        
        total_rewards = []
        total_distances = []
        total_velocities = []
        
        for episode in range(num_episodes):
            obs = env.reset()
            episode_reward = 0
            episode_length = 0
            max_steps = 1000
            
            initial_pos = None
            velocities = []
            
            while episode_length < max_steps:
                # Get action from agent
                action = agent.test_step(obs, steps=episode_length)
                if torch.is_tensor(action):
                    action = action.cpu().numpy()
                
                # Take step
                obs, infos = env.step(action)
                reward = infos['rewards'][0]
                done = infos['resets'][0]
                
                episode_reward += reward
                episode_length += 1
                
                # Track distance and velocity
                if hasattr(env.env, 'physics'):
                    current_pos = env.env.physics.named.data.xpos['head'][:2]
                    if initial_pos is None:
                        initial_pos = current_pos.copy()
                    
                    velocity = env.env.physics.named.data.sensordata['head_vel']
                    velocity_mag = np.linalg.norm(velocity[:2])
                    velocities.append(velocity_mag)
                
                if done:
                    break
            
            # Calculate episode metrics
            if hasattr(env.env, 'physics') and initial_pos is not None:
                final_pos = env.env.physics.named.data.xpos['head'][:2]
                episode_distance = np.linalg.norm(final_pos - initial_pos)
                avg_velocity = np.mean(velocities) if velocities else 0.0
            else:
                episode_distance = 0.0
                avg_velocity = 0.0
            
            total_rewards.append(episode_reward)
            total_distances.append(episode_distance)
            total_velocities.append(avg_velocity)
            
            print(f"  Episode {episode + 1}: Reward={episode_reward:.2f}, "
                  f"Distance={episode_distance:.3f}, Velocity={avg_velocity:.3f}")
        
        # Calculate final metrics
        avg_reward = np.mean(total_rewards)
        avg_distance = np.mean(total_distances)
        avg_velocity = np.mean(total_velocities)
        
        print(f"\n=== SIMPLE SWIMMING RESULTS ===")
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"Average Distance: {avg_distance:.3f}")
        print(f"Average Velocity: {avg_velocity:.3f}")
        print(f"Target Speed: {self.desired_speed} m/s")
        
        # Success criteria
        success = avg_velocity >= self.desired_speed * 0.5  # At least 50% of target speed
        print(f"Training Success: {'✅ YES' if success else '❌ NO'}")
        
        return {
            'avg_reward': avg_reward,
            'avg_distance': avg_distance,
            'avg_velocity': avg_velocity,
            'success': success,
            'rewards': total_rewards,
            'distances': total_distances,
            'velocities': total_velocities
        }
    
    def load_model(self, filename):
        """Load a trained simple swimmer model."""
        # Handle both relative and absolute paths
        if os.path.isabs(filename):
            load_path = filename
        else:
            load_path = os.path.join(self.output_dir, filename)
        
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Model file not found: {load_path}")
        
        print(f"Loading simple swimmer model from: {load_path}")
        
        # Create environment and model
        self.env = self.create_simple_environment()
        n_joints = self.env.action_space.shape[0]
        self.model = self.create_simple_ncap_model(n_joints)
        
        # Load model weights
        checkpoint = torch.load(load_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        
        # Create agent
        self.agent = self.create_simple_a2c_agent(self.model)
        self.agent.initialize(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            seed=42
        )
        
        print(f"Simple swimmer model loaded successfully")
        print(f"Action space: {n_joints} joints")


def test_simple_swimming_performance():
    """Test function to compare simple swimming performance."""
    print("=== TESTING SIMPLE SWIMMING PERFORMANCE ===")
    
    # Create trainer
    trainer = SimpleSwimmerTrainer(
        n_links=6,
        training_steps=50000,  # Shorter for testing
        save_steps=10000
    )
    
    # Train model
    agent, env, model = trainer.train()
    
    # Evaluate performance
    results = trainer.evaluate_simple_swimming(agent, env)
    
    return results


if __name__ == "__main__":
    test_simple_swimming_performance() 