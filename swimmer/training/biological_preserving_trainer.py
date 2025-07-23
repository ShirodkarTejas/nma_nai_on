#!/usr/bin/env python3
"""
Biological Preserving NCAP Trainer
Specialized trainer that maintains biological constraints during training.
"""

import torch
import torch.nn as nn
import numpy as np
import os
import tonic
import tonic.torch
from ..environments.mixed_environment import ImprovedMixedSwimmerEnv
from ..environments.tonic_wrapper import TonicSwimmerWrapper
from ..models.simple_ncap import SimpleNCAPSwimmer
from ..utils.training_logger import TrainingLogger
from .swimmer_trainer import SwimmerTrainer


class BiologicalPreservingTrainer(SwimmerTrainer):
    """
    Trainer that preserves biological NCAP parameters during training.
    Uses constraint methods to maintain oscillatory behavior.
    """
    
    def __init__(self, n_links=5, training_steps=50000, save_steps=10000,
                 output_dir='outputs/training', log_episodes=5):
        super().__init__(
            model_type='ncap',
            algorithm='ppo',  # Use PPO with clipped updates
            n_links=n_links,
            training_steps=training_steps,
            save_steps=save_steps,
            output_dir=output_dir,
            log_episodes=log_episodes
        )
        
        # Biological preservation parameters
        self.preserve_strength = 0.9  # How strongly to preserve biological params
        self.min_oscillator_strength = 0.5  # Minimum oscillator weight
        self.min_coupling_strength = 0.3   # Minimum proprioceptive coupling
        self.max_muscle_strength = 3.0     # Maximum muscle strength
        
        # Training parameters optimized for biological preservation
        self.bio_learning_rate = 1e-4      # Lower learning rate for stability
        self.bio_gradient_clip = 0.1       # Very conservative gradient clipping
        self.constraint_frequency = 10     # Apply constraints every N steps
        
        # Tracking
        self.biological_metrics = {
            'oscillator_strength': [],
            'coupling_strength': [],
            'muscle_balance': [],
            'parameter_drift': []
        }
        
    def create_biological_ncap_model(self, n_joints):
        """Create NCAP model optimized for biological preservation."""
        print("🧬 Creating Biologically Preserved NCAP Model...")
        
        # Use simplified NCAP with strong initialization
        model = SimpleNCAPSwimmer(
            n_joints=n_joints,
            oscillator_period=60,
            use_weight_sharing=True,
            use_weight_constraints=True,
            include_proprioception=True,
            include_head_oscillators=True
        )
        
        # Apply strong biological initialization
        self._apply_biological_initialization(model)
        
        return model
    
    def _apply_biological_initialization(self, model):
        """Apply strong biological initialization for preserved training."""
        with torch.no_grad():
            # Strong oscillator for reliable rhythm
            if 'bneuron_osc' in model.params:
                model.params['bneuron_osc'].data = torch.tensor(2.0)  # Strong oscillator
            
            # Strong proprioceptive coupling for wave propagation
            if 'bneuron_prop' in model.params:
                model.params['bneuron_prop'].data = torch.tensor(1.5)  # Strong coupling
            
            # Balanced muscle activation
            if 'muscle_ipsi' in model.params:
                model.params['muscle_ipsi'].data = torch.tensor(1.5)   # Strong excitation
            if 'muscle_contra' in model.params:
                model.params['muscle_contra'].data = torch.tensor(-1.5) # Strong inhibition
            
            print(f"✅ Applied strong biological initialization:")
            for name, param in model.params.items():
                print(f"  {name}: {param.item():.3f}")
    
    def create_biological_tonic_model(self, n_joints):
        """Create Tonic-compatible model with biological preservation."""
        from tonic.torch import models, normalizers
        
        # Create simplified NCAP actor
        ncap_swimmer = self.create_biological_ncap_model(n_joints)
        
        class BiologicalNCAPActor(nn.Module):
            def __init__(self, swimmer_module):
                super().__init__()
                self.swimmer = swimmer_module
                
            def forward(self, observations):
                # Extract joint positions
                joint_pos = observations[:, :self.swimmer.n_joints]
                
                # Get NCAP actions
                actions = self.swimmer(joint_pos)
                
                # Apply action noise for exploration
                if self.training:
                    actions = actions + 0.05 * torch.randn_like(actions)
                
                return torch.distributions.Normal(actions, 0.1)
            
            def initialize(self, observation_space, action_space):
                self.action_size = action_space.shape[0]
        
        # Create actor-critic model
        model = models.ActorCritic(
            actor=BiologicalNCAPActor(ncap_swimmer),
            critic=models.Critic(
                encoder=models.ObservationEncoder(),
                torso=models.MLP((32, 32), nn.Tanh),  # Smaller critic to avoid interference
                head=models.ValueHead(),
            ),
            observation_normalizer=normalizers.MeanStd(),
        )
        
        return model
    
    def create_biological_tonic_agent(self, model):
        """Create PPO agent with biological preservation."""
        from tonic.torch import updaters
        
        # Conservative actor updater
        actor_updater = updaters.ClippedRatio(
            optimizer=lambda params: torch.optim.Adam(params, lr=self.bio_learning_rate),
            ratio_clip=0.1,     # Very conservative clipping
            entropy_coeff=0.01, # Small entropy for stability
            gradient_clip=self.bio_gradient_clip
        )
        
        # Standard critic updater
        critic_updater = updaters.VRegression(
            optimizer=lambda params: torch.optim.Adam(params, lr=3e-4),
            gradient_clip=0.5
        )
        
        # Smaller replay buffer for more frequent updates
        from tonic import replays
        replay = replays.Segment(size=2048, batch_iterations=32)
        
        return tonic.torch.agents.PPO(
            model=model,
            replay=replay,
            actor_updater=actor_updater,
            critic_updater=critic_updater
        )
    
    def apply_biological_constraints(self, model):
        """Apply biological constraints to preserve swimming behavior."""
        if not hasattr(model, 'actor') or not hasattr(model.actor, 'swimmer'):
            return
        
        ncap = model.actor.swimmer
        if not hasattr(ncap, 'params'):
            return
        
        with torch.no_grad():
            # Preserve oscillator strength
            if 'bneuron_osc' in ncap.params:
                current_val = ncap.params['bneuron_osc'].item()
                if abs(current_val) < self.min_oscillator_strength:
                    # Restore minimum oscillator strength
                    ncap.params['bneuron_osc'].data = torch.tensor(
                        self.min_oscillator_strength if current_val >= 0 else -self.min_oscillator_strength
                    )
            
            # Preserve coupling strength
            if 'bneuron_prop' in ncap.params:
                current_val = ncap.params['bneuron_prop'].item()
                if current_val < self.min_coupling_strength:
                    ncap.params['bneuron_prop'].data = torch.tensor(self.min_coupling_strength)
            
            # Maintain muscle balance
            if 'muscle_ipsi' in ncap.params and 'muscle_contra' in ncap.params:
                ipsi_val = ncap.params['muscle_ipsi'].item()
                contra_val = ncap.params['muscle_contra'].item()
                
                # Ensure muscles don't become too weak or too strong
                ipsi_val = max(self.min_coupling_strength, min(ipsi_val, self.max_muscle_strength))
                contra_val = max(-self.max_muscle_strength, min(contra_val, -self.min_coupling_strength))
                
                ncap.params['muscle_ipsi'].data = torch.tensor(ipsi_val)
                ncap.params['muscle_contra'].data = torch.tensor(contra_val)
    
    def monitor_biological_health(self, model):
        """Monitor the health of biological parameters."""
        if not hasattr(model, 'actor') or not hasattr(model.actor, 'swimmer'):
            return
        
        ncap = model.actor.swimmer
        if not hasattr(ncap, 'params'):
            return
        
        # Calculate biological metrics
        oscillator_strength = abs(ncap.params.get('bneuron_osc', torch.tensor(0)).item())
        coupling_strength = ncap.params.get('bneuron_prop', torch.tensor(0)).item()
        
        ipsi_strength = ncap.params.get('muscle_ipsi', torch.tensor(0)).item()
        contra_strength = abs(ncap.params.get('muscle_contra', torch.tensor(0)).item())
        muscle_balance = abs(ipsi_strength - contra_strength)
        
        # Store metrics
        self.biological_metrics['oscillator_strength'].append(oscillator_strength)
        self.biological_metrics['coupling_strength'].append(coupling_strength)
        self.biological_metrics['muscle_balance'].append(muscle_balance)
        
        # Check for biological health
        healthy = (
            oscillator_strength > self.min_oscillator_strength and
            coupling_strength > self.min_coupling_strength and
            muscle_balance < 1.0  # Muscles shouldn't be too imbalanced
        )
        
        if not healthy:
            print(f"⚠️  Biological health warning:")
            print(f"   Oscillator: {oscillator_strength:.3f} (min: {self.min_oscillator_strength})")
            print(f"   Coupling: {coupling_strength:.3f} (min: {self.min_coupling_strength})")
            print(f"   Muscle balance: {muscle_balance:.3f} (max: 1.0)")
        
        return healthy
    
    def train(self):
        """Train with biological preservation."""
        print("🧬 Starting Biological Preserving NCAP Training...")
        
        # Create environment
        env = ImprovedMixedSwimmerEnv(n_links=self.n_links)
        tonic_env = TonicSwimmerWrapper(env)
        
        # Create model and agent
        model = self.create_biological_tonic_model(self.n_links)
        agent = self.create_biological_tonic_agent(model)
        
        # Initialize agent
        agent.initialize(
            observation_space=tonic_env.observation_space,
            action_space=tonic_env.action_space
        )
        
        print(f"🏊 Training for {self.training_steps} steps with biological preservation...")
        
        # Training loop with biological monitoring
        step = 0
        episode = 0
        
        while step < self.training_steps:
            # Run episode
            obs = tonic_env.reset()
            episode_reward = 0
            episode_steps = 0
            
            while episode_steps < 3000:  # Max episode length
                # Get action from agent
                action = agent.step(obs)
                obs, reward, done, info = tonic_env.step(action)
                
                episode_reward += reward
                step += 1
                episode_steps += 1
                
                # Apply biological constraints periodically
                if step % self.constraint_frequency == 0:
                    self.apply_biological_constraints(model)
                
                # Monitor biological health
                if step % (self.constraint_frequency * 10) == 0:
                    self.monitor_biological_health(model)
                
                if done or step >= self.training_steps:
                    break
            
            episode += 1
            
            # Logging
            if episode % self.log_episodes == 0:
                print(f"Episode {episode}, Step {step}: Reward = {episode_reward:.3f}")
                
                # Check biological health
                healthy = self.monitor_biological_health(model)
                if healthy:
                    print("  ✅ Biological parameters healthy")
                
            # Save model
            if step % self.save_steps == 0:
                save_path = os.path.join(self.output_dir, f'biological_ncap_{self.n_links}links_step{step}.pt')
                torch.save(model.state_dict(), save_path)
                print(f"💾 Saved model to {save_path}")
        
        print("🎯 Training completed with biological preservation!")
        
        # Final evaluation
        self.evaluate_biological_model(model, env)
        
        return model
    
    def evaluate_biological_model(self, model, env):
        """Evaluate the biologically preserved model."""
        print("🧪 Evaluating biological preservation...")
        
        # Test oscillatory behavior
        obs = env.reset()
        actions = []
        
        for step in range(120):  # 2 seconds
            with torch.no_grad():
                # Get observation tensor
                obs_tensor = torch.tensor(obs['joints'], dtype=torch.float32).unsqueeze(0)
                action_dist = model.actor(obs_tensor)
                action = action_dist.mean.squeeze(0).cpu().numpy()
                actions.append(action.copy())
            
            obs, reward, done, info = env.step(action)
        
        actions = np.array(actions)
        
        # Analyze oscillatory strength
        action_ranges = actions.max(axis=0) - actions.min(axis=0)
        avg_range = action_ranges.mean()
        
        print(f"📊 Post-training oscillatory analysis:")
        print(f"  Average action range: {avg_range:.3f}")
        print(f"  Action ranges per joint: {action_ranges}")
        
        if avg_range > 0.5:
            print("✅ Strong oscillatory behavior preserved!")
        else:
            print("❌ Oscillatory behavior weakened during training")
        
        env.close()


def create_biological_preserving_trainer(**kwargs):
    """Factory function for the biological preserving trainer."""
    return BiologicalPreservingTrainer(**kwargs) 