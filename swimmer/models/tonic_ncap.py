#!/usr/bin/env python3
"""
Tonic-compatible NCAP Model
NCAP model that works with Tonic's PPO agent.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .ncap_swimmer import NCAPSwimmer

class SwimmerActor(nn.Module):
    """Actor component for Tonic compatibility."""
    
    def __init__(self, swimmer_module, distribution=None):
        super().__init__()
        self.swimmer = swimmer_module
        self.distribution = distribution or (lambda x: torch.distributions.Normal(x, 0.1))
        
    def forward(self, observations):
        """Forward pass through the actor."""
        # Extract joint positions from observations
        if observations.dim() == 1:
            observations = observations.unsqueeze(0)  # Add batch dimension
        
        joint_positions = observations[:, :self.swimmer.n_joints]
        
        # Get NCAP output
        ncap_output = self.swimmer(joint_positions)
        
        # Return the distribution
        return self.distribution(ncap_output)

class SwimmerCritic(nn.Module):
    """Critic component for Tonic compatibility."""
    
    def __init__(self, n_joints, critic_sizes=(64, 64), critic_activation=nn.Tanh):
        super().__init__()
        self.n_joints = n_joints
        
        # Build critic network
        layers = []
        input_size = n_joints
        
        for size in critic_sizes:
            layers.extend([
                nn.Linear(input_size, size),
                critic_activation()
            ])
            input_size = size
        
        layers.append(nn.Linear(input_size, 1))
        
        self.critic_network = nn.Sequential(*layers)
        
    def forward(self, observations):
        """Forward pass through the critic."""
        # Extract joint positions from observations
        if observations.dim() == 1:
            observations = observations.unsqueeze(0)  # Add batch dimension
        
        joint_positions = observations[:, :self.n_joints]
        
        # Get value estimate
        value = self.critic_network(joint_positions)
        
        return value

class TonicNCAPModel(nn.Module):
    """
    Tonic-compatible NCAP model that provides both policy and value functions.
    """
    
    def __init__(self, n_joints, oscillator_period=60, memory_size=10, 
                 critic_sizes=(64, 64), critic_activation=nn.Tanh, action_noise=0.1):
        super().__init__()
        
        self.n_joints = n_joints
        self.oscillator_period = oscillator_period
        
        # Create the NCAP swimmer model
        self.ncap = NCAPSwimmer(
            n_joints=n_joints,
            oscillator_period=oscillator_period,
            memory_size=memory_size
        )
        
        # Create actor and critic components
        self.actor = SwimmerActor(
            swimmer_module=self.ncap,
            distribution=lambda x: torch.distributions.Normal(x, action_noise)
        )
        
        self.critic = SwimmerCritic(
            n_joints=n_joints,
            critic_sizes=critic_sizes,
            critic_activation=critic_activation
        )
        
        # Tonic expects these normalizers (set to None for now)
        self.observation_normalizer = None
        self.return_normalizer = None
        
        # Initialize weights
        self._init_weights()
        
        # Move to device immediately after all components are created
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ncap = self.ncap.to(device)
        self.actor = self.actor.to(device)
        self.critic = self.critic.to(device)
        print(f"TonicNCAPModel moved to device: {device}")
    
    def _init_weights(self):
        """Initialize network weights."""
        for module in [self.critic.critic_network]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.constant_(layer.bias, 0.0)
    
    def forward(self, observations):
        """
        Forward pass through the model.
        
        Args:
            observations: Tensor of shape (batch_size, obs_dim)
            
        Returns:
            tuple: (policy_output, value_output)
        """
        # Ensure observations are on the same device as the model
        device = next(self.parameters()).device
        observations = observations.to(device)
        
        # Get actor and critic outputs
        policy_output = self.actor(observations)
        value_output = self.critic(observations)
        
        return policy_output, value_output
    
    def get_policy(self, observations):
        """Get policy output only."""
        return self.actor(observations)
    
    def get_value(self, observations):
        """Get value output only."""
        return self.critic(observations)
    
    def initialize(self, observation_space, action_space):
        """
        Initialize the model with observation and action spaces.
        Required by Tonic agents.
        
        Args:
            observation_space: Gym observation space
            action_space: Gym action space
        """
        # Store spaces for reference
        self.observation_space = observation_space
        self.action_space = action_space
        
        # Verify dimensions match
        obs_dim = observation_space.shape[0]
        action_dim = action_space.shape[0]
        
        print(f"Initializing TonicNCAPModel with obs_dim={obs_dim}, action_dim={action_dim}")
        
        # Ensure our model dimensions are compatible
        if obs_dim < self.n_joints:
            raise ValueError(f"Observation space too small: {obs_dim} < {self.n_joints}")
        
        if action_dim != self.n_joints:
            raise ValueError(f"Action space mismatch: {action_dim} != {self.n_joints}")
        
        print("TonicNCAPModel initialized successfully")

    def to(self, device):
        """Override to ensure all components are moved to the same device."""
        super().to(device)
        self.ncap = self.ncap.to(device)
        self.actor = self.actor.to(device)
        self.critic = self.critic.to(device)
        return self

def create_tonic_ncap_model(n_joints=6, oscillator_period=60, memory_size=10,
                           critic_sizes=(64, 64), critic_activation=nn.Tanh, action_noise=0.1):
    """
    Factory function to create a Tonic-compatible NCAP model.
    
    Args:
        n_joints: Number of joints in the swimmer
        oscillator_period: Period of the oscillator
        memory_size: Size of the memory buffer
        critic_sizes: Sizes of critic network layers
        critic_activation: Activation function for critic
        action_noise: Standard deviation for action distribution
        
    Returns:
        TonicNCAPModel: The created model
    """
    return TonicNCAPModel(
        n_joints=n_joints,
        oscillator_period=oscillator_period,
        memory_size=memory_size,
        critic_sizes=critic_sizes,
        critic_activation=critic_activation,
        action_noise=action_noise
    ) 