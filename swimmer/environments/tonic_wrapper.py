#!/usr/bin/env python3
"""
Tonic Environment Wrapper
Makes our mixed environment swimmer compatible with Tonic training framework.
"""

import gym
import numpy as np
from gym import spaces
from .mixed_environment import ImprovedMixedSwimmerEnv

class TonicSwimmerWrapper(gym.Env):
    """
    Gym wrapper for our mixed environment swimmer to make it compatible with Tonic.
    """
    
    def __init__(self, n_links=6, time_feature=True):
        super().__init__()
        
        # Create the underlying environment
        self.env = ImprovedMixedSwimmerEnv(n_links=n_links)
        
        # Get action space from environment
        action_spec = self.env.action_spec
        self.action_space = spaces.Box(
            low=action_spec.minimum,
            high=action_spec.maximum,
            dtype=np.float32
        )
        
        # Create observation space
        # We'll use joint positions and velocities
        obs_dim = n_links * 2  # positions + velocities
        if time_feature:
            obs_dim += 1  # Add time feature
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        self.time_feature = time_feature
        self.step_count = 0
        self.max_steps = 1000  # Maximum episode length
        
    def reset(self):
        """Reset the environment."""
        obs = self.env.reset()
        self.step_count = 0
        
        # Convert observation to gym format
        gym_obs = self._process_observation(obs)
        
        return gym_obs
    
    def start(self):
        """Start the environment (Tonic compatibility)."""
        obs = self.reset()
        # Tonic expects a list of observations for multiple workers
        return [obs]
    
    def step(self, action):
        """Take a step in the environment."""
        # Take step in underlying environment
        obs, reward, done, info = self.env.step(action)
        
        # Update step count
        self.step_count += 1
        
        # Check if episode should end
        if self.step_count >= self.max_steps:
            done = True
        
        # Convert observation to gym format
        gym_obs = self._process_observation(obs)
        
        # Add additional info
        info['step_count'] = self.step_count
        info['current_environment'] = self.env.env.task.get_current_environment(self.env.physics)
        
        # Convert to Tonic format: (observations, infos)
        # Tonic expects infos to be a dictionary with 'rewards', 'resets', 'terminations', 'observations'
        infos = {
            'observations': [gym_obs],  # List of observations for each worker
            'rewards': np.array([reward], dtype=np.float32),
            'resets': np.array([done], dtype=bool),
            'terminations': np.array([done], dtype=bool)
        }
        
        return gym_obs, infos
    
    def _process_observation(self, obs):
        """Process observation to gym format."""
        if isinstance(obs, dict):
            # Extract joint positions and velocities
            joint_pos = obs.get('joints', np.zeros(self.env.action_spec.shape[0]))
            joint_vel = obs.get('joint_velocities', np.zeros(self.env.action_spec.shape[0]))
        else:
            # Assume obs is joint positions
            joint_pos = obs
            joint_vel = np.zeros_like(obs)
        
        # Combine into single observation vector
        gym_obs = np.concatenate([joint_pos, joint_vel])
        
        # Add time feature if requested
        if self.time_feature:
            time_feature = np.array([self.step_count / self.max_steps], dtype=np.float32)
            gym_obs = np.concatenate([gym_obs, time_feature])
        
        return gym_obs.astype(np.float32)
    
    def render(self, mode='rgb_array'):
        """Render the environment."""
        if hasattr(self.env, 'render'):
            return self.env.render(mode)
        return None
    
    def close(self):
        """Close the environment."""
        if hasattr(self.env, 'close'):
            self.env.close()
    
    @property
    def name(self):
        """Environment name for Tonic."""
        return f"mixed-swimmer-{self.env.action_spec.shape[0]}links" 