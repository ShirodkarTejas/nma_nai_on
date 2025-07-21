#!/usr/bin/env python3
"""
Simple Swimmer Environment for Basic Training
Matches the notebook's simple swimmer environment with forward velocity reward.
"""

import numpy as np
import collections
import gym
from gym import spaces
from dm_control import suite
from dm_control.suite import swimmer
from dm_control.rl import control
from dm_control.utils import rewards
from dm_control.mujoco.wrapper import mjbindings


_SWIM_SPEED = 0.1  # Match notebook's target speed


class SimpleSwim(swimmer.Swimmer):
    """Simple swim task that only rewards forward movement - matches notebook."""
    
    def __init__(self, desired_speed=_SWIM_SPEED, **kwargs):
        super().__init__(**kwargs)
        self._desired_speed = desired_speed

    def initialize_episode(self, physics):
        super().initialize_episode(physics)
        # Hide target by setting alpha to 0 (like notebook)
        physics.named.model.mat_rgba['target', 'a'] = 0
        physics.named.model.mat_rgba['target_default', 'a'] = 0
        physics.named.model.mat_rgba['target_highlight', 'a'] = 0

    def get_observation(self, physics):
        """Returns observation of joint angles and body velocities - matches notebook."""
        obs = collections.OrderedDict()
        obs['joints'] = physics.joints()
        obs['body_velocities'] = physics.body_velocities()
        return obs

    def get_reward(self, physics):
        """Simple forward velocity reward - exactly matches notebook."""
        # Note: negative sign for forward direction (matches notebook)
        forward_velocity = -physics.named.data.sensordata['head_vel'][1]
        return rewards.tolerance(
            forward_velocity,
            bounds=(self._desired_speed, float('inf')),
            margin=self._desired_speed,
            value_at_margin=0.,
            sigmoid='linear',
        )


# Register simple swim task
if not hasattr(swimmer, 'simple_swim_registered'):
    @swimmer.SUITE.add()
    def simple_swim(
        n_links=6,
        desired_speed=_SWIM_SPEED,
        time_limit=swimmer._DEFAULT_TIME_LIMIT,
        random=None,
        environment_kwargs={},
    ):
        """Returns the simple swim task - matches notebook exactly."""
        model_string, assets = swimmer.get_model_and_assets(n_links)
        physics = swimmer.Physics.from_xml_string(model_string, assets=assets)
        task = SimpleSwim(desired_speed=desired_speed, random=random)
        return control.Environment(
            physics,
            task,
            time_limit=time_limit,
            control_timestep=swimmer._CONTROL_TIMESTEP,
            **environment_kwargs,
        )
    
    swimmer.simple_swim_registered = True


class SimpleSwimmerEnv:
    """Simple swimmer environment wrapper for basic training."""
    
    def __init__(self, n_links=6, desired_speed=_SWIM_SPEED):
        self.env = suite.load('swimmer', 'simple_swim', 
                             task_kwargs={'random': 1, 'n_links': n_links, 'desired_speed': desired_speed})
        self.physics = self.env.physics
        self.action_spec = self.env.action_spec()
        self.observation_spec = self.env.observation_spec()
        self.done = False
        
    def reset(self):
        """Reset the environment."""
        time_step = self.env.reset()
        self.done = False
        return time_step.observation
    
    def step(self, action):
        """Take a step in the environment."""
        time_step = self.env.step(action)
        self.done = time_step.last()
        return time_step.observation, time_step.reward, self.done, {}
    
    def render(self, mode='rgb_array', height=480, width=640):
        """Render the environment."""
        return self.physics.render(camera_id=0, height=height, width=width)
    
    def close(self):
        """Close the environment."""
        pass


class TonicSimpleSwimmerWrapper(gym.Env):
    """
    Tonic wrapper for simple swimmer environment.
    Uses simple forward velocity reward like the notebook.
    """
    
    def __init__(self, n_links=6, time_feature=True, desired_speed=_SWIM_SPEED):
        super().__init__()
        
        # Create the simple swimmer environment
        self.env = SimpleSwimmerEnv(n_links=n_links, desired_speed=desired_speed)
        
        # Get action space from environment
        action_spec = self.env.action_spec
        self.action_space = spaces.Box(
            low=action_spec.minimum,
            high=action_spec.maximum,
            dtype=np.float32
        )
        
        # Create observation space - match notebook format
        obs_dim = n_links * 2  # joint positions + joint velocities
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
        
        # Convert to Tonic format: (observations, infos)
        infos = {
            'observations': [gym_obs],
            'rewards': np.array([reward], dtype=np.float32),
            'resets': np.array([done], dtype=bool),
            'terminations': np.array([done], dtype=bool)
        }
        
        return gym_obs, infos
    
    def _process_observation(self, obs):
        """Process observation to gym format - match notebook."""
        if isinstance(obs, dict):
            # Extract joint positions and body velocities (matches notebook)
            joint_pos = obs.get('joints', np.zeros(self.env.action_spec.shape[0]))
            body_vel = obs.get('body_velocities', np.zeros(self.env.action_spec.shape[0]))
        else:
            # Fallback
            joint_pos = obs
            body_vel = np.zeros_like(obs)
        
        # Combine into single observation vector (matches notebook format)
        gym_obs = np.concatenate([joint_pos, body_vel])
        
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
        return f"simple-swimmer-{self.env.action_spec.shape[0]}links" 