#!/usr/bin/env python3
"""
Progressive Mixed Environment for Swimming and Crawling
Starts with simple swimming physics and gradually introduces land zones.
"""

import numpy as np
import collections
from dm_control import suite
from dm_control.suite import swimmer
from dm_control.rl import control
from dm_control.utils import rewards
import gym
from gym import spaces
from .physics_fix import apply_swimming_physics_fix

_SWIM_SPEED = 0.15  # Slightly higher target for more dynamic swimming

class ProgressiveSwimCrawl(swimmer.Swimmer):
    """Progressive task that starts with simple swimming and adds land zones over time."""
    
    def __init__(self, 
                 desired_speed=_SWIM_SPEED,
                 land_zones=None,
                 water_viscosity=0.001,
                 land_viscosity=1.0,
                 training_progress=0.0,  # 0.0 = pure swimming, 1.0 = full mixed
                 **kwargs):
        super().__init__(**kwargs)
        self._desired_speed = desired_speed
        self._land_zones = land_zones or []
        self._water_viscosity = water_viscosity
        self._land_viscosity = land_viscosity
        self._training_progress = training_progress
        
        # Progressive complexity based on training progress
        self._current_land_zones = self._get_progressive_land_zones()
        
    def _get_progressive_land_zones(self):
        """Get land zones based on training progress."""
        if self._training_progress < 0.3:
            # Phase 1: Pure swimming (0-30% of training)
            return []
        elif self._training_progress < 0.6:
            # Phase 2: Single small land zone (30-60% of training)
            return [{'center': [3.0, 0], 'radius': 0.8}]
        elif self._training_progress < 0.8:
            # Phase 3: Two land zones (60-80% of training)
            return [
                {'center': [-2.0, 0], 'radius': 0.8},
                {'center': [3.0, 0], 'radius': 0.8}
            ]
        else:
            # Phase 4: Full complexity (80-100% of training)
            return self._land_zones if self._land_zones else [
                {'center': [-2.5, 0], 'radius': 1.0},
                {'center': [2.5, 0], 'radius': 1.0}
            ]

    def initialize_episode(self, physics):
        super().initialize_episode(physics)
        # Hide target for cleaner visualization
        physics.named.model.mat_rgba['target', 'a'] = 1
        physics.named.model.mat_rgba['target_default', 'a'] = 1
        physics.named.model.mat_rgba['target_highlight', 'a'] = 1
        
        # Set progressive environment properties
        self._update_environment_physics(physics)

    def _update_environment_physics(self, physics):
        """Update physics based on current training progress."""
        # Progressive viscosity changes
        if self._training_progress < 0.3:
            # Pure water phase
            physics.model.opt.viscosity = self._water_viscosity
        else:
            # Mixed environment phase - set base viscosity
            physics.model.opt.viscosity = self._water_viscosity
        
        # Store current land zones for reward calculation
        self._current_land_zones = self._get_progressive_land_zones()

    def get_observation(self, physics):
        """Enhanced observation with environment information."""
        obs = collections.OrderedDict()
        obs['joints'] = physics.joints()
        obs['body_velocities'] = physics.body_velocities()
        
        # Add environment context based on training progress
        if self._training_progress >= 0.3:
            head_pos = physics.named.data.xpos['head'][:2]
            
            # Calculate environment state
            in_water = True
            in_land = False
            current_viscosity = self._water_viscosity
            
            for zone in self._current_land_zones:
                distance = np.linalg.norm(head_pos - zone['center'])
                if distance < zone['radius']:
                    in_water = False
                    in_land = True
                    current_viscosity = self._land_viscosity
                    break
            
            obs['fluid_viscosity'] = np.array([current_viscosity], dtype=np.float32)
            obs['environment_type'] = np.array([1.0 if in_water else 0.0, 1.0 if in_land else 0.0], dtype=np.float32)
            obs['in_water_zone'] = np.array([1.0 if in_water else 0.0], dtype=np.float32)
            obs['in_land_zone'] = np.array([1.0 if in_land else 0.0], dtype=np.float32)
        
        return obs

    def get_reward(self, physics):
        """Progressive reward that adapts to training phase."""
        head_pos = physics.named.data.xpos['head'][:2]
        forward_velocity = -physics.named.data.sensordata['head_vel'][1]
        
        if self._training_progress < 0.3:
            # Phase 1: Pure swimming reward
            return rewards.tolerance(
                forward_velocity,
                bounds=(self._desired_speed, float('inf')),
                margin=self._desired_speed,
                value_at_margin=0.,
                sigmoid='linear',
            )
        else:
            # Phase 2+: Mixed environment reward
            # Determine current environment
            in_land = False
            for zone in self._current_land_zones:
                distance = np.linalg.norm(head_pos - zone['center'])
                if distance < zone['radius']:
                    in_land = True
                    break
            
            if in_land:
                # Land reward: encourage crawling motion
                # Reward both forward movement and joint activity
                joint_activity = np.sum(np.abs(physics.joints()))
                movement_reward = rewards.tolerance(
                    forward_velocity,
                    bounds=(self._desired_speed * 0.3, float('inf')),  # Lower speed target on land
                    margin=self._desired_speed * 0.3,
                    value_at_margin=0.,
                    sigmoid='linear',
                )
                activity_reward = rewards.tolerance(
                    joint_activity,
                    bounds=(0.1, float('inf')),
                    margin=0.1,
                    value_at_margin=0.,
                    sigmoid='linear',
                ) * 0.3
                
                return movement_reward + activity_reward
            else:
                # Water reward: swimming
                return rewards.tolerance(
                    forward_velocity,
                    bounds=(self._desired_speed, float('inf')),
                    margin=self._desired_speed,
                    value_at_margin=0.,
                    sigmoid='linear',
                )

    def update_training_progress(self, progress):
        """Update training progress (0.0 to 1.0)."""
        self._training_progress = np.clip(progress, 0.0, 1.0)
        self._current_land_zones = self._get_progressive_land_zones()

@swimmer.SUITE.add()
def progressive_swim_crawl(
    n_links=6,
    desired_speed=_SWIM_SPEED,
    training_progress=0.0,
    time_limit=swimmer._DEFAULT_TIME_LIMIT,
    random=None,
    environment_kwargs={},
):
    """Progressive swim and crawl task."""
    model_string, assets = swimmer.get_model_and_assets(n_links)
    physics = swimmer.Physics.from_xml_string(model_string, assets=assets)
    
    # Note: Gear ratio fix disabled for curriculum training to maintain biological authenticity
    # The progressive curriculum phases provide natural scaffolding for learning
    # apply_swimming_physics_fix(physics, gear_ratio=0.1)
    
    task = ProgressiveSwimCrawl(
        desired_speed=desired_speed,
        training_progress=training_progress,
        random=random
    )
    
    return control.Environment(
        physics,
        task,
        time_limit=time_limit,
        control_timestep=swimmer._CONTROL_TIMESTEP,
        **environment_kwargs,
    )

class ProgressiveMixedSwimmerEnv:
    """Progressive wrapper that manages training curriculum."""
    
    def __init__(self, n_links=5, desired_speed=_SWIM_SPEED, time_limit=3000):
        self.n_links = n_links
        self.desired_speed = desired_speed
        self.time_limit = time_limit
        self.training_progress = 0.0
        self.total_episodes = 0
        self.target_episodes = 1000000  # 1M episodes for full curriculum
        self.manual_progress_override = False  # For testing purposes
        
        # Create initial environment
        self._create_environment()
        
    def _create_environment(self):
        """Create environment with current training progress."""
        self.env = suite.load(
            'swimmer', 
            'progressive_swim_crawl',
            task_kwargs={
                'random': 1, 
                'n_links': self.n_links,
                'training_progress': self.training_progress
            }
        )
        self.physics = self.env.physics
        self.action_spec = self.env.action_spec()
        self.observation_spec = self.env.observation_spec()
        
    def update_training_progress(self, episode_count):
        """Update training progress based on episode count."""
        old_progress = self.training_progress
        self.training_progress = min(1.0, episode_count / self.target_episodes)
        
        # Check if we need to recreate environment for new phase
        phase_old = int(old_progress * 4)  # 4 phases
        phase_new = int(self.training_progress * 4)
        
        if phase_new > phase_old:
            print(f"\n🎓 TRAINING PHASE CHANGE: {phase_old} → {phase_new}")
            print(f"   Progress: {old_progress:.2%} → {self.training_progress:.2%}")
            
            if phase_new == 1:
                print("   📈 Phase 1: Adding first land zone")
            elif phase_new == 2:
                print("   📈 Phase 2: Adding second land zone")
            elif phase_new == 3:
                print("   📈 Phase 3: Full complexity mixed environment")
            
            self._create_environment()
        
        # Update task progress
        if hasattr(self.env, '_task'):
            self.env._task.update_training_progress(self.training_progress)
    
    def set_manual_progress(self, progress):
        """Manually set training progress for testing purposes."""
        self.manual_progress_override = True
        old_progress = self.training_progress
        self.training_progress = np.clip(progress, 0.0, 1.0)
        
        # Check if we need to recreate environment for new phase
        phase_old = int(old_progress * 4)  # 4 phases
        phase_new = int(self.training_progress * 4)
        
        if phase_new != phase_old:
            print(f"🔧 Manual phase change: {phase_old} → {phase_new} (progress: {self.training_progress:.3f})")
            self._create_environment()
        
        # Update task progress
        if hasattr(self.env, '_task'):
            self.env._task.update_training_progress(self.training_progress)
            print(f"🔧 Task progress updated to: {self.training_progress:.3f}")
        
    def reset(self):
        """Reset environment and update training progress."""
        self.total_episodes += 1
        
        # Only auto-update progress if not in manual override mode
        if not self.manual_progress_override:
            self.update_training_progress(self.total_episodes)
        
        time_step = self.env.reset()
        return time_step.observation
    
    def step(self, action):
        """Take a step in the environment."""
        time_step = self.env.step(action)
        return time_step.observation, time_step.reward, time_step.last(), {}
    
    def render(self, mode='rgb_array', height=480, width=640):
        """Render the environment."""
        return self.physics.render(camera_id=0, height=height, width=width)
    
    def close(self):
        """Close the environment."""
        pass

class TonicProgressiveMixedWrapper(gym.Env):
    """
    Gym wrapper for progressive mixed environment compatible with Tonic.
    """
    
    def __init__(self, n_links=5, time_feature=True, desired_speed=_SWIM_SPEED):
        super().__init__()
        
        # Create the underlying environment
        self.env = ProgressiveMixedSwimmerEnv(n_links=n_links, desired_speed=desired_speed)
        
        # Get action space from environment
        action_spec = self.env.action_spec
        self.action_space = spaces.Box(
            low=action_spec.minimum,
            high=action_spec.maximum,
            dtype=np.float32
        )
        
        # Create observation space (adaptive to training progress)
        # Start with simple obs, expand as complexity increases
        base_obs_dim = (n_links - 1) + (n_links * 3)  # joints + body velocities
        max_env_features = 5  # viscosity + env_type(2) + zones(2)
        time_dim = 1 if time_feature else 0
        
        total_obs_dim = base_obs_dim + max_env_features + time_dim
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_obs_dim,),
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
        
        return gym_obs, reward, done, info
    
    def _process_observation(self, obs):
        """Process observation to gym format with padding for consistency."""
        if isinstance(obs, dict):
            # Extract basic components
            joint_pos = obs.get('joints', np.zeros(self.env.action_spec.shape[0]))
            body_vel = obs.get('body_velocities', np.zeros(len(joint_pos) * 3 + 3))
            
            # Extract environmental features (if present)
            env_features = []
            if 'fluid_viscosity' in obs:
                env_features.extend(obs['fluid_viscosity'])
            else:
                env_features.append(0.001)  # Default water viscosity
            
            if 'environment_type' in obs:
                env_features.extend(obs['environment_type'])
            else:
                env_features.extend([1.0, 0.0])  # Default: in water
                
            if 'in_water_zone' in obs and 'in_land_zone' in obs:
                env_features.extend([obs['in_water_zone'][0], obs['in_land_zone'][0]])
            else:
                env_features.extend([1.0, 0.0])  # Default: in water
        else:
            # Handle array format
            n_joints = self.env.action_spec.shape[0]
            joint_pos = obs[:n_joints]
            body_vel = obs[n_joints:n_joints + len(joint_pos) * 3 + 3]
            env_features = [0.001, 1.0, 0.0, 1.0, 0.0]  # Default values
        
        # Combine into single observation vector
        gym_obs = np.concatenate([joint_pos, body_vel, env_features])
        
        # Add time feature if requested
        if self.time_feature:
            time_feature = np.array([self.step_count / self.max_steps], dtype=np.float32)
            gym_obs = np.concatenate([gym_obs, time_feature])
        
        # Pad to expected size if needed
        expected_size = self.observation_space.shape[0]
        if len(gym_obs) < expected_size:
            padding = np.zeros(expected_size - len(gym_obs))
            gym_obs = np.concatenate([gym_obs, padding])
        elif len(gym_obs) > expected_size:
            gym_obs = gym_obs[:expected_size]
        
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
        progress = self.env.training_progress
        phase = int(progress * 4)
        return f"progressive-mixed-swimmer-{self.env.n_links}links-phase{phase}"

    @property 
    def training_progress(self):
        """Get current training progress."""
        return self.env.training_progress 