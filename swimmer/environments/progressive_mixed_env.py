#!/usr/bin/env python3
"""
Progressive Mixed Environment for Swimming and Crawling
Starts with simple swimming physics and gradually introduces land zones.
Enhanced with goal-directed navigation targets.
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
        self._current_targets = self._get_progressive_targets()
        
        # Goal tracking
        self._current_target_index = 0
        self._targets_reached = 0
        self._target_radius = 2.0  # Distance to consider target "reached" (larger for untrained swimmers)
        self._target_visit_timer = 0  # Auto-advance targets if swimmer gets stuck
        
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
    
    def _get_progressive_targets(self):
        """Get navigation targets based on training progress."""
        if self._training_progress < 0.3:
            # Phase 1: Simple forward movement targets
            return [
                {'position': [2.0, 0], 'type': 'swim'},
                {'position': [4.0, 0], 'type': 'swim'},
                {'position': [6.0, 0], 'type': 'swim'}
            ]
        elif self._training_progress < 0.6:
            # Phase 2: Navigate to and around single land zone
            return [
                {'position': [1.5, 0], 'type': 'swim'},    # Approach land zone
                {'position': [3.0, 0], 'type': 'land'},    # Enter land zone
                {'position': [4.5, 0], 'type': 'swim'}     # Exit land zone
            ]
        elif self._training_progress < 0.8:
            # Phase 3: Navigate between two land zones
            return [
                {'position': [-2.0, 0], 'type': 'land'},   # First land zone
                {'position': [0.0, 0], 'type': 'swim'},    # Middle water
                {'position': [3.0, 0], 'type': 'land'},    # Second land zone
                {'position': [5.0, 0], 'type': 'swim'}     # Beyond zones
            ]
        else:
            # Phase 4: Complex navigation pattern
            return [
                {'position': [-2.5, 0], 'type': 'land'},   # Left land
                {'position': [0.0, 1.0], 'type': 'swim'},  # North water
                {'position': [2.5, 0], 'type': 'land'},    # Right land
                {'position': [0.0, -1.0], 'type': 'swim'}, # South water
                {'position': [4.0, 0], 'type': 'swim'}     # Final target
            ]

    def initialize_episode(self, physics):
        super().initialize_episode(physics)
        # Hide target for cleaner visualization
        physics.named.model.mat_rgba['target', 'a'] = 1
        physics.named.model.mat_rgba['target_default', 'a'] = 1
        physics.named.model.mat_rgba['target_highlight', 'a'] = 1
        
        # Reset goal tracking
        self._current_target_index = 0
        self._targets_reached = 0
        self._target_visit_timer = 0
        
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
        self._current_targets = self._get_progressive_targets()

    def get_observation(self, physics):
        """Enhanced observation with environment and goal information."""
        obs = collections.OrderedDict()
        obs['joints'] = physics.joints()
        obs['body_velocities'] = physics.body_velocities()
        
        # Current position
        head_pos = physics.named.data.xpos['head'][:2]
        
        # Add environment context based on training progress
        if self._training_progress >= 0.3:
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
        
        # Add goal navigation information
        if self._current_targets and self._current_target_index < len(self._current_targets):
            current_target = self._current_targets[self._current_target_index]
            target_pos = current_target['position']
            
            # Distance and direction to current target
            target_vector = np.array(target_pos) - head_pos
            distance_to_target = np.linalg.norm(target_vector)
            direction_to_target = target_vector / (distance_to_target + 1e-6)  # Normalize with small epsilon
            
            obs['target_distance'] = np.array([distance_to_target], dtype=np.float32)
            obs['target_direction'] = direction_to_target.astype(np.float32)
            obs['target_position'] = np.array(target_pos, dtype=np.float32)
            obs['target_type'] = np.array([1.0 if current_target['type'] == 'swim' else 0.0], dtype=np.float32)
            obs['targets_completed'] = np.array([self._targets_reached], dtype=np.float32)
        else:
            # No target or all targets completed
            obs['target_distance'] = np.array([0.0], dtype=np.float32)
            obs['target_direction'] = np.array([0.0, 0.0], dtype=np.float32)
            obs['target_position'] = np.array([0.0, 0.0], dtype=np.float32)
            obs['target_type'] = np.array([1.0], dtype=np.float32)  # Default to swim type
            obs['targets_completed'] = np.array([len(self._current_targets)], dtype=np.float32)
        
        return obs

    def get_reward(self, physics):
        """Enhanced reward with goal-directed navigation."""
        head_pos = physics.named.data.xpos['head'][:2]
        forward_velocity = -physics.named.data.sensordata['head_vel'][1]
        
        # Base movement reward
        if self._training_progress < 0.3:
            # Phase 1: Pure swimming reward
            base_reward = rewards.tolerance(
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
                
                base_reward = movement_reward + activity_reward
            else:
                # Water reward: swimming
                base_reward = rewards.tolerance(
                    forward_velocity,
                    bounds=(self._desired_speed, float('inf')),
                    margin=self._desired_speed,
                    value_at_margin=0.,
                    sigmoid='linear',
                )
        
        # Goal-directed navigation reward
        navigation_reward = 0.0
        
        if self._current_targets and self._current_target_index < len(self._current_targets):
            current_target = self._current_targets[self._current_target_index]
            target_pos = current_target['position']
            
            # Distance to target
            distance_to_target = np.linalg.norm(head_pos - target_pos)
            
            # Strong reward for approaching target (inverse distance)
            approach_reward = 1.0 / (1.0 + distance_to_target)
            navigation_reward += approach_reward * 0.5
            
            # Increment visit timer for auto-advance
            self._target_visit_timer += 1
            
            # Check for target completion (either reached or time limit)
            target_reached = distance_to_target < self._target_radius
            time_limit_reached = self._target_visit_timer > 300  # Auto-advance after 300 steps (~10 seconds)
            
            if target_reached or time_limit_reached:
                if target_reached:
                    navigation_reward += 2.0  # Large bonus for actually reaching target
                    # Only log occasionally to reduce spam
                    if self._targets_reached % 10 == 0:  # Log every 10th target reached
                        print(f"🎯 Target {self._targets_reached + 1} reached! Distance: {distance_to_target:.2f}m")
                else:
                    navigation_reward += 0.5  # Smaller bonus for time-based advance
                    # Only log occasionally to reduce spam
                    if self._targets_reached % 20 == 0:  # Log every 20th auto-advance
                        print(f"⏰ Target {self._targets_reached + 1} auto-advanced after {self._target_visit_timer} steps")
                
                # Move to next target
                self._current_target_index += 1
                self._targets_reached += 1
                self._target_visit_timer = 0  # Reset timer
                
                # Only log phase completion, not every target transition
                if self._current_target_index >= len(self._current_targets):
                    # Completed all targets in this phase
                    if self._targets_reached % 50 == 0:  # Log every 50 completions
                        print(f"🏆 Phase targets completed! ({self._targets_reached} total targets)")
                    navigation_reward += 5.0  # Bonus for completing all targets
                    # Reset to first target for continuous cycling
                    self._current_target_index = 0
            
            # Directional reward - encourage movement towards target
            if distance_to_target > 0.1:  # Avoid division by zero
                target_direction = (np.array(target_pos) - head_pos) / distance_to_target
                current_velocity = physics.named.data.sensordata['head_vel'][:2]
                velocity_magnitude = np.linalg.norm(current_velocity)
                
                if velocity_magnitude > 0.01:  # Only if actually moving
                    velocity_direction = current_velocity / velocity_magnitude
                    directional_alignment = np.dot(target_direction, velocity_direction)
                    navigation_reward += directional_alignment * 0.3  # Reward for moving in right direction
        
        # Combine rewards
        total_reward = base_reward + navigation_reward
        
        return total_reward

    def update_training_progress(self, progress):
        """Update training progress (0.0 to 1.0)."""
        old_progress = self._training_progress
        self._training_progress = np.clip(progress, 0.0, 1.0)
        
        # Check if targets need to be updated
        old_phase = int(old_progress * 4)
        new_phase = int(self._training_progress * 4)
        
        if new_phase != old_phase:
            # Reset target tracking for new phase
            self._current_target_index = 0
            self._targets_reached = 0
            self._target_visit_timer = 0
            
        self._current_land_zones = self._get_progressive_land_zones()
        self._current_targets = self._get_progressive_targets()

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
        goal_features = 7  # target_distance(1) + target_direction(2) + target_position(2) + target_type(1) + targets_completed(1)
        time_dim = 1 if time_feature else 0
        
        total_obs_dim = base_obs_dim + max_env_features + goal_features + time_dim
        
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
            
            # Extract goal-directed features
            goal_features = []
            if 'target_distance' in obs:
                goal_features.extend(obs['target_distance'])
            else:
                goal_features.append(0.0)  # No target
            
            if 'target_direction' in obs:
                goal_features.extend(obs['target_direction'])
            else:
                goal_features.extend([0.0, 0.0])  # No direction
            
            if 'target_position' in obs:
                goal_features.extend(obs['target_position'])
            else:
                goal_features.extend([0.0, 0.0])  # No target position
            
            if 'target_type' in obs:
                goal_features.extend(obs['target_type'])
            else:
                goal_features.append(1.0)  # Default to swim type
            
            if 'targets_completed' in obs:
                goal_features.extend(obs['targets_completed'])
            else:
                goal_features.append(0.0)  # No targets completed
                
        else:
            # Handle array format
            n_joints = self.env.action_spec.shape[0]
            joint_pos = obs[:n_joints]
            body_vel = obs[n_joints:n_joints + len(joint_pos) * 3 + 3]
            env_features = [0.001, 1.0, 0.0, 1.0, 0.0]  # Default values
            goal_features = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]  # Default goal values
        
        # Combine into single observation vector
        gym_obs = np.concatenate([joint_pos, body_vel, env_features, goal_features])
        
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