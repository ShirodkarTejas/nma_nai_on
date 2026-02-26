#!/usr/bin/env python3
"""
Mixed Environment Swimmer Environment Classes
Contains environment logic for mixed water/land swimming scenarios.
"""

import numpy as np
import collections
import os
import torch.nn as nn
from dm_control import suite
from dm_control.suite import swimmer
from dm_control.rl import control
from dm_control.utils import rewards
from dm_control.mujoco.wrapper import mjbindings
from dm_control.mujoco.wrapper.mjbindings import enums

# Import from other modules
from ..models.ncap_swimmer import NCAPSwimmer, NCAPSwimmerActor
from ..utils.helpers import flatten_observation
from ..utils.visualization import create_comprehensive_visualization, create_parameter_log, add_zone_overlay
from .environment_types import EnvironmentType

# Set environment variable for rendering only if not already set
# This allows main.py to control the rendering backend
if 'MUJOCO_GL' not in os.environ:
    os.environ['MUJOCO_GL'] = 'glfw'  # Windows-compatible rendering

_SWIM_SPEED = 0.15      # lowered target speed to give higher velocity rewards
_CRAWL_SPEED = 0.08     # slightly higher crawl speed target

# --- Mixed Environment Task ---
class MixedEnvironmentSwim(swimmer.Swimmer):
    """Task with mixed water and land zones for real-time adaptation."""
    
    def __init__(self, desired_speed=_SWIM_SPEED, **kwargs):
        super().__init__(**kwargs)
        self._desired_speed = desired_speed
        self.current_viscosity = 1e-4  # default before initialize_episode
        # Default environment is now WATER; define a few *land* patches.
        self.water_zones = []  # Everywhere is water unless inside a land zone
        self.land_zones = [
            {'center': [-1.5, 0], 'radius': 1.5},   # Left land island (closer for evaluation transitions)
            {'center': [1.5, 0], 'radius': 1.5},    # Right land island (closer for evaluation transitions)
        ]

        # Curriculum helpers
        self._episode_counter = 0  # increments every initialize_episode
        
    def initialize_episode(self, physics):
        super().initialize_episode(physics)
        
        # Default physics is now WATER
        self.apply_environment_physics(physics, EnvironmentType.WATER)

        # -------- Curriculum: gradually shrink land islands --------
        self._episode_counter += 1

        # ---------- Water viscosity ----------
        # Fixed at 0.005: thin enough for the 6-link agent to feel the effects
        # of its undulatory movements clearly during training, while still
        # providing meaningful drag feedback (not so thin that the physics
        # becomes numerically noisy, not so thick that thrust is impossible).
        self.current_viscosity = 0.005

        # Apply initial physics with this viscosity
        self.apply_environment_physics(physics, EnvironmentType.WATER)
        
        shrink_steps = [0, 10, 20, 40]  # episode indices where we shrink radius
        radii         = [1.0, 0.6, 0.4, 0.2]

        # Determine appropriate radius based on episode count
        target_radius = radii[min(len(radii)-1, sum(self._episode_counter >= s for s in shrink_steps)-1)]
        for zone in self.land_zones:
            zone['radius'] = target_radius
        
        # Hide target
        physics.named.model.mat_rgba['target', 'a'] = 1
        physics.named.model.mat_rgba['target_default', 'a'] = 1
        physics.named.model.mat_rgba['target_highlight', 'a'] = 1
        
        # Store initial position
        self._initial_position = physics.named.data.xpos['head', :2].copy()
        
        # Note: Can't set starting position, swimmer starts near [0,0]
        # Zones are designed around this default starting position
        self._debug_printed = False
        
        # Initialize tracking for visualization
        self.position_history = []
        self.env_history = []
        self.action_history = []
        self._land_reached = False
        
    def get_current_environment(self, physics):
        """Detect current environment based on swimmer position."""
        head_pos = physics.named.data.xpos['head', :2]
        
        # Debug: print position and zone distances
        if hasattr(self, '_debug_printed') and not self._debug_printed:
            print(f"Starting position: {head_pos}")
            for i, zone in enumerate(self.water_zones):
                distance = np.linalg.norm(head_pos - np.array(zone['center']))
                print(f"Water zone {i}: center={zone['center']}, radius={zone['radius']}, distance={distance}")
            for i, zone in enumerate(self.land_zones):
                distance = np.linalg.norm(head_pos - np.array(zone['center']))
                print(f"Land zone {i}: center={zone['center']}, radius={zone['radius']}, distance={distance}")
            self._debug_printed = True
        
        # Check if in land zone first (islands of land)
        for zone in self.land_zones:
            distance = np.linalg.norm(head_pos - np.array(zone['center']))
            if distance <= zone['radius']:
                return EnvironmentType.LAND

        # Check if in water zone (currently none, but keep for flexibility)
        for zone in self.water_zones:
            distance = np.linalg.norm(head_pos - np.array(zone['center']))
            if distance <= zone['radius']:
                return EnvironmentType.WATER

        # Default environment is WATER
        return EnvironmentType.WATER
    
    def apply_environment_physics(self, physics, env_type):
        """Apply physics based on current environment."""
        if env_type == EnvironmentType.WATER:
            # Water physics – use episode-specific viscosity
            viscosity = getattr(self, 'current_viscosity', 1e-4)
            physics.model.opt.viscosity = viscosity
            # Leave density at MuJoCo default to avoid extra quadratic drag
            
            # Very low friction for fast swimming
            # Restore near-default capsule friction (MuJoCo swimmer.xml ≈ 0.1, 0.005, 0.0001)
            for geom_id in range(physics.model.ngeom):
                if physics.model.geom_type[geom_id] == enums.mjtGeom.mjGEOM_CAPSULE:
                    physics.model.geom_friction[geom_id] = [0.1, 0.005, 0.0001]
                    
        elif env_type == EnvironmentType.LAND:
            # Land physics - high friction, no viscosity
            physics.model.opt.viscosity = 0.0  # No fluid drag on land
            # Leave density unchanged (default ~1.0-air)
            
            # High friction for crawling
            for geom_id in range(physics.model.ngeom):
                if physics.model.geom_type[geom_id] == enums.mjtGeom.mjGEOM_CAPSULE:
                    physics.model.geom_friction[geom_id] = [0.2, 0.05, 0.01]  # Moderate land friction
    
    def get_observation(self, physics):
        obs = collections.OrderedDict()
        obs['joints'] = physics.joints()
        obs['body_velocities'] = physics.body_velocities()

        # ---------- New feature: expose current viscosity ----------
        obs['fluid_viscosity'] = np.array([self.current_viscosity], dtype=np.float32)
        
        # Detect current environment
        current_env = self.get_current_environment(physics)
        
        # Apply physics based on current environment
        self.apply_environment_physics(physics, current_env)
        
        # Track position history for visualization
        head_pos = physics.named.data.xpos['head', :2]
        if not hasattr(self, 'position_history'):
            self.position_history = []
        self.position_history.append(head_pos.copy())
        
        # Track environment history for visualization
        if not hasattr(self, 'env_history'):
            self.env_history = []
        self.env_history.append(current_env)
        
        # Track action history for visualization
        if not hasattr(self, 'action_history'):
            self.action_history = []
        
        # Environment encoding
        if current_env == EnvironmentType.WATER:
            obs['environment_type'] = np.array([1.0, 0.0])  # [water, land] one-hot
        else:
            obs['environment_type'] = np.array([0.0, 1.0])  # [water, land] one-hot
        
        # Add environment transition indicators
        obs['in_water_zone'] = np.array([1.0 if current_env == EnvironmentType.WATER else 0.0])
        obs['in_land_zone'] = np.array([1.0 if current_env == EnvironmentType.LAND else 0.0])
        
        return obs
    
    def get_reward(self, physics):
        # Determine current medium first
        current_env = self.get_current_environment(physics)

        # Primary reward: velocity projected toward the nearest target (land-zone center).
        head_pos = np.asarray(physics.named.data.xpos['head', :2], dtype=np.float64)
        head_vel_xy = np.asarray(physics.named.data.sensordata['head_vel'][:2], dtype=np.float64)
        target_speeds = []
        for zone in self.land_zones:
            target_vec = np.asarray(zone['center'], dtype=np.float64) - head_pos
            norm = np.linalg.norm(target_vec)
            if norm < 1e-8:
                continue
            target_dir = target_vec / norm
            target_speeds.append(float(np.dot(head_vel_xy, target_dir)))
        target_velocity = max(target_speeds) if target_speeds else float(head_vel_xy[0])

        # Use different speed targets depending on medium
        target_speed = _SWIM_SPEED if current_env == EnvironmentType.WATER else _CRAWL_SPEED

        # Increase target-directed locomotion weighting to avoid stationary local optima.
        reward = 4.5 * rewards.tolerance(
            target_velocity,
            bounds=(target_speed, float('inf')),
            margin=target_speed,
            value_at_margin=0.,
            sigmoid='linear',
        )

        # ----------------------------------------------------------------
        # FORWARD MOMENTUM BONUS
        # Extra reward for sustained, meaningful forward velocity.  This
        # distinguishes real forward swimming from the zero-mean "shivering"
        # that the primary tolerance reward cannot fully prevent.
        # ----------------------------------------------------------------
        if target_velocity > 0.05:
            reward += 1.0 * target_velocity

        # ----------------------------------------------------------------
        # ANTI-TWITCHING: joint-velocity (activity) penalty
        # Increased from 0.0005 to 0.002 so high-frequency twitching is
        # clearly more costly than generating directed thrust.
        # ----------------------------------------------------------------
        joint_velocities = physics.data.qvel
        activity_penalty = np.sum(np.square(joint_velocities))
        reward -= activity_penalty * 0.002

        # ----------------------------------------------------------------
        # POSTURE PENALTY (anti-curling):
        # Light penalty for extreme joint angles that remain static.
        # ----------------------------------------------------------------
        n_joints = int(np.size(physics.data.ctrl))
        if n_joints > 0:
            joint_angles = np.asarray(physics.data.qpos[-n_joints:], dtype=np.float64)
            joint_speeds = np.asarray(physics.data.qvel[-n_joints:], dtype=np.float64)
            extreme_angles = np.maximum(0.0, np.abs(joint_angles) - 1.0)
            static_mask = (np.abs(joint_speeds) < 0.03).astype(np.float64)
            posture_penalty = float(np.mean(extreme_angles * static_mask))
            reward -= 0.03 * posture_penalty

        # ----------------------------------------------------------------
        # ANTI-TWITCHING: torque penalties
        # Two-tier torque penalty:
        #   1. A general soft penalty for all torque (scaled by viscosity).
        #   2. A HARD PENALTY for any joint torque exceeding ±0.8 to
        #      discourage high-frequency micro-contractions that produce no
        #      net displacement but rack up tiny positive velocity rewards.
        # ----------------------------------------------------------------
        ctrl = physics.data.ctrl

        # Soft penalty on total torque (reduced so policy is not over-penalized for actuation)
        torque_penalty = np.sum(np.square(ctrl))
        vis_scale = min(self.current_viscosity / 0.3, 1.0)  # 0..1
        reward -= torque_penalty * 0.0006 * vis_scale

        # Hard penalty on torques exceeding 0.8.
        excessive_torque = np.sum(np.maximum(0.0, np.abs(ctrl) - 0.8) ** 2)
        reward -= excessive_torque * 0.03

        # Small constant incentive to keep moving (helps exploration).
        reward += 0.01

        # -------- Milestone bonuses --------
        # One-time bonus when the swimmer reaches land for the first time in episode
        if current_env == EnvironmentType.LAND and not getattr(self, '_land_reached', False):
            self._land_reached = True
            reward += 0.3

        # Distance-based shaping every 0.5 m from initial position
        dist_from_start = np.linalg.norm(head_pos - self._initial_position)
        reward += 0.1 * (dist_from_start // 0.5)

        # Bonus for approaching water targets (if any water_zones defined)
        if self.water_zones:
            head_pos = physics.named.data.xpos['head', :2]
            min_dist_to_water = min(
                np.linalg.norm(head_pos - np.array(zone['center'])) for zone in self.water_zones
            )
            reward += (1.0 / (1.0 + min_dist_to_water**2)) * 0.1

        return reward

# --- Register mixed environment task ---
if not hasattr(swimmer, 'mixed_env_registered'):
    @swimmer.SUITE.add()
    def mixed_environment(
        n_links=6,
        desired_speed=_SWIM_SPEED,
        time_limit=3000,
        random=None,
    ):
        """Returns the mixed environment swim task."""
        model_string, assets = swimmer.get_model_and_assets(n_links)
        physics = swimmer.Physics.from_xml_string(model_string, assets=assets)
        task = MixedEnvironmentSwim(desired_speed=desired_speed, random=random)
        return control.Environment(physics, task, time_limit=time_limit, n_sub_steps=4)
    
    swimmer.mixed_env_registered = True

# --- Environment Wrapper Class ---
class MixedSwimmerEnv:
    """Wrapper class for the mixed environment swimmer."""
    
    def __init__(self, n_links=6, desired_speed=_SWIM_SPEED, time_limit=3000, speed_factor=1.0):
        # Create base environment
        self.env = suite.load('swimmer', 'mixed_environment', 
                             task_kwargs={'random': 1, 'n_links': n_links})
        # Accelerate simulation by scaling MuJoCo timestep (bigger timestep = faster simulated time)
        if speed_factor != 1.0:
            self.env.physics.model.opt.timestep *= speed_factor
            # Optionally reduce n_sub_steps to keep stability (unchanged for now)
        self.speed_factor = speed_factor
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
        self.env.task.action_history.append(action)
        time_step = self.env.step(action)
        self.done = time_step.last()
        return time_step.observation, time_step.reward, self.done, {}
    
    def render(self, mode='rgb_array', height=480, width=640):
        """Render the environment."""
        return self.physics.render(camera_id=0, height=height, width=width)
    
    def close(self):
        """Close the environment."""
        pass

# --- Test Function ---
def test_mixed_environment(n_links=6, oscillator_period=60, amplitude=8.0, trained_model_path=None):
    """Test mixed environment with square wave oscillators or trained model."""
    import torch
    import numpy as np
    import os
    import time
    import imageio
    
    print("=== TESTING MIXED ENVIRONMENT ===")
    
    # Create mixed environment
    env = suite.load('swimmer', 'mixed_environment', task_kwargs={'random': 1, 'n_links': n_links})
    
    obs_spec = env.observation_spec()
    action_spec = env.action_spec()
    time_step = env.reset()
    obs = time_step.observation
    n_joints = action_spec.shape[0]
    
    print(f"Observation shape: {flatten_observation(obs).shape}, Action shape: {action_spec.shape}")
    
    if trained_model_path:
        print(f"Loading trained model from: {trained_model_path}")
        # Load trained model
        from ..models.tonic_ncap import create_tonic_ncap_model
        from ..training.custom_tonic_agent import CustomPPO
        from ..environments.tonic_wrapper import TonicSwimmerWrapper
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = create_tonic_ncap_model(n_joints=n_joints, oscillator_period=oscillator_period, memory_size=10)
        model.to(device)
        
        checkpoint = torch.load(trained_model_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.eval()
        
        # Create Tonic environment for compatibility
        tonic_env = TonicSwimmerWrapper(n_links=n_links, time_feature=True)
        
        # Create agent
        agent = CustomPPO(model=model)
        agent.initialize(
            observation_space=tonic_env.observation_space,
            action_space=tonic_env.action_space,
            seed=42
        )
        
        # Create a simple actor wrapper for the trained model
        class TrainedModelActor:
            def __init__(self, agent, tonic_env):
                self.agent = agent
                self.tonic_env = tonic_env
                self.step_count = 0
            
            def __call__(self, obs):
                # Convert observation to Tonic format
                if isinstance(obs, dict):
                    tonic_obs = np.concatenate([
                        obs['joints'],
                        obs['environment_type'],
                        obs['head_position'],
                        obs['in_water_zone'],
                        obs['in_land_zone']
                    ])
                else:
                    tonic_obs = obs
                
                # Get action from trained model
                with torch.no_grad():
                    action = self.agent.test_step(tonic_obs, steps=self.step_count)
                    if torch.is_tensor(action):
                        action = action.cpu().numpy()
                
                self.step_count += 1
                return action
        
        actor = TrainedModelActor(agent, tonic_env)
        print("Using trained model for evaluation")
    else:
        # Create adaptive swimmer with square wave oscillators
        ncap_module = NCAPSwimmer(n_joints=n_joints, oscillator_period=oscillator_period)
        ncap_module.base_amplitude = nn.Parameter(torch.tensor(amplitude))
        actor = NCAPSwimmerActor(ncap_module)
        print("Using default NCAP model for evaluation")
    
    # Performance tracking
    start_time = time.time()
    physics = env.physics
    initial_head_pos = physics.named.data.xpos['head'].copy()
    velocities = []
    rewards_list = []
    distances = []
    environment_history = []
    
    # Video generation
    os.makedirs("results/mixed_env", exist_ok=True)
    video_filename = f"results/mixed_env/mixed_env_adaptation_{n_links}links.mp4"
    plot_filename = f"results/mixed_env/mixed_env_analysis_{n_links}links.png"
    log_filename = f"results/mixed_env/parameter_log_{n_links}links.txt"
    frame_count = 0
    max_frames = 1800  # 60 seconds at 30 fps (much longer training)
    frames = []
    
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
        # Get action from improved adaptive actor
        action = actor(obs)
        action = np.clip(action, action_spec.minimum, action_spec.maximum)
        
        time_step = env.step(action)
        obs = time_step.observation
        
        # Track environment changes
        new_env = env.task.get_current_environment(physics)
        if current_env != new_env and current_env is not None:
            env_transitions += 1
            print(f"Environment transition: {current_env} -> {new_env} at frame {frame_count}")
        current_env = new_env
        environment_history.append(current_env)
        
        # Track performance metrics
        current_head_pos = physics.named.data.xpos['head']
        current_velocity = physics.named.data.sensordata['head_vel']
        current_reward = time_step.reward if time_step.reward is not None else 0.0
        
        # Calculate metrics
        distance = np.linalg.norm(current_head_pos[:2] - initial_head_pos[:2])
        distances.append(distance)
        
        velocity_mag = np.linalg.norm(current_velocity[:2])
        velocities.append(velocity_mag)
        
        rewards_list.append(current_reward)
        
        # Capture frame
        try:
            camera = physics.render(camera_id=0, height=480, width=640)
            if camera.dtype != np.uint8:
                camera = (camera * 255).astype(np.uint8)
            
            # Add visual zone overlay to the frame
            frame_with_zones = add_zone_overlay(camera, env.task, current_env)
            frames.append(frame_with_zones)
        except Exception as e:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame[:, :, 0] = 255
            frames.append(frame)
        
        frame_count += 1
        if frame_count % 60 == 0:
            print(f"Captured {frame_count} frames, Environment: {current_env}, Transitions: {env_transitions}")
        
        if time_step.last():
            time_step = env.reset()
            obs = time_step.observation
    
    # Calculate final metrics
    total_time = time.time() - start_time
    total_distance = distances[-1] if distances else 0.0
    avg_velocity = np.mean(velocities) if velocities else 0.0
    max_velocity = np.max(velocities) if velocities else 0.0
    avg_reward = np.mean(rewards_list) if rewards_list else 0.0
    
    print(f"\n=== MIXED ENVIRONMENT PERFORMANCE METRICS ===")
    print(f"Total distance traveled: {total_distance:.4f}")
    print(f"Average velocity: {avg_velocity:.4f}")
    print(f"Maximum velocity: {max_velocity:.4f}")
    print(f"Average reward: {avg_reward:.4f}")
    print(f"Environment transitions: {env_transitions}")
    
    # Save video
    if frames:
        imageio.mimsave(video_filename, frames, fps=30, quality=8)
        print(f"Video saved as {video_filename}")
    
    # Create and save comprehensive visualization
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
    
    create_comprehensive_visualization(env.task, results, plot_filename)
    create_parameter_log(env.task, results, n_links, oscillator_period, amplitude, log_filename)
    
    return results 
