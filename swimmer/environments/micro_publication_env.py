#!/usr/bin/env python3
"""
Clean micro-publication environment for NMAP swimmer experiments.
Ported from nma_nai_on-main/micro_publication/environment.py with adapted imports.
Uses compute_navigation_reward for a principled, interpretable reward function.
"""

import collections
import os
from typing import Dict, List

import numpy as np
from dm_control import suite
from dm_control.rl import control
from dm_control.suite import swimmer
try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

from .micro_publication_config import AnisotropyConfig, ObservationConfig, RewardConfig
from .micro_publication_rewards import combine_reward_components, compute_navigation_reward

if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "glfw"

_SWIM_SPEED = 0.15
_MAX_DRAG_VELOCITY = 2.0
_MAX_FORCE_COMPONENT = 0.25


class MicroPublicationSwimCrawl(swimmer.Swimmer):
    """Smaller, publication-oriented mixed-media task with clean reward decomposition."""

    def __init__(
        self,
        desired_speed=_SWIM_SPEED,
        training_progress=0.0,
        observation_config: Dict = None,
        anisotropy_config: Dict = None,
        reward_config: Dict = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._desired_speed = desired_speed
        self._training_progress = training_progress
        self.observation_config = ObservationConfig(**(observation_config or {}))
        self.anisotropy_config = AnisotropyConfig(**(anisotropy_config or {}))
        self.reward_config = RewardConfig(**(reward_config or {}))

        self._water_viscosity = 0.001
        self._land_viscosity = 0.05
        self._current_land_zones = self._get_land_zones()
        self._current_targets = self._get_targets()
        self._current_target_index = 0
        self._targets_reached = 0
        self._target_radius = 0.8
        self._target_visit_timer = 0
        self._environment_transitions = 0
        self._last_environment = None
        self._previous_body_positions = None
        self._segment_body_names: List[str] = []
        self._initial_target_distance = None
        self._last_distance = None
        self._last_in_land = False
        self._land_start_probability = 0.35

    def _get_land_zones(self):
        if self._training_progress < 0.3:
            return []
        if self._training_progress < 0.6:
            return [{"center": [3.0, 0.0], "radius": 1.8}]
        return [
            {"center": [-2.0, 0.0], "radius": 1.4},
            {"center": [3.5, 0.0], "radius": 1.4},
        ]

    def _get_targets(self):
        if self._training_progress < 0.3:
            return [
                {"position": [1.5, 0.0], "type": "swim"},
                {"position": [2.5, 0.0], "type": "swim"},
            ]
        if self._training_progress < 0.6:
            return [
                {"position": [4.4, 0.0], "type": "land"},
                {"position": [-0.9, 0.0], "type": "swim"},
                {"position": [3.9, 0.9], "type": "land"},
                {"position": [-1.2, 0.0], "type": "swim"},
            ]
        return [
            {"position": [-3.1, 0.0], "type": "land"},
            {"position": [0.5, 0.0], "type": "swim"},
            {"position": [-2.6, 0.9], "type": "land"},
            {"position": [4.9, 0.0], "type": "land"},
            {"position": [1.5, 0.0], "type": "swim"},
            {"position": [4.2, -0.9], "type": "land"},
            {"position": [-0.5, 0.0], "type": "swim"},
        ]

    def initialize_episode(self, physics):
        super().initialize_episode(physics)
        self._current_land_zones = self._get_land_zones()
        self._current_targets = self._get_targets()
        self._current_target_index = 0
        self._targets_reached = 0
        self._target_visit_timer = 0
        self._environment_transitions = 0
        self._last_environment = None
        self._initial_target_distance = None
        self._last_distance = None
        self._last_in_land = False
        self._set_starting_position(physics)
        self._segment_body_names = self._identify_segment_bodies(physics)
        self._previous_body_positions = self._get_body_positions(physics)
        self._clear_applied_forces(physics)

    def _set_starting_position(self, physics):
        def _apply_root_xy(x, y):
            try:
                physics.named.data.qpos["root"][0] = x
                physics.named.data.qpos["root"][1] = y
                return True
            except Exception:
                pass
            try:
                physics.data.qpos[0] = x
                physics.data.qpos[1] = y
                return True
            except Exception:
                return False

        if getattr(self, "force_land_start", False) and self._current_land_zones:
            zone = self._current_land_zones[0]
            direction = np.array([0.0, 0.0], dtype=np.float32) - np.array(zone["center"], dtype=np.float32)
            norm = np.linalg.norm(direction)
            if norm < 1e-6:
                direction = np.array([1.0, 0.0], dtype=np.float32)
            else:
                direction = direction / norm
            tangent = np.array([-direction[1], direction[0]], dtype=np.float32)
            radial = zone["radius"] * np.random.uniform(0.75, 0.95)
            lateral = zone["radius"] * np.random.uniform(-0.15, 0.15)
            start = np.array(zone["center"], dtype=np.float32) + direction * radial + tangent * lateral
            _apply_root_xy(float(start[0]), float(start[1]))
            return

        if self._current_land_zones and self._training_progress >= 0.3 and np.random.rand() < self._land_start_probability:
            zone = self._current_land_zones[np.random.randint(len(self._current_land_zones))]
            direction = np.array([0.0, 0.0], dtype=np.float32) - np.array(zone["center"], dtype=np.float32)
            norm = np.linalg.norm(direction)
            if norm < 1e-6:
                direction = np.array([1.0, 0.0], dtype=np.float32)
            else:
                direction = direction / norm
            tangent = np.array([-direction[1], direction[0]], dtype=np.float32)
            radial = zone["radius"] * np.random.uniform(0.88, 0.98)
            lateral = zone["radius"] * np.random.uniform(-0.12, 0.12)
            start = np.array(zone["center"], dtype=np.float32) + direction * radial + tangent * lateral
            _apply_root_xy(float(start[0]), float(start[1]))
            return

        _apply_root_xy(0.0, 0.0)

    def update_training_progress(self, training_progress):
        self._training_progress = training_progress
        self._current_land_zones = self._get_land_zones()
        self._current_targets = self._get_targets()

    def _identify_segment_bodies(self, physics):
        try:
            all_names = list(physics.named.data.xpos.axes.row.names)
        except Exception:
            return []
        return [
            name for name in all_names
            if ("head" in name or "link" in name or "torso" in name or "body" in name)
        ]

    def _get_body_positions(self, physics):
        positions = []
        for body_name in self._segment_body_names:
            try:
                positions.append(np.array(physics.named.data.xpos[body_name][:2], dtype=np.float32))
            except Exception:
                continue
        return np.array(positions, dtype=np.float32) if positions else np.zeros((0, 2), dtype=np.float32)

    def _clear_applied_forces(self, physics):
        try:
            physics.data.xfrc_applied[:] = 0.0
        except Exception:
            pass

    def _current_environment_state(self, physics):
        head_pos = physics.named.data.xpos["head"][:2]
        for zone in self._current_land_zones:
            if np.linalg.norm(head_pos - np.array(zone["center"], dtype=np.float32)) < zone["radius"]:
                return "land", self._land_viscosity, False, True
        return "water", self._water_viscosity, True, False

    def _should_apply_anisotropy(self, environment_name: str):
        if environment_name == "water" and self.anisotropy_config.apply_in_water:
            return True
        if environment_name == "land" and self.anisotropy_config.apply_in_land:
            return True
        return False

    def _compute_drag_force(self, velocity, tangent):
        tangent = tangent / (np.linalg.norm(tangent) + 1e-8)
        normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)
        safe_velocity = np.nan_to_num(np.asarray(velocity, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        safe_velocity = np.clip(safe_velocity, -_MAX_DRAG_VELOCITY, _MAX_DRAG_VELOCITY)
        v_tangent = float(np.dot(safe_velocity, tangent))
        v_normal = float(np.dot(safe_velocity, normal))

        tangential_term = self.anisotropy_config.tangential_gain * v_tangent * tangent
        normal_term = self.anisotropy_config.normal_gain * v_normal * normal

        if self.anisotropy_config.mode == "full" and self.anisotropy_config.quadratic_drag:
            tangential_term += self.anisotropy_config.tangential_gain * abs(v_tangent) * v_tangent * tangent
            normal_term += self.anisotropy_config.normal_gain * abs(v_normal) * v_normal * normal

        if self.anisotropy_config.mode in {"proxy", "full"}:
            normal_term *= self.anisotropy_config.drag_ratio

        force = -(tangential_term + normal_term)
        force = np.nan_to_num(force, nan=0.0, posinf=0.0, neginf=0.0)
        force = np.clip(force, -_MAX_FORCE_COMPONENT, _MAX_FORCE_COMPONENT)
        return force.astype(np.float32)

    def _apply_anisotropic_drag(self, physics):
        if self.anisotropy_config.mode == "off":
            self._clear_applied_forces(physics)
            return

        current_positions = self._get_body_positions(physics)
        if len(current_positions) == 0:
            return

        if self._previous_body_positions is None or len(self._previous_body_positions) != len(current_positions):
            self._previous_body_positions = current_positions.copy()
            self._clear_applied_forces(physics)
            return

        environment_name, _, _, _ = self._current_environment_state(physics)
        if not self._should_apply_anisotropy(environment_name):
            self._previous_body_positions = current_positions.copy()
            self._clear_applied_forces(physics)
            return

        dt = max(float(swimmer._CONTROL_TIMESTEP), 1e-6)
        velocities = (current_positions - self._previous_body_positions) / dt
        velocities = np.nan_to_num(velocities, nan=0.0, posinf=0.0, neginf=0.0)
        velocities = np.clip(velocities, -_MAX_DRAG_VELOCITY, _MAX_DRAG_VELOCITY)
        self._clear_applied_forces(physics)

        for idx, body_name in enumerate(self._segment_body_names[:len(current_positions)]):
            if len(current_positions) == 1:
                tangent = np.array([1.0, 0.0], dtype=np.float32)
            elif idx < len(current_positions) - 1:
                tangent = current_positions[idx + 1] - current_positions[idx]
            else:
                tangent = current_positions[idx] - current_positions[idx - 1]

            drag_force_xy = self._compute_drag_force(velocities[idx], tangent)

            try:
                body_id = physics.model.name2id(body_name, "body")
                physics.data.xfrc_applied[body_id, 0] = drag_force_xy[0]
                physics.data.xfrc_applied[body_id, 1] = drag_force_xy[1]
            except Exception:
                continue

        self._previous_body_positions = current_positions.copy()

    def before_step(self, action, physics):
        self._apply_anisotropic_drag(physics)
        return super().before_step(action, physics)

    def get_observation(self, physics):
        obs = collections.OrderedDict()
        obs["joints"] = physics.joints()
        obs["body_velocities"] = physics.body_velocities()

        env_name, current_viscosity, in_water, in_land = self._current_environment_state(physics)
        physics.model.opt.viscosity = current_viscosity

        if self._last_environment is not None and self._last_environment != env_name:
            self._environment_transitions += 1
        self._last_environment = env_name

        obs["fluid_viscosity"] = np.array(
            [current_viscosity if self.observation_config.expose_viscosity else 0.0],
            dtype=np.float32,
        )
        if self.observation_config.expose_environment:
            obs["environment_type"] = np.array([1.0 if in_water else 0.0, 1.0 if in_land else 0.0], dtype=np.float32)
            obs["in_water_zone"] = np.array([1.0 if in_water else 0.0], dtype=np.float32)
            obs["in_land_zone"] = np.array([1.0 if in_land else 0.0], dtype=np.float32)
        else:
            obs["environment_type"] = np.array([0.0, 0.0], dtype=np.float32)
            obs["in_water_zone"] = np.array([0.0], dtype=np.float32)
            obs["in_land_zone"] = np.array([0.0], dtype=np.float32)

        if self.observation_config.expose_target and self._current_targets:
            current_target = self._current_targets[self._current_target_index % len(self._current_targets)]
            head_pos = physics.named.data.xpos["head"][:2]
            target_pos = np.array(current_target["position"], dtype=np.float32)
            target_vector = target_pos - head_pos
            distance = np.linalg.norm(target_vector)
            direction = target_vector / (distance + 1e-6)
            obs["target_distance"] = np.array([distance], dtype=np.float32)
            obs["target_direction"] = direction.astype(np.float32)
            obs["target_position"] = target_pos.astype(np.float32)
            obs["target_type"] = np.array([1.0 if current_target["type"] == "swim" else 0.0], dtype=np.float32)
            obs["targets_completed"] = np.array([self._targets_reached], dtype=np.float32)
        else:
            obs["target_distance"] = np.array([0.0], dtype=np.float32)
            obs["target_direction"] = np.array([0.0, 0.0], dtype=np.float32)
            obs["target_position"] = np.array([0.0, 0.0], dtype=np.float32)
            obs["target_type"] = np.array([1.0], dtype=np.float32)
            obs["targets_completed"] = np.array([self._targets_reached], dtype=np.float32)

        return obs

    def get_reward(self, physics):
        head_pos = physics.named.data.xpos["head"][:2]
        _, _, _, in_land = self._current_environment_state(physics)
        current_target = self._current_targets[self._current_target_index % len(self._current_targets)]
        target_pos = np.array(current_target["position"], dtype=np.float32)
        distance_to_target = float(np.linalg.norm(head_pos - target_pos))

        if self._initial_target_distance is None:
            self._initial_target_distance = distance_to_target
            self._last_distance = distance_to_target

        joint_activity = float(np.sum(np.abs(physics.data.qvel)))
        just_entered_land = in_land and not self._last_in_land
        just_entered_water = (not in_land) and self._last_in_land
        components = compute_navigation_reward(
            distance_to_target=distance_to_target,
            initial_distance=self._initial_target_distance,
            last_distance=self._last_distance,
            target_type=current_target["type"],
            in_land=in_land,
            just_entered_land=just_entered_land,
            just_entered_water=just_entered_water,
            transitions=self._environment_transitions,
            joint_activity=joint_activity,
            visit_timer=self._target_visit_timer,
            config=self.reward_config,
        )

        target_reached = distance_to_target < self._target_radius
        reward = combine_reward_components(components, target_reached)

        self._last_distance = distance_to_target
        self._target_visit_timer += 1
        self._last_in_land = in_land

        if target_reached:
            self._current_target_index = (self._current_target_index + 1) % len(self._current_targets)
            self._targets_reached += 1
            self._target_visit_timer = 0
            self._initial_target_distance = None
            self._last_distance = None

        return reward


@swimmer.SUITE.add()
def micro_publication_swim_crawl(
    n_links=6,
    desired_speed=_SWIM_SPEED,
    training_progress=0.0,
    observation_config=None,
    anisotropy_config=None,
    reward_config=None,
    time_limit=swimmer._DEFAULT_TIME_LIMIT,
    random=None,
    environment_kwargs={},
):
    model_string, assets = swimmer.get_model_and_assets(n_links)
    physics = swimmer.Physics.from_xml_string(model_string, assets=assets)
    task = MicroPublicationSwimCrawl(
        desired_speed=desired_speed,
        training_progress=training_progress,
        observation_config=observation_config,
        anisotropy_config=anisotropy_config,
        reward_config=reward_config,
        random=random,
    )
    return control.Environment(
        physics,
        task,
        time_limit=time_limit,
        control_timestep=swimmer._CONTROL_TIMESTEP,
        **environment_kwargs,
    )


class MicroPublicationSwimmerEnv:
    """Episode-level wrapper that manages curriculum progression."""

    def __init__(
        self,
        n_links=6,
        desired_speed=_SWIM_SPEED,
        time_limit=3000,
        observation_config=None,
        anisotropy_config=None,
        reward_config=None,
        land_start_probability=0.35,
    ):
        self.n_links = n_links
        self.desired_speed = desired_speed
        self.time_limit = time_limit
        self._training_progress = 0.0
        self.target_episodes = 20000
        self.total_episodes = 0
        self.observation_config = observation_config or {}
        self.anisotropy_config = anisotropy_config or {}
        self.reward_config = reward_config or {}
        self.force_land_start = False
        self.manual_progress_override = False
        self.land_start_probability = land_start_probability
        self.prefer_transition_evaluation = True
        self._create_environment()

    def _create_environment(self):
        self.env = suite.load(
            "swimmer",
            "micro_publication_swim_crawl",
            task_kwargs={
                "random": 1,
                "n_links": self.n_links,
                "training_progress": self._training_progress,
                "observation_config": self.observation_config,
                "anisotropy_config": self.anisotropy_config,
                "reward_config": self.reward_config,
            },
        )
        self.physics = self.env.physics
        self.action_spec = self.env.action_spec()
        self.observation_spec = self.env.observation_spec()
        if hasattr(self.env, "_task"):
            self.env._task._land_start_probability = self.land_start_probability
            self.env._task.force_land_start = self.force_land_start

    def update_training_progress(self, episode_count):
        old_phase = int(self._training_progress * 4)
        self._training_progress = min(1.0, episode_count / max(self.target_episodes, 1))
        new_phase = int(self._training_progress * 4)
        if new_phase != old_phase:
            self._create_environment()
        if hasattr(self.env, "_task"):
            self.env._task.update_training_progress(self._training_progress)

    def set_manual_progress(self, progress, force_land_start=False):
        self.manual_progress_override = True
        self.force_land_start = force_land_start
        self.training_progress = float(np.clip(progress, 0.0, 1.0))
        if hasattr(self.env, "_task"):
            self.env._task.force_land_start = self.force_land_start

    def reset(self):
        self.total_episodes += 1
        if not self.manual_progress_override:
            self.update_training_progress(self.total_episodes)
        self._clear_physics_state()
        return self.env.reset().observation

    def step(self, action):
        try:
            safe_action = np.nan_to_num(np.asarray(action, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            safe_action = np.clip(safe_action, self.action_spec.minimum, self.action_spec.maximum)
            time_step = self.env.step(safe_action)
            raw_reward = 0.0 if time_step.reward is None else time_step.reward
            reward = float(np.nan_to_num(raw_reward, nan=0.0, posinf=0.0, neginf=0.0))
            return time_step.observation, reward, time_step.last(), {}
        except control.PhysicsError:
            self._clear_physics_state()
            recovery_obs = self.env.reset().observation
            return recovery_obs, -1.0, True, {"physics_error": True}

    def _clear_physics_state(self):
        try:
            self.physics.data.xfrc_applied[:] = 0.0
        except Exception:
            pass
        try:
            if hasattr(self.env, "_task"):
                self.env._task._previous_body_positions = None
        except Exception:
            pass

    def render(self, mode="rgb_array", height=480, width=640):
        return self.physics.render(camera_id=0, height=height, width=width)

    @property
    def head_position(self):
        return self.physics.named.data.xpos["head"][:2].copy()

    @property
    def training_progress(self):
        return self._training_progress

    @training_progress.setter
    def training_progress(self, value):
        value = float(np.clip(value, 0.0, 1.0))
        phase_old = int(self._training_progress * 4)
        phase_new = int(value * 4)
        self._training_progress = value
        if phase_new != phase_old:
            self._create_environment()
        if hasattr(self, "env") and hasattr(self.env, "_task"):
            self.env._task.update_training_progress(self._training_progress)

    def close(self):
        pass


class TonicMicroPublicationWrapper(gym.Env):
    """Gym wrapper for TonicMicroPublicationWrapper, compatible with our CurriculumNCAPTrainer."""

    def __init__(
        self,
        n_links=6,
        time_feature=True,
        desired_speed=_SWIM_SPEED,
        observation_config=None,
        anisotropy_config=None,
        reward_config=None,
        land_start_probability=0.35,
    ):
        super().__init__()
        self.env = MicroPublicationSwimmerEnv(
            n_links=n_links,
            desired_speed=desired_speed,
            observation_config=observation_config,
            anisotropy_config=anisotropy_config,
            reward_config=reward_config,
            land_start_probability=land_start_probability,
        )

        action_spec = self.env.action_spec
        self.action_space = spaces.Box(
            low=action_spec.minimum,
            high=action_spec.maximum,
            dtype=np.float32,
        )

        # n_links-1 joints + n_links*3+3 body velocities + 5 env features + 7 goal + 1 time
        n_joints = n_links - 1
        base_obs_dim = n_joints + (n_links * 3 + 3)
        env_features = 5
        goal_features = 7
        time_dim = 1 if time_feature else 0
        total_obs_dim = base_obs_dim + env_features + goal_features + time_dim

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_obs_dim,),
            dtype=np.float32,
        )
        self.time_feature = time_feature
        self.step_count = 0
        self.max_steps = 600

    def reset(self):
        obs = self.env.reset()
        self.step_count = 0
        return self._process_observation(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.step_count += 1
        if self.step_count >= self.max_steps:
            done = True
        return self._process_observation(obs), reward, done, info

    def _process_observation(self, obs):
        if isinstance(obs, dict):
            joint_pos = obs.get("joints", np.zeros(self.env.action_spec.shape[0]))
            body_vel = obs.get("body_velocities", np.zeros(len(joint_pos) * 3 + 3))
            env_features = list(obs.get("fluid_viscosity", [0.0]))
            env_features.extend(list(obs.get("environment_type", [0.0, 0.0])))
            env_features.extend([obs.get("in_water_zone", [0.0])[0], obs.get("in_land_zone", [0.0])[0]])
            goal_features = list(obs.get("target_distance", [0.0]))
            goal_features.extend(list(obs.get("target_direction", [0.0, 0.0])))
            goal_features.extend(list(obs.get("target_position", [0.0, 0.0])))
            goal_features.extend(list(obs.get("target_type", [1.0])))
            goal_features.extend(list(obs.get("targets_completed", [0.0])))
        else:
            n_joints = self.env.action_spec.shape[0]
            joint_pos = obs[:n_joints]
            body_vel = obs[n_joints:n_joints + len(joint_pos) * 3 + 3]
            env_features = [0.0, 0.0, 0.0, 0.0, 0.0]
            goal_features = [0.0] * 7

        gym_obs = np.concatenate([joint_pos, body_vel, env_features, goal_features])
        if self.time_feature:
            gym_obs = np.concatenate([gym_obs, np.array([self.step_count / self.max_steps], dtype=np.float32)])
        expected = self.observation_space.shape[0]
        if len(gym_obs) < expected:
            gym_obs = np.concatenate([gym_obs, np.zeros(expected - len(gym_obs), dtype=np.float32)])
        elif len(gym_obs) > expected:
            gym_obs = gym_obs[:expected]
        return gym_obs.astype(np.float32)

    def render(self, mode="rgb_array"):
        return self.env.render(mode=mode)

    def close(self):
        self.env.close()

    @property
    def head_position(self):
        return self.env.head_position

    @property
    def training_progress(self):
        return self.env.training_progress

    @property
    def name(self):
        return f"micro-publication-swimmer-{self.env.n_links}links"
