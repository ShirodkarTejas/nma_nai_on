#!/usr/bin/env python3
"""
Configuration dataclasses for the micro-publication environment path.
Ported from the reference repo (nma_nai_on-main/micro_publication/config.py).
"""

from dataclasses import asdict, dataclass, field
from typing import Dict, List


@dataclass
class ObservationConfig:
    expose_environment: bool = True
    expose_viscosity: bool = True
    expose_target: bool = True

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class AnisotropyConfig:
    mode: str = "off"          # off, proxy, full
    drag_ratio: float = 10.0
    tangential_gain: float = 0.02
    normal_gain: float = 0.2
    quadratic_drag: bool = False
    apply_in_water: bool = False
    apply_in_land: bool = True

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class RewardConfig:
    progress_weight: float = 2.0
    completion_reward: float = 10.0
    land_target_bonus: float = 1.5
    land_target_in_zone_bonus: float = 2.0
    pre_land_progress_scale: float = 0.15
    land_entry_bonus: float = 3.0
    water_entry_bonus: float = 3.0
    activity_reward_scale: float = 0.01
    max_activity_reward: float = 0.1
    environment_transition_bonus: float = 0.25
    max_transition_bonus: float = 2.0
    wrong_medium_penalty_after: int = 80
    wrong_medium_penalty: float = 0.03
    stagnation_penalty_after: int = 180
    stagnation_penalty: float = 0.02
    no_timeout_target_switch: bool = True

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class TrainingConfig:
    model_type: str = "enhanced_ncap"
    algorithm: str = "ppo"
    training_steps: int = 3_000_000
    save_steps: int = 100_000
    log_episodes: int = 50
    n_links: int = 6
    oscillator_period: int = 60
    num_workers: int = 1
    use_multi_gpu: bool = False
    use_locomotion_only_early_training: bool = True
    learning_rate: float = 3e-5
    land_start_probability: float = 0.35

    def to_dict(self) -> Dict:
        return asdict(self)
