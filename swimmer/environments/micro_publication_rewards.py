#!/usr/bin/env python3
"""
Reward helpers for the micro-publication environment.
Ported from the reference repo (nma_nai_on-main/micro_publication/rewards.py).
"""

from typing import Dict

import numpy as np

from .micro_publication_config import RewardConfig


def compute_navigation_reward(
    distance_to_target: float,
    initial_distance: float,
    last_distance: float,
    target_type: str,
    in_land: bool,
    just_entered_land: bool,
    just_entered_water: bool,
    transitions: int,
    joint_activity: float,
    visit_timer: int,
    config: RewardConfig,
) -> Dict[str, float]:
    """Build interpretable reward components from navigation state."""
    target_multiplier = 1.0
    if target_type == "land":
        target_multiplier = config.land_target_bonus
        if in_land:
            target_multiplier = config.land_target_in_zone_bonus

    progress_reward = 0.0
    if initial_distance > 1e-6:
        progress_ratio = max(0.0, initial_distance - distance_to_target) / initial_distance
        progress_reward = progress_ratio * config.progress_weight * target_multiplier
    elif last_distance is not None:
        progress_reward = max(0.0, last_distance - distance_to_target) * config.progress_weight * target_multiplier

    # For land targets, weakly reward approach until the agent actually enters land.
    if target_type == "land" and not in_land:
        progress_reward *= config.pre_land_progress_scale

    completion_reward = config.completion_reward * target_multiplier
    activity_reward = min(joint_activity * config.activity_reward_scale, config.max_activity_reward)
    transition_bonus = min(transitions * config.environment_transition_bonus, config.max_transition_bonus)
    land_entry_bonus = config.land_entry_bonus if (target_type == "land" and just_entered_land) else 0.0
    water_entry_bonus = config.water_entry_bonus if (target_type == "swim" and just_entered_water) else 0.0
    in_wrong_medium = (target_type == "land" and not in_land) or (target_type == "swim" and in_land)
    wrong_medium_penalty = 0.0
    if in_wrong_medium and visit_timer >= config.wrong_medium_penalty_after:
        wrong_medium_penalty = config.wrong_medium_penalty

    stagnation_penalty = 0.0
    if visit_timer >= config.stagnation_penalty_after:
        stagnation_penalty = config.stagnation_penalty

    return {
        "progress_reward": progress_reward,
        "completion_reward": completion_reward,
        "activity_reward": activity_reward,
        "transition_bonus": transition_bonus,
        "land_entry_bonus": land_entry_bonus,
        "water_entry_bonus": water_entry_bonus,
        "wrong_medium_penalty": wrong_medium_penalty,
        "stagnation_penalty": stagnation_penalty,
    }


def combine_reward_components(components: Dict[str, float], target_reached: bool) -> float:
    total = components["progress_reward"]
    total += components["activity_reward"]
    total += components["transition_bonus"]
    total += components.get("land_entry_bonus", 0.0)
    total += components.get("water_entry_bonus", 0.0)
    total -= components.get("wrong_medium_penalty", 0.0)
    total -= components["stagnation_penalty"]
    if target_reached:
        total += components["completion_reward"]
    return float(total)
