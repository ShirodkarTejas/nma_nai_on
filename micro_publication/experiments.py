#!/usr/bin/env python3
"""
Named experiment presets for the micro-publication package.
"""

from copy import deepcopy

from .config import (
    AnisotropyConfig,
    ExperimentConfig,
    ObservationConfig,
    RewardConfig,
    TrainingConfig,
)


def _base_experiment() -> ExperimentConfig:
    return ExperimentConfig(
        name="baseline_short",
        description="Short baseline run with privileged substrate cues and isotropic medium switching.",
        observation=ObservationConfig(
            expose_environment=True,
            expose_viscosity=True,
            expose_target=True,
        ),
        anisotropy=AnisotropyConfig(
            mode="off",
            drag_ratio=10.0,
            tangential_gain=0.02,
            normal_gain=0.2,
            quadratic_drag=False,
            apply_in_water=False,
            apply_in_land=True,
        ),
        reward=RewardConfig(),
        training=TrainingConfig(),
        tags=["baseline", "short_run", "micro_publication"],
    )


def build_experiment(name: str) -> ExperimentConfig:
    experiment = _base_experiment()

    if name == "baseline_short":
        return experiment

    if name == "cue_ablation_short":
        experiment.name = name
        experiment.description = "Short run with privileged environment and viscosity observations hidden."
        experiment.observation.expose_environment = False
        experiment.observation.expose_viscosity = False
        experiment.tags = ["cue_ablation", "short_run", "micro_publication"]
        return experiment

    if name == "anisotropy_proxy_short":
        experiment.name = name
        experiment.description = "Short run with directional drag proxy on land."
        experiment.anisotropy.mode = "proxy"
        experiment.anisotropy.drag_ratio = 4.0
        experiment.anisotropy.normal_gain = 0.06
        experiment.anisotropy.tangential_gain = 0.015
        experiment.tags = ["anisotropy_proxy", "short_run", "micro_publication"]
        return experiment

    if name == "anisotropy_full_short":
        experiment.name = name
        experiment.description = "Short run with full per-segment anisotropic drag model enabled."
        experiment.anisotropy.mode = "full"
        experiment.anisotropy.apply_in_water = True
        experiment.anisotropy.apply_in_land = True
        experiment.anisotropy.quadratic_drag = True
        experiment.anisotropy.drag_ratio = 3.0
        experiment.anisotropy.normal_gain = 0.04
        experiment.anisotropy.tangential_gain = 0.01
        experiment.tags = ["anisotropy_full", "short_run", "micro_publication"]
        return experiment

    if name == "report_only_matrix":
        experiment.name = name
        experiment.description = "Manifest-only pseudo-experiment for generating the run matrix."
        experiment.training.training_steps = 0
        experiment.tags = ["report_only", "matrix", "micro_publication"]
        return experiment

    raise ValueError(f"Unknown experiment preset: {name}")


def list_experiments():
    return [
        "baseline_short",
        "cue_ablation_short",
        "anisotropy_proxy_short",
        "anisotropy_full_short",
        "report_only_matrix",
    ]
