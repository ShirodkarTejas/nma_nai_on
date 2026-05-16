#!/usr/bin/env python3
"""Metrics and run-summary helpers for micro-publication experiments."""

from typing import Dict, List
import os
import re

import numpy as np


def summarize_scalar_series(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    arr = np.asarray(values, dtype=np.float32)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


PHASE_NAME_TO_ID = {
    "Pure Swimming": 0,
    "Single Land Zone": 1,
    "Two Land Zones": 2,
    "Full Complexity": 3,
}


def _safe_mean(values):
    values = [float(v) for v in values if v is not None]
    return float(np.mean(values)) if values else 0.0


def _phase_success(phase_id: int, final_performance: Dict, trajectory: Dict) -> Dict:
    mean_distance = float(final_performance.get("mean_distance", 0.0))
    mean_reward = float(final_performance.get("mean_reward", 0.0))
    transitions = int(trajectory.get("transitions", 0))
    water_time = int(trajectory.get("water_time", 0))
    land_time = int(trajectory.get("land_time", 0))
    total_time = max(water_time + land_time, 1)
    land_fraction = land_time / total_time

    if phase_id == 0:
        successful = mean_distance > 0.02
    else:
        successful = transitions > 0 and water_time > 0 and land_time > 0

    return {
        "successful": bool(successful),
        "success_label": "pass" if successful else "fail",
        "transitions": transitions,
        "water_time": water_time,
        "land_time": land_time,
        "land_fraction": float(land_fraction),
        "mean_distance": mean_distance,
        "mean_reward": mean_reward,
    }


def parse_training_summary(path: str) -> Dict:
    if not path or not os.path.exists(path):
        return {"final_performance": {}, "trajectory": {}}

    with open(path, "r", encoding="utf-8") as handle:
        lines = [line.rstrip() for line in handle]

    final_performance = {}
    trajectory = {}
    section = None
    current_phase = None

    phase_header_pattern = re.compile(r"^\*\*(.+?)\*\*:")
    perf_pattern = re.compile(r"Mean Distance: ([0-9.]+)m ± ([0-9.]+)")
    reward_pattern = re.compile(r"Mean Reward: ([0-9.\-]+) ± ([0-9.]+)")
    final_distance_pattern = re.compile(r"Final Distance: ([0-9.]+)m")
    velocity_pattern = re.compile(r"Max Velocity: ([0-9.]+)")
    transitions_pattern = re.compile(r"Environment Transitions: (\d+)")
    water_time_pattern = re.compile(r"Time in Water: (\d+) steps")
    land_time_pattern = re.compile(r"Time on Land: (\d+) steps")

    for line in lines:
        if line.startswith("## Final Performance by Phase"):
            section = "final"
            current_phase = None
            continue
        if line.startswith("## Trajectory Analysis"):
            section = "trajectory"
            current_phase = None
            continue
        if line.startswith("## Training Progress"):
            section = None
            current_phase = None
            continue

        header_match = phase_header_pattern.match(line.strip())
        if header_match and section in {"final", "trajectory"}:
            phase_name = header_match.group(1)
            if phase_name in PHASE_NAME_TO_ID:
                current_phase = str(PHASE_NAME_TO_ID[phase_name])
                if section == "final":
                    final_performance.setdefault(current_phase, {})
                else:
                    trajectory.setdefault(current_phase, {})
            continue

        if current_phase is None:
            continue

        stripped = line.strip()
        if section == "final":
            match = perf_pattern.search(stripped)
            if match:
                final_performance[current_phase]["mean_distance"] = float(match.group(1))
                final_performance[current_phase]["std_distance"] = float(match.group(2))
                continue
            match = reward_pattern.search(stripped)
            if match:
                final_performance[current_phase]["mean_reward"] = float(match.group(1))
                final_performance[current_phase]["std_reward"] = float(match.group(2))
                continue

        if section == "trajectory":
            match = final_distance_pattern.search(stripped)
            if match:
                trajectory[current_phase]["final_distance"] = float(match.group(1))
                continue
            match = velocity_pattern.search(stripped)
            if match:
                trajectory[current_phase]["max_velocity"] = float(match.group(1))
                continue
            match = transitions_pattern.search(stripped)
            if match:
                trajectory[current_phase]["transitions"] = int(match.group(1))
                continue
            match = water_time_pattern.search(stripped)
            if match:
                trajectory[current_phase]["water_time"] = int(match.group(1))
                continue
            match = land_time_pattern.search(stripped)
            if match:
                trajectory[current_phase]["land_time"] = int(match.group(1))
                continue

    return {"final_performance": final_performance, "trajectory": trajectory}


def build_experiment_summary(eval_results: Dict, parsed_summary: Dict = None) -> Dict:
    phases = {}
    for phase, results in sorted(eval_results.items()):
        phases[str(phase)] = dict(results)

    parsed_summary = parsed_summary or {}
    final_performance = parsed_summary.get("final_performance", {})
    trajectory = parsed_summary.get("trajectory", {})

    phase_metrics = {}
    for phase_id in range(4):
        key = str(phase_id)
        phase_metrics[key] = _phase_success(
            phase_id=phase_id,
            final_performance=final_performance.get(key, {}),
            trajectory=trajectory.get(key, {}),
        )

    summary = {
        "phases": phases,
        "mean_phase_distance": float(np.mean([
            results.get("mean_distance", 0.0) for results in eval_results.values()
        ])) if eval_results else 0.0,
        "mean_phase_reward": float(np.mean([
            results.get("mean_reward", 0.0) for results in eval_results.values()
        ])) if eval_results else 0.0,
        "phase_metrics": phase_metrics,
        "final_performance": final_performance,
        "trajectory": trajectory,
        "mixed_phase_success_rate": _safe_mean([
            1.0 if phase_metrics[str(phase)]["successful"] else 0.0
            for phase in [1, 2, 3]
        ]),
        "mixed_phase_transition_mean": _safe_mean([
            phase_metrics[str(phase)]["transitions"] for phase in [1, 2, 3]
        ]),
    }
    return summary
