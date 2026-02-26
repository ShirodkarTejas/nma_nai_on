#!/usr/bin/env python3
"""Strict 4-step ablation runner with direct-to-results logging."""

# gym_bridge must be the first real import.
try:
    import NMAP.gym_bridge  # noqa: F401
except ModuleNotFoundError:
    import gym_bridge  # noqa: F401

import os
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = ROOT / "results"

# ---------------------------------------------------------------------------
# Architecture parameters — edit here to change geometry for all experiments.
# ---------------------------------------------------------------------------
NUM_SEGMENTS = 6          # Number of swimmer body segments (links)


EXPERIMENTS = [
    {
        "name": "01_baseline",
        "cmd": [
            "python",
            "-m", "NMAP.main",
            "--mode", "train_curriculum",
            "--training_steps", "2000000",
            "--algorithm", "ppo",
            "--n_links", str(NUM_SEGMENTS),
            "--use_locomotion_only_early_training",
        ],
    },
    {
        "name": "02_oscillation",
        "cmd": [
            "python",
            "-m", "NMAP.main",
            "--mode", "train_curriculum",
            "--training_steps", "2000000",
            "--algorithm", "ppo",
            "--n_links", str(NUM_SEGMENTS),
            "--use_locomotion_only_early_training",
            "--force_oscillation",
        ],
    },
    {
        "name": "03_init",
        "cmd": [
            "python",
            "-m", "NMAP.main",
            "--mode", "train_curriculum",
            "--training_steps", "2000000",
            "--algorithm", "ppo",
            "--n_links", str(NUM_SEGMENTS),
            "--use_locomotion_only_early_training",
            "--force_oscillation",
            "--sparse_init",
        ],
    },
    {
        "name": "04_full_nmap",
        "cmd": [
            "python",
            "-m", "NMAP.main",
            "--mode", "train_curriculum",
            "--training_steps", "2000000",
            "--algorithm", "ppo",
            "--n_links", str(NUM_SEGMENTS),
            "--use_locomotion_only_early_training",
            "--force_oscillation",
            "--sparse_init",
            "--sparse_reg_lambda", "0.05",
        ],
    },
]


def _build_subprocess_env() -> dict[str, str]:
    """Build environment vars so subprocesses can import NMAP and tonic."""
    env = os.environ.copy()
    pythonpath_parts = [
        str(Path.cwd().resolve()),
        str(ROOT.resolve()),
        str(ROOT.parent.resolve()),
        str((ROOT / "tonic").resolve()),
    ]
    existing = env.get("PYTHONPATH")
    if existing:
        pythonpath_parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    return env


def run_all_experiments() -> None:
    os.chdir(ROOT)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    env = _build_subprocess_env()

    legacy_outputs = ROOT / "outputs"
    if legacy_outputs.exists():
        shutil.rmtree(legacy_outputs)

    for exp in EXPERIMENTS:
        exp_name = exp["name"]
        result_dir = RESULTS_ROOT / exp_name

        if result_dir.exists():
            shutil.rmtree(result_dir)
        result_dir.mkdir(parents=True, exist_ok=True)

        cmd = exp["cmd"] + ["--log_dir", f"results/{exp_name}"]

        print("\n" + "=" * 100)
        print(f"STARTING EXPERIMENT: {exp_name}")
        print(f"COMMAND: {' '.join(cmd)}")
        print("=" * 100)

        subprocess.run(cmd, check=True, env=env)

        print("=" * 100)
        print(f"SUCCESS: {exp_name}")
        print(f"RESULTS WRITTEN DIRECTLY TO: {result_dir}")
        print("=" * 100)


if __name__ == "__main__":
    run_all_experiments()
