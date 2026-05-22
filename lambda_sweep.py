#!/usr/bin/env python3
"""
lambda hyperparameter tuning experiment and plotting
"""

# gym_bridge must be the first real import.
try:
    import NMAP.gym_bridge  # noqa: F401
except ModuleNotFoundError:
    import gym_bridge  # noqa: F401

import os
import numpy as np
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = ROOT / "results"

# ---------------------------------------------------------------------------
# Experiment parameters 
# ---------------------------------------------------------------------------
NUM_SEGMENTS   = 6           # Number of swimmer body segments (links)
TRAINING_STEPS = 1_000_000   # Number of training steps
SAVE_STEPS     = 100_000     # Checkpoint cadence (every 100 k)
LOG_EPISODES   = 50          # Logging cadence


def _base_cmd(extra_flags=None):
    """Return the common CLI prefix shared by all experiments."""
    cmd = [
        "python",
        "-m", "NMAP.main",
        "--mode", "train_curriculum",
        "--training_steps", str(TRAINING_STEPS),
        "--save_steps",     str(SAVE_STEPS),
        "--log_episodes",   str(LOG_EPISODES),
        "--algorithm",      "ppo",
        "--n_links",        str(NUM_SEGMENTS),
        "--use_locomotion_only_early_training",
    ]
    if extra_flags:
        cmd.extend(extra_flags)
    return cmd

LAMBDA_SWEEP =  [0.0005, 0.001,0.002]
# np.linspace(0.0009, 0.02, num=10).tolist()



SWEEP_EXPERIMENTS = []

for lam in LAMBDA_SWEEP:

    SWEEP_EXPERIMENTS.append({
        "name": f"sweep_lambda_{str(lam).replace('.', 'p')}",
        "cmd": _base_cmd([ "--sparse_reg_lambda", str(lam)]),
    })

    SWEEP_EXPERIMENTS.append({
        "name": f"sweep_full_lambda_{str(lam).replace('.', 'p')}_{TRAINING_STEPS//1_000_000}Msteps",
        "cmd": _base_cmd(["--sparse_init", "--sparse_reg_lambda", str(lam)]),
    })

    SWEEP_EXPERIMENTS.append(
        {"name": f"sweep_nmap_lambda_{str(lam).replace('.', 'p')}_{TRAINING_STEPS//1_000_000}Msteps",
        "cmd": _base_cmd([
            "--sparse_init",
            "--sparse_reg_lambda", str(lam),
            "--force_oscillation"])}
        )




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


def run_analysis() -> None:
    """Run the lambda sweep analysis script to generate plots."""
    plot_script = ROOT / "plot_lambda_sweep.py"
    if not plot_script.exists():
        print(f"\nWARNING: Analysis script not found at {plot_script}; skipping plots.")
        return

    env = _build_subprocess_env()
    analysis_dir = RESULTS_ROOT / "analysis_plots"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 100)
    print("PLOTTING THE RESULTS OF THE LAMBDA SWEEP")
    print(f"  Script : {plot_script}")
    print(f"  Output : {analysis_dir}/")
    print("=" * 100)

    result = subprocess.run(
        ["python", str(plot_script)],
        env=env,
        check=False,
    )
    if result.returncode == 0:
        print("=" * 100)
        print("ANALYSIS COMPLETE")
        print(f"  Plots  : {analysis_dir}/")
        print("=" * 100)
    else:
        print(f"WARNING: Analysis script exited with code {result.returncode}.")
        print("  Training results are intact; re-run `python results/plot_lambda_sweep.py` manually.")


def run_all_experiments() -> None:

    os.chdir(ROOT)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    env = _build_subprocess_env()

    legacy_outputs = ROOT / "outputs"
    if legacy_outputs.exists():
        shutil.rmtree(legacy_outputs)

    for exp in SWEEP_EXPERIMENTS:
        exp_name = exp["name"]
        result_dir = RESULTS_ROOT / exp_name

        if result_dir.exists():
            print(f"Skipping experiment '{exp_name}'.")
            continue
        else:
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

    # -----------------------------------------------------------------------
    # Post-training: generate analysis plots and RESULTS.md automatically.
    # -----------------------------------------------------------------------
    run_analysis()


if __name__ == "__main__":
    run_all_experiments()
