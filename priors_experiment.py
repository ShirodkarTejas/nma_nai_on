#!/usr/bin/env python3
"""Connectomic-priors ablation: 5 conditions that isolate each contribution.

Ablation design — each condition adds exactly one variable:
  01_ncap_baseline   — canonical NCAP, random init, no priors, no osc. enforcement
  02_ncap_sparse_init — + Cook-2019 synapse-count weight initialisation only
  03_ncap_reg_only   — + topological L2 regularisation only (no sparse init)
  04_ncap_full_priors — + both init AND regularisation (full connectomic priors)
  05_ncap_full_nmap  — + forced-oscillation variance penalty (complete NMAP stack)

Reading the ablation:
  02 vs 01 → contribution of connectomic initialisation alone
  03 vs 01 → contribution of wiring-economy regularisation alone
  04 vs 01 → combined connectomic prior contribution
  04 vs 02 → marginal gain from adding regularisation on top of init
  04 vs 03 → marginal gain from adding init on top of regularisation
  05 vs 04 → contribution of oscillation enforcement (upstream NMAP addition)

Environment: MicroPublicationSwimCrawl (micro_publication_env.py)
  • Clean, separated reward: compute_navigation_reward + combine_reward_components
  • progress_weight=2.0, completion_reward=10.0 (no 500-point jackpot)
  • land_start_probability=0.35, max_steps=600 per episode

Key settings:
  • n_links=6 
  • TRAINING_STEPS=3M 
  • sparse_reg_lambda=0.05
  • action_scaling_factor=1.8 
"""

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
# Experiment parameters — edit here to change shared values for all runs.
# ---------------------------------------------------------------------------
NUM_SEGMENTS   = 6           # Number of swimmer body segments (links) — matches reference repo
TRAINING_STEPS = 1_000_000   # matches reference repo recommendation
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


EXPERIMENTS = [
    # ------------------------------------------------------------------
    # 01: Canonical NCAP — shared weights, random init, no priors.
    # Establishes what the biological oscillator circuit achieves on its own.
    # ------------------------------------------------------------------
    {
        "name": "01_ncap_baseline",
        "cmd": _base_cmd(),
    },
    # ------------------------------------------------------------------
    # 02: + Cook-2019 weight initialisation only (no regularisation).
    # Adds sparse_init: forces use_weight_sharing=False and seeds weights
    # from mean synapse counts per pathway. Isolates init contribution.
    # ------------------------------------------------------------------
    {
        "name": "02_ncap_sparse_init",
        "cmd": _base_cmd(["--sparse_init"]),
    },
    # ------------------------------------------------------------------
    # 03: + topological L2 regularisation only (no sparse init).
    # Penalises long-range connections by anatomical distance.
    # Isolates regularisation contribution independent of init.
    # ------------------------------------------------------------------
    {
        "name": "03_ncap_reg_only",
        "cmd": _base_cmd(["--sparse_reg_lambda", "0.05"]),
    },    
    # ------------------------------------------------------------------
    # 04: + both init and regularisation (full connectomic priors).
    # Combines Cook-2019 init + wiring-economy L2 penalty.
    # 02 vs 04 shows reg contribution; 03 vs 04 shows init contribution.
    # ------------------------------------------------------------------
    {
        "name": "04_ncap_full_priors",
        "cmd": _base_cmd(["--sparse_init", "--sparse_reg_lambda", "0.002875"]),
    },
    # ------------------------------------------------------------------
    # 05: + forced-oscillation variance penalty (full NMAP stack).
    # Adds minimum action-variance penalty (force_oscillation) on top of
    # both connectomic priors.  Oscillation term is the main upstream
    # addition in the reference repo beyond baseline NCAP.
    # ------------------------------------------------------------------
    {
        "name": "05_ncap_full_nmap",
        "cmd": _base_cmd([
            "--sparse_init",
            "--sparse_reg_lambda", "0.005", 
            "--force_oscillation",
        ]),
    }

    
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


def run_analysis() -> None:
    """Run the ablation analysis script to generate plots and RESULTS.md."""
    plot_script = ROOT / "plot_ablation.py"
    if not plot_script.exists():
        print(f"\nWARNING: Analysis script not found at {plot_script}; skipping plots.")
        return

    env = _build_subprocess_env()
    analysis_dir = RESULTS_ROOT / "analysis_plots"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 100)
    print("RUNNING ABLATION ANALYSIS")
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
        print(f"  Report : {RESULTS_ROOT}/RESULTS.md")
        print("=" * 100)
    else:
        print(f"WARNING: Analysis script exited with code {result.returncode}.")
        print("  Training results are intact; re-run `python results/plot_ablation.py` manually.")


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

    # -----------------------------------------------------------------------
    # Post-training: generate analysis plots and RESULTS.md automatically.
    # -----------------------------------------------------------------------
    run_analysis()
    

if __name__ == "__main__":
    run_all_experiments()
