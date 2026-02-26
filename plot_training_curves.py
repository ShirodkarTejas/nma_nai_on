#!/usr/bin/env python3
"""
Comparative training-curve plotter for all NMAP ablation experiments.

Supports two log formats produced by this codebase:

1. Tonic log.csv  — written by tonic.logger to  results/<exp>/tonic/log.csv
   Columns include steps (integer) and episode/score or train/episode_score.

2. TrainingLogger metrics.json — written by CurriculumNCAPTrainer to
   results/<exp>/curriculum_training/logs/**/metrics.json
   Format: {"episode_reward": [{"step": N, "value": V, "timestamp": T}, ...], ...}

The script walks results/ for every recognised log file, extracts a (steps, reward)
series, applies a rolling mean (window=10 by default), and plots all runs on one
figure saved to results/comparative_training_curves.png.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RESULTS_ROOT = Path(__file__).resolve().parent / "results"
OUTPUT_PATH = RESULTS_ROOT / "comparative_training_curves.png"
ROLLING_WINDOW = 10

# Ordered list of column names to try when reading a tonic log.csv file.
# The first match wins.
TONIC_REWARD_COLS = [
    "train/episode_score",
    "train/episode_score/mean",
    "episode_score",
    "score",
    "reward",
    "episode_reward",
]
TONIC_STEP_COLS = ["steps", "step", "train/steps"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """Return the first candidate column name that exists in *df*."""
    for col in candidates:
        if col in df.columns:
            return col
    # Fall back: try case-insensitive prefix match
    lower_cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_cols:
            return lower_cols[cand.lower()]
    return None


def load_tonic_csv(path: Path) -> Optional[pd.DataFrame]:
    """
    Load a tonic log.csv and return a DataFrame with columns [steps, reward].
    Returns None when the file is empty or the required columns are missing.
    """
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"  [warn] Could not read {path}: {exc}")
        return None

    if df.empty:
        print(f"  [warn] {path} is empty — skipping.")
        return None

    step_col = _find_column(df, TONIC_STEP_COLS)
    reward_col = _find_column(df, TONIC_REWARD_COLS)

    if step_col is None or reward_col is None:
        print(f"  [warn] {path}: could not find step/reward columns "
              f"(available: {list(df.columns)}) — skipping.")
        return None

    out = pd.DataFrame({"steps": pd.to_numeric(df[step_col], errors="coerce"),
                        "reward": pd.to_numeric(df[reward_col], errors="coerce")})
    out = out.dropna().reset_index(drop=True)
    return out if not out.empty else None


def load_metrics_json(path: Path) -> Optional[pd.DataFrame]:
    """
    Load a TrainingLogger metrics.json and return a DataFrame with
    columns [steps, reward].
    Returns None when the file is empty or rewards are missing.
    """
    try:
        with open(path) as fh:
            data = json.load(fh)
    except Exception as exc:
        print(f"  [warn] Could not read {path}: {exc}")
        return None

    # Prefer mean_reward_10 (smoothed in-training) over raw episode_reward.
    for key in ("mean_reward_10", "episode_reward", "reward"):
        if key in data and data[key]:
            records = data[key]
            steps = [r.get("step", i) for i, r in enumerate(records)]
            values = [r.get("value", float("nan")) for r in records]
            out = pd.DataFrame({"steps": steps, "reward": values})
            out = out.dropna().reset_index(drop=True)
            if not out.empty:
                return out

    print(f"  [warn] {path}: no usable reward metric found — skipping.")
    return None


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_runs(results_root: Path) -> dict[str, pd.DataFrame]:
    """
    Walk *results_root* and collect one time-series per experiment.

    Priority order for each experiment directory:
      1. curriculum_training/logs/**/metrics.json  (CurriculumNCAPTrainer)
      2. tonic/log.csv                              (NCAPTrainer / Tonic)
      3. logs/**/metrics.json                       (NCAPTrainer TrainingLogger)
    """
    runs: dict[str, pd.DataFrame] = {}

    if not results_root.exists():
        print(f"[warn] Results directory not found: {results_root}")
        return runs

    for exp_dir in sorted(results_root.iterdir()):
        if not exp_dir.is_dir():
            continue
        exp_name = exp_dir.name

        # --- Attempt 1: curriculum TrainingLogger metrics.json ---------------
        curriculum_logs = exp_dir / "curriculum_training" / "logs"
        if curriculum_logs.exists():
            json_files = sorted(curriculum_logs.rglob("metrics.json"))
            for jf in json_files:
                df = load_metrics_json(jf)
                if df is not None:
                    print(f"  [ok] {exp_name}: loaded curriculum metrics.json "
                          f"({len(df)} rows) from {jf.relative_to(results_root)}")
                    runs[exp_name] = df
                    break  # Use the first valid file found

        if exp_name in runs:
            continue

        # --- Attempt 2: Tonic log.csv ----------------------------------------
        tonic_csv = exp_dir / "tonic" / "log.csv"
        if tonic_csv.exists():
            df = load_tonic_csv(tonic_csv)
            if df is not None:
                print(f"  [ok] {exp_name}: loaded tonic log.csv ({len(df)} rows)")
                runs[exp_name] = df
                continue

        # --- Attempt 3: NCAPTrainer TrainingLogger metrics.json --------------
        trainer_logs = exp_dir / "logs"
        if trainer_logs.exists():
            json_files = sorted(trainer_logs.rglob("metrics.json"))
            for jf in json_files:
                df = load_metrics_json(jf)
                if df is not None:
                    print(f"  [ok] {exp_name}: loaded trainer metrics.json "
                          f"({len(df)} rows) from {jf.relative_to(results_root)}")
                    runs[exp_name] = df
                    break

        if exp_name not in runs:
            print(f"  [skip] {exp_name}: no usable log found yet "
                  "(run will appear once training produces data).")

    return runs


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_runs(runs: dict[str, pd.DataFrame],
              output_path: Path,
              window: int = ROLLING_WINDOW) -> None:
    """Plot all runs on one figure with rolling-mean smoothing."""

    if not runs:
        print("[info] No completed runs to plot. "
              "Start the experiments and re-run this script.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for idx, (name, df) in enumerate(sorted(runs.items())):
        color = colors[idx % len(colors)]
        steps = df["steps"].to_numpy()
        reward = df["reward"].to_numpy()

        # Raw trace (faint)
        ax.plot(steps, reward, color=color, alpha=0.20, linewidth=0.8)

        # Rolling mean (bold)
        smoothed = (pd.Series(reward)
                    .rolling(window=window, min_periods=1, center=True)
                    .mean()
                    .to_numpy())
        ax.plot(steps, smoothed, color=color, linewidth=2.0,
                label=f"{name}  (roll-{window})")

    ax.set_xlabel("Environment Steps", fontsize=12)
    ax.set_ylabel("Episode Reward", fontsize=12)
    ax.set_title("NMAP Ablation — Comparative Training Curves", fontsize=14)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[saved] {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Scanning {RESULTS_ROOT} for training logs ...\n")
    runs = discover_runs(RESULTS_ROOT)
    print(f"\nFound {len(runs)} run(s): {list(runs.keys())}")
    plot_runs(runs, OUTPUT_PATH)


if __name__ == "__main__":
    main()
