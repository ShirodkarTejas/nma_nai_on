#!/usr/bin/env python3
"""
Sparse Regularization Lambda Sweep — Analysis
=============================================

Analyzes experiments of the form:
    sweep_lambda_<value>

Where λ controls topological regularization strength.
"""

from __future__ import annotations

import json
import math
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
step_count = 1 # in M steps, used for parsing experiment names
SWEEP_PREFIXES = [
    "sweep_nmap_lambda",
]

PHASE_NAMES = ["Pure Swimming", "Single Land Zone", "Two Land Zones", "Full Complexity"]

# Add your target lambdas here. Set to None to run all.
TARGET_LAMBDAS = None

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_lambda_from_name(name: str):
    match = re.search(rf"lambda_(\d+p\d+)_{step_count}Msteps", name)
    if not match:
        return None
    return float(match.group(1).replace("p", "."))

def find_experiments(prefix: str, base_dir, target_lambdas=None):
    experiments = []
    lambda_map = {}

    for path in base_dir.iterdir():
        if not path.is_dir():
            continue

        name = path.name

        if not name.startswith(prefix):
            continue

        lam = parse_lambda_from_name(name)
        if lam is None:
            continue

        # Filter by target lambdas if specified
        if target_lambdas is not None:
            if not any(math.isclose(lam, t_lam, rel_tol=1e-5) for t_lam in target_lambdas):
                continue

        experiments.append(name)
        lambda_map[name] = lam

    # sort by lambda 
    experiments.sort(key=lambda x: lambda_map[x])

    return experiments, lambda_map

def _metrics_path(exp: str) -> Optional[Path]:
    # FIXED: Added "results" to the path
    log_dir = SCRIPT_DIR / "results" / exp / "curriculum_training" / "logs" / "enhanced_ncap"
    if not log_dir.exists():
        return None
    for sub in sorted(log_dir.iterdir()):
        p = sub / "metrics.json"
        if p.exists():
            return p
    return None

def _load_metrics(exp: str):
    p = _metrics_path(exp)
    if p is None:
        return {}
    return json.load(open(p))

def _series(metrics: dict, key: str):
    entries = metrics.get(key, [])
    if not entries:
        return np.array([]), np.array([])
    steps = np.array([e["step"] for e in entries], float)
    vals  = np.array([e["value"] for e in entries], float)
    return steps, vals

def _final_eval(exp: str):
    # FIXED: Added "results" to the path
    ckpt_dir = SCRIPT_DIR / "results" / exp / "curriculum_training" / "checkpoints" / "enhanced_ncap"
    if not ckpt_dir.exists():
        return None
    
    pts = list(ckpt_dir.glob("*.pt"))
    if not pts:
        return None

    def get_step(p):
        m = re.search(r"_step_(\d+)", p.name)
        return int(m.group(1)) if m else -1
    
    pts.sort(key=get_step)

    try:
        import torch
        ckpt = torch.load(pts[-1], map_location="cpu", weights_only=False)
        er = ckpt.get("eval_results")
        if not er:
            return None
        return {int(k): v for k, v in er.items()}
    except Exception as e:
        warnings.warn(f"Failed to load {pts[-1]}: {e}")
        return None

# ---------------------------------------------------------------------------
# Plot 1 — Learning curves
# ---------------------------------------------------------------------------
def plot_learning_curves(all_metrics, EXPERIMENTS, LAMBDA_MAP, COLORS, OUT_DIR):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Learning Curves — Lambda Sweep")
    ax.set_xlabel("Steps (×10⁶)")
    ax.set_ylabel("Mean Distance")

    plotted_anything = False
    for exp in EXPERIMENTS:
        m = all_metrics.get(exp, {})
        steps, vals = _series(m, "mean_distance_10")
        if not len(steps):
            continue

        lam = LAMBDA_MAP[exp]
        ax.plot(steps / 1e6, vals,
                color=COLORS[exp],
                label=f"{lam}")
        plotted_anything = True

    if plotted_anything:
        ax.legend(title="λ")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "01_learning_curves.png", dpi=150)
    plt.close()

# ---------------------------------------------------------------------------
# Plot 2 — Lambda vs final performance
# ---------------------------------------------------------------------------
def plot_lambda_performance(all_metrics, all_evals, EXPERIMENTS, LAMBDA_MAP, OUT_DIR):
    lambdas = []
    final_train_dist = []
    final_eval_dist = []

    for exp in EXPERIMENTS:
        lam = LAMBDA_MAP[exp]

        m = all_metrics.get(exp, {})
        _, dist = _series(m, "mean_distance_10")
        if len(dist):
            final_train_dist.append(dist[-1])
        else:
            final_train_dist.append(np.nan)

        er = all_evals.get(exp)
        if er:
            vals = [er[p]["mean_distance"] for p in er]
            final_eval_dist.append(np.mean(vals))
        else:
            final_eval_dist.append(np.nan)

        lambdas.append(lam)

    lambdas = np.array(lambdas)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(lambdas, final_train_dist, "o-", label="Train Distance")
    ax.plot(lambdas, final_eval_dist, "o-", label="Eval Distance")

    ax.set_xscale("log")
    ax.set_xlabel("λ (log scale)")
    ax.set_ylabel("Distance")
    ax.set_title("Distance vs Regularization Strength")

    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "02_lambda_performance.png", dpi=150)
    plt.close()

# ---------------------------------------------------------------------------
# Plot 3 — Phase-specific performance vs λ (Distance & Reward)
# ---------------------------------------------------------------------------
def plot_lambda_phase_performance(all_evals, EXPERIMENTS, LAMBDA_MAP, OUT_DIR):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sorted_exps = sorted(EXPERIMENTS, key=lambda e: LAMBDA_MAP[e])
    lams = [LAMBDA_MAP[e] for e in sorted_exps]

    for phase in range(4):
        vals = []
        for exp in sorted_exps:
            er = all_evals.get(exp)
            if er and phase in er:
                vals.append(er[phase]["mean_distance"])
            else:
                vals.append(np.nan)

        axes[0].plot(lams, vals, "o-", label=PHASE_NAMES[phase])

    axes[0].set_xscale("log")
    axes[0].set_xlabel("λ (log scale)")
    axes[0].set_ylabel("Distance")
    axes[0].set_title("Eval Distance vs λ")
    axes[0].legend()

    for phase in range(4):
        vals = []
        for exp in sorted_exps:
            er = all_evals.get(exp)
            if er and phase in er:
                vals.append(er[phase]["mean_reward"])
            else:
                vals.append(np.nan)

        axes[1].plot(lams, vals, "o-", label=PHASE_NAMES[phase])

    axes[1].set_xscale("log")
    axes[1].set_xlabel("λ (log scale)")
    axes[1].set_ylabel("Reward")
    axes[1].set_title("Eval Reward vs λ")
    axes[1].legend()

    fig.suptitle("Phase-wise Eval Performance vs λ")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "03_lambda_phase_performance.png", dpi=150)
    plt.close()
    
# ---------------------------------------------------------------------------
# Plot 4 — Summary table
# ---------------------------------------------------------------------------
def plot_summary_table(all_metrics, all_evals, EXPERIMENTS, LAMBDA_MAP, OUT_DIR):
    rows = []

    for exp in EXPERIMENTS:
        lam = LAMBDA_MAP[exp]
        m = all_metrics.get(exp, {})
        _, dist = _series(m, "mean_distance_10")
        final_train = dist[-1] if len(dist) else np.nan

        er = all_evals.get(exp)
        if er:
            mean_eval_dist = np.mean([er[p]["mean_distance"] for p in er])
            mean_eval_reward = np.mean([er[p]["mean_reward"] for p in er])
        else:
            mean_eval_dist, mean_eval_reward = np.nan, np.nan

        rows.append([lam, final_train, mean_eval_dist, mean_eval_reward])

    if not rows:                          
        print("  Skipping summary table: no data.")
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    table = ax.table(
        cellText=[[f"{r[0]}", f"{r[1]:.3f}", f"{r[2]:.3f}", f"{r[3]:.1f}"] for r in rows],
        colLabels=["λ", "Final Train Dist", "Mean Eval Dist", "Mean Eval Reward"],
        loc="center"
    )
    fig.savefig(OUT_DIR / "04_summary_table.png", dpi=150)
    plt.close()


def run_analysis(prefix: str):
    print(f"\n=== Running analysis for {prefix} ===")

    OUT_DIR = SCRIPT_DIR / 'results' / f"analysis_plots_{prefix}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Pass TARGET_LAMBDAS down here
    EXPERIMENTS, LAMBDA_MAP = find_experiments(prefix, SCRIPT_DIR / "results", target_lambdas=TARGET_LAMBDAS)

    if not EXPERIMENTS:                  
        print(f"  No experiments found for prefix '{prefix}' matching target lambdas, skipping.")
        return

    print("\nDetected experiments:")
    for exp in EXPERIMENTS:
        print(f"  {exp}  ->  λ={LAMBDA_MAP[exp]}")
        
    # FIXED: The loading and plotting was incorrectly indented inside the print loop.
    cmap = plt.get_cmap("viridis")
    COLORS = {
        exp: cmap(i / (len(EXPERIMENTS) - 1)) if len(EXPERIMENTS) > 1 else cmap(0)
        for i, exp in enumerate(EXPERIMENTS)
    }

    # Load data once
    all_metrics = {exp: _load_metrics(exp) for exp in EXPERIMENTS}
    all_evals   = {exp: _final_eval(exp) for exp in EXPERIMENTS}

    # Plot once
    plot_learning_curves(all_metrics, EXPERIMENTS, LAMBDA_MAP, COLORS, OUT_DIR)
    plot_lambda_performance(all_metrics, all_evals, EXPERIMENTS, LAMBDA_MAP, OUT_DIR)
    plot_lambda_phase_performance(all_evals, EXPERIMENTS, LAMBDA_MAP, OUT_DIR)
    plot_summary_table(all_metrics, all_evals, EXPERIMENTS, LAMBDA_MAP, OUT_DIR)

    print(f"Done. Outputs in {OUT_DIR}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Running sweep analyses for target lambdas: {TARGET_LAMBDAS}...")

    for prefix in SWEEP_PREFIXES:
        run_analysis(prefix)

    print("\nAll analyses complete.")

if __name__ == "__main__":
    main()