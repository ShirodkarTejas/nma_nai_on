#!/usr/bin/env python3
"""Connectomic-Priors Ablation — unified analysis script.

Data sources (auto-discovered per condition):
  metrics.json      — mean_distance_10, mean_reward_10, phase, neuromod_variance
                      logged every ~50 episodes during training
  checkpoints.json  — per-phase distance + reward evaluated at each checkpoint step
  *.pt files        — model state dicts for weight-evolution plot (optional / slow)

Plots written to results/analysis_plots/:
  01_training_curves.png        dual-axis reward + distance over training steps
  02_checkpoint_progression.png distance per phase across all checkpoint steps (2×2)
  03_phase_bars.png             final-checkpoint distance bars grouped by phase
  04_reward_vs_distance.png     reward vs distance scatter per phase (gaming diagnostic)
  05_summary_bar.png            overall mean distance per condition (single number)
  06_neuromod_variance.png      neuromodulation signal variance diagnostic
  07_prior_diagnostics.png      Cook-2019 synapse counts + soma distances
  08_weight_evolution.png       key parameter trajectories over checkpoints (needs .pt)

RESULTS.md — quantitative summary table + interpretation guide

Ablation reading guide:
  02 vs 01 → connectomic init alone
  03 vs 01 → wiring-economy regularisation alone
  04 vs 01 → full connectomic priors (init + reg)
  04 vs 02 → marginal gain of reg on top of init
  04 vs 03 → marginal gain of init on top of reg
  05 vs 04 → oscillation enforcement (upstream NMAP)
"""

from __future__ import annotations

import json
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NMAP_ROOT = Path(__file__).resolve().parent      # …/NMAP/
RESULTS_ROOT = NMAP_ROOT / "results"   # …/NMAP/results/              
OUTPUT_DIR   = RESULTS_ROOT / "analysis_plots"

EXPERIMENTS: List[Tuple[str, str, str]] = [
    ("01_ncap_baseline",    "01 Baseline",    "#4C72B0"),
    ("02_ncap_sparse_init", "02 Sparse Init", "#DD8452"),
    ("03_ncap_reg_only",    "03 Reg Only",    "#55A868"),
    ("04_ncap_full_priors", "04 Full Priors", "#C44E52"),
    ("05_ncap_full_nmap",   "05 Full NMAP",   "#8172B2"),
]

PHASE_LABELS = {
    0: "Ph0: Pure Swimming",
    1: "Ph1: Single Land",
    2: "Ph2: Two Land Zones",
    3: "Ph3: Full Complexity",
}

PHASE_COLORS = ["#2196F3", "#FF9800", "#4CAF50", "#AA23C2"]
PHASE_BG     = ["#e8f4f8", "#fef9e7", "#eafaf1", "#fdf2f8"]

LOG_SUBDIR = "curriculum_training/logs/enhanced_ncap"
CKPT_SUBDIR = "curriculum_training/checkpoints/enhanced_ncap"

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor":   "white",
        "axes.facecolor":     "#FAFAFA",
        "axes.grid":          True,
        "grid.color":         "#E0E0E0",
        "grid.linestyle":     "--",
        "grid.linewidth":     0.7,
        "grid.alpha":         0.7,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.edgecolor":     "#333333",
        "font.family":        "sans-serif",
        "font.size":          12,
        "axes.titlesize":     13,
        "axes.labelsize":     11,
        "legend.fontsize":    9,
        "lines.linewidth":    2.0,
        "figure.dpi":         150,
    })


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _log_dir(exp_dir: Path) -> Optional[Path]:
    base = exp_dir / LOG_SUBDIR
    if not base.exists():
        return None
    candidates = sorted(base.iterdir())
    return candidates[0] if candidates else None


def load_metrics(exp_dir: Path) -> dict:
    d = _log_dir(exp_dir)
    if d is None:
        return {}
    p = d / "metrics.json"
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def load_checkpoints(exp_dir: Path) -> list:
    """Load checkpoints.json — list of dicts with step + performance_metrics."""
    d = _log_dir(exp_dir)
    if d is None:
        return []
    p = d / "checkpoints.json"
    if not p.exists():
        return []
    with open(p) as f:
        return json.load(f)


def load_pt_checkpoints(exp_dir: Path) -> Dict[int, dict]:
    """Load model state dicts from *.pt files (for weight evolution). Returns {} if none found."""
    ckpt_dir = exp_dir / CKPT_SUBDIR
    if not ckpt_dir.exists():
        return {}
    try:
        import torch
    except ImportError:
        return {}
    result = {}
    for pt in sorted(ckpt_dir.glob("*.pt")):
        m = re.search(r"_step_(\d+)", pt.name)
        if not m:
            continue
        step = int(m.group(1))
        try:
            ckpt = torch.load(pt, map_location="cpu", weights_only=False)
            if "model_state_dict" in ckpt:
                result[step] = ckpt["model_state_dict"]
        except Exception as exc:
            warnings.warn(f"Could not load {pt.name}: {exc}")
    return result


def collect_all() -> dict:
    """Load all conditions. Returns dict keyed by label with metrics, checkpoints, pt_history."""
    data = {}
    for exp_name, label, color in EXPERIMENTS:
        exp_dir = RESULTS_ROOT / exp_name
        if not exp_dir.exists():
            print(f"  [SKIP] {exp_name} — directory not found")
            continue
        metrics     = load_metrics(exp_dir)
        checkpoints = load_checkpoints(exp_dir)
        if not metrics and not checkpoints:
            print(f"  [SKIP] {exp_name} — no data files found")
            continue
        data[label] = {
            "color":      color,
            "exp_name":   exp_name,
            "metrics":    metrics,
            "checkpoints": checkpoints,
        }
        n_m = len(metrics.get("mean_distance_10", []))
        print(f"  [OK]   {exp_name} — {n_m} metric pts, {len(checkpoints)} checkpoints")
    return data


def _series(metrics: dict, key: str) -> Tuple[np.ndarray, np.ndarray]:
    entries = metrics.get(key, [])
    if not entries:
        return np.array([]), np.array([])
    steps  = np.array([e["step"]  for e in entries], dtype=float)
    values = np.array([e["value"] for e in entries], dtype=float)
    return steps, values


# ---------------------------------------------------------------------------
# Phase-shading helper (uses actual step data — not hardcoded fractions)
# ---------------------------------------------------------------------------

def _shade_phases(ax, steps: list, phases: list) -> None:
    """Colour background bands where curriculum phase changes (data-driven)."""
    if not steps or not phases:
        return
    max_step = float(max(steps))

    # Find step boundaries where phase changes
    boundaries = [float(steps[0])]
    boundary_phases = [phases[0]]
    cur = phases[0]
    for s, p in zip(steps[1:], phases[1:]):
        if p != cur:
            boundaries.append(float(s))
            boundary_phases.append(p)
            cur = p
    boundaries.append(max_step)

    for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        ph = boundary_phases[i] if i < len(boundary_phases) else 3
        ax.axvspan(start, end, alpha=0.12, color=PHASE_BG[min(ph, 3)], zorder=0, lw=0)

    # Phase-transition vertical lines
    for x in boundaries[1:-1]:
        ax.axvline(x, color="gray", ls="--", lw=0.8, alpha=0.5, zorder=1)


def _get_phase_steps(data: dict) -> Tuple[list, list]:
    """Extract step/phase arrays from the first condition that has them."""
    for d in data.values():
        m = d["metrics"]
        if m and "phase" in m:
            steps  = [e["step"]  for e in m["phase"]]
            phases = [int(e["value"]) for e in m["phase"]]
            return steps, phases
    return [], []


# ---------------------------------------------------------------------------
# Plot 1 — Training curves (dual-axis: reward + distance)
# ---------------------------------------------------------------------------

def plot_training_curves(data: dict, out_dir: Path) -> None:
    _style()
    fig, (ax_d, ax_r) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training Performance (Rolling 10-ep Mean)",
                 fontsize=14, fontweight="bold")

    phase_steps, phase_vals = _get_phase_steps(data)

    plotted = False
    for label, d in data.items():
        m     = d["metrics"]
        color = d["color"]
        steps_d, dist  = _series(m, "mean_distance_10")
        steps_r, rew   = _series(m, "mean_reward_10")
        if steps_d.size:
            ax_d.plot(steps_d / 1e6, dist, "o-", color=color,
                      lw=2, ms=3, alpha=0.88, label=label)
            plotted = True
        if steps_r.size:
            ax_r.plot(steps_r / 1e6, rew, "o-", color=color,
                      lw=2, ms=3, alpha=0.88, label=label)

    if not plotted:
        print("  [WARN] No metrics data for training curves.")
        plt.close(fig); return

    # Shade both panels with the same phase data (converted to ×10⁶)
    if phase_steps:
        ps_m = [s / 1e6 for s in phase_steps]
        _shade_phases(ax_d, ps_m, phase_vals)
        _shade_phases(ax_r, ps_m, phase_vals)

    for ax, ylabel, title in [
        (ax_d, "Mean Distance (m)",   "Distance"),
        (ax_r, "Mean Episode Reward", "Reward"),
    ]:
        ax.set_xlabel("Training Steps (×10⁶)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="medium")
        ax.set_xlim(left=0)
        ax.legend(loc="upper left", frameon=True, edgecolor="#CCCCCC",
                  framealpha=0.9, ncol=1)

    fig.tight_layout()
    path = out_dir / "01_training_curves.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ---------------------------------------------------------------------------
# Plot 2 — Checkpoint progression per phase (2×2 grid)
# ---------------------------------------------------------------------------

def plot_checkpoint_progression(data: dict, out_dir: Path) -> None:
    eligible = [(lbl, d) for lbl, d in data.items()
                if len(d["checkpoints"]) > 0]
    if not eligible:
        print("  [WARN] No checkpoint data for progression plot.")
        return

    _style()
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.flatten()
    fig.suptitle("Distance per Phase Across Checkpoint Steps",
                 fontsize=14, fontweight="bold")

    for ph_idx, ax in enumerate(axes):
        for label, d in eligible:
            steps, dists = [], []
            for ck in d["checkpoints"]:
                perf = ck.get("performance_metrics", {})
                ph   = perf.get(str(ph_idx), perf.get(ph_idx, {}))
                dist = ph.get("mean_distance")
                if dist is not None:
                    steps.append(ck["step"] / 1e6)
                    dists.append(dist)
            if steps:
                ax.plot(steps, dists, "o-", color=d["color"],
                        ms=5, lw=2, label=label, alpha=0.9)

        ax.set_title(PHASE_LABELS[ph_idx], fontsize=11)
        ax.set_xlabel("Training Step (×10⁶)", fontsize=9)
        ax.set_ylabel("Mean Distance (m)", fontsize=9)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=7, loc="upper left", frameon=True, edgecolor="#CCCCCC")

    fig.tight_layout()
    path = out_dir / "02_checkpoint_progression.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ---------------------------------------------------------------------------
# Plot 3 — Final-checkpoint phase bars
# ---------------------------------------------------------------------------

def plot_phase_bars(data: dict, out_dir: Path) -> None:
    eligible = [(lbl, d) for lbl, d in data.items() if d["checkpoints"]]
    if not eligible:
        print("  [WARN] No checkpoint data for phase bar chart.")
        return

    _style()
    phases  = [0, 1, 2, 3]
    n_cond  = len(eligible)
    width   = 0.8 / n_cond
    x       = np.arange(len(phases))

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.suptitle("Final-Checkpoint Distance per Phase",
                 fontsize=14, fontweight="bold")

    for i, (label, d) in enumerate(eligible):
        final = d["checkpoints"][-1]
        perf  = final.get("performance_metrics", {})
        dists, errs = [], []
        for ph in phases:
            ph_data = perf.get(str(ph), perf.get(ph, {}))
            dists.append(ph_data.get("mean_distance", 0.0))
            errs.append(ph_data.get("std_distance",  0.0))

        offset = (i - n_cond / 2 + 0.5) * width
        ax.bar(x + offset, dists, width * 0.9, label=label,
               color=d["color"], alpha=0.88, zorder=3,
               yerr=errs, capsize=3, error_kw={"linewidth": 1})

    ax.set_xticks(x)
    ax.set_xticklabels([PHASE_LABELS[p] for p in phases], fontsize=10)
    ax.set_ylabel("Mean Distance (m)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", frameon=True, edgecolor="#CCCCCC",
              framealpha=0.9, ncol=2)

    fig.tight_layout()
    path = out_dir / "03_phase_bars.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ---------------------------------------------------------------------------
# Plot 4 — Reward vs Distance scatter (gaming diagnostic)
# ---------------------------------------------------------------------------

def plot_reward_vs_distance(data: dict, out_dir: Path) -> None:
    eligible = [(lbl, d) for lbl, d in data.items() if d["checkpoints"]]
    if not eligible:
        print("  [WARN] No checkpoint data for reward vs distance scatter.")
        return

    _style()
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=False)
    fig.suptitle("Reward vs Distance per Phase at Final Checkpoint\n"
                 "(divergence = reward gaming; trust distance, not reward)",
                 fontsize=11, y=1.02)

    for ph_idx, ax in enumerate(axes):
        for label, d in eligible:
            final = d["checkpoints"][-1]
            perf  = final.get("performance_metrics", {})
            ph    = perf.get(str(ph_idx), perf.get(ph_idx, {}))
            dist  = ph.get("mean_distance")
            rew   = ph.get("mean_reward")
            if dist is None or rew is None:
                continue
            ax.scatter(rew, dist, color=d["color"], s=120, zorder=5,
                       edgecolors="white", linewidth=0.8, label=label)
            ax.annotate(label.split()[0], (rew, dist),
                        textcoords="offset points", xytext=(5, 4),
                        fontsize=7, color=d["color"])

        ax.set_title(PHASE_LABELS[ph_idx], fontsize=9)
        ax.set_xlabel("Mean Reward", fontsize=8)
        if ph_idx == 0:
            ax.set_ylabel("Mean Distance (m)", fontsize=8)
        ax.grid(True, alpha=0.3)

    handles = [mpatches.Patch(color=d["color"], label=lbl)
               for lbl, d in eligible]
    axes[-1].legend(handles=handles, fontsize=7, loc="lower right",
                    frameon=True, edgecolor="#CCCCCC")

    fig.tight_layout()
    path = out_dir / "04_reward_vs_distance.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ---------------------------------------------------------------------------
# Plot 5 — Summary bar (overall mean distance, one number per condition)
# ---------------------------------------------------------------------------

def plot_summary_bar(data: dict, out_dir: Path) -> None:
    eligible = [(lbl, d) for lbl, d in data.items() if d["checkpoints"]]
    if not eligible:
        print("  [WARN] No checkpoint data for summary bar.")
        return

    _style()
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.suptitle("Overall Mean Distance — Final Checkpoint\n"
                 "(average across all 4 phases)",
                 fontsize=13, fontweight="bold")

    labels_out, means_out, stds_out, colors_out = [], [], [], []
    for label, d in eligible:
        perf = d["checkpoints"][-1].get("performance_metrics", {})
        ph_means, ph_stds = [], []
        for ph in range(4):
            ph_data = perf.get(str(ph), perf.get(ph, {}))
            m = ph_data.get("mean_distance")
            s = ph_data.get("std_distance", 0.0)
            if m is not None:
                ph_means.append(m); ph_stds.append(s)
        if not ph_means:
            continue
        labels_out.append(label)
        means_out.append(float(np.mean(ph_means)))
        stds_out.append(float(np.mean(ph_stds)))
        colors_out.append(d["color"])

    y = np.arange(len(labels_out))
    bars = ax.barh(y, means_out, xerr=stds_out, color=colors_out,
                   alpha=0.88, capsize=4, height=0.55,
                   error_kw={"linewidth": 1.2})
    for bar, val in zip(bars, means_out):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)

    ax.set_yticks(y)
    ax.set_yticklabels(labels_out)
    ax.set_xlabel("Mean Distance (m)")
    ax.set_xlim(left=0)
    ax.invert_yaxis()

    fig.tight_layout()
    path = out_dir / "05_summary_bar.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ---------------------------------------------------------------------------
# Plot 6 — Neuromodulation variance
# ---------------------------------------------------------------------------

def plot_neuromod_variance(data: dict, out_dir: Path) -> None:
    has_data = any(
        len(d["metrics"].get("neuromod_variance", [])) > 0
        for d in data.values()
    )
    if not has_data:
        print("  [SKIP] neuromod_variance not in metrics (re-run experiments to populate).")
        return

    _style()
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle("Neuromodulation Signal Variance (land_flag)",
                 fontsize=14, fontweight="bold")
    ax.set_title("Near-zero in Phase 0 expected; should rise in Phase 2+", fontsize=11)

    phase_steps, phase_vals = _get_phase_steps(data)

    plotted = False
    for label, d in data.items():
        steps, vals = _series(d["metrics"], "neuromod_variance")
        if not steps.size:
            continue
        ax.plot(steps / 1e6, vals, color=d["color"], lw=2.5,
                label=label, alpha=0.9)
        plotted = True

    if not plotted:
        plt.close(fig); return

    if phase_steps:
        _shade_phases(ax, [s / 1e6 for s in phase_steps], phase_vals)

    ax.set_xlabel("Training Steps (×10⁶)")
    ax.set_ylabel("Variance of land_flag within episode")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=True, edgecolor="#CCCCCC", framealpha=0.9)

    fig.tight_layout()
    path = out_dir / "06_neuromod_variance.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ---------------------------------------------------------------------------
# Plot 7 — Prior diagnostics (Cook-2019 synapse counts + soma distances)
# ---------------------------------------------------------------------------

def plot_prior_diagnostics(out_dir: Path) -> None:
    _style()
    sys.path.insert(0, str(NMAP_ROOT.parent))
    sys.path.insert(0, str(NMAP_ROOT))

    priors = None
    try:
        from NMAP.connectome_priors.swimmer_priors import generate_ncap_segment_priors
        priors = generate_ncap_segment_priors(num_segments=4)
        print("  Loaded Cook-2019 priors successfully.")
    except Exception as exc:
        try:
            from connectome_priors.swimmer_priors import generate_ncap_segment_priors
            priors = generate_ncap_segment_priors(num_segments=4)
        except Exception:
            warnings.warn(f"Could not load priors: {exc}")

    pathways  = ["ipsi_db", "ipsi_vb", "contra_db", "contra_vb", "next_db", "next_vb"]
    pw_labels = [
        "ipsi_db\n(DB→dors.m.)",
        "ipsi_vb\n(VB→vent.m.)",
        "contra_db\n(DB→DD)",
        "contra_vb\n(VB→VD)",
        "next_db\n(DB↔DB gap)",
        "next_vb\n(VB↔VB gap)",
    ]
    x = np.arange(len(pathways))

    syn_vals  = [priors.get(f"syn_{p}",  0) for p in pathways] if priors else [0] * 6
    dist_vals = [priors.get(f"dist_{p}", 1) for p in pathways] if priors else [1] * 6

    fig, (ax_syn, ax_dist) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Cook-2019 Connectome Prior Diagnostics",
                 fontsize=14, fontweight="bold")

    chem_color = PHASE_COLORS[0]
    elec_color = PHASE_COLORS[1]
    bar_colors = [chem_color] * 4 + [elec_color] * 2

    ax_syn.bar(x, syn_vals, color=bar_colors, alpha=0.9, zorder=3, edgecolor="black")
    ax_syn.set_title("Mean Synapse Count per Pathway", fontweight="medium")
    ax_syn.set_ylabel("Average synapse count")
    ax_syn.set_xticks(x); ax_syn.set_xticklabels(pw_labels, rotation=25, ha="right", fontsize=10)
    ax_syn.set_xlabel("Pathway (scales weight initialisation)")
    ax_syn.legend(handles=[
        mpatches.Patch(color=chem_color, label="Chemical synapses"),
        mpatches.Patch(color=elec_color, label="Electrical gap junctions"),
    ], frameon=True, edgecolor="#CCCCCC")

    ax_dist.bar(x, dist_vals, color=PHASE_COLORS[2], alpha=0.9, zorder=3, edgecolor="black")
    ax_dist.set_title("Normalised Soma Distance per Pathway", fontweight="medium")
    ax_dist.set_ylabel("Normalised distance  [0=short, 1=long]")
    ax_dist.set_xticks(x); ax_dist.set_xticklabels(pw_labels, rotation=25, ha="right", fontsize=10)
    ax_dist.set_xlabel("Pathway (longer ↔ stronger L2 penalty)")
    ax_dist.set_ylim(0, 1.15)
    ax_dist.axhline(1.0, color="red", lw=1.5, ls="--", alpha=0.7, label="Max = 1.0")
    ax_dist.legend(frameon=True, edgecolor="#CCCCCC")

    if priors is None:
        for ax in (ax_syn, ax_dist):
            ax.text(0.5, 0.5, "Cook-2019 XLSX not found",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=12, color="#333333",
                    bbox=dict(boxstyle="round", fc="#FFF3E0", ec="#FFB74D"))

    fig.tight_layout()
    path = out_dir / "07_prior_diagnostics.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ---------------------------------------------------------------------------
# Plot 8 — Weight evolution (optional: loads .pt files)
# ---------------------------------------------------------------------------

TRACKED_PARAMS = [
    ("params.muscle_ipsi",      "muscle_ipsi\n(ipsi excit.)"),
    ("params.muscle_contra",    "muscle_contra\n(contra inhib.)"),
    ("params.bneuron_prop",     "bneuron_prop\n(propriocept.)"),
    ("params.bneuron_osc",      "bneuron_osc\n(CPG drive)"),
    ("params.muscle_d_d_0",     "muscle_d_d[0]\n(joint-0 ipsi)"),
    ("params.muscle_d_v_0",     "muscle_d_v[0]\n(joint-0 contra)"),
    ("params.bneuron_d_prop_1", "bneuron_d_prop[1]\n(proprio D)"),
    ("params.bneuron_v_prop_1", "bneuron_v_prop[1]\n(proprio V)"),
]


def plot_weight_evolution(data: dict, out_dir: Path) -> None:
    print("  Loading .pt checkpoints for weight evolution (may be slow)…")
    all_histories = {}
    for label, d in data.items():
        h = load_pt_checkpoints(RESULTS_ROOT / d["exp_name"])
        if h:
            all_histories[label] = h

    if not all_histories:
        print("  [SKIP] No .pt checkpoints found for weight evolution.")
        return

    _style()
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle("Key Parameter Weight Evolution Across Training Checkpoints",
                 fontsize=14, fontweight="bold")
    gs   = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.32)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(4)]

    for ax_idx, (param, plabel) in enumerate(TRACKED_PARAMS):
        ax = axes[ax_idx]
        any_data = False
        for label, history in all_histories.items():
            exp_name = data[label]["exp_name"]
            color    = data[label]["color"]
            steps, vals = [], []
            for s in sorted(history.keys()):
                sd = history[s]
                if param in sd:
                    try:
                        vals.append(float(sd[param].item()))
                        steps.append(s)
                    except Exception:
                        pass
            if steps:
                ax.plot([s / 1e6 for s in steps], vals, "o-",
                        color=color, lw=2, ms=4,
                        label=label, alpha=0.88)
                any_data = True

        ax.set_title(plabel, fontsize=10, fontweight="medium")
        ax.set_xlabel("Steps (×10⁶)", fontsize=9)
        ax.set_ylabel("Weight value", fontsize=9)
        ax.tick_params(labelsize=8)
        if "contra" in param or "d_v" in param or "v_d" in param:
            ax.axhspan(-1, 0, alpha=0.07, color="red", lw=0)
        else:
            ax.axhspan(0, 1, alpha=0.07, color="green", lw=0)
        if any_data and ax_idx == 0:
            ax.legend(fontsize=8, frameon=True, edgecolor="#CCCCCC")

    path = out_dir / "08_weight_evolution.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ---------------------------------------------------------------------------
# RESULTS.md
# ---------------------------------------------------------------------------

def write_results_md(data: dict, out_dir: Path) -> None:
    lines = [
        "# Connectomic-Priors Ablation — Results",
        "",
        "> Generated by `results/plot_ablation.py`. Plots are in `results/analysis_plots/`.",
        "",
        "## Plots",
        "| File | Description |",
        "|------|-------------|",
        "| `01_training_curves.png` | Rolling-10 reward + distance over training steps |",
        "| `02_checkpoint_progression.png` | Distance per phase across all checkpoint steps |",
        "| `03_phase_bars.png` | Final-checkpoint distance grouped by phase |",
        "| `04_reward_vs_distance.png` | Reward vs distance scatter — gaming diagnostic |",
        "| `05_summary_bar.png` | Single overall score per condition |",
        "| `06_neuromod_variance.png` | Neuromodulation signal variance |",
        "| `07_prior_diagnostics.png` | Cook-2019 synapse counts + soma distances |",
        "| `08_weight_evolution.png` | Key parameter trajectories (requires .pt files) |",
        "",
        "## Ablation Reading Guide",
        "```",
        "02 vs 01 → connectomic init alone",
        "03 vs 01 → wiring-economy regularisation alone",
        "04 vs 01 → full connectomic priors (init + reg)",
        "04 vs 02 → marginal gain of reg on top of init",
        "04 vs 03 → marginal gain of init on top of reg",
        "05 vs 04 → oscillation enforcement (upstream NMAP)",
        "```",
        "",
        "## Final Checkpoint Summary",
        "",
        "| Condition | Ph0 Pure Swim | Ph1 Single Land | Ph2 Two Land | Ph3 Full | Mean |",
        "|-----------|---------------|-----------------|--------------|----------|------|",
    ]

    for label, d in data.items():
        if not d["checkpoints"]:
            lines.append(f"| {label} | — | — | — | — | — |")
            continue
        perf = d["checkpoints"][-1].get("performance_metrics", {})
        vals = []
        for ph in range(4):
            ph_data = perf.get(str(ph), perf.get(ph, {}))
            vals.append(ph_data.get("mean_distance", float("nan")))
        mean_val = float(np.nanmean(vals))
        cols = " | ".join(f"{v:.3f}" if not np.isnan(v) else "—" for v in vals)
        lines.append(f"| {label} | {cols} | {mean_val:.3f} |")

    lines += [
        "",
        "*(Distance: higher = navigated further = better)*",
        "",
        "## Training Distance Summary",
        "",
        "| Condition | Min (m) | Max (m) | Final (m) |",
        "|-----------|---------|---------|-----------|",
    ]

    for label, d in data.items():
        steps, dists = _series(d["metrics"], "mean_distance_10")
        if dists.size:
            lines.append(f"| {label} | {dists.min():.3f} | {dists.max():.3f} | {dists[-1]:.3f} |")
        else:
            lines.append(f"| {label} | — | — | — |")

    lines.append("")
    out = RESULTS_ROOT / "RESULTS.md"
    out.write_text("\n".join(lines))
    print(f"  Saved: RESULTS.md")

# ---------------------------------------------------------------------------
# Plot 9 — Speed per phase at final checkpoint (line plot, one point per phase)
# ---------------------------------------------------------------------------

# Evaluation episode length in steps per phase (from PHASE_DURATION_CONFIG).
# Speed = distance / (steps / control_freq), where control_freq = 60 Hz.
_EVAL_STEPS_PER_PHASE = {0: 400, 1: 600, 2: 800, 3: 1200}
_CONTROL_FREQ_HZ      = 60.0


def plot_speed_per_phase(data: dict, out_dir: Path) -> None:
    """09_speed_per_phase.png

    Mean speed (m/s) per curriculum phase at the final checkpoint,
    derived from mean_distance / episode_duration.  Mirrors the style
    of the reference NMAP slide ('One number exposes the whole difference').
    """
    eligible = [(lbl, d) for lbl, d in data.items() if d["checkpoints"]]
    if not eligible:
        print("  [WARN] No checkpoint data for speed plot.")
        return

    _style()
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle("Mean Speed per Curriculum Phase — Final Checkpoint",
                 fontsize=14, fontweight="bold")

    phases      = [0, 1, 2, 3]
    phase_names = [PHASE_LABELS[p].split(": ")[1] for p in phases]  # short labels

    for label, d in eligible:
        perf   = d["checkpoints"][-1].get("performance_metrics", {})
        speeds = []
        for ph in phases:
            ph_data  = perf.get(str(ph), perf.get(ph, {}))
            dist     = ph_data.get("mean_distance")
            if dist is None:
                speeds.append(np.nan)
                continue
            duration_s = _EVAL_STEPS_PER_PHASE[ph] / _CONTROL_FREQ_HZ
            speeds.append(dist / duration_s)

        ax.plot(phases, speeds, "o-",
                color=d["color"], lw=2.5, ms=8,
                label=label, alpha=0.92, zorder=3)

    ax.set_xticks(phases)
    ax.set_xticklabels(phase_names, fontsize=10)
    ax.set_ylabel("Mean Speed (m/s)", fontsize=11)
    ax.set_xlabel("Curriculum Phase", fontsize=11)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=True, edgecolor="#CCCCCC", framealpha=0.9,
              fontsize=9, loc="upper left")

    # Light phase background shading
    for ph, bg in enumerate(PHASE_BG):
        ax.axvspan(ph - 0.4, ph + 0.4, alpha=0.15, color=bg, zorder=0, lw=0)

    fig.tight_layout()
    path = out_dir / "09_speed_per_phase.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nConnectomic Priors Ablation Analysis")
    print(f"  Results root : {RESULTS_ROOT}")
    print(f"  Output dir   : {OUTPUT_DIR}")
    print()

    print("Loading experiment data...")
    data = collect_all()
    if not data:
        print("\nERROR: No experiment data found.")
        return

    print(f"\nGenerating {len(data)} condition(s)...")
    plot_training_curves(data, OUTPUT_DIR)
    plot_checkpoint_progression(data, OUTPUT_DIR)
    plot_phase_bars(data, OUTPUT_DIR)
    plot_reward_vs_distance(data, OUTPUT_DIR)
    plot_summary_bar(data, OUTPUT_DIR)
    plot_neuromod_variance(data, OUTPUT_DIR)
    plot_prior_diagnostics(OUTPUT_DIR)
    plot_weight_evolution(data, OUTPUT_DIR)
    plot_speed_per_phase(data, OUTPUT_DIR)
    write_results_md(data, OUTPUT_DIR)

    print(f"\nDone. Outputs in: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
