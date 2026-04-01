#!/usr/bin/env python3
"""
CLI entry point for the dedicated micro-publication package.
"""

import argparse
import os

from micro_publication import list_experiments
from micro_publication.reporting import ensure_dir, make_matrix_report, write_json, write_markdown_summary
from micro_publication.trainer import run_named_experiment


def _write_matrix_manifest(output_dir: str):
    ensure_dir(output_dir)
    experiments = [name for name in list_experiments() if name != "report_only_matrix"]
    write_json(
        os.path.join(output_dir, "matrix_manifest.json"),
        {"experiments": experiments},
    )
    lines = [
        "## Experiments",
        *[f"- `{name}`" for name in experiments if name != "report_only_matrix"],
        "",
        "## Recommended Order",
        "- `baseline_short`",
        "- `cue_ablation_short`",
        "- `anisotropy_proxy_short`",
        "- `anisotropy_full_short`",
        "",
        "## Commands",
        "- `python3 micro_publication_main.py --experiment baseline_short`",
        "- `python3 micro_publication_main.py --experiment cue_ablation_short`",
        "- `python3 micro_publication_main.py --experiment anisotropy_proxy_short`",
        "- `python3 micro_publication_main.py --experiment anisotropy_full_short`",
    ]
    write_markdown_summary(
        os.path.join(output_dir, "matrix_manifest.md"),
        "Micro-Publication Experiment Matrix",
        lines,
    )


def _notes_for_experiment(experiment_name: str) -> str:
    mapping = {
        "baseline_short": "Privileged substrate cues on, isotropic medium switching.",
        "cue_ablation_short": "Privileged environment and viscosity cues hidden.",
        "anisotropy_proxy_short": "Directional drag proxy on land.",
        "anisotropy_full_short": "Full per-segment directional drag in both media.",
    }
    return mapping.get(experiment_name, "")


def main():
    parser = argparse.ArgumentParser(description="Micro-publication experiment runner")
    parser.add_argument("--experiment", choices=list_experiments(), default="baseline_short")
    parser.add_argument("--matrix_only", action="store_true", help="Write the experiment matrix manifest without training")
    parser.add_argument("--run_matrix", action="store_true", help="Run all short experiment presets sequentially")
    args = parser.parse_args()

    matrix_dir = os.path.join("outputs", "micro_publication")
    _write_matrix_manifest(matrix_dir)
    if args.matrix_only or args.experiment == "report_only_matrix":
        return

    if args.run_matrix:
        rows = []
        for experiment_name in [name for name in list_experiments() if name != "report_only_matrix"]:
            result = run_named_experiment(experiment_name)
            summary = result.get("summary", {})
            rows.append({
                "experiment": experiment_name,
                "output_dir": result.get("output_dir", ""),
                "mean_phase_distance": summary.get("mean_phase_distance", 0.0),
                "mean_phase_reward": summary.get("mean_phase_reward", 0.0),
                "mixed_phase_success_rate": summary.get("mixed_phase_success_rate", 0.0),
                "mixed_phase_transition_mean": summary.get("mixed_phase_transition_mean", 0.0),
                "phase_metrics": summary.get("phase_metrics", {}),
                "notes": _notes_for_experiment(experiment_name),
            })
        make_matrix_report(os.path.join(matrix_dir, "matrix"), rows)
        return

    run_named_experiment(args.experiment)


if __name__ == "__main__":
    main()
