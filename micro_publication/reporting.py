#!/usr/bin/env python3
"""
Reporting utilities for the micro-publication experiment package.
"""

import json
import os
from typing import Dict, Iterable


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _make_json_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_make_json_safe(v) for v in value]
    return str(value)


def write_json(path: str, payload: Dict) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_make_json_safe(payload), handle, indent=2, sort_keys=True)


def write_markdown_summary(path: str, title: str, lines: Iterable[str]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f"# {title}\n\n")
        for line in lines:
            handle.write(f"{line}\n")


def make_experiment_report(output_dir: str, config: Dict, summary: Dict, runtime_notes=None) -> None:
    runtime_notes = runtime_notes or []
    write_json(os.path.join(output_dir, "experiment_config.json"), config)
    write_json(os.path.join(output_dir, "experiment_summary.json"), summary)

    lines = [
        f"- Name: `{config['name']}`",
        f"- Description: {config['description']}",
        f"- Tags: {', '.join(config.get('tags', []))}",
        f"- Training steps: `{config['training']['training_steps']}`",
        f"- Observation cues: environment=`{config['observation']['expose_environment']}`, viscosity=`{config['observation']['expose_viscosity']}`",
        f"- Anisotropy mode: `{config['anisotropy']['mode']}`",
        "",
        "## Summary",
        f"- Mean phase distance: `{summary.get('mean_phase_distance', 0.0):.4f}`",
        f"- Mean phase reward: `{summary.get('mean_phase_reward', 0.0):.4f}`",
        f"- Mixed-phase success rate: `{summary.get('mixed_phase_success_rate', 0.0):.2f}`",
        f"- Mixed-phase mean transitions: `{summary.get('mixed_phase_transition_mean', 0.0):.2f}`",
    ]

    phase_metrics = summary.get("phase_metrics", {})
    if phase_metrics:
        lines.extend([
            "",
            "## Phase Metrics",
            "| Phase | Success | Transitions | Water | Land | Land Fraction | Mean Reward | Mean Distance |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ])
        phase_labels = {
            "0": "Pure Swimming",
            "1": "Single Land Zone",
            "2": "Two Land Zones",
            "3": "Full Complexity",
        }
        for phase_id in ["0", "1", "2", "3"]:
            phase = phase_metrics.get(phase_id, {})
            lines.append(
                f"| {phase_labels[phase_id]} | `{phase.get('success_label', 'n/a')}` | "
                f"{phase.get('transitions', 0)} | {phase.get('water_time', 0)} | {phase.get('land_time', 0)} | "
                f"{phase.get('land_fraction', 0.0):.2f} | {phase.get('mean_reward', 0.0):.2f} | {phase.get('mean_distance', 0.0):.3f} |"
            )

    if runtime_notes:
        lines.extend(["", "## Runtime Notes"])
        for note in runtime_notes:
            lines.append(f"- {note}")

    write_markdown_summary(
        os.path.join(output_dir, "experiment_summary.md"),
        title=f"Micro-Publication Experiment: {config['name']}",
        lines=lines,
    )


def make_matrix_report(output_dir: str, rows) -> None:
    ensure_dir(output_dir)
    write_json(os.path.join(output_dir, "comparison.json"), {"runs": rows})

    lines = [
        "| Experiment | Mean Phase Distance | Mean Phase Reward | Mixed Success | Mean Transitions | Notes |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['experiment']}` | {row.get('mean_phase_distance', 0.0):.4f} | "
            f"{row.get('mean_phase_reward', 0.0):.4f} | {row.get('mixed_phase_success_rate', 0.0):.2f} | "
            f"{row.get('mixed_phase_transition_mean', 0.0):.2f} | {row.get('notes', '')} |"
        )

    lines.extend([
        "",
        "## Phase Comparison",
        "| Experiment | P1 | P2 | P3 |",
        "|---|---|---|---|",
    ])
    for row in rows:
        metrics = row.get("phase_metrics", {})
        def _fmt(phase_id):
            phase = metrics.get(str(phase_id), {})
            return f"{phase.get('success_label', 'n/a')} / t={phase.get('transitions', 0)} / land={phase.get('land_fraction', 0.0):.2f}"
        lines.append(f"| `{row['experiment']}` | {_fmt(1)} | {_fmt(2)} | {_fmt(3)} |")

    lines.extend([
        "",
        "## Run Folders",
    ])
    for row in rows:
        lines.append(f"- `{row['experiment']}`: `{row.get('output_dir', '')}`")

    write_markdown_summary(
        os.path.join(output_dir, "comparison.md"),
        "Micro-Publication Matrix Comparison",
        lines,
    )
