#!/usr/bin/env python3
"""Local artifact naming for the micro-publication package."""

from __future__ import annotations

import os


class MicroPublicationArtifactNamer:
    def __init__(self, experiment_name: str, run_root: str, model_type: str, n_links: int, algorithm: str, oscillator_period: int):
        self.experiment_name = experiment_name
        self.run_root = run_root
        self.base_id = (
            f"{experiment_name}__{model_type}_{algorithm}_{n_links}links_"
            f"oscillator_period{oscillator_period}_training_modemicro_publication"
        )

    def _dir(self, name: str) -> str:
        path = os.path.join(self.run_root, name)
        os.makedirs(path, exist_ok=True)
        return path

    def checkpoint_name(self, step: int) -> str:
        return os.path.join(self._dir("checkpoints"), f"{self.base_id}_checkpoint_step_{step}.pt")

    def final_model_name(self) -> str:
        return os.path.join(self._dir("models"), f"{self.base_id}_final_model.pt")

    def evaluation_video_name(self, evaluation_type: str) -> str:
        return os.path.join(self._dir("videos"), f"{self.base_id}_eval_{evaluation_type}_final.mp4")

    def training_video_name(self, step: int, phase: str | None = None) -> str:
        suffix = f"_phase_{phase}" if phase else ""
        return os.path.join(self._dir("videos"), f"{self.base_id}_training{suffix}_step_{step}.mp4")

    def analysis_plot_name(self, analysis_type: str, phase: str | None = None, step: int | None = None) -> str:
        parts = [self.base_id, analysis_type]
        if phase:
            parts.append(f"phase_{phase}")
        if step is not None:
            parts.append(f"step_{step}")
        return os.path.join(self._dir("plots"), "_".join(parts) + ".png")

    def experiment_summary_name(self) -> str:
        return os.path.join(self._dir("summaries"), f"{self.base_id}_experiment_summary.md")

    def training_log_path(self) -> str:
        return os.path.join(self._dir("logs"), f"{self.base_id}_training_log.jsonl")
