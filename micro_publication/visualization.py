#!/usr/bin/env python3
"""Local plotting and video helpers for the micro-publication package."""

from __future__ import annotations

import os

import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PHASE_NAMES = ["Pure Swimming", "Single Land Zone", "Two Land Zones", "Full Complexity"]


def _save_video_previews(save_path: str, frames, fps: int = 30):
    if not frames:
        return
    base_path, _ = os.path.splitext(save_path)
    gif_path = base_path + "_preview.gif"
    png_path = base_path + "_contact_sheet.png"

    preview_count = min(120, len(frames))
    preview_indices = np.linspace(0, len(frames) - 1, preview_count, dtype=int)
    imageio.mimsave(gif_path, [frames[i] for i in preview_indices], fps=min(fps, 12))

    sheet_count = min(12, len(frames))
    sheet_indices = np.linspace(0, len(frames) - 1, sheet_count, dtype=int)
    cols = 4
    rows = int(np.ceil(sheet_count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 2.4))
    axes = np.atleast_1d(axes).reshape(rows, cols)
    for axis in axes.flat:
        axis.axis("off")
    for axis, idx in zip(axes.flat, sheet_indices):
        axis.imshow(frames[idx])
        axis.set_title(f"f{idx}", fontsize=8)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_video(save_path: str, frames, fps: int = 30):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    writer = imageio.get_writer(
        save_path,
        format="FFMPEG",
        mode="I",
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=16,
    )
    try:
        for frame in frames:
            writer.append_data(np.asarray(frame, dtype=np.uint8))
    finally:
        writer.close()
    _save_video_previews(save_path, frames, fps=fps)


def create_training_plot(phase_rewards, phase_distances, eval_results, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    for phase in range(4):
        rewards = np.asarray(phase_rewards.get(phase, []), dtype=np.float32)
        distances = np.asarray(phase_distances.get(phase, []), dtype=np.float32)
        axes[0].plot(rewards, label=PHASE_NAMES[phase], alpha=0.8)
        axes[1].plot(distances, label=PHASE_NAMES[phase], alpha=0.8)

    axes[0].set_title("Reward by Episode")
    axes[1].set_title("Distance by Episode")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)

    mean_rewards = [eval_results.get(phase, {}).get("mean_reward", 0.0) for phase in range(4)]
    mean_distances = [eval_results.get(phase, {}).get("mean_distance", 0.0) for phase in range(4)]
    axes[2].bar(PHASE_NAMES, mean_rewards)
    axes[2].set_title("Eval Mean Reward")
    axes[2].tick_params(axis="x", rotation=20)
    axes[3].bar(PHASE_NAMES, mean_distances)
    axes[3].set_title("Eval Mean Distance")
    axes[3].tick_params(axis="x", rotation=20)

    fig.tight_layout()
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def collect_rollout(agent, env, num_steps: int, phase_label: str):
    frames = []
    obs = env.reset()
    initial = env.env.head_position.copy()
    last_env = None
    transitions = 0
    water_time = 0
    land_time = 0

    for step in range(num_steps):
        frame = env.render(mode="rgb_array")
        if frame is not None:
            frames.append(frame)
        action = agent.test_step(obs)
        obs, _, done, info = env.step(action)
        if info.get("physics_error"):
            break
        in_land = bool(env.env.env._task._current_environment_state(env.env.physics)[3])
        current_env = "land" if in_land else "water"
        if current_env == "land":
            land_time += 1
        else:
            water_time += 1
        if last_env is not None and current_env != last_env:
            transitions += 1
        last_env = current_env
        if done:
            break

    final = env.env.head_position.copy()
    return {
        "frames": frames,
        "stats": {
            "phase_label": phase_label,
            "final_distance": float(np.linalg.norm(final - initial)),
            "transitions": int(transitions),
            "water_time": int(water_time),
            "land_time": int(land_time),
        },
    }


def create_phase_video(agent, env, save_path: str, phase_progress: float, force_land_start: bool, num_steps: int, phase_label: str):
    env.env.set_manual_progress(phase_progress, force_land_start=force_land_start)
    rollout = collect_rollout(agent, env, num_steps=num_steps, phase_label=phase_label)
    if rollout["frames"]:
        save_video(save_path, rollout["frames"], fps=30)
    return rollout["stats"]


def create_phase_comparison_video(agent, env_factory, save_path: str, phase_steps, force_land_start_by_phase=None):
    if force_land_start_by_phase is None:
        force_land_start_by_phase = [False, False, False, False]
    all_frames = []
    all_stats = {}
    for phase, num_steps in enumerate(phase_steps):
        env = env_factory()
        env.env.set_manual_progress((phase + 0.5) * 0.25, force_land_start=force_land_start_by_phase[phase])
        rollout = collect_rollout(agent, env, num_steps=num_steps, phase_label=PHASE_NAMES[phase])
        all_frames.extend(rollout["frames"])
        all_stats[phase] = rollout["stats"]
        if phase < 3 and rollout["frames"]:
            all_frames.extend([np.zeros_like(rollout["frames"][0], dtype=np.uint8) for _ in range(20)])
        env.close()
    if all_frames:
        save_video(save_path, all_frames, fps=30)
    return all_stats
