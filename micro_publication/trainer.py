#!/usr/bin/env python3
"""
Trainer wrapper dedicated to the micro-publication package.
"""

import os

from swimmer.training.curriculum_trainer import CurriculumNCAPTrainer
from swimmer.utils.artifact_naming import ArtifactNamer

from .experiments import build_experiment
from .metrics import build_experiment_summary, parse_training_summary
from .reporting import ensure_dir, make_experiment_report, write_json
from .environment import TonicMicroPublicationWrapper


class ExperimentArtifactNamer(ArtifactNamer):
    """
    Redirect inherited trainer artifacts into the micro-publication run folder.
    """

    def __init__(self, experiment_name: str, run_root: str, model_type: str, n_links: int, algorithm: str = "ppo", additional_config=None):
        super().__init__(model_type, n_links, algorithm, additional_config)
        self.experiment_name = experiment_name
        self.run_root = run_root
        self.base_id = f"{experiment_name}__{self.base_id}"

    def checkpoint_name(self, step: int, base_dir: str = "", use_model_subfolder: bool = True) -> str:
        target_dir = os.path.join(self.run_root, "checkpoints")
        os.makedirs(target_dir, exist_ok=True)
        return os.path.join(target_dir, f"{self.base_id}_checkpoint_step_{step}.pt")

    def final_model_name(self, base_dir: str = "", use_model_subfolder: bool = True) -> str:
        target_dir = os.path.join(self.run_root, "models")
        os.makedirs(target_dir, exist_ok=True)
        return os.path.join(target_dir, f"{self.base_id}_final_model.pt")

    def evaluation_video_name(self, step=None, evaluation_type: str = "mixed_env", base_dir: str = "", use_model_subfolder: bool = True) -> str:
        target_dir = os.path.join(self.run_root, "videos")
        os.makedirs(target_dir, exist_ok=True)
        filename = f"{self.base_id}_eval_{evaluation_type}"
        if step is not None:
            filename += f"_step_{step}"
        else:
            filename += "_final"
        return os.path.join(target_dir, filename + ".mp4")

    def training_video_name(self, step: int, phase=None, base_dir: str = "", use_model_subfolder: bool = True) -> str:
        target_dir = os.path.join(self.run_root, "videos")
        os.makedirs(target_dir, exist_ok=True)
        suffix = f"_phase_{phase}" if phase else ""
        return os.path.join(target_dir, f"{self.base_id}_training{suffix}_step_{step}.mp4")

    def analysis_plot_name(self, analysis_type: str, step=None, phase=None, base_dir: str = "", use_model_subfolder: bool = True) -> str:
        target_dir = os.path.join(self.run_root, "plots")
        os.makedirs(target_dir, exist_ok=True)
        parts = [self.base_id, analysis_type]
        if phase:
            parts.append(f"phase_{phase}")
        if step is not None:
            parts.append(f"step_{step}")
        return os.path.join(target_dir, "_".join(parts) + ".png")

    def training_log_dir(self, base_dir: str = "", use_model_subfolder: bool = True) -> str:
        target_dir = os.path.join(self.run_root, "logs", self.base_id)
        os.makedirs(target_dir, exist_ok=True)
        return target_dir

    def experiment_summary_name(self, base_dir: str = "", use_model_subfolder: bool = True) -> str:
        target_dir = os.path.join(self.run_root, "summaries")
        os.makedirs(target_dir, exist_ok=True)
        return os.path.join(target_dir, f"{self.base_id}_experiment_summary.md")


class MicroPublicationTrainer(CurriculumNCAPTrainer):
    def __init__(self, experiment):
        self.experiment = experiment
        run_root = os.path.join(experiment.output_root, "runs", experiment.name)
        ensure_dir(run_root)
        self.output_root = run_root

        super().__init__(
            n_links=experiment.training.n_links,
            learning_rate=experiment.training.learning_rate,
            training_steps=experiment.training.training_steps,
            save_steps=experiment.training.save_steps,
            log_episodes=experiment.training.log_episodes,
            oscillator_period=experiment.training.oscillator_period,
            model_type=experiment.training.model_type,
            algorithm=experiment.training.algorithm,
            num_workers=experiment.training.num_workers,
            use_multi_gpu=experiment.training.use_multi_gpu,
            use_locomotion_only_early_training=experiment.training.use_locomotion_only_early_training,
            expose_environment_observation=experiment.observation.expose_environment,
            expose_viscosity_observation=experiment.observation.expose_viscosity,
            anisotropic_drag_mode=experiment.anisotropy.mode if experiment.anisotropy.mode != "full" else "proxy",
            anisotropic_drag_ratio=experiment.anisotropy.drag_ratio,
            anisotropic_drag_gain=experiment.anisotropy.normal_gain,
            anisotropic_drag_land_only=not experiment.anisotropy.apply_in_water,
        )
        self.artifact_namer = ExperimentArtifactNamer(
            experiment_name=experiment.name,
            run_root=self.output_root,
            model_type=experiment.training.model_type,
            n_links=experiment.training.n_links,
            algorithm=experiment.training.algorithm,
            additional_config={
                "oscillator_period": experiment.training.oscillator_period,
                "training_mode": "micro_publication",
            },
        )

    def create_environment(self):
        return TonicMicroPublicationWrapper(
            n_links=self.n_links,
            time_feature=True,
            desired_speed=0.15,
            observation_config={
                "expose_environment": self.experiment.observation.expose_environment,
                "expose_viscosity": self.experiment.observation.expose_viscosity,
                "expose_target": self.experiment.observation.expose_target,
            },
            anisotropy_config={
                "mode": self.experiment.anisotropy.mode,
                "drag_ratio": self.experiment.anisotropy.drag_ratio,
                "tangential_gain": self.experiment.anisotropy.tangential_gain,
                "normal_gain": self.experiment.anisotropy.normal_gain,
                "quadratic_drag": self.experiment.anisotropy.quadratic_drag,
                "apply_in_water": self.experiment.anisotropy.apply_in_water,
                "apply_in_land": self.experiment.anisotropy.apply_in_land,
            },
            reward_config=self.experiment.reward.to_dict(),
            land_start_probability=self.experiment.training.land_start_probability,
        )

    def create_model(self):
        model = super().create_model()
        actual_model = model.module if hasattr(model, "module") else model
        with_goal_sensitivity = hasattr(actual_model, "goal_sensitivity")
        with_goal_persistence = hasattr(actual_model, "goal_persistence")
        if with_goal_sensitivity:
            actual_model.goal_sensitivity.data.fill_(self.experiment.training.goal_sensitivity_override)
        if with_goal_persistence:
            actual_model.goal_persistence.data.fill_(self.experiment.training.goal_persistence_override)
        return model

    def run(self):
        write_json(os.path.join(self.output_root, "experiment_manifest.json"), self.experiment.to_dict())
        train_result = self.train() if self.experiment.training.training_steps > 0 else {}
        extra_result = None
        if isinstance(train_result, tuple):
            final_eval = {}
            extras = []
            for item in train_result:
                if isinstance(item, dict) and not final_eval:
                    final_eval = item
                else:
                    extras.append(item)
            extra_result = extras if extras else None
        else:
            final_eval = train_result
        inherited_summary_path = self.artifact_namer.experiment_summary_name()
        parsed_summary = parse_training_summary(inherited_summary_path)
        summary = build_experiment_summary(final_eval or {}, parsed_summary=parsed_summary)
        if extra_result is not None:
            summary["extra_result"] = list(extra_result)
        make_experiment_report(
            output_dir=self.output_root,
            config=self.experiment.to_dict(),
            summary=summary,
            runtime_notes=[
                "This package is a clean micro-publication path separated from the legacy curriculum code.",
                "The full anisotropic mode is implemented as a per-segment directional drag model in the new environment package.",
            ],
        )
        return {
            "experiment": self.experiment.name,
            "output_dir": self.output_root,
            "summary": summary,
            "final_eval": final_eval,
        }


def run_named_experiment(name: str):
    experiment = build_experiment(name)
    trainer = MicroPublicationTrainer(experiment)
    return trainer.run()
