#!/usr/bin/env python3
"""Main entry point for NMAP training and evaluation."""

try:
    import NMAP.gym_bridge  # noqa: F401
except ModuleNotFoundError:
    import gym_bridge  # noqa: F401

import argparse
import sys
from pathlib import Path


def _bootstrap_paths():
    """Ensure local package roots are importable when running as a script."""
    package_root = Path(__file__).resolve().parent
    repo_root = package_root.parent
    local_tonic_root = repo_root / "tonic"

    for path in (repo_root, local_tonic_root):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def _load_trainers():
    _bootstrap_paths()
    from NMAP.swimmer.training import NCAPTrainer, CurriculumNCAPTrainer
    return NCAPTrainer, CurriculumNCAPTrainer


def build_parser():
    parser = argparse.ArgumentParser(description='NMAP swimmer training and evaluation')
    parser.add_argument(
        '--mode',
        choices=['train', 'train_curriculum', 'evaluate', 'evaluate_curriculum'],
        default='train',
        help='Execution mode.',
    )
    parser.add_argument('--algorithm', choices=['ppo', 'a2c'], default='ppo', help='RL algorithm to use for training/evaluation.')
    parser.add_argument('--n_links', type=int, default=6, help='Number of links in the swimmer.')
    parser.add_argument('--training_steps', type=int, default=100000, help='Number of training steps.')
    parser.add_argument(
        '--log_dir',
        type=str,
        default='results/manual_run',
        help='Root directory for training logs, checkpoints, and evaluation artifacts.',
    )
    parser.add_argument('--save_steps', type=int, default=20000, help='Checkpoint/save interval in steps.')
    parser.add_argument('--log_episodes', type=int, default=5, help='Episode logging interval.')
    parser.add_argument('--load_model', type=str, default=None, help='Path to a trained tonic model for evaluation.')
    parser.add_argument(
        '--resume_checkpoint',
        type=str,
        default=None,
        help='Checkpoint path for curriculum resume/evaluation.',
    )
    parser.add_argument('--eval_episodes', type=int, default=20, help='Episodes per phase for curriculum evaluation.')
    parser.add_argument('--eval_video_steps', type=int, default=5000, help='Video/evaluation horizon in steps.')
    parser.add_argument(
        '--model_type',
        choices=['biological_ncap', 'enhanced_ncap'],
        default='enhanced_ncap',
        help='NCAP model type used by curriculum training.',
    )
    parser.add_argument(
        '--use_locomotion_only_early_training',
        action='store_true',
        help='Enable locomotion-only mode during early curriculum phases.',
    )
    parser.add_argument(
        '--sparse_init',
        action='store_true',
        help='Enable Cook2019 6-pathway sparse initialization for NCAP weights.',
    )
    parser.add_argument(
        '--sparse_reg_lambda',
        type=float,
        default=0.0,
        help='Sparse-prior regularization coefficient (0.0 disables regularization).',
    )
    parser.add_argument(
        '--force_oscillation',
        action='store_true',
        help='Enforce minimum action variance during actor updates.',
    )
    return parser


def run_train(args):
    ncap_trainer_cls, _ = _load_trainers()
    trainer = ncap_trainer_cls(
        n_links=args.n_links,
        algorithm=args.algorithm,
        training_steps=args.training_steps,
        save_steps=args.save_steps,
        log_episodes=args.log_episodes,
        log_dir=args.log_dir,
        sparse_init=args.sparse_init,
        sparse_reg_lambda=args.sparse_reg_lambda,
        force_oscillation=args.force_oscillation,
    )
    trainer.train()


def run_train_curriculum(args):
    _, curriculum_trainer_cls = _load_trainers()
    trainer = curriculum_trainer_cls(
        n_links=args.n_links,
        learning_rate=3e-5,
        training_steps=args.training_steps,
        save_steps=args.save_steps,
        log_episodes=args.log_episodes,
        log_dir=args.log_dir,
        resume_from_checkpoint=args.resume_checkpoint,
        model_type=args.model_type,
        algorithm=args.algorithm,
        use_locomotion_only_early_training=args.use_locomotion_only_early_training,
        sparse_init=args.sparse_init,
        sparse_reg_lambda=args.sparse_reg_lambda,
        force_oscillation=args.force_oscillation,
    )
    trainer.train()


def run_evaluate(args):
    if args.load_model is None:
        raise SystemExit('⛔ --load_model is required for evaluate mode')

    ncap_trainer_cls, _ = _load_trainers()
    trainer = ncap_trainer_cls(
        n_links=args.n_links,
        algorithm=args.algorithm,
        log_dir=args.log_dir,
        sparse_init=args.sparse_init,
        sparse_reg_lambda=args.sparse_reg_lambda,
        force_oscillation=args.force_oscillation,
    )
    trainer.load_tonic_model(args.load_model)
    trainer.evaluate_mixed_environment(max_frames=args.eval_video_steps)


def run_evaluate_curriculum(args):
    if args.resume_checkpoint is None:
        raise SystemExit('⛔ --resume_checkpoint is required for evaluate_curriculum mode')

    _, curriculum_trainer_cls = _load_trainers()
    trainer = curriculum_trainer_cls(
        n_links=args.n_links,
        training_steps=0,
        log_dir=args.log_dir,
        resume_from_checkpoint=args.resume_checkpoint,
        model_type=args.model_type,
        algorithm=args.algorithm,
        use_locomotion_only_early_training=args.use_locomotion_only_early_training,
        sparse_init=args.sparse_init,
        sparse_reg_lambda=args.sparse_reg_lambda,
        force_oscillation=args.force_oscillation,
    )
    trainer.evaluate_only(eval_episodes=args.eval_episodes, video_steps=args.eval_video_steps)


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == 'train':
        run_train(args)
    elif args.mode == 'train_curriculum':
        run_train_curriculum(args)
    elif args.mode == 'evaluate':
        run_evaluate(args)
    elif args.mode == 'evaluate_curriculum':
        run_evaluate_curriculum(args)


if __name__ == '__main__':
    main()
