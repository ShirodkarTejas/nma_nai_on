#!/usr/bin/env python3
"""
Improved Mixed Environment Swimmer with Training Integration
Main script for running the improved mixed environment swimmer with proper RL training.
"""

import argparse
from swimmer.training import ImprovedNCAPTrainer

def main():
    parser = argparse.ArgumentParser(description='Improved Mixed Environment Swimmer')
    parser.add_argument('--mode', choices=['train_improved', 'train_biological', 'train_curriculum', 'evaluate', 'evaluate_curriculum'], default='train_improved',
                       help='Mode to run: train_improved (stable NCAP), train_biological (preserve biology), train_curriculum (progressive swim+crawl), evaluate, or evaluate_curriculum (eval only)')
    parser.add_argument('--model', choices=['ncap', 'mlp'], default='ncap',
                       help='Model type to use')
    parser.add_argument('--algorithm', choices=['ppo', 'a2c'], default='ppo',
                       help='RL algorithm to use')
    parser.add_argument('--n_links', type=int, default=6,
                       help='Number of links in the swimmer')
    parser.add_argument('--training_steps', type=int, default=100000,
                       help='Number of training steps')
    parser.add_argument('--save_steps', type=int, default=20000,
                       help='Steps between model saves')
    parser.add_argument('--log_episodes', type=int, default=5,
                       help='Episodes between progress logs')
    parser.add_argument('--load_model', type=str, default=None,
                       help='Path to load a trained model')
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume curriculum training from')
    parser.add_argument('--eval_episodes', type=int, default=20,
                       help='Number of episodes per phase for evaluation')
    parser.add_argument('--eval_video_steps', type=int, default=400,
                       help='Number of steps per video for evaluation')
    parser.add_argument('--model_type', choices=['biological_ncap', 'enhanced_ncap'], default='enhanced_ncap',
                       help='NCAP model type for curriculum training')
    
    args = parser.parse_args()
    
    # train_simple and legacy modes removed
    if args.mode == 'train_improved':
        trainer = ImprovedNCAPTrainer(
            n_links=args.n_links,
            training_steps=args.training_steps,
            save_steps=args.save_steps,
            log_episodes=args.log_episodes
        )
        trainer.train()
    elif args.mode == 'train_biological':
        from swimmer.training.simple_biological_trainer import SimpleBiologicalTrainer
        trainer = SimpleBiologicalTrainer(
            n_links=args.n_links,
            training_steps=args.training_steps,
            save_steps=args.save_steps,
            log_episodes=args.log_episodes
        )
        trainer.train()
    elif args.mode == 'train_curriculum':
        print("🎓 Starting curriculum training for swimming and crawling...")
        from swimmer.training.curriculum_trainer import CurriculumNCAPTrainer
        trainer = CurriculumNCAPTrainer(
            n_links=args.n_links,
            learning_rate=3e-5,  # Conservative learning rate for long training
            training_steps=args.training_steps,
            save_steps=args.save_steps,
            log_episodes=args.log_episodes,
            resume_from_checkpoint=args.resume_checkpoint,
            model_type=args.model_type,
            algorithm=args.algorithm
        )
        trainer.train()
    elif args.mode == 'evaluate_curriculum':
        if args.resume_checkpoint is None:
            print("⛔  --resume_checkpoint is required for curriculum evaluation"); return
        
        print("📊 Starting curriculum evaluation from checkpoint...")
        from swimmer.training.curriculum_trainer import CurriculumNCAPTrainer
        trainer = CurriculumNCAPTrainer(
            n_links=args.n_links,
            training_steps=0,  # No training
            resume_from_checkpoint=args.resume_checkpoint,
            model_type=args.model_type,
            algorithm=args.algorithm
        )
        trainer.evaluate_only(
            eval_episodes=args.eval_episodes,
            video_steps=args.eval_video_steps
        )
    elif args.mode == 'evaluate':
        if args.load_model is None:
            print("⛔  --load_model is required for evaluation"); return

        trainer = ImprovedNCAPTrainer(n_links=args.n_links)
        trainer.load_tonic_model(args.load_model)
        trainer.evaluate_mixed_environment(max_frames=1800)
        
if __name__ == "__main__":
    main() 