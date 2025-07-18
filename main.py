#!/usr/bin/env python3
"""
Improved Mixed Environment Swimmer with Training Integration
Main script for running the improved mixed environment swimmer with proper RL training.
"""

import argparse
import os
import sys
from swimmer.training import SwimmerTrainer
from swimmer.environments import test_improved_mixed_environment

def main():
    parser = argparse.ArgumentParser(description='Improved Mixed Environment Swimmer')
    parser.add_argument('--mode', choices=['test', 'train', 'evaluate'], default='test',
                       help='Mode to run: test (legacy), train (RL), or evaluate')
    parser.add_argument('--model', choices=['ncap', 'mlp'], default='ncap',
                       help='Model type to use')
    parser.add_argument('--algorithm', choices=['ppo'], default='ppo',
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
    
    args = parser.parse_args()
    
    if args.mode == 'test':
        # Run the improved mixed environment test
        print("Running improved mixed environment test...")
        
        # Use trained model if provided, otherwise use default
        if args.load_model:
            print(f"Using trained model: {args.load_model}")
            results = test_improved_mixed_environment(
                n_links=args.n_links, 
                oscillator_period=60, 
                amplitude=3.0,
                trained_model_path=args.load_model
            )
        else:
            print("Using default NCAP model")
            results = test_improved_mixed_environment(
                n_links=args.n_links, 
                oscillator_period=60, 
                amplitude=3.0
            )
        
        if results:
            print("\n=== IMPROVED ADAPTATION SUMMARY ===")
            print(f"Environment transitions detected: {results['env_transitions']}")
            print(f"Final performance: {results['avg_velocity']:.4f} avg velocity")
            print(f"Total distance: {results['total_distance']:.4f}")
            print(f"Maximum velocity: {results['max_velocity']:.4f}")
            print(f"Video: outputs/improved_mixed_env/improved_adaptation_{args.n_links}links.mp4")
            print(f"Plots: outputs/improved_mixed_env/improved_environment_analysis_{args.n_links}links.png")
    
    elif args.mode == 'train':
        # Create trainer and train
        trainer = SwimmerTrainer(
            model_type=args.model,
            algorithm=args.algorithm,
            n_links=args.n_links,
            training_steps=args.training_steps,
            save_steps=args.save_steps,
            log_episodes=args.log_episodes
        )
        
        print(f"Starting training with {args.model.upper()} model and {args.algorithm.upper()} algorithm")
        print(f"Training for {args.training_steps} steps with saves every {args.save_steps} steps")
        print(f"Progress logged every {args.log_episodes} episodes")
        
        trainer.train()
        
        # Evaluate after training
        print("\nEvaluating trained model...")
        eval_results = trainer.evaluate(num_episodes=5)
        
        print("\n=== TRAINING COMPLETED ===")
        print(f"Final evaluation results:")
        print(f"  Average Reward: {eval_results['avg_reward']:.4f}")
        print(f"  Average Episode Length: {eval_results['avg_length']:.1f}")
        print(f"Training logs and plots saved to: outputs/training_logs/")
    
    elif args.mode == 'evaluate':
        if args.load_model is None:
            print("Error: --load_model is required for evaluation mode")
            sys.exit(1)
        
        # Create trainer and load model
        trainer = SwimmerTrainer(
            model_type=args.model,
            algorithm=args.algorithm,
            n_links=args.n_links
        )
        
        print(f"Loading model from: {args.load_model}")
        
        # Use appropriate loading method based on model type
        if args.model == 'ncap' and args.algorithm == 'ppo':
            # Load Tonic model
            trainer.load_tonic_model(args.load_model)
        else:
            # Load standard model
            trainer.load_model(args.load_model)
        
        # Evaluate in mixed environment using existing infrastructure
        print("Evaluating loaded model in mixed environment...")
        mixed_env_results = trainer.evaluate_mixed_environment()
        
        if mixed_env_results:
            print("\n=== MIXED ENVIRONMENT EVALUATION RESULTS ===")
            print(f"Environment transitions detected: {mixed_env_results['env_transitions']}")
            print(f"Final performance: {mixed_env_results['avg_velocity']:.4f} avg velocity")
            print(f"Total distance: {mixed_env_results['total_distance']:.4f}")
            print(f"Maximum velocity: {mixed_env_results['max_velocity']:.4f}")
            print(f"Video: outputs/improved_mixed_env/trained_model_evaluation_{args.n_links}links.mp4")
            print(f"Plots: outputs/improved_mixed_env/trained_model_analysis_{args.n_links}links.png")
        
        # Also run simple environment evaluation for comparison
        print("\nEvaluating in simple environment for comparison...")
        eval_results = trainer.evaluate(num_episodes=2)
        
        print("\n=== SIMPLE ENVIRONMENT EVALUATION RESULTS ===")
        print(f"Average Reward: {eval_results['avg_reward']:.4f}")
        print(f"Average Episode Length: {eval_results['avg_length']:.1f}")
        print(f"Reward range: {min(eval_results['rewards']):.2f} - {max(eval_results['rewards']):.2f}")

if __name__ == "__main__":
    main() 