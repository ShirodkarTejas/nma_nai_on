#!/usr/bin/env python3
"""
Improved Mixed Environment Swimmer with Training Integration
Main script for running the improved mixed environment swimmer with proper RL training.
"""

import argparse
import os
import sys
from swimmer.training import SwimmerTrainer, ImprovedNCAPTrainer, SimpleSwimmerTrainer
from swimmer.environments import test_improved_mixed_environment

def main():
    parser = argparse.ArgumentParser(description='Improved Mixed Environment Swimmer')
    parser.add_argument('--mode', choices=['test', 'train', 'train_improved', 'train_simple', 'evaluate'], default='test',
                       help='Mode to run: test (legacy), train (RL), train_improved (stable NCAP), train_simple (basic swimming), or evaluate')
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
    
    args = parser.parse_args()
    
    if args.mode == 'test':
        # Legacy test function
        print("Running legacy test function...")
        test_improved_mixed_environment(
            n_links=args.n_links,
            oscillator_period=60,
            amplitude=3.0
        )
    
    elif args.mode == 'evaluate':
        # Evaluate trained model
        if args.load_model is None:
            print("Error: --load_model required for evaluation mode")
            return
        
        print(f"Evaluating model: {args.load_model}")
        # Add evaluation logic here
        
    elif args.mode == 'train_simple':
        # Use improved NCAP training on simple swimming environment
        if args.model != 'ncap':
            print("Simple training is only available for NCAP model. Setting model to 'ncap'.")
        
        trainer = ImprovedNCAPTrainer(
            n_links=args.n_links,
            training_steps=args.training_steps,
            save_steps=args.save_steps,
            log_episodes=args.log_episodes
        )
        
        print(f"Starting NCAP training on simple swimming environment")
        print(f"Training for {args.training_steps} steps focusing on forward movement")
        print(f"Using proven stability fixes from ImprovedNCAPTrainer")
        
        # Use the simple swimming method
        agent, env, model = trainer.train_simple_swimming()
        
        # Evaluate the simple swimming performance
        print("\nEvaluating simple swimming performance...")
        simple_results = trainer.evaluate_simple_swimming()
        
        print("\n=== SIMPLE SWIMMING TRAINING COMPLETED ===")
        if simple_results['success']:
            print("✅ SUCCESS: Model learned basic forward swimming!")
            print(f"Distance achieved: {simple_results['avg_distance']:.3f} (target: ≥1.0)")
            print(f"Velocity achieved: {simple_results['avg_velocity']:.3f} m/s (target: ≥0.03)")
            print(f"Model saved to: outputs/training/simple_ncap_{args.n_links}links.pt")
            print("\n🎯 NEXT STEP: Use this model for transfer learning to mixed environments")
        else:
            print("⚠️  Basic swimming needs improvement")
            print("Consider training for more steps or adjusting hyperparameters")
        
        print(f"Training logs saved to: outputs/training_logs/")
    
    elif args.mode == 'train_improved':
        # Use improved NCAP training with stability measures
        if args.model != 'ncap':
            print("Improved training is only available for NCAP model. Setting model to 'ncap'.")
        
        trainer = ImprovedNCAPTrainer(
            n_links=args.n_links,
            training_steps=args.training_steps,
            save_steps=args.save_steps,
            log_episodes=args.log_episodes
        )
        
        print(f"Starting improved NCAP training with stability measures")
        print(f"Training for {args.training_steps} steps with saves every {args.save_steps} steps")
        print(f"Using A2C algorithm optimized for NCAP stability")
        
        trainer.train()
        
        # Compare with default NCAP
        print("\nComparing trained model with default NCAP...")
        comparison = trainer.compare_with_default_ncap()
        
        print("\n=== IMPROVED TRAINING COMPLETED ===")
        if comparison['success']:
            print("✅ SUCCESS: Training achieved comparable performance to default NCAP!")
            print(f"Trained model saved to: outputs/training/improved_ncap_{args.n_links}links")
        else:
            print("⚠️  Training completed but performance needs improvement")
            print("Consider adjusting hyperparameters or training for more steps")
        
        print(f"Training logs and analysis saved to: outputs/training_logs/")
    
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

if __name__ == "__main__":
    main() 