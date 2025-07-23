#!/usr/bin/env python3
"""
Simple Biological Trainer
A simplified version that focuses on preserving biological behavior.
"""

import torch
import torch.nn as nn
import numpy as np
import os
from swimmer.training.improved_ncap_trainer import ImprovedNCAPTrainer
from swimmer.models.simple_ncap import SimpleNCAPSwimmer
from ..environments.physics_fix import apply_swimming_physics_fix


class SimpleBiologicalTrainer(ImprovedNCAPTrainer):
    """
    Simplified trainer that preserves biological NCAP parameters.
    Extends the existing ImprovedNCAPTrainer with biological preservation.
    """
    
    def __init__(self, n_links=5, training_steps=50000, save_steps=10000,
                 output_dir='outputs/training', log_episodes=5):
        # Initialize with conservative parameters for biological preservation
        super().__init__(
            n_links=n_links,
            training_steps=training_steps,
            save_steps=save_steps,
            output_dir=output_dir,
            log_episodes=log_episodes
        )
        
        # Override with biological preservation settings
        self.ncap_learning_rate = 5e-5  # Even more conservative
        self.ncap_gradient_clip = 0.1   # Stricter clipping
        self.parameter_constraint_strength = 0.1
        
        # Biological preservation parameters  
        self.min_oscillator_strength = 1.2  # Strong minimum for reliable oscillation
        self.min_coupling_strength = 0.8    # Strong minimum for wave propagation
        
        # Physics fix parameters
        self.gear_ratio = 0.1  # Optimal gear ratio for effective swimming
        
        print("🧬 Initialized Simple Biological Trainer with conservative settings")
        print(f"🔧 Will apply gear ratio fix: {self.gear_ratio}")
    
    def create_improved_ncap_model(self, n_joints):
        """Create NCAP model with strong biological initialization."""
        print("🧬 Creating Simple Biological NCAP Model...")
        
        # Use simplified NCAP
        model = SimpleNCAPSwimmer(
            n_joints=n_joints,
            oscillator_period=60,
            use_weight_sharing=True,
            use_weight_constraints=True,
            include_proprioception=True,
            include_head_oscillators=True
        )
        
        # Apply strong initialization
        with torch.no_grad():
            if 'bneuron_osc' in model.params:
                model.params['bneuron_osc'].data = torch.tensor(2.0)
            if 'bneuron_prop' in model.params:
                model.params['bneuron_prop'].data = torch.tensor(1.8)
            if 'muscle_ipsi' in model.params:
                model.params['muscle_ipsi'].data = torch.tensor(1.5)
            if 'muscle_contra' in model.params:
                model.params['muscle_contra'].data = torch.tensor(-1.5)
        
        print(f"✅ Strong biological initialization applied:")
        for name, param in model.params.items():
            print(f"  {name}: {param.item():.3f}")
        
        model.to(self.device)
        return model
    
    def apply_biological_constraints(self, model):
        """Apply biological constraints during training."""
        if not hasattr(model, 'ncap'):
            return
        
        ncap = model.ncap
        if not hasattr(ncap, 'params'):
            return
        
        with torch.no_grad():
            # Enforce minimum oscillator strength
            if 'bneuron_osc' in ncap.params:
                current_val = ncap.params['bneuron_osc'].item()
                if abs(current_val) < self.min_oscillator_strength:
                    sign = 1 if current_val >= 0 else -1
                    ncap.params['bneuron_osc'].data = torch.tensor(sign * self.min_oscillator_strength)
            
            # Enforce minimum coupling strength
            if 'bneuron_prop' in ncap.params:
                current_val = ncap.params['bneuron_prop'].item()
                if current_val < self.min_coupling_strength:
                    ncap.params['bneuron_prop'].data = torch.tensor(self.min_coupling_strength)
            
            # Keep muscles balanced and strong with STRICT sign enforcement
            if 'muscle_ipsi' in ncap.params and 'muscle_contra' in ncap.params:
                ipsi_val = ncap.params['muscle_ipsi'].item()
                contra_val = ncap.params['muscle_contra'].item()
                
                # CRITICAL: Fix any sign violations immediately!
                sign_violation = False
                if ipsi_val < 0:  # Ipsi should be positive
                    ncap.params['muscle_ipsi'].data = torch.tensor(self.min_coupling_strength)
                    print(f"⚠️ FIXED: muscle_ipsi sign violation {ipsi_val:.3f} → +{self.min_coupling_strength:.3f}")
                    sign_violation = True
                    
                if contra_val >= 0:  # Contra should be negative (ZERO is also wrong!)
                    ncap.params['muscle_contra'].data = torch.tensor(-self.min_coupling_strength)
                    print(f"⚠️ FIXED: muscle_contra sign violation {contra_val:.3f} → -{self.min_coupling_strength:.3f}")
                    sign_violation = True
                
                # Then ensure minimum strength (after fixing signs)
                ipsi_val = ncap.params['muscle_ipsi'].item()
                contra_val = ncap.params['muscle_contra'].item()
                
                if ipsi_val < self.min_coupling_strength:
                    ncap.params['muscle_ipsi'].data = torch.tensor(self.min_coupling_strength)
                    
                if contra_val > -self.min_coupling_strength:  # Too weak (but negative)
                    ncap.params['muscle_contra'].data = torch.tensor(-self.min_coupling_strength)
                
                if sign_violation:
                    print(f"🧬 After constraint: ipsi=+{ncap.params['muscle_ipsi'].item():.3f}, contra={ncap.params['muscle_contra'].item():.3f}")
    
    def monitor_parameter_stability(self, model):
        """Override to include biological monitoring."""
        super().monitor_parameter_stability(model)
        
        # Apply biological constraints
        self.apply_biological_constraints(model)
        
        # Log biological parameters
        if hasattr(model, 'ncap') and hasattr(model.ncap, 'params'):
            params = model.ncap.params
            if len(self.training_metrics['interval_rewards']) % 10 == 0:  # Every 10 intervals
                print(f"🧬 Biological parameters:")
                for name, param in params.items():
                    print(f"  {name}: {param.item():.3f}")
    
    def apply_physics_fix(self, env):
        """Apply gear ratio fix to the environment."""
        try:
            physics = env.physics if hasattr(env, 'physics') else env.env.physics
            apply_swimming_physics_fix(physics, self.gear_ratio)
            print(f"✅ Applied gear ratio fix ({self.gear_ratio}) for effective swimming")
        except Exception as e:
            print(f"⚠️ Could not apply physics fix: {e}")
    
    def create_tonic_environment(self):
        """Override to create simple environment optimized for NCAP."""
        # Use simple environment instead of complex mixed environment
        from ..environments.simple_swimmer import TonicSimpleSwimmerWrapper
        env = TonicSimpleSwimmerWrapper(n_links=self.n_links, time_feature=True, desired_speed=0.1)
        
        print(f"🏊 Using simple swimmer environment for better NCAP performance")
        print(f"   Environment: {env.name}")
        print(f"   Observation space: {env.observation_space.shape}")
        print(f"   Action space: {env.action_space.shape}")
        
        return env
    
    def _run_interval_training(self, agent, env, model):
        """Override to apply biological constraints more frequently."""
        print(f"🧬 Running biological interval training with frequent constraint checks...")
        
        # Use shorter intervals for more frequent biological monitoring
        original_training_steps = self.training_steps
        interval_size = min(500, self.training_steps // 4)  # Shorter intervals for more checks
        completed_steps = 0
        
        while completed_steps < self.training_steps:
            remaining_steps = self.training_steps - completed_steps
            current_interval = min(interval_size, remaining_steps)
            
            print(f"\n🧬 Biological training interval: {completed_steps} to {completed_steps + current_interval}")
            
            # Apply biological constraints BEFORE each interval
            print("🔧 Pre-interval biological constraints...")
            self.apply_biological_constraints(model)
            
            # Create a mini-trainer for this interval
            from tonic import Trainer
            trainer = Trainer(
                steps=current_interval,
                save_steps=current_interval + 1,  # Don't save during interval
                test_episodes=0  # Skip testing during training
            )
            
            trainer.initialize(
                agent=agent,
                environment=env,
                test_environment=env
            )
            
            # Run the training interval
            try:
                trainer.run()
                print(f"✅ Biological interval {completed_steps}-{completed_steps + current_interval} completed")
            except Exception as e:
                print(f"⚠️ Training interval had issues: {e}")
                self.apply_biological_constraints(model)
            
            # Apply biological constraints AFTER each interval
            print("🔧 Post-interval biological constraints...")
            self.apply_biological_constraints(model)
            
            # Evaluate performance
            interval_reward = self._evaluate_interval_performance(agent, env)
            self.training_metrics['interval_rewards'].append(interval_reward)
            
            # Check for improvement (early stopping logic from parent)
            if interval_reward > self.training_metrics['best_reward'] + self.min_improvement_threshold:
                print(f"📈 Biological training improved: {interval_reward:.3f} (prev best: {self.training_metrics['best_reward']:.3f})")
                self.training_metrics['best_reward'] = interval_reward
                self.training_metrics['patience_counter'] = 0
            else:
                self.training_metrics['patience_counter'] += 1
                print(f"📉 No improvement: {interval_reward:.3f} (patience: {self.training_metrics['patience_counter']}/{self.early_stopping_patience})")
            
            # Early stopping
            if self.training_metrics['patience_counter'] >= self.early_stopping_patience:
                print(f"\n🛑 EARLY STOPPING TRIGGERED in biological training!")
                break
            
            completed_steps += current_interval
        
        # Final biological constraints
        print("🔧 Final biological constraints...")
        self.apply_biological_constraints(model)
        
        # Show final biological state
        if hasattr(model, 'ncap') and hasattr(model.ncap, 'params'):
            print(f"\n🧬 Final biological parameters:")
            for name, param in model.ncap.params.items():
                print(f"  {name}: {param.item():.3f}")
        
        print(f"🧬 Biological training completed: {completed_steps} total steps")
    
    def train(self):
        """Train with biological preservation and physics fix."""
        print("🧬 Starting Simple Biological NCAP Training...")
        print(f"   Learning rate: {self.ncap_learning_rate}")
        print(f"   Gradient clip: {self.ncap_gradient_clip}")
        print(f"   Min oscillator: {self.min_oscillator_strength}")
        print(f"   Min coupling: {self.min_coupling_strength}")
        print(f"   Gear ratio fix: {self.gear_ratio}")
        
        # Use the existing improved trainer logic but with biological model
        result = super().train()
        
        return result


def create_simple_biological_trainer(**kwargs):
    """Factory function for the simple biological trainer."""
    return SimpleBiologicalTrainer(**kwargs) 