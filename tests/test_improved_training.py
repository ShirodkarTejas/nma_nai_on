#!/usr/bin/env python3
"""
Test script for improved NCAP training
Tests the new stability-focused training approach.
"""

import sys
import os
from swimmer.training import ImprovedNCAPTrainer

def test_improved_ncap_training():
    """Test the improved NCAP training with stability measures."""
    print("=== TESTING IMPROVED NCAP TRAINING ===")
    
    # Create improved trainer with conservative settings
    trainer = ImprovedNCAPTrainer(
        n_links=6,
        training_steps=5000,  # Start with short training for testing
        save_steps=1000,
        log_episodes=5
    )
    
    try:
        # Run training
        print("Starting improved NCAP training...")
        trainer.train()
        
        # Compare with default
        print("Comparing with default NCAP...")
        comparison = trainer.compare_with_default_ncap()
        
        # Print results
        if comparison['success']:
            print("✅ SUCCESS: Improved training achieved comparable performance!")
        else:
            print("❌ NEEDS WORK: Training still has issues to address")
            
        return comparison
        
    except Exception as e:
        print(f"❌ ERROR during training: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_stability_measures():
    """Test the stability monitoring and parameter constraint features."""
    print("\n=== TESTING STABILITY MEASURES ===")
    
    trainer = ImprovedNCAPTrainer(n_links=6)
    
    # Create a test model
    model = trainer.create_improved_ncap_model(6)
    tonic_model = trainer.create_tonic_ncap_model(6)
    
    # Test parameter monitoring
    print("Testing parameter stability monitoring...")
    stable = trainer.monitor_parameter_stability(tonic_model)
    print(f"Model stability check: {'✅ STABLE' if stable else '❌ UNSTABLE'}")
    
    # Test parameter constraints
    print("Testing parameter constraints...")
    trainer.apply_parameter_constraints(tonic_model)
    
    # Check values are in expected ranges
    amplitude = tonic_model.ncap.base_amplitude.item()
    frequency = tonic_model.ncap.base_frequency.item()
    
    print(f"Amplitude after constraints: {amplitude:.3f}")
    print(f"Frequency after constraints: {frequency:.3f}")
    
    amplitude_ok = 0.1 <= amplitude <= 3.0
    frequency_ok = 0.05 <= frequency <= 0.5
    
    print(f"Amplitude in range: {'✅' if amplitude_ok else '❌'}")
    print(f"Frequency in range: {'✅' if frequency_ok else '❌'}")
    
    return amplitude_ok and frequency_ok

if __name__ == "__main__":
    print("Starting improved NCAP training tests...")
    
    # Test stability measures first
    stability_ok = test_stability_measures()
    print(f"Stability measures test: {'✅ PASSED' if stability_ok else '❌ FAILED'}")
    
    if stability_ok:
        # Run training test
        training_result = test_improved_ncap_training()
        if training_result and training_result['success']:
            print("\n🎉 ALL TESTS PASSED! Improved training is working.")
        else:
            print("\n⚠️  Training completed but needs further improvements.")
    else:
        print("\n❌ Stability measures failed, skipping training test.")
    
    print("\nTest completed.") 