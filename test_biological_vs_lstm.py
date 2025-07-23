#!/usr/bin/env python3
"""
Test: Biological NCAP vs LSTM-based NCAP
Compare environment switching capability with and without LSTM memory.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os

# Create outputs directory if it doesn't exist
os.makedirs('outputs', exist_ok=True)

from swimmer.models.biological_ncap import BiologicalNCAPSwimmer, BiologicalNCAPActor
from swimmer.models.ncap_swimmer import NCAPSwimmer, NCAPSwimmerActor

def test_environment_adaptation(model_name, swimmer_model, num_episodes=10, episode_length=200):
    """Test how well a model adapts to different environments."""
    print(f"\n🧪 Testing {model_name}...")
    
    # Create test environments
    water_env_type = np.array([1.0, 0.0, 0.1], dtype=np.float32)  # Water (low viscosity)
    land_env_type = np.array([0.0, 1.0, 0.8], dtype=np.float32)   # Land (high viscosity)
    
    results = {
        'water_torques': [],
        'land_torques': [],
        'water_periods': [],
        'land_periods': [],
        'adaptation_difference': []
    }
    
    for episode in range(num_episodes):
        print(f"  Episode {episode + 1}/{num_episodes}")
        
        # Reset model
        swimmer_model.reset()
        
        # Test water environment
        water_torques = []
        for step in range(episode_length):
            # Simulate joint positions (swimming motion)
            joint_pos = 0.3 * np.sin(np.arange(swimmer_model.n_joints) * 0.5 + step * 0.1)
            torques = swimmer_model(joint_pos, environment_type=water_env_type)
            
            if isinstance(torques, torch.Tensor):
                torques = torques.detach().cpu().numpy()
            water_torques.append(np.mean(np.abs(torques)))
        
        # Reset and test land environment
        swimmer_model.reset()
        
        land_torques = []
        for step in range(episode_length):
            # Same joint positions, different environment
            joint_pos = 0.3 * np.sin(np.arange(swimmer_model.n_joints) * 0.5 + step * 0.1)
            torques = swimmer_model(joint_pos, environment_type=land_env_type)
            
            if isinstance(torques, torch.Tensor):
                torques = torques.detach().cpu().numpy()
            land_torques.append(np.mean(np.abs(torques)))
        
        # Store results
        results['water_torques'].append(np.mean(water_torques))
        results['land_torques'].append(np.mean(land_torques))
        
        # Check oscillator period adaptation (for biological model)
        if hasattr(swimmer_model, 'current_oscillator_period'):
            # Test period adaptation by checking what periods the model uses
            swimmer_model.reset()
            _ = swimmer_model(joint_pos, environment_type=water_env_type)
            water_period = swimmer_model.current_oscillator_period
            
            swimmer_model.reset()
            _ = swimmer_model(joint_pos, environment_type=land_env_type)
            land_period = swimmer_model.current_oscillator_period
            
            results['water_periods'].append(water_period)
            results['land_periods'].append(land_period)
        
        # Calculate adaptation difference
        adaptation_diff = abs(np.mean(land_torques) - np.mean(water_torques))
        results['adaptation_difference'].append(adaptation_diff)
    
    return results

def analyze_adaptation_capability():
    """Compare biological vs LSTM-based adaptation."""
    
    print("🔬 Environment Adaptation Capability Test")
    print("=" * 50)
    
    # Create both models
    n_joints = 6
    
    print("Creating Biological NCAP (no LSTM)...")
    bio_swimmer = BiologicalNCAPSwimmer(
        n_joints=n_joints,
        oscillator_period=60,
        include_environment_adaptation=True
    )
    
    print("Creating Complex NCAP (with LSTM)...")
    lstm_swimmer = NCAPSwimmer(
        n_joints=n_joints,
        oscillator_period=60,
        memory_size=10,
        include_environment_adaptation=True
    )
    
    # Test both models
    bio_results = test_environment_adaptation("Biological NCAP", bio_swimmer)
    lstm_results = test_environment_adaptation("LSTM NCAP", lstm_swimmer)
    
    # Analysis
    print("\n📊 ADAPTATION ANALYSIS")
    print("=" * 50)
    
    # Calculate adaptation metrics
    bio_water_avg = np.mean(bio_results['water_torques'])
    bio_land_avg = np.mean(bio_results['land_torques'])
    bio_adaptation = abs(bio_land_avg - bio_water_avg)
    
    lstm_water_avg = np.mean(lstm_results['water_torques'])
    lstm_land_avg = np.mean(lstm_results['land_torques'])
    lstm_adaptation = abs(lstm_land_avg - lstm_water_avg)
    
    print(f"🧬 Biological NCAP:")
    print(f"   Water torque: {bio_water_avg:.4f}")
    print(f"   Land torque:  {bio_land_avg:.4f}")
    print(f"   Adaptation:   {bio_adaptation:.4f}")
    
    if bio_results['water_periods']:
        bio_water_period = np.mean(bio_results['water_periods'])
        bio_land_period = np.mean(bio_results['land_periods'])
        print(f"   Water period: {bio_water_period:.1f}")
        print(f"   Land period:  {bio_land_period:.1f}")
    
    print(f"\n🤖 LSTM NCAP:")
    print(f"   Water torque: {lstm_water_avg:.4f}")
    print(f"   Land torque:  {lstm_land_avg:.4f}")
    print(f"   Adaptation:   {lstm_adaptation:.4f}")
    
    # Comparison
    print(f"\n🏆 COMPARISON:")
    print(f"   Biological adaptation: {bio_adaptation:.4f}")
    print(f"   LSTM adaptation:       {lstm_adaptation:.4f}")
    
    if bio_adaptation > lstm_adaptation * 0.8:  # Within 20%
        print("   ✅ Biological model shows comparable adaptation!")
        print("   🧠 LSTM may not be necessary for environment switching")
    else:
        print("   ⚠️  LSTM model shows significantly better adaptation")
        print("   🧠 LSTM may provide important memory benefits")
    
    # Biological plausibility assessment
    print(f"\n🔬 BIOLOGICAL PLAUSIBILITY:")
    print(f"   Biological NCAP: ⭐⭐⭐⭐⭐ (Highly plausible)")
    print(f"   LSTM NCAP:       ⭐⭐ (Artificial memory system)")
    
    # Parameter count comparison
    bio_params = sum(p.numel() for p in bio_swimmer.parameters())
    lstm_params = sum(p.numel() for p in lstm_swimmer.parameters())
    
    print(f"\n📈 MODEL COMPLEXITY:")
    print(f"   Biological NCAP: {bio_params} parameters")
    print(f"   LSTM NCAP:       {lstm_params} parameters")
    print(f"   Reduction:       {lstm_params - bio_params} parameters removed")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Torque comparison
    axes[0, 0].bar(['Water', 'Land'], [bio_water_avg, bio_land_avg], 
                   alpha=0.7, label='Biological NCAP', color='green')
    axes[0, 0].bar(['Water', 'Land'], [lstm_water_avg, lstm_land_avg], 
                   alpha=0.7, label='LSTM NCAP', color='blue', width=0.5)
    axes[0, 0].set_title('Torque Output by Environment')
    axes[0, 0].set_ylabel('Average Torque Magnitude')
    axes[0, 0].legend()
    
    # Adaptation over episodes
    axes[0, 1].plot(bio_results['adaptation_difference'], 'g-', label='Biological NCAP', linewidth=2)
    axes[0, 1].plot(lstm_results['adaptation_difference'], 'b-', label='LSTM NCAP', linewidth=2)
    axes[0, 1].set_title('Adaptation Difference Over Episodes')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('|Land - Water| Torque')
    axes[0, 1].legend()
    
    # Parameter counts
    axes[1, 0].bar(['Biological NCAP', 'LSTM NCAP'], [bio_params, lstm_params], 
                   color=['green', 'blue'], alpha=0.7)
    axes[1, 0].set_title('Model Complexity')
    axes[1, 0].set_ylabel('Number of Parameters')
    
    # Biological plausibility
    plausibility_scores = [5, 2]  # Out of 5 stars
    axes[1, 1].bar(['Biological NCAP', 'LSTM NCAP'], plausibility_scores,
                   color=['green', 'blue'], alpha=0.7)
    axes[1, 1].set_title('Biological Plausibility')
    axes[1, 1].set_ylabel('Stars (out of 5)')
    axes[1, 1].set_ylim(0, 5)
    
    plt.tight_layout()
    plt.savefig('outputs/biological_vs_lstm_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Comparison plot saved to: outputs/biological_vs_lstm_comparison.png")
    
    return bio_adaptation >= lstm_adaptation * 0.8  # True if biological is comparable

if __name__ == "__main__":
    # Run the comparison test
    can_swim_without_lstm = analyze_adaptation_capability()
    
    print(f"\n🎯 FINAL ANSWER:")
    if can_swim_without_lstm:
        print("✅ YES - The swimmer CAN learn environment switching without LSTM!")
        print("🧠 Biological mechanisms (neuromodulation) are sufficient")
        print("🔬 Recommend removing LSTM for better biological authenticity")
    else:
        print("❌ NO - LSTM provides significant adaptation benefits")
        print("🤖 Memory system appears necessary for environment switching")
        print("⚖️  Trade-off between biological plausibility and performance") 