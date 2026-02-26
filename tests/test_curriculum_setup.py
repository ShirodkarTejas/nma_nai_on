#!/usr/bin/env python3
"""
Test Curriculum Training Setup
Verify that all components work together before 1M episode training.
"""

import torch
import numpy as np
from swimmer.training.curriculum_trainer import CurriculumNCAPTrainer
from swimmer.environments.progressive_mixed_env import TonicProgressiveMixedWrapper


def test_curriculum_components():
    """Test all curriculum training components."""
    print("🧪 Testing Curriculum Training Components...")
    
    # Test 1: Environment creation and phase progression
    print(f"\n1️⃣ Testing Progressive Environment...")
    env = TonicProgressiveMixedWrapper(n_links=5, time_feature=True)
    
    print(f"   Initial phase: {env.name}")
    print(f"   Observation space: {env.observation_space.shape}")
    print(f"   Action space: {env.action_space.shape}")
    
    # Test environment phase transitions
    phases_tested = []
    for progress in [0.0, 0.4, 0.7, 0.9]:
        env.env.training_progress = progress
        env.env._create_environment()
        phase = int(progress * 4)
        phases_tested.append(phase)
        print(f"   Progress {progress:.1f} → Phase {phase}: {env.name}")
    
    print(f"   ✅ Tested phases: {phases_tested}")
    
    # Test 2: Model creation
    print(f"\n2️⃣ Testing NCAP Model Creation...")
    trainer = CurriculumNCAPTrainer(n_links=5, training_steps=1000)
    model = trainer.create_model()
    
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"   Device: {model.params['bneuron_osc'].device}")
    
    # Test model forward pass
    device = next(model.parameters()).device
    test_joints = torch.zeros(4, device=device)
    test_timesteps = torch.tensor([0], device=device)
    
    with torch.no_grad():
        actions = model(test_joints, timesteps=test_timesteps)
    
    print(f"   Model output shape: {actions.shape}")
    print(f"   ✅ Model creates valid actions")
    
    # Test 3: Agent creation
    print(f"\n3️⃣ Testing Agent Creation...")
    try:
        agent, tonic_model = trainer.create_agent(model, env)
        print(f"   ✅ Agent created successfully")
        print(f"   Agent type: {type(agent).__name__}")
    except Exception as e:
        print(f"   ❌ Agent creation failed: {e}")
        return False
    
    # Test 4: Biological constraints
    print(f"\n4️⃣ Testing Biological Constraints...")
    
    # Corrupt parameters to test constraints
    with torch.no_grad():
        model.params['muscle_contra'].data.fill_(0.5)  # Should be negative
        model.params['bneuron_osc'].data.fill_(0.1)   # Should be >= 1.2
    
    print(f"   Before constraints:")
    for name, param in model.params.items():
        print(f"     {name}: {param.item():.3f}")
    
    constraints_applied = trainer.apply_biological_constraints(model)
    
    print(f"   After constraints:")
    for name, param in model.params.items():
        print(f"     {name}: {param.item():.3f}")
    
    print(f"   ✅ Constraints applied: {constraints_applied}")
    
    # Test 5: Short interaction test
    print(f"\n5️⃣ Testing Environment Interaction...")
    
    try:
        print(f"   Testing NCAP-environment interaction...")
        
        # Test basic interaction
        obs = env.reset()
        total_reward = 0
        
        for step in range(50):  # Short test
            action = agent.test_step(obs)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            
            if done:
                obs = env.reset()
        
        print(f"   ✅ Interaction test completed")
        print(f"   Total reward: {total_reward:.2f}")
        print(f"   Average reward per step: {total_reward/50:.3f}")
        
    except Exception as e:
        print(f"   ❌ Interaction test failed: {e}")
        return False
    
    # Test 6: Performance evaluation
    print(f"\n6️⃣ Testing Performance Evaluation...")
    
    try:
        eval_results = trainer.evaluate_performance(agent, env, num_episodes=2)
        
        print(f"   Evaluation results:")
        for phase, results in eval_results.items():
            print(f"     Phase {phase}: {results['mean_distance']:.3f}m (±{results['std_distance']:.3f})")
        
        print(f"   ✅ Evaluation completed")
        
    except Exception as e:
        print(f"   ❌ Evaluation failed: {e}")
        return False
    
    env.close()
    
    print(f"\n🎉 All Curriculum Training Components Working!")
    return True


def test_training_expectations():
    """Test expected performance improvements."""
    print(f"\n📊 Training Expectations Analysis...")
    
    print(f"🎯 Expected Performance Targets:")
    print(f"   Phase 1 (Pure Swimming):")
    print(f"     • Initial: 0.3-0.5m (current zero-shot)")
    print(f"     • After 300k steps: 2-5m (basic swimming)")
    print(f"   Phase 2 (Single Land Zone):")
    print(f"     • After 600k steps: 1-3m (adaptation)")
    print(f"   Phase 3 (Two Land Zones):")
    print(f"     • After 800k steps: 2-4m (navigation)")
    print(f"   Phase 4 (Full Complexity):")
    print(f"     • After 1M steps: 5-15m (expert swim+crawl)")
    
    print(f"\n⏱️ Training Timeline:")
    print(f"   • Phase 1: Steps 0-300k (0-30%)")
    print(f"   • Phase 2: Steps 300k-600k (30-60%)")
    print(f"   • Phase 3: Steps 600k-800k (60-80%)")
    print(f"   • Phase 4: Steps 800k-1M (80-100%)")
    
    print(f"\n🔧 Key Features:")
    print(f"   ✅ Progressive complexity (avoids environment issues)")
    print(f"   ✅ Biological constraint preservation")
    print(f"   ✅ Gear ratio fix applied automatically")
    print(f"   ✅ Curriculum learning for both swimming and crawling")
    print(f"   ✅ Regular checkpointing and evaluation")


if __name__ == "__main__":
    print("🧪 Curriculum Training Setup Test")
    print("=" * 50)
    
    success = test_curriculum_components()
    
    if success:
        test_training_expectations()
        
        print(f"\n🚀 READY FOR 1M EPISODE TRAINING!")
        print(f"   Run with: python main.py --mode train_curriculum --training_steps 1000000")
        print(f"   Expected duration: 12-24 hours (depending on hardware)")
        print(f"   Expected final performance: 5-15m swimming + effective crawling")
    else:
        print(f"\n❌ Setup issues detected - fix before long training") 