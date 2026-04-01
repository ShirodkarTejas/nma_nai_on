import torch
import numpy as np
from swimmer.models.enhanced_biological_ncap import EnhancedBiologicalNCAPSwimmer

def test_model():
    n_links = 6
    n_joints = n_links - 1
    model = EnhancedBiologicalNCAPSwimmer(n_joints=n_joints)
    
    # Create dummy inputs
    batch_size = 1
    joint_pos = torch.zeros(batch_size, n_joints)
    env_type = torch.tensor([[1.0, 0.0, 0.5]]) # Water
    target_dir = torch.tensor([[1.0, 0.0]])
    
    print("Testing torque variation over 10 steps...")
    prev_torque = None
    for t in range(10):
        timesteps = torch.tensor([float(t)])
        with torch.no_grad():
            torques = model(joint_pos, environment_type=env_type, target_direction=target_dir, timesteps=timesteps)
        
        t_val = torques[0, 0].item()
        print(f"Step {t}: Torque[0,0] = {t_val:.4f}")
        
        if prev_torque is not None and torch.allclose(torques, prev_torque):
            print(f"⚠️ WARNING: Torques are IDENTICAL at step {t}")
        
        prev_torque = torques.clone()

if __name__ == "__main__":
    test_model()
