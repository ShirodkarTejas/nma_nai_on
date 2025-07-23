#!/usr/bin/env python3
"""
Compare original notebook implementation vs our complex environment.
This will help isolate whether the issue is environment/physics or model.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from dm_control import suite
import collections
from dm_control.utils import rewards
from dm_control.rl import control
import dm_control.suite.swimmer as swimmer

# ============================================================================
# EXACT NOTEBOOK IMPLEMENTATION (simplified)
# ============================================================================

def excitatory(w, upper=None):
    return w.clamp(min=0, max=upper)

def inhibitory(w, lower=None):
    return w.clamp(min=lower, max=0)

def graded(x):
    return x.clamp(min=0, max=1)

def excitatory_constant(shape=(1,), value=1.):
    return nn.Parameter(torch.full(shape, value))

def inhibitory_constant(shape=(1,), value=-1.):
    return nn.Parameter(torch.full(shape, value))

class NotebookSwimmerModule(nn.Module):
    """Exact implementation from the notebook - simplified NCAP."""
    
    def __init__(self, n_joints=5, oscillator_period=60):
        super().__init__()
        self.n_joints = n_joints
        self.oscillator_period = oscillator_period
        self.timestep = 0
        
        # Simple shared parameters from notebook
        self.params = nn.ParameterDict()
        self.params['bneuron_prop'] = excitatory_constant()
        self.params['bneuron_osc'] = excitatory_constant()
        self.params['muscle_ipsi'] = excitatory_constant()
        self.params['muscle_contra'] = inhibitory_constant()
    
    def reset(self):
        self.timestep = 0
    
    def forward(self, joint_pos, timesteps=None):
        """Simplified forward pass matching the notebook."""
        
        # Separate into dorsal and ventral sensor values
        joint_pos_d = joint_pos.clamp(min=0, max=1)
        joint_pos_v = joint_pos.clamp(min=-1, max=0).neg()
        
        joint_torques = []
        
        for i in range(self.n_joints):
            bneuron_d = bneuron_v = torch.zeros_like(joint_pos[..., 0, None])
            
            # Proprioceptive input from previous joint
            if i > 0:
                bneuron_d = bneuron_d + joint_pos_d[..., i-1, None] * excitatory(self.params['bneuron_prop'])
                bneuron_v = bneuron_v + joint_pos_v[..., i-1, None] * excitatory(self.params['bneuron_prop'])
            
            # Head oscillator (first joint only)
            if i == 0:
                if timesteps is not None:
                    phase = timesteps.round().remainder(self.oscillator_period)
                    mask = phase < self.oscillator_period // 2
                    oscillator_d = torch.zeros_like(timesteps)
                    oscillator_v = torch.zeros_like(timesteps)
                    oscillator_d[mask] = 1.
                    oscillator_v[~mask] = 1.
                else:
                    phase = self.timestep % self.oscillator_period
                    if phase < self.oscillator_period // 2:
                        oscillator_d, oscillator_v = 1.0, 0.0
                    else:
                        oscillator_d, oscillator_v = 0.0, 1.0
                
                bneuron_d = bneuron_d + oscillator_d * excitatory(self.params['bneuron_osc'])
                bneuron_v = bneuron_v + oscillator_v * excitatory(self.params['bneuron_osc'])
            
            # B-neuron activation
            bneuron_d = graded(bneuron_d)
            bneuron_v = graded(bneuron_v)
            
            # Muscle activation with antagonistic control
            muscle_d = graded(
                bneuron_d * excitatory(self.params['muscle_ipsi']) +
                bneuron_v * inhibitory(self.params['muscle_contra'])
            )
            muscle_v = graded(
                bneuron_v * excitatory(self.params['muscle_ipsi']) +
                bneuron_d * inhibitory(self.params['muscle_contra'])
            )
            
            # Joint torque from antagonistic muscle contraction
            joint_torque = muscle_d - muscle_v
            joint_torques.append(joint_torque)
        
        self.timestep += 1
        return torch.cat(joint_torques, -1)


# ============================================================================
# NOTEBOOK ENVIRONMENT SETUP
# ============================================================================

_SWIM_SPEED = 0.1

class NotebookSwim(swimmer.Swimmer):
    """Exact swim task from notebook."""
    
    def __init__(self, desired_speed=_SWIM_SPEED, **kwargs):
        super().__init__(**kwargs)
        self._desired_speed = desired_speed

    def initialize_episode(self, physics):
        super().initialize_episode(physics)
        # Hide target
        physics.named.model.mat_rgba['target', 'a'] = 1
        physics.named.model.mat_rgba['target_default', 'a'] = 1
        physics.named.model.mat_rgba['target_highlight', 'a'] = 1

    def get_observation(self, physics):
        """Returns observation of joint angles and body velocities."""
        obs = collections.OrderedDict()
        obs['joints'] = physics.joints()
        obs['body_velocities'] = physics.body_velocities()
        return obs

    def get_reward(self, physics):
        """Simple forward swimming reward."""
        forward_velocity = -physics.named.data.sensordata['head_vel'][1]
        return rewards.tolerance(
            forward_velocity,
            bounds=(self._desired_speed, float('inf')),
            margin=self._desired_speed,
            value_at_margin=0.,
            sigmoid='linear',
        )

@swimmer.SUITE.add()
def notebook_swim(
    n_links=6,
    desired_speed=_SWIM_SPEED,
    time_limit=swimmer._DEFAULT_TIME_LIMIT,
    random=None,
    environment_kwargs={},
):
    """Notebook swim task."""
    model_string, assets = swimmer.get_model_and_assets(n_links)
    physics = swimmer.Physics.from_xml_string(model_string, assets=assets)
    task = NotebookSwim(desired_speed=desired_speed, random=random)
    return control.Environment(
        physics,
        task,
        time_limit=time_limit,
        control_timestep=swimmer._CONTROL_TIMESTEP,
        **environment_kwargs,
    )


# ============================================================================
# COMPARISON TEST
# ============================================================================

def test_notebook_vs_our_environment():
    """Compare notebook implementation vs our complex environment."""
    print("🔬 Testing Notebook vs Our Environment Implementation...")
    
    # Create notebook-style NCAP model
    notebook_model = NotebookSwimmerModule(n_joints=5, oscillator_period=60)
    
    print(f"📊 Notebook model parameters:")
    for name, param in notebook_model.params.items():
        print(f"  {name}: {param.item():.3f}")
    
    # Test 1: Notebook model + Notebook environment
    print(f"\n🧪 Test 1: Notebook NCAP + Notebook Environment")
    distance_notebook = test_swimming_in_environment(notebook_model, "notebook")
    
    # Test 2: Notebook model + Our complex environment  
    print(f"\n🧪 Test 2: Notebook NCAP + Our Complex Environment")
    distance_complex = test_swimming_in_environment(notebook_model, "complex")
    
    # Test 3: Notebook model + Our environment with gear fix
    print(f"\n🧪 Test 3: Notebook NCAP + Our Environment + Gear Fix")
    distance_gear_fix = test_swimming_in_environment(notebook_model, "gear_fix")
    
    # Analysis
    print(f"\n📈 Results Comparison:")
    print(f"  Notebook env:     {distance_notebook:.4f}m")
    print(f"  Complex env:      {distance_complex:.4f}m") 
    print(f"  Gear fix env:     {distance_gear_fix:.4f}m")
    
    print(f"\n💡 Analysis:")
    
    # Check if notebook environment works better
    if distance_notebook > distance_complex * 2:
        print("❌ ENVIRONMENT ISSUE: Notebook environment works much better!")
        print("   - Our complex environment may have physics problems")
        print("   - Recommendation: Use simpler environment or fix physics")
    elif distance_gear_fix > distance_complex * 2:
        print("❌ GEAR RATIO ISSUE: Gear fix is essential")
        print("   - Environment needs gear ratio correction")
        print("   - Recommendation: Always apply gear fix")
    elif distance_notebook < 0.05:
        print("❌ MODEL ISSUE: Even notebook model doesn't swim well")
        print("   - Fundamental NCAP implementation problem")
        print("   - Recommendation: Check model architecture")
    else:
        print("✅ ENVIRONMENTS SIMILAR: No major environment issues")
        print("   - Focus on biological parameter preservation")
    
    return {
        'notebook': distance_notebook,
        'complex': distance_complex,
        'gear_fix': distance_gear_fix
    }


def test_swimming_in_environment(model, env_type):
    """Test swimming performance in different environments."""
    
    print(f"  Creating {env_type} environment...")
    
    if env_type == "notebook":
        # Create notebook-style environment
        try:
            env = suite.load('swimmer', 'notebook_swim', task_kwargs={'random': 1, 'n_links': 6})
            physics = env.physics
        except Exception as e:
            print(f"  ⚠️ Failed to create notebook environment: {e}")
            # Fall back to standard swimmer
            env = suite.load('swimmer', 'swimmer', task_kwargs={'random': 1, 'n_links': 6})
            physics = env.physics
        
    elif env_type == "complex":
        # Create our complex environment
        from swimmer.environments.mixed_environment import ImprovedMixedSwimmerEnv
        wrapped_env = ImprovedMixedSwimmerEnv(n_links=5)
        physics = wrapped_env.physics
        env = wrapped_env.env
        
    elif env_type == "gear_fix":
        # Create our environment with gear fix
        from swimmer.environments.physics_fix import create_fixed_swimmer_env
        wrapped_env = create_fixed_swimmer_env(n_links=5, gear_ratio=0.1)
        physics = wrapped_env.physics  
        env = wrapped_env
    
    # Reset and get initial position
    print(f"  Resetting environment...")
    time_step = env.reset()
    
    # Handle dm_env TimeStep vs direct observation
    if hasattr(time_step, 'observation'):
        obs = time_step.observation
    else:
        obs = time_step
    
    print(f"  Observation type: {type(obs)}")
    if isinstance(obs, dict):
        print(f"  Observation keys: {list(obs.keys())}")
        for key, value in obs.items():
            if hasattr(value, 'shape'):
                print(f"    {key}: shape {value.shape}")
    elif obs is not None:
        print(f"  Observation shape: {obs.shape if hasattr(obs, 'shape') else 'no shape'}")
    
    initial_pos = physics.named.data.xpos['head', :2].copy()
    
    # Run swimming test
    for step in range(120):  # 2 seconds
        # Get joint positions (handle different observation formats)
        if isinstance(obs, dict):
            if 'joints' in obs:
                joint_pos = torch.tensor(obs['joints'], dtype=torch.float32)
            else:
                # Try to find joint data in the dict
                for key, value in obs.items():
                    if value is not None and hasattr(value, '__len__'):
                        joint_pos = torch.tensor(value[:5], dtype=torch.float32)
                        break
                else:
                    print(f"  ⚠️ No valid joint data found in obs: {obs}")
                    return 0.0
        elif obs is not None:
            joint_pos = torch.tensor(obs[:5], dtype=torch.float32)
        else:
            print(f"  ⚠️ Observation is None")
            return 0.0
        
        # Get action from model
        with torch.no_grad():
            action = model(joint_pos, timesteps=torch.tensor([step], dtype=torch.float32))
            action = action.cpu().numpy()
        
        # Handle different action dimensions (5 vs 4 actuators)
        if len(action) == 5 and env_type != "notebook":
            action = action[:4]  # Use first 4 for 5-link swimmer
        
        # Step environment
        try:
            time_step = env.step(action)
            
            # Handle different return formats
            if hasattr(time_step, 'observation'):
                # dm_env TimeStep format
                obs = time_step.observation
            elif len(time_step) == 4:
                # Gym format (obs, reward, done, info)
                obs, reward, done, info = time_step
            else:
                obs = time_step
        except Exception as e:
            print(f"  ⚠️ Step error: {e}")
            break
    
    # Calculate final distance
    final_pos = physics.named.data.xpos['head', :2].copy()
    distance = np.linalg.norm(final_pos - initial_pos)
    
    print(f"  Distance: {distance:.4f}m")
    
    if hasattr(env, 'close'):
        env.close()
    
    return distance


if __name__ == "__main__":
    test_notebook_vs_our_environment() 