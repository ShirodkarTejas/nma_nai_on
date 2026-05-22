#!/usr/bin/env python3
"""
Simplified NCAP Implementation - Exact Match to Original Notebook
This version removes all complexity and matches the notebook exactly.
"""

import torch
import torch.nn as nn
import numpy as np


# Weight constraints from notebook
def excitatory(w, upper=None):
    return w.clamp(min=0, max=upper)

def inhibitory(w, lower=None):
    return w.clamp(min=lower, max=0)

def unsigned(w, lower=None, upper=None):
    return w if lower is None and upper is None else w.clamp(min=lower, max=upper)

# Activation constraints
def graded(x):
    """Graded activation function - key biological constraint!"""
    return x.clamp(min=0, max=1)

# Weight initialization functions
def excitatory_constant(shape=(1,), value=1.):
    return nn.Parameter(torch.full(shape, value))

def inhibitory_constant(shape=(1,), value=-1.):
    return nn.Parameter(torch.full(shape, value))


class SimpleNCAPSwimmer(nn.Module):
    """
    Simplified NCAP implementation that exactly matches the original notebook.
    No environment adaptation, no LSTM, just pure biological architecture.
    """
    
    def __init__(self, n_joints=5, oscillator_period=60, use_weight_sharing=True, 
                 use_weight_constraints=True, include_proprioception=True, 
                 include_head_oscillators=True):
        super().__init__()
        self.n_joints = n_joints
        self.oscillator_period = oscillator_period
        self.include_proprioception = include_proprioception
        self.include_head_oscillators = include_head_oscillators
        
        # Timestep counter for oscillations
        self.timestep = 0
        
        # Weight sharing and constraint functions
        self.ws = lambda nonshared, shared: shared if use_weight_sharing else nonshared
        
        if use_weight_constraints:
            self.exc = excitatory
            self.inh = inhibitory
            exc_param = excitatory_constant
            inh_param = inhibitory_constant
        else:
            self.exc = unsigned
            self.inh = unsigned
            exc_param = lambda: nn.Parameter(torch.tensor(1.0))
            inh_param = lambda: nn.Parameter(torch.tensor(-1.0))
        
        # Core NCAP parameters (exact notebook architecture)
        self.params = nn.ParameterDict()
        
        if use_weight_sharing:
            # Shared parameters (default NCAP)
            if self.include_proprioception:
                self.params['bneuron_prop'] = exc_param()
            if self.include_head_oscillators:
                self.params['bneuron_osc'] = exc_param()
            self.params['muscle_ipsi'] = exc_param()      # Ipsilateral excitation
            self.params['muscle_contra'] = inh_param()    # Contralateral inhibition
        else:
            # Individual parameters for each joint
            for i in range(self.n_joints):
                if self.include_proprioception and i > 0:
                    self.params[f'bneuron_d_prop_{i}'] = exc_param()
                    self.params[f'bneuron_v_prop_{i}'] = exc_param()
                
                if self.include_head_oscillators and i == 0:
                    self.params[f'bneuron_d_osc_{i}'] = exc_param()
                    self.params[f'bneuron_v_osc_{i}'] = exc_param()
                
                self.params[f'muscle_d_d_{i}'] = exc_param()
                self.params[f'muscle_d_v_{i}'] = inh_param()
                self.params[f'muscle_v_v_{i}'] = exc_param()
                self.params[f'muscle_v_d_{i}'] = inh_param()
        
        print(f"✅ Created SimpleNCAPSwimmer with {len(self.params)} biological parameters")
    
    def reset(self):
        """Reset timestep."""
        self.timestep = 0
    
    def forward(self, joint_pos, timesteps=None, **kwargs):
        """
        Forward pass using exact notebook NCAP architecture.
        
        Args:
            joint_pos: Joint positions in radians
            timesteps: Current timestep (for oscillator)
        
        Returns:
            Joint torques in [-1, 1] range
        """
        # Handle device and input conversion
        if not isinstance(joint_pos, torch.Tensor):
            joint_pos = torch.tensor(joint_pos, dtype=torch.float32)
        
        if timesteps is None:
            timesteps = torch.tensor([self.timestep], dtype=torch.float32)
        elif not isinstance(timesteps, torch.Tensor):
            timesteps = torch.tensor(timesteps, dtype=torch.float32)
        
        # Handle batch dimension
        if joint_pos.dim() == 1:
            joint_pos = joint_pos.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Normalize joint positions to [-1, 1] (exact notebook method)
        joint_limit = 2 * np.pi / (self.n_joints + 1)
        joint_pos_norm = torch.clamp(joint_pos / joint_limit, min=-1, max=1)
        
        # Separate into dorsal and ventral sensor values [0, 1] (KEY BIOLOGICAL CONSTRAINT)
        joint_pos_d = joint_pos_norm.clamp(min=0, max=1)     # Dorsal: positive positions
        joint_pos_v = joint_pos_norm.clamp(min=-1, max=0).neg()  # Ventral: negative positions (flipped)
        
        exc = self.exc
        inh = self.inh
        ws = self.ws
        
        joint_torques = []  # Will store torques for each joint
        
        # Process each joint using exact notebook NCAP architecture
        for i in range(self.n_joints):
            # Initialize B-neurons (biological interneurons)
            bneuron_d = bneuron_v = torch.zeros_like(joint_pos_norm[..., 0, None])  # shape (..., 1)
            
            # 1. PROPRIOCEPTION: B-neurons receive input from previous joint
            if self.include_proprioception and i > 0:
                bneuron_d = bneuron_d + joint_pos_d[..., i-1, None] * exc(
                    self.params[ws(f'bneuron_d_prop_{i}', 'bneuron_prop')]
                )
                bneuron_v = bneuron_v + joint_pos_v[..., i-1, None] * exc(
                    self.params[ws(f'bneuron_v_prop_{i}', 'bneuron_prop')]
                )
            
            # 2. HEAD OSCILLATORS: Drive the first joint (biological CPG)
            if self.include_head_oscillators and i == 0:
                if timesteps is not None:
                    phase = timesteps.round().remainder(self.oscillator_period)
                    mask = phase < self.oscillator_period // 2
                    oscillator_d = torch.zeros_like(timesteps)
                    oscillator_v = torch.zeros_like(timesteps)
                    oscillator_d[mask] = 1.0
                    oscillator_v[~mask] = 1.0
                else:
                    # Use internal timestep counter
                    phase = self.timestep % self.oscillator_period
                    if phase < self.oscillator_period // 2:
                        oscillator_d, oscillator_v = 1.0, 0.0
                    else:
                        oscillator_d, oscillator_v = 0.0, 1.0
                
                bneuron_d = bneuron_d + oscillator_d * exc(
                    self.params[ws(f'bneuron_d_osc_{i}', 'bneuron_osc')]
                )
                bneuron_v = bneuron_v + oscillator_v * exc(
                    self.params[ws(f'bneuron_v_osc_{i}', 'bneuron_osc')]
                )
            
            # 3. B-NEURON ACTIVATION (key biological constraint)
            bneuron_d = graded(bneuron_d)  # Clamp to [0, 1]
            bneuron_v = graded(bneuron_v)  # Clamp to [0, 1]
            
            # 4. MUSCLE ACTIVATION (antagonistic pairs)
            muscle_d = graded(
                bneuron_d * exc(self.params[ws(f'muscle_d_d_{i}', 'muscle_ipsi')]) +
                bneuron_v * inh(self.params[ws(f'muscle_d_v_{i}', 'muscle_contra')])
            )
            muscle_v = graded(
                bneuron_v * exc(self.params[ws(f'muscle_v_v_{i}', 'muscle_ipsi')]) +
                bneuron_d * inh(self.params[ws(f'muscle_v_d_{i}', 'muscle_contra')])
            )
            
            # 5. JOINT TORQUE: Antagonistic muscle contraction (KEY OUTPUT COMPUTATION)
            joint_torque = muscle_d - muscle_v  # This gives range [-1, 1]!
            joint_torques.append(joint_torque)
        
        # Combine all joint torques
        torques = torch.cat(joint_torques, -1)  # shape (..., n_joints)
        
        # Update timestep
        self.timestep += 1
        
        # Handle output dimension
        if squeeze_output:
            torques = torques.squeeze(0)
        
        return torques


class SimpleNCAPActor(nn.Module):
    """Simple actor wrapper for the simplified NCAP swimmer."""
    
    def __init__(self, swimmer_module):
        super().__init__()
        self.swimmer = swimmer_module
        
    def forward(self, observations):
        """Process observations and return actions."""
        # Extract joint positions from observations
        joint_pos = observations[:self.swimmer.n_joints]
        
        # Get actions from NCAP
        actions = self.swimmer(joint_pos)
        
        return actions

    def __call__(self, observations):
        """Make the actor callable."""
        with torch.no_grad():
            actions = self.forward(observations)
            if torch.is_tensor(actions):
                actions = actions.cpu().numpy()
        return actions 