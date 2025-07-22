#!/usr/bin/env python3
"""
Improved NCAP Swimmer Implementation
Based on proper notebook architecture with biological constraints.
"""

import torch
import torch.nn as nn
import numpy as np
import collections

# ==================================================================================================
# Weight constraints from notebook

def excitatory(w, upper=None):
    return w.clamp(min=0, max=upper)

def inhibitory(w, lower=None):
    return w.clamp(min=lower, max=0)

def unsigned(w, lower=None, upper=None):
    return w if lower is None and upper is None else w.clamp(min=lower, max=upper)

# ==================================================================================================
# Activation constraints

def graded(x):
    """Graded activation function - key biological constraint!"""
    return x.clamp(min=0, max=1)

# ==================================================================================================
# Weight initialization

def excitatory_uniform(shape=(1,), lower=0., upper=1.):
    assert lower >= 0
    return nn.init.uniform_(nn.Parameter(torch.empty(shape)), a=lower, b=upper)

def inhibitory_uniform(shape=(1,), lower=-1., upper=0.):
    assert upper <= 0
    return nn.init.uniform_(nn.Parameter(torch.empty(shape)), a=lower, b=upper)

def excitatory_constant(shape=(1,), value=1.):
    return nn.Parameter(torch.full(shape, value))

def inhibitory_constant(shape=(1,), value=-1.):
    return nn.Parameter(torch.full(shape, value))


class NCAPSwimmer(nn.Module):
    """
    Proper NCAP Swimmer Implementation with Biological Constraints
    Based on the notebook's SwimmerModule architecture with mixed environment adaptations.
    """
    
    def __init__(self, n_joints, oscillator_period=60, memory_size=10,
                 use_weight_sharing=True, use_weight_constraints=True,
                 include_proprioception=True, include_head_oscillators=True,
                 include_environment_adaptation=True):
        super().__init__()
        self.n_joints = n_joints
        self.oscillator_period = oscillator_period
        self.memory_size = memory_size
        self.include_proprioception = include_proprioception
        self.include_head_oscillators = include_head_oscillators
        self.include_environment_adaptation = include_environment_adaptation
        
        # Device setup
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"NCAP Swimmer using device: {self._device}")
        
        # Timestep counter for oscillations
        self.timestep = 0
        
        # Memory for environment adaptation
        self.env_memory = collections.deque(maxlen=memory_size)
        self.action_memory = collections.deque(maxlen=memory_size)
        
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
        
        # Core NCAP parameters (biological architecture)
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
        
        # Environment adaptation modules (our addition)
        if self.include_environment_adaptation:
            self.env_modulation = nn.Linear(2, n_joints)      # Environment type -> joint modulation
            self.amplitude_modulation = nn.Linear(2, 1)       # Environment type -> amplitude scaling
            self.memory_encoder = nn.LSTM(input_size=2, hidden_size=16, num_layers=1, batch_first=True)
            self.memory_decoder = nn.Linear(16, n_joints)
            
            # Initialize environment adaptation weights
            self._init_environment_weights()
        
        # Move to device
        self.to(self._device)
        if self._device.type == 'cuda':
            print(f"NCAP model on GPU: {next(self.parameters()).device}")
            print(f"GPU memory after NCAP: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    
    def _init_environment_weights(self):
        """Initialize environment adaptation weights."""
        with torch.no_grad():
            # Initialize modulation layers with small weights
            nn.init.uniform_(self.env_modulation.weight, -0.1, 0.1)
            nn.init.constant_(self.env_modulation.bias, 0.0)
            
            nn.init.uniform_(self.amplitude_modulation.weight, -0.1, 0.1)
            nn.init.constant_(self.amplitude_modulation.bias, 0.0)
            
            # Initialize LSTM and decoder
            for name, param in self.memory_encoder.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0.0)
            
            nn.init.xavier_uniform_(self.memory_decoder.weight)
            nn.init.constant_(self.memory_decoder.bias, 0.0)
    
    def reset(self):
        """Reset timestep and memory."""
        self.timestep = 0
        self.env_memory.clear()
        self.action_memory.clear()
    
    def get_memory_context(self):
        """Get memory context for adaptation."""
        if not self.include_environment_adaptation or len(self.env_memory) == 0:
            return torch.zeros(self.n_joints, device=self._device)
        
        try:
            # Convert memory to tensor
            env_history = torch.tensor(list(self.env_memory), dtype=torch.float32, device=self._device).unsqueeze(0)
            
            # Encode memory
            lstm_out, _ = self.memory_encoder(env_history)
            context = self.memory_decoder(lstm_out[:, -1, :])  # Use last output
            
            return context.squeeze(0)  # Remove batch dimension
        except Exception as e:
            # Fallback to zeros if memory processing fails
            return torch.zeros(self.n_joints, device=self._device)
    
    def forward(self, joint_pos, environment_type=None, timesteps=None, **kwargs):
        """
        Forward pass using proper NCAP biological architecture.
        
        Args:
            joint_pos: Joint positions in radians
            environment_type: Environment type [water_weight, land_weight] 
            timesteps: Current timestep (for oscillator)
        
        Returns:
            Joint torques in [-1, 1] range (properly bounded!)
        """
        # ------------------------------------------------------------------
        # Hard biological constraints – keep every weight within its valid
        # range *before* any computations.  This prevents values that the
        # optimiser pushed out-of-range from generating huge activations that
        # later explode to NaN.
        # ------------------------------------------------------------------
        self._constrain_parameters()
        
        # Handle device and input conversion
        if not isinstance(joint_pos, torch.Tensor):
            joint_pos = torch.tensor(joint_pos, dtype=torch.float32, device=self._device)
        elif joint_pos.device != self._device:
            joint_pos = joint_pos.to(self._device)
        
        if timesteps is None:
            timesteps = torch.tensor([self.timestep], dtype=torch.float32, device=self._device)
        elif not isinstance(timesteps, torch.Tensor):
            timesteps = torch.tensor(timesteps, dtype=torch.float32, device=self._device)
        elif timesteps.device != self._device:
            timesteps = timesteps.to(self._device)
        
        # Handle batch dimension
        if joint_pos.dim() == 1:
            joint_pos = joint_pos.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Store environment information for adaptation
        if environment_type is not None and self.include_environment_adaptation:
            self.env_memory.append(environment_type.copy() if hasattr(environment_type, 'copy') else environment_type)
        
        # Normalize joint positions to [-1, 1] (proper NCAP input range)
        joint_limit = 2 * np.pi / (self.n_joints + 1)  # As in notebook
        joint_pos_norm = torch.clamp(joint_pos / joint_limit, min=-1, max=1)
        
        # Separate into dorsal and ventral sensor values [0, 1] (KEY BIOLOGICAL CONSTRAINT)
        joint_pos_d = joint_pos_norm.clamp(min=0, max=1)     # Dorsal: positive positions
        joint_pos_v = joint_pos_norm.clamp(min=-1, max=0).neg()  # Ventral: negative positions (flipped)
        
        exc = self.exc
        inh = self.inh
        ws = self.ws
        
        joint_torques = []  # Will store torques for each joint
        
        # Process each joint using proper NCAP architecture
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
        base_torques = torch.cat(joint_torques, -1)  # shape (..., n_joints)
        
        # 6. ENVIRONMENT ADAPTATION (our addition)
        if environment_type is not None and self.include_environment_adaptation:
            try:
                env_tensor = torch.tensor(environment_type, dtype=torch.float32, device=self._device)
                if env_tensor.dim() == 1:
                    env_tensor = env_tensor.unsqueeze(0)
                
                # Environment modulation (small effect)
                env_modulation = self.env_modulation(env_tensor)
                env_modulation = torch.clamp(env_modulation, -0.3, 0.3)  # Keep small
                
                # Memory context
                memory_context = self.get_memory_context()
                memory_context = torch.clamp(memory_context, -0.2, 0.2)  # Keep small
                
                # Apply adaptations (small influence to preserve biological behavior)
                final_torques = base_torques + 0.1 * env_modulation.squeeze(0) + 0.05 * memory_context
                
            except Exception as e:
                print(f"Warning: Environment adaptation failed: {e}")
                final_torques = base_torques
        else:
            final_torques = base_torques
        
        # 7. FINAL BOUNDS (ensure output is in [-1, 1] as in proper NCAP)
        final_torques = torch.clamp(final_torques, -1.0, 1.0)
        
        # Add small exploration noise during training
        if self.training:
            final_torques = final_torques + 0.05 * torch.randn_like(final_torques)

        # Final safety check
        if torch.isnan(final_torques).any():
            print("WARNING: NaN detected in NCAP output, replacing with zeros")
            final_torques = torch.zeros_like(final_torques)
        
        # Store action in memory for adaptation
        if self.include_environment_adaptation:
            self.action_memory.append(final_torques.detach().cpu().numpy().copy())
        
        # Update timestep
        self.timestep += 1
        
        # Handle output dimension
        if squeeze_output:
            final_torques = final_torques.squeeze(0)
        
        return final_torques

    # ------------------------------------------------------------------
    # Parameter safety clamp
    # ------------------------------------------------------------------
    def _constrain_parameters(self):
        """Clamp NCAP parameters to biologically valid ranges.

        • Excitatory weights  : 0 … 1
        • Inhibitory weights : –1 … 0
        • Adaptation layers  : –0.3 … 0.3 (small influence)
        """
        with torch.no_grad():
            for name, p in self.params.items():
                # Heuristic: muscles and b-neurons obey their designated sign
                if 'muscle' in name or 'bneuron' in name or 'prop' in name or 'osc' in name:
                    if 'contra' in name or 'v_d_' in name or 'muscle_contra' in name or '_v_' in name and 'muscle_v_d_' not in name:
                        # Inhibitory
                        p.clamp_(-1.0, 0.0)
                    else:
                        # Excitatory
                        p.clamp_(0.0, 1.0)

            # Environment modulation and memory adaptor weights should stay small
            small_modules = ['env_modulation', 'amplitude_modulation', 'memory_decoder']
            for mod_name in small_modules:
                if hasattr(self, mod_name):
                    mod = getattr(self, mod_name)
                    if hasattr(mod, 'weight'):
                        mod.weight.data.clamp_(-0.3, 0.3)
                    if hasattr(mod, 'bias') and mod.bias is not None:
                        mod.bias.data.clamp_(-0.3, 0.3)


class NCAPSwimmerActor(nn.Module):
    """Actor wrapper for NCAP swimmer to work with mixed environment."""
    
    def __init__(self, swimmer_module):
        super().__init__()
        self.swimmer = swimmer_module
        # Move actor to same device as swimmer
        self.to(self.swimmer._device)
        
    def forward(self, observations):
        """Process observations and return actions."""
        environment_type = None
        
        # Extract data from observations
        if isinstance(observations, dict):
            joint_pos = observations['joints']
            environment_type = observations.get('environment_type', None)
        else:
            # Assume observations are joint positions
            joint_pos = observations[:self.swimmer.n_joints]
        
        # Ensure joint_pos is on the correct device
        if not isinstance(joint_pos, torch.Tensor):
            joint_pos = torch.tensor(joint_pos, dtype=torch.float32, device=self.swimmer._device)
        else:
            joint_pos = joint_pos.to(self.swimmer._device)
            
        # Get actions from NCAP
        actions = self.swimmer(joint_pos, environment_type=environment_type)
        
        return actions

    def __call__(self, observations):
        """Make the actor callable."""
        with torch.no_grad():
            actions = self.forward(observations)
            if torch.is_tensor(actions):
                actions = actions.cpu().numpy()
        return actions 