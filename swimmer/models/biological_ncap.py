#!/usr/bin/env python3
"""
Biologically Authentic NCAP Swimmer Implementation
Removes LSTM memory system in favor of direct neuromodulation-like adaptation.
More biologically plausible than the complex NCAP version.
"""

import torch
import torch.nn as nn
import numpy as np

# ==================================================================================================
# Weight constraints from notebook

def excitatory(w, upper=None):
    return w.clamp(min=0, max=upper)

def inhibitory(w, lower=None):
    return w.clamp(max=0, min=lower)

def unsigned(w):
    return w

def excitatory_constant():
    return nn.Parameter(torch.tensor(1.0))

def inhibitory_constant():
    return nn.Parameter(torch.tensor(-1.0))

# ==================================================================================================
# Activation constraints

def graded(x):
    """Graded activation function - key biological constraint!"""
    return x.clamp(min=0, max=1)

class BiologicalNCAPSwimmer(nn.Module):
    """
    Biologically Authentic NCAP Swimmer Implementation
    
    Key improvements over complex NCAP:
    - NO LSTM memory system (biologically implausible)
    - Direct parameter modulation (like neuromodulation)
    - Environment-sensitive oscillator periods
    - Viscosity-based amplitude scaling
    - All adaptations through core biological parameters
    """
    
    def __init__(self, n_joints, oscillator_period=60,
                 use_weight_sharing=True, use_weight_constraints=True,
                 include_proprioception=True, include_head_oscillators=True,
                 include_environment_adaptation=True):
        super().__init__()
        self.n_joints = n_joints
        self.base_oscillator_period = oscillator_period
        self.include_proprioception = include_proprioception
        self.include_head_oscillators = include_head_oscillators
        self.include_environment_adaptation = include_environment_adaptation
        
        # Device setup
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Biological NCAP Swimmer using device: {self._device}")
        
        # Timestep counter for oscillations
        self.timestep = 0
        self.current_oscillator_period = oscillator_period
        
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
        
        # **BIOLOGICAL ENVIRONMENT ADAPTATION** (no LSTM!)
        if self.include_environment_adaptation:
            # Direct neuromodulation-like parameters
            self.viscosity_sensitivity = nn.Parameter(torch.tensor(0.5))  # How much viscosity affects amplitude
            self.environment_bias = nn.Parameter(torch.tensor(0.0))       # Land vs water bias
            self.oscillator_sensitivity = nn.Parameter(torch.tensor(0.3)) # How much environment affects oscillator period
            
            # Environment-specific parameter modulation (like neuromodulation)
            self.land_adaptation = nn.Parameter(torch.tensor(0.1))        # Strength of land-specific adaptation
            self.water_adaptation = nn.Parameter(torch.tensor(0.1))       # Strength of water-specific adaptation
            
            print(f"✅ Added biological environment adaptation (no LSTM)")
        
        # Move to device
        self.to(self._device)
        if self._device.type == 'cuda':
            print(f"Biological NCAP model on GPU: {next(self.parameters()).device}")
    
    def reset(self):
        """Reset timestep."""
        self.timestep = 0
        self.current_oscillator_period = self.base_oscillator_period
    
    def forward(self, joint_pos, environment_type=None, timesteps=None, **kwargs):
        """
        Forward pass using biological NCAP architecture with neuromodulation-like adaptation.
        
        Args:
            joint_pos: Joint positions in radians
            environment_type: Environment type [water_weight, land_weight, viscosity_norm]
            timesteps: Current timestep (for oscillator)
        
        Returns:
            Joint torques in [-1, 1] range
        """
        # Constrain parameters to biological ranges
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
        
        # **BIOLOGICAL ENVIRONMENT ADAPTATION** (instead of LSTM)
        amplitude_scale = 1.0
        period_modulation = 1.0
        environment_modulation = 0.0
        
        if environment_type is not None and self.include_environment_adaptation:
            try:
                water_flag, land_flag, viscosity_norm = environment_type
                
                # **Viscosity-based amplitude scaling** (like changing muscle strength)
                # Higher viscosity = need more force, like real swimming/crawling
                amplitude_scale = 1.0 + self.viscosity_sensitivity * viscosity_norm
                amplitude_scale = torch.clamp(amplitude_scale, 0.3, 2.0)  # Biological limits
                
                # **Environment-specific period modulation** (like changing gait frequency)
                # Land = slower, more deliberate movements; Water = faster, fluid movements
                if land_flag > 0.5:  # In land
                    period_modulation = 1.0 + self.oscillator_sensitivity * 0.5  # Slower on land
                    environment_modulation = self.land_adaptation
                else:  # In water
                    period_modulation = 1.0 - self.oscillator_sensitivity * 0.3  # Faster in water  
                    environment_modulation = self.water_adaptation
                
                period_modulation = torch.clamp(period_modulation, 0.5, 2.0)  # Biological limits
                
                # Update oscillator period for more authentic biological adaptation
                self.current_oscillator_period = int(self.base_oscillator_period * period_modulation.item())
                self.current_oscillator_period = max(10, min(self.current_oscillator_period, 120))
                
            except Exception as e:
                print(f"Warning: Biological adaptation failed: {e}")
                amplitude_scale = 1.0
                environment_modulation = 0.0
        
        # Normalize joint positions to [-1, 1] (proper NCAP input range)
        joint_limit = 2 * np.pi / (self.n_joints + 1)  # As in notebook
        joint_pos_norm = torch.clamp(joint_pos / joint_limit, min=-1, max=1)
        
        # Separate into dorsal and ventral sensor values [0, 1] (KEY BIOLOGICAL CONSTRAINT)
        joint_pos_d = joint_pos_norm.clamp(min=0, max=1)     # Dorsal: positive positions
        joint_pos_v = joint_pos_norm.clamp(min=-1, max=0).neg()  # Ventral: negative positions (flipped)
        
        exc = self.exc
        inh = self.inh
        ws = self.ws
        
        joint_torques = []
        
        # Process each joint using proper NCAP architecture
        for i in range(self.n_joints):
            # Initialize B-neurons (biological interneurons)
            bneuron_d = bneuron_v = torch.zeros_like(joint_pos_norm[..., 0, None])  # shape (..., 1)
            
            # 1. PROPRIOCEPTION: B-neurons receive input from previous joint
            if self.include_proprioception and i > 0:
                prop_strength_d = exc(self.params[ws(f'bneuron_d_prop_{i}', 'bneuron_prop')])
                prop_strength_v = exc(self.params[ws(f'bneuron_v_prop_{i}', 'bneuron_prop')])
                
                # **BIOLOGICAL ADAPTATION**: Modulate proprioception by environment
                if self.include_environment_adaptation:
                    prop_strength_d = prop_strength_d * (1.0 + environment_modulation)
                    prop_strength_v = prop_strength_v * (1.0 + environment_modulation)
                
                bneuron_d = bneuron_d + joint_pos_d[..., i-1, None] * prop_strength_d
                bneuron_v = bneuron_v + joint_pos_v[..., i-1, None] * prop_strength_v
            
            # 2. HEAD OSCILLATORS: Drive the first joint (biological CPG)
            if self.include_head_oscillators and i == 0:
                if timesteps is not None:
                    phase = timesteps.round().remainder(self.current_oscillator_period)
                    mask = phase < self.current_oscillator_period // 2
                    oscillator_d = torch.zeros_like(timesteps)
                    oscillator_v = torch.zeros_like(timesteps)
                    oscillator_d[mask] = 1.0
                    oscillator_v[~mask] = 1.0
                else:
                    # Use internal timestep counter with adapted period
                    phase = self.timestep % self.current_oscillator_period
                    if phase < self.current_oscillator_period // 2:
                        oscillator_d, oscillator_v = 1.0, 0.0
                    else:
                        oscillator_d, oscillator_v = 0.0, 1.0
                
                osc_strength_d = exc(self.params[ws(f'bneuron_d_osc_{i}', 'bneuron_osc')])
                osc_strength_v = exc(self.params[ws(f'bneuron_v_osc_{i}', 'bneuron_osc')])
                
                # **BIOLOGICAL ADAPTATION**: Modulate oscillator strength by environment
                if self.include_environment_adaptation:
                    osc_strength_d = osc_strength_d * (1.0 + environment_modulation)
                    osc_strength_v = osc_strength_v * (1.0 + environment_modulation)
                
                bneuron_d = bneuron_d + oscillator_d * osc_strength_d
                bneuron_v = bneuron_v + oscillator_v * osc_strength_v
            
            # 3. B-NEURON ACTIVATION (key biological constraint)
            bneuron_d = graded(bneuron_d)  # Clamp to [0, 1]
            bneuron_v = graded(bneuron_v)  # Clamp to [0, 1]
            
            # 4. MUSCLE ACTIVATION (antagonistic pairs)
            muscle_ipsi_strength = exc(self.params[ws(f'muscle_d_d_{i}', 'muscle_ipsi')])
            muscle_contra_strength = inh(self.params[ws(f'muscle_d_v_{i}', 'muscle_contra')])
            
            # **BIOLOGICAL ADAPTATION**: Environment affects muscle activation strength
            if self.include_environment_adaptation:
                muscle_ipsi_strength = muscle_ipsi_strength * (1.0 + environment_modulation)
                muscle_contra_strength = muscle_contra_strength * (1.0 + environment_modulation)
            
            muscle_d = graded(
                bneuron_d * muscle_ipsi_strength +
                bneuron_v * muscle_contra_strength
            )
            muscle_v = graded(
                bneuron_v * exc(self.params[ws(f'muscle_v_v_{i}', 'muscle_ipsi')]) * (1.0 + environment_modulation if self.include_environment_adaptation else 1.0) +
                bneuron_d * inh(self.params[ws(f'muscle_v_d_{i}', 'muscle_contra')]) * (1.0 + environment_modulation if self.include_environment_adaptation else 1.0)
            )
            
            # 5. JOINT TORQUE: Antagonistic muscle contraction (KEY OUTPUT COMPUTATION)
            joint_torque = muscle_d - muscle_v  # This gives range [-1, 1]!
            joint_torques.append(joint_torque)
        
        # Combine all joint torques
        base_torques = torch.cat(joint_torques, -1)
        
        # **BIOLOGICAL AMPLITUDE SCALING** (instead of LSTM memory)
        final_torques = base_torques * amplitude_scale
        
        # Add environment bias (like neuromodulator effect)
        if self.include_environment_adaptation:
            final_torques = final_torques + self.environment_bias * 0.1  # Small bias effect
        
        # 6. FINAL BOUNDS (ensure output is in [-1, 1] as in proper NCAP)
        final_torques = torch.clamp(final_torques, -1.0, 1.0)
        
        # Add small exploration noise during training
        if self.training:
            final_torques = final_torques + 0.05 * torch.randn_like(final_torques)

        # Final safety check
        if torch.isnan(final_torques).any():
            print("WARNING: NaN detected in Biological NCAP output, replacing with zeros")
            final_torques = torch.zeros_like(final_torques)
        
        # Update timestep
        self.timestep += 1
        
        # Handle output dimension
        if squeeze_output:
            final_torques = final_torques.squeeze(0)
        
        return final_torques

    def _constrain_parameters(self):
        """Clamp NCAP parameters to biologically valid ranges."""
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

            # Constrain biological adaptation parameters to reasonable ranges
            if self.include_environment_adaptation:
                self.viscosity_sensitivity.data.clamp_(0.0, 2.0)      # Positive sensitivity
                self.environment_bias.data.clamp_(-0.5, 0.5)         # Small bias range
                self.oscillator_sensitivity.data.clamp_(0.0, 1.0)    # Positive modulation
                self.land_adaptation.data.clamp_(-0.3, 0.3)          # Small adaptation range
                self.water_adaptation.data.clamp_(-0.3, 0.3)         # Small adaptation range


class BiologicalNCAPActor(nn.Module):
    """Actor wrapper for biological NCAP swimmer to work with mixed environment."""
    
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
            viscosity = observations.get('fluid_viscosity', None)

            if environment_type is not None and viscosity is not None:
                # Build [water, land, viscosity_norm] vector
                water_flag, land_flag = environment_type
                # Normalize viscosity to 0..1 logarithmically between 1e-4 and 1.5
                vis = float(viscosity[0]) if isinstance(viscosity, (list, np.ndarray)) else float(viscosity)
                vis_norm = np.clip((np.log10(vis) - np.log10(1e-4)) / (np.log10(1.5) - np.log10(1e-4)), 0.0, 1.0)
                environment_type = np.array([water_flag, land_flag, vis_norm], dtype=np.float32)
        else:
            # Assume observations are joint positions
            joint_pos = observations[:self.swimmer.n_joints]
        
        # Ensure joint_pos is on the correct device
        if not isinstance(joint_pos, torch.Tensor):
            joint_pos = torch.tensor(joint_pos, dtype=torch.float32, device=self.swimmer._device)
        else:
            joint_pos = joint_pos.to(self.swimmer._device)
            
        # Get actions from Biological NCAP
        actions = self.swimmer(joint_pos, environment_type=environment_type)
        
        return actions

    def __call__(self, observations):
        """Make the actor callable."""
        with torch.no_grad():
            actions = self.forward(observations)
            if torch.is_tensor(actions):
                actions = actions.cpu().numpy()
        return actions 