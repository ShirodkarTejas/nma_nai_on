#!/usr/bin/env python3
"""
Enhanced Biological NCAP with Relaxation Oscillators and Goal-Directed Navigation

Based on: "Phase response analyses support a relaxation oscillator model of 
locomotor rhythm generation in Caenorhabditis elegans" (eLife, 2021)
https://elifesciences.org/articles/69905

Key improvements:
1. Asymmetric relaxation oscillator (70/30 phase split)
2. Goal-directed sensory input integration
3. Dramatic frequency adaptation (3-5x changes)
4. Proprioceptive threshold switching
5. Gradual rise/rapid fall dynamics
"""

import torch
import torch.nn as nn
import numpy as np
from .biological_ncap import (excitatory, inhibitory, unsigned, graded, 
                            excitatory_constant, inhibitory_constant)

class RelaxationOscillator(nn.Module):
    """
    Biologically authentic relaxation oscillator based on C. elegans research.
    
    Features:
    - Asymmetric phase durations (60% dorsal, 40% ventral) - REDUCED asymmetry
    - Gradual rise, rapid fall dynamics
    - Proprioceptive threshold switching
    - Goal-directed frequency modulation
    """
    
    def __init__(self, base_period=60, asymmetry_ratio=0.6):  # REDUCED from 0.7 to 0.6
        super().__init__()
        self.base_period = base_period
        self.asymmetry_ratio = asymmetry_ratio  # 0.6 = 60% dorsal, 40% ventral
        
        # Relaxation oscillator state
        self.dorsal_activity = 0.0
        self.ventral_activity = 0.0
        self.phase_accumulator = 0.0
        self.current_phase = 'dorsal'  # 'dorsal' or 'ventral'
        
        # Learnable thresholds for switching (proprioceptive-like)
        self.dorsal_threshold = nn.Parameter(torch.tensor(0.8))
        self.ventral_threshold = nn.Parameter(torch.tensor(0.8))
        
        # Rise and fall rates (asymmetric dynamics) - REDUCED rates for stability
        self.dorsal_rise_rate = nn.Parameter(torch.tensor(0.03))    # Slower gradual rise
        self.dorsal_fall_rate = nn.Parameter(torch.tensor(0.2))     # Slower rapid fall
        self.ventral_rise_rate = nn.Parameter(torch.tensor(0.05))   # Moderate rise
        self.ventral_fall_rate = nn.Parameter(torch.tensor(0.3))    # Rapid fall
        
    def forward(self, timestep, goal_bias=0.0, environment_factor=1.0):
        """
        Generate relaxation oscillator pattern with goal-directed modulation.
        
        Args:
            timestep: Current simulation timestep
            goal_bias: Goal-directed bias [-1, 1] (negative = turn left, positive = turn right)
            environment_factor: Environment frequency scaling [0.2, 3.0]
        
        Returns:
            tuple: (dorsal_activity, ventral_activity) in [0, 1] range
        """
        # Apply environmental frequency scaling (dramatic changes like real C. elegans)
        effective_period = self.base_period / environment_factor
        
        # Calculate asymmetric phase durations
        dorsal_duration = effective_period * self.asymmetry_ratio
        ventral_duration = effective_period * (1.0 - self.asymmetry_ratio)
        
        # Determine current phase position
        cycle_position = (timestep % effective_period)
        
        if cycle_position < dorsal_duration:
            # Dorsal phase (gradual rise)
            phase_progress = cycle_position / dorsal_duration
            
            # Gradual rise with MINIMAL goal-directed bias (FIXED: reduced from strong bias)
            target_dorsal = 1.0 + max(0.0, goal_bias) * 0.1  # REDUCED bias effect to 10%
            self.dorsal_activity = min(1.0, phase_progress * target_dorsal)
            
            # Rapid fall for ventral
            self.ventral_activity = max(0.0, self.ventral_activity - self.ventral_fall_rate)
            
        else:
            # Ventral phase (faster rise)
            phase_progress = (cycle_position - dorsal_duration) / ventral_duration
            
            # Faster rise with MINIMAL goal-directed bias (FIXED: reduced from strong bias)
            target_ventral = 1.0 + max(0.0, -goal_bias) * 0.1  # REDUCED bias effect to 10%
            self.ventral_activity = min(1.0, phase_progress * target_ventral)
            
            # Rapid fall for dorsal
            self.dorsal_activity = max(0.0, self.dorsal_activity - self.dorsal_fall_rate)
        
        # Apply proprioceptive threshold switching (prevents getting stuck) - FIXED: tensor boolean
        dorsal_threshold_val = float(torch.clamp(self.dorsal_threshold, 0.6, 0.9).item())
        ventral_threshold_val = float(torch.clamp(self.ventral_threshold, 0.6, 0.9).item())
        
        if float(self.dorsal_activity) > dorsal_threshold_val:
            self.ventral_activity = min(1.0, self.ventral_activity + 0.05)  # REDUCED from 0.1
        
        if float(self.ventral_activity) > ventral_threshold_val:
            self.dorsal_activity = min(1.0, self.dorsal_activity + 0.05)  # REDUCED from 0.1
        
        # Convert to tensor with proper device handling
        dorsal_tensor = torch.tensor(float(self.dorsal_activity), dtype=torch.float32)
        ventral_tensor = torch.tensor(float(self.ventral_activity), dtype=torch.float32)
        
        return torch.clamp(dorsal_tensor, 0, 1), torch.clamp(ventral_tensor, 0, 1)

class EnhancedBiologicalNCAPSwimmer(nn.Module):
    """
    Enhanced Biological NCAP with relaxation oscillators and goal-directed navigation.
    
    Improvements over basic NCAP:
    1. Asymmetric relaxation oscillators (70/30 phase split)
    2. Goal-directed sensory input integration  
    3. Dramatic frequency adaptation (3-5x changes)
    4. Proprioceptive threshold switching
    5. Target-seeking behavior
    """
    
    def __init__(self, n_joints, oscillator_period=60,
                 use_weight_sharing=True, use_weight_constraints=True,
                 include_proprioception=True, include_head_oscillators=True,
                 include_environment_adaptation=True, include_goal_direction=True):
        super().__init__()
        self.n_joints = n_joints
        self.base_oscillator_period = oscillator_period
        self.include_proprioception = include_proprioception
        self.include_head_oscillators = include_head_oscillators
        self.include_environment_adaptation = include_environment_adaptation
        self.include_goal_direction = include_goal_direction
        
        # Device setup
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Enhanced Biological NCAP Swimmer using device: {self._device}")
        
        # Biological relaxation oscillator
        self.relaxation_oscillator = RelaxationOscillator(oscillator_period)
        
        # Timestep counter
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
        
        # **ENHANCED BIOLOGICAL ADAPTATION** 
        if self.include_environment_adaptation:
            # Dramatic frequency adaptation (3-5x changes like real C. elegans)
            self.water_frequency_scale = nn.Parameter(torch.tensor(2.5))     # 2.5x faster in water
            self.land_frequency_scale = nn.Parameter(torch.tensor(0.5))      # 2x slower on land
            
            # Environment-specific amplitude scaling
            self.water_amplitude_scale = nn.Parameter(torch.tensor(1.2))     # Higher amplitude in water
            self.land_amplitude_scale = nn.Parameter(torch.tensor(0.8))      # Lower amplitude on land
            
            print(f"✅ Enhanced biological adaptation with dramatic frequency changes")
        
        # **GOAL-DIRECTED NAVIGATION** (new feature)
        if self.include_goal_direction:
            # Goal-directed bias parameters (like chemotaxis in C. elegans)
            self.goal_sensitivity = nn.Parameter(torch.tensor(0.3))          # How much goals affect oscillator
            self.goal_persistence = nn.Parameter(torch.tensor(0.1))          # How long goal bias persists
            self.directional_bias = 0.0  # Current goal-directed bias
            
            print(f"✅ Goal-directed navigation with sensory-motor integration")
        
        # Move to device
        self.to(self._device)
        if self._device.type == 'cuda':
            print(f"Enhanced Biological NCAP model on GPU: {next(self.parameters()).device}")
    
    def reset(self):
        """Reset timestep and oscillator state."""
        self.timestep = 0
        self.current_oscillator_period = self.base_oscillator_period
        self.directional_bias = 0.0
        
        # Reset relaxation oscillator
        self.relaxation_oscillator.dorsal_activity = 0.0
        self.relaxation_oscillator.ventral_activity = 0.0
    
    def _constrain_parameters(self):
        """Enforce biological constraints on parameters."""
        with torch.no_grad():
            for name, param in self.params.items():
                if 'muscle_contra' in name or any(x in name for x in ['d_v', 'v_d']):
                    # Inhibitory parameters: constrain to [-1, 0]
                    param.data = torch.clamp(param.data, -1.0, 0.0)
                else:
                    # Excitatory parameters: constrain to [0, 1]
                    param.data = torch.clamp(param.data, 0.0, 1.0)
            
            # Constrain adaptation parameters
            if self.include_environment_adaptation:
                self.water_frequency_scale.data = torch.clamp(self.water_frequency_scale.data, 1.5, 3.0)
                self.land_frequency_scale.data = torch.clamp(self.land_frequency_scale.data, 0.2, 0.8)
                self.water_amplitude_scale.data = torch.clamp(self.water_amplitude_scale.data, 0.8, 1.5)
                self.land_amplitude_scale.data = torch.clamp(self.land_amplitude_scale.data, 0.5, 1.2)
            
            if self.include_goal_direction:
                self.goal_sensitivity.data = torch.clamp(self.goal_sensitivity.data, 0.05, 0.2)  # REDUCED range
                self.goal_persistence.data = torch.clamp(self.goal_persistence.data, 0.02, 0.1)  # REDUCED range
                
            # **ADDITIONAL SAFETY CONSTRAINTS** - Prevent tail-chasing parameter drift
            for name, p in self.params.items():
                if 'osc' in name:  # Oscillator parameters
                    p.data = torch.clamp(p.data, 0.0, 0.8)  # Reduced max oscillator strength
                elif 'prop' in name:  # Proprioceptive parameters  
                    p.data = torch.clamp(p.data, 0.0, 0.6)  # Reduced max proprioceptive strength
    
    def forward(self, joint_pos, environment_type=None, target_direction=None, timesteps=None, **kwargs):
        """
        Forward pass with enhanced biological oscillator and goal-directed navigation.
        
        Args:
            joint_pos: Joint positions in radians
            environment_type: Environment type [water_weight, land_weight, viscosity_norm]
            target_direction: Target direction vector [x, y] for goal-directed movement
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
        
        # **ENHANCED ENVIRONMENT ADAPTATION** (dramatic frequency changes)
        amplitude_scale = 1.0
        frequency_scale = 1.0
        environment_modulation = 0.0
        
        if environment_type is not None and self.include_environment_adaptation:
            try:
                # Robust unpacking - handle different environment_type formats (FIXED: handle tensors)
                if isinstance(environment_type, torch.Tensor):
                    # Convert tensor to numpy array first, then to list for safe unpacking
                    env_array = environment_type.detach().cpu().numpy()
                    if env_array.ndim > 1:
                        env_array = env_array.flatten()  # Flatten if multidimensional
                    env_values = env_array.tolist()
                    
                    if len(env_values) >= 3:
                        water_flag, land_flag, viscosity_norm = float(env_values[0]), float(env_values[1]), float(env_values[2])
                    elif len(env_values) == 2:
                        water_flag, land_flag = float(env_values[0]), float(env_values[1])
                        viscosity_norm = 0.1  # Default viscosity
                    else:
                        water_flag = float(env_values[0]) if env_values else 1.0
                        land_flag = 1.0 - water_flag
                        viscosity_norm = 0.1  # Default viscosity
                        
                elif hasattr(environment_type, '__len__') and len(environment_type) >= 3:
                    water_flag, land_flag, viscosity_norm = float(environment_type[0]), float(environment_type[1]), float(environment_type[2])
                elif hasattr(environment_type, '__len__') and len(environment_type) == 2:
                    water_flag, land_flag = float(environment_type[0]), float(environment_type[1])
                    viscosity_norm = 0.1  # Default viscosity
                elif hasattr(environment_type, '__len__') and len(environment_type) == 1:
                    # Single value - assume it's water_flag
                    water_flag = float(environment_type[0])
                    land_flag = 1.0 - water_flag
                    viscosity_norm = 0.1  # Default viscosity
                else:
                    # Scalar value
                    water_flag = float(environment_type)
                    land_flag = 1.0 - water_flag
                    viscosity_norm = 0.1  # Default viscosity
                
                # FIXED: Handle tensor boolean properly
                if isinstance(land_flag, torch.Tensor):
                    land_flag_value = float(land_flag.item())
                elif isinstance(land_flag, (list, tuple)):
                    land_flag_value = float(land_flag[0])
                else:
                    land_flag_value = float(land_flag)
                
                if land_flag_value > 0.5:  # In land environment
                    frequency_scale = self.land_frequency_scale.item()    # Much slower (0.5x)
                    amplitude_scale = self.land_amplitude_scale.item()    # Reduced amplitude
                    environment_modulation = -0.1  # Inhibitory modulation
                else:  # In water environment
                    frequency_scale = self.water_frequency_scale.item()   # Much faster (2.5x)
                    amplitude_scale = self.water_amplitude_scale.item()   # Increased amplitude  
                    environment_modulation = 0.1   # Excitatory modulation
                
                # Additional viscosity-based scaling
                amplitude_scale *= (1.0 + 0.5 * viscosity_norm)  # More force in thick fluid
                
            except Exception as e:
                print(f"Warning: Enhanced biological adaptation failed: {e}")
                amplitude_scale = 1.0
                frequency_scale = 1.0
                environment_modulation = 0.0
        
        # **GOAL-DIRECTED NAVIGATION** (new sensory-motor integration) - FIXED: reduced intensity
        goal_bias = 0.0
        if target_direction is not None and self.include_goal_direction:
            try:
                # Convert target direction to goal bias for oscillator - FIXED: handle tensors
                if isinstance(target_direction, torch.Tensor):
                    target_values = target_direction.detach().cpu().numpy().tolist()
                    if len(target_values) >= 2:
                        target_x, target_y = target_values[:2]
                    else:
                        target_x = target_values[0] if target_values else 0.0
                        target_y = 0.0
                else:
                    target_x, target_y = target_direction[:2]
                
                # FIXED: Much smaller lateral bias to prevent tail-chasing
                lateral_bias = target_x * self.goal_sensitivity.item() * 0.1  # REDUCED by 10x
                
                # Update directional bias with persistence (like working memory)
                self.directional_bias = (self.directional_bias * (1.0 - self.goal_persistence.item()) + 
                                       lateral_bias * self.goal_persistence.item())
                # FIXED: Proper tensor construction - avoid torch.tensor() warning
                if isinstance(self.directional_bias, torch.Tensor):
                    goal_bias = torch.clamp(self.directional_bias.clone().detach(), -0.1, 0.1).item()
                else:
                    goal_bias = max(-0.1, min(0.1, float(self.directional_bias)))
                
            except Exception as e:
                goal_bias = 0.0
        
        # **BIOLOGICAL RELAXATION OSCILLATOR** (key improvement) - FIXED: reduced goal bias
        if self.include_head_oscillators:
            oscillator_d, oscillator_v = self.relaxation_oscillator(
                timesteps.item() if timesteps.numel() == 1 else timesteps[0].item(),
                goal_bias=goal_bias * 0.5,  # FURTHER REDUCED goal bias effect
                environment_factor=frequency_scale
            )
            oscillator_d = oscillator_d.to(self._device)
            oscillator_v = oscillator_v.to(self._device)
        else:
            oscillator_d = oscillator_v = torch.tensor(0.0, device=self._device)
        
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
        
        # Process each joint using enhanced NCAP architecture
        for i in range(self.n_joints):
            # Initialize B-neurons (biological interneurons)
            bneuron_d = bneuron_v = torch.zeros_like(joint_pos_norm[..., 0, None])  # shape (..., 1)
            
            # 1. PROPRIOCEPTION: B-neurons receive input from previous joint
            if self.include_proprioception and i > 0:
                prop_strength_d = exc(self.params[ws(f'bneuron_d_prop_{i}', 'bneuron_prop')])
                prop_strength_v = exc(self.params[ws(f'bneuron_v_prop_{i}', 'bneuron_prop')])
                
                # **ENHANCED ADAPTATION**: Modulate proprioception by environment and goals
                if self.include_environment_adaptation:
                    prop_strength_d = prop_strength_d * (1.0 + environment_modulation)
                    prop_strength_v = prop_strength_v * (1.0 + environment_modulation)
                
                bneuron_d = bneuron_d + joint_pos_d[..., i-1, None] * prop_strength_d
                bneuron_v = bneuron_v + joint_pos_v[..., i-1, None] * prop_strength_v
            
            # 2. HEAD OSCILLATORS: Drive the first joint with RELAXATION OSCILLATOR + TRAVELING WAVE
            if self.include_head_oscillators and i == 0:
                osc_strength_d = exc(self.params[ws(f'bneuron_d_osc_{i}', 'bneuron_osc')])
                osc_strength_v = exc(self.params[ws(f'bneuron_v_osc_{i}', 'bneuron_osc')])
                
                # **ENHANCED ADAPTATION**: Environment and goal modulation
                if self.include_environment_adaptation:
                    osc_strength_d = osc_strength_d * (1.0 + environment_modulation)
                    osc_strength_v = osc_strength_v * (1.0 + environment_modulation)
                
                bneuron_d = bneuron_d + oscillator_d * osc_strength_d
                bneuron_v = bneuron_v + oscillator_v * osc_strength_v
            
            # **TRAVELING WAVE PATTERN**: Create phase delays for posterior joints (ANTI-TAIL-CHASING)
            elif self.include_head_oscillators and i > 0:
                # Calculate phase delay for traveling wave (key anti-tail-chasing mechanism)
                phase_delay = i * 15  # 15 steps delay between adjacent joints (like original NCAP)
                delayed_timestep = max(0, (timesteps.item() if timesteps.numel() == 1 else timesteps[0].item()) - phase_delay)
                
                # Generate delayed oscillator pattern for this joint
                delayed_oscillator_d, delayed_oscillator_v = self.relaxation_oscillator(
                    delayed_timestep,
                    goal_bias=0.0,  # No goal bias on posterior joints - PREVENTS TAIL-CHASING
                    environment_factor=frequency_scale
                )
                delayed_oscillator_d = delayed_oscillator_d.to(self._device)
                delayed_oscillator_v = delayed_oscillator_v.to(self._device)
                
                osc_strength_d = exc(self.params[ws(f'bneuron_d_osc_{i}', 'bneuron_osc')])
                osc_strength_v = exc(self.params[ws(f'bneuron_v_osc_{i}', 'bneuron_osc')])
                
                # **ENHANCED ADAPTATION**: Environment modulation only (no goal bias)
                if self.include_environment_adaptation:
                    osc_strength_d = osc_strength_d * (1.0 + environment_modulation)
                    osc_strength_v = osc_strength_v * (1.0 + environment_modulation)
                
                # Apply delayed oscillator pattern - creates traveling wave
                bneuron_d = bneuron_d + delayed_oscillator_d * osc_strength_d * 0.8  # Slightly reduced strength
                bneuron_v = bneuron_v + delayed_oscillator_v * osc_strength_v * 0.8  # Slightly reduced strength
            
            # 3. B-NEURON ACTIVATION (key biological constraint)
            bneuron_d = graded(bneuron_d)  # Clamp to [0, 1]
            bneuron_v = graded(bneuron_v)  # Clamp to [0, 1]
            
            # 4. MUSCLE ACTIVATION (antagonistic pairs) - FIXED: removed problematic goal bias
            muscle_ipsi_strength = exc(self.params[ws(f'muscle_d_d_{i}', 'muscle_ipsi')])
            muscle_contra_strength = inh(self.params[ws(f'muscle_d_v_{i}', 'muscle_contra')])
            
            # **ENHANCED ADAPTATION**: Environment affects muscle activation strength
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
        
        # **ENHANCED BIOLOGICAL AMPLITUDE SCALING**
        final_torques = base_torques * amplitude_scale
        
        # 6. FINAL BOUNDS (ensure output is in [-1, 1] as in proper NCAP)
        final_torques = torch.clamp(final_torques, -1.0, 1.0)
        
        # Add small exploration noise during training
        if self.training:
            final_torques = final_torques + 0.02 * torch.randn_like(final_torques)  # REDUCED noise
        
        # **SAFETY CHECKS** (like original biological NCAP)
        if torch.isnan(final_torques).any():
            print("WARNING: NaN detected in Enhanced Biological NCAP output, replacing with zeros")
            final_torques = torch.zeros_like(final_torques)
        
        # **PREVENT EXCESSIVE TAIL MOVEMENT** - Limit posterior joint magnitudes
        for i in range(self.n_joints):
            if i > 1:  # Posterior joints
                final_torques[..., i] = final_torques[..., i] * 0.8  # REDUCED posterior joint strength
        
        # Increment timestep
        self.timestep += 1
        
        # Remove batch dimension if added
        if squeeze_output:
            final_torques = final_torques.squeeze(0)
        
        return final_torques

class EnhancedBiologicalNCAPActor(nn.Module):
    """Actor wrapper for enhanced biological NCAP with goal-directed navigation."""
    
    def __init__(self, swimmer_module):
        super().__init__()
        self.swimmer = swimmer_module
        # Move actor to same device as swimmer
        self.to(self.swimmer._device)
        
    def forward(self, observations):
        """Process observations and return actions with goal-directed behavior."""
        environment_type = None
        target_direction = None
        
        # Extract data from observations
        if isinstance(observations, dict):
            joint_pos = observations['joints']
            environment_type = observations.get('environment_type', None)
            viscosity = observations.get('fluid_viscosity', None)
            
            # **NEW**: Extract target information for goal-directed navigation
            if 'target_direction' in observations:
                target_direction = observations['target_direction']
            elif 'target_position' in observations:
                # Calculate direction from current position if available
                try:
                    if 'body_velocities' in observations:
                        # Estimate current position from body velocities (rough approximation)
                        target_pos = observations['target_position']
                        target_direction = target_pos / (np.linalg.norm(target_pos) + 1e-6)
                    else:
                        target_direction = observations['target_position']
                except:
                    target_direction = None

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
            
        # Get actions from Enhanced Biological NCAP with goal-directed behavior
        actions = self.swimmer(
            joint_pos, 
            environment_type=environment_type,
            target_direction=target_direction
        )
        
        return actions 