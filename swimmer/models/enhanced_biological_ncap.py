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
        Stateless implementation for batch/DataParallel compatibility.
        """
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor(timestep, dtype=torch.float32, device=self.dorsal_threshold.device)
        
        # Ensure goal_bias and environment_factor are tensors of correct shape
        if not isinstance(goal_bias, torch.Tensor):
            goal_bias = torch.tensor(goal_bias, dtype=torch.float32, device=timestep.device)
        if not isinstance(environment_factor, torch.Tensor):
            environment_factor = torch.tensor(environment_factor, dtype=torch.float32, device=timestep.device)
            
        # Reshape to match timestep batch if necessary (Handling both scalar and batched tensors)
        if goal_bias.dim() > 0: goal_bias = goal_bias.view_as(timestep)
        elif goal_bias.dim() == 0: goal_bias = goal_bias.expand_as(timestep)
        
        if environment_factor.dim() > 0: environment_factor = environment_factor.view_as(timestep)
        elif environment_factor.dim() == 0: environment_factor = environment_factor.expand_as(timestep)

        # Apply environmental frequency scaling
        effective_period = self.base_period / (environment_factor + 1e-8)
        
        # Calculate asymmetric phase durations
        dorsal_duration = effective_period * self.asymmetry_ratio
        ventral_duration = effective_period * (1.0 - self.asymmetry_ratio)
        
        # Determine current phase position
        cycle_position = timestep % effective_period
        
        # Determine current phase masks
        dorsal_mask = cycle_position < dorsal_duration
        ventral_mask = ~dorsal_mask
        
        # --- Dorsal Phase Dynamics ---
        # Gradual rise for dorsal
        phase_progress_d = cycle_position / (dorsal_duration + 1e-8)
        target_dorsal = 1.0 + goal_bias.clamp(min=0.0) * 0.1
        activity_d_rise = (phase_progress_d * target_dorsal).clamp(0, 1)
        
        # Rapid fall for ventral (stateless approximation)
        # It fell from 1.0 starting at the beginning of the dorsal phase
        activity_v_fall = (1.0 - self.ventral_fall_rate * cycle_position).clamp(0, 1)
        
        # --- Ventral Phase Dynamics ---
        # Gradual rise for ventral
        cycle_pos_v = cycle_position - dorsal_duration
        phase_progress_v = cycle_pos_v / (ventral_duration + 1e-8)
        target_ventral = 1.0 + (-goal_bias).clamp(min=0.0) * 0.1
        activity_v_rise = (phase_progress_v * target_ventral).clamp(0, 1)
        
        # Rapid fall for dorsal (stateless approximation)
        activity_d_fall = (1.0 - self.dorsal_fall_rate * cycle_pos_v).clamp(0, 1)
        
        # Combine using masks
        dorsal_activity = torch.where(dorsal_mask, activity_d_rise, activity_d_fall)
        ventral_activity = torch.where(ventral_mask, activity_v_rise, activity_v_fall)
        
        # Apply proprioceptive threshold switching effects (simplified stateless version)
        # If one is very high, it slightly boosts the other (coupling)
        d_thresh = self.dorsal_threshold.clamp(0.6, 0.9)
        v_thresh = self.ventral_threshold.clamp(0.6, 0.9)
        
        dorsal_boosted = torch.where(ventral_activity > v_thresh, (dorsal_activity + 0.05).clamp(0, 1), dorsal_activity)
        ventral_boosted = torch.where(dorsal_activity > d_thresh, (ventral_activity + 0.05).clamp(0, 1), ventral_activity)
        
        return dorsal_boosted, ventral_boosted

class EnhancedBiologicalNCAPSwimmer(nn.Module):
    """
    Enhanced Biological NCAP with relaxation oscillators and goal-directed navigation.
    
    Improvements over basic NCAP:
    1. Asymmetric relaxation oscillators (70/30 phase split)
    2. Goal-directed sensory input integration  
    3. Dramatic frequency adaptation (3-5x changes)
    4. Proprioceptive threshold switching
    5. Target-seeking behavior
    6. **NEW**: Locomotion-only mode for interference-free training
    """
    
    def __init__(self, n_joints, oscillator_period=60,
                 use_weight_sharing=True, use_weight_constraints=True,
                 include_proprioception=True, include_head_oscillators=True,
                 include_environment_adaptation=True, include_goal_direction=True,
                 locomotion_only_mode=False,  # **NEW**: Pure locomotion training mode
                 action_scaling_factor=1.5):  # **NEW**: Scale final torques for stronger swimming
        super().__init__()
        self.n_joints = n_joints
        self.base_oscillator_period = oscillator_period
        self.include_proprioception = include_proprioception
        self.include_head_oscillators = include_head_oscillators
        self.include_environment_adaptation = include_environment_adaptation
        self.include_goal_direction = include_goal_direction and not locomotion_only_mode  # **DISABLE** goal direction in locomotion mode
        self.locomotion_only_mode = locomotion_only_mode  # **NEW**: Flag for pure locomotion training
        self.action_scaling_factor = action_scaling_factor  # **NEW**: Scale actions for stronger movement
        
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
        
        # **GOAL-DIRECTED NAVIGATION** (disabled in locomotion_only_mode)
        if self.include_goal_direction:
            # Goal-directed bias parameters (like chemotaxis in C. elegans)
            self.goal_sensitivity = nn.Parameter(torch.tensor(0.3))          # How much goals affect oscillator
            self.goal_persistence = nn.Parameter(torch.tensor(0.1))          # How long goal bias persists
            self.directional_bias = 0.0  # Current goal-directed bias
            
            print(f"✅ Goal-directed navigation with sensory-motor integration")
        elif self.locomotion_only_mode:
            print(f"🏊 LOCOMOTION-ONLY MODE: Goal-directed navigation DISABLED for pure swimming training")
        
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
                    # Inhibitory parameters: **RELAXED** constraints for stronger inhibition
                    param.data = torch.clamp(param.data, -2.0, 0.0)  # **INCREASED** from -1.0 to -2.0
                else:
                    # Excitatory parameters: **RELAXED** constraints for stronger excitation
                    param.data = torch.clamp(param.data, 0.0, 2.0)  # **INCREASED** from 1.0 to 2.0
            
            # Constrain adaptation parameters with **INCREASED** amplitude scaling
            if self.include_environment_adaptation:
                self.water_frequency_scale.data = torch.clamp(self.water_frequency_scale.data, 1.5, 4.0)  # **INCREASED** max from 3.0 to 4.0
                self.land_frequency_scale.data = torch.clamp(self.land_frequency_scale.data, 0.2, 0.8)
                self.water_amplitude_scale.data = torch.clamp(self.water_amplitude_scale.data, 1.0, 2.5)  # **INCREASED** range from 0.8-1.5 to 1.0-2.5
                self.land_amplitude_scale.data = torch.clamp(self.land_amplitude_scale.data, 0.5, 1.5)  # **INCREASED** max from 1.2 to 1.5
            
            if self.include_goal_direction:
                self.goal_sensitivity.data = torch.clamp(self.goal_sensitivity.data, 0.05, 0.2)  # REDUCED range
                self.goal_persistence.data = torch.clamp(self.goal_persistence.data, 0.02, 0.1)  # REDUCED range
                
            # **RELAXED CONSTRAINTS** - Allow stronger parameters for better swimming performance
            for name, p in self.params.items():
                if 'osc' in name:  # Oscillator parameters
                    p.data = torch.clamp(p.data, 0.0, 2.0)  # **INCREASED** from 0.8 to 2.0 for stronger oscillations
                elif 'prop' in name:  # Proprioceptive parameters  
                    p.data = torch.clamp(p.data, 0.0, 1.5)  # **INCREASED** from 0.6 to 1.5 for stronger coupling
    
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
                # Batched environment adaptation
                env_tensor = torch.as_tensor(environment_type, device=joint_pos.device, dtype=torch.float32)
                if env_tensor.dim() == 1:
                    env_tensor = env_tensor.unsqueeze(0).expand(joint_pos.shape[0], -1)
                env_tensor = torch.nan_to_num(env_tensor, nan=0.0, posinf=1.0, neginf=0.0)
                
                water_flag = env_tensor[:, 0]
                land_flag = env_tensor[:, 1]
                viscosity_norm = env_tensor[:, 2] if env_tensor.shape[1] >= 3 else torch.zeros_like(water_flag) + 0.1
                viscosity_norm = torch.nan_to_num(viscosity_norm, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
                
                # Use masks for adaptation scaling
                land_mask = land_flag > 0.5
                frequency_scale = torch.where(land_mask, self.land_frequency_scale, self.water_frequency_scale)
                amplitude_scale = torch.where(land_mask, self.land_amplitude_scale, self.water_amplitude_scale)
                environment_modulation = torch.where(land_mask, -0.1, 0.1)
                
                # Additional viscosity-based scaling
                amplitude_scale = amplitude_scale * (1.0 + 0.5 * viscosity_norm)
                
            except Exception as e:
                print(f"Warning: Enhanced biological adaptation failed: {e}")
                amplitude_scale = torch.ones(joint_pos.shape[0], device=joint_pos.device)
                frequency_scale = torch.ones(joint_pos.shape[0], device=joint_pos.device)
                environment_modulation = torch.zeros(joint_pos.shape[0], device=joint_pos.device)
        else:
            amplitude_scale = torch.ones(joint_pos.shape[0], device=joint_pos.device)
            frequency_scale = torch.ones(joint_pos.shape[0], device=joint_pos.device)
            environment_modulation = torch.zeros(joint_pos.shape[0], device=joint_pos.device)
        
        # **GOAL-DIRECTED NAVIGATION** - Batched implementation
        goal_bias = torch.zeros(joint_pos.shape[0], device=joint_pos.device)
        if target_direction is not None and self.include_goal_direction:
            try:
                target_tensor = torch.as_tensor(target_direction, device=joint_pos.device, dtype=torch.float32)
                if target_tensor.dim() == 1:
                    target_tensor = target_tensor.unsqueeze(0).expand(joint_pos.shape[0], -1)
                target_tensor = torch.nan_to_num(target_tensor, nan=0.0, posinf=0.0, neginf=0.0)
                
                target_x = target_tensor[:, 0]
                
                # Lateral bias from target
                lateral_bias = target_x * self.goal_sensitivity * 0.1
                
                # For batched operations, ensure directional_bias is a tensor
                if not isinstance(self.directional_bias, torch.Tensor):
                    self.directional_bias = torch.zeros_like(lateral_bias)
                
                # Reshape directional_bias if batch size changed
                if self.directional_bias.shape != lateral_bias.shape:
                    self.directional_bias = torch.zeros_like(lateral_bias)
                else:
                    # Break graph history and sanitize persistent state each forward.
                    self.directional_bias = torch.nan_to_num(
                        self.directional_bias.detach(),
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0
                    )
                
                self.directional_bias = (self.directional_bias * (1.0 - self.goal_persistence) + 
                                       lateral_bias * self.goal_persistence)
                goal_bias = self.directional_bias.clamp(-0.1, 0.1)
                
            except Exception as e:
                print(f"Warning: Goal-directed navigation failed: {e}")
                goal_bias = torch.zeros(joint_pos.shape[0], device=joint_pos.device)
        
        # **BIOLOGICAL RELAXATION OSCILLATOR** (key improvement) - Batched implementation
        if self.include_head_oscillators:
            oscillator_d, oscillator_v = self.relaxation_oscillator(
                timesteps,
                goal_bias=goal_bias * 0.5,  # FURTHER REDUCED goal bias effect
                environment_factor=frequency_scale
            )
            # Relaxation oscillator already returns tensors on correct device
        else:
            oscillator_d = oscillator_v = torch.zeros_like(timesteps)
        
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
                    prop_strength_d = prop_strength_d * (1.0 + environment_modulation.unsqueeze(-1))
                    prop_strength_v = prop_strength_v * (1.0 + environment_modulation.unsqueeze(-1))
                
                bneuron_d = bneuron_d + joint_pos_d[..., i-1, None] * prop_strength_d
                bneuron_v = bneuron_v + joint_pos_v[..., i-1, None] * prop_strength_v
            
            # 2. HEAD OSCILLATORS: Drive the first joint with RELAXATION OSCILLATOR + TRAVELING WAVE
            if self.include_head_oscillators and i == 0:
                osc_strength_d = exc(self.params[ws(f'bneuron_d_osc_{i}', 'bneuron_osc')])
                osc_strength_v = exc(self.params[ws(f'bneuron_v_osc_{i}', 'bneuron_osc')])
                
                # **ENHANCED ADAPTATION**: Environment and goal modulation
                if self.include_environment_adaptation:
                    osc_strength_d = osc_strength_d * (1.0 + environment_modulation.unsqueeze(-1))
                    osc_strength_v = osc_strength_v * (1.0 + environment_modulation.unsqueeze(-1))
                
                bneuron_d = bneuron_d + oscillator_d.unsqueeze(-1) * osc_strength_d
                bneuron_v = bneuron_v + oscillator_v.unsqueeze(-1) * osc_strength_v
            
            # **TRAVELING WAVE PATTERN**: Create phase delays for posterior joints (ANTI-TAIL-CHASING)
            elif self.include_head_oscillators and i > 0:
                # Calculate phase delay for traveling wave (key anti-tail-chasing mechanism)
                phase_delay = i * 15  # 15 steps delay between adjacent joints (like original NCAP)
                delayed_timestep = (timesteps - phase_delay).clamp(min=0)
                
                # Generate delayed oscillator pattern for this joint
                # Pass frequency_scale as tensor if it's already one
                delayed_oscillator_d, delayed_oscillator_v = self.relaxation_oscillator(
                    delayed_timestep,
                    goal_bias=0.0,  # No goal bias on posterior joints - PREVENTS TAIL-CHASING
                    environment_factor=frequency_scale
                )
                
                osc_strength_d = exc(self.params[ws(f'bneuron_d_osc_{i}', 'bneuron_osc')])
                osc_strength_v = exc(self.params[ws(f'bneuron_v_osc_{i}', 'bneuron_osc')])
                
                # **ENHANCED ADAPTATION**: Environment modulation only (no goal bias)
                if self.include_environment_adaptation:
                    osc_strength_d = osc_strength_d * (1.0 + environment_modulation.unsqueeze(-1))
                    osc_strength_v = osc_strength_v * (1.0 + environment_modulation.unsqueeze(-1))
                
                # Apply delayed oscillator pattern - creates traveling wave
                bneuron_d = bneuron_d + delayed_oscillator_d.unsqueeze(-1) * osc_strength_d * 0.8  # Slightly reduced strength
                bneuron_v = bneuron_v + delayed_oscillator_v.unsqueeze(-1) * osc_strength_v * 0.8  # Slightly reduced strength
            
            # 3. B-NEURON ACTIVATION (key biological constraint)
            bneuron_d = graded(bneuron_d)  # Clamp to [0, 1]
            bneuron_v = graded(bneuron_v)  # Clamp to [0, 1]
            
            # 4. MUSCLE ACTIVATION (antagonistic pairs) - FIXED: removed problematic goal bias
            muscle_ipsi_strength = exc(self.params[ws(f'muscle_d_d_{i}', 'muscle_ipsi')])
            muscle_contra_strength = inh(self.params[ws(f'muscle_d_v_{i}', 'muscle_contra')])
            
            # **ENHANCED ADAPTATION**: Environment affects muscle activation strength
            if self.include_environment_adaptation:
                muscle_ipsi_strength = muscle_ipsi_strength * (1.0 + environment_modulation.unsqueeze(-1))
                muscle_contra_strength = muscle_contra_strength * (1.0 + environment_modulation.unsqueeze(-1))
            
            muscle_d = graded(
                bneuron_d * muscle_ipsi_strength +
                bneuron_v * muscle_contra_strength
            )
            muscle_v = graded(
                bneuron_v * exc(self.params[ws(f'muscle_v_v_{i}', 'muscle_ipsi')]) * (1.0 + environment_modulation.unsqueeze(-1) if self.include_environment_adaptation else 1.0) +
                bneuron_d * inh(self.params[ws(f'muscle_v_d_{i}', 'muscle_contra')]) * (1.0 + environment_modulation.unsqueeze(-1) if self.include_environment_adaptation else 1.0)
            )
            
            # 5. JOINT TORQUE: Antagonistic muscle contraction (KEY OUTPUT COMPUTATION)
            joint_torque = muscle_d - muscle_v  # This gives range [-1, 1]!
            joint_torques.append(joint_torque)
        
        # Combine all joint torques
        base_torques = torch.cat(joint_torques, -1)
        
        # **ENHANCED BIOLOGICAL AMPLITUDE SCALING**
        final_torques = base_torques * amplitude_scale.unsqueeze(-1)
        final_torques = torch.nan_to_num(final_torques, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 6. FINAL BOUNDS (ensure biological range is maintained)
        final_torques = torch.clamp(final_torques, -1.0, 1.0)
        
        # **NEW**: Apply action scaling for stronger swimming after normalization
        final_torques = final_torques * self.action_scaling_factor
        
        # Add small exploration noise during training
        if self.training:
            final_torques = final_torques + 0.02 * torch.randn_like(final_torques)  # REDUCED noise
            final_torques = torch.nan_to_num(final_torques, nan=0.0, posinf=0.0, neginf=0.0)
        
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
