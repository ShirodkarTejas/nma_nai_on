#!/usr/bin/env python3
"""
NCAP Swimmer Model Implementation
Contains the NCAP (Neural Central Pattern Generator) model for swimmer control.
"""

import torch
import torch.nn as nn
import numpy as np
import collections
import abc

class BaseSwimmerModel(nn.Module, metaclass=abc.ABCMeta):
    """
    Abstract base class for swimmer models.
    """
    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        pass

class NCAPSwimmer(BaseSwimmerModel):
    def __init__(self, n_joints, oscillator_period=60, memory_size=10):
        super().__init__()
        self.n_joints = n_joints
        self.oscillator_period = oscillator_period
        self.timestep = 0
        self.memory_size = memory_size
        self.env_memory = collections.deque(maxlen=memory_size)
        self.action_memory = collections.deque(maxlen=memory_size)
        
        # Check for GPU
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"NCAP Swimmer using device: {self._device}")
        
        # Restore original oscillator parameters but with smaller amplitude
        self.phase_offsets = nn.Parameter(torch.linspace(oscillator_period, 0, n_joints))
        self.base_amplitude = nn.Parameter(torch.tensor(1.0))  # Reduced from 8.0 to 1.0
        self.base_frequency = nn.Parameter(torch.tensor(2*np.pi/oscillator_period))
        
        # Restore original networks
        self.env_modulation = nn.Linear(2, n_joints)
        self.amplitude_modulation = nn.Linear(2, 1)
        self.memory_encoder = nn.LSTM(input_size=2, hidden_size=16, num_layers=1, batch_first=True)
        self.memory_decoder = nn.Linear(16, n_joints)
        
        # Initialize weights properly to prevent instability
        self._init_weights()
        
        # Move model to device
        self.to(self._device)
        if self._device.type == 'cuda':
            print(f"NCAP model on GPU: {next(self.parameters()).device}")
            print(f"GPU memory after NCAP: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

    def _init_weights(self):
        """Initialize weights with small values to prevent instability."""
        nn.init.xavier_uniform_(self.env_modulation.weight, gain=0.1)
        nn.init.constant_(self.env_modulation.bias, 0.0)
        nn.init.xavier_uniform_(self.amplitude_modulation.weight, gain=0.1)
        nn.init.constant_(self.amplitude_modulation.bias, 0.0)
        nn.init.xavier_uniform_(self.memory_decoder.weight, gain=0.1)
        nn.init.constant_(self.memory_decoder.bias, 0.0)

    @property
    def device(self):
        return self._device

    def update_memory(self, environment_type, action):
        self.env_memory.append(environment_type)
        self.action_memory.append(action)

    def get_memory_context(self):
        if len(self.env_memory) < 2:
            return torch.zeros(self.n_joints, device=self._device)
        env_history = torch.tensor(list(self.env_memory), dtype=torch.float32, device=self._device).unsqueeze(0)
        lstm_out, _ = self.memory_encoder(env_history)
        memory_context = self.memory_decoder(lstm_out[:, -1, :])
        return memory_context.squeeze()

    def generate_square_wave_oscillator(self, timesteps):
        phase = timesteps % self.oscillator_period
        mask = phase < self.oscillator_period // 2
        oscillator = torch.where(mask, torch.ones_like(timesteps), torch.zeros_like(timesteps))
        return oscillator

    def forward(self, joint_pos, environment_type=None, timesteps=None, **kwargs):
        # Ensure inputs are on the correct device
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
        batch_size = joint_pos.shape[0] if joint_pos.dim() > 1 else 1
        if joint_pos.dim() == 1:
            joint_pos = joint_pos.unsqueeze(0)
        
        # Expand timesteps to match batch size
        if timesteps.dim() == 0:
            timesteps = timesteps.unsqueeze(0)
        if timesteps.shape[0] == 1 and batch_size > 1:
            timesteps = timesteps.expand(batch_size)
        
        oscillators = []
        for i in range(self.n_joints):
            offset_timesteps = timesteps + self.phase_offsets[i]
            oscillator = self.generate_square_wave_oscillator(offset_timesteps)
            oscillators.append(oscillator)
        
        oscillators = torch.stack(oscillators, dim=-1)  # Shape: (batch_size, n_joints)
        
        if environment_type is not None:
            if not isinstance(environment_type, torch.Tensor):
                env_tensor = torch.tensor(environment_type, dtype=torch.float32, device=self._device).unsqueeze(0)
            else:
                env_tensor = environment_type.to(self._device).unsqueeze(0)
            joint_modulations = self.env_modulation(env_tensor)
            oscillators = oscillators + joint_modulations.squeeze(0) * 0.5
            amp_mod = self.amplitude_modulation(env_tensor)
            amplitude = self.base_amplitude * (1.0 + amp_mod.squeeze())
            memory_context = self.get_memory_context()
            oscillators = oscillators + memory_context * 0.3
        else:
            amplitude = self.base_amplitude
            
        oscillators = oscillators * amplitude
        self.timestep += 1
        
        # Add bounds to prevent NaN values and ensure stability
        oscillators = torch.clamp(oscillators, -2.0, 2.0)
        
        # Check for NaN values and replace with zeros if found
        if torch.isnan(oscillators).any():
            print("WARNING: NaN values detected in NCAP output, replacing with zeros")
            oscillators = torch.where(torch.isnan(oscillators), torch.zeros_like(oscillators), oscillators)
        
        # Return with batch dimension preserved
        return oscillators

class NCAPSwimmerActor(nn.Module):
    def __init__(self, swimmer_module):
        super().__init__()
        self.swimmer = swimmer_module
        # Move actor to same device as swimmer
        self.to(self.swimmer.device)
        
    def forward(self, observations):
        environment_type = None
        if isinstance(observations, dict) and 'environment_type' in observations:
            environment_type = observations['environment_type']
        if isinstance(observations, dict):
            joint_pos = observations['joints']
        else:
            joint_pos = observations[:self.swimmer.n_joints]
        
        # Ensure joint_pos is on the correct device
        if not isinstance(joint_pos, torch.Tensor):
            joint_pos = torch.tensor(joint_pos, dtype=torch.float32, device=self.swimmer.device).unsqueeze(0)
        else:
            joint_pos = joint_pos.to(self.swimmer.device).unsqueeze(0)
            
        timesteps = torch.tensor([self.swimmer.timestep], dtype=torch.float32, device=self.swimmer.device)
        actions = self.swimmer(joint_pos, environment_type=environment_type, timesteps=timesteps)
        if environment_type is not None:
            self.swimmer.update_memory(environment_type, actions.detach().cpu().numpy())
        return actions.detach().cpu().numpy() 