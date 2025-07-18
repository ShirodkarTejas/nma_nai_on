#!/usr/bin/env python3
"""
Custom Tonic A2C Agent with proper device handling
"""

import torch
from tonic.torch.agents import a2c


class CustomA2C(a2c.A2C):
    """Custom A2C agent that handles device conversion properly."""
    
    def __init__(
        self, model=None, replay=None, actor_updater=None, critic_updater=None
    ):
        # Use a smaller replay buffer for faster training
        if replay is None:
            from tonic import replays
            replay = replays.Segment(size=1024, batch_iterations=20)  # Smaller buffer, fewer iterations
        
        super().__init__(
            model=model, replay=replay, actor_updater=actor_updater,
            critic_updater=critic_updater)
    
    def step(self, observations, steps):
        # Sample actions and get their log-probabilities for training.
        actions, log_probs = self._step(observations)
        # Move to CPU before converting to numpy
        actions = actions.cpu().numpy()
        log_probs = log_probs.cpu().numpy()

        # Keep some values for the next update.
        self.last_observations = observations.copy()
        self.last_actions = actions.copy()
        self.last_log_probs = log_probs.copy()

        return actions
    
    def test_step(self, observations, steps):
        # Sample actions for testing.
        actions = self._test_step(observations)
        # Move to CPU before converting to numpy
        return actions.cpu().numpy()
    
    def update(self, observations=None, rewards=None, resets=None, terminations=None, steps=None, **kwargs):
        # Handle both positional and keyword arguments
        if observations is None and 'observations' in kwargs:
            observations = kwargs['observations']
        if rewards is None and 'rewards' in kwargs:
            rewards = kwargs['rewards']
        if resets is None and 'resets' in kwargs:
            resets = kwargs['resets']
        if terminations is None and 'terminations' in kwargs:
            terminations = kwargs['terminations']
        
        # Store the last transitions in the replay.
        self.replay.store(
            observations=self.last_observations, actions=self.last_actions,
            next_observations=observations, rewards=rewards, resets=resets,
            terminations=terminations, log_probs=self.last_log_probs)

        # Prepare to update the normalizers.
        if self.model.observation_normalizer:
            self.model.observation_normalizer.record(self.last_observations)
        if self.model.return_normalizer:
            self.model.return_normalizer.record(rewards)

        # Update the model if the replay is ready.
        if self.replay.ready():
            self._update()
    
    def _step(self, observations):
        # Convert observations to tensor and move to the same device as the model
        observations = torch.as_tensor(observations, dtype=torch.float32)
        
        # Get the device of the model
        model_device = next(self.model.parameters()).device
        observations = observations.to(model_device)
        
        with torch.no_grad():
            distributions = self.model.actor(observations)
            if hasattr(distributions, 'sample_with_log_prob'):
                actions, log_probs = distributions.sample_with_log_prob()
            else:
                actions = distributions.sample()
                log_probs = distributions.log_prob(actions)
            log_probs = log_probs.sum(dim=-1)
        return actions, log_probs

    def _test_step(self, observations):
        # Convert observations to tensor and move to the same device as the model
        observations = torch.as_tensor(observations, dtype=torch.float32)
        
        # Get the device of the model
        model_device = next(self.model.parameters()).device
        observations = observations.to(model_device)
        
        with torch.no_grad():
            return self.model.actor(observations).sample()

    def _evaluate(self, observations, next_observations):
        # Convert observations to tensor and move to the same device as the model
        observations = torch.as_tensor(observations, dtype=torch.float32)
        next_observations = torch.as_tensor(next_observations, dtype=torch.float32)
        
        # Get the device of the model
        model_device = next(self.model.parameters()).device
        observations = observations.to(model_device)
        next_observations = next_observations.to(model_device)
        
        with torch.no_grad():
            values = self.model.critic(observations)
            next_values = self.model.critic(next_observations)
        
        # Return the values as they are - don't squeeze, let the replay buffer handle the shape
        return values, next_values
    
    def _update(self):
        # Compute the lambda-returns.
        batch = self.replay.get_full('observations', 'next_observations')
        
        values, next_values = self._evaluate(**batch)
        
        # Move to CPU before converting to numpy
        values, next_values = values.cpu().numpy(), next_values.cpu().numpy()
        
        self.replay.compute_returns(values, next_values)

        # Update the actor once.
        keys = 'observations', 'actions', 'advantages', 'log_probs'
        batch = self.replay.get_full(*keys)
        
        # Convert to tensors and move to the correct device
        model_device = next(self.model.parameters()).device
        batch = {k: torch.as_tensor(v, device=model_device) for k, v in batch.items()}
        
        infos = self.actor_updater(**batch)
        for k, v in infos.items():
            from tonic import logger
            logger.store('actor/' + k, v.cpu().numpy())

        # Update the critic multiple times.
        for batch in self.replay.get('observations', 'returns'):
            # Convert to tensors and move to the correct device
            batch = {k: torch.as_tensor(v, device=model_device) for k, v in batch.items()}
            infos = self.critic_updater(**batch)
            for k, v in infos.items():
                from tonic import logger
                logger.store('critic/' + k, v.cpu().numpy())

        # Update the normalizers.
        if self.model.observation_normalizer:
            self.model.observation_normalizer.update()
        if self.model.return_normalizer:
            self.model.return_normalizer.update()


class CustomPPO(CustomA2C):
    """Custom PPO agent that inherits device handling from CustomA2C."""
    
    def __init__(self, model=None, replay=None, actor_updater=None, critic_updater=None):
        from tonic.torch import updaters
        actor_updater = actor_updater or updaters.ClippedRatio()
        super().__init__(
            model=model, replay=replay, actor_updater=actor_updater,
            critic_updater=critic_updater)
    
    def _update(self):
        # Add gradient clipping to prevent NaN values
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Call parent update method
        super()._update() 