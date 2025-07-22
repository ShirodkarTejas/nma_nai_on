#!/usr/bin/env python3
"""
Custom Tonic A2C Agent with proper device handling
"""

import torch
from tonic.torch.agents import a2c
import numpy as np


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
        # ---------------------------------------------------------------------------------
        # 1) PRE-CHECK: Sanitize model weights before computing the update to make sure we
        #    never propagate NaNs forward (they would otherwise corrupt the loss and 
        #    gradients).  This also clamps weights to a reasonable range, mitigating
        #    silent explosions that slip past gradient-clipping.
        # ---------------------------------------------------------------------------------
        self._sanitize_model_parameters()

        # Compute the lambda-returns.
        batch = self.replay.get_full('observations', 'next_observations')
        
        values, next_values = self._evaluate(**batch)

        # Replace any numerical issues that slipped through the sanitiser.
        values = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        next_values = torch.nan_to_num(next_values, nan=0.0, posinf=0.0, neginf=0.0)

        # Move to CPU before converting to numpy (Tonic expects numpy arrays).
        values, next_values = values.cpu().numpy(), next_values.cpu().numpy()
        
        self.replay.compute_returns(values, next_values)

        # Update the actor once.
        keys = 'observations', 'actions', 'advantages', 'log_probs'
        batch = self.replay.get_full(*keys)

        # ------------------------------------------------------------------
        # Sanitize the batch (advantages, log_probs, etc.) to kill NaNs.
        # ------------------------------------------------------------------
        for k, v in batch.items():
            batch[k] = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)

        # Convert to tensors and move to the correct device
        model_device = next(self.model.parameters()).device
        batch = {k: torch.as_tensor(v, device=model_device) for k, v in batch.items()}
        
        infos = self.actor_updater(**batch)

        # Abort and skip critic update if the actor loss came out as NaN
        if any(torch.isnan(v).any().item() for v in infos.values() if torch.is_tensor(v)):
            print("[Update-Skip] Actor updater produced NaN – skipping critic update this iter.")
            return
        for k, v in infos.items():
            from tonic import logger
            logger.store('actor/' + k, v.cpu().numpy())

        # Update the critic multiple times.
        for batch in self.replay.get('observations', 'returns'):
            # Convert to tensors and move to the correct device
            # Sanitize returns before tensor conversion
            batch = {k: np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0) for k, v in batch.items()}
            batch = {k: torch.as_tensor(v, device=model_device) for k, v in batch.items()}
            infos = self.critic_updater(**batch)

            if any(torch.isnan(v).any().item() for v in infos.values() if torch.is_tensor(v)):
                print("[Update-Skip] Critic updater produced NaN – breaking out of critic loop.")
                break
            for k, v in infos.items():
                from tonic import logger
                logger.store('critic/' + k, v.cpu().numpy())

        # Update the normalizers.
        if self.model.observation_normalizer:
            self.model.observation_normalizer.update()
        if self.model.return_normalizer:
            self.model.return_normalizer.update()

        # ---------------------------------------------------------------------------------
        # 2) POST-CHECK: After the optimizer steps we validate the parameters again to
        #    catch any NaNs/Infs that slipped in due to bad gradients or optimizer state.
        # ---------------------------------------------------------------------------------
        self._sanitize_model_parameters()


    # -------------------------------------------------------------------------
    # Helper utilities
    # -------------------------------------------------------------------------
    def _sanitize_model_parameters(self, clip_value: float = 10.0):
        """Detect NaNs/Infs and clamp model parameters to ±clip_value.

        This is a *safety net* against numerical explosions that would otherwise
        propagate NaNs through the NCAP network and into the environment.  It
        should have negligible effect on learning when parameters stay within
        healthy bounds, but it completely stops runaway values.
        """

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    # Zero out bad gradients so they don't pollute the optimizer.
                    param.grad.nan_to_num_(nan=0.0, posinf=clip_value, neginf=-clip_value)

                # Replace NaNs/Infs in the *weights* themselves.
                if torch.isnan(param).any() or torch.isinf(param).any():
                    print(f"[Sanitize] NaN/Inf detected in '{name}', resetting problematic values.")
                    param.nan_to_num_(nan=0.0, posinf=clip_value, neginf=-clip_value)

                # Hard-clip the parameter magnitudes.
                param.clamp_(-clip_value, clip_value)


class CustomPPO(CustomA2C):
    """Custom PPO agent that inherits device handling from CustomA2C."""
    
    def __init__(self, model=None, replay=None, actor_updater=None, critic_updater=None):
        """Create a PPO agent with safe gradient norms.

        If the caller does not provide custom *actor_updater* or *critic_updater*
        we instantiate the defaults with *gradient_clip*=1.0 so that the
        gradients are clipped **after** the backward pass but **before** the
        optimizer step (this is handled internally by the Tonic updaters).
        This prevents exploding gradients that have been causing NaNs during
        mixed-environment training.
        """

        from tonic.torch import updaters

        # Default actor updater with gradient clipping.
        if actor_updater is None:
            actor_updater = updaters.ClippedRatio(gradient_clip=1.0)

        # Default critic updater with gradient clipping.
        if critic_updater is None:
            critic_updater = updaters.VRegression(gradient_clip=1.0)

        super().__init__(
            model=model,
            replay=replay,
            actor_updater=actor_updater,
            critic_updater=critic_updater,
        )
        # No need for an explicit _update override: gradient clipping is now
        # handled inside the updaters themselves.

    # Remove the previous _update implementation that attempted to clip
    # gradients *before* they were computed. The base class implementation is
    # sufficient once the updaters perform clipping internally. 