#!/usr/bin/env python3
"""
Custom Tonic A2C Agent with proper device handling
"""

import numpy as np
import torch
from tonic.torch.agents import a2c, ppo

# ---------------------------------------------------------------
# Utility: fast conversion from nested lists to Torch tensor.
# ---------------------------------------------------------------

def _to_tensor(arr, device, dtype=torch.float32):
    """Convert a possibly list-of-arrays observation batch to torch tensor fast."""
    if isinstance(arr, list):
        arr = np.asarray(arr, dtype=np.float32)
    return torch.as_tensor(arr, dtype=dtype, device=device)


class CustomA2C(a2c.A2C):
    """Custom A2C agent that handles device conversion properly."""
    
    def __init__(
        self,
        model=None,
        replay=None,
        actor_updater=None,
        critic_updater=None,
        prior_reg_lambda: float = 0.0,
        prior_reg_gradient_clip: float = 1.0,
        force_oscillation: bool = False,
        force_oscillation_min_variance: float = 0.1,
        force_oscillation_penalty_scale: float = 1.0,
    ):
        self.prior_reg_lambda = float(prior_reg_lambda)
        self.prior_reg_gradient_clip = float(prior_reg_gradient_clip)
        self.force_oscillation = bool(force_oscillation)
        self.force_oscillation_min_variance = float(force_oscillation_min_variance)
        self.force_oscillation_penalty_scale = float(force_oscillation_penalty_scale)

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
        model_device = next(self.model.parameters()).device
        observations = _to_tensor(observations, model_device)
        
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
        model_device = next(self.model.parameters()).device
        observations = _to_tensor(observations, model_device)
        
        with torch.no_grad():
            return self.model.actor(observations).sample()

    def _evaluate(self, observations, next_observations):
        # Convert observations to tensor and move to the same device as the model
        model_device = next(self.model.parameters()).device
        observations = _to_tensor(observations, model_device)
        next_observations = _to_tensor(next_observations, model_device)
        
        with torch.no_grad():
            # Keep value tensors strictly 1D [batch] for return computation.
            values = self.model.critic(observations).view(-1)
            next_values = self.model.critic(next_observations).view(-1)
        
        return values, next_values

    def _safe_critic_update(self, observations: torch.Tensor, returns: torch.Tensor):
        """Run critic update with explicit shape-safe MSE to prevent broadcasting bugs."""
        # Returns from replay can be [B] while critics may output [B,1]; flatten both.
        targets = returns.view(-1)
        values = self.model.critic(observations).view(-1)

        # Defensive alignment for any unexpected shape divergence.
        if values.numel() != targets.numel():
            n = min(values.numel(), targets.numel())
            values = values[:n]
            targets = targets[:n]

        # If the updater exposes optimizer/model (VRegression), use explicit MSE update.
        if (
            hasattr(self.critic_updater, "optimizer")
            and hasattr(self.critic_updater, "model")
            and isinstance(getattr(self.critic_updater, "optimizer"), torch.optim.Optimizer)
        ):
            self.critic_updater.optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(values, targets)
            loss.backward()
            gradient_clip = float(getattr(self.critic_updater, "gradient_clip", 0.0))
            if gradient_clip > 0:
                critic_params = [p for p in self.model.critic.parameters() if p.requires_grad]
                torch.nn.utils.clip_grad_norm_(critic_params, gradient_clip)
            self.critic_updater.optimizer.step()
            return {"loss": loss.detach(), "v": values.detach()}

        # Fallback to updater API with already flattened targets.
        return self.critic_updater(observations=observations, returns=targets)

    def _get_actor_optimizer(self):
        for attr in ("optimizer", "_optimizer", "opt"):
            candidate = getattr(self.actor_updater, attr, None)
            if isinstance(candidate, torch.optim.Optimizer):
                return candidate
        return None

    def _compute_topological_prior_loss(self, model_device):
        if self.prior_reg_lambda <= 0.0:
            return torch.zeros((), device=model_device, dtype=torch.float32)

        if not hasattr(self.model, "compute_topological_prior_loss"):
            return torch.zeros((), device=model_device, dtype=torch.float32)

        prior_loss = self.model.compute_topological_prior_loss(self.prior_reg_lambda)
        if not torch.is_tensor(prior_loss):
            prior_loss = torch.as_tensor(prior_loss, device=model_device, dtype=torch.float32)
        return torch.nan_to_num(prior_loss, nan=0.0, posinf=0.0, neginf=0.0)

    def _compute_force_oscillation_penalty(self, observations: torch.Tensor, model_device):
        """Penalize low action variance to avoid policy collapse to near-zero actions."""
        zero = torch.zeros((), device=model_device, dtype=torch.float32)
        if not self.force_oscillation:
            return zero, zero
        if observations is None or not torch.is_tensor(observations):
            return zero, zero

        distributions = self.model.actor(observations)
        if hasattr(distributions, "mean"):
            actions = distributions.mean
        elif hasattr(distributions, "loc"):
            actions = distributions.loc
        else:
            actions = distributions.sample()

        if actions.dim() == 1:
            actions = actions.unsqueeze(0)

        action_variance = actions.var(dim=0, unbiased=False).mean()
        min_variance = torch.as_tensor(self.force_oscillation_min_variance, device=model_device, dtype=action_variance.dtype)
        variance_gap = torch.relu(min_variance - action_variance)
        penalty = self.force_oscillation_penalty_scale * variance_gap
        return penalty, action_variance

    def _apply_actor_auxiliary_step(self, auxiliary_loss: torch.Tensor):
        if auxiliary_loss is None or not torch.is_tensor(auxiliary_loss):
            return False
        if not auxiliary_loss.requires_grad:
            return False
        if float(torch.abs(auxiliary_loss.detach()).item()) <= 0.0:
            return False

        actor_optimizer = self._get_actor_optimizer()
        if actor_optimizer is None:
            return False

        actor_optimizer.zero_grad(set_to_none=True)
        auxiliary_loss.backward()
        if self.prior_reg_gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.actor.parameters(), self.prior_reg_gradient_clip)
        actor_optimizer.step()
        return True
    
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
        prior_loss = self._compute_topological_prior_loss(model_device)
        force_oscillation_penalty, action_variance = self._compute_force_oscillation_penalty(
            observations=batch.get("observations"),
            model_device=model_device,
        )
        auxiliary_actor_loss = prior_loss + force_oscillation_penalty
        auxiliary_step_applied = self._apply_actor_auxiliary_step(auxiliary_actor_loss)

        infos["prior_loss"] = prior_loss.detach()
        infos["force_oscillation_penalty"] = force_oscillation_penalty.detach()
        infos["action_variance"] = action_variance.detach()
        infos["force_oscillation_enabled"] = torch.as_tensor(float(self.force_oscillation), device=model_device)
        infos["auxiliary_step_applied"] = torch.as_tensor(float(auxiliary_step_applied), device=model_device)
        if "loss" in infos and torch.is_tensor(infos["loss"]):
            infos["loss_with_prior"] = infos["loss"] + prior_loss.detach()
            infos["loss_with_forced_oscillation"] = infos["loss"] + force_oscillation_penalty.detach()
            infos["loss_with_auxiliary"] = infos["loss"] + prior_loss.detach() + force_oscillation_penalty.detach()

        # Abort and skip critic update if the actor loss came out as NaN
        if any(torch.isnan(v).any().item() for v in infos.values() if torch.is_tensor(v)):
            print("[Update-Skip] Actor updater produced NaN – skipping critic update this iter.")
            return
        for k, v in infos.items():
            from tonic import logger
            if torch.is_tensor(v):
                logger.store('actor/' + k, v.detach().cpu().numpy())
            else:
                logger.store('actor/' + k, v)

        # Update the critic multiple times.
        for batch in self.replay.get('observations', 'returns'):
            # Convert to tensors and move to the correct device
            # Sanitize returns before tensor conversion
            batch = {k: np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0) for k, v in batch.items()}
            batch = {k: torch.as_tensor(v, device=model_device) for k, v in batch.items()}
            batch['returns'] = batch['returns'].view(-1)
            infos = self._safe_critic_update(
                observations=batch['observations'],
                returns=batch['returns'],
            )

            if any(torch.isnan(v).any().item() for v in infos.values() if torch.is_tensor(v)):
                print("[Update-Skip] Critic updater produced NaN – breaking out of critic loop.")
                break
            for k, v in infos.items():
                from tonic import logger
                if torch.is_tensor(v):
                    logger.store('critic/' + k, v.detach().cpu().numpy())
                else:
                    logger.store('critic/' + k, v)

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


class CustomPPO(ppo.PPO):
    """Custom PPO agent with the same biological constraints as CustomA2C."""

    # Reuse the device-safe stepping and biological helper logic.
    step = CustomA2C.step
    test_step = CustomA2C.test_step
    update = CustomA2C.update
    _step = CustomA2C._step
    _test_step = CustomA2C._test_step
    _evaluate = CustomA2C._evaluate
    _safe_critic_update = CustomA2C._safe_critic_update
    _get_actor_optimizer = CustomA2C._get_actor_optimizer
    _compute_topological_prior_loss = CustomA2C._compute_topological_prior_loss
    _compute_force_oscillation_penalty = CustomA2C._compute_force_oscillation_penalty
    _apply_actor_auxiliary_step = CustomA2C._apply_actor_auxiliary_step
    _sanitize_model_parameters = CustomA2C._sanitize_model_parameters

    def __init__(
        self,
        model=None,
        replay=None,
        actor_updater=None,
        critic_updater=None,
        prior_reg_lambda: float = 0.0,
        prior_reg_gradient_clip: float = 1.0,
        force_oscillation: bool = False,
        force_oscillation_min_variance: float = 0.1,
        force_oscillation_penalty_scale: float = 1.0,
    ):
        self.prior_reg_lambda = float(prior_reg_lambda)
        self.prior_reg_gradient_clip = float(prior_reg_gradient_clip)
        self.force_oscillation = bool(force_oscillation)
        self.force_oscillation_min_variance = float(force_oscillation_min_variance)
        self.force_oscillation_penalty_scale = float(force_oscillation_penalty_scale)

        if replay is None:
            from tonic import replays
            replay = replays.Segment(size=1024, batch_iterations=20)

        super().__init__(
            model=model,
            replay=replay,
            actor_updater=actor_updater,
            critic_updater=critic_updater,
        )

    def _update_actor_critic(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        advantages: torch.Tensor,
        log_probs: torch.Tensor,
        returns: torch.Tensor,
    ):
        actor_infos = self.actor_updater(observations, actions, advantages, log_probs)
        critic_infos = self._safe_critic_update(
            observations=observations,
            returns=returns.view(-1),
        )
        return dict(actor=actor_infos, critic=critic_infos)

    def _update(self):
        from tonic import logger

        self._sanitize_model_parameters()

        batch = self.replay.get_full('observations', 'next_observations')
        values, next_values = self._evaluate(**batch)
        values = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        next_values = torch.nan_to_num(next_values, nan=0.0, posinf=0.0, neginf=0.0)
        self.replay.compute_returns(values.cpu().numpy(), next_values.cpu().numpy())

        train_actor = True
        actor_iterations = 0
        critic_iterations = 0
        keys = 'observations', 'actions', 'advantages', 'log_probs', 'returns'
        model_device = next(self.model.parameters()).device

        for replay_batch in self.replay.get(*keys):
            replay_batch = {
                k: np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
                for k, v in replay_batch.items()
            }

            if train_actor:
                batch_t = {k: torch.as_tensor(v, device=model_device) for k, v in replay_batch.items()}
                infos = self._update_actor_critic(**batch_t)
                actor_iterations += 1

                prior_loss = self._compute_topological_prior_loss(model_device)
                force_oscillation_penalty, action_variance = self._compute_force_oscillation_penalty(
                    observations=batch_t.get("observations"),
                    model_device=model_device,
                )
                auxiliary_actor_loss = prior_loss + force_oscillation_penalty
                auxiliary_step_applied = self._apply_actor_auxiliary_step(auxiliary_actor_loss)

                actor_infos = infos.setdefault("actor", {})
                actor_infos["prior_loss"] = prior_loss.detach()
                actor_infos["force_oscillation_penalty"] = force_oscillation_penalty.detach()
                actor_infos["action_variance"] = action_variance.detach()
                actor_infos["force_oscillation_enabled"] = torch.as_tensor(
                    float(self.force_oscillation), device=model_device
                )
                actor_infos["auxiliary_step_applied"] = torch.as_tensor(
                    float(auxiliary_step_applied), device=model_device
                )
                if "loss" in actor_infos and torch.is_tensor(actor_infos["loss"]):
                    actor_infos["loss_with_prior"] = actor_infos["loss"] + prior_loss.detach()
                    actor_infos["loss_with_forced_oscillation"] = actor_infos["loss"] + force_oscillation_penalty.detach()
                    actor_infos["loss_with_auxiliary"] = (
                        actor_infos["loss"] + prior_loss.detach() + force_oscillation_penalty.detach()
                    )

                stop_signal = actor_infos.get(
                    "stop",
                    torch.as_tensor(False, device=model_device),
                )
                if torch.is_tensor(stop_signal):
                    train_actor = not bool(stop_signal.detach().cpu().item())
                else:
                    train_actor = not bool(stop_signal)
            else:
                critic_batch = {
                    'observations': torch.as_tensor(replay_batch['observations'], device=model_device),
                    'returns': torch.as_tensor(replay_batch['returns'], device=model_device).view(-1),
                }
                infos = {
                    "critic": self._safe_critic_update(
                        observations=critic_batch['observations'],
                        returns=critic_batch['returns'],
                    )
                }

            critic_iterations += 1

            has_nan = any(
                torch.isnan(v).any().item()
                for group in infos.values()
                if isinstance(group, dict)
                for v in group.values()
                if torch.is_tensor(v)
            )
            if has_nan:
                print("[Update-Skip] PPO updater produced NaN – skipping logger write this iter.")
                continue

            for group_name, group_infos in infos.items():
                if not isinstance(group_infos, dict):
                    continue
                for key, value in group_infos.items():
                    if torch.is_tensor(value):
                        logger.store(f'{group_name}/{key}', value.detach().cpu().numpy())
                    else:
                        logger.store(f'{group_name}/{key}', value)

        logger.store('actor/iterations', actor_iterations)
        logger.store('critic/iterations', critic_iterations)

        if self.model.observation_normalizer:
            self.model.observation_normalizer.update()
        if self.model.return_normalizer:
            self.model.return_normalizer.update()

        self._sanitize_model_parameters()
