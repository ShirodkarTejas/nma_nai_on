"""Thin wrappers around the baseline NCAP swimmer with optional priors."""

from __future__ import annotations

from typing import Callable, Iterable, Mapping, Optional, Tuple

import torch
from torch import nn

try:
    from NMAP.swimmer.models.ncap_swimmer import NCAPSwimmer, NCAPSwimmerActor
    from NMAP.swimmer.models.tonic_ncap import create_tonic_ncap_model
except Exception:
    from swimmer.models.ncap_swimmer import NCAPSwimmer, NCAPSwimmerActor  # type: ignore
    from swimmer.models.tonic_ncap import create_tonic_ncap_model  # type: ignore


class SwimmerModuleWithPriors(NCAPSwimmer):
    """Baseline NCAP swimmer extended with optional structural priors."""

    def __init__(
        self,
        *args,
        connectome_priors: Optional[Mapping[str, torch.Tensor]] = None,
        distance_priors: Optional[Mapping[str, torch.Tensor]] = None,
        synapse_prior_lambda: float = 0.0,
        distance_penalty_lambda: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.synapse_prior_lambda = synapse_prior_lambda
        self.distance_penalty_lambda = distance_penalty_lambda
        self.init_priors: Optional[Mapping[str, object]] = None
        self.distance_priors: Optional[Mapping[str, object]] = None
        self._load_priors(connectome_priors, distance_priors)

    def _load_priors(
        self,
        connectome_priors: Optional[Mapping[str, object]],
        distance_priors: Optional[Mapping[str, object]],
    ) -> None:
        device = self._device if hasattr(self, "_device") else next(self.parameters()).device

        def _move(obj):
            if torch.is_tensor(obj):
                return obj.to(device)
            if isinstance(obj, dict):
                return {k: _move(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return type(obj)(_move(v) for v in obj)
            return obj

        self.init_priors = _move(connectome_priors) if connectome_priors else None
        self.distance_priors = _move(distance_priors) if distance_priors else None
        self._apply_initialization_priors()

    def _normalized_mean(self, values: torch.Tensor, mask: torch.Tensor) -> Optional[torch.Tensor]:
        if values is None or mask is None:
            return None
        if not torch.is_tensor(values):
            values = torch.as_tensor(values, dtype=torch.float32, device=self._device)
        if not torch.is_tensor(mask):
            mask = torch.as_tensor(mask, dtype=torch.bool, device=self._device)
        if mask.numel() == 0 or not mask.any():
            return None
        valid = values[mask]
        max_val = valid.max()
        denom = max_val if float(max_val) > 0 else torch.tensor(1.0, device=values.device)
        return (valid / denom).mean()

    def _scale_params(self, keyword: str, factor: float) -> None:
        if factor is None:
            return
        for name, param in self.params.items():
            if keyword in name:
                with torch.no_grad():
                    param.mul_(1.0 + 0.1 * factor)

    def _apply_initialization_priors(self) -> None:
        """Scale initial parameters using masked priors where available."""
        if not self.init_priors:
            return
        prop = self.init_priors.get("prop_to_B") if isinstance(self.init_priors, Mapping) else None
        if prop and self.include_proprioception:
            vals = []
            for n_key, m_key in (("n_syn_d", "mask_d"), ("n_syn_v", "mask_v")):
                norm_val = self._normalized_mean(prop.get(n_key), prop.get(m_key))
                if norm_val is not None:
                    vals.append(norm_val)
            if vals:
                self._scale_params("prop", torch.stack(vals).mean().item())

        bb = self.init_priors.get("B_to_B_next") if isinstance(self.init_priors, Mapping) else None
        if bb:
            vals = []
            for n_key, m_key in (("n_syn_d", "mask_d"), ("n_syn_v", "mask_v")):
                norm_val = self._normalized_mean(bb.get(n_key), bb.get(m_key))
                if norm_val is not None:
                    vals.append(norm_val)
            if vals:
                self._scale_params("osc", torch.stack(vals).mean().item())

    def prior_regularization(self) -> torch.Tensor:
        """Compute structural regularization loss based on registered priors."""
        device = self._device if hasattr(self, "_device") else next(self.parameters()).device
        loss = torch.zeros((), device=device)

        def _dist_sum(dist, mask) -> Optional[torch.Tensor]:
            if dist is None or mask is None:
                return None
            dist_t = torch.as_tensor(dist, dtype=torch.float32, device=device)
            mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device)
            if mask_t.numel() == 0 or not mask_t.any():
                return None
            return dist_t[mask_t].sum()

        def _params_with_keyword(keyword: str):
            return [p for name, p in self.params.items() if keyword in name]

        if self.synapse_prior_lambda > 0 and self.init_priors:
            syn_parts = []
            prop = self.init_priors.get("prop_to_B") if isinstance(self.init_priors, Mapping) else None
            if prop:
                for n_key, m_key in (("n_syn_d", "mask_d"), ("n_syn_v", "mask_v")):
                    norm_val = self._normalized_mean(prop.get(n_key), prop.get(m_key))
                    if norm_val is not None:
                        syn_parts.append(norm_val)
            bb = self.init_priors.get("B_to_B_next") if isinstance(self.init_priors, Mapping) else None
            if bb:
                for n_key, m_key in (("n_syn_d", "mask_d"), ("n_syn_v", "mask_v")):
                    norm_val = self._normalized_mean(bb.get(n_key), bb.get(m_key))
                    if norm_val is not None:
                        syn_parts.append(norm_val)
            if syn_parts:
                target = torch.stack(syn_parts).mean()
                weight_energy = torch.stack([p.pow(2).mean() for p in self.params.values()]).mean()
                loss = loss + self.synapse_prior_lambda * target * weight_energy

        if self.distance_penalty_lambda > 0 and self.distance_priors:
            prop = self.distance_priors.get("prop_to_B") if isinstance(self.distance_priors, Mapping) else None
            if prop:
                dist_values = []
                for d_key, m_key in (("dist_d", "mask_d"), ("dist_v", "mask_v")):
                    s = _dist_sum(prop.get(d_key), prop.get(m_key))
                    if s is not None:
                        dist_values.append(s)
                if dist_values:
                    dist_total = torch.stack(dist_values).sum()
                    for p in _params_with_keyword("prop"):
                        loss = loss + self.distance_penalty_lambda * dist_total * p.pow(2)

            bb = self.distance_priors.get("B_to_B_next") if isinstance(self.distance_priors, Mapping) else None
            if bb:
                dist_values = []
                for d_key, m_key in (("dist_d", "mask_d"), ("dist_v", "mask_v")):
                    s = _dist_sum(bb.get(d_key), bb.get(m_key))
                    if s is not None:
                        dist_values.append(s)
                if dist_values:
                    dist_total = torch.stack(dist_values).sum()
                    for p in _params_with_keyword("osc"):
                        loss = loss + self.distance_penalty_lambda * dist_total * p.pow(2)

            muscles = self.distance_priors.get("B_to_muscle") if isinstance(self.distance_priors, Mapping) else None
            if muscles:
                for dist_key, mask_key in (("dist_b_to_m_d", "mask_b_to_m_d"), ("dist_b_to_m_v", "mask_b_to_m_v")):
                    s = _dist_sum(muscles.get(dist_key), muscles.get(mask_key))
                    if s is None:
                        continue
                    for p in _params_with_keyword("muscle"):
                        loss = loss + self.distance_penalty_lambda * s * p.pow(2)

        return loss


def ppo_swimmer_model_with_priors(
    n_joints: int = 6,
    synapse_prior_lambda: float = 0.0,
    distance_penalty_lambda: float = 0.0,
    connectome_priors: Optional[Mapping[str, torch.Tensor]] = None,
    distance_priors: Optional[Mapping[str, torch.Tensor]] = None,
    **kwargs,
):
    """Create a tonic-compatible model that swaps in SwimmerModuleWithPriors."""
    model = create_tonic_ncap_model(n_joints=n_joints, **kwargs)
    model.ncap = SwimmerModuleWithPriors(
        n_joints=n_joints,
        synapse_prior_lambda=synapse_prior_lambda,
        distance_penalty_lambda=distance_penalty_lambda,
        connectome_priors=connectome_priors,
        distance_priors=distance_priors,
    )
    model.actor.swimmer = model.ncap
    return model


# Re-exports for convenience
BaseSwimmerModule = NCAPSwimmer
BaseSwimmerActor = NCAPSwimmerActor
SwimmerModule = BaseSwimmerModule  # backward compatibility
SwimmerActor = BaseSwimmerActor

__all__ = [
    "BaseSwimmerModule",
    "BaseSwimmerActor",
    "SwimmerModule",
    "SwimmerActor",
    "SwimmerModuleWithPriors",
    "ppo_swimmer_model_with_priors",
    "create_tonic_ncap_model",
]
