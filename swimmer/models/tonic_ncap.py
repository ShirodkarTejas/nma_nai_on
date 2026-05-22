#!/usr/bin/env python3
"""
Tonic-compatible NCAP model with sparse biological prior penalties.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .ncap_swimmer import NCAPSwimmer

try:
    from NMAP.connectome_priors.swimmer_priors import generate_ncap_segment_priors, refresh_inventory_files
except Exception:
    generate_ncap_segment_priors = None
    refresh_inventory_files = None


class SwimmerActor(nn.Module):
    """Actor component for Tonic compatibility."""

    def __init__(self, swimmer_module, distribution=None):
        super().__init__()
        self.swimmer = swimmer_module
        self.distribution = distribution or (lambda x: torch.distributions.Normal(x, 0.1))

    def forward(self, observations):
        if observations.dim() == 1:
            observations = observations.unsqueeze(0)
        joint_positions = observations[:, : self.swimmer.n_joints]
        ncap_output = self.swimmer(joint_positions)
        return self.distribution(ncap_output)


class SwimmerCritic(nn.Module):
    """Critic component for Tonic compatibility."""

    def __init__(self, n_joints, critic_sizes=(64, 64), critic_activation=nn.Tanh):
        super().__init__()
        self.n_joints = n_joints

        layers = []
        input_size = n_joints
        for size in critic_sizes:
            layers.extend([nn.Linear(input_size, size), critic_activation()])
            input_size = size
        layers.append(nn.Linear(input_size, 1))
        self.critic_network = nn.Sequential(*layers)

    def forward(self, observations):
        if observations.dim() == 1:
            observations = observations.unsqueeze(0)
        joint_positions = observations[:, : self.n_joints]
        # Ensure value output is 1D [batch] to avoid accidental MSE broadcasting.
        return self.critic_network(joint_positions).squeeze(-1)


class TonicNCAPModel(nn.Module):
    """Tonic-compatible NCAP model with sparse pathway prior penalties."""

    _PATHWAYS = (
        "ipsi_db",
        "ipsi_vb",
        "contra_db",
        "contra_vb",
        "next_db",
        "next_vb",
    )

    def __init__(
        self,
        n_joints,
        oscillator_period=60,
        memory_size=10,
        critic_sizes=(64, 64),
        critic_activation=nn.Tanh,
        action_noise=0.1,
        num_segments: Optional[int] = None,
        prior_modulation_scale: float = 0.15,
    ):
        super().__init__()

        self.n_joints = int(n_joints)
        self.oscillator_period = oscillator_period
        self._requested_num_segments = int(num_segments) if num_segments is not None else None
        self.num_segments = self._requested_num_segments if self._requested_num_segments is not None else self.n_joints

        # Retained only for call-site compatibility.
        self.prior_modulation_scale = float(prior_modulation_scale)

        self.ncap = NCAPSwimmer(
            n_joints=self.n_joints,
            oscillator_period=oscillator_period,
            memory_size=memory_size,
            use_weight_sharing=False,
        )

        self.actor = SwimmerActor(
            swimmer_module=self.ncap,
            distribution=lambda x: torch.distributions.Normal(x, action_noise),
        )
        self.critic = SwimmerCritic(
            n_joints=self.n_joints,
            critic_sizes=critic_sizes,
            critic_activation=critic_activation,
        )

        self.observation_normalizer = None
        self.return_normalizer = None

        self.sparse_prior_scalars: Dict[str, float] = {}
        self.sparse_prior_metadata: Dict[str, object] = {}
        self._load_sparse_priors(self.num_segments)

        self._init_weights()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        print(f"TonicNCAPModel moved to device: {device}")

    def _default_sparse_priors(self) -> Dict[str, float]:
        return {
            "dist_ipsi_db": 1.0,
            "dist_ipsi_vb": 1.0,
            "dist_contra_db": 1.0,
            "dist_contra_vb": 1.0,
            "dist_next_db": 1.0,
            "dist_next_vb": 1.0,
            "syn_ipsi_db": 0.0,
            "syn_ipsi_vb": 0.0,
            "syn_contra_db": 0.0,
            "syn_contra_vb": 0.0,
            "syn_next_db": 0.0,
            "syn_next_vb": 0.0,
        }

    def _load_sparse_priors(self, num_segments: int) -> None:
        if generate_ncap_segment_priors is None:
            self.sparse_prior_scalars = self._default_sparse_priors()
            self.sparse_prior_metadata = {"status": "unavailable", "reason": "connectome_priors import failed"}
            return

        try:
            inventory_refresh = {"status": "skipped"}
            if refresh_inventory_files is not None:
                try:
                    inventory_refresh = refresh_inventory_files()
                    inventory_refresh["status"] = "ok"
                except Exception as refresh_exc:
                    inventory_refresh = {"status": "error", "reason": str(refresh_exc)}

            priors = generate_ncap_segment_priors(num_segments=int(num_segments))
            defaults = self._default_sparse_priors()
            scalars = {}
            for key in defaults:
                value = float(priors.get(key, defaults[key]))
                scalars[key] = value if torch.isfinite(torch.tensor(value)).item() else defaults[key]
            metadata = priors.get("metadata", {}) if isinstance(priors, dict) else {}
            if isinstance(metadata, dict):
                metadata["inventory_refresh"] = inventory_refresh
            self.sparse_prior_scalars = scalars
            self.sparse_prior_metadata = {
                "status": "ok",
                "sources": priors.get("sources", {}),
                "metadata": metadata,
                "counts": {
                    "ipsi_db": priors.get("count_ipsi_db", 0),
                    "ipsi_vb": priors.get("count_ipsi_vb", 0),
                    "contra_db": priors.get("count_contra_db", 0),
                    "contra_vb": priors.get("count_contra_vb", 0),
                    "next_db": priors.get("count_next_db", 0),
                    "next_vb": priors.get("count_next_vb", 0),
                },
            }
        except Exception as exc:
            self.sparse_prior_scalars = self._default_sparse_priors()
            self.sparse_prior_metadata = {"status": "error", "reason": str(exc)}

    def _iter_pathway_params(self, pathway: str):
        for name, param in self.ncap.params.items():
            if pathway == "ipsi_db":
                if name.startswith("muscle_d_d_") or name == "muscle_ipsi":
                    yield param
            elif pathway == "ipsi_vb":
                if name.startswith("muscle_v_v_") or name == "muscle_ipsi":
                    yield param
            elif pathway == "contra_db":
                # DB -> ventral muscle pathway corresponds to dorsal-B to ventral-muscle block.
                if name.startswith("muscle_v_d_") or name == "muscle_contra":
                    yield param
            elif pathway == "contra_vb":
                # VB -> dorsal muscle pathway corresponds to ventral-B to dorsal-muscle block.
                if name.startswith("muscle_d_v_") or name == "muscle_contra":
                    yield param
            elif pathway == "next_db":
                if name.startswith("bneuron_d_prop_") or name == "bneuron_prop":
                    yield param
            elif pathway == "next_vb":
                if name.startswith("bneuron_v_prop_") or name == "bneuron_prop":
                    yield param

    def _pathway_l2(self, pathway: str) -> torch.Tensor:
        terms = [(p**2).sum() for p in self._iter_pathway_params(pathway)]
        if not terms:
            return torch.zeros((), dtype=torch.float32, device=next(self.parameters()).device)
        return torch.stack(terms).sum()

    def _initialize_sparse_ncap_weights(self) -> None:
        if self.prior_modulation_scale <= 0.0:
            # Tabula-rasa baseline: sign-constrained random init, no synapse-prior scaling.
            with torch.no_grad():
                for pathway in self._PATHWAYS:
                    inhibitory = pathway.startswith("contra_")
                    low, high = (-0.25, -0.05) if inhibitory else (0.05, 0.25)
                    for param in self._iter_pathway_params(pathway):
                        param.uniform_(low, high)
            return

        syn_values = []
        for pathway in self._PATHWAYS:
            syn = float(self.sparse_prior_scalars.get(f"syn_{pathway}", 0.0))
            if syn > 0 and torch.isfinite(torch.tensor(syn)).item():
                syn_values.append(syn)
        syn_scale = max(syn_values) if syn_values else 1.0

        # Small jitter keeps blocks non-identical while preserving pathway priors.
        jitter_fraction = float(max(0.0, min(self.prior_modulation_scale, 0.45)))

        with torch.no_grad():
            for pathway in self._PATHWAYS:
                syn = float(self.sparse_prior_scalars.get(f"syn_{pathway}", 0.0))
                norm_strength = (syn / syn_scale) if syn_scale > 0 else 0.0
                norm_strength = float(min(max(norm_strength, 0.0), 1.0))
                base_strength = float(max(0.05, norm_strength))
                jitter = base_strength * jitter_fraction

                inhibitory = pathway.startswith("contra_")
                low = max(0.0, base_strength - jitter)
                high = min(1.0, base_strength + jitter)
                if low > high:
                    low, high = high, low

                for param in self._iter_pathway_params(pathway):
                    if inhibitory:
                        neg_low = -high
                        neg_high = -low
                        param.uniform_(neg_low, neg_high)
                    else:
                        param.uniform_(low, high)

    def compute_topological_prior_loss(self, prior_reg_lambda: float) -> torch.Tensor:
        lam = float(prior_reg_lambda)
        if lam <= 0.0:
            return torch.zeros((), dtype=torch.float32, device=next(self.parameters()).device)
        d = self.sparse_prior_scalars

        loss = torch.zeros((), dtype=torch.float32, device=next(self.parameters()).device)
        for pathway in self._PATHWAYS:
            dist_coeff = float(d.get(f"dist_{pathway}", 1.0))
            loss = loss + dist_coeff * self._pathway_l2(pathway)
        return lam * loss

    def get_sparse_prior_scalars(self) -> Dict[str, float]:
        return dict(self.sparse_prior_scalars)

    def _init_weights(self):
        for layer in self.critic.critic_network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)
        self._initialize_sparse_ncap_weights()

    def forward(self, observations):
        device = next(self.parameters()).device
        observations = observations.to(device)
        policy_output = self.actor(observations)
        value_output = self.critic(observations)
        return policy_output, value_output

    def get_policy(self, observations):
        return self.actor(observations)

    def get_value(self, observations):
        return self.critic(observations)

    def initialize(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space

        obs_dim = observation_space.shape[0]
        action_dim = action_space.shape[0]
        print(f"Initializing TonicNCAPModel with obs_dim={obs_dim}, action_dim={action_dim}")

        if obs_dim < self.n_joints:
            raise ValueError(f"Observation space too small: {obs_dim} < {self.n_joints}")
        if action_dim != self.n_joints:
            raise ValueError(f"Action space mismatch: {action_dim} != {self.n_joints}")

        resolved_segments = self._requested_num_segments if self._requested_num_segments is not None else action_dim
        if int(resolved_segments) != int(self.num_segments):
            self.num_segments = int(resolved_segments)
            self._load_sparse_priors(self.num_segments)
            self._initialize_sparse_ncap_weights()

        print(
            "TonicNCAPModel initialized successfully "
            f"(num_segments={self.num_segments}, prior_status={self.sparse_prior_metadata.get('status', 'unknown')})"
        )

    def to(self, device):
        super().to(device)
        self.ncap = self.ncap.to(device)
        self.actor = self.actor.to(device)
        self.critic = self.critic.to(device)
        return self


def create_tonic_ncap_model(
    n_joints=6,
    oscillator_period=60,
    memory_size=10,
    critic_sizes=(64, 64),
    critic_activation=nn.Tanh,
    action_noise=0.1,
    num_segments: Optional[int] = None,
    prior_modulation_scale: float = 0.15,
):
    """Factory function to create a Tonic-compatible NCAP model."""
    return TonicNCAPModel(
        n_joints=n_joints,
        oscillator_period=oscillator_period,
        memory_size=memory_size,
        critic_sizes=critic_sizes,
        critic_activation=critic_activation,
        action_noise=action_noise,
        num_segments=num_segments,
        prior_modulation_scale=prior_modulation_scale,
    )
