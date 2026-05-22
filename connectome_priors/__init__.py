"""NCAP swimmer overlay with biological priors."""

from .swimmer_priors import generate_ncap_segment_priors, refresh_inventory_files
from .c_elegans_connectome import discover_connectome_file
from .c_elegans_geometry import discover_neuron_geometry

__all__ = [
    "refresh_inventory_files",
    "generate_ncap_segment_priors",
    "discover_connectome_file",
    "discover_neuron_geometry",
]

# Optional torch-backed wrappers: keep import of lightweight prior utilities working
# even when torch/tonic runtime dependencies are absent.
try:
    from .swimmer import (
        BaseSwimmerActor,
        BaseSwimmerModule,
        SwimmerActor,
        SwimmerModule,
        SwimmerModuleWithPriors,
        create_tonic_ncap_model,
        ppo_swimmer_model_with_priors,
    )
except Exception:
    pass
else:
    __all__.extend(
        [
            "BaseSwimmerModule",
            "BaseSwimmerActor",
            "SwimmerModule",
            "SwimmerActor",
            "SwimmerModuleWithPriors",
            "create_tonic_ncap_model",
            "ppo_swimmer_model_with_priors",
        ]
    )
