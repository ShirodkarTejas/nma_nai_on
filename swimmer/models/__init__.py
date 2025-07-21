"""
Models package for swimmer neural networks.
Contains NCAP and other swimmer model implementations.
"""

from .base_swimmer import BaseSwimmerModel
from .ncap_swimmer import NCAPSwimmer, NCAPSwimmerActor
from .tonic_ncap import create_tonic_ncap_model, TonicNCAPModel

__all__ = [
    'BaseSwimmerModel',
    'NCAPSwimmer', 
    'NCAPSwimmerActor',
    'create_tonic_ncap_model',
    'TonicNCAPModel'
] 