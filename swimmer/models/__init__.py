"""
Models package for swimmer training.
Contains NCAP and other neural network models.
"""

from .ncap_swimmer import NCAPSwimmer, NCAPSwimmerActor, BaseSwimmerModel
from .tonic_ncap import TonicNCAPModel, create_tonic_ncap_model

__all__ = [
    'NCAPSwimmer',
    'NCAPSwimmerActor', 
    'BaseSwimmerModel',
    'TonicNCAPModel',
    'create_tonic_ncap_model'
] 