"""
Environments package for swimmer training.
Contains environment classes and wrappers.
"""

from .environment_types import EnvironmentType
from .mixed_environment import ImprovedMixedSwimmerEnv
from .tonic_wrapper import TonicSwimmerWrapper

__all__ = [
    'EnvironmentType',
    'ImprovedMixedSwimmerEnv'
] 