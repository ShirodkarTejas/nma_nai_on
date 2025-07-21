"""
Environments package for swimmer training.
Contains environment classes and wrappers.
"""

from .environment_types import EnvironmentType
from .mixed_environment import ImprovedMixedSwimmerEnv, test_improved_mixed_environment
from .tonic_wrapper import TonicSwimmerWrapper
from .simple_swimmer import SimpleSwimmerEnv, TonicSimpleSwimmerWrapper

__all__ = [
    'EnvironmentType',
    'ImprovedMixedSwimmerEnv',
    'test_improved_mixed_environment',
    'TonicSwimmerWrapper',
    'SimpleSwimmerEnv',
    'TonicSimpleSwimmerWrapper'
] 