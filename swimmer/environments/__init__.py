"""
Environments package for swimmer training.
Contains environment classes and wrappers.
"""

# Lazy imports to avoid EGL conflicts
def _lazy_import_environments():
    """Lazy import to avoid EGL conflicts during package import."""
    from .environment_types import EnvironmentType
    from .mixed_environment import MixedSwimmerEnv
    from .tonic_wrapper import TonicSwimmerWrapper
    return EnvironmentType, MixedSwimmerEnv, TonicSwimmerWrapper

# Only import non-dm_control components immediately
from .environment_types import EnvironmentType

__all__ = [
    'EnvironmentType',
    '_lazy_import_environments'
] 
