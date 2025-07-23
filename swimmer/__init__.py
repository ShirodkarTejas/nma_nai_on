"""
Swimmer Package
A C. elegans-inspired swimmer agent with NCAP models and RL training.
"""

__version__ = "1.0.0"
__author__ = "NeuroAI Course"

# Import main components for easy access - LAZY to avoid EGL conflicts
# These imports are deferred to avoid triggering dm_control imports before environment setup

def _lazy_import_environments():
    """Lazy import to avoid EGL conflicts during package import."""
    from .environments import ImprovedMixedSwimmerEnv, EnvironmentType
    return ImprovedMixedSwimmerEnv, EnvironmentType

def _lazy_import_models():
    """Lazy import to avoid EGL conflicts during package import.""" 
    from .models import NCAPSwimmer, NCAPSwimmerActor
    return NCAPSwimmer, NCAPSwimmerActor

def _lazy_import_training():
    """Lazy import to avoid EGL conflicts during package import."""
    from .training import SwimmerTrainer
    return SwimmerTrainer

def _lazy_import_utils():
    """Lazy import to avoid EGL conflicts during package import."""
    from .utils import TrainingLogger
    return TrainingLogger

# Skip all imports at package level to avoid EGL conflicts
# All imports will be done lazily when needed

__all__ = [
    '_lazy_import_environments',
    '_lazy_import_models', 
    '_lazy_import_training',
    '_lazy_import_utils'
] 