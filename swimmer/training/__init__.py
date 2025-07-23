"""
Training package for swimmer models.
Contains trainer classes and RL training utilities.
"""

# Use lazy imports to avoid EGL conflicts
def _lazy_import_trainers():
    """Lazy import to avoid EGL conflicts during package import."""
    from .swimmer_trainer import SwimmerTrainer
    from .improved_ncap_trainer import ImprovedNCAPTrainer
    return SwimmerTrainer, ImprovedNCAPTrainer

__all__ = ['_lazy_import_trainers'] 