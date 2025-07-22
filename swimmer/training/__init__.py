"""
Training package for swimmer models.
Contains trainer classes and RL training utilities.
"""

from .swimmer_trainer import SwimmerTrainer
from .improved_ncap_trainer import ImprovedNCAPTrainer

__all__ = ['SwimmerTrainer', 'ImprovedNCAPTrainer'] 