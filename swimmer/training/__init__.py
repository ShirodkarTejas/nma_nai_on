"""
Training package for swimmer models.
Contains trainer classes and RL training utilities.
"""

from .swimmer_trainer import SwimmerTrainer, PPOAgent
from .improved_ncap_trainer import ImprovedNCAPTrainer
from .simple_swimmer_trainer import SimpleSwimmerTrainer

__all__ = [
    'SwimmerTrainer',
    'PPOAgent',
    'ImprovedNCAPTrainer',
    'SimpleSwimmerTrainer'
] 