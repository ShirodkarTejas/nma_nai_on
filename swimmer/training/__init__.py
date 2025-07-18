"""
Training package for swimmer models.
Contains trainer classes and RL training utilities.
"""

from .swimmer_trainer import SwimmerTrainer, PPOAgent

__all__ = [
    'SwimmerTrainer',
    'PPOAgent'
] 