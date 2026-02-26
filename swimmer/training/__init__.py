"""
Training package for swimmer models.
Contains trainer classes and RL training utilities.
"""

from .swimmer_trainer import SwimmerTrainer
from .ncap_trainer import NCAPTrainer
from .curriculum_trainer import CurriculumNCAPTrainer

__all__ = [
    'SwimmerTrainer',
    'NCAPTrainer',
    'CurriculumNCAPTrainer',
]
