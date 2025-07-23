"""
Training package for swimmer models.
Contains trainer classes and RL training utilities.
"""

# Import trainer classes directly to make them available
from .swimmer_trainer import SwimmerTrainer
from .improved_ncap_trainer import ImprovedNCAPTrainer
from .curriculum_trainer import CurriculumNCAPTrainer
from .simple_biological_trainer import SimpleBiologicalTrainer
from .biological_preserving_trainer import BiologicalPreservingTrainer

__all__ = [
    'SwimmerTrainer',
    'ImprovedNCAPTrainer', 
    'CurriculumNCAPTrainer',
    'SimpleBiologicalTrainer',
    'BiologicalPreservingTrainer'
] 