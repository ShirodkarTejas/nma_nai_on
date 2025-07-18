"""
Swimmer Package
A C. elegans-inspired swimmer agent with NCAP models and RL training.
"""

__version__ = "1.0.0"
__author__ = "NeuroAI Course"

# Import main components for easy access
from .environments import ImprovedMixedSwimmerEnv, EnvironmentType
from .models import NCAPSwimmer, NCAPSwimmerActor
from .training import SwimmerTrainer
from .utils import TrainingLogger

__all__ = [
    'ImprovedMixedSwimmerEnv',
    'EnvironmentType', 
    'NCAPSwimmer',
    'NCAPSwimmerActor',
    'SwimmerTrainer',
    'TrainingLogger'
] 