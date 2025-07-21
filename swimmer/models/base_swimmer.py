#!/usr/bin/env python3
"""
Base swimmer model class
"""

import torch.nn as nn
import abc

class BaseSwimmerModel(nn.Module, metaclass=abc.ABCMeta):
    """
    Abstract base class for swimmer models.
    """
    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        pass 