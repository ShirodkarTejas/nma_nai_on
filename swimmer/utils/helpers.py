#!/usr/bin/env python3
"""
Helper functions for swimmer training.
Contains utility functions for data processing and manipulation.
"""

import numpy as np

def flatten_observation(obs):
    """Flatten observation dictionary or array into a single numpy array."""
    if isinstance(obs, dict):
        return np.concatenate([obs[key].flatten() for key in sorted(obs.keys())])
    elif isinstance(obs, (list, tuple)):
        return np.concatenate([flatten_observation(o) for o in obs])
    else:
        return obs.flatten() 