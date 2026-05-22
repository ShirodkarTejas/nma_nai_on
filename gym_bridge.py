#!/usr/bin/env python3
"""Global compatibility bridge mapping `gym` imports to gymnasium or a safe dummy."""

import importlib
import sys
import types


def _build_dummy_gym(import_error: Exception):
    class _MissingGymBase:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "Neither 'gymnasium' nor 'gym' compatibility layer is available. "
                "Install gymnasium to use environment wrappers."
            ) from import_error

    class _MissingBox:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "Cannot create gym spaces because gymnasium is not installed."
            ) from import_error

    spaces_module = types.ModuleType("gym.spaces")
    spaces_module.Box = _MissingBox

    wrappers_module = types.ModuleType("gym.wrappers")
    wrappers_module.Wrapper = _MissingGymBase

    dummy = types.ModuleType("gym")
    dummy.Env = _MissingGymBase
    dummy.Wrapper = _MissingGymBase
    dummy.spaces = spaces_module
    dummy.wrappers = wrappers_module

    sys.modules["gym.spaces"] = spaces_module
    sys.modules["gym.wrappers"] = wrappers_module
    return dummy


try:
    import gymnasium as gym
except ImportError as exc:
    gym = _build_dummy_gym(exc)
else:
    if hasattr(gym, "spaces"):
        sys.modules.setdefault("gym.spaces", gym.spaces)
    try:
        wrappers_module = importlib.import_module("gymnasium.wrappers")
        gym.wrappers = wrappers_module
        sys.modules.setdefault("gym.wrappers", wrappers_module)
    except Exception:
        pass

sys.modules["gym"] = gym
