from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROBOCASA_DIR = Path(__file__).resolve().parents[1] / "robocasa"
sys.path.insert(0, str(ROBOCASA_DIR))

from pi05_rollout import _apply_base_action_mode  # noqa: E402


def test_full_does_not_modify_action():
    action = np.arange(12, dtype=np.float32)
    assert np.array_equal(_apply_base_action_mode(action, "full", 0.15), action)


def test_frozen_only_zeros_base_axes():
    action = np.arange(12, dtype=np.float32)
    result = _apply_base_action_mode(action, "frozen", 0.15)
    assert np.array_equal(result[7:10], np.zeros(3))
    assert np.array_equal(result[:7], action[:7])
    assert np.array_equal(result[10:], action[10:])


def test_residual_only_clips_base_axes():
    action = np.arange(12, dtype=np.float32) - 5
    result = _apply_base_action_mode(action, "residual", 0.15)
    assert np.allclose(result[7:10], [0.15, 0.15, 0.15])
    assert np.array_equal(result[:7], action[:7])
    assert np.array_equal(result[10:], action[10:])


def test_invalid_mode():
    with pytest.raises(ValueError, match="base_action_mode"):
        _apply_base_action_mode(np.zeros(12), "invalid", 0.15)
