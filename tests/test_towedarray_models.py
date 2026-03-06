"""Unit tests for towed-array follower model helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import sys
import types

import numpy as np

from .support import FakeBase, FakeProperty, load_module_from_repo


def _install_fake_stonesoup_towedarray(monkeypatch) -> None:
    stonesoup_module = types.ModuleType("stonesoup")

    base_module = types.ModuleType("stonesoup.base")
    base_module.Base = FakeBase
    base_module.Property = FakeProperty

    movable_module = types.ModuleType("stonesoup.movable")
    movable_movable_module = types.ModuleType("stonesoup.movable.movable")

    class MovingMovable(FakeBase):
        def __iter__(self):
            return iter(getattr(self, "states", []))

        @property
        def position(self):
            return self.states[-1].state_vector

        def move(self, timestamp, **kwargs):
            _ = kwargs
            current_state = self.states[-1]
            next_vector = self.transition_model.function(current_state)
            self.states.append(GroundTruthState(next_vector, timestamp=timestamp))

    movable_movable_module.MovingMovable = MovingMovable
    movable_module.movable = movable_movable_module

    platform_module = types.ModuleType("stonesoup.platform")
    platform_base_module = types.ModuleType("stonesoup.platform.base")

    class MultiTransitionMovingPlatform(FakeBase):
        @property
        def states(self):
            return self.movement_controller.states

        @property
        def state(self):
            return self.states[-1]

    platform_base_module.MultiTransitionMovingPlatform = MultiTransitionMovingPlatform
    platform_module.base = platform_base_module

    array_module = types.ModuleType("stonesoup.types.array")
    array_module.StateVector = np.array
    array_module.StateVectors = np.ndarray

    groundtruth_module = types.ModuleType("stonesoup.types.groundtruth")

    @dataclass
    class GroundTruthState:
        state_vector: np.ndarray
        timestamp: datetime | None = None

    groundtruth_module.GroundTruthState = GroundTruthState

    state_module = types.ModuleType("stonesoup.types.state")
    state_module.State = GroundTruthState

    stonesoup_module.base = base_module
    stonesoup_module.movable = movable_module
    stonesoup_module.platform = platform_module

    monkeypatch.setitem(sys.modules, "stonesoup", stonesoup_module)
    monkeypatch.setitem(sys.modules, "stonesoup.base", base_module)
    monkeypatch.setitem(sys.modules, "stonesoup.movable", movable_module)
    monkeypatch.setitem(sys.modules, "stonesoup.movable.movable", movable_movable_module)
    monkeypatch.setitem(sys.modules, "stonesoup.platform", platform_module)
    monkeypatch.setitem(sys.modules, "stonesoup.platform.base", platform_base_module)
    monkeypatch.setitem(sys.modules, "stonesoup.types.array", array_module)
    monkeypatch.setitem(sys.modules, "stonesoup.types.groundtruth", groundtruth_module)
    monkeypatch.setitem(sys.modules, "stonesoup.types.state", state_module)


def _load_towedarray(monkeypatch):
    _install_fake_stonesoup_towedarray(monkeypatch)
    return load_module_from_repo("bluepebble/platform/towedarray.py", "bluepebble.platform.towedarray")


def test_follower_model_uses_default_direction_when_positions_overlap(monkeypatch) -> None:
    """Follower should fall back to -x direction when leader and follower overlap."""
    module = _load_towedarray(monkeypatch)

    leader = types.SimpleNamespace(position=np.array([[10.0], [5.0], [2.0]]))
    follower_state = types.SimpleNamespace(state_vector=np.array([[10.0], [5.0], [2.0]]))
    model = module._FollowerModel(leader=leader, offset=4.0)

    new_position = model.function(follower_state)
    assert new_position.shape == (3, 3)
    np.testing.assert_allclose(np.diag(new_position), np.array([14.0, 5.0, 2.0]))


def test_follower_model_moves_toward_leader_with_configured_offset(monkeypatch) -> None:
    """Follower should end up exactly `offset` behind the leader directionally."""
    module = _load_towedarray(monkeypatch)

    leader = types.SimpleNamespace(position=np.array([[3.0], [4.0], [0.0]]))
    follower_state = types.SimpleNamespace(state_vector=np.array([[0.0], [0.0], [0.0]]))
    model = module._FollowerModel(leader=leader, offset=1.0)

    new_position = model.function(follower_state)
    np.testing.assert_allclose(new_position.flatten(), np.array([2.4, 3.2, 0.0]), atol=1e-12)


def test_towed_array_follower_model_handles_zero_horizontal_distance(monkeypatch) -> None:
    """If slant offset is shorter than depth gap, horizontal offset should be zero."""
    module = _load_towedarray(monkeypatch)

    leader = types.SimpleNamespace(position=np.array([[5.0], [1.0], [120.0]]))
    follower_state = types.SimpleNamespace(state_vector=np.array([[5.0], [1.0], [80.0]]))
    model = module._TowedArrayFollowerModel(leader=leader, offset=10.0, array_depth_m=60.0)

    new_position = model.function(follower_state)
    np.testing.assert_allclose(new_position.flatten(), np.array([5.0, 1.0, 60.0]))


def test_towed_array_follower_model_keeps_depth_and_applies_horizontal_offset(monkeypatch) -> None:
    """Follower should preserve depth and apply horizontal geometry from slant range."""
    module = _load_towedarray(monkeypatch)

    leader = types.SimpleNamespace(position=np.array([[10.0], [0.0], [100.0]]))
    follower_state = types.SimpleNamespace(state_vector=np.array([[4.0], [0.0], [100.0]]))
    model = module._TowedArrayFollowerModel(leader=leader, offset=13.0, array_depth_m=95.0)

    new_position = model.function(follower_state)
    # horizontal_offset = sqrt(13^2 - 5^2) = 12
    np.testing.assert_allclose(new_position.flatten(), np.array([-2.0, 0.0, 95.0]), atol=1e-12)


def test_platform_state_position_returns_host_xyz(monkeypatch) -> None:
    """PlatformState.position should expose host x/y/z slices from host state vector."""
    module = _load_towedarray(monkeypatch)

    host_vector = np.array([[1.0], [9.0], [2.0], [8.0], [3.0], [7.0]])
    host_state = types.SimpleNamespace(state_vector=host_vector)
    host = module.HostState(state=host_state, heading_rad=0.0)
    array = module.ArrayState(
        num_sensors=1,
        state_vector=np.array([[0.0], [0.0], [0.0]]),
        ref_state_vector=np.array([[0.0], [0.0], [0.0]]),
    )
    state = module.PlatformState(timestamp=datetime(2026, 1, 1), host=host, array=array)

    np.testing.assert_allclose(state.position.flatten(), np.array([1.0, 2.0, 3.0]))