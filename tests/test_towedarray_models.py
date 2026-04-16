"""Unit tests for TowedArrayPlatform."""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pytest

from .support import FakeBase, FakeProperty, load_module_from_repo


def _column_vector(data, *args, **kwargs):
    """Return a 2-D column vector from 1-D input; pass through otherwise."""
    arr = np.array(data, *args, **kwargs)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


@dataclass
class _FakeState:
    state_vector: np.ndarray
    timestamp: datetime | None = None


@dataclass
class _FakeElement:
    """Minimal hydrophone element with a mutable state history."""

    states: list = field(default_factory=list)


class _FakeLinearHydrophoneArray:
    """Minimal LinearHydrophoneArray that records move() calls for platform tests."""

    def __init__(self, elements, element_spacing_m=1.0, reference_element_idx=0):
        self.elements = list(elements)
        self.element_spacing_m = float(element_spacing_m)
        self.reference_element_idx = int(reference_element_idx)

    @property
    def num_elements(self):
        return len(self.elements)

    def move(self, leader_pos, trail_direction_xy, cable_length_m, array_depth_m, timestamp):
        lp = np.asarray(leader_pos, dtype=float).flatten()
        td = np.asarray(trail_direction_xy, dtype=float).flatten()
        depth_diff = abs(float(lp[2]) - array_depth_m)
        if cable_length_m**2 < depth_diff**2:
            raise ValueError(
                f"Cable length ({cable_length_m}m) is too short for the depth "
                f"difference ({depth_diff:.3f}m) between tow point and array depth."
            )
        horizontal_cable = float(np.sqrt(cable_length_m**2 - depth_diff**2))
        first_xy = lp[:2] + horizontal_cable * td
        for i, element in enumerate(self.elements):
            element_xy = first_xy + i * self.element_spacing_m * td
            state = _FakeState(
                state_vector=_column_vector(
                    [float(element_xy[0]), float(element_xy[1]), array_depth_m]
                ),
                timestamp=timestamp,
            )
            element.states.append(state)


def _install_fake_stonesoup_towedarray(monkeypatch) -> None:
    stonesoup_module = types.ModuleType("stonesoup")
    base_module = types.ModuleType("stonesoup.base")
    base_module.Base = FakeBase
    base_module.Property = FakeProperty

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

    sensor_package = types.ModuleType("bluepebble.sensor")
    sensor_package.__path__ = []
    sensor_array_module = types.ModuleType("bluepebble.sensor.array")
    sensor_array_module.LinearHydrophoneArray = _FakeLinearHydrophoneArray

    stonesoup_module.base = base_module
    stonesoup_module.platform = platform_module

    monkeypatch.setitem(sys.modules, "stonesoup", stonesoup_module)
    monkeypatch.setitem(sys.modules, "stonesoup.base", base_module)
    monkeypatch.setitem(sys.modules, "stonesoup.platform", platform_module)
    monkeypatch.setitem(sys.modules, "stonesoup.platform.base", platform_base_module)
    monkeypatch.setitem(sys.modules, "bluepebble.sensor", sensor_package)
    monkeypatch.setitem(sys.modules, "bluepebble.sensor.array", sensor_array_module)


def _load_towedarray(monkeypatch):
    _install_fake_stonesoup_towedarray(monkeypatch)
    return load_module_from_repo(
        "bluepebble/platform/towedarray.py",
        "bluepebble.platform.towedarray",
    )


_T0 = datetime(2024, 1, 1)
_T1 = _T0 + timedelta(seconds=1)
_T2 = _T0 + timedelta(seconds=2)


def _make_array(
    num_elements: int = 2,
    element_spacing_m: float = 5.0,
) -> _FakeLinearHydrophoneArray:
    """Return a FakeLinearHydrophoneArray with the given number of fresh elements."""
    elements = [_FakeElement() for _ in range(num_elements)]
    return _FakeLinearHydrophoneArray(elements=elements, element_spacing_m=element_spacing_m)


class _FakeController:
    """Minimal movement controller with a constant-velocity model."""

    def __init__(self, init_state: _FakeState, velocity_x: float = 5.0):
        self.states = [init_state]
        self._velocity_x = velocity_x

    def move(self, timestamp: datetime, **kwargs) -> None:
        current = self.states[-1]
        s = current.state_vector.copy()
        s[0, 0] += self._velocity_x  # advance x by vx
        self.states.append(_FakeState(state_vector=s, timestamp=timestamp))


def _make_platform(
    module,
    num_elements: int = 2,
    cable_length_m: float = 10.0,
    element_spacing_m: float = 5.0,
    array_depth_m: float = 5.0,
    host_velocity_x: float = 5.0,
    host_z: float | None = None,
    **extra,
):
    """Return a TowedArrayPlatform wired to a fake constant-velocity host."""
    z = array_depth_m if host_z is None else host_z
    # 6-D state: [x, vx, y, vy, z, vz]
    sv = np.array([[0.0], [host_velocity_x], [0.0], [0.0], [z], [0.0]], dtype=float)
    init_state = _FakeState(state_vector=sv, timestamp=_T0)
    controller = _FakeController(init_state, velocity_x=host_velocity_x)
    sensor_array = _make_array(num_elements=num_elements, element_spacing_m=element_spacing_m)

    kwargs = dict(
        movement_controller=controller,
        position_mapping=[0, 2, 4],
        sensor_array=sensor_array,
        cable_length_m=cable_length_m,
        array_depth_m=array_depth_m,
    )
    kwargs.update(extra)
    return module.TowedArrayPlatform(**kwargs)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_platform_constructs_and_initialises_sensor_array(monkeypatch) -> None:
    """TowedArrayPlatform should call sensor_array.move() during construction."""
    module = _load_towedarray(monkeypatch)
    platform = _make_platform(module, num_elements=2)

    # Each element should have exactly one state from the initialisation call.
    for element in platform.sensor_array.elements:
        assert len(element.states) == 1
        assert element.states[0].timestamp == _T0


def test_platform_with_explicit_velocity_mapping(monkeypatch) -> None:
    """Explicitly supplied velocity_mapping should be stored and returned unchanged."""
    module = _load_towedarray(monkeypatch)
    platform = _make_platform(module, velocity_mapping=[1, 3, 5])

    assert platform.velocity_mapping == [1, 3, 5]
    assert platform._resolved_velocity_mapping() == [1, 3, 5]


def test_platform_constructs_with_no_initial_states(monkeypatch) -> None:
    """When the controller has no states, sensor_array.move() should not be called."""
    module = _load_towedarray(monkeypatch)
    _install_fake_stonesoup_towedarray(monkeypatch)
    module = load_module_from_repo(
        "bluepebble/platform/towedarray.py",
        "bluepebble.platform.towedarray",
    )
    sensor_array = _make_array(num_elements=1)

    class _EmptyController:
        states = []

    platform = module.TowedArrayPlatform(
        movement_controller=_EmptyController(),
        position_mapping=[0, 2, 4],
        sensor_array=sensor_array,
        cable_length_m=10.0,
        array_depth_m=5.0,
    )

    # No move() called → no element states.
    for element in platform.sensor_array.elements:
        assert len(element.states) == 0


# ---------------------------------------------------------------------------
# _resolved_velocity_mapping
# ---------------------------------------------------------------------------


def test_resolved_velocity_mapping_returns_explicit_when_set(monkeypatch) -> None:
    """Explicit velocity_mapping should be returned unchanged."""
    module = _load_towedarray(monkeypatch)
    platform = _make_platform(module, velocity_mapping=[2, 4, 6])

    assert list(platform._resolved_velocity_mapping()) == [2, 4, 6]


def test_resolved_velocity_mapping_falls_back_to_position_plus_one(monkeypatch) -> None:
    """When velocity_mapping is None the fallback is position_mapping + 1."""
    module = _load_towedarray(monkeypatch)
    platform = _make_platform(module)  # velocity_mapping defaults to None

    assert platform.velocity_mapping is None
    assert list(platform._resolved_velocity_mapping()) == [1, 3, 5]


# ---------------------------------------------------------------------------
# _trail_direction_xy
# ---------------------------------------------------------------------------


def test_trail_direction_xy_returns_opposite_to_velocity(monkeypatch) -> None:
    """Trail direction should be the unit vector opposite to the host velocity."""
    module = _load_towedarray(monkeypatch)
    platform = _make_platform(module, host_velocity_x=5.0)

    sv = platform.states[0].state_vector
    trail = platform._trail_direction_xy(sv)

    # Host moves in +x, so array trails in -x direction.
    np.testing.assert_allclose(trail, np.array([-1.0, 0.0]), atol=1e-12)


def test_trail_direction_xy_uses_fallback_when_velocity_zero(monkeypatch) -> None:
    """Zero host velocity should not raise; the default -x heading is used instead."""
    module = _load_towedarray(monkeypatch)
    platform = _make_platform(module, host_velocity_x=0.0)

    sv = platform.states[0].state_vector
    trail = platform._trail_direction_xy(sv)

    np.testing.assert_allclose(trail, np.array([-1.0, 0.0]), atol=1e-12)


# ---------------------------------------------------------------------------
# Cable length validation
# ---------------------------------------------------------------------------


def test_initialise_raises_when_cable_too_short_for_depth(monkeypatch) -> None:
    """A ValueError should be raised when the cable cannot span the depth difference."""
    module = _load_towedarray(monkeypatch)

    with pytest.raises(ValueError, match="too short"):
        # host z=0.0, array_depth_m=100.0 → depth_diff=100 > cable_length_m=50
        _make_platform(module, cable_length_m=50.0, array_depth_m=100.0, host_z=0.0)


def test_initialise_at_exact_depth_boundary_does_not_raise(monkeypatch) -> None:
    """When cable exactly spans the depth difference, horizontal offset is zero and no error is raised."""  # noqa: E501
    module = _load_towedarray(monkeypatch)

    # depth_diff == cable_length_m == 10 → horizontal_cable = 0
    platform = _make_platform(
        module, num_elements=1, cable_length_m=10.0, array_depth_m=10.0, host_z=0.0
    )

    sensor_pos = platform.sensor_array.elements[0].states[-1].state_vector.flatten()
    np.testing.assert_allclose(sensor_pos[0], 0.0, atol=1e-10)  # no horizontal displacement
    np.testing.assert_allclose(sensor_pos[2], 10.0)  # correct depth


# ---------------------------------------------------------------------------
# move()
# ---------------------------------------------------------------------------


def test_move_advances_host_and_appends_element_states(monkeypatch) -> None:
    """move() should advance the host and append one new state per element."""
    module = _load_towedarray(monkeypatch)
    platform = _make_platform(module, num_elements=2)

    # After construction: 1 state per element.
    assert all(len(e.states) == 1 for e in platform.sensor_array.elements)

    platform.move(_T1)

    # After one move: 2 states per element.
    assert all(len(e.states) == 2 for e in platform.sensor_array.elements)
    assert all(e.states[1].timestamp == _T1 for e in platform.sensor_array.elements)


def test_multiple_moves_accumulate_element_states(monkeypatch) -> None:
    """Each call to move() should append exactly one state per element."""
    module = _load_towedarray(monkeypatch)
    platform = _make_platform(module, num_elements=1)

    platform.move(_T1)
    platform.move(_T2)

    assert len(platform.sensor_array.elements[0].states) == 3
    assert platform.sensor_array.elements[0].states[-1].timestamp == _T2


def test_move_host_advances_position(monkeypatch) -> None:
    """Host position should increase by vx after one move step."""
    module = _load_towedarray(monkeypatch)
    platform = _make_platform(module, host_velocity_x=5.0)

    platform.move(_T1)

    x_after = platform.state.state_vector[0, 0]
    np.testing.assert_allclose(x_after, 5.0, atol=1e-10)


# ---------------------------------------------------------------------------
# host_path
# ---------------------------------------------------------------------------


def test_host_path_returns_array_of_positions(monkeypatch) -> None:
    """host_path should stack host positions into a 2-D array."""
    module = _load_towedarray(monkeypatch)
    platform = _make_platform(module)
    platform.move(_T1)

    path = platform.host_path

    assert path is not None
    assert path.shape == (2, 3)
    np.testing.assert_allclose(path[0], [0.0, 0.0, 5.0])  # initial position
    np.testing.assert_allclose(path[1, 0], 5.0, atol=1e-10)  # x advanced by vx=5


def test_host_path_returns_none_when_no_states(monkeypatch) -> None:
    """host_path should return None when the host has no recorded states."""
    module = _load_towedarray(monkeypatch)
    platform = _make_platform(module)

    platform.movement_controller.states.clear()

    assert platform.host_path is None


# ---------------------------------------------------------------------------
# sensor_paths
# ---------------------------------------------------------------------------


def test_sensor_paths_returns_list_per_element(monkeypatch) -> None:
    """sensor_paths should return one position array per array element."""
    module = _load_towedarray(monkeypatch)
    platform = _make_platform(module, num_elements=2)
    platform.move(_T1)

    paths = platform.sensor_paths

    assert len(paths) == 2
    for path in paths:
        assert path.shape[0] == 2  # two timesteps
        assert path.shape[1] == 3  # 3-D position


def test_sensor_paths_returns_empty_list_when_no_elements(monkeypatch) -> None:
    """sensor_paths should return [] when the sensor array has no elements."""
    module = _load_towedarray(monkeypatch)
    platform = _make_platform(module, num_elements=0)

    assert platform.sensor_paths == []


def test_sensor_paths_returns_empty_array_for_element_with_no_states(monkeypatch) -> None:
    """An element with no recorded states should produce a (0, 3) shape array."""
    module = _load_towedarray(monkeypatch)
    platform = _make_platform(module, num_elements=1)

    platform.sensor_array.elements[0].states.clear()
    paths = platform.sensor_paths

    assert len(paths) == 1
    assert paths[0].shape == (0, 3)


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------


def test_repr_contains_configuration_parameters(monkeypatch) -> None:
    """__repr__ should include key platform configuration parameters."""
    module = _load_towedarray(monkeypatch)
    platform = _make_platform(module, num_elements=4, cable_length_m=20.0)

    r = repr(platform)

    assert "num_elements=4" in r
    assert "cable_length_m=20.0" in r
