"""Unit tests for towed-array follower model helpers."""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pytest

from .support import FakeBase, FakeProperty, load_module_from_repo


def _column_vector(data, *args, **kwargs):
    """Create a 2-D column vector (N×1) from 1-D input; pass through otherwise."""
    arr = np.array(data, *args, **kwargs)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


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
            sv = self.states[-1].state_vector
            pm = getattr(self, "position_mapping", None)
            if pm is not None:
                return sv[pm]
            return sv

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
    array_module.StateVector = _column_vector
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
    return load_module_from_repo(
        "bluepebble/platform/towedarray.py",
        "bluepebble.platform.towedarray",
    )


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


# ---------------------------------------------------------------------------
# _TowedArrayFollowerModel – zero XY distance fallback
# ---------------------------------------------------------------------------


def test_towed_array_follower_model_uses_default_xy_direction_when_positions_overlap(
    monkeypatch,
) -> None:
    """When leader and follower share the same XY position, fall back to -x direction."""
    module = _load_towedarray(monkeypatch)

    # Leader is directly above follower (same XY), so dist_to_leader_xy == 0
    leader = types.SimpleNamespace(position=np.array([[5.0], [3.0], [100.0]]))
    follower_state = types.SimpleNamespace(state_vector=np.array([[5.0], [3.0], [90.0]]))
    model = module._TowedArrayFollowerModel(leader=leader, offset=30.0, array_depth_m=90.0)

    new_position = model.function(follower_state)

    # depth_difference = |100 - 90| = 10; horizontal_offset = sqrt(30^2 - 10^2) = sqrt(800)
    # direction_vec_xy falls back to [[-1], [0]]
    expected_x = 5.0 + np.sqrt(800)
    np.testing.assert_allclose(new_position.flatten(), [expected_x, 3.0, 90.0], atol=1e-12)


# ---------------------------------------------------------------------------
# TowedArrayPlatform – helpers
# ---------------------------------------------------------------------------

_T0 = datetime(2024, 1, 1)
_T1 = _T0 + timedelta(seconds=1)
_T2 = _T0 + timedelta(seconds=2)


def _make_platform(
    module,
    num_sensors: int = 2,
    cable_length_m: float = 10.0,
    sensor_spacing_m: float = 5.0,
    array_depth_m: float = 5.0,
    host_velocity_x: float = 5.0,
    host_z: float | None = None,
    **extra,
):
    """Return a ``TowedArrayPlatform`` wired to a fake constant-velocity host."""
    GroundTruthState = module.GroundTruthState
    MovingMovable = module.MovingMovable

    # 6-D state: [x, vx, y, vy, z, vz]; host z defaults to array_depth_m
    z = array_depth_m if host_z is None else host_z
    sv = np.array([[0.0], [host_velocity_x], [0.0], [0.0], [z], [0.0]], dtype=float)
    init_state = GroundTruthState(state_vector=sv, timestamp=_T0)

    class _CVModel:
        """Advance x by vx and y by vy each step."""

        def function(self, state):
            s = state.state_vector.copy()
            s[0, 0] += s[1, 0]
            s[2, 0] += s[3, 0]
            return s

    controller = MovingMovable(
        states=[init_state],
        position_mapping=[0, 2, 4],
        transition_model=_CVModel(),
    )

    kwargs = dict(
        movement_controller=controller,
        position_mapping=[0, 2, 4],
        num_sensors=num_sensors,
        cable_length_m=cable_length_m,
        sensor_spacing_m=sensor_spacing_m,
        array_depth_m=array_depth_m,
    )
    kwargs.update(extra)
    return module.TowedArrayPlatform(**kwargs)


# ---------------------------------------------------------------------------
# TowedArrayPlatform – construction
# ---------------------------------------------------------------------------


def test_platform_constructs_and_initialises_sensors(monkeypatch) -> None:
    """TowedArrayPlatform should create the correct number of towed sensors."""
    module = _load_towedarray(monkeypatch)

    platform = _make_platform(module, num_sensors=3)

    assert len(platform.towed_sensors) == 3
    # Initial state should be recorded in history
    assert len(platform.platform_history) == 1
    assert platform.platform_history[0].timestamp == _T0


def test_platform_constructs_with_explicit_velocity_mapping(monkeypatch) -> None:
    """When velocity_mapping is provided it should be used instead of the fallback."""
    module = _load_towedarray(monkeypatch)

    platform = _make_platform(module, velocity_mapping=[1, 3, 5])

    assert platform.velocity_mapping == [1, 3, 5]
    assert platform._resolved_velocity_mapping() == [1, 3, 5]


# ---------------------------------------------------------------------------
# TowedArrayPlatform – _resolved_velocity_mapping
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
# TowedArrayPlatform – _initialise_sensor_array edge cases
# ---------------------------------------------------------------------------


def test_initialise_sensor_array_raises_when_segment_too_short_for_depth(
    monkeypatch,
) -> None:
    """A ValueError is raised when cable length cannot span the depth difference."""
    module = _load_towedarray(monkeypatch)

    with pytest.raises(ValueError, match="too short"):
        # host z == 0, array_depth_m == 100 → depth_diff == 100 > cable_length_m == 50
        _make_platform(module, cable_length_m=50.0, array_depth_m=100.0, host_z=0.0)


def test_initialise_sensor_array_uses_zero_velocity_fallback_heading(monkeypatch) -> None:
    """Zero host velocity should not raise; the default -x heading is used instead."""
    module = _load_towedarray(monkeypatch)

    # host_velocity_x == 0 triggers the backwards_heading fallback
    platform = _make_platform(module, num_sensors=1, host_velocity_x=0.0)

    assert len(platform.towed_sensors) == 1
    # Sensor should be placed 10 m in the +x direction (opposite of -x heading)
    sensor_pos = platform.towed_sensors[0].states[-1].state_vector.flatten()
    np.testing.assert_allclose(sensor_pos[0], -10.0, atol=1e-10)


# ---------------------------------------------------------------------------
# TowedArrayPlatform – _capture_platform_state
# ---------------------------------------------------------------------------


def test_capture_platform_state_early_returns_on_missing_host_state(monkeypatch) -> None:
    """_capture_platform_state should silently skip when host state is absent."""
    module = _load_towedarray(monkeypatch)
    platform = _make_platform(module)

    history_before = len(platform.platform_history)
    # _T2 has never been added to host or sensor states
    platform._capture_platform_state(_T2)

    assert len(platform.platform_history) == history_before


# ---------------------------------------------------------------------------
# TowedArrayPlatform – move
# ---------------------------------------------------------------------------


def test_move_advances_host_and_sensors_and_records_state(monkeypatch) -> None:
    """move() should update every movable and append a new PlatformState."""
    module = _load_towedarray(monkeypatch)
    platform = _make_platform(module, num_sensors=2)

    assert len(platform.platform_history) == 1

    platform.move(_T1)

    assert len(platform.platform_history) == 2
    assert platform.platform_history[1].timestamp == _T1
    # Host should have advanced by one velocity step
    host_pos_after = platform.platform_history[1].host.state.state_vector[0, 0]
    np.testing.assert_allclose(host_pos_after, 5.0, atol=1e-10)


# ---------------------------------------------------------------------------
# TowedArrayPlatform – get_platform_state_at
# ---------------------------------------------------------------------------


def test_get_platform_state_at_returns_state_for_matching_timestamp(monkeypatch) -> None:
    """get_platform_state_at should return the correct PlatformState."""
    module = _load_towedarray(monkeypatch)
    platform = _make_platform(module)
    platform.move(_T1)

    result = platform.get_platform_state_at(_T1)

    assert result is not None
    assert result.timestamp == _T1


def test_get_platform_state_at_returns_none_for_unknown_timestamp(monkeypatch) -> None:
    """get_platform_state_at should return None when the timestamp is not in history."""
    module = _load_towedarray(monkeypatch)
    platform = _make_platform(module)

    assert platform.get_platform_state_at(_T2) is None


# ---------------------------------------------------------------------------
# TowedArrayPlatform – get_host_state_at
# ---------------------------------------------------------------------------


def test_get_host_state_at_returns_state_for_matching_timestamp(monkeypatch) -> None:
    """get_host_state_at should return the GroundTruthState for the given time."""
    module = _load_towedarray(monkeypatch)
    platform = _make_platform(module)
    platform.move(_T1)

    result = platform.get_host_state_at(_T1)

    assert result is not None
    assert result.timestamp == _T1


def test_get_host_state_at_returns_none_for_unknown_timestamp(monkeypatch) -> None:
    """get_host_state_at should return None for a time not in the host's history."""
    module = _load_towedarray(monkeypatch)
    platform = _make_platform(module)

    assert platform.get_host_state_at(_T2) is None


# ---------------------------------------------------------------------------
# TowedArrayPlatform – get_sensor_states_at
# ---------------------------------------------------------------------------


def test_get_sensor_states_at_returns_all_sensor_states(monkeypatch) -> None:
    """get_sensor_states_at should return one state per sensor in order."""
    module = _load_towedarray(monkeypatch)
    platform = _make_platform(module, num_sensors=2)
    platform.move(_T1)

    states = platform.get_sensor_states_at(_T1)

    assert states is not None
    assert len(states) == 2
    assert all(s.timestamp == _T1 for s in states)


def test_get_sensor_states_at_returns_none_when_sensor_missing(monkeypatch) -> None:
    """get_sensor_states_at returns None if any sensor lacks a state at that time."""
    module = _load_towedarray(monkeypatch)
    platform = _make_platform(module, num_sensors=2)

    # _T2 has never been added to any sensor
    assert platform.get_sensor_states_at(_T2) is None


# ---------------------------------------------------------------------------
# TowedArrayPlatform – host_path
# ---------------------------------------------------------------------------


def test_host_path_returns_array_of_positions(monkeypatch) -> None:
    """host_path should stack host positions into a 2-D array."""
    module = _load_towedarray(monkeypatch)
    platform = _make_platform(module)
    platform.move(_T1)

    path = platform.host_path

    assert path is not None
    assert path.shape == (2, 3)
    np.testing.assert_allclose(path[0], [0.0, 0.0, 5.0])
    np.testing.assert_allclose(path[1, 0], 5.0)  # x advanced by vx=5


def test_host_path_returns_none_when_no_states(monkeypatch) -> None:
    """host_path should return None when the host has no recorded states."""
    module = _load_towedarray(monkeypatch)
    platform = _make_platform(module)

    platform.movement_controller.states.clear()

    assert platform.host_path is None


# ---------------------------------------------------------------------------
# TowedArrayPlatform – sensor_paths
# ---------------------------------------------------------------------------


def test_sensor_paths_returns_list_per_sensor(monkeypatch) -> None:
    """sensor_paths should return one position array per towed sensor."""
    module = _load_towedarray(monkeypatch)
    platform = _make_platform(module, num_sensors=2)
    platform.move(_T1)

    paths = platform.sensor_paths

    assert len(paths) == 2
    for path in paths:
        assert path.shape[0] == 2  # two timesteps
        assert path.shape[1] == 3  # 3-D position (from flattened (3,1) vectors)


def test_sensor_paths_returns_empty_list_when_no_sensors(monkeypatch) -> None:
    """sensor_paths should return [] when towed_sensors is empty."""
    module = _load_towedarray(monkeypatch)
    platform = _make_platform(module, num_sensors=0)

    assert platform.sensor_paths == []


def test_sensor_paths_returns_empty_array_for_sensor_with_no_states(monkeypatch) -> None:
    """sensor_paths should append a zero-row array for a sensor with cleared states."""
    module = _load_towedarray(monkeypatch)
    platform = _make_platform(module, num_sensors=1)

    platform.towed_sensors[0].states.clear()
    paths = platform.sensor_paths

    assert len(paths) == 1
    assert paths[0].shape == (0, 3)


# ---------------------------------------------------------------------------
# TowedArrayPlatform – __repr__
# ---------------------------------------------------------------------------


def test_repr_contains_configuration_parameters(monkeypatch) -> None:
    """__repr__ should include the key numeric configuration parameters."""
    module = _load_towedarray(monkeypatch)
    platform = _make_platform(module, num_sensors=4, cable_length_m=20.0)

    r = repr(platform)

    assert "num_sensors=4" in r
    assert "cable_length_m=20.0" in r


# ---------------------------------------------------------------------------
# Edge conditions
# ---------------------------------------------------------------------------


# -- _FollowerModel --------------------------------------------------------


def test_follower_model_with_zero_offset_places_follower_at_leader(monkeypatch) -> None:
    """offset=0 should place the follower exactly at the leader's position."""
    module = _load_towedarray(monkeypatch)

    leader = types.SimpleNamespace(position=np.array([[7.0], [2.0], [3.0]]))
    follower_state = types.SimpleNamespace(state_vector=np.array([[1.0], [0.0], [0.0]]))
    model = module._FollowerModel(leader=leader, offset=0.0)

    new_position = model.function(follower_state)

    np.testing.assert_allclose(new_position.flatten(), [7.0, 2.0, 3.0], atol=1e-12)


def test_follower_model_extra_kwargs_are_ignored(monkeypatch) -> None:
    """Extra keyword arguments must be silently ignored (TransitionModel compatibility)."""
    module = _load_towedarray(monkeypatch)

    leader = types.SimpleNamespace(position=np.array([[3.0], [4.0], [0.0]]))
    follower_state = types.SimpleNamespace(state_vector=np.array([[0.0], [0.0], [0.0]]))
    model = module._FollowerModel(leader=leader, offset=1.0)

    result_without = model.function(follower_state)
    result_with = model.function(follower_state, noise=True, time_interval=0.1)

    np.testing.assert_allclose(result_without.flatten(), result_with.flatten(), atol=1e-12)


# -- _TowedArrayFollowerModel ----------------------------------------------


def test_towed_array_follower_model_exact_depth_boundary_gives_zero_horizontal(
    monkeypatch,
) -> None:
    """When offset exactly equals depth_difference, horizontal_offset must be 0."""
    module = _load_towedarray(monkeypatch)

    # depth_difference = |15 - 5| = 10; offset = 10 → offset**2 == depth_diff**2
    leader = types.SimpleNamespace(position=np.array([[5.0], [3.0], [15.0]]))
    follower_state = types.SimpleNamespace(state_vector=np.array([[0.0], [0.0], [5.0]]))
    model = module._TowedArrayFollowerModel(leader=leader, offset=10.0, array_depth_m=5.0)

    new_position = model.function(follower_state)

    # horizontal_offset = sqrt(100 - 100) = 0 → follower placed at leader's XY
    np.testing.assert_allclose(new_position.flatten(), [5.0, 3.0, 5.0], atol=1e-10)


def test_towed_array_follower_model_zero_depth_difference_uses_full_slant_as_horizontal(
    monkeypatch,
) -> None:
    """When leader is already at array depth, full slant distance becomes horizontal offset."""
    module = _load_towedarray(monkeypatch)

    # depth_difference = 0 → horizontal_offset = offset
    leader = types.SimpleNamespace(position=np.array([[10.0], [0.0], [5.0]]))
    follower_state = types.SimpleNamespace(state_vector=np.array([[0.0], [0.0], [5.0]]))
    model = module._TowedArrayFollowerModel(leader=leader, offset=6.0, array_depth_m=5.0)

    new_position = model.function(follower_state)

    # direction is toward leader: (10-0)/10 = [1, 0]; new_xy = [10,0] - 6*[1,0] = [4,0]
    np.testing.assert_allclose(new_position.flatten(), [4.0, 0.0, 5.0], atol=1e-12)


# -- _initialise_sensor_array (exception path) -----------------------------


def test_initialise_sensor_array_raises_valueerror_when_states_empty(monkeypatch) -> None:
    """IndexError on states[0] must be re-raised as ValueError with a clear message."""
    module = _load_towedarray(monkeypatch)
    MovingMovable = module.MovingMovable

    class _NoopModel:
        def function(self, state):
            return state.state_vector.copy()

    controller = MovingMovable(
        states=[], position_mapping=[0, 2, 4], transition_model=_NoopModel()
    )

    with pytest.raises(ValueError, match="initial state"):
        module.TowedArrayPlatform(
            movement_controller=controller,
            position_mapping=[0, 2, 4],
            num_sensors=1,
            cable_length_m=10.0,
            sensor_spacing_m=5.0,
            array_depth_m=5.0,
        )


def test_initialise_sensor_array_exact_boundary_does_not_raise(monkeypatch) -> None:
    """offset**2 == depth_diff**2 must not raise; horizontal sensor placement is 0."""
    module = _load_towedarray(monkeypatch)

    # host z=0, array_depth_m=10, cable=10 → depth_diff==10==cable → boundary equality
    platform = _make_platform(
        module, num_sensors=1, cable_length_m=10.0, array_depth_m=10.0, host_z=0.0
    )

    assert len(platform.towed_sensors) == 1
    sensor_pos = platform.towed_sensors[0].states[-1].state_vector.flatten()
    # horizontal_separation = 0, so sensor is directly below host (same x, y)
    np.testing.assert_allclose(sensor_pos[0], 0.0, atol=1e-10)
    np.testing.assert_allclose(sensor_pos[2], 10.0)


# -- get_position_from_object (state_vector fallback) ----------------------


def test_initialise_sensor_array_accepts_leader_with_state_vector_but_no_states(
    monkeypatch,
) -> None:
    """get_position_from_object should fall back to .state_vector when .states is absent."""
    module = _load_towedarray(monkeypatch)
    GroundTruthState = module.GroundTruthState
    MovingMovable = module.MovingMovable

    # Build a platform, then replace sensor 0 with an object that has only state_vector
    # (no .states list), and re-run _initialise_sensor_array with num_sensors=2 so that
    # sensor 1 uses that patched object as its leader — exercising the fallback branch.
    sv = np.array([[0.0], [5.0], [0.0], [0.0], [5.0], [0.0]], dtype=float)
    init_state = GroundTruthState(state_vector=sv, timestamp=_T0)

    class _NoopModel:
        def function(self, state):
            return state.state_vector.copy()

    controller = MovingMovable(
        states=[init_state], position_mapping=[0, 2, 4], transition_model=_NoopModel()
    )

    # Replace towed_sensors[0] with a namespace that only has state_vector
    fake_sensor_sv = np.array([[-10.0], [0.0], [5.0]])
    fake_sensor = types.SimpleNamespace(state_vector=fake_sensor_sv)

    # Manually call _initialise_sensor_array after replacing movement_controller
    p = module.TowedArrayPlatform(
        movement_controller=controller,
        position_mapping=[0, 2, 4],
        num_sensors=1,
        cable_length_m=10.0,
        sensor_spacing_m=5.0,
        array_depth_m=5.0,
    )

    # Patch sensor 0 to be a state_vector-only object and force re-init of a second sensor
    p.towed_sensors[0] = fake_sensor

    # Now re-run _initialise_sensor_array (only sensor 1 would use state_vector fallback)
    p.num_sensors = 2
    p._initialise_sensor_array()

    # Both sensors should exist; the second one used fake_sensor as its leader
    assert len(p.towed_sensors) == 2


# -- _capture_platform_state (sensor-absent early return) ------------------


def test_capture_platform_state_skips_when_sensor_states_missing_at_timestamp(
    monkeypatch,
) -> None:
    """Early return triggers when the host has a state but sensors do not."""
    module = _load_towedarray(monkeypatch)
    GroundTruthState = module.GroundTruthState
    platform = _make_platform(module, num_sensors=2)

    # Inject a host state at _T2 without moving the sensors
    new_sv = platform.movement_controller.states[-1].state_vector.copy()
    platform.movement_controller.states.append(
        GroundTruthState(state_vector=new_sv, timestamp=_T2)
    )

    history_len = len(platform.platform_history)
    platform._capture_platform_state(_T2)

    # Sensors have no state at _T2 → early return → history unchanged
    assert len(platform.platform_history) == history_len


# -- _capture_platform_state content correctness ---------------------------


def test_capture_platform_state_records_correct_heading(monkeypatch) -> None:
    """Heading stored in HostState should equal arctan2(vy, vx) from the state vector."""
    module = _load_towedarray(monkeypatch)
    # host moving in +x only → heading = arctan2(0, vx) = 0
    platform = _make_platform(module, num_sensors=1, host_velocity_x=5.0)

    heading = platform.platform_history[0].host.heading_rad

    np.testing.assert_allclose(heading, 0.0, atol=1e-12)


def test_capture_platform_state_uses_non_default_reference_sensor(monkeypatch) -> None:
    """ref_state_vector should correspond to the sensor at reference_sensor_idx."""
    module = _load_towedarray(monkeypatch)
    platform = _make_platform(module, num_sensors=3, reference_sensor_idx=2)

    state = platform.platform_history[0]
    expected_ref = platform.towed_sensors[2].states[-1].state_vector

    np.testing.assert_allclose(state.array.ref_state_vector, expected_ref)


def test_array_state_combined_vector_is_hstack_of_sensor_vectors(monkeypatch) -> None:
    """array.state_vector must be the column-wise stack of all sensor state vectors."""
    module = _load_towedarray(monkeypatch)
    platform = _make_platform(module, num_sensors=3)

    state = platform.platform_history[0]
    expected = np.hstack([s.states[-1].state_vector for s in platform.towed_sensors])

    np.testing.assert_allclose(state.array.state_vector, expected)


# -- get_sensor_states_at --------------------------------------------------


def test_get_sensor_states_at_returns_empty_list_when_no_sensors(monkeypatch) -> None:
    """With no towed sensors the function should return [] (not None)."""
    module = _load_towedarray(monkeypatch)
    platform = _make_platform(module, num_sensors=0)

    result = platform.get_sensor_states_at(_T0)

    assert result == []


# -- successive moves ------------------------------------------------------


def test_multiple_moves_accumulate_platform_history(monkeypatch) -> None:
    """Each call to move() should append exactly one entry to platform_history."""
    module = _load_towedarray(monkeypatch)
    platform = _make_platform(module, num_sensors=1)

    assert len(platform.platform_history) == 1  # initial capture

    platform.move(_T1)
    assert len(platform.platform_history) == 2

    platform.move(_T2)
    assert len(platform.platform_history) == 3
    assert platform.platform_history[-1].timestamp == _T2
