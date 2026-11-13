"""Unit tests for SonarPlatform, TowedArray and the towed-array follower models."""

from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest
from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel,
    ConstantVelocity,
)
from stonesoup.types.array import StateVector
from stonesoup.types.groundtruth import GroundTruthState
from stonesoup.types.state import State

from .support import isolate_real_bluepebble_imports, restore_real_bluepebble_imports

_T0 = datetime(2024, 1, 1)
_T1 = _T0 + timedelta(seconds=1)
_T2 = _T0 + timedelta(seconds=2)


@pytest.fixture
def bp():
    """Import the real platform and array modules without leaking caching into other tests."""
    snapshot = isolate_real_bluepebble_imports()
    from bluepebble.arrays import base, towedarray
    from bluepebble.platform import SonarPlatform

    yield SimpleNamespace(SonarPlatform=SonarPlatform, base=base, towedarray=towedarray)
    restore_real_bluepebble_imports(snapshot)


def _make_platform(bp, velocity_x=5.0, z=5.0, velocity_mapping=(1, 3, 5), **kwargs):
    """Return a SonarPlatform moving at a constant velocity along +x."""
    state = GroundTruthState([0.0, velocity_x, 0.0, 0.0, z, 0.0], timestamp=_T0)
    platform_kwargs = dict(
        states=state,
        position_mapping=[0, 2, 4],
        transition_models=[CombinedLinearGaussianTransitionModel([ConstantVelocity(0.0)] * 3)],
        transition_times=[timedelta(seconds=60)],
    )
    if velocity_mapping is not None:
        platform_kwargs["velocity_mapping"] = list(velocity_mapping)
    platform_kwargs.update(kwargs)
    return bp.SonarPlatform(**platform_kwargs)


def _make_mounted(
    bp,
    num_sensors=2,
    cable_length_m=10.0,
    sensor_spacing_m=5.0,
    array_depth_m=5.0,
    platform_kwargs=None,
    **array_kwargs,
):
    """Return a ``(platform, array)`` pair with a TowedArray mounted on a SonarPlatform."""
    platform = _make_platform(bp, **(platform_kwargs or {}))
    array = bp.towedarray.TowedArray(
        num_sensors=num_sensors,
        cable_length_m=cable_length_m,
        sensor_spacing_m=sensor_spacing_m,
        array_depth_m=array_depth_m,
        **array_kwargs,
    )
    platform.add_sensor(array)
    return platform, array


# ---------------------------------------------------------------------------
# Follower models
# ---------------------------------------------------------------------------


def test_follower_model_uses_default_direction_when_positions_overlap(bp) -> None:
    """Follower should fall back to the -x direction when leader and follower overlap."""
    leader = SimpleNamespace(position=StateVector([10.0, 5.0, 2.0]))
    model = bp.towedarray._FollowerModel(leader=leader, offset=4.0)

    new_position = model.function(State(StateVector([10.0, 5.0, 2.0])))

    np.testing.assert_allclose(new_position.ravel(), [14.0, 5.0, 2.0])


def test_follower_model_moves_toward_leader_with_configured_offset(bp) -> None:
    """Follower should end up exactly ``offset`` behind the leader."""
    leader = SimpleNamespace(position=StateVector([3.0, 4.0, 0.0]))
    model = bp.towedarray._FollowerModel(leader=leader, offset=1.0)

    new_position = model.function(State(StateVector([0.0, 0.0, 0.0])))

    np.testing.assert_allclose(new_position.ravel(), [2.4, 3.2, 0.0], atol=1e-12)


def test_follower_model_with_zero_offset_places_follower_at_leader(bp) -> None:
    """A zero offset puts the follower on the leader."""
    leader = SimpleNamespace(position=StateVector([7.0, -2.0, 3.0]))
    model = bp.towedarray._FollowerModel(leader=leader, offset=0.0)

    new_position = model.function(State(StateVector([0.0, 0.0, 0.0])))

    np.testing.assert_allclose(new_position.ravel(), [7.0, -2.0, 3.0])


def test_follower_model_extra_kwargs_are_ignored(bp) -> None:
    """Transition-model keyword arguments (e.g. time_interval) should not change the result."""
    leader = SimpleNamespace(position=StateVector([3.0, 4.0, 0.0]))
    model = bp.towedarray._FollowerModel(leader=leader, offset=1.0)
    state = State(StateVector([0.0, 0.0, 0.0]))

    np.testing.assert_array_equal(
        model.function(state, time_interval=timedelta(seconds=1), noise=False),
        model.function(state),
    )


def test_towed_array_follower_model_handles_zero_horizontal_distance(bp) -> None:
    """If the slant offset is shorter than the depth gap, the horizontal offset is zero."""
    leader = SimpleNamespace(position=StateVector([5.0, 1.0, 120.0]))
    model = bp.towedarray._TowedArrayFollowerModel(leader=leader, offset=10.0, array_depth_m=60.0)

    new_position = model.function(State(StateVector([5.0, 1.0, 80.0])))

    np.testing.assert_allclose(new_position.ravel(), [5.0, 1.0, 60.0])


def test_towed_array_follower_model_keeps_depth_and_applies_horizontal_offset(bp) -> None:
    """Follower should keep the array depth and take its horizontal offset from the slant."""
    leader = SimpleNamespace(position=StateVector([10.0, 0.0, 100.0]))
    model = bp.towedarray._TowedArrayFollowerModel(leader=leader, offset=13.0, array_depth_m=95.0)

    new_position = model.function(State(StateVector([4.0, 0.0, 100.0])))

    # horizontal offset = sqrt(13^2 - 5^2) = 12
    np.testing.assert_allclose(new_position.ravel(), [-2.0, 0.0, 95.0], atol=1e-12)


def test_towed_array_follower_model_uses_default_xy_direction_when_positions_overlap(bp) -> None:
    """When leader and follower share an XY position, fall back to the -x direction."""
    leader = SimpleNamespace(position=StateVector([5.0, 3.0, 100.0]))
    model = bp.towedarray._TowedArrayFollowerModel(leader=leader, offset=30.0, array_depth_m=90.0)

    new_position = model.function(State(StateVector([5.0, 3.0, 90.0])))

    np.testing.assert_allclose(new_position.ravel(), [5.0 + np.sqrt(800), 3.0, 90.0])


def test_towed_array_follower_model_exact_depth_boundary_gives_zero_horizontal(bp) -> None:
    """A slant offset equal to the depth gap leaves no horizontal offset."""
    leader = SimpleNamespace(position=StateVector([0.0, 0.0, 20.0]))
    model = bp.towedarray._TowedArrayFollowerModel(leader=leader, offset=10.0, array_depth_m=10.0)

    new_position = model.function(State(StateVector([-5.0, 0.0, 10.0])))

    np.testing.assert_allclose(new_position.ravel(), [0.0, 0.0, 10.0], atol=1e-12)


def test_towed_array_follower_model_zero_depth_difference_uses_full_slant_as_horizontal(
    bp,
) -> None:
    """At the leader's depth, the whole slant offset is horizontal."""
    leader = SimpleNamespace(position=StateVector([10.0, 0.0, 50.0]))
    model = bp.towedarray._TowedArrayFollowerModel(leader=leader, offset=4.0, array_depth_m=50.0)

    new_position = model.function(State(StateVector([0.0, 0.0, 50.0])))

    np.testing.assert_allclose(new_position.ravel(), [6.0, 0.0, 50.0], atol=1e-12)


def test_platform_state_position_returns_platform_xyz(bp) -> None:
    """PlatformState.position should take x/y/z from the platform's state vector."""
    platform_state = State(StateVector([1.0, 9.0, 2.0, 8.0, 3.0, 7.0]))
    state = bp.towedarray.PlatformState(
        timestamp=_T0,
        host=bp.towedarray.HostState(state=platform_state, heading_rad=0.0),
        array=bp.base.ArrayState(
            num_sensors=1,
            state_vector=np.zeros((3, 1)),
            ref_state_vector=StateVector([0.0, 0.0, 0.0]),
        ),
    )

    np.testing.assert_allclose(state.position.ravel(), [1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# SonarPlatform
# ---------------------------------------------------------------------------


def test_platform_depth_reads_z_from_position(bp) -> None:
    """SonarPlatform.depth should be the z-component of position."""
    platform = _make_platform(bp, z=150.0)

    assert platform.depth == 150.0


def test_platform_move_cascades_to_mounted_arrays(bp) -> None:
    """move() should advance the platform, then call every mounted array's hook."""
    calls = []

    class _RecordingArray(bp.base.SensorArray):
        def _on_platform_moved(self, timestamp, **kwargs):
            calls.append((timestamp, self.position[0, 0]))

    platform = _make_platform(bp)
    platform.add_sensor(_RecordingArray())
    platform.move(_T1)

    assert calls == [(_T1, 5.0)]


def test_platform_initialises_arrays_passed_to_its_constructor(bp) -> None:
    """Arrays given through ``sensors=`` should be mounted exactly as add_sensor mounts them."""
    array = bp.towedarray.TowedArray(
        num_sensors=3, cable_length_m=10.0, sensor_spacing_m=5.0, array_depth_m=5.0
    )
    platform = _make_platform(bp, sensors=[array])

    assert platform.arrays == (array,)
    assert array.movement_controller is platform.movement_controller
    assert len(array.towed_sensors) == 3
    assert [state.timestamp for state in array.platform_history] == [_T0]


def test_platform_arrays_lists_only_sensor_arrays(bp) -> None:
    """Stone Soup sensors mounted alongside arrays are not listed as arrays."""
    from stonesoup.sensor.passive import PassiveElevationBearing

    platform, array = _make_mounted(bp)
    platform.add_sensor(
        PassiveElevationBearing(ndim_state=6, mapping=[0, 2, 4], noise_covar=np.eye(2))
    )

    assert len(platform.sensors) == 2
    assert platform.arrays == (array,)


# ---------------------------------------------------------------------------
# TowedArray – mounting and initial layout
# ---------------------------------------------------------------------------


def test_unmounted_array_raises_clear_error(bp) -> None:
    """Querying an array before it is mounted should say how to mount it."""
    array = bp.towedarray.TowedArray(
        num_sensors=2, cable_length_m=10.0, sensor_spacing_m=5.0, array_depth_m=5.0
    )

    assert array.towed_sensors == [] and array.platform_history == []
    with pytest.raises(ValueError, match="add_sensor"):
        _ = array.array


def test_mounting_creates_sensors_and_records_initial_state(bp) -> None:
    """Mounting should lay out every sensor and record the platform's current state."""
    _, array = _make_mounted(bp, num_sensors=3)

    assert len(array.towed_sensors) == 3
    assert [state.timestamp for state in array.platform_history] == [_T0]


def test_sensors_trail_astern_at_cable_length_then_spacing(bp) -> None:
    """At the platform's depth, sensors sit cable_length_m, then sensor_spacing_m, astern."""
    _, array = _make_mounted(bp, num_sensors=3, cable_length_m=10.0, sensor_spacing_m=5.0)

    np.testing.assert_allclose(array.array.state_vector[0], [-10.0, -15.0, -20.0])
    np.testing.assert_allclose(array.array.state_vector[2], [5.0, 5.0, 5.0])


def test_mounting_offset_moves_the_tow_point(bp) -> None:
    """Stone Soup's mounting_offset sets the tow point relative to the platform reference."""
    platform, array = _make_mounted(
        bp, num_sensors=1, cable_length_m=10.0, mounting_offset=StateVector([-10.0, 0.0, 0.0])
    )
    np.testing.assert_allclose(array.position.ravel(), [-10.0, 0.0, 5.0])
    np.testing.assert_allclose(array.array.state_vector[:, 0], [-20.0, 0.0, 5.0])

    platform.move(_T1)

    np.testing.assert_allclose(array.array.state_vector[:, 0], [-15.0, 0.0, 5.0])


def test_velocity_mapping_comes_from_the_platform(bp) -> None:
    """The heading should use the platform movement controller's velocity mapping."""
    platform, array = _make_mounted(bp, platform_kwargs={"velocity_mapping": None})

    assert list(platform.movement_controller.velocity_mapping) == [1, 3, 5]
    assert array.platform_history[0].host.heading_rad == 0.0


def test_initialise_raises_when_segment_too_short_for_depth(bp) -> None:
    """A ValueError is raised when the cable cannot span the depth difference."""
    with pytest.raises(ValueError, match="too short"):
        _make_mounted(bp, cable_length_m=50.0, array_depth_m=100.0, platform_kwargs={"z": 0.0})


def test_initialise_exact_boundary_does_not_raise(bp) -> None:
    """A cable exactly as long as the depth difference hangs straight down."""
    _, array = _make_mounted(
        bp, num_sensors=1, cable_length_m=100.0, array_depth_m=100.0, platform_kwargs={"z": 0.0}
    )

    np.testing.assert_allclose(array.array.state_vector[:, 0], [0.0, 0.0, 100.0])


def test_initialise_uses_zero_velocity_fallback_heading(bp) -> None:
    """A stationary platform lays the array out along -x."""
    _, array = _make_mounted(bp, num_sensors=1, platform_kwargs={"velocity_x": 0.0})

    np.testing.assert_allclose(array.towed_sensors[0].state.state_vector[0, 0], -10.0)


# ---------------------------------------------------------------------------
# TowedArray – state capture and lookups
# ---------------------------------------------------------------------------


def test_capture_platform_state_skips_missing_platform_state(bp) -> None:
    """_capture_platform_state should do nothing for a timestamp the platform never reached."""
    _, array = _make_mounted(bp)

    array._capture_platform_state(_T2)

    assert len(array.platform_history) == 1


def test_capture_platform_state_skips_when_sensor_states_missing(bp) -> None:
    """If the platform reached a timestamp but the sensors did not, nothing is recorded."""
    platform, array = _make_mounted(bp)
    platform.movement_controller.move(_T1)  # bypass the cascade, so the sensors stay at _T0

    array._capture_platform_state(_T1)

    assert [state.timestamp for state in array.platform_history] == [_T0]


def test_move_advances_platform_and_sensors_and_records_state(bp) -> None:
    """platform.move() should update every sensor and append a new PlatformState."""
    platform, array = _make_mounted(bp, num_sensors=2)

    platform.move(_T1)

    assert [state.timestamp for state in array.platform_history] == [_T0, _T1]
    assert array.platform_history[1].host.state.state_vector[0, 0] == 5.0
    assert all(sensor.state.timestamp == _T1 for sensor in array.towed_sensors)


def test_get_platform_state_at_returns_matching_state_or_none(bp) -> None:
    """get_platform_state_at should find recorded timestamps and return None otherwise."""
    platform, array = _make_mounted(bp)
    platform.move(_T1)

    assert array.get_platform_state_at(_T1).timestamp == _T1
    assert array.get_platform_state_at(_T2) is None


def test_heading_due_minus_x_is_minus_pi(bp) -> None:
    """Heading follows the [-pi, pi) convention: due -x is -pi, not arctan2's +pi."""
    platform, array = _make_mounted(bp, platform_kwargs={"velocity_x": -5.0})
    platform.move(_T1)

    assert array.get_platform_state_at(_T1).host.heading_rad == -np.pi


def test_captured_states_match_every_sensor_over_several_moves(bp) -> None:
    """Each captured PlatformState holds the platform and sensor states at its own timestamp."""
    platform, array = _make_mounted(bp, num_sensors=3)
    timestamps = [_T0 + timedelta(seconds=i) for i in range(6)]
    for timestamp in timestamps[1:]:
        platform.move(timestamp)

    for timestamp in timestamps:
        captured = array.get_platform_state_at(timestamp)
        sensor_states = array.get_sensor_states_at(timestamp)
        assert captured is not None and sensor_states is not None
        assert captured.host.state is array.get_host_state_at(timestamp)
        np.testing.assert_array_equal(
            captured.array.state_vector, np.hstack([s.state_vector for s in sensor_states])
        )


def test_get_platform_state_at_returns_the_first_capture_for_a_repeated_timestamp(bp) -> None:
    """Moving to the same time twice records two states; lookups return the first."""
    platform, array = _make_mounted(bp)
    platform.move(_T1)
    platform.move(_T1)

    assert [state.timestamp for state in array.platform_history] == [_T0, _T1, _T1]
    assert array.get_platform_state_at(_T1) is array.platform_history[1]


def test_get_platform_state_at_finds_states_added_to_the_history_directly(bp) -> None:
    """Entries appended to platform_history outside move() are still found."""
    platform, array = _make_mounted(bp)
    platform.move(_T1)
    extra = bp.towedarray.PlatformState(
        timestamp=_T2, host=array.platform_history[1].host, array=array.platform_history[1].array
    )
    array.platform_history.append(extra)

    assert array.get_platform_state_at(_T2) is extra


def test_get_host_state_at_returns_matching_state_or_none(bp) -> None:
    """get_host_state_at should return the platform's own state, or None if absent."""
    platform, array = _make_mounted(bp)
    platform.move(_T1)

    assert array.get_host_state_at(_T1) is platform.states[-1]
    assert array.get_host_state_at(_T2) is None


def test_get_sensor_states_at_returns_all_states_or_none(bp) -> None:
    """get_sensor_states_at returns one state per sensor, or None if any is missing."""
    platform, array = _make_mounted(bp, num_sensors=2)
    platform.move(_T1)

    states = array.get_sensor_states_at(_T1)
    assert [state.timestamp for state in states] == [_T1, _T1]
    assert array.get_sensor_states_at(_T2) is None


def test_reference_sensor_and_axis(bp) -> None:
    """ref_state_vector is the chosen sensor; the axis points forward along the line."""
    _, array = _make_mounted(bp, num_sensors=3, reference_sensor_idx=2)

    state = array.array
    reference = array.towed_sensors[2].state.state_vector
    np.testing.assert_array_equal(state.ref_state_vector, reference)
    np.testing.assert_allclose(state.axis.ravel(), [1.0, 0.0, 0.0])
    assert state.num_sensors == 3


def test_host_and_sensor_paths(bp) -> None:
    """host_path and sensor_paths should hold one row per recorded timestamp."""
    platform, array = _make_mounted(bp, num_sensors=2)
    platform.move(_T1)
    platform.move(_T2)

    np.testing.assert_allclose(array.host_path[:, 0], [0.0, 5.0, 10.0])
    assert [path.shape for path in array.sensor_paths] == [(3, 3), (3, 3)]


def test_array_with_no_sensors_has_empty_lookups(bp) -> None:
    """An array with no sensors records nothing and returns empty sensor lookups."""
    _, array = _make_mounted(bp, num_sensors=0)

    assert array.platform_history == []
    assert array.get_sensor_states_at(_T0) == []
    assert array.sensor_paths == []


def test_sensor_paths_returns_empty_array_for_sensor_with_no_states(bp) -> None:
    """A sensor whose history has been cleared gives an empty (0, 3) path."""
    _, array = _make_mounted(bp, num_sensors=1)
    array.towed_sensors[0].states.clear()

    assert array.sensor_paths[0].shape == (0, 3)


def test_repr_contains_configuration_parameters(bp) -> None:
    """The repr should show the array's configuration."""
    _, array = _make_mounted(bp, num_sensors=4, cable_length_m=20.0)

    assert repr(array) == (
        "TowedArray(num_sensors=4, cable_length_m=20.0, sensor_spacing_m=5.0, array_depth_m=5.0)"
    )
