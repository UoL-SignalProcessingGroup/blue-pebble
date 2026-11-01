"""Tests for ArrayState, RigidArray's shared geometry, and the towed array's axis."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest
from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel,
    ConstantVelocity,
)
from stonesoup.types.array import StateVector, StateVectors
from stonesoup.types.groundtruth import GroundTruthState

from .support import isolate_real_bluepebble_imports, restore_real_bluepebble_imports

START = datetime(2026, 1, 1)


@pytest.fixture
def modules():
    """Import the real platform/sensor classes without leaking caching into other tests."""
    snapshot = isolate_real_bluepebble_imports()
    from bluepebble.arrays import ArrayState, TowedArray
    from bluepebble.platform import SonarPlatform

    yield SonarPlatform, ArrayState, TowedArray
    restore_real_bluepebble_imports(snapshot)


def test_towed_axis_runs_forward_along_the_line(modules) -> None:
    """A straight line towed east has its axis pointing east (aft sensor -> forward)."""
    SonarPlatform, _, TowedArray = modules
    model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.0)] * 3)
    platform = SonarPlatform(
        states=[GroundTruthState(StateVector([0.0, 3.0, 0.0, 0.0, 0.0, 0.0]), timestamp=START)],
        position_mapping=[0, 2, 4],
        transition_models=[model],
        transition_times=[timedelta(hours=1)],
    )
    towed = TowedArray(
        num_sensors=8,
        cable_length_m=100.0,
        sensor_spacing_m=1.0,
        array_depth_m=-50.0,
        mounting_offset=StateVector([-30.0, 0.0, 0.0]),
    )
    platform.add_sensor(towed)
    later = START + timedelta(seconds=10)
    platform.move(later)

    array = towed.get_platform_state_at(later).array

    np.testing.assert_allclose(np.asarray(array.axis).ravel(), [1.0, 0.0, 0.0], atol=1e-9)
    assert array.axis_azimuth_rad == pytest.approx(0.0, abs=1e-9)


def test_axis_azimuth_without_axis_raises(modules) -> None:
    """An array with no axis (e.g. a single hydrophone) has no azimuth to report."""
    _, ArrayState, _ = modules
    array = ArrayState(
        num_sensors=1,
        state_vector=StateVectors(np.zeros((3, 1))),
        ref_state_vector=StateVector([0.0, 0.0, 0.0]),
    )

    with pytest.raises(ValueError, match="no axis"):
        _ = array.axis_azimuth_rad


def test_sensor_gains_attenuate_sensors_facing_away(modules) -> None:
    """Sensors whose normal points away from a direction get the baffle gain; others get 1."""
    _, ArrayState, _ = modules
    array = ArrayState(
        num_sensors=2,
        state_vector=np.zeros((3, 2)),
        ref_state_vector=StateVector([0.0, 0.0, 0.0]),
        sensor_normals=StateVectors([[0.0, 0.0], [1.0, -1.0], [0.0, 0.0]]),
        baffle_attenuation_db=20.0,
    )

    np.testing.assert_allclose(array.sensor_gains([0.0, 1.0, 0.0]), [1.0, 0.1])
    assert array.transmit_gain([0.0, 1.0, 0.0]) == pytest.approx(0.55)


def test_sensor_gains_are_one_without_normals(modules) -> None:
    """Omnidirectional, unbaffled sensors have unit gain in every direction."""
    _, ArrayState, _ = modules
    array = ArrayState(
        num_sensors=3,
        state_vector=np.zeros((3, 3)),
        ref_state_vector=StateVector([0.0, 0.0, 0.0]),
        baffle_attenuation_db=20.0,
    )

    np.testing.assert_array_equal(array.sensor_gains([1.0, 0.0, 0.0]), np.ones(3))


@pytest.fixture
def rigid():
    """Import the real classes needed to test RigidArray's shared geometry."""
    snapshot = isolate_real_bluepebble_imports()
    from bluepebble.arrays import RigidArray, SingleSensorArray
    from bluepebble.platform import SonarPlatform

    yield SonarPlatform, RigidArray, SingleSensorArray
    restore_real_bluepebble_imports(snapshot)


def test_rigid_array_reference_matches_stone_soup_mounting(rigid) -> None:
    """The reference point equals Stone Soup's own mounted position, while moving and climbing.

    This pins RigidArray's use of Stone Soup's private ``_get_rotation_matrix`` to the
    rotation Stone Soup applies to ``mounting_offset``.
    """
    from stonesoup.models.transition.linear import KnownTurnRate

    SonarPlatform, _, SingleSensorArray = rigid
    turn = CombinedLinearGaussianTransitionModel(
        [KnownTurnRate([0.0, 0.0], np.radians(5.0)), ConstantVelocity(0.0)]
    )
    platform = SonarPlatform(
        states=[GroundTruthState(StateVector([0.0, 3.0, 0.0, 4.0, -5.0, 0.5]), timestamp=START)],
        position_mapping=[0, 2, 4],
        transition_models=[turn],
        transition_times=[timedelta(hours=1)],
    )
    array = SingleSensorArray(mounting_offset=StateVector([-12.0, 3.0, -2.0]))
    platform.add_sensor(array)

    for k in range(1, 20):
        platform.move(START + timedelta(seconds=k))
        np.testing.assert_allclose(
            np.asarray(array.array.ref_state_vector), np.asarray(array.position), atol=1e-12
        )


def test_rigid_array_subclass_only_supplies_a_layout(rigid) -> None:
    """A new rigid array needs only its local layout; rotation_offset turns it."""
    SonarPlatform, RigidArray, _ = rigid

    class _Line(RigidArray):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._local_offsets = np.array([[0.0, 1.0, 2.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    platform = SonarPlatform(
        states=[GroundTruthState(StateVector([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]), timestamp=START)],
        position_mapping=[0, 2, 4],
        transition_models=[CombinedLinearGaussianTransitionModel([ConstantVelocity(0.0)] * 3)],
        transition_times=[timedelta(hours=1)],
    )
    line = _Line(rotation_offset=StateVector([0.0, 0.0, np.pi / 2]))
    platform.add_sensor(line)

    # Heading north (+y), then yawed 90 degrees to port (-x).
    np.testing.assert_allclose(line.array.state_vector[0], [0.0, -1.0, -2.0], atol=1e-12)
    np.testing.assert_allclose(line.array.state_vector[1], 0.0, atol=1e-12)
    assert line.num_sensors == 3
    assert line.array.sensor_normals is None and line.array.axis is None


def test_fixed_platform_heading_is_its_orientation_yaw(rigid) -> None:
    """A fixed platform has no velocity mapping; its heading is its orientation's yaw."""
    from stonesoup.platform.base import FixedPlatform
    from stonesoup.types.state import State

    _, _, SingleSensorArray = rigid
    platform = FixedPlatform(
        states=State(StateVector([0.0, 0.0, 0.0]), timestamp=START),
        position_mapping=[0, 1, 2],
        orientation=StateVector([0.0, 0.0, np.pi]),
    )
    array = SingleSensorArray()
    platform.add_sensor(array)

    assert array._velocity_mapping is None
    assert array.get_platform_state_at(START).host.heading_rad == -np.pi
