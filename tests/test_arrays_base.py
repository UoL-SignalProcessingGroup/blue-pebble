"""Tests for ArrayState (axis, baffle gains) and the towed array's reported axis."""

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
