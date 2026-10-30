"""Tests for the towed array's reported axis and ArrayState.axis_azimuth_rad."""

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
    from bluepebble.platform.base import HostPlatform
    from bluepebble.sensors.base import ArrayState
    from bluepebble.sensors.towed_array import TowedArraySensor

    yield HostPlatform, ArrayState, TowedArraySensor
    restore_real_bluepebble_imports(snapshot)


def test_towed_axis_runs_forward_along_the_line(modules) -> None:
    """A straight line towed east has its axis pointing east (aft element -> forward)."""
    HostPlatform, _, TowedArraySensor = modules
    model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.0)] * 3)
    host = HostPlatform(
        states=[GroundTruthState(StateVector([0.0, 3.0, 0.0, 0.0, 0.0, 0.0]), timestamp=START)],
        position_mapping=[0, 2, 4],
        transition_models=[model],
        transition_times=[timedelta(hours=1)],
    )
    towed = TowedArraySensor(
        host=host,
        num_sensors=8,
        cable_length_m=100.0,
        sensor_spacing_m=1.0,
        array_depth_m=-50.0,
        towed_array_bow_offset_m=30.0,
    )
    later = START + timedelta(seconds=10)
    host.move(later)

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
