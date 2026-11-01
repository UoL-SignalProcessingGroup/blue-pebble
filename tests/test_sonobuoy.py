"""Tests for Sonobuoy and SingleSensorArray."""

from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest
from stonesoup.types.array import StateVector
from stonesoup.types.state import State

from .support import isolate_real_bluepebble_imports, restore_real_bluepebble_imports

START = datetime(2026, 1, 1)


@pytest.fixture
def bp():
    """Import the real classes without leaking caching into other tests."""
    snapshot = isolate_real_bluepebble_imports()
    from bluepebble.arrays import SingleSensorArray
    from bluepebble.platform import Sonobuoy

    yield SimpleNamespace(Sonobuoy=Sonobuoy, SingleSensorArray=SingleSensorArray)
    restore_real_bluepebble_imports(snapshot)


def _buoy(bp, x=100.0, y=-200.0, depth=50.0):
    return bp.Sonobuoy(
        states=State(StateVector([x, y, 0.0]), timestamp=START),
        position_mapping=[0, 1, 2],
        hydrophone_depth_m=depth,
    )


def test_hydrophone_is_a_single_sensor(bp) -> None:
    """The buoy's array reports exactly one sensor."""
    buoy = _buoy(bp)

    assert buoy.arrays == (buoy.hydrophone,)
    assert buoy.hydrophone.array.num_sensors == 1


def test_hydrophone_sits_below_the_buoy(bp) -> None:
    """The hydrophone is at the buoy's x, y and hydrophone_depth_m below it."""
    array = _buoy(bp).hydrophone.array

    np.testing.assert_allclose(array.state_vector[:, 0], [100.0, -200.0, -50.0])
    np.testing.assert_array_equal(
        np.asarray(array.ref_state_vector).ravel(), array.state_vector[:, 0]
    )


@pytest.mark.parametrize("depth", [0.0, 10.0, 300.0])
def test_hydrophone_depth_is_reflected(bp, depth) -> None:
    """Different hydrophone depths change only the hydrophone's z."""
    array = _buoy(bp, depth=depth).hydrophone.array

    assert array.state_vector[2, 0] == -depth


def test_single_sensor_array_is_unbaffled_with_no_axis(bp) -> None:
    """A single unbaffled sensor has unit gain everywhere and no axis."""
    array = _buoy(bp).hydrophone.array

    assert array.axis is None
    np.testing.assert_array_equal(array.sensor_gains([0.0, -1.0, 0.0]), [1.0])


def test_negative_depth_raises(bp) -> None:
    """Depth is given as a positive distance below the buoy."""
    with pytest.raises(ValueError, match="non-negative"):
        _buoy(bp, depth=-50.0)


def test_moving_the_buoy_records_timestamps(bp) -> None:
    """The buoy stays put but records each timestamp it is moved to."""
    buoy = _buoy(bp)
    later = START + timedelta(seconds=30)

    assert buoy.hydrophone.get_platform_state_at(later) is None
    buoy.move(later)

    state = buoy.hydrophone.get_platform_state_at(later)
    np.testing.assert_allclose(state.array.state_vector[:, 0], [100.0, -200.0, -50.0])
    np.testing.assert_allclose(np.asarray(state.position).ravel(), [100.0, -200.0, 0.0])
    assert state.host.heading_rad == 0.0


def test_single_sensor_array_mounts_on_a_plain_stone_soup_platform(bp) -> None:
    """SingleSensorArray works on Stone Soup's FixedPlatform, without a Blue Pebble platform."""
    from stonesoup.platform.base import FixedPlatform

    platform = FixedPlatform(
        states=State(StateVector([1.0, 2.0, 0.0]), timestamp=START), position_mapping=[0, 1, 2]
    )
    hydrophone = bp.SingleSensorArray(mounting_offset=[0.0, 0.0, -20.0])
    platform.add_sensor(hydrophone)

    np.testing.assert_allclose(hydrophone.array.state_vector[:, 0], [1.0, 2.0, -20.0])
    np.testing.assert_allclose(np.asarray(hydrophone.position).ravel(), [1.0, 2.0, -20.0])
