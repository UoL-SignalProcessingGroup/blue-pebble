"""Tests for the sonobuoy platform's array-receiver interface."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest
from stonesoup.types.array import StateVector
from stonesoup.types.state import State

from .support import isolate_real_bluepebble_imports, restore_real_bluepebble_imports


@pytest.fixture
def OmniSonobuoyPlatform():
    """Import OmniSonobuoyPlatform without leaking real caching into other tests.

    See ``isolate_real_bluepebble_imports`` for why this matters.
    """
    snapshot = isolate_real_bluepebble_imports()
    from bluepebble.platform.sonobuoy import OmniSonobuoyPlatform as cls

    yield cls
    restore_real_bluepebble_imports(snapshot)


def _make_buoy(OmniSonobuoyPlatform, hydrophone_depth_m: float = 50.0):
    return OmniSonobuoyPlatform(
        states=[State(StateVector([1000.0, 500.0]), timestamp=datetime.now())],
        position_mapping=[0, 1],
        hydrophone_depth_m=hydrophone_depth_m,
    )


def test_array_has_single_sensor(OmniSonobuoyPlatform) -> None:
    """The sonobuoy's array should report exactly one sensor."""
    buoy = _make_buoy(OmniSonobuoyPlatform)

    assert buoy.array.num_sensors == 1


def test_array_state_vector_matches_position_3d(OmniSonobuoyPlatform) -> None:
    """The array's state_vector should be the hydrophone's 3D position."""
    buoy = _make_buoy(OmniSonobuoyPlatform, hydrophone_depth_m=50.0)

    np.testing.assert_allclose(
        np.asarray(buoy.array.state_vector).flatten(),
        [1000.0, 500.0, 50.0],
    )


def test_array_ref_state_vector_matches_state_vector(OmniSonobuoyPlatform) -> None:
    """With only one sensor, the reference state vector should equal the array state vector."""
    buoy = _make_buoy(OmniSonobuoyPlatform)

    np.testing.assert_allclose(
        np.asarray(buoy.array.ref_state_vector).flatten(),
        np.asarray(buoy.array.state_vector).flatten(),
    )


def test_array_reflects_hydrophone_depth_changes(OmniSonobuoyPlatform) -> None:
    """Different hydrophone depths should be reflected in the array's z coordinate."""
    buoy = _make_buoy(OmniSonobuoyPlatform, hydrophone_depth_m=75.0)

    assert float(np.asarray(buoy.array.state_vector).flatten()[2]) == 75.0
