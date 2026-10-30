"""Tests for the hull-mounted BowArraySensor's baffle reporting."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest
from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel,
    ConstantVelocity,
)
from stonesoup.types.array import StateVector
from stonesoup.types.groundtruth import GroundTruthState

from .support import isolate_real_bluepebble_imports, restore_real_bluepebble_imports

START = datetime(2026, 1, 1)


@pytest.fixture
def modules():
    """Import the real platform/sensor classes without leaking caching into other tests."""
    snapshot = isolate_real_bluepebble_imports()
    from bluepebble.platform.base import HostPlatform
    from bluepebble.sensors.bow_array import BowArraySensor

    yield HostPlatform, BowArraySensor
    restore_real_bluepebble_imports(snapshot)


def _make_bow(HostPlatform, BowArraySensor, **overrides):
    model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.0)] * 3)
    host = HostPlatform(
        states=[GroundTruthState(StateVector([0.0, 0.0, 0.0, 5.0, -5.0, 0.0]), timestamp=START)],
        position_mapping=[0, 2, 4],
        transition_models=[model],
        transition_times=[timedelta(hours=1)],
    )
    kwargs = dict(
        dome_radius_m=2.0,
        azimuth_extent_rad=np.radians(60.0),
        elevation_extent_rad=np.radians(10.0),
        element_spacing_m=0.3,
        element_size_m=0.1,
        freq_max_hz=2000.0,
    )
    kwargs.update(overrides)
    return BowArraySensor(host=host, **kwargs)


def test_normals_face_along_heading(modules) -> None:
    """Every dome element faces the host's heading (north here)."""
    bow = _make_bow(*modules)

    normals = np.asarray(bow.array.element_normals)
    expected = np.tile([[0.0], [1.0], [0.0]], (1, normals.shape[1]))
    np.testing.assert_allclose(normals, expected, atol=1e-12)


def test_baffle_level_reaches_array_state(modules) -> None:
    """The sensor's baffle level is reported on every array snapshot."""
    bow = _make_bow(*modules, baffle_attenuation_db=25.0)

    assert bow.array.baffle_attenuation_db == 25.0
    assert bow.get_platform_state_at(START).array.baffle_attenuation_db == 25.0


def test_reference_sits_at_dome_depth(modules) -> None:
    """The reference (used for propagation depth) is lowered with the dome, below the host."""
    bow = _make_bow(*modules, array_depth_offset_m=6.0)

    ref = np.asarray(bow.array.ref_state_vector).flatten()
    np.testing.assert_allclose(ref, [0.0, 0.0, -11.0], atol=1e-12)
    np.testing.assert_allclose(np.asarray(bow.array.state_vector)[2].mean(), -11.0, atol=1e-9)


def test_axis_follows_heading(modules) -> None:
    """The array axis points along the host's heading (north here)."""
    bow = _make_bow(*modules)

    np.testing.assert_allclose(np.asarray(bow.array.axis).ravel(), [0.0, 1.0, 0.0], atol=1e-12)
    assert bow.get_platform_state_at(START).array.axis_azimuth_rad == pytest.approx(np.pi / 2)
