"""Tests for BowArray: dome layout, mounting, baffling and geometry over time."""

from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest
from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel,
    ConstantVelocity,
    KnownTurnRate,
)
from stonesoup.types.array import StateVector
from stonesoup.types.groundtruth import GroundTruthState

from .support import isolate_real_bluepebble_imports, restore_real_bluepebble_imports

START = datetime(2026, 1, 1)


@pytest.fixture
def bp():
    """Import the real platform and array classes without leaking caching into other tests."""
    snapshot = isolate_real_bluepebble_imports()
    from bluepebble.arrays import BowArray
    from bluepebble.platform import SonarPlatform

    yield SimpleNamespace(SonarPlatform=SonarPlatform, BowArray=BowArray)
    restore_real_bluepebble_imports(snapshot)


def _make_platform(bp, vx=0.0, vy=5.0, transition_models=None, transition_times=None):
    return bp.SonarPlatform(
        states=[GroundTruthState(StateVector([0.0, vx, 0.0, vy, -5.0, 0.0]), timestamp=START)],
        position_mapping=[0, 2, 4],
        transition_models=transition_models
        or [CombinedLinearGaussianTransitionModel([ConstantVelocity(0.0)] * 3)],
        transition_times=transition_times or [timedelta(hours=1)],
    )


def _make_bow(bp, platform=None, **overrides):
    kwargs = dict(
        dome_radius_m=2.0,
        azimuth_extent_rad=np.radians(60.0),
        elevation_extent_rad=np.radians(10.0),
        sensor_spacing_m=0.3,
        sensor_size_m=0.1,
        freq_max_hz=2000.0,
    )
    kwargs.update(overrides)
    bow = bp.BowArray(**kwargs)
    (platform or _make_platform(bp)).add_sensor(bow)
    return bow


def test_normals_face_along_heading(bp) -> None:
    """Every dome sensor faces the platform's heading (north here)."""
    bow = _make_bow(bp)

    normals = np.asarray(bow.array.sensor_normals)
    expected = np.tile([[0.0], [1.0], [0.0]], (1, bow.num_sensors))
    np.testing.assert_allclose(normals, expected, atol=1e-12)


def test_baffle_level_reaches_array_state(bp) -> None:
    """The array's baffle level is reported on every array state."""
    bow = _make_bow(bp, baffle_attenuation_db=25.0)

    assert bow.array.baffle_attenuation_db == 25.0
    assert bow.get_platform_state_at(START).array.baffle_attenuation_db == 25.0


def test_reference_sits_at_dome_depth(bp) -> None:
    """mounting_offset lowers the reference point (used for propagation) with the dome."""
    bow = _make_bow(bp, mounting_offset=StateVector([0.0, 0.0, -6.0]))

    np.testing.assert_allclose(np.asarray(bow.array.ref_state_vector).ravel(), [0.0, 0.0, -11.0])
    np.testing.assert_allclose(bow.array.state_vector[2].mean(), -11.0, atol=1e-9)
    np.testing.assert_allclose(np.asarray(bow.position).ravel(), [0.0, 0.0, -11.0])


def test_axis_follows_heading(bp) -> None:
    """The array axis points along the platform's heading (north here)."""
    bow = _make_bow(bp)

    np.testing.assert_allclose(np.asarray(bow.array.axis).ravel(), [0.0, 1.0, 0.0], atol=1e-12)
    assert bow.get_platform_state_at(START).array.axis_azimuth_rad == pytest.approx(np.pi / 2)


def test_dome_sensors_lie_on_the_sphere_within_the_patch(bp) -> None:
    """Sensors sit dome_radius_m from the dome centre, within the azimuth/elevation extents."""
    bow = _make_bow(bp, mounting_offset=StateVector([0.0, 0.0, -3.0]))

    offsets = bow.array.state_vector - np.asarray(bow.array.ref_state_vector)
    np.testing.assert_allclose(np.linalg.norm(offsets, axis=0), 2.0)
    bearings = np.arctan2(offsets[0], offsets[1])  # off the +y heading
    assert np.abs(bearings).max() <= np.radians(30.0) + 1e-9


@pytest.mark.parametrize(
    ("spacing", "size", "match"),
    [(0.1, 0.1, "do not touch"), (0.5, 0.1, "grating-lobe")],
)
def test_invalid_sensor_spacing_raises(bp, spacing, size, match) -> None:
    """Touching sensors, or spacing beyond lambda/2 at freq_max_hz, are rejected."""
    with pytest.raises(ValueError, match=match):
        _make_bow(bp, sensor_spacing_m=spacing, sensor_size_m=size)


def test_history_follows_the_platform_through_a_turn(bp) -> None:
    """Each timestamp's geometry uses the platform's state then, not its latest state."""
    turn = CombinedLinearGaussianTransitionModel(
        [KnownTurnRate([0.0, 0.0], np.radians(9.0)), ConstantVelocity(0.0)]
    )
    platform = _make_platform(
        bp, vx=5.0, vy=0.0, transition_models=[turn], transition_times=[timedelta(hours=1)]
    )
    bow = _make_bow(bp, platform=platform)
    for k in range(1, 11):
        platform.move(START + timedelta(seconds=k))

    for k in (0, 5, 10):
        state = bow.get_platform_state_at(START + timedelta(seconds=k))
        heading = state.host.heading_rad
        assert state.array.axis_azimuth_rad == pytest.approx(heading)
        np.testing.assert_allclose(
            np.asarray(state.array.ref_state_vector), np.asarray(state.position)
        )
    assert bow.get_platform_state_at(START + timedelta(seconds=10)).host.heading_rad == (
        pytest.approx(np.radians(90.0))
    )


def test_unreached_timestamp_returns_none(bp) -> None:
    """A timestamp the platform never reached has no geometry."""
    bow = _make_bow(bp)

    assert bow.get_platform_state_at(START + timedelta(seconds=1)) is None
