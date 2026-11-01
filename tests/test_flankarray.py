"""Tests for FlankArray: one panel per hull side, its layout, mounting checks and geometry."""

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

from .support import isolate_real_bluepebble_imports, restore_real_bluepebble_imports

START = datetime(2026, 1, 1)


@pytest.fixture
def bp():
    """Import the real platform and array classes without leaking caching into other tests."""
    snapshot = isolate_real_bluepebble_imports()
    from bluepebble.arrays import FlankArray, TowedArray
    from bluepebble.platform import SonarPlatform

    yield SimpleNamespace(
        SonarPlatform=SonarPlatform, FlankArray=FlankArray, TowedArray=TowedArray
    )
    restore_real_bluepebble_imports(snapshot)


def _make_platform(bp, vx=0.0, vy=0.0):
    return bp.SonarPlatform(
        states=[GroundTruthState(StateVector([0.0, vx, 0.0, vy, -5.0, 0.0]), timestamp=START)],
        position_mapping=[0, 2, 4],
        transition_models=[CombinedLinearGaussianTransitionModel([ConstantVelocity(0.0)] * 3)],
        transition_times=[timedelta(hours=1)],
    )


def _panel(bp, side="port", centre=(-10.0, 4.0, 0.0), **overrides):
    """Return a 2 m single-row panel (5 sensors at 0.5 m) centred at ``centre``."""
    kwargs = dict(
        side=side,
        aperture_length_m=2.0,
        sensor_spacing_m=0.5,
        sensor_size_m=0.1,
        freq_max_hz=1500.0,
        mounting_offset=StateVector(list(centre)),
    )
    kwargs.update(overrides)
    return bp.FlankArray(**kwargs)


def _mounted(bp, platform=None, **kwargs):
    panel = _panel(bp, **kwargs)
    (platform or _make_platform(bp)).add_sensor(panel)
    return panel


def test_side_sets_the_rotation_offset(bp) -> None:
    """Port panels are yawed +90 degrees and starboard panels -90 degrees."""
    port = _panel(bp, side="port")
    stbd = _panel(bp, side="starboard", centre=(-10.0, -4.0, 0.0))

    np.testing.assert_allclose(np.asarray(port.rotation_offset).ravel(), [0, 0, np.pi / 2])
    np.testing.assert_allclose(np.asarray(stbd.rotation_offset).ravel(), [0, 0, -np.pi / 2])


def test_normals_point_out_of_each_side(bp) -> None:
    """Heading east, port sensors face north (+y) and starboard sensors face south."""
    platform = _make_platform(bp, vx=5.0)
    port = _mounted(bp, platform=platform, side="port")
    stbd = _mounted(bp, platform=platform, side="starboard", centre=(-10.0, -4.0, 0.0))

    np.testing.assert_allclose(port.array.sensor_normals[1], 1.0)
    np.testing.assert_allclose(stbd.array.sensor_normals[1], -1.0)
    np.testing.assert_allclose(port.array.state_vector[1], 4.0)
    np.testing.assert_allclose(stbd.array.state_vector[1], -4.0)


def test_grid_rows_columns_and_reference(bp) -> None:
    """Rows are centred on the panel's depth, columns on its position along the hull."""
    panel = _mounted(
        bp, side="starboard", centre=(-3.05, -4.0, -3.0), aperture_height_m=1.0
    )

    positions = panel.array.state_vector
    assert panel.num_sensors == 5 * 3
    np.testing.assert_allclose(sorted(set(np.round(positions[2], 6))), [-8.5, -8.0, -7.5])
    np.testing.assert_allclose(positions[1], -4.0)
    columns = sorted(set(np.round(positions[0], 6)))
    np.testing.assert_allclose(columns, [-4.05, -3.55, -3.05, -2.55, -2.05])
    reference = np.asarray(panel.array.ref_state_vector).ravel()
    np.testing.assert_allclose(reference, [-3.05, -4.0, -8.0])


def test_geometry_rotates_with_heading_and_history(bp) -> None:
    """Heading north puts the panel south of the bow, and each timestamp has its own geometry."""
    platform = _make_platform(bp, vy=5.0)
    panel = _mounted(bp, platform=platform)
    later = START + timedelta(seconds=10)
    platform.move(later)

    positions_now = panel.array.state_vector
    positions_start = panel.get_platform_state_at(START).array.state_vector
    assert np.all(positions_now[1] < 50.0)  # aft of the bow, which is at y = 50
    np.testing.assert_allclose(positions_now[0], -4.0)  # port is -x when heading north
    np.testing.assert_allclose(positions_now[1] - positions_start[1], 50.0)
    assert panel.get_platform_state_at(START + timedelta(seconds=5)) is None


def test_baffle_level_and_axis(bp) -> None:
    """The baffle level reaches the array state; the axis points along the hull."""
    panel = _mounted(bp, platform=_make_platform(bp, vy=5.0), baffle_attenuation_db=20.0)

    assert panel.array.baffle_attenuation_db == 20.0
    assert panel.get_platform_state_at(START).array.baffle_attenuation_db == 20.0
    assert panel.array.axis_azimuth_rad == pytest.approx(np.pi / 2)


def test_panel_must_fit_between_bow_and_tow_point(bp) -> None:
    """A panel reaching aft of a mounted towed array's tow point is rejected at mounting."""
    platform = _make_platform(bp, vx=5.0)
    platform.add_sensor(
        bp.TowedArray(
            num_sensors=4,
            cable_length_m=100.0,
            sensor_spacing_m=1.0,
            array_depth_m=-50.0,
            mounting_offset=StateVector([-20.0, 0.0, 0.0]),
        )
    )
    _mounted(bp, platform=platform, centre=(-10.0, 4.0, 0.0))  # fits

    with pytest.raises(ValueError, match="tow point"):
        _mounted(bp, platform=platform, centre=(-19.5, 4.0, 0.0))


def test_panel_forward_of_the_bow_is_rejected(bp) -> None:
    """A panel whose footprint reaches ahead of the platform's reference point is rejected."""
    with pytest.raises(ValueError, match="forward of the bow"):
        _mounted(bp, centre=(-0.5, 4.0, 0.0))


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"side": "keel"}, "side"),
        ({"mounting_offset": None}, "needs a mounting_offset"),
        ({"centre": (-10.0, -4.0, 0.0)}, "must be positive for a port panel"),
        ({"rotation_offset": StateVector([0.0, 0.0, 0.0])}, "rotation_offset is set by side"),
        ({"aperture_height_m": -1.0}, "non-negative"),
        ({"sensor_spacing_m": 0.05}, "do not touch"),
    ],
)
def test_invalid_configuration_raises(bp, overrides, match) -> None:
    """Configuration errors are reported at construction."""
    with pytest.raises(ValueError, match=match):
        _panel(bp, **overrides)
