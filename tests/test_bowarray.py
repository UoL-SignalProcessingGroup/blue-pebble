"""Tests for BowArray: dome and cylinder layouts, mounting, baffling and geometry over time."""

from __future__ import annotations

import warnings
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


# The brief's check case: 40 staves x 11 rows, as a 1.64 m x 1.44 m cylinder of 440 sensors.
RADIUS_M = 0.82
HEIGHT_M = 1.44


def _make_cylinder(bp, platform=None, **overrides):
    kwargs = dict(
        dome_radius_m=RADIUS_M,
        azimuth_extent_rad=2 * np.pi,
        sensor_spacing_m=0.131,
        sensor_size_m=0.1,
        freq_max_hz=5000.0,
        shape="cylindrical",
        height_m=HEIGHT_M,
    )
    kwargs.update(overrides)
    bow = bp.BowArray(**kwargs)
    (platform or _make_platform(bp)).add_sensor(bow)
    return bow


def _offsets(bow):
    """Sensor positions relative to the array's reference point, in the world frame."""
    return bow.array.state_vector - np.asarray(bow.array.ref_state_vector)


def _azimuths_off_north(offsets):
    """Azimuth of each sensor off a +y heading, positive to port (towards -x)."""
    return np.arctan2(-offsets[0], offsets[1])


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


def test_default_shape_keeps_the_dome_layout(bp) -> None:
    """The default is the spherical dome, laid out as before the shape property existed."""
    bow = _make_bow(bp)

    # Worked by hand: 10 deg of a 2 m sphere is 0.35 m of arc, so floor(0.35 / 0.3) + 1 = 2
    # rows at +/-5 deg; each row spans 60 deg * 2 m * cos(5 deg) = 2.09 m, so
    # floor(2.09 / 0.3) + 1 = 7 sensors.
    elevations = np.repeat(np.radians([-5.0, 5.0]), 7)
    azimuths = np.tile(np.radians(np.linspace(-30.0, 30.0, 7)), 2)
    expected = 2.0 * np.vstack(
        [
            -np.cos(elevations) * np.sin(azimuths),
            np.cos(elevations) * np.cos(azimuths),
            np.sin(elevations),
        ]
    )
    np.testing.assert_allclose(_offsets(bow), expected, atol=1e-12)

    explicit = _make_bow(bp, shape="spherical")
    np.testing.assert_array_equal(_offsets(explicit), _offsets(bow))
    np.testing.assert_array_equal(
        np.asarray(explicit.array.sensor_normals), np.asarray(bow.array.sensor_normals)
    )


def test_cylinder_sensors_lie_on_the_cylinder(bp) -> None:
    """The check case gives 40 staves of 11 sensors, at the stated ring and row pitches."""
    bow = _make_cylinder(bp, mounting_offset=StateVector([0.0, 0.0, -3.0]))

    offsets = _offsets(bow)
    assert bow.num_sensors == 440
    np.testing.assert_allclose(np.hypot(offsets[0], offsets[1]), RADIUS_M)
    assert np.abs(offsets[2]).max() <= HEIGHT_M / 2

    # Ordered by row (lowest first), then by azimuth.
    rows = offsets.reshape(3, 11, 40)
    np.testing.assert_allclose(rows[2], rows[2][:, :1] * np.ones(40), atol=1e-12)
    row_pitches = np.diff(rows[2][:, 0])
    np.testing.assert_allclose(row_pitches, 0.1309, atol=5e-5)
    np.testing.assert_allclose(rows[2][:, 0].mean(), 0.0, atol=1e-12)

    stave_azimuths = _azimuths_off_north(rows[:, 0, :])
    np.testing.assert_allclose(RADIUS_M * np.diff(stave_azimuths), 0.1288, atol=5e-5)
    assert np.abs(stave_azimuths).min() == pytest.approx(0.0, abs=1e-12)
    for row in range(1, 11):
        np.testing.assert_allclose(rows[:2, row, :], rows[:2, 0, :], atol=1e-12)


def test_cylinder_sector_is_a_subset_of_the_full_ring(bp) -> None:
    """A 120 deg sector is the full ring's staves within 60 deg of boresight, unmoved."""
    full = _offsets(_make_cylinder(bp))
    sector = _offsets(_make_cylinder(bp, azimuth_extent_rad=np.radians(120.0)))

    assert sector.shape == (3, 13 * 11)
    in_sector = np.abs(_azimuths_off_north(full)) <= np.radians(60.0) + 1e-9
    np.testing.assert_allclose(sector, full[:, in_sector], atol=1e-12)


@pytest.mark.parametrize(("vx", "vy"), [(0.0, 5.0), (5.0, 0.0)])
def test_cylinder_normals_face_outwards(bp, vx, vy) -> None:
    """Each stave's normal is horizontal and radial, and turns with the heading."""
    bow = _make_cylinder(bp, platform=_make_platform(bp, vx=vx, vy=vy))

    normals = np.asarray(bow.array.sensor_normals)
    offsets = _offsets(bow)
    np.testing.assert_allclose(normals[2], 0.0, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(normals, axis=0), 1.0)
    np.testing.assert_allclose(normals[:2], offsets[:2] / RADIUS_M, atol=1e-12)
    # The boresight stave faces along the heading.
    heading = np.array([vx, vy, 0.0]) / np.hypot(vx, vy)
    assert (normals.T @ heading).max() == pytest.approx(1.0)


def test_full_ring_baffles_the_staves_facing_away_from_a_contact(bp) -> None:
    """For a contact dead astern, aft-facing staves hear it and forward-facing ones do not."""
    bow = _make_cylinder(bp, baffle_attenuation_db=20.0)

    gains = bow.array.sensor_gains([0.0, -1.0, 0.0])  # Heading is +y, so astern is -y

    # Staves abeam are on the 90 deg cutoff, where rounding decides the side.
    along_heading = _offsets(bow)[1]
    forward, aft = along_heading > 1e-9, along_heading < -1e-9
    assert forward.sum() == aft.sum() == 19 * 11
    np.testing.assert_allclose(gains[aft], 1.0)
    np.testing.assert_allclose(gains[forward], 0.1)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"freq_max_hz": 6000.0}, "ring pitch.*grating-lobe"),  # lambda/2 = 0.125 m
        ({"freq_max_hz": 5800.0}, "row pitch.*grating-lobe"),  # 0.1287 < 0.1293 < 0.1309 m
        ({"sensor_size_m": 0.13}, "ring pitch.*do not touch"),  # Chord is 0.1287 m
    ],
)
def test_cylinder_pitch_beyond_the_limits_raises(bp, overrides, match) -> None:
    """The ring's chord pitch and the row pitch are each checked, not the nominal spacing."""
    with pytest.raises(ValueError, match=match):
        _make_cylinder(bp, **overrides)


def test_cylinder_spacing_check_uses_the_pitches_as_laid_out(bp) -> None:
    """A nominal spacing above lambda/2 passes when the pitches it lays out are below it."""
    # lambda/2 is 0.136 m at 5.5 kHz. A 0.14 m spacing on a 0.25 m radius gives 12 staves a
    # 0.129 m chord apart, and on a 0.3 m height gives 3 rows 0.1 m apart.
    bow = _make_cylinder(
        bp,
        dome_radius_m=0.25,
        height_m=0.3,
        sensor_spacing_m=0.14,
        sensor_size_m=0.05,
        freq_max_hz=5500.0,
    )

    assert bow.num_sensors == 12 * 3


def test_single_stave_has_no_pitch_to_check(bp) -> None:
    """One stave of one sensor has no neighbours, so any frequency passes the spacing check."""
    bow = _make_cylinder(
        bp, azimuth_extent_rad=np.radians(1.0), height_m=0.12, freq_max_hz=50000.0
    )

    np.testing.assert_allclose(_offsets(bow), [[0.0], [RADIUS_M], [0.0]], atol=1e-12)


@pytest.mark.parametrize(
    ("make", "overrides", "match"),
    [
        (_make_bow, {"height_m": 1.0}, "height_m must be None"),
        (_make_bow, {"elevation_extent_rad": None}, "elevation_extent_rad is required"),
        (_make_bow, {"shape": "conical"}, "shape"),
        (_make_cylinder, {"elevation_extent_rad": 0.2}, "elevation_extent_rad must be None"),
        (_make_cylinder, {"height_m": None}, "height_m is required"),
        (_make_cylinder, {"height_m": 0.0}, "must be positive"),
        (_make_cylinder, {"azimuth_extent_rad": 7.0}, "azimuth_extent_rad"),
    ],
)
def test_mixed_up_shape_properties_raise(bp, make, overrides, match) -> None:
    """Each shape rejects the other's property and needs its own."""
    with pytest.raises(ValueError, match=match):
        make(bp, **overrides)


def test_legacy_positional_arguments_warn_and_keep_their_meaning(bp) -> None:
    """Three or more positional arguments are read in the order BowArray took up to 0.7.0."""
    with pytest.warns(DeprecationWarning, match="by keyword"):
        bow = bp.BowArray(2.0, np.radians(60.0), np.radians(10.0), 0.3, 0.1, 2000.0)

    assert bow.elevation_extent_rad == np.radians(10.0)
    assert (bow.sensor_spacing_m, bow.sensor_size_m, bow.freq_max_hz) == (0.3, 0.1, 2000.0)
    assert bow.num_sensors == 14

    with pytest.raises(TypeError, match="multiple values for elevation_extent_rad"):
        bp.BowArray(2.0, np.radians(60.0), np.radians(10.0), elevation_extent_rad=0.2)
    with pytest.raises(TypeError, match="at most 10 positional"):
        bp.BowArray(*range(11))


def test_two_positional_arguments_do_not_warn(bp) -> None:
    """The radius and azimuth extent kept their positions, so passing them is not deprecated."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        bow = bp.BowArray(
            2.0,
            np.radians(60.0),
            elevation_extent_rad=np.radians(10.0),
            sensor_spacing_m=0.3,
            sensor_size_m=0.1,
            freq_max_hz=2000.0,
        )

    assert bow.num_sensors == 14
