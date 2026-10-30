"""Tests for the hull-mounted FlankArraySensor."""

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
    from bluepebble.sensors.flank_array import FlankArraySensor
    from bluepebble.sensors.towed_array import TowedArraySensor

    yield HostPlatform, FlankArraySensor, TowedArraySensor
    restore_real_bluepebble_imports(snapshot)


def _make_host(HostPlatform, vx: float = 0.0, vy: float = 0.0):
    model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.0)] * 3)
    return HostPlatform(
        states=[GroundTruthState(StateVector([0.0, vx, 0.0, vy, -5.0, 0.0]), timestamp=START)],
        position_mapping=[0, 2, 4],
        transition_models=[model],
        transition_times=[timedelta(hours=1)],
    )


def _flank_kwargs(**overrides):
    kwargs = dict(
        hull_beam_m=8.0,
        element_spacing_m=0.5,
        element_size_m=0.1,
        freq_max_hz=1500.0,
        towed_array_bow_offset_m=20.0,
        flank_bow_offset_m=2.0,
    )
    kwargs.update(overrides)
    return kwargs


def test_elements_fill_hull_section_between_bow_and_tow_point(modules) -> None:
    """With no aperture length given, panels fill the section and footprints stay inside it."""
    HostPlatform, FlankArraySensor, _ = modules
    flank = FlankArraySensor(host=_make_host(HostPlatform), **_flank_kwargs())

    positions = np.asarray(flank.array.state_vector)
    x = positions[0]
    half = 0.05
    assert x.max() + half <= -2.0 + 1e-9
    assert x.min() - half >= -20.0 - 1e-9
    # 17.9m of centre span at 0.5m spacing -> 36 columns, one row, two sides.
    assert flank.array.num_sensors == 2 * 36
    np.testing.assert_allclose(sorted(set(np.round(positions[1], 6))), [-4.0, 4.0])


def test_aperture_that_does_not_fit_raises(modules) -> None:
    """An aperture longer than the bow-to-tow-point section is rejected."""
    HostPlatform, FlankArraySensor, _ = modules
    with pytest.raises(ValueError, match="does not fit"):
        FlankArraySensor(host=_make_host(HostPlatform), **_flank_kwargs(aperture_length_m=18.0))


def test_tow_offset_taken_from_attached_towed_array(modules) -> None:
    """Omitting the tow offset uses the attached TowedArraySensor's tow point."""
    HostPlatform, FlankArraySensor, TowedArraySensor = modules
    host = _make_host(HostPlatform, vx=5.0)
    TowedArraySensor(
        host=host,
        num_sensors=4,
        cable_length_m=100.0,
        sensor_spacing_m=1.0,
        array_depth_m=-50.0,
        towed_array_bow_offset_m=30.0,
    )
    flank = FlankArraySensor(host=host, **_flank_kwargs(towed_array_bow_offset_m=None))

    assert flank.towed_array_bow_offset_m == 30.0


def test_missing_tow_offset_without_towed_array_raises(modules) -> None:
    """With no tow offset and no towed array, construction fails clearly."""
    HostPlatform, FlankArraySensor, _ = modules
    with pytest.raises(ValueError, match="TowedArraySensor"):
        FlankArraySensor(
            host=_make_host(HostPlatform), **_flank_kwargs(towed_array_bow_offset_m=None)
        )


def test_grid_rows_sides_and_depth_offset(modules) -> None:
    """Rows are centred on the depth offset, single side gives one panel."""
    HostPlatform, FlankArraySensor, _ = modules
    flank = FlankArraySensor(
        host=_make_host(HostPlatform),
        **_flank_kwargs(
            aperture_length_m=2.0, aperture_height_m=1.0, array_depth_offset_m=3.0,
            sides="starboard",
        ),
    )

    positions = np.asarray(flank.array.state_vector)
    assert flank.array.num_sensors == 5 * 3
    np.testing.assert_allclose(sorted(set(np.round(positions[2], 6))), [-8.5, -8.0, -7.5])
    np.testing.assert_allclose(positions[1], -4.0)
    # Reference is the panel centroid, at the panel's depth (host -5 m, offset 3 m).
    ref = np.asarray(flank.array.ref_state_vector).flatten()
    np.testing.assert_allclose(ref, [-3.05, -4.0, -8.0])


def test_geometry_rotates_with_heading_and_history(modules) -> None:
    """Heading north puts the panels south of the bow, and history is per-timestamp."""
    HostPlatform, FlankArraySensor, _ = modules
    host = _make_host(HostPlatform, vy=5.0)
    flank = FlankArraySensor(host=host, **_flank_kwargs())

    later = START + timedelta(seconds=10)
    host.move(later)

    positions_now = np.asarray(flank.array.state_vector)
    assert np.all(positions_now[1] < 50.0 - 2.0 + 1e-9)
    np.testing.assert_allclose(sorted(set(np.round(positions_now[0], 6))), [-4.0, 4.0])

    positions_start = np.asarray(flank.get_platform_state_at(START).array.state_vector)
    np.testing.assert_allclose(positions_now[1] - positions_start[1], 50.0)


def test_baffle_level_reaches_array_state(modules) -> None:
    """The sensor's baffle level is reported on every array snapshot."""
    HostPlatform, FlankArraySensor, _ = modules
    host = _make_host(HostPlatform, vx=5.0)
    flank = FlankArraySensor(host=host, **_flank_kwargs(baffle_attenuation_db=20.0))

    assert flank.array.baffle_attenuation_db == 20.0
    assert flank.get_platform_state_at(START).array.baffle_attenuation_db == 20.0


def test_axis_follows_heading(modules) -> None:
    """The array axis points along the host's heading (north here)."""
    HostPlatform, FlankArraySensor, _ = modules
    flank = FlankArraySensor(host=_make_host(HostPlatform, vy=5.0), **_flank_kwargs())

    np.testing.assert_allclose(np.asarray(flank.array.axis).ravel(), [0.0, 1.0, 0.0], atol=1e-12)
    assert flank.array.axis_azimuth_rad == pytest.approx(np.pi / 2)


def test_panel_states_split_port_and_starboard(modules) -> None:
    """Each panel holds only its own side's elements, referenced at its own centroid."""
    HostPlatform, FlankArraySensor, _ = modules
    host = _make_host(HostPlatform, vx=5.0)
    flank = FlankArraySensor(host=host, **_flank_kwargs(baffle_attenuation_db=20.0))
    full = flank.get_platform_state_at(START).array
    full_positions = np.asarray(full.state_vector)

    panels = flank.panel_states(START)

    assert set(panels) == {"port", "starboard"}
    port_idx, port = panels["port"]
    stbd_idx, stbd = panels["starboard"]
    assert len(port_idx) + len(stbd_idx) == full.num_sensors
    assert not set(port_idx) & set(stbd_idx)

    # Heading east: port is +y (north), starboard is -y.
    np.testing.assert_allclose(np.asarray(port.array.state_vector)[1], 4.0)
    np.testing.assert_allclose(np.asarray(stbd.array.state_vector)[1], -4.0)
    np.testing.assert_allclose(np.asarray(port.array.state_vector), full_positions[:, port_idx])
    np.testing.assert_allclose(
        np.asarray(port.array.ref_state_vector).ravel(), full_positions[:, port_idx].mean(axis=1)
    )
    np.testing.assert_allclose(np.asarray(port.array.element_normals)[1], 1.0)
    assert port.array.num_sensors == len(port_idx)
    assert port.array.baffle_attenuation_db == 20.0
    assert port.array.axis_azimuth_rad == pytest.approx(0.0)


def test_panel_states_only_returns_fitted_sides(modules) -> None:
    """A single-sided sensor has a single panel, and no timestamp means the live geometry."""
    HostPlatform, FlankArraySensor, _ = modules
    flank = FlankArraySensor(host=_make_host(HostPlatform, vx=5.0), **_flank_kwargs(sides="port"))

    panels = flank.panel_states()

    assert list(panels) == ["port"]
    np.testing.assert_array_equal(panels["port"][0], np.arange(flank.array.num_sensors))
