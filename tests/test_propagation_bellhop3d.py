"""Tests for Bellhop3DAcousticPropagationModel.

The file-format and mapping tests use a stub executable, so they run without Bellhop. Tests
marked ``needs_bellhop`` run the real ``bellhopcxx`` and are skipped when it is not on PATH
(as in CI).
"""

from __future__ import annotations

import stat
from pathlib import Path
from shutil import which
from types import SimpleNamespace

import numpy as np
import pytest
from stonesoup.types.array import StateVector
from stonesoup.types.groundtruth import GroundTruthState

from .support import isolate_real_bluepebble_imports, restore_real_bluepebble_imports

needs_bellhop = pytest.mark.skipif(which("bellhopcxx") is None, reason="bellhopcxx not on PATH")

C = 1500.0


@pytest.fixture
def bp():
    """Import the real model and environment classes without leaking caching."""
    snapshot = isolate_real_bluepebble_imports()
    from bluepebble.models.environment import Constant, FlatBathymetry, WedgeBathymetry
    from bluepebble.models.propagation import (
        Bellhop3DAcousticPropagationModel,
        BellhopAcousticPropagationModel,
    )
    from bluepebble.models.propagation.bellhop3d import _contiguous_bearings_deg

    yield SimpleNamespace(
        Bellhop3DAcousticPropagationModel=Bellhop3DAcousticPropagationModel,
        BellhopAcousticPropagationModel=BellhopAcousticPropagationModel,
        Constant=Constant,
        FlatBathymetry=FlatBathymetry,
        WedgeBathymetry=WedgeBathymetry,
        contiguous_bearings_deg=_contiguous_bearings_deg,
    )
    restore_real_bluepebble_imports(snapshot)


def _model(bp, bathymetry=None, **kwargs):
    return bp.Bellhop3DAcousticPropagationModel(
        ssp=bp.Constant(speed=C),
        bathymetry=bathymetry or bp.FlatBathymetry(depth=-4000.0),
        **kwargs,
    )


def _platform(sensors_xyz):
    sensors = np.asarray(sensors_xyz, dtype=float)
    return SimpleNamespace(
        array=SimpleNamespace(
            state_vector=sensors, ref_state_vector=StateVector(sensors.mean(axis=1, keepdims=True))
        )
    )


def _source(xyz):
    return GroundTruthState(StateVector(list(xyz)), metadata={"position_mapping": [0, 1, 2]})


def _arr_text_3d(blocks, depths=(100.0,), ranges_m=(1000.0,), bearings=(0.0,)):
    """Return a 3D ASCII arrivals file: one block per (bearing, depth, range), range fastest.

    Each arrival is ``(amplitude, phase_deg, delay_s, rcv_azimuth_deg, n_top, n_bottom)``.
    """
    lines = [
        "'3D'",
        "   1000.0",
        "   1   0.0",
        "   1   0.0",
        "   1   50.0",
        f"   {len(depths)} " + " ".join(str(d) for d in depths),
        f"   {len(ranges_m)} " + " ".join(str(r) for r in ranges_m),
        f"   {len(bearings)} " + " ".join(str(b) for b in bearings),
        f"   {max([len(b) for b in blocks] + [1])}",
    ]
    for block in blocks:
        lines.append(f"   {len(block)}")
        lines += [
            f"   {amp} {phase} {delay} 0.0 -5.0 1.5 5.0 {azim} {top} {bot}"
            for amp, phase, delay, azim, top, bot in block
        ]
    return "\n".join(lines) + "\n"


def _stub(tmp_path: Path, arr_text: str) -> str:
    """Return a stand-in bellhopcxx that writes ``arr_text`` and keeps its inputs and args."""
    (tmp_path / "canned.arr").write_text(arr_text)
    exe = tmp_path / "fake_bellhopcxx"
    exe.write_text(
        "#!/bin/sh\nfor stem; do :; done\n"
        f'echo "$@" >> "{tmp_path}/args.log"\n'
        f'cp "{tmp_path}/canned.arr" "$stem.arr"\n'
        f'cp "$stem.env" "{tmp_path}/seen.env"\n'
        f'cp "$stem.bty" "{tmp_path}/seen.bty"\n'
    )
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    return str(exe)


# ---------------------------------------------------------------------------
# Files, parsing and receiver mapping (stub executable)
# ---------------------------------------------------------------------------


def test_env_and_bty_describe_the_3d_geometry(bp, tmp_path) -> None:
    """The source is at its world (x, y) in km and receivers on a bearing/depth/range grid."""
    model = _model(
        bp,
        bathymetry=bp.WedgeBathymetry(depth_at_origin=-300.0, x_gradient=0.0, y_gradient=-0.01),
        n_bathy_samples=5,
        exe_path=_stub(tmp_path, _arr_text_3d([[]] * 2, ranges_m=(1000.0, 2000.0))),
    )
    sensors = [[1500.0, 2500.0], [200.0, 200.0], [-80.0, -80.0]]

    model.propagate_spectrum(_platform(sensors), _source([500.0, 200.0, -50.0]), np.array([800.0]))

    env = (tmp_path / "seen.env").read_text().splitlines()
    assert "0.500000000 /" in env and "0.200000000 /" in env and "50.0000 /" in env
    assert "80.0000 /" in env
    assert "1.000000 2.000000 /" in env
    assert "0.000000 /" in env  # both receivers due +x of the source
    assert "'AG   3'" in env
    # The bearing fan is the receivers' bearing +/- 5 degrees at 0.1 degree spacing.
    assert env[env.index("'AG   3'") + 3 : env.index("'AG   3'") + 5] == [
        "101",
        "-5.000000 5.000000 /",
    ]
    # Box: 1.05 x the farthest range, and 1.05 x the deepest point (y = 2.3 km, 323 m).
    assert env[-1] == "0.0 2.100000 2.100000 339.15"

    bty = (tmp_path / "seen.bty").read_text().splitlines()
    assert bty[:2] == ["'C'", "5"]  # curvilinear by default
    x_km = np.array(bty[2].split(), dtype=float)
    y_km = np.array(bty[4].split(), dtype=float)
    np.testing.assert_allclose(x_km, 0.5 + np.linspace(-2.1, 2.1, 5))
    np.testing.assert_allclose(y_km, 0.2 + np.linspace(-2.1, 2.1, 5))
    depths = np.array([row.split() for row in bty[5:]], dtype=float)
    np.testing.assert_allclose(depths, np.tile((300.0 + 10.0 * y_km)[:, None], (1, 5)))
    assert "--3D" in (tmp_path / "args.log").read_text()


def test_linear_bathymetry_writes_a_rectilinear_grid(bp, tmp_path) -> None:
    """With bathymetry_interpolation="linear" the grid is joined by flat facets ('R')."""
    model = _model(
        bp,
        bathymetry_interpolation="linear",
        n_bathy_samples=5,
        exe_path=_stub(tmp_path, _arr_text_3d([[]] * 2, ranges_m=(1000.0, 2000.0))),
    )
    sensors = [[1500.0, 2500.0], [200.0, 200.0], [-80.0, -80.0]]

    model.propagate_spectrum(_platform(sensors), _source([500.0, 200.0, -50.0]), np.array([800.0]))

    assert (tmp_path / "seen.bty").read_text().splitlines()[0] == "'R'"


def test_nx2d_writes_its_run_type_and_flag(bp, tmp_path) -> None:
    """With nx2d, the environment asks for N x 2D ('AG   2') and bellhopcxx gets --Nx2D."""
    model = _model(
        bp,
        nx2d=True,
        n_bathy_samples=5,
        exe_path=_stub(tmp_path, _arr_text_3d([[]] * 2, ranges_m=(1000.0, 2000.0))),
    )
    sensors = [[1500.0, 2500.0], [200.0, 200.0], [-80.0, -80.0]]

    model.propagate_spectrum(_platform(sensors), _source([500.0, 200.0, -50.0]), np.array([800.0]))

    env = (tmp_path / "seen.env").read_text().splitlines()
    assert "'AG   2'" in env and "'AG   3'" not in env
    args = (tmp_path / "args.log").read_text()
    assert "--Nx2D" in args and "--3D" not in args


def test_receivers_are_mapped_to_their_bearing_depth_and_range(bp, tmp_path) -> None:
    """Sensors in any order get the arrivals at their own (bearing, depth, range) point."""
    # Grid: bearings (0, 90) x depths (50, 80) x ranges (1000, 2000); delay = block index.
    blocks = [[(1e-3, 0.0, float(k), 0.0, 0, 0)] for k in range(8)]
    arr = _arr_text_3d(blocks, depths=(50, 80), ranges_m=(1000, 2000), bearings=(0, 90))
    model = _model(bp, exe_path=_stub(tmp_path, arr))
    sensors = [
        [0.0, 2000.0, 1000.0, 0.0],
        [1000.0, 0.0, 0.0, 2000.0],
        [-80.0, -50.0, -80.0, -50.0],
    ]

    _, delays = model.propagate_spectrum(
        _platform(sensors), _source([0.0, 0.0, -10.0]), np.array([500.0])
    )

    # (90, 80, 1000) is block 6, (0, 50, 2000) block 1, (0, 80, 1000) block 2,
    # (90, 50, 2000) block 5.
    np.testing.assert_allclose(delays, [6.0, 1.0, 2.0, 5.0])


def test_parse_reads_the_3d_arrival_columns(bp, tmp_path) -> None:
    """Each 3D arrival row gives amplitude, phase, delay, angles, azimuths and bounces."""
    arr = _arr_text_3d([[(5e-4, 180.0, 0.68, 2.5, 1, 0), (4e-4, 0.0, 0.67, -1.0, 0, 0)]])
    model = _model(bp, exe_path=_stub(tmp_path, arr))

    rays = model.compute_eigenrays([0.0, 0.0, -50.0], [1000.0, 0.0, -100.0], 500.0)

    assert [r.amplitude for r in rays] == [5e-4, 4e-4]
    assert rays[0].phase_rad == pytest.approx(np.pi)
    assert [r.delay_s for r in rays] == [0.68, 0.67]
    assert (rays[0].src_angle_deg, rays[0].rcv_angle_deg) == (-5.0, 5.0)
    assert (rays[0].src_azimuth_deg, rays[0].rcv_azimuth_deg) == (1.5, 2.5)
    assert (rays[0].n_surface_bounces, rays[0].n_bottom_bounces) == (1, 0)


def test_bearings_either_side_of_minus_x_stay_neighbours(bp) -> None:
    """Receivers straddling the -x axis get bearings near 180, not near -180 and 180."""
    bearings = bp.contiguous_bearings_deg(np.array([-1000.0, -1000.0]), np.array([1.0, -1.0]))

    assert abs(bearings[1] - bearings[0]) < 0.2
    assert np.all(np.abs(np.abs(bearings) - 180.0) < 0.1)


def test_bearing_fan_never_exceeds_a_full_circle(bp) -> None:
    """Receivers all round the source get a 360 degree fan, not a wider one."""
    low, high = _model(bp)._bearing_fan_deg(np.array([-176.0, 0.0, 176.0]))

    assert high - low == pytest.approx(360.0)


def test_separate_planes_does_not_apply_and_is_ignored_with_a_warning(bp) -> None:
    """Every sensor is traced at its own bearing, so separate_planes is dropped."""
    with pytest.warns(UserWarning, match="separate_planes does not apply"):
        model = _model(bp, separate_planes=True)
    assert (model.far_field, model.separate_planes) == (False, False)


@pytest.mark.parametrize(
    ("geometry", "far_field"),
    [("reference", False), ("per_sensor", False), ("far_field", True)],
)
def test_deprecated_receiver_geometry_maps_to_far_field(bp, geometry, far_field) -> None:
    """receiver_geometry still works in 3D, with a DeprecationWarning."""
    with pytest.warns(DeprecationWarning, match="receiver_geometry is deprecated"):
        model = _model(bp, receiver_geometry=geometry)
    assert (model.far_field, model.separate_planes) == (far_field, False)


# ---------------------------------------------------------------------------
# Physics, against the real bellhopcxx
# ---------------------------------------------------------------------------


def _path(rays, top, bottom):
    """Return the amplitude and amplitude-weighted delay of one bounce class."""
    rays = [r for r in rays if (r.n_surface_bounces, r.n_bottom_bounces) == (top, bottom)]
    amplitudes = np.array([r.amplitude for r in rays])
    delays = np.array([r.delay_s for r in rays])
    return amplitudes.sum(), float(amplitudes @ delays / amplitudes.sum())


@needs_bellhop
def test_direct_path_matches_spherical_spreading(bp) -> None:
    """In deep isovelocity water the direct path has amplitude 1/R and delay R/c."""
    receiver = np.array([1500.0, 1500.0, -150.0])
    distance = np.linalg.norm(receiver - [0.0, 0.0, -100.0])

    rays = _model(bp).compute_eigenrays([0.0, 0.0, -100.0], receiver, 1000.0)
    amplitude, delay = _path(rays, 0, 0)

    assert amplitude == pytest.approx(1 / distance, rel=1e-3)
    assert delay == pytest.approx(distance / C, abs=5e-6)


@needs_bellhop
def test_agrees_with_the_2d_model_over_a_flat_bottom(bp) -> None:
    """With bathymetry the same on every bearing, 3D and 2D give the same field."""
    rng = np.random.default_rng(3)
    bearings = np.radians(rng.uniform(-5, 5, 6))
    ranges = rng.uniform(1500, 2500, 6)
    sensors = np.vstack(
        [ranges * np.cos(bearings), ranges * np.sin(bearings), -rng.uniform(50, 150, 6)]
    )
    source = _source([0.0, 0.0, -40.0])
    f = np.linspace(400.0, 800.0, 21)
    flat = bp.FlatBathymetry(depth=-200.0)

    H_2d, d_2d = bp.BellhopAcousticPropagationModel(
        ssp=bp.Constant(speed=C), bathymetry=flat
    ).propagate_spectrum(_platform(sensors), source, f)
    H_3d, d_3d = _model(bp, bathymetry=flat).propagate_spectrum(_platform(sensors), source, f)

    np.testing.assert_allclose(d_3d, d_2d, atol=2e-6)
    assert np.abs(H_3d - H_2d).max() / np.abs(H_2d).max() < 0.02


@needs_bellhop
def test_nx2d_matches_the_2d_model_in_separate_planes(bp) -> None:
    """N x 2D and 2D separate planes trace the same vertical planes, so agree, on a slope too.

    Over a seabed sloping across the line of sight both follow each receiver's own plane,
    with its own seabed profile, and neither leaves it.
    """
    rng = np.random.default_rng(5)
    bearings = np.radians(rng.uniform(-5, 5, 6))
    ranges = rng.uniform(1500, 2500, 6)
    sensors = np.vstack(
        [ranges * np.cos(bearings), ranges * np.sin(bearings), -rng.uniform(50, 150, 6)]
    )
    source = _source([0.0, 0.0, -40.0])
    f = np.linspace(400.0, 800.0, 21)
    wedge = bp.WedgeBathymetry(depth_at_origin=-300.0, x_gradient=0.0, y_gradient=-0.02)
    settings = dict(ssp=bp.Constant(speed=C), bathymetry=wedge, n_bathy_samples=400)

    H_2d, d_2d = bp.BellhopAcousticPropagationModel(
        separate_planes=True, **settings
    ).propagate_spectrum(_platform(sensors), source, f)
    H_nx, d_nx = bp.Bellhop3DAcousticPropagationModel(nx2d=True, **settings).propagate_spectrum(
        _platform(sensors), source, f
    )

    np.testing.assert_allclose(d_nx, d_2d, atol=2e-6)
    assert np.abs(H_nx - H_2d).max() / np.abs(H_2d).max() < 0.02


@needs_bellhop
def test_3d_bottom_bounce_off_a_cross_slope_follows_the_image_source(bp) -> None:
    """Over a sloping plane the bottom-reflected path is the straight line from the image.

    The reflection leaves the vertical plane through source and receiver, so 2D per-sensor
    runs (which assume the slope lies in that plane) get its delay wrong; 3D gets it right.
    """
    slope = -0.03
    wedge = bp.WedgeBathymetry(depth_at_origin=-300.0, x_gradient=0.0, y_gradient=slope)
    source = np.array([0.0, 0.0, -50.0])
    receiver = np.array([3000.0, 0.0, -100.0])
    normal = np.array([0.0, -slope, 1.0])  # seabed plane: normal . x + 300 = 0
    image = source - 2 * (normal @ source + 300.0) / (normal @ normal) * normal
    exact_s = np.linalg.norm(receiver - image) / C
    lossless = dict(bottom_attenuation_db_per_wavelength=0.0)

    _, delay_3d = _path(
        _model(bp, bathymetry=wedge, **lossless).compute_eigenrays(source, receiver, 1000.0), 0, 1
    )
    _, delay_2d = _path(
        bp.BellhopAcousticPropagationModel(
            ssp=bp.Constant(speed=C), bathymetry=wedge, **lossless
        ).compute_eigenrays(source, receiver, 1000.0),
        0,
        1,
    )

    assert delay_3d == pytest.approx(exact_s, abs=3e-6)
    assert abs(delay_2d - exact_s) > 10e-6
