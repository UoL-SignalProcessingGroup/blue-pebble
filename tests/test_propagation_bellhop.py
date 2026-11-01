"""Tests for BellhopAcousticPropagationModel and Eigenray.

The file-format and error tests use a stub executable, so they run without Bellhop. The tests
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

from .support import isolate_real_bluepebble_imports, restore_real_bluepebble_imports

needs_bellhop = pytest.mark.skipif(which("bellhopcxx") is None, reason="bellhopcxx not on PATH")

# A 2D arrivals file in bellhopcxx's ASCII format: one source, receiver and range, 3 arrivals.
ARR_TEXT = """'2D'
   1000.0000000000000
           1   100.000000
           1   100.000000
           1   1000.0000000000000
           3
           3
   5.00000000E-04 0.00000000 0.666658000 0.00000000 0.29800000 -0.29800000 0 0
   4.88210000E-04 180.000000 0.679860000 0.00000000 -11.60900000 11.60900000 1 0
   1.99400000E-05 360.961000 5.242514000 0.00000000 82.45200000 82.45200000 2 1
"""


@pytest.fixture
def bp():
    """Import the real model and environment classes without leaking caching."""
    snapshot = isolate_real_bluepebble_imports()
    from bluepebble.models.environment import Constant, FlatBathymetry
    from bluepebble.models.propagation import BellhopAcousticPropagationModel, Eigenray

    yield SimpleNamespace(
        BellhopAcousticPropagationModel=BellhopAcousticPropagationModel,
        Eigenray=Eigenray,
        Constant=Constant,
        FlatBathymetry=FlatBathymetry,
    )
    restore_real_bluepebble_imports(snapshot)


def _model(bp, depth=-4000.0, **kwargs):
    return bp.BellhopAcousticPropagationModel(
        ssp=bp.Constant(speed=1500.0), bathymetry=bp.FlatBathymetry(depth=depth), **kwargs
    )


def _stub_exe(tmp_path: Path, body: str) -> str:
    """Write an executable shell script standing in for bellhopcxx; it gets the file stem."""
    exe = tmp_path / "fake_bellhopcxx"
    exe.write_text("#!/bin/sh\n" + body)
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    return str(exe)


def test_parse_arr_reads_every_arrival_with_its_phase(bp, tmp_path) -> None:
    """Each arrival line becomes an Eigenray, with Bellhop's phase converted to radians."""
    arr = tmp_path / "x.arr"
    arr.write_text(ARR_TEXT)

    rays = bp.BellhopAcousticPropagationModel._parse_arr(arr)

    assert len(rays) == 3
    direct, surface, bottom = rays
    assert direct == bp.Eigenray(5.0e-4, 0.0, 0.666658, 0.298, -0.298, 0, 0)
    assert surface.phase_rad == pytest.approx(np.pi)
    assert (surface.n_surface_bounces, surface.n_bottom_bounces) == (1, 0)
    assert bottom.phase_rad == pytest.approx(np.radians(360.961))
    assert bottom.delay_s == pytest.approx(5.242514)


def test_env_and_bty_files_describe_the_geometry(bp, tmp_path) -> None:
    """The environment file holds the depths, range, bottom and beam fan given."""
    model = _model(
        bp,
        bottom_sound_speed_mps=1650.0,
        bottom_density_g_cm3=1.8,
        bottom_attenuation_db_per_wavelength=0.8,
        beam_angle_limits_deg=(-60.0, 60.0),
        n_ssp_samples=5,
    )
    stem = tmp_path / "bp"
    model._write_env(stem, 100.0, 250.0, 2000.0, 4000.0, 1500.0)
    model._write_bty(stem, np.array([0.0, 2.0]), np.array([4000.0, 3900.0]))

    env = stem.with_suffix(".env").read_text().splitlines()
    assert env[1] == "1500.0000"
    assert env[3] == "'SVW'"
    assert env[5:10] == [f"  {d:.2f}  1500.0000  /" for d in np.linspace(0, 4000, 5)]
    assert env[11] == "4000.00  1650.0  0.0  1.8  0.8  /"
    assert env[13] == "100.0000 /" and env[15] == "250.0000 /" and env[17] == "2.000000 /"
    assert env[18] == "'AG'" and env[20] == "-60.0 60.0 /"
    assert stem.with_suffix(".bty").read_text().splitlines() == [
        "'L'",
        "2",
        "0.000000 4000.00",
        "2.000000 3900.00",
    ]


def test_bathymetry_profile_runs_along_the_source_receiver_line(bp) -> None:
    """Ranges run from 0 at the source to the receiver, with positive water depths."""
    range_km, depth_m = _model(bp, n_bathy_samples=3)._bathymetry_profile(
        np.array([0.0, 0.0]), np.array([3000.0, 4000.0]), 5000.0
    )

    np.testing.assert_allclose(range_km, [0.0, 2.5, 5.0])
    np.testing.assert_allclose(depth_m, 4000.0)


def test_compute_eigenrays_converts_world_positions_to_bellhop_depths(bp, tmp_path) -> None:
    """World z (negative below the surface) is written as Bellhop's positive depth."""
    exe = _stub_exe(
        tmp_path,
        f'cp "$1.env" "{tmp_path}/seen.env"\ncat > "$1.arr" <<EOF\n{ARR_TEXT}EOF\n',
    )
    model = _model(bp, exe_path=exe)

    rays = model.compute_eigenrays([10.0, 20.0, -100.0], [610.0, 820.0, -250.0], 1000.0)

    assert len(rays) == 3
    env = (tmp_path / "seen.env").read_text().splitlines()
    assert env[-9] == "100.0000 /" and env[-7] == "250.0000 /"
    assert env[-5] == "1.000000 /"  # 1 km between (10, 20) and (610, 820)


@pytest.mark.parametrize(
    ("source", "receiver", "match"),
    [
        ([0.0, 0.0, 0.0], [1000.0, 0.0, -100.0], "below the surface"),
        ([0.0, 0.0, -100.0], [0.0, 0.0, -200.0], "different horizontal positions"),
    ],
)
def test_compute_eigenrays_rejects_invalid_geometry(bp, source, receiver, match) -> None:
    """Points at or above the surface, or directly above one another, are rejected."""
    with pytest.raises(ValueError, match=match):
        _model(bp).compute_eigenrays(source, receiver, 1000.0)


def test_missing_executable_says_how_to_install(bp) -> None:
    """An executable that cannot be found raises with a pointer to bellhopcuda."""
    with pytest.raises(FileNotFoundError, match="bellhopcuda"):
        _model(bp, exe_path="definitely-not-bellhopcxx").compute_eigenrays(
            [0.0, 0.0, -100.0], [1000.0, 0.0, -100.0], 1000.0
        )


def test_failing_run_raises_with_its_output(bp, tmp_path) -> None:
    """A non-zero exit from bellhopcxx is reported with its stderr."""
    exe = _stub_exe(tmp_path, 'echo "bad env file" >&2\nexit 3\n')

    with pytest.raises(RuntimeError, match=r"(?s)exit 3.*bad env file"):
        _model(bp, exe_path=exe).compute_eigenrays(
            [0.0, 0.0, -100.0], [1000.0, 0.0, -100.0], 1000.0
        )


def _paths(rays, n_surface, n_bottom):
    return [r for r in rays if (r.n_surface_bounces, r.n_bottom_bounces) == (n_surface, n_bottom)]


def _weighted_delay_s(rays):
    """Amplitude-weighted mean delay of the arrivals making up one geometric path.

    Neighbouring beams bracket the receiver, so each arrival's delay is slightly off the
    path's; weighting by amplitude recovers it.
    """
    amplitudes = np.array([r.amplitude for r in rays])
    return float(np.dot(amplitudes, [r.delay_s for r in rays]) / amplitudes.sum())


@needs_bellhop
def test_bellhop_direct_path_matches_spherical_spreading(bp) -> None:
    """In deep isovelocity water the direct path has delay R/c, amplitude 1/R and no phase.

    Bellhop returns the path as arrivals from neighbouring beams, so their amplitudes are
    summed and their delays averaged by amplitude.
    """
    rays = _model(bp).compute_eigenrays([0.0, 0.0, -100.0], [1000.0, 0.0, -100.0], 1000.0)
    direct = _paths(rays, 0, 0)

    assert direct
    assert sum(r.amplitude for r in direct) == pytest.approx(1 / 1000.0, rel=1e-4)
    assert _weighted_delay_s(direct) == pytest.approx(1000.0 / 1500.0, abs=1e-5)
    assert all(r.phase_rad == pytest.approx(0.0, abs=1e-6) for r in direct)


@needs_bellhop
def test_bellhop_surface_reflection_is_a_phase_inverted_image_path(bp) -> None:
    """The surface-reflected path follows the image source above the surface (Lloyd's mirror).

    Its length is sqrt(R^2 + (z_s + z_r)^2), and reflecting off the pressure-release surface
    inverts it (phase pi).
    """
    rays = _model(bp).compute_eigenrays([0.0, 0.0, -100.0], [1000.0, 0.0, -150.0], 1000.0)
    surface = _paths(rays, 1, 0)
    image_path_m = np.hypot(1000.0, 100.0 + 150.0)

    assert surface
    assert sum(r.amplitude for r in surface) == pytest.approx(1 / image_path_m, rel=1e-4)
    assert _weighted_delay_s(surface) == pytest.approx(image_path_m / 1500.0, abs=1e-5)
    assert all(r.phase_rad == pytest.approx(np.pi, abs=1e-6) for r in surface)
