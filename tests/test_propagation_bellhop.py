"""Tests for BellhopAcousticPropagationModel and Eigenray.

The file-format, spectrum and error tests use a stub executable, so they run without Bellhop.
Tests marked ``needs_bellhop`` run the real ``bellhopcxx`` and are skipped when it is not on
PATH (as in CI); ``needs_rtrs`` also needs the rtrs package.
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

try:
    import rtrs  # noqa: F401

    HAVE_RTRS = True
except ImportError:
    HAVE_RTRS = False

needs_bellhop = pytest.mark.skipif(which("bellhopcxx") is None, reason="bellhopcxx not on PATH")
needs_rtrs = pytest.mark.skipif(not HAVE_RTRS, reason="rtrs not installed")

C = 1500.0
# Bottom matched to the water, so it reflects nothing.
MATCHED_BOTTOM = dict(
    bottom_sound_speed_mps=1500.0,
    bottom_density_g_cm3=1.0,
    bottom_attenuation_db_per_wavelength=0.0,
)


def _arr_text(blocks, depths=(100.0,), ranges_m=(1000.0,)):
    """Return a 2D ASCII arrivals file: one block of arrival rows per grid point."""
    lines = [
        "'2D'",
        "   1000.0",
        "   1 100.0",
        f"   {len(depths)} " + " ".join(str(d) for d in depths),
        f"   {len(ranges_m)} " + " ".join(str(r) for r in ranges_m),
        f"   {max([len(b) for b in blocks] + [1])}",
    ]
    for block in blocks:
        lines.append(f"   {len(block)}")
        lines += [
            f"   {amp} {phase_deg} {delay} 0.0 0.1 -0.1 {top} {bot}"
            for amp, phase_deg, delay, top, bot in block
        ]
    return "\n".join(lines) + "\n"


# One receiver, 3 arrivals: direct, surface-reflected, and a two-surface one-bottom path.
ONE_RECEIVER = [
    [
        (5.0e-4, 0.0, 0.666658, 0, 0),
        (4.8821e-4, 180.0, 0.679860, 1, 0),
        (1.994e-5, 360.961, 5.242514, 2, 1),
    ]
]


@pytest.fixture
def bp():
    """Import the real model, environment and simulator classes without leaking caching."""
    snapshot = isolate_real_bluepebble_imports()
    from bluepebble.models.environment import Constant, FlatBathymetry
    from bluepebble.models.propagation import (
        AcousticPropagationModel,
        BellhopAcousticPropagationModel,
        Eigenray,
        SpectrumPropagationModel,
        rtrsAcousticPropagationModel,
    )
    from bluepebble.simulator import DiscretePassiveSonarArraySimulator

    yield SimpleNamespace(
        BellhopAcousticPropagationModel=BellhopAcousticPropagationModel,
        AcousticPropagationModel=AcousticPropagationModel,
        SpectrumPropagationModel=SpectrumPropagationModel,
        rtrsAcousticPropagationModel=rtrsAcousticPropagationModel,
        DiscretePassiveSonarArraySimulator=DiscretePassiveSonarArraySimulator,
        Eigenray=Eigenray,
        Constant=Constant,
        FlatBathymetry=FlatBathymetry,
    )
    restore_real_bluepebble_imports(snapshot)


def _model(bp, depth=-4000.0, **kwargs):
    return bp.BellhopAcousticPropagationModel(
        ssp=bp.Constant(speed=C), bathymetry=bp.FlatBathymetry(depth=depth), **kwargs
    )


def _platform(sensors_xyz, reference_xyz=None):
    """Return a minimal PlatformState stand-in: the array geometry propagation models read."""
    sensors = np.asarray(sensors_xyz, dtype=float)
    reference = (
        sensors.mean(axis=1, keepdims=True)
        if reference_xyz is None
        else np.asarray(reference_xyz, dtype=float).reshape(3, 1)
    )
    return SimpleNamespace(
        array=SimpleNamespace(state_vector=sensors, ref_state_vector=StateVector(reference))
    )


def _source(xyz, frequencies_hz=(500.0,), amplitudes_upa=(1.0,)):
    return GroundTruthState(
        StateVector(list(xyz)),
        metadata={
            "position_mapping": [0, 1, 2],
            "frequencies_hz": list(frequencies_hz),
            "amplitudes_upa": list(amplitudes_upa),
        },
    )


def _stub_exe(tmp_path: Path, body: str) -> str:
    """Write an executable shell script standing in for bellhopcxx.

    It is called as ``exe --memory=... stem``; the script sees the stem as ``$stem``.
    """
    exe = tmp_path / "fake_bellhopcxx"
    exe.write_text("#!/bin/sh\nfor stem; do :; done\n" + body)
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    return str(exe)


def _stub_writing(tmp_path: Path, arr_text: str, log_runs: bool = False) -> str:
    """Return a stub that writes ``arr_text`` as the arrivals file (and can count runs)."""
    (tmp_path / "canned.arr").write_text(arr_text)
    body = f'cp "{tmp_path}/canned.arr" "$stem.arr"\ncp "$stem.env" "{tmp_path}/seen.env"\n'
    if log_runs:
        body += f'echo run >> "{tmp_path}/runs.log"\n'
    return _stub_exe(tmp_path, body)


# ---------------------------------------------------------------------------
# Files and parsing
# ---------------------------------------------------------------------------


def test_parse_arr_reads_every_arrival_with_its_phase(bp, tmp_path) -> None:
    """Each arrival line becomes an Eigenray, with Bellhop's phase converted to radians."""
    arr = tmp_path / "x.arr"
    arr.write_text(_arr_text(ONE_RECEIVER))

    (rays,) = bp.BellhopAcousticPropagationModel._parse_arr(arr)

    assert len(rays) == 3
    direct, surface, bottom = rays
    assert direct == bp.Eigenray(5.0e-4, 0.0, 0.666658, 0.1, -0.1, 0, 0)
    assert surface.phase_rad == pytest.approx(np.pi)
    assert (surface.n_surface_bounces, surface.n_bottom_bounces) == (1, 0)
    assert bottom.phase_rad == pytest.approx(np.radians(360.961))


def test_parse_arr_returns_one_list_per_grid_point(bp, tmp_path) -> None:
    """A 2 x 3 receiver grid gives 6 blocks, range varying fastest."""
    blocks = [[(1e-3, 0.0, 0.1 * k, 0, 0)] for k in range(6)]
    arr = tmp_path / "x.arr"
    arr.write_text(_arr_text(blocks, depths=(50.0, 80.0), ranges_m=(100.0, 200.0, 300.0)))

    parsed = bp.BellhopAcousticPropagationModel._parse_arr(arr)

    assert [rays[0].delay_s for rays in parsed] == pytest.approx([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])


def test_env_and_bty_files_describe_the_geometry(bp, tmp_path) -> None:
    """The environment file holds the receiver grid, bottom, beam fan and beam count."""
    model = _model(
        bp,
        bottom_sound_speed_mps=1650.0,
        bottom_density_g_cm3=1.8,
        bottom_attenuation_db_per_wavelength=0.8,
        beam_angle_limits_deg=(-60.0, 60.0),
        num_beams=500,
        n_ssp_samples=5,
    )
    stem = tmp_path / "bp"
    model._write_env(
        stem, 100.0, np.array([50.0, 250.0]), np.array([2000.0, 2500.0]), 4000.0, 1500.0
    )
    model._write_bty(stem, np.array([0.0, 2.0]), np.array([4000.0, 3900.0]))

    env = stem.with_suffix(".env").read_text().splitlines()
    assert env[1] == "1500.0000"
    assert env[3] == "'SVW'"
    assert env[4] == "0 0.0 4000.000000"
    assert env[5:10] == [f"  {d:.6f}  1500.0000  /" for d in np.linspace(0, 4000, 5)]
    assert env[11] == "4000.000000  1650.0  0.0  1.8  0.8  /"
    assert env[12:18] == [
        "1",
        "100.0000 /",
        "2",
        "50.0000 250.0000 /",
        "2",
        "2.000000 2.500000 /",
    ]
    assert env[18:21] == ["'AG'", "500", "-60.0 60.0 /"]
    assert stem.with_suffix(".bty").read_text().splitlines() == [
        "'C'",
        "2",
        "0.000000 4000.000000",
        "2.000000 3900.000000",
    ]
    model._write_bty(stem, np.array([0.0, 2.0]), np.array([4000.0, 3900.0]), "L")
    assert stem.with_suffix(".bty").read_text().splitlines()[0] == "'L'"


def test_unknown_bathymetry_interpolation_is_rejected(bp) -> None:
    """bathymetry_interpolation must be curvilinear or linear."""
    with pytest.raises(ValueError, match="bathymetry_interpolation"):
        _model(bp, bathymetry_interpolation="spline")


def test_bathymetry_profile_runs_from_the_source_along_the_plane(bp) -> None:
    """Ranges run from 0 at the source to the far end, with positive water depths."""
    range_km, depth_m = _model(bp, n_bathy_samples=3)._bathymetry_profile(
        np.array([0.0, 0.0]), np.array([3000.0, 4000.0]), 5000.0
    )

    np.testing.assert_allclose(range_km, [0.0, 2.5, 5.0])
    np.testing.assert_allclose(depth_m, 4000.0)


def test_compute_eigenrays_converts_world_positions_to_bellhop_depths(bp, tmp_path) -> None:
    """World z (negative below the surface) is written as Bellhop's positive depth."""
    model = _model(bp, exe_path=_stub_writing(tmp_path, _arr_text(ONE_RECEIVER)))

    rays = model.compute_eigenrays([10.0, 20.0, -100.0], [610.0, 820.0, -250.0], 1000.0)

    assert len(rays) == 3
    env = (tmp_path / "seen.env").read_text().splitlines()
    run_type = env.index("'AG'")
    assert env[run_type - 5] == "100.0000 /" and env[run_type - 3] == "250.0000 /"
    assert env[run_type - 1] == "1.000000 /"  # 1 km between (10, 20) and (610, 820)


# ---------------------------------------------------------------------------
# Spectrum propagation interface
# ---------------------------------------------------------------------------


def test_is_a_spectrum_propagation_model_usable_by_the_passive_simulators(bp) -> None:
    """The model implements both propagation interfaces, so simulators accept it."""
    model = _model(bp)

    assert isinstance(model, bp.AcousticPropagationModel)
    assert isinstance(model, bp.SpectrumPropagationModel)
    simulator = bp.DiscretePassiveSonarArraySimulator(
        array=object(), propagation_model=model, signal_models=[object()]
    )
    assert simulator.propagation_model is model


def test_transfer_function_sums_arrivals_with_phase_and_delay(bp, tmp_path) -> None:
    """H(f) = sum a e^{i phi} e^{-i 2 pi f tau}, conjugate at -f, real at 0 Hz."""
    model = _model(bp, exe_path=_stub_writing(tmp_path, _arr_text(ONE_RECEIVER)))
    f = np.array([-300.0, 0.0, 300.0, 700.0])

    H, delays = model.propagate_spectrum(
        _platform([[1000.0], [0.0], [-100.0]]), _source([0.0, 0.0, -100.0]), f
    )

    amp, phase_deg, delay = (np.array([row[k] for row in ONE_RECEIVER[0]]) for k in range(3))
    expected_positive = np.sum(
        amp[:, None]
        * np.exp(1j * np.radians(phase_deg)[:, None])
        * np.exp(-2j * np.pi * delay[:, None] * f[None, 2:]),
        axis=0,
    )
    np.testing.assert_allclose(H[0, 2:], expected_positive)
    np.testing.assert_allclose(H[0, 0], np.conj(H[0, 2]))
    assert H[0, 1] == pytest.approx(np.sum(amp * np.cos(np.radians(phase_deg))))
    assert delays[0] == pytest.approx(0.666658)


def test_transfer_function_on_an_fft_grid_matches_the_direct_sum(bp, tmp_path) -> None:
    """On an evenly spaced grid the factored sum equals the direct one, at both signs and 0 Hz.

    Simulators pass FFT grids, where the transfer function is summed by factoring each
    frequency index into blocks; that must give the same numbers as summing every arrival at
    every frequency.
    """
    model = _model(bp, exe_path=_stub_writing(tmp_path, _arr_text(ONE_RECEIVER)))
    f = np.fft.fftfreq(4000, 1 / 8000.0)

    H, _ = model.propagate_spectrum(
        _platform([[1000.0], [0.0], [-100.0]]), _source([0.0, 0.0, -100.0]), f
    )

    amp, phase_deg, delay = (np.array([row[k] for row in ONE_RECEIVER[0]]) for k in range(3))
    sign = np.sign(f)
    expected = np.sum(
        amp[:, None]
        * np.where(
            sign[None, :] == 0,
            np.cos(np.radians(phase_deg))[:, None],
            np.exp(1j * sign[None, :] * np.radians(phase_deg)[:, None]),
        )
        * np.exp(-2j * np.pi * delay[:, None] * f[None, :]),
        axis=0,
    )
    np.testing.assert_allclose(H[0], expected, rtol=0, atol=1e-9 * np.abs(expected).max())


def test_transfer_function_on_an_fft_grid_with_gaps_matches_the_direct_sum(bp, tmp_path) -> None:
    """An FFT grid with bins left out, as a band threshold leaves it, matches the direct sum."""
    model = _model(bp, exe_path=_stub_writing(tmp_path, _arr_text(ONE_RECEIVER)))
    f = np.fft.fftfreq(4000, 1 / 8000.0)
    f = f[(f > 500) & (f < 3500) & ~((f > 1200) & (f < 1300)) & ~((f > 2000) & (f < 2010))]

    H, _ = model.propagate_spectrum(
        _platform([[1000.0], [0.0], [-100.0]]), _source([0.0, 0.0, -100.0]), f
    )

    amp, phase_deg, delay = (np.array([row[k] for row in ONE_RECEIVER[0]]) for k in range(3))
    expected = np.sum(
        amp[:, None]
        * np.exp(1j * np.radians(phase_deg)[:, None])
        * np.exp(-2j * np.pi * delay[:, None] * f[None, :]),
        axis=0,
    )
    np.testing.assert_allclose(H[0], expected, rtol=0, atol=1e-9 * np.abs(expected).max())


def test_max_workers_below_one_is_rejected(bp) -> None:
    """max_workers must be at least 1, or None."""
    with pytest.raises(ValueError, match="max_workers"):
        _model(bp, separate_planes=True, max_workers=0)


def test_receivers_are_mapped_to_their_grid_points(bp, tmp_path) -> None:
    """Sensors in any order get the arrivals at their own (depth, range) grid point."""
    # Grid: depths (50, 80) x ranges (1000, 2000); each block's delay names its grid point.
    blocks = [[(1e-3, 0.0, delay, 0, 0)] for delay in (1.0, 2.0, 3.0, 4.0)]
    model = _model(
        bp,
        exe_path=_stub_writing(
            tmp_path, _arr_text(blocks, depths=(50.0, 80.0), ranges_m=(1000.0, 2000.0))
        ),
    )
    sensors = [[2000.0, 1000.0, 1000.0], [0.0, 0.0, 0.0], [-50.0, -80.0, -50.0]]

    _, delays = model.propagate_spectrum(
        _platform(sensors), _source([0.0, 0.0, -10.0]), np.array([500.0])
    )

    # (50 m, 2000 m) is block 1, (80 m, 1000 m) block 2, (50 m, 1000 m) block 0.
    np.testing.assert_allclose(delays, [2.0, 3.0, 1.0])


def test_sensor_with_no_arrivals_gets_zero_and_nan_delay(bp, tmp_path) -> None:
    """A receiver Bellhop found no path to contributes nothing and has no delay."""
    model = _model(bp, exe_path=_stub_writing(tmp_path, _arr_text([[]])))

    H, delays = model.propagate_spectrum(
        _platform([[1000.0], [0.0], [-100.0]]), _source([0.0, 0.0, -100.0]), np.array([100.0])
    )

    assert H[0, 0] == 0
    assert np.isnan(delays[0])


@pytest.mark.parametrize(
    ("settings", "runs"),
    [
        ({}, 1),
        ({"separate_planes": True}, 3),
        ({"far_field": True}, 1),
    ],
)
def test_settings_set_the_number_of_runs(bp, tmp_path, settings, runs) -> None:
    """One Bellhop run for the whole array, except separate_planes=True, one per plane.

    The sensors are on three bearings from the source, so each has its own plane.
    """
    if settings == {}:
        arr = _arr_text([ONE_RECEIVER[0]] * 3, ranges_m=(1000.0, 1000.5, 1002.0))
    else:
        arr = _arr_text(ONE_RECEIVER)
    model = _model(bp, **settings, exe_path=_stub_writing(tmp_path, arr, log_runs=True))
    sensors = [[1000.0, 1000.0, 1002.0], [0.0, 30.0, -60.0], [-100.0, -100.0, -100.0]]

    model.propagate_spectrum(_platform(sensors), _source([0.0, 0.0, -100.0]), np.array([100.0]))

    assert (tmp_path / "runs.log").read_text().count("run") == runs


def test_separate_planes_shares_a_run_between_sensors_in_the_same_plane(bp) -> None:
    """Sensors on the same bearing from the source, at any range or depth, share one run."""
    model = _model(bp, separate_planes=True)
    sensors = np.array(
        [[1000.0, 1000.0, 2000.0, 1000.0], [0.0, 0.0, 0.0, 30.0], [-100.0, -50.0, -100.0, -100.0]]
    )

    groups = model._plane_groups(np.array([0.0, 0.0, -10.0]), sensors)

    assert sorted(sorted(group.tolist()) for group, _ in groups) == [[0, 1, 2], [3]]


def test_plane_tolerance_shares_runs_between_nearby_planes(bp) -> None:
    """A tolerance groups planes within it, splitting groups that span more than it."""
    bearings_deg = np.array([0.0, 0.004, 0.008, 0.02, 0.05])
    sensors = np.stack(
        [
            1000 * np.cos(np.radians(bearings_deg)),
            1000 * np.sin(np.radians(bearings_deg)),
            np.full(5, -100.0),
        ]
    )
    source = np.array([0.0, 0.0, -10.0])

    groups = {
        tolerance: sorted(
            sorted(group.tolist())
            for group, _ in _model(
                bp, separate_planes=True, plane_tolerance_deg=tolerance
            )._plane_groups(source, sensors)
        )
        for tolerance in (0.0, 0.01, 1.0)
    }

    assert groups[0.0] == [[0], [1], [2], [3], [4]]
    assert groups[0.01] == [[0, 1, 2], [3], [4]]
    assert groups[1.0] == [[0, 1, 2, 3, 4]]


def test_plane_groups_do_not_straddle_the_bearing_wrap(bp) -> None:
    """Planes either side of 180 degrees from the source group together."""
    model = _model(bp, separate_planes=True, plane_tolerance_deg=0.1)
    sensors = np.array([[-1000.0, -1000.0], [0.5, -0.5], [-100.0, -100.0]])

    groups = model._plane_groups(np.array([0.0, 0.0, -10.0]), sensors)

    assert len(groups) == 1


def test_negative_plane_tolerance_is_rejected(bp) -> None:
    """A negative plane_tolerance_deg raises a ValueError."""
    with pytest.raises(ValueError, match="plane_tolerance_deg"):
        _model(bp, separate_planes=True, plane_tolerance_deg=-0.1)


def test_far_field_shifts_the_reference_arrivals_by_plane_wave_delays(bp, tmp_path) -> None:
    """Each sensor gets the reference point's arrivals, delayed by -(r_n - r_ref) . u / c."""
    model = _model(
        bp,
        far_field=True,
        exe_path=_stub_writing(tmp_path, _arr_text(ONE_RECEIVER)),
    )
    reference = np.array([1000.0, 0.0, -100.0])
    sensors = reference[:, None] + np.array([[-3.0, 0.0, 2.0], [1.0, 0.0, -4.0], [0.5, 0.0, 1.0]])
    source = np.array([0.0, 0.0, -40.0])
    f = np.array([-300.0, 300.0, 700.0])

    H, delays = model.propagate_spectrum(_platform(sensors, reference), _source(source), f)

    u = (source - reference) / np.linalg.norm(source - reference)
    offsets_s = -((sensors - reference[:, None]).T @ u) / C
    H_ref, d_ref = _model(
        bp, exe_path=_stub_writing(tmp_path, _arr_text(ONE_RECEIVER))
    ).propagate_spectrum(_platform(reference[:, None]), _source(source), f)
    np.testing.assert_allclose(delays, d_ref[0] + offsets_s)
    np.testing.assert_allclose(H, H_ref * np.exp(-2j * np.pi * offsets_s[:, None] * f[None]))
    assert delays[0] < d_ref[0] < delays[2]  # sensor 0 is displaced towards the source


def test_far_field_rejects_a_source_at_the_reference_point(bp, tmp_path) -> None:
    """The plane-wave direction is undefined when the source sits on the reference point."""
    model = _model(bp, far_field=True)
    with pytest.raises(ValueError, match="reference point"):
        model._plane_wave_offsets_s(np.zeros(3), np.zeros((3, 2)), np.zeros(3))


def test_propagate_reports_loss_and_earliest_arrival(bp, tmp_path) -> None:
    """propagate() gives -20 log10 |H| at the loudest tonal and the earliest arrival."""
    model = _model(bp, exe_path=_stub_writing(tmp_path, _arr_text(ONE_RECEIVER)))
    source = _source([0.0, 0.0, -100.0], frequencies_hz=(200.0, 400.0), amplitudes_upa=(1.0, 5.0))
    platform = _platform([[1000.0], [0.0], [-100.0]])

    loss_db, time_s = model.propagate(platform, source)
    H, _ = model.propagate_spectrum(platform, source, np.array([400.0]))

    assert loss_db == pytest.approx(-20 * np.log10(abs(H[0, 0])))
    assert time_s == pytest.approx(0.666658)


def test_defaults_trace_every_sensor_in_one_shared_plane(bp) -> None:
    """With neither setting given, one run in the reference plane traces to every sensor."""
    model = _model(bp)
    assert (model.far_field, model.separate_planes) == (False, False)


@pytest.mark.parametrize(
    ("settings", "match"),
    [
        ({"receiver_geometry": "nearest"}, "receiver_geometry"),
        ({"receiver_geometry": "per_sensor", "far_field": True}, "not both"),
    ],
)
def test_invalid_settings_raise(bp, settings, match) -> None:
    """An unknown receiver_geometry, or one mixed with the new settings, is rejected."""
    with pytest.raises(ValueError, match=match):
        _model(bp, **settings)


def test_separate_planes_with_far_field_falls_back(bp) -> None:
    """Far field has one receiver, so one plane: separate_planes is dropped, with a warning."""
    with pytest.warns(UserWarning, match="ignoring separate_planes=True"):
        model = _model(bp, separate_planes=True, far_field=True)
    assert (model.far_field, model.separate_planes) == (True, False)


@pytest.mark.parametrize(
    ("geometry", "far_field", "separate_planes", "replacement"),
    [
        ("reference", False, False, "the defaults"),
        ("per_sensor", False, True, "separate_planes=True"),
        ("far_field", True, False, "far_field=True"),
    ],
)
def test_deprecated_receiver_geometry_maps_to_the_flags(
    bp, geometry, far_field, separate_planes, replacement
) -> None:
    """receiver_geometry still works, with a DeprecationWarning naming its replacement."""
    with pytest.warns(DeprecationWarning, match=f"use {replacement} instead"):
        model = _model(bp, receiver_geometry=geometry)
    assert (model.far_field, model.separate_planes) == (far_field, separate_planes)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Against analytic solutions and rtrs (real bellhopcxx)
# ---------------------------------------------------------------------------


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
def test_arrivals_over_a_slope_converge_as_bathymetry_points_are_added(bp) -> None:
    """Over a plane slope, doubling the bathymetry points leaves the transfer function unchanged.

    A plane is exact at any number of points, so the answer shouldn't move. When depths were
    rounded to the centimetre, every point kinked the slope, and the bottom-reflected arrivals
    changed by -6 to -15 dB as points were added instead of converging.
    """
    from bluepebble.models.environment import WedgeBathymetry

    slope = WedgeBathymetry(depth_at_origin=-300.0, x_gradient=0.0, y_gradient=-0.08)
    source, receiver = np.array([2165.0, -1250.0, -80.0]), np.array([[0.0], [0.0], [-11.0]])
    f = np.linspace(1000.0, 3000.0, 201)
    transfer = {
        n: bp.BellhopAcousticPropagationModel(
            ssp=bp.Constant(speed=C),
            bathymetry=slope,
            bottom_attenuation_db_per_wavelength=0.0,
            n_bathy_samples=n,
        ).propagate_spectrum(_platform(receiver), _source(source), f)[0]
        for n in (200, 400)
    }

    difference = np.sum(np.abs(transfer[200] - transfer[400]) ** 2)
    assert 10 * np.log10(difference / np.sum(np.abs(transfer[400]) ** 2)) < -40.0


@needs_bellhop
def test_arrivals_over_a_rounded_seamount_converge_with_curvilinear_bathymetry(bp) -> None:
    """Over a curved seabed, curvilinear bathymetry converges with a few hundred points.

    The path runs right over the summit, the hardest case. Straight segments act as flat
    mirrors, so with linear bathymetry the arrivals were still changing by as much as the
    answer itself from 400 to 800 points; curvilinear converges to about -35 dB there.
    """
    from bluepebble.models.environment import Munk, SeamountBathymetry

    seamount = SeamountBathymetry(
        summit_position=(1200.0, 700.0, -150.0),
        radius=1500.0,
        plateau_depth=-300.0,
        shape="gaussian",
    )
    source, receiver = np.array([1732.0, 1000.0, -80.0]), np.array([[3.9], [0.3], [-11.0]])
    f = np.linspace(1000.0, 3000.0, 201)
    transfer = {
        n: bp.BellhopAcousticPropagationModel(
            ssp=Munk(),
            bathymetry=seamount,
            bottom_attenuation_db_per_wavelength=0.0,
            n_bathy_samples=n,
        ).propagate_spectrum(_platform(receiver), _source(source), f)[0]
        for n in (400, 800)
    }

    difference = np.sum(np.abs(transfer[400] - transfer[800]) ** 2)
    assert 10 * np.log10(difference / np.sum(np.abs(transfer[800]) ** 2)) < -30.0


@needs_bellhop
def test_parallel_separate_planes_match_runs_one_at_a_time(bp) -> None:
    """Running the planes at once gives each sensor the same arrivals as one at a time.

    Multithreaded ``bellhopcxx`` lists a receiver's arrivals in a varying order, even run by
    run, so they are compared as sets.
    """
    bearings_rad = np.radians([0.0, 5.0, 10.0, 15.0, 20.0])
    sensors = np.stack(
        [
            np.concatenate([1000 * np.cos(bearings_rad), 1500 * np.cos(bearings_rad)]),
            np.concatenate([1000 * np.sin(bearings_rad), 1500 * np.sin(bearings_rad)]),
            np.full(10, -100.0),
        ]
    )
    platform, source = _platform(sensors), _source([0.0, 0.0, -50.0])

    one_at_a_time = _model(bp, separate_planes=True, max_workers=1)._sensor_arrivals(
        platform, source, 1000.0
    )
    in_parallel = _model(bp, separate_planes=True, max_workers=4)._sensor_arrivals(
        platform, source, 1000.0
    )

    def by_delay(arrivals):
        return [sorted(rays, key=lambda r: (r.delay_s, r.amplitude)) for rays in arrivals]

    assert by_delay(in_parallel) == by_delay(one_at_a_time)
    assert all(rays for rays in in_parallel)


@needs_bellhop
def test_bellhop_direct_path_matches_spherical_spreading(bp) -> None:
    """In deep isovelocity water the direct path has delay R/c, amplitude 1/R and no phase."""
    rays = _model(bp).compute_eigenrays([0.0, 0.0, -100.0], [1000.0, 0.0, -100.0], 1000.0)
    direct = _paths(rays, 0, 0)

    assert direct
    assert sum(r.amplitude for r in direct) == pytest.approx(1 / 1000.0, rel=1e-4)
    assert _weighted_delay_s(direct) == pytest.approx(1000.0 / C, abs=1e-5)
    assert all(r.phase_rad == pytest.approx(0.0, abs=1e-6) for r in direct)


@needs_bellhop
def test_bellhop_surface_reflection_is_a_phase_inverted_image_path(bp) -> None:
    """The surface-reflected path follows the image source above the surface, inverted."""
    rays = _model(bp).compute_eigenrays([0.0, 0.0, -100.0], [1000.0, 0.0, -150.0], 1000.0)
    surface = _paths(rays, 1, 0)
    image_path_m = np.hypot(1000.0, 100.0 + 150.0)

    assert surface
    assert sum(r.amplitude for r in surface) == pytest.approx(1 / image_path_m, rel=1e-4)
    assert _weighted_delay_s(surface) == pytest.approx(image_path_m / C, abs=1e-5)
    assert all(r.phase_rad == pytest.approx(np.pi, abs=1e-6) for r in surface)


@needs_bellhop
def test_spectrum_matches_lloyds_mirror(bp) -> None:
    """With a non-reflecting bottom, H(f) is the analytic direct-minus-image field.

    This fixes the delay convention e^{-i 2 pi f tau}: the wrong sign would be wrong by
    order 100%.
    """
    model = _model(bp, **MATCHED_BOTTOM)
    source = np.array([0.0, 0.0, -100.0])
    sensors = np.array([[1000.0, 1002.5, 1005.0], [300.0] * 3, [-150.0] * 3])
    f = np.linspace(200.0, 1000.0, 81)

    H, delays = model.propagate_spectrum(_platform(sensors), _source(source), f)

    r_direct = np.linalg.norm(sensors - source[:, None], axis=0)
    r_image = np.linalg.norm(sensors - (source * [1, 1, -1])[:, None], axis=0)
    analytic = (
        np.exp(-2j * np.pi * f[None] * r_direct[:, None] / C) / r_direct[:, None]
        - np.exp(-2j * np.pi * f[None] * r_image[:, None] / C) / r_image[:, None]
    )
    assert np.abs(H - analytic).max() / np.abs(analytic).max() < 0.01
    np.testing.assert_allclose(delays, r_direct / C, atol=5e-6)


@needs_bellhop
def test_bottom_reflection_phase_is_the_rayleigh_coefficient(bp) -> None:
    """Below the critical angle a fluid bottom reflects totally, with the Rayleigh phase.

    In Blue Pebble's convention (signals e^{+i omega t}, delay e^{-i 2 pi f tau}) the
    reflection coefficient's phase is positive; Bellhop's reported phase matches it, which is
    why H uses e^{+i phi}.
    """
    model = _model(
        bp,
        depth=-200.0,
        bottom_sound_speed_mps=1700.0,
        bottom_density_g_cm3=1.5,
        bottom_attenuation_db_per_wavelength=0.0,
    )
    rays = model.compute_eigenrays([0.0, 0.0, -100.0], [2000.0, 0.0, -100.0], 1000.0)
    bottom = _paths(rays, 0, 1)

    grazing = np.arctan2(200.0, 2000.0)
    root = np.sqrt(np.cos(grazing) ** 2 - (C / 1700.0) ** 2)  # real below the critical angle
    reflection = (1.5 * np.sin(grazing) + 1j * root) / (1.5 * np.sin(grazing) - 1j * root)
    amplitudes = np.array([r.amplitude for r in bottom])
    phase = np.angle(np.sum(amplitudes * np.exp(1j * np.array([r.phase_rad for r in bottom]))))

    assert abs(reflection) == pytest.approx(1.0)
    assert phase == pytest.approx(np.angle(reflection), abs=np.radians(0.5))
    assert amplitudes.sum() == pytest.approx(1 / np.hypot(2000.0, 200.0), rel=1e-3)


@needs_bellhop
def test_reference_and_per_sensor_geometry_agree_over_a_flat_bottom(bp) -> None:
    """With bathymetry the same on every bearing, one shared run equals one run per sensor."""
    rng = np.random.default_rng(1)
    sensors = np.vstack(
        [1900 + rng.uniform(0, 100, 10), 50 + rng.uniform(-20, 20, 10), -rng.uniform(30, 150, 10)]
    )
    source = _source([0.0, 0.0, -40.0])
    f = np.linspace(200.0, 1000.0, 41)

    H_ref, d_ref = _model(bp, depth=-200.0).propagate_spectrum(_platform(sensors), source, f)
    H_per, d_per = _model(bp, depth=-200.0, separate_planes=True).propagate_spectrum(
        _platform(sensors), source, f
    )

    np.testing.assert_allclose(H_ref, H_per, rtol=0, atol=1e-12 * np.abs(H_per).max())
    np.testing.assert_allclose(d_ref, d_per)


@needs_bellhop
def test_far_field_approaches_reference_geometry_for_a_distant_source(bp) -> None:
    """For a small array far from the source, plane-wave delays match per-sensor ray tracing."""
    rng = np.random.default_rng(2)
    reference = np.array([5000.0, 0.0, -100.0])
    sensors = reference[:, None] + rng.uniform(-2.0, 2.0, (3, 8))
    source = _source([0.0, 0.0, -100.0])
    f = np.linspace(400.0, 600.0, 21)

    _, d_ref = _model(bp, depth=-4000.0).propagate_spectrum(
        _platform(sensors, reference), source, f
    )
    _, d_far = _model(bp, depth=-4000.0, far_field=True).propagate_spectrum(
        _platform(sensors, reference), source, f
    )

    # Differences across the array, which is what beamforming sees, agree to well under a
    # sample at 48 kHz; curvature over a 4 m aperture at 5 km is about 1 mm.
    np.testing.assert_allclose(d_far - d_far.mean(), d_ref - d_ref.mean(), atol=2e-6)


@needs_bellhop
@needs_rtrs
def test_bellhop_agrees_with_rtrs_in_the_same_environment(bp) -> None:
    """Over the same sediment half-space, Bellhop and a converged rtrs give the same H(f).

    rtrs needs a finer angular resolution than its default to converge here.
    """
    ssp, bathymetry = bp.Constant(speed=C), bp.FlatBathymetry(depth=-200.0)
    bellhop = bp.BellhopAcousticPropagationModel(
        ssp=ssp,
        bathymetry=bathymetry,
        bottom_sound_speed_mps=1700.0,
        bottom_density_g_cm3=1.5,
        bottom_attenuation_db_per_wavelength=0.5,
    )
    rtrs_model = bp.rtrsAcousticPropagationModel(
        ssp=ssp,
        bathymetry=bathymetry,
        step_m=5.0,
        elevation_resolution=0.05,
        azimuth_resolution=0.05,
        water_density_g_cm3=1.0,
        bottom_model={
            "model": "acoustic",
            "compressional_speed_m_s": 1700.0,
            "density_g_cm3": 1.5,
            "compressional_attenuation_db_per_wavelength": 0.5,
        },
    )
    sensors = np.vstack([2000.0 - np.arange(8) * 2.5, np.full(8, 50.0), np.full(8, -60.0)])
    source = _source([0.0, 0.0, -40.0])
    f = np.linspace(200.0, 1000.0, 81)

    H_bellhop, _ = bellhop.propagate_spectrum(_platform(sensors), source, f)
    H_rtrs, _ = rtrs_model.propagate_spectrum(_platform(sensors), source, f)

    rms_difference = np.sqrt(np.mean(np.abs(H_bellhop - H_rtrs) ** 2))
    assert rms_difference / np.sqrt(np.mean(np.abs(H_rtrs) ** 2)) < 0.03
    level_difference_db = 20 * np.log10(np.abs(H_bellhop) / np.abs(H_rtrs))
    assert np.median(np.abs(level_difference_db)) < 0.2
