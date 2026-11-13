"""Tests for BellhopAcousticPropagationModel and Eigenray.

The file-format, spectrum and error tests use a stub executable, so they run without Bellhop.
Tests marked ``needs_bellhop`` run the real ``bellhopcxx`` and are skipped when it is not on
PATH (as in CI); ``needs_rtrs`` also needs the rtrs package.
"""

from __future__ import annotations

import dataclasses
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
    """Return a 2D ASCII arrivals file: one block of arrival rows per grid point.

    A row is ``(amplitude, phase_deg, delay_s, n_top, n_bottom)``, with an optional sixth
    entry for the imaginary delay Bellhop reports when it applies volume attenuation.
    """
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
            f"   {amp} {phase_deg} {delay} {imag_delay[0] if imag_delay else 0.0} 0.1 -0.1 "
            f"{top} {bot}"
            for amp, phase_deg, delay, top, bot, *imag_delay in block
        ]
    return "\n".join(lines) + "\n"


def _imag_delay_s(path_length_m, alpha_db_per_km, frequency_hz):
    """Return the imaginary delay Bellhop reports for a path absorbed at ``alpha_db_per_km``.

    Bellhop integrates the delay with a complex sound speed, which gives
    2 pi f Im(tau) = -alpha L with alpha in Np/m (1 Np/m = 8685.8896 dB/km).
    """
    return -alpha_db_per_km / 8685.8896 * path_length_m / (2 * np.pi * frequency_hz)


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
    from bluepebble.models.environment import Constant, FlatBathymetry, Linear
    from bluepebble.models.propagation import (
        AcousticPropagationModel,
        BellhopAcousticPropagationModel,
        Eigenray,
        SpectrumPropagationModel,
        bellhop,
        rtrsAcousticPropagationModel,
        thorp_db_per_km,
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
        Linear=Linear,
        FlatBathymetry=FlatBathymetry,
        thorp_db_per_km=thorp_db_per_km,
        transfer_function=bellhop._transfer_function,
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
    assert env[5:10] == [f"  {d:.2f}  1500.0000  /" for d in np.linspace(0, 4000, 5)]
    assert env[11] == "4000.00  1650.0  0.0  1.8  0.8  /"
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
        "'L'",
        "2",
        "0.000000 4000.00",
        "2.000000 3900.00",
    ]


def test_thorp_adds_bellhops_volume_attenuation_letter_to_the_top_option(bp, tmp_path) -> None:
    """With "thorp" the top option gains 'T', so Bellhop reports each path's absorption."""
    stem = tmp_path / "bp"
    _model(bp, volume_attenuation="thorp")._write_env(
        stem, 100.0, np.array([50.0]), np.array([2000.0]), 4000.0, 1500.0
    )

    assert stem.with_suffix(".env").read_text().splitlines()[3] == "'SVWT'"


def test_parse_arr_turns_the_imaginary_delay_into_a_path_length_only_when_asked(
    bp, tmp_path
) -> None:
    """The path length is the imaginary delay times the given ratio, and None without one."""
    arr = tmp_path / "x.arr"
    arr.write_text(_arr_text([[(1e-4, 0.0, 6.7, 0, 0, -2.0e-5)]]))

    ((plain,),) = bp.BellhopAcousticPropagationModel._parse_arr(arr)
    ((measured,),) = bp.BellhopAcousticPropagationModel._parse_arr(
        arr, imag_delay_to_path_length_mps=-5.0e8
    )

    assert plain.path_length_m is None
    assert measured.path_length_m == pytest.approx(10_000.0)


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


@pytest.mark.parametrize(("geometry", "runs"), [("reference", 1), ("per_sensor", 3)])
def test_receiver_geometry_sets_the_number_of_runs(bp, tmp_path, geometry, runs) -> None:
    """One Bellhop run for the whole array with "reference", one per sensor otherwise."""
    if geometry == "reference":
        arr = _arr_text([ONE_RECEIVER[0]] * 3, ranges_m=(1000.0, 1001.0, 1002.0))
    else:
        arr = _arr_text(ONE_RECEIVER)
    model = _model(
        bp, receiver_geometry=geometry, exe_path=_stub_writing(tmp_path, arr, log_runs=True)
    )
    sensors = [[1000.0, 1001.0, 1002.0], [0.0, 0.0, 0.0], [-100.0, -100.0, -100.0]]

    model.propagate_spectrum(_platform(sensors), _source([0.0, 0.0, -100.0]), np.array([100.0]))

    assert (tmp_path / "runs.log").read_text().count("run") == runs


def test_propagate_reports_loss_and_earliest_arrival(bp, tmp_path) -> None:
    """propagate() gives -20 log10 |H| at the loudest tonal and the earliest arrival."""
    model = _model(bp, exe_path=_stub_writing(tmp_path, _arr_text(ONE_RECEIVER)))
    source = _source([0.0, 0.0, -100.0], frequencies_hz=(200.0, 400.0), amplitudes_upa=(1.0, 5.0))
    platform = _platform([[1000.0], [0.0], [-100.0]])

    loss_db, time_s = model.propagate(platform, source)
    H, _ = model.propagate_spectrum(platform, source, np.array([400.0]))

    assert loss_db == pytest.approx(-20 * np.log10(abs(H[0, 0])))
    assert time_s == pytest.approx(0.666658)


def test_invalid_receiver_geometry_raises(bp) -> None:
    """Only the two documented receiver geometries are accepted."""
    with pytest.raises(ValueError, match="receiver_geometry"):
        _model(bp, receiver_geometry="nearest")


# ---------------------------------------------------------------------------
# Seawater absorption
# ---------------------------------------------------------------------------

# Thorp's absorption (Jensen et al. Eq. 1.34) worked by hand, in dB/km.
THORP_400_HZ = 0.0202374
THORP_3200_HZ = 0.2162044
FFT_GRID_HZ = np.fft.fftfreq(64, 1 / 12800.0)  # Bellhop runs at its centre, 3200 Hz


def test_invalid_volume_attenuation_raises(bp) -> None:
    """Only "none" and "thorp" are accepted."""
    with pytest.raises(ValueError, match="volume_attenuation"):
        _model(bp, volume_attenuation="francois-garrison")


def test_default_results_are_those_of_v0_8_0(bp, tmp_path) -> None:
    """Without absorption, the transfer function and loss are unchanged from v0.8.0.

    The values were saved from the v0.8.0 code. They are compared to rounding and not bit
    for bit, since the order BLAS sums the arrivals in can differ between platforms.
    """
    exe = _stub_writing(tmp_path, _arr_text(ONE_RECEIVER))
    platform = _platform([[1000.0], [0.0], [-100.0]])
    source = _source([0.0, 0.0, -100.0], frequencies_hz=(200.0, 400.0), amplitudes_upa=(1.0, 5.0))

    H, delays = _model(bp, exe_path=exe).propagate_spectrum(platform, source, FFT_GRID_HZ)
    loss_db, time_s = _model(bp, exe_path=exe).propagate(platform, source)
    losses_db, _ = _model(bp, exe_path=exe, use_all_frequencies=True).propagate(platform, source)

    np.testing.assert_allclose(
        H[0, [0, 1, 16, 31, 32, 33, 63]],
        [
            3.172719529952874e-05 + 0j,
            -0.000745883535737235 - 0.0005211413327306173j,
            0.00031048702949672795 - 0.0006317033853083173j,
            -0.000439378603821695 - 0.00012125989088183757j,
            -0.0007534379011726957 - 0.000608047601094006j,
            -0.000439378603821695 + 0.00012125989088183757j,
            -0.000745883535737235 + 0.0005211413327306173j,
        ],
        rtol=1e-13,
    )
    assert delays[0] == 0.666658 and time_s == 0.666658
    assert loss_db == pytest.approx(62.566225654384546, rel=1e-13)
    np.testing.assert_allclose(losses_db, [60.82006098502632, 62.566225654384546], rtol=1e-13)


def test_thorp_absorbs_each_frequency_bin_by_its_own_amount(bp, tmp_path) -> None:
    """|H| falls by alpha(|f|) R at every bin, whatever frequency Bellhop ran at.

    One direct arrival over R = 41 km, on the full complex FFT grid a simulator passes. The
    stub reports the path as Bellhop does, as the imaginary delay at the 3200 Hz run frequency.
    """
    R, amplitude = 41_000.0, 1 / 41_000.0
    row = (amplitude, 0.0, R / C, 0, 0, _imag_delay_s(R, THORP_3200_HZ, 3200.0))
    exe = _stub_writing(tmp_path, _arr_text([[row]]))
    platform, source = _platform([[R], [0.0], [-100.0]]), _source([0.0, 0.0, -100.0])

    H_without, _ = _model(bp, exe_path=exe).propagate_spectrum(platform, source, FFT_GRID_HZ)
    H_with, delays = _model(bp, exe_path=exe, volume_attenuation="thorp").propagate_spectrum(
        platform, source, FFT_GRID_HZ
    )

    np.testing.assert_allclose(np.abs(H_without[0]), amplitude)
    alpha_db_per_km = bp.thorp_db_per_km(FFT_GRID_HZ)
    np.testing.assert_allclose(
        -20 * np.log10(np.abs(H_with[0]) / np.abs(H_without[0])),
        alpha_db_per_km * R / 1000,
        rtol=1e-6,
    )
    # 17.6 dB at 5400 Hz against 8.9 dB at the run frequency: not one absorption for all bins.
    assert -20 * np.log10(np.abs(H_with[0, 27]) * R) == pytest.approx(0.4291274 * 41, rel=1e-6)
    np.testing.assert_allclose(np.angle(H_with[0]), np.angle(H_without[0]), atol=1e-12)
    positive, negative = FFT_GRID_HZ[1:32], FFT_GRID_HZ[:32:-1]
    np.testing.assert_array_equal(positive, -negative)
    np.testing.assert_allclose(H_with[0, :32:-1], np.conj(H_with[0, 1:32]))
    assert H_with[0, 0].imag == 0
    assert delays[0] == pytest.approx(R / C)


def test_thorp_raises_the_propagate_loss_by_the_absorption_at_each_tonal(bp, tmp_path) -> None:
    """propagate() adds alpha(f) L to the geometric loss, at the tonal Bellhop ran at."""
    L = 20_000.0
    row = (1 / L, 0.0, L / C, 0, 0, _imag_delay_s(L, THORP_400_HZ, 400.0))
    exe = _stub_writing(tmp_path, _arr_text([[row]]))
    platform = _platform([[L], [0.0], [-100.0]])
    source = _source([0.0, 0.0, -100.0], frequencies_hz=(400.0,))

    loss_db, _ = _model(bp, exe_path=exe, volume_attenuation="thorp").propagate(platform, source)

    assert loss_db == pytest.approx(20 * np.log10(L) + THORP_400_HZ * 20, abs=1e-5)


def test_arrival_absorption_is_one_without_absorption_and_needs_path_lengths_with_it(bp) -> None:
    """The helper returns ones for "none", and rejects arrivals that carry no path length."""
    rays = [bp.Eigenray(1e-3, 0.0, 1.0, 0.0, 0.0, 0, 0)] * 2
    f = np.array([-500.0, 0.0, 500.0])

    np.testing.assert_array_equal(_model(bp)._arrival_absorption(rays, f), np.ones((2, 3)))
    with pytest.raises(ValueError, match="path_length_m"):
        _model(bp, volume_attenuation="thorp")._arrival_absorption(rays, f)


def test_transfer_function_takes_absorption_by_keyword_only(bp) -> None:
    """The two-argument call downstream code uses still works; absorption scales arrivals."""
    rays = [
        bp.Eigenray(2e-3, 0.0, 1.0, 0.0, 0.0, 0, 0),
        bp.Eigenray(1e-3, np.pi, 1.5, 0.0, 0.0, 1, 0),
    ]
    f = np.array([250.0, 500.0])

    plain = bp.transfer_function(rays, f)
    first_only = bp.transfer_function(rays, f, absorption=np.array([[1.0, 0.5], [0.0, 0.0]]))

    np.testing.assert_allclose(
        plain, 2e-3 * np.exp(-2j * np.pi * f) - 1e-3 * np.exp(-3j * np.pi * f)
    )
    np.testing.assert_allclose(first_only, [2e-3, 1e-3] * np.exp(-2j * np.pi * f))
    with pytest.raises(TypeError):
        bp.transfer_function(rays, f, np.ones((2, 2)))


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


def _weighted_path_length_m(rays):
    """Amplitude-weighted mean path length of the arrivals making up one geometric path."""
    amplitudes = np.array([r.amplitude for r in rays])
    return float(np.dot(amplitudes, [r.path_length_m for r in rays]) / amplitudes.sum())


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
    H_per, d_per = _model(bp, depth=-200.0, receiver_geometry="per_sensor").propagate_spectrum(
        _platform(sensors), source, f
    )

    np.testing.assert_allclose(H_ref, H_per, rtol=0, atol=1e-12 * np.abs(H_per).max())
    np.testing.assert_allclose(d_ref, d_per)


@needs_bellhop
@pytest.mark.parametrize("frequency_hz", [500.0, 1000.0, 5400.0, 10000.0])
def test_bellhops_thorp_absorption_is_thorp_db_per_km(bp, frequency_hz) -> None:
    """In isovelocity water each path's length is its straight-line (or image) distance.

    The path length is Bellhop's own absorption, read from its imaginary delay, divided by
    ``thorp_db_per_km``, so it equals the geometric distance only if the two formulas agree.
    """
    rays = _model(bp, volume_attenuation="thorp").compute_eigenrays(
        [0.0, 0.0, -100.0], [10_000.0, 0.0, -150.0], frequency_hz
    )
    direct, surface = _paths(rays, 0, 0), _paths(rays, 1, 0)

    assert direct and surface
    # Bellhop stores delays in single precision: both came out 3e-7 short.
    assert _weighted_path_length_m(direct) == pytest.approx(np.hypot(10_000.0, 50.0), rel=1e-5)
    assert _weighted_path_length_m(surface) == pytest.approx(np.hypot(10_000.0, 250.0), rel=1e-5)


@needs_bellhop
def test_thorp_leaves_bellhops_arrivals_geometric(bp) -> None:
    """Asking Bellhop for absorption changes only what it adds to the seabed's attenuation.

    Paths that miss the bottom keep their amplitude, phase and delay exactly. Bellhop adds the
    Thorp term to the half-space too, which moved the bottom-reflected amplitudes by under
    1e-6 of their value and their phases by under 0.002 degrees.
    """
    source, receiver = [0.0, 0.0, -100.0], [10_000.0, 0.0, -150.0]

    def sorted_arrivals(**kwargs):
        rays = _model(bp, **kwargs).compute_eigenrays(source, receiver, 5400.0)
        # bellhopcxx writes arrivals in a different order from run to run.
        return sorted(
            rays, key=lambda r: (r.n_surface_bounces, r.n_bottom_bounces, r.src_angle_deg)
        )

    plain, absorbed = sorted_arrivals(), sorted_arrivals(volume_attenuation="thorp")

    assert len(plain) == len(absorbed) and any(r.n_bottom_bounces for r in plain)
    for before, after in zip(plain, absorbed, strict=True):
        assert before.path_length_m is None and after.path_length_m > 10_000.0
        if before.n_bottom_bounces == 0:
            assert dataclasses.replace(after, path_length_m=None) == before
        else:
            assert after.amplitude == pytest.approx(before.amplitude, rel=1e-5)
            assert after.phase_rad == pytest.approx(before.phase_rad, abs=np.radians(0.01))
            assert after.delay_s == before.delay_s


@needs_bellhop
@pytest.mark.parametrize("range_m", [10e3, 25e3, 50e3, 60e3])
def test_refracted_path_is_absorbed_over_its_arc_length(bp, range_m) -> None:
    """In a linear profile the direct ray is a circular arc, whose length is analytic.

    With c = c0 + g z, a ray between two points at the same depth, a range r apart, leaves at
    tan(theta) = r g / (2 c_s) on a circle of radius c_s / (g cos(theta)), so its length is
    2 theta c_s / (g cos(theta)): 61.1 km at 60 km. Bellhop's path length came out within 6e-6
    of it, where the depth-mean sound speed times the travel time was up to 3% out.
    """
    c0, gradient, depth_m = 1500.0, 0.017, 100.0
    model = bp.BellhopAcousticPropagationModel(
        ssp=bp.Linear(surface_speed=c0, gradient=gradient),
        bathymetry=bp.FlatBathymetry(depth=-6000.0),
        volume_attenuation="thorp",
    )
    c_source = c0 + gradient * depth_m
    launch_rad = np.arctan(range_m * gradient / (2 * c_source))
    arc_m = 2 * launch_rad * c_source / (gradient * np.cos(launch_rad))

    direct = _paths(
        model.compute_eigenrays([0.0, 0.0, -depth_m], [range_m, 0.0, -depth_m], 5400.0), 0, 0
    )

    assert direct
    assert _weighted_path_length_m(direct) == pytest.approx(arc_m, rel=5e-5)
    amplitudes = np.array([r.amplitude for r in direct])
    absorbed = amplitudes @ model._arrival_absorption(direct, [5400.0])[:, 0]
    # 0.4291274 dB/km is Thorp's formula at 5.4 kHz, worked by hand.
    assert -20 * np.log10(absorbed / amplitudes.sum()) == pytest.approx(
        0.4291274 * arc_m / 1000, abs=0.005
    )


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
