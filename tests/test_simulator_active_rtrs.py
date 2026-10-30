"""Tests for the RTRS two-hop Fourier-synthesis active sonar simulator."""

from __future__ import annotations

import sys
from datetime import datetime
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from stonesoup.types.array import StateVector
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState

from .support import isolate_real_bluepebble_imports, restore_real_bluepebble_imports

T0 = datetime(2024, 1, 1)


@pytest.fixture(scope="module")
def bp():
    """Import the bluepebble modules under test without leaking caching into other tests.

    See ``isolate_real_bluepebble_imports`` for why this matters: a plain top-level
    ``from bluepebble.simulator.active_rtrs import ...`` forces a real import of the
    ``bluepebble.simulator`` and ``bluepebble.signal``/``bluepebble.types`` packages, which
    permanently caches attributes on the shared ``bluepebble`` package and breaks other tests
    (``test_package_api.py``'s lazy-import assertions, ``test_simulator_acoustic.py``'s
    fake-dependency-loaded copy of ``continuous.py``) if done at module import time.

    Module-scoped (imported once, cleaned up once) rather than per-test: re-importing
    ``bluepebble.signal.active`` fresh on every test collides with other test files' real
    ``stonesoup.base`` submodule caching in ways that otherwise raise spurious
    ``AttributeError: Cannot use Property on this class type`` errors.
    """
    snapshot = isolate_real_bluepebble_imports()

    from bluepebble.detector.active import matched_filter
    from bluepebble.models.environment import Constant, FlatBathymetry
    from bluepebble.models.propagation import rtrsAcousticPropagationModel
    from bluepebble.models.scattering import ConstantTargetStrength
    from bluepebble.platform.sonobuoy import OmniSonobuoyPlatform
    from bluepebble.sensors.base import ArrayState
    from bluepebble.signal.active import CWSignal, LFMSignal
    from bluepebble.simulator.active_rtrs import (
        RtrsActiveSonarSimulatorArray,
        RtrsActiveSonarSimulatorOmni,
        _occupied_band_mask,
    )

    yield SimpleNamespace(
        matched_filter=matched_filter,
        Constant=Constant,
        FlatBathymetry=FlatBathymetry,
        rtrsAcousticPropagationModel=rtrsAcousticPropagationModel,
        ConstantTargetStrength=ConstantTargetStrength,
        OmniSonobuoyPlatform=OmniSonobuoyPlatform,
        CWSignal=CWSignal,
        LFMSignal=LFMSignal,
        RtrsActiveSonarSimulatorOmni=RtrsActiveSonarSimulatorOmni,
        RtrsActiveSonarSimulatorArray=RtrsActiveSonarSimulatorArray,
        ArrayState=ArrayState,
        _occupied_band_mask=_occupied_band_mask,
    )

    restore_real_bluepebble_imports(snapshot)


def _install_fake_rtrs(monkeypatch, delay_s: float = 0.01) -> list[dict]:
    """Install a deterministic fake rtrs backend: unit-magnitude H(f) with phase consistent
    with delay_s, plus the same delay reported separately via delay_s.

    Real rtrs's ray-traced phase inherently encodes the propagation delay (H(f) ~ e^{-2j*pi*f*
    delay} in the simulation's phase convention, before acoustic.py's e^{+i wt} -> e^{-i wt}
    conjugate correction, so the fake must return the pre-conjugate e^{+2j*pi*f*delay} here).
    Since active_rtrs.py no longer re-applies delay as a separate phase ramp (see
    RtrsActiveSonarSimulatorOmni docstring), a flat delay-independent fake would silently drop
    the round-trip delay from the reconstructed waveform.
    """
    calls: list[dict] = []

    def fake_run_simulation(env_config: dict) -> dict:
        calls.append(env_config)
        freqs = np.asarray(env_config["source"]["freq_hz"], dtype=float)
        nfreq = len(freqs)
        phase = np.exp(2j * np.pi * freqs * delay_s)
        return {
            "pressure_field": {
                "shape": [nfreq, 1, 1, 1],
                "pressure_re": np.real(phase).tolist(),
                "pressure_im": np.imag(phase).tolist(),
                "delay_s": [delay_s],
            }
        }

    fake_rtrs = ModuleType("rtrs")
    fake_rtrs.run_simulation = fake_run_simulation
    monkeypatch.setitem(sys.modules, "rtrs", fake_rtrs)
    return calls


def _make_scenario(bp):
    buoy = bp.OmniSonobuoyPlatform(
        states=[GroundTruthState(StateVector([0.0, 0.0]), timestamp=T0)],
        position_mapping=[0, 1],
        hydrophone_depth_m=50.0,
    )
    target_path = GroundTruthPath(
        states=[GroundTruthState(StateVector([2000.0, 0.0, 0.0, 0.0, 80.0, 0.0]), timestamp=T0)]
    )
    propagation_model = bp.rtrsAcousticPropagationModel(
        ssp=bp.Constant(speed=1500.0),
        bathymetry=bp.FlatBathymetry(depth=-200.0),
        step_m=20.0,
    )
    return buoy, target_path, propagation_model


def _make_lfm(bp, **overrides):
    params = dict(
        freq_min_hz=800.0,
        freq_max_hz=1200.0,
        rise_time_s=0.005,
        source_level_db=0.0,
        duration_s=0.05,
        sampling_rate_hz=10_000,
    )
    params.update(overrides)
    return bp.LFMSignal(**params)


def test_reciprocity_default_makes_single_rtrs_call_per_target(bp, monkeypatch) -> None:
    """With use_reciprocity=True (default), only one propagate_spectrum call per target."""
    calls = _install_fake_rtrs(monkeypatch)
    buoy, target_path, propagation_model = _make_scenario(bp)

    sim = bp.RtrsActiveSonarSimulatorOmni(
        platform=buoy,
        propagation_model=propagation_model,
        signal=_make_lfm(bp),
        scattering_model=bp.ConstantTargetStrength(target_strength_db=0.0),
        ground_truth_paths=[target_path],
        ping_timestamps=[T0],
        receive_duration_s=0.05,
    )

    list(sim.sensor_data_gen())

    assert len(calls) == 1


def test_disabling_reciprocity_makes_two_calls_with_reversed_roles(bp, monkeypatch) -> None:
    """With use_reciprocity=False, two calls are made with source/receiver roles reversed."""
    calls = _install_fake_rtrs(monkeypatch)
    buoy, target_path, propagation_model = _make_scenario(bp)

    sim = bp.RtrsActiveSonarSimulatorOmni(
        platform=buoy,
        propagation_model=propagation_model,
        signal=_make_lfm(bp),
        scattering_model=bp.ConstantTargetStrength(target_strength_db=0.0),
        ground_truth_paths=[target_path],
        ping_timestamps=[T0],
        receive_duration_s=0.05,
        use_reciprocity=False,
    )

    list(sim.sensor_data_gen())

    assert len(calls) == 2
    # First call: buoy source (depth 50 -> corrected sign), target receiver.
    assert calls[0]["receivers"]["x_rcvr_m"] == [2000.0]
    # Second call: target source, buoy receiver.
    assert calls[1]["receivers"]["x_rcvr_m"] == [0.0]


@pytest.mark.parametrize("make_signal", [
    lambda bp: _make_lfm(bp, freq_min_hz=800.0, freq_max_hz=1200.0),
    lambda bp: bp.CWSignal(
        frequency_hz=1000.0, source_level_db=0.0, duration_s=0.05, sampling_rate_hz=10_000
    ),
])
def test_dense_query_matches_native_occupied_band_bins(bp, monkeypatch, make_signal) -> None:
    """propagate_spectrum is queried at every native FFT bin in the pulse's occupied band.

    No sparse sampling or interpolation: benchmarking showed propagate_spectrum's cost is
    dominated by ray/beam-tracing geometry setup rather than the number of frequencies
    requested, so both LFM and CW pulses are queried densely, directly at native resolution.
    """
    calls = _install_fake_rtrs(monkeypatch)
    buoy, target_path, propagation_model = _make_scenario(bp)
    signal = make_signal(bp)
    receive_duration_s = 0.05

    sim = bp.RtrsActiveSonarSimulatorOmni(
        platform=buoy,
        propagation_model=propagation_model,
        signal=signal,
        scattering_model=bp.ConstantTargetStrength(target_strength_db=0.0),
        ground_truth_paths=[target_path],
        ping_timestamps=[T0],
        receive_duration_s=receive_duration_s,
    )

    list(sim.sensor_data_gen())

    n_receive = int((signal.duration_s + receive_duration_s) * signal.sampling_rate_hz)
    pulse_spectrum = np.fft.fft(signal.generate(), n_receive)
    native_freqs_hz = np.fft.fftfreq(n_receive, d=1.0 / signal.sampling_rate_hz)
    mask = bp._occupied_band_mask(pulse_spectrum, energy_threshold=1e-3)

    assert np.sum(mask) > 1  # a single-point query would defeat the point of this test
    np.testing.assert_allclose(calls[0]["source"]["freq_hz"], native_freqs_hz[mask])


def test_matched_filter_peak_lands_at_expected_round_trip_delay(bp, monkeypatch) -> None:
    """End-to-end: with a known fixed one-way delay, the MF peak should land at 2x that delay.

    Uses a flat (unity, zero-phase) mocked H(f) so the echo shape is undistorted and the only
    thing determining where the matched-filter peak lands is the round-trip delay placement
    logic -- exactly the mechanism the RTRS plan requires (real rtrs delay, not phase alone).
    """
    one_way_delay_s = 0.01
    _install_fake_rtrs(monkeypatch, delay_s=one_way_delay_s)
    buoy, target_path, propagation_model = _make_scenario(bp)
    signal = _make_lfm(bp)

    sim = bp.RtrsActiveSonarSimulatorOmni(
        platform=buoy,
        propagation_model=propagation_model,
        signal=signal,
        scattering_model=bp.ConstantTargetStrength(target_strength_db=0.0),
        ground_truth_paths=[target_path],
        ping_timestamps=[T0],
        receive_duration_s=0.05,
    )

    _, batch = next(iter(sim.sensor_data_gen()))
    data = next(iter(batch))

    mf = bp.matched_filter(data.received_waveform, data.transmit_pulse)
    peak_idx = int(np.argmax(mf))

    expected_idx = round(2 * one_way_delay_s * signal.sampling_rate_hz)
    assert peak_idx == expected_idx


def test_target_beyond_receive_window_is_skipped(bp, monkeypatch) -> None:
    """A target whose round-trip delay exceeds the receive buffer contributes nothing."""
    _install_fake_rtrs(monkeypatch, delay_s=10.0)  # round trip: 20 s, far beyond the buffer
    buoy, target_path, propagation_model = _make_scenario(bp)

    sim = bp.RtrsActiveSonarSimulatorOmni(
        platform=buoy,
        propagation_model=propagation_model,
        signal=_make_lfm(bp),
        scattering_model=bp.ConstantTargetStrength(target_strength_db=0.0),
        ground_truth_paths=[target_path],
        ping_timestamps=[T0],
        receive_duration_s=0.05,
    )

    _, batch = next(iter(sim.sensor_data_gen()))
    data = next(iter(batch))

    np.testing.assert_allclose(data.received_waveform, 0.0)


def test_no_ground_truth_state_at_timestamp_is_skipped(bp, monkeypatch) -> None:
    """A target with no state at the ping timestamp should not raise and adds no energy."""
    _install_fake_rtrs(monkeypatch)
    buoy, _, propagation_model = _make_scenario(bp)
    empty_path = GroundTruthPath(
        states=[GroundTruthState(
            StateVector([2000.0, 0.0, 0.0, 0.0, 80.0, 0.0]), timestamp=datetime(2023, 1, 1)
        )]
    )

    sim = bp.RtrsActiveSonarSimulatorOmni(
        platform=buoy,
        propagation_model=propagation_model,
        signal=_make_lfm(bp),
        scattering_model=bp.ConstantTargetStrength(target_strength_db=0.0),
        ground_truth_paths=[empty_path],
        ping_timestamps=[T0],
        receive_duration_s=0.05,
    )

    _, batch = next(iter(sim.sensor_data_gen()))
    data = next(iter(batch))

    np.testing.assert_allclose(data.received_waveform, 0.0)


def test_occupied_band_mask_all_zero_pulse_returns_no_band(bp) -> None:
    """A pulse spectrum that is identically zero should produce an entirely-False mask."""
    mask = bp._occupied_band_mask(np.zeros(10, dtype=complex), energy_threshold=1e-3)

    assert not np.any(mask)


def test_array_baffle_attenuates_both_legs(bp) -> None:
    """RtrsActiveSonarSimulatorArray applies the hull baffle on transmit and receive.

    The rtrs channel is stubbed to unit transfer functions, so the only difference between a
    one-element array facing the target and one facing away is the baffle on both legs.
    """
    from stonesoup.types.array import StateVectors

    class _UnitChannelSim(bp.RtrsActiveSonarSimulatorArray):
        def _round_trip_channels(self, array, ref_pos, timestamp, target_pos, freqs_hz):
            n = array.num_sensors
            return (np.ones(len(freqs_hz), dtype=complex), 0.01,
                    np.ones((n, len(freqs_hz)), dtype=complex), np.full(n, 0.01))

    def run(normal: list[float]) -> np.ndarray:
        array = bp.ArrayState(
            num_sensors=1,
            state_vector=StateVectors(np.array([[0.0], [0.0], [-50.0]])),
            ref_state_vector=StateVector([0.0, 0.0, -50.0]),
            element_normals=StateVectors(np.array([normal]).T),
            baffle_attenuation_db=20.0,
        )
        sensor = SimpleNamespace(get_platform_state_at=lambda ts: SimpleNamespace(array=array))
        sim = _UnitChannelSim(
            platform=sensor,
            propagation_model=None,
            signal=_make_lfm(bp),
            scattering_model=bp.ConstantTargetStrength(target_strength_db=0.0),
            ground_truth_paths=[GroundTruthPath(states=[GroundTruthState(
                StateVector([2000.0, 0.0, 0.0, 0.0, -80.0, 0.0]), timestamp=T0)])],
            ping_timestamps=[T0],
            receive_duration_s=0.05,
        )
        _, batch = next(iter(sim.sensor_data_gen()))
        return next(iter(batch)).received_waveform

    facing = run([1.0, 0.0, 0.0])
    behind = run([-1.0, 0.0, 0.0])

    ratio_db = 20.0 * np.log10(np.abs(behind).max() / np.abs(facing).max())
    assert ratio_db == pytest.approx(-40.0, abs=1e-6)
