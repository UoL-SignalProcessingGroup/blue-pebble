"""Tests for beamforming helpers and implementations."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from .support import (
    install_fake_stonesoup,
    install_fake_stonesoup_plotter_modules,
    install_repo_package,
    load_package_module_from_repo,
)


class ConstantSSP:
    """Minimal constant sound-speed profile for steering-delay tests."""

    def __init__(self, speed: float):
        """Store the constant speed to return."""
        self.speed = speed

    def calculate(self, depth):
        """Return the configured sound speed."""
        return float(self.speed)


def _load_beamformer_module(monkeypatch):
    """Load the beamforming modules with lightweight Stone Soup and model scaffolding.

    The beamforming code is split across ``base.py``, ``conventional.py``, ``adaptive.py``
    and ``steering.py``; the latter three import from ``base`` via relative imports, so
    ``base`` must be loaded first to seed ``sys.modules`` before the rest resolve against it.
    """
    install_fake_stonesoup(monkeypatch)
    install_fake_stonesoup_plotter_modules(monkeypatch)
    install_repo_package(monkeypatch, "bluepebble", "bluepebble")
    install_repo_package(monkeypatch, "bluepebble.sigproc", "bluepebble/sigproc")
    install_repo_package(monkeypatch, "bluepebble.models", "bluepebble/models")

    environment_module = ModuleType("bluepebble.models.environment")
    environment_module.SoundSpeedProfile = ConstantSSP
    monkeypatch.setitem(sys.modules, "bluepebble.models.environment", environment_module)

    combined = SimpleNamespace()
    for name in ("base", "conventional", "adaptive", "steering"):
        module = load_package_module_from_repo(
            f"bluepebble/sigproc/{name}.py",
            f"bluepebble.sigproc.{name}",
        )
        combined.__dict__.update(vars(module))
    return combined


def test_delay_and_sum_rejects_invalid_domain(monkeypatch) -> None:
    """Delay-and-sum beamforming should reject unsupported domains."""
    beamformer = _load_beamformer_module(monkeypatch)

    with pytest.raises(ValueError, match="Invalid beamforming domain"):
        beamformer.DelayAndSumBeamformer(sampling_rate_hz=1000.0, domain="space")


def test_delay_and_sum_rejects_zero_sum_shading(monkeypatch) -> None:
    """Shading weights must sum to a finite non-zero value."""
    beamformer = _load_beamformer_module(monkeypatch)

    with pytest.raises(ValueError, match="finite non-zero"):
        beamformer.DelayAndSumBeamformer(
            sampling_rate_hz=1000.0,
            domain="time",
            shading=np.array([1.0, -1.0]),
        )


def test_delay_and_sum_rejects_sensor_delay_mismatch(monkeypatch) -> None:
    """The steering-delay matrix must match the number of sensors."""
    beamformer = _load_beamformer_module(monkeypatch)
    model = beamformer.DelayAndSumBeamformer(sampling_rate_hz=1000.0, domain="time")

    with pytest.raises(ValueError, match="Number of sensors"):
        model.beamform(
            np.ones((2, 8), dtype=np.complex128),
            np.zeros((1, 3), dtype=float),
        )


def test_delay_and_sum_rejects_shading_length_mismatch(monkeypatch) -> None:
    """Explicit shading must provide one weight per sensor."""
    beamformer = _load_beamformer_module(monkeypatch)
    model = beamformer.DelayAndSumBeamformer(
        sampling_rate_hz=1000.0,
        shading=np.array([1.0, 1.0, 1.0]),
        domain="time",
    )

    with pytest.raises(ValueError, match="Shading length"):
        model.beamform(
            np.ones((2, 8), dtype=np.complex128),
            np.zeros((1, 2), dtype=float),
        )


def test_delay_and_sum_time_domain_returns_weighted_average_for_zero_delays(
    monkeypatch,
) -> None:
    """Zero steering delays should reduce to a weighted sensor average in time mode."""
    beamformer = _load_beamformer_module(monkeypatch)
    model = beamformer.DelayAndSumBeamformer(sampling_rate_hz=8.0, domain="time")

    sensor_signals = np.array(
        [[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]],
        dtype=np.complex128,
    )
    beamformed = model.beamform(sensor_signals, np.zeros((1, 2), dtype=float))

    np.testing.assert_array_equal(
        beamformed,
        np.array([[1.5, 3.0, 4.5, 6.0]], dtype=np.complex128),
    )


def test_delay_and_sum_frequency_domain_matches_zero_delay_average(monkeypatch) -> None:
    """Frequency-domain DAS should match the zero-delay weighted average."""
    beamformer = _load_beamformer_module(monkeypatch)
    model = beamformer.DelayAndSumBeamformer(sampling_rate_hz=8.0, domain="frequency")

    sensor_signals = np.array(
        [[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]],
        dtype=np.complex128,
    )
    beamformed = model.beamform(sensor_signals, np.zeros((1, 2), dtype=float))

    np.testing.assert_allclose(
        beamformed,
        np.array([[1.5, 3.0, 4.5, 6.0]], dtype=np.complex128),
    )


# A 16-sensor line array at half-wavelength spacing for a 6 kHz carrier, hearing a plane wave
# from 60 degrees whose envelope is a 40 Hz tone.
_SOUND_SPEED = 1500.0
_CARRIER_HZ = 6000.0
_ENVELOPE_HZ = 40.0
_SENSOR_X_M = np.arange(16) * 0.125
_BEARING_GRID_RAD = np.deg2rad(np.arange(181.0))
_SOURCE_BEAM = 60


def _arrival_delays_s(bearings_rad):
    """Return each sensor's plane-wave arrival delay, shape ``(num_bearings, num_sensors)``."""
    return -np.outer(np.cos(np.atleast_1d(bearings_rad)), _SENSOR_X_M) / _SOUND_SPEED


def _plane_wave(sampling_rate_hz, num_samples, baseband):
    """Return the plane wave at each sensor, at passband or demodulated by the carrier."""
    t = np.arange(num_samples) / sampling_rate_hz
    delays = _arrival_delays_s(_BEARING_GRID_RAD[_SOURCE_BEAM])[0][:, np.newaxis]
    passband = np.exp(2j * np.pi * (_CARRIER_HZ + _ENVELOPE_HZ) * (t - delays))
    return passband * np.exp(-2j * np.pi * _CARRIER_HZ * t) if baseband else passband


def _beam_pattern(beamformer, data, sampling_rate_hz, domain, f0=0.0):
    """Return the mean output power of each beam of ``_BEARING_GRID_RAD``."""
    output = beamformer.DelayAndSumBeamformer(
        sampling_rate_hz=sampling_rate_hz, domain=domain, f0=f0
    ).beamform(data, _arrival_delays_s(_BEARING_GRID_RAD))
    return np.mean(np.abs(output) ** 2, axis=1)


@pytest.mark.parametrize("domain", ["time", "frequency"])
def test_delay_and_sum_steers_baseband_data_with_its_carrier(monkeypatch, domain) -> None:
    """With f0 at the carrier, a baseband plane wave peaks at its bearing at full power."""
    beamformer = _load_beamformer_module(monkeypatch)
    data = _plane_wave(1000.0, 1000, baseband=True)

    pattern = _beam_pattern(beamformer, data, 1000.0, domain, f0=_CARRIER_HZ)

    assert int(np.argmax(pattern)) == _SOURCE_BEAM
    # The time domain aligns the 40 Hz envelope only to the nearest 1 ms sample.
    tolerance = 1e-12 if domain == "frequency" else 0.01
    assert pattern[_SOURCE_BEAM] == pytest.approx(1.0, abs=tolerance)


@pytest.mark.parametrize("domain", ["time", "frequency"])
def test_delay_and_sum_misses_baseband_data_without_its_carrier(monkeypatch, domain) -> None:
    """With f0 left at 0 the same wave is not found: the carrier phase is what steers it."""
    beamformer = _load_beamformer_module(monkeypatch)
    data = _plane_wave(1000.0, 1000, baseband=True)

    pattern = _beam_pattern(beamformer, data, 1000.0, domain)

    assert abs(int(np.argmax(pattern)) - _SOURCE_BEAM) > 30
    assert pattern.max() < 1e-3


def test_delay_and_sum_baseband_beam_pattern_is_the_passband_pattern(monkeypatch) -> None:
    """Frequency-domain beams of baseband data match those of the same wave at passband.

    The passband pattern peaking at the source's bearing also fixes the sign of the steering
    phase: a flipped sign would put the peak at the mirror bearing, 120 degrees.
    """
    beamformer = _load_beamformer_module(monkeypatch)

    passband = _beam_pattern(
        beamformer, _plane_wave(16_000.0, 16_000, baseband=False), 16_000.0, "frequency"
    )
    baseband = _beam_pattern(
        beamformer, _plane_wave(1000.0, 1000, baseband=True), 1000.0, "frequency", f0=_CARRIER_HZ
    )

    assert int(np.argmax(passband)) == _SOURCE_BEAM
    np.testing.assert_allclose(baseband, passband, atol=1e-10)


@pytest.mark.parametrize(
    ("domain", "expected"),
    [
        (
            "time",
            [
                0.3601909663076421 + 0.9676990002757425j,
                -0.8065981415520554 + 0.6063287636772928j,
                -0.3701822189142408 - 0.8977539328564543j,
            ],
        ),
        (
            "frequency",
            [
                0.4664855020187025 + 0.9226881041953516j,
                -0.7028473461600652 + 0.7249329969871152j,
                -0.5292586822354861 - 0.8135565133124373j,
            ],
        ),
    ],
)
def test_delay_and_sum_passband_results_are_those_of_v0_8_0(monkeypatch, domain, expected) -> None:
    """With f0 left at 0, both domains give what they gave before they used f0.

    The values were saved from the code before the change (commit 0d00482). They are compared
    to rounding and not bit for bit, since FFT results can differ between platforms.
    """
    beamformer = _load_beamformer_module(monkeypatch)
    n = np.arange(256)
    signals = np.array(
        [(1 + 0.1 * j) * np.exp(1j * (0.37 * j + 0.011 * n + 2e-5 * n**2)) for j in range(6)]
    )
    delays = np.outer([-1.0, 0.5, 2.0], np.arange(6) * 1.3e-4)

    output = beamformer.DelayAndSumBeamformer(domain=domain, sampling_rate_hz=8000.0).beamform(
        signals, delays
    )

    np.testing.assert_allclose(output[[0, 1, 2], [10, 100, 200]], expected, rtol=1e-13)


def test_delay_and_sum_normalises_explicit_shading(monkeypatch) -> None:
    """Explicit shading should be normalised before beamforming."""
    beamformer = _load_beamformer_module(monkeypatch)
    model = beamformer.DelayAndSumBeamformer(
        sampling_rate_hz=8.0,
        domain="time",
        shading=np.array([1.0, 3.0]),
    )

    beamformed = model.beamform(
        np.array([[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]], dtype=np.complex128),
        np.zeros((1, 2), dtype=float),
    )

    np.testing.assert_allclose(model.shading, np.array([0.25, 0.75]))
    np.testing.assert_allclose(
        beamformed,
        np.array([[1.75, 3.5, 5.25, 7.0]], dtype=np.complex128),
    )


def test_stft_rejects_short_inputs(monkeypatch) -> None:
    """The shared STFT helper should reject signals shorter than ``nfft``."""
    beamformer = _load_beamformer_module(monkeypatch)

    with pytest.raises(ValueError, match="less than window size"):
        beamformer._stft(
            np.ones((2, 3), dtype=np.complex128),
            nfft=4,
            overlap=0,
        )


def test_stft_uses_unit_hop_when_overlap_exceeds_window(monkeypatch) -> None:
    """Overlap larger than ``nfft`` should fall back to a hop of one sample."""
    beamformer = _load_beamformer_module(monkeypatch)

    stft = beamformer._stft(
        np.ones((2, 5), dtype=np.complex128),
        nfft=4,
        overlap=5,
    )

    assert stft.shape == (2, 2, 4)


def test_delay_and_sum_broadband_power_returns_zero_when_no_bins_are_active(monkeypatch) -> None:
    """Broadband power should be zero if the requested frequency band excludes all bins."""
    beamformer = _load_beamformer_module(monkeypatch)
    model = beamformer.DelayAndSumBeamformer(
        sampling_rate_hz=8.0,
        domain="broadband_power",
        nfft=4,
        overlap=0,
        fmin=100.0,
        fmax=200.0,
    )

    power = model.beamform(
        np.array([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]], dtype=np.complex128),
        np.zeros((1, 2), dtype=float),
    )

    np.testing.assert_array_equal(power, np.zeros((1, 1), dtype=np.float64))


def test_delay_and_sum_time_domain_crops_for_large_integer_delays(monkeypatch) -> None:
    """Large sample delays should crop edge artefacts from the time-domain output."""
    beamformer = _load_beamformer_module(monkeypatch)
    model = beamformer.DelayAndSumBeamformer(
        sampling_rate_hz=4.0,
        domain="time",
        shading=np.array([1.0, 3.0]),
    )

    beamformed = model.beamform(
        np.array(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
            ],
            dtype=np.complex128,
        ),
        np.array([[0.5, 0.0]], dtype=float),
    )

    np.testing.assert_allclose(
        beamformed,
        np.array([[23.75, 31.5, 39.25, 47.0, 52.75, 60.5]], dtype=np.complex128),
    )


def test_mvdr_rejects_sensor_delay_mismatch(monkeypatch) -> None:
    """MVDR beamforming should validate the steering-delay dimensions."""
    beamformer = _load_beamformer_module(monkeypatch)
    model = beamformer.MinimumVarianceDistortionlessResponseBeamformer(
        sampling_rate_hz=1000.0,
        nfft=4,
    )

    with pytest.raises(ValueError, match="Number of sensors"):
        model.beamform(
            np.ones((2, 8), dtype=np.complex128),
            np.zeros((1, 3), dtype=float),
        )


def test_mvdr_rejects_nfft_smaller_than_array_delay_span(monkeypatch) -> None:
    """MVDR should reject FFT windows that are too short for the steering geometry."""
    beamformer = _load_beamformer_module(monkeypatch)
    model = beamformer.MinimumVarianceDistortionlessResponseBeamformer(
        sampling_rate_hz=10.0,
        nfft=1,
    )

    with pytest.raises(ValueError, match="nfft too small"):
        model.beamform(
            np.ones((2, 4), dtype=np.complex128),
            np.array([[0.0, 0.2]], dtype=float),
        )


def test_mvdr_returns_finite_power_for_simple_input(monkeypatch) -> None:
    """MVDR should return a finite power map for a simple deterministic signal."""
    beamformer = _load_beamformer_module(monkeypatch)
    model = beamformer.MinimumVarianceDistortionlessResponseBeamformer(
        sampling_rate_hz=8.0,
        nfft=4,
        overlap=0,
    )

    power = model.beamform(
        np.array([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]], dtype=np.complex128),
        np.array([[0.0, 0.0], [0.0, 0.1]], dtype=float),
    )

    assert power.shape == (2, 1)
    assert np.isfinite(power).all()
    assert np.all(power >= 0.0)


def test_mvdr_returns_zero_when_no_frequency_bins_are_active(monkeypatch) -> None:
    """MVDR should return zero power when the selected frequency range excludes all bins."""
    beamformer = _load_beamformer_module(monkeypatch)
    model = beamformer.MinimumVarianceDistortionlessResponseBeamformer(
        sampling_rate_hz=8.0,
        nfft=4,
        overlap=0,
        fmin=100.0,
        fmax=200.0,
    )

    power = model.beamform(
        np.ones((2, 4), dtype=np.complex128),
        np.zeros((1, 2), dtype=float),
    )

    np.testing.assert_array_equal(power, np.zeros((1, 1), dtype=np.float64))


def _multiband_test_inputs(num_sensors: int = 4, num_samples: int = 128, num_dirs: int = 5):
    """Build deterministic sensor data and steering delays for multiband tests."""
    rng = np.random.default_rng(20260715)
    signals = (
        rng.standard_normal((num_sensors, num_samples))
        + 1j * rng.standard_normal((num_sensors, num_samples))
    ).astype(np.complex128)
    delays = rng.standard_normal((num_dirs, num_sensors)) * 1e-4
    return signals, delays


def test_frequency_band_rejects_non_positive_span(monkeypatch) -> None:
    """A band must span a positive frequency range."""
    beamformer = _load_beamformer_module(monkeypatch)

    with pytest.raises(ValueError, match="must span a positive frequency range"):
        beamformer.FrequencyBand(label="bad", fmin=100.0, fmax=100.0)


def test_beamformer_rejects_duplicate_band_labels(monkeypatch) -> None:
    """Duplicate band labels would silently collide downstream, so reject them."""
    beamformer = _load_beamformer_module(monkeypatch)

    with pytest.raises(ValueError, match="Band labels must be unique"):
        beamformer.MinimumVarianceDistortionlessResponseBeamformer(
            sampling_rate_hz=500.0,
            bands=[
                beamformer.FrequencyBand(label="dupe", fmin=70.0, fmax=80.0),
                beamformer.FrequencyBand(label="dupe", fmin=95.0, fmax=105.0),
            ],
        )


def test_beamformer_rejects_empty_bands_list(monkeypatch) -> None:
    """An empty bands list is a configuration error, not single-band mode."""
    beamformer = _load_beamformer_module(monkeypatch)

    with pytest.raises(ValueError, match="at least one FrequencyBand"):
        beamformer.MinimumVarianceDistortionlessResponseBeamformer(
            sampling_rate_hz=500.0,
            bands=[],
        )


def test_delay_and_sum_rejects_bands_outside_broadband_power(monkeypatch) -> None:
    """Time and frequency DAS integrate no bins, so bands are meaningless there."""
    beamformer = _load_beamformer_module(monkeypatch)

    with pytest.raises(ValueError, match="requires domain='broadband_power'"):
        beamformer.DelayAndSumBeamformer(
            sampling_rate_hz=500.0,
            domain="time",
            bands=[beamformer.FrequencyBand(label="a", fmin=70.0, fmax=80.0)],
        )


def test_mvdr_single_band_output_stays_two_dimensional(monkeypatch) -> None:
    """Leaving bands unset must preserve the original 2D output contract."""
    beamformer = _load_beamformer_module(monkeypatch)
    signals, delays = _multiband_test_inputs()

    power = beamformer.MinimumVarianceDistortionlessResponseBeamformer(
        sampling_rate_hz=500.0, nfft=32, overlap=16, fmin=70.0, fmax=105.0
    ).beamform(signals, delays)

    assert power.ndim == 2
    assert power.shape[0] == delays.shape[0]


@pytest.mark.parametrize("normalise", [False, True])
def test_mvdr_multiband_matches_separate_single_band_runs(monkeypatch, normalise) -> None:
    """A K-band run must equal K single-band runs exactly, including overlapping bands.

    This is the core guarantee of multiband processing: sharing the per-bin adaptive solve
    across bands is a performance optimisation only, and must not perturb the arithmetic.
    """
    beamformer = _load_beamformer_module(monkeypatch)
    signals, delays = _multiband_test_inputs()
    edges = [("70-105 Hz", 70.0, 105.0), ("70-80 Hz", 70.0, 80.0), ("95-105 Hz", 95.0, 105.0)]
    shared = dict(sampling_rate_hz=500.0, nfft=32, overlap=16, normalise_by_bandwidth=normalise)

    multiband = beamformer.MinimumVarianceDistortionlessResponseBeamformer(
        bands=[beamformer.FrequencyBand(label=lbl, fmin=lo, fmax=hi) for lbl, lo, hi in edges],
        **shared,
    ).beamform(signals, delays)

    assert multiband.shape == (len(edges), delays.shape[0], multiband.shape[2])

    for band_idx, (_, fmin, fmax) in enumerate(edges):
        single = beamformer.MinimumVarianceDistortionlessResponseBeamformer(
            fmin=fmin, fmax=fmax, **shared
        ).beamform(signals, delays)
        np.testing.assert_array_equal(multiband[band_idx], single)


def test_delay_and_sum_multiband_matches_separate_single_band_runs(monkeypatch) -> None:
    """Broadband-power DAS must satisfy the same K-band equivalence as MVDR."""
    beamformer = _load_beamformer_module(monkeypatch)
    signals, delays = _multiband_test_inputs()
    edges = [("wide", 70.0, 105.0), ("low", 70.0, 80.0), ("high", 95.0, 105.0)]
    shared = dict(sampling_rate_hz=500.0, domain="broadband_power", nfft=32, overlap=16)

    multiband = beamformer.DelayAndSumBeamformer(
        bands=[beamformer.FrequencyBand(label=lbl, fmin=lo, fmax=hi) for lbl, lo, hi in edges],
        **shared,
    ).beamform(signals, delays)

    for band_idx, (_, fmin, fmax) in enumerate(edges):
        single = beamformer.DelayAndSumBeamformer(fmin=fmin, fmax=fmax, **shared).beamform(
            signals, delays
        )
        np.testing.assert_array_equal(multiband[band_idx], single)


@pytest.mark.parametrize(
    ("f0", "fmin", "fmax"),
    [
        (0.0, 37.3, 190.0),  # starts between anchor bins and spans several anchors
        (200.0, None, None),  # every bin, across the jump from positive to negative bins
    ],
)
def test_delay_and_sum_broadband_power_matches_direct_steering(
    monkeypatch, f0, fmin, fmax
) -> None:
    """Steering phases built by recurrence must match an exp per bin to rounding."""
    beamformer = _load_beamformer_module(monkeypatch)
    signals, delays = _multiband_test_inputs(num_sensors=6, num_samples=1024, num_dirs=7)
    fs, nfft, overlap = 500.0, 256, 128
    power = beamformer.DelayAndSumBeamformer(
        sampling_rate_hz=fs,
        domain="broadband_power",
        nfft=nfft,
        overlap=overlap,
        f0=f0,
        fmin=fmin,
        fmax=fmax,
    ).beamform(signals, delays)

    stft = beamformer._stft(signals, nfft, overlap)
    f_bins = beamformer._stft_bin_frequencies(nfft, fs, f0)
    weights = np.full(signals.shape[0], 1.0 / signals.shape[0])
    expected = np.zeros_like(power)
    for i in beamformer._active_bin_indices(f_bins, fmin, fmax):
        steering = np.exp(1j * 2 * np.pi * f_bins[i] * delays) * weights
        expected += np.abs(steering @ stft[:, :, i]) ** 2

    np.testing.assert_allclose(power, expected, rtol=1e-12)


def test_delay_and_sum_multiband_bit_identical_when_band_starts_between_anchors(
    monkeypatch,
) -> None:
    """A band's power must not depend on which other bands share its beamforming pass."""
    beamformer = _load_beamformer_module(monkeypatch)
    signals, delays = _multiband_test_inputs(num_sensors=6, num_samples=1024, num_dirs=7)
    # Bins are 1.95 Hz apart, so "high" starts at bin 62, between the anchors at 32 and 64,
    # and in the multiband run "wide" reaches that bin through the recurrence.
    edges = [("wide", 30.0, 200.0), ("high", 120.0, 200.0), ("low", 30.0, 60.0)]
    shared = dict(sampling_rate_hz=500.0, domain="broadband_power", nfft=256, overlap=128)

    multiband = beamformer.DelayAndSumBeamformer(
        bands=[beamformer.FrequencyBand(label=lbl, fmin=lo, fmax=hi) for lbl, lo, hi in edges],
        **shared,
    ).beamform(signals, delays)

    for band_idx, (_, fmin, fmax) in enumerate(edges):
        single = beamformer.DelayAndSumBeamformer(fmin=fmin, fmax=fmax, **shared).beamform(
            signals, delays
        )
        np.testing.assert_array_equal(multiband[band_idx], single)


def test_multiband_band_with_no_active_bins_yields_zero_slice(monkeypatch) -> None:
    """An out-of-range band should zero its own slice without affecting its neighbours."""
    beamformer = _load_beamformer_module(monkeypatch)
    signals, delays = _multiband_test_inputs()

    power = beamformer.MinimumVarianceDistortionlessResponseBeamformer(
        sampling_rate_hz=500.0,
        nfft=32,
        overlap=16,
        bands=[
            beamformer.FrequencyBand(label="in range", fmin=70.0, fmax=105.0),
            beamformer.FrequencyBand(label="out of range", fmin=1e6, fmax=2e6),
        ],
    ).beamform(signals, delays)

    np.testing.assert_array_equal(power[1], np.zeros_like(power[1]))
    assert np.any(power[0] > 0.0)


def test_normalise_by_bandwidth_divides_each_band_by_its_bin_count(monkeypatch) -> None:
    """Bandwidth normalisation should remove the level offset caused by band width alone."""
    beamformer = _load_beamformer_module(monkeypatch)
    signals, delays = _multiband_test_inputs()
    bands = [
        beamformer.FrequencyBand(label="narrow", fmin=70.0, fmax=80.0),
        beamformer.FrequencyBand(label="wide", fmin=70.0, fmax=105.0),
    ]
    shared = dict(sampling_rate_hz=500.0, nfft=32, overlap=16, bands=bands)

    raw = beamformer.MinimumVarianceDistortionlessResponseBeamformer(**shared).beamform(
        signals, delays
    )
    normalised = beamformer.MinimumVarianceDistortionlessResponseBeamformer(
        normalise_by_bandwidth=True, **shared
    ).beamform(signals, delays)

    f_bins = beamformer._stft_bin_frequencies(32, 500.0, 0.0)
    for band_idx, band in enumerate(bands):
        num_bins = len(beamformer._active_bin_indices(f_bins, band.fmin, band.fmax))
        np.testing.assert_allclose(normalised[band_idx], raw[band_idx] / num_bins)


_STFT_BEAMFORMERS = ["delay_and_sum", "mvdr"]


def _band_power(beamformer, kind, signals, delays, **kwargs):
    """Return an STFT beamformer's power map at 500 Hz with 64-sample frames."""
    shared = dict(sampling_rate_hz=500.0, nfft=64, overlap=32, **kwargs)
    if kind == "mvdr":
        return beamformer.MinimumVarianceDistortionlessResponseBeamformer(**shared).beamform(
            signals, delays
        )
    return beamformer.DelayAndSumBeamformer(domain="broadband_power", **shared).beamform(
        signals, delays
    )


def _negative_frequency_noise(num_sensors: int = 4, num_samples: int = 1024):
    """Return noise confined to -3fs/8 to -fs/8, so every frame's power is in negative bins."""
    signals, _ = _multiband_test_inputs(num_sensors, num_samples)
    spectrum = np.fft.fft(signals, axis=1)
    frequencies = np.fft.fftfreq(num_samples)
    spectrum[:, (frequencies > -0.125) | (frequencies < -0.375)] = 0.0
    return np.fft.ifft(spectrum, axis=1)


@pytest.mark.parametrize("kind", _STFT_BEAMFORMERS)
def test_default_band_of_passband_data_starts_at_zero_hertz(monkeypatch, kind) -> None:
    """With ``f0 == 0`` a lower edge of ``None`` selects the bins that ``fmin=0`` does."""
    beamformer = _load_beamformer_module(monkeypatch)
    signals, delays = _multiband_test_inputs(num_samples=1024)

    default = _band_power(beamformer, kind, signals, delays)
    from_zero = _band_power(beamformer, kind, signals, delays, fmin=0.0)

    assert np.any(default > 0.0)
    np.testing.assert_array_equal(default, from_zero)


@pytest.mark.parametrize("kind", _STFT_BEAMFORMERS)
def test_default_band_of_passband_data_leaves_out_negative_frequencies(monkeypatch, kind) -> None:
    """Negative-frequency content is integrated only when a negative ``fmin`` asks for it.

    The noise lies a quarter of the band from 0 Hz and from fs/2. The Hann window's leakage
    into the non-negative bins was measured at 5e-8 of the power for both beamformers.
    """
    beamformer = _load_beamformer_module(monkeypatch)
    signals = _negative_frequency_noise()
    _, delays = _multiband_test_inputs()

    default = _band_power(beamformer, kind, signals, delays)
    whole_spectrum = _band_power(beamformer, kind, signals, delays, fmin=-250.0)

    assert whole_spectrum.sum() > 0.0
    assert default.sum() < 1e-6 * whole_spectrum.sum()


@pytest.mark.parametrize("kind", _STFT_BEAMFORMERS)
def test_default_band_of_baseband_data_is_every_bin(monkeypatch, kind) -> None:
    """With a carrier, ``None`` edges still select the whole spectrum, as before v0.9.0."""
    beamformer = _load_beamformer_module(monkeypatch)
    signals, delays = _multiband_test_inputs(num_samples=1024)
    f0 = 2000.0

    default = _band_power(beamformer, kind, signals, delays, f0=f0)
    every_bin = _band_power(
        beamformer, kind, signals, delays, f0=f0, fmin=f0 - 250.0, fmax=f0 + 250.0
    )
    upper_half = _band_power(beamformer, kind, signals, delays, f0=f0, fmin=f0)

    np.testing.assert_array_equal(default, every_bin)
    assert np.all(upper_half < default)


def test_default_band_divides_by_the_non_negative_bin_count(monkeypatch) -> None:
    """Bandwidth normalisation counts bins 0 to nfft / 2 of passband data, not all nfft."""
    beamformer = _load_beamformer_module(monkeypatch)
    signals, delays = _multiband_test_inputs(num_samples=1024)

    raw = _band_power(beamformer, "delay_and_sum", signals, delays)
    normalised = _band_power(
        beamformer, "delay_and_sum", signals, delays, normalise_by_bandwidth=True
    )

    np.testing.assert_allclose(normalised, raw / (64 // 2 + 1))


def test_steering_calculator_returns_expected_horizontal_delays(monkeypatch) -> None:
    """Steering delays should match the 2D projected sensor offsets."""
    beamformer = _load_beamformer_module(monkeypatch)
    calculator = beamformer.SteeringCalculator(
        ssp=ConstantSSP(1500.0),
        steering_sector_rad=(0.0, np.pi / 2),
        num_beams=2,
        frame="world",
    )
    platform = SimpleNamespace(
        array=SimpleNamespace(
            state_vector=np.array([[0.0, 1.0], [0.0, 0.0], [-10.0, -10.0]]),
            ref_state_vector=np.array([[0.0], [0.0], [-10.0]]),
        )
    )

    delays = calculator.calculate(platform)

    np.testing.assert_allclose(
        delays,
        np.array([[0.0, -1.0 / 1500.0], [0.0, 0.0]]),
        atol=1e-18,
    )


def _straight_array_platform(num_sensors: int = 8, spacing_m: float = 1.0) -> SimpleNamespace:
    """Build a fake platform whose sensors lie exactly on the x-axis (array axis = 0 rad)."""
    x = np.arange(num_sensors, dtype=np.float64) * spacing_m
    state_vector = np.array(
        [x, np.zeros(num_sensors), np.full(num_sensors, -10.0)],
    )
    return SimpleNamespace(
        array=SimpleNamespace(
            state_vector=state_vector,
            ref_state_vector=state_vector[:, [0]],
        )
    )


def test_array_axis_is_wrapped_to_minus_pi_not_plus_pi(monkeypatch) -> None:
    """An array lying along -x has axis -pi, the [-pi, pi) convention, not arctan2's +pi."""
    beamformer = _load_beamformer_module(monkeypatch)
    platform = _straight_array_platform(spacing_m=-1.0)  # last sensor at -x from the first

    assert beamformer.SteeringCalculator._array_axis_rad(platform) == -np.pi


def _rotated_array_platform(axis_rad: float, num_sensors: int = 8) -> SimpleNamespace:
    """Build a fake straight array whose axis points along ``axis_rad``."""
    offsets = np.arange(num_sensors, dtype=np.float64)
    state_vector = np.array(
        [offsets * np.cos(axis_rad), offsets * np.sin(axis_rad), np.full(num_sensors, -10.0)],
    )
    return SimpleNamespace(
        array=SimpleNamespace(state_vector=state_vector, ref_state_vector=state_vector[:, [0]])
    )


def _steering_calculator(beamformer, sector, num_beams, **kwargs):
    return beamformer.SteeringCalculator(
        ssp=ConstantSSP(1500.0), steering_sector_rad=sector, num_beams=num_beams, **kwargs
    )


FULL_CIRCLE_8 = np.linspace(-np.pi, np.pi, 8, endpoint=False)


@pytest.mark.parametrize(
    ("sector", "num_beams", "expected"),
    [
        # A partial sector includes both endpoints.
        ((-np.pi / 2, np.pi / 2), 5, np.linspace(-np.pi / 2, np.pi / 2, 5)),
        # Anticlockwise from start to end, so this is the 90 deg wedge through pi, wrapped.
        ((3 * np.pi / 4, -3 * np.pi / 4), 3, np.array([3 * np.pi / 4, -np.pi, -3 * np.pi / 4])),
        # Every way of writing a full circle gives the same canonical grid from -pi.
        ((-np.pi, np.pi), 8, FULL_CIRCLE_8),
        ((0.0, 2 * np.pi), 8, FULL_CIRCLE_8),
        ((np.pi, -np.pi), 8, FULL_CIRCLE_8),
        ((np.pi / 2, -3 * np.pi / 2), 8, FULL_CIRCLE_8),
    ],
    ids=[
        "half-plane",
        "wedge-through-pi",
        "full-minus-pi-to-pi",
        "full-zero-to-two-pi",
        "full-pi-to-minus-pi",
        "full-quarter-offset",
    ],
)
def test_uniform_steering_grid_follows_sector_rules(
    monkeypatch, sector, num_beams, expected
) -> None:
    """Uniform grids apply the endpoint, direction and full-circle rules."""
    beamformer = _load_beamformer_module(monkeypatch)

    calculator = _steering_calculator(beamformer, sector, num_beams, frame="world")

    bearings = calculator.steering_bearings()

    np.testing.assert_allclose(bearings, expected, atol=1e-12)
    assert np.all((bearings >= -np.pi) & (bearings < np.pi))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"steering_sector_rad": (0.0, 0.0)}, "is empty"),
        ({"steering_sector_rad": (0.0, 3 * np.pi)}, "more than a full circle"),
        ({"steering_sector_rad": (0.0, 1.0, 2.0)}, "must be \\(start, end\\)"),
        ({"num_beams": 1}, "num_beams"),
        ({"spacing": "log"}, "spacing"),
        ({"spacing": "sine", "steering_sector_rad": (0.0, 3.5)}, "within one side"),
        ({"spacing": "sine", "steering_sector_rad": (-0.2, 0.2)}, "crosses the array axis"),
        ({"frame": "compass"}, "frame"),
    ],
    ids=[
        "empty",
        "over-full",
        "three-values",
        "one-beam",
        "unknown-spacing",
        "sine-too-wide",
        "sine-crosses-axis",
        "unknown-frame",
    ],
)
def test_steering_calculator_rejects_invalid_grid(monkeypatch, kwargs, match) -> None:
    """Invalid sectors, beam counts and spacings are rejected at construction."""
    beamformer = _load_beamformer_module(monkeypatch)
    settings = {"steering_sector_rad": (-np.pi, np.pi), "num_beams": 8, **kwargs}

    with pytest.raises(ValueError, match=match):
        beamformer.SteeringCalculator(ssp=ConstantSSP(1500.0), **settings)


def test_steering_calculator_names_replacement_for_removed_azimuths(monkeypatch) -> None:
    """The removed steering_azimuths_rad raises a TypeError that names its replacement."""
    beamformer = _load_beamformer_module(monkeypatch)

    with pytest.raises(TypeError, match="steering_sector_rad"):
        beamformer.SteeringCalculator(
            ssp=ConstantSSP(1500.0), steering_azimuths_rad=np.linspace(-1.0, 1.0, 5)
        )


@pytest.mark.parametrize(
    ("sector", "broadside_rad"),
    [
        ((0.0, np.pi), np.pi / 2),  # endfire to endfire on the +y side
        ((-np.pi / 2 - 0.3, -np.pi / 2 + 0.5), -np.pi / 2),  # part of the -y side
    ],
    ids=["endfire-to-endfire", "partial-other-side"],
)
def test_sine_steering_grid_is_evenly_spaced_in_sine_from_broadside(
    monkeypatch, sector, broadside_rad
) -> None:
    """Sine spacing places beams evenly in sin(angle from broadside), sector ends included."""
    beamformer = _load_beamformer_module(monkeypatch)
    num_beams = 9
    calculator = _steering_calculator(beamformer, sector, num_beams, spacing="sine", frame="world")

    bearings = calculator.steering_bearings(_straight_array_platform())

    u = np.sin(bearings - broadside_rad)
    u_ends = np.sin(np.array(sector) - broadside_rad)
    np.testing.assert_allclose(u, np.linspace(u_ends[0], u_ends[1], num_beams), atol=1e-12)
    np.testing.assert_allclose(np.cos(bearings[[0, -1]] - np.array(sector)), 1.0, atol=1e-12)


def test_sine_steering_grid_rejects_sector_crossing_the_axis(monkeypatch) -> None:
    """A sine-spaced sector straddling the array axis cannot be spaced monotonically in u."""
    beamformer = _load_beamformer_module(monkeypatch)
    calculator = _steering_calculator(beamformer, (-0.2, 0.2), 5, spacing="sine", frame="world")

    with pytest.raises(ValueError, match="crosses the array axis"):
        calculator.steering_bearings(_straight_array_platform())


@pytest.mark.parametrize(
    ("spacing", "frame"), [("sine", "world"), ("uniform", "array")], ids=["world-sine", "array"]
)
def test_steering_grid_needs_a_platform(monkeypatch, spacing, frame) -> None:
    """Grids that depend on the array's geometry raise when asked for without a platform."""
    beamformer = _load_beamformer_module(monkeypatch)
    calculator = _steering_calculator(beamformer, (0.0, np.pi), 5, spacing=spacing, frame=frame)

    with pytest.raises(ValueError, match="platform"):
        calculator.steering_bearings()


def test_sine_steering_grid_is_fixed_while_the_heading_holds(monkeypatch) -> None:
    """The first axis fixes the grid; reversed sensor order is fine, a turn raises."""
    beamformer = _load_beamformer_module(monkeypatch)
    calculator = _steering_calculator(beamformer, (0.0, np.pi), 9, spacing="sine", frame="world")
    first = calculator.steering_bearings(_rotated_array_platform(0.0))

    # Reversing the sensor order flips the axis by pi, which is the same line.
    np.testing.assert_array_equal(
        calculator.steering_bearings(_straight_array_platform(spacing_m=-1.0)), first
    )
    with pytest.raises(ValueError, match="has turned"):
        calculator.calculate(_rotated_array_platform(0.05))


def test_array_frame_sine_sector_across_the_axis_suggests_a_side(monkeypatch) -> None:
    """A world-frame sector carried into the array frame gets a hint naming the fixes."""
    beamformer = _load_beamformer_module(monkeypatch)

    with pytest.raises(ValueError, match=r"STARBOARD.*PORT.*frame='world'"):
        _steering_calculator(beamformer, (-np.pi / 2, np.pi / 2), 9, spacing="sine")


def test_world_frame_sine_sector_across_the_axis_has_no_frame_hint(monkeypatch) -> None:
    """The hint is about the array frame, so the world frame's error leaves it out."""
    beamformer = _load_beamformer_module(monkeypatch)
    calculator = _steering_calculator(
        beamformer, (-np.pi / 2, np.pi / 2), 9, spacing="sine", frame="world"
    )

    with pytest.raises(ValueError, match="crosses the array axis") as excinfo:
        calculator.steering_bearings(_rotated_array_platform(0.0))  # the axis along +x
    assert "STARBOARD" not in str(excinfo.value)


def test_array_forward_points_from_the_last_sensor_to_sensor_0(monkeypatch) -> None:
    """Sensor 0 is the front of the array, so forward is the reverse of the sensor order."""
    beamformer = _load_beamformer_module(monkeypatch)
    platform = _rotated_array_platform(0.3)  # sensor 0 at the origin, the rest along 0.3 rad

    forward = beamformer.SteeringCalculator._array_forward_rad(platform)

    assert forward == pytest.approx(0.3 - np.pi)


def test_named_sectors_are_the_tuples_they_stand_for(monkeypatch) -> None:
    """PORT, STARBOARD and FULL_CIRCLE spell out sectors whose order is easy to get wrong."""
    beamformer = _load_beamformer_module(monkeypatch)

    assert beamformer.PORT == (0.0, np.pi)
    assert beamformer.STARBOARD == (-np.pi, 0.0)
    assert beamformer.FULL_CIRCLE == (-np.pi, np.pi)


def test_reversed_starboard_endpoints_steer_port(monkeypatch) -> None:
    """Sectors run anticlockwise from start to end, so (0, -pi) is port, not starboard."""
    beamformer = _load_beamformer_module(monkeypatch)
    platform = _straight_array_platform(spacing_m=-1.0)

    reversed_bearings = _steering_calculator(beamformer, (0.0, -np.pi), 5).steering_bearings(
        platform
    )
    port_bearings = _steering_calculator(beamformer, beamformer.PORT, 5).steering_bearings(
        platform
    )

    np.testing.assert_allclose(np.sort(reversed_bearings), np.sort(port_bearings), atol=1e-12)


@pytest.mark.parametrize(
    ("side", "side_y_sign"),
    [("STARBOARD", -1.0), ("PORT", 1.0)],
    ids=["starboard", "port"],
)
def test_array_frame_sides_follow_the_sign_convention(monkeypatch, side, side_y_sign) -> None:
    """Heading +x, STARBOARD (-pi, 0) points towards -y and PORT (0, pi) towards +y."""
    beamformer = _load_beamformer_module(monkeypatch)
    platform = _straight_array_platform(spacing_m=-1.0)  # sensor 0 in front, heading +x
    sector = getattr(beamformer, side)

    bearings = _steering_calculator(beamformer, sector, 5).steering_bearings(platform)

    np.testing.assert_allclose(bearings[2], side_y_sign * np.pi / 2, atol=1e-12)
    assert np.all(side_y_sign * np.sin(bearings) >= -1e-12)


def _dome_platform(axis) -> SimpleNamespace:
    """Build a fake dome-like array: sensors spread across y, facing along ``axis``."""
    y = np.linspace(-2.0, 2.0, 9)
    state_vector = np.array([3.0 - 0.1 * y**2, y, np.full(y.size, -10.0)])
    return SimpleNamespace(
        array=SimpleNamespace(
            state_vector=state_vector,
            ref_state_vector=np.array([[0.0], [0.0], [-10.0]]),
            axis=None if axis is None else np.asarray(axis, dtype=float).reshape(3, 1),
        )
    )


@pytest.mark.parametrize("axis_rad", [0.0, 0.7, -2.5])
def test_array_frame_follows_the_stated_axis(monkeypatch, axis_rad) -> None:
    """An array state's axis sets the array frame, not the line through its end sensors.

    The dome's end sensors lie across the ship, so reading them would turn its fan by 90
    degrees.
    """
    beamformer = _load_beamformer_module(monkeypatch)
    platform = _dome_platform([np.cos(axis_rad), np.sin(axis_rad), 0.0])

    bearings = _steering_calculator(beamformer, (-np.pi / 2, np.pi / 2), 5).steering_bearings(
        platform
    )

    np.testing.assert_allclose(np.angle(np.exp(1j * (bearings[2] - axis_rad))), 0.0, atol=1e-12)
    assert beamformer.SteeringCalculator._array_axis_rad(platform) == pytest.approx(axis_rad)


@pytest.mark.parametrize("axis", [None, [0.0, 0.0, 1.0]], ids=["no axis", "vertical axis"])
def test_array_frame_falls_back_to_the_end_sensors(monkeypatch, axis) -> None:
    """Without a horizontal stated axis, forward runs from the last sensor to sensor 0."""
    beamformer = _load_beamformer_module(monkeypatch)
    platform = _dome_platform(axis)

    forward = beamformer.SteeringCalculator._array_forward_rad(platform)

    assert forward == pytest.approx(-np.pi / 2)


def test_towed_array_axis_leaves_the_frame_unchanged(monkeypatch) -> None:
    """A towed array states the same forward axis its end sensors give, so nothing moves."""
    beamformer = _load_beamformer_module(monkeypatch)
    without = _rotated_array_platform(0.4)
    positions = without.array.state_vector
    forward = positions[:, 0] - positions[:, -1]
    stated = SimpleNamespace(
        array=SimpleNamespace(
            state_vector=positions,
            ref_state_vector=without.array.ref_state_vector,
            axis=(forward / np.linalg.norm(forward)).reshape(3, 1),
        )
    )
    calculator = _steering_calculator(beamformer, (0.0, np.pi), 9)

    np.testing.assert_allclose(
        calculator.steering_bearings(stated), calculator.steering_bearings(without), atol=1e-12
    )


@pytest.mark.parametrize("spacing", ["uniform", "sine"])
def test_array_frame_grid_turns_with_the_array(monkeypatch, spacing) -> None:
    """Turning a straight array turns every bearing with it and leaves the delays unchanged.

    For a sine grid this is what lets it steer through a turn, which the world frame refuses.
    """
    beamformer = _load_beamformer_module(monkeypatch)
    calculator = _steering_calculator(beamformer, (0.0, np.pi), 9, spacing=spacing)
    turn_rad = 1.1
    before = _rotated_array_platform(0.0)
    after = _rotated_array_platform(turn_rad)

    rotation = calculator.steering_bearings(after) - calculator.steering_bearings(before)

    np.testing.assert_allclose(np.angle(np.exp(1j * (rotation - turn_rad))), 0.0, atol=1e-12)
    np.testing.assert_allclose(
        calculator.calculate(after), calculator.calculate(before), rtol=1e-12, atol=1e-18
    )


def test_steering_calculator_mirror_half_plane_rejects_partial_sector(monkeypatch) -> None:
    """A partial sector cannot be mirror-paired."""
    beamformer = _load_beamformer_module(monkeypatch)

    with pytest.raises(ValueError, match="full-circle steering_sector_rad"):
        beamformer.SteeringCalculator(
            ssp=ConstantSSP(1500.0),
            steering_sector_rad=(-np.pi / 2, np.pi / 2),
            num_beams=9,
            mirror_half_plane=True,
        )


def test_mirror_plan_requires_mirror_half_plane(monkeypatch) -> None:
    """Calling mirror_plan() without the flag is a usage error, not a silent no-op."""
    beamformer = _load_beamformer_module(monkeypatch)
    calculator = beamformer.SteeringCalculator(
        ssp=ConstantSSP(1500.0), steering_sector_rad=(-np.pi, np.pi), num_beams=8
    )

    with pytest.raises(RuntimeError, match="mirror_half_plane=True"):
        calculator.mirror_plan(_straight_array_platform())


def test_steering_calculator_mirror_half_plane_computes_half_the_grid(monkeypatch) -> None:
    """With mirroring enabled, calculate() should return only the primary-half delays."""
    beamformer = _load_beamformer_module(monkeypatch)
    num_beams = 16
    calculator = beamformer.SteeringCalculator(
        ssp=ConstantSSP(1500.0),
        steering_sector_rad=(-np.pi, np.pi),
        num_beams=num_beams,
        mirror_half_plane=True,
    )
    platform = _straight_array_platform()

    delays = calculator.calculate(platform)
    plan = calculator.mirror_plan(platform)

    assert delays.shape == (int(plan.primary_mask.sum()), 8)
    assert delays.shape[0] < num_beams
    # The array lies exactly on the axis, so no rotation is needed to align back to the grid.
    assert plan.roll_shift == 0
    # mirror_idx is an involution: mirroring twice returns the original beam.
    np.testing.assert_array_equal(plan.mirror_idx[plan.mirror_idx], np.arange(num_beams))


def test_expand_mirrored_reconstructs_full_grid_from_synthetic_half(monkeypatch) -> None:
    """expand_mirrored should exactly reproduce a hand-built, mirror-symmetric array."""
    beamformer = _load_beamformer_module(monkeypatch)
    num_beams = 6
    mirror_idx = (-np.arange(num_beams)) % num_beams  # [0, 5, 4, 3, 2, 1]
    primary_mask = np.arange(num_beams) <= mirror_idx  # True for k in {0, 1, 2, 3}

    # Must respect the pairing itself: index 1 mirrors index 5, index 2 mirrors index 4
    # (0 and 3 are self-mirrored), so a genuine mirror-symmetric map has full[k] ==
    # full[mirror_idx[k]] for every k -- exactly the property a real straight array's
    # beam pattern has.
    full = np.array([10.0, 20.0, 30.0, 40.0, 30.0, 20.0]).reshape(num_beams, 1)
    plan = beamformer.MirrorPlan(primary_mask=primary_mask, mirror_idx=mirror_idx, roll_shift=0)

    primary = full[primary_mask]
    expanded = beamformer.Beamformer.expand_mirrored(primary, plan, direction_axis=0)

    np.testing.assert_array_equal(expanded, full)


def test_expand_mirrored_applies_roll_shift(monkeypatch) -> None:
    """A non-zero roll_shift should rotate the reconstruction onto the display grid."""
    beamformer = _load_beamformer_module(monkeypatch)
    num_beams = 6
    mirror_idx = (-np.arange(num_beams)) % num_beams
    primary_mask = np.arange(num_beams) <= mirror_idx
    plan = beamformer.MirrorPlan(primary_mask=primary_mask, mirror_idx=mirror_idx, roll_shift=2)

    axis_centred = np.array([10.0, 20.0, 30.0, 40.0, 30.0, 20.0]).reshape(num_beams, 1)
    primary = axis_centred[primary_mask]
    expanded = beamformer.Beamformer.expand_mirrored(primary, plan, direction_axis=0)

    np.testing.assert_array_equal(expanded, np.roll(axis_centred, 2, axis=0))


@pytest.mark.parametrize(
    "sector",
    [(-np.pi, np.pi), (0.0, 2 * np.pi), (np.pi / 2, -3 * np.pi / 2)],
    ids=["minus-pi-to-pi", "zero-to-two-pi", "quarter-offset"],
)
def test_delay_and_sum_mirror_plan_matches_full_grid_for_straight_array(
    monkeypatch, sector
) -> None:
    """For a straight array, mirrored beamforming should match a full-grid computation.

    The array here lies exactly on its axis (0 rad), so the reconstruction has no
    rotation/roll approximation to absorb -- this isolates the mirror-pairing logic
    itself and should match a full 360 deg computation to floating-point precision.
    Every way of writing a full circle must match: mirror pairing is only exact when beam
    0 lies on the array axis, which the canonical full circle guarantees (issue #98).
    """
    beamformer = _load_beamformer_module(monkeypatch)
    platform = _straight_array_platform(num_sensors=8, spacing_m=1.0)
    num_beams = 16
    sound_speed = 1500.0
    full_calculator = beamformer.SteeringCalculator(
        ssp=ConstantSSP(sound_speed), steering_sector_rad=sector, num_beams=num_beams
    )
    full_delays = full_calculator.calculate(platform)

    # A coherent tone arriving from one of the primary grid's own bearings, built from the
    # exact geometric delay for that direction -- a physically consistent plane wave.
    source_delay_s = full_delays[2]

    fs = 500.0
    num_samples = 256
    t = np.arange(num_samples) / fs
    f0 = 100.0
    raw_signals = np.exp(1j * 2 * np.pi * f0 * (t[None, :] - source_delay_s[:, None]))
    raw_signals = raw_signals.astype(np.complex128)

    das_kwargs = dict(
        sampling_rate_hz=fs, domain="broadband_power", nfft=64, overlap=32, fmin=50.0, fmax=150.0
    )
    das = beamformer.DelayAndSumBeamformer(**das_kwargs)

    full_power = das.beamform(raw_signals, full_delays)

    mirror_calculator = beamformer.SteeringCalculator(
        ssp=ConstantSSP(sound_speed),
        steering_sector_rad=sector,
        num_beams=num_beams,
        mirror_half_plane=True,
    )
    half_delays = mirror_calculator.calculate(platform)
    plan = mirror_calculator.mirror_plan(platform)
    mirrored_power = das.beamform(raw_signals, half_delays, mirror_plan=plan)

    assert mirrored_power.shape == full_power.shape
    np.testing.assert_allclose(mirrored_power, full_power, rtol=1e-9, atol=1e-9)


def test_world_frame_mirror_plan_offsets_every_beam_when_the_axis_is_off_grid(monkeypatch) -> None:
    """Off the grid, world-frame mirroring steers every beam at its bearing plus a remainder.

    roll_shift rounds the axis to whole beams, so the expanded output matches direct steering
    at the grid shifted by the uncorrected remainder, not the grid itself (issue #99).
    """
    beamformer = _load_beamformer_module(monkeypatch)
    num_beams = 16
    beam_spacing_rad = 2 * np.pi / num_beams
    axis_rad = 0.3 * beam_spacing_rad
    platform = _rotated_array_platform(axis_rad)
    rng = np.random.default_rng(0)
    raw_signals = rng.standard_normal((8, 256)) + 1j * rng.standard_normal((8, 256))
    das = beamformer.DelayAndSumBeamformer(
        sampling_rate_hz=500.0,
        domain="broadband_power",
        nfft=64,
        overlap=32,
        fmin=50.0,
        fmax=150.0,
    )
    mirror = _steering_calculator(
        beamformer, (-np.pi, np.pi), num_beams, mirror_half_plane=True, frame="world"
    )
    plan = mirror.mirror_plan(platform)
    mirrored = das.beamform(raw_signals, mirror.calculate(platform), mirror_plan=plan)

    remainder_rad = axis_rad - plan.roll_shift * beam_spacing_rad
    shifted = mirror.steering_bearings() + remainder_rad
    direct = das.beamform(raw_signals, mirror._delays_for_azimuths(platform, shifted))

    assert remainder_rad != 0.0
    np.testing.assert_allclose(mirrored, direct, rtol=1e-9, atol=1e-9)


def test_array_frame_mirror_plan_is_exact_when_the_axis_is_off_grid(monkeypatch) -> None:
    """In the array frame the mirrored output matches direct steering at the grid itself.

    Beam 0 always lies on the array axis, so no roll back onto the grid is needed and the
    world frame's off-grid offset (issue #99) does not arise.
    """
    beamformer = _load_beamformer_module(monkeypatch)
    num_beams = 16
    platform = _rotated_array_platform(0.3 * 2 * np.pi / num_beams)
    rng = np.random.default_rng(0)
    raw_signals = rng.standard_normal((8, 256)) + 1j * rng.standard_normal((8, 256))
    das = beamformer.DelayAndSumBeamformer(
        sampling_rate_hz=500.0,
        domain="broadband_power",
        nfft=64,
        overlap=32,
        fmin=50.0,
        fmax=150.0,
    )
    mirror = _steering_calculator(beamformer, (-np.pi, np.pi), num_beams, mirror_half_plane=True)
    plan = mirror.mirror_plan(platform)
    mirrored = das.beamform(raw_signals, mirror.calculate(platform), mirror_plan=plan)

    bearings = mirror.steering_bearings(platform)
    direct = das.beamform(raw_signals, mirror._delays_for_azimuths(platform, bearings))

    assert plan.roll_shift == 0
    np.testing.assert_allclose(mirrored, direct, rtol=1e-9, atol=1e-9)


def _vertical_array_platform(num_sensors: int = 4, spacing_m: float = 1.0) -> SimpleNamespace:
    """Build a fake platform whose sensors hang in a vertical line below sensor 0."""
    z = -10.0 - np.arange(num_sensors, dtype=np.float64) * spacing_m
    state_vector = np.array([np.zeros(num_sensors), np.zeros(num_sensors), z])
    return SimpleNamespace(
        array=SimpleNamespace(state_vector=state_vector, ref_state_vector=state_vector[:, [0]])
    )


def _crossed_array_platform(num_sensors: int = 16, spacing_m: float = 0.5) -> SimpleNamespace:
    """Build a fake platform with a line of sensors along +x and another down -z from sensor 0."""
    offsets = np.arange(num_sensors, dtype=np.float64) * spacing_m
    horizontal = np.array([offsets, np.zeros(num_sensors), np.full(num_sensors, -50.0)])
    vertical = np.array(
        [np.zeros(num_sensors - 1), np.zeros(num_sensors - 1), -50.0 - offsets[1:]]
    )
    state_vector = np.hstack([horizontal, vertical])
    return SimpleNamespace(
        array=SimpleNamespace(state_vector=state_vector, ref_state_vector=state_vector[:, [0]])
    )


def _world_calculator(beamformer, elevation_rad: float = 0.0, **kwargs):
    """Build a world-frame calculator over the first quadrant at one elevation."""
    settings = {"steering_sector_rad": (0.0, np.pi / 2), "num_beams": 4, **kwargs}
    return beamformer.SteeringCalculator(
        ssp=ConstantSSP(1500.0), frame="world", steering_elevation_rad=elevation_rad, **settings
    )


def test_zero_elevation_leaves_delays_bit_identical(monkeypatch) -> None:
    """The default elevation reproduces the horizontal delays exactly."""
    beamformer = _load_beamformer_module(monkeypatch)
    platform = _crossed_array_platform(num_sensors=4)
    calculator = _world_calculator(beamformer)
    bearings = calculator.steering_bearings(platform)
    relative = platform.array.state_vector - platform.array.ref_state_vector
    horizontal = np.array([np.cos(bearings), np.sin(bearings), np.zeros_like(bearings)])

    np.testing.assert_array_equal(
        calculator.calculate(platform), -(horizontal.T @ relative) / 1500.0
    )


@pytest.mark.parametrize("elevation_deg", [-60.0, -3.0, 0.0, 25.0, 90.0])
def test_vertical_array_delays_follow_the_sine_of_elevation(monkeypatch, elevation_deg) -> None:
    """A vertical line's delays are -dz sin(e) / c on every bearing; at e = 0 they vanish."""
    beamformer = _load_beamformer_module(monkeypatch)
    platform = _vertical_array_platform()
    elevation_rad = np.radians(elevation_deg)
    dz = platform.array.state_vector[2] - platform.array.ref_state_vector[2]

    delays = _world_calculator(beamformer, elevation_rad).calculate(platform)

    expected = np.broadcast_to(-dz * np.sin(elevation_rad) / 1500.0, delays.shape)
    np.testing.assert_allclose(delays, expected, rtol=0.0, atol=1e-15)


@pytest.mark.parametrize("elevation_deg", [-45.0, -3.0, 30.0])
def test_horizontal_array_delays_scale_with_the_cosine_of_elevation(
    monkeypatch, elevation_deg
) -> None:
    """Steering a horizontal line off the horizontal shrinks its delays by cos(e)."""
    beamformer = _load_beamformer_module(monkeypatch)
    platform = _straight_array_platform()
    elevation_rad = np.radians(elevation_deg)

    level = _world_calculator(beamformer).calculate(platform)
    tilted = _world_calculator(beamformer, elevation_rad).calculate(platform)

    np.testing.assert_allclose(tilted, np.cos(elevation_rad) * level, rtol=1e-12, atol=1e-18)


@pytest.mark.parametrize("elevation_rad", [-np.pi / 2 - 1e-6, np.pi / 2 + 1e-6, 2.0])
def test_steering_calculator_rejects_elevation_beyond_vertical(monkeypatch, elevation_rad) -> None:
    """Elevations outside [-pi/2, pi/2] do not name a direction uniquely, so they raise."""
    beamformer = _load_beamformer_module(monkeypatch)

    with pytest.raises(ValueError, match="steering_elevation_rad"):
        _world_calculator(beamformer, elevation_rad)


def test_mirror_plan_matches_full_grid_off_the_horizontal(monkeypatch) -> None:
    """Mirror pairing stays exact for a straight horizontal array steered below horizontal."""
    beamformer = _load_beamformer_module(monkeypatch)
    platform = _straight_array_platform(num_sensors=8, spacing_m=1.0)
    settings = {
        "ssp": ConstantSSP(1500.0),
        "steering_sector_rad": (-np.pi, np.pi),
        "num_beams": 16,
        "steering_elevation_rad": np.radians(-3.0),
    }
    full_calculator = beamformer.SteeringCalculator(**settings)
    mirror_calculator = beamformer.SteeringCalculator(**settings, mirror_half_plane=True)
    full_delays = full_calculator.calculate(platform)

    fs = 500.0
    t = np.arange(256) / fs
    raw_signals = np.exp(2j * np.pi * 100.0 * (t[None, :] - full_delays[2][:, None]))
    das = beamformer.DelayAndSumBeamformer(
        sampling_rate_hz=fs, domain="broadband_power", nfft=64, overlap=32, fmin=50.0, fmax=150.0
    )

    full_power = das.beamform(raw_signals, full_delays)
    mirrored_power = das.beamform(
        raw_signals,
        mirror_calculator.calculate(platform),
        mirror_plan=mirror_calculator.mirror_plan(platform),
    )

    np.testing.assert_allclose(mirrored_power, full_power, rtol=1e-9, atol=1e-9)


def test_delay_and_sum_locates_a_plane_wave_in_bearing_and_elevation(monkeypatch) -> None:
    """Scanning elevations with one calculator each finds a broadband plane wave's direction.

    A crossed horizontal and vertical line resolves both angles. On a 1 deg grid the
    beam-power peak is within a grid step of the source, which sits between grid points. On
    0.05 deg cuts through the source, the peak is at its bearing and elevation, so the
    steering geometry adds no bias of its own.
    """
    beamformer = _load_beamformer_module(monkeypatch)
    platform = _crossed_array_platform()
    true_bearing_rad, true_elevation_rad = np.radians(30.4), np.radians(-10.3)

    def calculator(sector_deg, num_beams, elevation_rad):
        return beamformer.SteeringCalculator(
            ssp=ConstantSSP(1500.0),
            steering_sector_rad=tuple(np.radians(sector_deg)),
            num_beams=num_beams,
            frame="world",
            steering_elevation_rad=elevation_rad,
        )

    source = calculator((0.0, 1.0), 2, true_elevation_rad)
    arrival_delays = source._delays_for_azimuths(platform, np.array([true_bearing_rad]))[0]
    fs, num_samples = 4000.0, 4096
    rng = np.random.default_rng(7)
    spectrum = np.fft.fft(rng.normal(size=num_samples) + 1j * rng.normal(size=num_samples))
    f = np.fft.fftfreq(num_samples, 1 / fs)
    raw_signals = np.fft.ifft(spectrum * np.exp(-2j * np.pi * f * arrival_delays[:, None]), axis=1)
    das = beamformer.DelayAndSumBeamformer(
        sampling_rate_hz=fs,
        domain="broadband_power",
        nfft=256,
        overlap=128,
        fmin=500.0,
        fmax=1400.0,
    )

    def power(sector_deg, num_beams, elevation_rad):
        steering = calculator(sector_deg, num_beams, elevation_rad)
        return steering.steering_bearings(platform), das.beamform(
            raw_signals, steering.calculate(platform)
        ).sum(axis=1)

    elevations_rad = np.radians(np.arange(-30.0, 31.0))
    coarse = np.array([power((0.0, 60.0), 61, elevation)[1] for elevation in elevations_rad])
    i_el, i_az = np.unravel_index(np.argmax(coarse), coarse.shape)
    assert abs(i_az - 30.4) <= 1.0
    assert abs(np.degrees(elevations_rad[i_el]) - (-10.3)) <= 1.0

    bearings, bearing_cut = power((28.0, 33.0), 101, true_elevation_rad)
    assert bearings[np.argmax(bearing_cut)] == pytest.approx(true_bearing_rad, abs=1e-9)
    fine_elevations_rad = np.radians(np.linspace(-12.5, -7.5, 101))
    elevation_cut = [power((30.4, 30.45), 2, elevation)[1][0] for elevation in fine_elevations_rad]
    assert fine_elevations_rad[np.argmax(elevation_cut)] == pytest.approx(
        true_elevation_rad, abs=1e-9
    )
