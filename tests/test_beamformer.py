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


def test_steering_calculator_returns_expected_horizontal_delays(monkeypatch) -> None:
    """Steering delays should match the 2D projected sensor offsets."""
    beamformer = _load_beamformer_module(monkeypatch)
    calculator = beamformer.SteeringCalculator(
        ssp=ConstantSSP(1500.0),
        steering_azimuths_rad=np.array([0.0, np.pi / 2]),
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


def test_steering_calculator_mirror_half_plane_rejects_non_uniform_grid(monkeypatch) -> None:
    """A non-uniform or partial-circle grid cannot be mirror-paired."""
    beamformer = _load_beamformer_module(monkeypatch)

    with pytest.raises(ValueError, match="uniform, full-circle steering grid"):
        beamformer.SteeringCalculator(
            ssp=ConstantSSP(1500.0),
            steering_azimuths_rad=np.array([0.0, 0.5, 3.0]),
            mirror_half_plane=True,
        )


def test_mirror_plan_requires_mirror_half_plane(monkeypatch) -> None:
    """Calling mirror_plan() without the flag is a usage error, not a silent no-op."""
    beamformer = _load_beamformer_module(monkeypatch)
    calculator = beamformer.SteeringCalculator(
        ssp=ConstantSSP(1500.0),
        steering_azimuths_rad=np.linspace(-np.pi, np.pi, 8, endpoint=False),
    )

    with pytest.raises(RuntimeError, match="mirror_half_plane=True"):
        calculator.mirror_plan(_straight_array_platform())


def test_steering_calculator_mirror_half_plane_computes_half_the_grid(monkeypatch) -> None:
    """With mirroring enabled, calculate() should return only the primary-half delays."""
    beamformer = _load_beamformer_module(monkeypatch)
    num_beams = 16
    calculator = beamformer.SteeringCalculator(
        ssp=ConstantSSP(1500.0),
        steering_azimuths_rad=np.linspace(-np.pi, np.pi, num_beams, endpoint=False),
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


def test_delay_and_sum_mirror_plan_matches_full_grid_for_straight_array(monkeypatch) -> None:
    """For a straight array, mirrored beamforming should match a full-grid computation.

    The array here lies exactly on its axis (0 rad), so the reconstruction has no
    rotation/roll approximation to absorb -- this isolates the mirror-pairing logic
    itself and should match a full 360 deg computation to floating-point precision.
    """
    beamformer = _load_beamformer_module(monkeypatch)
    platform = _straight_array_platform(num_sensors=8, spacing_m=1.0)
    num_beams = 16
    grid = np.linspace(-np.pi, np.pi, num_beams, endpoint=False)
    sound_speed = 1500.0

    # A coherent tone arriving from one of the primary grid's own bearings, built from the
    # exact geometric delay for that direction -- a physically consistent plane wave.
    theta0 = grid[2]
    source_calculator = beamformer.SteeringCalculator(
        ssp=ConstantSSP(sound_speed), steering_azimuths_rad=np.array([theta0])
    )
    source_delay_s = source_calculator.calculate(platform)[0]

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

    full_calculator = beamformer.SteeringCalculator(
        ssp=ConstantSSP(sound_speed), steering_azimuths_rad=grid
    )
    full_power = das.beamform(raw_signals, full_calculator.calculate(platform))

    mirror_calculator = beamformer.SteeringCalculator(
        ssp=ConstantSSP(sound_speed), steering_azimuths_rad=grid, mirror_half_plane=True
    )
    half_delays = mirror_calculator.calculate(platform)
    plan = mirror_calculator.mirror_plan(platform)
    mirrored_power = das.beamform(raw_signals, half_delays, mirror_plan=plan)

    assert mirrored_power.shape == full_power.shape
    np.testing.assert_allclose(mirrored_power, full_power, rtol=1e-9, atol=1e-9)
