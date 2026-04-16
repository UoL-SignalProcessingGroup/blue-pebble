"""Tests for beamforming helpers and implementations."""

from __future__ import annotations

import sys
from types import ModuleType

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
    """Load ``beamformer.py`` with lightweight Stone Soup and model scaffolding."""
    install_fake_stonesoup(monkeypatch)
    install_fake_stonesoup_plotter_modules(monkeypatch)
    install_repo_package(monkeypatch, "bluepebble", "bluepebble")
    install_repo_package(monkeypatch, "bluepebble.sigproc", "bluepebble/sigproc")
    install_repo_package(monkeypatch, "bluepebble.models", "bluepebble/models")

    environment_module = ModuleType("bluepebble.models.environment")
    environment_module.SoundSpeedProfile = ConstantSSP
    monkeypatch.setitem(sys.modules, "bluepebble.models.environment", environment_module)

    return load_package_module_from_repo(
        "bluepebble/sigproc/beamformer.py",
        "bluepebble.sigproc.beamformer",
    )


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


def test_delay_and_sum_stft_rejects_short_inputs(monkeypatch) -> None:
    """The internal STFT helper should reject signals shorter than ``nfft``."""
    beamformer = _load_beamformer_module(monkeypatch)

    with pytest.raises(ValueError, match="less than window size"):
        beamformer.DelayAndSumBeamformer._stft(
            np.ones((2, 3), dtype=np.complex128),
            nfft=4,
            overlap=0,
        )


def test_delay_and_sum_stft_uses_unit_hop_when_overlap_exceeds_window(monkeypatch) -> None:
    """Overlap larger than ``nfft`` should fall back to a hop of one sample."""
    beamformer = _load_beamformer_module(monkeypatch)

    stft = beamformer.DelayAndSumBeamformer._stft(
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


def test_steering_calculator_returns_expected_horizontal_delays(monkeypatch) -> None:
    """Steering delays should match the 2D projected sensor offsets."""
    from datetime import datetime

    from stonesoup.types.array import StateVector
    from stonesoup.types.state import State

    from bluepebble.sensor.array import LinearHydrophoneArray
    from bluepebble.sensor.hydrophone import Hydrophone, HydrophoneResponse

    beamformer = _load_beamformer_module(monkeypatch)
    calculator = beamformer.SteeringCalculator(
        ssp=ConstantSSP(1500.0),
        steering_azimuths_rad=np.array([0.0, np.pi / 2]),
    )

    ts = datetime(2024, 1, 1)
    response = HydrophoneResponse(sensitivity_db=0.0)
    elements = [
        Hydrophone(response=response),
        Hydrophone(response=HydrophoneResponse(sensitivity_db=0.0)),
    ]
    elements[0].states.append(State(state_vector=StateVector([0.0, 0.0, -10.0]), timestamp=ts))
    elements[1].states.append(State(state_vector=StateVector([1.0, 0.0, -10.0]), timestamp=ts))
    array = LinearHydrophoneArray(elements=elements, element_spacing_m=1.0)

    delays = calculator.calculate(array, ts)

    np.testing.assert_allclose(
        delays,
        np.array([[0.0, -1.0 / 1500.0], [0.0, 0.0]]),
        atol=1e-18,
    )
