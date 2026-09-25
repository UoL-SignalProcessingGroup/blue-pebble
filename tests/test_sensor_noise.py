"""Tests for sensor self-noise spectral models."""

from __future__ import annotations

import numpy as np
import pytest
from stonesoup.types.state import State

from .support import (
    install_fake_stonesoup,
    install_repo_package,
    load_package_module_from_repo,
)


def _load_noise_module(monkeypatch):
    """Load the noise module with minimal dependency scaffolding."""
    install_fake_stonesoup(monkeypatch)
    install_repo_package(monkeypatch, "bluepebble", "bluepebble")
    install_repo_package(monkeypatch, "bluepebble.sensor", "bluepebble/sensor")
    return load_package_module_from_repo(
        "bluepebble/sensor/noise.py",
        "bluepebble.sensor.noise",
    )


# Flow state comfortably inside the turbulent regime (Re_x ~ 7.4e6).
_SPEED_MPS = 10.0
_POSITION_M = 1.0


def _state(speed_mps: float = _SPEED_MPS) -> State:
    """Return a 3-D position-velocity State with all speed on the x-axis."""
    return State(state_vector=np.array([[0.0], [speed_mps], [0.0], [0.0], [0.0], [0.0]]))


class TestGoodyFlowNoiseSpectrum:
    """Tests for GoodyFlowNoiseSpectrum."""

    def test_declares_pressure_domain(self, monkeypatch):
        """Flow noise acts in the pressure domain."""
        mod = _load_noise_module(monkeypatch)
        assert mod.GoodyFlowNoiseSpectrum.domain == "pressure"
        assert mod.GoodyFlowNoiseSpectrum().domain == "pressure"

    def test_psd_is_non_negative_and_finite(self, monkeypatch):
        """PSD values are finite and non-negative across the band."""
        mod = _load_noise_module(monkeypatch)
        model = mod.GoodyFlowNoiseSpectrum()
        freqs = np.logspace(0, 4, 64)
        psd = model.psd(freqs, _state(), streamwise_position_m=_POSITION_M)
        assert psd.shape == freqs.shape
        assert np.all(np.isfinite(psd))
        assert np.all(psd >= 0.0)

    def test_negative_frequencies_folded(self, monkeypatch):
        """Negative frequencies are handled via their absolute value."""
        mod = _load_noise_module(monkeypatch)
        model = mod.GoodyFlowNoiseSpectrum()
        freqs = np.array([100.0, 500.0, 2000.0])
        psd_pos = model.psd(freqs, _state(), streamwise_position_m=_POSITION_M)
        psd_neg = model.psd(-freqs, _state(), streamwise_position_m=_POSITION_M)
        np.testing.assert_allclose(psd_pos, psd_neg, rtol=1e-12)

    def test_low_frequency_plateau_slope(self, monkeypatch):
        """Below the outer-scale corner, the spectrum follows the Goody omega^2 rise."""
        mod = _load_noise_module(monkeypatch)
        model = mod.GoodyFlowNoiseSpectrum()
        freqs = np.array([1.0, 2.0, 4.0, 8.0])
        psd = model.psd(freqs, _state(), streamwise_position_m=_POSITION_M)
        slope, _ = np.polyfit(np.log10(freqs), np.log10(psd), 1)
        assert slope == pytest.approx(2.0, abs=0.15)

    def test_high_frequency_rolloff_slope(self, monkeypatch):
        """Above the viscous corner, PSD decays as omega^-5."""
        mod = _load_noise_module(monkeypatch)
        model = mod.GoodyFlowNoiseSpectrum()
        freqs = np.logspace(5, 6, 16)
        psd = model.psd(freqs, _state(), streamwise_position_m=_POSITION_M)
        slope, _ = np.polyfit(np.log10(freqs), np.log10(psd), 1)
        assert slope == pytest.approx(-5.0, abs=0.15)

    def test_psd_increases_monotonically_with_speed(self, monkeypatch):
        """At every frequency of interest, faster flow yields higher PSD."""
        mod = _load_noise_module(monkeypatch)
        model = mod.GoodyFlowNoiseSpectrum()
        freqs = np.logspace(1, 4, 32)

        psd_slow = model.psd(freqs, _state(speed_mps=5.0), streamwise_position_m=_POSITION_M)
        psd_fast = model.psd(freqs, _state(speed_mps=10.0), streamwise_position_m=_POSITION_M)
        assert np.all(psd_fast > psd_slow)

    def test_peak_psd_shifts_to_lower_frequency_with_position(self, monkeypatch):
        """Thicker boundary layer pushes the spectral peak to lower frequency."""
        mod = _load_noise_module(monkeypatch)
        model = mod.GoodyFlowNoiseSpectrum()
        freqs = np.logspace(0, 5, 2048)

        peak_near = freqs[np.argmax(model.psd(freqs, _state(), streamwise_position_m=0.5))]
        peak_far = freqs[np.argmax(model.psd(freqs, _state(), streamwise_position_m=5.0))]
        assert peak_far < peak_near

    def test_speed_is_velocity_norm(self, monkeypatch):
        """Speed is the Euclidean norm of the mapped velocity components."""
        mod = _load_noise_module(monkeypatch)
        model = mod.GoodyFlowNoiseSpectrum()
        freqs = np.array([100.0, 1000.0])

        state_axis = State(state_vector=np.array([[0.0], [10.0], [0.0], [0.0], [0.0], [0.0]]))
        # Same speed magnitude split across all three velocity components.
        component = 10.0 / np.sqrt(3.0)
        state_diagonal = State(
            state_vector=np.array([[0.0], [component], [0.0], [component], [0.0], [component]])
        )

        psd_axis = model.psd(freqs, state_axis, streamwise_position_m=_POSITION_M)
        psd_diag = model.psd(freqs, state_diagonal, streamwise_position_m=_POSITION_M)
        np.testing.assert_allclose(psd_diag, psd_axis, rtol=1e-12)

    def test_velocity_mapping_is_overridable(self, monkeypatch):
        """Non-default velocity_mapping selects different state-vector indices."""
        mod = _load_noise_module(monkeypatch)
        freqs = np.array([100.0, 1000.0])

        # Speed lives at index 0 in this contrived layout.
        state = State(state_vector=np.array([[10.0], [0.0], [0.0]]))
        model = mod.GoodyFlowNoiseSpectrum(velocity_mapping=(0,))

        reference = mod.GoodyFlowNoiseSpectrum().psd(
            freqs, _state(speed_mps=10.0), streamwise_position_m=_POSITION_M
        )
        result = model.psd(freqs, state, streamwise_position_m=_POSITION_M)
        np.testing.assert_allclose(result, reference, rtol=1e-12)

    def test_level_db_conversion(self, monkeypatch):
        """level_db equals 10 log10(psd / 1e-12)."""
        mod = _load_noise_module(monkeypatch)
        model = mod.GoodyFlowNoiseSpectrum()
        freqs = np.array([50.0, 500.0, 5000.0])
        psd = model.psd(freqs, _state(), streamwise_position_m=_POSITION_M)
        expected = 10.0 * np.log10(psd / 1.0e-12)
        np.testing.assert_allclose(
            model.level_db(freqs, _state(), streamwise_position_m=_POSITION_M),
            expected,
            rtol=1e-12,
        )

    def test_missing_streamwise_position_raises(self, monkeypatch):
        """Flow noise requires an explicit streamwise position."""
        mod = _load_noise_module(monkeypatch)
        model = mod.GoodyFlowNoiseSpectrum()
        freqs = np.array([100.0])
        with pytest.raises(ValueError, match="streamwise_position_m is required"):
            model.psd(freqs, _state())

    def test_non_positive_speed_raises(self, monkeypatch):
        """Zero-velocity platform state is rejected."""
        mod = _load_noise_module(monkeypatch)
        model = mod.GoodyFlowNoiseSpectrum()
        freqs = np.array([100.0])
        stationary = State(state_vector=np.zeros((6, 1)))
        with pytest.raises(ValueError, match="Speed derived from platform_state"):
            model.psd(freqs, stationary, streamwise_position_m=_POSITION_M)

    def test_non_positive_position_raises(self, monkeypatch):
        """Zero or negative streamwise position is rejected."""
        mod = _load_noise_module(monkeypatch)
        model = mod.GoodyFlowNoiseSpectrum()
        freqs = np.array([100.0])
        with pytest.raises(ValueError, match="streamwise_position_m must be strictly positive"):
            model.psd(freqs, _state(), streamwise_position_m=0.0)
        with pytest.raises(ValueError, match="streamwise_position_m must be strictly positive"):
            model.psd(freqs, _state(), streamwise_position_m=-0.1)

    def test_laminar_regime_warns(self, monkeypatch):
        """A low Reynolds number triggers a UserWarning."""
        mod = _load_noise_module(monkeypatch)
        model = mod.GoodyFlowNoiseSpectrum()
        freqs = np.array([100.0])
        # Re_x = 0.1 * 0.01 / 1.35e-6 ~ 740 << 5e5
        with pytest.warns(UserWarning, match="Reynolds"):
            model.psd(freqs, _state(speed_mps=0.1), streamwise_position_m=0.01)

    def test_constants_are_overridable(self, monkeypatch):
        """Goody constants can be overridden for sensitivity studies."""
        mod = _load_noise_module(monkeypatch)
        baseline = mod.GoodyFlowNoiseSpectrum()
        tweaked = mod.GoodyFlowNoiseSpectrum(c2=6.0)
        freqs = np.array([100.0, 1000.0])
        psd_baseline = baseline.psd(freqs, _state(), streamwise_position_m=_POSITION_M)
        psd_tweaked = tweaked.psd(freqs, _state(), streamwise_position_m=_POSITION_M)
        np.testing.assert_allclose(psd_tweaked / psd_baseline, 2.0, rtol=0.05)


class TestSensorNoiseSpectrumBase:
    """Tests for behaviour on the abstract base class."""

    def test_level_db_rejects_non_pressure_domain(self, monkeypatch):
        """level_db raises for voltage-domain noise sources."""
        mod = _load_noise_module(monkeypatch)

        class VoltageNoise(mod.SensorNoiseSpectrum):
            domain = "voltage"

            def psd(self, frequencies_hz, platform_state, *, streamwise_position_m=None):
                return np.ones_like(np.asarray(frequencies_hz, dtype=float))

        model = VoltageNoise()
        with pytest.raises(ValueError, match="pressure-domain"):
            model.level_db(np.array([100.0]), _state())
