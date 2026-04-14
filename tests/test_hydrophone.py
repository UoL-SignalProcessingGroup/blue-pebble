"""Tests for hydrophone LTI system models."""

from __future__ import annotations

import numpy as np
import pytest

from .support import (
    install_fake_stonesoup,
    install_repo_package,
    load_package_module_from_repo,
)


def _load_hydrophone_module(monkeypatch):
    """Load the hydrophone module with minimal dependency scaffolding."""
    install_fake_stonesoup(monkeypatch)
    install_repo_package(monkeypatch, "bluepebble", "bluepebble")
    install_repo_package(monkeypatch, "bluepebble.models", "bluepebble/models")
    return load_package_module_from_repo(
        "bluepebble/models/hydrophone.py",
        "bluepebble.models.hydrophone",
    )


# ---------------------------------------------------------------------------
# FlatFrequencyResponse
# ---------------------------------------------------------------------------


class TestFlatFrequencyResponse:
    """Tests for FlatFrequencyResponse."""

    def test_returns_ones(self, monkeypatch):
        mod = _load_hydrophone_module(monkeypatch)
        flat = mod.FlatFrequencyResponse()
        freqs = np.array([100.0, 500.0, 1000.0, 5000.0])
        result = flat.evaluate(freqs)
        np.testing.assert_array_equal(result, np.ones(4, dtype=np.complex128))
        assert result.dtype == np.complex128

    def test_shape_matches_input(self, monkeypatch):
        mod = _load_hydrophone_module(monkeypatch)
        flat = mod.FlatFrequencyResponse()
        freqs = np.linspace(0, 10000, 128)
        result = flat.evaluate(freqs)
        assert result.shape == (128,)


# ---------------------------------------------------------------------------
# TabulatedFrequencyResponse
# ---------------------------------------------------------------------------


class TestTabulatedFrequencyResponse:
    """Tests for TabulatedFrequencyResponse."""

    def test_magnitude_only_at_tabulated_points(self, monkeypatch):
        mod = _load_hydrophone_module(monkeypatch)
        freqs = np.array([100.0, 1000.0, 10000.0])
        mag_db = np.array([-6.0, 0.0, -3.0])
        tab = mod.TabulatedFrequencyResponse(
            frequencies_hz=freqs,
            magnitude_db=mag_db,
        )
        result = tab.evaluate(freqs)
        expected_mag = 10.0 ** (mag_db / 20.0)
        np.testing.assert_allclose(np.abs(result), expected_mag, rtol=1e-12)
        np.testing.assert_allclose(np.angle(result), 0.0, atol=1e-15)

    def test_magnitude_and_phase(self, monkeypatch):
        mod = _load_hydrophone_module(monkeypatch)
        freqs = np.array([100.0, 1000.0, 10000.0])
        mag_db = np.array([0.0, -6.0, -12.0])
        phase_deg = np.array([0.0, 45.0, 90.0])
        tab = mod.TabulatedFrequencyResponse(
            frequencies_hz=freqs,
            magnitude_db=mag_db,
            phase_deg=phase_deg,
        )
        result = tab.evaluate(freqs)
        expected_mag = 10.0 ** (mag_db / 20.0)
        expected_phase = np.deg2rad(phase_deg)
        np.testing.assert_allclose(np.abs(result), expected_mag, rtol=1e-12)
        np.testing.assert_allclose(np.angle(result), expected_phase, atol=1e-14)

    def test_interpolation_in_db_domain(self, monkeypatch):
        mod = _load_hydrophone_module(monkeypatch)
        freqs = np.array([100.0, 1000.0])
        mag_db = np.array([0.0, -20.0])
        tab = mod.TabulatedFrequencyResponse(
            frequencies_hz=freqs,
            magnitude_db=mag_db,
        )
        # Midpoint in frequency => midpoint in dB
        query = np.array([550.0])
        result = tab.evaluate(query)
        expected_db = -10.0
        expected_mag = 10.0 ** (expected_db / 20.0)
        np.testing.assert_allclose(np.abs(result), expected_mag, rtol=1e-12)

    def test_extrapolation_clamps_to_endpoints(self, monkeypatch):
        mod = _load_hydrophone_module(monkeypatch)
        freqs = np.array([100.0, 1000.0])
        mag_db = np.array([-6.0, -12.0])
        tab = mod.TabulatedFrequencyResponse(
            frequencies_hz=freqs,
            magnitude_db=mag_db,
        )
        # Below and above tabulated range
        query = np.array([10.0, 50000.0])
        result = tab.evaluate(query)
        expected = np.array(
            [
                10.0 ** (-6.0 / 20.0),
                10.0 ** (-12.0 / 20.0),
            ]
        )
        np.testing.assert_allclose(np.abs(result), expected, rtol=1e-12)

    def test_negative_frequencies_mapped_via_abs(self, monkeypatch):
        mod = _load_hydrophone_module(monkeypatch)
        freqs = np.array([100.0, 1000.0])
        mag_db = np.array([-6.0, -12.0])
        tab = mod.TabulatedFrequencyResponse(
            frequencies_hz=freqs,
            magnitude_db=mag_db,
        )
        pos_result = tab.evaluate(np.array([500.0]))
        neg_result = tab.evaluate(np.array([-500.0]))
        np.testing.assert_array_equal(pos_result, neg_result)


# ---------------------------------------------------------------------------
# HydrophoneModel
# ---------------------------------------------------------------------------


class TestFirstOrderHighPassResponse:
    """Tests for FirstOrderHighPassResponse."""

    def test_unity_magnitude_well_above_cutoff(self, monkeypatch):
        mod = _load_hydrophone_module(monkeypatch)
        hp = mod.FirstOrderHighPassResponse(cutoff_hz=100.0)
        f = np.array([1e6])
        result = hp.evaluate(f)
        np.testing.assert_allclose(np.abs(result), 1.0, rtol=1e-4)

    def test_minus_3db_at_cutoff(self, monkeypatch):
        mod = _load_hydrophone_module(monkeypatch)
        hp = mod.FirstOrderHighPassResponse(cutoff_hz=1000.0)
        result = hp.evaluate(np.array([1000.0]))
        np.testing.assert_allclose(np.abs(result), 1.0 / np.sqrt(2), rtol=1e-12)

    def test_rolls_off_below_cutoff(self, monkeypatch):
        mod = _load_hydrophone_module(monkeypatch)
        hp = mod.FirstOrderHighPassResponse(cutoff_hz=1000.0)
        result = hp.evaluate(np.array([100.0]))
        np.testing.assert_allclose(
            np.abs(result), 100.0 / np.sqrt(100.0**2 + 1000.0**2), rtol=1e-12
        )

    def test_phase_at_cutoff_is_45_degrees(self, monkeypatch):
        mod = _load_hydrophone_module(monkeypatch)
        hp = mod.FirstOrderHighPassResponse(cutoff_hz=500.0)
        result = hp.evaluate(np.array([500.0]))
        np.testing.assert_allclose(np.degrees(np.angle(result)), 45.0, atol=1e-10)

    def test_output_dtype_is_complex128(self, monkeypatch):
        mod = _load_hydrophone_module(monkeypatch)
        hp = mod.FirstOrderHighPassResponse(cutoff_hz=1000.0)
        result = hp.evaluate(np.array([100.0, 1000.0, 10000.0]))
        assert result.dtype == np.complex128


class TestFirstOrderLowPassResponse:
    """Tests for FirstOrderLowPassResponse."""

    def test_unity_magnitude_well_below_cutoff(self, monkeypatch):
        mod = _load_hydrophone_module(monkeypatch)
        lp = mod.FirstOrderLowPassResponse(cutoff_hz=10000.0)
        result = lp.evaluate(np.array([1.0]))
        np.testing.assert_allclose(np.abs(result), 1.0, rtol=1e-4)

    def test_minus_3db_at_cutoff(self, monkeypatch):
        mod = _load_hydrophone_module(monkeypatch)
        lp = mod.FirstOrderLowPassResponse(cutoff_hz=1000.0)
        result = lp.evaluate(np.array([1000.0]))
        np.testing.assert_allclose(np.abs(result), 1.0 / np.sqrt(2), rtol=1e-12)

    def test_rolls_off_above_cutoff(self, monkeypatch):
        mod = _load_hydrophone_module(monkeypatch)
        lp = mod.FirstOrderLowPassResponse(cutoff_hz=1000.0)
        result = lp.evaluate(np.array([10000.0]))
        np.testing.assert_allclose(
            np.abs(result), 1000.0 / np.sqrt(1000.0**2 + 10000.0**2), rtol=1e-12
        )

    def test_phase_at_cutoff_is_minus_45_degrees(self, monkeypatch):
        mod = _load_hydrophone_module(monkeypatch)
        lp = mod.FirstOrderLowPassResponse(cutoff_hz=500.0)
        result = lp.evaluate(np.array([500.0]))
        np.testing.assert_allclose(np.degrees(np.angle(result)), -45.0, atol=1e-10)

    def test_output_dtype_is_complex128(self, monkeypatch):
        mod = _load_hydrophone_module(monkeypatch)
        lp = mod.FirstOrderLowPassResponse(cutoff_hz=1000.0)
        result = lp.evaluate(np.array([100.0, 1000.0, 10000.0]))
        assert result.dtype == np.complex128


class TestHydrophoneModel:
    """Tests for HydrophoneModel."""

    def test_sensitivity_scaling(self, monkeypatch):
        mod = _load_hydrophone_module(monkeypatch)
        hydro = mod.HydrophoneModel(sensitivity_db=-180.0)
        freqs = np.array([100.0, 500.0, 1000.0])
        result = hydro.transfer_function(freqs)
        expected = 10.0 ** (-180.0 / 20.0)
        np.testing.assert_allclose(np.abs(result), expected, rtol=1e-12)

    def test_sensitivity_with_tabulated_response(self, monkeypatch):
        mod = _load_hydrophone_module(monkeypatch)
        tab = mod.TabulatedFrequencyResponse(
            frequencies_hz=np.array([100.0, 1000.0]),
            magnitude_db=np.array([0.0, -6.0]),
        )
        hydro = mod.HydrophoneModel(
            sensitivity_db=-160.0,
            frequency_response=tab,
        )
        freqs = np.array([100.0, 1000.0])
        result = hydro.transfer_function(freqs)
        sensitivity_linear = 10.0 ** (-160.0 / 20.0)
        response_linear = 10.0 ** (np.array([0.0, -6.0]) / 20.0)
        expected = sensitivity_linear * response_linear
        np.testing.assert_allclose(np.abs(result), expected, rtol=1e-12)

    def test_none_frequency_response_defaults_to_flat(self, monkeypatch):
        mod = _load_hydrophone_module(monkeypatch)
        hydro = mod.HydrophoneModel(sensitivity_db=-170.0, frequency_response=None)
        freqs = np.array([100.0, 1000.0, 5000.0])
        result = hydro.transfer_function(freqs)
        expected = 10.0 ** (-170.0 / 20.0)
        np.testing.assert_allclose(np.abs(result), expected, rtol=1e-12)

    def test_zero_db_sensitivity_with_flat_response_is_unity(self, monkeypatch):
        mod = _load_hydrophone_module(monkeypatch)
        hydro = mod.HydrophoneModel(sensitivity_db=0.0)
        freqs = np.array([100.0, 1000.0])
        result = hydro.transfer_function(freqs)
        np.testing.assert_allclose(result, np.ones(2, dtype=np.complex128), rtol=1e-15)


# ---------------------------------------------------------------------------
# evaluate_hydrophone_transfer_functions
# ---------------------------------------------------------------------------


class TestEvaluateHydrophoneTransferFunctions:
    """Tests for the module-level helper function."""

    def test_single_model_uniform(self, monkeypatch):
        mod = _load_hydrophone_module(monkeypatch)
        hydro = mod.HydrophoneModel(sensitivity_db=-180.0)
        freqs = np.array([100.0, 500.0, 1000.0])
        result = mod.evaluate_hydrophone_transfer_functions(hydro, 4, freqs)
        assert result.shape == (4, 3)
        # All rows should be identical
        for i in range(1, 4):
            np.testing.assert_array_equal(result[0], result[i])

    def test_per_sensor_list(self, monkeypatch):
        mod = _load_hydrophone_module(monkeypatch)
        models = [
            mod.HydrophoneModel(sensitivity_db=-180.0),
            mod.HydrophoneModel(sensitivity_db=-170.0),
            mod.HydrophoneModel(sensitivity_db=-160.0),
        ]
        freqs = np.array([100.0, 1000.0])
        result = mod.evaluate_hydrophone_transfer_functions(models, 3, freqs)
        assert result.shape == (3, 2)
        # Each row should differ
        for i, model in enumerate(models):
            expected = 10.0 ** (model.sensitivity_db / 20.0)
            np.testing.assert_allclose(np.abs(result[i]), expected, rtol=1e-12)

    def test_wrong_length_list_raises(self, monkeypatch):
        mod = _load_hydrophone_module(monkeypatch)
        models = [
            mod.HydrophoneModel(sensitivity_db=-180.0),
            mod.HydrophoneModel(sensitivity_db=-170.0),
        ]
        freqs = np.array([100.0])
        with pytest.raises(ValueError, match="must match"):
            mod.evaluate_hydrophone_transfer_functions(models, 5, freqs)
