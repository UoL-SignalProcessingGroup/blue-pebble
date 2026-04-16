"""Tests for hydrophone LTI system models, Hydrophone element, and LinearHydrophoneArray."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest
from stonesoup.types.array import StateVector
from stonesoup.types.state import State

from .support import (
    install_fake_stonesoup,
    install_repo_package,
    load_package_module_from_repo,
)


def _load_hydrophone_module(monkeypatch):
    """Load the hydrophone module with minimal dependency scaffolding."""
    install_fake_stonesoup(monkeypatch)
    install_repo_package(monkeypatch, "bluepebble", "bluepebble")
    install_repo_package(monkeypatch, "bluepebble.sensor", "bluepebble/sensor")
    return load_package_module_from_repo(
        "bluepebble/sensor/hydrophone.py",
        "bluepebble.sensor.hydrophone",
    )


def _load_array_module(monkeypatch):
    """Load the array module with minimal dependency scaffolding."""
    install_fake_stonesoup(monkeypatch)
    install_repo_package(monkeypatch, "bluepebble", "bluepebble")
    install_repo_package(monkeypatch, "bluepebble.sensor", "bluepebble/sensor")
    install_repo_package(  # noqa: E501
        monkeypatch, "bluepebble.sensor.hydrophone", "bluepebble/sensor/hydrophone.py"
    )
    return load_package_module_from_repo(
        "bluepebble/sensor/array.py",
        "bluepebble.sensor.array",
    )


# ---------------------------------------------------------------------------
# FlatFrequencyResponse
# ---------------------------------------------------------------------------


class TestFlatFrequencyResponse:
    """Tests for FlatFrequencyResponse."""

    def test_returns_ones(self, monkeypatch):
        """Unity response at every frequency."""
        mod = _load_hydrophone_module(monkeypatch)
        flat = mod.FlatFrequencyResponse()
        freqs = np.array([100.0, 500.0, 1000.0, 5000.0])
        result = flat.evaluate(freqs)
        np.testing.assert_array_equal(result, np.ones(4, dtype=np.complex128))
        assert result.dtype == np.complex128

    def test_shape_matches_input(self, monkeypatch):
        """Output shape equals input shape."""
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
        """Magnitude matches table entries exactly at tabulated frequencies."""
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
        """Magnitude and phase are both interpolated correctly."""
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
        """Midpoint query interpolates in the dB domain."""
        mod = _load_hydrophone_module(monkeypatch)
        freqs = np.array([100.0, 1000.0])
        mag_db = np.array([0.0, -20.0])
        tab = mod.TabulatedFrequencyResponse(
            frequencies_hz=freqs,
            magnitude_db=mag_db,
        )
        query = np.array([550.0])
        result = tab.evaluate(query)
        expected_db = -10.0
        expected_mag = 10.0 ** (expected_db / 20.0)
        np.testing.assert_allclose(np.abs(result), expected_mag, rtol=1e-12)

    def test_extrapolation_clamps_to_endpoints(self, monkeypatch):
        """Out-of-range queries extrapolate from the nearest endpoint."""
        mod = _load_hydrophone_module(monkeypatch)
        freqs = np.array([100.0, 1000.0])
        mag_db = np.array([-6.0, -12.0])
        tab = mod.TabulatedFrequencyResponse(
            frequencies_hz=freqs,
            magnitude_db=mag_db,
        )
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
        """Negative frequencies are treated as their absolute value."""
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
# FirstOrderHighPassResponse
# ---------------------------------------------------------------------------


class TestFirstOrderHighPassResponse:
    """Tests for FirstOrderHighPassResponse."""

    def test_unity_magnitude_well_above_cutoff(self, monkeypatch):
        """Magnitude approaches unity well above the cutoff."""
        mod = _load_hydrophone_module(monkeypatch)
        hp = mod.FirstOrderHighPassResponse(cutoff_hz=100.0)
        f = np.array([1e6])
        result = hp.evaluate(f)
        np.testing.assert_allclose(np.abs(result), 1.0, rtol=1e-4)

    def test_minus_3db_at_cutoff(self, monkeypatch):
        """Magnitude is -3 dB at the cutoff frequency."""
        mod = _load_hydrophone_module(monkeypatch)
        hp = mod.FirstOrderHighPassResponse(cutoff_hz=1000.0)
        result = hp.evaluate(np.array([1000.0]))
        np.testing.assert_allclose(np.abs(result), 1.0 / np.sqrt(2), rtol=1e-12)

    def test_rolls_off_below_cutoff(self, monkeypatch):
        """Magnitude follows expected roll-off below the cutoff."""
        mod = _load_hydrophone_module(monkeypatch)
        hp = mod.FirstOrderHighPassResponse(cutoff_hz=1000.0)
        result = hp.evaluate(np.array([100.0]))
        np.testing.assert_allclose(
            np.abs(result), 100.0 / np.sqrt(100.0**2 + 1000.0**2), rtol=1e-12
        )

    def test_phase_at_cutoff_is_45_degrees(self, monkeypatch):
        """Phase is +45 degrees at the cutoff frequency."""
        mod = _load_hydrophone_module(monkeypatch)
        hp = mod.FirstOrderHighPassResponse(cutoff_hz=500.0)
        result = hp.evaluate(np.array([500.0]))
        np.testing.assert_allclose(np.degrees(np.angle(result)), 45.0, atol=1e-10)

    def test_output_dtype_is_complex128(self, monkeypatch):
        """Output dtype is always complex128."""
        mod = _load_hydrophone_module(monkeypatch)
        hp = mod.FirstOrderHighPassResponse(cutoff_hz=1000.0)
        result = hp.evaluate(np.array([100.0, 1000.0, 10000.0]))
        assert result.dtype == np.complex128


# ---------------------------------------------------------------------------
# FirstOrderLowPassResponse
# ---------------------------------------------------------------------------


class TestFirstOrderLowPassResponse:
    """Tests for FirstOrderLowPassResponse."""

    def test_unity_magnitude_well_below_cutoff(self, monkeypatch):
        """Magnitude approaches unity well below the cutoff."""
        mod = _load_hydrophone_module(monkeypatch)
        lp = mod.FirstOrderLowPassResponse(cutoff_hz=10000.0)
        result = lp.evaluate(np.array([1.0]))
        np.testing.assert_allclose(np.abs(result), 1.0, rtol=1e-4)

    def test_minus_3db_at_cutoff(self, monkeypatch):
        """Magnitude is -3 dB at the cutoff frequency."""
        mod = _load_hydrophone_module(monkeypatch)
        lp = mod.FirstOrderLowPassResponse(cutoff_hz=1000.0)
        result = lp.evaluate(np.array([1000.0]))
        np.testing.assert_allclose(np.abs(result), 1.0 / np.sqrt(2), rtol=1e-12)

    def test_rolls_off_above_cutoff(self, monkeypatch):
        """Magnitude follows expected roll-off above the cutoff."""
        mod = _load_hydrophone_module(monkeypatch)
        lp = mod.FirstOrderLowPassResponse(cutoff_hz=1000.0)
        result = lp.evaluate(np.array([10000.0]))
        np.testing.assert_allclose(
            np.abs(result), 1000.0 / np.sqrt(1000.0**2 + 10000.0**2), rtol=1e-12
        )

    def test_phase_at_cutoff_is_minus_45_degrees(self, monkeypatch):
        """Phase is -45 degrees at the cutoff frequency."""
        mod = _load_hydrophone_module(monkeypatch)
        lp = mod.FirstOrderLowPassResponse(cutoff_hz=500.0)
        result = lp.evaluate(np.array([500.0]))
        np.testing.assert_allclose(np.degrees(np.angle(result)), -45.0, atol=1e-10)

    def test_output_dtype_is_complex128(self, monkeypatch):
        """Output dtype is always complex128."""
        mod = _load_hydrophone_module(monkeypatch)
        lp = mod.FirstOrderLowPassResponse(cutoff_hz=1000.0)
        result = lp.evaluate(np.array([100.0, 1000.0, 10000.0]))
        assert result.dtype == np.complex128


# ---------------------------------------------------------------------------
# HydrophoneResponse
# ---------------------------------------------------------------------------


class TestHydrophoneResponse:
    """Tests for HydrophoneResponse (renamed from HydrophoneModel)."""

    def test_sensitivity_scaling(self, monkeypatch):
        """Linear sensitivity is applied uniformly across frequencies."""
        mod = _load_hydrophone_module(monkeypatch)
        hydro = mod.HydrophoneResponse(sensitivity_db=-180.0)
        freqs = np.array([100.0, 500.0, 1000.0])
        result = hydro.transfer_function(freqs)
        expected = 10.0 ** (-180.0 / 20.0)
        np.testing.assert_allclose(np.abs(result), expected, rtol=1e-12)

    def test_sensitivity_with_tabulated_response(self, monkeypatch):
        """Sensitivity and tabulated response are combined correctly."""
        mod = _load_hydrophone_module(monkeypatch)
        tab = mod.TabulatedFrequencyResponse(
            frequencies_hz=np.array([100.0, 1000.0]),
            magnitude_db=np.array([0.0, -6.0]),
        )
        hydro = mod.HydrophoneResponse(
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
        """None frequency_response applies unity (flat) response."""
        mod = _load_hydrophone_module(monkeypatch)
        hydro = mod.HydrophoneResponse(sensitivity_db=-170.0, frequency_response=None)
        freqs = np.array([100.0, 1000.0, 5000.0])
        result = hydro.transfer_function(freqs)
        expected = 10.0 ** (-170.0 / 20.0)
        np.testing.assert_allclose(np.abs(result), expected, rtol=1e-12)

    def test_phase_offset_rotates_transfer_function(self, monkeypatch):
        """Constant phase offset rotates the transfer function phasor."""
        mod = _load_hydrophone_module(monkeypatch)
        hydro = mod.HydrophoneResponse(sensitivity_db=0.0, phase_offset_deg=90.0)
        freqs = np.array([100.0, 1000.0])
        result = hydro.transfer_function(freqs)
        np.testing.assert_allclose(np.abs(result), 1.0, rtol=1e-12)
        np.testing.assert_allclose(np.degrees(np.angle(result)), 90.0, atol=1e-10)

    def test_phase_offset_is_frequency_independent(self, monkeypatch):
        """Phase offset is the same at every frequency."""
        mod = _load_hydrophone_module(monkeypatch)
        hydro = mod.HydrophoneResponse(sensitivity_db=0.0, phase_offset_deg=45.0)
        freqs = np.linspace(10.0, 10000.0, 50)
        result = hydro.transfer_function(freqs)
        np.testing.assert_allclose(np.degrees(np.angle(result)), 45.0, atol=1e-10)

    def test_phase_offset_default_is_zero(self, monkeypatch):
        """Default phase offset produces zero phase."""
        mod = _load_hydrophone_module(monkeypatch)
        hydro = mod.HydrophoneResponse(sensitivity_db=0.0)
        freqs = np.array([500.0])
        result = hydro.transfer_function(freqs)
        np.testing.assert_allclose(np.angle(result), 0.0, atol=1e-15)

    def test_zero_db_sensitivity_with_flat_response_is_unity(self, monkeypatch):
        """0 dB sensitivity with flat response is unity transfer function."""
        mod = _load_hydrophone_module(monkeypatch)
        hydro = mod.HydrophoneResponse(sensitivity_db=0.0)
        freqs = np.array([100.0, 1000.0])
        result = hydro.transfer_function(freqs)
        np.testing.assert_allclose(result, np.ones(2, dtype=np.complex128), rtol=1e-15)


# ---------------------------------------------------------------------------
# Hydrophone
# ---------------------------------------------------------------------------


class TestHydrophoneElement:
    """Tests for the Hydrophone element class."""

    def test_initial_state_is_none(self, monkeypatch):
        """State is None before any move is called."""
        mod = _load_hydrophone_module(monkeypatch)
        hydro = mod.Hydrophone(response=mod.HydrophoneResponse(sensitivity_db=0.0))
        assert hydro.state is None

    def test_states_list_starts_empty(self, monkeypatch):
        """States list is empty on construction."""
        mod = _load_hydrophone_module(monkeypatch)
        hydro = mod.Hydrophone(response=mod.HydrophoneResponse(sensitivity_db=0.0))
        assert hydro.states == []

    def test_state_returns_last_appended(self, monkeypatch):
        """State property returns the most recently appended State."""
        mod = _load_hydrophone_module(monkeypatch)
        hydro = mod.Hydrophone(response=mod.HydrophoneResponse(sensitivity_db=0.0))
        ts1 = datetime(2024, 1, 1)
        ts2 = datetime(2024, 1, 2)
        s1 = State(state_vector=StateVector([0.0, 0.0, -10.0]), timestamp=ts1)
        s2 = State(state_vector=StateVector([1.0, 0.0, -10.0]), timestamp=ts2)
        hydro.states.append(s1)
        hydro.states.append(s2)
        assert hydro.state is s2


# ---------------------------------------------------------------------------
# LinearHydrophoneArray
# ---------------------------------------------------------------------------


def _make_array(n: int, spacing: float, sensitivity_db: float = 0.0):
    """Build a LinearHydrophoneArray with n Hydrophone elements."""
    from bluepebble.sensor.array import LinearHydrophoneArray
    from bluepebble.sensor.hydrophone import Hydrophone, HydrophoneResponse

    elements = [
        Hydrophone(response=HydrophoneResponse(sensitivity_db=sensitivity_db)) for _ in range(n)
    ]
    return LinearHydrophoneArray(elements=elements, element_spacing_m=spacing)


class TestLinearHydrophoneArray:
    """Tests for LinearHydrophoneArray."""

    def test_num_elements_derived_from_list(self):
        """num_elements equals len(elements)."""
        arr = _make_array(8, 0.15)
        assert arr.num_elements == 8

    def test_state_none_before_move(self):
        """State is None before move() is called."""
        arr = _make_array(4, 0.1)
        assert arr.state is None

    def test_move_appends_states(self):
        """move() appends one State per element per call."""
        arr = _make_array(4, 0.1)
        ts = datetime(2024, 1, 1)
        arr.move(
            leader_pos=[0.0, 0.0, 0.0],
            trail_direction_xy=[-1.0, 0.0],
            cable_length_m=50.0,
            array_depth_m=-10.0,
            timestamp=ts,
        )
        assert all(len(e.states) == 1 for e in arr.elements)
        assert arr.state.timestamp == ts

    def test_move_positions_correct(self):
        """Element positions are spaced correctly along the trail direction."""
        arr = _make_array(3, 2.0)
        ts = datetime(2024, 1, 1)
        arr.move(
            leader_pos=[0.0, 0.0, 0.0],
            trail_direction_xy=[-1.0, 0.0],
            cable_length_m=10.0,
            array_depth_m=0.0,
            timestamp=ts,
        )
        positions = [e.state.state_vector.flatten() for e in arr.elements]
        # cable is horizontal (depth diff = 0), so first element at x=-10
        np.testing.assert_allclose(positions[0], [-10.0, 0.0, 0.0], atol=1e-10)
        np.testing.assert_allclose(positions[1], [-12.0, 0.0, 0.0], atol=1e-10)
        np.testing.assert_allclose(positions[2], [-14.0, 0.0, 0.0], atol=1e-10)

    def test_move_depth_offset_applied(self):
        """All elements are placed at array_depth_m regardless of host depth."""
        arr = _make_array(2, 1.0)
        ts = datetime(2024, 1, 1)
        arr.move(
            leader_pos=[0.0, 0.0, 0.0],
            trail_direction_xy=[-1.0, 0.0],
            cable_length_m=50.0,
            array_depth_m=-20.0,
            timestamp=ts,
        )
        for element in arr.elements:
            assert element.state.state_vector[2, 0] == pytest.approx(-20.0)

    def test_move_cable_too_short_raises(self):
        """ValueError if cable is shorter than depth difference."""
        arr = _make_array(2, 1.0)
        ts = datetime(2024, 1, 1)
        with pytest.raises(ValueError, match="too short"):
            arr.move(
                leader_pos=[0.0, 0.0, 0.0],
                trail_direction_xy=[-1.0, 0.0],
                cable_length_m=5.0,
                array_depth_m=-100.0,
                timestamp=ts,
            )

    def test_element_states_at_returns_correct_states(self):
        """element_states_at returns one state per element at the queried timestamp."""
        arr = _make_array(3, 1.0)
        ts1 = datetime(2024, 1, 1)
        ts2 = datetime(2024, 1, 2)
        for ts in [ts1, ts2]:
            arr.move(
                leader_pos=[0.0, 0.0, 0.0],
                trail_direction_xy=[-1.0, 0.0],
                cable_length_m=10.0,
                array_depth_m=0.0,
                timestamp=ts,
            )
        states = arr.element_states_at(ts1)
        assert len(states) == 3
        assert all(s.timestamp == ts1 for s in states)

    def test_element_states_at_missing_timestamp_raises(self):
        """ValueError if the timestamp was never recorded."""
        arr = _make_array(2, 1.0)
        ts = datetime(2024, 1, 1)
        with pytest.raises(ValueError, match="No state found"):
            arr.element_states_at(ts)

    def test_position_matrix_at_shape(self):
        """position_matrix_at returns a (3, num_elements) matrix."""
        arr = _make_array(5, 1.0)
        ts = datetime(2024, 1, 1)
        arr.move(
            leader_pos=[0.0, 0.0, 0.0],
            trail_direction_xy=[-1.0, 0.0],
            cable_length_m=10.0,
            array_depth_m=0.0,
            timestamp=ts,
        )
        mat = arr.position_matrix_at(ts)
        assert mat.shape == (3, 5)

    def test_transfer_functions_shape(self):
        """transfer_functions returns (num_elements, num_frequencies)."""
        arr = _make_array(4, 1.0)
        freqs = np.array([100.0, 500.0, 1000.0])
        tf = arr.transfer_functions(freqs)
        assert tf.shape == (4, 3)
        assert tf.dtype == np.complex128

    def test_transfer_functions_per_element(self):
        """Each row in transfer_functions matches the corresponding element response."""
        from bluepebble.sensor.array import LinearHydrophoneArray
        from bluepebble.sensor.hydrophone import Hydrophone, HydrophoneResponse

        sensitivities = [-180.0, -170.0, -160.0]
        elements = [
            Hydrophone(response=HydrophoneResponse(sensitivity_db=s)) for s in sensitivities
        ]
        arr = LinearHydrophoneArray(elements=elements, element_spacing_m=1.0)
        freqs = np.array([500.0])
        tf = arr.transfer_functions(freqs)
        for i, s in enumerate(sensitivities):
            expected = 10.0 ** (s / 20.0)
            np.testing.assert_allclose(np.abs(tf[i, 0]), expected, rtol=1e-12)

    def test_reference_element_idx_selects_state(self):
        """State property reflects the reference_element_idx."""
        from bluepebble.sensor.array import LinearHydrophoneArray
        from bluepebble.sensor.hydrophone import Hydrophone, HydrophoneResponse

        elements = [Hydrophone(response=HydrophoneResponse(sensitivity_db=0.0)) for _ in range(4)]
        arr = LinearHydrophoneArray(
            elements=elements, element_spacing_m=1.0, reference_element_idx=2
        )
        ts = datetime(2024, 1, 1)
        arr.move(
            leader_pos=[0.0, 0.0, 0.0],
            trail_direction_xy=[-1.0, 0.0],
            cable_length_m=10.0,
            array_depth_m=0.0,
            timestamp=ts,
        )
        assert arr.state is arr.elements[2].state

    def test_move_accumulates_history(self):
        """Calling move multiple times accumulates states on each element."""
        arr = _make_array(2, 1.0)
        timestamps = [datetime(2024, 1, i) for i in range(1, 4)]
        for ts in timestamps:
            arr.move(
                leader_pos=[0.0, 0.0, 0.0],
                trail_direction_xy=[-1.0, 0.0],
                cable_length_m=10.0,
                array_depth_m=0.0,
                timestamp=ts,
            )
        assert all(len(e.states) == 3 for e in arr.elements)
