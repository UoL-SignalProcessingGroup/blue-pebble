"""Tests for ``bluepebble.types.sensordata``."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from tests.support import (
    install_fake_stonesoup,
    install_fake_stonesoup_simulator_modules,
    install_repo_package,
    load_package_module_from_repo,
)


def _load_types_modules(monkeypatch):
    install_fake_stonesoup(monkeypatch)
    install_fake_stonesoup_simulator_modules(monkeypatch)

    install_repo_package(monkeypatch, "bluepebble", "bluepebble")
    install_repo_package(monkeypatch, "bluepebble.types", "bluepebble/types")

    sensordata_module = load_package_module_from_repo(
        "bluepebble/types/sensordata.py",
        "bluepebble.types.sensordata",
    )
    types_module = load_package_module_from_repo(
        "bluepebble/types/__init__.py",
        "bluepebble.types",
    )
    return sensordata_module, types_module


def test_passive_sonar_sensor_data_stores_payload(monkeypatch):
    """Payload fields should preserve assigned raw, beamformed, and timestamp values."""
    sensordata_module, _ = _load_types_modules(monkeypatch)
    payload_cls = sensordata_module.PassiveSonarSensorData

    raw_signals = np.ones((2, 8), dtype=np.complex64)
    beamformed_data = np.arange(6, dtype=np.float64).reshape(2, 3)
    timestamp = datetime(2026, 1, 1, 12, 0, 0)

    payload = payload_cls(
        raw_signals=raw_signals,
        beamformed_data=beamformed_data,
        timestamp=timestamp,
    )

    np.testing.assert_array_equal(payload.raw_signals, raw_signals)
    np.testing.assert_array_equal(payload.beamformed_data, beamformed_data)
    assert payload.timestamp == timestamp


def test_passive_sonar_sensor_data_defaults_beamformed_data_to_none(monkeypatch):
    """Beamformed payload should default to ``None`` when not provided."""
    sensordata_module, _ = _load_types_modules(monkeypatch)
    payload_cls = sensordata_module.PassiveSonarSensorData

    payload = payload_cls(
        raw_signals=np.zeros((1, 4), dtype=np.complex64),
        timestamp=datetime(2026, 1, 1, 12, 0, 0),
    )

    assert payload.beamformed_data is None


def test_passive_sonar_sensor_data_stores_multiband_payload(monkeypatch):
    """A 3D beamformed payload should keep its band axis and the labels naming it."""
    sensordata_module, _ = _load_types_modules(monkeypatch)
    payload_cls = sensordata_module.PassiveSonarSensorData

    beamformed_data = np.arange(24, dtype=np.float64).reshape(2, 3, 4)

    payload = payload_cls(
        raw_signals=np.ones((3, 8), dtype=np.complex64),
        beamformed_data=beamformed_data,
        timestamp=datetime(2026, 1, 1, 12, 0, 0),
        band_labels=["70-80 Hz", "95-105 Hz"],
    )

    np.testing.assert_array_equal(payload.beamformed_data, beamformed_data)
    assert payload.band_labels == ["70-80 Hz", "95-105 Hz"]


def test_passive_sonar_sensor_data_defaults_band_labels_to_none(monkeypatch):
    """Single-band payloads should leave ``band_labels`` unset."""
    sensordata_module, _ = _load_types_modules(monkeypatch)
    payload_cls = sensordata_module.PassiveSonarSensorData

    payload = payload_cls(
        raw_signals=np.zeros((1, 4), dtype=np.complex64),
        beamformed_data=np.zeros((2, 3), dtype=np.float64),
        timestamp=datetime(2026, 1, 1, 12, 0, 0),
    )

    assert payload.band_labels is None


def test_passive_sonar_sensor_data_rejects_band_labels_without_data(monkeypatch):
    """Labels with nothing to label is a configuration error."""
    sensordata_module, _ = _load_types_modules(monkeypatch)
    payload_cls = sensordata_module.PassiveSonarSensorData

    with pytest.raises(ValueError, match="without any beamformed_data"):
        payload_cls(
            raw_signals=np.zeros((1, 4), dtype=np.complex64),
            timestamp=datetime(2026, 1, 1, 12, 0, 0),
            band_labels=["70-80 Hz"],
        )


def test_passive_sonar_sensor_data_rejects_band_labels_on_2d_data(monkeypatch):
    """Labelling a single-band map would mis-key every downstream detection."""
    sensordata_module, _ = _load_types_modules(monkeypatch)
    payload_cls = sensordata_module.PassiveSonarSensorData

    with pytest.raises(ValueError, match="requires 3D beamformed_data"):
        payload_cls(
            raw_signals=np.zeros((1, 4), dtype=np.complex64),
            beamformed_data=np.zeros((2, 3), dtype=np.float64),
            timestamp=datetime(2026, 1, 1, 12, 0, 0),
            band_labels=["70-80 Hz"],
        )


def test_passive_sonar_sensor_data_rejects_band_label_count_mismatch(monkeypatch):
    """Label count must match the band axis, or slices are silently misnamed."""
    sensordata_module, _ = _load_types_modules(monkeypatch)
    payload_cls = sensordata_module.PassiveSonarSensorData

    with pytest.raises(ValueError, match="2 band labels but beamformed_data has 3 bands"):
        payload_cls(
            raw_signals=np.zeros((1, 4), dtype=np.complex64),
            beamformed_data=np.zeros((3, 2, 4), dtype=np.float64),
            timestamp=datetime(2026, 1, 1, 12, 0, 0),
            band_labels=["70-80 Hz", "95-105 Hz"],
        )


def test_types_package_exports_passive_sonar_sensor_data(monkeypatch):
    """Types package should re-export ``PassiveSonarSensorData`` in ``__all__``."""
    sensordata_module, types_module = _load_types_modules(monkeypatch)

    assert "PassiveSonarSensorData" in types_module.__all__
    assert types_module.PassiveSonarSensorData is sensordata_module.PassiveSonarSensorData


@pytest.mark.parametrize(
    ("beamformed_shape", "band_labels"),
    [((3, 4), None), ((2, 3, 4), ["70-80 Hz", "95-105 Hz"])],
    ids=["single-band", "multiband"],
)
def test_passive_sonar_sensor_data_stores_one_bearing_per_beam(
    monkeypatch, beamformed_shape, band_labels
):
    """Bearings match the beam axis: the first, or the second behind a band axis."""
    sensordata_module, _ = _load_types_modules(monkeypatch)
    payload_cls = sensordata_module.PassiveSonarSensorData
    bearings = np.array([-0.5, 0.0, 0.5])

    payload = payload_cls(
        raw_signals=np.zeros((1, 4), dtype=np.complex64),
        beamformed_data=np.zeros(beamformed_shape, dtype=np.float64),
        timestamp=datetime(2026, 1, 1, 12, 0, 0),
        band_labels=band_labels,
        steering_bearings_rad=bearings,
    )

    np.testing.assert_array_equal(payload.steering_bearings_rad, bearings)


@pytest.mark.parametrize(
    ("beamformed_data", "bearings", "match"),
    [
        (None, np.zeros(3), "without any beamformed_data"),
        (np.zeros((3, 4)), np.zeros(4), "one bearing per beam"),
        (np.zeros((3, 4)), np.zeros((3, 1)), "one bearing per beam"),
    ],
    ids=["no-data", "wrong-count", "not-1d"],
)
def test_passive_sonar_sensor_data_rejects_mismatched_bearings(
    monkeypatch, beamformed_data, bearings, match
):
    """Bearings that don't line up with the beams would mislabel every detection."""
    sensordata_module, _ = _load_types_modules(monkeypatch)
    payload_cls = sensordata_module.PassiveSonarSensorData

    with pytest.raises(ValueError, match=match):
        payload_cls(
            raw_signals=np.zeros((1, 4), dtype=np.complex64),
            beamformed_data=beamformed_data,
            timestamp=datetime(2026, 1, 1, 12, 0, 0),
            steering_bearings_rad=bearings,
        )


def test_active_sonar_sensor_data_stores_one_row_per_sensor(monkeypatch):
    """Active sensor data keeps the waveform, pulse and ping time, one row per sensor."""
    sensordata_module, types_module = _load_types_modules(monkeypatch)
    received = np.zeros((3, 100), dtype=np.complex128)
    pulse = np.ones(20, dtype=np.complex128)
    timestamp = datetime(2026, 1, 1, 12, 0, 0)

    data = sensordata_module.ActiveSonarSensorData(
        received_waveform=received, transmit_pulse=pulse, timestamp=timestamp
    )

    assert data.num_sensors == 3
    np.testing.assert_array_equal(data.transmit_pulse, pulse)
    assert data.timestamp == timestamp
    assert "ActiveSonarSensorData" in types_module.__all__


@pytest.mark.parametrize(
    ("received_shape", "pulse_shape", "match"),
    [((100,), (20,), "one row"), ((1, 100), (1, 20), "one-dimensional")],
)
def test_active_sonar_sensor_data_rejects_wrong_shapes(
    monkeypatch, received_shape, pulse_shape, match
):
    """A 1-D received waveform or a 2-D pulse is rejected."""
    sensordata_module, _ = _load_types_modules(monkeypatch)

    with pytest.raises(ValueError, match=match):
        sensordata_module.ActiveSonarSensorData(
            received_waveform=np.zeros(received_shape, dtype=np.complex128),
            transmit_pulse=np.ones(pulse_shape, dtype=np.complex128),
            timestamp=datetime(2026, 1, 1),
        )


def test_sensor_data_is_at_passband_unless_given_a_demodulation_frequency(monkeypatch):
    """Both payloads default to passband (0 Hz) and keep a carrier they are given."""
    sensordata_module, _ = _load_types_modules(monkeypatch)
    timestamp = datetime(2026, 1, 1)
    passive = dict(raw_signals=np.zeros((1, 4), dtype=np.complex64), timestamp=timestamp)
    active = dict(
        received_waveform=np.zeros((1, 100), dtype=np.complex128),
        transmit_pulse=np.ones(20, dtype=np.complex128),
        timestamp=timestamp,
    )

    for payload_cls, fields in (
        (sensordata_module.PassiveSonarSensorData, passive),
        (sensordata_module.ActiveSonarSensorData, active),
    ):
        assert payload_cls(**fields).demodulation_frequency_hz == 0.0
        with_carrier = payload_cls(**fields, demodulation_frequency_hz=5400.0)
        assert with_carrier.demodulation_frequency_hz == 5400.0
