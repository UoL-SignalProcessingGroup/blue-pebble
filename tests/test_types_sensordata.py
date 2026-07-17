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
