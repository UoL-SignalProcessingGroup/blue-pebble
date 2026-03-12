"""Tests for ``bluepebble.types.sensordata``."""

from __future__ import annotations

from datetime import datetime

import numpy as np

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


def test_types_package_exports_passive_sonar_sensor_data(monkeypatch):
    """Types package should re-export ``PassiveSonarSensorData`` in ``__all__``."""
    sensordata_module, types_module = _load_types_modules(monkeypatch)

    assert "PassiveSonarSensorData" in types_module.__all__
    assert types_module.PassiveSonarSensorData is sensordata_module.PassiveSonarSensorData
