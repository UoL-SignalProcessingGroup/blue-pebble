"""Tests for passive detector behaviour."""

from __future__ import annotations

import sys
from datetime import datetime
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from .support import (
    install_fake_stonesoup,
    install_repo_package,
    load_package_module_from_repo,
)


def _install_fake_passive_dependencies(monkeypatch) -> None:
    """Install minimal Stone Soup and detector dependencies for passive detector imports."""
    install_fake_stonesoup(monkeypatch)

    stonesoup_module = sys.modules["stonesoup"]
    stonesoup_base_module = sys.modules["stonesoup.base"]

    buffered_generator_module = ModuleType("stonesoup.buffered_generator")

    class FakeBufferedGenerator:
        """Minimal decorator container for generator methods."""

        @staticmethod
        def generator_method(func):
            """Return the wrapped generator unchanged for tests."""
            return func

    buffered_generator_module.BufferedGenerator = FakeBufferedGenerator

    reader_module = ModuleType("stonesoup.reader")
    reader_base_module = ModuleType("stonesoup.reader.base")
    reader_base_module.DetectionReader = type(
        "DetectionReader",
        (stonesoup_base_module.Base,),
        {},
    )
    reader_module.base = reader_base_module

    types_module = ModuleType("stonesoup.types")
    detection_module = ModuleType("stonesoup.types.detection")

    class FakeDetection:
        """Simple hashable detection object for set insertion assertions."""

        def __init__(self, state_vector, timestamp, metadata):
            self.state_vector = np.asarray(state_vector, dtype=float)
            self.timestamp = timestamp
            self.metadata = metadata

    detection_module.Detection = FakeDetection
    types_module.detection = detection_module

    sensordata_module = ModuleType("stonesoup.types.sensordata")
    sensordata_module.SensorData = type("SensorData", (), {})
    types_module.sensordata = sensordata_module

    stonesoup_module.buffered_generator = buffered_generator_module
    stonesoup_module.reader = reader_module
    stonesoup_module.types = types_module

    algorithms_module = ModuleType("bluepebble.detector.algorithms")
    algorithms_module.DetectionAlgorithm = type("DetectionAlgorithm", (), {})

    monkeypatch.setitem(
        sys.modules,
        "stonesoup.buffered_generator",
        buffered_generator_module,
    )
    monkeypatch.setitem(sys.modules, "stonesoup.reader", reader_module)
    monkeypatch.setitem(sys.modules, "stonesoup.reader.base", reader_base_module)
    monkeypatch.setitem(sys.modules, "stonesoup.types", types_module)
    monkeypatch.setitem(sys.modules, "stonesoup.types.detection", detection_module)
    monkeypatch.setitem(sys.modules, "stonesoup.types.sensordata", sensordata_module)
    monkeypatch.setitem(sys.modules, "bluepebble.detector.algorithms", algorithms_module)

    # Stub bluepebble.types.sensordata so that loading passive.py does not import
    # the real module against the fake Stone Soup environment.  Left in sys.modules
    # via monkeypatch so it is cleaned up after the test and does not corrupt the
    # PassiveSonarSensorData class used by other test modules.
    bp_types_module = ModuleType("bluepebble.types")
    bp_sensordata_module = ModuleType("bluepebble.types.sensordata")
    bp_sensordata_module.PassiveSonarSensorData = type("PassiveSonarSensorData", (), {})
    bp_types_module.sensordata = bp_sensordata_module
    monkeypatch.setitem(sys.modules, "bluepebble.types", bp_types_module)
    monkeypatch.setitem(sys.modules, "bluepebble.types.sensordata", bp_sensordata_module)


def _load_passive_detector_module(monkeypatch):
    """Load ``detector/passive.py`` with lightweight dependency scaffolding."""
    _install_fake_passive_dependencies(monkeypatch)
    install_repo_package(monkeypatch, "bluepebble", "bluepebble")
    install_repo_package(monkeypatch, "bluepebble.detector", "bluepebble/detector")
    return load_package_module_from_repo(
        "bluepebble/detector/passive.py",
        "bluepebble.detector.passive",
    )


def test_run_detection_chain_uses_sparse_map_between_algorithms(monkeypatch) -> None:
    """Each chain stage should only pass forward detected indices and values."""
    passive = _load_passive_detector_module(monkeypatch)

    class FirstStage:
        def detect(self, data_map):
            return np.array([[1.0, data_map[1]], [3.0, data_map[3]]])

    class SecondStage:
        def detect(self, data_map):
            np.testing.assert_array_equal(
                np.isneginf(data_map),
                np.array([True, False, True, False]),
            )
            np.testing.assert_allclose(data_map[[1, 3]], np.array([10.0, 5.0]))
            return np.array([[3.0, data_map[3]]])

    detector = passive.PassiveSonarDetector(
        detection_chain=[FirstStage(), SecondStage()],
        sensor_data_gen=iter(()),
        steering_azimuths_rad=np.array([], dtype=float),
    )

    final_detections = detector._run_detection_chain(np.array([1.0, 10.0, 2.0, 5.0]))

    np.testing.assert_array_equal(final_detections, np.array([[3.0, 5.0]]))


def test_detections_gen_power_mode_emits_bearing_detections(monkeypatch) -> None:
    """Power mode should create detections at the expected steering index."""
    passive = _load_passive_detector_module(monkeypatch)
    timestamp = datetime(2026, 1, 1, 12, 0, 0)
    sensor_data = SimpleNamespace(
        beamformed_data=np.array([[1.0 + 0.0j, 1.0 + 0.0j], [2.0 + 0.0j, 2.0 + 0.0j]]),
        timestamp=timestamp,
    )

    class SelectSecondBeam:
        def detect(self, data_map):
            return np.array([[1.0, data_map[1]]])

    detector = passive.PassiveSonarDetector(
        detection_chain=[SelectSecondBeam()],
        sensor_data_gen=iter([(timestamp, [sensor_data])]),
        steering_azimuths_rad=np.array([0.1, 0.5]),
    )

    generated = list(detector.detections_gen(beamformer_output_type="power"))

    assert len(generated) == 1
    yielded_timestamp, detections = generated[0]
    assert yielded_timestamp == timestamp
    assert len(detections) == 1

    detection = next(iter(detections))
    assert detection.timestamp == timestamp
    assert float(detection.state_vector[0, 0]) == pytest.approx(0.5)
    assert detection.metadata["snr_db"] == pytest.approx(4.0)
    np.testing.assert_allclose(detector.snr_history, np.array([[1.0, 4.0]]))


def test_detections_gen_rejects_unknown_output_type(monkeypatch) -> None:
    """Unsupported beamformer output modes should raise a clear validation error."""
    passive = _load_passive_detector_module(monkeypatch)
    timestamp = datetime(2026, 1, 1, 12, 0, 0)
    sensor_data = SimpleNamespace(
        beamformed_data=np.array([[1.0 + 0.0j, 1.0 + 0.0j]]),
        timestamp=timestamp,
    )

    detector = passive.PassiveSonarDetector(
        detection_chain=[],
        sensor_data_gen=iter([(timestamp, [sensor_data])]),
        steering_azimuths_rad=np.array([0.1]),
    )

    with pytest.raises(ValueError, match="Unsupported beamformer_output_type"):
        next(detector.detections_gen(beamformer_output_type="unknown"))
