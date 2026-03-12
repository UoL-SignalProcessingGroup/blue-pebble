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


def test_detections_gen_snr_percentile_mode_uses_percentile_noise_floor(monkeypatch) -> None:
    """snr_percentile mode should estimate noise from the configured percentile."""
    passive = _load_passive_detector_module(monkeypatch)
    timestamp = datetime(2026, 1, 1, 12, 0, 0)
    # Two beams: power [1, 9]. 10th-percentile noise ≈ 1. SNR ≈ [0, ~9.5] dB.
    beamformed_data = np.array([[1.0 + 0.0j], [3.0 + 0.0j]])
    sensor_data = SimpleNamespace(beamformed_data=beamformed_data, timestamp=timestamp)

    all_detections_class = []

    class CaptureAll:
        def detect(self, data_map):
            all_detections_class.append(data_map.copy())
            return np.array([[1.0, data_map[1]]])

    detector = passive.PassiveSonarDetector(
        detection_chain=[CaptureAll()],
        sensor_data_gen=iter([(timestamp, [sensor_data])]),
        steering_azimuths_rad=np.array([0.1, 0.5]),
    )

    list(detector.detections_gen(beamformer_output_type="snr_percentile", snr_percentile_val=10))

    assert len(all_detections_class) == 1
    snr_map = all_detections_class[0]
    # Beam 1 (power=9) should have higher SNR than beam 0 (power=1)
    assert snr_map[1] > snr_map[0]
    # snr_history should contain one row with two values
    assert detector.snr_history.shape == (1, 2)


def test_detections_gen_median_power_mode_uses_median_noise_floor(monkeypatch) -> None:
    """median_power mode should estimate noise from the median directional power."""
    passive = _load_passive_detector_module(monkeypatch)
    timestamp = datetime(2026, 1, 1, 12, 0, 0)
    # Three beams with equal power — SNR should be ~0 dB for all.
    beamformed_data = np.array([[2.0 + 0.0j], [2.0 + 0.0j], [2.0 + 0.0j]])
    sensor_data = SimpleNamespace(beamformed_data=beamformed_data, timestamp=timestamp)

    captured = []

    class Capture:
        def detect(self, data_map):
            captured.append(data_map.copy())
            return np.empty((0, 2))

    detector = passive.PassiveSonarDetector(
        detection_chain=[Capture()],
        sensor_data_gen=iter([(timestamp, [sensor_data])]),
        steering_azimuths_rad=np.array([0.1, 0.2, 0.3]),
    )

    list(detector.detections_gen(beamformer_output_type="median_power"))

    assert len(captured) == 1
    # All beams have the same power so SNR should be ~0 dB for all
    np.testing.assert_allclose(captured[0], np.zeros(3), atol=1e-6)


def test_detections_gen_log_power_mode_returns_log_power_directly(monkeypatch) -> None:
    """log_power mode should pass 10*log10(mean power) directly to the detection chain."""
    passive = _load_passive_detector_module(monkeypatch)
    timestamp = datetime(2026, 1, 1, 12, 0, 0)
    # Single beam, two samples: mean power = (4+4)/2 = 4 → 10*log10(4) ≈ 6.02 dB
    beamformed_data = np.array([[2.0 + 0.0j, 2.0 + 0.0j]])
    sensor_data = SimpleNamespace(beamformed_data=beamformed_data, timestamp=timestamp)

    captured = []

    class Capture:
        def detect(self, data_map):
            captured.append(data_map.copy())
            return np.empty((0, 2))

    detector = passive.PassiveSonarDetector(
        detection_chain=[Capture()],
        sensor_data_gen=iter([(timestamp, [sensor_data])]),
        steering_azimuths_rad=np.array([0.0]),
    )

    list(detector.detections_gen(beamformer_output_type="log_power"))

    assert len(captured) == 1
    np.testing.assert_allclose(captured[0], [10 * np.log10(4.0)], rtol=1e-5)


def test_detections_gen_skips_none_beamformed_data(monkeypatch) -> None:
    """Sensor data with beamformed_data=None should be silently skipped."""
    passive = _load_passive_detector_module(monkeypatch)
    timestamp = datetime(2026, 1, 1, 12, 0, 0)
    sensor_data = SimpleNamespace(beamformed_data=None, timestamp=timestamp)

    detector = passive.PassiveSonarDetector(
        detection_chain=[],
        sensor_data_gen=iter([(timestamp, [sensor_data])]),
        steering_azimuths_rad=np.array([0.1]),
    )

    result = list(detector.detections_gen())

    assert len(result) == 1
    _, detections = result[0]
    assert len(detections) == 0


def test_detections_gen_skips_empty_beamformed_data(monkeypatch) -> None:
    """Sensor data with a zero-size beamformed array should be silently skipped."""
    passive = _load_passive_detector_module(monkeypatch)
    timestamp = datetime(2026, 1, 1, 12, 0, 0)
    sensor_data = SimpleNamespace(
        beamformed_data=np.empty((0, 0), dtype=complex), timestamp=timestamp
    )

    detector = passive.PassiveSonarDetector(
        detection_chain=[],
        sensor_data_gen=iter([(timestamp, [sensor_data])]),
        steering_azimuths_rad=np.array([0.1]),
    )

    result = list(detector.detections_gen())

    assert len(result) == 1
    _, detections = result[0]
    assert len(detections) == 0


def test_detections_gen_empty_chain_produces_no_detections(monkeypatch) -> None:
    """An empty detection_chain should yield a timestep with no detections."""
    passive = _load_passive_detector_module(monkeypatch)
    timestamp = datetime(2026, 1, 1, 12, 0, 0)
    sensor_data = SimpleNamespace(
        beamformed_data=np.array([[1.0 + 0.0j, 2.0 + 0.0j]]),
        timestamp=timestamp,
    )

    detector = passive.PassiveSonarDetector(
        detection_chain=[],
        sensor_data_gen=iter([(timestamp, [sensor_data])]),
        steering_azimuths_rad=np.array([0.1]),
    )

    result = list(detector.detections_gen())

    assert len(result) == 1
    _, detections = result[0]
    assert len(detections) == 0


def test_snr_history_empty_before_any_detections(monkeypatch) -> None:
    """snr_history should return an empty array on a freshly constructed detector."""
    passive = _load_passive_detector_module(monkeypatch)

    detector = passive.PassiveSonarDetector(
        detection_chain=[],
        sensor_data_gen=iter(()),
        steering_azimuths_rad=np.array([0.1]),
    )

    history = detector.snr_history
    assert history.shape == (0,)
    assert history.dtype == np.float64


def test_detections_gen_accumulates_across_multiple_sensor_data_per_step(monkeypatch) -> None:
    """Multiple sensor_data objects in one timestep should each contribute detections."""
    passive = _load_passive_detector_module(monkeypatch)
    timestamp = datetime(2026, 1, 1, 12, 0, 0)

    # Two sensor data objects at the same timestamp, each with one beam
    sd1 = SimpleNamespace(beamformed_data=np.array([[3.0 + 0.0j]]), timestamp=timestamp)
    sd2 = SimpleNamespace(beamformed_data=np.array([[5.0 + 0.0j]]), timestamp=timestamp)

    class SelectFirst:
        def detect(self, data_map):
            return np.array([[0.0, data_map[0]]])

    detector = passive.PassiveSonarDetector(
        detection_chain=[SelectFirst()],
        sensor_data_gen=iter([(timestamp, [sd1, sd2])]),
        steering_azimuths_rad=np.array([0.7]),
    )

    result = list(detector.detections_gen(beamformer_output_type="power"))

    assert len(result) == 1
    _, detections = result[0]
    # Both sensor_data objects trigger a detection at the same bearing
    assert len(detections) == 2


def test_run_detection_chain_short_circuits_on_empty_stage_output(monkeypatch) -> None:
    """If any stage in the chain returns no detections the chain should stop immediately."""
    passive = _load_passive_detector_module(monkeypatch)

    second_stage_called = []

    class EmptyStage:
        def detect(self, data_map):
            return np.empty((0, 2), dtype=np.float64)

    class ShouldNotRun:
        def detect(self, data_map):
            second_stage_called.append(True)
            return np.array([[0.0, data_map[0]]])

    detector = passive.PassiveSonarDetector(
        detection_chain=[EmptyStage(), ShouldNotRun()],
        sensor_data_gen=iter(()),
        steering_azimuths_rad=np.array([0.1, 0.5]),
    )

    result = detector._run_detection_chain(np.array([1.0, 2.0]))

    assert result.shape == (0, 2)
    assert second_stage_called == []


def test_detections_gen_all_zero_beamformed_data_does_not_raise(monkeypatch) -> None:
    """All-zero beamformed data should not raise due to epsilon guard in SNR computation."""
    passive = _load_passive_detector_module(monkeypatch)
    timestamp = datetime(2026, 1, 1, 12, 0, 0)
    sensor_data = SimpleNamespace(
        beamformed_data=np.zeros((3, 4), dtype=complex), timestamp=timestamp
    )

    detector = passive.PassiveSonarDetector(
        detection_chain=[],
        sensor_data_gen=iter([(timestamp, [sensor_data])]),
        steering_azimuths_rad=np.array([0.1, 0.2, 0.3]),
    )

    result = list(detector.detections_gen(beamformer_output_type="snr_percentile"))

    assert len(result) == 1
    assert np.isfinite(detector.snr_history).all()


def test_detections_gen_empty_sensor_data_set_yields_empty_detections(monkeypatch) -> None:
    """A timestep with an empty sensor_data iterable should yield an empty detection set."""
    passive = _load_passive_detector_module(monkeypatch)
    timestamp = datetime(2026, 1, 1, 12, 0, 0)

    detector = passive.PassiveSonarDetector(
        detection_chain=[],
        sensor_data_gen=iter([(timestamp, [])]),
        steering_azimuths_rad=np.array([0.1]),
    )

    result = list(detector.detections_gen())

    assert len(result) == 1
    _, detections = result[0]
    assert len(detections) == 0


def test_snr_history_accumulates_one_row_per_timestep(monkeypatch) -> None:
    """snr_history should grow by one row for each processed timestep."""
    passive = _load_passive_detector_module(monkeypatch)
    t1 = datetime(2026, 1, 1, 12, 0, 0)
    t2 = datetime(2026, 1, 1, 12, 0, 1)

    def make_sd(t):
        return SimpleNamespace(beamformed_data=np.array([[1.0 + 0.0j, 1.0 + 0.0j]]), timestamp=t)

    detector = passive.PassiveSonarDetector(
        detection_chain=[],
        sensor_data_gen=iter([(t1, [make_sd(t1)]), (t2, [make_sd(t2)])]),
        steering_azimuths_rad=np.array([0.1]),
    )

    list(detector.detections_gen(beamformer_output_type="power"))

    assert detector.snr_history.shape == (2, 1)


def test_detections_gen_custom_snr_percentile_val(monkeypatch) -> None:
    """A custom snr_percentile_val should be used instead of the default 10."""
    passive = _load_passive_detector_module(monkeypatch)
    timestamp = datetime(2026, 1, 1, 12, 0, 0)
    # Four beams with powers [1, 2, 3, 4]. 50th-percentile noise ≈ 2.5.
    # With 10th-percentile noise ≈ 1.3 the SNR values differ.
    beamformed_data = np.array(
        [[1.0 + 0.0j], [np.sqrt(2) + 0.0j], [np.sqrt(3) + 0.0j], [2.0 + 0.0j]]
    )
    sensor_data = SimpleNamespace(beamformed_data=beamformed_data, timestamp=timestamp)

    captured_50 = []
    captured_10 = []

    class Capture50:
        def detect(self, data_map):
            captured_50.append(data_map.copy())
            return np.empty((0, 2))

    class Capture10:
        def detect(self, data_map):
            captured_10.append(data_map.copy())
            return np.empty((0, 2))

    for _captured, pct, chain in [
        (captured_50, 50, [Capture50()]),
        (captured_10, 10, [Capture10()]),
    ]:
        detector = passive.PassiveSonarDetector(
            detection_chain=chain,
            sensor_data_gen=iter([(timestamp, [sensor_data])]),
            steering_azimuths_rad=np.array([0.1, 0.2, 0.3, 0.4]),
        )
        list(
            detector.detections_gen(
                beamformer_output_type="snr_percentile", snr_percentile_val=pct
            )
        )

    # Higher percentile → higher noise floor → lower SNR values
    assert captured_50[0].max() < captured_10[0].max()


def test_run_detection_chain_with_empty_input_array(monkeypatch) -> None:
    """_run_detection_chain should return empty detections for a zero-length input."""
    passive = _load_passive_detector_module(monkeypatch)

    class AlwaysDetect:
        def detect(self, data_map):
            return np.empty((0, 2), dtype=np.float64)

    detector = passive.PassiveSonarDetector(
        detection_chain=[AlwaysDetect()],
        sensor_data_gen=iter(()),
        steering_azimuths_rad=np.array([], dtype=float),
    )

    result = detector._run_detection_chain(np.array([], dtype=np.float64))
    assert result.shape == (0, 2)


def test_detections_gen_progress_bar_wraps_iterator(monkeypatch) -> None:
    """When progress_bar=True the sensor_data iterator should be wrapped with tqdm."""
    passive = _load_passive_detector_module(monkeypatch)
    timestamp = datetime(2026, 1, 1, 12, 0, 0)
    sensor_data = SimpleNamespace(beamformed_data=None, timestamp=timestamp)

    wrapped = []

    def fake_tqdm(iterable, **kwargs):
        wrapped.append(kwargs)
        return iterable

    monkeypatch.setattr(passive, "tqdm", fake_tqdm)

    detector = passive.PassiveSonarDetector(
        detection_chain=[],
        sensor_data_gen=iter([(timestamp, [sensor_data])]),
        steering_azimuths_rad=np.array([0.1]),
    )

    list(detector.detections_gen(progress_bar=True, total_timesteps=5))

    assert len(wrapped) == 1
    assert wrapped[0]["total"] == 5
