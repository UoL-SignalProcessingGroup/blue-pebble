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
    """Load ``detector/passive.py`` with lightweight dependency scaffolding.

    ``algorithms.py`` is loaded for real rather than stubbed, since ``passive.py``
    imports ``_CFARDetectorBase`` from it as a type annotation.
    """
    _install_fake_passive_dependencies(monkeypatch)
    install_repo_package(monkeypatch, "bluepebble", "bluepebble")
    install_repo_package(monkeypatch, "bluepebble.detector", "bluepebble/detector")
    load_package_module_from_repo(
        "bluepebble/detector/algorithms.py",
        "bluepebble.detector.algorithms",
    )
    return load_package_module_from_repo(
        "bluepebble/detector/passive.py",
        "bluepebble.detector.passive",
    )


class _FakeDetector:
    """Minimal stand-in for a ``_CFARDetectorBase``: independent detect()/snr_map()."""

    def __init__(self, detect_fn, snr_fn=None):
        self._detect_fn = detect_fn
        self._snr_fn = snr_fn or (lambda data: np.zeros(np.asarray(data).shape[0]))

    def detect(self, beamformed_data):
        return self._detect_fn(np.asarray(beamformed_data))

    def snr_map(self, beamformed_data):
        return self._snr_fn(np.asarray(beamformed_data))


# --------------------------------------------------------------------------
# beam_snr (standalone utility)
# --------------------------------------------------------------------------


def test_beam_power_returns_the_frame_averaged_linear_power(monkeypatch) -> None:
    """beam_power is the unnormalised power map, not an SNR."""
    passive = _load_passive_detector_module(monkeypatch)
    data = np.array([[1.0 + 0.0j], [3.0 + 0.0j]])

    np.testing.assert_allclose(passive.beam_power(data), [1.0, 9.0])


def test_beam_power_in_decibels(monkeypatch) -> None:
    """decibels=True is 10*log10 of the same quantity."""
    passive = _load_passive_detector_module(monkeypatch)
    data = np.array([[2.0 + 0.0j, 2.0 + 0.0j]])

    np.testing.assert_allclose(
        passive.beam_power(data, decibels=True), [10 * np.log10(4.0)], rtol=1e-6
    )


def test_beam_power_does_not_square_real_power_input(monkeypatch) -> None:
    """The reason this helper exists: BeamformedData may already be power.

    Squaring a real-power beamformer's output a second time double-applies the power law,
    which is silent in the output and breaks any Pfa calibration downstream. Amplitude and the
    power it corresponds to must give the same answer.
    """
    passive = _load_passive_detector_module(monkeypatch)
    amplitude = np.array([[1.0 + 0.0j], [3.0 + 0.0j]])
    real_power = np.abs(amplitude) ** 2

    np.testing.assert_allclose(
        passive.beam_power(amplitude), passive.beam_power(real_power)
    )


def test_snr_is_zero_db_when_every_beam_carries_equal_power(monkeypatch) -> None:
    """With a flat scan the noise floor equals every beam, whatever the percentile."""
    passive = _load_passive_detector_module(monkeypatch)
    data = np.array([[2.0 + 0.0j], [2.0 + 0.0j], [2.0 + 0.0j]])

    np.testing.assert_allclose(
        passive.beam_snr(data, percentile=50), np.zeros(3), atol=1e-6
    )


def test_percentile_50_is_the_median_reference(monkeypatch) -> None:
    """The former 'median_power' mode is exactly percentile=50, so it needs no own option."""
    passive = _load_passive_detector_module(monkeypatch)
    rng = np.random.default_rng(0)
    data = rng.exponential(1.0, size=(9, 3))

    from_percentile = passive.beam_snr(data, percentile=50)
    power = passive.beam_power(data)
    eps = np.finfo(float).eps
    by_hand = 10 * np.log10((power + eps) / (np.median(power) + eps))

    np.testing.assert_allclose(from_percentile, by_hand)


def test_a_higher_percentile_lowers_the_reported_snr(monkeypatch) -> None:
    """A higher percentile is a higher assumed noise floor."""
    passive = _load_passive_detector_module(monkeypatch)
    data = np.array([[1.0 + 0.0j], [np.sqrt(2) + 0.0j], [np.sqrt(3) + 0.0j], [2.0 + 0.0j]])

    low = passive.beam_snr(data, percentile=10)
    high = passive.beam_snr(data, percentile=50)

    assert high.max() < low.max()


# --------------------------------------------------------------------------
# PassiveSonarDetector
# --------------------------------------------------------------------------


def test_detections_gen_emits_bearing_detections_with_snr_metadata(monkeypatch) -> None:
    """A detection at beam index 1 should map to steering_azimuths_rad[1] and carry its SNR."""
    passive = _load_passive_detector_module(monkeypatch)
    timestamp = datetime(2026, 1, 1, 12, 0, 0)
    sensor_data = SimpleNamespace(
        beamformed_data=np.array([[1.0 + 0.0j, 1.0 + 0.0j], [2.0 + 0.0j, 2.0 + 0.0j]]),
        timestamp=timestamp,
    )
    detector = passive.PassiveSonarDetector(
        detector=_FakeDetector(detect_fn=lambda data: np.array([[1.0, 4.0]])),
        sensor_data_gen=iter([(timestamp, [sensor_data])]),
        steering_azimuths_rad=np.array([0.1, 0.5]),
    )

    generated = list(detector.detections_gen())

    assert len(generated) == 1
    yielded_timestamp, detections = generated[0]
    assert yielded_timestamp == timestamp
    assert len(detections) == 1

    detection = next(iter(detections))
    assert detection.timestamp == timestamp
    assert float(detection.state_vector[0, 0]) == pytest.approx(0.5)
    assert detection.metadata["snr_db"] == pytest.approx(4.0)


def test_detections_gen_skips_none_beamformed_data(monkeypatch) -> None:
    """Sensor data with beamformed_data=None should be silently skipped."""
    passive = _load_passive_detector_module(monkeypatch)
    timestamp = datetime(2026, 1, 1, 12, 0, 0)
    sensor_data = SimpleNamespace(beamformed_data=None, timestamp=timestamp)

    detector = passive.PassiveSonarDetector(
        detector=_FakeDetector(detect_fn=lambda data: np.empty((0, 2))),
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
        detector=_FakeDetector(detect_fn=lambda data: np.empty((0, 2))),
        sensor_data_gen=iter([(timestamp, [sensor_data])]),
        steering_azimuths_rad=np.array([0.1]),
    )

    result = list(detector.detections_gen())

    assert len(result) == 1
    _, detections = result[0]
    assert len(detections) == 0


def test_detections_gen_no_detections_when_detector_finds_nothing(monkeypatch) -> None:
    """A detector reporting no detections should yield an empty set, not raise."""
    passive = _load_passive_detector_module(monkeypatch)
    timestamp = datetime(2026, 1, 1, 12, 0, 0)
    sensor_data = SimpleNamespace(
        beamformed_data=np.array([[1.0 + 0.0j, 2.0 + 0.0j]]),
        timestamp=timestamp,
    )

    detector = passive.PassiveSonarDetector(
        detector=_FakeDetector(detect_fn=lambda data: np.empty((0, 2), dtype=np.float64)),
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
        detector=_FakeDetector(detect_fn=lambda data: np.empty((0, 2))),
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

    sd1 = SimpleNamespace(beamformed_data=np.array([[3.0 + 0.0j]]), timestamp=timestamp)
    sd2 = SimpleNamespace(beamformed_data=np.array([[5.0 + 0.0j]]), timestamp=timestamp)

    detector = passive.PassiveSonarDetector(
        detector=_FakeDetector(detect_fn=lambda data: np.array([[0.0, data[0, 0].real]])),
        sensor_data_gen=iter([(timestamp, [sd1, sd2])]),
        steering_azimuths_rad=np.array([0.7]),
    )

    result = list(detector.detections_gen())

    assert len(result) == 1
    _, detections = result[0]
    # Both sensor_data objects trigger a detection at the same bearing, with different SNRs.
    assert len(detections) == 2
    assert {d.metadata["snr_db"] for d in detections} == {3.0, 5.0}


def test_detections_gen_empty_sensor_data_set_yields_empty_detections(monkeypatch) -> None:
    """A timestep with an empty sensor_data iterable should yield an empty detection set."""
    passive = _load_passive_detector_module(monkeypatch)
    timestamp = datetime(2026, 1, 1, 12, 0, 0)

    detector = passive.PassiveSonarDetector(
        detector=_FakeDetector(detect_fn=lambda data: np.empty((0, 2))),
        sensor_data_gen=iter([(timestamp, [])]),
        steering_azimuths_rad=np.array([0.1]),
    )

    result = list(detector.detections_gen())

    assert len(result) == 1
    _, detections = result[0]
    assert len(detections) == 0


def test_snr_history_accumulates_one_row_per_timestep_from_snr_map(monkeypatch) -> None:
    """snr_history should record snr_map()'s output, one row per timestep.

    snr_reference="local" is set explicitly: this test is about the snr_map() plumbing, and
    the default reports a scan-wide percentile that never consults it.
    """
    passive = _load_passive_detector_module(monkeypatch)
    t1 = datetime(2026, 1, 1, 12, 0, 0)
    t2 = datetime(2026, 1, 1, 12, 0, 1)

    def make_sd(t):
        return SimpleNamespace(beamformed_data=np.array([[1.0 + 0.0j, 1.0 + 0.0j]]), timestamp=t)

    detector = passive.PassiveSonarDetector(
        detector=_FakeDetector(
            detect_fn=lambda data: np.empty((0, 2)),
            snr_fn=lambda data: np.array([7.0]),
        ),
        sensor_data_gen=iter([(t1, [make_sd(t1)]), (t2, [make_sd(t2)])]),
        steering_azimuths_rad=np.array([0.1]),
        snr_reference="local",
    )

    list(detector.detections_gen())

    assert detector.snr_history.shape == (2, 1)
    np.testing.assert_allclose(detector.snr_history, [[7.0], [7.0]])


def test_detect_and_snr_map_are_independent_calls(monkeypatch) -> None:
    """detect() and snr_map() are called separately, so their outputs need not agree.

    snr_reference="local" is set explicitly so snr_map() is the reported source; the default
    global reference would bypass it and defeat the point of the test.
    """
    passive = _load_passive_detector_module(monkeypatch)
    timestamp = datetime(2026, 1, 1, 12, 0, 0)
    sensor_data = SimpleNamespace(beamformed_data=np.array([[1.0 + 0.0j]]), timestamp=timestamp)

    detector = passive.PassiveSonarDetector(
        detector=_FakeDetector(
            detect_fn=lambda data: np.array([[0.0, 4.0]]),
            snr_fn=lambda data: np.array([99.0]),
        ),
        sensor_data_gen=iter([(timestamp, [sensor_data])]),
        steering_azimuths_rad=np.array([0.1]),
        snr_reference="local",
    )

    generated = list(detector.detections_gen())

    detection = next(iter(generated[0][1]))
    assert detection.metadata["snr_db"] == pytest.approx(4.0)
    np.testing.assert_allclose(detector.snr_history, [[99.0]])


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
        detector=_FakeDetector(detect_fn=lambda data: np.empty((0, 2))),
        sensor_data_gen=iter([(timestamp, [sensor_data])]),
        steering_azimuths_rad=np.array([0.1]),
    )

    list(detector.detections_gen(progress_bar=True, total_timesteps=5))

    assert len(wrapped) == 1
    assert wrapped[0]["total"] == 5


def test_snr_gives_identical_results_for_real_power_and_complex_amplitude(
    monkeypatch,
) -> None:
    """Real power input must not be squared again relative to the equivalent amplitude.

    Regression test for the same _directional_power bug covered in test_detector_algorithms.py:
    BeamformedData is deliberately either complex amplitude or already-real power (see
    bluepebble.types.sensordata.BeamformedData), and this function must treat both consistently.
    """
    passive = _load_passive_detector_module(monkeypatch)
    amplitude = np.array([[1.0 + 0.0j], [3.0 + 0.0j]])
    real_power = np.abs(amplitude) ** 2

    np.testing.assert_allclose(
        passive.beam_snr(amplitude),
        passive.beam_snr(real_power),
    )


def test_band_detector_reports_the_global_noise_floor_by_default(monkeypatch) -> None:
    """The default is the scan-wide percentile, matching pre-refactor behaviour.

    The detector's own snr_map() must not be consulted in this mode, which the impossible
    sentinel below checks: if it leaked into the result the assertion would see -99.
    """
    passive = _load_passive_detector_module(monkeypatch)
    impossible = np.full(6, -99.0)
    band = passive.BandDetector(
        detector=_FakeDetector(lambda d: np.empty((0, 2)), snr_fn=lambda d: impossible)
    )

    snr, _ = band.detect(np.ones((6, 2)))

    assert band.snr_reference == "global"
    assert not np.array_equal(snr, impossible)
    np.testing.assert_allclose(snr, passive.beam_snr(np.ones((6, 2))))


def test_band_detector_can_report_the_local_noise_floor(monkeypatch) -> None:
    """'local' asks for the detector's own training-cell estimate instead.

    That is the quantity its threshold was actually compared against, so it is the honest
    answer to "what did this detector see", at the cost of not being comparable between
    bearings -- a strong target flattens its own apparent SNR by sitting in its training cells.
    """
    passive = _load_passive_detector_module(monkeypatch)
    marker = np.full(6, -99.0)
    band = passive.BandDetector(
        detector=_FakeDetector(lambda d: np.empty((0, 2)), snr_fn=lambda d: marker),
        snr_reference="local",
    )

    snr, _ = band.detect(np.ones((6, 2)))

    np.testing.assert_array_equal(snr, marker)


def test_band_detector_can_report_against_a_global_noise_floor(monkeypatch) -> None:
    """'global' restores the pre-refactor scan-wide percentile reference.

    The two references answer different questions -- local follows a noise field that varies
    with bearing, global is comparable between bearings -- so both are legitimate and the
    caller picks. The detector's own snr_map() is not consulted in this mode, which the
    sentinel below checks by making it return something impossible.
    """
    passive = _load_passive_detector_module(monkeypatch)
    data = np.array([[1.0 + 0.0j], [1.0 + 0.0j], [10.0 + 0.0j], [1.0 + 0.0j]])
    band = passive.BandDetector(
        detector=_FakeDetector(lambda d: np.empty((0, 2)), snr_fn=lambda d: np.full(4, -99.0)),
        snr_reference="global",
    )

    snr, _ = band.detect(data)

    expected = passive.beam_snr(data, percentile=10)
    np.testing.assert_allclose(snr, expected)
    assert not np.any(snr == -99.0)


def test_snr_percentile_selects_the_global_noise_floor(monkeypatch) -> None:
    """A higher percentile is a higher noise floor, so reported SNR falls."""
    passive = _load_passive_detector_module(monkeypatch)
    data = np.array([[1.0 + 0.0j], [2.0 + 0.0j], [3.0 + 0.0j], [20.0 + 0.0j]])
    make = lambda pct: passive.BandDetector(  # noqa: E731
        detector=_FakeDetector(lambda d: np.empty((0, 2))),
        snr_reference="global",
        snr_percentile=pct,
    )

    low, _ = make(10).detect(data)
    high, _ = make(90).detect(data)

    assert high.max() < low.max()


def test_snr_reference_does_not_change_what_is_detected(monkeypatch) -> None:
    """The option only sets what is reported; thresholding always uses the local estimate."""
    passive = _load_passive_detector_module(monkeypatch)
    data = np.array([[1.0 + 0.0j], [9.0 + 0.0j], [1.0 + 0.0j]])
    detections = np.array([[1.0, 12.0]])
    make = lambda ref: passive.BandDetector(  # noqa: E731
        detector=_FakeDetector(lambda d: detections), snr_reference=ref
    )

    _, local_hits = make("local").detect(data)
    _, global_hits = make("global").detect(data)

    np.testing.assert_array_equal(local_hits, global_hits)


def test_unknown_snr_reference_is_rejected(monkeypatch) -> None:
    """A typo should name the valid options rather than silently pick one."""
    passive = _load_passive_detector_module(monkeypatch)
    band = passive.BandDetector(
        detector=_FakeDetector(lambda d: np.empty((0, 2))), snr_reference="globl"
    )

    with pytest.raises(ValueError, match="snr_reference must be one of"):
        band.detect(np.ones((4, 2)))
