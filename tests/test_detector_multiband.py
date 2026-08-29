"""Tests for the multiband passive sonar detector and its per-band readers."""

from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest

from .test_detector_passive import _load_passive_detector_module

TIMESTAMP = datetime(2026, 1, 1, 12, 0, 0)


def _sensor_data(band_maps, timestamp=TIMESTAMP, band_labels=("low", "high")):
    """Build a fake sensor-data payload with a leading band axis."""
    return SimpleNamespace(
        beamformed_data=np.asarray(band_maps, dtype=np.float64),
        band_labels=list(band_labels),
        timestamp=timestamp,
    )


def _two_band_source(num_steps=1):
    """Yield steps whose 'low' band has a strong peak and 'high' band a weaker one."""
    for step in range(num_steps):
        timestamp = TIMESTAMP + timedelta(seconds=5 * step)
        low = np.array([[1.0], [1.0], [8.0], [1.0]])
        high = np.array([[1.0], [3.0], [1.0], [1.0]])
        yield timestamp, [_sensor_data([low, high], timestamp=timestamp)]


def _detector(passive, band_detectors, source=None, default_detector=None, num_steps=1):
    """Construct a multiband detector over the two-band fake source."""
    return passive.MultibandPassiveSonarDetector(
        band_detectors=band_detectors,
        default_detector=default_detector,
        sensor_data_gen=source if source is not None else _two_band_source(num_steps),
        steering_azimuths_rad=np.array([-1.5, -0.5, 0.5, 1.5]),
    )


class _ThresholdDetector:
    """Minimal CFAR-like detector: raw mean power, thresholded directly (no dB scaling)."""

    def __init__(self, threshold):
        self.threshold = threshold

    def snr_map(self, beamformed_data):
        """Return raw mean power per beam."""
        data = np.asarray(beamformed_data)
        return np.mean(np.abs(data) ** 2, axis=1)

    def detect(self, beamformed_data):
        """Return ``[index, power]`` rows for cells whose power exceeds the threshold."""
        power = self.snr_map(beamformed_data)
        indices = np.nonzero(power > self.threshold)[0]
        if indices.size == 0:
            return np.empty((0, 2), dtype=np.float64)
        return np.column_stack((indices, power[indices])).astype(np.float64)


def _band_detector(passive, threshold):
    """Build a BandDetector wrapping a _ThresholdDetector at the given power threshold."""
    return passive.BandDetector(detector=_ThresholdDetector(threshold))


def test_per_band_detectors_are_applied_independently(monkeypatch) -> None:
    """Each band should be detected with its own detector, not a shared one."""
    passive = _load_passive_detector_module(monkeypatch)

    # 'low' peaks at power 64 (8.0**2), 'high' at power 9 (3.0**2).
    detector = _detector(
        passive,
        {"low": _band_detector(passive, 16.0), "high": _band_detector(passive, 16.0)},
    )
    detections = [d for _, batch in detector.detections_gen() for d in batch]
    assert [d.metadata["band"] for d in detections] == ["low"]

    # Lowering only the 'high' band's threshold must bring its peak in, and nothing else.
    detector = _detector(
        passive,
        {"low": _band_detector(passive, 16.0), "high": _band_detector(passive, 4.0)},
    )
    detections = [d for _, batch in detector.detections_gen() for d in batch]
    assert sorted(d.metadata["band"] for d in detections) == ["high", "low"]


def test_detections_carry_band_and_snr_metadata(monkeypatch) -> None:
    """Band provenance and SNR must survive onto every detection."""
    passive = _load_passive_detector_module(monkeypatch)
    detector = _detector(
        passive,
        {"low": _band_detector(passive, 16.0)},
        default_detector=_band_detector(passive, 16.0),
    )

    detections = [d for _, batch in detector.detections_gen() for d in batch]

    assert len(detections) == 1
    detection = detections[0]
    assert detection.metadata["band"] == "low"
    assert isinstance(detection.metadata["snr_db"], float)
    assert detection.metadata["snr_db"] == pytest.approx(64.0)
    # The peak sits in beam index 2 of the 'low' band.
    assert detection.state_vector[0][0] == pytest.approx(0.5)


def test_union_equals_the_sum_of_per_band_readers(monkeypatch) -> None:
    """The squashed view must contain exactly what the separate views do."""
    passive = _load_passive_detector_module(monkeypatch)
    band_detectors = {
        "low": _band_detector(passive, 4.0),
        "high": _band_detector(passive, 4.0),
    }

    union_detector = _detector(passive, band_detectors, num_steps=3)
    union_count = sum(len(batch) for _, batch in union_detector.detections_gen())

    split_detector = _detector(passive, band_detectors, num_steps=3)
    low_reader = split_detector.band_reader("low")
    high_reader = split_detector.band_reader("high")
    split_count = sum(len(batch) for _, batch in low_reader.detections_gen()) + sum(
        len(batch) for _, batch in high_reader.detections_gen()
    )

    assert union_count == 6
    assert split_count == union_count


def test_band_readers_share_one_pass_over_the_sensor_data(monkeypatch) -> None:
    """Two readers must not consume the source twice, and each must see every step."""
    passive = _load_passive_detector_module(monkeypatch)
    steps_pulled = []

    def counting_source():
        for step in range(3):
            timestamp = TIMESTAMP + timedelta(seconds=5 * step)
            steps_pulled.append(step)
            low = np.array([[1.0], [1.0], [8.0], [1.0]])
            high = np.array([[1.0], [3.0], [1.0], [1.0]])
            yield timestamp, [_sensor_data([low, high], timestamp=timestamp)]

    detector = _detector(
        passive,
        {"low": _band_detector(passive, 4.0), "high": _band_detector(passive, 4.0)},
        source=counting_source(),
    )
    low_reader = detector.band_reader("low")
    high_reader = detector.band_reader("high")

    # Drain one reader completely before the other starts: the second must replay from
    # its own buffer rather than re-pulling the exhausted source.
    low_batches = list(low_reader.detections_gen())
    high_batches = list(high_reader.detections_gen())

    assert len(low_batches) == 3
    assert len(high_batches) == 3
    assert steps_pulled == [0, 1, 2], "source should be consumed exactly once"


def test_interleaved_band_readers_stay_aligned(monkeypatch) -> None:
    """Readers pulled in lockstep should see the same timestamps in the same order."""
    passive = _load_passive_detector_module(monkeypatch)
    detector = _detector(
        passive,
        {"low": _band_detector(passive, 4.0), "high": _band_detector(passive, 4.0)},
        num_steps=3,
    )
    low_gen = detector.band_reader("low").detections_gen()
    high_gen = detector.band_reader("high").detections_gen()

    for _ in range(3):
        low_time, low_batch = next(low_gen)
        high_time, high_batch = next(high_gen)
        assert low_time == high_time
        assert {d.metadata["band"] for d in low_batch} == {"low"}
        assert {d.metadata["band"] for d in high_batch} == {"high"}


def test_subscribing_after_iteration_starts_is_rejected(monkeypatch) -> None:
    """A late reader would silently miss consumed steps, so it must fail loudly."""
    passive = _load_passive_detector_module(monkeypatch)
    detector = _detector(
        passive,
        {"low": _band_detector(passive, 4.0), "high": _band_detector(passive, 4.0)},
        num_steps=3,
    )
    low_gen = detector.band_reader("low").detections_gen()
    next(low_gen)

    with pytest.raises(RuntimeError, match="Cannot subscribe after iteration has started"):
        detector.band_reader("high")


def test_unknown_band_without_default_detector_raises(monkeypatch) -> None:
    """An unrecognised band label must not be silently dropped."""
    passive = _load_passive_detector_module(monkeypatch)
    detector = _detector(passive, {"low": _band_detector(passive, 4.0)})

    with pytest.raises(KeyError, match="No detector for band 'high'"):
        list(detector.detections_gen())


def test_default_detector_covers_unlisted_bands(monkeypatch) -> None:
    """A default detector should absorb bands with no explicit entry."""
    passive = _load_passive_detector_module(monkeypatch)
    detector = _detector(
        passive,
        {"low": _band_detector(passive, 4.0)},
        default_detector=_band_detector(passive, 4.0),
    )

    detections = [d for _, batch in detector.detections_gen() for d in batch]
    assert sorted(d.metadata["band"] for d in detections) == ["high", "low"]


def test_snr_history_is_keyed_by_band(monkeypatch) -> None:
    """SNR history should be one map per band, each one row per timestep."""
    passive = _load_passive_detector_module(monkeypatch)
    detector = _detector(
        passive,
        {"low": _band_detector(passive, 4.0), "high": _band_detector(passive, 4.0)},
        num_steps=4,
    )

    assert detector.snr_history == {}

    list(detector.detections_gen())

    history = detector.snr_history
    assert sorted(history) == ["high", "low"]
    assert history["low"].shape == (4, 4)
    assert history["high"].shape == (4, 4)
    # The 'low' band's peak sits in beam 2 and is stronger than the 'high' band's.
    assert np.argmax(history["low"][0]) == 2
    assert history["low"][0].max() > history["high"][0].max()


def test_each_band_uses_its_own_detectors_snr_scale(monkeypatch) -> None:
    """Per-band SNR history must reflect that band's own detector, not a shared scale."""
    passive = _load_passive_detector_module(monkeypatch)

    class DbDetector:
        """A detector reporting SNR in dB relative to the mean, instead of raw power."""

        def snr_map(self, beamformed_data):
            power = np.mean(np.abs(np.asarray(beamformed_data)) ** 2, axis=1)
            eps = np.finfo(float).eps
            return 10 * np.log10((power + eps) / (np.mean(power) + eps))

        def detect(self, beamformed_data):
            return np.empty((0, 2), dtype=np.float64)

    detector = _detector(
        passive,
        {
            "low": passive.BandDetector(detector=_ThresholdDetector(threshold=1e9)),
            "high": passive.BandDetector(detector=DbDetector()),
        },
    )
    list(detector.detections_gen())
    history = detector.snr_history

    # 'low' uses raw linear power, so the 8.0 peak survives as 64.0 untouched.
    np.testing.assert_allclose(history["low"][0], [1.0, 1.0, 64.0, 1.0])
    # 'high' uses dB-relative-to-mean, so its flat non-peak cells sit below 0 dB.
    assert history["high"][0][0] < 0.0
    assert history["high"][0][1] > history["high"][0][0]


def test_rejects_single_band_beamformed_data(monkeypatch) -> None:
    """2D data means the beamformer had no bands; that is a configuration error."""
    passive = _load_passive_detector_module(monkeypatch)

    def single_band_source():
        yield (
            TIMESTAMP,
            [
                SimpleNamespace(
                    beamformed_data=np.array([[1.0], [8.0]]),
                    band_labels=None,
                    timestamp=TIMESTAMP,
                )
            ],
        )

    detector = _detector(
        passive,
        {"low": _band_detector(passive, 4.0)},
        source=single_band_source(),
    )

    with pytest.raises(ValueError, match="requires beamformed data with a band axis"):
        list(detector.detections_gen())


def test_skips_empty_and_missing_beamformed_data(monkeypatch) -> None:
    """Steps without usable beamformed data should yield no detections, not raise."""
    passive = _load_passive_detector_module(monkeypatch)

    def sparse_source():
        yield TIMESTAMP, [SimpleNamespace(beamformed_data=None, timestamp=TIMESTAMP)]
        yield (
            TIMESTAMP,
            [
                SimpleNamespace(
                    beamformed_data=np.empty((0, 0)),
                    band_labels=None,
                    timestamp=TIMESTAMP,
                )
            ],
        )

    detector = _detector(
        passive,
        {"low": _band_detector(passive, 4.0)},
        source=sparse_source(),
    )

    batches = list(detector.detections_gen())
    assert [batch for _, batch in batches] == [set(), set()]
    assert detector.snr_history == {}
