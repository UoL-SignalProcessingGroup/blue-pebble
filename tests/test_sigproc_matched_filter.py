"""Tests for MatchedFilter, parabolic_peak_offset and bistatic_range."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import get_window

from .support import isolate_real_bluepebble_imports, restore_real_bluepebble_imports


@pytest.fixture
def bp():
    """Import the real helpers without leaking caching into other tests."""
    snapshot = isolate_real_bluepebble_imports()
    from bluepebble.detector import bistatic_range, matched_filter
    from bluepebble.sigproc import MatchedFilter, parabolic_peak_offset

    yield MatchedFilter, parabolic_peak_offset, bistatic_range, matched_filter
    restore_real_bluepebble_imports(snapshot)


def _chirp(n: int = 200) -> np.ndarray:
    t = np.arange(n) / n
    return np.exp(1j * np.pi * 40.0 * t**2)


def test_matched_filter_peaks_at_echo_delay(bp) -> None:
    """The correlation peak sits at the sample where the replica starts in the data."""
    MatchedFilter, *_ = bp
    replica = _chirp()
    data = np.zeros(2000, dtype=complex)
    data[700:900] = 0.5 * replica

    out = MatchedFilter().filter(data, replica)

    assert out.shape == data.shape
    assert int(np.argmax(out)) == 700
    assert out[700] == pytest.approx(0.5 * np.sum(np.abs(replica) ** 2))


def test_matched_filter_rows_match_single_channel_and_window_tapers(bp) -> None:
    """Each row of a 2-D input is filtered independently, using the windowed replica."""
    MatchedFilter, *_ = bp
    rng = np.random.default_rng(1)
    replica = _chirp()
    data = rng.normal(size=(3, 1000)) + 1j * rng.normal(size=(3, 1000))
    mf = MatchedFilter(window="hann")

    rows = mf.filter(data, replica)

    for i in range(3):
        np.testing.assert_allclose(rows[i], mf.filter(data[i], replica))
    tapered = replica * get_window("hann", len(replica))
    np.testing.assert_allclose(rows, MatchedFilter().filter(data, tapered))


def test_matched_filter_function_is_normalised_matched_filter(bp) -> None:
    """detector.matched_filter is MatchedFilter scaled to a peak of 1."""
    MatchedFilter, _, _, matched_filter = bp
    rng = np.random.default_rng(2)
    replica = _chirp()
    data = rng.normal(size=1500) + 1j * rng.normal(size=1500)

    raw = MatchedFilter().filter(data, replica)

    np.testing.assert_allclose(matched_filter(data, replica), raw / raw.max())


def test_parabolic_peak_offset_recovers_vertex(bp) -> None:
    """A sampled parabola gives back its exact vertex; edges and flat data give 0."""
    _, parabolic_peak_offset, *_ = bp
    x = np.arange(10, dtype=float)
    y = -((x - 4.3) ** 2)

    assert 4 + parabolic_peak_offset(y, 4) == pytest.approx(4.3)
    assert parabolic_peak_offset(y, 0) == 0.0
    assert parabolic_peak_offset(y, 9) == 0.0
    assert parabolic_peak_offset(np.ones(5), 2) == 0.0


def test_bistatic_range_matches_geometry(bp) -> None:
    """The returned range puts the target on the bistatic ellipse; monostatic halves the path."""
    *_, bistatic_range, _ = bp
    rx = np.array([10.0, -5.0])
    tx = np.array([60.0, 0.0])
    target = np.array([900.0, 1400.0])
    path = np.linalg.norm(target - tx) + np.linalg.norm(target - rx)
    bearing = np.arctan2(*(target - rx)[::-1])

    assert bistatic_range(path, bearing, rx, tx) == pytest.approx(np.linalg.norm(target - rx))
    assert bistatic_range(3000.0, 0.7, rx, rx) == pytest.approx(1500.0)
