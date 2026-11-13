"""Tests for MatchedFilter and parabolic_peak_offset."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import correlate, get_window

from .support import isolate_real_bluepebble_imports, restore_real_bluepebble_imports


@pytest.fixture
def sigproc():
    """Import the real sigproc package without leaking caching into other tests."""
    snapshot = isolate_real_bluepebble_imports()
    import bluepebble.sigproc

    yield bluepebble.sigproc
    restore_real_bluepebble_imports(snapshot)


def _chirp(n: int = 200) -> np.ndarray:
    t = np.arange(n) / n
    return np.exp(1j * np.pi * 40.0 * t**2)


def test_matched_filter_peaks_at_echo_delay(sigproc) -> None:
    """The correlation peak sits where the replica starts in the data, at the echo's energy."""
    replica = _chirp()
    data = np.zeros(2000, dtype=complex)
    data[700:900] = 0.5 * replica

    out = sigproc.MatchedFilter().filter(data, replica)

    assert out.shape == data.shape
    assert int(np.argmax(out)) == 700
    assert out[700] == pytest.approx(0.5 * np.sum(np.abs(replica) ** 2))


def test_matched_filter_matches_scipy_correlate(sigproc) -> None:
    """A single channel equals scipy's direct correlation, from zero lag onwards."""
    rng = np.random.default_rng(3)
    replica = _chirp(50)
    data = rng.normal(size=400) + 1j * rng.normal(size=400)

    full = np.abs(correlate(data, replica, mode="full", method="direct"))
    expected = full[len(replica) - 1 : len(replica) - 1 + len(data)]

    np.testing.assert_allclose(sigproc.MatchedFilter().filter(data, replica), expected, atol=1e-10)


def test_matched_filter_rows_match_single_channel_and_window_tapers(sigproc) -> None:
    """Each row of a 2-D input is filtered independently, using the windowed replica."""
    rng = np.random.default_rng(1)
    replica = _chirp()
    data = rng.normal(size=(3, 1000)) + 1j * rng.normal(size=(3, 1000))
    matched_filter = sigproc.MatchedFilter(window="hann")

    rows = matched_filter.filter(data, replica)

    for i in range(3):
        np.testing.assert_allclose(rows[i], matched_filter.filter(data[i], replica), atol=1e-12)
    tapered = replica * get_window("hann", len(replica))
    np.testing.assert_allclose(rows, sigproc.MatchedFilter().filter(data, tapered), atol=1e-12)


@pytest.mark.parametrize(
    ("data_shape", "replica_shape", "match"),
    [
        ((100,), (5, 2), "replica must be one-dimensional"),
        ((2, 3, 100), (10,), "data must be"),
        ((20,), (30,), "longer than the data"),
    ],
)
def test_matched_filter_rejects_bad_shapes(sigproc, data_shape, replica_shape, match) -> None:
    """Shapes the correlation is not defined for raise a ValueError naming the problem."""
    with pytest.raises(ValueError, match=match):
        sigproc.MatchedFilter().filter(np.zeros(data_shape), np.ones(replica_shape))


def test_parabolic_peak_offset_recovers_vertex(sigproc) -> None:
    """A sampled parabola gives back its exact vertex; edges and flat data give 0."""
    x = np.arange(10, dtype=float)
    y = -((x - 4.3) ** 2)

    assert 4 + sigproc.parabolic_peak_offset(y, 4) == pytest.approx(4.3)
    assert sigproc.parabolic_peak_offset(y, 0) == 0.0
    assert sigproc.parabolic_peak_offset(y, 9) == 0.0
    assert sigproc.parabolic_peak_offset(np.ones(5), 2) == 0.0
