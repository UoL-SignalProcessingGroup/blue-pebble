"""Tests for the active sonar transmit pulses (CW and LFM)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from .support import isolate_real_bluepebble_imports, restore_real_bluepebble_imports

FS = 48_000


@pytest.fixture
def bp():
    """Import the real signal classes without leaking caching into other tests."""
    snapshot = isolate_real_bluepebble_imports()
    from bluepebble.signal import CWSignal, LFMSignal

    yield SimpleNamespace(CWSignal=CWSignal, LFMSignal=LFMSignal)
    restore_real_bluepebble_imports(snapshot)


def _lfm(bp, **overrides):
    kwargs = dict(
        freq_min_hz=4000.0,
        freq_max_hz=8000.0,
        rise_time_s=0.005,
        source_level_db=200.0,
        duration_s=0.1,
        sampling_rate_hz=FS,
    )
    kwargs.update(overrides)
    return bp.LFMSignal(**kwargs)


def test_cw_is_a_tone_at_its_frequency_and_level(bp) -> None:
    """A CW pulse has constant amplitude 10^(SL/20) and its spectral peak at frequency_hz."""
    cw = bp.CWSignal(
        frequency_hz=3000.0, source_level_db=180.0, duration_s=0.1, sampling_rate_hz=FS
    )
    pulse = cw.generate()

    assert pulse.shape == (cw.num_samples,) == (4800,)
    np.testing.assert_allclose(np.abs(pulse), 1e9)
    freqs = np.fft.fftfreq(pulse.size, 1 / FS)
    assert freqs[np.argmax(np.abs(np.fft.fft(pulse)))] == pytest.approx(3000.0)


def test_cw_phase_offset_rotates_the_pulse(bp) -> None:
    """phase_rad sets the carrier phase at t = 0."""
    cw = bp.CWSignal(
        frequency_hz=3000.0,
        source_level_db=0.0,
        phase_rad=np.pi / 3,
        duration_s=0.01,
        sampling_rate_hz=FS,
    )

    assert np.angle(cw.generate()[0]) == pytest.approx(np.pi / 3)


def test_lfm_sweeps_from_min_to_max(bp) -> None:
    """The instantaneous frequency rises linearly from freq_min_hz to freq_max_hz."""
    pulse = _lfm(bp).generate()
    t = np.arange(pulse.size) / FS
    inst_freq = np.diff(np.unwrap(np.angle(pulse))) * FS / (2 * np.pi)
    expected = 4000.0 + (8000.0 - 4000.0) / 0.1 * (t[:-1] + 0.5 / FS)

    np.testing.assert_allclose(inst_freq, expected, atol=1e-6)
    assert _lfm(bp).carrier_frequency_hz == 6000.0


def test_lfm_envelope_tapers_to_zero_at_the_edges(bp) -> None:
    """The envelope starts at 0, is flat at 10^(SL/20) in the middle, and falls towards 0."""
    lfm = _lfm(bp)
    envelope = np.abs(lfm.generate()) / lfm.amplitude
    rise = int(0.005 * FS)

    assert envelope[0] == pytest.approx(0.0)
    np.testing.assert_allclose(envelope[rise + 1 : -rise - 1], 1.0)
    assert np.all(np.diff(envelope[: rise + 1]) > 0)
    assert np.all(np.diff(envelope[-rise:]) < 0)
    assert envelope[-1] < 0.05


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"freq_max_hz": 4000.0}, "must exceed freq_min_hz"),
        ({"rise_time_s": 0.06}, "at most half"),
        ({"rise_time_s": 0.0}, "positive"),
        ({"sampling_rate_hz": 8000}, "must exceed the pulse's highest frequency"),
        ({"duration_s": 0.0}, "duration_s"),
    ],
)
def test_lfm_rejects_invalid_configuration(bp, overrides, match) -> None:
    """Sweeps, tapers and sampling rates that cannot work are rejected."""
    with pytest.raises(ValueError, match=match):
        _lfm(bp, **overrides)


def test_cw_rejects_undersampled_tone(bp) -> None:
    """A tone at or above the sampling rate would alias."""
    with pytest.raises(ValueError, match="highest frequency"):
        bp.CWSignal(frequency_hz=FS, source_level_db=0.0, duration_s=0.1, sampling_rate_hz=FS)
