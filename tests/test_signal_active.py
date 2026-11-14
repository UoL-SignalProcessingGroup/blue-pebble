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


@pytest.mark.parametrize("phase_rad", [0.0, 0.7])
def test_baseband_lfm_is_the_passband_lfm_mixed_down_by_its_carrier(bp, phase_rad) -> None:
    """A baseband LFM equals the passband LFM times exp(-j 2 pi f_c t) at every sample."""
    decimation = 6
    passband = _lfm(bp, phase_rad=phase_rad)
    baseband = _lfm(bp, phase_rad=phase_rad, sampling_rate_hz=FS // decimation, baseband=True)
    t = np.arange(baseband.num_samples) * decimation / FS

    mixed_down = passband.generate()[::decimation] * np.exp(-2j * np.pi * 6000.0 * t)

    np.testing.assert_allclose(baseband.generate(), mixed_down, atol=1e-9 * passband.amplitude)
    assert baseband.demodulation_frequency_hz == 6000.0
    assert passband.demodulation_frequency_hz == 0.0


def test_baseband_cw_is_the_passband_cw_mixed_down_by_its_carrier(bp) -> None:
    """A baseband CW pulse is the constant envelope A exp(j phase) of the passband tone."""
    settings = dict(frequency_hz=3000.0, source_level_db=180.0, phase_rad=0.4, duration_s=0.1)
    passband = bp.CWSignal(sampling_rate_hz=FS, **settings)
    baseband = bp.CWSignal(sampling_rate_hz=FS // 48, baseband=True, **settings)
    t = np.arange(baseband.num_samples) * 48 / FS

    mixed_down = passband.generate()[::48] * np.exp(-2j * np.pi * 3000.0 * t)

    np.testing.assert_allclose(baseband.generate(), mixed_down, atol=1e-9 * passband.amplitude)
    assert baseband.demodulation_frequency_hz == 3000.0


@pytest.mark.parametrize(
    ("sampling_rate_hz", "match"),
    [
        # The 4-8 kHz sweep reaches 2 kHz either side of its 6 kHz carrier.
        (4000, "twice the pulse's largest offset"),
        (12_000, "below the carrier"),
    ],
)
def test_baseband_lfm_rejects_unusable_sampling_rates(bp, sampling_rate_hz, match) -> None:
    """A baseband rate must hold the sweep and keep every bin a positive frequency."""
    with pytest.raises(ValueError, match=match):
        _lfm(bp, sampling_rate_hz=sampling_rate_hz, baseband=True)


def test_baseband_sampling_rate_only_has_to_cover_the_bandwidth(bp) -> None:
    """A rate below the pulse's highest frequency works at baseband and not at passband."""
    assert _lfm(bp, sampling_rate_hz=5000, baseband=True).generate().shape == (500,)
    with pytest.raises(ValueError, match="highest frequency"):
        _lfm(bp, sampling_rate_hz=5000)


@pytest.mark.parametrize(
    ("lowest_hz", "highest_hz"),
    [(900.0, 1300.0), (700.0, 1100.0)],
)
def test_baseband_check_uses_the_larger_offset_from_the_carrier(bp, lowest_hz, highest_hz) -> None:
    """A band 300 Hz to one side of a 1 kHz carrier needs more than 600 Hz, whichever side."""

    class OffCentre(bp.CWSignal):
        @property
        def lowest_frequency_hz(self):
            return lowest_hz

        @property
        def highest_frequency_hz(self):
            return highest_hz

    settings = dict(frequency_hz=1000.0, source_level_db=0.0, duration_s=0.1, baseband=True)

    OffCentre(sampling_rate_hz=700, **settings)
    with pytest.raises(ValueError, match="twice the pulse's largest offset"):
        OffCentre(sampling_rate_hz=600, **settings)
