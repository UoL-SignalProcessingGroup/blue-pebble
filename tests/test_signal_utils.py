"""Tests for small deterministic signal utility helpers."""

from __future__ import annotations

import numpy as np
import pytest

from .support import load_module_from_repo

signal_utils = load_module_from_repo("bluepebble/signal/utils.py", "bluepebble_signal_utils_test")


def test_compute_stft_rejects_signal_shorter_than_frame_len() -> None:
    """Signals shorter than frame_len should raise ValueError with a helpful message."""
    short_signal = np.ones(7, dtype=np.float32)

    with pytest.raises(ValueError, match="shorter than frame_len"):
        signal_utils.compute_stft(short_signal, frame_len=8)


def test_compute_stft_rejects_non_positive_hop_factor() -> None:
    """hop_factor of zero or negative should raise ValueError before dividing."""
    signal = np.ones(16, dtype=np.float32)

    with pytest.raises(ValueError, match="hop_factor must be a positive integer"):
        signal_utils.compute_stft(signal, frame_len=8, hop_factor=0)

    with pytest.raises(ValueError, match="hop_factor must be a positive integer"):
        signal_utils.compute_stft(signal, frame_len=8, hop_factor=-2)


def test_compute_stft_rejects_hop_factor_larger_than_frame_len() -> None:
    """hop_factor > frame_len produces hop=0 and should raise ValueError."""
    signal = np.ones(16, dtype=np.float32)

    with pytest.raises(ValueError, match="hop_factor.*larger than frame_len"):
        signal_utils.compute_stft(signal, frame_len=8, hop_factor=16)


def test_compute_stft_rejects_unknown_window() -> None:
    """Unsupported window names should raise a clear validation error."""
    signal = np.ones(16, dtype=np.float32)

    with pytest.raises(ValueError, match="Unknown window type"):
        signal_utils.compute_stft(signal, frame_len=8, window="triangle")


def test_compute_and_inverse_stft_round_trip_real_signal() -> None:
    """Forward and inverse STFT should approximately reconstruct a real signal."""
    signal = np.sin(2.0 * np.pi * 0.05 * np.arange(64)).astype(np.float32)

    stft, _frequencies, hop, window = signal_utils.compute_stft(
        signal,
        frame_len=16,
        hop_factor=4,
    )
    reconstructed = signal_utils.inverse_stft(stft, frame_len=16, hop=hop, window=window)

    np.testing.assert_allclose(reconstructed.real, signal[hop:-hop], atol=1e-5, rtol=1e-5)


def _negative_bin_fraction(waveform: np.ndarray) -> float:
    """Return the share of a waveform's power in the bins ``fftfreq`` labels negative."""
    power = np.abs(np.fft.fft(waveform)) ** 2
    return float(power[np.fft.fftfreq(len(waveform)) < 0].sum() / power.sum())


def _band_limited_noise(num_samples: int) -> np.ndarray:
    """Return seeded real noise with no power at 0 Hz or fs/2."""
    spectrum = np.fft.rfft(np.random.default_rng(3).standard_normal(num_samples))
    spectrum[0] = 0.0
    if num_samples % 2 == 0:
        spectrum[-1] = 0.0
    return np.fft.irfft(spectrum, num_samples)


@pytest.mark.parametrize("num_samples", [256, 255])
def test_analytic_signal_has_no_power_in_the_negative_bins(num_samples: int) -> None:
    """White noise fills every bin, the fs/2 bin of an even length included."""
    noise = np.random.default_rng(1).standard_normal(num_samples)

    assert _negative_bin_fraction(signal_utils.analytic_signal(noise)) < 1e-12


@pytest.mark.parametrize("num_samples", [256, 255])
def test_analytic_signal_of_a_cosine_is_its_complex_exponential(num_samples: int) -> None:
    """``A cos(wt + phi)`` has the analytic form ``A exp(i(wt + phi)) / sqrt(2)``."""
    amplitude, cycles, phase = 3.0, 20, 0.7
    angle = 2.0 * np.pi * cycles * np.arange(num_samples) / num_samples + phase

    analytic = signal_utils.analytic_signal(amplitude * np.cos(angle))

    np.testing.assert_allclose(
        analytic, amplitude * np.exp(1j * angle) / np.sqrt(2.0), atol=1e-12 * amplitude
    )


@pytest.mark.parametrize("num_samples", [256, 255])
def test_analytic_signal_keeps_the_waveform_and_its_rms(num_samples: int) -> None:
    """With no power at 0 Hz or fs/2, the real part is ``x / sqrt(2)`` and the RMS is that of x."""
    noise = _band_limited_noise(num_samples)

    analytic = signal_utils.analytic_signal(noise)

    np.testing.assert_allclose(analytic.real, noise / np.sqrt(2.0), atol=1e-12)
    assert np.sqrt(np.mean(np.abs(analytic) ** 2)) == pytest.approx(
        np.sqrt(np.mean(noise**2)), rel=1e-12
    )


def test_analytic_signal_halves_the_power_of_a_constant() -> None:
    """The documented loss at 0 Hz: a constant keeps ``1 / sqrt(2)`` of its amplitude."""
    np.testing.assert_allclose(
        signal_utils.analytic_signal(np.full(16, 2.0)), np.full(16, np.sqrt(2.0)), atol=1e-12
    )


def test_analytic_signal_rejects_complex_and_two_dimensional_input() -> None:
    """A complex waveform has no single real pressure to convert."""
    with pytest.raises(ValueError, match="real waveform"):
        signal_utils.analytic_signal(np.ones(8, dtype=np.complex128))
    with pytest.raises(ValueError, match="1D"):
        signal_utils.analytic_signal(np.ones((2, 8)))


def test_apply_fade_in_tapers_only_the_leading_segment() -> None:
    """Fade-in should smoothly taper the leading samples and leave the tail untouched."""
    signal = np.ones(8, dtype=np.float32)

    faded = signal_utils.apply_fade_in(signal, fade_samples=4)

    assert faded[0] == pytest.approx(0.0)
    assert faded[1] < faded[2] < faded[3] < 1.0
    np.testing.assert_allclose(faded[4:], signal[4:])


def test_apply_fade_in_is_noop_for_non_positive_or_full_length_fades() -> None:
    """Degenerate fade lengths should return the original signal unchanged."""
    signal = np.arange(5, dtype=np.float32)

    np.testing.assert_array_equal(signal_utils.apply_fade_in(signal, fade_samples=0), signal)
    np.testing.assert_array_equal(signal_utils.apply_fade_in(signal, fade_samples=5), signal)
