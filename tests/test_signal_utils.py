"""Tests for small deterministic signal utility helpers."""

from __future__ import annotations

import numpy as np
import pytest

from .support import load_module_from_repo

signal_utils = load_module_from_repo("bluepebble/signal/utils.py", "bluepebble_signal_utils_test")


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
