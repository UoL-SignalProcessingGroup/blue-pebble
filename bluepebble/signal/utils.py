"""Signal processing utilities for STFT-based broadband processing."""

from typing import Any, Literal, TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

ComplexArray: TypeAlias = NDArray[np.complexfloating[Any, Any]]
FloatArray: TypeAlias = NDArray[np.floating[Any]]
WindowType: TypeAlias = Literal["hann", "hamming", "blackman"]
StftResult: TypeAlias = tuple[ComplexArray, FloatArray, int, FloatArray]
SignalArray: TypeAlias = ComplexArray | FloatArray


def compute_stft(
    signal: ArrayLike,
    frame_len: int,
    hop_factor: int = 4,
    window: WindowType = "hann",
) -> StftResult:
    """Compute Short-Time Fourier Transform of a signal.

    Uses overlap-add method matching BroadbandArrayProcessor implementation.

    Parameters
    ----------
    signal : ArrayLike
        Input time-domain signal (complex or real).
    frame_len : int
        STFT frame length in samples (power of 2 recommended).
    hop_factor : int, optional
        Hop size = frame_len // hop_factor (4 gives 75% overlap).
    window : WindowType, optional
        Window type ('hann', 'hamming', 'blackman').

    Returns
    -------
    StftResult
        Tuple containing:
            stft : ComplexArray
                STFT matrix of shape (num_frames, num_freq_bins).
            frequencies : FloatArray
                Frequency array for the bins.
            hop : int
                Hop size in samples.
            window_array : FloatArray
                The window array used.

    """
    signal_array = np.asarray(signal)
    if signal_array.ndim != 1:
        msg = "Input signal must be 1D"
        raise ValueError(msg)

    if len(signal_array) < frame_len:
        msg = (
            f"Signal length ({len(signal_array)}) is shorter than frame_len ({frame_len}). "
            "Increase duration_s, increase sampling_rate_hz, or reduce frame_len so that "
            "num_samples >= frame_len."
        )
        raise ValueError(msg)

    if hop_factor < 1:
        msg = f"hop_factor must be a positive integer, got {hop_factor}"
        raise ValueError(msg)

    hop = frame_len // hop_factor
    if hop < 1:
        msg = (
            f"hop_factor ({hop_factor}) is larger than frame_len ({frame_len}), "
            "which produces a hop of zero samples. "
            "Use a hop_factor <= frame_len."
        )
        raise ValueError(msg)

    has_imag = np.iscomplexobj(signal_array) and np.any(np.abs(np.imag(signal_array)) > 0.0)

    # Get window
    if window == "hann":
        w = np.hanning(frame_len)
    elif window == "hamming":
        w = np.hamming(frame_len)
    elif window == "blackman":
        w = np.blackman(frame_len)
    else:
        msg = f"Unknown window type: {window}"
        raise ValueError(msg)

    # Calculate number of frames and pad signal
    num_frames = int(np.ceil((len(signal_array) - frame_len) / hop)) + 1
    pad_amount = (num_frames - 1) * hop + frame_len - len(signal_array)
    signal_padded = np.concatenate([signal_array, np.zeros(pad_amount, dtype=signal_array.dtype)])

    # Compute STFT
    stft_bins = frame_len if has_imag else (frame_len // 2 + 1)
    stft = np.zeros((num_frames, stft_bins), dtype=np.complex64)

    for i in range(num_frames):
        start = i * hop
        frame = signal_padded[start : start + frame_len]
        if has_imag:
            frame_complex = np.asarray(frame * w, dtype=np.complex64)
            stft[i, :] = np.fft.fft(frame_complex)
        else:
            frame_real = np.asarray(np.real(frame) * w, dtype=np.float32)
            stft[i, :] = np.fft.rfft(frame_real)

    # Generate frequency array (assuming sampling rate of 1.0, caller must scale)
    frequencies = np.fft.fftfreq(frame_len, 1.0) if has_imag else np.fft.rfftfreq(frame_len, 1.0)

    return stft, frequencies, hop, w


def inverse_stft(
    stft: ArrayLike,
    frame_len: int,
    hop: int,
    window: ArrayLike,
) -> ComplexArray:
    """Reconstruct time-domain signal from STFT using overlap-add.

    Matches BroadbandArrayProcessor._inverse_stft() implementation.

    Parameters
    ----------
    stft : ArrayLike
        STFT matrix of shape (num_frames, num_freq_bins).
    frame_len : int
        STFT frame length in samples.
    hop : int
        Hop size in samples.
    window : ArrayLike
        Window array used in forward STFT.

    Returns
    -------
    ComplexArray
        Reconstructed time-domain signal.

    """
    stft_array = np.asarray(stft)
    window_array = np.asarray(window)

    num_frames = stft_array.shape[0]
    signal_len = (num_frames - 1) * hop + frame_len
    reconstructed = np.zeros(signal_len, dtype=np.complex64)
    window_sum = np.zeros(signal_len, dtype=np.float32)

    onesided_bins = frame_len // 2 + 1
    twosided_bins = frame_len
    is_twosided = stft_array.shape[1] == twosided_bins
    is_onesided = stft_array.shape[1] == onesided_bins

    if not (is_twosided or is_onesided):
        msg = (
            f"Invalid STFT shape {stft_array.shape}. Expected num_freq_bins "
            f"to be {onesided_bins} (rFFT) or {twosided_bins} (FFT)."
        )
        raise ValueError(msg)

    for i in range(num_frames):
        start = i * hop
        frame_freq = stft_array[i, :]
        frame_time = (
            np.fft.ifft(frame_freq, n=frame_len)
            if is_twosided
            else np.fft.irfft(frame_freq, n=frame_len)
        )
        frame_time = np.asarray(frame_time, dtype=np.complex64)

        # Apply window and accumulate
        reconstructed[start : start + frame_len] += frame_time * window_array
        window_sum[start : start + frame_len] += window_array**2

    # Normalize by window overlap
    # Avoid division by zero in regions with proper overlap
    # Areas with very low window_sum are edge artifacts and should be trimmed
    eps = 1e-10
    valid_mask = window_sum > eps
    reconstructed[valid_mask] /= window_sum[valid_mask]

    # Trim edge artifacts: Remove regions where window normalization is incomplete
    # This happens at the start (first hop_len samples) and end (last hop_len samples)
    # where there isn't full overlap-add coverage
    trim_start = hop  # Remove first hop samples (incomplete overlap)
    trim_end = hop  # Remove last hop samples (incomplete overlap)

    if len(reconstructed) > trim_start + trim_end:
        reconstructed = reconstructed[trim_start:-trim_end]

    return reconstructed


def apply_fade_in(signal: ArrayLike, fade_samples: int) -> SignalArray:
    """Apply smooth cosine-taper fade-in to signal arrival.

    Uses a raised cosine (Tukey) window for smooth signal arrival, matching the
    BroadbandArrayProcessor implementation.

    Parameters
    ----------
    signal : ArrayLike
        Input signal.
    fade_samples : int
        Number of samples for fade-in duration.

    Returns
    -------
    SignalArray
        Signal with fade-in applied.

    """
    signal_array = np.asarray(signal)
    if fade_samples <= 0 or fade_samples >= len(signal_array):
        return cast(SignalArray, signal_array)

    # Cosine taper: 0.5 * (1 - cos(pi * t / T))
    # This produces a smooth S-curve from 0 to 1
    fade = 0.5 * (1.0 - np.cos(np.pi * np.arange(fade_samples) / fade_samples))
    signal_faded = signal_array.copy()
    signal_faded[:fade_samples] *= fade
    return cast(SignalArray, signal_faded)


def apply_fade_out(signal: ArrayLike, fade_samples: int) -> SignalArray:
    """Apply smooth cosine-taper fade-out to the end of a signal.

    Uses the same raised-cosine profile as :func:`apply_fade_in`, reversed so
    the signal transitions smoothly from 1 to 0 over ``fade_samples``.

    Parameters
    ----------
    signal : ArrayLike
        Input signal.
    fade_samples : int
        Number of samples for fade-out duration.

    Returns
    -------
    SignalArray
        Signal with fade-out applied.

    """
    signal_array = np.asarray(signal)
    if fade_samples <= 0 or fade_samples >= len(signal_array):
        return cast(SignalArray, signal_array)

    # Reuse the same raised-cosine taper and reverse it for a 1 -> 0 ramp.
    fade = 0.5 * (1.0 - np.cos(np.pi * np.arange(fade_samples) / fade_samples))
    signal_faded = signal_array.copy()
    signal_faded[-fade_samples:] *= fade[::-1]
    return cast(SignalArray, signal_faded)
