"""Signal processing utilities for STFT-based broadband processing."""

from typing import Any, Literal, TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

ComplexArray: TypeAlias = NDArray[np.complexfloating[Any, Any]]
FloatArray: TypeAlias = NDArray[np.floating[Any]]
WindowType: TypeAlias = Literal["hann", "hamming", "blackman"]
StftResult: TypeAlias = tuple[ComplexArray, FloatArray, int, FloatArray]
SignalArray: TypeAlias = ComplexArray | FloatArray


def _stft_layout(num_samples: int, frame_len: int, hop_factor: int) -> tuple[int, int]:
    """Return ``(hop, num_frames)`` for a signal of ``num_samples``.

    The signal is zero-padded at its end to a whole number of frames, so
    ``num_frames = ceil((num_samples - frame_len) / hop) + 1``.

    Raises
    ------
    ValueError
        If the signal is shorter than a frame, or ``hop_factor`` gives no positive hop.

    """
    if num_samples < frame_len:
        msg = (
            f"Signal length ({num_samples}) is shorter than frame_len ({frame_len}). "
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

    return hop, int(np.ceil((num_samples - frame_len) / hop)) + 1


def _stft_window(window: WindowType, frame_len: int) -> FloatArray:
    """Return the named analysis window of ``frame_len`` samples."""
    if window == "hann":
        return np.hanning(frame_len)
    if window == "hamming":
        return np.hamming(frame_len)
    if window == "blackman":
        return np.blackman(frame_len)
    msg = f"Unknown window type: {window}"
    raise ValueError(msg)


def compute_stft(
    signal: ArrayLike,
    frame_len: int,
    hop_factor: int = 4,
    window: WindowType = "hann",
    *,
    two_sided: bool = False,
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
    two_sided : bool, optional
        Return all ``frame_len`` bins in ``numpy.fft.fftfreq`` order whatever the signal.
        Otherwise a signal with no imaginary part gets the ``frame_len // 2 + 1`` bins of a
        real transform, in ``numpy.fft.rfftfreq`` order.

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

    hop, num_frames = _stft_layout(len(signal_array), frame_len, hop_factor)

    full_fft = two_sided or bool(
        np.iscomplexobj(signal_array) and np.any(np.abs(np.imag(signal_array)) > 0.0)
    )

    w = _stft_window(window, frame_len)

    pad_amount = (num_frames - 1) * hop + frame_len - len(signal_array)
    signal_padded = np.concatenate([signal_array, np.zeros(pad_amount, dtype=signal_array.dtype)])

    # Compute STFT
    stft_bins = frame_len if full_fft else (frame_len // 2 + 1)
    stft = np.zeros((num_frames, stft_bins), dtype=np.complex64)

    for i in range(num_frames):
        start = i * hop
        frame = signal_padded[start : start + frame_len]
        if full_fft:
            frame_complex = np.asarray(frame * w, dtype=np.complex64)
            stft[i, :] = np.fft.fft(frame_complex)
        else:
            frame_real = np.asarray(np.real(frame) * w, dtype=np.float32)
            stft[i, :] = np.fft.rfft(frame_real)

    # Generate frequency array (assuming sampling rate of 1.0, caller must scale)
    frequencies = np.fft.fftfreq(frame_len, 1.0) if full_fft else np.fft.rfftfreq(frame_len, 1.0)

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


def analytic_signal(signal: ArrayLike) -> NDArray[np.complex128]:
    r"""Return the one-sided (analytic) form of a real waveform, keeping its RMS.

    .. math::

        x_a = \frac{x + i\,\mathcal{H}\{x\}}{\sqrt{2}}

    where :math:`\mathcal{H}` is the Hilbert transform. It is applied in the frequency domain:
    the positive-frequency bins of ``x`` are scaled by :math:`\sqrt{2}`, the 0 Hz bin by
    :math:`1/\sqrt{2}`, and the negative-frequency bins are set to zero. For an even length the
    fs/2 bin is set to zero as well, because ``numpy.fft.fftfreq`` labels it negative.

    The real pressure is recovered as :math:`\sqrt{2}\,\mathrm{Re}\{x_a\}`.

    Assumptions
    -----------
    - ``x`` has no power at fs/2. Any there is discarded, and the real part of the result then
      differs from :math:`x/\sqrt{2}` by that component.
    - ``x`` has no power at 0 Hz. Half of any there is lost, so the RMS of the result falls
      short of the RMS of ``x`` by that amount.
    - ``x`` is one period of a periodic waveform, as for any transform taken over the whole
      record. A waveform cut from a longer one gains small tails at its ends.

    Parameters
    ----------
    signal : ArrayLike
        Real waveform, one-dimensional.

    Returns
    -------
    NDArray[np.complex128]
        One-sided waveform of the same length, with the RMS of ``signal``.

    Raises
    ------
    ValueError
        If ``signal`` is complex or not one-dimensional.

    """
    signal_array = np.asarray(signal)
    if np.iscomplexobj(signal_array):
        raise ValueError("analytic_signal takes a real waveform, got a complex one")
    if signal_array.ndim != 1:
        raise ValueError("Input signal must be 1D")

    num_samples = len(signal_array)
    gain = np.zeros(num_samples)
    if num_samples:
        gain[0] = 1.0 / np.sqrt(2.0)
        # Bins 1 to ceil(N / 2) - 1 are the positive frequencies below fs/2.
        gain[1 : (num_samples + 1) // 2] = np.sqrt(2.0)
    return np.fft.ifft(np.fft.fft(signal_array.astype(np.float64)) * gain)


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
