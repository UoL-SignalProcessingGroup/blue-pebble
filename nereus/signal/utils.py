"""Signal processing utilities for STFT-based broadband processing."""

import numpy as np


def compute_stft(
    signal: np.ndarray,
    frame_len: int,
    hop_factor: int = 4,
    window: str = "hann",
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """Compute Short-Time Fourier Transform of a signal.

    Uses overlap-add method matching BroadbandArrayProcessor implementation.

    Parameters
    ----------
    signal : np.ndarray
        Input time-domain signal (complex or real).
    frame_len : int
        STFT frame length in samples (power of 2 recommended).
    hop_factor : int, optional
        Hop size = frame_len // hop_factor (4 gives 75% overlap).
    window : str, optional
        Window type ('hann', 'hamming', 'blackman').

    Returns
    -------
    tuple[np.ndarray, np.ndarray, int, np.ndarray]
        Tuple containing:
            stft : np.ndarray
                STFT matrix of shape (num_frames, num_freq_bins).
            frequencies : np.ndarray
                Frequency array for the bins.
            hop : int
                Hop size in samples.
            window_array : np.ndarray
                The window array used.

    """
    signal = np.asarray(signal, dtype=np.complex64)
    hop = frame_len // hop_factor

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
    num_frames = int(np.ceil((len(signal) - frame_len) / hop)) + 1
    pad_amount = (num_frames - 1) * hop + frame_len - len(signal)
    signal_padded = np.concatenate([signal, np.zeros(pad_amount, dtype=signal.dtype)])

    # Compute STFT
    stft_bins = frame_len // 2 + 1
    stft = np.zeros((num_frames, stft_bins), dtype=np.complex64)

    for i in range(num_frames):
        start = i * hop
        frame = signal_padded[start : start + frame_len] * w
        # Handle complex signals by taking real part for FFT
        # (For complex baseband signals, use full FFT instead of rfft)
        if np.iscomplexobj(signal):
            # For complex signals, compute FFT of real part
            frame_real = np.real(frame).astype(np.float32)
            stft[i, :] = np.fft.rfft(frame_real)
        else:
            frame_real = np.asarray(frame, dtype=np.float32)
            stft[i, :] = np.fft.rfft(frame_real)

    # Generate frequency array (assuming sampling rate of 1.0, caller must scale)
    frequencies = np.fft.rfftfreq(frame_len, 1.0)

    return stft, frequencies, hop, w


def inverse_stft(
    stft: np.ndarray,
    frame_len: int,
    hop: int,
    window: np.ndarray,
) -> np.ndarray:
    """Reconstruct time-domain signal from STFT using overlap-add.

    Matches BroadbandArrayProcessor._inverse_stft() implementation.

    Parameters
    ----------
    stft : np.ndarray
        STFT matrix of shape (num_frames, num_freq_bins).
    frame_len : int
        STFT frame length in samples.
    hop : int
        Hop size in samples.
    window : np.ndarray
        Window array used in forward STFT.

    Returns
    -------
    np.ndarray
        Reconstructed time-domain signal.

    """
    num_frames = stft.shape[0]
    signal_len = (num_frames - 1) * hop + frame_len
    reconstructed = np.zeros(signal_len, dtype=np.complex64)
    window_sum = np.zeros(signal_len, dtype=np.float32)

    for i in range(num_frames):
        start = i * hop
        frame_freq = stft[i, :]
        frame_time = np.fft.irfft(frame_freq, n=frame_len)
        frame_time = np.asarray(frame_time, dtype=np.complex64)

        # Apply window and accumulate
        reconstructed[start : start + frame_len] += frame_time * window
        window_sum[start : start + frame_len] += window**2

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


def apply_fade_in(signal: np.ndarray, fade_samples: int) -> np.ndarray:
    """Apply smooth cosine-taper fade-in to signal arrival.

    Uses a raised cosine (Tukey) window for smooth signal arrival, matching the
    BroadbandArrayProcessor implementation.

    Parameters
    ----------
    signal : np.ndarray
        Input signal.
    fade_samples : int
        Number of samples for fade-in duration.

    Returns
    -------
    np.ndarray
        Signal with fade-in applied.

    """
    if fade_samples <= 0 or fade_samples >= len(signal):
        return signal

    # Cosine taper: 0.5 * (1 - cos(pi * t / T))
    # This produces a smooth S-curve from 0 to 1
    fade = 0.5 * (1.0 - np.cos(np.pi * np.arange(fade_samples) / fade_samples))
    signal_faded = signal.copy()
    signal_faded[:fade_samples] *= fade
    return signal_faded
