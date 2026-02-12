"""Anthropogenic signal models for sensor arrays.

© Copyright 2025 Joshua J. Wakefield.
Licensed under the MIT License.
"""

import numpy as np
from stonesoup.base import Property

from .base import Signal


class TonalSignal(Signal):
    """Generates a signal from a source defined by a sum of pure tones.

    This model is ideal for sources that can be represented as a combination
    of sinusoids, each with a specific frequency, amplitude, and phase.
    """

    def _generate_base_signal(self, source) -> np.ndarray:
        """Satisfy the abstract base class.

        This method is never called for TonalSignal as it overrides `generate`.
        """
        # This implementation is not used but is required by the base class.
        # It will never be called because the `generate` method is overridden.
        pass

    def generate(
        self, source, sensor_delays_s, tloss_db, propagation_time_s
    ) -> np.ndarray:
        """Generate the signal received across all sensors from a single source.

        Parameters
        ----------
        source : State
            The source state. Must contain ``amplitudes_upa``,
            ``frequencies_hz``, and ``phases_rad`` in its metadata dictionary.
        sensor_delays_s : numpy.ndarray
            The relative time delay for each sensor in the array.
        tloss_db : float or numpy.ndarray
            The transmission loss in decibels. Can be a scalar or an array
            matching the number of frequencies.
        propagation_time_s : float
            The time in seconds for the signal to propagate from the source to
            the array's origin.

        Returns
        -------
        numpy.ndarray
            An array of complex signals received by the sensors, with shape
            (num_sensors, num_samples).

        Raises
        ------
        ValueError
            If ``tloss_db`` is an array and its length does not match the
            number of frequencies in the source metadata.

        """
        # Create a 1D array representing the time vector for the signal snapshot
        time_array_s = np.arange(self.num_samples) / self.sampling_rate_hz

        amplitudes_upa = source.metadata["amplitudes_upa"]
        frequencies_hz = source.metadata["frequencies_hz"]
        phases_rad = source.metadata["phases_rad"]

        # Handle per-frequency or single transmission loss
        if isinstance(tloss_db, (np.ndarray, list)):
            # Per-frequency transmission loss
            tloss_db_array = np.array(tloss_db)
            if len(tloss_db_array) != len(amplitudes_upa):
                raise ValueError(
                    f"Length of tloss_db array ({len(tloss_db_array)}) must match "
                    f"number of frequencies ({len(amplitudes_upa)})"
                )
            received_amplitude_upa = amplitudes_upa * 10 ** (-tloss_db_array / 20.0)
        else:
            # Single transmission loss applied to all frequencies
            received_amplitude_upa = amplitudes_upa * 10 ** (-tloss_db / 20.0)

        # --- Use NumPy broadcasting to perform calculations efficiently ---
        # Reshape arrays to dimensions: (sensors, tonals, samples)
        time_reshaped = time_array_s[np.newaxis, np.newaxis, :]
        delays_reshaped = sensor_delays_s[:, np.newaxis, np.newaxis]
        freq_reshaped = frequencies_hz[np.newaxis, :, np.newaxis]
        phase_reshaped = phases_rad[np.newaxis, :, np.newaxis]

        # Calculate the instantaneous phase
        # Shape = (num_sensors, num_tonals, num_samples)
        total_phase = (
            2
            * np.pi
            * freq_reshaped
            * (time_reshaped - propagation_time_s - delays_reshaped)
            + phase_reshaped
        )

        # Create the complex signal components using Euler's formula
        amp_reshaped = received_amplitude_upa[np.newaxis, :, np.newaxis]
        all_tonal_components = amp_reshaped * np.exp(1j * total_phase)

        # Sum the components along the tonal axis (axis=1) to get the final
        # signal at each sensor.
        sensor_signals = np.sum(all_tonal_components, axis=1)

        return sensor_signals


class BlendedTonalSignal(Signal):
    """Generates blended tonal signals for phase/amplitude continuity.

    This signal model maintains state across timesteps to ensure both phase and
    amplitude continuity. It tracks the accumulated phase offset and previous
    signal state to ensure smooth transitions between timesteps.

    Parameters
    ----------
    blend_fraction : float, optional
        Fraction of signal duration to use for blending. Default is 0.1 (10% of
        signal duration).

    """

    blend_fraction = Property(
        float, default=0.1, doc="Fraction of signal to blend for continuity"
    )

    def __init__(self, *args, **kwargs):
        """Initialize the blended signal generator with state tracking."""
        super().__init__(*args, **kwargs)
        # Dictionary to store state for each source
        # Key: source_key, Value: dict with 'previous_signal', 'last_time',
        # 'phase_offset'
        self._source_states = {}

    def _get_source_key(self, source, num_sensors: int) -> tuple:
        """Generate a unique key for this source and sensor configuration.

        Parameters
        ----------
        source : State
            The source state object.
        num_sensors : int
            Number of sensors in the array.

        Returns
        -------
        tuple
            A tuple that uniquely identifies this source-sensor combination.

        """
        # Create a hashable key from source metadata
        # Use frequency tuple as identifier (assumes frequencies don't change)
        freq_tuple = tuple(source.metadata["frequencies_hz"])
        return (freq_tuple, num_sensors)

    def _generate_base_signal(self, source) -> np.ndarray:
        """Satisfy the abstract base class.

        This method is never called for BlendedTonalSignal as it overrides `generate`.
        """
        pass

    def generate(
        self, source, sensor_delays_s, tloss_db, propagation_time_s
    ) -> np.ndarray:
        """Generate blended signal with blend for phase and amplitude continuity.

        Parameters
        ----------
        source : State
            The source state. Must contain ``amplitudes_upa``,
            ``frequencies_hz``, and ``phases_rad`` in its metadata dictionary.
        sensor_delays_s : numpy.ndarray
            The relative time delay for each sensor in the array.
        tloss_db : float or numpy.ndarray
            The transmission loss in decibels.
        propagation_time_s : float
            The time in seconds for the signal to propagate from the source to
            the array's origin.

        Returns
        -------
        numpy.ndarray
            An array of complex signals received by the sensors, with shape
            (num_sensors, num_samples).

        Raises
        ------
        ValueError
            If ``tloss_db`` length does not match the number of frequencies.

        """
        num_sensors = len(sensor_delays_s)
        source_key = self._get_source_key(source, num_sensors)

        # Retrieve or initialize source state
        if source_key not in self._source_states:
            self._source_states[source_key] = {
                "previous_signal": None,
                "last_time": None,
                "last_phase": None,
                "cumulative_time": 0.0,
            }

        state = self._source_states[source_key]

        # Calculate time offset for phase continuity
        if state["last_time"] is not None:
            # Calculate elapsed time since last generation
            time_delta = (source.timestamp - state["last_time"]).total_seconds()
            state["cumulative_time"] += time_delta
        else:
            state["cumulative_time"] = 0.0

        # Create time array relative to cumulative time
        time_array_s = (
            np.arange(self.num_samples) / self.sampling_rate_hz
            + state["cumulative_time"]
        )

        amplitudes_upa = source.metadata["amplitudes_upa"]
        frequencies_hz = source.metadata["frequencies_hz"]
        phases_rad = source.metadata["phases_rad"]

        # Handle per-frequency or single transmission loss
        if isinstance(tloss_db, (np.ndarray, list)):
            tloss_db_array = np.array(tloss_db)
            if len(tloss_db_array) != len(amplitudes_upa):
                raise ValueError(
                    f"Length of tloss_db array ({len(tloss_db_array)}) must match "
                    f"number of frequencies ({len(amplitudes_upa)})"
                )
            received_amplitude_upa = amplitudes_upa * 10 ** (-tloss_db_array / 20.0)
        else:
            received_amplitude_upa = amplitudes_upa * 10 ** (-tloss_db / 20.0)

        # Reshape arrays to dimensions: (sensors, tonals, samples)
        time_reshaped = time_array_s[np.newaxis, np.newaxis, :]
        delays_reshaped = sensor_delays_s[:, np.newaxis, np.newaxis]
        freq_reshaped = frequencies_hz[np.newaxis, :, np.newaxis]
        phase_reshaped = phases_rad[np.newaxis, :, np.newaxis]

        # Calculate the instantaneous phase using cumulative time
        # Shape = (num_sensors, num_tonals, num_samples)
        total_phase = (
            2
            * np.pi
            * freq_reshaped
            * (time_reshaped - propagation_time_s - delays_reshaped)
            + phase_reshaped
        )

        # Create the complex signal components using Euler's formula
        amp_reshaped = received_amplitude_upa[np.newaxis, :, np.newaxis]
        all_tonal_components = amp_reshaped * np.exp(1j * total_phase)

        # Sum the components along the tonal axis (axis=1) to get the final
        # signal at each sensor.
        sensor_signals = np.sum(all_tonal_components, axis=1)

        # Apply blending if we have previous signal
        if state["previous_signal"] is not None:
            blend_samples = int(self.num_samples * self.blend_fraction)

            if blend_samples > 0:
                # Create smooth blending window (cosine taper)
                fade_out = np.cos(np.linspace(0, np.pi / 2, blend_samples)) ** 2
                fade_in = np.sin(np.linspace(0, np.pi / 2, blend_samples)) ** 2

                # Get the tail of the previous signal
                prev_signal = state["previous_signal"]
                prev_tail = prev_signal[:, -blend_samples:]

                # Blend the beginning of current signal with end of previous
                current_head = sensor_signals[:, :blend_samples].copy()
                blended_section = (
                    prev_tail * fade_out[np.newaxis, :]
                    + current_head * fade_in[np.newaxis, :]
                )

                # Replace the beginning of current signal with blended section
                sensor_signals[:, :blend_samples] = blended_section

        # Store current state for next timestep
        state["previous_signal"] = sensor_signals.copy()
        state["last_time"] = source.timestamp

        return sensor_signals

    def reset(self):
        """Clear all stored previous signals.

        This should be called when starting a new simulation or when
        continuity should be reset.
        """
        self._source_states.clear()


class OverlapAddTonalSignal(Signal):
    """Generates tonal signals using overlap-add with perfect reconstruction.

    This signal model uses the overlap-add method with synthesis windows to
    achieve perfect phase and amplitude continuity across timesteps without
    transient artifacts. It employs a fixed 50% overlap with Hann windows.

    The method generates signals that are twice the requested duration, applies
    windowing, and buffers the second half for overlap-adding with the next
    timestep.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the overlap-add signal generator with state tracking."""
        super().__init__(*args, **kwargs)
        # Dictionary to store state for each source
        # Key: source_key, Value: dict with buffered overlap segment
        self._source_states: dict[tuple, dict] = {}
        # Pre-compute synthesis windows
        self._synthesis_window: np.ndarray | None = None

    def _get_source_key(self, source, num_sensors: int) -> tuple:
        """Generate a unique key for this source and sensor configuration.

        Parameters
        ----------
        source : State
            The source state object.
        num_sensors : int
            Number of sensors in the array.

        Returns
        -------
        tuple
            A tuple that uniquely identifies this source-sensor combination.

        """
        freq_tuple = tuple(source.metadata["frequencies_hz"])
        return (freq_tuple, num_sensors)

    def _get_synthesis_window(self) -> np.ndarray:
        """Generate or retrieve the cached synthesis window.

        Uses Hann window for perfect reconstruction with 50% overlap.
        The window satisfies the Constant Overlap-Add (COLA) constraint:
        w[n] + w[n + hop_size] = 1 for all n

        Returns
        -------
        numpy.ndarray
            The synthesis window array.

        """
        if self._synthesis_window is None:
            # Generate Hann window of length 2*num_samples
            # This ensures 50% overlap when hop_size = num_samples
            extended_length = 2 * self.num_samples
            self._synthesis_window = np.hanning(extended_length)
        return self._synthesis_window

    def _generate_base_signal(self, source) -> np.ndarray:
        """Satisfy the abstract base class.

        This method is never called for OverlapAddTonalSignal as it overrides
        `generate`.
        """
        pass

    def generate(
        self, source, sensor_delays_s, tloss_db, propagation_time_s
    ) -> np.ndarray:
        """Generate signal using overlap-add method with perfect reconstruction.

        Parameters
        ----------
        source : State
            The source state. Must contain ``amplitudes_upa``,
            ``frequencies_hz``, and ``phases_rad`` in its metadata dictionary.
        sensor_delays_s : numpy.ndarray
            The relative time delay for each sensor in the array.
        tloss_db : float or numpy.ndarray
            The transmission loss in decibels.
        propagation_time_s : float
            The time in seconds for the signal to propagate from the source to
            the array's origin.

        Returns
        -------
        numpy.ndarray
            An array of complex signals received by the sensors, with shape
            (num_sensors, num_samples).

        Raises
        ------
        ValueError
            If ``tloss_db`` length does not match the number of frequencies.

        """
        num_sensors = len(sensor_delays_s)
        source_key = self._get_source_key(source, num_sensors)

        # Retrieve or initialize source state
        if source_key not in self._source_states:
            self._source_states[source_key] = {
                "overlap_buffer": None,
                "last_time": None,
                "cumulative_time": 0.0,
            }

        state = self._source_states[source_key]

        # Update cumulative time for phase continuity
        if state["last_time"] is not None:
            time_delta = (source.timestamp - state["last_time"]).total_seconds()
            state["cumulative_time"] += time_delta
        else:
            state["cumulative_time"] = 0.0

        # Extract source parameters
        amplitudes_upa = source.metadata["amplitudes_upa"]
        frequencies_hz = source.metadata["frequencies_hz"]
        phases_rad = source.metadata["phases_rad"]

        # Handle per-frequency or single transmission loss
        if isinstance(tloss_db, (np.ndarray, list)):
            tloss_db_array = np.array(tloss_db)
            if len(tloss_db_array) != len(amplitudes_upa):
                raise ValueError(
                    f"Length of tloss_db array ({len(tloss_db_array)}) must match "
                    f"number of frequencies ({len(amplitudes_upa)})"
                )
            received_amplitude_upa = amplitudes_upa * 10 ** (-tloss_db_array / 20.0)
        else:
            received_amplitude_upa = amplitudes_upa * 10 ** (-tloss_db / 20.0)

        # Generate extended signal (2x duration for 50% overlap)
        extended_samples = 2 * self.num_samples
        time_array_s = (
            np.arange(extended_samples) / self.sampling_rate_hz
            + state["cumulative_time"]
        )

        # --- Use NumPy broadcasting to perform calculations efficiently ---
        # Reshape arrays to dimensions: (sensors, tonals, samples)
        time_reshaped = time_array_s[np.newaxis, np.newaxis, :]
        delays_reshaped = sensor_delays_s[:, np.newaxis, np.newaxis]
        freq_reshaped = frequencies_hz[np.newaxis, :, np.newaxis]
        phase_reshaped = phases_rad[np.newaxis, :, np.newaxis]

        # Calculate the instantaneous phase
        # Shape = (num_sensors, num_tonals, extended_samples)
        total_phase = (
            2
            * np.pi
            * freq_reshaped
            * (time_reshaped - propagation_time_s - delays_reshaped)
            + phase_reshaped
        )

        # Create the complex signal components using Euler's formula
        amp_reshaped = received_amplitude_upa[np.newaxis, :, np.newaxis]
        all_tonal_components = amp_reshaped * np.exp(1j * total_phase)

        # Sum the components along the tonal axis (axis=1)
        extended_signal = np.sum(all_tonal_components, axis=1)

        # Apply synthesis window
        synthesis_window = self._get_synthesis_window()
        windowed_signal = extended_signal * synthesis_window[np.newaxis, :]

        # Split into two halves
        first_half = windowed_signal[:, : self.num_samples]
        second_half = windowed_signal[:, self.num_samples :]

        # Overlap-add with buffered segment from previous timestep
        if state["overlap_buffer"] is not None:
            # Add the buffered overlap to the first half
            output_signal = first_half + state["overlap_buffer"]
        else:
            # First timestep: no buffer exists, use first half as-is
            # This will have windowing artifacts only at the very start
            output_signal = first_half

        # Store second half for next timestep's overlap-add
        state["overlap_buffer"] = second_half.copy()
        state["last_time"] = source.timestamp

        return output_signal

    def reset(self):
        """Clear all stored overlap buffers and state.

        This should be called when starting a new simulation or when
        continuity should be reset.
        """
        self._source_states.clear()
        self._synthesis_window = None


class BroadbandTonalSignal(Signal):
    """Generates broadband tonal signals using STFT-based frequency-domain processing.

    This signal model is designed for continuous broadband acoustic simulation
    using the Short-Time Fourier Transform (STFT) method. It generates a long-
    duration source signal once, computes its STFT representation, and provides
    frequency-domain data for propagation models to apply transfer functions.

    Unlike time-domain signal models, this class does NOT generate signals at
    each timestep. Instead, it provides the STFT of a continuous source signal
    that propagation models can process in the frequency domain with time-varying
    transfer functions.

    Parameters
    ----------
    frame_len : int, optional
        STFT frame length in samples (power of 2 recommended). Default is 1024.
    hop_factor : int, optional
        Hop factor, where hop size = frame_len // hop_factor. Default is 4
        (75% overlap).
    window_type : str, optional
        Window type for STFT (e.g., 'hann', 'hamming', 'blackman'). Default
        is 'hann'.

    """

    frame_len = Property(int, default=1024, doc="STFT frame length in samples")
    hop_factor = Property(
        int, default=4, doc="Hop factor (hop = frame_len // hop_factor)"
    )
    window_type = Property(str, default="hann", doc="Window type for STFT")

    def __init__(self, *args, **kwargs):
        """Initialize broadband signal generator with STFT computation."""
        super().__init__(*args, **kwargs)
        self._stft_cache = None
        self._frequencies = None
        self._hop = None
        self._window = None
        self._source_signal = None

    def _generate_base_signal(self, source) -> np.ndarray:
        """Generate the complete source signal for STFT processing.

        This method creates the full-duration time-domain signal that will
        be transformed to frequency domain for propagation.

        Parameters
        ----------
        source : State
            The source state with tonal parameters in metadata.

        Returns
        -------
        numpy.ndarray
            Complex time-domain signal of shape (num_samples,).

        """
        time_array_s = np.arange(self.num_samples) / self.sampling_rate_hz

        amplitudes_upa = source.metadata["amplitudes_upa"]
        frequencies_hz = source.metadata["frequencies_hz"]
        phases_rad = source.metadata["phases_rad"]

        # Reshape for broadcasting: (tonals, samples)
        freq_reshaped = frequencies_hz[:, np.newaxis]
        phase_reshaped = phases_rad[:, np.newaxis]
        amp_reshaped = amplitudes_upa[:, np.newaxis]

        # Calculate phase for each tonal
        total_phase = (
            2 * np.pi * freq_reshaped * time_array_s[np.newaxis, :] + phase_reshaped
        )

        # Generate complex signal components
        tonal_components = amp_reshaped * np.exp(1j * total_phase)

        # Sum all tonals
        signal = np.sum(tonal_components, axis=0)

        return signal

    def compute_stft(self, source) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
        """Compute and cache the STFT of the source signal.

        This method should be called once per source before simulation begins.
        Results are cached for subsequent access.

        Parameters
        ----------
        source : State
            The source state with tonal parameters in metadata.

        Returns
        -------
        tuple
            A tuple containing:
                - stft (numpy.ndarray): STFT matrix of shape
                  (num_frames, num_freq_bins).
                - frequencies (numpy.ndarray): Frequency array in Hz.
                - hop (int): Hop size in samples.
                - window (numpy.ndarray): Window array used.

        """
        # Check cache
        if self._stft_cache is not None:
            return self._stft_cache, self._frequencies, self._hop, self._window

        # Generate source signal
        self._source_signal = self._generate_base_signal(source)

        # Import STFT utilities
        from .utils import compute_stft

        # Compute STFT
        stft, freq_normalized, hop, window = compute_stft(
            self._source_signal, self.frame_len, self.hop_factor, self.window_type
        )

        # Scale frequencies by sampling rate
        frequencies = freq_normalized * self.sampling_rate_hz

        # Cache results
        self._stft_cache = stft
        self._frequencies = frequencies
        self._hop = hop
        self._window = window

        return stft, frequencies, hop, window

    def get_stft(self) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
        """Get the cached STFT data.

        Returns
        -------
        tuple
            A tuple containing (stft, frequencies, hop, window).

        Raises
        ------
        RuntimeError
            If STFT has not been computed yet.

        """
        if self._stft_cache is None:
            msg = "STFT not computed yet. Call compute_stft() first."
            raise RuntimeError(msg)

        return self._stft_cache, self._frequencies, self._hop, self._window

    def get_source_signal(self) -> np.ndarray:
        """Get the cached source time-domain signal.

        Returns
        -------
        numpy.ndarray
            Time-domain source signal.

        Raises
        ------
        RuntimeError
            If signal has not been generated yet.

        """
        if self._source_signal is None:
            msg = "Source signal not generated yet. Call compute_stft() first."
            raise RuntimeError(msg)

        return self._source_signal

    def generate(
        self, source, sensor_delays_s, tloss_db, propagation_time_s
    ) -> np.ndarray:
        """Not used for broadband processing - use compute_stft() instead.

        This method is required by the base Signal class but is not used
        in broadband STFT-based processing. The simulator should use
        compute_stft() and process in frequency domain.

        Raises
        ------
        NotImplementedError
            Always raises - not applicable for broadband processing.

        """
        msg = (
            "BroadbandTonalSignal does not support per-timestep generation. "
            "Use compute_stft() and process in frequency domain via "
            "BroadbandPassiveSonarArraySimulator."
        )
        raise NotImplementedError(msg)

    def reset(self):
        """Clear cached STFT and source signal data.

        Call this when starting a new simulation with different source parameters.
        """
        self._stft_cache = None
        self._frequencies = None
        self._hop = None
        self._window = None
        self._source_signal = None


class BroadbandShipSignal(Signal):
    """Generates ship signals with broadband tonals and colored noise.

    This signal model combines:
    1. Broadband tonals with finite bandwidth
    2. Wideband colored noise
    3. STFT-based frequency-domain processing for efficient propagation

    Parameters
    ----------
    frame_len : int, optional
        STFT frame length in samples (power of 2 recommended). Default is 1024.
    hop_factor : int, optional
        Hop factor, where hop size = frame_len // hop_factor. Default is 4.
    window_type : str, optional
        Window type for STFT (e.g., 'hann'). Default is 'hann'.
    tonal_bandwidth_hz : float, optional
        Bandwidth of each tonal component in Hz. Creates realistic spectral
        spreading around nominal frequencies. Default is 2.0.
    noise_amplitude_upa : float, optional
        RMS amplitude of background noise in µPa. Set to 0.0 to disable noise.
        Default is 0.0.
    noise_spectral_exponent : float, optional
        Spectral shape exponent for colored noise. -2.0 is pink noise (1/f),
        -1.0 is flicker, 0.0 is white. Default is -2.0.
    noise_freq_range_hz : tuple, optional
        Tuple of (min_freq, max_freq) for noise generation. Default is
        (20.0, 200.0), covering typical machinery noise ranges.
    noise_variance : float, optional
        Variance multiplier applied to all generated white noise before any
        bandlimiting or normalization (default 1.0). This controls the base
        random field variance.
    tonal_noise_is_constant : bool, optional
        If True, reuse the same band-limited tonal noise across calls; phase
        and amplitude are still applied per call. Default is False.
    use_powerlaw_noise : bool, optional
        If True, build broadband noise deterministically from the power-law
        spectrum (no random white-noise seed). Default is False.
    noise_is_constant : bool, optional
        If True, use same noise realization for all signal generations
        (constant scalar over time). If False, generate new random noise each
        time. Default is True.

    Examples
    --------
    Merchant vessel with propeller tonals and machinery noise:

    >>> signal_model = BroadbandShipSignal(
    ...     duration_s=60.0,
    ...     sampling_rate_hz=500.0,
    ...     frame_len=500,
    ...     hop_factor=4,
    ...     tonal_bandwidth_hz=3.0,  # Broader tonals
    ...     noise_amplitude_upa=10**(50/20),  # 50 dB re 1 µPa background
    ...     noise_spectral_exponent=-2.0,  # Pink noise
    ...     noise_freq_range_hz=(30.0, 150.0)
    ... )

    """

    frame_len = Property(int, default=1024, doc="STFT frame length in samples")
    hop_factor = Property(
        int, default=4, doc="Hop factor (hop = frame_len // hop_factor)"
    )
    window_type = Property(str, default="hann", doc="Window type for STFT")
    tonal_bandwidth_hz = Property(
        float, default=2.0, doc="Bandwidth of each tonal component (Hz)"
    )
    noise_amplitude_upa = Property(
        float, default=0.0, doc="RMS amplitude of background noise (µPa)"
    )
    noise_spectral_exponent = Property(
        float, default=-2.0, doc="Spectral shape exponent (-2=pink, 0=white)"
    )
    noise_freq_range_hz = Property(
        tuple, default=(20.0, 200.0), doc="Frequency range for noise (Hz)"
    )
    noise_variance = Property(
        float,
        default=1.0,
        doc=(
            "Variance multiplier for generated white noise before shaping; "
            "std = sqrt(variance)."
        ),
    )
    tonal_noise_is_constant = Property(
        bool,
        default=False,
        doc=(
            "If True, reuse the same band-limited tonal noise across calls; "
            "phase and amplitude are still applied per call."
        ),
    )
    use_powerlaw_noise = Property(
        bool,
        default=False,
        doc=(
            "If True, build broadband noise deterministically from the "
            "power-law spectrum (no random white-noise seed)."
        ),
    )
    noise_is_constant = Property(
        bool,
        default=True,
        doc="If True, use same noise realization across calls; "
        "if False, generate new noise each time",
    )

    def __init__(self, *args, **kwargs):
        """Initialize realistic ship signal generator."""
        super().__init__(*args, **kwargs)
        self._stft_cache = None
        self._frequencies = None
        self._hop = None
        self._window = None
        self._source_signal = None
        self._noise_realization = None  # Cache for constant noise mode
        self._tonal_realizations: list[np.ndarray] | None = None

    def _generate_base_signal(self, source) -> np.ndarray:
        """Generate the complete source signal with broadband tonals and noise.

        This method creates:
        1. Broadband tonals using band-limited white noise modulated by tonal
        frequencies
        2. Wideband colored noise for background machinery/cavitation sounds

        Parameters
        ----------
        source : State
            The source state with tonal parameters in metadata.

        Returns
        -------
        numpy.ndarray
            Complex time-domain signal of shape (num_samples,).

        """
        amplitudes_upa = source.metadata["amplitudes_upa"]
        frequencies_hz = source.metadata["frequencies_hz"]
        phases_rad = source.metadata["phases_rad"]

        # Initialize output signal
        signal = np.zeros(self.num_samples, dtype=np.complex128)

        tonal_cache_available = (
            self.tonal_noise_is_constant
            and self._tonal_realizations is not None
            and len(self._tonal_realizations) == len(frequencies_hz)
        )
        tonal_cache: list[np.ndarray] = []

        # Generate broadband tonals (each tonal has finite bandwidth)
        for idx, (freq, amp, phase) in enumerate(
            zip(frequencies_hz, amplitudes_upa, phases_rad, strict=False)
        ):
            # Create narrow-band noise centered at tonal frequency
            # Bandwidth determined by tonal_bandwidth_hz
            if tonal_cache_available:
                base_noise = self._tonal_realizations[idx]
            else:
                noise_real = np.random.randn(self.num_samples)
                noise_imag = np.random.randn(self.num_samples)
                noise = noise_real + 1j * noise_imag

                # Bandpass filter: Create filter in frequency domain
                freq_bins = np.fft.fftfreq(self.num_samples, 1 / self.sampling_rate_hz)

                # Gaussian bandpass centered at tonal frequency
                # Bandwidth controls the spectral width
                # (sigma = bandwidth / 2sqrt2ln2 ~= bandwidth / 2.355)
                sigma_hz = self.tonal_bandwidth_hz / 2.355
                bandpass_filter = np.exp(-((freq_bins - freq) ** 2) / (2 * sigma_hz**2))
                bandpass_filter += np.exp(
                    -((freq_bins + freq) ** 2) / (2 * sigma_hz**2)
                )  # Negative freq

                # Apply filter in frequency domain
                noise_fft = np.fft.fft(noise)
                filtered_noise_fft = noise_fft * bandpass_filter
                filtered_noise = np.fft.ifft(filtered_noise_fft)

                # Normalize to unit RMS for later amplitude scaling
                rms = np.sqrt(np.mean(np.abs(filtered_noise) ** 2))
                base_noise = filtered_noise if rms == 0 else filtered_noise / rms

                if self.tonal_noise_is_constant:
                    tonal_cache.append(base_noise)

            phase_shift = np.exp(1j * phase)
            tonal_component = amp * base_noise * phase_shift

            signal += tonal_component

        if self.tonal_noise_is_constant and not tonal_cache_available:
            self._tonal_realizations = tonal_cache

        # Add wideband colored noise if amplitude > 0
        if self.noise_amplitude_upa > 0:
            # Check cache for constant mode
            if self.noise_is_constant and self._noise_realization is not None:
                colored_noise = self._noise_realization
            else:
                freq_bins = np.fft.fftfreq(self.num_samples, 1 / self.sampling_rate_hz)
                freq_abs = np.abs(freq_bins)
                freq_abs[freq_abs < 1.0] = 1.0  # Avoid division by zero at DC

                # Spectral envelope (power-law) and bandpass mask
                spectral_shape = freq_abs ** (self.noise_spectral_exponent / 2.0)
                freq_min, freq_max = self.noise_freq_range_hz
                bandpass = np.where(
                    (freq_abs >= freq_min) & (freq_abs <= freq_max),
                    1.0,
                    0.0,
                )
                noise_filter = spectral_shape * bandpass

                if self.use_powerlaw_noise:
                    # Deterministic: use the magnitude spectrum directly
                    colored_noise_fft = noise_filter
                else:
                    # Stochastic: start from white noise then shape
                    noise_std = np.sqrt(self.noise_variance)
                    noise_real = noise_std * np.random.randn(self.num_samples)
                    noise_imag = noise_std * np.random.randn(self.num_samples)
                    white_noise = noise_real + 1j * noise_imag
                    white_noise_fft = np.fft.fft(white_noise)
                    colored_noise_fft = white_noise_fft * noise_filter

                colored_noise = np.fft.ifft(colored_noise_fft)

                # Normalize to desired RMS amplitude
                rms = np.sqrt(np.mean(np.abs(colored_noise) ** 2))
                if rms > 0:
                    colored_noise = colored_noise * (self.noise_amplitude_upa / rms)

                # Cache for constant mode
                if self.noise_is_constant:
                    self._noise_realization = colored_noise.copy()

            signal += colored_noise

        return signal

    def compute_stft(self, source) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
        """Compute and cache the STFT of the source signal.

        This method should be called once per source before simulation begins.
        Results are cached for subsequent access.

        Parameters
        ----------
        source : State
            The source state with tonal parameters in metadata.

        Returns
        -------
        tuple
            A tuple containing:
                - stft (numpy.ndarray): STFT matrix of shape
                  (num_frames, num_freq_bins).
                - frequencies (numpy.ndarray): Frequency array in Hz.
                - hop (int): Hop size in samples.
                - window (numpy.ndarray): Window array used.

        """
        # Check cache
        if self._stft_cache is not None:
            return self._stft_cache, self._frequencies, self._hop, self._window

        # Generate source signal
        self._source_signal = self._generate_base_signal(source)

        # Import STFT utilities
        from .utils import compute_stft

        # Compute STFT
        stft, freq_normalized, hop, window = compute_stft(
            self._source_signal, self.frame_len, self.hop_factor, self.window_type
        )

        # Scale frequencies by sampling rate
        frequencies = freq_normalized * self.sampling_rate_hz

        # Cache results
        self._stft_cache = stft
        self._frequencies = frequencies
        self._hop = hop
        self._window = window

        return stft, frequencies, hop, window

    def get_stft(self) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
        """Get the cached STFT data.

        Returns
        -------
        tuple
            A tuple containing (stft, frequencies, hop, window).

        Raises
        ------
        RuntimeError
            If STFT has not been computed yet.

        """
        if self._stft_cache is None:
            msg = "STFT not computed yet. Call compute_stft() first."
            raise RuntimeError(msg)

        return self._stft_cache, self._frequencies, self._hop, self._window

    def get_source_signal(self) -> np.ndarray:
        """Get the cached source time-domain signal.

        Returns
        -------
        numpy.ndarray
            Time-domain source signal.

        Raises
        ------
        RuntimeError
            If signal has not been generated yet.

        """
        if self._source_signal is None:
            msg = "Source signal not generated yet. Call compute_stft() first."
            raise RuntimeError(msg)

        return self._source_signal

    def generate(
        self, source, sensor_delays_s, tloss_db, propagation_time_s
    ) -> np.ndarray:
        """Not used for broadband processing - use compute_stft() instead.

        This method is required by the base Signal class but is not used
        in broadband STFT-based processing. The simulator should use
        compute_stft() and process in frequency domain.

        Raises
        ------
        NotImplementedError
            Always raises - not applicable for broadband processing.

        """
        msg = (
            "BroadbandShipSignal does not support per-timestep generation. "
            "Use compute_stft() and process in frequency domain via "
            "BroadbandPassiveSonarArraySimulator."
        )
        raise NotImplementedError(msg)

    def reset(self):
        """Clear cached STFT and source signal data.

        Call this when starting a new simulation with different source parameters.
        """
        self._stft_cache = None
        self._frequencies = None
        self._hop = None
        self._window = None
        self._source_signal = None
        self._noise_realization = None
        self._tonal_realizations = None
