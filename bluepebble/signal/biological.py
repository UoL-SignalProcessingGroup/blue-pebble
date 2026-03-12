"""Models for biological acoustic signals."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, TypeAlias, TypedDict

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import interpolate, signal
from scipy.stats import levy_stable
from stonesoup.base import Property

from ..models.environment.sound_speed_profile import SoundSpeedProfile
from .base import Signal
from .effects import Effect

if TYPE_CHECKING:
    from stonesoup.types.state import State

FloatArray: TypeAlias = NDArray[np.float64]
ComplexArray: TypeAlias = NDArray[np.complex128]


class CallEvent(TypedDict):
    """Dictionary structure for a generated whale call event."""

    start_time: float
    duration: float

    contour_freqs: list[float]
    amplitude: float


def _get_source_metadata(source: State) -> Mapping[str, object]:
    """Return validated source metadata mapping."""
    metadata = getattr(source, "metadata", None)
    if not isinstance(metadata, Mapping):
        msg = "Source state metadata must be mapping-like"
        raise ValueError(msg)
    return metadata


def _get_source_amplitude_upa(source: State) -> float:
    """Read scalar source amplitude metadata."""
    metadata = _get_source_metadata(source)
    if "amplitude_upa" not in metadata:
        msg = "Source metadata missing required key: amplitude_upa"
        raise ValueError(msg)
    return float(metadata["amplitude_upa"])


def _get_source_position(source: State) -> FloatArray:
    """Extract source position vector using metadata mapping."""
    metadata = _get_source_metadata(source)
    position_mapping = metadata.get("position_mapping")
    if position_mapping is None:
        msg = "Source metadata missing required key: position_mapping"
        raise ValueError(msg)

    mapping_array = np.asarray(position_mapping, dtype=int)
    if mapping_array.ndim != 1 or mapping_array.size == 0:
        msg = "Source position_mapping must be one-dimensional and non-empty"
        raise ValueError(msg)

    return np.asarray(source.state_vector[mapping_array.tolist()], dtype=float)


def _design_butterworth_filter(
    order: int,
    cutoff_hz: ArrayLike | float,
    btype: str,
    sampling_rate_hz: float,
) -> tuple[FloatArray, FloatArray]:
    """Design a Butterworth filter with explicit ``None`` guard for static typing."""
    coeffs = signal.butter(order, cutoff_hz, btype=btype, fs=sampling_rate_hz)
    if coeffs is None:
        msg = "scipy.signal.butter returned no filter coefficients"
        raise RuntimeError(msg)
    b, a = coeffs
    return np.asarray(b, dtype=np.float64), np.asarray(a, dtype=np.float64)


def _get_snap_rate_from_temp(temperature_celsius: float, slope: float, intercept: float) -> float:
    """Calculate snap rate per second from water temperature.

    This function uses a linear regression model derived from the annual dataset in Bohnenstiehl
    et al. (2016), Figure 5b. It captures the strong seasonal trend of higher snap rates in warmer
    months and lower rates in colder months.

    Parameters
    ----------
    temperature_celsius  : float
        Water temperature in Celsius.
    slope  : float
        Slope of the linear regression model.
    intercept  : float
        Intercept of the linear regression model.

    Returns
    -------
    float
        Snap rate per second. Returns 0 if the calculated rate is negative.

    """
    # Snaps per minute -> snaps per second
    return max(0, (slope * temperature_celsius) + intercept) / 60.0


class PointSourceSnappingShrimpSignal(Signal):
    """Generates a point-source signal representing a colony of snapping shrimp.

    This model simulates the sound of a snapping shrimp colony using a non-homogeneous Poisson
    process. The snap rate is dependent on water temperature and can be modulated by diurnal
    (daily) and tidal cycles.

    By treating the colony as a single point source, the generated signal is coherent across the
    array, allowing for localisation and tracking.
    """

    TEMP_TO_RATE_SLOPE = 137.0
    TEMP_TO_RATE_INTERCEPT = -685.0

    # --- Temporal Distribution Parameters ---
    temperature_celsius: float = Property(float, doc="Water temperature in Celsius.")
    start_time_hours: float = Property(
        float, default=0.0, doc="Simulation start time in hours from midnight (0-24)."
    )
    diurnal_amplitude: float = Property(
        float, default=0.0, doc="Amplitude of diurnal snap rate modulation (0-1)."
    )
    diurnal_phase_hours: float = Property(
        float, default=0.0, doc="Phase offset of diurnal cycle in hours."
    )
    tidal_amplitude: float = Property(
        float, default=0.0, doc="Amplitude of tidal snap rate modulation (0-1)."
    )
    tidal_phase_hours: float = Property(
        float, default=0.0, doc="Phase offset of tidal cycle in hours."
    )

    # --- Snap Amplitude Distribution Parameters ---
    alpha: float = Property(
        float,
        default=1.5,
        doc="Alpha parameter for the Symmetric Alpha-Stable distribution.",
    )

    # --- Individual Snap Waveform Parameters ---
    delay_duration: float = Property(float, default=0.0006, doc="Pre-snap delay in seconds.")
    onset_duration: float = Property(float, default=0.0001, doc="Snap onset duration in seconds.")
    snap_duration: float = Property(float, default=0.0014, doc="Snap impulse duration in seconds.")
    onset_level: float = Property(float, default=0.15, doc="Relative amplitude of the onset.")
    onset_freq: float = Property(
        float, default=2500, doc="Frequency of the onset sine wave in Hz."
    )
    snap_decay: float = Property(float, default=1000, doc="Exponential decay rate for the snap.")
    low_cutoff_hz: float = Property(float, default=2000, doc="Bandpass filter low cutoff in Hz.")
    high_cutoff_hz: float = Property(
        float, default=15000, doc="Bandpass filter high cutoff in Hz."
    )

    # --- Post-processing Effects ---
    effects: list[Effect] | None = Property(
        list[Effect], default=None, doc="List of effects to apply to the signal."
    )

    def _create_snap_template(self) -> FloatArray:
        """Generate the prototypical waveform for a single shrimp snap.

        Returns
        -------
        FloatArray
            Waveform template for a single shrimp snap.

        """
        delay_samps = int(self.delay_duration * self.sampling_rate_hz)
        onset_samps = int(self.onset_duration * self.sampling_rate_hz)
        snap_samps = int(self.snap_duration * self.sampling_rate_hz)

        delay = np.zeros(delay_samps)

        t_onset = np.linspace(0, self.onset_duration, onset_samps, endpoint=False)
        onset_sine = np.sin(2 * np.pi * self.onset_freq * t_onset)
        onset_wave = onset_sine * np.linspace(0, 1, onset_samps) * self.onset_level

        t_snap = np.linspace(0, self.snap_duration, snap_samps, endpoint=False)
        snap_noise = np.random.uniform(-1, 1, len(t_snap))
        snap_envelope = np.exp(-self.snap_decay * t_snap)
        snap_impulse = snap_noise * snap_envelope

        raw_waveform = np.concatenate([delay, onset_wave, snap_impulse])
        b, a = _design_butterworth_filter(
            order=4,
            cutoff_hz=[self.low_cutoff_hz, self.high_cutoff_hz],
            btype="bandpass",
            sampling_rate_hz=self.sampling_rate_hz,
        )
        return signal.filtfilt(b, a, raw_waveform)

    def _rate_function(self, t: FloatArray, base_rate: float) -> FloatArray:
        """Calculate the time-varying snap rate.

        Parameters
        ----------
        t : FloatArray
            Time vector in seconds.
        base_rate : float
            Base snap rate in snaps per second.

        Returns
        -------
        FloatArray
            Time-varying snap rate.

        """
        diurnal_mod = self.diurnal_amplitude * np.sin(
            2 * np.pi * t / (24 * 3600) + (self.diurnal_phase_hours * np.pi / 12.0)
        )
        tidal_mod = self.tidal_amplitude * np.sin(
            2 * np.pi * t / (12.42 * 3600) + (self.tidal_phase_hours * np.pi / 6.21)
        )
        return np.maximum(0, base_rate * (1 + diurnal_mod + tidal_mod))

    def _generate_base_signal(self, source: State) -> FloatArray:
        """Generate the base snapping shrimp signal for a single point source.

        This creates a 1-D time series of snapping events for the point colony. Event amplitudes
        are sampled from a Symmetric Alpha-Stable distribution and snaps that would fall outside
        the buffer are ignored.

        Parameters
        ----------
        source : State
            Source state object providing metadata (e.g. amplitude_upa).

        Returns
        -------
        FloatArray
            One-dimensional time-domain signal for the configured duration.

        """
        base_lambda_rate = _get_snap_rate_from_temp(
            self.temperature_celsius,
            self.TEMP_TO_RATE_SLOPE,
            self.TEMP_TO_RATE_INTERCEPT,
        )
        snap_template = self._create_snap_template()

        signal_buffer = np.zeros(self.num_samples)
        start_time_s = self.start_time_hours * 3600

        lambda_max = base_lambda_rate * (1 + self.diurnal_amplitude + self.tidal_amplitude)
        if lambda_max <= 0:
            return signal_buffer

        num_candidates = int(self.duration_s * lambda_max * 1.2)
        if num_candidates == 0:
            return signal_buffer

        intervals = np.random.exponential(1.0 / lambda_max, num_candidates)
        candidate_times = start_time_s + np.cumsum(intervals)
        candidate_times = candidate_times[candidate_times < start_time_s + self.duration_s]
        if len(candidate_times) == 0:
            return signal_buffer

        actual_rates = self._rate_function(candidate_times, base_lambda_rate)
        accepted_mask = np.random.uniform(0, 1, len(candidate_times)) < (actual_rates / lambda_max)
        snap_times = candidate_times[accepted_mask]
        if len(snap_times) == 0:
            return signal_buffer

        snap_len = len(snap_template)

        # Snap placement algorithm:

        # 1. Calculate all start indices
        start_indices = ((snap_times - start_time_s) * self.sampling_rate_hz).astype(int)

        # 2. Filter out snaps that would be placed out of bounds
        valid_mask = (start_indices >= 0) & (start_indices + snap_len < self.num_samples)
        start_indices = start_indices[valid_mask]
        num_snaps = len(start_indices)

        if num_snaps == 0:
            return signal_buffer

        # 3. Generate amplitudes only for valid snaps
        scale_upa = _get_source_amplitude_upa(source)
        amplitudes = np.asarray(
            levy_stable.rvs(self.alpha, 0, scale=scale_upa, size=num_snaps),
            dtype=np.float64,
        )

        # 4. Create an index array for placing snaps
        # This creates a 2D array where each row corresponds to the indices
        # for one snap in the final signal_buffer.
        snap_indices = np.arange(snap_len) + start_indices[:, np.newaxis]

        # 5. Scale templates and place them in the buffer
        # We create a scaled version of the template for each snap and then
        # use np.add.at to add these values to the correct locations.
        scaled_snaps = snap_template[np.newaxis, :] * amplitudes[:, np.newaxis]
        np.add.at(signal_buffer, snap_indices, scaled_snaps)

        return signal_buffer

    def generate(
        self,
        source: State,
        sensor_delays_s: ArrayLike,
        tloss_db: ArrayLike | float,
        propagation_time_s: float,
    ) -> ComplexArray:
        """Generate shrimp snaps, propagate them, and apply effects.

        The point-source model generates a base 1-D signal, applies transmission loss and
        per-sensor phase shifts via FFT-based propagation (handled by the base `Signal.generate`
        implementation), and then applies any post-processing effects.

        Parameters
        ----------
        source : State
            The source state object used to read metadata.
        sensor_delays_s : ArrayLike
            1-D array of per-sensor delays in seconds.
        tloss_db : ArrayLike | float
            Transmission loss to the array origin (dB).
        propagation_time_s : float
            Propagation time from source to origin (s).

        Returns
        -------
        ComplexArray
            Complex signal matrix with shape ``(num_sensors, num_samples)``.

        """
        # Call the base class generate method to handle propagation
        signals = super().generate(source, sensor_delays_s, tloss_db, propagation_time_s)
        # Apply post-processing effects if any are specified. Validate that
        # each item in `effects` is an `Effect` instance before applying.
        if self.effects:
            for effect in self.effects:
                if isinstance(effect, Effect):
                    signals = effect.apply(signals, self.sampling_rate_hz)

        return signals.astype(np.complex128)


class DiffuseSnappingShrimpSignal(Signal):
    """Generates a diffuse field signal representing a colony of snapping shrimp.

    This model simulates the sound of a snapping shrimp colony as a diffuse
    field composed of many incoherent point sources distributed over an area.
    This creates a realistic, non-localizable ambient noise signal.
    """

    # --- Model Constants ---
    TEMP_TO_RATE_SLOPE = 137.0
    TEMP_TO_RATE_INTERCEPT = -685.0

    # --- Diffuse Field Parameters ---
    num_diffuse_sources: int = Property(
        int, default=100, doc="Number of incoherent point sources in the diffuse field."
    )
    colony_radius_m: float = Property(
        float,
        default=50.0,
        doc="Radius of the circular area over which sources are distributed.",
    )

    # --- Temporal and Waveform Properties ---
    temperature_celsius: float = Property(float, doc="Water temperature in Celsius.")
    start_time_hours: float = Property(
        float, default=0.0, doc="Simulation start time in hours from midnight (0-24)."
    )
    diurnal_amplitude: float = Property(
        float, default=0.0, doc="Amplitude of diurnal snap rate modulation (0-1)."
    )
    diurnal_phase_hours: float = Property(
        float, default=0.0, doc="Phase offset of diurnal cycle in hours."
    )
    tidal_amplitude: float = Property(
        float, default=0.0, doc="Amplitude of tidal snap rate modulation (0-1)."
    )
    tidal_phase_hours: float = Property(
        float, default=0.0, doc="Phase offset of tidal cycle in hours."
    )
    alpha: float = Property(
        float,
        default=1.5,
        doc="Alpha parameter for the Symmetric Alpha-Stable distribution.",
    )
    delay_duration: float = Property(float, default=0.0006, doc="Pre-snap delay in seconds.")
    onset_duration: float = Property(float, default=0.0001, doc="Snap onset duration in seconds.")
    snap_duration: float = Property(float, default=0.0014, doc="Snap impulse duration in seconds.")
    onset_level: float = Property(float, default=0.15, doc="Relative amplitude of the onset.")
    onset_freq: float = Property(
        float, default=2500, doc="Frequency of the onset sine wave in Hz."
    )
    snap_decay: float = Property(float, default=1000, doc="Exponential decay rate for the snap.")
    low_cutoff_hz: float = Property(float, default=2000, doc="Bandpass filter low cutoff in Hz.")
    high_cutoff_hz: float = Property(
        float, default=15000, doc="Bandpass filter high cutoff in Hz."
    )
    ssp: SoundSpeedProfile = Property(SoundSpeedProfile, doc="Sound speed profile object.")
    effects: list[Effect] | None = Property(
        list[Effect], default=None, doc="List of effects to apply to the signal."
    )

    def _create_snap_template(self) -> FloatArray:
        """Generate the prototypical waveform for a single shrimp snap."""
        delay_samps = int(self.delay_duration * self.sampling_rate_hz)
        onset_samps = int(self.onset_duration * self.sampling_rate_hz)
        snap_samps = int(self.snap_duration * self.sampling_rate_hz)
        delay = np.zeros(delay_samps)
        t_onset = np.linspace(0, self.onset_duration, onset_samps, endpoint=False)
        onset_sine = np.sin(2 * np.pi * self.onset_freq * t_onset)
        onset_wave = onset_sine * np.linspace(0, 1, onset_samps) * self.onset_level
        t_snap = np.linspace(0, self.snap_duration, snap_samps, endpoint=False)
        snap_noise = np.random.uniform(-1, 1, len(t_snap))
        snap_envelope = np.exp(-self.snap_decay * t_snap)
        snap_impulse = snap_noise * snap_envelope
        raw_waveform = np.concatenate([delay, onset_wave, snap_impulse])
        b, a = _design_butterworth_filter(
            order=4,
            cutoff_hz=[self.low_cutoff_hz, self.high_cutoff_hz],
            btype="bandpass",
            sampling_rate_hz=self.sampling_rate_hz,
        )
        return signal.filtfilt(b, a, raw_waveform)

    def _rate_function(self, t: FloatArray, base_rate: float) -> FloatArray:
        """Calculate the time-varying snap rate.

        Parameters
        ----------
        t : FloatArray
            Time vector in seconds.
        base_rate : float
            Base snap rate.

        Returns
        -------
        FloatArray
            Time-varying snap rate.

        """
        diurnal_mod = self.diurnal_amplitude * np.sin(
            2 * np.pi * t / (24 * 3600) + (self.diurnal_phase_hours * np.pi / 12.0)
        )
        tidal_mod = self.tidal_amplitude * np.sin(
            2 * np.pi * t / (12.42 * 3600) + (self.tidal_phase_hours * np.pi / 6.21)
        )
        return np.maximum(0, base_rate * (1 + diurnal_mod + tidal_mod))

    def _generate_base_signal(self, source: State, lambda_rate_fraction: float) -> FloatArray:
        """Generate a sparse base signal for a single sub-source.

        This is used by the diffuse-field model to simulate one incoherent contributor in the
        colony.

        Parameters
        ----------
        source : State
            Source state object providing metadata (e.g. amplitude_upa).
        lambda_rate_fraction : float
            Fraction of the total snap rate to assign to this sub-source
            (e.g. 1/num_diffuse_sources).

        Returns
        -------
        FloatArray
            The generated base signal.

        """
        base_lambda_rate = _get_snap_rate_from_temp(
            self.temperature_celsius,
            self.TEMP_TO_RATE_SLOPE,
            self.TEMP_TO_RATE_INTERCEPT,
        )
        # Scale the snap rate for this sub-source
        sparse_lambda_rate = base_lambda_rate * lambda_rate_fraction

        snap_template = self._create_snap_template()
        signal_buffer = np.zeros(self.num_samples)
        start_time_s = self.start_time_hours * 3600
        lambda_max = sparse_lambda_rate * (1 + self.diurnal_amplitude + self.tidal_amplitude)
        if lambda_max <= 0:
            return signal_buffer
        num_candidates = int(self.duration_s * lambda_max * 1.5) + 1  # Ensure at least 1
        if num_candidates == 0:
            return signal_buffer
        intervals = np.random.exponential(1.0 / lambda_max, num_candidates)
        candidate_times = start_time_s + np.cumsum(intervals)
        candidate_times = candidate_times[candidate_times < start_time_s + self.duration_s]
        if len(candidate_times) == 0:
            return signal_buffer
        actual_rates = self._rate_function(candidate_times, sparse_lambda_rate)
        accepted_mask = np.random.uniform(0, 1, len(candidate_times)) < (actual_rates / lambda_max)
        snap_times = candidate_times[accepted_mask]
        if len(snap_times) == 0:
            return signal_buffer
        snap_len = len(snap_template)
        start_indices = ((snap_times - start_time_s) * self.sampling_rate_hz).astype(int)
        valid_mask = (start_indices >= 0) & (start_indices + snap_len < self.num_samples)
        start_indices = start_indices[valid_mask]
        num_snaps = len(start_indices)
        if num_snaps == 0:
            return signal_buffer

        scale_upa = _get_source_amplitude_upa(source)
        amplitudes = np.asarray(
            levy_stable.rvs(self.alpha, 0, scale=scale_upa, size=num_snaps),
            dtype=np.float64,
        )

        snap_indices = np.arange(snap_len) + start_indices[:, np.newaxis]
        scaled_snaps = snap_template[np.newaxis, :] * amplitudes[:, np.newaxis]
        np.add.at(signal_buffer, snap_indices, scaled_snaps)
        return signal_buffer

    def generate(
        self,
        source: State,
        sensor_delays_s: ArrayLike,
        tloss_db: ArrayLike | float,
        propagation_time_s: float,
    ) -> ComplexArray:
        """Generate a diffuse field by summing many incoherent point sources.

        Parameters
        ----------
        source
            The source state object used to read metadata.
        sensor_delays_s : ArrayLike
            1-D array of per-sensor delays in seconds.
        tloss_db : ArrayLike | float
            Transmission loss to the array origin (dB).
        propagation_time_s : float
            Propagation time from source to origin (s).

        Returns
        -------
        ComplexArray
            Complex signal for each sensor with shape
            `(num_sensors, num_samples)` and dtype `np.complex128`.

        """
        sensor_delays = np.asarray(sensor_delays_s, dtype=float)
        final_signals = np.zeros((len(sensor_delays), self.num_samples), dtype=np.complex128)
        fft_freqs_hz = np.fft.fftfreq(self.num_samples, 1 / self.sampling_rate_hz)

        source_position = _get_source_position(source)
        speed_of_sound_mps = self.ssp.calculate(source_position[2])

        for _ in range(self.num_diffuse_sources):
            # 1. Generate a sparse base signal for one sub-source
            # Each sub-source contributes a fraction of the total snap rate
            lambda_fraction = 1.0 / self.num_diffuse_sources
            base_signal = self._generate_base_signal(source, lambda_fraction)

            # 2. Randomly perturb the propagation time to simulate a different
            # location inside the colony. Sample uniformly inside a circle to
            # avoid clustering at the center.
            r = self.colony_radius_m * np.sqrt(np.random.uniform(0.0, 1.0))
            # theta = np.random.uniform(0.0, 2.0 * np.pi)  # not used
            # radial offset (m) used as a simple proxy for additional path
            # length; for diffuse ambient this approximation is sufficient.
            radial_offset_m = r
            time_perturbation_s = radial_offset_m / speed_of_sound_mps
            perturbed_prop_time_s = propagation_time_s + time_perturbation_s

            # 3. Propagate this individual signal
            tloss_array = np.asarray(tloss_db, dtype=float)
            amplitude_scaling = 10 ** (-tloss_array / 20.0)
            base_signal *= amplitude_scaling
            base_signal_fft = np.fft.fft(base_signal)

            total_delays_s = perturbed_prop_time_s + sensor_delays
            phase_shifts = np.exp(
                -1j * 2 * np.pi * total_delays_s[:, np.newaxis] * fft_freqs_hz[np.newaxis, :]
            )
            signals_fft = base_signal_fft[np.newaxis, :] * phase_shifts
            sub_source_signals = np.fft.ifft(signals_fft, axis=1)

            # 4. Add to the final signal field
            final_signals += sub_source_signals

        # 5. Apply post-processing effects if any are specified
        if self.effects:
            for effect in self.effects:
                if isinstance(effect, Effect):
                    final_signals = effect.apply(final_signals, self.sampling_rate_hz)

        return final_signals.astype(np.complex128)


class WhaleCallSignal(Signal):
    """Generates a sequence of whale calls with realistic variation.

    This model simulates whale calls with various parameters, including temporal distribution,
    harmonic structure, and timbre. It supports both random call sequences and structured songs.
    """

    # --- Call Temporal Distribution Parameters ---
    mean_call_interval_s: float = Property(
        float, default=10.0, doc="Mean interval between calls in seconds."
    )
    interval_jitter_s: float = Property(
        float, default=2.0, doc="Standard deviation of call interval jitter in seconds."
    )

    # --- Harmonic Structure Parameters ---
    min_harmonics: int = Property(int, default=8, doc="Minimum number of harmonics per call.")
    max_harmonics: int = Property(int, default=40, doc="Maximum number of harmonics per call.")
    harmonic_decay_db: float = Property(
        float, default=6.0, doc="Amplitude decay per harmonic in dB."
    )

    # --- Individual Call Waveform Parameters ---
    call_duration_s: float = Property(float, default=2.0, doc="Duration of each call in seconds.")
    duration_jitter_s: float = Property(
        float, default=0.2, doc="Standard deviation of call duration jitter in seconds."
    )
    start_freq_hz: float = Property(
        float, default=1000, doc="Starting frequency of the call in Hz."
    )
    start_freq_jitter_hz: float = Property(
        float, default=100, doc="Standard deviation of starting frequency jitter in Hz."
    )
    end_freq_hz: float = Property(float, default=5000, doc="Ending frequency of the call in Hz.")
    end_freq_jitter_hz: float = Property(
        float, default=500, doc="Standard deviation of ending frequency jitter in Hz."
    )
    # --- Dynamic Contour Parameters ---
    num_contour_points: int = Property(
        int,
        default=2,
        doc="Number of points defining the frequency contour. 2 is a simple sweep.",
    )
    contour_variability_hz: float = Property(
        float,
        default=50.0,
        doc="Max frequency change between contour points, creating variability.",
    )
    sweep_method: str = Property(
        str,
        default="logarithmic",
        doc=("Frequency sweep method ('linear', 'quadratic', 'logarithmic', 'hyperbolic')."),
    )
    vibrato_rate_hz: float = Property(
        float, default=0.0, doc="Speed of the vibrato in oscillations per second (Hz)."
    )
    vibrato_depth_hz: float = Property(float, default=0.0, doc="Intensity of the vibrato in Hz.")
    low_cutoff_hz: float = Property(
        float, default=50, doc="Bandpass filter low cutoff frequency in Hz."
    )
    high_cutoff_hz: float = Property(
        float,
        default=1500.0,
        doc="High cutoff frequency for the bandpass filter in Hz.",
    )
    envelope_taper_ratio: float = Property(
        default=0.25,
        doc="Ratio of the call duration to taper for a smooth amplitude envelope.",
    )

    # --- Biphonation Parameters ---
    add_biphonation: bool = Property(
        bool,
        default=False,
        doc="If True, adds a second, non-harmonic voice to the call.",
    )
    biphonic_freq_ratio: float = Property(
        float,
        default=1.5,
        doc=(
            "Frequency ratio of the biphonic voice to the fundamental "
            "(e.g., 1.5 for a perfect fifth)."
        ),
    )
    biphonic_jitter_ratio: float = Property(
        float,
        default=0.1,
        doc="Amount of random variation in the biphonic frequency ratio.",
    )
    biphonic_amplitude_ratio: float = Property(
        float,
        default=0.5,
        doc="Amplitude of the biphonic voice relative to the fundamental (0-1).",
    )

    # --- Sub-harmonic Parameters ---
    sub_harmonic_ratios: list[float] | None = Property(
        list,
        default=None,
        doc="List of frequency ratios for sub-harmonics (e.g., [0.5, 0.25]).",
    )
    sub_harmonic_amplitude_ratio: float = Property(
        float,
        default=0.3,
        doc="Amplitude of the sub-harmonics relative to the fundamental (0-1).",
    )

    # --- Timbre / Texture Parameters ---
    add_breathy_noise: bool = Property(
        bool,
        default=False,
        doc="If True, adds a 'breathy' or 'noisy' texture to the call.",
    )
    breathy_noise_amount: float = Property(
        float,
        default=0.1,
        doc="Mix amount of breathy noise (0-1). Higher is more noisy.",
    )
    breathy_noise_lp_cutoff_hz: float = Property(
        float,
        default=1500.0,
        doc="Low-pass cutoff for the breathy noise, controlling its 'color'.",
    )

    # --- Post Processing Parameters ---
    effects: list[Effect] | None = Property(
        list[Effect],
        default=None,
        doc="List of audio effects to apply post-generation.",
    )

    # --- Song Structure Parameters ---
    song_structure_enabled: bool = Property(
        bool,
        default=False,
        doc="If True, generates calls based on a defined song structure.",
    )
    song_themes: list[list[float]] | None = Property(
        list,
        default=None,
        doc="A list of themes, where each theme is a list of frequency offsets (Hz).",
    )
    song_phrases: list[list[int]] | None = Property(
        list,
        default=None,
        doc="A list of phrases, where each phrase is a list of theme indices.",
    )
    theme_base_freq_hz: float = Property(
        float,
        default=150.0,
        doc="The base frequency from which themes are generated.",
    )
    theme_freq_jitter_hz: float = Property(
        float,
        default=10.0,
        doc="Jitter applied to each point in a theme's contour.",
    )
    theme_duration_s: float = Property(
        float,
        default=2.0,
        doc="The base duration for a call generated from a theme.",
    )

    def _create_call_template(self, duration: float, contour_freqs: list[float]) -> FloatArray:
        """Generate the prototypical waveform for a single whale call.

        Parameters
        ----------
        duration  : float
            Duration of the call in seconds.
        contour_freqs  : list[float]
            Frequency contour of the call.

        Returns
        -------
        FloatArray
            The waveform of a single whale call.

        """
        # --- 1. Create time vector and frequency contour ---
        num_samples_call = int(duration * self.sampling_rate_hz)
        if num_samples_call == 0:
            return np.array([])
        t = np.linspace(0, duration, num_samples_call, endpoint=False)

        # Create time points for interpolation, matching the frequency points
        contour_times = np.linspace(0, duration, len(contour_freqs))

        # Interpolate to create a smooth, dynamic frequency contour
        # Using cubic interpolation for smoother curves
        if len(contour_freqs) >= 4:
            interp_func = interpolate.interp1d(contour_times, contour_freqs, kind="cubic")
        else:
            interp_func = interpolate.interp1d(contour_times, contour_freqs, kind="linear")
        f0_chirp_freq = interp_func(t)

        # --- Add vibrato for a more organic, controlled warble ---
        if self.vibrato_rate_hz > 0 and self.vibrato_depth_hz > 0:
            vibrato_modulation = self.vibrato_depth_hz * np.sin(
                2 * np.pi * self.vibrato_rate_hz * t
            )
            f0_chirp_freq += vibrato_modulation

        call_template = np.sin(2 * np.pi * np.cumsum(f0_chirp_freq) / self.sampling_rate_hz)

        # --- 3. Generate and add harmonics ---
        # Add harmonic variability
        actual_num_harmonics = np.random.randint(self.min_harmonics, self.max_harmonics + 1)

        # Loop through the fundamental (i=1) and all its harmonics
        for i in range(1, actual_num_harmonics + 1):
            # Determine amplitude for the current harmonic
            if i == 1:
                harmonic_amplitude = 1.0  # Fundamental has full amplitude
            else:
                harmonic_amplitude = 10 ** (-self.harmonic_decay_db * (i - 1) / 20.0)

            # --- Generate the main harmonic itself ---
            current_harmonic_freq = f0_chirp_freq * i
            harmonic_phase = 2 * np.pi * np.cumsum(current_harmonic_freq) / self.sampling_rate_hz
            harmonic_signal = harmonic_amplitude * np.sin(harmonic_phase)
            call_template += harmonic_signal

            # --- Generate sub-harmonics FOR THIS HARMONIC ---
            if self.sub_harmonic_ratios:
                for ratio in self.sub_harmonic_ratios:
                    if 0 < ratio < 1:  # Ensure it's a sub-harmonic
                        sub_harmonic_freq = current_harmonic_freq * ratio
                        sub_harmonic_phase = (
                            2 * np.pi * np.cumsum(sub_harmonic_freq) / self.sampling_rate_hz
                        )
                        # Amplitude is relative to the parent harmonic's amplitude
                        sub_harmonic_amp = harmonic_amplitude * self.sub_harmonic_amplitude_ratio
                        sub_harmonic_signal = sub_harmonic_amp * np.sin(sub_harmonic_phase)
                        call_template += sub_harmonic_signal

        # --- 4. Add Biphonation (a second, independent voice) ---
        if self.add_biphonation:
            # Determine the frequency for the biphonic voice for this specific call
            jitter = np.random.uniform(-self.biphonic_jitter_ratio, self.biphonic_jitter_ratio)
            actual_ratio = self.biphonic_freq_ratio + jitter

            biphonic_contour = [freq * actual_ratio for freq in contour_freqs]
            if len(biphonic_contour) >= 4:
                interp_func_biphonic = interpolate.interp1d(
                    contour_times, biphonic_contour, kind="cubic"
                )
                biphonic_chirp_freq = interp_func_biphonic(t)
            else:
                interp_func_biphonic = interpolate.interp1d(
                    contour_times, biphonic_contour, kind="linear"
                )
                biphonic_chirp_freq = interp_func_biphonic(t)

            biphonic_signal = self.biphonic_amplitude_ratio * np.sin(
                2 * np.pi * np.cumsum(biphonic_chirp_freq) / self.sampling_rate_hz
            )
            call_template += biphonic_signal

        # --- 5. Add Breathy Noise for a more realistic timbre ---
        if self.add_breathy_noise and self.breathy_noise_amount > 0:
            # Generate white noise
            noise = np.random.randn(len(call_template))

            # Create a low-pass filter for the noise to make it 'breathy'
            b_noise, a_noise = _design_butterworth_filter(
                order=2,
                cutoff_hz=self.breathy_noise_lp_cutoff_hz,
                btype="low",
                sampling_rate_hz=self.sampling_rate_hz,
            )
            filtered_noise = signal.filtfilt(b_noise, a_noise, noise)

            # Normalise noise and mix it with the tonal signal
            filtered_noise /= np.max(np.abs(filtered_noise))
            call_template = (
                1 - self.breathy_noise_amount
            ) * call_template + self.breathy_noise_amount * filtered_noise

        # --- 6. Apply bandpass filter ---
        b, a = _design_butterworth_filter(
            order=4,
            cutoff_hz=[self.low_cutoff_hz, self.high_cutoff_hz],
            btype="bandpass",
            sampling_rate_hz=self.sampling_rate_hz,
        )
        filtered_call = signal.filtfilt(b, a, call_template)

        # --- 7. Apply amplitude envelope for natural attack/decay ---
        if self.envelope_taper_ratio > 0:
            num_samples = len(filtered_call)
            window = signal.get_window(
                ("tukey", self.envelope_taper_ratio),
                num_samples,
                fftbins=False,
            )
            filtered_call *= window

        return filtered_call

    def _generate_random_call_sequence(self, source: State) -> list[CallEvent]:
        """Generate a sequence of random, unstructured whale calls.

        Parameters
        ----------
        source : State
            Source state object providing metadata (e.g. amplitude_upa).

        Returns
        -------
        list[CallEvent]
            A list of call events, each containing start time,
        duration, contour frequencies, and amplitude.

        """
        potential_events = []
        current_time_s = np.random.uniform(0, self.mean_call_interval_s)

        # 1. Generate all potential call events
        while current_time_s < self.duration_s:
            duration = self.call_duration_s + np.random.uniform(
                -self.duration_jitter_s, self.duration_jitter_s
            )

            amplitude = _get_source_amplitude_upa(source)

            # --- Generate a dynamic frequency contour ---
            num_points = max(2, self.num_contour_points)
            contour_freqs = [
                self.start_freq_hz
                + np.random.uniform(-self.start_freq_jitter_hz, self.start_freq_jitter_hz)
            ]
            for _ in range(num_points - 1):
                next_freq = contour_freqs[-1]
                contour_freqs.append(max(self.low_cutoff_hz, next_freq))  # Ensure freq > 0

            potential_events.append(
                {
                    "start_time": current_time_s,
                    "duration": max(0.1, duration),
                    "contour_freqs": contour_freqs,
                    "amplitude": amplitude,
                }
            )

            interval = self.mean_call_interval_s + np.random.uniform(
                -self.interval_jitter_s, self.interval_jitter_s
            )
            current_time_s += max(0.1, interval)

        return potential_events

    def _generate_structured_song_sequence(self, source: State) -> list[CallEvent]:
        """Generate a structured song based on themes and phrases.

        Parameters
        ----------
        source : State
            Source state object providing metadata (e.g. amplitude_upa).

        Returns
        -------
        list[CallEvent]
            A list of call events, each containing start time, duration, contour frequencies, and
            amplitude.

        """
        if not self.song_themes or not self.song_phrases:
            return []

        potential_events = []
        current_time_s = np.random.uniform(0, self.mean_call_interval_s)

        mean_amp = _get_source_amplitude_upa(source)

        for phrase in self.song_phrases:
            for theme_index in phrase:
                if theme_index >= len(self.song_themes):
                    continue  # Skip invalid theme index

                # --- Generate contour from the theme ---
                theme_offsets = self.song_themes[theme_index]
                base_freq = self.theme_base_freq_hz + np.random.uniform(
                    -self.start_freq_jitter_hz, self.start_freq_jitter_hz
                )
                contour_freqs = [base_freq]
                for offset in theme_offsets:
                    jitter = np.random.uniform(
                        -self.theme_freq_jitter_hz, self.theme_freq_jitter_hz
                    )
                    next_freq = contour_freqs[-1] + offset + jitter
                    contour_freqs.append(max(self.low_cutoff_hz, next_freq))

                # --- Set duration and amplitude ---
                duration = self.theme_duration_s + np.random.uniform(
                    -self.duration_jitter_s, self.duration_jitter_s
                )
                amplitude = mean_amp

                potential_events.append(
                    {
                        "start_time": current_time_s,
                        "duration": max(0.1, duration),
                        "contour_freqs": contour_freqs,
                        "amplitude": amplitude,
                    }
                )

                # --- Update time for the next call ---
                interval = self.mean_call_interval_s + np.random.uniform(
                    -self.interval_jitter_s, self.interval_jitter_s
                )
                current_time_s += max(0.1, interval)

                if current_time_s > self.duration_s:
                    return potential_events  # Stop if we exceed total duration

        return potential_events

    def _generate_call_sequence(self, source: State) -> list[CallEvent]:
        """Generate a sequential sequence of whale calls from a single source.

        Prevents overlapping calls.

        Parameters
        ----------
        source : State
            Source state object providing metadata (e.g. amplitude_upa).

        Returns
        -------
        list[CallEvent]
            A list of call events, each containing start time, duration, contour frequencies,
            and amplitude.

        """
        if self.song_structure_enabled:
            potential_events = self._generate_structured_song_sequence(source)
        else:
            potential_events = self._generate_random_call_sequence(source)

        if not potential_events:
            return []

        # Sort events by their intended start time
        potential_events.sort(key=lambda x: x["start_time"])

        # Adjust timing to ensure no overlaps
        sequential_events = []
        last_event_end_time = 0.0
        for event in potential_events:
            # If the current event starts before the last one ended,
            # shift its start time.
            if event["start_time"] < last_event_end_time:
                event["start_time"] = last_event_end_time

            sequential_events.append(event)
            last_event_end_time = event["start_time"] + event["duration"]

        return sequential_events

    def _generate_base_signal(self, source: State) -> FloatArray:
        """Generate the base 1D time-domain signal for the whale calls.

        Parameters
        ----------
        source : State
            Source state object providing metadata (e.g. amplitude_upa).

        Returns
        -------
        FloatArray
            The generated base signal containing the whale calls.

        """
        base_signal = np.zeros(self.num_samples, dtype=np.float64)
        call_events = self._generate_call_sequence(source)

        for event in call_events:
            call_template = self._create_call_template(
                duration=event["duration"],
                contour_freqs=event["contour_freqs"],
            )
            if call_template.size == 0:
                continue

            start_sample = int(event["start_time"] * self.sampling_rate_hz)
            end_sample = start_sample + len(call_template)

            if end_sample < self.num_samples:
                base_signal[start_sample:end_sample] += call_template * event["amplitude"]
        return base_signal

    def generate(
        self,
        source: State,
        sensor_delays_s: ArrayLike,
        tloss_db: ArrayLike | float,
        propagation_time_s: float,
    ) -> ComplexArray:
        """Generate whale calls, propagate them, and apply effects.

        Parameters
        ----------
        source
            The source state object used to read metadata.
        sensor_delays_s : ArrayLike
            1-D array of per-sensor delays in seconds.
        tloss_db : ArrayLike | float
            Transmission loss to the array origin (dB).
        propagation_time_s : float
            Propagation time from source to origin (s).

        Returns
        -------
        ComplexArray
            Complex signal for each sensor with shape
            `(num_sensors, num_samples)` and dtype `np.complex128`.

        """
        # Call the base class generate method to handle propagation
        signals = super().generate(source, sensor_delays_s, tloss_db, propagation_time_s)

        # Apply post-processing effects if any are specified
        if self.effects:
            for effect in self.effects:
                if isinstance(effect, Effect):
                    signals = effect.apply(signals, self.sampling_rate_hz)

        return np.asarray(signals, dtype=np.complex128)
