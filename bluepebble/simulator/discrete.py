"""Discrete acoustic sensor simulators module."""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Iterator
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray
from stonesoup.base import Property
from stonesoup.types.sensordata import SensorData

from ..signal.base import Signal
from .base import PassiveSonarArraySimulatorBase

if TYPE_CHECKING:
    from stonesoup.types.state import State

Complex64Array: TypeAlias = NDArray[np.complex64]
Complex128Array: TypeAlias = NDArray[np.complex128]
SensorBatch: TypeAlias = tuple[datetime, set[SensorData]]


class _StftSourceSignalModel(Protocol):
    """Protocol for signal models exposing STFT-backed source caching."""

    def compute_stft(self, source: State) -> object:
        """Compute and cache a source STFT."""
        ...

    def get_source_signal(self) -> NDArray[np.complexfloating[Any, Any]]:
        """Return cached source signal."""
        ...


class _BaseSignalModel(Protocol):
    """Protocol for signal models exposing direct base-signal synthesis."""

    def _generate_base_signal(
        self,
        source: State,
    ) -> NDArray[np.floating[Any] | np.complexfloating[Any, Any]]:
        """Generate base source waveform."""
        ...


class _SpectrumPropagationModel(Protocol):
    """Protocol for propagation models with frequency-domain transfer support."""

    def propagate_spectrum(
        self,
        platform: object,
        source: State,
        frequencies_hz: NDArray[np.float64],
    ) -> tuple[NDArray[np.complexfloating[Any, Any]], float]:
        """Return per-sensor transfer functions and propagation time."""
        ...


class DiscretePassiveSonarArraySimulator(PassiveSonarArraySimulatorBase):
    """Discrete broadband simulator with per-timestamp spectrum rendering.

    This simulator synthesizes one independent snapshot per platform timestamp.
    It is intended for broadband scenarios where each timestep can be processed
    as a standalone frame, without enforcing waveform continuity across adjacent
    timestamps.

    Processing stages
    -----------------
    1. Resolve one signal model per target (or broadcast a single shared model).
    2. Build one full source waveform per target via
       ``compute_stft/get_source_signal`` or ``_generate_base_signal``.
    3. Partition source waveforms into timestamp-aligned chunks using platform
       time spacing and sampling rate.
    4. For each timestamp, evaluate ``H(f)`` from ``propagate_spectrum`` at the
       chunk FFT bins for each active target.
    5. Apply ``H(f)`` in the frequency domain and IFFT per sensor, summing
       contributions from all targets.
    6. Add optional ambient noise, run optional beamforming, and emit one
       ``PassiveSonarSensorData`` payload.

    Notes
    -----
    - This method does not interpolate transfer functions across timesteps.
    - No overlap-add or continuity smoothing is applied between emitted chunks.
    - Timestamp boundaries that produce zero-length chunks are clamped to a
      minimum one-sample snapshot to keep FFT processing valid.

    Tradeoffs
    ---------
    - Lower complexity and simpler reasoning than continuous STFT/WOLA methods.
    - Best suited to analyses where per-timestep independence is acceptable.

    """

    signal_models: list[Signal] = Property(
        list,
        doc="List of broadband signal models (one per target, or single-element list for all)",
    )

    def _resolve_signal_models(self, num_targets: int) -> list[Signal]:
        """Resolve one signal model per target.

        Parameters
        ----------
        num_targets : int
            Number of targets represented in the current scenario.

        Returns
        -------
        list of Signal
            Resolved signal-model list where each target has one model.

        """
        return self._resolve_models(self.signal_models, num_targets, "signal models")

    @staticmethod
    def _get_target_first_state(target_path: Iterable[State]) -> State | None:
        """Return the first state from a target path.

        Parameters
        ----------
        target_path : Iterable of State
            Target state sequence.

        Returns
        -------
        State or None
            First state if available, else ``None`` for empty paths.

        """
        try:
            return next(iter(target_path))
        except StopIteration:
            return None

    @staticmethod
    def _get_broadband_source_signal(
        signal_model: Signal,
        first_state: State | None,
    ) -> Complex128Array:
        """Get a source waveform from supported signal-model interfaces.

        Parameters
        ----------
        signal_model : Signal
            Signal model implementing either
            ``compute_stft/get_source_signal`` or ``_generate_base_signal``.
        first_state : State or None
            First target state, used to initialise lazy source generation.

        Returns
        -------
        numpy.ndarray
            Complex source waveform as ``complex128``.

        Raises
        ------
        ValueError
            If source initialisation requires state context but no state is
            available.
        TypeError
            If the signal model does not expose a supported source API.

        """
        if hasattr(signal_model, "compute_stft") and hasattr(signal_model, "get_source_signal"):
            stft_signal_model = cast(_StftSourceSignalModel, signal_model)
            try:
                source_signal = stft_signal_model.get_source_signal()
            except RuntimeError as err:
                if first_state is None:
                    msg = (
                        "Cannot initialize broadband source signal for an empty target path. "
                        "Provide a target state or pre-compute the source signal."
                    )
                    raise ValueError(msg) from err
                stft_signal_model.compute_stft(first_state)
                source_signal = stft_signal_model.get_source_signal()
            return np.asarray(source_signal, dtype=np.complex128)

        if hasattr(signal_model, "_generate_base_signal"):
            base_signal_model = cast(_BaseSignalModel, signal_model)
            if first_state is None:
                msg = (
                    "Cannot initialize source signal for an empty target path when using "
                    "_generate_base_signal."
                )
                raise ValueError(msg)
            return np.asarray(
                base_signal_model._generate_base_signal(first_state),
                dtype=np.complex128,
            )

        msg = (
            "Signal model must implement either compute_stft/get_source_signal "
            "or _generate_base_signal."
        )
        raise TypeError(msg)

    def sensor_data_gen(self) -> Iterator[SensorBatch]:
        """Yield one independent broadband snapshot per platform timestamp.

        Yields
        ------
        tuple of (datetime, set of SensorData)
            Timestamp and simulated sensor-data set for that timestamp.

        Raises
        ------
        AttributeError
            If the configured propagation model does not implement
            ``propagate_spectrum``.
        ValueError
            If signal model configuration is invalid.

        """
        if not hasattr(self.propagation_model, "propagate_spectrum"):
            msg = (
                "DiscreteBroadbandPassiveSonarArraySimulator requires propagation_model "
                "to implement propagate_spectrum"
            )
            raise AttributeError(msg)
        spectrum_propagation_model = cast(_SpectrumPropagationModel, self.propagation_model)

        all_timestamps = self._sorted_timestamps()
        ground_truth_paths = self.ground_truth_paths or []
        signal_models_list = self._resolve_signal_models(len(ground_truth_paths))

        if len(signal_models_list) == 0:
            msg = "signal models must contain at least one model"
            raise ValueError(msg)

        num_sensors = int(self.platform.num_sensors)
        sampling_rate_hz = float(signal_models_list[0].sampling_rate_hz)
        total_samples = int(signal_models_list[0].num_samples)

        if total_samples <= 0:
            msg = "signal model num_samples must be greater than zero"
            raise ValueError(msg)

        t0 = all_timestamps[0]
        step_times_s = np.array(
            [(ts - t0).total_seconds() for ts in all_timestamps],
            dtype=np.float64,
        )
        step_sample_idx = np.rint(step_times_s * sampling_rate_hz).astype(np.int64)
        step_sample_idx = np.clip(step_sample_idx, 0, total_samples)
        step_sample_idx = np.maximum.accumulate(step_sample_idx)

        source_signal_by_target: list[Complex64Array] = []
        for target_idx, target_path in enumerate(ground_truth_paths):
            target_signal_model = signal_models_list[target_idx]

            if int(target_signal_model.num_samples) != total_samples:
                msg = (
                    "All signal models must share the same num_samples for discrete "
                    "broadband simulation."
                )
                raise ValueError(msg)

            if float(target_signal_model.sampling_rate_hz) != sampling_rate_hz:
                msg = (
                    "All signal models must share the same sampling_rate_hz for "
                    "discrete broadband simulation."
                )
                raise ValueError(msg)

            first_state = self._get_target_first_state(target_path)
            source_signal = self._get_broadband_source_signal(target_signal_model, first_state)

            if len(source_signal) < total_samples:
                pad_len = total_samples - len(source_signal)
                source_signal = np.concatenate(
                    [source_signal, np.zeros(pad_len, dtype=np.complex64)]
                )
            elif len(source_signal) > total_samples:
                source_signal = source_signal[:total_samples]

            source_signal_by_target.append(np.asarray(source_signal, dtype=np.complex64))

        n_steps = len(all_timestamps)
        for step_idx, timestamp in enumerate(all_timestamps):
            platform_state = self.platform.get_platform_state_at(timestamp)

            start_sample = int(step_sample_idx[step_idx])
            if step_idx < n_steps - 1:
                end_sample = int(step_sample_idx[step_idx + 1])
            else:
                end_sample = total_samples

            # Ensure a non-empty chunk for FFT processing.
            if end_sample <= start_sample:
                if start_sample >= total_samples:
                    start_sample = max(0, total_samples - 1)
                    end_sample = total_samples
                else:
                    end_sample = min(total_samples, start_sample + 1)

            num_samples_snapshot = end_sample - start_sample
            frequencies_hz = np.fft.fftfreq(num_samples_snapshot, d=1.0 / sampling_rate_hz)
            sensor_signals = np.zeros((num_sensors, num_samples_snapshot), dtype=np.complex64)

            for target_idx, target_path in enumerate(ground_truth_paths):
                target_state = self._target_state_at(target_path, timestamp)
                if target_state is None:
                    continue

                source_chunk = source_signal_by_target[target_idx][start_sample:end_sample]
                source_fft = np.fft.fft(source_chunk)

                H_sensors, _ = spectrum_propagation_model.propagate_spectrum(
                    platform_state,
                    target_state,
                    frequencies_hz,
                )
                target_fft = np.asarray(H_sensors, dtype=np.complex64) * source_fft[np.newaxis, :]
                sensor_signals += np.fft.ifft(target_fft, axis=1).astype(np.complex64)

            noise = self._generate_noise(
                num_sensors=num_sensors,
                num_samples=num_samples_snapshot,
                sampling_rate_hz=sampling_rate_hz,
            )
            if noise is not None:
                sensor_signals += noise

            beamformed_data = self._beamform_if_configured(
                timestamp=timestamp,
                sensor_signals=sensor_signals,
            )

            sensor_data = self._make_sensor_data(
                timestamp=timestamp,
                sensor_signals=sensor_signals,
                beamformed_data=beamformed_data,
            )
            yield timestamp, {sensor_data}


class DeprecatedDiscretePassiveSonarArraySimulator(PassiveSonarArraySimulatorBase):
    """Deprecated discrete-time passive-sonar array simulator.

    .. warning::
       This class is deprecated and retained for backwards compatibility only.
       Prefer ``DiscretePassiveSonarArraySimulator`` for new broadband work.

    This simulator produces one sensor-data snapshot per platform timestamp and supports two
    propagation modes:

    1. ``"transmission_loss"`` (default): uses ``propagate`` + per-sensor delays with the signal
        model's ``generate`` method.
    2. ``"spectrum"``: applies ``propagate_spectrum`` transfer functions to each target's source
       spectrum and reconstructs sensor channels by IFFT.

    The spectrum mode is more computationally expensive but can represent broadband multipath
    phase and frequency-dependent attenuation more faithfully when the propagation model provides
    high-quality transfer functions.

    Attributes
    ----------
    platform : TowedArrayPlatform
        The towed array platform providing geometry.
    propagation_model : AcousticPropagationModel
        Model for acoustic propagation.
    signal_models : list of Signal
        Signal models for target generation. Provide one model to share across all targets, or one
        model per target.
    noise_model : AmbientNoise
        Model for generating ambient noise.
    beamformer : Beamformer
        The beamforming algorithm to apply.
    steering_calculator : SteeringCalculator
        Calculator for steering delays.
    ground_truth_paths : list
        A list of ``GroundTruthPath`` objects representing targets.
    propagation_method : str
        Propagation path used for target rendering. Supported values are ``"transmission_loss"``
        and ``"spectrum"``.

    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialise simulator and emit a deprecation warning.

        Parameters
        ----------
        *args : object
            Positional arguments forwarded to the base simulator.
        **kwargs : object
            Keyword arguments forwarded to the base simulator.

        """
        warnings.warn(
            "DeprecatedDiscretePassiveSonarArraySimulator is deprecated; "
            "use DiscretePassiveSonarArraySimulator instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)

    signal_models: list[Signal] = Property(
        list,
        doc="List of acoustic signal models (one per target, or single-element list for all)",
    )
    propagation_method: str = Property(
        str,
        default="transmission_loss",
        doc="Propagation method: 'transmission_loss' or 'spectrum'",
    )

    def _resolve_signal_models(self, num_targets: int) -> list[Signal]:
        """Resolve signal model mapping for targets.

        Parameters
        ----------
        num_targets : int
            Number of targets to model at the current timestamp.

        Returns
        -------
        list of Signal
            A list of signal models with one entry per target when ``num_targets > 1``.

        """
        return self._resolve_models(self.signal_models, num_targets, "signal models")

    def sensor_data_gen(self) -> Iterator[SensorBatch]:
        """Generate sensor data for each timestamp in the platform's trajectory.

        This generator iterates through all unique timestamps defined in the platform's movement
        controller, yielding a set of sensor data for each point in time.

        Yields
        ------
        tuple of (datetime, set of SensorData)
            Timestamp and simulated sensor-data set for that timestep.

        """
        all_timestamps = self._sorted_timestamps()

        # Generate sensor data for each timestamp
        for timestamp in all_timestamps:
            sensor_data = self._generate_sensor_data_at(timestamp)
            yield timestamp, {sensor_data}

    def _generate_sensor_data_at(self, timestamp: datetime) -> SensorData:
        """Generate a single snapshot of sensor data at a specific timestamp.

        This method performs the core simulation steps for a single moment in time. It generates
        signals for all active targets, sums them, adds ambient noise, and then processes the
        result through a beamformer.

        Parameters
        ----------
        timestamp : datetime
            The timestamp for which to generate data.

        Returns
        -------
        SensorData
            Simulated passive-sonar sensor snapshot for the given timestamp.

        Raises
        ------
        ValueError
            If ``propagation_method`` is unsupported.
        AttributeError
            If ``propagation_method="spectrum"`` is selected but the
            propagation model does not implement ``propagate_spectrum``.

        """
        platform = self.platform.get_platform_state_at(timestamp)
        method = str(self.propagation_method).lower()
        if method not in {"transmission_loss", "spectrum"}:
            msg = (
                f"Unsupported propagation_method '{self.propagation_method}'. "
                "Expected 'transmission_loss' or 'spectrum'."
            )
            raise ValueError(msg)

        # Get sensor positions at this timestamp
        num_sensors = self.platform.num_sensors
        ground_truth_paths = self.ground_truth_paths or []
        signal_models_list = self._resolve_signal_models(len(ground_truth_paths))
        num_samples = signal_models_list[0].num_samples

        # Initialise combined signal array
        sensor_signals = np.zeros((num_sensors, num_samples), dtype=np.complex128)

        # Generate signals from all targets present at this timestamp
        for target_idx, path in enumerate(ground_truth_paths):
            target_state = self._target_state_at(path, timestamp)

            if target_state is None:
                continue

            target_signal_model = signal_models_list[target_idx]

            if method == "spectrum":
                if not hasattr(self.propagation_model, "propagate_spectrum"):
                    msg = (
                        "propagation_method='spectrum' requires propagation_model "
                        "to implement propagate_spectrum"
                    )
                    raise AttributeError(msg)
                spectrum_propagation_model = cast(
                    _SpectrumPropagationModel,
                    self.propagation_model,
                )

                # Build physical frequency axis matching FFT bins.
                sampling_rate_hz = float(target_signal_model.sampling_rate_hz)
                frequencies = np.fft.fftfreq(num_samples, d=1.0 / sampling_rate_hz)
                H_sensors, _ = spectrum_propagation_model.propagate_spectrum(
                    platform,
                    target_state,
                    frequencies,
                )

                if hasattr(target_signal_model, "_generate_base_signal"):
                    base_signal_model = cast(_BaseSignalModel, target_signal_model)
                    base_signal = np.asarray(
                        base_signal_model._generate_base_signal(target_state),
                        dtype=np.complex128,
                    )
                    if len(base_signal) < num_samples:
                        pad = num_samples - len(base_signal)
                        base_signal = np.concatenate(
                            [base_signal, np.zeros(pad, dtype=np.complex128)]
                        )
                    elif len(base_signal) > num_samples:
                        base_signal = base_signal[:num_samples]
                    source_fft = np.fft.fft(base_signal)
                    target_fft = (
                        np.asarray(H_sensors, dtype=np.complex128) * source_fft[np.newaxis, :]
                    )
                    target_signal = np.fft.ifft(target_fft, axis=1).astype(np.complex128)
                else:
                    # Fallback for non-standard signal models.
                    tloss_db, prop_time_s = self.propagation_model.propagate(
                        platform, target_state
                    )
                    sensor_delays_s = self.propagation_model.compute_sensor_delays(
                        platform, target_state
                    )
                    target_signal = target_signal_model.generate(
                        target_state,
                        sensor_delays_s,
                        tloss_db,
                        prop_time_s,
                    )
            else:
                # Calculate propagation effects
                tloss_db, prop_time_s = self.propagation_model.propagate(platform, target_state)

                # Calculate sensor delays
                sensor_delays_s = self.propagation_model.compute_sensor_delays(
                    platform, target_state
                )

                # Generate target signal with acoustic properties from metadata
                target_signal = target_signal_model.generate(
                    target_state,
                    sensor_delays_s,
                    tloss_db,
                    prop_time_s,
                )

            sensor_signals += target_signal

        # Add environmental noise
        noise = self._generate_noise(
            num_sensors=num_sensors,
            num_samples=num_samples,
        )
        if noise is not None:
            sensor_signals += noise

        beamformed_data = self._beamform_if_configured(
            timestamp=timestamp,
            sensor_signals=sensor_signals,
        )

        return self._make_sensor_data(
            timestamp=timestamp,
            sensor_signals=sensor_signals,
            beamformed_data=beamformed_data,
        )
