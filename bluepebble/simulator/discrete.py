"""Discrete acoustic sensor simulatiors module."""

from collections.abc import Iterator
from datetime import datetime

import numpy as np
from stonesoup.base import Property
from stonesoup.types.sensordata import SensorData

from ..signal.base import Signal
from .base import PassiveSonarArraySimulatorBase


class DiscretePassiveSonarArraySimulator(PassiveSonarArraySimulatorBase):
    """Discrete-time passive-sonar array simulator.

    This simulator produces one sensor-data snapshot per platform timestamp and supports two
    propagation modes:

    1. ``"transmission_loss"`` (default): uses ``propagate`` + per-sensor delays with the signal model's
       ``generate`` method.
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
        Propagation path used for target rendering. Supported values are ``"transmission_loss"`` and
        ``"spectrum"``.

    """

    signal_models = Property(
        list,
        doc="List of acoustic signal models (one per target, or single-element list for all)",
    )
    propagation_method = Property(
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

    def sensor_data_gen(self) -> Iterator[tuple[datetime, set[SensorData]]]:
        """Generate sensor data for each timestamp in the platform's trajectory.

        This generator iterates through all unique timestamps defined in the platform's movement
        controller, yielding a set of sensor data for each point in time.

        Yields
        ------
        tuple
            A tuple containing the timestamp and a set of ``PassiveSonarSensorData`` objects for
            that timestamp.

        """
        all_timestamps = self._sorted_timestamps()

        # Generate sensor data for each timestamp
        for timestamp in all_timestamps:
            sensor_data = self._generate_sensor_data_at(timestamp)
            yield timestamp, {sensor_data}

    def _generate_sensor_data_at(self, timestamp) -> SensorData:
        """Generate a single snapshot of sensor data at a specific timestamp.

        This method performs the core simulation steps for a single moment in time. It generates
        signals for all active targets, sums them, adds ambient noise, and then processes the
        result through a beamformer.

        Parameters
        ----------
        timestamp  : datetime
            The timestamp for which to generate data.

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

                # Build physical frequency axis matching FFT bins.
                sampling_rate_hz = float(target_signal_model.sampling_rate_hz)
                frequencies = np.fft.fftfreq(num_samples, d=1.0 / sampling_rate_hz)
                H_sensors, _ = self.propagation_model.propagate_spectrum(
                    platform,
                    target_state,
                    frequencies,
                )

                if hasattr(target_signal_model, "_generate_base_signal"):
                    base_signal = np.asarray(
                        target_signal_model._generate_base_signal(target_state),
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
                    target_fft = np.asarray(H_sensors, dtype=np.complex128) * source_fft[np.newaxis, :]
                    target_signal = np.fft.ifft(target_fft, axis=1).astype(np.complex128)
                else:
                    # Fallback for non-standard signal models.
                    tloss_db, prop_time_s = self.propagation_model.propagate(platform, target_state)
                    sensor_delays_s = self.propagation_model.compute_sensor_delays(platform, target_state)
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
                sensor_delays_s = self.propagation_model.compute_sensor_delays(platform, target_state)

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