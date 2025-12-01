"""Anthropogenic signal models for sensor arrays.

© Copyright 2025 Joshua J. Wakefield.
Licensed under the MIT License.
"""

import numpy as np

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

        Args:
            source: The source state. Must contain `amplitudes_upa`,
                `frequencies_hz`, and `phases_rad` in its metadata dictionary.
            sensor_delays_s: The relative time delay for each sensor in the array.
            tloss_db: The transmission loss in decibels.
            propagation_time_s: The time in seconds for the signal to propagate
                from the source to the array's origin.

        Returns:
            An array of complex signals received by the sensors,
            with shape (num_sensors, num_samples).

        """
        # Create a 1D array representing the time vector for the signal snapshot
        time_array_s = np.arange(self.num_samples) / self.sampling_rate_hz

        amplitudes_upa = source.metadata["amplitudes_upa"]
        frequencies_hz = source.metadata["frequencies_hz"]
        phases_rad = source.metadata["phases_rad"]

        # Attenuate the source amplitude(s) based on transmission loss
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
