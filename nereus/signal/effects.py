"""Signal post-processing effects.

© Copyright 2025 Joshua J. Wakefield.
Licensed under the MIT License.
"""

import numpy as np
from stonesoup.base import Base, Property


class Effect(Base):
    """A base class for all signal post-processing effects."""

    def apply(self, signals: np.ndarray, sampling_rate_hz: int) -> np.ndarray:
        """Apply the effect to the signal. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement the apply method.")


class Reverb(Effect):
    """Applies a simple convolutional reverb effect to the signal."""

    duration_s = Property(
        float, default=1.0, doc="The decay time of the reverb tail in seconds."
    )
    wet_dry_mix = Property(
        float,
        default=0.3,
        doc="Mix between wet (reverb) and dry signal (0=dry, 1=wet).",
    )

    def apply(self, signals: np.ndarray, sampling_rate_hz: int) -> np.ndarray:
        """Apply a simple convolutional reverb effect to the signal.

        Args:
            signals: An array of complex signals with shape (num_sensors, num_samples).
            sampling_rate_hz: The sampling rate in Hertz.

        Returns:
            An array of complex signals with the reverb effect applied,
            with shape (num_sensors, num_samples).

        """
        if not (0 < self.wet_dry_mix <= 1 and self.duration_s > 0):
            return signals

        # Generate a synthetic Impulse Response (IR) for the reverb effect
        ir_samples = int(self.duration_s * sampling_rate_hz)
        time = np.arange(ir_samples) / sampling_rate_hz
        decay = np.exp(-5.0 * time / self.duration_s)  # Exponential decay
        ir = decay * np.random.randn(ir_samples)  # White noise modulated by decay
        ir /= np.max(np.abs(ir))  # Normalise the IR

        # Apply reverb to each sensor channel independently
        reverbed_signals = np.copy(signals)
        for i in range(signals.shape[0]):
            dry_signal = signals[i, :]
            wet_signal = np.convolve(dry_signal, ir, mode="same")
            reverbed_signals[i, :] = (
                1 - self.wet_dry_mix
            ) * dry_signal + self.wet_dry_mix * wet_signal

        return reverbed_signals
