"""Signal post-processing effects."""

from abc import ABC, abstractmethod
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray
from stonesoup.base import Base, Property

ComplexArray: TypeAlias = NDArray[np.complexfloating[Any, Any]]


class Effect(Base, ABC):
    """A base class for all signal post-processing effects."""

    @abstractmethod
    def apply(self, signals: ComplexArray, sampling_rate_hz: int) -> ComplexArray:
        """Apply the effect to the signal. Must be implemented by subclasses."""
        ...


class Reverb(Effect):
    """Applies a simple convolutional reverb effect to the signal."""

    duration_s: float = Property(default=1.0, doc="The decay time of the reverb tail in seconds.")
    wet_dry_mix: float = Property(
        default=0.3,
        doc="Mix between wet (reverb) and dry signal (0=dry, 1=wet).",
    )

    def apply(self, signals: ComplexArray, sampling_rate_hz: int) -> ComplexArray:
        """Apply a simple convolutional reverb effect to the signal.

        Parameters
        ----------
        signals : ComplexArray
            An array of complex signals with shape (num_sensors, num_samples).
        sampling_rate_hz : int
            The sampling rate in Hertz.

        Returns
        -------
        ComplexArray
            An array of complex signals with the reverb effect applied,
            with shape (num_sensors, num_samples).

        """
        if self.duration_s <= 0:
            msg = f"duration_s must be positive, got {self.duration_s}"
            raise ValueError(msg)
        if not 0 < self.wet_dry_mix <= 1:
            msg = f"wet_dry_mix must be in (0, 1], got {self.wet_dry_mix}"
            raise ValueError(msg)

        # Generate a synthetic Impulse Response (IR) for the reverb effect
        ir_samples = int(self.duration_s * sampling_rate_hz)
        time = np.arange(ir_samples) / sampling_rate_hz
        decay = np.exp(-5.0 * time / self.duration_s)  # Exponential decay
        ir = decay * np.random.randn(ir_samples)  # White noise modulated by decay
        ir_max = np.max(np.abs(ir))
        if ir_max > 0:
            ir /= ir_max  # Normalise the IR

        # Apply reverb to each sensor channel independently
        reverbed_signals = np.copy(signals)
        for i in range(signals.shape[0]):
            dry_signal = signals[i, :]
            wet_signal = np.convolve(dry_signal, ir, mode="same")
            reverbed_signals[i, :] = (
                1 - self.wet_dry_mix
            ) * dry_signal + self.wet_dry_mix * wet_signal

        return reverbed_signals
