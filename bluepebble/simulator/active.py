"""Active sonar simulation: pings, target echoes and the received waveform at each sensor."""

from collections.abc import Iterator, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, TypeAlias

import numpy as np
from numpy.typing import ArrayLike
from scipy.fft import fft, fftfreq, ifft, next_fast_len
from stonesoup.base import Property
from stonesoup.types.array import StateVector
from stonesoup.types.groundtruth import GroundTruthState

from ..arrays.base import ArrayState, HostState, PlatformState
from ..models.propagation import SpectrumPropagationModel
from ..models.propagation.acoustic import _get_source_position
from ..models.scattering import TargetScatteringModel
from ..signal.active import ActiveSignal
from ..types.sensordata import ActiveSonarSensorData
from .base import ComplexArray, FloatArray, SonarArraySimulatorBase

if TYPE_CHECKING:
    from stonesoup.types.state import State

    from ..arrays import SensorArray

ActiveSensorBatch: TypeAlias = tuple[datetime, set[ActiveSonarSensorData]]

# Targets closer than this horizontally to the transmitter or receiver are skipped: range-
# dependent propagation models have no horizontal direction to trace along.
_MIN_HORIZONTAL_RANGE_M = 1.0


class ActiveSonarSimulator(SonarArraySimulatorBase):
    r"""Simulate the echoes an active sonar receives from each ping.

    Each ping transmits ``signal`` from the transmitter's reference point. Every target echoes
    it back to each sensor of ``array``. Both legs come from ``propagation_model``, so any
    :class:`~bluepebble.models.propagation.SpectrumPropagationModel` works (spherical
    spreading, rtrs or Bellhop). Received sensor ``i`` sees, in the frequency domain,

    .. math::

        R_i(f) = \sum_{\text{targets}} g_\text{tx}\, g_i\, P(f)\,
                 H_\text{out}(f)\, S(f)\, H_{\text{in},i}(f),

    where :math:`P` is the pulse spectrum, :math:`H_\text{out}` the transfer function from the
    transmitter to the target, :math:`S` the target's scattering, :math:`H_{\text{in},i}` the
    transfer function from the target to sensor ``i``, and :math:`g_\text{tx}` and
    :math:`g_i` the baffle amplitude gains towards the target
    (:meth:`~bluepebble.arrays.ArrayState.transmit_gain` and
    :meth:`~bluepebble.arrays.ArrayState.sensor_gains`). The transfer functions carry the
    absolute travel times, so the inverse FFT puts each echo at its round-trip delay.
    Sample 0 of ``received_waveform`` is the transmit instant.

    With no ``transmitter`` the sonar is monostatic. A transmitter on another platform makes
    it bistatic, e.g. a hull array pinging for a sonobuoy field.

    Assumptions
    -----------
    - Geometry is frozen at the ping time for the whole ping, so there is no Doppler and no
      target or platform motion between transmission and reception.
    - Each target is a point scatterer; there is no reverberation and no direct blast from
      transmitter to receiver.
    - The pulse is transmitted from the transmitter's reference point, with the array's mean
      baffle gain towards the target as its directivity.
    - Baffle gains use the straight-line direction to the target, not the ray angles.
    - Echoes later than the FFT length, twice the listen window plus the pulse, would wrap;
      everything earlier is placed exactly.
    """

    propagation_model: SpectrumPropagationModel = Property(
        doc="Propagation model giving both legs' transfer functions",
    )
    signal: ActiveSignal = Property(doc="Transmitted pulse")
    scattering_models: Sequence[TargetScatteringModel] = Property(
        doc="Target scattering: one model shared by every target, or one per ground-truth path",
    )
    listen_duration_s: float = Property(
        doc="Length of the received waveform after each transmission, in seconds",
    )
    transmitter: "SensorArray | None" = Property(
        default=None,
        doc="Array whose reference point transmits. None for monostatic, transmitting from "
        "``array``.",
    )
    ping_timestamps: Sequence[datetime] | None = Property(
        default=None,
        doc="Transmit times. None pings at every timestamp the receiving array's platform "
        "has a state for.",
    )
    band_threshold_db: float = Property(
        default=60.0,
        doc="Transfer functions are evaluated only at frequency bins where the pulse spectrum "
        "is within this many dB of its peak; other bins are zero.",
    )

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialise the simulator and check its settings.

        Parameters
        ----------
        *args : object
            Positional arguments forwarded to ``SonarArraySimulatorBase``.
        **kwargs : object
            Keyword arguments forwarded to ``SonarArraySimulatorBase``.

        Raises
        ------
        TypeError
            If ``propagation_model`` is not a ``SpectrumPropagationModel``.
        ValueError
            If ``listen_duration_s`` or ``band_threshold_db`` is not positive.

        """
        super().__init__(*args, **kwargs)
        if not isinstance(self.propagation_model, SpectrumPropagationModel):
            raise TypeError(
                f"{type(self).__name__} needs a SpectrumPropagationModel, got "
                f"{type(self.propagation_model).__name__}."
            )
        if isinstance(self.scattering_models, TargetScatteringModel):
            self.scattering_models = [self.scattering_models]
        if not self.listen_duration_s > 0:
            raise ValueError(f"listen_duration_s ({self.listen_duration_s}) must be positive.")
        if not self.band_threshold_db > 0:
            raise ValueError(f"band_threshold_db ({self.band_threshold_db}) must be positive.")

    def sensor_data_gen(self) -> Iterator[ActiveSensorBatch]:
        """Yield the received waveform for each ping.

        Every platform involved must have been moved through each ping timestamp first.

        Yields
        ------
        tuple of (datetime, set of ActiveSonarSensorData)
            The ping's transmit time and a one-element set holding its sensor data.

        Raises
        ------
        ValueError
            If the receiving array or the transmitter has no state at a ping time.

        """
        fs = float(self.signal.sampling_rate_hz)
        pulse = np.asarray(self.signal.generate(), dtype=np.complex128)
        num_samples = int(round(self.listen_duration_s * fs))
        num_fft = next_fast_len(2 * (num_samples + pulse.size))
        pulse_spectrum = fft(pulse, num_fft)
        band = _band_mask(pulse_spectrum, self.band_threshold_db)
        band_frequencies_hz = fftfreq(num_fft, 1.0 / fs)[band]
        band_pulse_spectrum = pulse_spectrum[band]

        paths = list(self.ground_truth_paths)
        scattering_models = (
            self._resolve_models(self.scattering_models, len(paths), "scattering_models")
            if paths
            else []
        )
        target_states = [self._states_by_timestamp(path) for path in paths]
        timestamps = (
            self._sorted_timestamps()
            if self.ping_timestamps is None
            else sorted(self.ping_timestamps)
        )

        for timestamp in timestamps:
            receiver = self._platform_state_at(timestamp)
            transmitter = self._transmitter_state_at(timestamp, receiver)
            transmitter_position = _column(transmitter.array.ref_state_vector)
            receiver_position = _column(receiver.array.ref_state_vector)

            spectrum = np.zeros((receiver.array.num_sensors, num_fft), dtype=np.complex128)
            for states, scattering_model in zip(target_states, scattering_models, strict=True):
                target = states.get(timestamp)
                if target is None:
                    continue
                echo = self._echo_spectrum(
                    receiver,
                    transmitter,
                    target,
                    scattering_model,
                    band_frequencies_hz,
                    band_pulse_spectrum,
                )
                if echo is not None:
                    spectrum[:, band] += echo

            waveform = ifft(spectrum, axis=1)[:, :num_samples]
            noise = self._generate_noise(receiver.array.num_sensors, num_samples)
            if noise is not None:
                waveform = waveform + noise
            beamformed, bearings = self._beamform_if_configured(timestamp, waveform)
            sensor_data = ActiveSonarSensorData(
                received_waveform=waveform,
                transmit_pulse=pulse,
                timestamp=timestamp,
                beamformed_data=beamformed,
                steering_bearings_rad=bearings,
                transmitter_position=transmitter_position,
                receiver_reference_position=receiver_position,
            )
            yield timestamp, {sensor_data}

    def _transmitter_state_at(self, timestamp: datetime, receiver: PlatformState) -> PlatformState:
        """Return the transmitter's state at ``timestamp``; the receiver's when monostatic."""
        if self.transmitter is None:
            return receiver
        state = self.transmitter.get_platform_state_at(timestamp)
        if state is None:
            raise ValueError(
                f"The transmitter's platform has no state at {timestamp}; move it through "
                "every ping timestamp first."
            )
        return state

    def _echo_spectrum(
        self,
        receiver: PlatformState,
        transmitter: PlatformState,
        target: "State",
        scattering_model: TargetScatteringModel,
        frequencies_hz: FloatArray,
        pulse_spectrum: ComplexArray,
    ) -> ComplexArray | None:
        """Return one target's echo spectrum at each sensor, or None if it is not heard.

        A target is not heard if either leg has no propagation path, if it is within
        1 m horizontally of the transmitter or receiver, or if its earliest echo arrives
        after the listen window.
        """
        target_position = _column(_get_source_position(target))
        transmitter_position = _column(transmitter.array.ref_state_vector)
        receiver_position = _column(receiver.array.ref_state_vector)
        if (
            min(
                np.hypot(*(target_position[:2] - transmitter_position[:2])),
                np.hypot(*(target_position[:2] - receiver_position[:2])),
            )
            < _MIN_HORIZONTAL_RANGE_M
        ):
            return None

        # The target is the receiver on the way out, so no leg relies on reciprocity.
        transfer_out, delay_out = self.propagation_model.propagate_spectrum(
            _point_platform_state(target_position, receiver.timestamp),
            _point_source(transmitter_position, receiver.timestamp),
            frequencies_hz,
        )
        transfer_in, delays_in = self.propagation_model.propagate_spectrum(
            receiver, target, frequencies_hz
        )
        delays_in = np.asarray(delays_in, dtype=float)
        delay_out_s = float(np.asarray(delay_out, dtype=float).reshape(-1)[0])
        if np.isnan(delay_out_s) or np.all(np.isnan(delays_in)):
            return None
        if delay_out_s + np.nanmin(delays_in) >= self.listen_duration_s:
            return None

        transmit_gain = transmitter.array.transmit_gain(target_position - transmitter_position)
        sensor_gains = receiver.array.sensor_gains(target_position - receiver_position)
        transmitted = (
            transmit_gain
            * pulse_spectrum
            * np.asarray(transfer_out).reshape(-1)
            * np.asarray(scattering_model.scatter(frequencies_hz))
        )
        return sensor_gains[:, np.newaxis] * np.asarray(transfer_in) * transmitted


def _band_mask(spectrum: ComplexArray, threshold_db: float) -> np.ndarray:
    """Return the bins where ``|spectrum|`` is within ``threshold_db`` of its peak."""
    magnitude = np.abs(spectrum)
    return magnitude >= magnitude.max() * 10.0 ** (-threshold_db / 20.0)


def _column(vector: ArrayLike) -> FloatArray:
    """Return a position as a flat ``(3,)`` float array."""
    return np.asarray(vector, dtype=float).reshape(-1)


def _point_platform_state(position: FloatArray, timestamp: datetime) -> PlatformState:
    """Return a one-sensor platform state at ``position``, for a propagation receiver."""
    state = GroundTruthState(StateVector(position), timestamp=timestamp)
    return PlatformState(
        timestamp=timestamp,
        host=HostState(state=state, heading_rad=0.0, position_mapping=(0, 1, 2)),
        array=ArrayState(
            num_sensors=1,
            state_vector=position.reshape(3, 1),
            ref_state_vector=StateVector(position),
        ),
    )


def _point_source(position: FloatArray, timestamp: datetime) -> GroundTruthState:
    """Return a source state at ``position``, for a propagation source."""
    return GroundTruthState(
        StateVector(position),
        timestamp=timestamp,
        metadata={"position_mapping": [0, 1, 2]},
    )
