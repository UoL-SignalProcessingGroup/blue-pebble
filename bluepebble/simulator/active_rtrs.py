"""Active sonar simulator using the RTRS two-hop Fourier-synthesis method."""

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
from numpy.typing import NDArray
from stonesoup.base import Base, Property
from stonesoup.types.array import StateVector, StateVectors
from stonesoup.types.groundtruth import GroundTruthState

from ..models.propagation import rtrsAcousticPropagationModel
from ..models.scattering import TargetScatteringModel
from ..platform.sonobuoy import OmniSonobuoyPlatform
from ..sensors.base import ArrayState
from ..signal.active import ActiveSignal
from ..signal.random import RandomSignal
from ..types.sensordata import ActiveSonarSensorData
from .active_bellhop import _target_state_at

if TYPE_CHECKING:
    from stonesoup.types.groundtruth import GroundTruthPath

ComplexArray: TypeAlias = NDArray[np.complexfloating[Any, Any]]
FloatArray: TypeAlias = NDArray[np.float64]
SensorBatch: TypeAlias = tuple[datetime, set[ActiveSonarSensorData]]

_POSITION_MAPPING_3D: list[int] = [0, 1, 2]


@dataclass(frozen=True)
class _PointReceiver:
    """Minimal receiver stand-in for a target, satisfying the ``platform.array`` contract.

    ``propagate_spectrum`` only reads ``platform.array.state_vector`` /
    ``.ref_state_vector``, so a full ``Platform`` is unnecessary here.
    """

    array: ArrayState


def _point_state(position: StateVector, timestamp: datetime) -> GroundTruthState:
    """Build a synthetic single-point source state for use with ``propagate_spectrum``."""
    return GroundTruthState(
        position,
        timestamp=timestamp,
        metadata={"position_mapping": _POSITION_MAPPING_3D},
    )


def _point_receiver(position: StateVector) -> _PointReceiver:
    """Build a synthetic single-element receiver at ``position``."""
    return _PointReceiver(
        array=ArrayState(
            num_sensors=1,
            state_vector=StateVectors(position),
            ref_state_vector=position,
        )
    )


def _occupied_band_mask(pulse_spectrum: ComplexArray, energy_threshold: float) -> NDArray[np.bool_]:
    """Return a mask over native FFT bins where the pulse spectrum carries real energy.

    Derived directly from the transmit pulse's own FFT magnitude rather than from a signal
    type's nominal bandwidth, so it correctly covers a CW pulse's finite-duration spectral
    mainlobe (not just its single carrier frequency) as well as an LFM pulse's swept band.
    """
    magnitude = np.abs(pulse_spectrum)
    peak = magnitude.max()
    if peak == 0.0:
        return np.zeros(magnitude.shape, dtype=bool)
    return magnitude >= energy_threshold * peak


class RtrsActiveSonarSimulatorOmni(Base):
    """Monostatic active sonar simulator using RTRS two-hop Fourier synthesis.

    Simulates a single omnidirectional sonobuoy that transmits a pulse and receives echoes from
    one or more targets, using the 3D `rtrs` ray tracer's frequency-domain transfer function
    ``H(f)`` (via :meth:`rtrsAcousticPropagationModel.propagate_spectrum`) in place of
    :class:`~bluepebble.simulator.active_bellhop.BellhopActiveSonarSimulatorOmni`'s discrete
    Bellhop eigenray pairs. For each ping, target:

    1. The propagation channel is queried directly at every native FFT bin within the pulse's
       occupied band (by acoustic reciprocity, one call stands in for both the outbound and
       inbound legs, unless ``use_reciprocity`` is ``False``). Benchmarking showed
       ``propagate_spectrum``'s cost is dominated by ray/beam-tracing geometry setup, done once
       per call regardless of how many frequencies are requested, so there is no benefit to
       querying a sparse subset and interpolating.
    2. The resulting transfer function is combined with the pulse spectrum and the target's
       scattering response: ``R(f) = X(f) * H(f)**2 * S(f)``.
    3. Contributions from all targets are summed in the frequency domain and inverse-FFT'd once
       per ping to produce the received time-domain waveform, with each target's exact
       round-trip delay -- rtrs's own ray-traced travel time -- already embedded in ``H(f)``'s
       phase at every queried bin.

    Parameters
    ----------
    platform : OmniSonobuoyPlatform
        The sonobuoy acting as both transmitter and receiver.
    propagation_model : rtrsAcousticPropagationModel
        RTRS-backed propagation model providing ``propagate_spectrum``.
    signal : ActiveSignal
        Transmit waveform (CWSignal or LFMSignal).
    scattering_model : TargetScatteringModel
        Target scattering response applied once per round trip.
    ground_truth_paths : list[GroundTruthPath]
        Target tracks. Each path must provide a state at every ping timestamp.
    ping_timestamps : list[datetime]
        Ordered list of times at which pings are transmitted.
    target_position_mapping : list[int]
        Indices into the target state vector that give ``[x, y, z]``.
        Defaults to ``[0, 2, 4]`` for a 6-D kinematic state
        ``[x, vx, y, vy, z, vz]``.
    receive_duration_s : float
        Listening window after pulse end in seconds. The receive buffer is
        ``signal.duration_s + receive_duration_s`` long.
    energy_threshold : float
        Fraction of the transmit pulse spectrum's peak magnitude below which a native FFT bin
        is treated as carrying no pulse energy, and excluded from the queried band. Default
        is ``1e-3``.
    use_reciprocity : bool
        If ``True`` (default), get the round-trip channel from a single
        ``propagate_spectrum`` call via acoustic reciprocity
        (``H(buoy->target) == H(target->buoy)``). If ``False``, fall back to two independent
        calls with source/receiver roles reversed for the outbound and inbound legs.
    noise_model : RandomSignal, optional
        Optional ambient noise model added to the raw hydrophone signal each ping.

    """

    platform: OmniSonobuoyPlatform = Property(
        doc="Omnidirectional sonobuoy (transmitter and receiver)"
    )
    propagation_model: rtrsAcousticPropagationModel = Property(
        doc="RTRS-backed propagation model providing propagate_spectrum"
    )
    signal: ActiveSignal = Property(
        doc="Transmit waveform (CWSignal or LFMSignal)"
    )
    scattering_model: TargetScatteringModel = Property(
        doc="Target scattering response applied once per round trip"
    )
    ground_truth_paths: list["GroundTruthPath"] = Property(
        doc="Target ground-truth tracks"
    )
    ping_timestamps: list[datetime] = Property(
        doc="Timestamps at which pings are transmitted"
    )
    target_position_mapping: list[int] = Property(
        default=None,
        doc="State-vector indices for [x, y, z]. Defaults to [0, 2, 4].",
    )
    receive_duration_s: float = Property(
        default=2.0,
        doc="Listening window after pulse end in seconds",
    )
    energy_threshold: float = Property(
        default=1e-3,
        doc="Fraction of the pulse spectrum's peak magnitude treated as the occupied band",
    )
    use_reciprocity: bool = Property(
        default=True,
        doc="Use a single propagate_spectrum call via acoustic reciprocity",
    )
    noise_model: RandomSignal | None = Property(
        default=None,
        doc="Optional ambient noise model added to the raw hydrophone signal each ping",
    )

    def _resolved_target_position_mapping(self) -> list[int]:
        """Return ``target_position_mapping``, defaulting to ``[0, 2, 4]`` when unset."""
        if self.target_position_mapping is None:
            return [0, 2, 4]
        return self.target_position_mapping

    def sensor_data_gen(self) -> Iterator[SensorBatch]:
        """Yield one ping's received waveform per timestamp.

        Yields
        ------
        tuple[datetime, set[ActiveSonarSensorData]]
            Ping timestamp and a one-element set containing the sensor data.

        """
        pulse = self.signal.generate()
        n_receive = int(
            (self.signal.duration_s + self.receive_duration_s) * self.signal.sampling_rate_hz
        )

        for timestamp in sorted(self.ping_timestamps):
            received = self._simulate_ping(timestamp, pulse, n_receive)
            if self.noise_model is not None:
                noise = self.noise_model.generate(num_sensors=1, num_samples=n_receive)
                received = received + noise[0]
            data = ActiveSonarSensorData(
                received_waveform=received,
                transmit_pulse=pulse,
                timestamp=timestamp,
            )
            yield timestamp, {data}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _simulate_ping(
        self,
        timestamp: datetime,
        pulse: ComplexArray,
        n_receive: int,
    ) -> ComplexArray:
        """Compute the received waveform for a single ping."""
        fs = self.signal.sampling_rate_hz
        pm = self._resolved_target_position_mapping()

        X = np.fft.fft(pulse, n_receive)
        native_freqs_hz = np.fft.fftfreq(n_receive, d=1.0 / fs)
        mask = _occupied_band_mask(X, self.energy_threshold)
        freqs_hz = native_freqs_hz[mask]

        buoy_pos = self.platform.position_3d

        R_total = np.zeros(n_receive, dtype=np.complex128)

        for target_path in self.ground_truth_paths:
            state = _target_state_at(target_path, timestamp)
            if state is None:
                continue

            tv = state.state_vector
            target_pos = StateVector(
                [
                    float(tv[pm[0], 0]),
                    float(tv[pm[1], 0]),
                    abs(float(tv[pm[2], 0])),
                ]
            )

            range_m = float(np.linalg.norm(buoy_pos[:2] - target_pos[:2]))
            if range_m < 1.0:
                continue  # degenerate case: target co-located with sonobuoy

            H_round, round_trip_delay_s = self._round_trip_channel(
                buoy_pos, target_pos, timestamp, freqs_hz
            )

            if round_trip_delay_s * fs >= n_receive:
                continue  # echo would arrive after the receive window ends

            S = np.zeros(mask.shape, dtype=np.complex128)
            S[mask] = self.scattering_model.scatter(native_freqs_hz[mask])

            # freqs_hz == native_freqs_hz[mask], queried in that exact order, so H_round
            # already carries rtrs's own exact per-bin phase/delay.
            H_native = np.zeros(native_freqs_hz.shape, dtype=np.complex128)
            H_native[mask] = H_round

            R_total += X * H_native * S

        return np.fft.ifft(R_total)

    def _round_trip_channel(
        self,
        buoy_pos: StateVector,
        target_pos: StateVector,
        timestamp: datetime,
        freqs_hz: FloatArray,
    ) -> tuple[ComplexArray, float]:
        """Return the round-trip transfer function and round-trip delay for one target.

        By acoustic reciprocity, ``H(buoy->target) == H(target->buoy)``, so
        ``use_reciprocity=True`` gets both legs from a single ``propagate_spectrum`` call
        (squaring the result) rather than two independent, role-reversed calls.
        """
        buoy_source = _point_state(buoy_pos, timestamp)
        target_receiver = _point_receiver(target_pos)

        H1, tau1 = self.propagation_model.propagate_spectrum(
            target_receiver, buoy_source, freqs_hz
        )
        H1 = H1[0]
        tau1 = float(np.asarray(tau1).reshape(-1)[0])

        if self.use_reciprocity:
            return H1**2, 2.0 * tau1

        target_source = _point_state(target_pos, timestamp)
        H2, tau2 = self.propagation_model.propagate_spectrum(
            self.platform, target_source, freqs_hz
        )
        H2 = H2[0]
        tau2 = float(np.asarray(tau2).reshape(-1)[0])
        return H1 * H2, tau1 + tau2
