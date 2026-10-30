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
from ..sensors.bow_array import BowArraySensor
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


def _point_and_array_receiver(ref_pos: StateVector, elem_pos: StateVectors) -> _PointReceiver:
    """Receiver set combining the array reference point with every dome element.

    Row 0 is the array reference point; rows 1: are the real elements. Letting one
    ``propagate_spectrum`` call (source = target) return both lets
    :class:`RtrsActiveSonarSimulatorArray` get the per-element inbound transfer functions
    directly from rtrs's own 3-D ray tracing *and*, via acoustic reciprocity, a stand-in for
    the single-point outbound leg (transmit modelled from the array reference, no TX
    beamforming -- matching :class:`~bluepebble.simulator.active_bellhop.
    BellhopActiveSonarSimulatorArray`'s convention) from the same ray-tracing pass, instead of
    a second, separate call.
    """
    combined = np.hstack([np.asarray(ref_pos).reshape(3, 1), np.asarray(elem_pos)])
    return _PointReceiver(
        array=ArrayState(
            num_sensors=combined.shape[1],
            state_vector=StateVectors(combined),
            ref_state_vector=ref_pos,
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


class RtrsActiveSonarSimulatorArray(Base):
    """Monostatic active sonar simulator using RTRS two-hop Fourier synthesis, for arrays.

    Like :class:`RtrsActiveSonarSimulatorOmni`, but the platform exposes multiple receive
    elements (``platform.array``) instead of a single hydrophone point, and per-element
    structure comes directly from rtrs's own 3-D ray tracing rather than from a far-field
    plane-wave delay/phase correction applied afterward (contrast
    :class:`~bluepebble.simulator.active_bellhop.BellhopActiveSonarSimulatorArray`, which must
    approximate the cross-element timing because 2-D Bellhop has no azimuth resolution to
    begin with). Each element gets its own genuinely ray-traced transfer function and delay,
    so near-field amplitude/multipath differences across the array would show up here if the
    array were large enough for them to matter.

    For each ping, target:

    1. The outbound leg (transmit, modelled from the array reference point only -- no TX
       beamforming, matching :class:`~bluepebble.simulator.active_bellhop.
       BellhopActiveSonarSimulatorArray`'s convention) and the inbound leg (receive, genuinely
       per-element) are obtained together. With ``use_reciprocity=True`` (default) this is a
       **single** ``propagate_spectrum`` call per target: source = target, receiver =
       ``[array reference point] + [every dome element]`` (see
       :func:`_point_and_array_receiver`). Row 0 of the result is ``H(target -> reference)``,
       which by acoustic reciprocity stands in for the outbound ``H(reference -> target)``
       leg; rows 1: are the real per-element inbound transfer functions, needed anyway and
       requiring no reciprocity assumption since this is already the true physical direction
       of travel (scattered energy radiates from the target to the array). With
       ``use_reciprocity=False``, two independent calls are made instead: a true outbound
       call (source = array reference, receiver = target point) and a true inbound call
       (source = target, receiver = the array only, no extra reference-point row).
    2. Round-trip transfer per element: ``H(f) = H_out(f) * H_in_i(f)``. Each element's own
       round-trip delay (``tau_out + tau_in_i``) is already embedded in that product's phase
       (rtrs's own per-receiver earliest-arrival delay), so -- as in
       :class:`RtrsActiveSonarSimulatorOmni` -- no separate sample-shift step is needed; it is
       only used to decide whether an element's echo falls inside the receive window.
    3. Target scattering ``S(f)`` is applied once per round trip (shared across elements, since
       there is no per-element scattering model). Hull baffling, when the array reports
       ``element_normals``, scales the outbound leg by the array's
       :meth:`~bluepebble.sensors.base.ArrayState.transmit_gain` and each element's inbound
       leg by its :meth:`~bluepebble.sensors.base.ArrayState.element_gains`, as in
       :class:`~bluepebble.simulator.active_bellhop.BellhopActiveSonarSimulatorArray`.
    4. Contributions from all targets are summed per element in the frequency domain and
       inverse-FFT'd once per ping (vectorised across elements) to produce the received
       ``(num_sensors, n_receive)`` waveform.

    Parameters
    ----------
    platform : BowArraySensor
        Array platform exposing ``.array`` (``ArrayState``) as transmitter/receiver.
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
        If ``True`` (default), get the outbound leg from the same call that provides the
        per-element inbound legs, via acoustic reciprocity. If ``False``, fall back to two
        independent calls (see above).
    noise_model : RandomSignal, optional
        Optional ambient noise model added to the raw hydrophone signal each ping.

    """

    platform: BowArraySensor = Property(
        doc="Array platform exposing .array (ArrayState) as transmitter/receiver"
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
        doc="Get the outbound leg via acoustic reciprocity from the same call as the "
        "per-element inbound legs, instead of a separate call",
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
        """Yield one ping's per-element received waveforms per timestamp.

        Does not advance ``platform.host`` itself: for a moving array, the caller must have
        already moved the host (e.g. via ``host.move(timestamp)`` in a loop) to every ping
        timestamp before iterating this generator -- see
        :meth:`~bluepebble.simulator.active_bellhop.BellhopActiveSonarSimulatorArray.
        sensor_data_gen` for the full rationale, which applies identically here.

        Yields
        ------
        tuple[datetime, set[ActiveSonarSensorData]]
            Ping timestamp and a one-element set containing the sensor data, whose
            ``received_waveform`` has shape ``(num_sensors, n_receive)``.

        """
        pulse = self.signal.generate()
        n_receive = int(
            (self.signal.duration_s + self.receive_duration_s) * self.signal.sampling_rate_hz
        )

        for timestamp in sorted(self.ping_timestamps):
            received = self._simulate_ping(timestamp, pulse, n_receive)
            if self.noise_model is not None:
                noise = self.noise_model.generate(
                    num_sensors=received.shape[0], num_samples=n_receive
                )
                received = received + noise
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
        """Compute the received waveforms (one row per element) for a single ping."""
        fs = self.signal.sampling_rate_hz
        pm = self._resolved_target_position_mapping()

        X = np.fft.fft(pulse, n_receive)
        native_freqs_hz = np.fft.fftfreq(n_receive, d=1.0 / fs)
        mask = _occupied_band_mask(X, self.energy_threshold)
        freqs_hz = native_freqs_hz[mask]

        array = self.platform.get_platform_state_at(timestamp).array
        num_sensors = array.num_sensors
        ref_pos = array.ref_state_vector

        R_total = np.zeros((num_sensors, n_receive), dtype=np.complex128)

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

            range_m = float(np.linalg.norm(
                np.asarray(ref_pos).flatten()[:2] - np.asarray(target_pos).flatten()[:2]
            ))
            if range_m < 1.0:
                continue  # degenerate case: target co-located with array reference

            H_out, tau_out, H_in, tau_in = self._round_trip_channels(
                array, ref_pos, timestamp, target_pos, freqs_hz
            )

            round_trip_delay_s = tau_out + tau_in  # (num_sensors,)
            in_window = round_trip_delay_s * fs < n_receive
            if not np.any(in_window):
                continue  # every element's echo would arrive after the receive window ends

            S = np.zeros(mask.shape, dtype=np.complex128)
            S[mask] = self.scattering_model.scatter(native_freqs_hz[mask])

            # Hull baffling on both legs. The direction uses the signed world target depth,
            # matching the array's own world-frame normals.
            to_target = np.array(
                [float(tv[pm[0], 0]), float(tv[pm[1], 0]), float(tv[pm[2], 0])]
            ) - np.asarray(ref_pos, dtype=float).ravel()
            baffle_gain = array.transmit_gain(to_target) * array.element_gains(to_target)

            H_round = H_out[np.newaxis, :] * H_in  # (num_sensors, n_occupied)
            H_round *= baffle_gain[:, np.newaxis]
            H_round[~in_window, :] = 0.0

            H_native = np.zeros((num_sensors, native_freqs_hz.shape[0]), dtype=np.complex128)
            H_native[:, mask] = H_round

            R_total += X[np.newaxis, :] * H_native * S[np.newaxis, :]

        return np.fft.ifft(R_total, axis=1)

    def _round_trip_channels(
        self,
        array: ArrayState,
        ref_pos: StateVector,
        timestamp: datetime,
        target_pos: StateVector,
        freqs_hz: FloatArray,
    ) -> tuple[ComplexArray, float, ComplexArray, FloatArray]:
        """Return ``(H_out, tau_out, H_in, tau_in)`` for one target.

        ``H_in``/``tau_in`` have shape ``(num_sensors,)`` / ``(num_sensors, n_freqs)``
        -- one genuine ray-traced transfer function and delay per dome element.
        """
        target_source = _point_state(target_pos, timestamp)

        if self.use_reciprocity:
            merged = _point_and_array_receiver(ref_pos, array.state_vector)
            H_all, tau_all = self.propagation_model.propagate_spectrum(
                merged, target_source, freqs_hz
            )
            tau_all = np.asarray(tau_all).reshape(-1)
            return H_all[0], float(tau_all[0]), H_all[1:], tau_all[1:]

        buoy_receiver = _point_receiver(ref_pos)
        H_out_arr, tau_out_arr = self.propagation_model.propagate_spectrum(
            buoy_receiver, target_source, freqs_hz
        )
        H_out = H_out_arr[0]
        tau_out = float(np.asarray(tau_out_arr).reshape(-1)[0])

        array_receiver = _PointReceiver(array=array)
        H_in, tau_in = self.propagation_model.propagate_spectrum(
            array_receiver, target_source, freqs_hz
        )
        return H_out, tau_out, H_in, np.asarray(tau_in).reshape(-1)
