"""Active sonar simulator using the Bellhop arrivals (eigenray) method."""

from collections.abc import Iterator
from datetime import datetime
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
from numpy.typing import NDArray
from stonesoup.base import Base, Property

from ..models.propagation import BellhopArrivalsModel
from ..platform.sonobuoy import OmniSonobuoyPlatform
from ..sensors.bow_array import BowArraySensor
from ..signal.active import ActiveSignal
from ..signal.random import RandomSignal
from ..types.sensordata import ActiveSonarSensorData

if TYPE_CHECKING:
    from stonesoup.types.groundtruth import GroundTruthPath
    from stonesoup.types.state import State

ComplexArray: TypeAlias = NDArray[np.complexfloating[Any, Any]]
SensorBatch: TypeAlias = tuple[datetime, set[ActiveSonarSensorData]]


def _target_state_at(
    path: "GroundTruthPath", timestamp: datetime
) -> "State | None":
    for state in path:
        if state.timestamp == timestamp:
            return state
    return None


class BellhopActiveSonarSimulatorOmni(Base):
    """Monostatic active sonar simulator using Bellhop eigenrays.

    Simulates a single omnidirectional sonobuoy that transmits a pulse and
    receives echoes from one or more targets. For each ping, the simulator:

    1. Runs Bellhop in arrivals mode twice per target (out-path and back-path).
    2. Combines eigenray pairs: each pair produces a shifted, scaled copy of
       the transmit pulse.
    3. Accumulates all contributions into the received waveform.

    Parameters
    ----------
    platform : OmniSonobuoyPlatform
        The sonobuoy acting as both transmitter and receiver.
    propagation_model : BellhopArrivalsModel
        Bellhop arrivals model for eigenray computation.
    signal : ActiveSignal
        Transmit waveform (CWSignal or LFMSignal).
    ground_truth_paths : list[GroundTruthPath]
        Target tracks. Each path must provide a state at every ping timestamp.
    target_strength_db : float
        Target strength in dB (scalar reflectivity; same value for all targets).
    ping_timestamps : list[datetime]
        Ordered list of times at which pings are transmitted.
    target_position_mapping : list[int]
        Indices into the target state vector that give ``[x, y, z]``.
        Defaults to ``[0, 2, 4]`` for a 6-D kinematic state
        ``[x, vx, y, vy, z, vz]``.
    receive_duration_s : float
        Listening window after pulse end in seconds. The receive buffer is
        ``signal.duration_s + receive_duration_s`` long.
    amplitude_cutoff : float
        Eigenrays with amplitude below ``amplitude_cutoff × max_amplitude``
        are discarded before combination. Default matches the MATLAB reference
        (``5e-3``).

    """

    platform: OmniSonobuoyPlatform = Property(
        doc="Omnidirectional sonobuoy (transmitter and receiver)"
    )
    propagation_model: BellhopArrivalsModel = Property(
        doc="Bellhop arrivals model for eigenray computation"
    )
    signal: ActiveSignal = Property(
        doc="Transmit waveform (CWSignal or LFMSignal)"
    )
    ground_truth_paths: list["GroundTruthPath"] = Property(
        doc="Target ground-truth tracks"
    )
    target_strength_db: float = Property(
        doc="Target strength in dB (scalar reflectivity)"
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
    amplitude_cutoff: float = Property(
        default=5e-3,
        doc="Fraction of the maximum eigenray amplitude below which rays are discarded",
    )
    noise_model: RandomSignal | None = Property(
        default=None,
        doc="Optional ambient noise model added to the raw hydrophone signal each ping",
    )

    def __post_init__(self) -> None:
        if self.target_position_mapping is None:
            self._property_target_position_mapping = [0, 2, 4]

    def sensor_data_gen(self) -> Iterator[SensorBatch]:
        """Yield one ping's received waveform per timestamp.

        Yields
        ------
        tuple[datetime, set[ActiveSonarSensorData]]
            Ping timestamp and a one-element set containing the sensor data.

        """
        pulse = self.signal.generate()
        target_amplitude = float(np.sqrt(10.0 ** (self.target_strength_db / 10.0)))
        n_receive = int(
            (self.signal.duration_s + self.receive_duration_s) * self.signal.sampling_rate_hz
        )

        for timestamp in sorted(self.ping_timestamps):
            received = self._simulate_ping(timestamp, pulse, target_amplitude, n_receive)
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
        target_amplitude: float,
        n_receive: int,
    ) -> ComplexArray:
        """Compute the received waveform for a single ping."""
        received = np.zeros(n_receive, dtype=np.complex128)

        fs = self.signal.sampling_rate_hz
        carrier_hz = self.signal.carrier_frequency_hz
        pm = self.target_position_mapping

        buoy_pos = self.platform.position_3d.flatten()
        sx, sy = float(buoy_pos[0]), float(buoy_pos[1])
        sd = float(buoy_pos[2])  # positive hydrophone depth

        for target_path in self.ground_truth_paths:
            state = _target_state_at(target_path, timestamp)
            if state is None:
                continue

            tv = state.state_vector
            tx = float(tv[pm[0], 0])
            ty = float(tv[pm[1], 0])
            td = abs(float(tv[pm[2], 0]))  # positive depth for Bellhop

            range_m = float(np.sqrt((sx - tx) ** 2 + (sy - ty) ** 2))
            if range_m < 1.0:
                continue  # degenerate case: target co-located with sonobuoy

            out_rays = self.propagation_model.compute_eigenrays(sd, td, range_m, carrier_hz)
            back_rays = self.propagation_model.compute_eigenrays(td, sd, range_m, carrier_hz)

            if not out_rays or not back_rays:
                continue

            out_rays = _apply_cutoff(out_rays, self.amplitude_cutoff)
            back_rays = _apply_cutoff(back_rays, self.amplitude_cutoff)

            n_pulse = len(pulse)
            for r_out in out_rays:
                for r_back in back_rays:
                    total_delay_s = r_out.delay_s + r_back.delay_s
                    total_amp = r_out.amplitude * r_back.amplitude * target_amplitude

                    n_shift = int(total_delay_s * fs)
                    if n_shift >= n_receive:
                        continue

                    frac_delay_s = total_delay_s - n_shift / fs
                    phase = np.exp(-2j * np.pi * carrier_hz * frac_delay_s)

                    copy_len = min(n_pulse, n_receive - n_shift)
                    received[n_shift : n_shift + copy_len] += (
                        total_amp * phase * pulse[:copy_len]
                    )

        return received


def _apply_cutoff(rays: list, cutoff_fraction: float) -> list:
    """Return rays whose amplitude exceeds ``cutoff_fraction x max_amplitude``."""
    if not rays:
        return rays
    max_amp = max(r.amplitude for r in rays)
    threshold = cutoff_fraction * max_amp
    return [r for r in rays if r.amplitude >= threshold]


class BellhopActiveSonarSimulatorArray(Base):
    """Monostatic active sonar simulator using Bellhop eigenrays, for multi-element arrays.

    Like :class:`BellhopActiveSonarSimulatorOmni`, but the platform exposes multiple
    receive elements (``platform.array``) instead of a single hydrophone point.
    Eigenrays are computed ONCE per target per ping from the array's *reference*
    position — not once per element — since the dome aperture is tiny compared to
    typical target ranges (running full Bellhop per element would be
    ``num_sensors``x the ray-tracing cost per ping for no real gain). Per-element
    differences are then applied as a far-field plane-wave delay/phase correction
    relative to the reference element.

    This means the receive elements only differ by delay/phase, not by amplitude or
    multipath structure — the whole dome shares one set of eigenrays. Worth keeping
    in mind before relying on this for anything where near-field effects matter.

    Parameters
    ----------
    platform : BowArraySensor
        The array acting as both transmitter and receiver.
    propagation_model : BellhopArrivalsModel
        Bellhop arrivals model for eigenray computation.
    signal : ActiveSignal
        Transmit waveform (CWSignal or LFMSignal).
    ground_truth_paths : list[GroundTruthPath]
        Target tracks. Each path must provide a state at every ping timestamp.
    target_strength_db : float
        Target strength in dB (scalar reflectivity; same value for all targets).
    ping_timestamps : list[datetime]
        Ordered list of times at which pings are transmitted.
    target_position_mapping : list[int]
        Indices into the target state vector that give ``[x, y, z]``.
        Defaults to ``[0, 2, 4]`` for a 6-D kinematic state
        ``[x, vx, y, vy, z, vz]``.
    receive_duration_s : float
        Listening window after pulse end in seconds. The receive buffer is
        ``signal.duration_s + receive_duration_s`` long.
    amplitude_cutoff : float
        Eigenrays with amplitude below ``amplitude_cutoff × max_amplitude``
        are discarded before combination. Default matches the MATLAB reference
        (``5e-3``).

    """

    platform: BowArraySensor = Property(
        doc="Array platform exposing .array (ArrayState) as transmitter/receiver"
    )
    propagation_model: BellhopArrivalsModel = Property(
        doc="Bellhop arrivals model for eigenray computation"
    )
    signal: ActiveSignal = Property(
        doc="Transmit waveform (CWSignal or LFMSignal)"
    )
    ground_truth_paths: list["GroundTruthPath"] = Property(
        doc="Target ground-truth tracks"
    )
    target_strength_db: float = Property(
        doc="Target strength in dB (scalar reflectivity)"
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
    amplitude_cutoff: float = Property(
        default=5e-3,
        doc="Fraction of the maximum eigenray amplitude below which rays are discarded",
    )
    noise_model: RandomSignal | None = Property(
        default=None,
        doc="Optional ambient noise model added to the raw hydrophone signal each ping",
    )

    def __post_init__(self) -> None:
        if self.target_position_mapping is None:
            self._property_target_position_mapping = [0, 2, 4]

    def sensor_data_gen(self) -> Iterator[SensorBatch]:
        """Yield one ping's per-element received waveforms per timestamp.

        Unlike the older self-driving behaviour, this no longer advances ``platform.host``
        itself: for a moving array, the caller must have already moved the host (e.g. via
        ``host.move(timestamp)`` in a loop, or a shared scenario driver) to every ping
        timestamp before iterating this generator, since a single host may now carry
        sensors driven by other simulators too. Per-ping geometry is read via
        ``platform.get_platform_state_at(timestamp)``, which looks up the host's recorded
        state for that exact timestamp -- not wherever the host ended up if it was already
        driven past it -- so it is safe to drive the host through every ping timestamp
        upfront, not just one step ahead of this generator. A host that is never moved (a
        stationary installation) needs no such driving and works unchanged.

        Yields
        ------
        tuple[datetime, set[ActiveSonarSensorData]]
            Ping timestamp and a one-element set containing the sensor data, whose
            ``received_waveform`` has shape ``(num_sensors, n_receive)`` (unlike the
            Omni simulator's 1-D waveform).

        """
        pulse = self.signal.generate()
        target_amplitude = float(np.sqrt(10.0 ** (self.target_strength_db / 10.0)))
        n_receive = int(
            (self.signal.duration_s + self.receive_duration_s) * self.signal.sampling_rate_hz
        )

        for timestamp in sorted(self.ping_timestamps):
            received = self._simulate_ping(timestamp, pulse, target_amplitude, n_receive)
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
        target_amplitude: float,
        n_receive: int,
    ) -> ComplexArray:
        """Compute the received waveforms (one row per element) for a single ping."""
        array = self.platform.get_platform_state_at(timestamp).array
        num_sensors = array.num_sensors
        received = np.zeros((num_sensors, n_receive), dtype=np.complex128)

        ref_pos = array.ref_state_vector.flatten()  # (3,) world frame: x, y, z
        elem_pos = array.state_vector  # (3, num_sensors) world frame

        fs = self.signal.sampling_rate_hz
        carrier_hz = self.signal.carrier_frequency_hz
        pm = self.target_position_mapping

        sx, sy = float(ref_pos[0]), float(ref_pos[1])
        # abs() as in OmniSonobuoyPlatform.position_3d / the target's td below: world z sign
        # convention (up-positive vs down-positive) is whatever the caller's state vectors use,
        # so normalise to a positive depth magnitude the same way the target depth already is.
        sd = abs(float(ref_pos[2]))

        for target_path in self.ground_truth_paths:
            state = _target_state_at(target_path, timestamp)
            if state is None:
                continue

            tv = state.state_vector
            tx = float(tv[pm[0], 0])
            ty = float(tv[pm[1], 0])
            td = abs(float(tv[pm[2], 0]))  # positive depth for Bellhop

            range_m = float(np.sqrt((sx - tx) ** 2 + (sy - ty) ** 2))
            if range_m < 1.0:
                continue  # degenerate case: target co-located with array reference

            # Eigenrays computed once from the array reference position, then reused
            # for every element via the per-element correction below.
            out_rays = self.propagation_model.compute_eigenrays(sd, td, range_m, carrier_hz)
            back_rays = self.propagation_model.compute_eigenrays(td, sd, range_m, carrier_hz)

            if not out_rays or not back_rays:
                continue

            out_rays = _apply_cutoff(out_rays, self.amplitude_cutoff)
            back_rays = _apply_cutoff(back_rays, self.amplitude_cutoff)

            # Per-element far-field (plane-wave) correction. Uses the true signed 3-D
            # geometry (not the abs()-depth used for the 2-D Bellhop calls above), since
            # ref_pos/elem_pos/target share one consistent world frame. Valid because
            # dome_radius_m << range_m; note 2-D Bellhop has no azimuth resolution, so this
            # only adds the geometric delay difference, not per-element refraction/multipath.
            target_xyz = np.array([tx, ty, float(tv[pm[2], 0])])
            to_target = target_xyz - ref_pos
            range_3d_m = float(np.linalg.norm(to_target))
            if range_3d_m < 1.0:
                continue  # degenerate case: target co-located with array reference
            unit_to_target = to_target / range_3d_m

            offsets = elem_pos - ref_pos[:, np.newaxis]  # (3, num_sensors)
            # Element i is `offsets[:, i] @ unit_to_target` closer to the target than the
            # reference, so its round-trip leg is shorter by that much -> earlier arrival.
            extra_delay_s = -(offsets.T @ unit_to_target) / self.platform.NOMINAL_SOUND_SPEED

            n_pulse = len(pulse)
            for r_out in out_rays:
                for r_back in back_rays:
                    # Outbound leg (r_out) is common to all elements: transmit is modelled
                    # from the array reference point, not per-element (no TX beamforming).
                    base_delay_s = r_out.delay_s + r_back.delay_s
                    total_amp = r_out.amplitude * r_back.amplitude * target_amplitude

                    total_delay_s = base_delay_s + extra_delay_s  # (num_sensors,)
                    n_shift = (total_delay_s * fs).astype(int)
                    frac_delay_s = total_delay_s - n_shift / fs
                    phase = np.exp(-2j * np.pi * carrier_hz * frac_delay_s)

                    for i in range(num_sensors):
                        if n_shift[i] >= n_receive:
                            continue
                        copy_len = min(n_pulse, n_receive - n_shift[i])
                        received[i, n_shift[i] : n_shift[i] + copy_len] += (
                            total_amp * phase[i] * pulse[:copy_len]
                        )

        return received


class BellhopActiveSonarSimulatorArrayPerElement(Base):
    """Monostatic active sonar simulator using Bellhop eigenrays, per receive element.

    Like :class:`BellhopActiveSonarSimulatorArray`, but eigenrays are computed
    separately for each receive element from its own true position, on both the
    out-path and back-path, instead of once from the array reference position
    with a far-field plane-wave delay/phase correction applied afterward. This
    captures real near-field amplitude and multipath differences between
    elements, at the cost of ``num_sensors``x the Bellhop calls per target per
    ping compared to :class:`BellhopActiveSonarSimulatorArray`.

    Parameters
    ----------
    platform : BowArraySensor
        The array acting as both transmitter and receiver.
    propagation_model : BellhopArrivalsModel
        Bellhop arrivals model for eigenray computation.
    signal : ActiveSignal
        Transmit waveform (CWSignal or LFMSignal).
    ground_truth_paths : list[GroundTruthPath]
        Target tracks. Each path must provide a state at every ping timestamp.
    target_strength_db : float
        Target strength in dB (scalar reflectivity; same value for all targets).
    ping_timestamps : list[datetime]
        Ordered list of times at which pings are transmitted.
    target_position_mapping : list[int]
        Indices into the target state vector that give ``[x, y, z]``.
        Defaults to ``[0, 2, 4]`` for a 6-D kinematic state
        ``[x, vx, y, vy, z, vz]``.
    receive_duration_s : float
        Listening window after pulse end in seconds. The receive buffer is
        ``signal.duration_s + receive_duration_s`` long.
    amplitude_cutoff : float
        Eigenrays with amplitude below ``amplitude_cutoff × max_amplitude``
        are discarded before combination. Default matches the MATLAB reference
        (``5e-3``).

    """

    platform: BowArraySensor = Property(
        doc="Array platform exposing .array (ArrayState) as transmitter/receiver"
    )
    propagation_model: BellhopArrivalsModel = Property(
        doc="Bellhop arrivals model for eigenray computation"
    )
    signal: ActiveSignal = Property(
        doc="Transmit waveform (CWSignal or LFMSignal)"
    )
    ground_truth_paths: list["GroundTruthPath"] = Property(
        doc="Target ground-truth tracks"
    )
    target_strength_db: float = Property(
        doc="Target strength in dB (scalar reflectivity)"
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
    amplitude_cutoff: float = Property(
        default=5e-3,
        doc="Fraction of the maximum eigenray amplitude below which rays are discarded",
    )
    noise_model: RandomSignal | None = Property(
        default=None,
        doc="Optional ambient noise model added to the raw hydrophone signal each ping",
    )

    def __post_init__(self) -> None:
        if self.target_position_mapping is None:
            self._property_target_position_mapping = [0, 2, 4]

    def sensor_data_gen(self) -> Iterator[SensorBatch]:
        """Yield one ping's per-element received waveforms per timestamp.

        See :meth:`BellhopActiveSonarSimulatorArray.sensor_data_gen` — behaviour (including
        the requirement that the caller drive ``platform.host`` to each ping timestamp
        beforehand) is identical; only the per-element eigenray computation in
        :meth:`_simulate_ping` differs.

        Yields
        ------
        tuple[datetime, set[ActiveSonarSensorData]]
            Ping timestamp and a one-element set containing the sensor data, whose
            ``received_waveform`` has shape ``(num_sensors, n_receive)``.

        """
        pulse = self.signal.generate()
        target_amplitude = float(np.sqrt(10.0 ** (self.target_strength_db / 10.0)))
        n_receive = int(
            (self.signal.duration_s + self.receive_duration_s) * self.signal.sampling_rate_hz
        )

        for timestamp in sorted(self.ping_timestamps):
            received = self._simulate_ping(timestamp, pulse, target_amplitude, n_receive)
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
        target_amplitude: float,
        n_receive: int,
    ) -> ComplexArray:
        """Compute the received waveforms (one row per element) for a single ping.

        Unlike :class:`BellhopActiveSonarSimulatorArray`, eigenrays are computed
        independently for every element (``num_sensors`` x 2 Bellhop calls per
        target here, vs. 2 total there), so no far-field correction is needed:
        each element's own out/back delays and amplitudes already reflect its
        true position.
        """
        array = self.platform.get_platform_state_at(timestamp).array
        num_sensors = array.num_sensors
        received = np.zeros((num_sensors, n_receive), dtype=np.complex128)
        elem_pos = array.state_vector  # (3, num_sensors) world frame

        fs = self.signal.sampling_rate_hz
        carrier_hz = self.signal.carrier_frequency_hz
        pm = self.target_position_mapping
        n_pulse = len(pulse)

        for target_path in self.ground_truth_paths:
            state = _target_state_at(target_path, timestamp)
            if state is None:
                continue

            tv = state.state_vector
            tx = float(tv[pm[0], 0])
            ty = float(tv[pm[1], 0])
            td = abs(float(tv[pm[2], 0]))  # positive depth for Bellhop

            for i in range(num_sensors):
                ex, ey = float(elem_pos[0, i]), float(elem_pos[1, i])
                ed = abs(float(elem_pos[2, i]))  # positive depth for Bellhop

                range_m = float(np.sqrt((ex - tx) ** 2 + (ey - ty) ** 2))
                if range_m < 1.0:
                    continue  # degenerate case: target co-located with this element

                out_rays = self.propagation_model.compute_eigenrays(ed, td, range_m, carrier_hz)
                back_rays = self.propagation_model.compute_eigenrays(td, ed, range_m, carrier_hz)

                if not out_rays or not back_rays:
                    continue

                out_rays = _apply_cutoff(out_rays, self.amplitude_cutoff)
                back_rays = _apply_cutoff(back_rays, self.amplitude_cutoff)

                for r_out in out_rays:
                    for r_back in back_rays:
                        total_delay_s = r_out.delay_s + r_back.delay_s
                        total_amp = r_out.amplitude * r_back.amplitude * target_amplitude

                        n_shift = int(total_delay_s * fs)
                        if n_shift >= n_receive:
                            continue

                        frac_delay_s = total_delay_s - n_shift / fs
                        phase = np.exp(-2j * np.pi * carrier_hz * frac_delay_s)

                        copy_len = min(n_pulse, n_receive - n_shift)
                        received[i, n_shift : n_shift + copy_len] += (
                            total_amp * phase * pulse[:copy_len]
                        )

        return received

