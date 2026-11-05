r"""
====================================
Tracking Targets with Bistatic Sonar
====================================

This example pings from a ship's bow array and listens on four arrays at once. One tracker fuses
all four arrays' detections.

- **Bow**: the dome that transmits, which also receives.
- **Port** and **Starboard**: a flank array on each side of the hull.
- **Towed**: a line array on a 150 m cable from a tow point 60 m aft. Its first sensor trails
  about 184 m behind the bow, 90 m deep.

Only the bow array transmits. For the other three arrays, each echo travels from the bow to the
target and back to a different array, so they are bistatic. Each bistatic detection is converted
to a range from its receiving array in three steps.

1. The echo's delay gives the range sum :math:`L = R_\text{tx} + R_\text{rx}`, the target's
   range from the transmitter plus its range from the receiver.
2. That places the target on an ellipse with the transmitter and receiver at its foci.
3. The receiving array's bearing picks the point on the ellipse, at range

   .. math::

       R_\text{rx} = \frac{L^2 - b^2}{2 (L - b \cos\psi)}

   from the receiver. Here :math:`b` is the transmitter-receiver baseline and :math:`\psi` is
   the angle between the target's bearing and the baseline, both seen from the receiver.

The tracker then gets a bearing and range measurement from the receiving array.

The towed array is a line, so it cannot tell a bearing from its mirror image about the array
axis. Each towed detection is resolved by checking which of its two images lies near a bow or
flank detection on the same ping.

This builds on :ref:`sphx_glr_auto_examples_locating_targets_with_a_bow_array.py`, which covers
beamforming and grouping detections across beams.

Assumptions
-----------
- Constant 1500 m/s sound speed over a flat 1000 m seabed matched to the water, so it reflects
  nothing.
- Each target is a point scatterer with a 15 dB target strength. There is no reverberation, and
  no direct blast from the bow array to the receivers.
- Geometry is frozen during each ping, so there is no Doppler.
- The bistatic conversion is horizontal, so it ignores the 100-150 m depth differences between
  the targets and the arrays. That lengthens each converted range by 1.5-6 m.
"""  # noqa: D205, D212, D400, D415

# %%
# Imports
# -------

from datetime import datetime, timedelta

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from stonesoup.dataassociator.neighbour import GNNWith2DAssignment
from stonesoup.deleter.error import CovarianceBasedDeleter
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.initiator.simple import MultiMeasurementInitiator
from stonesoup.measures import Mahalanobis
from stonesoup.models.measurement.nonlinear import CartesianToBearingRange
from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel,
    ConstantVelocity,
)
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.types.angle import Bearing
from stonesoup.types.array import StateVector
from stonesoup.types.detection import Detection
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.types.state import GaussianState
from stonesoup.updater.kalman import ExtendedKalmanUpdater

from bluepebble import set_seed
from bluepebble.arrays import BowArray, FlankArray, TowedArray
from bluepebble.detector import (
    ActiveSonarDetector,
    CACFARDetector,
    NoiseCalibrator,
    matched_filter_power,
    matched_filter_scans_from_sensor_data,
)
from bluepebble.models import ConstantTargetStrength
from bluepebble.models.environment import Constant, FlatBathymetry
from bluepebble.models.propagation import BellhopAcousticPropagationModel
from bluepebble.platform import SonarPlatform
from bluepebble.signal import LFMSignal
from bluepebble.signal.random import WhiteNoiseSignal
from bluepebble.sigproc import DelayAndSumBeamformer, SteeringCalculator, parabolic_peak_offset
from bluepebble.simulator import ActiveSonarSimulator

# %%
# Configuration
# -------------

set_seed(7)

START = datetime(2026, 1, 1)
NUM_PINGS = 6
PING_TIMES = [START + i * timedelta(seconds=15) for i in range(NUM_PINGS)]

SOUND_SPEED_MPS = 1500.0
WATER_DEPTH_M = 1000.0
SAMPLING_RATE_HZ = 8000
FREQ_MIN_HZ, FREQ_MAX_HZ = 1500.0, 2500.0
SOURCE_LEVEL_DB = 200.0
TARGET_STRENGTH_DB = 15.0
NOISE_LEVEL_DB = 72.0
LISTEN_DURATION_S = 4.0
TARGET_PFA = 1e-7
EDGE_RAD = np.radians(5.0)
MERGE_DISTANCE_M = 150.0

# %%
# Ship and arrays
# ---------------
#
# The ship steams along +x at 3 m/s with its reference point at the bow, 5 m down. Each array is
# mounted at an offset from that point.
#
# - **Bow**: a dome 6 m below the reference point, which transmits and receives.
# - **Port** and **Starboard**: flank arrays, 24 m long, centred 22 m aft and 4 m out on each
#   side.
# - **Towed**: 64 sensors at 0.25 m spacing, 90 m deep on a 150 m cable from a tow point 60 m
#   aft.
#
# The hull baffles the flank arrays and the dome, so sound from behind a sensor's face is 20 dB
# down.

platform = SonarPlatform(
    states=[GroundTruthState(StateVector([0.0, 3.0, 0.0, 0.0, -5.0, 0.0]), timestamp=START)],
    position_mapping=[0, 2, 4],
    transition_models=[CombinedLinearGaussianTransitionModel([ConstantVelocity(0.0)] * 3)],
    transition_times=[timedelta(hours=1)],
)
arrays = {
    "Bow": BowArray(
        dome_radius_m=4.0,
        azimuth_extent_rad=np.radians(90.0),
        elevation_extent_rad=np.radians(16.0),
        sensor_spacing_m=0.2,
        sensor_size_m=0.1,
        freq_max_hz=FREQ_MAX_HZ,
        baffle_attenuation_db=20.0,
        mounting_offset=StateVector([0.0, 0.0, -6.0]),
    ),
    **{
        name: FlankArray(
            side=side,
            aperture_length_m=24.0,
            sensor_spacing_m=0.25,
            sensor_size_m=0.1,
            freq_max_hz=FREQ_MAX_HZ,
            baffle_attenuation_db=20.0,
            mounting_offset=StateVector([-22.0, y_m, -6.0]),
        )
        for name, side, y_m in (("Port", "port", 4.0), ("Starboard", "starboard", -4.0))
    },
    "Towed": TowedArray(
        num_sensors=64,
        cable_length_m=150.0,
        sensor_spacing_m=0.25,
        array_depth_m=-90.0,
        mounting_offset=StateVector([-60.0, 0.0, 0.0]),
    ),
}
for array in arrays.values():
    platform.add_sensor(array)
for ping_time in PING_TIMES[1:]:
    platform.move(ping_time)
bow = arrays["Bow"]
for name, array in arrays.items():
    print(f"{name}: {array.num_sensors} sensors")

# %%
# Array layout
# ------------
#
# The layout shows every sensor at the first ping, seen from above. The towed array trails on its
# 150 m cable from the tow point, 60 m aft of the bow.

state0 = {name: array.get_platform_state_at(START).array for name, array in arrays.items()}
figure = go.Figure(
    [
        go.Scatter(
            x=np.ravel(state.state_vector[0]),
            y=np.ravel(state.state_vector[1]),
            mode="markers",
            marker={"size": 4},
            name=name,
        )
        for name, state in state0.items()
    ]
)
figure.update_layout(
    template="plotly_white",
    title="Array Layout (First Ping)",
    xaxis_title="x (m)",
    yaxis_title="y (m)",
    yaxis_scaleanchor="x",
).show()

# %%
# Targets
# -------
#
# Three targets sit 1.8-2.2 km from the bow and 120-160 m deep, each on its own course.
#
# - **T1**: 10 degrees to port of the bow.
# - **T2**: 75 degrees to port.
# - **T3**: 55 degrees to starboard.

TARGETS = {
    "T1": (2000.0, 10.0, 140.0, (-2.0, 1.0)),
    "T2": (1800.0, 75.0, 160.0, (3.0, 0.0)),
    "T3": (2200.0, -55.0, 120.0, (2.0, 2.0)),
}
targets = {
    name: GroundTruthPath(
        [
            GroundTruthState(
                StateVector(
                    [
                        r * np.cos(np.radians(b)) + vx * t,
                        vx,
                        r * np.sin(np.radians(b)) + vy * t,
                        vy,
                        -depth,
                        0.0,
                    ]
                ),
                timestamp=ping_time,
                metadata={"position_mapping": [0, 2, 4]},
            )
            for ping_time in PING_TIMES
            for t in [(ping_time - START).total_seconds()]
        ]
    )
    for name, (r, b, depth, (vx, vy)) in TARGETS.items()
}


def target_xyz(name, ping_time):
    """Return a target's true position at a ping."""
    state = next(s for s in targets[name] if s.timestamp == ping_time)
    return np.ravel(state.state_vector[[0, 2, 4]]).astype(float)


def reference_xyz(array, ping_time):
    """Return an array's reference point at a ping."""
    return np.ravel(array.get_platform_state_at(ping_time).array.ref_state_vector).astype(float)


# %%
# Simulation
# ----------
#
# Each array has its own simulator and its own beams, with the bow array as the transmitter.
# Each fan is measured anticlockwise from its array's forward axis, so it turns with the ship.
#
# - **Bow**: 90 degrees either side of the bow.
# - **Port** and **Starboard**: 20-160 degrees to their own side.
# - **Towed**: 15-165 degrees to port. A line array can't tell port from starboard, so these
#   beams also cover the mirror-image bearings to starboard. The 15 degrees nearest each end of
#   the array axis are left out.

SECTORS_DEG = {
    "Bow": (-90.0, 90.0),
    "Port": (20.0, 160.0),
    "Starboard": (-160.0, -20.0),
    "Towed": (15.0, 165.0),
}
pulse = LFMSignal(
    freq_min_hz=FREQ_MIN_HZ,
    freq_max_hz=FREQ_MAX_HZ,
    rise_time_s=0.03,
    duration_s=0.5,
    sampling_rate_hz=SAMPLING_RATE_HZ,
    source_level_db=SOURCE_LEVEL_DB,
)
propagation_model = BellhopAcousticPropagationModel(
    ssp=Constant(speed=SOUND_SPEED_MPS),
    bathymetry=FlatBathymetry(depth=-WATER_DEPTH_M),
    bottom_sound_speed_mps=SOUND_SPEED_MPS,
    bottom_density_g_cm3=1.0,
    bottom_attenuation_db_per_wavelength=0.0,
)


def simulator(name, ground_truth_paths, seed, ping_times=PING_TIMES):
    """Return a simulator for one receiving array, pinged from the bow."""
    start_deg, end_deg = SECTORS_DEG[name]
    return ActiveSonarSimulator(
        array=arrays[name],
        transmitter=None if name == "Bow" else bow,
        propagation_model=propagation_model,
        signal=pulse,
        scattering_models=ConstantTargetStrength(target_strength_db=TARGET_STRENGTH_DB),
        ground_truth_paths=ground_truth_paths,
        listen_duration_s=LISTEN_DURATION_S,
        noise_model=WhiteNoiseSignal(
            amplitude_upa=10 ** (NOISE_LEVEL_DB / 20),
            duration_s=LISTEN_DURATION_S,
            sampling_rate_hz=SAMPLING_RATE_HZ,
            seed=seed,
        ),
        beamformer=DelayAndSumBeamformer(domain="frequency", sampling_rate_hz=SAMPLING_RATE_HZ),
        steering_calculator=SteeringCalculator(
            ssp=Constant(speed=SOUND_SPEED_MPS),
            steering_sector_rad=(np.radians(start_deg), np.radians(end_deg)),
            num_beams=int(round((end_deg - start_deg) / 0.5)) + 1,
            steering_elevation_rad=np.radians(-3.0),
        ),
        ping_timestamps=ping_times,
    )


pings = {
    name: [
        next(iter(data))
        for _, data in simulator(name, list(targets.values()), seed).sensor_data_gen()
    ]
    for seed, name in enumerate(arrays, start=1)
}

# %%
# Matched-filter output by array
# ------------------------------
#
# Each panel shows one array's matched-filter level on the beam nearest each target at the first
# ping. It is plotted against path length :math:`c\tau`, where :math:`c` is the sound speed and
# :math:`\tau` the echo delay. The dashed lines are the true path lengths.
#
# - **Bow**: the path length is twice the range.
# - **Flank and towed**: the path length is the range sum :math:`R_\text{tx} + R_\text{rx}`
#   defined in the introduction.

figure = make_subplots(rows=len(arrays), cols=1, subplot_titles=list(arrays))
for row, (name, array) in enumerate(arrays.items(), start=1):
    ping = pings[name][0]
    power = matched_filter_power(ping)
    path_m = np.arange(power.shape[1]) * SOUND_SPEED_MPS / SAMPLING_RATE_HZ
    noise_floor = np.median(power)
    bearings = np.asarray(ping.steering_bearings_rad)
    for target in targets:
        xyz = target_xyz(target, START)
        offset = xyz - reference_xyz(array, START)
        beam = int(
            np.argmin(np.abs(np.angle(np.exp(1j * (bearings - np.arctan2(offset[1], offset[0]))))))
        )
        true_path_m = np.linalg.norm(xyz - reference_xyz(bow, START)) + np.linalg.norm(offset)
        window = (path_m > true_path_m - 40) & (path_m < true_path_m + 80)
        figure.add_trace(
            go.Scatter(
                x=path_m[window],
                y=10 * np.log10(power[beam, window] / noise_floor),
                mode="lines",
                name=target,
                showlegend=row == 1,
                line={"color": {"T1": "steelblue", "T2": "seagreen", "T3": "darkorange"}[target]},
            ),
            row=row,
            col=1,
        )
        figure.add_vline(x=true_path_m, line_dash="dash", line_color="grey", row=row, col=1)
    figure.update_yaxes(title_text="dB re median", row=row, col=1)
figure.update_xaxes(title_text="Path length (m)", row=len(arrays), col=1)
figure.update_layout(
    template="plotly_white", title="Matched-Filter Output (First Ping)", height=900
).show()

# %%
# Detection
# ---------
#
# Each array's CFAR (Constant False Alarm Rate) detector is calibrated on one noise-only ping of
# its own. Detections across beams are grouped as in the bow-array example, strongest first by
# matched-filter power. Three of the grouping thresholds are stricter.
#
# - **Range: 30 m rather than 5 m.** A beam steered away from an echo adds the sensors' copies
#   out of step. That smears the echo by up to the aperture's travel time, :math:`\pm 12` m for
#   the 24 m flank arrays.
# - **Bearing: 30 degrees rather than 10.** The dome's face spans 45 degrees either side of the
#   bow. A target beyond it, like T2 at 75 degrees, sees only a small part of the dome, so its
#   mainlobe is over 20 degrees wide.
# - **Power: 6 dB rather than 10 dB.** A weaker detection is dropped at 6 dB down, because some
#   of the dome's sidelobes, 20-30 degrees off the mainlobe, are only 6.5-9 dB down.
#
# Two real targets within 30 m of range sum, and within 30 degrees or 6 dB of each other, would
# merge.
#
# Two more rules drop leakage from targets outside a fan.
#
# - **Fan edges**: detections within 5 degrees of either edge of a fan are dropped, since a
#   target just outside it shows up there.
# - **Far sidelobes**: any detection more than 20 dB below the array's strongest on that ping is
#   dropped. This removes a target's leakage through far sidelobes, but would also lose a
#   genuinely quieter target.


def group_detections(detections, ping):
    """Keep one detection per target; return ``(bearing_rad, range_m, power_db)`` for each."""
    bearings = np.asarray(ping.steering_bearings_rad)
    fan_middle = float(np.angle(np.exp(1j * bearings[0]) + np.exp(1j * bearings[-1])))
    fan_half_width = abs(float(np.angle(np.exp(1j * (bearings[-1] - bearings[0]))))) / 2
    power = matched_filter_power(ping)
    candidates = []
    for detection in detections:
        bearing, range_m = (float(v) for v in detection.state_vector[:, 0])
        beam = int(np.argmin(np.abs(np.angle(np.exp(1j * (bearings - bearing))))))
        sample = min(
            int(round(detection.metadata["delay_s"] * SAMPLING_RATE_HZ)), power.shape[1] - 1
        )
        candidates.append((10 * np.log10(power[beam, sample]), bearing, range_m, beam, sample))
    kept, dropped = [], []
    for power_db, bearing, range_m, beam, sample in sorted(candidates, reverse=True):
        if any(
            abs(range_m - r) < 30.0
            and (abs(np.angle(np.exp(1j * (bearing - b)))) < np.radians(30) or power_db < p - 6)
            for b, r, p in kept + dropped
        ):
            continue
        if abs(np.angle(np.exp(1j * (bearing - fan_middle)))) > fan_half_width - EDGE_RAD:
            # An edge detection is likely a target outside the fan, so it still suppresses its
            # weaker neighbours.
            dropped.append((bearing, range_m, power_db))
            continue
        across_beams = power[:, sample]
        is_peak = 0 < beam < len(bearings) - 1 and across_beams[beam] >= max(
            across_beams[beam - 1], across_beams[beam + 1]
        )
        offset = parabolic_peak_offset(10 * np.log10(across_beams), beam) if is_peak else 0.0
        kept.append((bearing + offset * (bearings[1] - bearings[0]), range_m, power_db))
    strongest_db = max((p for _, _, p in kept), default=0.0)
    return [d for d in kept if d[2] >= strongest_db - 20.0]


grouped = {}
for seed, name in enumerate(arrays, start=11):
    cfar = CACFARDetector(
        num_guard_cells=40,
        num_training_cells=200,
        target_pfa=TARGET_PFA,
        circular=False,
        peak_distance=20,
    )
    noise_only = simulator(name, [], seed, ping_times=PING_TIMES[:1])
    NoiseCalibrator(cfar).calibrate_from_noise(
        matched_filter_scans_from_sensor_data(noise_only.sensor_data_gen())
    )
    detector = ActiveSonarDetector(
        detector=cfar,
        sensor_data_gen=[(ping.timestamp, {ping}) for ping in pings[name]],
        sampling_rate_hz=SAMPLING_RATE_HZ,
        sound_speed_ms=SOUND_SPEED_MPS,
        min_range_m=200.0,
    )
    grouped[name] = [
        group_detections(detections, ping)
        for (_, detections), ping in zip(detector.detections_gen(), pings[name], strict=True)
    ]
    print(f"{name}: {[len(g) for g in grouped[name]]} detections per ping")

# %%
# Flank-array baffles
# -------------------
#
# Each flank array is a line along the hull, so like the towed array it hears a target at its
# true bearing and at the mirror image across the hull. The hull baffles the far side by 20 dB,
# so a target heard on both flanks is on the side that hears it louder. A flank detection is
# dropped when the other flank has a detection that meets all three of these conditions.
#
# - **Range sum**: the same, within 30 m.
# - **Bearing**: the mirror image, within 10 degrees.
# - **Power**: at least 10 dB stronger.


def is_mirror_ghost(detection, other_side):
    """Return whether the other flank hears a detection's mirror image 10 dB louder."""
    bearing, range_m, power_db = detection
    return any(
        abs(range_m - r) < 30.0
        and abs(np.angle(np.exp(1j * (bearing + b)))) < np.radians(10)
        and p >= power_db + 10
        for b, r, p in other_side
    )


for index in range(NUM_PINGS):
    port, starboard = grouped["Port"][index], grouped["Starboard"][index]
    grouped["Port"][index] = [d for d in port if not is_mirror_ghost(d, starboard)]
    grouped["Starboard"][index] = [d for d in starboard if not is_mirror_ghost(d, port)]
for name in ("Port", "Starboard"):
    print(f"{name} after the baffle check: {[len(g) for g in grouped[name]]} detections per ping")

# %%
# Bistatic conversion
# -------------------
#
# The bow array's detections are monostatic ranges already. The others' are range sums, each
# converted to a range from its receiving array with the ellipse formula above.


def bistatic_range_m(range_sum_m, bearing_rad, receiver_xy, transmitter_xy):
    """Return the range from the receiver of a target on a bistatic ellipse, horizontally."""
    baseline = transmitter_xy - receiver_xy
    b = float(np.linalg.norm(baseline))
    cos_psi = float(np.array([np.cos(bearing_rad), np.sin(bearing_rad)]) @ baseline) / b
    return (range_sum_m**2 - b**2) / (2 * (range_sum_m - b * cos_psi))


measurements = {name: [] for name in arrays}
for name in arrays:
    for ping_time, kept in zip(PING_TIMES, grouped[name], strict=True):
        receiver_xy = reference_xyz(arrays[name], ping_time)[:2]
        transmitter_xy = reference_xyz(bow, ping_time)[:2]
        converted = []
        for bearing, range_m, power_db in kept:
            if name != "Bow":
                range_m = bistatic_range_m(range_m, bearing, receiver_xy, transmitter_xy)
            converted.append((bearing, range_m, power_db))
        measurements[name].append(converted)

# %%
# Towed-array ambiguity
# ---------------------
#
# A towed detection's bearing :math:`\theta` has a mirror image :math:`2\alpha - \theta` about
# the array axis :math:`\alpha`. Which image is kept depends on the bow and flank detections on
# the same ping.
#
# - **One image within 150 m of one of them**: that image is kept.
# - **Neither image near one**: both are kept for the tracker to sort out.


def position(array, ping_time, bearing, range_m):
    """Return the horizontal position at a bearing and range from an array's reference point."""
    return reference_xyz(array, ping_time)[:2] + range_m * np.array(
        [np.cos(bearing), np.sin(bearing)]
    )


towed = arrays["Towed"]
for index, ping_time in enumerate(PING_TIMES):
    axis = np.ravel(towed.get_platform_state_at(ping_time).array.axis)
    alpha = float(np.arctan2(axis[1], axis[0]))
    others = [
        position(arrays[name], ping_time, b, r)
        for name in ("Bow", "Port", "Starboard")
        for b, r, _ in measurements[name][index]
    ]
    resolved = []
    for bearing, range_m, power_db in measurements["Towed"][index]:
        images = [bearing, 2 * alpha - bearing]
        near = [
            image
            for image in images
            if any(
                np.linalg.norm(position(towed, ping_time, image, range_m) - xy) < 150.0
                for xy in others
            )
        ]
        resolved += [(image, range_m, power_db) for image in (near or images)]
    measurements["Towed"][index] = resolved

# %%
# Errors by array
# ---------------
#
# For each array and ping, each target's nearest measurement is compared with its true bearing
# and horizontal range from that array. Each array misses some targets, as expected.
#
# - T1, 10 degrees off the bow, is outside both flank fans and within 15 degrees of the towed
#   array's axis, so only the bow array hears it.
# - T2 and T3 are each on one side, so the far flank array's baffle hides them.
#
# - **Bearings**: within a fraction of a degree.
# - **Ranges**: errors of 2-11 m, mostly from the horizontal bistatic conversion ignoring depth,
#   plus the surface-multipath offset seen in the bow-array example.

print(f"\n{'Array':>10} {'Target':>6} {'RMS bearing (deg)':>18} {'RMS range (m)':>14}", end="")
print(f" {'Pings':>6}")
for name, array in arrays.items():
    for target in targets:
        bearing_errors, range_errors = [], []
        for index, ping_time in enumerate(PING_TIMES):
            offset = (target_xyz(target, ping_time) - reference_xyz(array, ping_time))[:2]
            true_b, true_r = np.arctan2(offset[1], offset[0]), np.hypot(*offset)
            candidates = [
                (b, r)
                for b, r, _ in measurements[name][index]
                if np.linalg.norm(
                    position(array, ping_time, b, r) - target_xyz(target, ping_time)[:2]
                )
                < 100.0
            ]
            if not candidates:
                continue
            b, r = min(candidates, key=lambda m: abs(m[1] - true_r))
            bearing_errors.append(np.degrees(np.angle(np.exp(1j * (b - true_b)))))
            range_errors.append(r - true_r)
        if bearing_errors:
            print(
                f"{name:>10} {target:>6} {np.sqrt(np.mean(np.square(bearing_errors))):>18.2f} "
                f"{np.sqrt(np.mean(np.square(range_errors))):>14.1f} {len(bearing_errors):>6}"
            )
        else:
            print(f"{name:>10} {target:>6} {'not detected':>18}")

# %%
# Tracking
# --------
#
# One tracker fuses every array's measurements by centralised, sequential measurement fusion.
#
# - **Centralised**: each array's detections go straight to the one tracker, with no tracker of
#   their own.
# - **Sequential**: within a ping the arrays update the tracks one after another, in the order
#   Bow, Port, Starboard, Towed. On each array's turn, global nearest neighbour (GNN) assignment
#   gives each track at most one of that array's detections. The update starts from the state the
#   previous array left.
#
# For independent Gaussian errors, this matches a single update with every array's measurements
# stacked together. Here the match is approximate, since each extended Kalman update
# relinearises at the latest estimate.
#
# The tracker is set up as follows.
#
# - **Measurements**: each detection becomes a bearing and range measurement from its receiving
#   array's reference point. The bearing standard deviation is set by the array's beamwidth, and
#   the range standard deviation is 15 m.
# - **Initiation**: unassociated detections start tentative tracks, confirmed after two updates.
# - **Duplicates**: several arrays hear the same target, so several tracks can start on it at
#   once. Of any two tracks within 150 m, the one with fewer updates is dropped.

BEARING_SD_DEG = {"Bow": 0.5, "Port": 0.75, "Starboard": 0.75, "Towed": 1.0}
transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.01)] * 2)
predictor = KalmanPredictor(transition_model)
updater = ExtendedKalmanUpdater(measurement_model=None)
hypothesiser = DistanceHypothesiser(predictor, updater, Mahalanobis(), missed_distance=4)
associator = GNNWith2DAssignment(hypothesiser)
deleter = CovarianceBasedDeleter(covar_trace_thresh=1e6)
initiator = MultiMeasurementInitiator(
    prior_state=GaussianState(
        StateVector([0.0, 0.0, 0.0, 0.0]), np.diag([0.0, 5.0**2, 0.0, 5.0**2])
    ),
    deleter=deleter,
    data_associator=GNNWith2DAssignment(hypothesiser),
    updater=updater,
    min_points=2,
)


def to_detection(name, ping_time, bearing, range_m):
    """Return a bearing and range detection from an array's reference point."""
    model = CartesianToBearingRange(
        ndim_state=4,
        mapping=(0, 2),
        noise_covar=np.diag([np.radians(BEARING_SD_DEG[name]) ** 2, 15.0**2]),
        translation_offset=StateVector(reference_xyz(arrays[name], ping_time)[:2]),
    )
    return Detection(
        StateVector([Bearing(bearing), range_m]),
        timestamp=ping_time,
        measurement_model=model,
        metadata={"array": name},
    )


def updates(track):
    """Return the number of measurement updates in a track."""
    return sum(1 for state in track if hasattr(state, "hypothesis"))


tracks = set()
for index, ping_time in enumerate(PING_TIMES):
    unassociated = set()
    updated = set()
    for name in arrays:
        detections = {to_detection(name, ping_time, b, r) for b, r, _ in measurements[name][index]}
        hypotheses = associator.associate(tracks, detections, ping_time)
        for track, hypothesis in hypotheses.items():
            if hypothesis:
                track.append(updater.update(hypothesis))
                updated.add(track)
        associated = {h.measurement for h in hypotheses.values() if h}
        unassociated |= detections - associated
    for track in tracks - updated:
        track.append(predictor.predict(track.state, timestamp=ping_time))
    tracks -= deleter.delete_tracks(tracks)
    tracks |= initiator.initiate(unassociated, ping_time)
    # Several arrays can start a track on the same target at the same ping. Of two tracks
    # closer than MERGE_DISTANCE_M, keep the one with more updates.
    for track in sorted(tracks, key=updates):
        if any(
            other is not track
            and updates(other) >= updates(track)
            and np.hypot(*(track.state_vector[[0, 2], 0] - other.state_vector[[0, 2], 0]))
            < MERGE_DISTANCE_M
            for other in tracks
        ):
            tracks.discard(track)

print(f"\n{len(tracks)} tracks at the last ping")
final_time = PING_TIMES[-1]
for number, track in enumerate(sorted(tracks, key=lambda t: float(t.state_vector[0, 0]))):
    xy = np.array([float(track.state_vector[0, 0]), float(track.state_vector[2, 0])])
    distances = {name: np.linalg.norm(xy - target_xyz(name, final_time)[:2]) for name in targets}
    nearest = min(distances, key=distances.get)
    used = sorted(
        {s.hypothesis.measurement.metadata["array"] for s in track if hasattr(s, "hypothesis")}
    )
    print(f"Track {number + 1}: {distances[nearest]:6.1f} m from {nearest}; arrays {used}")

# %%
# Tracks
# ------
#
# The top panel shows the ship, the true target tracks, every array's converted detections and
# the tracker's tracks. The panels below zoom in on each target.

colours = {"Bow": "black", "Port": "crimson", "Starboard": "seagreen", "Towed": "purple"}
figure = make_subplots(
    rows=2,
    cols=3,
    specs=[[{"colspan": 3}, None, None], [{}, {}, {}]],
    subplot_titles=["Overview", *targets],
    row_heights=[0.6, 0.4],
)
cells = [(1, 1)] + [(2, column) for column in range(1, 4)]
for row, col in cells:
    first = (row, col) == (1, 1)
    ship_xy = np.array(
        [[float(s.state_vector[0, 0]), float(s.state_vector[2, 0])] for s in platform.states]
    )
    figure.add_trace(
        go.Scatter(
            x=ship_xy[:, 0],
            y=ship_xy[:, 1],
            mode="lines",
            name="Ship",
            line={"color": "grey"},
            showlegend=first,
        ),
        row=row,
        col=col,
    )
    for target, path in targets.items():
        xy = np.array([[float(s.state_vector[0, 0]), float(s.state_vector[2, 0])] for s in path])
        figure.add_trace(
            go.Scatter(
                x=xy[:, 0],
                y=xy[:, 1],
                mode="lines",
                line={"color": "black", "dash": "dash"},
                name="Truth",
                showlegend=first and target == "T1",
            ),
            row=row,
            col=col,
        )
    for name, array in arrays.items():
        xy = np.array(
            [
                position(array, t, b, r)
                for t, ms in zip(PING_TIMES, measurements[name], strict=True)
                for b, r, _ in ms
            ]
        )
        if xy.size:
            figure.add_trace(
                go.Scatter(
                    x=xy[:, 0],
                    y=xy[:, 1],
                    mode="markers",
                    name=name,
                    marker={"size": 5, "color": colours[name]},
                    showlegend=first,
                ),
                row=row,
                col=col,
            )
    for number, track in enumerate(tracks):
        xy = np.array([[float(s.state_vector[0, 0]), float(s.state_vector[2, 0])] for s in track])
        figure.add_trace(
            go.Scatter(
                x=xy[:, 0],
                y=xy[:, 1],
                mode="lines+markers",
                line={"color": "gold", "width": 3},
                name="Tracks",
                showlegend=first and number == 0,
            ),
            row=row,
            col=col,
        )
for column, target in enumerate(targets, start=1):
    xy = np.array([target_xyz(target, t)[:2] for t in PING_TIMES])
    centre = xy.mean(axis=0)
    figure.update_xaxes(range=[centre[0] - 80, centre[0] + 80], row=2, col=column)
    figure.update_yaxes(range=[centre[1] - 80, centre[1] + 80], row=2, col=column)
figure.update_layout(template="plotly_white", title="Detections and Tracks", height=950).show()
