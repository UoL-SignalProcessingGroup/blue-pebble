"""The bistatic active sonar scenario shared by the paper's scripts.

Every script in this folder builds its ship, arrays, targets, water, seabeds, pulse, noise,
simulators and detector from here, so that they all simulate the same thing. Change a setting
here and it changes for all of them. Each function returns new objects, so a script can't
alter another's by mutating them.

The scenario combines the bistatic tracking example's ship, arrays and moving targets with the
Bellhop comparison's refracting water and reflective, sloping seabeds, so that the choice of
propagation model changes the echoes the arrays receive.

- **Ship**: steams along +x at 3 m/s, pinging from its bow array every 10 s.
- **Arrays**: the bow dome transmits; the bow, two flank arrays and a towed array receive.
- **Water**: Munk sound-speed profile, 300 m deep at the origin.
- **Seabeds**: flat (the control), a wedge deepening towards +y (port), or a seamount; each a
  lossless fluid half-space (1700 m/s, 1.5 g/cm³), so bottom echoes are strong.
- **Targets**: three point scatterers, 1.8-2.6 km away and 100-120 m deep, each on its own
  straight course, well above the seabed under them throughout.
"""

from datetime import datetime, timedelta

import numpy as np
from stonesoup.dataassociator.neighbour import GNNWith2DAssignment
from stonesoup.deleter.error import CovarianceBasedDeleter
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.initiator.simple import MultiMeasurementInitiator
from stonesoup.measures import Euclidean, Mahalanobis
from stonesoup.metricgenerator.ospametric import GOSPAMetric
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

from bluepebble.arrays import BowArray, FlankArray, TowedArray
from bluepebble.detector import (
    CACFARDetector,
    NoiseCalibrator,
    OSCFARDetector,
    matched_filter_scans_from_sensor_data,
)
from bluepebble.detector._theory import solve_ca_cfar_alpha, solve_os_cfar_alpha_single_look
from bluepebble.models import ConstantTargetStrength
from bluepebble.models.environment import (
    FlatBathymetry,
    Munk,
    SeamountBathymetry,
    WedgeBathymetry,
)
from bluepebble.models.propagation import (
    Bellhop3DAcousticPropagationModel,
    BellhopAcousticPropagationModel,
    SphericalAcousticPropagationModel,
)
from bluepebble.platform import SonarPlatform
from bluepebble.signal import LFMSignal
from bluepebble.signal.random import WhiteNoiseSignal
from bluepebble.sigproc import DelayAndSumBeamformer, SteeringCalculator, parabolic_peak_offset
from bluepebble.simulator import ActiveSonarSimulator

# --------------------------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------------------------

START = datetime(2026, 1, 1)
PING_INTERVAL_S = 10.0
NUM_PINGS = 40
PING_TIMES = [START + i * timedelta(seconds=PING_INTERVAL_S) for i in range(NUM_PINGS)]

# --------------------------------------------------------------------------------------------
# Acoustics
# --------------------------------------------------------------------------------------------

SAMPLING_RATE_HZ = 8000
FREQ_MIN_HZ, FREQ_MAX_HZ = 1500.0, 2500.0
CENTRE_HZ = (FREQ_MIN_HZ + FREQ_MAX_HZ) / 2
PULSE_DURATION_S = 0.5
SOURCE_LEVEL_DB = 200.0
TARGET_STRENGTH_DB = 15.0
NOISE_LEVEL_DB = 72.0
LISTEN_DURATION_S = 4.0
TARGET_PFA = 1e-7
MIN_RANGE_M = 200.0  # earlier detections are dropped (a range sum for the bistatic arrays)

# Independent noise for each array: the same in every configuration's pings, and different
# again for the noise-only calibration pings.
NOISE_SEEDS = {"Bow": 1, "Port": 2, "Starboard": 3, "Towed": 4}
CALIBRATION_SEEDS = {"Bow": 11, "Port": 12, "Starboard": 13, "Towed": 14}

# --------------------------------------------------------------------------------------------
# Water and seabeds
# --------------------------------------------------------------------------------------------

DEPTH_AT_ORIGIN_M = 300.0
WEDGE_Y_GRADIENT = -0.08  # 8 m deeper per 100 m towards +y (port)
BOTTOM = {
    "bottom_sound_speed_mps": 1700.0,
    "bottom_density_g_cm3": 1.5,
    "bottom_attenuation_db_per_wavelength": 0.0,
}
SEABED_NAMES = ("Flat", "Wedge", "Seamount")


def sound_speed_profile():
    """Return the Munk sound-speed profile: 1548 m/s at the surface, 1523 m/s at 300 m."""
    return Munk()


def seabed(name):
    """Return the seabed called ``name``: "Flat", "Wedge" or "Seamount"."""
    if name == "Flat":
        return FlatBathymetry(depth=-DEPTH_AT_ORIGIN_M)
    if name == "Wedge":
        return WedgeBathymetry(
            depth_at_origin=-DEPTH_AT_ORIGIN_M, x_gradient=0.0, y_gradient=WEDGE_Y_GRADIENT
        )
    if name == "Seamount":
        # A rounded seamount rising from the 300 m plateau to 150 m, 1.4 km off the ship's
        # track to port; its steepest slope, 500 m out, is 18%.
        return SeamountBathymetry(
            summit_position=(1200.0, 1400.0, -150.0),
            radius=1500.0,
            plateau_depth=-DEPTH_AT_ORIGIN_M,
            shape="gaussian",
            resolution=50.0,
        )
    raise ValueError(f"Unknown seabed {name!r}; expected one of {SEABED_NAMES}")


# Delays are converted to range with the sound speed at the bow array, as a sonar would.
BOW_DEPTH_M = 11.0
SOUND_SPEED_MPS = float(sound_speed_profile().calculate(BOW_DEPTH_M))

# --------------------------------------------------------------------------------------------
# Propagation configurations
# --------------------------------------------------------------------------------------------

# 2D separate planes (one 2D run per element, each in its own plane) is left out: N x 2D is
# Bellhop's standard implementation of the same approximation, so the paper uses that.
REFERENCE_MODE = "3D all"
MODES = {
    "2D all": (BellhopAcousticPropagationModel, {}),
    "2D far field": (BellhopAcousticPropagationModel, {"far_field": True}),
    "3D all": (Bellhop3DAcousticPropagationModel, {}),
    "3D far field": (Bellhop3DAcousticPropagationModel, {"far_field": True}),
    "Nx2D": (Bellhop3DAcousticPropagationModel, {"nx2d": True}),
}
MODEL_CLASSES = {cls.__name__: cls for cls, _ in MODES.values()}


def propagation_model(mode, seabed_name, settings):
    """Return configuration ``mode`` over seabed ``seabed_name`` with ray/grid ``settings``.

    ``settings`` are the model's ray and grid settings, such as those saved by
    ``bellhop_propagation_model_configuration.py``.
    """
    model_class, options = MODES[mode]
    return model_class(
        ssp=sound_speed_profile(),
        bathymetry=seabed(seabed_name),
        **BOTTOM,
        **options,
        **settings,
    )


def propagation_model_from_record(record):
    """Return a propagation model from one configuration's entry in the saved settings."""
    model_class = MODEL_CLASSES[record["model_class"]]
    return model_class(
        ssp=sound_speed_profile(),
        bathymetry=seabed(record["seabed"]),
        **BOTTOM,
        **record["options"],
        **record["settings"],
    )


# --------------------------------------------------------------------------------------------
# Ship and arrays
# --------------------------------------------------------------------------------------------

SHIP_SPEED_MPS = 3.0
ARRAY_NAMES = ("Bow", "Port", "Starboard", "Towed")
# Beam fans, anticlockwise from each array's forward axis. The towed fan also covers the mirror
# image to starboard, since a line array can't tell the two apart.
SECTORS_DEG = {
    "Bow": (-90.0, 90.0),
    "Port": (20.0, 160.0),
    "Starboard": (-160.0, -20.0),
    "Towed": (15.0, 165.0),
}
BEAM_SPACING_DEG = 0.5
STEERING_ELEVATION_DEG = -3.0


def ship(ping_times=PING_TIMES):
    """Return ``(platform, arrays)``: the ship, moved through ``ping_times``, and its arrays.

    - **Bow**: a dome 6 m below the ship's 5 m deep reference point; transmits and receives.
    - **Port** and **Starboard**: flank arrays, 24 m long, 22 m aft and 4 m out each side.
    - **Towed**: 64 sensors at 0.25 m spacing, 90 m deep on a 150 m cable from 60 m aft.

    The hull baffles the dome and flank arrays: sound from behind a sensor's face is 20 dB down.
    """
    platform = SonarPlatform(
        states=[
            GroundTruthState(
                StateVector([0.0, SHIP_SPEED_MPS, 0.0, 0.0, -5.0, 0.0]), timestamp=START
            )
        ],
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
    for ping_time in ping_times[1:]:
        platform.move(ping_time)
    return platform, arrays


def reference_xyz(array, ping_time):
    """Return an array's reference point at a ping."""
    return np.ravel(array.get_platform_state_at(ping_time).array.ref_state_vector).astype(float)


def array_axis_rad(array, ping_time):
    """Return the bearing of a line array's axis at a ping."""
    axis = np.ravel(array.get_platform_state_at(ping_time).array.axis)
    return float(np.arctan2(axis[1], axis[0]))


# --------------------------------------------------------------------------------------------
# Targets
# --------------------------------------------------------------------------------------------

# name: (range from the origin (m), bearing (deg, anticlockwise from +x), depth (m), (vx, vy))
TARGETS = {
    "T1": (2500.0, 10.0, 100.0, (1.0, 0.5)),
    "T2": (2000.0, 60.0, 120.0, (3.0, 0.0)),
    "T3": (2200.0, -50.0, 100.0, (2.0, 1.5)),
}


def targets(ping_times=PING_TIMES):
    """Return ``{name: GroundTruthPath}``, each target's true state at every ping."""
    paths = {}
    for name, (r, b, depth, (vx, vy)) in TARGETS.items():
        x0, y0 = r * np.cos(np.radians(b)), r * np.sin(np.radians(b))
        paths[name] = GroundTruthPath(
            [
                GroundTruthState(
                    StateVector([x0 + vx * t, vx, y0 + vy * t, vy, -depth, 0.0]),
                    timestamp=ping_time,
                    metadata={"position_mapping": [0, 2, 4]},
                )
                for ping_time in ping_times
                for t in [(ping_time - START).total_seconds()]
            ]
        )
    return paths


def target_xyz(path, ping_time):
    """Return a target's true position at a ping."""
    state = next(s for s in path if s.timestamp == ping_time)
    return np.ravel(state.state_vector[[0, 2, 4]]).astype(float)


def true_range_and_bearing(arrays, name, xyz, ping_time):
    """Return what array ``name`` should measure for a target at ``xyz``: (range, bearing).

    The range is the slant range for the bow array, which transmits and receives, and the
    range sum (transmitter to target to receiver) for the others. The bearing is the
    horizontal bearing from the array's reference point, in radians.
    """
    receiver = reference_xyz(arrays[name], ping_time)
    transmitter = reference_xyz(arrays["Bow"], ping_time)
    offset = xyz - receiver
    bearing = float(np.arctan2(offset[1], offset[0]))
    if name == "Bow":
        return float(np.linalg.norm(offset)), bearing
    return float(np.linalg.norm(xyz - transmitter) + np.linalg.norm(offset)), bearing


def bistatic_range_m(range_sum_m, bearing_rad, receiver_xy, transmitter_xy):
    """Return the range from the receiver of a target on a bistatic ellipse, horizontally."""
    baseline = transmitter_xy - receiver_xy
    b = float(np.linalg.norm(baseline))
    cos_psi = float(np.array([np.cos(bearing_rad), np.sin(bearing_rad)]) @ baseline) / b
    return (range_sum_m**2 - b**2) / (2 * (range_sum_m - b * cos_psi))


# --------------------------------------------------------------------------------------------
# Pulse, simulators and detector
# --------------------------------------------------------------------------------------------


def pulse():
    """Return the transmitted pulse: a 0.5 s LFM sweep over 1.5-2.5 kHz."""
    return LFMSignal(
        freq_min_hz=FREQ_MIN_HZ,
        freq_max_hz=FREQ_MAX_HZ,
        rise_time_s=0.03,
        duration_s=PULSE_DURATION_S,
        sampling_rate_hz=SAMPLING_RATE_HZ,
        source_level_db=SOURCE_LEVEL_DB,
    )


def steering_calculator(name):
    """Return array ``name``'s steering calculator: beams every 0.5 deg across its fan."""
    start_deg, end_deg = SECTORS_DEG[name]
    return SteeringCalculator(
        ssp=sound_speed_profile(),
        steering_sector_rad=(np.radians(start_deg), np.radians(end_deg)),
        num_beams=int(round((end_deg - start_deg) / BEAM_SPACING_DEG)) + 1,
        steering_elevation_rad=np.radians(STEERING_ELEVATION_DEG),
    )


def simulator(
    arrays,
    name,
    model,
    ground_truth_paths,
    ping_times,
    seed=None,
    beamformer="default",
):
    """Return a simulator for receiving array ``name``, pinged from the bow array.

    ``seed`` defaults to the array's entry in ``NOISE_SEEDS``. ``beamformer`` defaults to
    frequency-domain delay-and-sum; pass None for the sensors' data alone (beamform it later),
    or any time-preserving beamformer.
    """
    if beamformer == "default":
        beamformer = DelayAndSumBeamformer(domain="frequency", sampling_rate_hz=SAMPLING_RATE_HZ)
    return ActiveSonarSimulator(
        array=arrays[name],
        transmitter=None if name == "Bow" else arrays["Bow"],
        propagation_model=model,
        signal=pulse(),
        scattering_models=ConstantTargetStrength(target_strength_db=TARGET_STRENGTH_DB),
        ground_truth_paths=list(ground_truth_paths),
        listen_duration_s=LISTEN_DURATION_S,
        noise_model=WhiteNoiseSignal(
            amplitude_upa=10 ** (NOISE_LEVEL_DB / 20),
            duration_s=LISTEN_DURATION_S,
            sampling_rate_hz=SAMPLING_RATE_HZ,
            seed=NOISE_SEEDS[name] if seed is None else seed,
        ),
        beamformer=beamformer,
        steering_calculator=None if beamformer is None else steering_calculator(name),
        ping_timestamps=list(ping_times),
    )


NUM_GUARD_CELLS = 40
NUM_TRAINING_CELLS = 200  # each side
OS_RANK = 300  # of the 400 training cells


def cfar_detector(kind="OS", target_pfa=TARGET_PFA, calibrated=True):
    """Return a CFAR detector along range; the paper's baseline is a calibrated OS-CFAR.

    - ``kind="OS"``: ordered-statistic CFAR. Its noise estimate is the 300th smallest of 400
      training cells (200 each side, beyond 40 guard cells). Ranking the cells ignores the few
      dozen an echo occupies, so closely spaced bottom echoes don't raise each other's
      thresholds.
    - ``kind="CA"``: cell-averaging CFAR over the same window: the mean of the training cells.

    With ``calibrated=True`` the detector is set for ``target_pfa`` and must be calibrated on
    noise-only data (:func:`calibrated_cfar_detectors`). With ``calibrated=False`` its threshold
    is the textbook (nominal) one for ``target_pfa``, which assumes independent single-look
    cells: what a detector would use with no noise-only data to calibrate on.
    """
    common = {
        "num_guard_cells": NUM_GUARD_CELLS,
        "num_training_cells": NUM_TRAINING_CELLS,
        "circular": False,
        "peak_distance": 20,
    }
    num_training_total = 2 * NUM_TRAINING_CELLS
    if kind == "OS":
        alpha = solve_os_cfar_alpha_single_look(target_pfa, num_training_total, OS_RANK)
        threshold = {"target_pfa": target_pfa} if calibrated else {"threshold_factor": alpha}
        return OSCFARDetector(rank=OS_RANK, **common, **threshold)
    if kind == "CA":
        alpha = solve_ca_cfar_alpha(target_pfa, num_training_total, num_frames=1)
        threshold = {"target_pfa": target_pfa} if calibrated else {"threshold_factor": alpha}
        return CACFARDetector(**common, **threshold)
    raise ValueError(f'kind must be "OS" or "CA", not {kind!r}')


def noise_scans(arrays, name, beamformer="default"):
    """Return array ``name``'s matched-filtered beams from a noise-only ping, for calibration.

    The ping has no targets, so it needs no propagation and uses the cheap spherical-spreading
    model. Its noise is independent of the scenario pings' (``CALIBRATION_SEEDS``).
    """
    noise_only = simulator(
        arrays,
        name,
        SphericalAcousticPropagationModel(ssp=sound_speed_profile()),
        [],
        PING_TIMES[:1],
        seed=CALIBRATION_SEEDS[name],
        beamformer=beamformer,
    )
    return list(matched_filter_scans_from_sensor_data(noise_only.sensor_data_gen()))


def calibrated_cfar_detectors(arrays, detector=cfar_detector, beamformer="default"):
    """Return ``{array name: detector}``, each calibrated on a noise-only ping of its own.

    ``detector`` is called with no arguments for a new detector; one in fixed (nominal) mode is
    returned uncalibrated.
    """
    detectors = {}
    for name in arrays:
        cfar = detector()
        if cfar.threshold_factor is None:
            NoiseCalibrator(cfar).calibrate_from_noise(noise_scans(arrays, name, beamformer))
        detectors[name] = cfar
    return detectors


# --------------------------------------------------------------------------------------------
# Grouping detections
# --------------------------------------------------------------------------------------------
# The bistatic tracking example's post-detection steps, shared so that every script turns
# detections into the same measurements.

EDGE_RAD = np.radians(5.0)
GROUP_RANGE_M = 30.0
GHOST_BEARING_RAD = np.radians(10.0)


def wrap(angle_rad):
    """Wrap an angle in radians to [-pi, pi)."""
    return float(np.angle(np.exp(1j * angle_rad)))


def group_detections(detections, ping, power):
    """Keep one detection per echo; return ``[(bearing_rad, range_m, power_db)]``."""
    bearings = np.asarray(ping.steering_bearings_rad)
    fan_middle = wrap(np.angle(np.exp(1j * bearings[0]) + np.exp(1j * bearings[-1])))
    fan_half_width = abs(wrap(bearings[-1] - bearings[0])) / 2
    candidates = []
    for detection in detections:
        bearing, range_m = (float(v) for v in detection.state_vector[:, 0])
        beam = int(np.argmin(np.abs(np.angle(np.exp(1j * (bearings - bearing))))))
        sample = min(
            int(round(detection.metadata["delay_s"] * SAMPLING_RATE_HZ)),
            power.shape[1] - 1,
        )
        candidates.append((10 * np.log10(power[beam, sample]), bearing, range_m, beam, sample))
    kept, dropped = [], []
    for power_db, bearing, range_m, beam, sample in sorted(candidates, reverse=True):
        if any(
            abs(range_m - r) < GROUP_RANGE_M
            and (abs(wrap(bearing - b)) < np.radians(30) or power_db < p - 6)
            for b, r, p in kept + dropped
        ):
            continue
        if abs(wrap(bearing - fan_middle)) > fan_half_width - EDGE_RAD:
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


def is_mirror_ghost(detection, other_side):
    """Return whether the other flank hears a detection's mirror image 10 dB louder."""
    bearing, range_m, power_db = detection
    return any(
        abs(range_m - r) < GROUP_RANGE_M
        and abs(wrap(bearing + b)) < GHOST_BEARING_RAD
        and p >= power_db + 10
        for b, r, p in other_side
    )


# --------------------------------------------------------------------------------------------
# Tracking and scoring
# --------------------------------------------------------------------------------------------
# One tracker fuses the arrays by centralised, sequential measurement fusion, as in the
# bistatic tracking example: each array's measurements in turn update the tracks by global
# nearest neighbour assignment, through an extended Kalman filter on a constant-velocity model.
# Each measurement is a bearing and range from its array's reference point. Unassociated
# measurements start tentative tracks, confirmed after ``min_points`` updates, and of two
# tracks closer than ``merge_distance_m`` the one with fewer updates is dropped.

# Each array's bearing standard deviation, set by its beamwidth; scaled by bearing_sd_scale.
BEARING_SD_DEG = {"Bow": 0.5, "Port": 0.75, "Starboard": 0.75, "Towed": 1.0}
# The tracker's settings as set from the sensors' specification: the untuned tracker.
TRACKER_DEFAULTS = {
    "process_noise": 0.01,  # constant-velocity process noise, m^2/s^3
    "bearing_sd_scale": 1.0,  # times BEARING_SD_DEG
    "range_sd_m": 15.0,
    "gate": 4.0,  # Mahalanobis distance beyond which a measurement can't update a track
    "prior_speed_sd_mps": 5.0,  # a new track's velocity uncertainty
    "min_points": 2,  # updates to confirm a track
    "merge_distance_m": 150.0,
}

GOSPA_C_M = 200.0  # cut-off distance: a track further than this from a target is false
GOSPA_P = 1
SETTLE_PINGS = 3  # pings left out of averages while tracks are first confirmed


def _updates(track):
    """Return the number of measurement updates in a track."""
    return sum(1 for state in track if hasattr(state, "hypothesis"))


def run_tracker(arrays, ping_times, chain_measurements, used, params=None):
    """Track one chain's measurements; return ``(tracks, states_at)``.

    ``chain_measurements`` holds, for each ping, ``{array: [(bearing_rad, range_m, ...)]}``
    with ranges from the receiving array. ``used`` are the arrays fed to the tracker, in update
    order. ``params`` override ``TRACKER_DEFAULTS``. ``states_at`` holds, for each ping, the
    state of every track alive then (the tracks themselves keep growing after that ping).
    """
    p = {**TRACKER_DEFAULTS, **(params or {})}
    transition_model = CombinedLinearGaussianTransitionModel(
        [ConstantVelocity(p["process_noise"])] * 2
    )
    predictor = KalmanPredictor(transition_model)
    updater = ExtendedKalmanUpdater(measurement_model=None)
    hypothesiser = DistanceHypothesiser(
        predictor, updater, Mahalanobis(), missed_distance=p["gate"]
    )
    associator = GNNWith2DAssignment(hypothesiser)
    deleter = CovarianceBasedDeleter(covar_trace_thresh=1e6)
    initiator = MultiMeasurementInitiator(
        prior_state=GaussianState(
            StateVector([0.0, 0.0, 0.0, 0.0]),
            np.diag([0.0, p["prior_speed_sd_mps"] ** 2, 0.0, p["prior_speed_sd_mps"] ** 2]),
        ),
        deleter=deleter,
        data_associator=GNNWith2DAssignment(hypothesiser),
        updater=updater,
        min_points=int(p["min_points"]),
    )

    def to_detection(name, ping_time, bearing, range_m):
        model = CartesianToBearingRange(
            ndim_state=4,
            mapping=(0, 2),
            noise_covar=np.diag(
                [
                    np.radians(BEARING_SD_DEG[name] * p["bearing_sd_scale"]) ** 2,
                    p["range_sd_m"] ** 2,
                ]
            ),
            translation_offset=StateVector(reference_xyz(arrays[name], ping_time)[:2]),
        )
        return Detection(
            StateVector([Bearing(bearing), range_m]),
            timestamp=ping_time,
            measurement_model=model,
            metadata={"array": name},
        )

    tracks, states_at = set(), []
    for ping_time, measured in zip(ping_times, chain_measurements, strict=True):
        unassociated, updated = set(), set()
        for name in used:
            detections = {to_detection(name, ping_time, m[0], m[1]) for m in measured[name]}
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
        for track in sorted(tracks, key=_updates):
            if any(
                other is not track
                and _updates(other) >= _updates(track)
                and np.hypot(*(track.state_vector[[0, 2], 0] - other.state_vector[[0, 2], 0]))
                < p["merge_distance_m"]
                for other in tracks
            ):
                tracks.discard(track)
        states_at.append([track.state for track in tracks])
    return tracks, states_at


def score_tracks(target_paths, ping_times, states_at):
    """Return per-ping GOSPA parts and the counts of tracks, false tracks and missed targets.

    GOSPA (generalised optimal sub-pattern assignment) with cut-off ``GOSPA_C_M`` and
    ``p = GOSPA_P``: the localisation error of the tracks assigned to targets, plus half the
    cut-off for every missed target and every false track.
    """
    gospa = GOSPAMetric(c=GOSPA_C_M, p=GOSPA_P, measure=Euclidean(mapping=[0, 2], mapping2=[0, 2]))
    scores = {key: [] for key in ("gospa_m", "localisation_m", "missed_targets", "false_tracks")}
    scores["tracks"] = []
    for ping_time, states in zip(ping_times, states_at, strict=True):
        truths = [
            next(s for s in path if s.timestamp == ping_time) for path in target_paths.values()
        ]
        metric, assignment = gospa.compute_gospa_metric(states, truths)
        missed = sum(1 for index in assignment if index == -1)
        scores["gospa_m"].append(float(metric.value["distance"]))
        scores["localisation_m"].append(float(metric.value["localisation"]))
        scores["missed_targets"].append(missed)
        scores["false_tracks"].append(len(states) - (len(truths) - missed))
        scores["tracks"].append(len(states))
    return scores
