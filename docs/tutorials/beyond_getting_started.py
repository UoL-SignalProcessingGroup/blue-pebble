"""
======================================================
Beyond Getting Started: Multiple Targets and Ambiguity
======================================================
"""  # noqa: D205, D212, D400, D415

# %%
# This tutorial picks up where Getting Started with Blue Pebble leaves off. The array,
# beamformer and tracker hand-off are the same; what changes is the scene. With several
# targets on both sides of the array, three things appear that a single target on one side
# never shows:
#
# - Every target has a mirror image, or *ghost*, on the other side of the array.
# - A turn of the towing platform tells real targets from their ghosts.
# - Neighbouring targets disturb a CFAR detector's noise estimate, which calls for a more
#   robust detector and a calibrated false-alarm rate.
#
# Each section below introduces one of these, in the order the scene raises them.

# %%
# Simulation Timing and Reproducibility
# -------------------------------------

# %%
from datetime import datetime, timedelta

import numpy as np

import bluepebble

seed = 42
bluepebble.set_seed(seed)
rng = bluepebble.get_rng()

sim_length_s = 900
sim_rate_s = 5.0
time_interval = timedelta(seconds=sim_rate_s)

num_steps = int(sim_length_s / sim_rate_s)
start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
timesteps = np.array([start_time + i * time_interval for i in range(num_steps)], dtype=object)

# %%
# A Turning Platform
# ------------------
#
# The platform sails a straight leg, turns 45 degrees to starboard, and sails a second
# straight leg. Stone Soup's :class:`~stonesoup.models.transition.linear.KnownTurnRate`
# model describes the turn; :class:`~.TowedArrayPlatform` works out where the towed array's
# sensors are throughout, including while the array bends through the turn. The turn
# matters later, when it separates real targets from their ghosts.

# %%
from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel,
    ConstantVelocity,
    KnownTurnRate,
)
from stonesoup.types.groundtruth import GroundTruthState

from bluepebble.platform import TowedArrayPlatform

# 5 m/s, heading a little north of east.
platform_start_vector = np.array([-7500.0, 4.88, -2000.0, 1.06, -5.0, 0.0])
platform_position_mapping = [0, 2, 4]
platform_velocity_mapping = [1, 3, 5]
platform_turn_rate_radps = np.deg2rad(1.0)

leg1_duration = timedelta(seconds=405)
turn_angle_rad = np.deg2rad(-45)
turn_duration = timedelta(
    seconds=round((abs(turn_angle_rad) / platform_turn_rate_radps) / sim_rate_s) * sim_rate_s
)
leg2_duration = timedelta(seconds=sim_length_s) - leg1_duration - turn_duration

straight_model = CombinedLinearGaussianTransitionModel(
    [ConstantVelocity(0.0), ConstantVelocity(0.0), ConstantVelocity(0.0)]
)
turning_model = CombinedLinearGaussianTransitionModel(
    [
        KnownTurnRate(
            turn_rate=np.sign(turn_angle_rad) * platform_turn_rate_radps,
            turn_noise_diff_coeffs=np.array([0.0, 0.0]),
        ),
        ConstantVelocity(0.0),  # depth
    ]
)

# The same array as Getting Started.
num_sensors = 64
tow_cable_length_m = 100.0
sensor_spacing_m = 3.0
array_depth_m = -50.0

platform = TowedArrayPlatform(
    states=GroundTruthState(platform_start_vector, timestamp=start_time),
    position_mapping=platform_position_mapping,
    velocity_mapping=platform_velocity_mapping,
    transition_models=[straight_model, turning_model, straight_model],
    transition_times=[leg1_duration, turn_duration, leg2_duration],
    num_sensors=num_sensors,
    cable_length_m=tow_cable_length_m,
    sensor_spacing_m=sensor_spacing_m,
    array_depth_m=array_depth_m,
)

for timestamp in timesteps[1:]:
    platform.move(timestamp)

# %%
# Several Targets
# ---------------
#
# Each target is an ordinary Stone Soup :class:`~stonesoup.types.groundtruth.GroundTruthPath`,
# built exactly as in Getting Started. Kinematics and acoustics stay separate:
#
# - State vectors describe where each target is and how it moves.
# - Metadata describes what each target sounds like.
#
# That separation pays off here: the three targets below share one motion model and the
# same tonal bandwidth and broadband noise, and differ only in their start states and tonals.

# %%
from stonesoup.types.groundtruth import GroundTruthPath

from bluepebble.plotter import plot_world

target_start_vectors = [
    np.array([-1.5e4, 9.0, 1.2e4, -10.0, -5.0, 0.0]),
    np.array([-8.0e3, 8.0, 4.0e3, 0.0, -5.0, 0.0]),
    np.array([-4.0e3, 4.24, 4.0e3, -4.24, -5.0, 0.0]),
]
target_transition_model = CombinedLinearGaussianTransitionModel(
    [ConstantVelocity(0), ConstantVelocity(0), ConstantVelocity(0)]
)
target_position_mapping = [0, 2, 4]

# Tonal bandwidth and broadband noise, the same for every target
target_tonal_bandwidth_hz = rng.uniform(0.5, 2.0)
target_noise_amplitude_upa = 10 ** (90 / 20)
target_noise_spectral_exponent = -1.0

target_truths = []
for start_vector in target_start_vectors:
    metadata = {
        "position_mapping": target_position_mapping,
        "amplitudes_upa": 10 ** (rng.uniform(87, 102, 4) / 20),
        "frequencies_hz": rng.uniform(50.0, 200.0, 4),
        "phases_rad": rng.uniform(0, 2 * np.pi, 4),
        "tonal_bandwidth_hz": target_tonal_bandwidth_hz,
        "noise_amplitude_upa": target_noise_amplitude_upa,
        "noise_spectral_exponent": target_noise_spectral_exponent,
    }
    target_states = [GroundTruthState(start_vector, timestamp=start_time, metadata=metadata)]
    for timestamp in timesteps[1:]:
        dt = timestamp - target_states[-1].timestamp
        new_state_vector = target_transition_model.function(
            target_states[-1], noise=False, time_interval=dt
        )
        target_states.append(
            GroundTruthState(new_state_vector, timestamp=timestamp, metadata=metadata)
        )
    target_truths.append(GroundTruthPath(target_states))

plot_world(truths=target_truths, platform=platform)

# %%
# A Higher-Fidelity Propagation Model
# -----------------------------------
#
# Getting Started used a simple cylindrical spreading model. Here
# :class:`~.rtrsAcousticPropagationModel` traces rays through a sound-speed profile and
# bathymetry instead. Nothing else in the workflow changes. To the simulator it is just
# another propagation model.
#
# The seabed is sand, which absorbs part of every steep bounce. Left unset, ``bottom_model``
# gives a rigid seabed that reflects everything, so steep multipath arrives far stronger
# than it would at sea.

# %%
from bluepebble.models.environment import FlatBathymetry, Linear
from bluepebble.models.propagation import rtrsAcousticPropagationModel

ssp = Linear(surface_speed=1500.0, gradient=0.2)
bathymetry = FlatBathymetry(depth=-150.0)

propagation_model = rtrsAcousticPropagationModel(
    ssp=ssp,
    bathymetry=bathymetry,
    use_all_frequencies=False,
    step_m=20.0,
    azimuth_search_width=2.0,
    azimuth_resolution=0.5,
    elevation_range=(-25.0, 25.0),
    elevation_resolution=1.0,
    bottom_model={
        "model": "acoustic",
        "compressional_speed_m_s": 1650.0,
        "density_g_cm3": 1.9,
        "compressional_attenuation_db_per_wavelength": 0.8,
    },
)

# %%
# Source and Noise Models
# -----------------------
#
# The ambient noise and source models are the same kinds as in Getting Started. The
# simulator takes one source model per target; each reads its own target's metadata.

# %%
from bluepebble.signal.anthropogenic import SyntheticAnthropogenicSignal
from bluepebble.signal.random import ColouredNoiseSignal

sampling_rate_hz = 500.0
frame_len = 500
hop_factor = 2
fade_in_ms = 1000.0
total_duration_s = num_steps * time_interval.total_seconds()

ambient_noise_model = ColouredNoiseSignal(
    amplitude_upa=10 ** (45 / 20),
    spectral_exponent=-1,
    duration_s=time_interval.total_seconds(),
    sampling_rate_hz=sampling_rate_hz,
)


def make_signal_model():
    """Build one target's source model; all targets share these settings."""
    return SyntheticAnthropogenicSignal(
        duration_s=total_duration_s,
        sampling_rate_hz=sampling_rate_hz,
        frame_len=frame_len,
        hop_factor=hop_factor,
        noise_freq_range_hz=(0.0, sampling_rate_hz / 2),
        tonal_noise_is_constant=True,
        noise_is_constant=True,
    )


signal_models = [make_signal_model() for _ in target_truths]

# %%
# Beamforming the Full Circle
# ---------------------------
#
# The beamformer matches Getting Started, with one difference: it steers the full circle,
# because there are now targets on both sides of the array.

# %%
from bluepebble.sigproc import (
    DelayAndSumBeamformer,
    SteeringCalculator,
    beams_per_mainlobe,
    cfar_window_for_mainlobe,
)
from bluepebble.simulator import ContinuousSTFTPassiveSonarArraySimulator

steering_azimuths_rad = np.linspace(-np.pi, np.pi, 360, endpoint=False)
fmin = 100.0
fmax = 245.0

beamformer = DelayAndSumBeamformer(
    sampling_rate_hz=sampling_rate_hz,
    fmin=fmin,
    fmax=fmax,
    shading=np.hanning(num_sensors),
    domain="broadband_power",
)
steering_calculator = SteeringCalculator(
    ssp=ssp,
    steering_azimuths_rad=steering_azimuths_rad,
)


def make_simulator(ground_truth_paths):
    """Build the simulator for this scene; an empty list gives ambient noise only."""
    return ContinuousSTFTPassiveSonarArraySimulator(
        platform=platform,
        propagation_model=propagation_model,
        signal_models=signal_models,
        noise_model=ambient_noise_model,
        beamformer=beamformer,
        steering_calculator=steering_calculator,
        ground_truth_paths=ground_truth_paths,
        fade_in_ms=fade_in_ms,
    )


# %%
# Detecting with a Calibrated OS-CFAR Detector
# --------------------------------------------
#
# Getting Started used CA-CFAR, which estimates each beam's noise level as the average of
# the beams either side of it. With several targets that average is easily disturbed; a
# neighbouring target, or a ghost, in those beams raises the estimate and can hide a real
# contact. :class:`~.OSCFARDetector` instead takes the k-th smallest of those beams, so a
# few bright ones make no difference.
#
# With more going on in the scene, it is also worth controlling the false-alarm rate rather
# than choosing a threshold by hand. Setting ``target_pfa`` and calibrating the detector on
# ambient noise does this; recorded, in practice, before the operation with the same array
# and settings but no targets present. Here the survey comes from the same scene with the
# targets removed. :ref:`sphx_glr_auto_examples_calibrating_cfar_from_noise.py` covers
# calibration in depth.

# %%
from itertools import islice

from bluepebble.detector import (
    NoiseCalibrator,
    OSCFARDetector,
    PassiveSonarDetector,
    beamformed_scans_from_sensor_data,
)
from bluepebble.plotter import plot_btr

# The window is sized from the beam width, as in Getting Started.
mainlobe_beams = beams_per_mainlobe(
    aperture_m=(num_sensors - 1) * sensor_spacing_m,
    frequency_hz=fmin,
    beam_spacing_rad=float(np.diff(steering_azimuths_rad)[0]),
    sound_speed_ms=1500.0,
    shading_factor=1.44,  # Hann
)
num_guard_cells, num_training_cells, peak_distance = cfar_window_for_mainlobe(mainlobe_beams)

cfar_detector = OSCFARDetector(
    num_guard_cells=num_guard_cells,
    num_training_cells=num_training_cells,
    # Three quarters of the way up the sorted beams: above the quietest, below any targets.
    rank=round(0.75 * 2 * num_training_cells),
    target_pfa=1e-3,
    peak_distance=peak_distance,
)

# 60 scans of 360 beams is just over the roughly 20,000 cells calibration needs.
num_survey_scans = 60
noise_scans = beamformed_scans_from_sensor_data(
    islice(make_simulator([]).sensor_data_gen(), num_survey_scans),
    progress_bar=True,
    total=num_survey_scans,
)
NoiseCalibrator(cfar_detector).calibrate_from_noise(noise_scans)

detector = PassiveSonarDetector(
    detector=cfar_detector,
    sensor_data_gen=make_simulator(target_truths).sensor_data_gen(progress_bar=True),
    steering_azimuths_rad=steering_azimuths_rad,
)
all_detections = list(detector.detections_gen(progress_bar=True, total_timesteps=num_steps))
detections_for_plotter = [d for _, detections in all_detections for d in detections]

plot_btr(
    data=detector.reported_snr_history,
    detections=detections_for_plotter,
    timesteps=timesteps,
    steering_azimuths=np.rad2deg(steering_azimuths_rad),
    bearing_convention="true",
).update_layout(
    template="plotly_white",
    autosize=True,
    width=None,
    height=700,
)

# %%
# Six Tracks, Three Targets
# -------------------------
#
# As in Getting Started, the plots show true bearings, clockwise from north.
#
# **Each target appears twice.** A line of hydrophones cannot tell which side of itself a
# sound came from: a source and its mirror image across the array's axis produce the same
# arrival times along the array, so the beamformer reports both. Every target therefore has
# a ghost, reflected about the array's axis. Here all three targets are to port and their
# ghosts to starboard, but nothing in the data says which side is which.
#
# **The turn tells them apart.** About seven and a half minutes in, three of the six tracks
# jump to new bearings while the other three carry straight on. A real target's bearing
# depends only on where the target is, so the turn leaves it unchanged. A ghost is
# reflected about the array's axis, so when the axis swings, the ghost swings with it, by
# twice as much. A towed array follows the platform's path rather than its heading, so the
# jump builds up over a minute or so as the array works its way through the bend. Turning
# to resolve this left-right ambiguity is standard practice with towed arrays.

# %%
# Tracking Every Bearing
# ----------------------
#
# The hand-off to Stone Soup is as in Getting Started, but with several contacts per scan
# the tracker must weigh up which detection belongs to which track. It uses JPDA (joint
# probabilistic data association) to do that, and starts a new track once enough nearby
# detections agree.
#
# The tracker knows nothing about ghosts: it follows all six bearings. The turn still shows
# through, though. The three real tracks run unbroken from start to finish, while the
# ghosts' tracks break or get tangled at the jump as the ghosts swing across to their new
# bearings.

# %%
bearing_truths = []
for target_truth in target_truths:
    bearing_states = []
    for state in target_truth:
        platform_state = platform.get_platform_state_at(state.timestamp)
        assert platform_state is not None
        ref_sensor_position = np.mean(platform_state.array.state_vector, axis=1)
        target_xy = np.array([state.state_vector[0], state.state_vector[2]])
        relative_position = target_xy - ref_sensor_position[:2]
        bearing = np.arctan2(relative_position[1], relative_position[0])
        bearing_states.append(GroundTruthState(np.array([bearing]), timestamp=state.timestamp))
    bearing_truths.append(GroundTruthPath(bearing_states))

# %%
from stonesoup.dataassociator.neighbour import GNNWith2DAssignment
from stonesoup.dataassociator.probability import JPDA
from stonesoup.deleter.time import UpdateTimeStepsDeleter
from stonesoup.functions import mod_bearing
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.hypothesiser.probability import PDAHypothesiser
from stonesoup.initiator.simple import MultiMeasurementInitiator
from stonesoup.measures import Mahalanobis
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.tracker.simple import MultiTargetMixtureTracker
from stonesoup.types.state import GaussianState
from stonesoup.updater.kalman import KalmanUpdater

predictor = KalmanPredictor(ConstantVelocity(0.000001))
measurement_model = LinearGaussian(
    ndim_state=2,
    mapping=[0],
    noise_covar=np.array([[np.deg2rad(2) ** 2]]),
)
updater = KalmanUpdater(measurement_model=measurement_model)

hypothesiser = PDAHypothesiser(
    predictor=predictor,
    updater=updater,
    clutter_spatial_density=3 / (2 * np.pi),  # about three false alarms per scan
    prob_detect=0.85,
)
data_associator = JPDA(hypothesiser=hypothesiser)

# A track that goes 5 scans without an update is deleted.
deleter = UpdateTimeStepsDeleter(time_steps_since_update=5)

first_detection = next(det for _, det_set in all_detections if det_set for det in det_set)
prior_state = GaussianState(
    np.array([[first_detection.state_vector[0, 0]], [0.0]]),
    np.diag([np.deg2rad(5) ** 2, np.deg2rad(0.5) ** 2]),
    timestamp=start_time,
)
initiator = MultiMeasurementInitiator(
    prior_state=prior_state,
    measurement_model=measurement_model,
    deleter=deleter,
    data_associator=GNNWith2DAssignment(
        DistanceHypothesiser(
            predictor=predictor,
            updater=updater,
            measure=Mahalanobis(),
            missed_distance=6,
        )
    ),
    updater=updater,
    min_points=10,
)

tracker = MultiTargetMixtureTracker(
    initiator=initiator,
    deleter=deleter,
    detector=all_detections,
    data_associator=data_associator,
    updater=updater,
)

# Detections carry their bearing as a Stone Soup Bearing, so each innovation wraps at +/-180
# degrees and a target crossing due west is tracked straight through. The Kalman filter keeps
# its state as a plain number, so the loop below wraps each updated bearing back into range.
tracks = set()
for _, current_tracks in tracker:
    for track in current_tracks:
        track[-1].state_vector[0, 0] = mod_bearing(float(track[-1].state_vector[0, 0]))
    tracks |= current_tracks

plot_btr(
    timesteps=timesteps,
    steering_azimuths=np.rad2deg(steering_azimuths_rad),
    bearing_convention="true",
    truths=bearing_truths,
    detections=detections_for_plotter,
    tracks=tracks,
).update_layout(
    template="plotly_white",
    autosize=True,
    width=None,
    height=700,
)

# %%
# Summary
# -------
#
# Compared with Getting Started, this tutorial added:
#
# 1. Several targets, each with its own acoustic metadata and source model.
# 2. Full-circle beamforming, which shows every target's ghost across the array's axis.
# 3. A platform turn, which separates real targets, whose bearings hold steady, from ghosts,
#    which jump.
# 4. A calibrated OS-CFAR detector, robust to neighbouring targets and with a controlled
#    false-alarm rate.
# 5. JPDA tracking, to associate several detections per scan.
