"""
================================
Getting Started with Blue Pebble
================================
"""  # noqa: D205, D212, D400, D415

# %%
# This tutorial shows how to use Blue Pebble as a Stone Soup plugin for a complete
# single-target passive-sonar workflow. The goal is not just to produce a result, but
# to show how Blue Pebble components fit around standard Stone Soup state, truth,
# detection, and tracking objects.
#
# Background & Key Concepts
# -------------------------
#
# Blue Pebble extends Stone Soup with underwater-acoustics and passive-sonar components
# that are not part of the core tracking library. In practice, the plugin adds
# array-platform models, acoustic propagation models, source and noise signal models,
# beamforming, and passive-sonar detection utilities. Stone Soup still provides the
# state representations, motion models, data-association logic, and trackers.
#
# In this tutorial you will assemble the following pipeline:
#
# - A towed array platform.
# - A single target truth path with plugin-specific acoustic metadata.
# - A cylindrical acoustic propagation model and broadband signal/noise models.
# - A broadband passive-sonar simulator, beamformer, and detector.
# - A Stone Soup bearing tracker driven by Blue Pebble detections.

# %%
# Simulation Timing and Reproducibility
# -------------------------------------
#
# Start by defining the timing configuration shared by every plugin component. In
# Blue Pebble, the integration interval drives more than state propagation: it also
# sets the cadence for signal generation, beamforming, and detection.

# %%
from datetime import datetime, timedelta

# sphinx_gallery_thumbnail_path = "_static/thumbnails/getting_started.png"
import numpy as np

import bluepebble
from bluepebble.detector import CACFARDetector

# Random seed for reproducibility
seed = 42
bluepebble.set_seed(seed)
rng = bluepebble.get_rng()

# Simulation parameters
sim_length_s = 900
sim_rate_s = 5.0
time_interval = timedelta(seconds=sim_rate_s)

num_steps = int(sim_length_s / sim_rate_s)
start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
timesteps = np.array([start_time + i * time_interval for i in range(num_steps)], dtype=object)

# %%
# Build a Passive Towed Array Platform
# ------------------------------------
#
# The first plugin-specific object is the host platform. :class:`~.TowedArrayPlatform` inherits
# from Stone Soup's moving-platform machinery, so you still provide familiar Stone Soup
# ingredients such as an initial :class:`~stonesoup.types.groundtruth.GroundTruthState`,
# position/velocity mappings, and :class:`~stonesoup.models.transition.linear.CombinedLinearGaussianTransitionModel`.  # noqa: E501
#
# Blue Pebble then adds the array-specific parameters that Stone Soup does not model by default:
#
# - `num_sensors`
# - `cable_length_m`
# - `sensor_spacing_m`
# - `array_depth_m`

# %%
from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel,
    ConstantVelocity,
)
from stonesoup.types.groundtruth import GroundTruthState

from bluepebble.platform import TowedArrayPlatform

# Define the platform's initial state and transition model
platform_start_vector = np.array([-2000.0, 5.0, 2000.0, 0.0, -5.0, 0.0])
platform_position_mapping = [0, 2, 4]
platform_velocity_mapping = [1, 3, 5]
platform_transition_model = CombinedLinearGaussianTransitionModel(
    [ConstantVelocity(0), ConstantVelocity(0), ConstantVelocity(0)]
)

# Define the towed array parameters
num_sensors = 64
tow_cable_length_m = 100.0
# 3 m spacing makes a long (189 m) array for sharp bearings, while staying fine enough for the
# highest frequency simulated (250 Hz).
sensor_spacing_m = 3.0
array_depth_m = -50.0

# Create the towed array platform and simulate its movement over time
platform_initial_state = GroundTruthState(platform_start_vector, timestamp=start_time)
platform = TowedArrayPlatform(
    states=platform_initial_state,
    position_mapping=platform_position_mapping,
    velocity_mapping=platform_velocity_mapping,
    transition_models=[platform_transition_model],
    transition_times=[timedelta(seconds=sim_length_s)],
    num_sensors=num_sensors,
    cable_length_m=tow_cable_length_m,
    sensor_spacing_m=sensor_spacing_m,
    array_depth_m=array_depth_m,
)

for timestamp in timesteps[1:]:
    platform.move(timestamp)

# %%
# Attach Acoustic Metadata to the Ground Truth
# --------------------------------------------
#
# Targets remain ordinary Stone Soup :class:`~stonesoup.types.groundtruth.GroundTruthPath` objects.
# The plugin-specific step is to attach acoustic source parameters to each state's `metadata` so
# the signal model and simulator can interpret the target as an emitting underwater source.
#
# Here the metadata describes a ship-like source: a few tonals over broadband noise.

# %%
from stonesoup.types.groundtruth import GroundTruthPath

from bluepebble.plotter import plot_world

# Define the target's initial state and transition model
target_start_vector = np.array([500.0, 0.0, 0.0, -3.0, -5.0, 0.0])
target_transition_model = CombinedLinearGaussianTransitionModel(
    [ConstantVelocity(0), ConstantVelocity(0), ConstantVelocity(0)]
)
target_position_mapping = [0, 2, 4]

# What the target emits, carried in its metadata: tonals, their bandwidth, broadband noise
target_amplitudes_upa = 10 ** (rng.uniform(97, 112, 4) / 20)
target_frequencies_hz = rng.uniform(25.0, 200.0, 4)
target_phases_rad = rng.uniform(0, 2 * np.pi, 4)
target_tonal_bandwidth_hz = rng.uniform(0.5, 2.0)
target_noise_amplitude_upa = 10 ** (90 / 20)
target_noise_spectral_exponent = -1.0  # Pink noise

target_metadata = {
    "amplitudes_upa": target_amplitudes_upa,
    "frequencies_hz": target_frequencies_hz,
    "phases_rad": target_phases_rad,
    "tonal_bandwidth_hz": target_tonal_bandwidth_hz,
    "noise_amplitude_upa": target_noise_amplitude_upa,
    "noise_spectral_exponent": target_noise_spectral_exponent,
    "position_mapping": target_position_mapping,
}

# Simulate the target's movement over time and create a ground truth path
target_states = [
    GroundTruthState(target_start_vector, timestamp=start_time, metadata=target_metadata)
]
for timestamp in timesteps[1:]:
    dt = timestamp - target_states[-1].timestamp
    new_state_vector = target_transition_model.function(
        target_states[-1], noise=False, time_interval=dt
    )
    target_states.append(
        GroundTruthState(new_state_vector, timestamp=timestamp, metadata=target_metadata)
    )

target_truth = GroundTruthPath(target_states)
target_truths = [target_truth]

plot_world(truths=target_truths, platform=platform)

# %%
# Choose a Blue Pebble Propagation Model
# ---------------------------------------
#
# Next, configure the acoustic environment. This is where Blue Pebble begins to add
# the underwater-propagation physics that sit outside Stone Soup's core remit.
#
# This tutorial uses :class:`~.CylindricalAcousticPropagationModel` with a simple linear sound-speed  # noqa: E501
# profile. Once you provide a propagation model, the simulator can
# use it to convert target/platform geometry into array-level acoustic observations.

# %%
from bluepebble.models.environment import Linear
from bluepebble.models.propagation import CylindricalAcousticPropagationModel

ssp = Linear(surface_speed=1500.0, gradient=0.2)
attenuation_factor = 0.5

propagation_model = CylindricalAcousticPropagationModel(
    ssp=ssp,
    attenuation_factor=attenuation_factor,
)

# %%
# Configure Source and Noise Models
# ---------------------------------
#
# With platform motion and target truth in place, define the acoustic content that will actually
# reach the array. This is another plugin boundary: Blue Pebble supplies source and ambient-noise
# generators that are aware of passive-sonar signal-processing settings.
#
# The two important patterns are:
#
# - Ambient background is modelled explicitly with :class:`~.ColouredNoiseSignal`, usually over one
#   integration interval at a time.
# - Target emissions are modelled explicitly with :class:`~.SyntheticAnthropogenicSignal`, using
#   the metadata attached to the Stone Soup truth states.

# %%
from bluepebble.signal.anthropogenic import SyntheticAnthropogenicSignal
from bluepebble.signal.random import ColouredNoiseSignal

sampling_rate_hz = 500.0
frame_len = 500
hop_factor = 2
fade_in_ms = 1000.0
total_duration_s = num_steps * time_interval.total_seconds()

ambient_amplitude_upa = 10 ** (45 / 20)
ambient_spectral_exponent = -1
ambient_noise_model = ColouredNoiseSignal(
    amplitude_upa=ambient_amplitude_upa,
    spectral_exponent=ambient_spectral_exponent,
    duration_s=time_interval.total_seconds(),
    sampling_rate_hz=sampling_rate_hz,
)

signal_model = SyntheticAnthropogenicSignal(
    duration_s=total_duration_s,
    sampling_rate_hz=sampling_rate_hz,
    frame_len=frame_len,
    hop_factor=hop_factor,
    noise_freq_range_hz=(0.0, sampling_rate_hz / 2),
    tonal_noise_is_constant=True,
    noise_is_constant=True,
)

# %%
# Run the Blue Pebble Simulator, Beamformer, and Detector
# ---------------------------------------------------------
#
# This section is the core plugin workflow. :class:`~.ContinuousSTFTPassiveSonarArraySimulator`
# brings together the platform, propagation model, source/noise models, steering
# calculation, and beamformer to produce beamformed sonar output over time.
#
# Once the simulator is in place, :class:`~.PassiveSonarDetector` applies a single CFAR-family
# ``detector`` (here, :class:`~.CACFARDetector`) directly to those outputs. Blue Pebble handles
# the signal-processing and detection side, then returns timestamped detections that can be
# analysed directly or passed into Stone Soup tracking components.

# %%
from bluepebble.detector import PassiveSonarDetector
from bluepebble.plotter import plot_btr
from bluepebble.sigproc import (
    DelayAndSumBeamformer,
    SteeringCalculator,
    beams_per_mainlobe,
    cfar_window_for_mainlobe,
)
from bluepebble.simulator import ContinuousSTFTPassiveSonarArraySimulator

# Hann weights suppress the beams' sidelobes (see the notes under the plot).
shading = np.hanning(num_sensors)
beamforming_domain = "broadband_power"
# A line array cannot tell which side of it a sound came from, so steer only the side the
# target is on: starboard, in 1 degree steps. Steering angles are azimuths, anticlockwise from
# east, so for this east-heading platform starboard is -180 to 0 degrees.
steering_azimuths_rad = np.linspace(-np.pi, 0.0, 181)
# Up to just below the highest simulated frequency (250 Hz), covering the target's tonals.
fmin = 100.0
fmax = 245.0

beamformer = DelayAndSumBeamformer(
    sampling_rate_hz=sampling_rate_hz,
    fmin=fmin,
    fmax=fmax,
    shading=shading,
    domain=beamforming_domain,
)

steering_calculator = SteeringCalculator(
    ssp=ssp,
    steering_azimuths_rad=steering_azimuths_rad,
)

simulator = ContinuousSTFTPassiveSonarArraySimulator(
    platform=platform,
    propagation_model=propagation_model,
    signal_models=[signal_model],
    noise_model=ambient_noise_model,
    beamformer=beamformer,
    steering_calculator=steering_calculator,
    ground_truth_paths=target_truths,
    fade_in_ms=fade_in_ms,
)

# The CFAR window and peak spacing are sized from the beam width at the lowest frequency, so
# the target's own beam stays out of its noise estimate.
mainlobe_beams = beams_per_mainlobe(
    aperture_m=(num_sensors - 1) * sensor_spacing_m,
    frequency_hz=fmin,
    beam_spacing_rad=float(np.diff(steering_azimuths_rad)[0]),
    sound_speed_ms=1500.0,
    shading_factor=1.44,  # Hann; uniform weights would be 0.886
)
num_guard_cells, num_training_cells, peak_distance = cfar_window_for_mainlobe(mainlobe_beams)
# Detect where a beam's power exceeds 1.5x its local noise estimate (about 1.8 dB). The
# detector API docs describe calibrating to a false-alarm rate instead.
threshold_factor = 1.5

cfar_detector = CACFARDetector(
    num_guard_cells=num_guard_cells,
    num_training_cells=num_training_cells,
    threshold_factor=threshold_factor,
    peak_distance=peak_distance,
    circular=False,  # half a circle of bearings does not wrap round
)

detector = PassiveSonarDetector(
    detector=cfar_detector,
    sensor_data_gen=simulator.sensor_data_gen(progress_bar=True),
    steering_azimuths_rad=steering_azimuths_rad,
)

all_detections = list(detector.detections_gen(progress_bar=True, total_timesteps=num_steps))
reported_snr = detector.reported_snr_history

detections_for_plotter = [d for _, detections in all_detections for d in detections]

print(f"Total no. of detections: {len(detections_for_plotter)}")

# %%
plot_btr(
    data=reported_snr,
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
# Reading the Bearing-Time Record
# --------------------------------
#
# The plots show true bearings, clockwise from north, as a sonar display would. The platform
# heads east, so dead ahead is 090 and starboard broadside 180.
#
# **The target is a band, not a line.** Each beam listens over a spread of bearings rather
# than a single one, so the target lights up several neighbouring beams. How wide that spread
# is depends on the array's length measured in wavelengths: a longer array, or a higher
# frequency, gives narrower beams. The band also widens towards the array axis (090, dead
# ahead). A line array judges bearing from the differences in arrival time along its length,
# and those change quickly with bearing broadside to the array (180) but hardly at all near
# its axis. That is why the trace is broader early in the run, when the
# target is closer to dead ahead.
#
# **Faint copies either side are sidelobes.** An array of finite length also picks up a
# little of the target at bearings well away from it. The Hann shading set above keeps these
# sidelobes weak, but they can still show as faint streaks beside the main trace.
#
# **Colour shows SNR against the scan's overall noise floor**: the dark background is ambient
# noise, and the bright band is the target.

# %%
# Feed Blue Pebble Detections into a Stone Soup Tracker
# ------------------------------------------------------
#
# The final step shows the hand-off back into Stone Soup. The bearing-time record and
# passive-sonar detections come from Blue Pebble, but the tracker itself is assembled
# from standard Stone Soup building blocks: a transition model, measurement model,
# predictor, updater, hypothesiser, data associator, and tracker.
#
# That separation is the main plugin pattern to remember:
#
# - Blue Pebble models the acoustic sensing process and detection generation.
# - Stone Soup models the Bayesian tracking logic once measurements exist.
#
# For plotting on the same axes, we also convert the target's true position to its bearing
# from the array at each timestep.

# %%
bearing_states = []
for target_state in target_truth:
    platform_state = platform.get_platform_state_at(target_state.timestamp)
    assert platform_state is not None
    ref_sensor_position = np.mean(platform_state.array.state_vector, axis=1)
    target_xy = np.array([target_state.state_vector[0], target_state.state_vector[2]])
    relative_position = target_xy - ref_sensor_position[:2]
    bearing = np.arctan2(relative_position[1], relative_position[0])
    bearing_states.append(GroundTruthState(np.array([bearing]), timestamp=target_state.timestamp))

bearing_truth = GroundTruthPath(bearing_states)
bearing_truths = [bearing_truth]

# %%
from stonesoup.dataassociator.probability import PDA
from stonesoup.deleter.time import UpdateTimeStepsDeleter
from stonesoup.functions import mod_bearing
from stonesoup.hypothesiser.probability import PDAHypothesiser
from stonesoup.initiator.simple import SinglePointInitiator
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.tracker.simple import SingleTargetMixtureTracker
from stonesoup.types.state import GaussianState
from stonesoup.types.track import Track
from stonesoup.updater.kalman import KalmanUpdater

predictor = KalmanPredictor(ConstantVelocity(0.000001))

measurement_model = LinearGaussian(
    ndim_state=2,
    mapping=[0],
    noise_covar=np.array([[np.deg2rad(1) ** 2]]),
)

updater = KalmanUpdater(measurement_model=measurement_model)

hypothesiser = PDAHypothesiser(
    predictor=predictor,
    updater=updater,
    clutter_spatial_density=5 / np.pi,
    prob_detect=0.95,
)
data_associator = PDA(hypothesiser=hypothesiser)

initial_bearing = float(bearing_truth[0].state_vector[0])

prior_state = GaussianState(
    np.array([initial_bearing, 0.0]),
    np.diag([np.deg2rad(5) ** 2, np.deg2rad(0.5) ** 2]),
    timestamp=start_time,
)

initiator = SinglePointInitiator(
    prior_state=prior_state,
    measurement_model=measurement_model,
    updater=updater,
)

deleter = UpdateTimeStepsDeleter(time_steps_since_update=99999)

kf = SingleTargetMixtureTracker(
    initiator=initiator,
    deleter=deleter,
    detector=all_detections,
    data_associator=data_associator,
    updater=updater,
)

tracks: set[Track] = set()

for _, current_tracks in kf:
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
# You now have the minimal single-target plugin workflow:
#
# 1. Build a Stone Soup platform and truth model.
# 2. Add Blue Pebble array, propagation, source, and noise components.
# 3. Run the passive sonar simulator and a CFAR-family passive-sonar detector.
# 4. Pass the resulting detections into a Stone Soup tracker.
