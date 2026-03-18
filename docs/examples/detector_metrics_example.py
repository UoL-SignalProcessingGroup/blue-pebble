"""Detector Metrics.

This example evaluates detector behaviour on a shared multi-target scenario and turns
the outputs into quantitative performance curves. Rather than stopping at a single
visual BTR inspection, it sweeps detector settings to show how sensitivity and false
alarms trade off across configurations.

**Background**

- In passive sonar, detector tuning is always a compromise between missed detections
  and false alarms.
- Visual inspection of one scenario can suggest whether a detector looks plausible,
  but it does not quantify how operating point changes affect performance.
- ROC and precision-recall views are especially useful when detection quality must be
  compared across parameter settings or detector chains.

**Key Concepts**

- Beamforming followed by thresholding and peak-selection detection.
- Confusion-count based evaluation over parameter sweeps.
- ROC and precision-recall metrics for passive-sonar detector comparison.
"""

# %%
from datetime import datetime, timedelta

import numpy as np
from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel,
    ConstantVelocity,
    KnownTurnRate,
)
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState

from bluepebble.detector import CACFARDetector, OSCFARDetector, PassiveSonarDetector, PeakDetector
from bluepebble.detector.metrics import SweepSpec, sweep_detection_parameter
from bluepebble.models.environment import FlatBathymetry, Linear
from bluepebble.models.propagation import rtrsAcousticPropagationModel
from bluepebble.platform import TowedArrayPlatform
from bluepebble.plotter import plot_btr, plot_roc_pr, plot_world
from bluepebble.signal.anthropogenic import SyntheticAnthropogenicSignal
from bluepebble.signal.random import ColouredNoiseSignal
from bluepebble.sigproc import DelayAndSumBeamformer, SteeringCalculator
from bluepebble.simulator import ContinuousSTFTPassiveSonarArraySimulator

# %% [markdown]
# Simulation Parameters
# ---------------------
#
# This section fixes the global timing and reproducibility settings used throughout the
# example.
#
# - `seed` ensures repeatable random draws for target signal metadata.
# - `sim_duration` and `time_interval` define the timeline used for platform motion,
#   truth propagation, simulation, and detector outputs.
# - `timesteps` is the master time axis reused in plotting and metric calculations.

# %%
# Random seed for reproducibility
seed = 2000
np.random.seed(seed)

# Simulation parameters (same scenario as multi_target_tutorial)
sim_duration = timedelta(seconds=900)
time_interval = timedelta(seconds=5)

num_steps = int(sim_duration.total_seconds() / time_interval.total_seconds())
start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
timesteps = np.array([start_time + i * time_interval for i in range(num_steps)], dtype=object)

# %% [markdown]
# Platform Scenario Setup
# -----------------------
#
# The platform follows a deterministic **straight-turn-straight** trajectory matching
# the tutorial scenario.
#
# This cell also defines the towed-array geometry (`num_sensors`, cable length, spacing,
# depth) and propagates the platform state over all `timesteps`.

# %%
# Define the platform's initial state and transition model
platform_start_vector = np.array([-7500.0, 1.15, -2000.0, 0.25, -5.0, 0.0])
platform_position_mapping = [0, 2, 4]
platform_velocity_mapping = [1, 3, 5]
platform_turn_rate_radps = np.deg2rad(1.0)

leg1_duration_s = timedelta(seconds=405)
turn1_angle_rad = np.deg2rad(-45)
turn1_duration_s = timedelta(
    seconds=round((abs(turn1_angle_rad) / platform_turn_rate_radps) / 5.0) * 5.0
)
leg2_duration_s = sim_duration - leg1_duration_s - turn1_duration_s

straight_model = CombinedLinearGaussianTransitionModel(
    [ConstantVelocity(0.0), ConstantVelocity(0.0), ConstantVelocity(0.0)]
)

turn_rate_rad1 = np.sign(turn1_angle_rad) * platform_turn_rate_radps
planar_turn1 = KnownTurnRate(
    turn_rate=turn_rate_rad1,
    turn_noise_diff_coeffs=np.array([0.0, 0.0]),
)
depth_model = ConstantVelocity(0.0)
turning_model1 = CombinedLinearGaussianTransitionModel([planar_turn1, depth_model])

transition_models = [straight_model, turning_model1, straight_model]
transition_times = [leg1_duration_s, turn1_duration_s, leg2_duration_s]

# Define the towed array parameters
num_sensors = 200
tow_cable_length_m = 100.0
sensor_spacing_m = 0.5
array_depth_m = -50.0

# Create the towed array platform and simulate its movement over time
platform_initial_state = GroundTruthState(platform_start_vector, timestamp=start_time)
platform = TowedArrayPlatform(
    states=platform_initial_state,
    position_mapping=platform_position_mapping,
    velocity_mapping=platform_velocity_mapping,
    transition_models=transition_models,
    transition_times=transition_times,
    num_sensors=num_sensors,
    cable_length_m=tow_cable_length_m,
    sensor_spacing_m=sensor_spacing_m,
    array_depth_m=array_depth_m,
)

for timestamp in timesteps[1:]:
    platform.move(timestamp)

# %% [markdown]
# Target Truth Generation
# -----------------------
#
# Three Cartesian target truth paths are created with constant-velocity dynamics.
#
# Each target also carries metadata used by the signal model (tonal amplitudes/frequencies/
# phases, tonal bandwidth, and broadband noise parameters).

# %%
# Define target initial states and transition model (same scenario as multi_target_tutorial)
target1_start_vector = np.array([-1.5e4, 9.0, 1.2e4, -10, -5.0, 0.0])
target2_start_vector = np.array([-1.1e4, -8.0, -5.7e3, 3.6, -5.0, 0.0])
target3_start_vector = np.array([-3.0e3, 10.4, -9.7e3, 9.7, -5.0, 0.0])

target_transition_model = CombinedLinearGaussianTransitionModel(
    [ConstantVelocity(0), ConstantVelocity(0), ConstantVelocity(0)]
)
target_position_mapping = [0, 2, 4]
target_velocity_mapping = [1, 3, 5]

target_truths = []

target_tonal_bandwidth_hz = np.random.uniform(0.5, 2.0)
target_noise_amplitude_upa = 10 ** (90 / 20)
target_noise_spectral_exponent = -1.0

for sv in [target1_start_vector, target2_start_vector, target3_start_vector]:
    metadata = {
        "position_mapping": target_position_mapping,
        "velocity_mapping": target_velocity_mapping,
        "amplitudes_upa": 10 ** (np.random.uniform(87, 102, 4) / 20),
        "frequencies_hz": np.random.uniform(50.0, 200.0, 4),
        "phases_rad": np.random.uniform(0, 2 * np.pi, 4),
        "tonal_bandwidth_hz": np.random.uniform(0.5, 2.0),
        "noise_amplitude_upa": 10 ** (np.random.uniform(65, 85) / 20),
        "target_tonal_bandwidth_hz": target_tonal_bandwidth_hz,
        "target_noise_amplitude_upa": target_noise_amplitude_upa,
        "noise_spectral_exponent": target_noise_spectral_exponent,
    }
    target_states = [GroundTruthState(sv, timestamp=start_time, metadata=metadata)]
    for timestamp in timesteps[1:]:
        dt = timestamp - target_states[-1].timestamp
        new_state_vector = target_transition_model.function(
            target_states[-1], noise=False, time_interval=dt
        )
        target_states.append(
            GroundTruthState(new_state_vector, timestamp=timestamp, metadata=metadata)
        )

    target_truths.append(GroundTruthPath(target_states))

# %% [markdown]
# Propagation Environment
# -----------------------
#
# Defines the acoustic environment and propagation model used to map source signals to
# the array:
#
# - Linear sound-speed profile (`ssp`)
# - Flat bathymetry
# - `rtrsAcousticPropagationModel` with configured spatial search/resolution settings
#
# Keeping these parameters fixed is important when comparing detector chains, so ROC/PR
# differences reflect detector behaviour rather than environmental changes.

# %%
# Propagation model
ssp = Linear(surface_speed=1500.0, gradient=0.2)
bathymetry = FlatBathymetry(depth=-150.0)
attenuation_factor = 0.5

propagation_model = rtrsAcousticPropagationModel(
    ssp=ssp,
    bathymetry=bathymetry,
    use_all_frequencies=False,
    step_m=20.0,
    azimuth_search_width=2.0,
    azimuth_resolution=0.5,
    elevation_range=(-25.0, 25.0),
    elevation_resolution=1.0,
)

# %% [markdown]
# Signal and Ambient Noise Models
# -------------------------------
#
# Constructs:
#
# - Broadband ambient noise model
# - Broadband ship-like source model used by targets
#
# These models determine the spectral content and difficulty of the detection task. The
# resulting SNR structure directly influences detector operating points and curve shape
# in later ROC/PR analysis.

# %%
# Signal and ambient-noise models
sampling_rate_hz = 500.0
frame_len = 500
hop_factor = 2
duration_s = num_steps * time_interval.total_seconds()

ambient_amplitude_upa = 10 ** (45 / 20)
ambient_spectral_exponent = -1
ambient_noise_model = ColouredNoiseSignal(
    amplitude_upa=ambient_amplitude_upa,
    spectral_exponent=ambient_spectral_exponent,
    duration_s=time_interval.total_seconds(),
    sampling_rate_hz=sampling_rate_hz,
)

def _make_signal_model():
    return SyntheticAnthropogenicSignal(
        duration_s=duration_s,
        sampling_rate_hz=sampling_rate_hz,
        frame_len=frame_len,
        hop_factor=hop_factor,
        tonal_bandwidth_hz=target_tonal_bandwidth_hz,
        noise_amplitude_upa=target_noise_amplitude_upa,
        noise_spectral_exponent=target_noise_spectral_exponent,
        noise_freq_range_hz=(0.0, sampling_rate_hz / 2),
        tonal_noise_is_constant=True,
        noise_is_constant=True,
    )


signal_models = [_make_signal_model() for _ in target_truths]

# %% [markdown]
# Beamforming and Detector Pipeline Setup
# ---------------------------------------
#
# This section wires together the runtime pipeline:
#
# - Steering grid and delay-and-sum beamformer
# - Broadband simulator
# - Baseline detection chain (`CACFARDetector` + `PeakDetector`)
#
# The baseline run generates detections and `snr_map`, which are then reused for
# parameter sweeps in the metrics section.

# %%
# Beamforming and detector setup
steering_azimuths_rad = np.linspace(-np.pi, np.pi, 361)

beamformer = DelayAndSumBeamformer(
    sampling_rate_hz=sampling_rate_hz,
    shading=None,
    domain="frequency",
)

steering_calculator = SteeringCalculator(
    ssp=ssp,
    steering_azimuths_rad=steering_azimuths_rad,
)

fade_in_ms = 1000.0

simulator = ContinuousSTFTPassiveSonarArraySimulator(
    platform=platform,
    propagation_model=propagation_model,
    signal_models=signal_models,
    noise_model=ambient_noise_model,
    beamformer=beamformer,
    steering_calculator=steering_calculator,
    ground_truth_paths=target_truths,
    fade_in_ms=fade_in_ms,
)

cfar_detector = CACFARDetector(
    num_guard_cells=6,
    num_training_cells=10,
    threshold_factor=1.05,
    mode="wrap",
)
peak_detector = PeakDetector(distance=8)

detection_chain = [cfar_detector, peak_detector]

detector = PassiveSonarDetector(
    detection_chain=detection_chain,
    sensor_data_gen=simulator.sensor_data_gen(),
    steering_azimuths_rad=steering_azimuths_rad,
)

# %% [markdown]
# Run Detection on Simulated Data
# -------------------------------
#
# Executes the detector over simulator outputs and captures:
#
# - `all_detections`: timestamped detection sets
# - `snr_map`: time-by-bearing SNR map used for visualisation and parameter sweeps
# - `detections_for_plotter`: flattened detections for overlay plots
#
# This provides the core inputs needed for both qualitative inspection and quantitative
# evaluation.

# %%
# Run simulation and collect detections/SNR map
all_detections = list(detector.detections_gen(progress_bar=False))
snr_map = detector.snr_history

detections_for_plotter = [d for _, detections in all_detections for d in detections]

print(f"Total no. of detections: {len(detections_for_plotter)}")

# %% [markdown]
# Relative Bearing Ground Truth Conversion
# ----------------------------------------
#
# Converts Cartesian target truths into **relative bearing truths** referenced to the
# towed-array platform at each timestamp.
#
# This representation is required for bearing-domain metric evaluation, where detections
# are associated to truth bearings using angular thresholds.

# %%
# Convert Cartesian target truths to relative bearing truths
relative_bearing_truths = []

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

    relative_bearing_truths.append(GroundTruthPath(bearing_states))

# Backward-compatible aliases used later in this example
target_ground_truths = target_truths
relative_bearing_ground_truths = relative_bearing_truths

# %% [markdown]
# Visualisation
# -------------
#
# Inspect:
#
# - World geometry (`plot_world`)
# - Bearing-Time Record (`plot_btr`) with SNR heatmap and truth overlays

# %%
# Ground truth and SNR visualisation
plot_world(truths=target_truths, platform=platform).show()

plot_btr(
    data=snr_map,
    timesteps=timesteps,
    steering_azimuths=np.rad2deg(steering_azimuths_rad),
    truths=relative_bearing_truths,
    figsize=(1000, 700),
).show()

# %% [markdown]
# Detection Metrics Sweep (ROC and PR)
# ------------------------------------
#
# Here multiple `SweepSpec` configurations are defined and detector parameters are swept
# to compare operating behaviour.
#
# For each sweep, the example reports:
#
# - ROC AUC
# - PR AUC
# - Best parameter by Youden's J statistic
# - Parameter nearest a target FPR
#
# Interpretation guidance:
#
# - Higher ROC AUC indicates better global discrimination between positives and negatives.
# - Higher PR AUC is often more informative when positive events are relatively sparse.
# - Comparing CFAR-only and CFAR+peak-clustering chains helps quantify how post-processing
#   changes false-alarm/recall trade-offs.

# %%
specs = [
    # Just CA-CFAR
    SweepSpec(
        detection_chain=[
            CACFARDetector(num_guard_cells=2, num_training_cells=5, threshold_factor=1.1),
        ],
        algorithm_index=0,
        param_name="threshold_factor",
        param_values=np.linspace(0.0, 5.0, 400),
        label="CA-CFAR",
    ),
    # Just OS-CFAR
    SweepSpec(
        detection_chain=[
            OSCFARDetector(
                num_guard_cells=3, num_training_cells=12, rank=24, threshold_factor=1.1
            ),
        ],
        algorithm_index=0,
        param_name="threshold_factor",
        param_values=np.linspace(0.0, 5.0, 400),
        label="OS-CFAR",
    ),
    # CA-CFAR with Peak Detection clustering
    SweepSpec(
        detection_chain=[
            CACFARDetector(num_guard_cells=2, num_training_cells=5, threshold_factor=1.1),
            PeakDetector(distance=3),
        ],
        algorithm_index=0,
        param_name="threshold_factor",
        param_values=np.linspace(0.0, 5.0, 400),
        label="CA-CFAR + peak clustering",
    ),
    # OS-CFAR with Peak Detection clustering
    SweepSpec(
        detection_chain=[
            OSCFARDetector(
                num_guard_cells=3, num_training_cells=12, rank=15, threshold_factor=1.1
            ),
            PeakDetector(distance=3),
        ],
        algorithm_index=0,
        param_name="threshold_factor",
        param_values=np.linspace(0.0, 5.0, 400),
        label="OS-CFAR + peak clustering",
    ),
]

TARGET_FPR = 0.05

results = sweep_detection_parameter(
    snr_map=snr_map,
    sweep_specs=specs,
    ground_truth_paths=relative_bearing_truths,
    steering_azimuths_rad=steering_azimuths_rad,
    association_threshold_rad=np.deg2rad(2.0),
)

for r in results:
    print(
        f"{r.label}: "
        f"AUC-ROC={r.auc_roc:.4f}  "
        f"AUC-PR={r.auc_pr:.4f}  "
        f"best_param (Youden-J)={r.best_param:.4f}  "
        f"param @ FPR={TARGET_FPR}={r.param_at_fpr(TARGET_FPR):.4f}"
    )

plot_roc_pr(results).show()
