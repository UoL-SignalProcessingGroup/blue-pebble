"""
===========================
Evaluating Detector Metrics
===========================

This example evaluates detector behaviour on a shared multi-target scenario and turns
the outputs into quantitative performance curves. Rather than stopping at a single
visual BTR inspection, it sweeps detector settings to show how sensitivity and false
alarms trade off across configurations.

In passive sonar, detector tuning is always a compromise between missed detections and
false alarms. Visual inspection of one scenario can suggest whether a detector looks
plausible, but it does not quantify how operating point changes affect performance.
ROC and precision-recall curves are especially useful when comparing operating points
across parameter settings or detector chains.
"""  # noqa: D205, D212, D400, D415

# %%
# Imports
# -------
#
# All dependencies are consolidated here for convenience.

from datetime import datetime, timedelta

import numpy as np
from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel,
    ConstantVelocity,
    KnownTurnRate,
)
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState

import bluepebble
from bluepebble.detector import CACFARDetector, OSCFARDetector, PassiveSonarDetector, PeakDetector
from bluepebble.detector.metrics import SweepSpec, sweep_detection_parameter
from bluepebble.models.environment import FlatBathymetry, Linear
from bluepebble.models.propagation import rtrsAcousticPropagationModel
from bluepebble.platform import TowedArrayPlatform
from bluepebble.plotter import plot_btr, plot_roc_pr, plot_world
from bluepebble.sensor import Hydrophone, HydrophoneResponse, LinearHydrophoneArray
from bluepebble.signal.anthropogenic import SyntheticAnthropogenicSignal
from bluepebble.signal.random import ColouredNoiseSignal
from bluepebble.sigproc import DelayAndSumBeamformer, SteeringCalculator
from bluepebble.simulator import ContinuousSTFTPassiveSonarArraySimulator

# %%
# Simulation Parameters
# ---------------------
#
# A fixed seed and shared timeline are established here so every stochastic signal
# component is reproducible and all downstream components (platform motion, target
# propagation, simulator, and metrics) operate on the same time axis.

seed = 2000
bluepebble.set_seed(seed)
rng = bluepebble.get_rng()

sim_length_s = 900
sim_rate_s = 5.0
time_interval = timedelta(seconds=sim_rate_s)

num_steps = int(sim_length_s / sim_rate_s)
start_time = datetime(2026, 1, 1, 0, 0, 0)
timesteps = np.array([start_time + i * time_interval for i in range(num_steps)], dtype=object)

# %%
# Platform Setup and Generation
# -----------------------------
#
# The platform follows a straight-turn-straight path. The mid-run heading change is
# important for this example: it shifts the apparent bearing of all three targets
# over time, which exercises the beamformer over a range of steering angles rather
# than just one static geometry.

platform_start_vector = np.array([-7500.0, 1.15, -2000.0, 0.25, -5.0, 0.0])
platform_position_mapping = [0, 2, 4]
platform_velocity_mapping = [1, 3, 5]
platform_turn_rate_radps = np.deg2rad(1.0)

leg1_duration_s = timedelta(seconds=405)
turn1_angle_rad = np.deg2rad(-45)
turn1_duration_s = timedelta(
    seconds=round((abs(turn1_angle_rad) / platform_turn_rate_radps) / sim_rate_s) * sim_rate_s
)
leg2_duration_s = timedelta(seconds=sim_length_s) - leg1_duration_s - turn1_duration_s

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

num_sensors = 200
tow_cable_length_m = 100.0
sensor_spacing_m = 0.5
array_depth_m = -50.0

hydrophone_response = HydrophoneResponse()
elements = [Hydrophone(response=hydrophone_response) for _ in range(num_sensors)]
sensor_array = LinearHydrophoneArray(elements=elements, element_spacing_m=sensor_spacing_m)

platform_initial_state = GroundTruthState(platform_start_vector, timestamp=start_time)
platform = TowedArrayPlatform(
    states=platform_initial_state,
    position_mapping=platform_position_mapping,
    velocity_mapping=platform_velocity_mapping,
    transition_models=transition_models,
    transition_times=transition_times,
    sensor_array=sensor_array,
    cable_length_m=tow_cable_length_m,
    array_depth_m=array_depth_m,
)

for timestamp in timesteps[1:]:
    platform.move(timestamp)

# %%
# Ground Truth Setup and Generation
# ---------------------------------
#
# Three targets are created with random tonal amplitudes, frequencies, and broadband
# noise levels drawn from the fixed seed. The variation in SNR across targets produces
# a non-trivial ROC/PR curve. A scenario where all targets are equally easy to detect
# gives a less informative sweep.
#
# Cartesian truths are immediately converted to relative bearing truths so the
# bearing-domain metric association is available for all downstream sections.
# The :func:`~bluepebble.plotter.plot_world` figure confirms the geometry before
# the detector is run.

target1_start_vector = np.array([-1.5e4, 9.0, 1.2e4, -10, -5.0, 0.0])
target2_start_vector = np.array([-1.1e4, -8.0, -5.7e3, 3.6, -5.0, 0.0])
target3_start_vector = np.array([-3.0e3, 10.4, -9.7e3, 9.7, -5.0, 0.0])

target_transition_model = CombinedLinearGaussianTransitionModel(
    [ConstantVelocity(0), ConstantVelocity(0), ConstantVelocity(0)]
)
target_position_mapping = [0, 2, 4]
target_velocity_mapping = [1, 3, 5]

target_tonal_bandwidth_hz = rng.uniform(0.5, 2.0)
target_noise_amplitude_upa = 10 ** (90 / 20)
target_noise_spectral_exponent = -1.0

target_truths = []

for sv in [target1_start_vector, target2_start_vector, target3_start_vector]:
    metadata = {
        "position_mapping": target_position_mapping,
        "velocity_mapping": target_velocity_mapping,
        "amplitudes_upa": 10 ** (rng.uniform(87, 102, 4) / 20),
        "frequencies_hz": rng.uniform(50.0, 200.0, 4),
        "phases_rad": rng.uniform(0, 2 * np.pi, 4),
        "tonal_bandwidth_hz": rng.uniform(0.5, 2.0),
        "noise_amplitude_upa": 10 ** (rng.uniform(65, 85) / 20),
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

relative_bearing_truths = []

for target_truth in target_truths:
    bearing_states = []
    for state in target_truth:
        positions = platform.sensor_array.position_matrix_at(state.timestamp)
        ref_sensor_position = np.mean(positions, axis=1)

        target_xy = np.array([state.state_vector[0], state.state_vector[2]])
        relative_position = target_xy - ref_sensor_position[:2]
        bearing = np.arctan2(relative_position[1], relative_position[0])

        bearing_states.append(GroundTruthState(np.array([bearing]), timestamp=state.timestamp))

    relative_bearing_truths.append(GroundTruthPath(bearing_states))

fig_world = plot_world(truths=target_truths, platform=platform).update_layout(
    title="World Picture: Target and Platform Trajectories",
    template="plotly_white",
    autosize=True,
    width=None,
    height=None,
)

# %%
# Propagation Model
# -----------------
#
# The acoustic environment and propagation model are held fixed across all detector
# configurations so that ROC/PR differences reflect detector chain behaviour rather
# than environmental changes.

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
)

# %%
# Signal and Ambient Noise Models
# -------------------------------
#
# The ambient noise model and per-target signal models set the spectral content and
# SNR distribution that determine the difficulty of the detection task.
# :class:`~.SyntheticAnthropogenicSignal` is one-shot per instance, so a factory
# function is used to produce fresh instances for each simulator construction.

sampling_rate_hz = 500.0
frame_len = 500
hop_factor = 2
total_duration_s = num_steps * time_interval.total_seconds()

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
        duration_s=total_duration_s,
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

# %%
# Beamformer and Detector Pipeline Setup
# ---------------------------------------
#
# A broadband delay-and-sum beamformer is used as the baseline. The baseline
# detector run produces the SNR map that the parameter sweep reuses, so all
# sweep configurations operate on the same acoustic data.

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

cfar_num_guard_cells = 6
cfar_num_training_cells = 10
cfar_threshold_factor = 1.05
peak_distance = 8

cfar_detector = CACFARDetector(
    num_guard_cells=cfar_num_guard_cells,
    num_training_cells=cfar_num_training_cells,
    threshold_factor=cfar_threshold_factor,
    mode="wrap",
)
peak_detector = PeakDetector(distance=peak_distance)

detector = PassiveSonarDetector(
    detection_chain=[cfar_detector, peak_detector],
    sensor_data_gen=simulator.sensor_data_gen(),
    steering_azimuths_rad=steering_azimuths_rad,
)

# %%
# Run Detection on Simulated Data
# -------------------------------
#
# The baseline detector is executed once to produce the SNR map and a flattened set
# of detections. The SNR map is passed directly to the parameter sweep, avoiding a
# second simulation run.

all_detections = list(detector.detections_gen(progress_bar=False))
snr_map = detector.snr_history

detections_for_plotter = [d for _, detections in all_detections for d in detections]

print(f"Total no. of detections: {len(detections_for_plotter)}")

# %%
# Baseline Detector Example
# -------------------------
#
# A bearing-time record of the baseline CA-CFAR run provides a qualitative check before the
# parameter sweep. Three target bearing tracks are visible sweeping across the BTR, and an
# ownship heading change at around the 00:07 mark causes all three apparent bearings to shift
# simultaneously. The white markers show where the detector fires, with dense clusters near the
# truth tracks suggesting that the CFAR threshold is reasonably set for this scenario, while
# isolated detections away from the truth lines indicate false alarms or background clutter.
# Inspecting the BTR before the sweep provides geometric intuition that helps interpret any
# anomalies in the ROC and PR curves that follow.

fig_btr = plot_btr(
    data=snr_map,
    timesteps=timesteps,
    steering_azimuths=np.rad2deg(steering_azimuths_rad),
    truths=relative_bearing_truths,
    detections=detections_for_plotter,
).update_layout(
    title="Bearing-Time Record: Baseline Detector",
    template="plotly_white",
    autosize=True,
    width=None,
    height=None,
)

# %%
# Sweep Configurations
# --------------------
#
# Four detector configurations are swept over ``threshold_factor`` to produce ROC and
# PR curves. The first two (CA-CFAR and OS-CFAR alone) show how the choice of CFAR
# variant affects the underlying threshold-to-performance mapping. The second two add
# a :class:`~.PeakDetector` stage, which clusters nearby detections into single peaks
# after thresholding. Because the sweep steps ``algorithm_index=0`` (the CFAR stage)
# while the peak-clustering distance is held fixed, the peak variants produce a
# fundamentally different sweep trajectory — see the Detection Metrics Results cell
# for details. ``target_fpr`` marks the operating point printed in the summary table.

target_fpr = 0.05

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
        label="CA-CFAR + Peak",
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
        label="OS-CFAR + Peak",
    ),
]

# %%
# Run Detection Metrics Sweep
# ---------------------------
#
# The sweep runs each configuration against the pre-computed SNR map, so no second
# simulation pass is required. Each :class:`~.SweepSpec` steps through
# ``param_values`` and records ROC and PR statistics at every operating point.

results = sweep_detection_parameter(
    snr_map=snr_map,
    sweep_specs=specs,
    ground_truth_paths=relative_bearing_truths,
    steering_azimuths_rad=steering_azimuths_rad,
    association_threshold_rad=np.deg2rad(2.0),
)

# %%
# Detection Metrics Results
# -------------------------
#
# The summary table and curves highlight two main results. First, CA-CFAR gives
# the strongest ROC performance (AUC-ROC = 0.8593), while OS-CFAR gives the best
# PR performance among the non-peak variants (AUC-PR = 0.1175), confirming that
# detector ranking depends on the metric used. Second, adding peak clustering
# yields the highest PR score for CA-CFAR + Peak (AUC-PR = 0.1347), but both
# peak-augmented variants produce near-zero AUC-ROC values. This suggests that
# sweeping the CFAR threshold alone, while holding the clustering distance fixed,
# produces a non-standard ROC trajectory: at low thresholds, many raw detections
# are merged into a smaller set of peaks, so false-positive and true-positive
# behaviour no longer evolves monotonically with threshold. As a result, PR is the
# more reliable summary metric for the peak-augmented configurations in this sweep.

headers = [
    "Detector",
    "AUC-ROC",
    "AUC-PR",
    "Best Param (Youden-J)",
    f"Param @ FPR={target_fpr}",
]

rows = [
    [
        r.label,
        f"{r.auc_roc:.4f}",
        f"{r.auc_pr:.4f}",
        f"{r.best_param:.4f}",
        f"{r.param_at_fpr(target_fpr):.4f}",
    ]
    for r in results
]

table = [headers, *rows]
col_widths = [max(len(row[i]) for row in table) for i in range(len(headers))]
separator = "-+-".join("-" * w for w in col_widths)

print(" | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)))
print(separator)
for row in rows:
    print(" | ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(row)))

fig_roc_pr = plot_roc_pr(results).update_layout(
    title="Receiver Operating Characteristic and Precision-Recall Curves",
    template="plotly_white",
    autosize=True,
    width=None,
    height=int(np.clip(260 * 2, 700, 2200)),
)

# %%
# Key Takeaways
# -------------
#
# * **ROC and PR can rank detectors differently** - CA-CFAR achieves the highest
#   AUC-ROC (0.8491), while CA-CFAR + Peak achieves the highest AUC-PR (0.1907).
#   Among the non-peak variants, OS-CFAR has the stronger AUC-PR (0.1655). This is
#   why both ROC and PR curves should be reported.
# * **Peak clustering changes how the ROC sweep behaves** - sweeping only the CFAR
#   threshold while holding ``PeakDetector.distance`` fixed produces a non-standard
#   ROC trajectory for the peak-augmented chains, yielding near-zero AUC-ROC despite
#   competitive AUC-PR values. In this setting, PR is the more interpretable summary
#   metric for the peak configurations.
# * **Sweep only the stage you want to characterise** - ``algorithm_index=0`` sweeps
#   the CFAR threshold while peak clustering remains fixed. To characterise the peak
#   stage itself, set ``algorithm_index=1`` and sweep ``PeakDetector.distance``.
# * **The SNR map is computed once and reused** - ``sweep_detection_parameter``
#   operates on the pre-computed ``snr_map``, so adding more sweep configurations
#   does not require rerunning the simulator.
