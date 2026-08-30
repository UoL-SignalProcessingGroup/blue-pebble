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
across parameter settings or detector configurations.
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
from bluepebble.detector import CACFARDetector, OSCFARDetector, PassiveSonarDetector
from bluepebble.detector.metrics import SweepSpec, sweep_detection_parameter
from bluepebble.models.environment import FlatBathymetry, Linear
from bluepebble.models.propagation import rtrsAcousticPropagationModel
from bluepebble.platform import TowedArrayPlatform
from bluepebble.plotter import plot_btr, plot_roc_pr, plot_world
from bluepebble.signal.anthropogenic import SyntheticAnthropogenicSignal
from bluepebble.signal.random import ColouredNoiseSignal
from bluepebble.sigproc import (
    DelayAndSumBeamformer,
    SteeringCalculator,
    beams_per_mainlobe,
    cfar_window_for_mainlobe,
)
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
        platform_state = platform.get_platform_state_at(state.timestamp)
        assert platform_state is not None
        ref_sensor_position = np.mean(platform_state.array.state_vector, axis=1)

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
# configurations so that ROC/PR differences reflect detector behaviour rather
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
# A broadband delay-and-sum beamformer is used as the baseline. ``'broadband_power'``
# integrates each STFT frame's bins into a real-valued beam power map, which is the per-look
# quantity a CFAR detector expects; the ``'frequency'`` domain would instead hand it one
# complex amplitude per frequency bin. That distinction matters here, because a detector
# calibrated on a false-alarm rate assumes each look is a power sample. The baseline detector
# run and the parameter sweep both consume these same frames, so every configuration is
# compared on identical acoustic data.

steering_azimuths_rad = np.linspace(-np.pi, np.pi, 361)

beamformer = DelayAndSumBeamformer(
    sampling_rate_hz=sampling_rate_hz,
    shading=None,
    domain="broadband_power",
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

# Guard and training cells are set by the array, not chosen. A source spans a mainlobe in
# bearing, so guard cells have to reach past it -- otherwise the training cells measure the
# target and the reported SNR is compressed. Evaluated at 50 Hz, the lowest tonal in the
# scenario and therefore the widest lobe: 15.3 beams across, giving a 16-beam guard band.
# peak_distance falls out of the same number, since two candidates closer than a mainlobe
# cannot be resolved as separate sources.
mainlobe_beams = beams_per_mainlobe(
    aperture_m=(num_sensors - 1) * sensor_spacing_m,
    frequency_hz=50.0,
    beam_spacing_rad=float(np.diff(steering_azimuths_rad)[0]),
    sound_speed_ms=1500.0,
)
cfar_num_guard_cells, cfar_num_training_cells, peak_distance = cfar_window_for_mainlobe(
    mainlobe_beams
)
# OS-CFAR picks the k-th smallest training cell, so its rank has to scale with the window.
# 0.75 of the total is the usual starting point; 0.5 is the lowest that does not trip the
# low-rank warning, and gives the contrast the two OS specs are here to show.
os_rank_high = round(0.75 * 2 * cfar_num_training_cells)
os_rank_low = round(0.50 * 2 * cfar_num_training_cells)
# Reproduces the pre-refactor threshold_factor=1.05 exactly, via CA-CFAR's single-look
# Pfa = (1 + alpha/N)^-N with N = 2 * num_training_cells.
cfar_target_pfa = 0.3594

cfar_detector = CACFARDetector(
    num_guard_cells=cfar_num_guard_cells,
    num_training_cells=cfar_num_training_cells,
    target_pfa=cfar_target_pfa,
    peak_distance=peak_distance,
    circular=True,
)

# The sweep needs the raw beamformed frames, which PassiveSonarDetector does not retain after
# detecting, so the simulator is drained once here and the collected steps are replayed into
# the detector. That keeps this to a single simulation pass while giving the sweep the raw
# data it needs.
sensor_steps = list(simulator.sensor_data_gen())
beamformed_data = [
    sensor_data.beamformed_data
    for _, sensor_data_set in sensor_steps
    for sensor_data in sensor_data_set
]

detector = PassiveSonarDetector(
    detector=cfar_detector,
    sensor_data_gen=sensor_steps,
    steering_azimuths_rad=steering_azimuths_rad,
)

# %%
# Run Detection on Simulated Data
# -------------------------------
#
# The baseline detector is executed once over the collected frames to produce its SNR map
# and a flattened set of detections. The same collected frames feed the parameter sweep, so
# no second simulation run is needed.

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
# Four configurations are swept over ``target_pfa``: CA-CFAR and OS-CFAR, each with peak
# consolidation off and on. Sweeping the requested false-alarm rate rather than a raw threshold
# multiplier means the x-axis asks the same question of both detector families, which need
# quite different multipliers to answer it.
#
# The consolidation axis is the interesting one, and it is what makes the ROC panel readable.
# A CFAR threshold calibrated for a given Pfa produces that rate of threshold crossings, so
# with ``consolidate_peaks=False`` the achieved false-positive rate follows the requested Pfa
# across the whole sweep and the curve covers the full ROC box. Consolidation then merges
# adjacent crossings into one detection per source, which is what you want operationally but
# which caps the achieved rate: on noise-only data at this beam count, reported detections
# hold at the requested Pfa to about 0.05 and then fall away, to roughly 78% of crossings at
# Pfa 0.2 and 37% at 0.9.
#
# Both are legitimate operating modes and the pair is the point: the unconsolidated curves show
# what the calibration delivers, the consolidated ones show what survives being turned into
# one detection per source. ``target_fpr`` marks the operating point printed in the summary
# table.

target_fpr = 0.05

# Spanning four decades of Pfa gives the ROC curve enough spread to be informative at both
# ends. Fewer points than the old threshold sweep used: each point is a full pass over every
# timestep, and Pfa is the meaningful axis, so log spacing covers the interesting range far
# more efficiently than 400 linear steps did.
pfa_values = np.logspace(-4, np.log10(0.9), 60)


specs = [
    # Raw CFAR output: every cell above the threshold. The count is then the achieved false-
    # alarm rate directly, so these curves span the full FPR range and can be read against the
    # requested target_pfa.
    SweepSpec(
        detector=CACFARDetector(
            num_guard_cells=cfar_num_guard_cells,
            num_training_cells=cfar_num_training_cells,
            target_pfa=float(pfa_values[0]),
            circular=True,
            consolidate_peaks=False,
        ),
        param_name="target_pfa",
        param_values=pfa_values,
        label="CA-CFAR",
    ),
    SweepSpec(
        detector=OSCFARDetector(
            num_guard_cells=cfar_num_guard_cells,
            num_training_cells=cfar_num_training_cells,
            rank=os_rank_high,
            target_pfa=float(pfa_values[0]),
            circular=True,
            consolidate_peaks=False,
            rng=np.random.default_rng(seed + 1),
        ),
        param_name="target_pfa",
        param_values=pfa_values,
        label="OS-CFAR",
    ),
    # The same two detectors reporting one detection per source, which is how they would
    # actually be run. Consolidation merges adjacent crossings, so achieved FPR saturates well
    # below the requested Pfa and these curves stop short of the top-right corner.
    SweepSpec(
        detector=CACFARDetector(
            num_guard_cells=cfar_num_guard_cells,
            num_training_cells=cfar_num_training_cells,
            target_pfa=float(pfa_values[0]),
            peak_distance=peak_distance,
            circular=True,
        ),
        param_name="target_pfa",
        param_values=pfa_values,
        label="CA-CFAR + Peak",
    ),
    SweepSpec(
        detector=OSCFARDetector(
            num_guard_cells=cfar_num_guard_cells,
            num_training_cells=cfar_num_training_cells,
            rank=os_rank_low,
            target_pfa=float(pfa_values[0]),
            peak_distance=peak_distance,
            circular=True,
            rng=np.random.default_rng(seed + 2),
        ),
        param_name="target_pfa",
        param_values=pfa_values,
        label="OS-CFAR + Peak",
    ),
]

# %%
# Run Detection Metrics Sweep
# ---------------------------
#
# The sweep runs each configuration against the beamformed frames collected above, so no
# second simulation pass is required. Each :class:`~.SweepSpec` steps through
# ``param_values``, deep-copying its detector and recalibrating it at every operating point,
# and records ROC and PR statistics for each.

results = sweep_detection_parameter(
    beamformed_data=beamformed_data,
    sweep_specs=specs,
    ground_truth_paths=relative_bearing_truths,
    steering_azimuths_rad=steering_azimuths_rad,
    association_threshold_rad=np.deg2rad(2.0),
)

# %%
# Detection Metrics Results
# -------------------------
#
# Read the ranking off the printed table below rather than from fixed numbers here: the
# figures move with the scenario, the seed, and the swept range, and quoting them in prose
# only guarantees they go stale.
#
# Two things are worth looking for. The first is that ROC and PR need not rank the detectors
# the same way, which is the whole reason for reporting both. Positives are rare here -- only
# a handful of the 361 beams hold a target at any timestep -- and PR is far more sensitive to
# that imbalance than ROC is, so a configuration that looks competitive on one can be clearly
# worse on the other.
#
# The second is the span each curve actually covers. A detector whose achieved FPR saturates
# short of 1.0 has not been measured across the full ROC box, and
# :attr:`~.SweepResult.auc_roc` extends its curve to ``(0, 0)`` and ``(1, 1)`` before
# integrating so that the comparison stays meaningful. Without that extension a saturating
# detector scores near zero purely because it stopped early -- an artefact of the sweep, not
# a property of the detector. It does mean that when a sweep covers only part of the range,
# much of the area comes from the extension rather than from measured points, so check
# :attr:`~.SweepResult.fpr` before reading much into a small gap between two AUC values.

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
# * **ROC and PR can rank detectors differently** - positives are rare in a bearing-time
#   map, so the two metrics weigh false alarms very differently. Report both, and read the
#   values off the table rather than assuming one ordering carries over to the other.
# * **Sweep the quantity that means something** - ``target_pfa`` is a false-alarm rate, so
#   the same swept value asks the same question of CA-CFAR and OS-CFAR even though they need
#   different threshold multipliers to answer it. A raw multiplier is not comparable across
#   detector families.
# * **The noise-floor estimator is the real design choice** - window width for CA-CFAR, rank
#   for OS-CFAR. Both shape what the detector treats as background, and both move the curves
#   more than peak consolidation does on this scenario.
# * **Clustering is now a detector parameter** - consolidation happens inside the detector, so
#   ``peak_distance`` is swept or held fixed like any other parameter. There is no separate
#   stage to index into.
# * **Watch the FPR span before comparing AUCs** - a detector whose achieved FPR saturates
#   early covers only part of the ROC box. ``auc_roc`` extends curves to the corners to keep
#   the comparison honest, but a narrow span still rests mostly on that extension. Check
#   :attr:`~.SweepResult.fpr` before reading much into small gaps.
# * **The beamformed frames are collected once and reused** - ``sweep_detection_parameter``
#   operates on the frames gathered for the baseline run, so adding more sweep configurations
#   does not require rerunning the simulator.
