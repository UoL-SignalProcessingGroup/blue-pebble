"""
================================
Designing Narrowband Beamformers
================================

This example compares three narrowband MVDR frequency selections on a shared
passive-sonar scenario containing two targets, each emitting a single tonal line at a
distinct frequency. The aim is to isolate how band choice changes peak sharpness,
clutter structure, detection count, and bearing stability when everything else in the
pipeline is held fixed.

The three bands are chosen deliberately: one straddles both target tonals, one covers
only the lower tonal, and one covers only the upper tonal. That makes the effect of
including or excluding a source frequency directly visible in the BTR and detections.
"""  # noqa: D205, D212, D400, D415

# %%
# Imports
# -------
#
# All dependencies are consolidated here for convenience.

from datetime import datetime, timedelta

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel,
    ConstantVelocity,
    KnownTurnRate,
)
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState

from bluepebble.detector import CACFARDetector, PassiveSonarDetector, PeakDetector
from bluepebble.models.environment import FlatBathymetry, Linear
from bluepebble.models.propagation import rtrsAcousticPropagationModel
from bluepebble.platform import TowedArrayPlatform
from bluepebble.plotter import apply_shared_colourscale, deduplicate_legend, plot_btr, plot_world
from bluepebble.signal.anthropogenic import SyntheticAnthropogenicSignal
from bluepebble.signal.random import ColouredNoiseSignal
from bluepebble.sigproc import MinimumVarianceDistortionlessResponseBeamformer, SteeringCalculator
from bluepebble.simulator import ContinuousSTFTPassiveSonarArraySimulator

# %%
# Simulation Parameters
# ---------------------
#
# This section fixes the random seed and defines the simulation clock. The seed
# controls the platform manoeuvre sequence and all stochastic signal components, making
# every run fully reproducible.

seed = 2000
np.random.seed(seed)

sim_length_s = 900  # seconds
sim_rate_s = 5.0  # seconds

start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
time_interval = timedelta(seconds=sim_rate_s)
num_steps = int(sim_length_s / sim_rate_s)
timesteps = np.array([start_time + i * time_interval for i in range(num_steps)], dtype=object)

total_duration_s = num_steps * time_interval.total_seconds()

# %%
# Platform Manoeuvre Generator
# ----------------------------
#
# This helper builds a sequence of straight and turning legs by drawing random
# durations and turn angles from the seeded RNG. It is scaffolding — the resulting
# transition model list is what matters downstream.


def _generate_random_manoeuvres(
    total_duration_s,
    timestep_s,
    min_leg_duration_s=60,
    max_leg_duration_s=300,
    manoeuvre_prob=0.5,
):
    manoeuvre_models = []
    manoeuvre_durations = []
    manoeuvre_descriptions = []

    current_time = 0.0
    while current_time < total_duration_s - min_leg_duration_s:
        leg_duration = np.random.uniform(min_leg_duration_s, max_leg_duration_s)
        leg_duration = round(leg_duration / timestep_s) * timestep_s
        leg_duration = min(leg_duration, total_duration_s - current_time)

        if np.random.rand() > manoeuvre_prob:
            model = CombinedLinearGaussianTransitionModel(
                [ConstantVelocity(0.0), ConstantVelocity(0.0), ConstantVelocity(0.0)]
            )
            description = f"Straight ({leg_duration:.0f}s)"
        else:
            turn_angle_deg = np.random.uniform(-90, 90)
            turn_rate_deg_per_s = 1.0

            turn_duration_s = abs(turn_angle_deg) / turn_rate_deg_per_s
            turn_duration_s = round(turn_duration_s / timestep_s) * timestep_s
            leg_duration = min(turn_duration_s, total_duration_s - current_time)

            turn_rate_rad = np.deg2rad(np.sign(turn_angle_deg) * turn_rate_deg_per_s)
            planar_turn = KnownTurnRate(
                turn_rate=turn_rate_rad,
                turn_noise_diff_coeffs=np.array([0.0, 0.0]),
            )
            model = CombinedLinearGaussianTransitionModel([planar_turn, ConstantVelocity(0.0)])

            direction = "Left" if turn_angle_deg > 0 else "Right"
            description = f"{direction} Turn {abs(turn_angle_deg):.0f}° ({leg_duration:.0f}s)"

        manoeuvre_models.append(model)
        manoeuvre_durations.append(timedelta(seconds=leg_duration))
        manoeuvre_descriptions.append(description)
        current_time += leg_duration

    return manoeuvre_models, manoeuvre_durations, manoeuvre_descriptions


# %%
# Platform Setup and Generation
# -----------------------------
#
# The platform manoeuvre sequence is drawn from the fixed seed set above, so the array
# heading changes are deterministic but varied. Changing heading mid-run is important
# for this example: it shifts the apparent bearing of both targets relative to the array,
# which exercises the beamformer across a wider range of steering angles.

platform_start_vector = np.array([5000.0, 0.0, 1000.0, 5.0, -5.0, 0.0])
platform_position_mapping = [0, 2, 4]
platform_velocity_mapping = [1, 3, 5]

num_sensors = 50
tow_cable_length_m = 100.0
sensor_spacing_m = 0.5
array_depth_m = -50.0

manoeuvre_models, manoeuvre_durations, manoeuvre_descriptions = _generate_random_manoeuvres(
    total_duration_s=total_duration_s,
    timestep_s=time_interval.total_seconds(),
    min_leg_duration_s=60,
    max_leg_duration_s=300,
    manoeuvre_prob=0.25,
)

initial_state = GroundTruthState(platform_start_vector, timestamp=start_time)
platform = TowedArrayPlatform(
    states=[initial_state],
    position_mapping=platform_position_mapping,
    velocity_mapping=platform_velocity_mapping,
    transition_models=manoeuvre_models,
    transition_times=manoeuvre_durations,
    num_sensors=num_sensors,
    cable_length_m=tow_cable_length_m,
    sensor_spacing_m=sensor_spacing_m,
    array_depth_m=array_depth_m,
)

for i in range(1, num_steps):
    platform.move(start_time + i * time_interval)

# %%
# Ground Truth Setup and Generation
# ---------------------------------
#
# Two targets are defined here, each radiating a single tonal: Target 1 at 75 Hz and
# Target 2 at 100 Hz. These frequencies were chosen to align with the three frequency
# bands in the Beamformer section. Cartesian state vectors are propagated over the full
# timeline, then converted to relative-bearing truths for BTR overlays. The
# :func:`~bluepebble.plotter.plot_world` figure confirms the geometry before beamforming.

constant_velocity_model = CombinedLinearGaussianTransitionModel(
    [ConstantVelocity(0.0), ConstantVelocity(0.0), ConstantVelocity(0.0)]
)

target_configs = [
    {
        "name": "Target 1",
        "start_vector": np.array([4500.0, 5.0, 2000.0, 0.0, -5.0, 0.0]),
        "transition_model": constant_velocity_model,
        "metadata": {
            "amplitudes_upa": 10 ** (np.array([90.0]) / 20),
            "frequencies_hz": [75.0],
            "phases_rad": [0.0],
            "position_mapping": [0, 2, 4],
            "velocity_mapping": [1, 3, 5],
            "tonal_bandwidth_hz": 2.0,
            "noise_amplitude_upa": 10 ** (70 / 20),
            "noise_spectral_exponent": -1.0,
        },
    },
    {
        "name": "Target 2",
        "start_vector": np.array([1000.0, 5.0, 3000.0, 3.0, -5.0, 0.0]),
        "transition_model": constant_velocity_model,
        "metadata": {
            "amplitudes_upa": 10 ** (np.array([100.0]) / 20),
            "frequencies_hz": [100.0],
            "phases_rad": [0.0],
            "position_mapping": [0, 2, 4],
            "velocity_mapping": [1, 3, 5],
            "tonal_bandwidth_hz": 2.0,
            "noise_amplitude_upa": 10 ** (60 / 20),
            "noise_spectral_exponent": -1.0,
        },
    },
]

target_ground_truths = []
relative_bearing_ground_truths = []

for target_config in target_configs:
    target_states = [
        GroundTruthState(
            target_config["start_vector"],
            timestamp=start_time,
            metadata=target_config["metadata"],
        )
    ]

    for i in range(1, num_steps):
        new_time = start_time + i * time_interval
        interval_now = new_time - target_states[-1].timestamp
        new_state_vector = target_config["transition_model"].function(
            target_states[-1],
            noise=False,
            time_interval=interval_now,
        )
        target_states.append(
            GroundTruthState(
                new_state_vector,
                timestamp=new_time,
                metadata=target_states[-1].metadata,
            )
        )

    target_ground_truth = GroundTruthPath(target_states)
    target_ground_truths.append(target_ground_truth)

    bearing_states = []
    for target_state in target_ground_truth.states:
        platform_state = platform.get_platform_state_at(target_state.timestamp)
        assert platform_state is not None
        ref_sensor_position = np.mean(platform_state.array.state_vector, axis=1)
        target_position = np.array([target_state.state_vector[0], target_state.state_vector[2]])
        relative_position = target_position - ref_sensor_position[:2]
        bearing_rad = np.arctan2(relative_position[1], relative_position[0])
        bearing_states.append(
            GroundTruthState(
                state_vector=np.array([bearing_rad]),
                timestamp=target_state.timestamp,
            )
        )

    relative_bearing_ground_truths.append(GroundTruthPath(bearing_states))

fig_world = plot_world(truths=target_ground_truths, platform=platform).update_layout(
    title="World Picture: Target and Platform Trajectories",
    template="plotly_white",
    autosize=True,
    width=None,
    height=None,
)

# %%
# Acoustic Environment and Signal Models
# --------------------------------------
#
# The propagation model and signal models are both held fixed across all three beamformer
# runs so that output differences come from band selection alone.
#
# :class:`~.rtrsAcousticPropagationModel` is configured with a linear SSP and a flat
# 150 m seabed. Each target is modelled as a single tonal line with low-level pink noise
# using :class:`~.SyntheticAnthropogenicSignal`; tonal frequencies come from each
# target's metadata. Ambient pink noise is added at the array once per integration
# interval.

ssp = Linear(surface_speed=1500.0, gradient=0.2)
bathymetry = FlatBathymetry(depth=-150.0)
prop_step_m = 20.0
prop_azimuth_search_width = 2.0
prop_azimuth_resolution = 0.5
prop_elevation_range = (-25.0, 25.0)
prop_elevation_resolution = 1.0

propagation_model = rtrsAcousticPropagationModel(
    ssp=ssp,
    bathymetry=bathymetry,
    step_m=prop_step_m,
    azimuth_search_width=prop_azimuth_search_width,
    azimuth_resolution=prop_azimuth_resolution,
    elevation_range=prop_elevation_range,
    elevation_resolution=prop_elevation_resolution,
)

sampling_rate_hz = 500.0
frame_len = 500
hop_factor = 2
fade_in_ms = 1000.0

ambient_amplitude_upa = 10 ** (50.0 / 20)
ambient_spectral_exponent = -1.0

ambient_noise_model = ColouredNoiseSignal(
    amplitude_upa=ambient_amplitude_upa,
    spectral_exponent=ambient_spectral_exponent,
    duration_s=time_interval.total_seconds(),
    sampling_rate_hz=sampling_rate_hz,
)


def _make_signal_models():
    models = []
    for target_ground_truth in target_ground_truths:
        target_metadata = next(iter(target_ground_truth)).metadata
        models.append(
            SyntheticAnthropogenicSignal(
                duration_s=total_duration_s,
                sampling_rate_hz=sampling_rate_hz,
                frame_len=frame_len,
                hop_factor=hop_factor,
                tonal_bandwidth_hz=target_metadata["tonal_bandwidth_hz"],
                noise_amplitude_upa=target_metadata["noise_amplitude_upa"],
                noise_spectral_exponent=target_metadata["noise_spectral_exponent"],
                noise_freq_range_hz=(0.0, sampling_rate_hz / 2),
                tonal_noise_is_constant=True,
                noise_is_constant=True,
            )
        )
    return models


# %%
# Beamformer
# ----------
#
# Three frequency bands are defined, chosen to probe the two target tonals in
# different combinations:
#
# - **70–105 Hz**: straddles both the 75 Hz and 100 Hz tonals simultaneously.
# - **70–80 Hz**: covers Target 1 (75 Hz) only; Target 2 is outside the band.
# - **95–105 Hz**: covers Target 2 (100 Hz) only; Target 1 is outside the band.
#
# One :class:`~.MinimumVarianceDistortionlessResponseBeamformer` is constructed per
# band. All other pipeline components are shared so any output differences are
# attributable solely to band selection.

steering_azimuths_rad = np.linspace(-np.pi, np.pi, 101)

freq_bands = [
    {"label": "70-105 Hz", "fmin_hz": 70.0, "fmax_hz": 105.0},
    {"label": "70-80 Hz", "fmin_hz": 70.0, "fmax_hz": 80.0},
    {"label": "95-105 Hz", "fmin_hz": 95.0, "fmax_hz": 105.0},
]

beamformers = []
steering_calculators = []
for config in freq_bands:
    beamformer = MinimumVarianceDistortionlessResponseBeamformer(
        sampling_rate_hz=sampling_rate_hz,
        fmin=config["fmin_hz"],
        fmax=config["fmax_hz"],
    )
    steering_calculator = SteeringCalculator(
        ssp=ssp,
        steering_azimuths_rad=steering_azimuths_rad,
    )

    beamformers.append(beamformer)
    steering_calculators.append(steering_calculator)

# %%
# Detector Pipeline Setup
# -----------------------
#
# This section builds one simulator + detector pipeline per beamformer configuration.
#
# All pipelines share the same scenario and detector chain so output differences come from
# beamformer band selection rather than different downstream logic.

cfar_num_guard_cells = 2
cfar_num_training_cells = 10
cfar_threshold_factor = 2.1
peak_distance = 3


def _make_detector(simulator, steering_azimuths_rad):
    cfar_detector = CACFARDetector(
        num_guard_cells=cfar_num_guard_cells,
        num_training_cells=cfar_num_training_cells,
        threshold_factor=cfar_threshold_factor,
    )
    peak_detector = PeakDetector(distance=peak_distance)
    return PassiveSonarDetector(
        detection_chain=[cfar_detector, peak_detector],
        sensor_data_gen=simulator.sensor_data_gen(),
        steering_azimuths_rad=steering_azimuths_rad,
    )


simulators = []
detectors = []
for beamformer, steering_calculator in zip(beamformers, steering_calculators, strict=False):
    simulator = ContinuousSTFTPassiveSonarArraySimulator(
        platform=platform,
        propagation_model=propagation_model,
        signal_models=_make_signal_models(),
        noise_model=ambient_noise_model,
        beamformer=beamformer,
        steering_calculator=steering_calculator,
        ground_truth_paths=target_ground_truths,
        fade_in_ms=fade_in_ms,
    )
    simulators.append(simulator)
    detectors.append(_make_detector(simulator, steering_azimuths_rad))

# %%
# Run Detection on Simulated Data
# -------------------------------
#
# This cell executes all three detector pipelines and collects the results. A summary
# line is printed for each band showing detection count and peak SNR — a first
# indication of how band choice affects sensitivity before the figures are inspected.

all_detections_per_bf = []
snr_maps = []
detections_flat_per_bf = []
comparison_summary = []

for detector, config in zip(detectors, freq_bands, strict=False):
    all_detections = list(detector.detections_gen(progress_bar=False, total_timesteps=num_steps))
    snr_map = detector.snr_history
    detections_flat = [d for _, detection_set in all_detections for d in detection_set]

    all_detections_per_bf.append(all_detections)
    snr_maps.append(snr_map)
    detections_flat_per_bf.append(detections_flat)
    comparison_summary.append(
        {
            "label": config["label"],
            "num_detections": len(detections_flat),
            "snr_peak_db": float(np.max(snr_map)),
        }
    )

for summary in comparison_summary:
    print(
        f"{summary['label']}: {summary['num_detections']} detections, "
        f"peak SNR {summary['snr_peak_db']:.2f} dB"
    )

# %%
# Results: SNR Maps and Detections Across Frequency Bands
# --------------------------------------------------------
#
# Each row corresponds to one frequency band. The left column shows the raw SNR map;
# the right column overlays detections and bearing ground truths. A shared colour scale
# across all heatmaps means level differences between bands are directly comparable.

n_bands = len(freq_bands)

fig_results = make_subplots(
    rows=n_bands,
    cols=2,
    shared_xaxes=True,
    shared_yaxes=False,
    vertical_spacing=0.04,
    horizontal_spacing=0.06,
    row_titles=[config["label"] for config in freq_bands],
    column_titles=["SNR Map", "SNR Map w/ Detections"],
)

for row, (snr_map, detections_flat) in enumerate(
    zip(snr_maps, detections_flat_per_bf, strict=False), start=1
):
    map_rows = min(len(timesteps), snr_map.shape[0])
    timesteps_map = np.array(timesteps[:map_rows])
    snr_map_plot = snr_map[:map_rows]

    plot_btr(
        data=snr_map_plot,
        timesteps=timesteps_map,
        steering_azimuths=np.rad2deg(steering_azimuths_rad),
        fig=fig_results,
        row=row,
        col=1,
    )
    plot_btr(
        data=snr_map_plot,
        detections=detections_flat,
        truths=relative_bearing_ground_truths,
        timesteps=timesteps_map,
        steering_azimuths=np.rad2deg(steering_azimuths_rad),
        fig=fig_results,
        row=row,
        col=2,
    )

    fig_results.update_yaxes(title_text="Time (HH:MM)", row=row, col=1)
    fig_results.update_yaxes(title_text="", showticklabels=False, row=row, col=2)

    if row < n_bands:
        fig_results.update_xaxes(title_text="", showticklabels=False, row=row, col=1)
        fig_results.update_xaxes(title_text="", showticklabels=False, row=row, col=2)

fig_results.update_xaxes(title_text="Bearing (°)", row=n_bands, col=1)
fig_results.update_xaxes(title_text="Bearing (°)", row=n_bands, col=2)

apply_shared_colourscale(
    fig_results,
    colorbar=dict(
        title=dict(text="SNR (dB)", side="right"),
        x=1.1,
        xanchor="left",
        y=0.5,
        yanchor="middle",
        len=1.0,
        thickness=24,
    ),
)

fig_results.update_layout(
    template="plotly_white",
    autosize=True,
    width=None,
    height=int(np.clip(260 * n_bands, 700, 2200)),
    margin=dict(r=120),
    title="SNR Maps and Detections Across Frequency Bands",
    legend=dict(x=0.5, y=-0.15, xanchor="center", yanchor="top", orientation="h"),
)

# %%
# Bearing-Cut Diagnostics
# -----------------------
#
# A single mid-scenario timestep is extracted from each band's SNR map and plotted as
# a bearing cut. Stacking all three bands in one figure makes it easier to compare peak
# width and height across bands, verify whether detections land on the truth bearing or
# are offset, and see whether the MVDR null structure changes with band selection.

middle_timestep_idx = num_steps // 2
middle_time = timesteps[middle_timestep_idx]
steering_azimuths_deg = np.rad2deg(steering_azimuths_rad)
colorway = px.colors.qualitative.Plotly

fig_cuts = make_subplots(
    rows=n_bands,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    row_titles=[config["label"] for config in freq_bands],
)

for row, (all_detections, snr_map) in enumerate(
    zip(all_detections_per_bf, snr_maps, strict=False), start=1
):
    detections_by_time = [detection_set for _, detection_set in all_detections]
    cut_idx = min(middle_timestep_idx, snr_map.shape[0] - 1)
    middle_snr = snr_map[cut_idx, :]

    y_span = max(float(np.ptp(middle_snr)), 1.0)
    y_pad = 0.05 * y_span
    y_min = float(np.min(middle_snr) - y_pad)
    y_max = float(np.max(middle_snr) + y_pad)

    fig_cuts.add_trace(
        go.Scatter(
            x=steering_azimuths_deg,
            y=middle_snr,
            mode="lines",
            line=dict(width=2, color="black"),
            showlegend=False,
        ),
        row=row,
        col=1,
    )

    for target_idx, gt_path in enumerate(relative_bearing_ground_truths):
        gt_bearing_deg = float(np.rad2deg(gt_path.states[cut_idx].state_vector[0]))
        fig_cuts.add_trace(
            go.Scatter(
                x=[gt_bearing_deg, gt_bearing_deg],
                y=[y_min, y_max],
                mode="lines",
                line=dict(color=colorway[target_idx % len(colorway)], dash="dash", width=2),
                name=f"Truth {target_idx + 1}",
            ),
            row=row,
            col=1,
        )

    middle_detections = list(detections_by_time[min(cut_idx, len(detections_by_time) - 1)])
    for det_idx, det in enumerate(middle_detections):
        det_bearing_deg = float(np.rad2deg(det.state_vector[0]))
        fig_cuts.add_trace(
            go.Scatter(
                x=[det_bearing_deg, det_bearing_deg],
                y=[y_min, y_max],
                mode="lines",
                line=dict(color="green", dash="dot", width=2),
                name="Detection",
                showlegend=det_idx == 0,
                opacity=0.75,
            ),
            row=row,
            col=1,
        )

    fig_cuts.update_yaxes(title_text="SNR (dB)", range=[y_min, y_max], row=row, col=1)
    if row < n_bands:
        fig_cuts.update_xaxes(showticklabels=False, row=row, col=1)

deduplicate_legend(fig_cuts)

fig_cuts.update_xaxes(
    title_text="Bearing (°)",
    row=n_bands,
    col=1,
    range=[-180, 180],
    tickmode="linear",
    tick0=-180,
    dtick=360 / 6.0 if 360 > 0 else 1.0,
    tickangle=-45,
)
fig_cuts.update_layout(
    template="plotly_white",
    autosize=True,
    width=None,
    height=int(np.clip(260 * n_bands, 700, 2200)),
    title=f"SNR Bearing Cuts at {middle_time.time()}",
    legend=dict(x=1.08, y=0.5, xanchor="left", yanchor="middle"),
)

# %%
# Key Takeaways
# -------------
#
# * **The wide band (70–105 Hz) sees both targets but competes with itself** — both
#   tonals contribute energy across the band, which can broaden peaks and produce
#   competing detections when the targets are close in bearing.
# * **Narrow single-target bands produce cleaner, better-localised peaks** — 70–80 Hz
#   isolates Target 1 and 95–105 Hz isolates Target 2; each should show a sharper peak
#   and fewer spurious detections than the wide band for that target.
# * **A target outside the band simply disappears** — 70–80 Hz produces no response
#   at the 100 Hz bearing, and 95–105 Hz produces none at the 75 Hz bearing.
# * **Swap** :class:`~.rtrsAcousticPropagationModel` for
#   :class:`~.CylindricalAcousticPropagationModel` to run the same comparison faster
#   without a ray-trace solver; the band-selection effects are the same.
# * **Adjust** ``cfar_threshold_factor`` to trade detection sensitivity against false
#   alarm rate independently of band choice.
