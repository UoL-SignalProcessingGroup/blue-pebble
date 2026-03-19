"""
================================
FUSION 2026 Single-Target Example
================================

This example assembles the single-target demonstration used for the FUSION 2026
workflow. It runs the acoustic simulation, produces detections, compares them with a
Stone Soup baseline, and prepares figures for presentation output.

Single-target scenarios are often the clearest way to explain the passive-sonar
pipeline before moving to more ambiguous multi-target cases. Keeping the example
export-oriented makes it easier to reuse in papers, talks, and reproducibility packages.
"""  # noqa: D205, D212, D400, D415

# %%
# Imports
# -------

# %%
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import get_window
from scipy.stats import uniform
from stonesoup.dataassociator.probability import PDA
from stonesoup.functions import gm_reduce_single
from stonesoup.hypothesiser.probability import PDAHypothesiser
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel,
    ConstantVelocity,
)
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.types.array import StateVectors
from stonesoup.types.detection import Detection
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.types.state import GaussianState
from stonesoup.types.track import Track
from stonesoup.types.update import GaussianStateUpdate
from stonesoup.updater.kalman import ExtendedKalmanUpdater

from bluepebble.detector import CACFARDetector, PassiveSonarDetector, PeakDetector
from bluepebble.models.environment import FlatBathymetry, Linear
from bluepebble.models.propagation import CylindricalAcousticPropagationModel
from bluepebble.platform import TowedArrayPlatform
from bluepebble.signal.anthropogenic import SyntheticAnthropogenicSignal
from bluepebble.signal.random import ColouredNoiseSignal
from bluepebble.sigproc import (
    DelayAndSumBeamformer,
    MinimumVarianceDistortionlessResponseBeamformer,
    SteeringCalculator,
)
from bluepebble.simulator import ContinuousSTFTPassiveSonarArraySimulator

# %%
# Configuration
# -------------

# %%
seed = 2000
np.random.seed(seed)

sim_length = 900
sim_rate = 5.0

sim_params = {
    "start_time": datetime.now().replace(hour=0, minute=0, second=0, microsecond=0),
    "time_interval": timedelta(seconds=sim_rate),
    "num_steps": int(sim_length / sim_rate),
    "sim_length": sim_length,
}

ship_params = {
    "start_vector": np.array([-2000.0, 5.0, 2000.0, 0.0, -5.0, 0.0]),
    "position_mapping": [0, 2, 4],
    "velocity_mapping": [1, 3, 5],
    "transition_model": CombinedLinearGaussianTransitionModel(
        [ConstantVelocity(0), ConstantVelocity(0), ConstantVelocity(0)]
    ),
}

array_params = {
    "num_sensors": 50,
    "tow_cable_length": 100.0,
    "sensor_spacing": 0.5,
    "array_depth": -50.0,
}

num_targets = 1
targets = []
for _ in range(num_targets):
    targets.append(
        {
            "start_vector": np.array([0.0, 0.0, 0.0, 8.0, -5.0, 0.0]),
            "position_mapping": [0, 2, 4],
            "velocity_mapping": [1, 3, 5],
            "transition_model": CombinedLinearGaussianTransitionModel(
                [ConstantVelocity(0), ConstantVelocity(0), ConstantVelocity(0)]
            ),
            "amplitudes_upa": 10 ** (np.random.uniform(87, 102, 4) / 20),
            "frequencies_hz": np.random.uniform(25.0, 200.0, 4),
            "phases_rad": np.random.uniform(0, 2 * np.pi, 4),
            "tonal_bandwidth_hz": np.random.uniform(0.5, 2.0),
            "noise_amplitude_upa": 10 ** (np.random.uniform(65, 85) / 20),
            "noise_spectral_exponent": -1.0,
        }
    )

total_duration_s = sim_params["num_steps"] * sim_params["time_interval"].total_seconds()

signal_params = {
    "duration_s": total_duration_s,
    "sampling_rate_hz": 500.0,
    "frame_len": 500,
    "hop_factor": 2,
    "fade_in_ms": 1000.0,
}

ambient_noise_params = {
    "amplitude_upa": 10 ** (np.random.uniform(45, 55) / 20),
    "spectral_exponent": -1,
}

prop_params = {
    "ssp": Linear(surface_speed=1500.0, gradient=0.2),
    "attenuation_factor": 0.5,
    "bathymetry": FlatBathymetry(depth=-150.0),
}

bf_params = {
    "beamformer_type": "MVDR",
    "shading": None,
    "domain": "broadband_power",
    "steering_azimuths_rad": np.linspace(-np.pi, np.pi, 181),
    "fmin": 100.0,
    "fmax": 125.0,
}

det_params = {
    "cfar_detector": {
        "num_guard_cells": 2,
        "num_training_cells": 8,
        "threshold_factor": 1.95,
    },
    "peak_detector": {
        "distance": 3,
    },
}

cfg = {
    "seed": seed,
    "num_targets": num_targets,
    "sim": sim_params,
    "ship": ship_params,
    "array": array_params,
    "targets": targets,
    "signal": signal_params,
    "ambient_noise": ambient_noise_params,
    "propagation": prop_params,
    "beamforming": bf_params,
    "detection": det_params,
    "total_duration_s": total_duration_s,
}

print(f"Total simulation duration: {cfg['total_duration_s']} seconds")
print(
    f"Number of timesteps: {cfg['sim']['num_steps']}, "
    f"Timestep interval: {cfg['sim']['time_interval'].total_seconds()} seconds"
)

# %%
# Scenario Build, Simulation, and Detection
# -----------------------------------------

# %%
sim = cfg["sim"]
ship = cfg["ship"]
array = cfg["array"]

initial_state = GroundTruthState(ship["start_vector"], timestamp=sim["start_time"])

platform = TowedArrayPlatform(
    states=initial_state,
    position_mapping=ship["position_mapping"],
    velocity_mapping=ship["velocity_mapping"],
    transition_models=[ship["transition_model"]],
    transition_times=[timedelta(seconds=sim["sim_length"])],
    num_sensors=array["num_sensors"],
    cable_length_m=array["tow_cable_length"],
    sensor_spacing_m=array["sensor_spacing"],
    array_depth_m=array["array_depth"],
)

for i in range(1, sim["num_steps"]):
    platform.move(sim["start_time"] + i * sim["time_interval"])

target_ground_truths: list[GroundTruthPath] = []
relative_bearing_ground_truths: list[GroundTruthPath] = []

for target_cfg in cfg["targets"]:
    target_states = [
        GroundTruthState(
            target_cfg["start_vector"],
            timestamp=sim["start_time"],
            metadata={
                "amplitudes_upa": target_cfg["amplitudes_upa"],
                "frequencies_hz": target_cfg["frequencies_hz"],
                "phases_rad": target_cfg["phases_rad"],
                "position_mapping": target_cfg["position_mapping"],
                "velocity_mapping": target_cfg["velocity_mapping"],
                "tonal_bandwidth_hz": target_cfg["tonal_bandwidth_hz"],
                "noise_amplitude_upa": target_cfg["noise_amplitude_upa"],
                "noise_spectral_exponent": target_cfg["noise_spectral_exponent"],
            },
        )
    ]

    transition_model = target_cfg["transition_model"]
    for i in range(1, sim["num_steps"]):
        new_time = sim["start_time"] + i * sim["time_interval"]
        time_interval = new_time - target_states[-1].timestamp
        new_state_vector = transition_model.function(
            target_states[-1], noise=False, time_interval=time_interval
        )
        target_states.append(
            GroundTruthState(
                new_state_vector,
                timestamp=new_time,
                metadata=target_states[-1].metadata,
            )
        )

    target_truth = GroundTruthPath(target_states)
    target_ground_truths.append(target_truth)

    gt_relative_bearings = []
    for target_state in target_truth.states:
        platform_state = platform.get_platform_state_at(target_state.timestamp)
        ref_sensor_position = np.mean(platform_state.array.state_vector, axis=1)
        target_pos = np.array([target_state.state_vector[0], target_state.state_vector[2]])
        relative_pos = target_pos - ref_sensor_position[:2]
        gt_relative_bearings.append(np.arctan2(relative_pos[1], relative_pos[0]))

    bearing_states = []
    for i, bearing in enumerate(np.asarray(gt_relative_bearings)):
        bearing_states.append(
            GroundTruthState(
                state_vector=np.array([bearing]),
                timestamp=sim["start_time"] + i * sim["time_interval"],
            )
        )

    relative_bearing_ground_truths.append(GroundTruthPath(bearing_states))

prop = cfg["propagation"]
signal = cfg["signal"]
ambient = cfg["ambient_noise"]
bf = cfg["beamforming"]

prop_model = CylindricalAcousticPropagationModel(
    ssp=prop["ssp"],
    attenuation_factor=prop["attenuation_factor"],
)

ambient_noise_model = ColouredNoiseSignal(
    amplitude_upa=ambient["amplitude_upa"],
    spectral_exponent=ambient["spectral_exponent"],
    duration_s=cfg["sim"]["time_interval"].total_seconds(),
    sampling_rate_hz=signal["sampling_rate_hz"],
)

shading = None
if bf["shading"] is not None:
    shading = get_window(bf["shading"], platform.num_sensors)

if bf["beamformer_type"] == "DAS":
    if bf["domain"] == "broadband_power":
        beamformer = DelayAndSumBeamformer(
            domain=bf["domain"],
            sampling_rate_hz=signal["sampling_rate_hz"],
            fmin=bf["fmin"],
            fmax=bf["fmax"],
        )
    else:
        beamformer = DelayAndSumBeamformer(
            sampling_rate_hz=signal["sampling_rate_hz"],
            shading=shading,
            domain=bf["domain"],
        )
elif bf["beamformer_type"] == "MVDR":
    beamformer = MinimumVarianceDistortionlessResponseBeamformer(
        sampling_rate_hz=signal["sampling_rate_hz"],
        fmin=bf["fmin"],
        fmax=bf["fmax"],
    )
else:
    raise ValueError(f"Unknown beamformer type: {bf['beamformer_type']}")

steering_calculator = SteeringCalculator(
    ssp=prop["ssp"],
    steering_azimuths_rad=bf["steering_azimuths_rad"],
)

# Keep same behaviour: use first target's signal settings for simulator signal model.
target_cfg = cfg["targets"][0]
signal_model = SyntheticAnthropogenicSignal(
    duration_s=signal["duration_s"],
    sampling_rate_hz=signal["sampling_rate_hz"],
    frame_len=signal["frame_len"],
    hop_factor=signal["hop_factor"],
    tonal_bandwidth_hz=target_cfg["tonal_bandwidth_hz"],
    noise_amplitude_upa=target_cfg["noise_amplitude_upa"],
    noise_spectral_exponent=target_cfg["noise_spectral_exponent"],
    noise_freq_range_hz=(0.0, signal["sampling_rate_hz"] / 2),
    tonal_noise_is_constant=True,
    noise_is_constant=True,
)

simulator = ContinuousSTFTPassiveSonarArraySimulator(
    platform=platform,
    propagation_model=prop_model,
    signal_models=[signal_model],
    noise_model=ambient_noise_model,
    beamformer=beamformer,
    steering_calculator=steering_calculator,
    ground_truth_paths=target_ground_truths,
    fade_in_ms=signal["fade_in_ms"],
)

print("Running detection chain...")

det = cfg["detection"]
bf = cfg["beamforming"]

cfar_detector = CACFARDetector(
    num_guard_cells=det["cfar_detector"]["num_guard_cells"],
    num_training_cells=det["cfar_detector"]["num_training_cells"],
    threshold_factor=det["cfar_detector"]["threshold_factor"],
)

detection_chain = [cfar_detector]
if det["peak_detector"]["distance"] > 0:
    detection_chain.append(PeakDetector(distance=det["peak_detector"]["distance"]))

detector = PassiveSonarDetector(
    detection_chain=detection_chain,
    sensor_data_gen=simulator.sensor_data_gen(),
    steering_azimuths_rad=bf["steering_azimuths_rad"],
)

all_detections = list(detector.detections_gen(progress_bar=False))
snr_map = detector.snr_history

timesteps = [
    cfg["sim"]["start_time"] + i * cfg["sim"]["time_interval"]
    for i in range(cfg["sim"]["num_steps"])
]

# %%
# Tracking and Stone Soup Baseline Detections
# -------------------------------------------

# %%
relative_bearing_ground_truth = relative_bearing_ground_truths[0]

transition_model = ConstantVelocity(0.000001)
predictor = KalmanPredictor(transition_model)

measurement_model = LinearGaussian(
    ndim_state=2,
    mapping=[0],
    noise_covar=np.array([[np.deg2rad(1) ** 2]]),
)
updater = ExtendedKalmanUpdater(measurement_model=measurement_model)

hypothesiser = PDAHypothesiser(
    predictor=predictor,
    updater=updater,
    clutter_spatial_density=5 / np.pi,
    prob_detect=0.95,
)
data_associator = PDA(hypothesiser=hypothesiser)

initial_bearing = relative_bearing_ground_truth[0].state_vector[0] + np.random.normal(
    0, np.deg2rad(2)
)
prior_state = GaussianState(
    np.array([initial_bearing, 0]),
    np.diag([np.deg2rad(5) ** 2, np.deg2rad(0.5) ** 2]),
    timestamp=cfg["sim"]["start_time"],
)

track = Track([prior_state])

for timestamp, detections in all_detections:
    hypotheses = data_associator.associate({track}, detections, timestamp)[track]

    posterior_states = []
    posterior_state_weights = []
    for hypothesis in hypotheses:
        if not hypothesis:
            posterior_states.append(hypothesis.prediction)
        else:
            posterior_states.append(updater.update(hypothesis))
        posterior_state_weights.append(hypothesis.probability)

    means = StateVectors([state.state_vector for state in posterior_states])
    covars = np.stack([state.covar for state in posterior_states], axis=2)
    weights = np.asarray(posterior_state_weights)
    post_mean, post_covar = gm_reduce_single(means, covars, weights)

    track.append(
        GaussianStateUpdate(
            post_mean,
            post_covar,
            hypotheses,
            hypotheses[0].measurement.timestamp,
        )
    )


deg_std = 0.5

detection_measurement_model = LinearGaussian(
    ndim_state=1,
    mapping=[0],
    noise_covar=np.array([[np.deg2rad(deg_std) ** 2]]),
)

ss_measurement_model = LinearGaussian(
    ndim_state=2,
    mapping=[0],
    noise_covar=np.array([[np.deg2rad(deg_std) ** 2]]),
)

fov_rad = np.deg2rad(360)
expected_false_alarms_per_scan = 1
clutter_spatial_density = expected_false_alarms_per_scan / fov_rad
prob_detection = 0.95

stone_soup_detections: list[tuple] = []
for i, timestamp in enumerate(timesteps):
    detections_at_time = []

    for gt_bearing_path in relative_bearing_ground_truths:
        if np.random.rand() < prob_detection:
            measurement = detection_measurement_model.function(gt_bearing_path[i], noise=True)
            detections_at_time.append(
                Detection(
                    state_vector=measurement,
                    timestamp=timestamp,
                    measurement_model=ss_measurement_model,
                )
            )
            detections_at_time.append(
                Detection(
                    state_vector=-measurement,
                    timestamp=timestamp,
                    measurement_model=ss_measurement_model,
                )
            )

    num_clutter = np.random.poisson(clutter_spatial_density)
    for _ in range(num_clutter):
        clutter_bearing = uniform.rvs(loc=-np.pi, scale=2 * np.pi)
        detections_at_time.append(
            Detection(
                state_vector=np.array([[clutter_bearing]]),
                timestamp=timestamp,
                measurement_model=ss_measurement_model,
            )
        )

    stone_soup_detections.append((timestamp, detections_at_time))

# %%
# Build Figures
# -------------

# %%
fig = go.Figure()

plat_x = []
plat_y = []
for timestamp in timesteps:
    platform_state = platform.get_platform_state_at(timestamp)
    plat_x.append(platform_state.host.state.state_vector[0] / 1000)
    plat_y.append(platform_state.host.state.state_vector[2] / 1000)

tgt_x = []
tgt_y = []
for target_ground_truth in target_ground_truths:
    tgt_x = [state.state_vector[0] / 1000 for state in target_ground_truth]
    tgt_y = [state.state_vector[2] / 1000 for state in target_ground_truth]

fig.add_trace(
    go.Scatter(
        x=plat_x,
        y=plat_y,
        mode="lines",
        line=dict(color="black", width=3),
        name="Platform",
    )
)
fig.add_trace(
    go.Scatter(
        x=tgt_x,
        y=tgt_y,
        mode="lines",
        line=dict(color="#ff4141", width=3, dash="5px,2px"),
        name="Target",
    )
)

fig.update_layout(
    width=600,
    height=600,
    font=dict(family="Times New Roman", size=16, color="black"),
    showlegend=True,
    legend=dict(x=0.5, y=-0.25, xanchor="center", orientation="h", yanchor="bottom"),
    plot_bgcolor="white",
    yaxis=dict(scaleanchor="x", scaleratio=1),
)

if plat_x and tgt_x:
    all_x = plat_x + tgt_x
    all_y = plat_y + tgt_y
    xrange = [min(all_x) - 0.5, max(all_x) + 0.5]
    yrange = [min(all_y) - 0.5, max(all_y) + 0.5]

    fig.update_xaxes(
        range=xrange,
        showgrid=True,
        gridcolor="rgba(200, 200, 200, 0.5)",
        linecolor="black",
        title="X Position (km)",
        zeroline=True,
        zerolinecolor="rgba(200, 200, 200, 0.5)",
        zerolinewidth=0.5,
    )
    fig.update_yaxes(
        range=yrange,
        showgrid=True,
        gridcolor="rgba(200, 200, 200, 0.5)",
        linecolor="black",
        title="Y Position (km)",
        zeroline=True,
        zerolinecolor="rgba(200, 200, 200, 0.5)",
        zerolinewidth=0.5,
    )

world_fig = fig

# %%
det_x = []
det_y = []
for _, detections in all_detections:
    for det in detections:
        det_x.append(np.rad2deg(det.state_vector[0]))
        det_y.append(det.timestamp)

track_x = [np.rad2deg(state.state_vector[0]) for state in track]
track_y = [state.timestamp for state in track]

gt_x = [np.rad2deg(state.state_vector[0]) for state in relative_bearing_ground_truth]
gt_y = [state.timestamp for state in relative_bearing_ground_truth]

fig = make_subplots(
    rows=1,
    cols=3,
    shared_xaxes=True,
    shared_yaxes=True,
    horizontal_spacing=0.06,
    subplot_titles=["(a)", "(b)", "(c)"],
)

fig.add_trace(
    go.Heatmap(
        z=snr_map,
        y=timesteps,
        x=np.rad2deg(cfg["beamforming"]["steering_azimuths_rad"]),
        colorscale="Viridis",
        colorbar=dict(
            title=dict(text="SNR (dB)", side="right", font=dict(size=16)),
            thickness=24,
            len=1.0,
            tickfont=dict(size=14),
        ),
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Heatmap(
        z=snr_map,
        y=timesteps,
        x=np.rad2deg(cfg["beamforming"]["steering_azimuths_rad"]),
        colorscale="Viridis",
        showscale=False,
    ),
    row=1,
    col=2,
)

fig.add_trace(
    go.Scatter(
        x=det_x,
        y=det_y,
        mode="markers",
        name="Detection",
        marker=dict(size=6, line=dict(width=1), color="white", opacity=1.0),
        hovertemplate="Bearing: %{x:.1f}°<br>Time: %{y|%H:%M:%S}<extra></extra>",
    ),
    row=1,
    col=2,
)

fig.add_trace(
    go.Scatter(
        x=det_x,
        y=det_y,
        mode="markers",
        name="Detection",
        showlegend=False,
        marker=dict(size=6, line=dict(width=1), color="white", opacity=0.8),
        hovertemplate="Bearing: %{x:.1f}°<br>Time: %{y|%H:%M:%S}<extra></extra>",
    ),
    row=1,
    col=3,
)

fig.add_trace(
    go.Scatter(
        x=track_x,
        y=track_y,
        mode="lines",
        name="Track",
        line=dict(color="#1f77b4", width=4),
        hovertemplate="Bearing: %{x:.1f}°<br>Time: %{y|%H:%M:%S}<extra></extra>",
    ),
    row=1,
    col=3,
)

fig.add_trace(
    go.Scatter(
        x=gt_x,
        y=gt_y,
        mode="lines",
        name="Ground Truth",
        line=dict(color="#ff4141", width=3, dash="dash"),
        hovertemplate="Bearing: %{x:.1f}°<br>Time: %{y|%H:%M:%S}<extra></extra>",
    ),
    row=1,
    col=3,
)

fig.update_xaxes(
    range=[-180, 180],
    tickmode="linear",
    tick0=-180,
    dtick=60,
    tickangle=-45,
    tickfont=dict(size=14),
    showgrid=True,
    gridcolor="rgba(200, 200, 200, 0.5)",
    title="Bearing (°)",
    ticks="outside",
    tickcolor="rgba(160, 160, 160, 1.0)",
)
fig.update_xaxes(title="", col=1)
fig.update_xaxes(
    title="",
    col=3,
    showline=True,
    linewidth=1,
    linecolor="rgba(160, 160, 160, 1.0)",
)

fig.update_yaxes(
    range=[timesteps[-1], timesteps[0]],
    showgrid=True,
    gridcolor="rgba(200, 200, 200, 0.5)",
    tickformat="%H:%M",
    tickfont=dict(size=14),
    autorange=False,
    title="Time (HH:MM)",
    tickcolor="rgba(160, 160, 160, 1.0)",
)
fig.update_yaxes(ticks="outside", col=1)
fig.update_yaxes(showticklabels=False, title="", col=2)
fig.update_yaxes(
    showticklabels=False,
    title="",
    col=3,
    showline=True,
    linewidth=1,
    linecolor="rgba(160, 160, 160, 1.0)",
)

fig.update_layout(
    width=1200,
    height=600,
    margin=dict(b=100),
    font=dict(family="Times New Roman", size=16, color="black"),
    showlegend=True,
    legend=dict(
        x=0.5,
        y=-0.3,
        xanchor="center",
        yanchor="bottom",
        bgcolor="rgba(255,255,255,0.0)",
        borderwidth=0,
        orientation="h",
    ),
    plot_bgcolor="white",
    paper_bgcolor="white",
)

tracker_fig = fig

# %%
det_x = []
det_y = []
for _, detections in all_detections:
    for det in detections:
        det_x.append(np.rad2deg(det.state_vector[0]))
        det_y.append(det.timestamp)

ss_det_x = []
ss_det_y = []
for t, detection_set in stone_soup_detections:
    for detection in detection_set:
        ss_det_x.append(np.rad2deg(detection.state_vector[0]))
        ss_det_y.append(t)

fig = make_subplots(
    rows=1,
    cols=2,
    shared_xaxes=True,
    shared_yaxes=True,
    horizontal_spacing=0.06,
    subplot_titles=["(a)", "(b)"],
)

scatter_style = dict(
    mode="markers",
    marker=dict(size=6, line=dict(width=0.5), color="white", opacity=0.8),
    hovertemplate="Bearing: %{x:.1f}°<br>Time: %{y|%H:%M:%S}<extra></extra>",
)

fig.add_trace(
    go.Scatter(x=det_x, y=det_y, name="Detection", showlegend=False, **scatter_style),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(x=ss_det_x, y=ss_det_y, name="Detection", showlegend=True, **scatter_style),
    row=1,
    col=2,
)

fig.update_xaxes(
    range=[-180, 180],
    tickmode="linear",
    tick0=-180,
    dtick=60,
    tickangle=-45,
    tickfont=dict(size=14),
    ticks="outside",
    tickcolor="rgba(160, 160, 160, 1.0)",
    showgrid=True,
    gridcolor="rgba(200, 200, 200, 0.5)",
    showline=True,
    linecolor="rgba(160, 160, 160, 1.0)",
)
fig.add_annotation(
    text="Bearing (°)",
    xref="paper",
    yref="paper",
    x=0.5,
    y=-0.18,
    showarrow=False,
    font=dict(family="Times New Roman", size=18, color="black"),
)

fig.update_yaxes(
    range=[timesteps[-1], timesteps[0]],
    tickformat="%H:%M",
    tickfont=dict(size=16),
    ticks="outside",
    tickcolor="rgba(160, 160, 160, 1.0)",
    showgrid=True,
    gridcolor="rgba(200, 200, 200, 0.5)",
    showline=True,
    linewidth=1,
    linecolor="rgba(160, 160, 160, 1.0)",
    autorange=False,
    title="Time (HH:MM)",
)
fig.update_yaxes(title="", ticks="", showticklabels=False, col=2)

fig.update_layout(
    width=600,
    height=600,
    margin=dict(b=100),
    font=dict(family="Times New Roman", size=16, color="black"),
    showlegend=False,
    legend=dict(
        x=1.0,
        y=1.02,
        xanchor="right",
        yanchor="bottom",
        bgcolor="rgba(255,255,255,0.0)",
        borderwidth=0,
        orientation="h",
    ),
    plot_bgcolor="white",
    paper_bgcolor="white",
)

plugin_vs_ss_fig = fig

# %%
# Export and Summary
# ------------------

# %%
figures = {
    "st_world_picture.pdf": world_fig,
    "st_bf_tracker.pdf": tracker_fig,
    "st_plugin_vs_ss.pdf": plugin_vs_ss_fig,
}
output_dir = "figs"
scale = 1.0
out_dir = Path(output_dir)
out_dir.mkdir(parents=True, exist_ok=True)

for filename, fig in figures.items():
    width = fig.layout.width
    height = fig.layout.height
    export_kwargs = {"scale": scale}
    if width is not None:
        export_kwargs["width"] = int(width)
    if height is not None:
        export_kwargs["height"] = int(height)
    fig.write_image(str(out_dir / filename), **export_kwargs)

tab = " " * 4

print()
print("=" * 80)
print("Random Scenario Summary:")
print("=" * 80)
print(f"Seed: {cfg['seed']}")
print(f"Number of Targets: {cfg['num_targets']}")
print(
    f"Simulation Duration: {cfg['total_duration_s']} s across {cfg['sim']['num_steps']} timesteps"
)
print(f"Timestep Interval: {cfg['sim']['time_interval'].total_seconds()} s")
print(f"Array: {cfg['array']['num_sensors']} sensors, {cfg['array']['sensor_spacing']} m spacing")
print(f"Signal Sampling Rate: {cfg['signal']['sampling_rate_hz']} Hz")
print(
    f"Ambient Noise Level: {20 * np.log10(cfg['ambient_noise']['amplitude_upa']):.1f} dB re 1 µPa"
)
print(f"Steering Directions: {len(cfg['beamforming']['steering_azimuths_rad'])}")
array_length_m = cfg["array"]["num_sensors"] * cfg["array"]["sensor_spacing"]
print(f"Array length: {array_length_m} m")

print()
print("Platform Initial State:")
print(
    f"{tab}Position: ({cfg['ship']['start_vector'][0]:.0f}, "
    f"{cfg['ship']['start_vector'][2]:.0f}) m"
)
print(
    f"{tab}Velocity: ({cfg['ship']['start_vector'][1]:.1f}, "
    f"{cfg['ship']['start_vector'][3]:.1f}) m/s"
)

print()
for target_idx, target_cfg in enumerate(cfg["targets"]):
    print(f"Target {target_idx + 1}:")
    print(f"{tab}Frequencies: {target_cfg['frequencies_hz']} Hz")
    print(f"{tab}Amplitudes: {20 * np.log10(target_cfg['amplitudes_upa'])} dB re 1 µPa")
    print(
        f"{tab}Start Position: ({target_cfg['start_vector'][0]:.0f}, "
        f"{target_cfg['start_vector'][2]:.0f}) m"
    )
    print(
        f"{tab}Velocity: ({target_cfg['start_vector'][1]:.1f}, "
        f"{target_cfg['start_vector'][3]:.1f}) m/s"
    )
    print(f"{tab}Tonal Bandwidth: {target_cfg['tonal_bandwidth_hz']:.2f} Hz")
    print(f"{tab}Noise Level: {20 * np.log10(target_cfg['noise_amplitude_upa']):.1f} dB re 1 µPa")
