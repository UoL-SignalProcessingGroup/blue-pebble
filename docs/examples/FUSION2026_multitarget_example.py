"""
================================
FUSION 2026 Multi-Target Example
================================

This example assembles the multi-target demonstration used for the FUSION 2026
workflow. It runs the acoustic simulation, produces detections, compares them with a
Stone Soup baseline, and packages figures for a presentation or publication pipeline.

Multi-target passive-sonar examples demonstrate the full chain from acoustic simulation
through detection to tracker behaviour. Using the same scenario for both plugin
detections and Stone Soup baseline comparisons shows where the plugin adds value.
"""  # noqa: D205, D212, D400, D415
# sphinx_gallery_skip_execution = True

# %%
# Imports
# -------

# %%
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import get_window
from scipy.stats import uniform
from stonesoup.dataassociator.neighbour import GNNWith2DAssignment
from stonesoup.dataassociator.probability import JPDA
from stonesoup.deleter.error import CovarianceBasedDeleter
from stonesoup.functions import gm_reduce_single, mod_bearing
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.hypothesiser.probability import PDAHypothesiser
from stonesoup.initiator.simple import MultiMeasurementInitiator
from stonesoup.measures import Mahalanobis
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel,
    ConstantVelocity,
    KnownTurnRate,
)
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.types.angle import Bearing
from stonesoup.types.array import StateVector, StateVectors
from stonesoup.types.detection import Detection, MissedDetection
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.types.state import GaussianState, State
from stonesoup.types.update import GaussianStateUpdate
from stonesoup.updater.kalman import ExtendedKalmanUpdater

import bluepebble
from bluepebble.detector import CACFARDetector, PassiveSonarDetector
from bluepebble.models.environment import FlatBathymetry, Linear
from bluepebble.models.propagation import rtrsAcousticPropagationModel
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
seed = 12
bluepebble.set_seed(seed)
rng = bluepebble.get_rng()

sim_length = 900
sim_rate = 5.0
sim_params = {
    "start_time": datetime.now().replace(hour=0, minute=0, second=0, microsecond=0),
    "time_interval": timedelta(seconds=sim_rate),
    "num_steps": int(sim_length / sim_rate),
    "sim_length": sim_length,
}

ship_params = {
    "position_mapping": [0, 2, 4],
    "velocity_mapping": [1, 3, 5],
    "transition_model": CombinedLinearGaussianTransitionModel(
        [ConstantVelocity(0), ConstantVelocity(0), ConstantVelocity(0)]
    ),
}

array_params = {
    "num_sensors": 200,
    "tow_cable_length": 100.0,
    "sensor_spacing": 0.5,
    "array_depth": -50.0,
}

target_svs = [
    StateVector(
        [
            -1.45627511e04,
            8.79368717,
            1.20214456e04,
            -9.79815002,
            -5.0,
            0.0,
        ]
    ),
    StateVector(
        [
            -1.09436946e04,
            -8.05826663,
            -5.70307247e03,
            3.60050555,
            -5.0,
            0.0,
        ]
    ),
    StateVector(
        [
            -2.98105118e03,
            10.3979014,
            -9.67307472e03,
            9.71453496,
            -5.0,
            0.0,
        ]
    ),
]

targets = []
for sv in target_svs:
    targets.append(
        {
            "start_vector": sv,
            "position_mapping": [0, 2, 4],
            "velocity_mapping": [1, 3, 5],
            "transition_model": CombinedLinearGaussianTransitionModel(
                [ConstantVelocity(0), ConstantVelocity(0), ConstantVelocity(0)]
            ),
            "amplitudes_upa": 10 ** (rng.uniform(87, 102, 4) / 20),
            "frequencies_hz": rng.uniform(25.0, 200.0, 4),
            "phases_rad": rng.uniform(0, 2 * np.pi, 4),
            "tonal_bandwidth_hz": rng.uniform(0.5, 2.0),
            "noise_amplitude_upa": 10 ** (rng.uniform(65, 85) / 20),
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
    "amplitude_upa": 10 ** (rng.uniform(45, 55) / 20),
    "spectral_exponent": -1,
}

prop_params = {
    "ssp": Linear(surface_speed=1500.0, gradient=0.2),
    "bathymetry": FlatBathymetry(depth=-150.0),
    "step_m": 20.0,
    "azimuth_search_width": 2.0,
    "azimuth_resolution": 0.5,
    "elevation_range": (-25.0, 25.0),
    "elevation_resolution": 1.0,
}

bf_params = {
    "beamformer_type": "DAS",
    "shading": None,
    "domain": "frequency",
    "steering_azimuths_rad": np.linspace(-np.pi, np.pi, 361),
}

det_params = {
    "cfar_detector": {
        "num_guard_cells": 6,
        "num_training_cells": 10,
        # Reproduces the pre-refactor threshold_factor exactly, via CA-CFAR's single-look
        # Pfa = (1 + alpha/N)^-N with N = 2 * num_training_cells.
        "target_pfa": 0.3594,
        "circular": True,
        "peak_distance": 3,
    },
}

ss_det_params = {
    "prob_detection": 0.95,
    "bearing_std_deg": 0.5,
    "fov_deg": 360.0,
    "expected_clutter_per_scan": 1,
    "include_ambiguity": True,
}

cfg = {
    "seed": seed,
    "sim": sim_params,
    "ship": ship_params,
    "array": array_params,
    "targets": targets,
    "signal": signal_params,
    "ambient_noise": ambient_noise_params,
    "propagation": prop_params,
    "beamforming": bf_params,
    "detection": det_params,
    "stone_soup_detection": ss_det_params,
    "total_duration_s": total_duration_s,
}

print(f"Total simulation duration: {cfg['total_duration_s']} seconds")
print(
    f"Number of timesteps: {cfg['sim']['num_steps']}, "
    f"Timestep interval: {cfg['sim']['time_interval'].total_seconds()} seconds"
)
print(f"Num targets: {len(cfg['targets'])}")

# %%
# Scenario Build and Detection
# ----------------------------

# %%
sim = cfg["sim"]
platform_turn_rate_radps = np.deg2rad(1.0)

leg1_duration_s = timedelta(seconds=405)
turn1_angle_rad = np.deg2rad(-45)
turn1_duration_s = timedelta(
    seconds=round((abs(turn1_angle_rad) / platform_turn_rate_radps) / 5.0) * 5.0
)
leg2_duration_s = timedelta(seconds=sim["sim_length"]) - leg1_duration_s - turn1_duration_s

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

plat_init_sv = StateVector([-7500, 1.92039757, -2000, 0.269915147, -5.0, 0.0])
platform_initial_state = State(plat_init_sv, timestamp=cfg["sim"]["start_time"])

platform = TowedArrayPlatform(
    states=platform_initial_state,
    position_mapping=cfg["ship"]["position_mapping"],
    velocity_mapping=cfg["ship"]["velocity_mapping"],
    transition_models=transition_models,
    transition_times=transition_times,
    num_sensors=cfg["array"]["num_sensors"],
    cable_length_m=cfg["array"]["tow_cable_length"],
    sensor_spacing_m=cfg["array"]["sensor_spacing"],
    array_depth_m=cfg["array"]["array_depth"],
)

for i in range(1, cfg["sim"]["num_steps"]):
    platform.move(cfg["sim"]["start_time"] + i * cfg["sim"]["time_interval"])

target_ground_truths: list[GroundTruthPath] = []
relative_bearing_ground_truths: list[GroundTruthPath] = []

for target_cfg in cfg["targets"]:
    target_states = [
        GroundTruthState(
            target_cfg["start_vector"],
            timestamp=cfg["sim"]["start_time"],
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
    for i in range(1, cfg["sim"]["num_steps"]):
        new_time = cfg["sim"]["start_time"] + i * cfg["sim"]["time_interval"]
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
                timestamp=cfg["sim"]["start_time"] + i * cfg["sim"]["time_interval"],
            )
        )

    relative_bearing_ground_truths.append(GroundTruthPath(bearing_states))

prop = cfg["propagation"]
signal = cfg["signal"]
ambient = cfg["ambient_noise"]
bf = cfg["beamforming"]

prop_model = rtrsAcousticPropagationModel(
    ssp=prop["ssp"],
    bathymetry=prop["bathymetry"],
    use_all_frequencies=False,
    step_m=prop["step_m"],
    azimuth_search_width=prop["azimuth_search_width"],
    azimuth_resolution=prop["azimuth_resolution"],
    elevation_range=prop["elevation_range"],
    elevation_resolution=prop["elevation_resolution"],
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
    beamformer = DelayAndSumBeamformer(
        sampling_rate_hz=signal["sampling_rate_hz"],
        shading=shading,
        domain=bf["domain"],
    )
elif bf["beamformer_type"] == "MVDR":
    beamformer = MinimumVarianceDistortionlessResponseBeamformer(
        sampling_rate_hz=signal["sampling_rate_hz"],
        fmin=bf.get("fmin"),
        fmax=bf.get("fmax"),
    )
else:
    raise ValueError(f"Unknown beamformer type: {bf['beamformer_type']}")

steering_calculator = SteeringCalculator(
    ssp=prop["ssp"],
    steering_azimuths_rad=bf["steering_azimuths_rad"],
)


def _make_signal_model(target_cfg):
    return SyntheticAnthropogenicSignal(
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


signal_models = [_make_signal_model(tc) for tc in cfg["targets"]]

simulator = ContinuousSTFTPassiveSonarArraySimulator(
    platform=platform,
    propagation_model=prop_model,
    signal_models=signal_models,
    noise_model=ambient_noise_model,
    beamformer=beamformer,
    steering_calculator=steering_calculator,
    ground_truth_paths=target_ground_truths,
    fade_in_ms=signal["fade_in_ms"],
)

print("Running detector...")

det = cfg["detection"]
cfar_detector = CACFARDetector(
    num_guard_cells=det["cfar_detector"]["num_guard_cells"],
    num_training_cells=det["cfar_detector"]["num_training_cells"],
    target_pfa=det["cfar_detector"]["target_pfa"],
    circular=det["cfar_detector"]["circular"],
    peak_distance=det["cfar_detector"]["peak_distance"],
)

# snr_history feeds the BTR figures below. It reports SNR against a scan-wide percentile,
# which is the reference these figures were produced with and keeps them comparable with the
# published versions. The detector thresholds against its own local training-cell estimate
# regardless; snr_reference only chooses what is reported.
detector = PassiveSonarDetector(
    detector=cfar_detector,
    sensor_data_gen=simulator.sensor_data_gen(),
    steering_azimuths_rad=cfg["beamforming"]["steering_azimuths_rad"],
)

all_detections = list(detector.detections_gen(progress_bar=False))
snr_map = detector.snr_history

timesteps = [
    cfg["sim"]["start_time"] + i * cfg["sim"]["time_interval"]
    for i in range(cfg["sim"]["num_steps"])
]

# %%
# Tracking and Stone Soup Baseline
# --------------------------------

# %%
transition_model = ConstantVelocity(0.000001)
predictor = KalmanPredictor(transition_model)

measurement_model = LinearGaussian(
    ndim_state=2,
    mapping=[0],
    noise_covar=np.array([[np.deg2rad(1) ** 2]]),
)
updater = ExtendedKalmanUpdater(measurement_model=measurement_model)

fov_rad = np.deg2rad(360)
expected_false_alarms_per_scan = 3
clutter_spatial_density = expected_false_alarms_per_scan / fov_rad

jpda_hypothesiser = PDAHypothesiser(
    predictor=predictor,
    updater=updater,
    clutter_spatial_density=clutter_spatial_density,
    prob_detect=0.85,
)
jpda_associator = JPDA(hypothesiser=jpda_hypothesiser)

init_hypothesiser = DistanceHypothesiser(
    predictor=predictor,
    updater=updater,
    measure=Mahalanobis(),
    missed_distance=6,
)
init_associator = GNNWith2DAssignment(init_hypothesiser)

deleter = CovarianceBasedDeleter(covar_trace_thresh=0.2)

initial_bearing = float(relative_bearing_ground_truths[0][0].state_vector[0]) + rng.normal(
    0, np.deg2rad(2)
)
initial_bearing = mod_bearing(initial_bearing)

prior_state = GaussianState(
    np.array([[initial_bearing], [0.0]]),
    np.diag([np.deg2rad(5) ** 2, np.deg2rad(0.5) ** 2]),
    timestamp=cfg["sim"]["start_time"],
)
Bearing(prior_state.state_vector[0, 0])

initiator = MultiMeasurementInitiator(
    prior_state=prior_state,
    measurement_model=measurement_model,
    deleter=deleter,
    data_associator=init_associator,
    updater=updater,
    min_points=30,
)

tracks, all_tracks = set(), set()

for timestamp, detections in all_detections:
    for det in detections:
        det.state_vector[0, 0] = mod_bearing(float(det.state_vector[0, 0]))

    associations = jpda_associator.associate(set(tracks), detections, timestamp)
    associated_detections = set()

    for track in list(tracks):
        track_hypotheses = associations[track]
        posterior_states = []
        posterior_weights = []

        for hyp in track_hypotheses:
            if hyp.measurement is None or isinstance(hyp.measurement, MissedDetection):
                state = hyp.prediction
                Bearing(state.state_vector[0, 0])
            else:
                state = updater.update(hyp)
                Bearing(state.state_vector[0, 0])
                associated_detections.add(hyp.measurement)

            posterior_states.append(state)
            posterior_weights.append(float(hyp.probability))

        means = StateVectors([s.state_vector for s in posterior_states])
        covars = np.stack([s.covar for s in posterior_states], axis=2)
        weights = np.asarray(posterior_weights, dtype=float)
        if weights.sum() > 0:
            weights /= weights.sum()

        post_mean, post_covar = gm_reduce_single(means, covars, weights)
        post_mean = post_mean.copy()
        post_mean[0, 0] = mod_bearing(float(post_mean[0, 0]))

        track.append(GaussianStateUpdate(post_mean, post_covar, track_hypotheses, timestamp))

    tracks -= deleter.delete_tracks(tracks)
    tracks |= initiator.initiate(detections - associated_detections, timestamp)
    all_tracks |= tracks

ss_cfg = cfg["stone_soup_detection"]
deg_std = ss_cfg["bearing_std_deg"]

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

prob_detection = ss_cfg["prob_detection"]
fov_rad = np.deg2rad(ss_cfg["fov_deg"])
expected_false_alarms_per_scan = ss_cfg["expected_clutter_per_scan"]
clutter_spatial_density = expected_false_alarms_per_scan / fov_rad

stone_soup_detections = []
for i, timestamp in enumerate(timesteps):
    detections_at_time = []

    for target_gt in target_ground_truths:
        target_state = target_gt[i]
        platform_state = platform.get_platform_state_at(timestamp)
        target_pos = np.array([target_state.state_vector[0], target_state.state_vector[2]])
        array_center = np.mean(platform_state.array.state_vector, axis=1)
        relative_pos = target_pos - array_center[:2]
        absolute_bearing = np.arctan2(relative_pos[1], relative_pos[0])
        true_bearing = np.arctan2(np.sin(absolute_bearing), np.cos(absolute_bearing))

        if np.random.rand() < prob_detection:
            true_bearing_state = GroundTruthState(
                state_vector=np.array([true_bearing]),
                timestamp=timestamp,
            )
            measurement = detection_measurement_model.function(
                true_bearing_state,
                noise=True,
            )
            detections_at_time.append(
                Detection(
                    state_vector=measurement,
                    timestamp=timestamp,
                    measurement_model=ss_measurement_model,
                )
            )

            if ss_cfg["include_ambiguity"]:
                array_positions = platform_state.array.state_vector
                array_head = array_positions[:2, -1]
                array_tail = array_positions[:2, 0]
                array_axis = array_head - array_tail
                array_heading = np.arctan2(array_axis[1], array_axis[0])

                ambiguous_bearing = 2 * array_heading - true_bearing
                ambiguous_bearing = np.arctan2(
                    np.sin(ambiguous_bearing), np.cos(ambiguous_bearing)
                )
                ambiguous_bearing_state = GroundTruthState(
                    state_vector=np.array([ambiguous_bearing]),
                    timestamp=timestamp,
                )
                ambiguous_measurement = detection_measurement_model.function(
                    ambiguous_bearing_state,
                    noise=True,
                )
                detections_at_time.append(
                    Detection(
                        state_vector=ambiguous_measurement,
                        timestamp=timestamp,
                        measurement_model=ss_measurement_model,
                    )
                )

    num_clutter = rng.poisson(clutter_spatial_density)
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
num_targets = len(target_ground_truths)

plat_x = [entry.host.state.state_vector[0] / 1000 for entry in platform.platform_history]
plat_y = [entry.host.state.state_vector[2] / 1000 for entry in platform.platform_history]

tgt_x = [[] for _ in range(num_targets)]
tgt_y = [[] for _ in range(num_targets)]
for idx, target_ground_truth in enumerate(target_ground_truths):
    tgt_x[idx] = [state.state_vector[0] / 1000 for state in target_ground_truth]
    tgt_y[idx] = [state.state_vector[2] / 1000 for state in target_ground_truth]

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=plat_x,
        y=plat_y,
        mode="lines",
        line=dict(color="black", width=3),
        name="Platform",
    )
)

colors = px.colors.qualitative.Dark24
names = [f"Target {i + 1}" for i in range(num_targets)]
for i in range(num_targets):
    fig.add_trace(
        go.Scatter(
            x=tgt_x[i],
            y=tgt_y[i],
            mode="lines",
            line=dict(color=colors[i], width=3, dash="5px,2px"),
            name=names[i],
        )
    )

fig.add_trace(
    go.Scatter(
        x=plat_x,
        y=plat_y,
        mode="lines+markers",
        marker=dict(size=3, color="black"),
        line=dict(color="black", width=1.5),
        showlegend=False,
        xaxis="x2",
        yaxis="y2",
    )
)

all_x = plat_x + [x for sublist in tgt_x for x in sublist]
all_y = plat_y + [y for sublist in tgt_y for y in sublist]

min_x, max_x = min(all_x) - 0.5, max(all_x) + 0.5
min_y, max_y = min(all_y) - 0.5, max(all_y) + 0.5

mid_x = (max_x + min_x) / 2
mid_y = (max_y + min_y) / 2
max_span = max(max_x - min_x, max_y - min_y)

main_x_range = [mid_x - max_span / 2, mid_x + max_span / 2]
main_y_range = [mid_y - max_span / 2, mid_y + max_span / 2]

p_min_x, p_max_x = min(plat_x), max(plat_x)
p_min_y, p_max_y = min(plat_y), max(plat_y)
p_mid_x = (p_min_x + p_max_x) / 2
p_mid_y = (p_min_y + p_max_y) / 2

zoom_pad = 0.2
zoom_span = max(p_max_x - p_min_x, p_max_y - p_min_y) + (zoom_pad * 2)

box_x = [p_mid_x - zoom_span / 2, p_mid_x + zoom_span / 2]
box_y = [p_mid_y - zoom_span / 2, p_mid_y + zoom_span / 2]

inset_dom_x = [0.6, 1.0]
inset_dom_y = [0.6, 1.0]

fig.update_layout(
    width=600,
    height=600,
    font=dict(family="Times New Roman", size=16, color="black"),
    showlegend=True,
    legend=dict(x=0.5, y=-0.15, xanchor="center", orientation="h"),
    plot_bgcolor="white",
    xaxis=dict(
        title="X Position (km)",
        range=main_x_range,
        showgrid=True,
        gridcolor="rgba(200,200,200,0.5)",
        linecolor="black",
        zeroline=True,
        zerolinecolor="rgba(200, 200, 200, 0.5)",
        zerolinewidth=0.5,
    ),
    yaxis=dict(
        title="Y Position (km)",
        range=main_y_range,
        showgrid=True,
        gridcolor="rgba(200,200,200,0.5)",
        linecolor="black",
        zeroline=True,
        zerolinecolor="rgba(200, 200, 200, 0.5)",
        zerolinewidth=0.5,
    ),
    xaxis2=dict(
        domain=inset_dom_x,
        anchor="y2",
        range=box_x,
        showgrid=True,
        gridcolor="rgba(200,200,200,0.25)",
        linecolor="rgba(0,0,0,0.5)",
        tickfont=dict(size=10),
    ),
    yaxis2=dict(
        domain=inset_dom_y,
        anchor="x2",
        range=box_y,
        showgrid=True,
        gridcolor="rgba(200,200,200,0.25)",
        linecolor="rgba(0,0,0,0.5)",
        tickfont=dict(size=10),
    ),
)

fig.add_shape(
    type="rect",
    xref="x domain",
    yref="y domain",
    x0=inset_dom_x[0],
    y0=inset_dom_y[0],
    x1=inset_dom_x[1],
    y1=inset_dom_y[1],
    fillcolor="white",
    line=dict(width=0),
    layer="below",
)

fig.add_shape(
    type="rect",
    xref="x",
    yref="y",
    x0=box_x[0],
    y0=box_y[0],
    x1=box_x[1],
    y1=box_y[1],
    line=dict(color="rgba(0,0,0,0.5)", width=1),
    fillcolor="rgba(0,0,0,0)",
)

inset_left_km = main_x_range[0] + (inset_dom_x[0] * (main_x_range[1] - main_x_range[0]))
inset_right_km = main_x_range[0] + (inset_dom_x[1] * (main_x_range[1] - main_x_range[0]))
inset_bottom_km = main_y_range[0] + (inset_dom_y[0] * (main_y_range[1] - main_y_range[0]))
inset_top_km = main_y_range[0] + (inset_dom_y[1] * (main_y_range[1] - main_y_range[0]))

fig.add_shape(
    type="line",
    xref="x",
    yref="y",
    x0=box_x[0],
    y0=box_y[1],
    x1=inset_left_km,
    y1=inset_top_km,
    line=dict(color="rgba(0,0,0,0.5)", width=1, dash="solid"),
    layer="above",
)

fig.add_shape(
    type="line",
    xref="x",
    yref="y",
    x0=box_x[1],
    y0=box_y[0],
    x1=inset_right_km,
    y1=inset_bottom_km,
    line=dict(color="rgba(0,0,0,0.5)", width=1, dash="solid"),
    layer="above",
)

for tgt_idx in range(num_targets):
    for i in range(40, len(tgt_x[tgt_idx]), 30):
        if i < 1 or i >= len(tgt_x[tgt_idx]):
            continue
        dx = tgt_x[tgt_idx][i] - tgt_x[tgt_idx][i - 1]
        dy = tgt_y[tgt_idx][i] - tgt_y[tgt_idx][i - 1]
        magnitude = (dx**2 + dy**2) ** 0.5
        if magnitude <= 0:
            continue
        ux, uy = dx / magnitude, dy / magnitude
        vx, vy = -uy, ux
        tip_x, tip_y = tgt_x[tgt_idx][i], tgt_y[tgt_idx][i]
        size = 0.7
        base_x = tip_x - (ux * size)
        base_y = tip_y - (uy * size)
        hw = size * 0.35
        path = (
            f"M {tip_x},{tip_y} "
            f"L {base_x + vx * hw},{base_y + vy * hw} "
            f"L {base_x - vx * hw},{base_y - vy * hw} Z"
        )
        fig.add_shape(
            type="path",
            path=path,
            fillcolor=colors[tgt_idx],
            line=dict(color=colors[tgt_idx], width=1),
            xref="x",
            yref="y",
            layer="above",
        )

for i in range(40, len(plat_x), 40):
    if i < 1 or i >= len(plat_x):
        continue
    dx = plat_x[i] - plat_x[i - 1]
    dy = plat_y[i] - plat_y[i - 1]
    magnitude = (dx**2 + dy**2) ** 0.5
    if magnitude <= 0:
        continue
    ux, uy = dx / magnitude, dy / magnitude
    vx, vy = -uy, ux
    tip_x, tip_y = plat_x[i], plat_y[i]
    size = 0.15
    base_x = tip_x - (ux * size)
    base_y = tip_y - (uy * size)
    hw = size * 0.35
    path = (
        f"M {tip_x},{tip_y} "
        f"L {base_x + vx * hw},{base_y + vy * hw} "
        f"L {base_x - vx * hw},{base_y - vy * hw} Z"
    )
    fig.add_shape(
        type="path",
        path=path,
        fillcolor="black",
        line=dict(color="black", width=1),
        xref="x2",
        yref="y2",
        layer="above",
    )

world_fig = fig

# %%
det_x = []
det_y = []
for _, detections in all_detections:
    for det in detections:
        det_x.append(np.rad2deg(det.state_vector[0]))
        det_y.append(det.timestamp)

track_x = [[np.rad2deg(state.state_vector[0]) for state in track] for track in all_tracks]
track_y = [[state.timestamp for state in track] for track in all_tracks]

gt_x = [
    [np.rad2deg(state.state_vector[0]) for state in relative_bearing_ground_truth]
    for relative_bearing_ground_truth in relative_bearing_ground_truths
]
gt_y = [
    [state.timestamp for state in relative_bearing_ground_truth]
    for relative_bearing_ground_truth in relative_bearing_ground_truths
]

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
        marker=dict(size=4, line=dict(width=1), color="white", opacity=0.5),
        hovertemplate="Bearing: %{x:.1f}°<br>Time: %{y|%H:%M:%S}<extra></extra>",
    ),
    row=1,
    col=3,
)

track_colors = [
    "#1f77b4",
    "#2ca02c",
    "#9467bd",
    "#17becf",
    "#000080",
    "#006400",
    "#4b0082",
]

truth_colors = [
    "#d62728",
    "#ff7f0e",
    "#e377c2",
    "#bcbd22",
    "#8c564b",
    "#ff0000",
    "#ff1493",
]

for i in range(len(track_x)):
    if track_x[i] and track_y[i]:
        wx = [track_x[i][0]]
        wy = [track_y[i][0]]
        for j in range(1, len(track_x[i])):
            if abs(track_x[i][j] - track_x[i][j - 1]) > 180:
                wx.append(None)
                wy.append(None)
            wx.append(track_x[i][j])
            wy.append(track_y[i][j])
    else:
        wx, wy = [], []
    fig.add_trace(
        go.Scatter(
            x=wx,
            y=wy,
            mode="lines",
            name="Track",
            line=dict(color=track_colors[i % len(track_colors)], width=4),
            showlegend=(i == 0),
        ),
        row=1,
        col=3,
    )

for i in range(len(relative_bearing_ground_truths)):
    if gt_x[i] and gt_y[i]:
        wx = [gt_x[i][0]]
        wy = [gt_y[i][0]]
        for j in range(1, len(gt_x[i])):
            if abs(gt_x[i][j] - gt_x[i][j - 1]) > 180:
                wx.append(None)
                wy.append(None)
            wx.append(gt_x[i][j])
            wy.append(gt_y[i][j])
    else:
        wx, wy = [], []
    fig.add_trace(
        go.Scatter(
            x=wx,
            y=wy,
            mode="lines",
            name="Ground Truth",
            line=dict(color=truth_colors[i % len(truth_colors)], width=3, dash="dash"),
            showlegend=(i == 0),
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
    go.Scatter(
        x=ss_det_x,
        y=ss_det_y,
        name="Detection",
        showlegend=True,
        **scatter_style,
    ),
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
# Export
# ------

# %%
figures = {
    "mt_world_picture.pdf": world_fig,
    "mt_bf_tracker.pdf": tracker_fig,
    "mt_plugin_vs_ss.pdf": plugin_vs_ss_fig,
}
output_dir = "figs"
scale = 1.0
out_dir = Path(output_dir)
out_dir.mkdir(parents=True, exist_ok=True)

# Static export needs Kaleido, which drives a headless Chrome. That is here to produce the
# paper figures, not to read the example, so a machine without the browser should skip it
# rather than take the whole docs build down -- Read the Docs has no Chrome at all.
try:
    for filename, fig in figures.items():
        width = fig.layout.width
        height = fig.layout.height
        export_kwargs = {"scale": scale}
        if width is not None:
            export_kwargs["width"] = int(width)
        if height is not None:
            export_kwargs["height"] = int(height)
        fig.write_image(str(out_dir / filename), **export_kwargs)
    print(f"Wrote {len(figures)} figures to {out_dir}/")
except Exception as exc:  # noqa: BLE001 - any export failure should be non-fatal here
    print(
        f"Skipped static figure export ({type(exc).__name__}). The interactive figures above "
        "are unaffected. To enable it: pip install 'kaleido>=1' and run plotly_get_chrome."
    )
