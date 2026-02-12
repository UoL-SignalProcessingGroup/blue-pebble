"""Pipeline helpers for the FUSION 2026 multi-target figures notebook."""

from __future__ import annotations

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

from nereus.detector import CFARDetector, PassiveSonarDetector, PeakDetector
from nereus.models.environment import FlatBathymetry, Linear
from nereus.models.propagation import rtrsAcousticPropagationModel
from nereus.platform import TowedArrayPlatform
from nereus.signal.ambient import ColouredNoise
from nereus.signal.anthropogenic import BroadbandShipSignal
from nereus.sigproc import (
    DelayAndSumBeamformer,
    MinimumVarianceDistortionlessResponseBeamformer,
    SteeringCalculator,
)
from nereus.simulator import BroadbandPassiveSonarArraySimulator


def build_config(seed: int = 12) -> dict:
    """Create a reproducible multi-target notebook configuration."""
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
                "amplitudes_upa": 10 ** (np.random.uniform(87, 102, 4) / 20),
                "frequencies_hz": np.random.uniform(25.0, 200.0, 4),
                "phases_rad": np.random.uniform(0, 2 * np.pi, 4),
                "tonal_bandwidth_hz": np.random.uniform(0.5, 2.0),
                "noise_amplitude_upa": 10 ** (np.random.uniform(65, 85) / 20),
                "noise_spectral_exponent": -1.0,
            }
        )

    total_duration_s = (
        sim_params["num_steps"] * sim_params["time_interval"].total_seconds()
    )

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
            "threshold_factor": 1.05,
            "mode": "wrap",
        },
        "peak_detector": {
            "distance": 3,
        },
    }

    ss_det_params = {
        "prob_detection": 0.95,
        "bearing_std_deg": 0.5,
        "fov_deg": 360.0,
        "expected_clutter_per_scan": 1,
        "include_ambiguity": True,
    }

    return {
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


def build_timesteps(cfg: dict) -> list[datetime]:
    """Build timestamp sequence for the scenario."""
    return [
        cfg["sim"]["start_time"] + i * cfg["sim"]["time_interval"]
        for i in range(cfg["sim"]["num_steps"])
    ]


def _platform_maneuver_models(cfg: dict) -> tuple[list, list]:
    """Create transition model schedule for the platform maneuver."""
    sim = cfg["sim"]
    platform_turn_rate_radps = np.deg2rad(1.0)

    leg1_duration_s = timedelta(seconds=405)
    turn1_angle_rad = np.deg2rad(-45)
    turn1_duration_s = timedelta(
        seconds=round((abs(turn1_angle_rad) / platform_turn_rate_radps) / 5.0) * 5.0
    )
    leg2_duration_s = (
        timedelta(seconds=sim["sim_length"]) - leg1_duration_s - turn1_duration_s
    )

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
    return transition_models, transition_times


def build_platform(cfg: dict) -> TowedArrayPlatform:
    """Build and propagate the towed-array platform for the scenario."""
    transition_models, transition_times = _platform_maneuver_models(cfg)

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

    return platform


def build_target_truths(
    cfg: dict, platform: TowedArrayPlatform
) -> tuple[list[GroundTruthPath], list[GroundTruthPath]]:
    """Generate target trajectories and relative-bearing truth paths."""
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
            target_pos = np.array(
                [target_state.state_vector[0], target_state.state_vector[2]]
            )
            relative_pos = target_pos - ref_sensor_position[:2]
            gt_relative_bearings.append(np.arctan2(relative_pos[1], relative_pos[0]))

        bearing_states = []
        for i, bearing in enumerate(np.asarray(gt_relative_bearings)):
            bearing_states.append(
                GroundTruthState(
                    state_vector=np.array([bearing]),
                    timestamp=cfg["sim"]["start_time"]
                    + i * cfg["sim"]["time_interval"],
                )
            )

        relative_bearing_ground_truths.append(GroundTruthPath(bearing_states))

    return target_ground_truths, relative_bearing_ground_truths


def build_simulator(
    cfg: dict,
    platform: TowedArrayPlatform,
    target_ground_truths: list[GroundTruthPath],
) -> BroadbandPassiveSonarArraySimulator:
    """Build simulator with propagation, beamforming, and noise models."""
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

    ambient_noise_model = ColouredNoise(
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

    target_cfg = cfg["targets"][0]
    signal_model = BroadbandShipSignal(
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

    return BroadbandPassiveSonarArraySimulator(
        platform=platform,
        propagation_model=prop_model,
        signal_model=signal_model,
        noise_model=ambient_noise_model,
        beamformer=beamformer,
        steering_calculator=steering_calculator,
        ground_truth_paths=target_ground_truths,
        fade_in_ms=signal["fade_in_ms"],
    )


def run_detection(
    cfg: dict, simulator: BroadbandPassiveSonarArraySimulator
) -> tuple[list, np.ndarray]:
    """Run detector chain on simulator output and return detections/SNR."""
    det = cfg["detection"]
    cfar_detector = CFARDetector(
        num_guard_cells=det["cfar_detector"]["num_guard_cells"],
        num_training_cells=det["cfar_detector"]["num_training_cells"],
        threshold_factor=det["cfar_detector"]["threshold_factor"],
        mode=det["cfar_detector"]["mode"],
    )
    peak_detector = PeakDetector(distance=det["peak_detector"]["distance"])

    detector = PassiveSonarDetector(
        detection_chain=[cfar_detector, peak_detector],
        sensor_data_gen=simulator.sensor_data_gen(),
        steering_azimuths_rad=cfg["beamforming"]["steering_azimuths_rad"],
    )

    all_detections = list(detector.detections_gen(progress_bar=True))
    return all_detections, detector.snr_history


def run_tracking_jpda(
    cfg: dict,
    all_detections: list,
    relative_bearing_ground_truths: list[GroundTruthPath],
) -> tuple[set, set]:
    """Run JPDA/GNN track initiation and maintenance for multi-target detections."""
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

    initial_bearing = float(
        relative_bearing_ground_truths[0][0].state_vector[0]
    ) + np.random.normal(0, np.deg2rad(2))
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
                if hyp.measurement is None or isinstance(
                    hyp.measurement, MissedDetection
                ):
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

            track.append(
                GaussianStateUpdate(post_mean, post_covar, track_hypotheses, timestamp)
            )

        tracks -= deleter.delete_tracks(tracks)
        tracks |= initiator.initiate(detections - associated_detections, timestamp)
        all_tracks |= tracks

    return tracks, all_tracks


def _handle_wraparound(
    x_vals: list[float], y_vals: list[datetime], threshold: float = 180
) -> tuple[list[float], list[datetime]]:
    """Insert None separators where bearing wraps to avoid long plot jumps."""
    if not x_vals or not y_vals:
        return [], []

    clean_x = [x_vals[0]]
    clean_y = [y_vals[0]]

    for i in range(1, len(x_vals)):
        if abs(x_vals[i] - x_vals[i - 1]) > threshold:
            clean_x.append(None)
            clean_y.append(None)
        clean_x.append(x_vals[i])
        clean_y.append(y_vals[i])

    return clean_x, clean_y


def build_world_figure(
    platform: TowedArrayPlatform,
    target_ground_truths: list[GroundTruthPath],
) -> go.Figure:
    """Build world-view figure with inset and directional arrows."""
    num_targets = len(target_ground_truths)

    plat_x = [
        entry.host.state.state_vector[0] / 1000 for entry in platform.platform_history
    ]
    plat_y = [
        entry.host.state.state_vector[2] / 1000 for entry in platform.platform_history
    ]

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

    def domain_to_data(dom_val: float, r_min: float, r_max: float) -> float:
        return r_min + (dom_val * (r_max - r_min))

    inset_left_km = domain_to_data(inset_dom_x[0], main_x_range[0], main_x_range[1])
    inset_right_km = domain_to_data(inset_dom_x[1], main_x_range[0], main_x_range[1])
    inset_bottom_km = domain_to_data(inset_dom_y[0], main_y_range[0], main_y_range[1])
    inset_top_km = domain_to_data(inset_dom_y[1], main_y_range[0], main_y_range[1])

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

    def add_arrowhead(
        figure: go.Figure,
        x: list[float],
        y: list[float],
        i: int,
        color: str,
        size: float,
        xref: str = "x",
        yref: str = "y",
    ) -> None:
        if i < 1 or i >= len(x):
            return
        dx = x[i] - x[i - 1]
        dy = y[i] - y[i - 1]
        magnitude = (dx**2 + dy**2) ** 0.5
        if magnitude <= 0:
            return

        ux, uy = dx / magnitude, dy / magnitude
        vx, vy = -uy, ux
        tip_x, tip_y = x[i], y[i]
        base_x = tip_x - (ux * size)
        base_y = tip_y - (uy * size)
        hw = size * 0.35
        path = (
            f"M {tip_x},{tip_y} "
            f"L {base_x + vx * hw},{base_y + vy * hw} "
            f"L {base_x - vx * hw},{base_y - vy * hw} Z"
        )
        figure.add_shape(
            type="path",
            path=path,
            fillcolor=color,
            line=dict(color=color, width=1),
            xref=xref,
            yref=yref,
            layer="above",
        )

    for tgt_idx in range(num_targets):
        for i in range(40, len(tgt_x[tgt_idx]), 30):
            add_arrowhead(fig, tgt_x[tgt_idx], tgt_y[tgt_idx], i, colors[tgt_idx], 0.7)

    for i in range(40, len(plat_x), 40):
        add_arrowhead(fig, plat_x, plat_y, i, "black", 0.15, xref="x2", yref="y2")

    return fig


def build_tracker_figure(
    timesteps: list[datetime],
    steering_azimuths_rad: np.ndarray,
    snr_map: np.ndarray,
    all_detections: list,
    all_tracks: set,
    relative_bearing_ground_truths: list[GroundTruthPath],
) -> go.Figure:
    """Build 3-panel beamformer/detection/track figure for multi-target case."""
    det_x = []
    det_y = []
    for _, detections in all_detections:
        for det in detections:
            det_x.append(np.rad2deg(det.state_vector[0]))
            det_y.append(det.timestamp)

    track_x = [
        [np.rad2deg(state.state_vector[0]) for state in track] for track in all_tracks
    ]
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
            x=np.rad2deg(steering_azimuths_rad),
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
            x=np.rad2deg(steering_azimuths_rad),
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
        wx, wy = _handle_wraparound(track_x[i], track_y[i])
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
        wx, wy = _handle_wraparound(gt_x[i], gt_y[i])
        fig.add_trace(
            go.Scatter(
                x=wx,
                y=wy,
                mode="lines",
                name="Ground Truth",
                line=dict(
                    color=truth_colors[i % len(truth_colors)], width=3, dash="dash"
                ),
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

    return fig


def _compute_bearing_world_frame(platform_state, target_position: np.ndarray) -> float:
    """Compute bearing from array center to target in world frame."""
    array_center = np.mean(platform_state.array.state_vector, axis=1)
    relative_pos = target_position - array_center[:2]
    absolute_bearing = np.arctan2(relative_pos[1], relative_pos[0])
    return np.arctan2(np.sin(absolute_bearing), np.cos(absolute_bearing))


def _generate_ambiguous_bearing(
    true_bearing_rad: float, array_heading_rad: float
) -> float:
    """Generate left-right ambiguous bearing around array axis heading."""
    ambiguous_bearing = 2 * array_heading_rad - true_bearing_rad
    return np.arctan2(np.sin(ambiguous_bearing), np.cos(ambiguous_bearing))


def generate_stonesoup_detections(
    cfg: dict,
    timesteps: list[datetime],
    target_ground_truths: list[GroundTruthPath],
    platform: TowedArrayPlatform,
) -> list[tuple[datetime, list[Detection]]]:
    """Generate Stone Soup detections with optional left-right ambiguity."""
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
            target_pos = np.array(
                [target_state.state_vector[0], target_state.state_vector[2]]
            )
            true_bearing = _compute_bearing_world_frame(platform_state, target_pos)

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

                    ambiguous_bearing = _generate_ambiguous_bearing(
                        true_bearing,
                        array_heading,
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

    return stone_soup_detections


def build_plugin_vs_ss_figure(
    timesteps: list[datetime],
    all_detections: list,
    stone_soup_detections: list[tuple[datetime, list[Detection]]],
) -> go.Figure:
    """Build plugin-vs-Stone-Soup detections comparison figure."""
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
        go.Scatter(
            x=det_x, y=det_y, name="Detection", showlegend=False, **scatter_style
        ),
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

    return fig


def write_figures(
    figures: dict[str, go.Figure], output_dir: str = "figs", scale: float = 1.0
) -> None:
    """Write figures to disk using each figure's configured layout size."""
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
