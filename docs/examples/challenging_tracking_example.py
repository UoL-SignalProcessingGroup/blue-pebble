"""Challenging Tracking Example.

This example builds a deliberately difficult multi-target passive-sonar scenario and
then tracks the resulting detections with Stone Soup. It is intended as a stress case
where manoeuvres, clutter-like detections, and overlapping bearing structure make
association meaningfully harder.

**Background**

- Bearing-only passive-sonar tracking is sensitive to missed detections, false alarms,
  and ambiguous target separation.
- When multiple targets manoeuvre while the observing platform also manoeuvres, the BTR
  structure becomes richer but also harder to associate scan-to-scan.
- This kind of scenario is useful for evaluating the limits of JPDA-style tracking
  pipelines built on top of Blue Pebble detections.

**Key Concepts**

- Coupled platform motion, target manoeuvres, and propagation effects.
- Broadband beamforming and threshold-based detection in a difficult geometry.
- Bearing-only tracking with Stone Soup under ambiguous association conditions.
"""

# %% [markdown]
# We first define global timing, reproducibility, and helper functions.
#
# This section seeds NumPy, builds `timesteps`, and defines helper utilities used by
# both platform and target generation.

# %%
from datetime import datetime, timedelta

import numpy as np
from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel,
    ConstantVelocity,
    KnownTurnRate,
)
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState

from bluepebble.models.environment import FlatBathymetry, Linear
from bluepebble.models.propagation import rtrsAcousticPropagationModel
from bluepebble.platform import TowedArrayPlatform

seed = 12
np.random.seed(seed)


def random_start_vector(
    x_lim: tuple,
    vx_lim: tuple,
    y_lim: tuple,
    vy_lim: tuple,
) -> np.ndarray:
    """Generate a random start vector for a target within the specified limits.

    Parameters
    ----------
    x_lim : tuple
        (min_x, max_x) limits for the x position.
    vx_lim : tuple
        (min_vx, max_vx) limits for the x velocity.
    y_lim : tuple
        (min_y, max_y) limits for the y position.
    vy_lim : tuple
        (min_vy, max_vy) limits for the y velocity.

    Returns
    -------
    np.ndarray
        A 6D state vector [x, vx, y, vy, z, vz] with z fixed at -5.0 and vz fixed at 0.0.

    """
    return np.array(
        [
            np.random.uniform(x_lim[0], x_lim[1]),
            np.random.uniform(vx_lim[0], vx_lim[1]),
            np.random.uniform(y_lim[0], y_lim[1]),
            np.random.uniform(vy_lim[0], vy_lim[1]),
            -5.0,
            0.0,
        ]
    )


def generate_random_manoeuvres(
    total_duration_s: float,
    timestep_s: float,
    min_leg_duration_s: float = 60.0,
    max_leg_duration_s: float = 300.0,
    manoeuvre_prob: float = 0.5,
) -> tuple:
    """Generate a random sequence of manoeuvres for a target.

    Parameters
    ----------
    total_duration_s : float
        Total duration of the manoeuvre sequence in seconds.
    timestep_s : float
        Time step for rounding manoeuvre durations in seconds.
    min_leg_duration_s : float, optional
        Minimum duration of each manoeuvre leg in seconds, by default 60.0.
    max_leg_duration_s : float, optional
        Maximum duration of each manoeuvre leg in seconds, by default 300.0.
    manoeuvre_prob : float, optional
        Probability of performing a manoeuvre (turn) instead of straight motion,
        by default 0.5.

    Returns
    -------
    tuple[list[CombinedLinearGaussianTransitionModel], list[timedelta]]
        A tuple of lists containing the transition models and durations for each
        manoeuvre leg.

    """
    manoeuvre_models = []
    manoeuvre_durations = []

    current_time = 0.0
    while current_time < total_duration_s - min_leg_duration_s:
        leg_duration = np.random.uniform(min_leg_duration_s, max_leg_duration_s)
        leg_duration = round(leg_duration / timestep_s) * timestep_s
        leg_duration = min(leg_duration, total_duration_s - current_time)

        if np.random.rand() > manoeuvre_prob:
            model = CombinedLinearGaussianTransitionModel(
                [ConstantVelocity(0.0), ConstantVelocity(0.0), ConstantVelocity(0.0)]
            )
        else:
            turn_angle_deg = np.random.uniform(-90.0, 90.0)
            turn_rate_deg_per_s = 1.0
            turn_duration_s = abs(turn_angle_deg) / turn_rate_deg_per_s
            turn_duration_s = round(turn_duration_s / timestep_s) * timestep_s
            leg_duration = min(turn_duration_s, total_duration_s - current_time)
            turn_rate_rad = np.deg2rad(np.sign(turn_angle_deg) * turn_rate_deg_per_s)
            model = CombinedLinearGaussianTransitionModel(
                [
                    KnownTurnRate(
                        turn_rate=turn_rate_rad,
                        turn_noise_diff_coeffs=np.array([0.0, 0.0]),
                    ),
                    ConstantVelocity(0.0),
                ]
            )

        manoeuvre_models.append(model)
        manoeuvre_durations.append(timedelta(seconds=leg_duration))
        current_time += leg_duration

    return manoeuvre_models, manoeuvre_durations


sim_duration = timedelta(seconds=1500)
time_interval = timedelta(seconds=5)
num_steps = int(sim_duration.total_seconds() / time_interval.total_seconds())
start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
timesteps = np.array([start_time + i * time_interval for i in range(num_steps)], dtype=object)

total_duration_s = sim_duration.total_seconds()
num_targets = 10
x_lim = (-6000.0, 6000.0)
y_lim = (-6000.0, 6000.0)

platform_start_vector = np.array([0.0, -5.0, 0.0, 0.0, -5.0, 0.0])
platform_position_mapping = [0, 2, 4]
platform_velocity_mapping = [1, 3, 5]

num_sensors = 250
tow_cable_length_m = 100.0
sensor_spacing_m = 1.0
array_depth_m = -30.0

# %% [markdown]
# Platform Setup and Generation
# -----------------------------
#
# This section builds a manoeuvring `TowedArrayPlatform` and propagates it across all
# timesteps.
#
# To keep this specific random seed in a harder but stable geometry, positive turn-rate
# legs are mirrored to right turns.

# %%
platform_manoeuvre_models, platform_manoeuvre_durations = generate_random_manoeuvres(
    total_duration_s=total_duration_s,
    timestep_s=time_interval.total_seconds(),
    min_leg_duration_s=60.0,
    max_leg_duration_s=300.0,
    manoeuvre_prob=0.25,
)

for i, model in enumerate(platform_manoeuvre_models):
    base_model = model.model_list[0]
    if hasattr(base_model, "turn_rate") and base_model.turn_rate > 0:
        platform_manoeuvre_models[i] = CombinedLinearGaussianTransitionModel(
            [
                KnownTurnRate(
                    turn_rate=-base_model.turn_rate,
                    turn_noise_diff_coeffs=np.array([0.0, 0.0]),
                ),
                ConstantVelocity(0.0),
            ]
        )

platform = TowedArrayPlatform(
    states=GroundTruthState(platform_start_vector, timestamp=start_time),
    position_mapping=platform_position_mapping,
    velocity_mapping=platform_velocity_mapping,
    transition_models=platform_manoeuvre_models,
    transition_times=platform_manoeuvre_durations,
    num_sensors=num_sensors,
    cable_length_m=tow_cable_length_m,
    sensor_spacing_m=sensor_spacing_m,
    array_depth_m=array_depth_m,
)

for timestamp in timesteps[1:]:
    platform.move(timestamp)

# %% [markdown]
# Ground Truth Setup and Generation
# ---------------------------------
#
# We generate `num_targets` target trajectories, each with random piecewise manoeuvres
# and independent signal metadata.
#
# Cartesian truths are then converted to relative-bearing truths for BTR overlays, and
# `plot_world` is used to inspect geometry.

# %%
from bluepebble.plotter import plot_world

target_truths = []
relative_bearing_truths = []
target_signal_metadata = []

default_straight_model = CombinedLinearGaussianTransitionModel(
    [ConstantVelocity(0.0), ConstantVelocity(0.0), ConstantVelocity(0.0)]
)

for _ in range(num_targets):
    target_start_vector = random_start_vector(x_lim, (-10.5, 10.5), y_lim, (-10.5, 10.5))
    manoeuvre_models, manoeuvre_durations = generate_random_manoeuvres(
        total_duration_s=total_duration_s,
        timestep_s=time_interval.total_seconds(),
        min_leg_duration_s=60.0,
        max_leg_duration_s=300.0,
        manoeuvre_prob=0.20,
    )

    if not manoeuvre_models:
        manoeuvre_models = [default_straight_model]
        manoeuvre_durations = [timedelta(seconds=total_duration_s)]

    metadata = {
        "position_mapping": [0, 2, 4],
        "velocity_mapping": [1, 3, 5],
        "amplitudes_upa": 10 ** (np.random.uniform(87, 102, 4) / 20),
        "frequencies_hz": np.random.uniform(25.0, 200.0, 4),
        "phases_rad": np.random.uniform(0.0, 2.0 * np.pi, 4),
        "tonal_bandwidth_hz": np.random.uniform(0.5, 2.0),
        "noise_amplitude_upa": 10 ** (np.random.uniform(75, 85) / 20),
        "noise_spectral_exponent": -1.0,
    }
    target_signal_metadata.append(metadata)

    step_counts = [
        int(round(d.total_seconds() / time_interval.total_seconds())) for d in manoeuvre_durations
    ]
    cumulative_steps = np.cumsum([0] + step_counts)

    states = [GroundTruthState(target_start_vector, timestamp=start_time, metadata=metadata)]
    manoeuvre_idx = 0

    for i, timestamp in enumerate(timesteps[1:], start=1):
        while (
            manoeuvre_idx + 1 < len(cumulative_steps) - 1
            and i >= cumulative_steps[manoeuvre_idx + 1]
        ):
            manoeuvre_idx += 1

        transition_model = manoeuvre_models[min(manoeuvre_idx, len(manoeuvre_models) - 1)]
        dt = timestamp - states[-1].timestamp
        next_state_vector = transition_model.function(states[-1], noise=False, time_interval=dt)
        states.append(GroundTruthState(next_state_vector, timestamp=timestamp, metadata=metadata))

    truth_path = GroundTruthPath(states)
    target_truths.append(truth_path)

    bearing_states = []
    for state in truth_path:
        platform_state = platform.get_platform_state_at(state.timestamp)
        assert platform_state is not None
        reference_xy = np.mean(platform_state.array.state_vector, axis=1)[:2]
        target_xy = np.array([float(state.state_vector[0]), float(state.state_vector[2])])
        relative_xy = target_xy - reference_xy
        bearing = np.arctan2(relative_xy[1], relative_xy[0])
        bearing_states.append(GroundTruthState(np.array([bearing]), timestamp=state.timestamp))

    relative_bearing_truths.append(GroundTruthPath(bearing_states))

plot_world(truths=target_truths, platform=platform).show()

# %% [markdown]
# Propagation Model
# -----------------
#
# This section configures linear SSP, flat bathymetry, and RTRS propagation controls for
# the random scenario.

# %%
ssp = Linear(surface_speed=1500.0, gradient=0.2)
bathymetry = FlatBathymetry(depth=-150.0)

propagation_model = rtrsAcousticPropagationModel(
    ssp=ssp,
    bathymetry=bathymetry,
    step_m=20.0,
    azimuth_search_width=2.0,
    azimuth_resolution=0.5,
    elevation_range=(-25.0, 25.0),
    elevation_resolution=1.0,
)

# %% [markdown]
# Signal Model
# ------------
#
# Ambient background and per-target source signals are defined with shared STFT settings.
#
# Each target gets its own `SyntheticAnthropogenicSignal` instance so randomised source
# metadata carries through simulation.

# %%
from bluepebble.signal.anthropogenic import SyntheticAnthropogenicSignal
from bluepebble.signal.random import ColouredNoiseSignal

sampling_rate_hz = 500.0
frame_len = 500
hop_factor = 2
fade_in_ms = 1000.0

ambient_noise_model = ColouredNoiseSignal(
    amplitude_upa=10 ** (np.random.uniform(50.0, 55.0) / 20),
    spectral_exponent=-1,
    duration_s=time_interval.total_seconds(),
    sampling_rate_hz=sampling_rate_hz,
)

signal_models = []
for metadata in target_signal_metadata:
    signal_models.append(
        SyntheticAnthropogenicSignal(
            duration_s=total_duration_s,
            sampling_rate_hz=sampling_rate_hz,
            frame_len=frame_len,
            hop_factor=hop_factor,
            tonal_bandwidth_hz=metadata["tonal_bandwidth_hz"],
            noise_amplitude_upa=metadata["noise_amplitude_upa"],
            noise_spectral_exponent=metadata["noise_spectral_exponent"],
            noise_freq_range_hz=(0.0, sampling_rate_hz / 2),
            tonal_noise_is_constant=True,
            noise_is_constant=True,
        )
    )

# %% [markdown]
# Beamformer
# ----------
#
# This stage runs broadband-power DAS beamforming, CA-CFAR + peak detection, and produces
# a BTR with detection overlays.

# %%
from bluepebble.detector import CACFARDetector, PassiveSonarDetector, PeakDetector
from bluepebble.plotter import plot_btr
from bluepebble.sigproc import DelayAndSumBeamformer, SteeringCalculator
from bluepebble.simulator import ContinuousSTFTPassiveSonarArraySimulator

steering_azimuths_rad = np.linspace(-np.pi, np.pi, 181)

beamformer = DelayAndSumBeamformer(
    domain="broadband_power",
    sampling_rate_hz=sampling_rate_hz,
    fmin=25.0,
    fmax=200.0,
)

steering_calculator = SteeringCalculator(ssp=ssp, steering_azimuths_rad=steering_azimuths_rad)

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
    num_guard_cells=2,
    num_training_cells=10,
    threshold_factor=1.10,
)
peak_detector = PeakDetector(distance=3)

detector = PassiveSonarDetector(
    detection_chain=[cfar_detector, peak_detector],
    sensor_data_gen=simulator.sensor_data_gen(),
    steering_azimuths_rad=steering_azimuths_rad,
)

all_detections = list(detector.detections_gen(progress_bar=False))
snr_map = detector.snr_history

detections_for_plotter = [d for _, detections in all_detections for d in detections]

print(f"Total no. of detections: {len(detections_for_plotter)}")

# %%
from plotly.subplots import make_subplots

fig = make_subplots(
    rows=1,
    cols=2,
    shared_yaxes=True,
    subplot_titles=("SNR Map", "SNR Map w/ Detections & Truths"),
)

plot_btr(
    data=snr_map,
    timesteps=timesteps,
    steering_azimuths=np.rad2deg(steering_azimuths_rad),
    fig=fig,
    row=1,
    col=1,
)
plot_btr(
    data=snr_map,
    truths=relative_bearing_truths,
    detections=detections_for_plotter,
    timesteps=timesteps,
    steering_azimuths=np.rad2deg(steering_azimuths_rad),
    fig=fig,
    row=1,
    col=2,
)

fig.update_layout(
    width=1200,
    height=700,
    yaxis2=dict(title=""),
    showlegend=True,
    legend=dict(x=1.15, y=1.0, yanchor="top", xanchor="left"),
)
fig.show()

# %% [markdown]
# Tracker
# -------
#
# Tracking is run on plugin detections using a bearing-only Kalman/JPDA setup.
#
# When detections are available, the resulting tracks are overlaid with truths and
# detections on the BTR.

# %%
from stonesoup.dataassociator.neighbour import GNNWith2DAssignment
from stonesoup.dataassociator.probability import JPDA
from stonesoup.deleter.error import CovarianceBasedDeleter
from stonesoup.functions import mod_bearing
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.hypothesiser.probability import PDAHypothesiser
from stonesoup.initiator.simple import MultiMeasurementInitiator
from stonesoup.measures import Mahalanobis
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.models.transition.linear import ConstantVelocity
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.tracker.simple import MultiTargetMixtureTracker
from stonesoup.types.state import GaussianState
from stonesoup.updater.kalman import KalmanUpdater

tracks = []

if detections_for_plotter:
    transition_model = ConstantVelocity(1e-6)
    predictor = KalmanPredictor(transition_model)

    measurement_model = LinearGaussian(
        ndim_state=2,
        mapping=[0],
        noise_covar=np.array([[np.deg2rad(6.0) ** 2]]),
    )
    updater = KalmanUpdater(measurement_model=measurement_model)

    fov_rad = np.deg2rad(360.0)
    expected_false_alarms_per_scan = 3
    clutter_spatial_density = expected_false_alarms_per_scan / fov_rad

    hypothesiser = PDAHypothesiser(
        predictor=predictor,
        updater=updater,
        clutter_spatial_density=clutter_spatial_density,
        prob_detect=0.85,
    )
    data_associator = JPDA(hypothesiser=hypothesiser)

    init_hypothesiser = DistanceHypothesiser(
        predictor=predictor,
        updater=updater,
        measure=Mahalanobis(),
        missed_distance=6,
    )
    init_associator = GNNWith2DAssignment(init_hypothesiser)

    deleter = CovarianceBasedDeleter(covar_trace_thresh=0.2)

    initial_bearing = float(detections_for_plotter[0].state_vector[0, 0])
    prior_state = GaussianState(
        np.array([[initial_bearing], [0.0]]),
        np.diag([np.deg2rad(5.0) ** 2, np.deg2rad(0.5) ** 2]),
        timestamp=start_time,
    )

    initiator = MultiMeasurementInitiator(
        prior_state=prior_state,
        measurement_model=measurement_model,
        deleter=deleter,
        data_associator=init_associator,
        updater=updater,
        min_points=20,
    )

    tracker = MultiTargetMixtureTracker(
        initiator=initiator,
        deleter=deleter,
        detector=all_detections,
        data_associator=data_associator,
        updater=updater,
    )

    track_set = set()
    for _, current_tracks in tracker:
        for track in current_tracks:
            track[-1].state_vector[0, 0] = mod_bearing(float(track[-1].state_vector[0, 0]))
        track_set |= current_tracks

    tracks = list(track_set)

    plot_btr(
        timesteps=timesteps,
        steering_azimuths=np.rad2deg(steering_azimuths_rad),
        truths=relative_bearing_truths,
        detections=detections_for_plotter,
        tracks=tracks,
        figsize=(900, 700),
    ).show()
else:
    print("No detections were produced, so tracker execution was skipped.")
