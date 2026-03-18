"""Multi-Narrowband Beamformer Comparison Example.

This example compares several narrowband MVDR frequency selections on a shared
passive-sonar scenario. The aim is to isolate how band choice changes the resulting
BTR structure, detection count, and mid-scan bearing behaviour when everything else
in the pipeline is held fixed.

**Background**

- Different frequency bands can emphasise different parts of a target signature or
  suppress different noise contributions.
- In narrowband beamforming, a seemingly small change in band limits can noticeably
  alter peak sharpness, clutter structure, and detection stability.
- A controlled comparison is most useful when geometry, propagation, and detector
  settings remain identical across runs.

**Key Concepts**

- MVDR beamforming under multiple narrowband selections.
- Controlled comparison using shared propagation and detection settings.
- Per-band BTR inspection, cross-band comparison, and bearing-cut diagnostics.
"""

# %% [markdown]
# Simulation Parameters
# ---------------------
#
# This section defines reproducibility and timing for the full run: random seed,
# simulation duration, step size, start time, and the full timestep sequence used by
# every pipeline.

# %%
from datetime import datetime, timedelta

import numpy as np

seed = 2000
np.random.seed(seed)

SIM_LENGTH = 900  # seconds
SIM_RATE = 5.0  # seconds

start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
time_interval = timedelta(seconds=SIM_RATE)
num_steps = int(SIM_LENGTH / SIM_RATE)
timesteps = np.array([start_time + i * time_interval for i in range(num_steps)], dtype=object)

total_duration_s = num_steps * time_interval.total_seconds()
print(f"Total simulation duration: {total_duration_s} seconds")

# %% [markdown]
# Platform Setup and Generation
# -----------------------------
#
# Here the ownship trajectory and towed-array geometry are defined.
#
# The host platform follows randomised multi-leg manoeuvres generated from a
# straight/turn process over the full simulation duration.

# %%
from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel,
    ConstantVelocity,
    KnownTurnRate,
)
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState

from bluepebble.platform import TowedArrayPlatform


def generate_random_manoeuvres(
    total_duration_s,
    timestep_s,
    min_leg_duration_s=60,
    max_leg_duration_s=300,
    manoeuvre_prob=0.5,
):
    """Generate random straight and turning manoeuvre segments."""
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


platform_start_vector = np.array([5000.0, 0.0, 1000.0, 5.0, -5.0, 0.0])
platform_position_mapping = [0, 2, 4]
platform_velocity_mapping = [1, 3, 5]

num_sensors = 50
tow_cable_length_m = 100.0
sensor_spacing_m = 0.5
array_depth_m = -50.0

manoeuvre_models, manoeuvre_durations, manoeuvre_descriptions = generate_random_manoeuvres(
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

print("Platform manoeuvre plan:")
for description in manoeuvre_descriptions:
    print(f"- {description}")

# %% [markdown]
# Ground Truth Setup and Generation
# ---------------------------------
#
# Target kinematics and source metadata are generated here.
#
# Each target configuration is kept in a single dictionary. Cartesian truths are then
# converted to relative-bearing truths for BTR overlays, and `plot_world` is used to
# validate the geometry before beamforming.

# %%
from bluepebble.plotter import plot_world

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

fig_world = plot_world(truths=target_ground_truths, platform=platform)
fig_world.update_layout(title="World Picture: Target and Platform Trajectories")
fig_world.show()

# %% [markdown]
# Propagation Model
# -----------------
#
# This section configures the acoustic environment used by RTRS propagation: sound-speed
# profile, bathymetry, and angular/range sampling controls.
#
# These settings are shared across all beamformer runs so output differences come from
# band selection rather than from changes in propagation.

# %%
from bluepebble.models.environment import FlatBathymetry, Linear
from bluepebble.models.propagation import rtrsAcousticPropagationModel

ssp = Linear(surface_speed=1500.0, gradient=0.2)
bathymetry = FlatBathymetry(depth=-150.0)
prop_step_m = 20.0
prop_azimuth_search_width = 2.0
prop_azimuth_resolution = 0.5
prop_elevation_range = (-25.0, 25.0)
prop_elevation_resolution = 1.0

prop_model = rtrsAcousticPropagationModel(
    ssp=ssp,
    bathymetry=bathymetry,
    step_m=prop_step_m,
    azimuth_search_width=prop_azimuth_search_width,
    azimuth_resolution=prop_azimuth_resolution,
    elevation_range=prop_elevation_range,
    elevation_resolution=prop_elevation_resolution,
)

# %% [markdown]
# Signal Model
# ------------
#
# Here target and ambient signals are defined.
#
# Each target gets its own broadband ship signal model so the scenario-specific metadata
# carries through simulation, while ambient coloured noise is added once per integration
# interval at the array.

# %%
from bluepebble.signal.anthropogenic import SyntheticAnthropogenicSignal
from bluepebble.signal.random import ColouredNoiseSignal

sampling_rate_hz = 500.0
frame_len = 500
hop_factor = 2
fade_in_ms = 1000.0

ambient_noise_amplitude_upa = 10 ** (50.0 / 20)
ambient_noise_spectral_exponent = -1.0

ambient_noise_model = ColouredNoiseSignal(
    amplitude_upa=ambient_noise_amplitude_upa,
    spectral_exponent=ambient_noise_spectral_exponent,
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

# %% [markdown]
# Beamformer
# ----------
#
# This section builds the steering grid and one MVDR beamformer per narrowband
# configuration.
#
# All detectors downstream use the same propagation model, simulator settings, and
# detection chain, so any difference in output should be attributable to frequency-band
# choice.

# %%
from bluepebble.sigproc import MinimumVarianceDistortionlessResponseBeamformer, SteeringCalculator

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

print("Frequency bands:")
for config in freq_bands:
    print(f"- {config['label']}")

# %% [markdown]
# Detector Pipeline Setup
# -----------------------
#
# This section builds one simulator + detector pipeline per beamformer configuration.
#
# All pipelines share the same scenario and detector chain so output differences come from
# beamformer band selection rather than different downstream logic.

# %%
from bluepebble.detector import CACFARDetector, PassiveSonarDetector, PeakDetector
from bluepebble.simulator import ContinuousSTFTPassiveSonarArraySimulator

cfar_num_guard_cells = 2
cfar_num_training_cells = 10
cfar_threshold_factor = 2.1
peak_distance = 3


def make_detector(simulator, steering_azimuths_rad):
    """Create a PassiveSonarDetector with a CACFARDetector followed by a PeakDetector."""
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
        propagation_model=prop_model,
        signal_models=_make_signal_models(),
        noise_model=ambient_noise_model,
        beamformer=beamformer,
        steering_calculator=steering_calculator,
        ground_truth_paths=target_ground_truths,
        fade_in_ms=fade_in_ms,
    )
    simulators.append(simulator)
    detectors.append(make_detector(simulator, steering_azimuths_rad))

# %% [markdown]
# Run Detection on Simulated Data
# -------------------------------
#
# This cell executes all detector pipelines, stores detections and SNR maps, and prints
# a short comparison summary for the configured bands.

# %%
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

# %% [markdown]
# Per-Band BTR Inspection
# -----------------------
#
# These figures inspect each beamformer configuration one at a time. For each band,
# the raw SNR field is shown first and then the same field with detections and bearing
# truth overlays.

# %%
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from bluepebble.plotter import plot_btr

for snr_map, detections_flat, config in zip(
    snr_maps,
    detections_flat_per_bf,
    freq_bands,
    strict=False,
):
    map_rows = min(len(timesteps), snr_map.shape[0])
    timesteps_map = np.array(timesteps[:map_rows])
    snr_map_plot = snr_map[:map_rows]

    fig_btr = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        horizontal_spacing=0.08,
        subplot_titles=(
            "SNR Map",
            "SNR Map w/ Detections & Ground Truth",
        ),
    )

    plot_btr(
        data=snr_map_plot,
        timesteps=timesteps_map,
        steering_azimuths=np.rad2deg(steering_azimuths_rad),
        fig=fig_btr,
        row=1,
        col=1,
    )
    plot_btr(
        data=snr_map_plot,
        detections=detections_flat,
        truths=relative_bearing_ground_truths,
        timesteps=timesteps_map,
        steering_azimuths=np.rad2deg(steering_azimuths_rad),
        fig=fig_btr,
        row=1,
        col=2,
    )

    fig_btr.update_layout(
        width=1280,
        height=640,
        title=f"Frequency Band: {config['label']}",
        legend=dict(
            x=1.12,
        ),
    )
    fig_btr.update_yaxes(title_text="Time (HH:MM)", row=1, col=1)
    fig_btr.update_yaxes(title_text="", row=1, col=2)
    fig_btr.show()

# %% [markdown]
# Cross-Band Comparison
# ---------------------
#
# This comparison puts all beamformer outputs on the same colour scale so band-to-band
# differences in concentration, smear, and target contrast are easier to judge directly.

# %%
fig_compare = make_subplots(
    rows=1,
    cols=len(freq_bands),
    shared_yaxes=True,
    horizontal_spacing=0.04,
    subplot_titles=[config["label"] for config in freq_bands],
)

for i, snr_map in enumerate(snr_maps, start=1):
    map_rows = min(len(timesteps), snr_map.shape[0])
    timesteps_map = np.array(timesteps[:map_rows])
    snr_map_plot = snr_map[:map_rows]
    plot_btr(
        data=snr_map_plot,
        timesteps=timesteps_map,
        steering_azimuths=np.rad2deg(steering_azimuths_rad),
        fig=fig_compare,
        row=1,
        col=i,
    )
    if i > 1:
        fig_compare.update_yaxes(title_text="", row=1, col=i)

vmin = 0.0
vmax = 40.0
heatmap_count = 0
for trace in fig_compare.data:
    if trace.type == "heatmap":
        trace.zmin = vmin
        trace.zmax = vmax
        trace.showscale = heatmap_count == 0
        if heatmap_count == 0:
            trace.colorbar = dict(
                title="SNR (dB)",
                x=1.02,
                xanchor="left",
                y=0.5,
                len=0.9,
                thickness=20,
            )
        heatmap_count += 1

fig_compare.update_layout(
    width=1400,
    height=600,
    margin=dict(r=120),
    title="SNR Comparison Across Frequency Bands",
    showlegend=False,
)
fig_compare.update_yaxes(title_text="Time (HH:MM)", row=1, col=1)
fig_compare.show()

# %% [markdown]
# Bearing-Cut Diagnostics
# -----------------------
#
# These bearing cuts inspect a single mid-scenario timestep for each band. They make it
# easier to compare peak sharpness, truth alignment, and any extra detections that are
# hard to judge from the full BTR alone.

# %%
for all_detections, snr_map, config in zip(
    all_detections_per_bf,
    snr_maps,
    freq_bands,
    strict=False,
):
    detections_by_time = [detection_set for _, detection_set in all_detections]

    middle_timestep_idx = min(num_steps // 2, snr_map.shape[0] - 1)
    middle_snr = snr_map[middle_timestep_idx, :]
    middle_time = timesteps[middle_timestep_idx]
    steering_azimuths_deg = np.rad2deg(steering_azimuths_rad)

    y_span = max(float(np.ptp(middle_snr)), 1.0)
    y_pad = 0.05 * y_span
    y_min = float(np.min(middle_snr) - y_pad)
    y_max = float(np.max(middle_snr) + y_pad)

    fig_middle = go.Figure()
    fig_middle.add_trace(
        go.Scatter(
            x=steering_azimuths_deg,
            y=middle_snr,
            mode="lines",
            line=dict(width=2, color="black"),
            showlegend=False,
        )
    )

    colorway = px.colors.qualitative.Plotly
    for target_idx, gt_path in enumerate(relative_bearing_ground_truths):
        gt_bearing_deg = float(np.rad2deg(gt_path.states[middle_timestep_idx].state_vector[0]))
        fig_middle.add_trace(
            go.Scatter(
                x=[gt_bearing_deg, gt_bearing_deg],
                y=[y_min, y_max],
                mode="lines",
                line=dict(color=colorway[target_idx % len(colorway)], dash="dash", width=2),
                name=f"Truth {target_idx + 1} ({gt_bearing_deg:.1f}°)",
            )
        )

    middle_detections = list(detections_by_time[middle_timestep_idx])
    for det_idx, det in enumerate(middle_detections):
        det_bearing_deg = float(np.rad2deg(det.state_vector[0]))
        fig_middle.add_trace(
            go.Scatter(
                x=[det_bearing_deg, det_bearing_deg],
                y=[y_min, y_max],
                mode="lines",
                line=dict(color="green", dash="dot", width=2),
                name="Detection",
                showlegend=det_idx == 0,
                opacity=0.75,
            )
        )

    fig_middle.update_layout(
        template="plotly_white",
        width=950,
        height=460,
        title=f"SNR Bearing Cut: {config['label']} at {middle_time.time()}",
        xaxis_title="Bearing (degrees)",
        yaxis_title="SNR (dB)",
    )
    fig_middle.update_yaxes(range=[y_min, y_max])
    fig_middle.show()
