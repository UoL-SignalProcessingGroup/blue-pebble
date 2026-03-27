"""
====================
Comparing Bathymetry
====================

This example compares the same passive-sonar scenario under two seafloor assumptions:
a flat seabed and an idealised seamount. The aim is to isolate how bathymetry alone
changes the propagation model, beamformed output, and downstream detections when
everything else in the scenario is held fixed.

In Blue Pebble, bathymetry is part of the acoustic environment. The seafloor shape can
alter propagation paths and the relative strength of arrivals that reach the array.
Differences in the final SNR maps and detections can therefore be attributed to seabed
geometry rather than to a different signal-processing chain.
"""  # noqa: D205, D212, D400, D415

# %%
# Imports
# -------
#
# All dependencies are consolidated here for convenience.
from copy import deepcopy
from datetime import datetime, timedelta

import numpy as np
from plotly.subplots import make_subplots
from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel,
    ConstantVelocity,
    KnownTurnRate,
)
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState

from bluepebble.detector import CACFARDetector, PassiveSonarDetector, PeakDetector
from bluepebble.models.environment import Constant, FlatBathymetry, SeamountBathymetry
from bluepebble.models.propagation import rtrsAcousticPropagationModel
from bluepebble.platform import TowedArrayPlatform
from bluepebble.plotter import apply_shared_colourscale, plot_btr, plot_world
from bluepebble.signal.anthropogenic import SyntheticAnthropogenicSignal
from bluepebble.signal.random import ColouredNoiseSignal
from bluepebble.sigproc import (
    DelayAndSumBeamformer,
    SteeringCalculator,
)
from bluepebble.simulator import ContinuousSTFTPassiveSonarArraySimulator

# %%
# Simulation Parameters
# ---------------------
#
# This section fixes the random seed and defines one shared simulation clock for the whole
# comparison. Keeping the timing identical across both bathymetry runs ensures that any
# later difference comes from propagation over the seabed, not from inconsistent platform
# or target updates.

seed = 2000
np.random.seed(seed)

sim_length_s = 900  # seconds
sim_rate_s = 5.0  # seconds

start_time = datetime(2026, 1, 1, 0, 0, 0)
time_interval = timedelta(seconds=sim_rate_s)
num_steps = int(sim_length_s / sim_rate_s)

total_duration_s = num_steps * time_interval.total_seconds()

# %%
# Platform Setup and Generation
# -----------------------------
#
# Here the ownship trajectory and towed-array geometry are defined once and then reused
# throughout the example. The host follows a deterministic multi-leg path so the array
# heading changes over time, which makes the bathymetry comparison more informative than
# a purely straight-line transit.

platform_turn_rate_radps = np.deg2rad(1.0)
leg1_duration_s = timedelta(seconds=405)
turn1_angle_rad = np.deg2rad(-85)
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

platform_start_vector = np.array([0.0, 1.8, 2000.0, 1.8, -5.0, 0.0])
platform_position_mapping = [0, 2, 4]
platform_velocity_mapping = [1, 3, 5]
platform_transition_models = [straight_model, turning_model1, straight_model]
platform_transition_times = [leg1_duration_s, turn1_duration_s, leg2_duration_s]

num_sensors = 200
tow_cable_length_m = 400.0
sensor_spacing_m = 0.5
array_depth_m = -50.0

initial_state = GroundTruthState(platform_start_vector, timestamp=start_time)
platform = TowedArrayPlatform(
    states=[initial_state],
    position_mapping=platform_position_mapping,
    velocity_mapping=platform_velocity_mapping,
    transition_models=platform_transition_models,
    transition_times=platform_transition_times,
    num_sensors=num_sensors,
    cable_length_m=tow_cable_length_m,
    sensor_spacing_m=sensor_spacing_m,
    array_depth_m=array_depth_m,
)

for i in range(1, num_steps):
    new_time = start_time + i * time_interval
    platform.move(new_time)

# %%
# Ground Truth Setup and Generation
# ---------------------------------
#
# Target kinematics and source metadata are generated here. Each target truth is
# propagated over the full timeline, and the corresponding relative-bearing truth is
# computed with respect to the array reference position.

target_start_vectors = [
    np.array([6000, 0.0, 1.0e3, 10, -5.0, 0.0]),
    np.array([4000, -8.0, -5.7e3, 0.0, -5.0, 0.0]),
]

target_transition_model = CombinedLinearGaussianTransitionModel(
    [ConstantVelocity(0.0), ConstantVelocity(0.0), ConstantVelocity(0.0)]
)
target_position_mapping = [0, 2, 4]
target_velocity_mapping = [1, 3, 5]

shared_target_tonal_bandwidth_hz = np.random.uniform(0.5, 2.0)
shared_target_noise_amplitude_upa = 10 ** (90 / 20)
shared_target_noise_spectral_exponent = -1.0

target_ground_truths = []
relative_bearing_ground_truths = []

for target_start_vector in target_start_vectors:
    target_amplitudes_upa = 10 ** (np.random.uniform(90, 102, 4) / 20)
    target_frequencies_hz = np.random.uniform(120.0, 250.0, 4)
    target_phases_rad = np.random.uniform(0, 2 * np.pi, 4)
    target_tonal_bandwidth_hz = np.random.uniform(0.5, 2.0)
    target_noise_amplitude_upa = 10 ** (np.random.uniform(70, 85) / 20)

    target_states = [
        GroundTruthState(
            target_start_vector,
            timestamp=start_time,
            metadata={
                "amplitudes_upa": target_amplitudes_upa,
                "frequencies_hz": target_frequencies_hz,
                "phases_rad": target_phases_rad,
                "position_mapping": target_position_mapping,
                "velocity_mapping": target_velocity_mapping,
                "tonal_bandwidth_hz": target_tonal_bandwidth_hz,
                "noise_amplitude_upa": target_noise_amplitude_upa,
                "target_tonal_bandwidth_hz": shared_target_tonal_bandwidth_hz,
                "target_noise_amplitude_upa": shared_target_noise_amplitude_upa,
                "noise_spectral_exponent": shared_target_noise_spectral_exponent,
            },
        )
    ]

    for i in range(1, num_steps):
        new_time = start_time + i * time_interval
        time_interval_now = new_time - target_states[-1].timestamp
        new_state_vector = target_transition_model.function(
            target_states[-1], noise=False, time_interval=time_interval_now
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
        target_pos = np.array([target_state.state_vector[0], target_state.state_vector[2]])
        relative_pos = target_pos - ref_sensor_position[:2]
        bearing_rad = np.arctan2(relative_pos[1], relative_pos[0])
        bearing_states.append(
            GroundTruthState(
                state_vector=np.array([bearing_rad]), timestamp=target_state.timestamp
            )
        )

    relative_bearing_ground_truths.append(GroundTruthPath(bearing_states))

# %%
# Propagation Model
# -----------------
#
# This section defines the acoustic environment used by RTRS propagation: one shared
# sound-speed profile, two bathymetry models, and the angular and range sampling controls
# for the solver.

ssp = Constant(speed=1500.0)
flat_bathymetry = FlatBathymetry(depth=-150.0)
seamount_bathymetry = SeamountBathymetry(
    summit_position=(4000.0, 3000.0, -25.0),
    radius=10000.0,
    plateau_depth=-150.0,
)
prop_step_m = 20.0
prop_azimuth_search_width = 10.0
prop_azimuth_resolution = 0.5
prop_elevation_range = (-15.0, 15.0)
prop_elevation_resolution = 0.5
prop_water_density_g_cm3 = 1.0
prop_bottom_model = {
    "model": "elastic",
    "compressional_speed_m_s": 1700.0,
    "shear_speed_m_s": 400.0,
    "density_g_cm3": 1.6,
    "compressional_attenuation_db_per_wavelength": 0.2,
    "shear_attenuation_db_per_wavelength": 0.3,
}
prop_store_ray_paths = False
prop_integration_method = "rk2"

flat_prop_model = rtrsAcousticPropagationModel(
    ssp=ssp,
    bathymetry=flat_bathymetry,
    use_all_frequencies=False,
    step_m=prop_step_m,
    azimuth_search_width=prop_azimuth_search_width,
    azimuth_resolution=prop_azimuth_resolution,
    elevation_range=prop_elevation_range,
    elevation_resolution=prop_elevation_resolution,
    water_density_g_cm3=prop_water_density_g_cm3,
    bottom_model=prop_bottom_model,
    store_ray_paths=prop_store_ray_paths,
    integration_method=prop_integration_method,
)

seamount_prop_model = rtrsAcousticPropagationModel(
    ssp=ssp,
    bathymetry=seamount_bathymetry,
    use_all_frequencies=False,
    step_m=prop_step_m,
    azimuth_search_width=prop_azimuth_search_width,
    azimuth_resolution=prop_azimuth_resolution,
    elevation_range=prop_elevation_range,
    elevation_resolution=prop_elevation_resolution,
    water_density_g_cm3=prop_water_density_g_cm3,
    bottom_model=prop_bottom_model,
    store_ray_paths=prop_store_ray_paths,
    integration_method=prop_integration_method,
)

# %%
# Geometry View: Trajectories over Bathymetry
# -------------------------------------------
#
# These plots use :func:`~bluepebble.plotter.plot_world` with bathymetry overlays to show the
# same kinematic scene against each seabed model. The trajectories do not change; only the
# seafloor under them does.

fig_bathy_flat = plot_world(
    truths=target_ground_truths,
    platform=platform,
    bathymetry=flat_bathymetry,
    figsize=(600, 600),
)

fig_bathy_seamount = plot_world(
    truths=target_ground_truths,
    platform=platform,
    bathymetry=seamount_bathymetry,
    figsize=(600, 600),
)

fig_bathy = make_subplots(
    rows=2,
    cols=1,
    subplot_titles=("Flat", "Seamount"),
    vertical_spacing=0.10,
    shared_xaxes=True,
    shared_yaxes=True,
)


def _add_bathymetry_traces(figure, source_figure, *, row, show_colourbar):
    for trace in source_figure.data:
        trace_copy = deepcopy(trace)
        if trace_copy.type == "heatmap":
            trace_copy.showscale = show_colourbar
            if show_colourbar:
                trace_copy.colorbar = dict(
                    title=dict(text="Depth (m)", side="right"),
                    thickness=24,
                    len=1.0,
                    x=0.8,
                    xanchor="left",
                    y=0.5,
                    yanchor="middle",
                )
        elif row == 2:
            trace_copy.showlegend = False
        figure.add_trace(trace_copy, row=row, col=1)

    suffix = "" if row == 1 else str(row)
    for annotation in source_figure.layout.annotations:
        ann = deepcopy(annotation)
        ann.xref = f"x{suffix}"
        ann.yref = f"y{suffix}"
        ann.axref = f"x{suffix}"
        ann.ayref = f"y{suffix}"
        figure.add_annotation(ann)


_add_bathymetry_traces(fig_bathy, fig_bathy_flat, row=1, show_colourbar=False)
_add_bathymetry_traces(fig_bathy, fig_bathy_seamount, row=2, show_colourbar=True)

x_title = fig_bathy_flat.layout.xaxis.title.text
y_title = fig_bathy_flat.layout.yaxis.title.text
shared_x_range = fig_bathy_flat.layout.xaxis.range
shared_y_range = fig_bathy_flat.layout.yaxis.range

del fig_bathy_flat, fig_bathy_seamount

common_axis_style = dict(
    constrain="domain",
    showgrid=True,
    gridcolor="#929292",
    zeroline=False,
)

fig_bathy.update_xaxes(range=shared_x_range, scaleratio=1, **common_axis_style)
fig_bathy.update_xaxes(title_text="", scaleanchor="y", showticklabels=False, row=1, col=1)
fig_bathy.update_xaxes(title_text=x_title, scaleanchor="y2", row=2, col=1)
fig_bathy.update_yaxes(title_text=y_title, range=shared_y_range, **common_axis_style)

fig_bathy.update_layout(
    template="plotly_white",
    autosize=True,
    width=None,
    height=750,
    legend=dict(x=0.5, y=-0.14, xanchor="center", yanchor="top", orientation="h"),
    margin=dict(b=90, t=60),
    title="Bathymetry Comparison",
)

# %%
# Signal Model
# ------------
#
# Here the source and ambient signal models are defined. Each target receives a broadband
# ship signal model, and coloured ambient noise is added at the array.

sampling_rate_hz = 500.0
frame_len = 500
hop_factor = 2
fade_in_ms = 1000.0
ambient_amplitude_upa = 10 ** (45 / 20)
ambient_spectral_exponent = -1  # pink noise

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
                tonal_bandwidth_hz=target_metadata["target_tonal_bandwidth_hz"],
                noise_amplitude_upa=target_metadata["target_noise_amplitude_upa"],
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
# This section sets the beamforming parameters and builds the steering calculator used by
# both simulators. The steering grid spans the full azimuth range so that any difference
# in arrival structure caused by the seabed is visible in the resulting bearing-time
# record.

steering_azimuths_rad = np.linspace(-np.pi, np.pi, 181)

beamformer = DelayAndSumBeamformer(
    sampling_rate_hz=sampling_rate_hz,
    domain="frequency",
)

steering_calculator = SteeringCalculator(
    ssp=ssp,
    steering_azimuths_rad=steering_azimuths_rad,
)

# %%
# Detector Pipeline Setup
# -----------------------
#
# Two simulators are created from the same scenario:
#
# - flat bathymetry
# - seamount bathymetry
#
# Both use the same detector chain, CA-CFAR followed by peak selection. That means any
# difference in the final output should come from the seabed model rather than from a
# different detection policy.

cfar_num_guard_cells = 6
cfar_num_training_cells = 10
cfar_threshold_factor = 1.05
cfar_mode = "wrap"
peak_distance = 8


def _make_detector(simulator: ContinuousSTFTPassiveSonarArraySimulator) -> PassiveSonarDetector:
    """Create a PassiveSonarDetector with a CACFARDetector followed by a PeakDetector."""
    cfar_detector = CACFARDetector(
        num_guard_cells=cfar_num_guard_cells,
        num_training_cells=cfar_num_training_cells,
        threshold_factor=cfar_threshold_factor,
        mode=cfar_mode,
    )
    peak_detector = PeakDetector(distance=peak_distance)

    return PassiveSonarDetector(
        detection_chain=[cfar_detector, peak_detector],
        sensor_data_gen=simulator.sensor_data_gen(),
        steering_azimuths_rad=steering_azimuths_rad,
    )


simulator_flat_bathymetry = ContinuousSTFTPassiveSonarArraySimulator(
    platform=platform,
    propagation_model=flat_prop_model,
    signal_models=_make_signal_models(),
    noise_model=ambient_noise_model,
    beamformer=beamformer,
    steering_calculator=steering_calculator,
    ground_truth_paths=target_ground_truths,
    fade_in_ms=fade_in_ms,
)

simulator_seamount_bathymetry = ContinuousSTFTPassiveSonarArraySimulator(
    platform=platform,
    propagation_model=seamount_prop_model,
    signal_models=_make_signal_models(),
    noise_model=ambient_noise_model,
    beamformer=beamformer,
    steering_calculator=steering_calculator,
    ground_truth_paths=target_ground_truths,
    fade_in_ms=fade_in_ms,
)

detector_flat_bathymetry = _make_detector(simulator_flat_bathymetry)
detector_seamount_bathymetry = _make_detector(simulator_seamount_bathymetry)

# %%
# Run Detection on Simulated Data
# -------------------------------
#
# This cell executes both detector pipelines and stores the resulting SNR maps and
# detections for later plotting. At this point the example branches acoustically into
# two cases, but the scenario definition remains otherwise identical.

all_detections_flat_bathymetry = list(
    detector_flat_bathymetry.detections_gen(progress_bar=False, total_timesteps=num_steps)
)
snr_map_flat_bathymetry = detector_flat_bathymetry.snr_history

all_detections_seamount_bathymetry = list(
    detector_seamount_bathymetry.detections_gen(progress_bar=False, total_timesteps=num_steps)
)
snr_map_seamount_bathymetry = detector_seamount_bathymetry.snr_history

timesteps = np.array([start_time + i * time_interval for i in range(num_steps)], dtype=object)
steering_azimuths_deg = np.rad2deg(steering_azimuths_rad)

detections_flat_bathymetry = [
    d for _, detection_set in all_detections_flat_bathymetry for d in detection_set
]
detections_seamount_bathymetry = [
    d for _, detection_set in all_detections_seamount_bathymetry for d in detection_set
]

print(f"Total no. of detections (flat bathymetry): {len(detections_flat_bathymetry)}")
print(f"Total no. of detections (seamount bathymetry): {len(detections_seamount_bathymetry)}")

# %%
# Results: Flat vs Seamount Bathymetry
# ------------------------------------
#
# The final figure compares the two bathymetry conditions in a 2x2 layout:
#
# - left column: raw SNR maps
# - right column: SNR maps with detection overlays
#
# Any shift in structure, contrast, or detection placement between the two rows reflects
# the change from flat seabed to seamount bathymetry under the same source, noise,
# beamforming, and detector settings.

row_titles = ["Flat", "Seamount"]
col_titles = ["SNR Map", "SNR Map w/ Detections"]

fig_snr = make_subplots(
    rows=2,
    cols=2,
    shared_xaxes=True,
    shared_yaxes=True,
    row_titles=row_titles,
    column_titles=col_titles,
)

snr_plot_configs = [
    (1, 1, snr_map_flat_bathymetry, None),
    (1, 2, snr_map_flat_bathymetry, detections_flat_bathymetry),
    (2, 1, snr_map_seamount_bathymetry, None),
    (2, 2, snr_map_seamount_bathymetry, detections_seamount_bathymetry),
]

for row, col, snr_map, detections in snr_plot_configs:
    plot_btr(
        data=snr_map,
        detections=detections,
        timesteps=timesteps,
        steering_azimuths=steering_azimuths_deg,
        fig=fig_snr,
        row=row,
        col=col,
    )

apply_shared_colourscale(
    fig_snr,
    colorbar=dict(
        title=dict(text="SNR (dB)", side="right"),
        x=1.02,
        y=0.5,
        yanchor="middle",
        len=1.0,
        thickness=24,
    ),
)

# Remove x-axis labels from row 1
for col in (1, 2):
    fig_snr.update_xaxes(title_text="", row=1, col=col, showticklabels=False)

# Remove y-axis labels from column 2
for row in (1, 2):
    fig_snr.update_yaxes(title_text="", row=row, col=2, showticklabels=False)

fig_snr.update_layout(
    template="plotly_white",
    autosize=True,
    height=700,
    showlegend=False,
    title="SNR Maps and Detections: Flat vs Seamount Bathymetry",
)
