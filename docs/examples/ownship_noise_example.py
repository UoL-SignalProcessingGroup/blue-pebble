"""
===================
Ownship Noise Example
===================

This example compares the same passive-sonar scenario in two conditions:
ambient noise only, and ambient noise with ownship self-noise added. Holding
all other pipeline settings fixed isolates the effect of ownship interference
on the beamformed output and downstream detection counts.
"""  # noqa: D205, D212, D400, D415

# %%
# Simulation Parameters
# ---------------------
#
# This section defines reproducibility and timing for the full run:
# random seed, simulation duration, step size, start time, and total timesteps.

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

total_duration_s = num_steps * time_interval.total_seconds()
print(f"Total simulation duration: {total_duration_s} seconds")

# %%
# Platform Setup and Generation
# -----------------------------
#
# Here the ownship trajectory and towed-array geometry are defined.
#
# The host platform follows a deterministic multi-leg path (straight-turn-straight),
# and a `GroundTruthPath` that represents ownship self-noise as a moving source is also
# created.

# %%
from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel,
    ConstantVelocity,
    KnownTurnRate,
)
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState

from bluepebble.platform import TowedArrayPlatform

platform_turn_rate_radps = np.deg2rad(1.0)
leg1_duration_s = timedelta(seconds=405)
turn1_angle_rad = np.deg2rad(-45)
turn1_duration_s = timedelta(
    seconds=round((abs(turn1_angle_rad) / platform_turn_rate_radps) / SIM_RATE) * SIM_RATE
)
leg2_duration_s = timedelta(seconds=SIM_LENGTH) - leg1_duration_s - turn1_duration_s

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

platform_start_vector = np.array([-7500.0, 1.15, -2000.0, 0.25, -5.0, 0.0])
platform_position_mapping = [0, 2, 4]
platform_velocity_mapping = [1, 3, 5]
platform_transition_models = [straight_model, turning_model1, straight_model]
platform_transition_times = [leg1_duration_s, turn1_duration_s, leg2_duration_s]

ownship_amplitudes_upa = 10 ** (np.array([80.0, 77.0, 81.0, 76.0]) / 20)
ownship_frequencies_hz = np.array([50.0, 100.0, 130.0, 200.0])
ownship_phases_rad = np.array([0.0, 0.0, 0.0, 0.0])
ownship_tonal_bandwidth_hz = 1.0
ownship_noise_amplitude_upa = 10 ** (74.0 / 20)
ownship_noise_spectral_exponent = -1.0

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

self_noise_states = [
    GroundTruthState(
        state.state_vector,
        timestamp=state.timestamp,
        metadata={
            "position_mapping": platform_position_mapping,
            "velocity_mapping": platform_velocity_mapping,
            "amplitudes_upa": ownship_amplitudes_upa,
            "frequencies_hz": ownship_frequencies_hz,
            "phases_rad": ownship_phases_rad,
            "tonal_bandwidth_hz": ownship_tonal_bandwidth_hz,
            "noise_amplitude_upa": ownship_noise_amplitude_upa,
            "noise_spectral_exponent": ownship_noise_spectral_exponent,
        },
    )
    for state in platform.movement_controller.states
]
self_noise_ground_truth = GroundTruthPath(self_noise_states)

# %%
# Ground Truth Setup and Generation
# ---------------------------------
#
# Target kinematics and source metadata are generated here.
#
# For each target, motion is propagated over all timesteps and the relative-bearing
# truth sequence is computed with respect to the array reference position.
#
# A world-view plot is shown to validate the geometry before beamforming.

# %%
from bluepebble.plotter import plot_world

target_start_vectors = [
    np.array([-1.5e4, 9.0, 1.2e4, -10, -5.0, 0.0]),
    np.array([-1.1e4, -8.0, -5.7e3, 3.6, -5.0, 0.0]),
    np.array([-3.0e3, 10.4, -9.7e3, 9.7, -5.0, 0.0]),
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
    target_amplitudes_upa = 10 ** (np.random.uniform(87, 102, 4) / 20)
    target_frequencies_hz = np.random.uniform(50.0, 200.0, 4)
    target_phases_rad = np.random.uniform(0, 2 * np.pi, 4)
    target_tonal_bandwidth_hz = np.random.uniform(0.5, 2.0)
    target_noise_amplitude_upa = 10 ** (np.random.uniform(65, 85) / 20)

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

fig1 = plot_world(truths=target_ground_truths, platform=platform)

# %%
# Propagation Model
# -----------------
#
# This section configures the acoustic environment used by RTRS propagation:
# sound-speed profile, bathymetry, and angular/range sampling controls.
#
# These settings determine transfer functions from each source to each sensor.

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

prop_model = rtrsAcousticPropagationModel(
    ssp=ssp,
    bathymetry=bathymetry,
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
# Signal Model
# ------------
#
# Here source and ambient signals are defined.
#
# Each target gets a broadband ship signal model, ambient coloured noise is added at the
# array, and an additional broadband model is created for ownship self noise to enable
# the with/without comparison.

# %%
from bluepebble.signal.anthropogenic import SyntheticAnthropogenicSignal
from bluepebble.signal.random import ColouredNoiseSignal

sampling_rate_hz = 500.0
frame_len = 500
hop_factor = 2
fade_in_ms = 1000.0
ambient_amplitude_upa = 10 ** (45 / 20)
ambient_spectral_exponent = -1

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


def _make_self_noise_model():
    return SyntheticAnthropogenicSignal(
        duration_s=total_duration_s,
        sampling_rate_hz=sampling_rate_hz,
        frame_len=frame_len,
        hop_factor=hop_factor,
        tonal_bandwidth_hz=ownship_tonal_bandwidth_hz,
        noise_amplitude_upa=ownship_noise_amplitude_upa,
        noise_spectral_exponent=ownship_noise_spectral_exponent,
        noise_freq_range_hz=(0.0, sampling_rate_hz / 2),
        tonal_noise_is_constant=True,
        noise_is_constant=True,
    )


# %%
# Beamformer
# ----------
#
# This section sets beamforming parameters (domain, shading, steering grid) and builds
# the beamformer + steering calculator used by the simulators.
# The steering grid spans full azimuth so detections can appear over 360 degrees.

# %%
from scipy.signal import get_window

from bluepebble.sigproc import (
    DelayAndSumBeamformer,
    MinimumVarianceDistortionlessResponseBeamformer,
    SteeringCalculator,
)

beamformer_type = "DAS"
beamformer_shading = None
beamformer_domain = "frequency"
steering_azimuths_rad = np.linspace(-np.pi, np.pi, 181)

shading = None
if beamformer_shading is not None:
    shading = get_window(beamformer_shading, platform.num_sensors)

if beamformer_type == "DAS":
    if beamformer_domain == "broadband_power":
        beamformer = DelayAndSumBeamformer(
            domain=beamformer_domain,
            sampling_rate_hz=sampling_rate_hz,
            fmin=0.0,
            fmax=sampling_rate_hz / 2,
        )
    else:
        beamformer = DelayAndSumBeamformer(
            sampling_rate_hz=sampling_rate_hz,
            shading=shading,
            domain=beamformer_domain,
        )
elif beamformer_type == "MVDR":
    beamformer = MinimumVarianceDistortionlessResponseBeamformer(
        sampling_rate_hz=sampling_rate_hz,
        fmin=0.0,
        fmax=sampling_rate_hz / 2,
    )
else:
    raise ValueError(f"Unknown beamformer type: {beamformer_type}")

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
# - without ownship noise
# - with ownship noise (ownship source appended)
#
# Both use the same detector chain (CA-CFAR followed by peak selection) so any difference
# in output is attributable to the added ownship interference.

# %%
from bluepebble.detector import CACFARDetector, PassiveSonarDetector, PeakDetector
from bluepebble.simulator import ContinuousSTFTPassiveSonarArraySimulator

cfar_num_guard_cells = 6
cfar_num_training_cells = 10
cfar_threshold_factor = 1.05
cfar_mode = "wrap"
peak_distance = 8


def make_detector(simulator: ContinuousSTFTPassiveSonarArraySimulator) -> PassiveSonarDetector:
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


simulator_without_ownship_noise = ContinuousSTFTPassiveSonarArraySimulator(
    platform=platform,
    propagation_model=prop_model,
    signal_models=_make_signal_models(),
    noise_model=ambient_noise_model,
    beamformer=beamformer,
    steering_calculator=steering_calculator,
    ground_truth_paths=target_ground_truths,
    fade_in_ms=fade_in_ms,
)

simulator_with_ownship_noise = ContinuousSTFTPassiveSonarArraySimulator(
    platform=platform,
    propagation_model=prop_model,
    signal_models=_make_signal_models() + [_make_self_noise_model()],
    noise_model=ambient_noise_model,
    beamformer=beamformer,
    steering_calculator=steering_calculator,
    ground_truth_paths=target_ground_truths + [self_noise_ground_truth],
    fade_in_ms=fade_in_ms,
)

detector_without_ownship_noise = make_detector(simulator_without_ownship_noise)
detector_with_ownship_noise = make_detector(simulator_with_ownship_noise)

# %%
# Run Detection on Simulated Data
# -------------------------------
#
# This cell executes both detector pipelines and stores SNR maps for each condition.

# %%
all_detections_without_ownship_noise = list(
    detector_without_ownship_noise.detections_gen(progress_bar=False, total_timesteps=num_steps)
)
snr_map_without_ownship_noise = detector_without_ownship_noise.snr_history

all_detections_with_ownship_noise = list(
    detector_with_ownship_noise.detections_gen(progress_bar=False, total_timesteps=num_steps)
)
snr_map_with_ownship_noise = detector_with_ownship_noise.snr_history

timesteps = np.array([start_time + i * time_interval for i in range(num_steps)], dtype=object)
steering_azimuths_deg = np.rad2deg(steering_azimuths_rad)

detections_without_ownship_noise = [
    d for _, detection_set in all_detections_without_ownship_noise for d in detection_set
]
detections_with_ownship_noise = [
    d for _, detection_set in all_detections_with_ownship_noise for d in detection_set
]

print(f"Total no. of detections w/o ownship noise: {len(detections_without_ownship_noise)}")
print(f"Total no. of detections w/ ownship noise: {len(detections_with_ownship_noise)}")

# %%
# Results: With vs Without Ownship Noise
# --------------------------------------
#
# The final figure compares both conditions in a 2x2 layout:
#
# - left column: raw SNR maps
# - right column: SNR maps with detection overlays

# %%
from plotly.subplots import make_subplots

from bluepebble.plotter import plot_btr

fig2 = make_subplots(
    rows=2,
    cols=2,
    shared_yaxes=True,
    subplot_titles=(
        "SNR Map",
        "SNR Map w/ Detections",
        "SNR Map",
        "SNR Map w/ Detections",
    ),
    vertical_spacing=0.15,
)

plot_btr(
    data=snr_map_without_ownship_noise,
    timesteps=timesteps,
    steering_azimuths=steering_azimuths_deg,
    fig=fig2,
    row=1,
    col=1,
)
plot_btr(
    data=snr_map_without_ownship_noise,
    detections=detections_without_ownship_noise,
    timesteps=timesteps,
    steering_azimuths=steering_azimuths_deg,
    fig=fig2,
    row=1,
    col=2,
)
plot_btr(
    data=snr_map_with_ownship_noise,
    timesteps=timesteps,
    steering_azimuths=steering_azimuths_deg,
    fig=fig2,
    row=2,
    col=1,
)
plot_btr(
    data=snr_map_with_ownship_noise,
    detections=detections_with_ownship_noise,
    timesteps=timesteps,
    steering_azimuths=steering_azimuths_deg,
    fig=fig2,
    row=2,
    col=2,
)

# Row-level headings.
fig2.add_annotation(
    x=0.5,
    y=1.08,
    xref="paper",
    yref="paper",
    text="Without Ownship Noise",
    showarrow=False,
    font=dict(size=16),
)
fig2.add_annotation(
    x=0.5,
    y=0.48,
    xref="paper",
    yref="paper",
    text="With Ownship Noise",
    showarrow=False,
    font=dict(size=16),
)

# Keep a single shared colorbar and shared colour scale for all heatmaps.
heatmap_traces = [trace for trace in fig2.data if trace.type == "heatmap"]
if heatmap_traces:
    shared_zmin = min(np.nanmin(np.asarray(trace.z, dtype=float)) for trace in heatmap_traces)
    shared_zmax = max(np.nanmax(np.asarray(trace.z, dtype=float)) for trace in heatmap_traces)

    for i, trace in enumerate(heatmap_traces):
        trace.zmin = shared_zmin
        trace.zmax = shared_zmax
        trace.showscale = i == 0

    heatmap_traces[0].colorbar = dict(
        title=dict(text="SNR (dB)", side="right"),
        x=1.02,
        y=0.5,
        yanchor="middle",
        len=0.75,
        thickness=30,
    )

# Remove y-axis labels from column 2.
fig2.update_yaxes(title_text="", row=1, col=2)
fig2.update_yaxes(title_text="", row=2, col=2)

fig2.update_layout(
    width=1400,
    height=1200,
    showlegend=False,
    margin=dict(r=90, t=110),
)
