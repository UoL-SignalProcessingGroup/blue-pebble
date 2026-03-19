"""
==============================
Measured Environmental Data Example
==============================

This example runs one scenario using measured environmental inputs:

- GEBCO bathymetry (seafloor)
- Copernicus temperature/salinity converted to sound speed via Leroy's equation

.. note::

   This example requires external data files (GEBCO bathymetry and Copernicus
   ocean reanalysis) that are not bundled with the repository.  It is excluded
   from the automated gallery build and must be run manually after downloading
   the required datasets.
"""  # noqa: D205, D212, D400, D415
# sphinx_gallery_skip_execution = True

# %%
# Simulation Parameters
# ---------------------
#
# Here core parameters for the simulation are set.

# %%
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

seed = 2000
np.random.seed(seed)

SIM_LENGTH = 1800  # seconds
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
# The ownship trajectory and towed-array geometry are configured here and propagated
# over the full simulation duration.

# %%
from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel,
    ConstantVelocity,
    KnownTurnRate,
)
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState

from bluepebble.platform import TowedArrayPlatform

turn_configs = [
    {"angle_deg": -85.0, "rate_deg_per_s": 1.0},
    {"angle_deg": 85.0, "rate_deg_per_s": 1.0},
]
leg_duration_seconds = [
    SIM_LENGTH / 3,
    SIM_LENGTH / 5,
    SIM_LENGTH / 3,
]
if len(leg_duration_seconds) != len(turn_configs) + 1:
    raise ValueError("Expect one leg duration per segment between turns")

leg_durations_s = [
    timedelta(seconds=round(seconds / SIM_RATE) * SIM_RATE) for seconds in leg_duration_seconds
]

straight_model = CombinedLinearGaussianTransitionModel(
    [ConstantVelocity(0.0), ConstantVelocity(0.0), ConstantVelocity(0.0)]
)

depth_model = ConstantVelocity(0.0)


def _build_turn(angle_deg: float, rate_deg_per_s: float):
    angle_rad = np.deg2rad(angle_deg)
    turn_rate_radps = np.deg2rad(rate_deg_per_s)
    planar_turn = KnownTurnRate(
        turn_rate=np.sign(angle_rad) * turn_rate_radps,
        turn_noise_diff_coeffs=np.array([0.0, 0.0]),
    )
    turn_duration_s = round((abs(angle_rad) / turn_rate_radps) / SIM_RATE) * SIM_RATE
    return CombinedLinearGaussianTransitionModel([planar_turn, depth_model]), timedelta(
        seconds=turn_duration_s
    )


platform_start_vector = np.array([0.0, 2.2, 20000.0, 2.2, -5.0, 0.0])
platform_position_mapping = [0, 2, 4]
platform_velocity_mapping = [1, 3, 5]

maneuvers = []
for idx, turn_config in enumerate(turn_configs):
    maneuvers.append({"type": "leg", "duration": leg_durations_s[idx]})
    maneuvers.append({"type": "turn", **turn_config})
maneuvers.append({"type": "leg", "duration": leg_durations_s[-1]})

platform_transition_models = []
platform_transition_times = []
for maneuver in maneuvers:
    if maneuver["type"] == "leg":
        platform_transition_models.append(straight_model)
        platform_transition_times.append(maneuver["duration"])
    else:
        angle_deg = maneuver["angle_deg"]
        rate_deg_per_s = maneuver["rate_deg_per_s"]
        turn_model, turn_duration = _build_turn(angle_deg, rate_deg_per_s)
        platform_transition_models.append(turn_model)
        platform_transition_times.append(turn_duration)

num_sensors = 70
tow_cable_length_m = 400.0
sensor_spacing_m = 1.1
array_depth_m = -100.0

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
# Target kinematics and source metadata are generated here, along with relative-bearing
# truth sequences used for BTR overlays.

# %%
from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel,
    ConstantVelocity,
)
from stonesoup.types.groundtruth import GroundTruthState

from bluepebble.plotter import plot_world

target_start_vectors = [
    np.array([-15000, 10.0, 20000, 10, -5.0, 0.0]),
    np.array([-10000, 8.0, -10000, 12.0, -5.0, 0.0]),
]

target_transition_model = CombinedLinearGaussianTransitionModel(
    [ConstantVelocity(0.0), ConstantVelocity(0.0), ConstantVelocity(0.0)]
)
target_position_mapping = [0, 2, 4]
target_velocity_mapping = [1, 3, 5]

target_ground_truths = []
relative_bearing_ground_truths = []

for target_start_vector in target_start_vectors:
    target_amplitudes_upa = 10 ** (np.random.uniform(90, 102, 4) / 20)
    target_frequencies_hz = np.random.uniform(120.0, 250.0, 4)
    target_phases_rad = np.random.uniform(0, 2 * np.pi, 4)
    target_tonal_bandwidth_hz = np.random.uniform(0.5, 2.0)
    target_noise_amplitude_upa = 10 ** (np.random.uniform(75, 85) / 20)

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
                "noise_spectral_exponent": -1.0,
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
# Measured Environment Models (GEBCO + Copernicus)
# ------------------------------------------------
#
# Load the measured environmental data and create the corresponding environment models.
# The bathymetry for this example is from the GEBCO 2025 dataset for a region south of
# the Faroe Islands, and the sound speed profile is from the Copernicus Marine Service
# for the same region. This data is not generated by this example and is loaded from the
# `docs/examples/measured_data` folder. Both datasets are in `.nc` NetCDF format.
#
# The files can be downloaded from:
#
# - GEBCO Compilation Group, The GEBCO Grid (GEBCO_2024 Grid).
# - E.U. Copernicus Marine Service Information
#   (https://doi.org/10.48670/moi-00016).
#
# The sound speed profile is calculated from the temperature and salinity profiles using
# Leroy's equation, which is implemented in the `LeroyCopernicusSoundSpeedProfile` class.
# The bathymetry is loaded from the GEBCO dataset using the `GEBCOBathymetry` class.

# %%
import os

from bluepebble.models.environment import GEBCOBathymetry, LeroyCopernicusSoundSpeedProfile
from bluepebble.models.propagation import rtrsAcousticPropagationModel

data_dir = Path(os.getcwd()) / "measured_data"
if not data_dir.exists():
    raise FileNotFoundError(f"Could not find measured_data directory at {data_dir}")

gebco_file = data_dir / "GEBCO_11_Apr_2025_cd1b685d47c9" / "gebco_2024_n61.25_s59.0_w-8.0_e-5.0.nc"
cop_temp_file = (
    data_dir / "cmems_mod_glo_phy-thetao_anfc_0.083deg"
    "_PT6H-i_thetao_8.00W-5.00W_59.00N-61.25N_0.49-5274.78m_2025-03-19.nc"
)
cop_sal_file = (
    data_dir / "cmems_mod_glo_phy-so_anfc_0.083deg"
    "_PT6H-i_so_8.00W-5.00W_59.00N-61.25N_0.49-5274.78m_2025-03-19.nc"
)

bathymetry = GEBCOBathymetry(
    file_path=str(gebco_file),
    resolution=500.0,
)

ssp = LeroyCopernicusSoundSpeedProfile(
    temperature_file_path=str(cop_temp_file),
    salinity_file_path=str(cop_sal_file),
    reference_lat_deg=bathymetry.reference_lat_deg,
    reference_lon_deg=bathymetry.reference_lon_deg,
)

prop_step_m = 25.0
prop_azimuth_search_width = 10.0
prop_azimuth_resolution = 0.5
prop_elevation_range = (-25.0, 25.0)
prop_elevation_resolution = 0.6
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
# Geometry View: Trajectories over Bathymetry
# -------------------------------------------
#
# These plots use `plot_world` with bathymetry overlays to show the kinematic scene
# against the measured seabed model.

# %%
fig_world = plot_world(
    truths=target_ground_truths,
    platform=platform,
    bathymetry=bathymetry,
    figsize=(700, 550),
)

# %%
# Bathymetry and Sound Speed Viewer
# ---------------------------------
#
# The Dash viewer supports notebook display modes through `jupyter_mode`: `inline`,
# `tab`, `external`, or `jupyterlab`.
# Use `None` for standard server mode.

# %%
from bluepebble.plotter import launch_bathymetry_and_sound_speed_viewer

launch_bathymetry_ssp_viewer = False
viewer_host = "127.0.0.1"
viewer_port = 8050
viewer_jupyter_mode = None

if launch_bathymetry_ssp_viewer:
    launch_bathymetry_and_sound_speed_viewer(
        bathymetry=bathymetry,
        ssp=ssp,
        host=viewer_host,
        port=viewer_port,
        debug=False,
        jupyter_mode=viewer_jupyter_mode,
    )
else:
    print(
        "Set launch_bathymetry_ssp_viewer=True to start the dashboard. "
        f"Current jupyter_mode={viewer_jupyter_mode!r}."
    )

# %%
# Signal Model
# ------------
#
# Here the source and ambient signal models are defined. Each target receives a broadband
# ship signal model, and coloured ambient noise is added at the array.

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

signal_models = []
for target_ground_truth in target_ground_truths:
    target_metadata = next(iter(target_ground_truth)).metadata
    signal_models.append(
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

# %%
# Beamformer
# ----------
#
# This section sets the beamforming parameters and builds the steering calculator.
# An MVDR beamformer is used.

# %%
from scipy.signal import get_window

from bluepebble.sigproc import (
    DelayAndSumBeamformer,
    MinimumVarianceDistortionlessResponseBeamformer,
    SteeringCalculator,
)

beamformer_type = "MVDR"
beamformer_shading = None
beamformer_domain = "frequency"
steering_azimuths_rad = np.linspace(-np.pi, np.pi, 181)
fmin = 120.0
fmax = 250.0

shading = None
if beamformer_shading is not None:
    shading = get_window(beamformer_shading, platform.num_sensors)

if beamformer_type == "DAS":
    if beamformer_domain == "broadband_power":
        beamformer = DelayAndSumBeamformer(
            domain=beamformer_domain,
            sampling_rate_hz=sampling_rate_hz,
            fmin=fmin,
            fmax=fmax,
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
        fmin=fmin,
        fmax=fmax,
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
# Set up the detector pipeline, which is a CA-CFAR detector with local maximum
# clustering.

# %%
from bluepebble.detector import CACFARDetector, PassiveSonarDetector, PeakDetector
from bluepebble.simulator import ContinuousSTFTPassiveSonarArraySimulator

cfar_num_guard_cells = 2
cfar_num_training_cells = 5
cfar_threshold_factor = 1.75
cfar_mode = "wrap"
peak_distance = 3

simulator = ContinuousSTFTPassiveSonarArraySimulator(
    platform=platform,
    propagation_model=prop_model,
    signal_models=signal_models,
    noise_model=ambient_noise_model,
    beamformer=beamformer,
    steering_calculator=steering_calculator,
    ground_truth_paths=target_ground_truths,
    fade_in_ms=fade_in_ms,
)

cfar_detector = CACFARDetector(
    num_guard_cells=cfar_num_guard_cells,
    num_training_cells=cfar_num_training_cells,
    threshold_factor=cfar_threshold_factor,
    mode=cfar_mode,
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
# This cell runs the simulation and applies the detector pipeline.

# %%
all_detections = list(detector.detections_gen(progress_bar=False, total_timesteps=num_steps))
snr_map = detector.snr_history

timesteps = [start_time + i * time_interval for i in range(num_steps)]
steering_azimuths_deg = np.rad2deg(steering_azimuths_rad)

detections = [d for _, detection_set in all_detections for d in detection_set]
print(f"Total no. of detections: {len(detections)}")

# %%
# Results
# -------
#
# Here the beamformer SNR map is shown, along with the detections overlaid on the ground
# truth.

# %%
from plotly.subplots import make_subplots

from bluepebble.plotter import plot_btr

fig_results = make_subplots(
    rows=1,
    cols=2,
    shared_yaxes=True,
    subplot_titles=("SNR Map", "SNR Map w/ Detections"),
    horizontal_spacing=0.08,
)

plot_btr(
    data=snr_map,
    timesteps=timesteps,
    steering_azimuths=steering_azimuths_deg,
    fig=fig_results,
    row=1,
    col=1,
)
plot_btr(
    data=snr_map,
    detections=detections,
    truths=relative_bearing_ground_truths,
    timesteps=timesteps,
    steering_azimuths=steering_azimuths_deg,
    fig=fig_results,
    row=1,
    col=2,
)

fig_results.update_yaxes(title_text="", row=1, col=2)
fig_results.update_layout(
    width=1300,
    height=600,
    showlegend=False,
    margin=dict(r=90, t=90),
)

# %%
# Acknowledgement
# ~~~~~~~~~~~~~~~
#
# This example uses external environmental data derived from:
#
# - GEBCO Compilation Group, The GEBCO Grid (GEBCO_2024 Grid).
# - E.U. Copernicus Marine Service Information (https://doi.org/10.48670/moi-00016).
