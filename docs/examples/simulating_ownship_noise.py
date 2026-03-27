"""
========================
Simulating Ownship Noise
========================

This example compares the same passive-sonar scenario in two conditions:
ambient noise only, and ambient noise with ownship self-noise added. Holding
all other pipeline settings fixed isolates the effect of ownship interference
on beamformed output and downstream detection counts.

The comparison is useful when tuning detector operating points in realistic
pipelines, where self-noise can produce bearing-dependent clutter that is not
present in ambient-only simulations.
"""  # noqa: D205, D212, D400, D415

# %%
# Imports
# -------
#
# All dependencies are consolidated here for convenience.

from datetime import datetime, timedelta

import numpy as np
from plotly.subplots import make_subplots
from scipy.signal import get_window
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
from bluepebble.plotter import apply_shared_colourscale, plot_btr, plot_world
from bluepebble.signal.anthropogenic import SyntheticAnthropogenicSignal
from bluepebble.signal.random import ColouredNoiseSignal
from bluepebble.sigproc import (
    DelayAndSumBeamformer,
    MinimumVarianceDistortionlessResponseBeamformer,
    SteeringCalculator,
)
from bluepebble.simulator import ContinuousSTFTPassiveSonarArraySimulator

# %%
# Simulation Timing and Reproducibility
# -------------------------------------
#
# A fixed seed and shared timeline keep stochastic source content reproducible
# and ensure both comparison branches run over exactly the same timestamps.

seed = 2000
np.random.seed(seed)

sim_length_s = 900
sim_rate_s = 5.0

start_time = datetime(2026, 1, 1, 0, 0, 0)
time_interval = timedelta(seconds=sim_rate_s)
num_steps = int(sim_length_s / sim_rate_s)
total_duration_s = num_steps * time_interval.total_seconds()
timesteps = np.array([start_time + i * time_interval for i in range(num_steps)], dtype=object)

# %%
# Platform Setup and Generation
# -----------------------------
#
# The platform follows a straight-turn-straight path with a towed linear array.

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

platform_initial_state = GroundTruthState(platform_start_vector, timestamp=start_time)
platform = TowedArrayPlatform(
    states=[platform_initial_state],
    position_mapping=platform_position_mapping,
    velocity_mapping=platform_velocity_mapping,
    transition_models=platform_transition_models,
    transition_times=platform_transition_times,
    num_sensors=num_sensors,
    cable_length_m=tow_cable_length_m,
    sensor_spacing_m=sensor_spacing_m,
    array_depth_m=array_depth_m,
)

for timestamp in timesteps[1:]:
    platform.move(timestamp)

# %%
# Ownship Self-Noise Ground Truth
# -------------------------------
# A second truth path is built from the platform states to represent ownship self-noise as a
# moving acoustic source. Self-noise in Blue Pebble is effectively an additional source with
# time-varying position and source metadata, so this path allows the same signal model to be
# applied to both targets and ownship noise under the same simulator framework.
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
# Three targets are created with reproducible randomised source metadata. Their
# Cartesian trajectories are converted to relative bearing truths for detector
# interpretation, then visualised to verify the simulation geometry. PassiveSonarDetector reports
# bearing-state detections, so truth needs to be in the same space for any downstream comparison.


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

target_ground_truths: list[GroundTruthPath] = []
relative_bearing_ground_truths: list[GroundTruthPath] = []

for target_start_vector in target_start_vectors:
    target_amplitudes_upa = 10 ** (np.random.uniform(87, 102, 4) / 20)
    target_frequencies_hz = np.random.uniform(50.0, 200.0, 4)
    target_phases_rad = np.random.uniform(0, 2 * np.pi, 4)
    target_tonal_bandwidth_hz = np.random.uniform(0.5, 2.0)
    target_noise_amplitude_upa = 10 ** (np.random.uniform(65, 85) / 20)

    target_metadata = {
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
    }

    target_states = [
        GroundTruthState(
            target_start_vector,
            timestamp=start_time,
            metadata=target_metadata,
        )
    ]

    for timestamp in timesteps[1:]:
        dt = timestamp - target_states[-1].timestamp
        new_state_vector = target_transition_model.function(
            target_states[-1], noise=False, time_interval=dt
        )
        target_states.append(
            GroundTruthState(
                new_state_vector,
                timestamp=timestamp,
                metadata=target_metadata,
            )
        )

    target_ground_truth = GroundTruthPath(target_states)
    target_ground_truths.append(target_ground_truth)

    bearing_states = []
    for target_state in target_ground_truth.states:
        platform_state = platform.get_platform_state_at(target_state.timestamp)
        assert platform_state is not None
        ref_sensor_position = np.mean(platform_state.array.state_vector, axis=1)

        target_xy = np.array([target_state.state_vector[0], target_state.state_vector[2]])
        relative_position = target_xy - ref_sensor_position[:2]
        bearing_rad = np.arctan2(relative_position[1], relative_position[0])

        bearing_states.append(
            GroundTruthState(np.array([bearing_rad]), timestamp=target_state.timestamp)
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
# Propagation Model
# -----------------
#
# The same environment and RTRS propagation settings are used in both branches,
# so output differences are attributable to ownship self-noise rather than
# environmental changes.

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
    water_density_g_cm3=1.0,
    bottom_model={
        "model": "elastic",
        "compressional_speed_m_s": 1700.0,
        "shear_speed_m_s": 400.0,
        "density_g_cm3": 1.6,
        "compressional_attenuation_db_per_wavelength": 0.2,
        "shear_attenuation_db_per_wavelength": 0.3,
    },
    store_ray_paths=False,
    integration_method="rk2",
)

# %%
# Acoustic Environment and Signal Models
# --------------------------------------
#
# Ambient coloured noise is fixed at the array and source models are created via factories. Fresh
# signal instances are required because these models are one-shot, so re-use would corrupt the
# second run.

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


def _make_signal_models() -> list[SyntheticAnthropogenicSignal]:
    models: list[SyntheticAnthropogenicSignal] = []
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


def _make_self_noise_model() -> SyntheticAnthropogenicSignal:
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
# A single beamformer configuration is shared between the two simulation branches so source/noise
# differences are compared under identical processing.

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
# Two simulators are created from the same scenario: one without ownship
# self-noise and one with ownship self-noise appended as an additional source.
#
# Keeping propagation, beamformer, and detector settings identical between the two simulators means
# any difference in output is solely attributable to the ownship source term.

cfar_num_guard_cells = 6
cfar_num_training_cells = 10
cfar_threshold_factor = 1.05
peak_distance = 8


def _make_detector(simulator: ContinuousSTFTPassiveSonarArraySimulator) -> PassiveSonarDetector:
    cfar_detector = CACFARDetector(
        num_guard_cells=cfar_num_guard_cells,
        num_training_cells=cfar_num_training_cells,
        threshold_factor=cfar_threshold_factor,
        mode="wrap",
    )
    peak_detector = PeakDetector(distance=peak_distance)

    return PassiveSonarDetector(
        detection_chain=[cfar_detector, peak_detector],
        sensor_data_gen=simulator.sensor_data_gen(),
        steering_azimuths_rad=steering_azimuths_rad,
    )


simulator_without_ownship_noise = ContinuousSTFTPassiveSonarArraySimulator(
    platform=platform,
    propagation_model=propagation_model,
    signal_models=_make_signal_models(),
    noise_model=ambient_noise_model,
    beamformer=beamformer,
    steering_calculator=steering_calculator,
    ground_truth_paths=target_ground_truths,
    fade_in_ms=fade_in_ms,
)

simulator_with_ownship_noise = ContinuousSTFTPassiveSonarArraySimulator(
    platform=platform,
    propagation_model=propagation_model,
    signal_models=_make_signal_models() + [_make_self_noise_model()],
    noise_model=ambient_noise_model,
    beamformer=beamformer,
    steering_calculator=steering_calculator,
    ground_truth_paths=target_ground_truths + [self_noise_ground_truth],
    fade_in_ms=fade_in_ms,
)

detector_without_ownship_noise = _make_detector(simulator_without_ownship_noise)
detector_with_ownship_noise = _make_detector(simulator_with_ownship_noise)

# %%
# Run Detection on Simulated Data
# -------------------------------
#
# Both detector pipelines are executed and their SNR histories are retained for
# side-by-side visual comparison.

all_detections_without_ownship_noise = list(
    detector_without_ownship_noise.detections_gen(progress_bar=False, total_timesteps=num_steps)
)
snr_map_without_ownship_noise = detector_without_ownship_noise.snr_history

all_detections_with_ownship_noise = list(
    detector_with_ownship_noise.detections_gen(progress_bar=False, total_timesteps=num_steps)
)
snr_map_with_ownship_noise = detector_with_ownship_noise.snr_history

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
# A two-row comparison grid highlights how ownship self-noise changes the raw
# SNR field and corresponding detection overlays under the same detector chain.
# In the ambient-only row the three target tracks are cleanly resolved, with
# detections following each line closely. In the ownship-noise row the same
# tracks remain present but a wideband elevation appears concentrated around
# the heading bearing — the direction from the array towards the towing vessel.
# Unlike a distant contact, which produces a narrow bearing track, ownship
# self-noise is broadband and spreads across a wide angular region centred on
# that bearing. The elevated power in this region drives additional detections,
# visible as scattered points near the heading in the right column.

n_rows = 2
row_titles = ["Without Ownship Noise", "With Ownship Noise"]
col_titles = ["SNR Map", "SNR Map w/ Detections"]
fig_results = make_subplots(
    rows=n_rows,
    cols=2,
    shared_xaxes=True,
    shared_yaxes=True,
    row_titles=row_titles,
    column_titles=col_titles,
)

result_panels = [
    (1, snr_map_without_ownship_noise, detections_without_ownship_noise),
    (2, snr_map_with_ownship_noise, detections_with_ownship_noise),
]

for row, snr_map, detections in result_panels:
    plot_btr(
        data=snr_map,
        timesteps=timesteps,
        steering_azimuths=steering_azimuths_deg,
        fig=fig_results,
        row=row,
        col=1,
    )
    plot_btr(
        data=snr_map,
        detections=detections,
        timesteps=timesteps,
        steering_azimuths=steering_azimuths_deg,
        fig=fig_results,
        row=row,
        col=2,
    )

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

for col in [1, 2]:
    fig_results.update_xaxes(title_text="", showticklabels=False, row=1, col=col)
for row in [1, 2]:
    fig_results.update_yaxes(title_text="", showticklabels=False, row=row, col=2)

fig_results.update_layout(
    title="Bearing-Time Comparison: Without vs With Ownship Self-Noise",
    template="plotly_white",
    autosize=True,
    width=None,
    height=int(np.clip(260 * n_rows, 700, 2200)),
    showlegend=False,
    margin=dict(r=120, t=120),
)

# %%
# Key Takeaways
# -------------
#
# * **Wideband heading clutter** - ownship self-noise appears as a broad
#   elevation centred on the heading angle. Its broadband character spreads
#   power across a wide angular region rather than the thin line a distant
#   tonal source would produce.
# * **Target tracks survive** - the three target lines remain visible in both
#   rows; ownship noise raises the noise floor but does not mask the targets at
#   these source levels.
# * **Increased false alarms** - the elevated background in the ownship-noise
#   row causes the CACFAR detector to produce additional detections during the
#   banding intervals, visible as inter-track scatter in the right column.
# * **Fair comparison** - keeping propagation, beamformer, and detector settings
#   identical between runs means any difference is solely attributable to the
#   ownship source term.
# * **Tow cable as mitigation** - increasing ``tow_cable_length_m`` separates
#   the array from the ownship source, which the propagation model converts into
#   additional transmission loss. A longer cable therefore reduces the self-noise
#   level at the array and narrows the banding. This mirrors the real-world
#   practice of deploying longer cables to improve self-noise rejection.
# * **Extension path** - increase ownship amplitude or reduce the CACFAR
#   threshold factor to find the operating point at which self-noise begins to
#   mask true targets.
