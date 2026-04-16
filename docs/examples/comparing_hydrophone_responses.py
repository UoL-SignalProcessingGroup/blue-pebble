"""
==============================
Comparing Hydrophone Responses
==============================

This example compares how four different hydrophone frequency response configurations
affect the beamformed output and downstream detections when receiving the same acoustic
scene. The scenario is held fixed throughout: one target radiates three tonal components
at 30 Hz, 100 Hz, and 200 Hz, propagated to the array under a spherical spreading model.
Only the :class:`~bluepebble.sensor.HydrophoneResponse` attached to each array element
changes between the four runs.

The configurations demonstrate each :class:`~bluepebble.sensor.FrequencyResponse`
subclass provided by Blue Pebble:

1. **Flat** (default) — uniform unity response; all three tonals arrive at equal strength.
2. **First-order high-pass** — -3 dB at 80 Hz; attenuates the 30 Hz tonal.
3. **First-order low-pass** — -3 dB at 120 Hz; attenuates the 200 Hz tonal.
4. **Tabulated** — a resonant peak centred near 100 Hz; emphasises the mid tonal and
   suppresses both flanking components.

The first figure shows the magnitude of each transfer function across the simulation
band. The second figure presents a bearing-time record for each configuration in a
stacked layout so the differential tonal visibility is immediately apparent.
"""  # noqa: D205, D212, D400, D415

# %%
# Imports
# -------
#
# All dependencies are consolidated here for convenience.
from datetime import datetime, timedelta

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel,
    ConstantVelocity,
)
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState

import bluepebble
from bluepebble.detector import CACFARDetector, PassiveSonarDetector, PeakDetector
from bluepebble.models.environment import Constant
from bluepebble.models.propagation import SphericalAcousticPropagationModel
from bluepebble.platform import TowedArrayPlatform
from bluepebble.plotter import apply_shared_colourscale, plot_btr, plot_world
from bluepebble.sensor import (
    FirstOrderHighPassResponse,
    FirstOrderLowPassResponse,
    Hydrophone,
    HydrophoneResponse,
    LinearHydrophoneArray,
    TabulatedFrequencyResponse,
)
from bluepebble.signal.anthropogenic import SyntheticAnthropogenicSignal
from bluepebble.signal.random import ColouredNoiseSignal
from bluepebble.sigproc import DelayAndSumBeamformer, SteeringCalculator
from bluepebble.simulator import ContinuousSTFTPassiveSonarArraySimulator

# %%
# Simulation Parameters
# ---------------------
#
# A single shared clock drives all four runs. A 300-second simulation at 5-second
# intervals keeps runtime manageable while still producing a readable bearing-time record.

seed = 1234
bluepebble.set_seed(seed)
rng = bluepebble.get_rng()

sim_length_s = 300
sim_rate_s = 5.0

start_time = datetime(2026, 1, 1, 0, 0, 0)
time_interval = timedelta(seconds=sim_rate_s)
num_steps = int(sim_length_s / sim_rate_s)
total_duration_s = num_steps * time_interval.total_seconds()

# %%
# Hydrophone Response Configurations
# -----------------------------------
#
# Four :class:`~bluepebble.sensor.HydrophoneResponse` objects are defined here, one per
# comparison branch. Each wraps a different :class:`~bluepebble.sensor.FrequencyResponse`
# subclass (or uses the flat default). The scalar sensitivity is left at 0 dB throughout
# so that only the frequency shaping differs between configurations.

# 1. Flat response — identity; no frequency shaping applied.
response_flat = HydrophoneResponse()

# 2. First-order high-pass at 80 Hz.
#    The 30 Hz tonal lies well below the cutoff and will be strongly attenuated.
response_highpass = HydrophoneResponse(
    frequency_response=FirstOrderHighPassResponse(cutoff_hz=80.0),
)

# 3. First-order low-pass at 120 Hz.
#    The 200 Hz tonal lies above the cutoff and will roll off at -20 dB/decade.
response_lowpass = HydrophoneResponse(
    frequency_response=FirstOrderLowPassResponse(cutoff_hz=120.0),
)

# 4. Tabulated response with a resonant peak near 100 Hz.
#    The shape approximates a hydrophone with elevated mid-band sensitivity
#    that rolls off at both low and high frequencies.
_tab_frequencies_hz = np.array([0.0, 20.0, 50.0, 80.0, 100.0, 130.0, 160.0, 200.0, 250.0])
_tab_magnitude_db = np.array([-20.0, -12.0, -4.0, 0.0, 3.0, 0.0, -5.0, -12.0, -20.0])
response_tabulated = HydrophoneResponse(
    frequency_response=TabulatedFrequencyResponse(
        frequencies_hz=_tab_frequencies_hz,
        magnitude_db=_tab_magnitude_db,
    ),
)

configs = [
    ("Flat (default)", response_flat),
    ("High-pass  (fc = 80 Hz)", response_highpass),
    ("Low-pass  (fc = 120 Hz)", response_lowpass),
    ("Tabulated  (peak ≈ 100 Hz)", response_tabulated),
]

# %%
# Frequency Response Curves
# --------------------------
#
# The magnitude of each transfer function is plotted here. This figure is generated
# analytically — no simulation is required — and serves as the reference for
# interpreting the bearing-time records that follow.
#
# Vertical dashed lines mark the three target tonal frequencies so the expected
# attenuation for each configuration can be read directly from the plot.

_f_axis = np.linspace(0.5, 250.0, 1000)
_tonal_freqs_hz = [30.0, 100.0, 200.0]
_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

fig_response = go.Figure()

for (label, response), color in zip(configs, _colors, strict=True):
    H = response.transfer_function(_f_axis)
    mag_db = 20.0 * np.log10(np.abs(H) + 1e-12)
    fig_response.add_trace(
        go.Scatter(
            x=_f_axis,
            y=mag_db,
            name=label,
            line=dict(color=color, width=2),
        )
    )

for freq in _tonal_freqs_hz:
    fig_response.add_vline(
        x=freq,
        line=dict(color="grey", width=1, dash="dash"),
        annotation_text=f"{int(freq)} Hz",
        annotation_position="top",
    )

fig_response.update_layout(
    template="plotly_white",
    title="Hydrophone Frequency Response Magnitudes",
    xaxis=dict(title="Frequency (Hz)", range=[0, 250]),
    yaxis=dict(title="Magnitude (dB)", range=[-25, 10]),
    legend=dict(x=0.01, y=0.01, xanchor="left", yanchor="bottom"),
    height=400,
)

# %%
# Platform Setup
# --------------
#
# Each comparison uses an independent :class:`~bluepebble.platform.TowedArrayPlatform`
# built from the same trajectory parameters. The only difference between platforms is the
# :class:`~bluepebble.sensor.HydrophoneResponse` assigned to each array element.

straight_model = CombinedLinearGaussianTransitionModel(
    [ConstantVelocity(0.0), ConstantVelocity(0.0), ConstantVelocity(0.0)]
)

platform_start_vector = np.array([0.0, 2.0, 2000.0, 2.0, -5.0, 0.0])
platform_position_mapping = [0, 2, 4]
platform_velocity_mapping = [1, 3, 5]

num_sensors = 100
tow_cable_length_m = 200.0
sensor_spacing_m = 0.5
array_depth_m = -50.0


def _make_platform(hydrophone_response: HydrophoneResponse) -> TowedArrayPlatform:
    """Build and advance a :class:`~bluepebble.platform.TowedArrayPlatform`.

    Each call returns an independent platform that has been moved through the full
    simulation timeline. Using separate instances ensures each simulator has its own
    element state history and its own hydrophone transfer function.

    Parameters
    ----------
    hydrophone_response : HydrophoneResponse
        Electro-acoustic response model to attach to every array element.

    Returns
    -------
    TowedArrayPlatform
        Platform advanced to the final simulation timestamp.

    """
    elements = [Hydrophone(response=hydrophone_response) for _ in range(num_sensors)]
    sensor_array = LinearHydrophoneArray(elements=elements, element_spacing_m=sensor_spacing_m)
    initial_state = GroundTruthState(platform_start_vector.copy(), timestamp=start_time)
    platform = TowedArrayPlatform(
        states=[initial_state],
        position_mapping=platform_position_mapping,
        velocity_mapping=platform_velocity_mapping,
        transition_models=[straight_model],
        transition_times=[timedelta(seconds=total_duration_s)],
        sensor_array=sensor_array,
        cable_length_m=tow_cable_length_m,
        array_depth_m=array_depth_m,
    )
    for i in range(1, num_steps):
        platform.move(start_time + i * time_interval)
    return platform


platforms = [_make_platform(response) for _, response in configs]

# %%
# Ground Truth Setup and Generation
# ----------------------------------
#
# One target is generated here. It radiates three tonal components at 30 Hz, 100 Hz,
# and 200 Hz with equal source levels of 90 dB re 1 µPa. Spreading these three tonals
# across the simulation band ensures each hydrophone configuration produces a visibly
# distinct bearing-time record.

target_start_vector = np.array([6000.0, -4.0, 4000.0, -2.0, -5.0, 0.0])
target_position_mapping = [0, 2, 4]
target_velocity_mapping = [1, 3, 5]
target_transition_model = CombinedLinearGaussianTransitionModel(
    [ConstantVelocity(0.0), ConstantVelocity(0.0), ConstantVelocity(0.0)]
)

target_amplitudes_upa = np.full(3, 10 ** (90.0 / 20.0))
target_frequencies_hz = np.array([30.0, 100.0, 200.0])
target_phases_rad = rng.uniform(0, 2 * np.pi, 3)

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
        },
    )
]

for i in range(1, num_steps):
    new_time = start_time + i * time_interval
    new_state_vector = target_transition_model.function(
        target_states[-1], noise=False, time_interval=time_interval
    )
    target_states.append(
        GroundTruthState(
            new_state_vector,
            timestamp=new_time,
            metadata=target_states[-1].metadata,
        )
    )

target_ground_truth = GroundTruthPath(target_states)

# Compute relative bearing ground truth using the first platform (positions are identical).
reference_platform = platforms[0]
bearing_states = []
for target_state in target_ground_truth.states:
    positions = reference_platform.sensor_array.position_matrix_at(target_state.timestamp)
    ref_position = np.mean(positions, axis=1)
    target_pos = np.array(
        [target_state.state_vector[0], target_state.state_vector[2]], dtype=float
    )
    relative_pos = target_pos - ref_position[:2]
    bearing_rad = np.arctan2(relative_pos[1], relative_pos[0])
    bearing_states.append(
        GroundTruthState(state_vector=np.array([bearing_rad]), timestamp=target_state.timestamp)
    )

relative_bearing_ground_truth = GroundTruthPath(bearing_states)

# %%
# Geometry View: Ownship and Target Trajectories
# -----------------------------------------------
#
# This figure shows the kinematic scene. Because all four platforms follow identical
# trajectories, only one is plotted here. The target moves slowly across the field.

fig_world = plot_world(
    truths=[target_ground_truth],
    platform=reference_platform,
    figsize=(600, 600),
)

# %%
# Propagation Model
# -----------------
#
# Spherical spreading is used here. It provides realistic per-sensor transfer functions
# (including inter-element phase delays) without the computational overhead of ray tracing.
# Because the propagation model is the same for all four runs, any differences in the
# bearing-time records come solely from the hydrophone response.

ssp = Constant(speed=1500.0)

prop_model = SphericalAcousticPropagationModel(ssp=ssp)

# %%
# Signal and Noise Models
# -----------------------
#
# A single set of signal parameters is shared across all four runs. No broadband noise is
# added to the source signal so that the effect of each frequency response on the three
# tonals is as clear as possible. Ambient coloured noise is added at the array.

sampling_rate_hz = 500.0
frame_len = 500
hop_factor = 2
fade_in_ms = 500.0
ambient_amplitude_upa = 10 ** (45.0 / 20.0)
ambient_spectral_exponent = -1  # pink noise

ambient_noise_model = ColouredNoiseSignal(
    amplitude_upa=ambient_amplitude_upa,
    spectral_exponent=ambient_spectral_exponent,
    duration_s=time_interval.total_seconds(),
    sampling_rate_hz=sampling_rate_hz,
)


def _make_signal_model() -> SyntheticAnthropogenicSignal:
    """Create an independent signal model for one simulator instance."""
    return SyntheticAnthropogenicSignal(
        duration_s=total_duration_s,
        sampling_rate_hz=sampling_rate_hz,
        frame_len=frame_len,
        hop_factor=hop_factor,
        tonal_bandwidth_hz=1.0,
        noise_amplitude_upa=0.0,
        tonal_noise_is_constant=True,
        noise_is_constant=True,
    )


# %%
# Beamformer
# ----------
#
# A frequency-domain delay-and-sum beamformer is configured once and shared across all
# four detector pipelines. The steering grid spans the full azimuth range.

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
# Four simulators are created — one per hydrophone configuration. Each is paired with its
# own platform so that the correct hydrophone transfer function is applied during signal
# generation. A shared CA-CFAR + peak-selection detector chain is applied identically to
# all four outputs so that detection differences can be attributed to the hydrophone
# response alone.

cfar_num_guard_cells = 6
cfar_num_training_cells = 10
cfar_threshold_factor = 1.05
cfar_mode = "wrap"
peak_distance = 8


def _make_detector(simulator: ContinuousSTFTPassiveSonarArraySimulator) -> PassiveSonarDetector:
    """Create a PassiveSonarDetector with a CACFARDetector followed by a PeakDetector."""
    return PassiveSonarDetector(
        detection_chain=[
            CACFARDetector(
                num_guard_cells=cfar_num_guard_cells,
                num_training_cells=cfar_num_training_cells,
                threshold_factor=cfar_threshold_factor,
                mode=cfar_mode,
            ),
            PeakDetector(distance=peak_distance),
        ],
        sensor_data_gen=simulator.sensor_data_gen(),
        steering_azimuths_rad=steering_azimuths_rad,
    )


simulators = [
    ContinuousSTFTPassiveSonarArraySimulator(
        platform=platform,
        propagation_model=prop_model,
        signal_models=[_make_signal_model()],
        noise_model=ambient_noise_model,
        beamformer=beamformer,
        steering_calculator=steering_calculator,
        ground_truth_paths=[target_ground_truth],
        fade_in_ms=fade_in_ms,
    )
    for platform in platforms
]

detectors = [_make_detector(sim) for sim in simulators]

# %%
# Run Detection on Simulated Data
# --------------------------------
#
# All four detector pipelines are executed sequentially. The SNR history and flat
# detection lists are retained for plotting.

all_detections = []
snr_maps = []

for (label, _), detector in zip(configs, detectors, strict=True):
    raw = list(detector.detections_gen(progress_bar=False, total_timesteps=num_steps))
    snr_maps.append(detector.snr_history)
    flat = [d for _, detection_set in raw for d in detection_set]
    all_detections.append(flat)
    print(f"{label}: {len(flat)} detections")

timesteps = np.array([start_time + i * time_interval for i in range(num_steps)], dtype=object)
steering_azimuths_deg = np.rad2deg(steering_azimuths_rad)

# %%
# Results: Bearing-Time Records by Hydrophone Configuration
# ----------------------------------------------------------
#
# The four SNR maps are stacked vertically. Each row uses the same colour scale so that
# tonal brightness can be compared directly across configurations. The vertical layout
# makes it straightforward to trace how each filter reshapes the three-tonal signature.
#
# - **Flat**: all three tonals appear with equal strength.
# - **High-pass**: the 30 Hz tonal is attenuated or absent; 100 Hz and 200 Hz persist.
# - **Low-pass**: the 200 Hz tonal is attenuated; 30 Hz and 100 Hz persist.
# - **Tabulated**: the 100 Hz tonal is boosted; the flanking components are suppressed.

row_titles = [label for label, _ in configs]

fig_btr = make_subplots(
    rows=4,
    cols=1,
    shared_xaxes=True,
    shared_yaxes=True,
    row_titles=row_titles,
    vertical_spacing=0.06,
)

for row, (snr_map, detections) in enumerate(zip(snr_maps, all_detections, strict=True), start=1):
    plot_btr(
        data=snr_map,
        detections=detections,
        timesteps=timesteps,
        steering_azimuths=steering_azimuths_deg,
        fig=fig_btr,
        row=row,
        col=1,
    )

apply_shared_colourscale(
    fig_btr,
    colorbar=dict(
        title=dict(text="SNR (dB)", side="right"),
        x=1.02,
        y=0.5,
        yanchor="middle",
        len=1.0,
        thickness=24,
    ),
)

# Remove x-axis labels from all rows except the last.
for row in range(1, 4):
    fig_btr.update_xaxes(title_text="", row=row, col=1, showticklabels=False)

fig_btr.update_layout(
    template="plotly_white",
    autosize=True,
    height=900,
    showlegend=False,
    title="SNR Maps: Flat vs High-pass vs Low-pass vs Tabulated Hydrophone Response",
    margin=dict(r=100),
)
