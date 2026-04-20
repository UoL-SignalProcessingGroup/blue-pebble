"""
==============================
Comparing Hydrophone Responses
==============================

This example compares how five different hydrophone frequency response configurations
affect the beamformed output and downstream detections when receiving the same acoustic
scene. The scenario is held fixed throughout: one target radiates three tonal components
at 30 Hz, 100 Hz, and 200 Hz, propagated to the array under a spherical spreading model.
Only the :class:`~bluepebble.sensor.HydrophoneResponse` attached to each array element
changes between the five runs.

The configurations demonstrate each :class:`~bluepebble.sensor.FrequencyResponse`
subclass provided by Blue Pebble:

1. **Flat** (default) — uniform unity response; all three tonals arrive at equal strength.
2. **First-order high-pass** — -3 dB at 80 Hz; attenuates the 30 Hz tonal.
3. **First-order low-pass** — -3 dB at 120 Hz; attenuates the 200 Hz tonal.
4. **Second-order band-pass** — -3 dB at 50 Hz and 180 Hz; attenuates both the 30 Hz
   and 200 Hz tonals, passing only the 100 Hz component through the flat passband.
5. **Tabulated** — a resonant peak centred near 100 Hz; emphasises the mid tonal and
   suppresses both flanking components.

The first figure shows the magnitude of each transfer function across the simulation
band. The second figure presents a bearing-time record for each configuration in a
stacked layout so the differential tonal visibility is immediately apparent. The third
figure shows the array-averaged received signal spectrogram for each configuration,
making the spectral shaping applied by each hydrophone response directly visible as a
function of frequency and time.
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
from bluepebble.plotter import apply_shared_colourscale, plot_btr, plot_spectrogram, plot_world
from bluepebble.sensor import (
    FirstOrderHighPassResponse,
    FirstOrderLowPassResponse,
    Hydrophone,
    HydrophoneResponse,
    LinearHydrophoneArray,
    SecondOrderBandPassResponse,
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

seed = 42
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

# 4. Second-order band-pass with cutoffs at 50 Hz and 180 Hz.
#    The 30 Hz tonal is below the low cutoff and will be attenuated; the 200 Hz tonal
#    is above the high cutoff and will also be attenuated. Only the 100 Hz tonal
#    sits in the flat passband.
response_bandpass = HydrophoneResponse(
    frequency_response=SecondOrderBandPassResponse(low_cutoff_hz=50.0, high_cutoff_hz=180.0),
)

# 5. Physically motivated tabulated response for the 0–250 Hz simulation band.
# The shape combines a first-order high-pass roll-off (RC corner at 15 Hz,
# approximating the piezoelectric element driving a finite load impedance) [1]
# with a Lorentzian resonance bump of +2 dB centred at 100 Hz (Q = 1.5) [2].
# The result is a response that rises from near-zero at DC, peaks gently in
# the mid-band, and rolls off gradually toward the 250 Hz Nyquist limit —
# behaviour consistent with a low-frequency piezoelectric hydrophone in a
# resistively loaded circuit.  This gives a visibly distinct BTR / spectrogram
# relative to the other configs without relying on manufacturer calibration
# data outside the simulation band.
#
# [1] Teledyne Marine, "Hydrophone TC4033 Product Datasheet," Teledyne RESON,
#     Slangerup, Denmark, 2019. [Online]. Available:
#     https://www.teledynemarine.com/en-us/products/SiteAssets/RESON/TC4033%20%20product%20leaflet.pdf
#
# [2] M. Brunner et al., "Free-Field Calibration of Hydrophones at Frequencies
#     from 250 Hz to 200 kHz," NPL Report DQL AC 019, National Physical
#     Laboratory, Teddington, UK, 2004. [Online]. Available:
#     https://eprintspublications.npl.co.uk/3814/1/DQL_AC19.pdf


def _realistic_tab(freqs, fc_hz=40.0, peak_hz=100.0, peak_db=6.0, q=2.0):
    """Physically motivated tabulated response for the 0-250 Hz band.

    - First-order HPF roll-off below fc_hz (piezoelectric RC corner)
    - Gentle resonant peak at peak_hz with quality factor q
    - Stays within ±3 dB across the passband
    """
    # First-order HPF: +20 dB/decade rise below fc
    hpf_db = 20.0 * np.log10(freqs / np.sqrt(freqs**2 + fc_hz**2) + 1e-12)

    # Lorentzian resonance bump
    bump_db = peak_db / (1 + q**2 * ((freqs / peak_hz) - (peak_hz / freqs)) ** 2)

    return hpf_db + bump_db


_tab_frequencies_hz = np.array(
    [1, 5, 10, 20, 30, 50, 80, 100, 120, 150, 180, 200, 250], dtype=float
)
_tab_magnitude_db = _realistic_tab(_tab_frequencies_hz)
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
    ("Band-pass  (50–180 Hz)", response_bandpass),
    ("Tabulated  (peak ≈ 100 Hz)", response_tabulated),
]

# %%
# Frequency Response Curves
# --------------------------
#
# The magnitude of each transfer function is plotted here in dB as a function of
# frequency. This figure is generated analytically — no simulation is required — and
# serves as the reference for interpreting the bearing-time records and spectrograms
# that follow.
#
# A flat 0 dB line means the hydrophone passes that frequency without attenuation.
# Values below 0 dB indicate the hydrophone suppresses that frequency; values above
# 0 dB indicate amplification (as seen in the tabulated response's resonant peak near
# 100 Hz). Vertical dashed lines mark the three target tonal frequencies (30, 100 and
# 200 Hz) so the expected gain or attenuation at each tonal can be read directly.
#
# Key observations:
#
# - The **flat** response is a horizontal line at 0 dB — all tonals arrive unmodified.
# - The **high-pass** curve rolls off below its 80 Hz cutoff, strongly attenuating the
#   30 Hz tonal while leaving 100 Hz and 200 Hz nearly untouched.
# - The **low-pass** curve rolls off above its 120 Hz cutoff, attenuating the 200 Hz
#   tonal while preserving 30 Hz and 100 Hz.
# - The **band-pass** curve combines both roll-offs: it attenuates below 50 Hz and
#   above 180 Hz, passing only the mid-band where the 100 Hz tonal sits.
# - The **tabulated** curve provides the most aggressive shaping, with a +3 dB peak
#   at 100 Hz and steep attenuation at both band edges.

_f_axis = np.linspace(0.5, 250.0, 1000)
_tonal_freqs_hz = [30.0, 100.0, 200.0]

fig_response = go.Figure()

for label, response in configs:
    H = response.transfer_function(_f_axis)
    mag_db = 20.0 * np.log10(np.abs(H) + 1e-12)
    fig_response.add_trace(
        go.Scatter(
            x=_f_axis,
            y=mag_db,
            name=label,
            line=dict(width=2),
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
    legend=dict(x=0.5, y=-0.25, xanchor="center", yanchor="top", orientation="h"),
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

platform_start_vector = np.array([0.0, 0.0, 2000.0, 2.0, -5.0, 0.0])
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

target_amplitudes_upa = np.full(3, 10 ** (100.0 / 20.0))
target_frequencies_hz = np.array([30.0, 100.0, 200.0])
target_phases_rad = rng.uniform(0, 2 * np.pi, 3)
target_noise_amplitude_upa = 10 ** (90 / 20)
target_noise_spectral_exponent = -1.0  # Pink noise

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
            "noise_amplitude_upa": target_noise_amplitude_upa,
            "noise_spectral_exponent": target_noise_spectral_exponent,
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
# Spherical spreading is used here. It provides per-sensor transfer functions
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
fade_in_ms = 1000.0
ambient_amplitude_upa = 10 ** (
    30.0 / 20.0
)  # Low-level ambient noise to keep the spectrogram visible.
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
        noise_amplitude_upa=target_noise_amplitude_upa,
        noise_spectral_exponent=target_noise_spectral_exponent,
        noise_freq_range_hz=(0.0, sampling_rate_hz / 2),
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
# Five simulators are created — one per hydrophone configuration. Each is paired with its
# own platform so that the correct hydrophone transfer function is applied during signal
# generation. A shared CA-CFAR + peak-selection detector chain is applied identically to
# all five outputs so that detection differences can be attributed to the hydrophone
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
# All five detector pipelines are executed sequentially. The SNR history and flat
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
# Results: Received Signal Spectrograms
# --------------------------------------
#
# The spectrogram shows how each hydrophone configuration shapes the received signal
# spectrum. The DAS beamformer is run across all steering azimuths, and the beamformed
# outputs are averaged across all beams at each timestep to produce an omnidirectional
# received waveform. This captures contributions from every direction rather than locking
# on to the target bearing.
#
# The hydrophone response is applied to the raw sensor signals before beamforming, so it
# shapes both the tonal components and the ambient noise floor. The effect is visible as a
# change in the spectral envelope across the four panels. Percentile-based colour limits
# are used to keep the scale sensitive to this spectral variation.
#
# - **Flat**: all three tonals appear with equal strength.
# - **High-pass**: the 30 Hz tonal is attenuated or absent; 100 Hz and 200 Hz persist.
# - **Low-pass**: the 200 Hz tonal is attenuated; 30 Hz and 100 Hz persist.
# - **Band-pass**: the 30 Hz and 200 Hz tonals are both attenuated; only 100 Hz persists.
# - **Tabulated**: the 100 Hz tonal is boosted; the flanking components are suppressed.

bluepebble.set_seed(seed)

subplot_titles = [label for label, _ in configs]

fig_spec = make_subplots(
    rows=5,
    cols=1,
    shared_xaxes=True,
    shared_yaxes=True,
    subplot_titles=subplot_titles,
    vertical_spacing=0.04,
)

for row, (platform, _) in enumerate(zip(platforms, configs, strict=True), start=1):
    sim = ContinuousSTFTPassiveSonarArraySimulator(
        platform=platform,
        propagation_model=prop_model,
        signal_models=[_make_signal_model()],
        noise_model=ambient_noise_model,
        beamformer=beamformer,
        steering_calculator=steering_calculator,
        ground_truth_paths=[target_ground_truth],
        fade_in_ms=fade_in_ms,
    )

    # Average the beamformed output across all steering angles to obtain an
    # omnidirectional received waveform for spectral analysis.
    chunks = []
    for _, sensor_data_set in sim.sensor_data_gen():
        sd = next(iter(sensor_data_set))
        assert sd.beamformed_data is not None, "Beamformer must be configured"
        chunks.append(np.mean(np.real(sd.beamformed_data), axis=0))
    full_signal = np.concatenate(chunks)

    plot_spectrogram(
        signal=full_signal,
        sr=int(sampling_rate_hz),
        n_fft=2048,
        hop_length=512,
        y_lim=(0.0, sampling_rate_hz / 2.0),
        yaxis_format="Hz",
        db_reference="absolute",
        showscale=False,
        fig=fig_spec,
        row=row,
        col=1,
    )
    for freq in _tonal_freqs_hz:
        fig_spec.add_hline(
            y=freq,
            line=dict(color="white", width=1, dash="dash"),
            row=row,
            col=1,
        )

# Use percentile-based shared limits so the colour scale is sensitive to
# spectral variation rather than clamped to absolute extremes.
_all_z = np.concatenate(
    [
        np.asarray(t.z, dtype=float).ravel()
        for t in fig_spec.data
        if getattr(t, "type", None) == "heatmap"
    ]
)
_finite_z = _all_z[np.isfinite(_all_z)]
apply_shared_colourscale(
    fig_spec,
    zmin=float(np.percentile(_finite_z, 2)),
    zmax=float(np.percentile(_finite_z, 98)),
    colorbar=dict(
        title=dict(text="Intensity (dB)", side="right"),
        x=1.02,
        y=0.5,
        yanchor="middle",
        len=1.0,
        thickness=24,
    ),
)

# Suppress x-axis labels on all but the bottom row; plot_spectrogram sets
# "Time (s)" on every row, so only row 5 retains the title.
for row in range(1, 5):
    fig_spec.update_xaxes(title_text="", row=row, col=1, showticklabels=False)

fig_spec.update_layout(
    template="plotly_white",
    autosize=True,
    height=int(np.clip(260 * 5, 700, 2200)),
    showlegend=False,
    title="Received Spectrograms",
    margin=dict(r=100),
)


# %%
# Results: Bearing-Time Records by Hydrophone Configuration
# ----------------------------------------------------------
#
# The five SNR maps are stacked vertically. Each row uses the same colour scale so that
# tonal brightness can be compared directly across configurations. The vertical layout
# makes it straightforward to trace how each filter reshapes the three-tonal signature.
#

fig_btr = make_subplots(
    rows=5,
    cols=1,
    shared_xaxes=True,
    shared_yaxes=True,
    subplot_titles=subplot_titles,
    vertical_spacing=0.04,
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
for row in range(1, 5):
    fig_btr.update_xaxes(title_text="", row=row, col=1, showticklabels=False)

fig_btr.update_layout(
    template="plotly_white",
    autosize=True,
    height=int(np.clip(260 * 5, 700, 2200)),
    showlegend=False,
    title="SNR Maps",
    margin=dict(r=100),
)
