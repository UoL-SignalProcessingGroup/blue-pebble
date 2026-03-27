"""
====================
Comparing Simulators
====================

This example compares five simulator backends on a 300-second passive sonar
scenario containing two vessels: a stationary tow ship carrying a 3-element
hydrophone array and a single target approaching from approximately 1.2 km.
Both a synthetic ship signal (tonal comb + coloured noise) and a real
hydrophone recording (SANCTSOUND CI05) are used as sources.

For each simulator mode the received signal at the centre array element is
plotted as a spectrogram, giving a 5 x 2 comparison table that lets you see
how continuous-processing artefacts differ across backends.

Simulator modes compared:

- **STFT Interp**: STFT-domain propagation with interpolation between transfer-function updates.
- **WOLA Interp**: Weighted overlap-add reconstruction with interpolated frame-to-frame
  propagation.
- **COLA**: Constant overlap-add STFT processing for stable frame stitching.
- **Fractional Delay**: Time-domain model using sub-sample delay alignment.
- **Discrete**: Per-step discrete simulation baseline without overlap-add processing.
"""  # noqa: D205, D212, D400, D415

# %%
# Imports
# -------
#
# All dependencies are consolidated here for convenience.

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from plotly.subplots import make_subplots
from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel,
    ConstantVelocity,
)
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState

from bluepebble.models.environment import Constant
from bluepebble.models.propagation import CylindricalAcousticPropagationModel
from bluepebble.platform import TowedArrayPlatform
from bluepebble.plotter import plot_spectrogram, plot_world
from bluepebble.signal.anthropogenic import (
    RecordedAnthropogenicSignal,
    SyntheticAnthropogenicSignal,
)
from bluepebble.simulator import (
    ContinuousFractionalDelayPassiveSonarArraySimulator,
    ContinuousSTFTPassiveSonarArraySimulator,
    DiscretePassiveSonarArraySimulator,
)

# %%
# Simulation Timing and Reproducibility
# --------------------------------------
#
# A fixed random seed ensures the tonal phases and noise realisations are identical on every run,
# making the spectrograms fully reproducible.

np.random.seed(1999)

sim_rate_s = 2.0
sim_length_s = 300.0
sim_start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
sim_time_interval = timedelta(seconds=sim_rate_s)
sim_num_steps = int(sim_length_s / sim_rate_s)

total_duration_s = sim_num_steps * sim_time_interval.total_seconds()

# %%
# Platform and Target
# -------------------
#
# The tow ship is stationary with a 3-element hydrophone array. The array is separated by a 10 m
# cable with 1 m sensor spacing at 10 m depth. The target vessel starts approximately 1.2 km away
# and moves slowly from the north-west.

platform_start_vector = np.array([0, 0, 0, 0, -10.0, 0])
platform_position_mapping = [0, 2, 4]
platform_velocity_mapping = [1, 3, 5]
platform_transition_model = CombinedLinearGaussianTransitionModel(
    [ConstantVelocity(0), ConstantVelocity(0), ConstantVelocity(0)]
)

num_sensors = 3
tow_cable_length_m = 10.0
sensor_spacing_m = 1.0
array_depth_m = -10.0

sensor_to_analyse = num_sensors // 2

initial_state = GroundTruthState(platform_start_vector, timestamp=sim_start_time)

platform = TowedArrayPlatform(
    states=[initial_state],
    position_mapping=platform_position_mapping,
    velocity_mapping=platform_velocity_mapping,
    transition_models=[platform_transition_model],
    transition_times=[timedelta(seconds=total_duration_s)],
    num_sensors=num_sensors,
    cable_length_m=tow_cable_length_m,
    sensor_spacing_m=sensor_spacing_m,
    array_depth_m=array_depth_m,
)

for i in range(1, sim_num_steps):
    new_time = sim_start_time + i * sim_time_interval
    platform.move(new_time)

target_states = [
    GroundTruthState(
        np.array([-1000, 0, 750, -5.0, -10.0, 0]),
        timestamp=sim_start_time,
        metadata={
            "amplitudes_upa": 10 ** (np.array([100.0, 90.0, 100.0, 85.0]) / 20),
            "frequencies_hz": np.array([20.0, 140.0, 200.0, 500.0]),
            "phases_rad": np.random.uniform(0, 2 * np.pi, 4),
            "position_mapping": [0, 2, 4],
            "velocity_mapping": [1, 3, 5],
        },
    )
]

transition_model = CombinedLinearGaussianTransitionModel(
    [ConstantVelocity(0.001), ConstantVelocity(0.001), ConstantVelocity(0)]
)
for i in range(1, sim_num_steps):
    new_time = sim_start_time + i * sim_time_interval
    time_interval = new_time - target_states[-1].timestamp
    new_state_vector = transition_model.function(
        target_states[-1], noise=False, time_interval=time_interval
    )
    new_state = GroundTruthState(
        new_state_vector,
        timestamp=new_time,
        metadata=target_states[-1].metadata,
    )
    target_states.append(new_state)

target_ground_truth = GroundTruthPath(target_states)

fig_world = plot_world(truths=[target_ground_truth], platform=platform)
fig_world = fig_world.update_layout(title="World Picture")

# %%
# Propagation Model
# -----------------
#
# :class:`~.CylindricalAcousticPropagationModel` combines cylindrical spreading
# (:math:`10 \log_{10} r`) with a constant absorption term. It is fast, analytical, and supports
# ``propagate_spectrum()``, which is required by the continuous simulator backends.  Swap it for
# :class:`~.rtrsAcousticPropagationModel` if you need more realistic ray-path geometry. The rest of
# the example is unchanged.

ssp = Constant(speed=1500.0)
propagation_model = CylindricalAcousticPropagationModel(ssp=ssp, attenuation_factor=0.5)

# %%
# Source Signal Models
# --------------------
#
# Two source types are compared side-by-side throughout the example:
#
# * **Synthetic** - a tonal comb at 20, 140, 200 and 500 Hz mixed with pink noise, representing a
#   generic vessel signature.
# * **Measured** - a 30-second segment from the SANCTSOUND CI05 hydrophone recording of a large
#   ship, tiled to match the full simulation duration. The file is included in
#   ``docs/examples/measured_data/``; the path search below works both in the gallery build (where
#   ``__file__`` is set) and in interactive use from any working directory.

sampling_rate_hz = 3000.0
frame_len = 500
hop_factor = 4
fade_in_ms = 100.0
fade_out_ms = 100.0
tonal_bandwidth_hz = 10.0
noise_amplitude_upa = 10 ** (80 / 20)
noise_spectral_exponent = -1.0

wav_name = "SanctSound_CI05_03_largeship_20190925T135956Z.wav"

_data_dir_candidates: list[Path] = []
if "__file__" in globals():
    _data_dir_candidates.append(Path(__file__).resolve().parent / "measured_data")

_data_dir_candidates.extend(
    [
        Path.cwd() / "measured_data",
        Path.cwd() / "docs" / "examples" / "measured_data",
    ]
)

measured_wav_path = next(
    (
        candidate / wav_name
        for candidate in _data_dir_candidates
        if (candidate / wav_name).exists()
    ),
    None,
)

if measured_wav_path is None:
    raise FileNotFoundError(
        "Measured WAV file not found. Checked: "
        + ", ".join(str(candidate / wav_name) for candidate in _data_dir_candidates)
    )


def _make_synthetic_signal_model():
    return SyntheticAnthropogenicSignal(
        duration_s=total_duration_s,
        sampling_rate_hz=sampling_rate_hz,
        frame_len=frame_len,
        hop_factor=hop_factor,
        tonal_bandwidth_hz=tonal_bandwidth_hz,
        noise_amplitude_upa=noise_amplitude_upa,
        noise_spectral_exponent=noise_spectral_exponent,
        noise_freq_range_hz=(0.0, sampling_rate_hz / 2),
        tonal_noise_is_constant=True,
        noise_is_constant=True,
    )


def _make_measured_signal_model():
    return RecordedAnthropogenicSignal(
        duration_s=total_duration_s,
        sampling_rate_hz=sampling_rate_hz,
        frame_len=frame_len,
        hop_factor=hop_factor,
        wav_path=str(measured_wav_path),
        segment_start_s=0.0,
        segment_duration_s=30.0,
        duration_match_mode="tile",
        level_db_re_1upa=85.0,
    )


synthetic_signal_model = _make_synthetic_signal_model()
measured_signal_model = _make_measured_signal_model()


# %%
# Source Spectrograms
# -------------------
#
# Before running the full simulation, inspect the source signals in isolation. This confirms the
# tonal structure is present in the synthetic signal and that the recorded signal has comparable
# bandwidth. The two subplots share the same frequency axis for easy comparison. Colour scaling
# uses the 5th–95th percentile range of each panel to avoid outliers compressing the dynamic
# range.

_sr = int(sampling_rate_hz)
_n_fft, _hop = 500, 250
freq_hi = sampling_rate_hz / 2

source_signal_synthetic = synthetic_signal_model.get_source_waveform(target_states[0])
source_signal_measured = measured_signal_model.get_source_waveform(target_states[0])

synthetic_source_real = np.real(source_signal_synthetic)
measured_source_real = np.real(source_signal_measured)

fig_source_spec = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.10,
    subplot_titles=["Synthetic Source Signal", "Measured Source Signal"],
)

for row, src in enumerate((synthetic_source_real, measured_source_real), start=1):
    fig_source_spec = plot_spectrogram(
        signal=src,
        row=row,
        sr=_sr,
        n_fft=_n_fft,
        hop_length=_hop,
        y_lim=(0, freq_hi),
        yaxis_format="Hz",
        fig=fig_source_spec,
        col=1,
        analysis_mode="psd",
        db_reference="absolute",
        z_percentiles=(5.0, 95.0),
        colorbar_title="dB re 1 uPa^2/Hz",
        hovertemplate=(
            "Time: %{x:.2f} s<br>"
            "Frequency: %{y:.1f} Hz<br>"
            "PSD: %{z:.2f} dB re 1 uPa^2/Hz"
            "<extra></extra>"
        ),
    )
    if row > 1:
        fig_source_spec.data[-1].showscale = False
        fig_source_spec.data[-1].colorbar = None

fig_source_spec = (
    fig_source_spec.update_yaxes(title_text="Frequency (Hz)", range=[0, freq_hi])
    .update_xaxes(title_text=None, row=1, col=1)
    .update_xaxes(title_text="Time (s)", row=2, col=1)
    .update_layout(template="plotly_white", height=700, title="Source Signal Spectrograms")
)

# %%
# Simulator Comparison Table
# --------------------------
#
# Each row corresponds to a simulator backend; each column to a source type. All five modes should
# reproduce the same tonal lines. Differences appear at segment boundaries (frame-stitching
# artefacts) and in inter-frame phase continuity. The Discrete mode processes each timestep
# independently, so it shows the starkest inter-frame transitions, while the COLA and WOLA modes
# are designed to minimise them. Colour scaling uses the 5th–95th percentile range so that
# transient artefacts do not dominate the colour axis.

sim_configs = [
    {"label": "STFT Interp", "kind": "stft", "mode": "stft_interp"},
    {"label": "WOLA Interp", "kind": "stft", "mode": "wola_interp"},
    {"label": "COLA", "kind": "stft", "mode": "cola"},
    {"label": "Fractional Delay", "kind": "fractional", "mode": None},
    {"label": "Discrete", "kind": "discrete", "mode": None},
]


def build_simulator(config, signal_model):
    """Build a simulator based on the config dict and signal model."""
    common_kwargs = {
        "platform": platform,
        "propagation_model": propagation_model,
        "signal_models": [signal_model],
        "noise_model": None,
        "beamformer": None,
        "steering_calculator": None,
        "ground_truth_paths": [target_ground_truth],
    }

    if config["kind"] == "stft":
        return ContinuousSTFTPassiveSonarArraySimulator(
            mode=config["mode"],
            fade_in_ms=fade_in_ms,
            fade_out_ms=fade_out_ms,
            **common_kwargs,
        )

    if config["kind"] == "fractional":
        return ContinuousFractionalDelayPassiveSonarArraySimulator(
            fade_in_ms=fade_in_ms,
            fade_out_ms=fade_out_ms,
            **common_kwargs,
        )

    if config["kind"] == "discrete":
        return DiscretePassiveSonarArraySimulator(**common_kwargs)

    raise ValueError(f"Unsupported simulator config: {config}")


def run_continuous_simulation(simulator_obj):
    """Run a simulator and return the received signal for the selected sensor."""
    all_sensor_signals = []
    for _, sensor_data_set in simulator_obj.sensor_data_gen():
        sensor_data = next(iter(sensor_data_set))
        all_sensor_signals.append(sensor_data.raw_signals)

    all_sensor_signals_array = np.concatenate(all_sensor_signals, axis=1)
    return all_sensor_signals_array[sensor_to_analyse, :]


comparison_results = []
for config in sim_configs:
    # Fresh model instances per config — compute_stft() is one-shot per instance.
    synthetic_sim = build_simulator(config, _make_synthetic_signal_model())
    measured_sim = build_simulator(config, _make_measured_signal_model())

    synthetic_received = run_continuous_simulation(synthetic_sim)
    measured_received = run_continuous_simulation(measured_sim)

    comparison_results.append(
        {
            "label": config["label"],
            "synthetic": synthetic_received,
            "measured": measured_received,
        }
    )

n_rows = len(comparison_results)

row_titles = [result["label"] for result in comparison_results]
column_titles = ["Synthetic", "Measured"]

fig_grid = make_subplots(
    rows=n_rows,
    cols=2,
    shared_xaxes=False,
    shared_yaxes=False,
    vertical_spacing=0.03,
    horizontal_spacing=0.08,
    row_titles=row_titles,
    column_titles=column_titles,
)

for row_idx, result in enumerate(comparison_results, start=1):
    for col_idx, key in enumerate(["synthetic", "measured"], start=1):
        fig_grid = plot_spectrogram(
            signal=result[key],
            sr=int(sampling_rate_hz),
            n_fft=_n_fft,
            hop_length=_hop,
            y_lim=(0, sampling_rate_hz / 2),
            yaxis_format="Hz",
            fig=fig_grid,
            row=row_idx,
            col=col_idx,
            analysis_mode="psd",
            db_reference="absolute",
            z_percentiles=(5.0, 95.0),
            showscale=(row_idx == 1 and col_idx == 2),
            colorbar_title="dB re 1 uPa^2/Hz",
            hovertemplate=(
                "Time: %{x:.2f} s<br>"
                "Frequency: %{y:.1f} Hz<br>"
                "PSD: %{z:.2f} dB re 1 uPa^2/Hz"
                "<extra></extra>"
            ),
        )

        fig_grid = fig_grid.update_xaxes(
            title_text="Time (s)" if row_idx == n_rows else None,
            showticklabels=(row_idx == n_rows),
            row=row_idx,
            col=col_idx,
        )
        fig_grid = fig_grid.update_yaxes(
            title_text="Frequency (Hz)" if col_idx == 1 else None,
            range=[0, sampling_rate_hz / 2],
            showticklabels=(col_idx == 1),
            row=row_idx,
            col=col_idx,
        )

fig_grid = fig_grid.update_layout(
    template="plotly_white",
    autosize=True,
    height=int(np.clip(260 * n_rows, 700, 2200)),
    title="Received Signal Spectrogram Table by Simulator Mode",
)

# %%
# Key Takeaways
# -------------
#
# * **All five modes preserve similar broad tonal content, but differ in artefact texture and
#   continuity**; COLA/WOLA appear smoother, while Fractional Delay/Discrete show more per-frame
#   discontinuity.
# * **Swap the propagation model** for :class:`~.rtrsAcousticPropagationModel` for more realistic
#   ray-path geometry and multipath structure.
# * **Adjust** ``num_sensors`` and ``sensor_to_analyse`` to explore multi-element effects
#   such as beam steering or spatial filtering.
# * **Remove the data dependency** by replacing :class:`~.RecordedAnthropogenicSignal` with a
#   second :class:`~.SyntheticAnthropogenicSignal` configured with different tonal frequencies or
#   noise colour.
