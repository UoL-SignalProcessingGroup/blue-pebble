"""Signal and Propagation Comparison Example.

This example compares one passive-sonar scenario across multiple signal models and
propagation models. It is designed to show how source-model assumptions and propagation
fidelity interact, rather than treating signal generation and propagation as independent
choices.

**Background**

- In underwater acoustics, both the emitted signal structure and the chosen propagation
  model can materially change what arrives at the array.
- A tonal source paired with a simple spreading law can produce very different received
  behaviour from a richer source paired with RTRS or Bellhop-style propagation.
- Comparative examples are useful when deciding which modelling complexity is justified
  for a study or demonstration.

**Key Concepts**

- Controlled comparison across signal-model and propagation-model combinations.
- Sensitivity of received waveform, spectrogram, and envelope structure to modelling
  assumptions.
- Interpreting trade-offs between simple and higher-fidelity propagation backends.
"""
# sphinx_gallery_skip_execution = True

# %% [markdown]
# ## Simulation Parameters
#
# This section sets only reproducibility and simulation timing:
# random seed, start time, timestep interval, and number of steps.

# %%
from datetime import datetime, timedelta

import numpy as np

seed = 1999
np.random.seed(seed)

SIM_RATE = 2.0
start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
time_interval = timedelta(seconds=SIM_RATE)
num_steps = 10

total_duration_s = num_steps * time_interval.total_seconds()
print(f"Total simulation duration: {total_duration_s} seconds")

# %% [markdown]
# ## Platform Setup and Generation
#
# This section defines the platform motion model and array geometry
# (start state, mappings, transition model, cable length, spacing, depth, sensor count),
# then propagates platform states over all timesteps.

# %%
from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel,
    ConstantVelocity,
)
from stonesoup.types.groundtruth import GroundTruthState

from bluepebble.platform import TowedArrayPlatform

platform_start_vector = np.array([0, 0, 0, 0, -10.0, 0])
platform_position_mapping = [0, 2, 4]
platform_velocity_mapping = [1, 3, 5]
platform_transition_model = CombinedLinearGaussianTransitionModel(
    [ConstantVelocity(0), ConstantVelocity(0), ConstantVelocity(0)]
)

tow_cable_length_m = 800.0
sensor_spacing_m = 1.0
array_depth_m = -200.0
num_sensors = 256

initial_state = GroundTruthState(platform_start_vector, timestamp=start_time)
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

for i in range(1, num_steps):
    new_time = start_time + i * time_interval
    platform.move(new_time)

# %% [markdown]
# ## Ground Truth Setup and Generation
#
# Target kinematics and tonal source metadata are defined here,
# then target states are propagated across all timesteps to build `target_ground_truth`.

# %%
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState

target_start_vector = np.array([-6000, 10, 2000, -1, -10.0, 0])
target_position_mapping = [0, 2, 4]
target_velocity_mapping = [1, 3, 5]
target_transition_model = CombinedLinearGaussianTransitionModel(
    [ConstantVelocity(0.001), ConstantVelocity(0.001), ConstantVelocity(0)]
)
target_amplitudes_upa = 10 ** (np.array([100.0, 100.0, 100.0]) / 20)
target_frequencies_hz = np.array([10.0, 50.0, 120.0])
target_phases_rad = np.array([0.0, 0.0, 0.0])

target_metadata = {
    "amplitudes_upa": target_amplitudes_upa,
    "frequencies_hz": target_frequencies_hz,
    "phases_rad": target_phases_rad,
    "position_mapping": target_position_mapping,
    "velocity_mapping": target_velocity_mapping,
}

target_states = [
    GroundTruthState(
        target_start_vector,
        timestamp=start_time,
        metadata=target_metadata,
    )
]

for i in range(1, num_steps):
    new_time = start_time + i * time_interval
    time_interval_now = new_time - target_states[-1].timestamp
    new_state_vector = target_transition_model.function(
        target_states[-1],
        noise=True,
        time_interval=time_interval_now,
    )
    target_states.append(
        GroundTruthState(
            new_state_vector,
            timestamp=new_time,
            metadata=target_metadata,
        )
    )

target_ground_truth = GroundTruthPath(target_states)

# %% [markdown]
# ## Propagation Models
#
# This section defines the acoustic environment (`Munk` SSP and flat bathymetry depth)
# and builds the propagation-model set used for comparison.
# `Bellhop` is included only when available in the runtime environment.

# %%
from bluepebble.models.environment import FlatBathymetry, Munk
from bluepebble.models.propagation import (
    BellhopAcousticPropagationModel,
    SphericalAcousticPropagationModel,
    rtrsAcousticPropagationModel,
)

ssp = Munk()
env_depth_m = 5000.0

propagation_models = {
    "Spherical": SphericalAcousticPropagationModel(ssp=ssp),
    "rtrs": rtrsAcousticPropagationModel(
        ssp=ssp,
        bathymetry=FlatBathymetry(depth=env_depth_m),
        use_all_frequencies=True,
    ),
}

bellhop_warning = None
try:
    propagation_models["Bellhop"] = BellhopAcousticPropagationModel(
        ssp=ssp,
        env_depth=env_depth_m,
    )
except FileNotFoundError as exc:
    bellhop_warning = str(exc)

if bellhop_warning is not None:
    print("Bellhop unavailable in this environment; skipping Bellhop model.")

propagation_model_colors = {
    "Spherical": "#1f77b4",
    "rtrs": "#ff7f0e",
    "Bellhop": "#2ca02c",
}

print(f"Enabled propagation models: {', '.join(propagation_models)}")

# %% [markdown]
# ## Signal Models
#
# This section declares the three anthropogenic source signal model classes
# that will be instantiated and run in later cells.

# %%
from bluepebble.signal.anthropogenic import (
    NarrowbandBlendedTonalSignal,
    NarrowbandOverlapAddTonalSignal,
    NarrowbandTonalSignal,
)

signal_model_specs = [
    ("NarrowbandTonalSignal", NarrowbandTonalSignal),
    ("NarrowbandBlendedTonalSignal", NarrowbandBlendedTonalSignal),
    ("NarrowbandOverlapAddTonalSignal", NarrowbandOverlapAddTonalSignal),
]

# %% [markdown]
# ## Beamformer
#
# This section sets beamforming parameters (`sampling_rate_hz`, shading, domain,
# steering azimuth grid) and builds the beamformer plus steering calculator.

# %%
from scipy.signal import get_window

from bluepebble.sigproc import DelayAndSumBeamformer, SteeringCalculator

sampling_rate_hz = 500.0
beamformer_shading = "blackman"
beamformer_domain = "frequency"
steering_azimuths_rad = np.linspace(0, np.pi, 181)

beamformer = DelayAndSumBeamformer(
    sampling_rate_hz=sampling_rate_hz,
    shading=get_window(beamformer_shading, platform.num_sensors),
    domain=beamformer_domain,
)

steering_calculator = SteeringCalculator(
    ssp=ssp,
    steering_azimuths_rad=steering_azimuths_rad,
)

print(f"Beamformer domain: {beamformer_domain}")
print(f"Steering grid size: {len(steering_azimuths_rad)}")

# %% [markdown]
# ## Simulation Pipeline Setup
#
# This section defines helper functions for signal-model reset, source generation,
# propagated simulation at one sensor index, plotting utilities, and WAV writing.

# %%
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.io import wavfile

from bluepebble.plotter import plot_spectrogram
from bluepebble.simulator import DiscretePassiveSonarArraySimulator

sensor_to_analyse = num_sensors // 2


def reset_signal_model(signal_model) -> None:
    """Reset the signal model to ensure consistent signal generation across simulations."""
    if hasattr(signal_model, "reset"):
        signal_model.reset()


def run_simulation(signal_model, propagation_model) -> np.ndarray:
    """Run a passive array simulation and return the received signal at the specified sensor."""
    reset_signal_model(signal_model)

    simulator = DiscretePassiveSonarArraySimulator(
        platform=platform,
        propagation_model=propagation_model,
        signal_models=[signal_model],
        noise_model=None,
        beamformer=beamformer,
        steering_calculator=steering_calculator,
        ground_truth_paths=[target_ground_truth],
    )

    sensor_chunks = []
    for _, sensor_data_set in simulator.sensor_data_gen():
        if not sensor_data_set:
            continue
        sensor_data = next(iter(sensor_data_set))
        sensor_chunks.append(sensor_data.raw_signals[sensor_to_analyse, :])

    if not sensor_chunks:
        raise RuntimeError("Simulation produced no sensor data.")

    return np.concatenate(sensor_chunks)


def generate_source_signal(signal_model) -> np.ndarray:
    """Generate the source signal using the signal model."""
    reset_signal_model(signal_model)

    source_chunks = []
    for state in target_states:
        chunk = signal_model.generate(
            source=state,
            sensor_delays_s=np.array([0.0]),
            tloss_db=0.0,
            propagation_time_s=0.0,
        )
        source_chunks.append(chunk[0, :])

    return np.concatenate(source_chunks)


def build_spectrogram_trace(signal: np.ndarray, show_scale: bool) -> go.Heatmap:
    """Build a Plotly Heatmap trace for the spectrogram of the given signal."""
    spec_fig = plot_spectrogram(
        signal=np.real(signal),
        sr=int(sampling_rate_hz),
        n_fft=int(1.5 * sampling_rate_hz),
        hop_length=int(0.75 * sampling_rate_hz),
        y_lim=(0, 250),
        yaxis_format="Hz",
        figsize=(800, 400),
    )

    trace = spec_fig.data[0]
    trace.update(showscale=show_scale)
    if show_scale:
        trace.colorbar = dict(title="Intensity (dB)", x=1.02, y=0.5, len=0.75)
    return trace


def plot_model_results(signal_model_name: str, source_signal: np.ndarray, received_signals: dict):
    """Plot the time-domain and spectrogram results for the source and received signals."""
    model_signals = {"Source": source_signal, **received_signals}
    rows = len(model_signals)

    subplot_titles = []
    for label in model_signals:
        subplot_titles.extend([f"{label} - Time Domain", f"{label} - Spectrogram"])

    fig = make_subplots(
        rows=rows,
        cols=2,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.09,
        vertical_spacing=0.07,
        shared_xaxes="columns",
    )

    for row_idx, (label, signal) in enumerate(model_signals.items(), start=1):
        time_axis = np.arange(len(signal)) / sampling_rate_hz

        trace_color = propagation_model_colors.get(label, "#222222")

        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=np.real(signal),
                mode="lines",
                line=dict(width=1, color=trace_color),
                name=f"{label} (time)",
                showlegend=False,
            ),
            row=row_idx,
            col=1,
        )

        fig.add_trace(
            build_spectrogram_trace(signal=signal, show_scale=row_idx == 1),
            row=row_idx,
            col=2,
        )

        for tonal_freq in target_frequencies_hz:
            fig.add_hline(
                y=float(tonal_freq),
                line_dash="dash",
                line_color="black",
                line_width=1,
                opacity=0.6,
                row=row_idx,
                col=2,
            )

        if row_idx == rows:
            fig.update_xaxes(title_text="Time (s)", row=row_idx, col=1)
        fig.update_yaxes(title_text="Amplitude (uPa)", row=row_idx, col=1)
        if row_idx == rows:
            fig.update_xaxes(title_text="Time (s)", row=row_idx, col=2)
        fig.update_yaxes(title_text="Frequency (Hz)", range=[0, 250], row=row_idx, col=2)

    fig.update_layout(
        template="plotly_white",
        width=1300,
        height=320 * rows + 120,
        title=dict(
            text=f"{signal_model_name} Comparison - Sensor {sensor_to_analyse}",
        ),
    )
    fig.show()


def save_as_wav(signal: np.ndarray, output_path: Path, sr: float) -> None:
    """Save the given signal as a WAV file, normalising to prevent clipping."""
    signal_real = np.real(signal) if np.iscomplexobj(signal) else signal
    peak = np.max(np.abs(signal_real))
    if peak > 0:
        signal_real = signal_real / peak
    wavfile.write(output_path, int(sr), np.int16(signal_real * 32767))


# %% [markdown]
# ## Run Simulation on Model Combinations
#
# This cell instantiates each signal model, runs every signal/propagation combination,
# and stores source signals, received signals, and per-run timing.

# %%
from time import perf_counter

signal_model_names = [name for name, _ in signal_model_specs]
source_signals = {}
received_signals = {}
runtimes_s = {}

signal_duration_s = SIM_RATE
blend_fraction = 0.40

for signal_name, signal_cls in signal_model_specs:
    if signal_name == "NarrowbandBlendedTonalSignal":
        signal_model = signal_cls(
            duration_s=signal_duration_s,
            sampling_rate_hz=sampling_rate_hz,
            blend_fraction=blend_fraction,
        )
    else:
        signal_model = signal_cls(
            duration_s=signal_duration_s,
            sampling_rate_hz=sampling_rate_hz,
        )

    source_signals[signal_name] = generate_source_signal(signal_model)
    received_signals[signal_name] = {}
    runtimes_s[signal_name] = {}

    for propagation_name, propagation_model in propagation_models.items():
        start_t = perf_counter()
        received_signals[signal_name][propagation_name] = run_simulation(
            signal_model,
            propagation_model,
        )
        runtimes_s[signal_name][propagation_name] = perf_counter() - start_t

for signal_name, timings in runtimes_s.items():
    timing_text = ", ".join(
        f"{propagation_name}: {runtime_s:.2f}s" for propagation_name, runtime_s in timings.items()
    )
    print(f"{signal_name} -> {timing_text}")

# %% [markdown]
# ## Time-Domain and Spectrogram Comparison
#
# For each signal model, this section plots source and received signals in the time domain
# and corresponding spectrograms for each propagation model.

# %%
for signal_name in signal_model_names:
    plot_model_results(
        signal_name,
        source_signals[signal_name],
        received_signals[signal_name],
    )

# %% [markdown]
# ## Power, Spectra, and Envelopes
#
# This section produces three diagnostics from the received signals:
# windowed power over time, frequency spectra, and absolute pressure envelopes.

# %%
signal_names = list(source_signals)

# Windowed received-signal power.
window_samples = int(signal_duration_s * sampling_rate_hz)
fig_power = make_subplots(
    rows=1,
    cols=len(signal_names),
    shared_yaxes=True,
    subplot_titles=signal_names,
)

for col_idx, signal_name in enumerate(signal_names, start=1):
    for propagation_name, signal in received_signals[signal_name].items():
        step_powers_db = []
        step_times_s = []

        for start_idx in range(0, len(signal), window_samples):
            end_idx = start_idx + window_samples
            if end_idx > len(signal):
                break
            window = signal[start_idx:end_idx]
            power_db = 10 * np.log10(np.mean(np.abs(window) ** 2) + 1e-12)
            step_powers_db.append(power_db)
            step_times_s.append(start_idx / sampling_rate_hz)

        color = propagation_model_colors.get(propagation_name, "#7f7f7f")

        fig_power.add_trace(
            go.Scatter(
                x=step_times_s,
                y=step_powers_db,
                mode="lines+markers",
                line=dict(width=2, color=color),
                marker=dict(color=color),
                name=propagation_name,
                legendgroup=propagation_name,
                showlegend=col_idx == 1,
            ),
            row=1,
            col=col_idx,
        )

    fig_power.update_xaxes(title_text="Time (s)", row=1, col=col_idx)

fig_power.update_yaxes(title_text="Power (dB)", row=1, col=1)
fig_power.update_layout(
    template="plotly_white",
    width=450 * len(signal_names),
    height=520,
    title=dict(text="Windowed Received-Signal Power"),
)
fig_power.show()

# Received-signal spectra.
fig_spectrum = make_subplots(
    rows=1,
    cols=len(signal_names),
    shared_yaxes=True,
    subplot_titles=signal_names,
)

for col_idx, signal_name in enumerate(signal_names, start=1):
    for propagation_name, signal in received_signals[signal_name].items():
        fft_signal = np.fft.fft(signal)
        fft_freq_hz = np.fft.fftfreq(len(signal), d=1 / sampling_rate_hz)
        positive_mask = fft_freq_hz > 0

        color = propagation_model_colors.get(propagation_name, "#7f7f7f")

        fig_spectrum.add_trace(
            go.Scatter(
                x=fft_freq_hz[positive_mask],
                y=20 * np.log10(np.abs(fft_signal[positive_mask]) + 1e-12),
                mode="lines",
                line=dict(width=1.3, color=color),
                name=propagation_name,
                legendgroup=propagation_name,
                showlegend=col_idx == 1,
            ),
            row=1,
            col=col_idx,
        )

    for tonal_freq in target_frequencies_hz:
        fig_spectrum.add_vline(
            x=float(tonal_freq),
            line_dash="dash",
            line_color="black",
            line_width=1,
            opacity=0.5,
            row=1,
            col=col_idx,
        )

    fig_spectrum.update_xaxes(title_text="Frequency (Hz)", range=[0, 250], row=1, col=col_idx)

fig_spectrum.update_yaxes(title_text="Magnitude (dB)", row=1, col=1)
fig_spectrum.update_layout(
    template="plotly_white",
    width=450 * len(signal_names),
    height=520,
    title=dict(text="Received-Signal Spectra"),
)
fig_spectrum.show()

# Absolute pressure envelopes.
fig_envelope = make_subplots(
    rows=len(signal_names),
    cols=1,
    shared_xaxes=True,
    subplot_titles=signal_names,
    vertical_spacing=0.08,
)

for row_idx, signal_name in enumerate(signal_names, start=1):
    for propagation_name, signal in received_signals[signal_name].items():
        time_axis = np.arange(len(signal)) / sampling_rate_hz
        color = propagation_model_colors.get(propagation_name, "#7f7f7f")

        fig_envelope.add_trace(
            go.Scatter(
                x=time_axis,
                y=np.abs(signal),
                mode="lines",
                line=dict(width=1.1, color=color),
                name=propagation_name,
                legendgroup=propagation_name,
                showlegend=row_idx == 1,
            ),
            row=row_idx,
            col=1,
        )

    for step in range(1, num_steps):
        fig_envelope.add_vline(
            x=step * signal_duration_s,
            line_dash="dash",
            line_color="black",
            line_width=1,
            opacity=0.5,
            row=row_idx,
            col=1,
        )

    fig_envelope.update_yaxes(title_text="|Pressure| (uPa)", row=row_idx, col=1)

fig_envelope.update_xaxes(title_text="Time (s)", row=len(signal_names), col=1)
fig_envelope.update_layout(
    template="plotly_white",
    width=1250,
    height=320 * len(signal_names) + 120,
    title=dict(text="Absolute Pressure Envelopes"),
)
fig_envelope.show()

# %% [markdown]
# ## Optional WAV Export
#
# Enable `export_wav` to write source and received signals to `wavs/`.
# When disabled, no files are written.

# %%
export_wav = False
output_dir = Path("wavs")

if export_wav:
    output_dir.mkdir(parents=True, exist_ok=True)

    for signal_name, signal in source_signals.items():
        output_path = output_dir / f"source_{signal_name.lower()}.wav"
        save_as_wav(signal, output_path, sampling_rate_hz)

    for signal_name, model_signals in received_signals.items():
        for propagation_name, signal in model_signals.items():
            safe_propagation_name = propagation_name.lower().replace(" ", "_")
            output_path = output_dir / (
                f"received_{signal_name.lower()}_{safe_propagation_name}.wav"
            )
            save_as_wav(signal, output_path, sampling_rate_hz)

    print(f"WAV files written to {output_dir.resolve()}")
else:
    print("WAV export disabled. Set export_wav = True to write files.")
