# %% [markdown]  # noqa: D100
# # Broadband Measured vs Synthetic Comparison Across Simulator Modes
#
# This example extends `bb_sig_analysis.py` by looping over multiple simulator
# implementations and plotting a spectrogram table for each mode.
#
# Retained from the original workflow:
# - Ground-truth world plot
# - Source spectrogram plots (synthetic and measured)
#
# Removed for brevity:
# - Frequency-spectrum plots
# - Time-series plots
# - WAV export

# %% [markdown]
# ## Setup and Reproducibility

# %%
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal as spsignal
from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel,
    ConstantVelocity,
)
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState

from bluepebble.models.environment import Constant, FlatBathymetry
from bluepebble.models.propagation import (
    CylindricalAcousticPropagationModel,
    rtrsAcousticPropagationModel,
)
from bluepebble.platform import TowedArrayPlatform
from bluepebble.plotter import plot_spectrogram, plot_world
from bluepebble.signal.anthropogenic import BroadbandRecordedSignal, BroadbandSyntheticSignal
from bluepebble.simulator import (
    ContinuousFractionalDelayPassiveSonarArraySimulator,
    ContinuousSTFTPassiveSonarArraySimulator,
    DiscretePassiveSonarArraySimulator,
)

np.random.seed(1999)

# %% [markdown]
# ## Simulation Parameters

# %%
SIM_RATE = 2.0
SIM_LENGTH_S = 300.0
SIM_PARAMS = {
    "start_time": datetime.now().replace(hour=0, minute=0, second=0, microsecond=0),
    "time_interval": timedelta(seconds=SIM_RATE),
    "num_steps": int(SIM_LENGTH_S / SIM_RATE),
}

total_duration_s = SIM_PARAMS["num_steps"] * SIM_PARAMS["time_interval"].total_seconds()

print("=== Broadband Simulator-Mode Comparison ===")
print(f"Total simulation duration: {total_duration_s} s")
print(f"Number of timesteps: {SIM_PARAMS['num_steps']}")
print(f"Timestep interval: {SIM_PARAMS['time_interval'].total_seconds()} s")

# %% [markdown]
# ## Platform and Array Parameters

# %%
SHIP_PARAMS = {
    "start_vector": np.array([0, 0, 0, 0, -10.0, 0]),
    "position_mapping": [0, 2, 4],
    "velocity_mapping": [1, 3, 5],
    "transition_model": CombinedLinearGaussianTransitionModel(
        [ConstantVelocity(0), ConstantVelocity(0), ConstantVelocity(0)]
    ),
}

ARRAY_PARAMS = {
    "num_sensors": 3,
    "tow_cable_length": 10.0,
    "sensor_spacing": 1.0,
    "array_depth": -10.0,
}

SENSOR_TO_ANALYZE = ARRAY_PARAMS["num_sensors"] // 2

# %% [markdown]
# ## Target Parameters

# %%
TARGET_PARAMS = {
    "start_vector": np.array([-1000, 0, 750, -5.0, -10.0, 0]),
    "position_mapping": [0, 2, 4],
    "velocity_mapping": [1, 3, 5],
    "transition_model": CombinedLinearGaussianTransitionModel(
        [ConstantVelocity(0.001), ConstantVelocity(0.001), ConstantVelocity(0)]
    ),
    "amplitudes_upa": 10 ** (np.array([100.0, 90.0, 100.0, 85.0]) / 20),
    "frequencies_hz": np.array([20.0, 140.0, 200.0, 500.0]),
    "phases_rad": np.random.uniform(0, 2 * np.pi, 4),
    "tonal_bandwidth_hz": 10.0,
    "noise_amplitude_upa": 10 ** (80 / 20),
    "noise_spectral_exponent": -1.0,
}

# %% [markdown]
# ## Signal Parameters

# %%
SIGNAL_PARAMS = {
    "duration_s": total_duration_s,
    "sampling_rate_hz": 3000.0,
    "frame_len": 500,
    "hop_factor": 4,
    "fade_in_ms": 100.0,
    "fade_out_ms": 100.0,
}

# %% [markdown]
# ## Platform Generation

# %%
initial_state = GroundTruthState(SHIP_PARAMS["start_vector"], timestamp=SIM_PARAMS["start_time"])

platform = TowedArrayPlatform(
    states=[initial_state],
    position_mapping=SHIP_PARAMS["position_mapping"],
    velocity_mapping=SHIP_PARAMS["velocity_mapping"],
    transition_models=[SHIP_PARAMS["transition_model"]],
    transition_times=[timedelta(seconds=total_duration_s)],
    num_sensors=ARRAY_PARAMS["num_sensors"],
    cable_length_m=ARRAY_PARAMS["tow_cable_length"],
    sensor_spacing_m=ARRAY_PARAMS["sensor_spacing"],
    array_depth_m=ARRAY_PARAMS["array_depth"],
)

for i in range(1, SIM_PARAMS["num_steps"]):
    new_time = SIM_PARAMS["start_time"] + i * SIM_PARAMS["time_interval"]
    platform.move(new_time)

# %% [markdown]
# ## Target Trajectory

# %%
target_states = [
    GroundTruthState(
        TARGET_PARAMS["start_vector"],
        timestamp=SIM_PARAMS["start_time"],
        metadata={
            "amplitudes_upa": TARGET_PARAMS["amplitudes_upa"],
            "frequencies_hz": TARGET_PARAMS["frequencies_hz"],
            "phases_rad": TARGET_PARAMS["phases_rad"],
            "position_mapping": TARGET_PARAMS["position_mapping"],
            "velocity_mapping": TARGET_PARAMS["velocity_mapping"],
        },
    )
]

transition_model = TARGET_PARAMS["transition_model"]
for i in range(1, SIM_PARAMS["num_steps"]):
    new_time = SIM_PARAMS["start_time"] + i * SIM_PARAMS["time_interval"]
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
fig_world.update_layout(title="World Picture")
fig_world.show()

# %% [markdown]
# ## Source Signal Models

# %%
synthetic_signal_model = BroadbandSyntheticSignal(
    duration_s=SIGNAL_PARAMS["duration_s"],
    sampling_rate_hz=SIGNAL_PARAMS["sampling_rate_hz"],
    frame_len=SIGNAL_PARAMS["frame_len"],
    hop_factor=SIGNAL_PARAMS["hop_factor"],
    tonal_bandwidth_hz=TARGET_PARAMS["tonal_bandwidth_hz"],
    noise_amplitude_upa=TARGET_PARAMS["noise_amplitude_upa"],
    noise_spectral_exponent=TARGET_PARAMS["noise_spectral_exponent"],
    noise_freq_range_hz=(0.0, SIGNAL_PARAMS["sampling_rate_hz"] / 2),
    tonal_noise_is_constant=True,
    noise_is_constant=True,
)

wav_name = "SanctSound_CI05_03_largeship_20190925T135956Z.wav"
data_dir = Path(__file__).resolve().parent / "measured_data"
measured_wav_path = data_dir / wav_name

measured_signal_model = BroadbandRecordedSignal(
    duration_s=SIGNAL_PARAMS["duration_s"],
    sampling_rate_hz=SIGNAL_PARAMS["sampling_rate_hz"],
    frame_len=SIGNAL_PARAMS["frame_len"],
    hop_factor=SIGNAL_PARAMS["hop_factor"],
    wav_path=str(measured_wav_path),
    segment_start_s=0.0,
    segment_duration_s=30.0,
    duration_match_mode="tile",
    level_db_re_1upa=85.0,
)

# %% [markdown]
# ## Propagation Model
#
# Try switching between rtrs and cyclindrical to see how the propagation model affects the
# simulation results.

# %%
ssp = Constant(speed=1500.0)
bathymetry = FlatBathymetry(depth=-100.0)

prop_model_to_use = "cylin"

if prop_model_to_use == "rtrs":
    prop_model = rtrsAcousticPropagationModel(
        ssp=ssp,
        bathymetry=bathymetry,
        use_all_frequencies=False,
        step_m=15.0,
        azimuth_search_width=10.0,
        azimuth_resolution=2.0,
        elevation_range=(-55.0, 55.0),
        elevation_resolution=2.0,
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
else:
    prop_model = CylindricalAcousticPropagationModel(
        attenuation_factor=5.0,
        ssp=ssp,
    )

# %% [markdown]
# ## Source Spectrograms

# %%
try:
    source_signal_synthetic = synthetic_signal_model.get_source_signal()
except RuntimeError:
    synthetic_signal_model.compute_stft(target_states[0])
    source_signal_synthetic = synthetic_signal_model.get_source_signal()

try:
    source_signal_measured = measured_signal_model.get_source_signal()
except RuntimeError:
    measured_signal_model.compute_stft(target_states[0])
    source_signal_measured = measured_signal_model.get_source_signal()

synthetic_source_real = np.real(source_signal_synthetic)
measured_source_real = np.real(source_signal_measured)

fig_synth_source_spec = plot_spectrogram(
    synthetic_source_real,
    int(SIGNAL_PARAMS["sampling_rate_hz"]),
    n_fft=500,
    hop_length=250,
    y_lim=(0, SIGNAL_PARAMS["sampling_rate_hz"] / 2),
    yaxis_format="hz",
    figsize=(9, 4),
)
fig_synth_source_spec.update_layout(title="Spectrogram - Synthetic Source Signal")
fig_synth_source_spec.show()

fig_meas_source_spec = plot_spectrogram(
    measured_source_real,
    int(SIGNAL_PARAMS["sampling_rate_hz"]),
    n_fft=500,
    hop_length=250,
    y_lim=(0, SIGNAL_PARAMS["sampling_rate_hz"] / 2),
    yaxis_format="hz",
    figsize=(9, 4),
)
fig_meas_source_spec.update_layout(title="Spectrogram - Measured Source Signal")
fig_meas_source_spec.show()

# %% [markdown]
# ## Simulator Comparison Table
#
# Rows correspond to simulator implementations and columns correspond to source
# types (synthetic / measured). Each cell is the received-signal spectrogram for
# the selected sensor.
#
# Simulator types used in this comparison:
# - **STFT Interp**: STFT-domain propagation with interpolation between
#   transfer-function updates.
# - **WOLA Interp**: Weighted overlap-add reconstruction with interpolated
#   frame-to-frame propagation.
# - **COLA**: Constant overlap-add STFT processing for stable frame stitching.
# - **Fractional Delay**: Time-domain model using sub-sample delay alignment.
# - **Discrete**: Per-step discrete simulation baseline without continuous
#   overlap-add processing.

# %%
SIMULATOR_CONFIGS = [
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
        "propagation_model": prop_model,
        "signal_models": [signal_model],
        "noise_model": None,
        "beamformer": None,
        "steering_calculator": None,
        "ground_truth_paths": [target_ground_truth],
    }

    if config["kind"] == "stft":
        return ContinuousSTFTPassiveSonarArraySimulator(
            mode=config["mode"],
            fade_in_ms=SIGNAL_PARAMS["fade_in_ms"],
            fade_out_ms=SIGNAL_PARAMS["fade_out_ms"],
            **common_kwargs,
        )

    if config["kind"] == "fractional":
        return ContinuousFractionalDelayPassiveSonarArraySimulator(
            fade_in_ms=SIGNAL_PARAMS["fade_in_ms"],
            fade_out_ms=SIGNAL_PARAMS["fade_out_ms"],
            **common_kwargs,
        )

    if config["kind"] == "discrete":
        return DiscretePassiveSonarArraySimulator(**common_kwargs)

    raise ValueError(f"Unsupported simulator config: {config}")


def run_continuous_simulation(simulator_obj):
    """Run a continuous simulator and return the received signal for the selected sensor."""
    all_sensor_signals = []
    for _, sensor_data_set in simulator_obj.sensor_data_gen():
        sensor_data = next(iter(sensor_data_set))
        all_sensor_signals.append(sensor_data.raw_signals)

    all_sensor_signals_array = np.concatenate(all_sensor_signals, axis=1)
    return all_sensor_signals_array[SENSOR_TO_ANALYZE, :]


def compute_spectrogram_db(signal_data, sampling_rate_hz, n_fft=500, hop_length=250):
    """Compute a spectrogram and return frequencies, times, and power in dB."""
    frequencies, times, spec_power = spsignal.spectrogram(
        np.real(signal_data),
        fs=sampling_rate_hz,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        scaling="density",
        mode="psd",
    )
    spec_db = 10 * np.log10(spec_power + 1e-16)
    return frequencies, times, spec_db


comparison_results = []
for config in SIMULATOR_CONFIGS:
    print(f"Running simulator mode: {config['label']}")

    synthetic_sim = build_simulator(config, synthetic_signal_model)
    measured_sim = build_simulator(config, measured_signal_model)

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
fig_grid = make_subplots(
    rows=n_rows,
    cols=2,
    shared_xaxes=False,
    shared_yaxes=False,
    vertical_spacing=0.03,
    horizontal_spacing=0.08,
    subplot_titles=[
        title
        for result in comparison_results
        for title in (
            f"{result['label']} - Synthetic (Sensor {SENSOR_TO_ANALYZE})",
            f"{result['label']} - Measured (Sensor {SENSOR_TO_ANALYZE})",
        )
    ],
)

for row_idx, result in enumerate(comparison_results, start=1):
    for col_idx, key in enumerate(["synthetic", "measured"], start=1):
        f_hz, t_s, s_db = compute_spectrogram_db(
            result[key],
            SIGNAL_PARAMS["sampling_rate_hz"],
            n_fft=500,
            hop_length=250,
        )

        fig_grid.add_trace(
            go.Heatmap(
                x=t_s,
                y=f_hz,
                z=s_db,
                colorscale="Viridis",
                zmin=-50,
                zmax=20,
                showscale=(row_idx == 1 and col_idx == 2),
                colorbar=dict(title="dB re 1 uPa^2/Hz")
                if (row_idx == 1 and col_idx == 2)
                else None,
            ),
            row=row_idx,
            col=col_idx,
        )

        fig_grid.update_yaxes(
            title_text="Frequency (Hz)" if col_idx == 1 else None,
            range=[0, SIGNAL_PARAMS["sampling_rate_hz"] / 2],
            row=row_idx,
            col=col_idx,
        )
        if row_idx == n_rows:
            fig_grid.update_xaxes(title_text="Time (s)", row=row_idx, col=col_idx)

fig_grid.update_layout(
    template="plotly_white",
    height=320 * n_rows,
    width=1200,
    title="Received Signal Spectrogram Table by Simulator Mode",
)
fig_grid.show()
