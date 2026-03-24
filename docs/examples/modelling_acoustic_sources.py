"""
==========================
Modelling Acoustic Sources
==========================

This example previews several standalone acoustic source and noise models, then combines
them into a simple composite soundscape. It is intended as a quick orientation for how
different source classes behave before they are embedded in a full propagation and
beamforming pipeline.

Passive sonar scenes often contain a mix of biological, anthropogenic, and ambient
contributors. Understanding the isolated time-frequency signature of each component
makes it easier to interpret later BTRs, spectrograms, and received mixtures.
"""  # noqa: D205, D212, D400, D415

# %%
# Imports
# -------
#
# All dependencies are consolidated here for convenience.

from datetime import datetime
from pathlib import Path

import numpy as np
from stonesoup.types.groundtruth import GroundTruthState

from bluepebble.plotter import plot_spectrogram
from bluepebble.signal.anthropogenic import (
    RecordedAnthropogenicSignal,
    SyntheticAnthropogenicSignal,
)
from bluepebble.signal.biological import PointSourceSnappingShrimpSignal, WhaleCallSignal
from bluepebble.signal.effects import Reverb
from bluepebble.signal.random import WhiteNoiseSignal

# %%
# Setup and Reproducibility
# -------------------------
#
# A fixed random seed ensures all stochastic signal components are identical on every run,
# making the spectrograms fully reproducible.
#
# Each panel uses a different ``n_fft`` chosen for its frequency content, so PSD
# (which normalises by frequency resolution, shifting the broadband noise floor by
# 10·log10(Δf) between panels) would make equivalent signals appear at different
# levels depending on FFT size. STFT magnitude with peak normalisation is used instead:
# for tonal signals the bin amplitude is independent of ``n_fft``, and for this example
# the goal is visual orientation rather than absolute spectral density.

seed = 2000
np.random.seed(seed)

sampling_rate_Hz = 48_000
signal_duration_s = 10.0
reference_time = datetime(2026, 1, 1, 0, 0, 0)

component_signals = {}


# %%
# WAV File Path Resolution
# ------------------------
#
# The measured-vessel section loads a WAV recording. The path search below works both in
# the gallery build (where ``__file__`` is set) and in interactive use from any working
# directory.

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

# %%
# Whale Call Signal
# -----------------
#
# This section simulates a structured humpback-style vocalisation.
#
# - The call is built from reusable themes and phrases.
# - Harmonics, vibrato, breathy noise, and reverb shape the timbre.
# - The result is a comparatively rich mid-frequency biological source.

whale_source = GroundTruthState(
    [0, 0, 0, 0],
    timestamp=reference_time,
    metadata={
        "amplitude_upa": 10 ** (180 / 20),
    },
)

theme_0 = [15, -15, 15, -15]
theme_1 = [10, 0]
theme_2 = [-20, -10, 10]
theme_3 = [0, 0, 0, 0]

phrase_a = [0, 1]
phrase_b = [2, 1]
phrase_c = [3]
song_phrases = [phrase_c, phrase_a, phrase_b]

reverb_effect = Reverb(duration_s=0.4, wet_dry_mix=0.8)

whale_signal_model = WhaleCallSignal(
    duration_s=signal_duration_s,
    sampling_rate_hz=sampling_rate_Hz,
    song_structure_enabled=True,
    song_themes=[theme_0, theme_1, theme_2, theme_3],
    song_phrases=song_phrases,
    theme_base_freq_hz=250,
    theme_freq_jitter_hz=10,
    theme_duration_s=1.2,
    mean_call_interval_s=2.0,
    interval_jitter_s=0.3,
    call_duration_s=1.0,
    duration_jitter_s=0.3,
    start_freq_hz=250,
    start_freq_jitter_hz=20,
    end_freq_hz=800,
    end_freq_jitter_hz=50,
    num_contour_points=3,
    contour_variability_hz=10,
    min_harmonics=20,
    max_harmonics=25,
    harmonic_decay_db=1,
    vibrato_rate_hz=2.5,
    vibrato_depth_hz=1.0,
    sub_harmonic_ratios=[0.5],
    sub_harmonic_amplitude_ratio=0.25,
    add_breathy_noise=True,
    breathy_noise_amount=0.15,
    breathy_noise_lp_cutoff_hz=1800,
    low_cutoff_hz=200,
    high_cutoff_hz=5000,
    envelope_taper_ratio=0.8,
    effects=[reverb_effect],
)

whale_calls_complex = whale_signal_model.generate(
    source=whale_source,
    sensor_delays_s=np.array([0.0]),
    tloss_db=90.0,
    propagation_time_s=0.0,
)
whale_calls_real = np.real(whale_calls_complex[0, :])
component_signals["whale_call"] = whale_calls_real

fig_whale = plot_spectrogram(
    signal=whale_calls_real,
    sr=int(sampling_rate_Hz),
    n_fft=4096,
    hop_length=1024,
    y_lim=(0, 5500),
    yaxis_format="kHz",
    analysis_mode="stft",
    db_reference="peak",
    z_lim=(-60.0, 0.0),
    colorbar_title="dB re peak",
    hovertemplate=(
        "Time: %{x:.2f} s<br>Frequency: %{y:.1f} Hz<br>Level: %{z:.1f} dB re peak<extra></extra>"
    ),
).update_layout(
    title="Whale Call Spectrogram",
    template="plotly_white",
    autosize=True,
    width=None,
    height=None,
)

# %%
# Snapping Shrimp Signal
# ----------------------
#
# This section models the broadband crackle associated with a snapping shrimp colony.
#
# - The snap is short, impulsive, and strongly broadband.
# - This makes it a useful contrast against the tonal or structured signals elsewhere
#   in the example.
# - The example uses a point-source far-field approximation rather than a diffuse colony
#   model.

shrimp_source = GroundTruthState(
    [0, 0, 0, 0, -50, 0],
    timestamp=reference_time,
    metadata={
        "amplitude_upa": 10 ** (195 / 20),
        "position_mapping": [0, 2, 4],
    },
)

shrimp_signal_model = PointSourceSnappingShrimpSignal(
    duration_s=signal_duration_s,
    sampling_rate_hz=sampling_rate_Hz,
    temperature_celsius=25,
    start_time_hours=18.0,
    diurnal_amplitude=0.25,
    diurnal_phase_hours=6,
    delay_duration=0.0006,
    onset_duration=0.0001,
    snap_duration=0.0014,
    onset_level=0.15,
    onset_freq=2500,
    snap_decay=1000,
    low_cutoff_hz=1500,
    high_cutoff_hz=10000,
)

shrimp_signal = shrimp_signal_model.generate(
    source=shrimp_source,
    sensor_delays_s=np.array([0.0]),
    tloss_db=80.0,
    propagation_time_s=10.0,
)
shrimp_signal_real = np.real(shrimp_signal[0, :])
component_signals["snapping_shrimp"] = shrimp_signal_real

fig_shrimp = plot_spectrogram(
    signal=shrimp_signal_real,
    sr=int(sampling_rate_Hz),
    n_fft=2048,
    hop_length=512,
    y_lim=(0, 20000),
    yaxis_format="kHz",
    analysis_mode="stft",
    db_reference="peak",
    z_lim=(-120.0, 0.0),
    colorbar_title="dB re peak",
    hovertemplate=(
        "Time: %{x:.2f} s<br>Frequency: %{y:.1f} Hz<br>Level: %{z:.1f} dB re peak<extra></extra>"
    ),
).update_layout(
    title="Snapping Shrimp Spectrogram",
    template="plotly_white",
    autosize=True,
    width=None,
    height=None,
)

# %%
# Commercial Vessel Tonals
# ------------------------
#
# This section generates a simplified low-frequency ship signature.
#
# - The signal is dominated by a handful of narrowband tonal lines.
# - These tones represent blade-rate and machinery components.
# - In contrast to the biological examples above, this source is continuous and spectrally
#   stable over the window shown here.

commercial_vessel_state = GroundTruthState(
    [0, 0, 0, 0, -10, 0],
    timestamp=reference_time,
    metadata={
        "frequencies_hz": np.array([50.0, 75.0, 125.0, 82.0]),
        "amplitudes_upa": 10 ** (np.array([175.0, 168.0, 162.0, 160.0]) / 20),
        "phases_rad": np.random.uniform(0, 2 * np.pi, 4),
        "position_mapping": [0, 2, 4],
    },
)

tonal_signal_model = SyntheticAnthropogenicSignal(
    duration_s=signal_duration_s,
    sampling_rate_hz=sampling_rate_Hz,
    frame_len=2048,
    hop_factor=4,
)
_ = tonal_signal_model.compute_stft(source=commercial_vessel_state)
tonal_signal_real = np.real(tonal_signal_model.get_source_signal())
component_signals["commercial_vessel"] = tonal_signal_real

fig_vessel_tonal = plot_spectrogram(
    signal=tonal_signal_real,
    sr=int(sampling_rate_Hz),
    n_fft=4096 * 6,
    hop_length=1024,
    y_lim=(0, 200),
    yaxis_format="Hz",
    analysis_mode="stft",
    db_reference="peak",
    z_lim=(-40.0, 0.0),
    colorbar_title="dB re peak",
    hovertemplate=(
        "Time: %{x:.2f} s<br>Frequency: %{y:.1f} Hz<br>Level: %{z:.1f} dB re peak<extra></extra>"
    ),
).update_layout(
    title="Commercial Vessel Tonal Spectrogram",
    template="plotly_white",
    autosize=True,
    width=None,
    height=None,
)

# %%
# Measured Vessel Noise
# ---------------------
#
# This section loads a real recording of a commercial vessel from a WAV file. This
# recording came from `Sanct Sounds <https://sanctsound.ioos.us/sounds.html#Vessels>`_
# and contains the recording of a large vessel. The file is included in
# ``docs/examples/measured_data/``; the path was resolved at the top of this script.

measured_vessel_state = GroundTruthState(
    [0, 0, 0, 0, -10, 0],
    timestamp=reference_time,
    metadata={
        "frequencies_hz": np.array([50.0, 75.0, 125.0, 82.0]),
        "amplitudes_upa": 10 ** (np.array([175.0, 168.0, 162.0, 160.0]) / 20),
        "phases_rad": np.random.uniform(0, 2 * np.pi, 4),
        "position_mapping": [0, 2, 4],
    },
)

measured_signal_model = RecordedAnthropogenicSignal(
    duration_s=signal_duration_s,
    sampling_rate_hz=sampling_rate_Hz,
    frame_len=500,
    hop_factor=2,
    wav_path=str(measured_wav_path),
    segment_start_s=0.0,
    segment_duration_s=30.0,
    duration_match_mode="tile",
    level_db_re_1upa=85.0,
)

# RecordedAnthropogenicSignal is frequency-domain only
_ = measured_signal_model.compute_stft(source=measured_vessel_state)
measured_signal_real = np.real(measured_signal_model.get_source_signal())
component_signals["measured_vessel"] = measured_signal_real

fig_vessel_measured = plot_spectrogram(
    signal=measured_signal_real,
    sr=int(sampling_rate_Hz),
    n_fft=4096 * 6,
    hop_length=1024,
    y_lim=(0, 4000),
    yaxis_format="Hz",
    analysis_mode="stft",
    db_reference="peak",
    z_lim=(-60.0, 0.0),
    colorbar_title="dB re peak",
    hovertemplate=(
        "Time: %{x:.2f} s<br>Frequency: %{y:.1f} Hz<br>Level: %{z:.1f} dB re peak<extra></extra>"
    ),
).update_layout(
    title="Measured Vessel Noise Spectrogram",
    template="plotly_white",
    autosize=True,
    width=None,
    height=None,
)

# %%
# Ambient White Noise
# -------------------
#
# This section generates a simple ambient baseline.
#
# - :class:`~.WhiteNoiseSignal` is used here as a deliberately simple reference model.
# - It does not attempt to reproduce a full ocean ambient spectrum.
# - The output is useful as a baseline when contrasting structured and unstructured energy.

ambient_noise = WhiteNoiseSignal(
    amplitude_upa=10 ** (90 / 20),
    duration_s=signal_duration_s,
    sampling_rate_hz=sampling_rate_Hz,
).generate()

ambient_noise_real = np.real(ambient_noise[0, :])
component_signals["ambient_white_noise"] = ambient_noise_real

fig_ambient = plot_spectrogram(
    signal=ambient_noise_real,
    sr=int(sampling_rate_Hz),
    n_fft=4096,
    hop_length=1024,
    y_lim=(0, 20000),
    yaxis_format="kHz",
    analysis_mode="stft",
    db_reference="peak",
    z_lim=(-60.0, 0.0),
    colorbar_title="dB re peak",
    hovertemplate=(
        "Time: %{x:.2f} s<br>Frequency: %{y:.1f} Hz<br>Level: %{z:.1f} dB re peak<extra></extra>"
    ),
).update_layout(
    title="Ambient White Noise Spectrogram",
    template="plotly_white",
    autosize=True,
    width=None,
    height=None,
)

# %%
# Composite Soundscape
# --------------------
#
# This final signal combines the individual synthetic components into one simple scene.
#
# - The vessel tonals dominate the low end.
# - The whale call contributes mid-band contour and harmonic structure.
# - The shrimp and white-noise components raise the broadband floor.

soundscape = (
    component_signals["whale_call"]
    + component_signals["snapping_shrimp"]
    + component_signals["commercial_vessel"]
    + component_signals["measured_vessel"]
    + component_signals["ambient_white_noise"]
)
component_signals["composite_soundscape"] = soundscape

fig_composite = plot_spectrogram(
    signal=soundscape,
    sr=int(sampling_rate_Hz),
    n_fft=4096,
    hop_length=1024,
    y_lim=(0, 5500),
    yaxis_format="kHz",
    analysis_mode="stft",
    db_reference="peak",
    z_lim=(-120.0, 0.0),
    colorbar_title="dB re peak",
    hovertemplate=(
        "Time: %{x:.2f} s<br>Frequency: %{y:.1f} Hz<br>Level: %{z:.1f} dB re peak<extra></extra>"
    ),
).update_layout(
    title="Composite Soundscape Spectrogram",
    template="plotly_white",
    autosize=True,
    width=None,
    height=None,
)
