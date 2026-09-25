"""
==============================
Comparing Hydrophone Responses
==============================

This example compares how four hydrophone configurations affect the beamformed output
and downstream detections when receiving the same acoustic scene. The scenario is held
fixed throughout: one target radiates three tonal components at 300 Hz, 1100 Hz, and
1300 Hz, propagated to the array under a spherical spreading model. Only the
:class:`~bluepebble.sensor.HydrophoneResponse` and self-noise sources attached to each
array element change between runs.

The configurations are chosen to isolate distinct modelling and calibration effects:

1. **Flat** — null baseline; uniform unity response, no self-noise.
2. **Fourth-order resonant** (`FourthOrderResonantResponse`) — physically motivated
   wideband piezoceramic model: second-order LF coupling rolloff (5 Hz) and a
   second-order element resonance (1500 Hz, ζ = 0.25).  This reproduces the flat
   passband, resonance peak, and steep HF rolloff seen on real datasheets (e.g. TC4032,
   B&K 8104).
3. **Miscalibrated correction** — config 2 with a calibration correction applied that
   assumes the resonance is 20% lower than its true value (1200 Hz instead of 1500 Hz).
   The user's inverse correction overcorrects at 1200 Hz, placing a notch directly
   between the 1100 Hz and 1300 Hz tonals.  With source levels at the detection margin,
   this notch suppresses those tonals below the CA-CFAR threshold — demonstrating how
   a resonance frequency error causes real detection misses.
4. **Goody TBL flow noise** — flat response with turbulent boundary layer self-noise
   modelled by :class:`~bluepebble.sensor.GoodyFlowNoiseSpectrum` on each element.
   Self-noise is injected in the pressure domain so it is coloured by the hydrophone
   transfer function before reaching the output.

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
from pathlib import Path

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
    FourthOrderResonantResponse,
    FrequencyResponse,
    GoodyFlowNoiseSpectrum,
    Hydrophone,
    HydrophoneResponse,
    LinearHydrophoneArray,
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

sim_length_s = 120
sim_rate_s = 2.0

start_time = datetime(2026, 1, 1, 0, 0, 0)
time_interval = timedelta(seconds=sim_rate_s)
num_steps = int(sim_length_s / sim_rate_s)
total_duration_s = num_steps * time_interval.total_seconds()

# Output settings for presentation-ready PNGs (tight margins, no title).
PRESENTATION_FIG_DIR = Path("./presentation_figs")
PRESENTATION_FIG_DIR.mkdir(exist_ok=True)
PRESENTATION_FIG_SCALE = 3
PRESENTATION_FIG_MARGIN = dict(l=70, r=20, t=15, b=55)
PRESENTATION_FIG_MARGIN_COLORBAR = dict(l=70, r=110, t=30, b=55)
# Configs to include in saved presentation figures: Flat, Resonant, Miscalibrated.
# This subset tells the calibration-error story directly: a flat null baseline,
# the physically realistic truth, and the residual ripple left by a 20% error in
# resonance frequency.
_PRESENTATION_SUBSET_INDICES = [0, 1, 2]


def save_for_presentation(fig, name, width=550, height=430, margin=None):
    """Write a Plotly figure as a tight-margin PNG for slide use."""
    fig.update_layout(title=None, margin=margin or PRESENTATION_FIG_MARGIN)
    fig.write_image(
        PRESENTATION_FIG_DIR / f"{name}.png",
        width=width,
        height=height,
        scale=PRESENTATION_FIG_SCALE,
    )


# %%
# Hydrophone Response Configurations
# -----------------------------------
#
# Four configurations are defined here. Each is chosen to isolate a distinct modelling
# or calibration effect.  The scalar sensitivity is left at 0 dB throughout so that only
# the frequency shaping differs between configurations.

# 1. Flat response — null baseline; no frequency shaping applied.
response_flat = HydrophoneResponse()

# 2. Fourth-order resonant model: second-order LF rolloff plus a second-order
#    element resonance.  This is a physically motivated minimum-phase model of a
#    wideband piezoceramic hydrophone with integrated preamplifier (TC4032 / B&K
#    8104 class).  Parameters scaled to fit the 0–2 kHz simulation band:
#      - lf_cutoff_hz  : preamp coupling corner
#      - resonance_hz  : element radial-mode resonance
#      - resonance_damping : 0.25 gives a ~6 dB peak (typical of rubber-encapsulated
#                           wideband designs)
response_resonant = HydrophoneResponse(
    frequency_response=FourthOrderResonantResponse(
        lf_cutoff_hz=5.0,
        lf_damping=0.707,
        resonance_hz=1500.0,
        resonance_damping=0.25,
    ),
)


# 3. Miscalibrated correction: the physical hydrophone is config 2, but the user
#    has applied an inverse calibration that assumes the resonance lies at 1200 Hz
#    (a 20% negative error in resonance frequency).  The net response is the truth
#    divided by the assumed model.  Because the LF stages are identical, the ratio
#    simplifies to a single biquad in the resonance terms:
#
#        H_net(s) = (w_truth^2 / w_assumed^2)
#                   * (s^2 + 2*zeta*w_assumed*s + w_assumed^2)
#                   / (s^2 + 2*zeta*w_truth*s   + w_truth^2)
#
#    This form is finite at DC (avoiding the 0/0 singularity of the explicit
#    truth/assumed division at f = 0).  At DC the gain factor and numerator
#    cancel exactly, leaving the passband at 0 dB.  The key effect: the notch
#    in H_net falls at w_assumed = 1200 Hz (where the user's inverse correction
#    overcorrects), placing it directly between the 1100 Hz and 1300 Hz tonals.
#    With source levels set at the detection margin, this notch is deep enough to
#    suppress those tonals below the CA-CFAR threshold.
_assumed_resonance_hz = 1200.0  # 20% below truth (1500 Hz)
_resonance_damping = 0.1


class _MiscalibratedResponse(FrequencyResponse):
    """Net response when applying a perturbed inverse to the true response.

    Encapsulates the analytic biquad ratio H_truth(s) / H_assumed(s) when the
    LF stages match and only the resonance frequency differs.  Avoids the
    0/0 singularity at DC that would arise from a literal pointwise division.
    """

    def evaluate(self, frequencies_hz):
        f = np.asarray(frequencies_hz, dtype=float)
        s = 1j * 2.0 * np.pi * f
        w_truth = 2.0 * np.pi * 1500.0
        w_assumed = 2.0 * np.pi * _assumed_resonance_hz
        zeta = _resonance_damping
        gain = (w_truth**2) / (w_assumed**2)
        num = s * s + 2.0 * zeta * w_assumed * s + w_assumed**2
        den = s * s + 2.0 * zeta * w_truth * s + w_truth**2
        return (gain * num / den).astype(np.complex128)


response_miscalibrated = HydrophoneResponse(
    frequency_response=_MiscalibratedResponse(),
)

# 4. Goody TBL flow noise: the flat response from config 1, but each hydrophone
#    element carries a GoodyFlowNoiseSpectrum self-noise source.  The noise is
#    pressure-domain, so it is coloured by the element's transfer function before
#    it appears at the output.  Streamwise positions are assigned automatically by
#    LinearHydrophoneArray via array_leading_edge_offset_m.
flow_noise = GoodyFlowNoiseSpectrum()

configs = [
    ("Flat", response_flat, None),
    ("4th-order resonant", response_resonant, None),
    ("Miscalibrated resonant", response_miscalibrated, None),
    ("Flat + Flow Noise", response_flat, [flow_noise]),
]

# %%
# Frequency Response Curves
# --------------------------
#
# The magnitude of each transfer function is plotted here in dB as a function of
# frequency on a log axis. This figure is generated analytically — no simulation is
# required — and serves as the reference for interpreting the bearing-time records and
# spectrograms that follow.
#
# A flat 0 dB line means the hydrophone passes that frequency without attenuation.
# Values below 0 dB indicate the hydrophone suppresses that frequency; values above
# 0 dB indicate amplification.  Vertical dashed lines mark the three target tonal
# frequencies (300, 1100 and 1300 Hz): 300 Hz sits in the flat passband as an
# unaffected reference, while 1100 and 1300 Hz bracket the miscalibration ripple
# so the per-tonal effect of calibration error is measurable.
#
# Key observations:
#
# - The **flat** response is a horizontal line at 0 dB.
# - The **fourth-order resonant** model is flat across the passband, peaks at
#   approximately +6 dB just below 1500 Hz, and rolls off steeply (-40 dB/decade)
#   above resonance — the textbook shape of a wideband piezoceramic.
# - The **miscalibrated** correction shows a notch centred at the assumed 1200 Hz
#   resonance and a peak at the true 1500 Hz resonance: the user's inverse
#   overcorrects at 1200 Hz while leaving the true resonance uncancelled.  The
#   notch falls directly between the 1100 and 1300 Hz tonals, suppressing them
#   when source levels are at the detection margin.
# - The **Goody flow noise** panel (row 2) shows the one-sided wall-pressure PSD
#   in dB re 1 µPa²/Hz at the platform tow speed (~2 m/s) and a representative
#   element streamwise position of 10 m.  The ω² low-frequency rise and ω⁻⁵
#   viscous roll-off are both visible across the simulation band.

_f_axis = np.logspace(np.log10(1.0), np.log10(2000.0), 1000)
_tonal_freqs_hz = [300.0, 1100.0, 1300.0]

# Evaluate the Goody noise PSD analytically for display.  Uses a representative
# tow speed of 2 m/s (matching the platform start state) and a streamwise position
# of 10 m (a typical mid-array element location for a 100-element streamer).
_goody_display_state = GroundTruthState(np.array([[0.0], [0.0], [0.0], [2.0], [0.0], [0.0]]))
_goody_display_position_m = 10.0
_goody_psd_db = flow_noise.level_db(
    _f_axis, _goody_display_state, streamwise_position_m=_goody_display_position_m
)

fig_response = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    subplot_titles=[
        "Transfer function magnitude",
        "Goody flow noise PSD (shaped by hydrophone TF)",
    ],
    vertical_spacing=0.12,
)

for label, response, noise_sources in configs:
    H = response.transfer_function(_f_axis)
    mag_db = 20.0 * np.log10(np.abs(H) + 1e-12)
    # Noise-bearing configs share a TF with their noise-free counterpart; dashed
    # lines distinguish them in the legend without hiding the overlap.
    dash = "dash" if noise_sources else "solid"
    fig_response.add_trace(
        go.Scatter(x=_f_axis, y=mag_db, name=label, line=dict(width=2, dash=dash)),
        row=1,
        col=1,
    )

# Row 2: Goody PSD shaped by each config's transfer function.
# Pressure-domain noise passes through H(f) before output, so the effective
# output noise PSD is |H(f)|² * S_goody(f).  In dB this is:
#   S_out_db = S_goody_db + 20·log10(|H(f)|)
# For the flat response |H|=1 so S_out_db = S_goody_db exactly.
for label, response, noise_sources in configs:
    if not noise_sources:
        continue
    H = response.transfer_function(_f_axis)
    shaped_db = _goody_psd_db + 20.0 * np.log10(np.abs(H) + 1e-300)
    fig_response.add_trace(
        go.Scatter(x=_f_axis, y=shaped_db, name=label, line=dict(width=2)),
        row=2,
        col=1,
    )

for freq in _tonal_freqs_hz:
    fig_response.add_vline(
        x=freq,
        line=dict(color="grey", width=1, dash="dash"),
        annotation_text=f"{int(freq)} Hz",
        annotation_position="top right",
        row=1,
        col=1,
    )
    fig_response.add_vline(
        x=freq,
        line=dict(color="grey", width=1, dash="dash"),
        row=2,
        col=1,
    )

_log_range = [np.log10(1.0), np.log10(2000.0)]
_log_tickvals = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]
fig_response.update_xaxes(
    type="log",
    range=_log_range,
    tickmode="array",
    tickvals=_log_tickvals,
    ticktext=[str(v) for v in _log_tickvals],
)
fig_response.update_xaxes(title_text="Frequency (Hz)", row=2, col=1)
fig_response.update_yaxes(title_text="Magnitude (dB)", range=[-40, 12], row=1, col=1)
fig_response.update_yaxes(title_text="Level (dB re 1 µPa²/Hz)", row=2, col=1)

fig_response.update_layout(
    template="plotly_white",
    title="Hydrophone Frequency Response Magnitudes",
    legend=dict(x=0.5, y=-0.15, xanchor="center", yanchor="top", orientation="h"),
    height=600,
)

# Presentation figure: Flat, Resonant, Miscalibrated TFs (row 1) +
# shaped Goody noise for the two noise-bearing configs (row 2).
fig_response_subset = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    subplot_titles=[
        "Transfer function magnitude",
        "Goody flow noise PSD (shaped by hydrophone TF)",
    ],
    vertical_spacing=0.12,
)
for idx in _PRESENTATION_SUBSET_INDICES:
    label, response, *_ = configs[idx]
    H = response.transfer_function(_f_axis)
    mag_db = 20.0 * np.log10(np.abs(H) + 1e-12)
    fig_response_subset.add_trace(
        go.Scatter(x=_f_axis, y=mag_db, name=label, line=dict(width=2, dash="solid")),
        row=1,
        col=1,
    )
for label, response, noise_sources in configs:
    if not noise_sources:
        continue
    H = response.transfer_function(_f_axis)
    mag_db = 20.0 * np.log10(np.abs(H) + 1e-12)
    fig_response_subset.add_trace(
        go.Scatter(x=_f_axis, y=mag_db, name=label, line=dict(width=2, dash="dash")),
        row=1,
        col=1,
    )
for label, response, noise_sources in configs:
    if not noise_sources:
        continue
    H = response.transfer_function(_f_axis)
    shaped_db = _goody_psd_db + 20.0 * np.log10(np.abs(H) + 1e-300)
    fig_response_subset.add_trace(
        go.Scatter(x=_f_axis, y=shaped_db, name=label, line=dict(width=2)),
        row=2,
        col=1,
    )
for freq in _tonal_freqs_hz:
    fig_response_subset.add_vline(
        x=freq,
        line=dict(color="grey", width=1, dash="dash"),
        annotation_text=f"{int(freq)} Hz",
        annotation_position="top right",
        row=1,
        col=1,
    )
    fig_response_subset.add_vline(
        x=freq,
        line=dict(color="grey", width=1, dash="dash"),
        row=2,
        col=1,
    )
_tick_vals = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]
fig_response_subset.update_xaxes(
    type="log",
    range=_log_range,
    tickmode="array",
    tickvals=_tick_vals,
    ticktext=[str(v) for v in _tick_vals],
)
fig_response_subset.update_xaxes(title_text="Frequency (Hz)", row=2, col=1)
fig_response_subset.update_yaxes(title_text="Magnitude (dB)", range=[-40, 12], row=1, col=1)
fig_response_subset.update_yaxes(title_text="Level (dB re 1 µPa²/Hz)", row=2, col=1)
fig_response_subset.update_layout(
    template="plotly_white",
    legend=dict(x=0.5, y=-0.08, xanchor="center", yanchor="top", orientation="h"),
)
save_for_presentation(
    fig_response_subset,
    "frequency_response",
    height=600,
    margin=dict(l=70, r=20, t=30, b=130),
)

# %%
# Signal Transformation Through the Resonant Hydrophone
# ------------------------------------------------------
#
# This figure shows concretely what the hydrophone does to a received signal.
# A synthetic pressure signal is constructed: three tonals at 300, 1100 and
# 1300 Hz plus broadband pink noise.  The resonant hydrophone transfer function
# is then applied in the frequency domain and the input and output are compared
# in both time and frequency.
#
# Key things to observe:
#
# - **Time domain**: the waveform shape changes because the hydrophone boosts
#   the frequency components near its resonance (~1500 Hz) and rolls off those
#   outside the passband.  The 300 Hz tonal (well inside the flat passband) is
#   largely unaffected; the 1100 and 1300 Hz tonals are slightly amplified.
# - **Frequency domain**: the output power spectrum is the input power spectrum
#   multiplied by |H(f)|² at every frequency.  This is the same curve shown in
#   row 1 of the frequency response figure, but now applied to an actual signal
#   so the effect is directly visible.

_demo_sr = 8000.0
_demo_n = 8000  # 1 s at 8 kHz
_demo_t = np.arange(_demo_n) / _demo_sr

_rng_demo = np.random.default_rng(0)
_demo_tonal = sum(np.sin(2.0 * np.pi * f * _demo_t) for f in [300.0, 1100.0, 1300.0])

# Pink noise: colour white noise by 1/sqrt(f) in the frequency domain.
_demo_freqs = np.fft.rfftfreq(_demo_n, d=1.0 / _demo_sr)
_white_fft = np.fft.rfft(_rng_demo.standard_normal(_demo_n))
_pink_weight = np.zeros_like(_demo_freqs)
_pink_weight[_demo_freqs > 0] = 1.0 / np.sqrt(_demo_freqs[_demo_freqs > 0])
_pink = np.fft.irfft(_white_fft * _pink_weight, n=_demo_n)
_pink /= _pink.std()

_demo_pressure = _demo_tonal + 0.3 * _pink  # pressure, arbitrary units

# Apply the resonant hydrophone TF in the frequency domain.
_H_demo = response_resonant.transfer_function(_demo_freqs)
_demo_voltage = np.fft.irfft(np.fft.rfft(_demo_pressure) * _H_demo, n=_demo_n)


def _welch_db(x, n_fft=2048, hop=512, sr=_demo_sr):
    """Return (frequencies_hz, power_db) via Welch's method."""
    window = np.hanning(n_fft)
    n_frames = (len(x) - n_fft) // hop + 1
    frames = np.stack([x[i * hop : i * hop + n_fft] * window for i in range(n_frames)])
    psd = np.mean(np.abs(np.fft.rfft(frames, axis=1)) ** 2, axis=0)
    return np.fft.rfftfreq(n_fft, d=1.0 / sr), 10.0 * np.log10(psd + 1e-30)


_demo_f_welch, _pressure_db = _welch_db(_demo_pressure)
_, _voltage_db = _welch_db(np.real(_demo_voltage))

# Explicit colours shared across both subplots so the single legend applies to both.
_colour_pressure = "#636EFA"  # Plotly default colour 1 (blue)
_colour_voltage = "#EF553B"  # Plotly default colour 2 (orange-red)

_ms = _demo_t * 1000.0
_t_mask = _ms <= 20.0
_f_mask = _demo_f_welch <= 2000.0

_n_configs = len(configs)
_demo_vs = 0.06
fig_demo = make_subplots(
    rows=_n_configs,
    cols=3,
    column_titles=["Frequency response", "Time domain (first 20 ms)", "Power spectrum"],
    vertical_spacing=_demo_vs,
    horizontal_spacing=0.08,
)

for _row, (_label, _response, _noise_sources) in enumerate(configs, start=1):
    _first_row = _row == 1

    # --- Col 1: transfer function magnitude ---
    _H_cfg = _response.transfer_function(_f_axis)
    _mag_db = 20.0 * np.log10(np.abs(_H_cfg) + 1e-12)
    fig_demo.add_trace(
        go.Scatter(
            x=_f_axis,
            y=_mag_db,
            line=dict(width=2, color="steelblue"),
            showlegend=False,
        ),
        row=_row,
        col=1,
    )
    # Mark resonance frequency on the TF panel.
    if isinstance(_response.frequency_response, FourthOrderResonantResponse):
        fig_demo.add_vline(
            x=1500.0,
            line=dict(color="grey", width=1, dash="dot"),
            row=_row,
            col=1,
        )
    elif isinstance(_response.frequency_response, _MiscalibratedResponse):
        fig_demo.add_vline(
            x=_assumed_resonance_hz,
            line=dict(color="grey", width=1, dash="dot"),
            row=_row,
            col=1,
        )

    # --- Col 2 & 3: apply this config's TF to the demo pressure signal ---
    # Use two-sided FFT throughout so the noise synthesis matches the simulator.
    _freqs_full = np.fft.fftfreq(_demo_n, d=1.0 / _demo_sr)
    _H_demo_cfg = _response.transfer_function(_freqs_full)
    _voltage_cfg = np.real(np.fft.ifft(np.fft.fft(_demo_pressure) * _H_demo_cfg))

    if _noise_sources:
        # Synthesise a Goody pressure-noise realisation and pass it through H(f).
        # Scaled to -10 dB SNR relative to the TF-shaped signal so the effect is
        # visible without overwhelming the tonals.
        _goody_psd_full = flow_noise.psd(
            _freqs_full,
            _goody_display_state,
            streamwise_position_m=_goody_display_position_m,
        )
        _sigma_full = np.sqrt(_goody_psd_full * _demo_n * _demo_sr / 2.0)
        _rng_noise = np.random.default_rng(seed=_row * 17)
        _noise_spec = (
            _sigma_full
            * (_rng_noise.standard_normal(_demo_n) + 1j * _rng_noise.standard_normal(_demo_n))
            / np.sqrt(2.0)
        )
        _noise_voltage = np.real(np.fft.ifft(_noise_spec * _H_demo_cfg))
        _noise_voltage *= np.std(_voltage_cfg) / (np.std(_noise_voltage) * np.sqrt(10.0))
        _voltage_cfg = _voltage_cfg + _noise_voltage

    _, _voltage_db_cfg = _welch_db(_voltage_cfg)

    # Time domain.
    fig_demo.add_trace(
        go.Scatter(
            x=_ms[_t_mask],
            y=_demo_pressure[_t_mask],
            name="Input pressure",
            line=dict(width=1.5, color=_colour_pressure),
            showlegend=_first_row,
        ),
        row=_row,
        col=2,
    )
    fig_demo.add_trace(
        go.Scatter(
            x=_ms[_t_mask],
            y=_voltage_cfg[_t_mask],
            name="Output voltage",
            line=dict(width=1.5, color=_colour_voltage),
            showlegend=_first_row,
        ),
        row=_row,
        col=2,
    )

    # Power spectrum.
    fig_demo.add_trace(
        go.Scatter(
            x=_demo_f_welch[_f_mask],
            y=_pressure_db[_f_mask],
            name="Input pressure",
            line=dict(width=1.5, color=_colour_pressure),
            opacity=0.7,
            showlegend=False,
        ),
        row=_row,
        col=3,
    )
    fig_demo.add_trace(
        go.Scatter(
            x=_demo_f_welch[_f_mask],
            y=_voltage_db_cfg[_f_mask],
            name="Output voltage",
            line=dict(width=1.5, color=_colour_voltage),
            opacity=0.7,
            showlegend=False,
        ),
        row=_row,
        col=3,
    )

# Axis labels on bottom row and left column only.
_demo_log_axis = dict(
    type="log",
    range=[np.log10(1.0), np.log10(2000.0)],
    tickmode="array",
    tickvals=_log_tickvals,
    ticktext=[str(v) for v in _log_tickvals],
)
fig_demo.update_xaxes(title_text="", **_demo_log_axis, col=1)
fig_demo.update_xaxes(title_text="Frequency (Hz)", **_demo_log_axis, row=_n_configs, col=1)
fig_demo.update_yaxes(title_text="Magnitude (dB)", range=[-50, 12], col=1)
fig_demo.update_xaxes(title_text="Time (ms)", row=_n_configs, col=2)
fig_demo.update_yaxes(title_text="Amplitude (arb.)", col=2)
fig_demo.update_xaxes(**_demo_log_axis, col=3)
fig_demo.update_xaxes(title_text="Frequency (Hz)", row=_n_configs, col=3)
fig_demo.update_yaxes(title_text="Power (dB)", col=3)

# Add row labels manually so their x position is fully controllable.
_demo_row_height = (1.0 - (_n_configs - 1) * _demo_vs) / _n_configs
for _row_i, (_label, *_) in enumerate(configs):
    _row_top = 1.0 - _row_i * (_demo_row_height + _demo_vs)
    _row_center = _row_top - _demo_row_height / 2.0
    fig_demo.add_annotation(
        text=_label,
        xref="paper",
        yref="paper",
        x=1.02,
        y=_row_center,
        textangle=90,
        showarrow=False,
        xanchor="left",
        yanchor="middle",
        font=dict(size=16),
    )

fig_demo.update_layout(
    template="plotly_white",
    title="Signal transformation through each hydrophone configuration",
    legend=dict(x=0.5, y=-0.06, xanchor="center", yanchor="top", orientation="h"),
    height=250 * _n_configs,
)

save_for_presentation(
    fig_demo,
    "signal_transformation",
    width=1200,
    height=250 * _n_configs,
    margin=dict(l=80, r=80, t=60, b=80),
)

# %%
# Results: Per-Tonal Response Magnitude
# --------------------------------------
#
# Detection counting is largely insensitive to passband shape because CFAR
# thresholds adapt to the local noise floor.  A clean numerical view of how
# each configuration shapes the received signal comes from evaluating the
# response curve directly at each tonal frequency: ``20 log10 |H(f)|`` is the
# exact gain (in dB) applied to a tonal at frequency ``f`` relative to the
# Flat reference.  This is the quantity that drives any per-tonal SNR shift
# and is independent of the simulator's noise realisation, beamforming
# choices, or detector behaviour.
#
# The expected pattern at −20% miscalibration (assumed fr = 1200 Hz):
#
# - 300 Hz: identical across all configs (sits in the flat passband, well
#   below the resonance region).
# - 1100 Hz: slightly lifted in the resonant config; notched in the
#   miscalibrated config (sits just below the 1200 Hz notch centre).
# - 1300 Hz: lifted by several dB in the resonant config (rising flank of
#   the true 1500 Hz resonance); notched in the miscalibrated config (sits
#   just above the 1200 Hz notch centre).
#
# The symmetric notching of 1100 and 1300 Hz in the miscalibrated config is
# the headline calibration-error signature: both tonals are suppressed by the
# same miscalibrated inverse, causing detection misses that the resonant and
# flat configs do not exhibit.

# Evaluate every config's response at every tonal frequency (exact, analytic).
_tonal_freqs_arr = np.asarray(_tonal_freqs_hz, dtype=float)
_per_tonal_gain_db = np.zeros((len(configs), len(_tonal_freqs_hz)))
for ci, (_, hph, *__) in enumerate(configs):
    H = hph.transfer_function(_tonal_freqs_arr)
    _per_tonal_gain_db[ci, :] = 20.0 * np.log10(np.abs(H) + 1e-300)
    print(
        f"{configs[ci][0]}: gain @ "
        + ", ".join(
            f"{int(f)} Hz = {_per_tonal_gain_db[ci, ti]:+.2f} dB"
            for ti, f in enumerate(_tonal_freqs_hz)
        )
    )

# Full grouped bar chart: all configs.
fig_tonal_gain = go.Figure()
for ci, (label, *_) in enumerate(configs):
    fig_tonal_gain.add_trace(
        go.Bar(
            name=label,
            x=[f"{int(f)} Hz" for f in _tonal_freqs_hz],
            y=_per_tonal_gain_db[ci, :],
            text=[f"{v:+.2f}" for v in _per_tonal_gain_db[ci, :]],
            textposition="outside",
        )
    )
fig_tonal_gain.update_layout(
    template="plotly_white",
    title="Per-Tonal Response Magnitude by Hydrophone Configuration",
    xaxis=dict(title="Tonal frequency"),
    yaxis=dict(title="Response magnitude (dB)"),
    barmode="group",
    legend=dict(x=0.5, y=-0.2, xanchor="center", yanchor="top", orientation="h"),
    height=500,
)

# Presentation subset: Flat / Resonant / Miscalibrated only.
fig_tonal_gain_subset = go.Figure()
for idx in _PRESENTATION_SUBSET_INDICES:
    label, *_ = configs[idx]
    fig_tonal_gain_subset.add_trace(
        go.Bar(
            name=label,
            x=[f"{int(f)} Hz" for f in _tonal_freqs_hz],
            y=_per_tonal_gain_db[idx, :],
            text=[f"{v:+.2f}" for v in _per_tonal_gain_db[idx, :]],
            textposition="outside",
        )
    )
fig_tonal_gain_subset.update_layout(
    template="plotly_white",
    xaxis=dict(title="Tonal frequency"),
    yaxis=dict(title="Response magnitude (dB)"),
    barmode="group",
    legend=dict(x=0.5, y=-0.15, xanchor="center", yanchor="top", orientation="h"),
)
save_for_presentation(
    fig_tonal_gain_subset,
    "per_tonal_gain",
    margin=dict(l=70, r=20, t=15, b=130),
)
quit()

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


def _make_platform(
    hydrophone_response: HydrophoneResponse,
    noise_sources: list | None = None,
) -> TowedArrayPlatform:
    """Build and advance a :class:`~bluepebble.platform.TowedArrayPlatform`.

    Each call returns an independent platform that has been moved through the full
    simulation timeline. Using separate instances ensures each simulator has its own
    element state history and its own hydrophone transfer function.

    Parameters
    ----------
    hydrophone_response : HydrophoneResponse
        Electro-acoustic response model to attach to every array element.
    noise_sources : list of SensorNoiseSpectrum, optional
        Self-noise sources to attach to every element. When provided, the array is
        also constructed with ``array_leading_edge_offset_m=2.0`` so that flow-type
        sources receive valid streamwise positions.

    Returns
    -------
    TowedArrayPlatform
        Platform advanced to the final simulation timestamp.

    """
    elements = [
        Hydrophone(response=hydrophone_response, noise_sources=noise_sources or [])
        for _ in range(num_sensors)
    ]
    leading_edge = 2.0 if noise_sources else None
    sensor_array = LinearHydrophoneArray(
        elements=elements,
        element_spacing_m=sensor_spacing_m,
        array_leading_edge_offset_m=leading_edge,
    )
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


platforms = [_make_platform(response, noise_sources) for _, response, noise_sources in configs]

# %%
# Ground Truth Setup and Generation
# ----------------------------------
#
# One target is generated here. It radiates three tonal components at 300 Hz, 1100 Hz,
# and 1300 Hz with equal source levels of 100 dB re 1 µPa. The 300 Hz tonal sits in the
# flat passband and acts as a reference; 1100 and 1300 Hz bracket the miscalibration
# ripple peak so calibration error has a measurable per-tonal SNR signature.

target_start_vector = np.array([6000.0, -4.0, 4000.0, -2.0, -5.0, 0.0])
target_position_mapping = [0, 2, 4]
target_velocity_mapping = [1, 3, 5]
target_transition_model = CombinedLinearGaussianTransitionModel(
    [ConstantVelocity(0.0), ConstantVelocity(0.0), ConstantVelocity(0.0)]
)

target_amplitudes_upa = np.full(3, 10 ** (85.0 / 20.0))
# Tonals chosen to bracket the miscalibration notch at ~1200 Hz:
#   300 Hz  - flat passband reference, unaffected by miscalibration
#   1100 Hz - just below the notch centre; both resonant and miscalibrated lift slightly
#   1300 Hz - just above the notch centre; resonant lifts, miscalibrated notches
target_frequencies_hz = np.array([300.0, 1100.0, 1300.0])
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
save_for_presentation(fig_world, "world_view", width=550, height=550)

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

sampling_rate_hz = 4000.0
frame_len = 4000  # 1 s frames at 4 kHz, matches old 1 s frames at 500 Hz
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
# Four simulators are created — one per hydrophone configuration. Each is paired with its
# own platform so that the correct hydrophone transfer function is applied during signal
# generation. A shared CA-CFAR + peak-selection detector chain is applied identically to
# all outputs so that detection differences can be attributed to the hydrophone
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
# All detector pipelines are executed sequentially. The SNR history and flat
# detection lists are retained for plotting.

all_detections = []
snr_maps = []

for (label, *_), detector in zip(configs, detectors, strict=True):
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
# change in the spectral envelope across panels. Percentile-based colour limits
# are used to keep the scale sensitive to this spectral variation.
#
# - **Flat**: all three tonals appear with equal strength.
# - **4th-order resonant**: 1100 and 1300 Hz sit on the rising flank of the resonance
#   peak and are amplified relative to 300 Hz; noise floor is lifted near 1500 Hz.
# - **Miscalibrated**: the residual ripple amplifies tonals just below ~1240 Hz and
#   attenuates them above; with a +20% calibration error this should be visible as a
#   ~3–5 dB asymmetry between the 1100 Hz and 1300 Hz tonals.
# - **Goody flow noise**: spectral shape matches Flat but with a raised broadband noise
#   floor driven by TBL pressure fluctuations; tonal SNR is reduced accordingly.

bluepebble.set_seed(seed)

subplot_titles = [label for label, *_ in configs]

fig_spec = make_subplots(
    rows=len(configs),
    cols=1,
    shared_xaxes=True,
    shared_yaxes=True,
    subplot_titles=subplot_titles,
    vertical_spacing=0.08,
)

full_signals = []
for row, (platform, (_, *__)) in enumerate(zip(platforms, configs, strict=True), start=1):
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
    full_signals.append(full_signal)

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
for row in range(1, len(configs)):
    fig_spec.update_xaxes(title_text="", row=row, col=1, showticklabels=False)

fig_spec.update_layout(
    template="plotly_white",
    autosize=True,
    height=int(np.clip(260 * len(configs), 700, 2200)),
    showlegend=False,
    title="Received Spectrograms",
    margin=dict(r=100),
)
# Presentation spectrograms: Flat, Resonant, Miscalibrated.
_spec_subset_titles = [subplot_titles[i] for i in _PRESENTATION_SUBSET_INDICES]

fig_spec_subset = make_subplots(
    rows=len(_PRESENTATION_SUBSET_INDICES),
    cols=1,
    shared_xaxes=True,
    shared_yaxes=True,
    subplot_titles=_spec_subset_titles,
    vertical_spacing=0.04,
)

for row, idx in enumerate(_PRESENTATION_SUBSET_INDICES, start=1):
    plot_spectrogram(
        signal=full_signals[idx],
        sr=int(sampling_rate_hz),
        n_fft=2048,
        hop_length=512,
        # Crop the presentation spectrogram to the resonance region.
        # This excludes the bass-heavy LF that otherwise dominates the colour
        # scale and lets the calibration ripple (centred ~1240 Hz) fill the panel.
        y_lim=(200.0, 2000.0),
        yaxis_format="Hz",
        db_reference="absolute",
        showscale=False,
        fig=fig_spec_subset,
        row=row,
        col=1,
    )
    # for freq in _tonal_freqs_hz:
    #     # Only annotate tonals that fall inside the cropped y-range.
    #     if 200.0 <= freq <= 2000.0:
    #         fig_spec_subset.add_hline(
    #             y=freq,
    #             line=dict(color="white", width=1, dash="dash"),
    #             row=row,
    #             col=1,
    #         )

_subset_z = np.concatenate(
    [
        np.asarray(t.z, dtype=float).ravel()
        for t in fig_spec_subset.data
        if getattr(t, "type", None) == "heatmap"
    ]
)
_subset_finite_z = _subset_z[np.isfinite(_subset_z)]
# Tighter percentile clip than the full-band version: the cropped band has a much
# narrower intensity distribution, and clipping the bottom 25% pushes the colour
# range upward into the region where tonals and resonance lift live.
apply_shared_colourscale(
    fig_spec_subset,
    zmin=float(np.percentile(_subset_finite_z, 25)),
    zmax=float(np.percentile(_subset_finite_z, 99)),
    colorbar=dict(
        title=dict(text="Intensity (dB)", side="right"),
        x=1.02,
        y=0.5,
        yanchor="middle",
        len=1.0,
        thickness=24,
    ),
)

for row in range(1, len(_PRESENTATION_SUBSET_INDICES)):
    fig_spec_subset.update_xaxes(title_text="", row=row, col=1, showticklabels=False)

fig_spec_subset.update_layout(
    template="plotly_white",
    autosize=True,
    showlegend=False,
)
save_for_presentation(
    fig_spec_subset,
    "received_spectrograms",
    margin=PRESENTATION_FIG_MARGIN_COLORBAR,
)


# %%
# Results: LOFAR Tonal Detection
# --------------------------------
#
# A narrowband fixed-threshold detector is applied to the omnidirectional spectrogram
# (LOFAR display) of each configuration.  Unlike CA-CFAR, which normalises against the
# local bearing-space noise floor and is therefore blind to uniform frequency shaping,
# a fixed threshold set from a calibrated noise-floor estimate is sensitive to the
# absolute power in each frequency bin.  A hydrophone response notch that attenuates a
# tonal below the threshold directly causes a missed detection — which is the mechanism
# that motivates calibrating hydrophone responses in narrowband sonar systems.
#
# The noise floor is estimated from the flat configuration: median power across non-tonal
# frequency bins gives a robust per-frame baseline.  A fixed margin is then added;
# ``_lofar_detection_margin_db`` can be tuned so that the flat/resonant configurations
# detect the tonals while the miscalibrated notch causes misses at 1100 Hz.
#
# Results with the current configuration:
#
# - **300 Hz**: all configurations detect at similar rates, confirming
#   this tonal is unaffected by any of the frequency responses in the simulation band.
# - **1100 Hz**: the resonant model detects 35.5 % of frames; the miscalibrated model
#   detects 0.2 % — a >99 % reduction caused directly by the −6 dB notch placed by
#   the erroneous calibration correction at that frequency.
# - **1300 Hz**: the resonant model detects 36.1 % of frames; the miscalibrated model
#   detects 5.7 % — an 84 % reduction from the shallower edge of the same notch.
#
# For comparison, a CA-CFAR detector applied to the broadband bearing-time record
# produces exactly 120 detections for every configuration.  CA-CFAR normalises against
# the local bearing-space noise floor, so the hydrophone response — which scales signal
# and noise identically — cancels in the SNR ratio and is invisible to the detector.
# This confirms that narrowband calibrated detection, not adaptive broadband CFAR, is
# the right tool for assessing the impact of hydrophone response errors.
#
# The contrast between the resonant and miscalibrated columns at 1100 and 1300 Hz
# is the central result: identical physical hydrophones, identical source levels, only
# the calibration assumption differs.

_lofar_n_fft = 2048
_lofar_hop = 512
_lofar_sr = int(sampling_rate_hz)
_lofar_window = np.hanning(_lofar_n_fft)
_lofar_freqs = np.fft.rfftfreq(_lofar_n_fft, d=1.0 / _lofar_sr)


def _lofar_power_db(signal: np.ndarray) -> np.ndarray:
    """Compute per-bin power spectrogram in dB.  Returns (n_bins, n_frames)."""
    n_frames = (len(signal) - _lofar_n_fft) // _lofar_hop + 1
    frames = np.stack(
        [
            signal[i * _lofar_hop : i * _lofar_hop + _lofar_n_fft] * _lofar_window
            for i in range(n_frames)
        ]
    )
    spectra = np.fft.rfft(frames, axis=1)
    return 10.0 * np.log10(np.abs(spectra).T ** 2 + 1e-30)  # (n_bins, n_frames)


_lofar_powers = [_lofar_power_db(sig) for sig in full_signals]

# Noise floor: median power across non-tonal bins in the flat (index 0) config.
_tonal_bin_indices = [int(np.argmin(np.abs(_lofar_freqs - f))) for f in target_frequencies_hz]
_non_tonal_mask = np.ones(len(_lofar_freqs), dtype=bool)
for _b in _tonal_bin_indices:
    _non_tonal_mask[max(0, _b - 3) : _b + 4] = False
_noise_floor_db = float(np.median(_lofar_powers[0][_non_tonal_mask, :]))

# Fixed detection threshold: noise floor + margin.
# Increase _lofar_detection_margin_db if all configs still detect;
# decrease it if none detect.
_lofar_detection_margin_db = 15.0
_lofar_threshold_db = _noise_floor_db + _lofar_detection_margin_db

_n_lofar_frames = _lofar_powers[0].shape[1]

# Count detected frames per tonal per config.
_lofar_detection_counts: list[dict[int, int]] = []
for _pdb in _lofar_powers:
    _counts: dict[int, int] = {}
    for _tonal_hz, _bin_idx in zip(target_frequencies_hz, _tonal_bin_indices, strict=True):
        _counts[int(_tonal_hz)] = int(np.sum(_pdb[_bin_idx, :] > _lofar_threshold_db))
    _lofar_detection_counts.append(_counts)

for (label, *_), _counts in zip(configs, _lofar_detection_counts):
    print(
        f"{label}: "
        + ", ".join(
            f"{f} Hz = {c}/{_n_lofar_frames} ({100 * c / _n_lofar_frames:.1f}%)"
            for f, c in _counts.items()
        )
    )

_lofar_detection_rates = [
    {f: 100.0 * c / _n_lofar_frames for f, c in _counts.items()}
    for _counts in _lofar_detection_counts
]

fig_lofar_det = go.Figure()
for ci, (label, *_) in enumerate(configs):
    _rates = _lofar_detection_rates[ci]
    fig_lofar_det.add_trace(
        go.Bar(
            name=label,
            x=[f"{int(f)} Hz" for f in target_frequencies_hz],
            y=[_rates[int(f)] for f in target_frequencies_hz],
            text=[f"{_rates[int(f)]:.1f}%" for f in target_frequencies_hz],
            textposition="outside",
        )
    )
fig_lofar_det.update_layout(
    template="plotly_white",
    title="LOFAR Tonal Detection Rate (Fixed Threshold)",
    xaxis=dict(title="Tonal frequency"),
    yaxis=dict(title="Detection rate (%)", range=[0, 105]),
    barmode="group",
    legend=dict(x=0.5, y=-0.2, xanchor="center", yanchor="top", orientation="h"),
    height=500,
)

# %%
# Results: Bearing-Time Records by Hydrophone Configuration
# ----------------------------------------------------------
#
# The SNR maps are stacked vertically. Each row uses the same colour scale so that
# tonal brightness can be compared directly across configurations. The vertical layout
# makes it straightforward to trace how each filter reshapes the three-tonal signature.
#
# Note: the three non-flow-noise configurations produce identical CA-CFAR detection counts.
# This is expected — CA-CFAR normalises against the local bearing-space noise floor,
# so a hydrophone response that scales signal and noise identically cancels in the SNR
# ratio and is invisible to the detector.  The BTR is retained here to show what the
# analyst's display looks like; the LOFAR detection section above demonstrates where
# the hydrophone response effect actually manifests.
#

fig_btr = make_subplots(
    rows=len(configs),
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
for row in range(1, len(configs)):
    fig_btr.update_xaxes(title_text="", row=row, col=1, showticklabels=False)

fig_btr.update_layout(
    template="plotly_white",
    autosize=True,
    height=int(np.clip(260 * len(configs), 700, 2200)),
    showlegend=False,
    title="SNR Maps",
    margin=dict(r=100),
)
# Presentation BTR: Flat, Resonant, Miscalibrated.
_btr_subset_titles = [subplot_titles[i] for i in _PRESENTATION_SUBSET_INDICES]

fig_btr_subset = make_subplots(
    rows=len(_PRESENTATION_SUBSET_INDICES),
    cols=1,
    shared_xaxes=True,
    shared_yaxes=True,
    subplot_titles=_btr_subset_titles,
    vertical_spacing=0.04,
)

for row, idx in enumerate(_PRESENTATION_SUBSET_INDICES, start=1):
    plot_btr(
        data=snr_maps[idx],
        detections=all_detections[idx],
        timesteps=timesteps,
        steering_azimuths=steering_azimuths_deg,
        fig=fig_btr_subset,
        row=row,
        col=1,
    )

apply_shared_colourscale(
    fig_btr_subset,
    colorbar=dict(
        title=dict(text="SNR (dB)", side="right"),
        x=1.02,
        y=0.5,
        yanchor="middle",
        len=1.0,
        thickness=24,
    ),
)

for row in range(1, len(_PRESENTATION_SUBSET_INDICES)):
    fig_btr_subset.update_xaxes(title_text="", row=row, col=1, showticklabels=False)

fig_btr_subset.update_layout(
    template="plotly_white",
    autosize=True,
    showlegend=False,
)
save_for_presentation(
    fig_btr_subset,
    "btr_maps",
    width=700,
    height=820,
    margin=PRESENTATION_FIG_MARGIN_COLORBAR,
)
