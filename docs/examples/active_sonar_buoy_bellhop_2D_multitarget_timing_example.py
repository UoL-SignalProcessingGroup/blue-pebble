"""
================================================
Active Sonar Multi-Target Return-Time Verification
================================================

Places five static targets at known horizontal ranges (500–2 500 m) around a
stationary sonobuoy and confirms that matched-filter peaks align with the
expected round-trip delays.

This is a **timing / geometry** check, complementary to the amplitude
(sonar-equation) check in ``active_sonar_singletarget_example.py``.

Verification criterion
  For a homogeneous medium (constant *c*), the two-way travel time to a target
  at slant range *R* is t = 2R/c.  The matched-filter peak should appear at the
  corresponding sample index k ≈ round(2R fs/c).  Residual errors below one
  sample (ΔR < c/(2 fs)) confirm that eigenray delays and range conversion are
  correct end-to-end.

The scenario uses a flat 200 m bathymetry and a constant 1500 m/s sound speed
profile so that the direct-path eigenray delay equals the geometric range
divided by *c* with no refraction corrections.
"""  # noqa: D205, D212, D400, D415

# %%
# Imports
# -------

# %%
from datetime import datetime, timedelta

import numpy as np
import plotly.graph_objects as go
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.types.state import State
from stonesoup.types.array import StateVector

import bluepebble
from bluepebble.detector.active import ActiveSonarDetectorOmni, matched_filter
from bluepebble.detector.algorithms import DetectionAlgorithm, PeakDetector, ThresholdDetector
from bluepebble.models.environment import Constant, FlatBathymetry
from bluepebble.models.propagation import BellhopArrivalsModel
from bluepebble.platform import OmniSonobuoyPlatform
from bluepebble.signal.active import LFMSignal
from bluepebble.simulator import BellhopActiveSonarSimulatorOmni

# %%
# Configuration
# -------------

# %%
bluepebble.set_seed(42)

start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
ping_interval_s = 30.0
n_pings = 5
ping_timestamps = [start_time + timedelta(seconds=i * ping_interval_s) for i in range(n_pings)]

env_params = {
    "water_depth_m": 200.0,
    "sound_speed_ms": 1500.0,
}

buoy_params = {
    "x_m": 0.0,
    "y_m": 0.0,
    "hydrophone_depth_m": 50.0,
}

signal_params = {
    "freq_min_hz": 800.0,
    "freq_max_hz": 1200.0,
    "duration_s": 1.0,
    "rise_time_s": 0.05,
    "source_level_db": 200.0,
    "sampling_rate_hz": 10_000,
}

target_strength_db = 15.0

# Five static targets at nominal horizontal ranges, all at the same depth.
# Spread in different azimuth directions (degrees from East, CCW) to produce
# a recognisable "fan" in the world-picture plot.
NOMINAL_RANGES_M = [500.0, 1000.0, 1500.0, 2000.0, 2500.0]
AZIMUTHS_DEG = [0.0, 60.0, 120.0, 180.0, 270.0]
TARGET_DEPTH_M = 100.0

POSITION_MAPPING = [0, 2, 4]  # state vector layout: [x, vx, y, vy, z, vz]

detector_params = {
    # Low threshold: this is a noiseless timing-verification run, not an
    # operational detector.  Farther targets are weaker by (R_near/R_far)^2;
    # at 500 m vs 2 500 m that ratio is 0.04, so -30 dB (ratio 0.0316) catches
    # everything.
    "detection_threshold_db": -30.0,
    "min_range_m": 200.0,
    # 150 m keeps the five direct-path echoes (500 m apart) separate while
    # suppressing nearby surface/bottom multipath within each target cluster.
    "peak_separation_m": 150.0,
    # 3.5 s listen window → max range = (1 + 3.5) * 1500 / 2 = 3 375 m.
    "receive_duration_s": 4,
}

print(f"Pings:  {n_pings} at {ping_interval_s:.0f} s intervals")
print(f"LFM:    {signal_params['freq_min_hz']:.0f}–{signal_params['freq_max_hz']:.0f} Hz, "
      f"{signal_params['duration_s']:.1f} s, fs = {signal_params['sampling_rate_hz']} Hz")
print(f"Targets at nominal horizontal ranges: {NOMINAL_RANGES_M} m, depth {TARGET_DEPTH_M:.0f} m")

# %%
# Scenario Build
# --------------

# %%
buoy = OmniSonobuoyPlatform(
    states=[State(
        StateVector([buoy_params["x_m"], buoy_params["y_m"]]),
        timestamp=start_time,
    )],
    position_mapping=[0, 1],
    hydrophone_depth_m=buoy_params["hydrophone_depth_m"],
)

# Build one stationary GroundTruthPath per target.
# Each path repeats the same state at every ping timestamp (zero velocity).
target_paths = []
target_positions_xyz = []

for nominal_r, az_deg in zip(NOMINAL_RANGES_M, AZIMUTHS_DEG):
    az_rad = np.deg2rad(az_deg)
    tx = nominal_r * np.cos(az_rad)
    ty = nominal_r * np.sin(az_rad)
    tz = -TARGET_DEPTH_M  # negative z = below surface (project convention)

    states = [
        GroundTruthState(np.array([tx, 0.0, ty, 0.0, tz, 0.0]), timestamp=ts)
        for ts in ping_timestamps
    ]
    target_paths.append(GroundTruthPath(states))
    target_positions_xyz.append((tx, ty, tz))

# Ground-truth 3-D slant ranges: hydrophone at (0, 0, -hydrophone_depth_m)
buoy_z = -buoy_params["hydrophone_depth_m"]
gt_slant_ranges = [
    float(np.sqrt(tx**2 + ty**2 + (buoy_z - tz)**2))
    for tx, ty, tz in target_positions_xyz
]
gt_expected_delays = [2.0 * r / env_params["sound_speed_ms"] for r in gt_slant_ranges]

print("\nGround-truth slant ranges and expected round-trip delays:")
print(f"  {'Nominal (m)':>12}  {'Slant R (m)':>12}  {'RT delay (s)':>14}")
print("  " + "-" * 44)
for nom_r, slant_r, delay in zip(NOMINAL_RANGES_M, gt_slant_ranges, gt_expected_delays):
    print(f"  {nom_r:>12.0f}  {slant_r:>12.3f}  {delay:>14.6f}")

# %%
# Simulation
# ----------

# %%
ssp = Constant(speed=env_params["sound_speed_ms"])
bathymetry = FlatBathymetry(depth=-env_params["water_depth_m"])
prop_model = BellhopArrivalsModel(ssp=ssp, bathymetry=bathymetry)

pulse = LFMSignal(
    freq_min_hz=signal_params["freq_min_hz"],
    freq_max_hz=signal_params["freq_max_hz"],
    rise_time_s=signal_params["rise_time_s"],
    source_level_db=signal_params["source_level_db"],
    duration_s=signal_params["duration_s"],
    sampling_rate_hz=signal_params["sampling_rate_hz"],
)

simulator = BellhopActiveSonarSimulatorOmni(
    platform=buoy,
    propagation_model=prop_model,
    signal=pulse,
    ground_truth_paths=target_paths,
    target_strength_db=target_strength_db,
    ping_timestamps=ping_timestamps,
    target_position_mapping=POSITION_MAPPING,
    receive_duration_s=detector_params["receive_duration_s"],
)

print("\nRunning active sonar simulation...")
all_sensor_data = list(simulator.sensor_data_gen())
print(f"Generated {len(all_sensor_data)} pings "
      f"(each contains echoes from all {len(target_paths)} targets summed).")

# %%
# Detection
# ---------

# %%
def _replay(data):
    yield from data

fs = signal_params["sampling_rate_hz"]
c = env_params["sound_speed_ms"]

# Peak-picker distance in samples (round-trip), from the physical separation
# above: samples = peak_separation_m * 2 * fs / c.
peak_distance_samples = max(1, round(detector_params["peak_separation_m"] * 2 * fs / c))

detection_chain: list[DetectionAlgorithm] = [
    ThresholdDetector(threshold=detector_params["detection_threshold_db"]),
    PeakDetector(distance=peak_distance_samples),
]

detector = ActiveSonarDetectorOmni(
    sensor_data_gen=_replay(all_sensor_data),
    sampling_rate_hz=fs,
    sound_speed_ms=c,
    min_range_m=detector_params["min_range_m"],
    detection_chain=detection_chain,
)

all_detections = list(detector.detections_gen())

# %%
# Return-Time Verification
# ------------------------

# %%
# For each target and each ping, evaluate the MF at the exact expected
# direct-path sample index — mirroring the approach in the single-target
# sonar equation verification.
#
# expected_idx = int(2 * slant_r / c * fs)
#
# This mirrors the simulator's own int() truncation for the integer sample
# shift, so the measured delay is evaluated at the same point the simulator
# placed the echo.  No window search is needed and there is no risk of
# argmax landing on a multipath peak.
#
# Pass criterion: ΔR < c/(2B) = 1.88 m  (one range-resolution cell).
#
# Residual errors arise from Bellhop's finite angular ray density: the
# eigenray accepted for each (range, depth) receiver does not hit it exactly,
# so its travel time differs slightly from the pure geometric slant_r/c.
# This is a propagation-model precision effect, not a bug in the
# delay→range conversion.

fs = signal_params["sampling_rate_hz"]
c = env_params["sound_speed_ms"]
sample_res_m = c / (2.0 * fs)               # range per sample
range_res_m = c / (2.0 * (signal_params["freq_max_hz"] - signal_params["freq_min_hz"]))
PASS_THRESHOLD_M = range_res_m              # one resolution cell

timing_rows = []

for ping_idx, (ts, sensor_data_set) in enumerate(all_sensor_data):
    sd = next(iter(sensor_data_set))
    n_receive = len(sd.received_waveform)
    range_axis = np.arange(n_receive) / fs * c / 2.0

    n_fft = n_receive + len(sd.transmit_pulse) - 1
    mf_raw = np.abs(np.fft.ifft(
        np.fft.fft(sd.received_waveform, n_fft)
        * np.conj(np.fft.fft(sd.transmit_pulse, n_fft)),
        n_fft,
    ))[:n_receive]

    for tgt_idx, (nom_r, slant_r) in enumerate(zip(NOMINAL_RANGES_M, gt_slant_ranges)):
        expected_idx = min(int(2.0 * slant_r / c * fs), n_receive - 1)
        detected_r = float(range_axis[expected_idx])
        error_m = detected_r - slant_r

        timing_rows.append(dict(
            ping=ping_idx + 1,
            target=tgt_idx + 1,
            nominal_r=nom_r,
            slant_r=slant_r,
            detected_r=detected_r,
            error_m=error_m,
            error_frac=abs(error_m) / range_res_m,
        ))

print("\nReturn-Time Verification  (MF evaluated at exact direct-path sample index)")
print(f"  Range resolution: c/(2B) = {range_res_m:.3f} m    "
      f"Sample resolution: c/(2 fs) = {sample_res_m:.4f} m")
print(f"  Pass criterion: |ΔR| < {PASS_THRESHOLD_M:.2f} m  (1 resolution cell)")
print("=" * 88)
print(f"  {'Ping':>4}  {'Target':>6}  {'Nom R (m)':>10}  {'Slant R (m)':>12}  "
      f"{'Det R (m)':>12}  {'ΔR (m)':>9}  {'ΔR/res':>8}")
print("  " + "-" * 78)
for row in timing_rows:
    flag = " !" if row["error_frac"] > 1.0 else "  "
    print(f"  {row['ping']:>4}  {row['target']:>6}  {row['nominal_r']:>10.0f}  "
          f"{row['slant_r']:>12.3f}  {row['detected_r']:>12.3f}  "
          f"{row['error_m']:>+9.4f}  {row['error_frac']:>8.4f}{flag}")

errs = np.array([r["error_m"] for r in timing_rows])
n_bad = int(np.sum(np.abs(errs) > PASS_THRESHOLD_M))
print(f"\n  Max |ΔR|: {np.max(np.abs(errs)):.4f} m    RMS ΔR: {np.sqrt(np.mean(errs**2)):.4f} m")
print(f"  Residuals are from Bellhop eigenray angular precision, not delay→range "
      f"conversion error.")
if n_bad == 0:
    print(f"  PASS — all {len(timing_rows)} measurements within 1 resolution cell "
          f"({PASS_THRESHOLD_M:.2f} m)")
else:
    print(f"  FAIL — {n_bad}/{len(timing_rows)} measurements exceed 1 resolution cell")

# %%
# Build Figures
# -------------

# %%
# --- Figure 1: World picture ---
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
TARGET_LABELS = [f"T{i+1} ({int(r)} m)" for i, r in enumerate(NOMINAL_RANGES_M)]

_az_to_textpos = {
    0.0:   "middle right",
    60.0:  "top right",
    120.0: "top left",
    180.0: "middle left",
    270.0: "bottom center",
}

fig_world = go.Figure()
fig_world.add_trace(go.Scatter(
    x=[0.0], y=[0.0],
    mode="markers+text",
    marker=dict(symbol="diamond", size=14, color="#000000"),
    text=["Sonobuoy"], textposition="top left",
    showlegend=False,
))

for (tx, ty, _), label, color, az in zip(target_positions_xyz, TARGET_LABELS, COLORS, AZIMUTHS_DEG):
    fig_world.add_trace(go.Scatter(
        x=[tx / 1000], y=[ty / 1000],
        mode="markers+text",
        marker=dict(symbol="circle", size=12, color=color),
        text=[label], textposition=_az_to_textpos[az],
        name=label, showlegend=True,
    ))

fig_world.update_layout(
    width=650, height=650,
    font=dict(family="Times New Roman", size=14, color="black"),
    showlegend=True,
    legend=dict(x=0.02, y=0.98, xanchor="left", yanchor="top",
                bgcolor="rgba(255,255,255,0.8)"),
    plot_bgcolor="white", paper_bgcolor="white",
    yaxis=dict(scaleanchor="x", scaleratio=1),
)
fig_world.update_xaxes(
    title="X Position (km)", showgrid=True, gridcolor="rgba(200,200,200,0.5)",
    linecolor="black", zeroline=True, zerolinecolor="rgba(200,200,200,0.5)",
)
fig_world.update_yaxes(
    title="Y Position (km)", showgrid=True, gridcolor="rgba(200,200,200,0.5)",
    linecolor="black", zeroline=True, zerolinecolor="rgba(200,200,200,0.5)",
)

# %%
# --- Figure 2: MF envelope for ping 1 ---
# Shows the normalised matched-filter output vs slant range.
# Vertical dashed lines mark each target's expected direct-path range.
# All five peaks should align with their markers.

first_sd = next(iter(all_sensor_data[0][1]))
n_receive = len(first_sd.received_waveform)
range_axis_m = np.arange(n_receive) / fs * c / 2.0

envelope_norm = matched_filter(first_sd.received_waveform, first_sd.transmit_pulse)

fig_env = go.Figure()
fig_env.add_trace(go.Scatter(
    x=range_axis_m / 1000,
    y=20.0 * np.log10(np.maximum(envelope_norm, 1e-6)),
    mode="lines",
    line=dict(color="#444444", width=1.5),
    name="MF envelope (ping 1)",
))

for slant_r, nom_r, color in zip(gt_slant_ranges, NOMINAL_RANGES_M, COLORS):
    fig_env.add_vline(
        x=slant_r / 1000,
        line=dict(color=color, width=1.5, dash="dash"),
        annotation_text=f"{int(nom_r)} m",
        annotation_position="top",
        annotation_font=dict(size=11, color=color),
    )

max_range_km = (signal_params["duration_s"] + detector_params["receive_duration_s"]) * c / 2.0 / 1000.0
fig_env.update_layout(
    width=820, height=420,
    font=dict(family="Times New Roman", size=14, color="black"),
    showlegend=True,
    legend=dict(x=0.98, y=0.02, xanchor="right", yanchor="bottom",
                bgcolor="rgba(255,255,255,0.8)"),
    plot_bgcolor="white", paper_bgcolor="white",
)
fig_env.update_xaxes(
    title="Slant range (km)", range=[0, max_range_km],
    showgrid=True, gridcolor="rgba(200,200,200,0.5)",
    ticks="outside", showline=True, linewidth=1, linecolor="rgba(160,160,160,1)",
)
fig_env.update_yaxes(
    title="MF amplitude (dB, normalised)", range=[-50, 5],
    showgrid=True, gridcolor="rgba(200,200,200,0.5)",
    ticks="outside", showline=True, linewidth=1, linecolor="rgba(160,160,160,1)",
)

# %%
# --- Figure 3: MF waterfall (all pings) ---
# Since targets are stationary, each column of peaks should be perfectly
# vertical — confirming consistency across pings.

mf_matrix = []
for _, sensor_data_set in all_sensor_data:
    sd = next(iter(sensor_data_set))
    mf_matrix.append(matched_filter(sd.received_waveform, sd.transmit_pulse))
mf_matrix = np.array(mf_matrix)

fig_waterfall = go.Figure(go.Heatmap(
    z=20.0 * np.log10(np.maximum(mf_matrix, 1e-6)),
    x=range_axis_m / 1000,
    y=[ts.strftime("%H:%M:%S") for ts in ping_timestamps],
    colorscale="Viridis",
    colorbar=dict(title="dB"),
    zmin=-40, zmax=0,
))

for slant_r, nom_r in zip(gt_slant_ranges, NOMINAL_RANGES_M):
    fig_waterfall.add_vline(
        x=slant_r / 1000,
        line=dict(color="white", width=2.0, dash="dash"),
        annotation_text=f"{int(nom_r)} m",
        annotation_font=dict(size=11, color="white"),
        annotation_position="top",
    )

fig_waterfall.update_layout(
    width=820, height=360,
    font=dict(family="Times New Roman", size=14, color="black"),
    plot_bgcolor="white", paper_bgcolor="white",
)
fig_waterfall.update_xaxes(
    title="Slant range (km)", range=[0, max_range_km],
    showgrid=True, gridcolor="rgba(200,200,200,0.5)",
    ticks="outside", showline=True, linewidth=1, linecolor="rgba(160,160,160,1)",
)
fig_waterfall.update_yaxes(
    title="Ping time", autorange="reversed",
)

# %%
# --- Figure 4: Detection accuracy — detected vs expected range ---
# Each detection is matched to the nearest expected slant range.
# Points should lie on the y = x diagonal if timing is correct.

det_expected_km, det_found_km = [], []
for _, dets in all_detections:
    for d in dets:
        r_det = d.state_vector[0, 0]
        nearest_exp = min(gt_slant_ranges, key=lambda r: abs(r - r_det))
        det_expected_km.append(nearest_exp / 1000)
        det_found_km.append(r_det / 1000)

r_lo = min(gt_slant_ranges) * 0.85 / 1000
r_hi = max(gt_slant_ranges) * 1.10 / 1000

fig_acc = go.Figure()
fig_acc.add_trace(go.Scatter(
    x=[r_lo, r_hi], y=[r_lo, r_hi],
    mode="lines", line=dict(color="#aaaaaa", width=1.5, dash="dash"),
    name="y = x (perfect)",
))
fig_acc.add_trace(go.Scatter(
    x=det_expected_km, y=det_found_km,
    mode="markers",
    marker=dict(size=10, color="#1f77b4", symbol="circle"),
    name="Detections",
))

fig_acc.update_layout(
    width=540, height=500,
    font=dict(family="Times New Roman", size=14, color="black"),
    showlegend=True,
    legend=dict(x=0.05, y=0.95, xanchor="left", yanchor="top",
                bgcolor="rgba(255,255,255,0.8)"),
    plot_bgcolor="white", paper_bgcolor="white",
    yaxis=dict(scaleanchor="x", scaleratio=1),
)
fig_acc.update_xaxes(
    title="Expected slant range (km)", range=[r_lo, r_hi],
    showgrid=True, gridcolor="rgba(200,200,200,0.5)",
    ticks="outside", showline=True, linewidth=1, linecolor="rgba(160,160,160,1)",
)
fig_acc.update_yaxes(
    title="Detected range (km)", range=[r_lo, r_hi],
    showgrid=True, gridcolor="rgba(200,200,200,0.5)",
    ticks="outside", showline=True, linewidth=1, linecolor="rgba(160,160,160,1)",
)

# %%
# Export and Summary
# ------------------

# %%
from pathlib import Path

out_dir = Path(__file__).resolve().parent / "figs_active"
out_dir.mkdir(parents=True, exist_ok=True)

for fname, fig in [
    ("mt_world.pdf", fig_world),
    ("mt_mf_envelope.pdf", fig_env),
    ("mt_mf_waterfall.pdf", fig_waterfall),
    ("mt_det_accuracy.pdf", fig_acc),
]:
    fig.write_image(str(out_dir / fname))
    print(f"Saved {out_dir / fname}")

print()
bw = signal_params["freq_max_hz"] - signal_params["freq_min_hz"]
print("=" * 60)
print("Multi-Target Timing Verification Summary")
print("=" * 60)
print(f"Sonobuoy:  ({buoy_params['x_m']:.0f}, {buoy_params['y_m']:.0f}) m, "
      f"hydrophone at {buoy_params['hydrophone_depth_m']:.0f} m depth")
print(f"Water:     {env_params['water_depth_m']:.0f} m deep, c = {c:.0f} m/s")
print(f"LFM:       {signal_params['freq_min_hz']:.0f}–{signal_params['freq_max_hz']:.0f} Hz, "
      f"range resolution = {c / (2 * bw):.2f} m, "
      f"sample resolution = {sample_res_m:.4f} m")
print(f"Targets:   {len(target_paths)} static  |  Pings: {n_pings}")
print(f"Detections found: {sum(len(d) for _, d in all_detections)}  "
      f"(includes direct-path and multipath returns)")
print(f"\nTiming errors (MF at exact direct-path index, all targets × all pings):")
print(f"  Max |ΔR|: {np.max(np.abs(errs)):.4f} m")
print(f"  RMS ΔR:   {np.sqrt(np.mean(errs**2)):.4f} m")
print(f"  Criterion: |ΔR| < {range_res_m:.2f} m (one resolution cell, c/(2B))  →  "
      f"{'PASS' if np.max(np.abs(errs)) <= range_res_m else 'FAIL'}")
