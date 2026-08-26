"""
==============================================
Active Sonar Single-Target Example (Bellhop)
==============================================

This example demonstrates the monostatic active sonar pipeline using a stationary
omnidirectional sonobuoy pinging a single moving target. It covers:

- Configuring an ``OmniSonobuoyPlatform`` with a hydrophone at depth.
- Generating LFM transmit pulses with a sinusoidal edge taper.
- Running the two-pass Bellhop eigenray simulation (sonobuoy → target → sonobuoy).
- Applying matched filtering to recover range from round-trip echo delay.
- Plotting the matched filter waterfall, range-vs-time estimates, and a world picture.

The scenario uses a flat 200 m bathymetry and a constant 1500 m/s sound speed profile
— the simplest possible environment, useful for verifying the pipeline before adding
realistic SSP data.
"""  # noqa: D205, D212, D400, D415

# %%
# Imports
# -------

# %%
from datetime import datetime, timedelta

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState

import bluepebble
from bluepebble.detector.active import ActiveSonarDetectorOmni, matched_filter
from bluepebble.detector.algorithms import DetectionAlgorithm, OSCFARDetector, PeakDetector, ThresholdDetector
from bluepebble.signal.random import ColouredNoiseSignal
from bluepebble.models.environment import Constant, Munk, FlatBathymetry, SeamountBathymetry
from bluepebble.models.propagation import BellhopArrivalsModel
from bluepebble.platform import OmniSonobuoyPlatform
from bluepebble.signal.active import LFMSignal
from bluepebble.simulator import BellhopActiveSonarSimulatorOmni
from stonesoup.types.state import State
from stonesoup.types.array import StateVector

from stonesoup.dataassociator.neighbour import NearestNeighbour
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.measures import Mahalanobis
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.types.state import GaussianState
from stonesoup.types.track import Track
from stonesoup.updater.kalman import KalmanUpdater

# %%
# Configuration
# -------------

# %%
seed = 42
bluepebble.set_seed(seed)
rng = bluepebble.get_rng()

start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

sim_params = {
    "start_time": start_time,
    "ping_interval_s": 30.0,
    "n_pings": 5,
    "amplitude_cutoff": 0.005,  # fraction of max eigenray amplitude below which rays are discarded
}

ping_timestamps = [
    start_time + timedelta(seconds=i * sim_params["ping_interval_s"])
    for i in range(sim_params["n_pings"])
]

env_params = {
    "water_depth_m": 200.0,
    "sound_speed_ms": 1500.0,
    'sound_speed_uncertainty_ms': 0,   # configure depending on SSP
    'nominal_range_m': 2000            # estimated range, use with SSP uncertainty to estimate range uncertainty
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

# Target: starts 2 km away, moves at 3 m/s — state vector: [x, vx, y, vy, z, vz]
target_params = {
    "start_vector": np.array([2000.0, 3.0, 0.0, 0.0, -150.0, 0.0]),
    "position_mapping": [0, 2, 4],
    "target_strength_db": 15.0,
}

ambient_noise_params = {
    "amplitude_upa": 10 ** (rng.uniform(45, 55) / 20),
    "spectral_exponent": -1,
}

det_params = {
    "min_range_m": 100.0,
    "receive_duration_s": 4.0,
    "cfar_detector": {
        "num_guard_cells": 25,   # = 1 range resolution cell (c/2B = 1.875 m = 25 samples at 10 kHz)
        "num_training_cells": 50,
        "rank": 75,              # 75th percentile of 100 training cells
        "threshold_factor": 8,
    },
    "min_amplitude_db": -40.0,   # dB re MF peak; rejects noise sidelobes well below target echo
    "peak_detector": {
        "distance": 3000,        # ~225 m at c=1500, fs=10000 — collapses multipath cluster
    },
}

cfg = {
    "seed": seed,
    "sim": sim_params,
    "env": env_params,
    "buoy": buoy_params,
    "signal": signal_params,
    "target": target_params,
    "ambient_noise": ambient_noise_params,
    "detection": det_params
}

print(f"Pings: {cfg['sim']['n_pings']} at {cfg['sim']['ping_interval_s']} s intervals")
print(f"LFM: {signal_params['freq_min_hz']:.0f}–{signal_params['freq_max_hz']:.0f} Hz, "
      f"{signal_params['duration_s']:.1f} s duration, "
      f"fs={signal_params['sampling_rate_hz']} Hz")
print(f"Target start range: {target_params['start_vector'][0]:.0f} m, "
      f"speed: {target_params['start_vector'][1]:.1f} m/s")
print(f"Ambient noise level: {20 * np.log10(ambient_noise_params['amplitude_upa']):.1f} dB re 1 µPa")

# %%
# Scenario Build
# --------------

# %%
# Stationary sonobuoy
buoy = OmniSonobuoyPlatform(
    states=[State(
        StateVector([buoy_params["x_m"], buoy_params["y_m"]]),
        timestamp=start_time,
    )],
    position_mapping=[0, 1],
    hydrophone_depth_m=buoy_params["hydrophone_depth_m"],
)

# Moving target — generate one state per ping timestamp
transition_model = CombinedLinearGaussianTransitionModel(
    [ConstantVelocity(0), ConstantVelocity(0), ConstantVelocity(0)]
)

target_states = [
    GroundTruthState(target_params["start_vector"], timestamp=start_time)
]
for ts in ping_timestamps[1:]:
    dt = ts - target_states[-1].timestamp
    new_sv = transition_model.function(target_states[-1], noise=False, time_interval=dt)
    target_states.append(GroundTruthState(new_sv, timestamp=ts))

target_truth = GroundTruthPath(target_states)

# Ground-truth slant ranges at each ping (3-D sonobuoy hydrophone → target)
gt_ranges = []
buoy_xy = buoy.position_3d.flatten()[:2]
buoy_z = -buoy.hydrophone_depth_m  # negative z = below surface, matching project convention
for state in target_truth:
    tv = state.state_vector
    tx = float(tv[target_params["position_mapping"][0]])
    ty = float(tv[target_params["position_mapping"][1]])
    tz = float(tv[target_params["position_mapping"][2]])
    gt_ranges.append(float(np.sqrt(
        (buoy_xy[0] - tx)**2 + (buoy_xy[1] - ty)**2 + (buoy_z - tz)**2
    )))

print("\nGround-truth ranges:")
for ts, r in zip(ping_timestamps, gt_ranges):
    expected_delay = 2.0 * r / env_params["sound_speed_ms"]
    print(f"  {ts.strftime('%H:%M:%S')}  {r:.1f} m  (round-trip delay {expected_delay:.3f} s)")

# %%
# Simulation
# ----------

# %%
# ssp = Constant(speed=env_params["sound_speed_ms"])
ssp = Munk()
# bathymetry = FlatBathymetry(depth=-env_params["water_depth_m"])
bathymetry = SeamountBathymetry()

prop_model = BellhopArrivalsModel(ssp=ssp, bathymetry=bathymetry)

ambient_noise_model = ColouredNoiseSignal(
    amplitude_upa=ambient_noise_params["amplitude_upa"],
    spectral_exponent=ambient_noise_params["spectral_exponent"],
    duration_s=signal_params["duration_s"] + det_params["receive_duration_s"],
    sampling_rate_hz=signal_params["sampling_rate_hz"],
)

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
    ground_truth_paths=[target_truth],
    target_strength_db=cfg["target"]["target_strength_db"],
    ping_timestamps=ping_timestamps,
    target_position_mapping=target_params["position_mapping"],
    receive_duration_s=det_params["receive_duration_s"],
    noise_model=ambient_noise_model,
    amplitude_cutoff=cfg["sim"]["amplitude_cutoff"],
)

print("\nRunning active sonar simulation...")
all_sensor_data = list(simulator.sensor_data_gen())
print(f"Generated {len(all_sensor_data)} pings.")

# %%
# Detection
# ---------

# %%
def sensor_data_replay(data):
    yield from data

det = cfg["detection"]

cfar_detector = OSCFARDetector(
    num_guard_cells=det["cfar_detector"]["num_guard_cells"],
    num_training_cells=det["cfar_detector"]["num_training_cells"],
    rank=det["cfar_detector"]["rank"],
    threshold_factor=det["cfar_detector"]["threshold_factor"],
)

detection_chain: list[DetectionAlgorithm] = [cfar_detector]
if det["min_amplitude_db"] is not None:
    detection_chain.append(ThresholdDetector(threshold=det["min_amplitude_db"]))
if det["peak_detector"]["distance"] > 0:
    detection_chain.append(PeakDetector(distance=det["peak_detector"]["distance"]))

detector = ActiveSonarDetectorOmni(
    sensor_data_gen=sensor_data_replay(all_sensor_data),
    sampling_rate_hz=cfg["signal"]["sampling_rate_hz"],
    sound_speed_ms=cfg["env"]["sound_speed_ms"],
    min_range_m=det["min_range_m"],
    detection_chain=detection_chain,
)

all_detections = list(detector.detections_gen())

print("\nDetections:")
for (ts, dets), gt_r in zip(all_detections, gt_ranges):
    if dets:
        for d in sorted(dets, key=lambda d: d.state_vector[0, 0]):
            err = d.state_vector[0, 0] - gt_r
            print(f"  {ts.strftime('%H:%M:%S')}  detected {d.state_vector[0, 0]:.1f} m  "
                  f"(GT {gt_r:.1f} m, error {err:+.1f} m)")
    else:
        print(f"  {ts.strftime('%H:%M:%S')}  no detection  (GT {gt_r:.1f} m)")

# %%
# Tracking — nearest-neighbour Kalman filter on range detections
# --------------------------------------------------------------

# %%
B = cfg['signal']['freq_max_hz'] - cfg['signal']['freq_min_hz']
c = cfg["env"]["sound_speed_ms"]

range_res = c / (2 * B)
range_uncertainty = (cfg['env']['nominal_range_m'] * cfg['env']['sound_speed_uncertainty_ms']) / c

kf_transition_model = ConstantVelocity(0.0001)
predictor = KalmanPredictor(kf_transition_model)

measurement_model = LinearGaussian(
    ndim_state=2,
    mapping=[0],
    noise_covar=np.array([[range_res ** 2 + range_uncertainty ** 2]]),
)
updater = KalmanUpdater(measurement_model=measurement_model)

hypothesiser = DistanceHypothesiser(predictor, updater, Mahalanobis(), missed_distance=5)
associator = NearestNeighbour(hypothesiser)

# Initialise prior from first ping with a detection
prior = None
start_idx = 0
for i, (ts, dets) in enumerate(all_detections):
    if dets:
        first_range = float(max(dets, key=lambda d: d.metadata["mf_amplitude"]).state_vector[0, 0])
        prior = GaussianState(
            state_vector=[[first_range], [0.0]],
            covar=np.array([[range_res ** 2, 0.0], [0.0, 10.0 ** 2]]),
            timestamp=ts,
        )
        start_idx = i + 1
        break

track = Track([prior]) if prior is not None else Track()

for ts, detections in all_detections[start_idx:]:
    hypotheses = associator.associate({track}, detections, timestamp=ts)
    hypothesis = hypotheses[track]
    if hypothesis:
        posterior = updater.update(hypothesis)
    else:
        posterior = hypothesis.prediction
    track.append(posterior)
    prior = posterior

print(f"\nKF track: {len(track)} states")
for state in track:
    print(f"  {state.timestamp.strftime('%H:%M:%S')}  range={float(state.mean[0]):.1f} m  "
          f"range_rate={float(state.mean[1]):.3f} m/s")

# %%
# Sonar Equation Verification
# ---------------------------

# %%
# Only valid for a constant SSP where TL = 20·log10(R) (spherical spreading).
# Skipped automatically when a non-constant SSP (e.g. Munk) is used, because
# refraction, ducting, and convergence zones break the spherical-spreading
# assumption and the comparison would be meaningless.

SL = signal_params["source_level_db"]  # dB re 1 µPa @ 1 m
TS = cfg["target"]["target_strength_db"]  # dB
c  = env_params["sound_speed_ms"]
fs = signal_params["sampling_rate_hz"]

sonar_eq_rows = []
fig_sonar_eq = None

if isinstance(ssp, Constant):
    # The monostatic active sonar equation predicts the received echo level (EL):
    #
    #   EL = SL - 2·TL + TS
    #
    # SL  Source Level [dB re 1 µPa @ 1 m] — set by ``source_level_db``
    # TL  One-way Transmission Loss [dB]   — 20·log10(R) for spherical spreading
    # TS  Target Strength [dB]             — set by ``target_strength_db``
    #
    # Bellhop's 2-D model applies the 3-D point-source correction internally, so
    # the direct-path eigenray amplitude scales as 1/R (spherical), not 1/sqrt(R).
    #
    # EL_meas is derived from the unnormalised matched-filter peak at the expected
    # direct-path round-trip delay, divided by the pulse energy.  This isolates the
    # direct-path echo from multipath arrivals: the LFM bandwidth B = 400 Hz gives
    # range resolution c/(2B) ≈ 1.9 m, resolving the direct path from the nearest
    # surface/bottom bounce which arrives ~7.5 m later in slant range.
    for i, (_, sensor_data_set) in enumerate(all_sensor_data):
        R        = gt_ranges[i]
        TL_geom  = 20.0 * np.log10(R)           # spherical spreading, one-way
        EL_pred  = SL - 2.0 * TL_geom + TS     # sonar equation prediction

        sd           = next(iter(sensor_data_set))
        pulse_energy = float(np.sum(np.abs(sd.transmit_pulse) ** 2))

        # Unnormalised matched filter
        n_fft   = len(sd.received_waveform) + len(sd.transmit_pulse) - 1
        mf_raw  = np.abs(np.fft.ifft(
            np.fft.fft(sd.received_waveform, n_fft)
            * np.conj(np.fft.fft(sd.transmit_pulse, n_fft)),
            n_fft,
        ))[: len(sd.received_waveform)]

        # Evaluate the unnormalised MF at the expected direct-path round-trip sample.
        # Using the exact sample avoids picking up the higher MF peak from the
        # direct × surface-bounce coherent pair which arrives ~2.7 ms later.
        expected_sample = min(int(2.0 * R / c * fs), len(mf_raw) - 1)
        mf_at_direct    = float(mf_raw[expected_sample])

        # A_received = Bellhop_out × Bellhop_back × A_target (no A_signal factor).
        # Multiply by A_signal (= 10^(SL/20)) via the SL term below.
        A_received = mf_at_direct / pulse_energy
        EL_meas    = SL + 20.0 * np.log10(max(A_received, 1e-30))

        # TL implied by the simulation (inverted sonar equation)
        TL_implied = (SL - EL_meas + TS) / 2.0

        sonar_eq_rows.append(dict(
            ping=i + 1, range_m=R,
            TL_geom=TL_geom, TL_implied=TL_implied,
            EL_pred=EL_pred, EL_meas=EL_meas,
            delta=EL_meas - EL_pred,
        ))

    print("\nSonar Equation Verification  (EL = SL - 2*TL + TS,  TL = 20*log10(R)  spherical)")
    print("=" * 86)
    print(f"  SL = {SL:.0f} dB re 1 uPa @ 1 m     TS = {TS:.0f} dB")
    print(f"  {'Ping':>4}  {'Range (m)':>10}  {'TL_geom':>10}  "
          f"{'TL_implied':>12}  {'EL_pred':>10}  {'EL_meas':>10}  {'Delta':>8}")
    print(f"  {'':>4}  {'':>10}  {'(dB)':>10}  "
          f"{'(dB)':>12}  {'(dB)':>10}  {'(dB)':>10}  {'(dB)':>8}")
    print("-" * 86)
    for row in sonar_eq_rows:
        print(f"  {row['ping']:>4}  {row['range_m']:>10.1f}  {row['TL_geom']:>10.1f}  "
              f"{row['TL_implied']:>12.1f}  {row['EL_pred']:>10.1f}  "
              f"{row['EL_meas']:>10.1f}  {row['delta']:>+8.1f}")
    _deltas = np.array([r["delta"] for r in sonar_eq_rows])
    print(f"\n  Mean delta: {np.mean(_deltas):+.1f} dB    RMS delta: "
          f"{np.sqrt(np.mean(_deltas ** 2)):.1f} dB")
    print("  (small delta expected from direct-path MF; ±1–2 dB residual from sidelobe leakage)")
else:
    print("\nSonar equation verification skipped (non-constant SSP).")

# %%
# Build Figures
# -------------

# %%
# --- Figure 1: World picture ---
fig_world = go.Figure()

fig_world.add_trace(go.Scatter(
    x=[buoy_params["x_m"] / 1000],
    y=[buoy_params["y_m"] / 1000],
    mode="markers",
    marker=dict(symbol="diamond", size=14, color="#1f77b4"),
    name="Sonobuoy",
))

tgt_x_km = [float(s.state_vector[target_params["position_mapping"][0]]) / 1000
             for s in target_truth]
tgt_y_km = [float(s.state_vector[target_params["position_mapping"][1]]) / 1000
             for s in target_truth]

fig_world.add_trace(go.Scatter(
    x=tgt_x_km, y=tgt_y_km,
    mode="lines+markers",
    line=dict(color="#ff4141", width=3),
    marker=dict(size=8),
    name="Target",
))

all_x = [buoy_params["x_m"] / 1000] + tgt_x_km
all_y = [buoy_params["y_m"] / 1000] + tgt_y_km
xrange = [min(all_x) - 0.5, max(all_x) + 0.5]
yrange = [min(all_y) - 0.5, max(all_y) + 0.5]

fig_world.update_layout(
    width=600, height=600,
    font=dict(family="Times New Roman", size=16, color="black"),
    showlegend=True,
    legend=dict(x=0.5, y=-0.25, xanchor="center", orientation="h", yanchor="bottom"),
    plot_bgcolor="white",
    paper_bgcolor="white",
    yaxis=dict(scaleanchor="x", scaleratio=1),
)
fig_world.update_xaxes(
    range=xrange,
    title="X Position (km)", showgrid=True, gridcolor="rgba(200, 200, 200, 0.5)",
    linecolor="black", zeroline=True,
    zerolinecolor="rgba(200, 200, 200, 0.5)", zerolinewidth=0.5,
)
fig_world.update_yaxes(
    range=yrange,
    title="Y Position (km)", showgrid=True, gridcolor="rgba(200, 200, 200, 0.5)",
    linecolor="black", zeroline=True,
    zerolinecolor="rgba(200, 200, 200, 0.5)", zerolinewidth=0.5,
)

# %%
# --- Figure 2: Range detections vs ground truth ---
det_times, det_ranges = [], []
for ts, dets in all_detections:
    for d in dets:
        det_times.append(ts)
        det_ranges.append(d.state_vector[0, 0])

fig_range = go.Figure()

fig_range.add_trace(go.Scatter(
    x=ping_timestamps, y=gt_ranges,
    mode="lines", line=dict(color="#ff4141", width=3, dash="dash"),
    name="Ground truth",
))
fig_range.add_trace(go.Scatter(
    x=det_times, y=det_ranges,
    mode="markers", marker=dict(size=12, color="#1f77b4", symbol="circle"),
    name="Detection",
))

track_times = [s.timestamp for s in track]
track_ranges = [float(s.mean[0]) for s in track]
fig_range.add_trace(go.Scatter(
    x=track_times, y=track_ranges,
    mode="lines+markers", line=dict(color="#2ca02c", width=2),
    marker=dict(size=8), name="KF Track",
))

fig_range.update_layout(
    width=600, height=400,
    font=dict(family="Times New Roman", size=16, color="black"),
    showlegend=True,
    legend=dict(x=0.5, y=-0.3, xanchor="center", orientation="h", yanchor="bottom",
                bgcolor="rgba(255,255,255,0.0)", borderwidth=0),
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(b=100),
)
fig_range.update_xaxes(
    title="Time", showgrid=True, gridcolor="rgba(200, 200, 200, 0.5)",
    tickformat="%H:%M:%S",
    ticks="outside", tickcolor="rgba(160, 160, 160, 1.0)",
    showline=True, linewidth=1, linecolor="rgba(160, 160, 160, 1.0)",
)
fig_range.update_yaxes(
    title="Slant range (m)", showgrid=True, gridcolor="rgba(200, 200, 200, 0.5)",
    ticks="outside", tickcolor="rgba(160, 160, 160, 1.0)",
    showline=True, linewidth=1, linecolor="rgba(160, 160, 160, 1.0)",
)

# %%
# --- Figure 3: MF envelope waterfall (all pings stacked) ---
fs = signal_params["sampling_rate_hz"]
c  = env_params["sound_speed_ms"]
n_receive = int((signal_params["duration_s"] + det_params["receive_duration_s"]) * fs)
range_axis = np.arange(n_receive) / fs * c / 2.0  # two-way delay → range

mf_matrix = []
for _, sensor_data_set in all_sensor_data:
    for sd in sensor_data_set:
        env = matched_filter(sd.received_waveform, sd.transmit_pulse)
        mf_matrix.append(env)

mf_matrix = np.array(mf_matrix)  # shape (n_pings, n_receive)

fig_mf = go.Figure(go.Heatmap(
    z=20 * np.log10(np.maximum(mf_matrix, 1e-6)),
    x=range_axis / 1000,
    y=[ts.strftime("%H:%M:%S") for ts in ping_timestamps],
    colorscale="Viridis",
    colorbar=dict(title="MF amplitude (dB)"),
    zmin=-60, zmax=0,
))

# Overlay ground truth range
fig_mf.add_trace(go.Scatter(
    x=[r / 1000 for r in gt_ranges],
    y=[ts.strftime("%H:%M:%S") for ts in ping_timestamps],
    mode="markers",
    marker=dict(symbol="x", size=12, color="#ff4141", line=dict(width=2)),
    name="Ground truth",
))

fig_mf.update_layout(
    width=700, height=420,
    font=dict(family="Times New Roman", size=16, color="black"),
    showlegend=True,
    legend=dict(x=0.5, y=-0.25, xanchor="center", orientation="h", yanchor="bottom",
                bgcolor="rgba(255,255,255,0.0)", borderwidth=0),
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(b=100),
)
fig_mf.update_xaxes(
    title="Slant range (km)",
    range=[0, (signal_params["duration_s"] + det_params["receive_duration_s"]) * c / 2 / 1000],
    showgrid=True, gridcolor="rgba(200, 200, 200, 0.5)",
    ticks="outside", tickcolor="rgba(160, 160, 160, 1.0)",
    showline=True, linewidth=1, linecolor="rgba(160, 160, 160, 1.0)",
)
fig_mf.update_yaxes(title="Ping time", autorange="reversed")

# %%
# --- Figure 4: MF scatter plot (per-sample, colored by ping) ---
# Each dot is one MF output sample above the noise floor, coloured by ping
# time.  Unlike the waterfall heatmap (which averages within a row pixel),
# this view shows the realistic spread of returns across multipath arrivals.
_db_floor = -50.0
_ping_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

fig_mf_scatter = go.Figure()
for ping_idx, ts in enumerate(ping_timestamps):
    color = _ping_colors[ping_idx % len(_ping_colors)]
    label = ts.strftime("%H:%M:%S")
    env_db = 20 * np.log10(np.maximum(mf_matrix[ping_idx], 1e-6))
    mask = env_db >= _db_floor
    # Invisible single-point trace: provides a large legend marker
    fig_mf_scatter.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="markers",
        marker=dict(size=10, color=color),
        name=label,
        showlegend=True,
    ))
    # Actual data: small markers, no legend entry
    fig_mf_scatter.add_trace(go.Scatter(
        x=range_axis[mask],
        y=env_db[mask],
        mode="markers",
        marker=dict(size=2, color=color),
        name=label,
        showlegend=False,
    ))

fig_mf_scatter.update_layout(
    width=900, height=400,
    font=dict(family="Times New Roman", size=16, color="black"),
    showlegend=True,
    legend=dict(
        title="Ping", x=0.02, y=0.98, xanchor="left", yanchor="top",
        bgcolor="rgba(255,255,255,0.85)", borderwidth=1,
    ),
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(b=60, t=20, l=20, r=20),
)
fig_mf_scatter.update_xaxes(
    title="Slant range (m)",
    range=[0, range_axis[-1]],
    showgrid=True, gridcolor="rgba(200, 200, 200, 0.5)",
    ticks="outside", tickcolor="rgba(160, 160, 160, 1.0)",
    showline=True, linewidth=1, linecolor="rgba(160, 160, 160, 1.0)",
)
fig_mf_scatter.update_yaxes(
    title="MF amplitude (dB)",
    range=[_db_floor, 0],
    showgrid=True, gridcolor="rgba(200, 200, 200, 0.5)",
    ticks="outside", tickcolor="rgba(160, 160, 160, 1.0)",
    showline=True, linewidth=1, linecolor="rgba(160, 160, 160, 1.0)",
)

# %%
# --- Figure 5: Sonar equation verification (constant SSP only) ---
if isinstance(ssp, Constant):
    r_axis = np.linspace(0.85 * min(gt_ranges), 1.20 * max(gt_ranges), 300)
    EL_curve = SL - 2.0 * 20.0 * np.log10(r_axis) + TS

    fig_sonar_eq = go.Figure()

    fig_sonar_eq.add_trace(go.Scatter(
        x=r_axis / 1000, y=EL_curve,
        mode="lines",
        line=dict(color="#ff4141", width=2, dash="dash"),
        name="Sonar equation (spherical TL)",
    ))
    fig_sonar_eq.add_trace(go.Scatter(
        x=[r["range_m"] / 1000 for r in sonar_eq_rows],
        y=[r["EL_meas"] for r in sonar_eq_rows],
        mode="markers",
        marker=dict(size=12, color="#1f77b4", symbol="circle"),
        name="Simulated EL",
    ))

    fig_sonar_eq.update_layout(
        width=800, height=500,
        font=dict(family="Times New Roman", size=16, color="black"),
        showlegend=True,
        legend=dict(
            x=0.97, y=0.97,
            xanchor="right", yanchor="top",
            orientation="v",
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(160,160,160,0.5)",
            borderwidth=1,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(b=60, t=20, l=20, r=20),
    )
    fig_sonar_eq.update_xaxes(
        title="Slant range (km)", showgrid=True, gridcolor="rgba(200, 200, 200, 0.5)",
        ticks="outside", tickcolor="rgba(160, 160, 160, 1.0)",
        showline=True, linewidth=1, linecolor="rgba(160, 160, 160, 1.0)",
    )
    fig_sonar_eq.update_yaxes(
        title="Echo level (dB re 1 µPa)", showgrid=True, gridcolor="rgba(200, 200, 200, 0.5)",
        ticks="outside", tickcolor="rgba(160, 160, 160, 1.0)",
        showline=True, linewidth=1, linecolor="rgba(160, 160, 160, 1.0)",
    )

# %%
# Export and Summary
# ------------------

# %%
from pathlib import Path

out_dir = Path(__file__).resolve().parent / "figs_active"
out_dir.mkdir(parents=True, exist_ok=True)

for fname, fig in [
    ("as_world.pdf", fig_world),
    ("as_range_detections.pdf", fig_range),
    ("as_mf_waterfall.pdf", fig_mf),
    ("as_mf_scatter.pdf", fig_mf_scatter),
    ("as_sonar_eq.pdf", fig_sonar_eq),
]:
    if fig is None:
        continue
    fig.write_image(str(out_dir / fname))
    print(f"Saved {out_dir / fname}")

print()
print("=" * 60)
print("Active Sonar Scenario Summary")
print("=" * 60)
print(f"Sonobuoy position:  ({buoy_params['x_m']:.0f}, {buoy_params['y_m']:.0f}) m")
print(f"Hydrophone depth:   {buoy_params['hydrophone_depth_m']:.0f} m")
print(f"Water depth:        {env_params['water_depth_m']:.0f} m")
print(f"Sound speed:        {env_params['sound_speed_ms']:.0f} m/s")
print(f"LFM bandwidth:      {signal_params['freq_max_hz'] - signal_params['freq_min_hz']:.0f} Hz")
bw = signal_params["freq_max_hz"] - signal_params["freq_min_hz"]
print(f"Range resolution:   ~{c / (2 * bw):.1f} m  (c / 2B)")
print(f"Target strength:    {cfg['target']['target_strength_db']:.0f} dB")
print(f"Detections found:   {sum(len(d) for _, d in all_detections)} / {cfg['sim']['n_pings']} pings")
