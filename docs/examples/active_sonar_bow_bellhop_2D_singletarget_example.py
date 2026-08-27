"""
==============================================
Active Sonar Single-Target Example (Bow Array)
==============================================

This example demonstrates the monostatic active sonar pipeline using a stationary
bow-mounted dome array pinging a single moving target. It covers:

- Configuring a ``HostPlatform`` (the ship) with a ``BowArraySensor`` mounted on it (a
  curved forward-facing dome of elements).
- Generating LFM transmit pulses with a sinusoidal edge taper.
- Running the two-pass Bellhop eigenray simulation per array element
  (dome → target → dome), with a far-field per-element delay/phase correction on
  the return leg (see ``BellhopActiveSonarSimulatorArray``).
- Delay-and-sum beamforming the per-element echoes toward the target bearing to
  recover a single combined waveform, exactly as a real receive beamformer would.
- Applying matched filtering to the beamformed waveform to recover range from
  round-trip echo delay.
- Plotting the matched filter waterfall, range-vs-time estimates, and a world picture.

The scenario uses a flat 200 m bathymetry and a constant 1500 m/s sound speed profile
— the simplest possible environment, useful for verifying the pipeline before adding
realistic SSP data.

Beamforming here steers exactly at the (known) ground-truth bearing to the target,
since the point of this script is to verify the array simulation + beamforming +
detection chain end-to-end, not to demonstrate bearing search.
"""  # noqa: D205, D212, D400, D415

# %%
# Imports
# -------

# %%
from datetime import datetime, timedelta

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import get_window
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState

import bluepebble
from bluepebble.detector.active import ActiveSonarDetectorOmni, matched_filter
from bluepebble.detector.algorithms import DetectionAlgorithm, OSCFARDetector, PeakDetector, ThresholdDetector
from bluepebble.signal.random import ColouredNoiseSignal
from bluepebble.models.environment import Constant, Munk, FlatBathymetry, SeamountBathymetry
from bluepebble.models.propagation import BellhopArrivalsModel
from bluepebble.platform import HostPlatform
from bluepebble.sensors import BowArraySensor
from bluepebble.sigproc import (
    DelayAndSumBeamformer,
    MinimumVarianceDistortionlessResponseBeamformer,
    SteeringCalculator,
)
from bluepebble.signal.active import LFMSignal
from bluepebble.simulator import BellhopActiveSonarSimulatorArray, BellhopActiveSonarSimulatorArrayPerElement
from bluepebble.types.sensordata import ActiveSonarSensorData
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
    "amplitude_cutoff": 0.005,
}

ping_timestamps = [
    start_time + timedelta(seconds=i * sim_params["ping_interval_s"])
    for i in range(sim_params["n_pings"])
]

env_params = {
    "water_depth_m": 200.0,
    "sound_speed_ms": 1500.0,
    "sound_speed_uncertainty_ms": 0,
    "nominal_range_m": 2000,
}

signal_params = {
    "freq_min_hz": 800.0,
    "freq_max_hz": 1200.0,
    "duration_s": 1.0,
    "rise_time_s": 0.05,
    "source_level_db": 200.0,
    "sampling_rate_hz": 10_000,
}

bow_params = {
    "x_m": 0.0,
    "y_m": 0.0,
    "depth_m": 50.0,
    "speed_ms": 5.0,
    "heading_rad": 0.0,
    "dome_radius_m": 2.5,
    "azimuth_extent_rad": np.radians(90.0),
    "elevation_extent_rad": np.radians(90.0),
    "element_spacing_m": 0.10,
    "element_size_m": 0.05,
}

# Target submarine starts 2 km ahead of the ship on the reciprocal heading, so the two
# close head-on over the run. State vector: [x, vx, y, vy, z, vz].
target_params = {
    "start_vector": np.array([2000.0, -3.0, 0.0, 0.0, -150.0, 0.0]),
    "position_mapping": [0, 2, 4],
    "target_strength_db": 15.0,
}

ambient_noise_params = {
    "amplitude_upa": 10 ** (rng.uniform(45, 55) / 20),
    "spectral_exponent": -1,
}

bf_params = {
    "beamformer_type": "DAS",
    "shading": None,
    "domain": "time",
}

det_params = {
    "min_range_m": 100.0,
    "receive_duration_s": 4.0,
    "cfar_detector": {
        "num_guard_cells": 25,
        "num_training_cells": 50,
        "rank": 75,
        "threshold_factor": 8,
    },
    "min_amplitude_db": -40.0,
    "peak_detector": {
        "distance": 3000,
    },
}

cfg = {
    "seed": seed,
    "sim": sim_params,
    "env": env_params,
    "array": bow_params,
    "signal": signal_params,
    "target": target_params,
    "ambient_noise": ambient_noise_params,
    "beamforming": bf_params,
    "detection": det_params
}

print(f"Pings: {cfg['sim']['n_pings']} at {cfg['sim']['ping_interval_s']} s intervals")
print(f"LFM: {signal_params['freq_min_hz']:.0f}–{signal_params['freq_max_hz']:.0f} Hz, "
      f"{signal_params['duration_s']:.1f} s duration, "
      f"fs={signal_params['sampling_rate_hz']} Hz")
print(f"Ship speed: {bow_params['speed_ms']:.1f} m/s (heading {np.degrees(bow_params['heading_rad']):.0f} deg)")
print(f"Target start range: {target_params['start_vector'][0]:.0f} m, "
      f"speed: {target_params['start_vector'][1]:.1f} m/s (closing)")
print(f"Ambient noise level: {20 * np.log10(ambient_noise_params['amplitude_upa']):.1f} dB re 1 µPa")

# %%
# Scenario Build
# --------------

# %%
# Ship host with a real constant-velocity transition model. It is driven to each ping
# timestamp explicitly below, before the simulator runs.
transition_model = CombinedLinearGaussianTransitionModel(
    [ConstantVelocity(0), ConstantVelocity(0), ConstantVelocity(0)]
)

ship_vx = bow_params["speed_ms"] * np.cos(bow_params["heading_rad"])
ship_vy = bow_params["speed_ms"] * np.sin(bow_params["heading_rad"])

host = HostPlatform(
    states=[State(
        StateVector([
            bow_params["x_m"], ship_vx,
            bow_params["y_m"], ship_vy,
            -bow_params["depth_m"], 0.0,
        ]),
        timestamp=start_time,
    )],
    position_mapping=[0, 2, 4],
    transition_models=[transition_model],
    transition_times=[
        timedelta(seconds=sim_params["ping_interval_s"] * sim_params["n_pings"])
    ],
)
bow_array = BowArraySensor(
    host=host,
    dome_radius_m=bow_params["dome_radius_m"],
    azimuth_extent_rad=bow_params["azimuth_extent_rad"],
    elevation_extent_rad=bow_params["elevation_extent_rad"],
    element_spacing_m=bow_params["element_spacing_m"],
    element_size_m=bow_params["element_size_m"],
    freq_max_hz=signal_params["freq_max_hz"],
)
num_elements = bow_array.array.num_sensors
print(f"\nShip (bow array): {num_elements} elements, {bow_params['dome_radius_m']:.1f} m dome radius")

# Moving submarine, one ground-truth state per ping timestamp, sharing the ship's
# zero-noise transition model.
target_states = [
    GroundTruthState(target_params["start_vector"], timestamp=start_time)
]
for ts in ping_timestamps[1:]:
    dt = ts - target_states[-1].timestamp
    new_sv = transition_model.function(target_states[-1], noise=False, time_interval=dt)
    target_states.append(GroundTruthState(new_sv, timestamp=ts))

target_truth = GroundTruthPath(target_states)

# %%
# Beamforming
# -----------

# %%
bf = cfg["beamforming"]

# This example verifies the detection chain rather than searching bearing, so the beam is
# steered once, at the ground-truth az/el from the ping-0 geometry.
array_ref = bow_array.array.ref_state_vector.flatten()
first_target_xyz = np.array([
    float(target_states[0].state_vector[target_params["position_mapping"][0]]),
    float(target_states[0].state_vector[target_params["position_mapping"][1]]),
    float(target_states[0].state_vector[target_params["position_mapping"][2]]),
])
to_target = first_target_xyz - array_ref
bearing_az_rad = float(np.arctan2(to_target[1], to_target[0]))
bearing_el_rad = float(np.arctan2(to_target[2], np.hypot(to_target[0], to_target[1])))
print(f"Steering bow array beam to bearing az={np.degrees(bearing_az_rad):.1f} deg, "
      f"el={np.degrees(bearing_el_rad):.1f} deg")

ssp = Munk()

shading = None
if bf["shading"] is not None:
    shading = get_window(bf["shading"], num_elements)

if bf["beamformer_type"] == "DAS":
    beamformer = DelayAndSumBeamformer(
        sampling_rate_hz=signal_params["sampling_rate_hz"],
        shading=shading,
        domain=bf["domain"],
    )
elif bf["beamformer_type"] == "MVDR":
    beamformer = MinimumVarianceDistortionlessResponseBeamformer(
        sampling_rate_hz=signal_params["sampling_rate_hz"],
        fmin=bf.get("fmin"),
        fmax=bf.get("fmax"),
    )
else:
    raise ValueError(f"Unknown beamformer type: {bf['beamformer_type']}")

steering_calculator = SteeringCalculator(
    ssp=ssp,
    steering_azimuths_rad=np.array([bearing_az_rad]),
    steering_elevations_rad=np.array([bearing_el_rad]),
)
steering_delays_s = steering_calculator.calculate(bow_array)

# %%
# Drive the host to every ping timestamp up front — the simulator no longer advances its
# platform internally, so the caller must. host.move() advances host.states 1:1 with pings.
for ts in ping_timestamps[1:]:
    host.move(ts)

# %%
# Simulation
# ----------

# %%
# ssp = Constant(speed=env_params["sound_speed_ms"])
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

simulator = BellhopActiveSonarSimulatorArray(
    platform=bow_array,
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
all_array_sensor_data = list(simulator.sensor_data_gen())
print(f"Generated {len(all_array_sensor_data)} pings, "
      f"{next(iter(all_array_sensor_data[0][1])).received_waveform.shape[0]} elements/ping.")

# %%
# Ground-Truth Ranges
# -------------------
# host.states holds the ship's own ping-by-ping track (driven above), 1:1 with the
# submarine's ground truth.

# %%
gt_ranges = []
for ship_state, tgt_state in zip(host.states, target_truth, strict=True):
    sx, sy, sz = (float(ship_state.state_vector[i]) for i in (0, 2, 4))
    tv = tgt_state.state_vector
    tx = float(tv[target_params["position_mapping"][0]])
    ty = float(tv[target_params["position_mapping"][1]])
    tz = float(tv[target_params["position_mapping"][2]])
    gt_ranges.append(float(np.sqrt((sx - tx) ** 2 + (sy - ty) ** 2 + (sz - tz) ** 2)))

print("\nGround-truth ranges (ship <-> submarine, closing):")
for ts, r in zip(ping_timestamps, gt_ranges, strict=True):
    expected_delay = 2.0 * r / env_params["sound_speed_ms"]
    print(f"  {ts.strftime('%H:%M:%S')}  {r:.1f} m  (round-trip delay {expected_delay:.3f} s)")

# %%
# Beamform Echoes
# ---------------

# %%
all_sensor_data = []
for ts, data_set in all_array_sensor_data:
    element_data = next(iter(data_set))
    beamformed_waveform = beamformer.beamform(
        element_data.received_waveform, steering_delays_s
    )[0]
    all_sensor_data.append((ts, {ActiveSonarSensorData(
        received_waveform=beamformed_waveform,
        transmit_pulse=element_data.transmit_pulse,
        timestamp=ts,
    )}))

# DAS crops a few edge samples, so read the true post-beamform sample count back.
n_receive = next(iter(all_sensor_data[0][1])).received_waveform.shape[0]

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
# Tracking
# --------

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
# Only valid for a constant SSP, where TL = 20*log10(R) (spherical spreading); skipped
# automatically for a non-constant SSP (e.g. Munk), where refraction breaks that assumption.

SL = signal_params["source_level_db"]
TS = cfg["target"]["target_strength_db"]
c = env_params["sound_speed_ms"]
fs = signal_params["sampling_rate_hz"]

sonar_eq_rows = []
fig_sonar_eq = None

if isinstance(ssp, Constant):
    # Monostatic active sonar equation for the received echo level:
    #   EL = SL - 2*TL + TS   (SL source level, TL one-way loss, TS target strength)
    # Bellhop's 2-D model applies the 3-D point-source correction internally, so the
    # direct-path eigenray amplitude scales as 1/R. EL is measured on the beamformed
    # waveform here, so it carries array gain and is not comparable to the omni example.
    # EL_meas is taken from the unnormalised matched-filter output at the exact direct-path
    # round-trip sample, divided by the pulse energy.
    for i, (_, sensor_data_set) in enumerate(all_sensor_data):
        R = gt_ranges[i]
        TL_geom = 20.0 * np.log10(R)
        EL_pred = SL - 2.0 * TL_geom + TS

        sd = next(iter(sensor_data_set))
        pulse_energy = float(np.sum(np.abs(sd.transmit_pulse) ** 2))

        n_fft = len(sd.received_waveform) + len(sd.transmit_pulse) - 1
        mf_raw = np.abs(np.fft.ifft(
            np.fft.fft(sd.received_waveform, n_fft)
            * np.conj(np.fft.fft(sd.transmit_pulse, n_fft)),
            n_fft,
        ))[: len(sd.received_waveform)]

        # Sample the MF at the exact direct-path round-trip delay, not its peak, which the
        # direct x surface-bounce coherent pair pulls ~2.7 ms late.
        expected_sample = min(int(2.0 * R / c * fs), len(mf_raw) - 1)
        mf_at_direct = float(mf_raw[expected_sample])

        # A_received omits the A_signal factor; it is restored via the SL term below.
        A_received = mf_at_direct / pulse_energy
        EL_meas = SL + 20.0 * np.log10(max(A_received, 1e-30))

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
# World picture
fig_world = go.Figure()

ship_x_km = [float(s.state_vector[0]) / 1000 for s in host.states]
ship_y_km = [float(s.state_vector[2]) / 1000 for s in host.states]

fig_world.add_trace(go.Scatter(
    x=ship_x_km, y=ship_y_km,
    mode="lines+markers",
    line=dict(color="#1f77b4", width=3),
    marker=dict(symbol="diamond", size=10),
    name="Ship",
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
    name="Submarine",
))

all_x = ship_x_km + tgt_x_km
all_y = ship_y_km + tgt_y_km
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
# Range detections vs ground truth
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
# MF envelope waterfall, all pings stacked
fs = signal_params["sampling_rate_hz"]
c = env_params["sound_speed_ms"]
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
    range=[0, range_axis[-1] / 1000],
    showgrid=True, gridcolor="rgba(200, 200, 200, 0.5)",
    ticks="outside", tickcolor="rgba(160, 160, 160, 1.0)",
    showline=True, linewidth=1, linecolor="rgba(160, 160, 160, 1.0)",
)
fig_mf.update_yaxes(title="Ping time", autorange="reversed")

# %%
# MF scatter, per-sample coloured by ping. Each dot is one MF output sample above the
# noise floor; unlike the waterfall heatmap it shows the spread of returns across multipath.
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
# Sonar equation verification, constant SSP only
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
    ("as_bow_world.pdf", fig_world),
    ("as_bow_range_detections.pdf", fig_range),
    ("as_bow_mf_waterfall.pdf", fig_mf),
    ("as_bow_mf_scatter.pdf", fig_mf_scatter),
    ("as_bow_sonar_eq.pdf", fig_sonar_eq),
]:
    if fig is None:
        continue
    fig.write_image(str(out_dir / fname))
    print(f"Saved {out_dir / fname}")

print()
print("=" * 60)
print("Active Sonar Scenario Summary (Ship vs Submarine, Bow Array)")
print("=" * 60)
print(f"Ship start position: ({bow_params['x_m']:.0f}, {bow_params['y_m']:.0f}) m, "
      f"depth {bow_params['depth_m']:.0f} m")
print(f"Ship end position:   ({ship_x_km[-1] * 1000:.0f}, {ship_y_km[-1] * 1000:.0f}) m")
print(f"Ship speed:          {bow_params['speed_ms']:.1f} m/s")
print(f"Submarine speed:     {abs(target_params['start_vector'][1]):.1f} m/s (closing)")
print(f"Range: {gt_ranges[0]:.0f} m -> {gt_ranges[-1]:.0f} m over the run")
print(f"Dome elements:       {num_elements}  (radius {bow_params['dome_radius_m']:.1f} m, "
      f"spacing {bow_params['element_spacing_m']:.2f} m)")
print(f"Water depth:        {env_params['water_depth_m']:.0f} m")
print(f"Sound speed:        {env_params['sound_speed_ms']:.0f} m/s")
print(f"LFM bandwidth:      {signal_params['freq_max_hz'] - signal_params['freq_min_hz']:.0f} Hz")
bw = signal_params["freq_max_hz"] - signal_params["freq_min_hz"]
print(f"Range resolution:   ~{c / (2 * bw):.1f} m  (c / 2B)")
print(f"Target strength:    {cfg['target']['target_strength_db']:.0f} dB")
print(f"Detections found:   {sum(len(d) for _, d in all_detections)} / {cfg['sim']['n_pings']} pings")
