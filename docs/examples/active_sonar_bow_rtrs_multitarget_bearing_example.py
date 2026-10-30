"""
================================================================
Active Sonar Multi-Target Example (Bow Array, RTRS, DAS + MVDR)
================================================================

RTRS counterpart of ``active_sonar_bow_bellhop_2D_multitarget_bearing_example.py``:
same three-target, bow-array, DAS-vs-MVDR bearing scenario, but the per-element echoes
come from :class:`.RtrsActiveSonarSimulatorArray` (true 3-D ray tracing via ``rtrs``) instead
of :class:`.BellhopActiveSonarSimulatorArray` (2-D Bellhop eigenrays + a far-field plane-wave
correction applied across the dome).

Why this comparison matters
----------------------------
2-D Bellhop has no azimuth resolution: a single vertical range-depth slice is computed once
per target, from the array's *reference* point, and every element's copy of it is then
time/phase-shifted by an explicit plane-wave formula built from the target's true bearing
(see :class:`.BellhopActiveSonarSimulatorArray`'s docstring). That correction is exact only in
the far-field/small-aperture limit -- real near-field or environment-driven differences
*between elements* are invisible to it by construction.

``rtrs`` instead ray-traces each dome element's own transfer function directly in 3-D
(:meth:`~bluepebble.models.propagation.rtrsAcousticPropagationModel.propagate_spectrum`
called with the whole array as receivers), so cross-element timing comes out of genuine
per-element geometry rather than a bolted-on correction. For the small dome used here
(radius ``<<`` target range) the two should agree closely on bearing -- this example exists
to make that comparison directly, on the same scenario, same beamformers, same detection
logic as the Bellhop version.

Notes / limitations
--------------------
- :class:`.RtrsActiveSonarSimulatorArray` still only beamforms *receive*; transmit is a
  single point at the array reference (no TX beamforming), matching the Bellhop array
  simulator's convention.
- ``use_reciprocity=True`` (default) gets the outbound leg from the same ray-tracing pass
  as the per-element inbound legs via acoustic reciprocity, instead of firing a second,
  separate outbound call -- see :class:`.RtrsActiveSonarSimulatorArray`'s docstring.
- ``rtrsAcousticPropagationModel`` launches a ray fan per call (azimuth x elevation); runtime
  scales with the dome's element count only through how many receivers are packed into that
  one call, not by re-launching per element -- but it is still far more expensive per call
  than 2-D Bellhop, so this scenario uses fewer pings/targets than a Bellhop run might.
- Environment is the same simplest-useful case as the Bellhop version -- flat 300 m
  bathymetry, constant 1500 m/s sound speed -- so any bearing differences between the two
  methods are attributable to the propagation/array modelling, not environmental complexity.
"""  # noqa: D205, D212, D400, D415

# %%
# Imports
# -------

# %%
from datetime import datetime, timedelta

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import find_peaks, get_window
from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel,
    ConstantVelocity,
)
from stonesoup.types.array import StateVector
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.types.state import State

import bluepebble
from bluepebble.detector import DetectorChain, OSCFARDetector, PeakDetector, ThresholdDetector
from bluepebble.models.environment import Constant, FlatBathymetry
from bluepebble.models.propagation import rtrsAcousticPropagationModel
from bluepebble.models.scattering import ConstantTargetStrength
from bluepebble.platform import HostPlatform
from bluepebble.sensors import BowArraySensor
from bluepebble.signal.active import LFMSignal
from bluepebble.signal.random import ColouredNoiseSignal
from bluepebble.sigproc import (
    DelayAndSumBeamformer,
    MatchedFilter,
    MinimumVarianceDistortionlessResponseBeamformer,
    SteeringCalculator,
    parabolic_peak_offset,
)
from bluepebble.simulator import RtrsActiveSonarSimulatorArray

# %%
# Configuration
# -------------

# %%
seed = 7
bluepebble.set_seed(seed)

start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

sim_params = {
    "ping_interval_s": 20.0,
    "n_pings": 3,
    "receive_duration_s": 4.0,
}

ping_timestamps = [
    start_time + timedelta(seconds=i * sim_params["ping_interval_s"])
    for i in range(sim_params["n_pings"])
]

env_params = {
    "water_depth_m": 300.0,
    "sound_speed_ms": 1500.0,
}

signal_params = {
    "freq_min_hz": 1500.0,
    "freq_max_hz": 2500.0,
    "duration_s": 0.5,
    "rise_time_s": 0.03,
    "source_level_db": 200.0,
    "sampling_rate_hz": 8000.0,
}

platform_params = {
    "x_m": 0.0,
    "y_m": 0.0,
    "depth_m": 8.0,      # keep > 0: a source/receiver at exactly 0 m depth degenerates
    "speed_ms": 3.0,
    "heading_deg": 0.0,
}

# Bow dome geometry (mounted on the ownship platform above) -- same as the Bellhop example.
bow_params = {
    "dome_radius_m": 4.0,
    "azimuth_extent_rad": np.radians(90.0),
    "elevation_extent_rad": np.radians(16.0),
    "element_spacing_m": 0.20,
    "element_size_m": 0.10,
    "array_depth_offset_m": 3,
}

rtrs_params = {
    "step_m": 20.0,
    "azimuth_search_width": 1.0,
    "azimuth_resolution": 0.5,
    "elevation_range": (-30.0, 30.0),
    "elevation_resolution": 2.0,
    "energy_threshold": 1e-3,
    "use_reciprocity": True,
}

# Each target set from a (range, bearing, depth) triple so the geometry is explicit.
def _target_start(range_m: float, bearing_deg: float, depth_m: float,
                  vx: float, vy: float) -> np.ndarray:
    az = np.radians(bearing_deg)
    return np.array([range_m * np.cos(az), vx, range_m * np.sin(az), vy, -depth_m, 0.0])

target_params = {
    "position_mapping": [0, 2, 4],
    "target_strength_db": 15.0,
    "starts": [
        _target_start(2000.0, -32.0, 140.0, vx=0.0, vy=0.5),
        _target_start(1500.0, 0.0, 160.0, vx=-3.0, vy=0.0),
        _target_start(2500.0, 28.0, 120.0, vx=0.0, vy=-3.0),
    ],
}

ambient_noise_params = {
    "amplitude_upa": 10 ** (72.0 / 20.0),
    "spectral_exponent": -1,
}

bf_params = {
    "steering_azimuths_rad": np.radians(np.arange(-55.0, 55.0 + 0.5, 0.5)),
    "steering_elevation_rad": np.radians(-3.0),
    "das_shading": "hann",
    "mvdr_nfft": 128,
    "mvdr_overlap": 64,
    "mvdr_fmin_hz": 1400.0,
    "mvdr_fmax_hz": 2600.0,
    "mvdr_search_halfwidth_beams": 12,
}

det_params = {
    "min_range_m": 150.0,
    "replica_window": "hann",              # matched-filter replica taper (range sidelobes)
    "surface_floor_db": -60.0,
    "bearing_lobe_dyn_range_db": 10.0,     # lobes within this of the profile peak
    "bearing_lobe_min_sep_beams": 12,
    "bearing_lobe_min_prominence_db": 2.5,
    "bearing_cut_agreement_deg": 4.0,      # reject lobe if range-cut bearing disagrees
    "range_peak_min_db": -16.0,
    "range_peak_min_sep_samples": 300,
    "cfar": {
        "num_guard_cells": 15,
        "num_training_cells": 40,
        "rank": 55,
        "threshold_factor": 8.0,
    },
}

fs = signal_params["sampling_rate_hz"]
c = env_params["sound_speed_ms"]

# %%
# Scenario Build
# -------------

# %%
transition_model = CombinedLinearGaussianTransitionModel(
    [ConstantVelocity(0), ConstantVelocity(0), ConstantVelocity(0)]
)

_heading_rad = np.radians(platform_params["heading_deg"])
ship_vx = platform_params["speed_ms"] * np.cos(_heading_rad)
ship_vy = platform_params["speed_ms"] * np.sin(_heading_rad)

host = HostPlatform(
    states=[State(
        StateVector([
            platform_params["x_m"], ship_vx,
            platform_params["y_m"], ship_vy,
            -platform_params["depth_m"], 0.0,
        ]),
        timestamp=start_time,
    )],
    position_mapping=[0, 2, 4],
    transition_models=[transition_model],
    transition_times=[timedelta(seconds=sim_params["ping_interval_s"] * sim_params["n_pings"])],
)

bow_array = BowArraySensor(
    host=host,
    dome_radius_m=bow_params["dome_radius_m"],
    azimuth_extent_rad=bow_params["azimuth_extent_rad"],
    elevation_extent_rad=bow_params["elevation_extent_rad"],
    element_spacing_m=bow_params["element_spacing_m"],
    element_size_m=bow_params["element_size_m"],
    freq_max_hz=signal_params["freq_max_hz"],
    array_depth_offset_m=bow_params["array_depth_offset_m"],
)
num_elements = bow_array.array.num_sensors
print(f"Bow dome: {num_elements} elements, {bow_params['dome_radius_m']:.1f} m radius")

target_truths: list[GroundTruthPath] = []
for start_vec in target_params["starts"]:
    states = [GroundTruthState(start_vec, timestamp=start_time)]
    for ts in ping_timestamps[1:]:
        dt = ts - states[-1].timestamp
        states.append(GroundTruthState(
            transition_model.function(states[-1], noise=False, time_interval=dt),
            timestamp=ts,
        ))
    target_truths.append(GroundTruthPath(states))

# Drive the ownship to every ping timestamp up front; the array simulator does not advance
# the host itself. With speed_ms = 0 this would be a no-op.
for ts in ping_timestamps[1:]:
    host.move(ts)

# %%
# Ground-Truth Bearings and Ranges
# --------------------------------

# %%
pm = target_params["position_mapping"]
array_refs = [
    bow_array.get_platform_state_at(ts).array.ref_state_vector.flatten()
    for ts in ping_timestamps
]

gt = []  # gt[ping][target] = dict(range_m, bearing_deg)
for p_idx in range(sim_params["n_pings"]):
    per_ping = []
    for path in target_truths:
        tv = path.states[p_idx].state_vector
        txyz = np.array([float(tv[pm[0]]), float(tv[pm[1]]), float(tv[pm[2]])])
        to_t = txyz - array_refs[p_idx]
        per_ping.append({
            "range_m": float(np.linalg.norm(to_t)),
            "bearing_deg": float(np.degrees(np.arctan2(to_t[1], to_t[0]))),
        })
    gt.append(per_ping)

print("\nGround truth (range m / bearing deg):")
for p_idx, ts in enumerate(ping_timestamps):
    row = "   ".join(f"T{k+1}: {g['range_m']:6.0f} / {g['bearing_deg']:+5.1f}"
                     for k, g in enumerate(gt[p_idx]))
    print(f"  {ts.strftime('%H:%M:%S')}  {row}")

# %%
# Beamformers and Steering
# -----------------------

# %%
ssp = Constant(speed=env_params["sound_speed_ms"])
bathymetry = FlatBathymetry(depth=-env_params["water_depth_m"])

azimuths = bf_params["steering_azimuths_rad"]
beam_step_rad = float(azimuths[1] - azimuths[0])
n_beams = len(azimuths)
azimuths_deg = np.degrees(azimuths)

steering_calculator = SteeringCalculator(
    ssp=ssp,
    steering_azimuths_rad=azimuths,
    steering_elevations_rad=np.array([bf_params["steering_elevation_rad"]]),
)
steering_delays_s = steering_calculator.calculate(bow_array)  # (n_beams, n_elements)

das = DelayAndSumBeamformer(
    sampling_rate_hz=fs,
    domain="time",
    shading=get_window(bf_params["das_shading"], num_elements),
)
mvdr = MinimumVarianceDistortionlessResponseBeamformer(
    sampling_rate_hz=fs,
    nfft=bf_params["mvdr_nfft"],
    overlap=bf_params["mvdr_overlap"],
    fmin=bf_params["mvdr_fmin_hz"],
    fmax=bf_params["mvdr_fmax_hz"],
)
mvdr_hop = bf_params["mvdr_nfft"] - bf_params["mvdr_overlap"]

# %%
# Simulation
# ----------

# %%
prop_model = rtrsAcousticPropagationModel(
    ssp=ssp,
    bathymetry=bathymetry,
    step_m=rtrs_params["step_m"],
    azimuth_search_width=rtrs_params["azimuth_search_width"],
    azimuth_resolution=rtrs_params["azimuth_resolution"],
    elevation_range=rtrs_params["elevation_range"],
    elevation_resolution=rtrs_params["elevation_resolution"],
)
scattering_model = ConstantTargetStrength(target_strength_db=target_params["target_strength_db"])

ambient_noise_model = ColouredNoiseSignal(
    amplitude_upa=ambient_noise_params["amplitude_upa"],
    spectral_exponent=ambient_noise_params["spectral_exponent"],
    duration_s=signal_params["duration_s"] + sim_params["receive_duration_s"],
    sampling_rate_hz=fs,
)

pulse = LFMSignal(
    freq_min_hz=signal_params["freq_min_hz"],
    freq_max_hz=signal_params["freq_max_hz"],
    rise_time_s=signal_params["rise_time_s"],
    source_level_db=signal_params["source_level_db"],
    duration_s=signal_params["duration_s"],
    sampling_rate_hz=fs,
)

simulator = RtrsActiveSonarSimulatorArray(
    platform=bow_array,
    propagation_model=prop_model,
    signal=pulse,
    scattering_model=scattering_model,
    ground_truth_paths=target_truths,
    ping_timestamps=ping_timestamps,
    target_position_mapping=pm,
    receive_duration_s=sim_params["receive_duration_s"],
    noise_model=ambient_noise_model,
    energy_threshold=rtrs_params["energy_threshold"],
    use_reciprocity=rtrs_params["use_reciprocity"],
)

print("\nRunning multi-target RTRS active sonar simulation...")
pings = list(simulator.sensor_data_gen())
print(f"Generated {len(pings)} pings, {next(iter(pings[0][1])).received_waveform.shape} per ping.")

# %%
# Processing Helpers
# -----------------
#
# Identical to the Bellhop example's -- the beamformers and detection logic only ever see
# ``received_waveform``/``transmit_pulse``, so nothing here needs to know which propagation
# method produced the data.

# %%
matched_filter = MatchedFilter(window=det_params["replica_window"])


def refine_bearing(cut_db: np.ndarray, k_lo: int, k_hi: int) -> tuple[int, float]:
    """Argmax of a beam response between ``k_lo`` and ``k_hi``, parabola-refined (deg)."""
    k = k_lo + int(np.argmax(cut_db[k_lo:k_hi + 1]))
    bearing = azimuths[k] + parabolic_peak_offset(cut_db, k) * beam_step_rad
    return k, float(np.degrees(bearing))


range_chain = DetectorChain(algorithms=[
    OSCFARDetector(
        num_guard_cells=det_params["cfar"]["num_guard_cells"],
        num_training_cells=det_params["cfar"]["num_training_cells"],
        rank=det_params["cfar"]["rank"],
        threshold_factor=det_params["cfar"]["threshold_factor"],
        mode="valid",
    ),
    ThresholdDetector(threshold=det_params["range_peak_min_db"]),
    PeakDetector(distance=det_params["range_peak_min_sep_samples"]),
])

# %%
# DAS: Bearing x Range Surface + Detection
# --------------------------------------

# %%
das_results = []       # das_results[ping] = list of dict(range_m, bearing_deg, amp_db)
das_surfaces_db = []
das_range_axes = []

for _ts, data_set in pings:
    sensor_data = next(iter(data_set))
    beams = das.beamform(sensor_data.received_waveform, steering_delays_s)   # (n_beams, n_crop)
    surface = matched_filter.filter(beams, sensor_data.transmit_pulse)
    surface_db = 20.0 * np.log10(np.maximum(surface / surface.max(), 1e-6))
    surface_db = np.maximum(surface_db, det_params["surface_floor_db"])

    n_samp = surface_db.shape[1]
    range_axis = np.arange(n_samp) / fs * c / 2.0
    min_sample = int(2.0 * det_params["min_range_m"] / c * fs)

    bearing_profile = surface_db[:, min_sample:].max(axis=1)
    lobe_idx, _ = find_peaks(
        bearing_profile,
        height=bearing_profile.max() - det_params["bearing_lobe_dyn_range_db"],
        distance=det_params["bearing_lobe_min_sep_beams"],
        prominence=det_params["bearing_lobe_min_prominence_db"],
    )

    ping_targets = []  # one detection per resolved bearing lobe
    for k in lobe_idx:
        samples = range_chain.detect(surface_db[k])[:, 0].astype(int)
        samples = samples[samples >= min_sample]
        if samples.size == 0:
            continue
        sample = int(samples.min())  # earliest return == direct path (multipath is later)

        k_lo = max(k - 2 * det_params["bearing_lobe_min_sep_beams"], 1)
        k_hi = min(k + 2 * det_params["bearing_lobe_min_sep_beams"], n_beams - 2)
        k_ref, bearing_deg = refine_bearing(surface_db[:, sample], k_lo, k_hi)
        # A true target's range-cut peaks at its own lobe bearing; a sidelobe ghost
        # of a stronger target peaks back at that target's bearing, not the ghost's.
        if abs(bearing_deg - azimuths_deg[k]) > det_params["bearing_cut_agreement_deg"]:
            continue

        ping_targets.append({
            "range_m": sample / fs * c / 2.0,
            "bearing_deg": bearing_deg,
            "amp_db": float(surface_db[k_ref, sample]),
        })
    ping_targets.sort(key=lambda d: d["bearing_deg"])
    das_results.append(ping_targets)
    das_surfaces_db.append(surface_db)
    das_range_axes.append(range_axis)

# %%
# MVDR: Bearing-Only Estimate at Each DAS Range
# --------------------------------------------

# %%
mvdr_results = []      # mvdr_results[ping][k] = dict(bearing_deg), aligned to das_results
mvdr_specs = []         # mvdr_specs[ping][k] = normalised bearing response (dB) over the fan

hw = bf_params["mvdr_search_halfwidth_beams"]

for (_ts, data_set), das_ping in zip(pings, das_results, strict=True):
    sensor_data = next(iter(data_set))
    power_map = mvdr.beamform(sensor_data.received_waveform, steering_delays_s)
    n_frames = power_map.shape[1]

    ping_bearings, ping_specs = [], []
    for tgt in das_ping:
        centre_frame = int(2.0 * tgt["range_m"] / c * fs) // mvdr_hop
        f_lo = max(centre_frame - 3, 0)
        f_hi = min(centre_frame + 4, n_frames)
        spec = power_map[:, f_lo:f_hi].mean(axis=1)
        spec_db = 10.0 * np.log10(np.maximum(spec / spec.max(), 1e-6))

        das_k = int(np.argmin(np.abs(azimuths_deg - tgt["bearing_deg"])))
        _, bearing_deg = refine_bearing(spec_db, max(das_k - hw, 1), min(das_k + hw, n_beams - 2))
        ping_bearings.append({"bearing_deg": bearing_deg})
        ping_specs.append(spec_db)

    mvdr_results.append(ping_bearings)
    mvdr_specs.append(ping_specs)

# %%
# Results Table
# ------------

# %%
def match_to_gt(estimates: list[dict], gt_ping: list[dict]) -> list[dict | None]:
    """Nearest-bearing association of each estimate to a ground-truth target."""
    out, used = [], set()
    for est in estimates:
        best_j, best_d = None, np.inf
        for j, g in enumerate(gt_ping):
            if j in used:
                continue
            d = abs(est["bearing_deg"] - g["bearing_deg"])
            if d < best_d:
                best_j, best_d = j, d
        if best_j is not None:
            used.add(best_j)
        out.append(gt_ping[best_j] if best_j is not None else None)
    return out


print("\n" + "=" * 80)
print("Per-ping estimates  (DAS range + bearing, MVDR bearing)  vs ground truth  [RTRS]")
print("=" * 80)
for p_idx, ts in enumerate(ping_timestamps):
    print(f"\n{ts.strftime('%H:%M:%S')}")
    print(f"  {'tgt':4}  {'GT rng':>7}  {'GT brg':>7}  {'DAS rng':>8}  {'rng err':>8}  "
          f"{'DAS brg':>8}  {'DAS err':>8}  {'MVDR brg':>9}  {'MVDR err':>9}")
    matches = match_to_gt(das_results[p_idx], gt[p_idx])
    for k, (das_est, g) in enumerate(zip(das_results[p_idx], matches, strict=True)):
        mvdr_est = mvdr_results[p_idx][k]
        if g is None:
            print(f"  T?    (unmatched)  DAS {das_est['bearing_deg']:+.1f} deg "
                  f"/ {das_est['range_m']:.0f} m")
            continue
        d_rng = das_est["range_m"] - g["range_m"]
        d_das = das_est["bearing_deg"] - g["bearing_deg"]
        d_mvdr = mvdr_est["bearing_deg"] - g["bearing_deg"]
        print(f"  T{k+1:<3}  {g['range_m']:7.0f}  {g['bearing_deg']:+7.1f}  "
              f"{das_est['range_m']:8.0f}  {d_rng:+8.0f}  "
              f"{das_est['bearing_deg']:+8.1f}  {d_das:+8.2f}  "
              f"{mvdr_est['bearing_deg']:+9.1f}  {d_mvdr:+9.2f}")

# %%
# Figures
# -------

# %%
_PLOT_PING = 0
_font = dict(family="Times New Roman", size=15, color="black")

# World picture
fig_world = go.Figure()
fig_world.add_trace(go.Scatter(
    x=[r[0] / 1000 for r in array_refs], y=[r[1] / 1000 for r in array_refs],
    mode="lines+markers",
    line=dict(color="#1f77b4", width=2),
    marker=dict(symbol="diamond", size=10, color="#1f77b4"), name="Bow array",
))
for k, path in enumerate(target_truths):
    fig_world.add_trace(go.Scatter(
        x=[float(s.state_vector[pm[0]]) / 1000 for s in path.states],
        y=[float(s.state_vector[pm[1]]) / 1000 for s in path.states],
        mode="lines+markers", line=dict(width=2), marker=dict(size=7), name=f"T{k+1}",
    ))
fig_world.update_layout(
    width=600, height=600, font=_font, plot_bgcolor="white", paper_bgcolor="white",
    legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center"),
    yaxis=dict(scaleanchor="x", scaleratio=1),
)
fig_world.update_xaxes(title="X (km)", showgrid=True, gridcolor="rgba(200,200,200,0.5)")
fig_world.update_yaxes(title="Y (km)", showgrid=True, gridcolor="rgba(200,200,200,0.5)")

# DAS matched-filter range profile per target beam (one ping), per-sample scatter.
surface_db = das_surfaces_db[_PLOT_PING]
range_axis = das_range_axes[_PLOT_PING]
_scatter_floor_db = -45.0
_target_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

fig_scatter = go.Figure()
matches_scatter = match_to_gt(das_results[_PLOT_PING], gt[_PLOT_PING])
for k, tgt in enumerate(das_results[_PLOT_PING]):
    color = _target_colors[k % len(_target_colors)]
    beam_k = int(np.argmin(np.abs(azimuths_deg - tgt["bearing_deg"])))
    row_db = surface_db[beam_k]
    mask = row_db >= _scatter_floor_db

    fig_scatter.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers", marker=dict(size=10, color=color),
        name=f"T{k+1}  (beam {azimuths_deg[beam_k]:+.1f} deg)", showlegend=True,
    ))
    fig_scatter.add_trace(go.Scatter(
        x=range_axis[mask], y=row_db[mask], mode="markers",
        marker=dict(size=3, color=color), showlegend=False,
    ))
    g = matches_scatter[k]
    if g is not None:
        fig_scatter.add_vline(x=g["range_m"], line=dict(color=color, dash="dash", width=1))

fig_scatter.update_layout(
    width=880, height=420, font=_font, plot_bgcolor="white", paper_bgcolor="white",
    legend=dict(title="Target (dashed = GT range)", x=0.98, y=0.98,
                xanchor="right", yanchor="top", bgcolor="rgba(255,255,255,0.85)", borderwidth=1),
    margin=dict(b=60, t=20, l=20, r=20),
)
fig_scatter.update_xaxes(
    title="Slant range (m)", range=[0, range_axis[-1]],
    showgrid=True, gridcolor="rgba(200,200,200,0.5)",
)
fig_scatter.update_yaxes(
    title="MF amplitude (dB)", range=[_scatter_floor_db, 2],
    showgrid=True, gridcolor="rgba(200,200,200,0.5)",
)

# DAS vs MVDR bearing response at each target's range (one ping)
n_t = max(len(das_results[_PLOT_PING]), 1)
matches_plot = match_to_gt(das_results[_PLOT_PING], gt[_PLOT_PING])
fig_bear = make_subplots(
    rows=1, cols=n_t,
    subplot_titles=[f"T{k+1}  (R~{t['range_m']:.0f} m)"
                    for k, t in enumerate(das_results[_PLOT_PING])] or ["(no detections)"],
)
for k, tgt in enumerate(das_results[_PLOT_PING]):
    centre_sample = min(int(2.0 * tgt["range_m"] / c * fs), surface_db.shape[1] - 1)
    das_cut = surface_db[:, centre_sample]
    das_cut = das_cut - das_cut.max()
    mvdr_cut = mvdr_specs[_PLOT_PING][k]
    g = matches_plot[k]

    fig_bear.add_trace(go.Scatter(
        x=azimuths_deg, y=das_cut, mode="lines",
        line=dict(color="#1f77b4", width=2), name="DAS (shaded)", showlegend=(k == 0),
    ), row=1, col=k + 1)
    fig_bear.add_trace(go.Scatter(
        x=azimuths_deg, y=mvdr_cut, mode="lines",
        line=dict(color="#d62728", width=2), name="MVDR", showlegend=(k == 0),
    ), row=1, col=k + 1)
    if g is not None:
        fig_bear.add_vline(x=g["bearing_deg"], line=dict(color="black", dash="dash", width=1),
                           row=1, col=k + 1)
        fig_bear.update_xaxes(range=[g["bearing_deg"] - 25, g["bearing_deg"] + 25],
                              row=1, col=k + 1)
    fig_bear.update_xaxes(title="Bearing (deg)", row=1, col=k + 1)
    fig_bear.update_yaxes(title="Norm. response (dB)" if k == 0 else None,
                          range=[-35, 2], row=1, col=k + 1)
fig_bear.update_layout(
    width=340 * n_t, height=380, font=_font, plot_bgcolor="white", paper_bgcolor="white",
    legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center"),
)

# %%
# Export
# ------

# %%
from pathlib import Path

out_dir = Path(__file__).resolve().parent / "figs_active"
out_dir.mkdir(parents=True, exist_ok=True)

for fname, fig in [
    ("as_bow_rtrs_mt_world.pdf", fig_world),
    ("as_bow_rtrs_mt_mf_scatter.pdf", fig_scatter),
    ("as_bow_rtrs_mt_bearing_das_vs_mvdr.pdf", fig_bear),
]:
    fig.write_image(str(out_dir / fname))
    print(f"Saved {out_dir / fname}")

print("\n" + "=" * 60)
print("Multi-Target Active Sonar Summary (Bow Array, RTRS)")
print("=" * 60)
b_hz = signal_params["freq_max_hz"] - signal_params["freq_min_hz"]
f_lo_hz = signal_params["freq_min_hz"]
f_hi_hz = signal_params["freq_max_hz"]
ts_db = target_params["target_strength_db"]
print(f"Ownship:           ({platform_params['x_m']:.0f}, {platform_params['y_m']:.0f}) m, "
      f"depth {platform_params['depth_m']:.0f} m, {platform_params['speed_ms']:.1f} m/s "
      f"@ {platform_params['heading_deg']:.0f} deg")
print(f"Dome elements:      {num_elements}")
print(f"Steering fan:       {azimuths_deg[0]:.0f} to {azimuths_deg[-1]:.0f} deg "
      f"@ {np.degrees(beam_step_rad):.2f} deg")
print(f"LFM:                {f_lo_hz:.0f}-{f_hi_hz:.0f} Hz, "
      f"{signal_params['duration_s']:.2f} s, fs={fs:.0f} Hz")
print(f"Range resolution:   ~{c / (2 * b_hz):.2f} m  (c / 2B)")
print(f"Targets:            {len(target_truths)}  (TS {ts_db:.0f} dB, shared)")
print(f"Pings:              {sim_params['n_pings']}")
print(f"use_reciprocity:    {rtrs_params['use_reciprocity']}")
