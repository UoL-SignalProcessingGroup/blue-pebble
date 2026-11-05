r"""
================================
Detecting Echoes with a Sonobuoy
================================

This example pings from a single sonobuoy hydrophone, detects the echoes and tracks a target in
range. It shows the active sonar chain at its simplest, with no beamforming:

- **Simulation**: :class:`~bluepebble.simulator.ActiveSonarSimulator` synthesises each ping's
  echoes through a Bellhop propagation model.
- **Detection**: :class:`~bluepebble.detector.ActiveSonarDetector` matched-filters the hydrophone
  signal and runs CFAR (Constant False Alarm Rate) detection along range. Its threshold is
  calibrated from noise-only pings for a stated probability of false alarm (Pfa).
- **Tracking**: a one-dimensional Kalman filter tracks an opening target's range.

Five static targets, from 500 m to 2.5 km, check the chain in two ways.

- **Timing**: each one's detected range should fall within two resolution cells of its true
  slant range.
- **Echo level**: each one's echo level is compared with the sonar equation.

A single hydrophone measures range only, so the static targets' bearings play no part.

Assumptions
-----------
- Constant 1500 m/s sound speed over a flat 200 m seabed matched to the water, so it reflects
  nothing. Each leg of the echo, buoy to target and target to buoy, then has two straight-line
  paths, direct and surface-reflected. A reflective seabed would add a train of bottom bounces
  up to 100 m behind each echo, some as strong as the direct path.
- Each target is a point scatterer with a 15 dB target strength. There is no reverberation.
- Geometry is frozen during each ping, so there is no Doppler.
"""  # noqa: D205, D212, D400, D415

# %%
# Imports
# -------

from datetime import datetime, timedelta

import numpy as np
import plotly.graph_objects as go
from stonesoup.dataassociator.neighbour import NearestNeighbour
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.measures import Mahalanobis
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.models.transition.linear import ConstantVelocity
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.types.array import StateVector
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.types.state import GaussianState
from stonesoup.types.track import Track
from stonesoup.updater.kalman import KalmanUpdater

from bluepebble import set_seed
from bluepebble.detector import (
    ActiveSonarDetector,
    NoiseCalibrator,
    OSCFARDetector,
    matched_filter_scans_from_sensor_data,
)
from bluepebble.models import ConstantTargetStrength
from bluepebble.models.environment import Constant, FlatBathymetry
from bluepebble.models.propagation import BellhopAcousticPropagationModel
from bluepebble.platform import Sonobuoy
from bluepebble.signal import LFMSignal
from bluepebble.signal.random import ColouredNoiseSignal
from bluepebble.sigproc import MatchedFilter
from bluepebble.simulator import ActiveSonarSimulator

# %%
# Configuration
# -------------
#
# The pulse is a 1 s linear frequency-modulated (LFM) sweep over 800-1200 Hz. With sound speed
# :math:`c` and bandwidth :math:`B`, the range resolution is :math:`c / 2B = 1.875` m. The listen
# window of 5 s covers echoes out to 3.75 km.

set_seed(7)

START = datetime(2026, 1, 1)
NUM_PINGS = 5
PING_INTERVAL = timedelta(seconds=30)
PING_TIMES = [START + i * PING_INTERVAL for i in range(NUM_PINGS)]

SOUND_SPEED_MPS = 1500.0
WATER_DEPTH_M = 200.0
HYDROPHONE_DEPTH_M = 50.0

SAMPLING_RATE_HZ = 10_000
FREQ_MIN_HZ, FREQ_MAX_HZ = 800.0, 1200.0
BANDWIDTH_HZ = FREQ_MAX_HZ - FREQ_MIN_HZ
RANGE_RESOLUTION_M = SOUND_SPEED_MPS / (2 * BANDWIDTH_HZ)
SOURCE_LEVEL_DB = 200.0
TARGET_STRENGTH_DB = 15.0
NOISE_LEVEL_DB = 70.0
LISTEN_DURATION_S = 5.0
TARGET_PFA = 1e-4

# %%
# Scenario
# --------
#
# - **Sonobuoy**: floats at the origin with its hydrophone 50 m down. The fixed buoy is moved
#   through every ping time so its hydrophone has a state at each one.
# - **Static targets**: five, 100 m deep at 500-2500 m.
# - **Moving target**: a sixth, also 100 m deep, opening from 3 km at 3 m/s on a bearing of 45
#   degrees.

buoy = Sonobuoy(
    states=GroundTruthState(StateVector([0.0, 0.0, 0.0]), timestamp=START),
    position_mapping=[0, 1, 2],
    hydrophone_depth_m=HYDROPHONE_DEPTH_M,
)
for ping_time in PING_TIMES[1:]:
    buoy.move(ping_time)
hydrophone_xyz = np.array([0.0, 0.0, -HYDROPHONE_DEPTH_M])


def target_path(x_m, y_m, depth_m, vx_mps=0.0, vy_mps=0.0):
    """Return a constant-velocity target path with a state at every ping."""
    return GroundTruthPath(
        [
            GroundTruthState(
                StateVector([x_m + vx_mps * t, vx_mps, y_m + vy_mps * t, vy_mps, -depth_m, 0.0]),
                timestamp=ping_time,
                metadata={"position_mapping": [0, 2, 4]},
            )
            for ping_time in PING_TIMES
            for t in [(ping_time - START).total_seconds()]
        ]
    )


static_ranges_m = [500.0, 1000.0, 1500.0, 2000.0, 2500.0]
static_bearings_deg = [0.0, 60.0, 120.0, 180.0, 270.0]
static_targets = [
    target_path(r * np.cos(np.radians(b)), r * np.sin(np.radians(b)), 100.0)
    for r, b in zip(static_ranges_m, static_bearings_deg, strict=True)
]
heading = np.radians(45.0)
moving_target = target_path(
    3000.0 * np.cos(heading),
    3000.0 * np.sin(heading),
    100.0,
    vx_mps=3.0 * np.cos(heading),
    vy_mps=3.0 * np.sin(heading),
)
targets = [*static_targets, moving_target]


def slant_range_m(state):
    """Return a target state's slant range from the hydrophone."""
    return float(np.linalg.norm(np.ravel(state.state_vector[[0, 2, 4]]) - hydrophone_xyz))


# %%
# Plan view
# ---------

figure = go.Figure()
for index, path in enumerate(targets):
    xy = np.array([[float(s.state_vector[0, 0]), float(s.state_vector[2, 0])] for s in path])
    figure.add_trace(
        go.Scatter(
            x=xy[:, 0],
            y=xy[:, 1],
            mode="lines+markers",
            name="Moving target" if path is moving_target else f"Static target {index + 1}",
        )
    )
figure.add_trace(
    go.Scatter(x=[0.0], y=[0.0], mode="markers", marker={"symbol": "star", "size": 14},
               name="Sonobuoy")
)
figure.update_layout(
    template="plotly_white",
    title="Plan View",
    xaxis_title="x (m)",
    yaxis_title="y (m)",
    yaxis_scaleanchor="x",
).show()

# %%
# Simulation
# ----------
#
# Bellhop traces both legs at every ping. Ambient noise is pink (power falling as :math:`1/f`)
# at 70 dB re 1 µPa across the band.

pulse = LFMSignal(
    freq_min_hz=FREQ_MIN_HZ,
    freq_max_hz=FREQ_MAX_HZ,
    rise_time_s=0.05,
    duration_s=1.0,
    sampling_rate_hz=SAMPLING_RATE_HZ,
    source_level_db=SOURCE_LEVEL_DB,
)
propagation_model = BellhopAcousticPropagationModel(
    ssp=Constant(speed=SOUND_SPEED_MPS),
    bathymetry=FlatBathymetry(depth=-WATER_DEPTH_M),
    bottom_sound_speed_mps=SOUND_SPEED_MPS,
    bottom_density_g_cm3=1.0,
    bottom_attenuation_db_per_wavelength=0.0,
)


def simulator(ground_truth_paths, seed):
    """Return a simulator for the buoy's hydrophone with its own noise seed."""
    return ActiveSonarSimulator(
        array=buoy.hydrophone,
        propagation_model=propagation_model,
        signal=pulse,
        scattering_models=ConstantTargetStrength(target_strength_db=TARGET_STRENGTH_DB),
        ground_truth_paths=ground_truth_paths,
        listen_duration_s=LISTEN_DURATION_S,
        noise_model=ColouredNoiseSignal(
            amplitude_upa=10 ** (NOISE_LEVEL_DB / 20),
            spectral_exponent=-1,
            duration_s=LISTEN_DURATION_S,
            sampling_rate_hz=SAMPLING_RATE_HZ,
            seed=seed,
        ),
    )


pings = [next(iter(data)) for _, data in simulator(targets, seed=1).sensor_data_gen()]

# %%
# Matched-filter waterfall
# ------------------------
#
# Each row is one ping's matched-filter output, normalised to its peak. The static targets'
# echoes stay put while the moving target's drifts out in range. Ranges are binned to 2 m for
# plotting.

matched_filter = MatchedFilter()
mf_power = np.array(
    [matched_filter.filter(ping.received_waveform[0], ping.transmit_pulse) ** 2 for ping in pings]
)
range_axis_m = np.arange(mf_power.shape[1]) * SOUND_SPEED_MPS / (2 * SAMPLING_RATE_HZ)
bin_width = int(round(2.0 / (range_axis_m[1] - range_axis_m[0])))
num_bins = mf_power.shape[1] // bin_width
binned = mf_power[:, : num_bins * bin_width].reshape(len(pings), num_bins, bin_width).max(axis=2)
binned_db = 10 * np.log10(binned / binned.max(axis=1, keepdims=True))
figure = go.Figure(
    go.Heatmap(
        x=range_axis_m[: num_bins * bin_width : bin_width],
        y=[f"{t:%H:%M:%S}" for t in PING_TIMES],
        z=binned_db,
        zmin=-60,
        zmax=0,
        colorscale="Viridis",
        colorbar={"title": "dB re peak"},
    )
)
for path in targets:
    figure.add_trace(
        go.Scatter(
            x=[slant_range_m(s) for s in path],
            y=[f"{t:%H:%M:%S}" for t in PING_TIMES],
            mode="markers",
            marker={"symbol": "x", "color": "white"},
            showlegend=False,
        )
    )
figure.update_layout(
    template="plotly_white",
    title="Matched-Filter Output (white crosses: true slant ranges)",
    xaxis_title="Range (m)",
    yaxis_title="Ping",
).show()

# %%
# Detection
# ---------
#
# An ordered-statistic CFAR detector runs along range on the matched-filter output power.
#
# - **Noise estimate**: the 600th smallest of 800 training cells. The surface-reflected arrivals
#   and range sidelobes of a nearby echo don't inflate it as they would a mean.
# - **Guard cells**: either side of the cell under test, they keep the echo's own mainlobe out of
#   the estimate. At sampling rate :math:`f_s` the mainlobe is about :math:`f_s / B = 25`
#   samples wide.
# - **Threshold**: calibrated for the requested Pfa from noise-only pings, made by the same
#   simulator with no targets.

cfar = OSCFARDetector(
    num_guard_cells=100,
    num_training_cells=400,
    rank=600,
    target_pfa=TARGET_PFA,
    circular=False,
    peak_distance=40,
)
NoiseCalibrator(cfar).calibrate_from_noise(
    matched_filter_scans_from_sensor_data(simulator([], seed=2).sensor_data_gen())
)
detector = ActiveSonarDetector(
    detector=cfar,
    sensor_data_gen=[(ping.timestamp, {ping}) for ping in pings],
    sampling_rate_hz=SAMPLING_RATE_HZ,
    sound_speed_ms=SOUND_SPEED_MPS,
    min_range_m=100.0,
)
detections_by_ping = [detections for _, detections in detector.detections_gen()]
for ping_time, detections in zip(PING_TIMES, detections_by_ping, strict=True):
    print(f"{ping_time:%H:%M:%S}: {len(detections)} detections")

# %%
# Timing check
# ------------
#
# Each static target's nearest detection on every ping is compared with its true slant range.
# On each leg the surface-reflected path arrives :math:`2 z_1 z_2 / R` behind the direct one,
# where :math:`z_1` and :math:`z_2` are the two depths and :math:`R` the range.
#
# - **Short range**: the gap is many resolution cells and the direct echo is clean, so the error
#   is a fraction of a cell.
# - **2-2.5 km**: the gap has closed to about one cell (2-2.5 m in range). The two arrivals
#   overlap in the matched filter, and their interference shifts the peak by up to a cell.
#
# So the check allows two cells, :math:`c / B`.

print(f"\nRange resolution c/2B = {RANGE_RESOLUTION_M:.3f} m")
print(f"{'Target':>8} {'True (m)':>10} {'Worst error (m)':>16}  Within c/B")
for index, path in enumerate(static_targets):
    true_m = slant_range_m(path.states[0])
    errors = [
        min((abs(float(d.state_vector[-1, 0]) - true_m) for d in detections), default=np.inf)
        for detections in detections_by_ping
    ]
    worst = max(errors)
    print(f"{index + 1:>8} {true_m:>10.1f} {worst:>16.3f}  {worst <= 2 * RANGE_RESOLUTION_M}")

# %%
# Detected against true range
# ---------------------------
#
# Each point is one static target's nearest detection on one ping, less its true slant range.
# The errors grow with range as the surface-reflected arrival closes on the direct one.

true_points, detected_points = [], []
for path in static_targets:
    true_m = slant_range_m(path.states[0])
    for detections in detections_by_ping:
        nearest = min(detections, key=lambda d: abs(float(d.state_vector[-1, 0]) - true_m))
        true_points.append(true_m)
        detected_points.append(float(nearest.state_vector[-1, 0]))
true_points, detected_points = np.array(true_points), np.array(detected_points)
figure = go.Figure(
    [
        go.Scatter(x=true_points, y=detected_points - true_points, mode="markers",
                   name="Detected - true"),
        go.Scatter(x=[0, 3000], y=[RANGE_RESOLUTION_M] * 2, mode="lines",
                   line={"dash": "dash", "color": "grey"}, name="±c/2B (one cell)"),
        go.Scatter(x=[0, 3000], y=[-RANGE_RESOLUTION_M] * 2, mode="lines",
                   line={"dash": "dash", "color": "grey"}, showlegend=False),
    ]
)
figure.update_layout(
    template="plotly_white", title="Range Error of the Static Targets",
    xaxis_title="True slant range (m)", yaxis_title="Detected - true (m)",
).show()

# %%
# Echo level and the sonar equation
# ---------------------------------
#
# A matched filter's peak is the echo's amplitude gain times the pulse energy
# :math:`\sum |p|^2`, where :math:`p` is the transmitted pulse. With source level :math:`SL`, an
# arrival's level is therefore :math:`SL + 20\log_{10}(\text{peak} / \sum |p|^2)`. For spherical
# spreading on both legs, the sonar equation predicts :math:`SL - 40\log_{10} R + TS`, where
# :math:`R` is the slant range and :math:`TS` the target strength.
#
# The arrivals compare with that line as follows.
#
# - **Direct-path echo**: read at its true delay, it follows the line to within 2.5 dB. The rest
#   of the difference comes from neighbouring arrivals that overlap it.
# - **Strongest arrival**: about 6 dB higher. It comes from the two mixed paths. One goes out
#   direct and comes back via the surface, and the other does the reverse. Both have the same
#   length, so they arrive together and add in phase, which doubles the amplitude of either.

pulse_energy = float(np.sum(np.abs(pings[0].transmit_pulse) ** 2))
window = int(round(10 * SAMPLING_RATE_HZ / BANDWIDTH_HZ))
static_true_m = np.array([slant_range_m(path.states[0]) for path in static_targets])


def arrival_level_db(start, stop):
    """Return the strongest arrival level between two samples of ping 1, in dB re 1 µPa."""
    return SOURCE_LEVEL_DB + 20 * np.log10(np.sqrt(mf_power[0, start:stop].max()) / pulse_energy)


direct_db, strongest_db = [], []
for true_m in static_true_m:
    centre = int(round(2 * true_m / SOUND_SPEED_MPS * SAMPLING_RATE_HZ))
    direct_db.append(arrival_level_db(centre - 3, centre + 4))
    strongest_db.append(arrival_level_db(centre - window, centre + window))
range_grid_m = np.linspace(300.0, 3000.0, 200)
predicted_db = SOURCE_LEVEL_DB - 40 * np.log10(range_grid_m) + TARGET_STRENGTH_DB
figure = go.Figure(
    [
        go.Scatter(x=range_grid_m, y=predicted_db, mode="lines", name="SL - 40 log R + TS"),
        go.Scatter(x=range_grid_m, y=predicted_db + 20 * np.log10(2), mode="lines",
                   line={"dash": "dash"}, name="+6 dB (two paths in phase)"),
        go.Scatter(x=static_true_m, y=direct_db, mode="markers", name="Direct path",
                   marker={"size": 10}),
        go.Scatter(x=static_true_m, y=strongest_db, mode="markers", name="Strongest arrival",
                   marker={"size": 10, "symbol": "diamond"}),
    ]
)
figure.update_layout(
    template="plotly_white", title="Echo Level Against Range (Ping 1)",
    xaxis_title="Slant range (m)", yaxis_title="Echo level (dB re 1 µPa)",
).show()

# %%
# Range tracking
# --------------
#
# A Kalman filter tracks the opening target's range.
#
# - **State**: range and range rate.
# - **Association**: each ping, the filter takes the detection nearest its prediction, gated at
#   four standard deviations.
# - **Measurement noise**: the range resolution is the measurement's standard deviation.

measurement_model = LinearGaussian(
    ndim_state=2, mapping=[0], noise_covar=np.array([[RANGE_RESOLUTION_M**2]])
)
predictor = KalmanPredictor(ConstantVelocity(1e-4))
updater = KalmanUpdater(measurement_model)
associator = NearestNeighbour(
    DistanceHypothesiser(predictor, updater, Mahalanobis(), missed_distance=4)
)

first_range_m = slant_range_m(moving_target.states[0])
first = min(detections_by_ping[0], key=lambda d: abs(float(d.state_vector[-1, 0]) - first_range_m))
track = Track(
    [
        GaussianState(
            StateVector([float(first.state_vector[-1, 0]), 0.0]),
            np.diag([RANGE_RESOLUTION_M**2, 5.0**2]),
            timestamp=PING_TIMES[0],
        )
    ]
)
for ping_time, detections in zip(PING_TIMES[1:], detections_by_ping[1:], strict=True):
    hypothesis = associator.associate({track}, detections, ping_time)[track]
    if hypothesis:
        track.append(updater.update(hypothesis))
    else:
        track.append(hypothesis.prediction)

final_truth_m = slant_range_m(moving_target.states[-1])
print(
    f"Final track range {float(track.state_vector[0, 0]):.1f} m (truth {final_truth_m:.1f} m), "
    f"range rate {float(track.state_vector[1, 0]):.2f} m/s"
)

# %%
# Range against time
# ------------------

figure = go.Figure()
detection_times = [t for t, ds in zip(PING_TIMES, detections_by_ping, strict=True) for _ in ds]
detection_ranges = [float(d.state_vector[-1, 0]) for ds in detections_by_ping for d in ds]
figure.add_trace(
    go.Scatter(x=detection_times, y=detection_ranges, mode="markers", name="Detections",
               marker={"size": 5, "color": "grey"})
)
for path in targets:
    figure.add_trace(
        go.Scatter(x=PING_TIMES, y=[slant_range_m(s) for s in path], mode="lines",
                   line={"dash": "dash", "color": "black"}, showlegend=path is moving_target,
                   name="Truth")
    )
figure.add_trace(
    go.Scatter(x=[s.timestamp for s in track], y=[float(s.state_vector[0, 0]) for s in track],
               mode="lines+markers", name="Kalman track", line={"color": "crimson"})
)
figure.update_layout(
    template="plotly_white", title="Range Against Time", xaxis_title="Time",
    yaxis_title="Slant range (m)",
).show()

# %%
# Swapping the propagation model
# ------------------------------
#
# The simulator works with any spectrum propagation model, so the same chain runs on rtrs (3-D
# ray tracing) by changing one line. rtrs needs a fine launch-angle grid for an accurate single
# hydrophone. At its 1 degree default, the direct-path echo comes out 2-8 dB low.
#
# .. code-block:: python
#
#     from bluepebble.models.propagation import rtrsAcousticPropagationModel
#
#     propagation_model = rtrsAcousticPropagationModel(
#         ssp=Constant(speed=SOUND_SPEED_MPS),
#         bathymetry=FlatBathymetry(depth=-WATER_DEPTH_M),
#         step_m=5.0,
#         elevation_resolution=0.05,
#         azimuth_resolution=0.05,
#     )
