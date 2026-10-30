"""
=====================================================================
Active Sonar Multi-Array Example (Bow + Flank + Towed, Fused Tracking)
=====================================================================

This example simulates three receive arrays on a single ownship and fuses their
detections into one set of tracks:

- a **bow array** (:class:`~bluepebble.sensors.BowArraySensor`), which transmits the
  ping and receives its own echoes (monostatic);
- **port and starboard flank panels** (:class:`~bluepebble.sensors.FlankArraySensor`),
  mounted on the hull between the bow and the tow point and baffled by the hull;
- a **towed line array** (:class:`~bluepebble.sensors.TowedArraySensor`), which has
  all-round coverage but a left/right ambiguity.

The flank and towed arrays receive the bow transmission, so their echoes are
bistatic: the matched-filter delay gives the path length transmitter → target →
receiver.

Processing steps:

1. Simulate each receiver with ``BellhopActiveSonarSimulatorArray``.
2. For each array and ping: blank the direct-blast window, beamform (delay-and-sum),
   matched filter each beam and extract (bearing, path length) detections. Confirm
   each detection and refine its bearing with an MVDR beamformer.
3. Convert path length to range from the receiver and resolve the towed-array
   left/right ambiguity.
4. Update a single tracker with the detections from every array.

Assumptions: the direct blast is not simulated (only its blind zone, by blanking);
ownship and flow noise are not modelled; propagation is 2-D Bellhop in an
isovelocity environment.
"""  # noqa: D205, D212, D400, D415

# %%
# Imports
# -------

from datetime import datetime, timedelta

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import find_peaks, get_window
from stonesoup.dataassociator.neighbour import GNNWith2DAssignment
from stonesoup.deleter.error import CovarianceBasedDeleter
from stonesoup.functions import mod_bearing
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.initiator.simple import MultiMeasurementInitiator
from stonesoup.measures import Mahalanobis
from stonesoup.models.measurement.nonlinear import CartesianToBearingRange
from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel,
    ConstantVelocity,
)
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.types.angle import Bearing
from stonesoup.types.array import StateVector
from stonesoup.types.detection import Detection
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.types.state import GaussianState, State
from stonesoup.updater.kalman import ExtendedKalmanUpdater

import bluepebble
from bluepebble.detector import (
    DetectorChain,
    OSCFARDetector,
    PeakDetector,
    ThresholdDetector,
    bistatic_range,
)
from bluepebble.models.environment import Constant, FlatBathymetry
from bluepebble.models.propagation import BellhopArrivalsModel
from bluepebble.platform import HostPlatform
from bluepebble.sensors import BowArraySensor, FlankArraySensor, TowedArraySensor
from bluepebble.signal.active import LFMSignal
from bluepebble.signal.random import ColouredNoiseSignal
from bluepebble.sigproc import (
    DelayAndSumBeamformer,
    MatchedFilter,
    MinimumVarianceDistortionlessResponseBeamformer,
    SteeringCalculator,
    parabolic_peak_offset,
)
from bluepebble.simulator import BellhopActiveSonarSimulatorArray

# %%
# Configuration
# -------------

# %%
seed = 7
bluepebble.set_seed(seed)

start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

sim_params = {
    "ping_interval_s": 15.0,
    "n_pings": 6,
    "amplitude_cutoff": 0.05,
    "receive_duration_s": 4.0,
}

ping_timestamps = [
    start_time + timedelta(seconds=i * sim_params["ping_interval_s"])
    for i in range(sim_params["n_pings"])
]

env_params = {
    "water_depth_m": 1000.0,   # deep enough that bottom bounces arrive well after the direct path
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
    "depth_m": 0.0,
    "speed_ms": 3.0,
    "heading_deg": 0.0,
}

bow_params = {
    "dome_radius_m": 4.0,
    "azimuth_extent_rad": np.radians(90.0),
    "elevation_extent_rad": np.radians(16.0),
    "element_spacing_m": 0.20,
    "element_size_m": 0.10,
    "array_depth_offset_m": 6.0,
    "baffle_attenuation_db": 20.0,
}

# The tow point is towed_array_bow_offset_m aft of the bow. The flank panels must fit
# between the bow and the tow point.
towed_params = {
    "num_sensors": 64,
    "sensor_spacing_m": 0.25,
    "cable_length_m": 150.0,
    "array_depth_m": -90.0,
    "towed_array_bow_offset_m": 60.0,
}

flank_params = {
    "hull_beam_m": 8.0,
    "flank_bow_offset_m": 10.0,
    "aperture_length_m": 24.0,
    "element_spacing_m": 0.25,
    "element_size_m": 0.10,
    "array_depth_offset_m": 6.0,
    "baffle_attenuation_db": 20.0,
}


def _target_start(range_m: float, bearing_deg: float, depth_m: float,
                  vx: float, vy: float) -> np.ndarray:
    az = np.radians(bearing_deg)
    return np.array([range_m * np.cos(az), vx, range_m * np.sin(az), vy, -depth_m, 0.0])


# All targets are forward of the beam, where the bow array transmits at full level.
# T1: ahead (bow). T2: port bow (port flank, towed). T3: starboard bow (bow,
# starboard flank, towed). T2 and T3 are not at mirror-image bearings, so they fall in
# different towed-array beams.
target_params = {
    "position_mapping": [0, 2, 4],
    "target_strength_db": 15.0,
    "starts": [
        _target_start(2000.0, 10.0, 140.0, vx=-2.0, vy=1.0),
        _target_start(1800.0, 75.0, 160.0, vx=3.0, vy=0.0),
        _target_start(2200.0, -55.0, 120.0, vx=2.0, vy=2.0),
    ],
}

ambient_noise_params = {
    "amplitude_upa": 10 ** (72.0 / 20.0),
    "spectral_exponent": -1,
}

# Steering fans are measured from each array's axis: the ownship heading for the hull
# arrays and the line direction for the towed array.
bf_params = {
    "beam_step_deg": 0.5,
    "steering_elevation_rad": np.radians(-3.0),
    "fans_deg": {
        "bow": (-60.0, 60.0),
        "flank_port": (20.0, 160.0),
        "flank_stbd": (-160.0, -20.0),
        "towed": (15.0, 165.0),  # one side only; the other side is its mirror image
    },
    # MVDR detection confirmation and bearing refinement. nfft must exceed the largest
    # aperture transit time (24 m flank panel: 16 ms), so 256 samples (32 ms) are used.
    "mvdr_nfft": 256,
    "mvdr_overlap": 192,
    "mvdr_fmin_hz": 1400.0,
    "mvdr_fmax_hz": 2600.0,
    "mvdr_guard_deg": 15.0,          # MVDR scan extent beyond each edge of the fan
    "mvdr_peak_prominence_db": 3.0,  # minimum prominence of a confirming MVDR peak
    "mvdr_agreement_deg": 3.0,       # maximum MVDR-DAS bearing difference
    "bearing_estimator": "mvdr",     # bearing passed to the tracker: "mvdr" or "das"
}

det_params = {
    "blank_guard_s": 0.05,                 # blanking margin after direct blast and pulse
    "replica_window": "hann",
    "surface_floor_db": -60.0,
    "bearing_lobe_dyn_range_db": 10.0,
    "bearing_lobe_min_sep_beams": 12,
    "bearing_lobe_min_prominence_db": 2.5,
    "bearing_cut_agreement_deg": 4.0,
    "fan_edge_margin_beams": 3,            # reject lobes at the edge of the fan
    "range_peak_min_db": -16.0,
    "range_peak_min_sep_samples": 300,
    # The guard band covers the direct echo and its surface reflections (about 15 m),
    # keeping them out of the training cells.
    "cfar": {
        "num_guard_cells": 100,
        "num_training_cells": 200,
        "rank": 300,
        "threshold_factor": 8.0,
    },
}

fusion_params = {
    # "flank": discard the towed mirror image not confirmed by a flank detection.
    # "tracker": pass both images to the tracker.
    "resolve_towed_ambiguity": "tracker",
    "ambiguity_gate_m": 150.0,
    "endfire_skip_deg": 20.0,
    # Measurement standard deviations per array (bearing in degrees, range in metres).
    "bearing_sigma_deg": {"bow": 0.5, "flank_port": 0.75, "flank_stbd": 0.75, "towed": 1.0},
    "range_sigma_m": 15.0,
    "process_noise": 0.01,
    "missed_distance": 4.0,
    "deleter_covar_trace": 1e6,
    "init_velocity_sd_ms": 5.0,
    "init_min_points": 2,
}

ARRAYS = ["bow", "flank_port", "flank_stbd", "towed"]  # tracker update order

fs = signal_params["sampling_rate_hz"]
c = env_params["sound_speed_ms"]

# %%
# Scenario Build
# --------------
# Create the ownship, its three arrays and the target ground truth. The towed array is
# created first so that the flank array can take the tow point from it.

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
    freq_max_hz=signal_params["freq_max_hz"],
    **bow_params,
)
towed_array = TowedArraySensor(
    host=host,
    **towed_params,
)
flank_array = FlankArraySensor(
    host=host,
    hull_beam_m=flank_params["hull_beam_m"],
    flank_bow_offset_m=flank_params["flank_bow_offset_m"],
    aperture_length_m=flank_params["aperture_length_m"],
    element_spacing_m=flank_params["element_spacing_m"],
    element_size_m=flank_params["element_size_m"],
    array_depth_offset_m=flank_params["array_depth_offset_m"],
    baffle_attenuation_db=flank_params["baffle_attenuation_db"],
    freq_max_hz=signal_params["freq_max_hz"],
)

print(f"Bow dome:     {bow_array.array.num_sensors} elements")
print(f"Flank panels: {flank_array.array.num_sensors} elements (both sides), "
      f"tow point {flank_array.towed_array_bow_offset_m:.0f} m aft of the bow")
print(f"Towed array:  {towed_params['num_sensors']} elements, "
      f"{towed_params['cable_length_m']:.0f} m cable")

pm = target_params["position_mapping"]
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

# Move the ownship, and with it the towed array, to each ping time.
for ts in ping_timestamps[1:]:
    host.move(ts)


def truth_xy(k: int, p_idx: int) -> np.ndarray:
    """Horizontal ground-truth position of target ``k`` at ping ``p_idx``."""
    tv = target_truths[k].states[p_idx].state_vector
    return np.array([float(tv[pm[0]]), float(tv[pm[1]])])


# %%
# Per-Array Geometry at Each Ping
# -------------------------------
# The flank array is split into port and starboard panels with
# ``FlankArraySensor.panel_states``; each panel is processed separately. Each array
# state's ``axis`` defines the reference direction for its steering fan.


# %%
# Simulation
# ----------
# Simulate the received data for each array. The bow array is monostatic; the flank
# and towed arrays use the bow array as transmitter. Hull baffling is applied to the
# bow and flank arrays; the towed array is unbaffled.

# %%
ssp = Constant(speed=env_params["sound_speed_ms"])
bathymetry = FlatBathymetry(depth=-env_params["water_depth_m"])
prop_model = BellhopArrivalsModel(ssp=ssp, bathymetry=bathymetry)

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

receivers = {"bow": bow_array, "flank": flank_array, "towed": towed_array}
received = {}  # received[receiver] = list of (timestamp, ActiveSonarSensorData)
for rx_name, rx in receivers.items():
    simulator = BellhopActiveSonarSimulatorArray(
        platform=rx,
        transmitter=None if rx is bow_array else bow_array,
        propagation_model=prop_model,
        signal=pulse,
        ground_truth_paths=target_truths,
        target_strength_db=target_params["target_strength_db"],
        ping_timestamps=ping_timestamps,
        target_position_mapping=pm,
        receive_duration_s=sim_params["receive_duration_s"],
        noise_model=ambient_noise_model,
        amplitude_cutoff=sim_params["amplitude_cutoff"],
    )
    print(f"Simulating {rx_name} ({sim_params['n_pings']} pings)...")
    received[rx_name] = [(ts, next(iter(data))) for ts, data in simulator.sensor_data_gen()]

# %%
# Per-Array Processing Helpers
# ----------------------------
# Processing applied to each array and ping:
#
# - **Blanking**: samples covering the direct blast and the pulse length are zeroed.
# - **Beamforming and detection**: delay-and-sum beamforming, matched filtering of
#   each beam, then OS-CFAR, threshold and peak detection along range. Steering delays
#   are shifted to be non-positive; the resulting common lag is removed when
#   converting to delay.
# - **MVDR confirmation**: a detection is kept only if the MVDR spectrum, averaged
#   over the echo, has a prominent peak within ``mvdr_agreement_deg`` of the DAS
#   bearing. This rejects sidelobes of strong targets outside the fan. The MVDR peak
#   gives the refined bearing.

# %%
matched_filter = MatchedFilter(window=det_params["replica_window"])
steering_elevations = np.array([bf_params["steering_elevation_rad"]])
mvdr = MinimumVarianceDistortionlessResponseBeamformer(
    sampling_rate_hz=fs,
    nfft=bf_params["mvdr_nfft"],
    overlap=bf_params["mvdr_overlap"],
    fmin=bf_params["mvdr_fmin_hz"],
    fmax=bf_params["mvdr_fmax_hz"],
)
mvdr_hop = bf_params["mvdr_nfft"] - bf_params["mvdr_overlap"]


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


def detect_on_array(
    waveform: np.ndarray,
    snapshot: object,
    fan_rad: np.ndarray,
    mvdr_fan_rad: np.ndarray,
    replica: np.ndarray,
    blank_s: float,
) -> tuple[list[dict], list[dict], np.ndarray, float]:
    """Run DAS + matched filter + detection, then the MVDR check, on one array/ping.

    Returns ``(detections, rejected, surface_db, lag_s)``. Each detection is
    ``dict(path_m, sample, bearing_das_rad, bearing_mvdr_rad, bearing_rad, amp_db,
    mvdr_spec_db)``; ``bearing_rad`` is the one selected by ``bearing_estimator``.
    ``rejected`` holds DAS detections MVDR did not confirm (``bearing_mvdr_rad`` None).
    """
    def steering_for(azimuths_rad: np.ndarray) -> np.ndarray:
        return SteeringCalculator(
            ssp=ssp, steering_azimuths_rad=azimuths_rad,
            steering_elevations_rad=steering_elevations,
        ).calculate(snapshot)

    steering = steering_for(fan_rad)
    lag_s = float(steering.max())
    das = DelayAndSumBeamformer(
        sampling_rate_hz=fs, domain="time",
        shading=get_window("hann", snapshot.array.num_sensors),
    )

    data = waveform.copy()
    n_blank = int(np.ceil(blank_s * fs))
    data[:, :n_blank] = 0.0

    beams = das.beamform(data, steering - lag_s)
    surface = matched_filter.filter(beams, replica)
    surface_db = 20.0 * np.log10(np.maximum(surface / surface.max(), 1e-6))
    surface_db = np.maximum(surface_db, det_params["surface_floor_db"])

    n_beams = len(fan_rad)
    beam_step = float(fan_rad[1] - fan_rad[0])
    min_sample = n_blank + int(np.ceil(lag_s * fs))
    edge = det_params["fan_edge_margin_beams"]
    sep = det_params["bearing_lobe_min_sep_beams"]

    bearing_profile = surface_db[:, min_sample:].max(axis=1)
    lobe_idx, _ = find_peaks(
        bearing_profile,
        height=bearing_profile.max() - det_params["bearing_lobe_dyn_range_db"],
        distance=sep,
        prominence=det_params["bearing_lobe_min_prominence_db"],
    )

    found = []
    for k in lobe_idx:
        if k < edge or k > n_beams - 1 - edge:
            continue  # pinned to the fan edge: a target outside the fan leaking in
        samples = range_chain.detect(surface_db[k])[:, 0].astype(int)
        samples = samples[samples >= min_sample]
        if samples.size == 0:
            continue
        sample = int(samples.min())  # earliest return in the beam == direct path

        k_lo, k_hi = max(k - 2 * sep, 1), min(k + 2 * sep, n_beams - 2)
        cut = surface_db[:, sample]
        k_ref = k_lo + int(np.argmax(cut[k_lo:k_hi + 1]))
        bearing = float(fan_rad[k_ref] + parabolic_peak_offset(cut, k_ref) * beam_step)
        if abs(bearing - fan_rad[k]) > np.radians(det_params["bearing_cut_agreement_deg"]):
            continue  # sidelobe ghost of a stronger target

        found.append({
            "path_m": (sample / fs - lag_s) * c,
            "sample": sample,
            "bearing_das_rad": bearing,
            "amp_db": float(surface_db[k_ref, sample]),
        })

    if not found:
        return [], [], surface_db, lag_s

    # MVDR over the fan and guard band, using the same blanked data.
    power = mvdr.beamform(data, steering_for(mvdr_fan_rad))  # (n_dirs, n_frames)
    n_frames = power.shape[1]
    n_pulse = len(replica)
    mvdr_step = float(mvdr_fan_rad[1] - mvdr_fan_rad[0])
    agree = np.radians(bf_params["mvdr_agreement_deg"])
    confirmed, rejected = [], []
    for det in found:
        # Samples occupied by the echo (the matched-filter peak marks its start).
        s0 = det["sample"] - int(round(lag_s * fs))
        f_lo = min(max(s0 // mvdr_hop, 0), n_frames - 1)
        f_hi = min(max((s0 + n_pulse - bf_params["mvdr_nfft"]) // mvdr_hop + 1, f_lo + 1),
                   n_frames)
        spec = power[:, f_lo:f_hi].mean(axis=1)
        spec_db = 10.0 * np.log10(np.maximum(spec / spec.max(), 1e-6))
        det["mvdr_spec_db"] = spec_db

        peaks, _ = find_peaks(spec_db, prominence=bf_params["mvdr_peak_prominence_db"])
        near = [j for j in peaks if abs(mvdr_fan_rad[j] - det["bearing_das_rad"]) <= agree]
        if not near:
            det["bearing_mvdr_rad"] = None
            det["bearing_rad"] = det["bearing_das_rad"]
            rejected.append(det)
            continue
        j = max(near, key=lambda i: spec_db[i])
        det["bearing_mvdr_rad"] = float(
            mvdr_fan_rad[j] + parabolic_peak_offset(spec_db, j) * mvdr_step
        )
        det["bearing_rad"] = (det["bearing_mvdr_rad"] if bf_params["bearing_estimator"] == "mvdr"
                              else det["bearing_das_rad"])
        confirmed.append(det)
    return confirmed, rejected, surface_db, lag_s


# %%
# Per-Array Detection
# -------------------
# Run the detection chain on every array for every ping and convert each detection to
# a range and position.

# %%
fans_rad = {
    name: np.radians(np.arange(lo, hi + 1e-9, bf_params["beam_step_deg"]))
    for name, (lo, hi) in bf_params["fans_deg"].items()
}
_guard = bf_params["mvdr_guard_deg"]
mvdr_fans_rad = {
    name: np.radians(np.arange(lo - _guard, hi + _guard + 1e-9, bf_params["beam_step_deg"]))
    for name, (lo, hi) in bf_params["fans_deg"].items()
}

per_ping = []        # per_ping[p] = {array: list of detection dicts (with rx/position info)}
plot_surfaces = {}   # first-ping matched-filter surfaces for the figure
mvdr_rejected = []   # (ping, detection) DAS detections MVDR did not confirm

for p_idx, ts in enumerate(ping_timestamps):
    tx_xy = np.asarray(bow_array.get_platform_state_at(ts).array.ref_state_vector)[:2, 0]

    flank_panels = flank_array.panel_states(ts)
    snapshots = {
        "bow": (None, bow_array.get_platform_state_at(ts)),
        "flank_port": flank_panels["port"],
        "flank_stbd": flank_panels["starboard"],
        "towed": (None, towed_array.get_platform_state_at(ts)),
    }
    waveforms = {
        "bow": received["bow"][p_idx][1],
        "flank_port": received["flank"][p_idx][1],
        "flank_stbd": received["flank"][p_idx][1],
        "towed": received["towed"][p_idx][1],
    }

    ping_dets = {}
    for name in ARRAYS:
        idx, snapshot = snapshots[name]
        sensor_data = waveforms[name]
        waveform = sensor_data.received_waveform
        if idx is not None:
            waveform = waveform[idx]

        rx_xy = np.asarray(snapshot.array.ref_state_vector)[:2, 0]
        axis = snapshot.array.axis_azimuth_rad
        baseline_m = float(np.linalg.norm(tx_xy - rx_xy))
        blank_s = baseline_m / c + signal_params["duration_s"] + det_params["blank_guard_s"]

        found, rejected, surface_db, lag_s = detect_on_array(
            waveform, snapshot, axis + fans_rad[name], axis + mvdr_fans_rad[name],
            sensor_data.transmit_pulse, blank_s,
        )
        for det in found + rejected:
            det["ping"] = p_idx
            det["array"] = name
            det["rx_xy"] = rx_xy
            det["axis_rad"] = axis
            det["range_m"] = bistatic_range(det["path_m"], det["bearing_rad"], rx_xy, tx_xy)
            det["xy"] = rx_xy + det["range_m"] * np.array(
                [np.cos(det["bearing_rad"]), np.sin(det["bearing_rad"])]
            )
        ping_dets[name] = found
        mvdr_rejected.extend((p_idx, det) for det in rejected)
        if p_idx == 0:
            plot_surfaces[name] = {
                "fan_rad": axis + fans_rad[name], "mvdr_fan_rad": axis + mvdr_fans_rad[name],
                "axis_rad": axis, "surface_db": surface_db,
                "lag_s": lag_s, "blank_s": blank_s, "detections": list(found),
                "rejected": list(rejected),
                "rx": np.asarray(snapshot.array.ref_state_vector).flatten(),
            }

    per_ping.append(ping_dets)
    summary = ", ".join(f"{n}: {len(d)}" for n, d in ping_dets.items())
    n_rej = sum(1 for p, _ in mvdr_rejected if p == p_idx)
    print(f"Ping {p_idx}: detections {summary}  (MVDR rejected {n_rej})")

# %%
# Towed-Array Left/Right Ambiguity
# --------------------------------
# A straight towed array measures only the angle from its axis, so each towed
# detection has a mirror image on the other side of the axis. Each image is given its
# own bistatic range.
#
# ``"flank"``: keep the image confirmed by a flank detection within
# ``ambiguity_gate_m``; keep both if neither or both are confirmed. ``"tracker"``: keep
# both images. On a straight ownship course this can produce a mirror track.

# %%
dropped_mirrors = []  # (ping, xy) of candidates removed by the flank check

for p_idx, ping_dets in enumerate(per_ping):
    tx_xy = np.asarray(bow_array.get_platform_state_at(ping_timestamps[p_idx])
                       .array.ref_state_vector)[:2, 0]
    flank_xy = [d["xy"] for n in ("flank_port", "flank_stbd") for d in ping_dets[n]]

    resolved = []
    for det in ping_dets["towed"]:
        off_axis = mod_bearing(det["bearing_rad"] - det["axis_rad"])
        endfire_skip_rad = np.radians(fusion_params["endfire_skip_deg"])
        if min(abs(off_axis), np.pi - abs(off_axis)) < endfire_skip_rad:
            resolved.append(det)
            continue

        mirror = dict(det)
        for key in ("bearing_rad", "bearing_das_rad", "bearing_mvdr_rad"):
            mirror[key] = mod_bearing(2.0 * det["axis_rad"] - det[key])
        mirror["range_m"] = bistatic_range(
            det["path_m"], mirror["bearing_rad"], det["rx_xy"], tx_xy
        )
        mirror["xy"] = det["rx_xy"] + mirror["range_m"] * np.array(
            [np.cos(mirror["bearing_rad"]), np.sin(mirror["bearing_rad"])]
        )
        candidates = [det, mirror]

        if fusion_params["resolve_towed_ambiguity"] == "flank":
            confirmed = [
                any(np.linalg.norm(cand["xy"] - f) < fusion_params["ambiguity_gate_m"]
                    for f in flank_xy)
                for cand in candidates
            ]
            if sum(confirmed) == 1:
                keep = candidates[confirmed.index(True)]
                drop = candidates[confirmed.index(False)]
                resolved.append(keep)
                dropped_mirrors.append((p_idx, drop["xy"]))
                continue
        resolved.extend(candidates)

    ping_dets["towed"] = resolved

print(f"\nTowed ambiguity ({fusion_params['resolve_towed_ambiguity']}): "
      f"{len(dropped_mirrors)} mirror candidates dropped by the flank check")

# %%
# Measurement-Level Fusion Tracker
# --------------------------------
# A single tracker with a 2-D constant-velocity state ``[x, vx, y, vy]``. Each
# detection has its own ``CartesianToBearingRange`` model, offset to the receiving
# array's position and using that array's bearing accuracy. At each ping the tracker
# is updated with each array's detections in turn.

# %%
transition_2d = CombinedLinearGaussianTransitionModel([
    ConstantVelocity(fusion_params["process_noise"]),
    ConstantVelocity(fusion_params["process_noise"]),
])

predictor = KalmanPredictor(transition_2d)
updater = ExtendedKalmanUpdater(measurement_model=None)
hypothesiser = DistanceHypothesiser(
    predictor, updater, measure=Mahalanobis(),
    missed_distance=fusion_params["missed_distance"],
)
data_associator = GNNWith2DAssignment(hypothesiser)
deleter = CovarianceBasedDeleter(covar_trace_thresh=fusion_params["deleter_covar_trace"])

vel_var = fusion_params["init_velocity_sd_ms"] ** 2
initiator = MultiMeasurementInitiator(
    prior_state=GaussianState(
        StateVector([0.0, 0.0, 0.0, 0.0]), np.diag([0.0, vel_var, 0.0, vel_var])
    ),
    measurement_model=None,
    deleter=deleter,
    data_associator=GNNWith2DAssignment(DistanceHypothesiser(
        predictor, updater, measure=Mahalanobis(),
        missed_distance=fusion_params["missed_distance"],
    )),
    updater=updater,
    min_points=fusion_params["init_min_points"],
)


def to_detection(det: dict, timestamp: datetime) -> Detection:
    """Wrap a per-array detection as a Stone Soup bearing-range ``Detection``."""
    sigma_b = np.radians(fusion_params["bearing_sigma_deg"][det["array"]])
    model = CartesianToBearingRange(
        ndim_state=4,
        mapping=(0, 2),
        noise_covar=np.diag([sigma_b ** 2, fusion_params["range_sigma_m"] ** 2]),
        translation_offset=StateVector(det["rx_xy"]),
    )
    return Detection(
        StateVector([Bearing(det["bearing_rad"]), det["range_m"]]),
        timestamp=timestamp,
        measurement_model=model,
        metadata={"array": det["array"]},
    )


tracks = set()
all_tracks = set()
for ts, ping_dets in zip(ping_timestamps, per_ping, strict=True):
    for name in ARRAYS:
        detections = {to_detection(d, ts) for d in ping_dets[name]}
        hypotheses = data_associator.associate(tracks, detections, ts)
        associated = set()
        for track, hypothesis in hypotheses.items():
            if hypothesis:
                track.append(updater.update(hypothesis))
                associated.add(hypothesis.measurement)
            else:
                track.append(hypothesis.prediction)
        tracks -= deleter.delete_tracks(tracks)
        tracks |= initiator.initiate(detections - associated, ts)
        all_tracks |= tracks

print(f"Confirmed tracks at end: {len(tracks)}")

# %%
# Results
# -------
# Per-array errors compare each detection with the nearest target at that ping. Track
# errors compare each track's final position with the nearest target at the last ping.

# %%
n_targets = len(target_truths)
print(f"\nMVDR rejected {len(mvdr_rejected)} DAS detections: " + ", ".join(
    f"{d['array']}@{np.degrees(d['bearing_das_rad']):.1f}deg (ping {p})"
    for p, d in mvdr_rejected
))


def _rms(values: list[float]) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


# In "tracker" mode the towed results include mirror images, which increases the
# towed-array errors.
print(f"\n{'Array':<12}{'dets':>6}{'DAS brg err':>13}{'MVDR brg err':>14}{'range err (m)':>15}")
for name in ARRAYS:
    das_err, mvdr_err, r_err = [], [], []
    for p_idx, ping_dets in enumerate(per_ping):
        for det in ping_dets[name]:
            truths = [truth_xy(k, p_idx) for k in range(n_targets)]
            nearest = min(truths, key=lambda t: np.linalg.norm(t - det["xy"]))
            to_t = nearest - det["rx_xy"]
            true_brg = np.arctan2(to_t[1], to_t[0])
            das_err.append(np.degrees(mod_bearing(det["bearing_das_rad"] - true_brg)))
            mvdr_err.append(np.degrees(mod_bearing(det["bearing_mvdr_rad"] - true_brg)))
            r_err.append(det["range_m"] - np.linalg.norm(to_t))
    if das_err:
        print(f"{name:<12}{len(das_err):>6}{_rms(das_err):>13.2f}{_rms(mvdr_err):>14.2f}"
              f"{_rms(r_err):>15.1f}")
    else:
        print(f"{name:<12}{0:>6}{'-':>13}{'-':>14}{'-':>15}")
print("(bearing errors are RMS in degrees; the tracker uses the "
      f"{bf_params['bearing_estimator'].upper()} bearing)")

print(f"\n{'Fused track':<12}{'nearest':>8}{'position err (m)':>18}{'arrays used':>30}")
final_truths = [truth_xy(k, -1) for k in range(n_targets)]
for i, track in enumerate(sorted(tracks, key=lambda t: float(t.state_vector[2]))):
    xy = np.array([float(track.state_vector[0]), float(track.state_vector[2])])
    k = int(np.argmin([np.linalg.norm(t - xy) for t in final_truths]))
    used = sorted({s.hypothesis.measurement.metadata["array"]
                   for s in track.states if hasattr(s, "hypothesis")})
    print(f"track {i:<6}{'T' + str(k + 1):>8}{np.linalg.norm(final_truths[k] - xy):>18.1f}"
          f"{', '.join(used):>30}")

# %%
# Figures
# -------

# %%
_font = dict(family="Times New Roman", size=15, color="black")
_colors = {"bow": "#1f77b4", "flank_port": "#2ca02c", "flank_stbd": "#9467bd",
           "towed": "#ff7f0e"}
_grid = dict(showgrid=True, gridcolor="rgba(200,200,200,0.5)")

# Array element positions relative to the bow at the first ping.
fig_layout = go.Figure()
host_xy0 = np.asarray(host.states[0].state_vector)[[0, 2], 0]
layout_arrays = {
    "bow": bow_array.get_platform_state_at(ping_timestamps[0]).array,
    "flank": flank_array.get_platform_state_at(ping_timestamps[0]).array,
    "towed": towed_array.get_platform_state_at(ping_timestamps[0]).array,
}
for name, arr in layout_arrays.items():
    pos = np.asarray(arr.state_vector)
    fig_layout.add_trace(go.Scatter(
        x=pos[0] - host_xy0[0], y=pos[1] - host_xy0[1], mode="markers",
        marker=dict(size=4, color=_colors.get(name, "#2ca02c")), name=name,
    ))
tow_x = -towed_params["towed_array_bow_offset_m"]
fig_layout.add_trace(go.Scatter(
    x=[0.0, tow_x], y=[0.0, 0.0], mode="markers+text", text=["bow ref", "tow point"],
    textposition="top center", marker=dict(symbol="x", size=10, color="black"),
    name="hull reference points",
))
fig_layout.update_layout(
    width=900, height=320, font=_font, plot_bgcolor="white", paper_bgcolor="white",
    title="Array layout relative to the bow (first ping)",
    yaxis=dict(scaleanchor="x", scaleratio=1),
)
fig_layout.update_xaxes(title="Along-track (m)", **_grid)
fig_layout.update_yaxes(title="Cross-track (m)", **_grid)

# True path length and in-fan bearing of each target for each array at ping 0. For the
# towed array, a starboard target appears at its mirror bearing.
tx0 = np.asarray(bow_array.get_platform_state_at(ping_timestamps[0]).array.ref_state_vector)
tx0 = tx0.flatten()
_target_colors = ["#d62728", "#17becf", "#bcbd22"]


def truth_in_fan(name: str, k: int) -> tuple[float, float, bool] | None:
    """Return ``(path_m, fan_bearing_rad, mirrored)`` for target ``k`` at ping 0."""
    info = plot_surfaces[name]
    tv = target_truths[k].states[0].state_vector
    target = np.array([float(tv[pm[0]]), float(tv[pm[1]]), float(tv[pm[2]])])
    path_m = float(np.linalg.norm(target - tx0) + np.linalg.norm(target - info["rx"]))
    to_t = target - info["rx"]
    rel = mod_bearing(float(np.arctan2(to_t[1], to_t[0])) - info["axis_rad"])
    lo, hi = np.radians(bf_params["fans_deg"][name])
    for candidate, mirrored in ((rel, False), (-rel, True)):
        if lo <= candidate <= hi and (name == "towed" or not mirrored):
            return path_m, info["axis_rad"] + candidate, mirrored
    return None


truth_paths = [
    t[0] for name in ARRAYS for k in range(n_targets) if (t := truth_in_fan(name, k))
]
_path_window_km = (min(truth_paths) / 1000 - 0.3, max(truth_paths) / 1000 + 1.2)
_scatter_floor_db = -45.0


def true_path_and_bearings(name: str, k: int) -> tuple[float, float, float | None]:
    """Target ``k`` at ping 0 as seen by array ``name``: ``(path_m, bearing, mirror)``.

    ``mirror`` is the towed array's left/right image of the bearing (None otherwise).
    """
    info = plot_surfaces[name]
    tv = target_truths[k].states[0].state_vector
    target = np.array([float(tv[pm[0]]), float(tv[pm[1]]), float(tv[pm[2]])])
    path_m = float(np.linalg.norm(target - tx0) + np.linalg.norm(target - info["rx"]))
    bearing = float(np.arctan2(target[1] - info["rx"][1], target[0] - info["rx"][0]))
    mirror = mod_bearing(2.0 * info["axis_rad"] - bearing) if name == "towed" else None
    return path_m, bearing, mirror


# DAS and MVDR bearing responses for each ping-0 detection, including those rejected
# by MVDR. The shaded region is the MVDR guard band.
bear_panels = [(name, det, True) for name in ARRAYS for det in plot_surfaces[name]["detections"]]
bear_panels += [(name, det, False) for name in ARRAYS for det in plot_surfaces[name]["rejected"]]
_n_cols = 3
_n_rows = int(np.ceil(len(bear_panels) / _n_cols))
_titles = []
for name, det, kept in bear_panels:
    k = min(range(n_targets),
            key=lambda j, n=name, d=det: abs(true_path_and_bearings(n, j)[0] - d["path_m"]))
    det["_plot_target"] = k
    _titles.append(f"{name}: T{k + 1}" + ("" if kept else "  -- REJECTED by MVDR"))
fig_bear = make_subplots(rows=_n_rows, cols=_n_cols, subplot_titles=_titles,
                         horizontal_spacing=0.06, vertical_spacing=0.16)
for i, (name, det, _kept) in enumerate(bear_panels):
    row, col = i // _n_cols + 1, i % _n_cols + 1
    info = plot_surfaces[name]
    das_cut = info["surface_db"][:, det["sample"]]
    das_cut = das_cut - das_cut.max()
    fan_deg = np.degrees(info["fan_rad"])
    mvdr_deg = np.degrees(info["mvdr_fan_rad"])

    fig_bear.add_trace(go.Scatter(
        x=fan_deg, y=das_cut, mode="lines", line=dict(color="#1f77b4", width=2),
        name="DAS (matched filter)", legendgroup="das", showlegend=(i == 0),
    ), row=row, col=col)
    fig_bear.add_trace(go.Scatter(
        x=mvdr_deg, y=det["mvdr_spec_db"], mode="lines", line=dict(color="#d62728", width=2),
        name="MVDR", legendgroup="mvdr", showlegend=(i == 0),
    ), row=row, col=col)
    # Add the shading after the traces so that it is placed on the correct subplot.
    for lo, hi in ((mvdr_deg[0], fan_deg[0]), (fan_deg[-1], mvdr_deg[-1])):
        fig_bear.add_vrect(x0=lo, x1=hi, fillcolor="lightgrey", opacity=0.5, line_width=0,
                           layer="below", row=row, col=col)

    _, truth_brg, mirror_brg = true_path_and_bearings(name, det["_plot_target"])
    marks = [np.degrees(det["bearing_das_rad"]), np.degrees(truth_brg)]
    fig_bear.add_vline(x=np.degrees(truth_brg), line=dict(color="black", dash="dash", width=1.5),
                       row=row, col=col)
    if mirror_brg is not None:
        fig_bear.add_vline(x=np.degrees(mirror_brg),
                           line=dict(color="grey", dash="dashdot", width=1.5), row=row, col=col)
        marks.append(np.degrees(mirror_brg))
    # Limit the axis to the detection, the truth and the MVDR scan extent.
    in_view = [m for m in marks if mvdr_deg[0] <= m <= mvdr_deg[-1]]
    fig_bear.update_xaxes(range=[max(min(in_view) - 15, mvdr_deg[0]),
                                 min(max(in_view) + 15, mvdr_deg[-1])],
                          title="World bearing (deg)", row=row, col=col, **_grid)
    fig_bear.update_yaxes(range=[-40, 2], title="Norm. response (dB)" if col == 1 else None,
                          row=row, col=col, **_grid)
for label, style in (("truth", dict(color="black", dash="dash")),
                     ("towed mirror of truth", dict(color="grey", dash="dashdot"))):
    fig_bear.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=style, name=label),
                       row=1, col=1)
fig_bear.add_trace(go.Scatter(
    x=[None], y=[None], mode="markers", name="outside accepted fan (MVDR guard band)",
    marker=dict(symbol="square", size=12, color="lightgrey"),
), row=1, col=1)
fig_bear.update_layout(
    width=1500, height=380 * _n_rows, font=_font, paper_bgcolor="white", plot_bgcolor="white",
    title="Beam response at each ping-0 detection: DAS vs MVDR",
    legend=dict(orientation="h", y=-0.12, x=0.5, xanchor="center"),
)

# Matched-filter output along each detection's beam at ping 0. Dashed lines mark the
# true path lengths.
fig_scatter = make_subplots(rows=1, cols=len(ARRAYS), subplot_titles=ARRAYS,
                            horizontal_spacing=0.05, shared_yaxes=True)
for col, name in enumerate(ARRAYS, start=1):
    info = plot_surfaces[name]
    path_km = (np.arange(info["surface_db"].shape[1]) / fs - info["lag_s"]) * c / 1000.0
    in_window = (path_km >= _path_window_km[0]) & (path_km <= _path_window_km[1])
    truths = {k: t for k in range(n_targets) if (t := truth_in_fan(name, k))}
    for det in info["detections"]:
        # Colour by the nearest target in path length and bearing.
        k = min(truths, key=lambda j: abs(truths[j][0] - det["path_m"]) / 10.0
                + abs(np.degrees(truths[j][1] - det["bearing_rad"])))
        color = _target_colors[k % len(_target_colors)]
        beam = int(np.argmin(np.abs(info["fan_rad"] - det["bearing_rad"])))
        row_db = info["surface_db"][beam]
        mask = in_window & (row_db >= _scatter_floor_db)
        fig_scatter.add_trace(go.Scatter(
            x=path_km[mask], y=row_db[mask], mode="markers",
            marker=dict(size=3, color=color),
            name=f"T{k+1}", legendgroup=f"T{k+1}", showlegend=False,
        ), row=1, col=col)
        fig_scatter.add_vline(x=truths[k][0] / 1000, line=dict(color=color, dash="dash", width=1),
                              row=1, col=col)
    fig_scatter.update_xaxes(title="Path length (km)", range=_path_window_km, row=1, col=col,
                             **_grid)
for k in range(n_targets):  # legend entries with readable marker size
    fig_scatter.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(size=10, color=_target_colors[k % len(_target_colors)]),
        name=f"T{k+1}", legendgroup=f"T{k+1}",
    ), row=1, col=1)
fig_scatter.update_yaxes(title="MF level (dB re ping max)", row=1, col=1, **_grid)
fig_scatter.update_layout(
    width=1500, height=420, font=_font, paper_bgcolor="white", plot_bgcolor="white",
    title="Matched-filter level along each detection's beam, ping 0 (dashed: true path)",
    legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center"),
)

# Plan view of detections, tracks and ground truth, with a zoomed panel for each target.
fig_world = make_subplots(
    rows=2, cols=n_targets, row_heights=[0.55, 0.45], vertical_spacing=0.1,
    specs=[[{"colspan": n_targets}] + [None] * (n_targets - 1), [{}] * n_targets],
    subplot_titles=["Overview"] + [f"T{k+1} (zoom)" for k in range(n_targets)],
)
own = np.array([np.asarray(s.state_vector)[[0, 2], 0] for s in host.states])
last = len(ping_timestamps) - 1
_legend_done = set()


def _legend_once(key: str) -> bool:
    if key in _legend_done:
        return False
    _legend_done.add(key)
    return True


fig_world.add_trace(go.Scatter(
    x=own[:, 0] / 1000, y=own[:, 1] / 1000, mode="lines+markers",
    line=dict(color="black", width=2), marker=dict(symbol="diamond", size=6), name="Ownship",
), row=1, col=1)
# Bearing lines from each array to its detections on the last ping.
for name in ARRAYS:
    for det in per_ping[last][name]:
        fig_world.add_trace(go.Scatter(
            x=[det["rx_xy"][0] / 1000, det["xy"][0] / 1000],
            y=[det["rx_xy"][1] / 1000, det["xy"][1] / 1000], mode="lines",
            line=dict(color=_colors[name], width=1, dash="dot"), opacity=0.7,
            name=name, legendgroup=name, showlegend=False,
        ), row=1, col=1)
if dropped_mirrors:
    pts = np.array([xy for _, xy in dropped_mirrors])
    fig_world.add_trace(go.Scatter(
        x=pts[:, 0] / 1000, y=pts[:, 1] / 1000, mode="markers",
        marker=dict(symbol="x", size=8, color=_colors["towed"]),
        name="towed mirror dropped by flank",
    ), row=1, col=1)

# Plot all tracks. A track that ends far from every target is typically a towed-array
# mirror track.
for track in all_tracks:
    xy = np.array([[float(s.state_vector[0]), float(s.state_vector[2])] for s in track.states])
    p_end = ping_timestamps.index(track.states[-1].timestamp)
    miss_m = min(np.linalg.norm(xy[-1] - truth_xy(k, p_end)) for k in range(n_targets))
    ghost = bool(miss_m > 500.0)
    fig_world.add_trace(go.Scatter(
        x=xy[:, 0] / 1000, y=xy[:, 1] / 1000, mode="lines+markers",
        line=dict(color="magenta" if ghost else "red", width=2, dash="dash" if ghost else None),
        marker=dict(size=4),
        name="ghost track" if ghost else "fused track",
        legendgroup="ghost" if ghost else "track",
        showlegend=ghost and _legend_once("ghost"),
    ), row=1, col=1)

for k in range(n_targets):
    truth = np.array([truth_xy(k, p) for p in range(len(ping_timestamps))])
    fig_world.add_trace(go.Scatter(
        x=truth[:, 0] / 1000, y=truth[:, 1] / 1000, mode="lines+text",
        text=[f"T{k+1}"] + [""] * (len(truth) - 1), textposition="top center",
        line=dict(color="grey", width=4), name="truth", legendgroup="truth",
        showlegend=False,
    ), row=1, col=1)

    col = k + 1
    fig_world.add_trace(go.Scatter(
        x=truth[:, 0], y=truth[:, 1], mode="lines+markers",
        line=dict(color="grey", width=3), marker=dict(size=8, color="grey"),
        name="truth (one marker per ping)", legendgroup="truth",
        showlegend=_legend_once("truth"),
    ), row=2, col=col)
    for name in ARRAYS:
        pts = np.array([d["xy"] for ping_dets in per_ping for d in ping_dets[name]
                        if np.linalg.norm(d["xy"] - truth.mean(axis=0)) < 500.0])
        if len(pts):
            fig_world.add_trace(go.Scatter(
                x=pts[:, 0], y=pts[:, 1], mode="markers",
                marker=dict(size=8, color=_colors[name], line=dict(width=1, color="white")),
                name=f"{name} detections", legendgroup=name, showlegend=_legend_once(name),
            ), row=2, col=col)
    for track in all_tracks:
        xy = np.array([[float(s.state_vector[0]), float(s.state_vector[2])]
                       for s in track.states])
        if np.linalg.norm(xy[-1] - truth[-1]) > 500.0:
            continue
        fig_world.add_trace(go.Scatter(
            x=xy[:, 0], y=xy[:, 1], mode="lines", line=dict(color="red", width=2),
            name="fused track", legendgroup="track", showlegend=_legend_once("track"),
        ), row=2, col=col)

    # Square window centred on the target, keeping a 1:1 aspect ratio.
    centre = (truth.min(axis=0) + truth.max(axis=0)) / 2
    half = max(np.ptp(truth, axis=0).max() / 2 + 40.0, 80.0)
    fig_world.update_xaxes(range=[centre[0] - half, centre[0] + half], title="X (m)",
                           row=2, col=col, **_grid)
    fig_world.update_yaxes(range=[centre[1] - half, centre[1] + half],
                           title="Y (m)" if col == 1 else None, row=2, col=col, **_grid)

fig_world.update_xaxes(title="X (km)", row=1, col=1, **_grid)
fig_world.update_yaxes(title="Y (km)", scaleanchor="x", scaleratio=1, row=1, col=1, **_grid)
fig_world.update_layout(
    width=1200, height=1150, font=_font, plot_bgcolor="white", paper_bgcolor="white",
    legend=dict(orientation="h", y=-0.08, x=0.5, xanchor="center"),
    title="Detections and fused tracks (top: dotted lines show what each array saw on the "
          "last ping)",
)

# %%
# Export
# ------

# %%
from pathlib import Path

out_dir = Path(__file__).resolve().parent / "figs_active"
out_dir.mkdir(parents=True, exist_ok=True)

for fname, fig in [
    ("as_multiarray_layout.pdf", fig_layout),
    ("as_multiarray_bearing_das_vs_mvdr.pdf", fig_bear),
    ("as_multiarray_mf_scatter.pdf", fig_scatter),
    ("as_multiarray_world.pdf", fig_world),
]:
    fig.write_image(str(out_dir / fname))
    print(f"Saved {out_dir / fname}")
