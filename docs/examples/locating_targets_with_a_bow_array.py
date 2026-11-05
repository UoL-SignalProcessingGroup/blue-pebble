r"""
=================================
Locating Targets with a Bow Array
=================================

This example pings from a ship's bow array and locates three targets in bearing and range.
Unlike a single hydrophone (:ref:`sphx_glr_auto_examples_detecting_echoes_with_a_sonobuoy.py`),
an array can beamform, so each echo is detected on the beam pointing at it.

- **Simulation**: :class:`~bluepebble.simulator.ActiveSonarSimulator` synthesises the echoes at
  every sensor of a :class:`~bluepebble.arrays.BowArray` through a Bellhop propagation model. It
  beamforms them with a frequency-domain delay-and-sum (DAS) beamformer, which keeps every
  sample. At sampling rate :math:`f_s`, sample :math:`k` of a beam is then still a round-trip
  delay of :math:`k / f_s`.
- **Detection**: :class:`~bluepebble.detector.ActiveSonarDetector` matched-filters each beam and
  runs CFAR (Constant False Alarm Rate) detection along range. Each detection has a bearing and
  a range.
- **Grouping**: a strong echo is detected on its own beam and on neighbouring beams. Detections
  at the same range and nearby bearings are grouped into one, and its bearing is refined between
  beams.

Finally, the DAS bearing response at each target's echo is compared with a minimum variance
distortionless response (MVDR) beamformer's. MVDR narrows the mainlobe and lowers the sidelobes
for the target dead ahead, but not for the two targets off the bow.

Assumptions
-----------
- Constant 1500 m/s sound speed over a flat 300 m seabed matched to the water, so it reflects
  nothing. Only the direct and surface-reflected paths remain.
- Each target is a point scatterer with a 15 dB target strength. There is no reverberation.
- Geometry is frozen during each ping, so there is no Doppler.
"""  # noqa: D205, D212, D400, D415
# sphinx_gallery_thumbnail_path = "_static/thumbnails/locating_targets_with_a_bow_array.png"

# %%
# Imports
# -------

from datetime import datetime, timedelta

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel,
    ConstantVelocity,
)
from stonesoup.types.array import StateVector
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState

from bluepebble import set_seed
from bluepebble.arrays import BowArray
from bluepebble.detector import (
    ActiveSonarDetector,
    CACFARDetector,
    NoiseCalibrator,
    matched_filter_power,
    matched_filter_scans_from_sensor_data,
)
from bluepebble.models import ConstantTargetStrength
from bluepebble.models.environment import Constant, FlatBathymetry
from bluepebble.models.propagation import BellhopAcousticPropagationModel
from bluepebble.platform import SonarPlatform
from bluepebble.signal import LFMSignal
from bluepebble.signal.random import WhiteNoiseSignal
from bluepebble.sigproc import (
    DelayAndSumBeamformer,
    MinimumVarianceDistortionlessResponseBeamformer,
    SteeringCalculator,
    parabolic_peak_offset,
)
from bluepebble.simulator import ActiveSonarSimulator

# %%
# Configuration
# -------------
#
# The pulse is a 0.5 s linear frequency-modulated (LFM) sweep over 1500-2500 Hz. With sound speed
# :math:`c` and bandwidth :math:`B`, the range resolution is :math:`c / 2B = 0.75` m.

set_seed(7)

START = datetime(2026, 1, 1)
NUM_PINGS = 3
PING_TIMES = [START + i * timedelta(seconds=20) for i in range(NUM_PINGS)]

SOUND_SPEED_MPS = 1500.0
WATER_DEPTH_M = 300.0
SAMPLING_RATE_HZ = 8000
FREQ_MIN_HZ, FREQ_MAX_HZ = 1500.0, 2500.0
RANGE_RESOLUTION_M = SOUND_SPEED_MPS / (2 * (FREQ_MAX_HZ - FREQ_MIN_HZ))
SOURCE_LEVEL_DB = 200.0
TARGET_STRENGTH_DB = 15.0
NOISE_LEVEL_DB = 72.0
LISTEN_DURATION_S = 4.0
TARGET_PFA = 1e-7

# %%
# Scenario
# --------
#
# - **Ship**: steams along +x at 3 m/s, with its hull reference point 8 m deep.
# - **Bow dome**: centred 3 m below the hull reference point, at 11 m. Its sensors cover 90
#   degrees in azimuth and 16 in elevation at 0.2 m spacing.
# - **Targets**: three, 1.5-2.5 km ahead and 120-160 m deep.

platform = SonarPlatform(
    states=[GroundTruthState(StateVector([0.0, 3.0, 0.0, 0.0, -8.0, 0.0]), timestamp=START)],
    position_mapping=[0, 2, 4],
    transition_models=[CombinedLinearGaussianTransitionModel([ConstantVelocity(0.0)] * 3)],
    transition_times=[timedelta(hours=1)],
)
bow = BowArray(
    dome_radius_m=4.0,
    azimuth_extent_rad=np.radians(90.0),
    elevation_extent_rad=np.radians(16.0),
    sensor_spacing_m=0.2,
    sensor_size_m=0.1,
    freq_max_hz=FREQ_MAX_HZ,
    mounting_offset=StateVector([0.0, 0.0, -3.0]),
)
platform.add_sensor(bow)
for ping_time in PING_TIMES[1:]:
    platform.move(ping_time)
print(f"Bow array: {bow.num_sensors} sensors")

# Each target is given by its range, bearing (degrees anticlockwise from the bow) and depth at
# the first ping, and its velocity.
TARGETS = {
    "T1": (2000.0, -32.0, 140.0, (0.0, 0.5)),
    "T2": (1500.0, 0.0, 160.0, (-3.0, 0.0)),
    "T3": (2500.0, 28.0, 120.0, (0.0, -3.0)),
}
targets = {
    name: GroundTruthPath(
        [
            GroundTruthState(
                StateVector(
                    [
                        r * np.cos(np.radians(b)) + vx * t,
                        vx,
                        r * np.sin(np.radians(b)) + vy * t,
                        vy,
                        -depth,
                        0.0,
                    ]
                ),
                timestamp=ping_time,
                metadata={"position_mapping": [0, 2, 4]},
            )
            for ping_time in PING_TIMES
            for t in [(ping_time - START).total_seconds()]
        ]
    )
    for name, (r, b, depth, (vx, vy)) in TARGETS.items()
}


def truth_at(name, ping_time):
    """Return a target's true slant range and bearing from the bow dome at a ping."""
    state = next(s for s in targets[name] if s.timestamp == ping_time)
    dome = np.ravel(bow.get_platform_state_at(ping_time).array.ref_state_vector)
    offset = np.ravel(state.state_vector[[0, 2, 4]]) - dome
    return float(np.linalg.norm(offset)), float(np.arctan2(offset[1], offset[0]))


# %%
# Beamforming
# -----------
#
# 221 beams cover 55 degrees either side of the bow in 0.5 degree steps, all steered 4 degrees
# below the horizontal, towards the targets' depression angles of about 3-6 degrees. The fan
# is measured from the array's forward axis, so it turns with the ship.

steering = SteeringCalculator(
    ssp=Constant(speed=SOUND_SPEED_MPS),
    steering_sector_rad=(np.radians(-55.0), np.radians(55.0)),
    num_beams=221,
    steering_elevation_rad=np.radians(-4.0),
)
beamformer = DelayAndSumBeamformer(domain="frequency", sampling_rate_hz=SAMPLING_RATE_HZ)

# %%
# Simulation
# ----------
#
# Bellhop traces each leg once per target and ping, in the vertical plane through the dome
# centre, giving every sensor its own amplitude and delay. White noise at 72 dB re 1 µPa is
# independent between sensors.

pulse = LFMSignal(
    freq_min_hz=FREQ_MIN_HZ,
    freq_max_hz=FREQ_MAX_HZ,
    rise_time_s=0.03,
    duration_s=0.5,
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


def simulator(ground_truth_paths, seed, ping_times=PING_TIMES):
    """Return a beamforming simulator for the bow array with its own noise seed."""
    return ActiveSonarSimulator(
        array=bow,
        propagation_model=propagation_model,
        signal=pulse,
        scattering_models=ConstantTargetStrength(target_strength_db=TARGET_STRENGTH_DB),
        ground_truth_paths=ground_truth_paths,
        listen_duration_s=LISTEN_DURATION_S,
        noise_model=WhiteNoiseSignal(
            amplitude_upa=10 ** (NOISE_LEVEL_DB / 20),
            duration_s=LISTEN_DURATION_S,
            sampling_rate_hz=SAMPLING_RATE_HZ,
            seed=seed,
        ),
        beamformer=beamformer,
        steering_calculator=steering,
        ping_timestamps=ping_times,
    )


pings = [next(iter(data)) for _, data in simulator(list(targets.values()), 1).sensor_data_gen()]

# %%
# Detection
# ---------
#
# - **CFAR**: runs along range on every beam's matched-filter output. Its threshold is calibrated
#   on a noise-only ping for the requested probability of false alarm (Pfa). With 221 beams of
#   32,000 range cells each, a Pfa of :math:`10^{-7}` allows about one false alarm per ping.
# - **Detections on several beams**: a target is detected on several beams, because its mainlobe
#   spans a few beams and its sidelobes, 10-15 dB down, reach further.
# - **Grouping**: detections are grouped greedily, strongest first by matched-filter power. The
#   CFAR detector's own signal-to-noise ratio (SNR) would rank them poorly. On the target's beam,
#   the echo's surface-reflected arrivals sit in the training cells and raise its noise estimate.
# - **Bearing refinement**: when a kept detection's beam is the peak, its bearing is refined by
#   fitting a parabola, in dB, to that beam's matched-filter power and its two neighbours'.

cfar = CACFARDetector(
    num_guard_cells=40,
    num_training_cells=200,
    target_pfa=TARGET_PFA,
    circular=False,
    peak_distance=20,
)
noise_only = simulator([], 2, ping_times=PING_TIMES[:1])
NoiseCalibrator(cfar).calibrate_from_noise(
    matched_filter_scans_from_sensor_data(noise_only.sensor_data_gen())
)
detector = ActiveSonarDetector(
    detector=cfar,
    sensor_data_gen=[(ping.timestamp, {ping}) for ping in pings],
    sampling_rate_hz=SAMPLING_RATE_HZ,
    sound_speed_ms=SOUND_SPEED_MPS,
    min_range_m=200.0,
)
raw_detections = [detections for _, detections in detector.detections_gen()]


def group_detections(detections, ping):
    """Keep one detection per target, with its bearing refined between beams.

    Detections are taken strongest first by matched-filter power. One within 5 m in range of a
    kept detection is dropped if it is within 10 degrees in bearing (the same mainlobe) or more
    than 10 dB weaker (a sidelobe). Returns a list of ``(bearing_rad, range_m, power_db)``.
    """
    bearings = np.asarray(ping.steering_bearings_rad)
    power = matched_filter_power(ping)
    candidates = []
    for detection in detections:
        bearing, range_m = (float(v) for v in detection.state_vector[:, 0])
        beam = int(np.argmin(np.abs(np.angle(np.exp(1j * (bearings - bearing))))))
        sample = min(
            int(round(detection.metadata["delay_s"] * SAMPLING_RATE_HZ)), power.shape[1] - 1
        )
        candidates.append((10 * np.log10(power[beam, sample]), bearing, range_m, beam, sample))
    kept = []
    for power_db, bearing, range_m, beam, sample in sorted(candidates, reverse=True):
        if any(
            abs(range_m - r) < 5.0
            and (abs(np.angle(np.exp(1j * (bearing - b)))) < np.radians(10) or power_db < p - 10)
            for b, r, p in kept
        ):
            continue
        across_beams = power[:, sample]
        is_peak = 0 < beam < len(bearings) - 1 and across_beams[beam] >= max(
            across_beams[beam - 1], across_beams[beam + 1]
        )
        offset = parabolic_peak_offset(10 * np.log10(across_beams), beam) if is_peak else 0.0
        kept.append((bearing + offset * (bearings[1] - bearings[0]), range_m, power_db))
    return kept


grouped = [group_detections(ds, ping) for ds, ping in zip(raw_detections, pings, strict=True)]
for ping_time, raw, kept in zip(PING_TIMES, raw_detections, grouped, strict=True):
    print(f"{ping_time:%H:%M:%S}: {len(raw)} beam detections grouped into {len(kept)}")

# %%
# Matched-filter output on each target's beam
# -------------------------------------------
#
# Each panel shows the matched-filter level along a target's nearest beam on the first ping,
# relative to the median across the ping. The dashed line is the target's true range. Each echo
# is followed closely by its surface-reflected arrivals.

ping = pings[0]
power = matched_filter_power(ping)
noise_floor = np.median(power)
range_axis_m = np.arange(power.shape[1]) * SOUND_SPEED_MPS / (2 * SAMPLING_RATE_HZ)
beam_bearings = np.asarray(ping.steering_bearings_rad)
figure = make_subplots(rows=len(targets), cols=1, subplot_titles=list(targets))
for row, name in enumerate(targets, start=1):
    true_r, true_b = truth_at(name, ping.timestamp)
    beam = int(np.argmin(np.abs(np.angle(np.exp(1j * (beam_bearings - true_b))))))
    window = (range_axis_m > true_r - 30) & (range_axis_m < true_r + 60)
    figure.add_trace(
        go.Scatter(
            x=range_axis_m[window],
            y=10 * np.log10(power[beam, window] / noise_floor),
            mode="lines",
            showlegend=False,
        ),
        row=row,
        col=1,
    )
    figure.add_vline(x=true_r, line_dash="dash", line_color="black", row=row, col=1)
    figure.update_xaxes(range=[true_r - 30, true_r + 60], row=row, col=1)
figure.update_xaxes(title_text="Range (m)", row=len(targets), col=1)
figure.update_yaxes(title_text="dB re median")
figure.update_layout(
    template="plotly_white", title="Matched-Filter Output (Ping 1)", height=700
).show()

# %%
# Bearing and range errors
# ------------------------
#
# Each target's grouped detection is the one nearest its true position on each ping.
#
# - **Bearings**: within a fraction of a degree.
# - **Ranges**: 0.5-1.4 m long, because of surface multipath.
#
# As in the sonobuoy example, the strongest arrival is the pair of mixed paths, which add in
# phase. One goes out direct and comes back via the surface, and the other does the reverse. In
# range, they arrive :math:`z_1 z_2 / R` behind the direct echo, where :math:`z_1` and :math:`z_2`
# are the dome and target depths and :math:`R` the range. The offset for each target is as
# follows.
#
# - **T1**: 0.8 m.
# - **T2**: 1.2 m.
# - **T3**: 0.5 m.


def nearest(kept, true_range_m, true_bearing_rad):
    """Return the grouped detection nearest a true position, in metres."""

    def distance(item):
        b, r, _ = item
        return np.hypot(
            r * np.cos(b) - true_range_m * np.cos(true_bearing_rad),
            r * np.sin(b) - true_range_m * np.sin(true_bearing_rad),
        )

    return min(kept, key=distance)


print(f"\n{'Ping':>8} {'Target':>6} {'Range err (m)':>14} {'Bearing err (deg)':>18}")
errors = {name: [] for name in targets}
for ping_time, kept in zip(PING_TIMES, grouped, strict=True):
    for name in targets:
        true_r, true_b = truth_at(name, ping_time)
        b, r, _ = nearest(kept, true_r, true_b)
        range_err, bearing_err = r - true_r, np.degrees(np.angle(np.exp(1j * (b - true_b))))
        errors[name].append((range_err, bearing_err))
        print(f"{ping_time:%H:%M:%S} {name:>6} {range_err:>14.2f} {bearing_err:>18.2f}")

# %%
# Plan view
# ---------
#
# The plan view shows the ship track, the target tracks and every grouped detection, placed at
# its bearing and range from the dome.

figure = go.Figure()
ship_xy = np.array(
    [[float(s.state_vector[0, 0]), float(s.state_vector[2, 0])] for s in platform.states]
)
figure.add_trace(go.Scatter(x=ship_xy[:, 0], y=ship_xy[:, 1], mode="lines+markers", name="Ship"))
for name, path in targets.items():
    xy = np.array([[float(s.state_vector[0, 0]), float(s.state_vector[2, 0])] for s in path])
    figure.add_trace(
        go.Scatter(x=xy[:, 0], y=xy[:, 1], mode="lines+markers", name=name, marker={"size": 12})
    )
detection_xy = np.array(
    [
        np.ravel(bow.get_platform_state_at(t).array.ref_state_vector)[:2]
        + r * np.array([np.cos(b), np.sin(b)])
        for t, kept in zip(PING_TIMES, grouped, strict=True)
        for b, r, _ in kept
    ]
)
figure.add_trace(
    go.Scatter(
        x=detection_xy[:, 0],
        y=detection_xy[:, 1],
        mode="markers",
        marker={"symbol": "x-thin-open", "size": 10, "color": "black", "line": {"width": 2}},
        name="Detections",
    )
)
figure.update_layout(
    template="plotly_white",
    title="Plan View",
    xaxis_title="x (m)",
    yaxis_title="y (m)",
    yaxis_scaleanchor="x",
).show()

# %%
# DAS and MVDR bearing responses
# ------------------------------
#
# For each target on the first ping, the sensor signals during its echo are beamformed again as
# broadband power, once with DAS and once with MVDR. Both integrate 1400-2600 Hz over 256-point
# frames.
#
# - **T2, dead ahead**: MVDR's -3 dB mainlobe is 2 degrees wide against DAS's 7 degrees. Its
#   highest sidelobe within 25 degrees is 16 dB down, against 8 dB for DAS. Its weights adapt to
#   null everything but the look direction.
# - **T1 and T3, about 30 degrees off the bow**: MVDR's mainlobe is no narrower than DAS's and
#   its sidelobes are higher.
#
# MVDR loses its advantage in two known cases. Either may be at work here.
#
# - **Mismatch**: the arriving wavefront differs from its steering vector [1]_.
# - **Coherent interference**: a coherent copy of the signal arrives from another direction
#   [2]_, as the surface-reflected echo does.
#
# .. [1] Cox, H. "Resolving power and sensitivity to mismatch of optimum array processors."
#        J. Acoust. Soc. Am. 54(3), 771-785, 1973.
# .. [2] Widrow, B., Duvall, K. M., Gooch, R. P. and Newman, W. C. "Signal cancellation
#        phenomena in adaptive antennas: causes and cures." IEEE Trans. Antennas Propag. 30(3),
#        469-478, 1982.

ping = pings[0]
platform_state = bow.get_platform_state_at(ping.timestamp)
steering_delays_s = steering.calculate(platform_state)
bearings_deg = np.degrees(steering.steering_bearings(platform_state))
stft_settings = dict(
    sampling_rate_hz=SAMPLING_RATE_HZ, nfft=256, overlap=192, fmin=1400.0, fmax=2600.0
)
das = DelayAndSumBeamformer(domain="broadband_power", **stft_settings)
mvdr = MinimumVarianceDistortionlessResponseBeamformer(**stft_settings)

figure = make_subplots(rows=1, cols=len(targets), subplot_titles=list(targets))
for column, name in enumerate(targets, start=1):
    true_r, true_b = truth_at(name, ping.timestamp)
    start = int(round(2 * true_r / SOUND_SPEED_MPS * SAMPLING_RATE_HZ))
    echo = ping.received_waveform[:, start : start + pulse.num_samples]
    for label, beamformer_, colour in (("DAS", das, "steelblue"), ("MVDR", mvdr, "crimson")):
        response = np.asarray(beamformer_.beamform(echo, steering_delays_s)).sum(axis=-1)
        figure.add_trace(
            go.Scatter(
                x=bearings_deg,
                y=10 * np.log10(response / response.max()),
                name=label,
                line={"color": colour},
                showlegend=column == 1,
            ),
            row=1,
            col=column,
        )
    figure.add_vline(x=np.degrees(true_b), line_dash="dash", line_color="black", row=1, col=column)
    figure.update_xaxes(
        title_text="Bearing (deg)",
        range=[np.degrees(true_b) - 25, np.degrees(true_b) + 25],
        row=1,
        col=column,
    )
figure.update_yaxes(title_text="Power (dB re peak)", range=[-30, 1], row=1, col=1)
figure.update_layout(
    template="plotly_white", title="DAS and MVDR Bearing Responses (Ping 1)", height=420
).show()
