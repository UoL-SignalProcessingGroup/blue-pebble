"""
================================================
Multiband Detection: Bands and Their Detections
================================================

Splitting a passive-sonar band into sub-bands lets a tonal compete only against the noise
in its own band, rather than against the noise spanning the whole processing band.

This example shows the mechanism at ``N = 5`` bands spanning
50-240 Hz: each band gets its own SNR map and its own detections, sized with a CFAR/peak
chain tuned to that band's mainlobe width. These detections per band can also be collapsed
into a single set of detections for downstream tracking.
"""  # noqa: D205, D212, D400, D415

# %%
# Imports
# -------
#
# All dependencies are consolidated here for convenience.

import math
from datetime import datetime, timedelta

import numpy as np
from plotly.subplots import make_subplots
from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel,
    ConstantVelocity,
    KnownTurnRate,
)
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState

import bluepebble
from bluepebble.detector import (
    BandDetector,
    CACFARDetector,
    MultibandPassiveSonarDetector,
    PeakDetector,
)
from bluepebble.models.environment import FlatBathymetry, Linear
from bluepebble.models.propagation import CylindricalAcousticPropagationModel
from bluepebble.platform import TowedArrayPlatform
from bluepebble.plotter import apply_shared_colourscale, deduplicate_legend, plot_btr, plot_world
from bluepebble.signal.anthropogenic import SyntheticAnthropogenicSignal
from bluepebble.signal.random import ColouredNoiseSignal
from bluepebble.sigproc import DelayAndSumBeamformer, FrequencyBand, SteeringCalculator
from bluepebble.simulator import ContinuousSTFTPassiveSonarArraySimulator

# %%
# Simulation Parameters
# ---------------------

seed = 2000
bluepebble.set_seed(seed)
rng = bluepebble.get_rng()

sim_length_s = 900  # seconds
sim_rate_s = 5.0  # seconds

start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
time_interval = timedelta(seconds=sim_rate_s)
num_steps = int(sim_length_s / sim_rate_s)
timesteps = np.array([start_time + i * time_interval for i in range(num_steps)], dtype=object)

total_duration_s = num_steps * time_interval.total_seconds()

sound_speed_ms = 1500.0

# %%
# Array Geometry
# --------------

num_sensors = 50
sensor_spacing_m = 1.25
tow_cable_length_m = 100.0
array_depth_m = -50.0

array_aperture_m = (num_sensors - 1) * sensor_spacing_m

num_beams = 120
steering_azimuths_rad = np.linspace(-np.pi, np.pi, num_beams, endpoint=False)
beam_spacing_deg = 360.0 / num_beams

band_fmin_hz = 50.0
band_fmax_hz = 240.0


def _beamwidth_deg(frequency_hz: float) -> float:
    """Return the -3 dB mainlobe width of the uniform line array at one frequency."""
    return float(np.rad2deg(0.886 * (sound_speed_ms / frequency_hz) / array_aperture_m))


print(f"Aperture {array_aperture_m:.2f} m, {num_beams} beams at {beam_spacing_deg:.2f} deg")

# %%
# Platform Manoeuvres
# --------------------


def _build_turn(angle_deg: float, rate_deg_per_s: float):
    angle_rad = np.deg2rad(angle_deg)
    turn_rate_radps = np.deg2rad(rate_deg_per_s)
    planar_turn = KnownTurnRate(
        turn_rate=np.sign(angle_rad) * turn_rate_radps,
        turn_noise_diff_coeffs=np.array([0.0, 0.0]),
    )
    turn_duration_s = round((abs(angle_rad) / turn_rate_radps) / sim_rate_s) * sim_rate_s
    return (
        CombinedLinearGaussianTransitionModel([planar_turn, ConstantVelocity(0.0)]),
        timedelta(seconds=turn_duration_s),
    )


straight_model = CombinedLinearGaussianTransitionModel(
    [ConstantVelocity(0.0), ConstantVelocity(0.0), ConstantVelocity(0.0)]
)

turn_configs = [
    {"angle_deg": -85.0, "rate_deg_per_s": 1.0},
    {"angle_deg": 85.0, "rate_deg_per_s": 1.0},
]
leg_duration_seconds = [sim_length_s / 3, sim_length_s / 5, sim_length_s / 3]
leg_durations_s = [
    timedelta(seconds=round(seconds / sim_rate_s) * sim_rate_s) for seconds in leg_duration_seconds
]

maneuvers = []
for idx, turn_config in enumerate(turn_configs):
    maneuvers.append({"type": "leg", "duration": leg_durations_s[idx]})
    maneuvers.append({"type": "turn", **turn_config})
maneuvers.append({"type": "leg", "duration": leg_durations_s[-1]})

platform_transition_models = []
platform_transition_times = []
for maneuver in maneuvers:
    if maneuver["type"] == "leg":
        platform_transition_models.append(straight_model)
        platform_transition_times.append(maneuver["duration"])
    else:
        turn_model, turn_duration = _build_turn(maneuver["angle_deg"], maneuver["rate_deg_per_s"])
        platform_transition_models.append(turn_model)
        platform_transition_times.append(turn_duration)

# %%
# Platform Setup and Generation
# -----------------------------

platform_start_vector = np.array([5000.0, 0.0, 1000.0, 5.0, -5.0, 0.0])

platform = TowedArrayPlatform(
    states=[GroundTruthState(platform_start_vector, timestamp=start_time)],
    position_mapping=[0, 2, 4],
    velocity_mapping=[1, 3, 5],
    transition_models=platform_transition_models,
    transition_times=platform_transition_times,
    num_sensors=num_sensors,
    cable_length_m=tow_cable_length_m,
    sensor_spacing_m=sensor_spacing_m,
    array_depth_m=array_depth_m,
)

for i in range(1, num_steps):
    platform.move(start_time + i * time_interval)

# %%
# Ground Truth Setup and Generation
# ---------------------------------

num_tonals_per_target = 4
tonal_min_hz = 75.0
tonal_max_hz = 240.0

target_configs = [
    {
        "name": "Target 1",
        "start_vector": np.array([4500.0, 5.0, 2000.0, 0.0, -5.0, 0.0]),
        "tonal_level_db": 82.0,
        "noise_level_db": 79.0,
    },
    {
        "name": "Target 2",
        "start_vector": np.array([1000.0, 5.0, 3000.0, 3.0, -5.0, 0.0]),
        "tonal_level_db": 80.0,
        "noise_level_db": 77.0,
    },
]

constant_velocity_model = CombinedLinearGaussianTransitionModel(
    [ConstantVelocity(0.0), ConstantVelocity(0.0), ConstantVelocity(0.0)]
)

for target_config in target_configs:
    frequencies = np.sort(rng.uniform(tonal_min_hz, tonal_max_hz, num_tonals_per_target))
    target_config["frequencies_hz"] = frequencies
    target_config["metadata"] = {
        "amplitudes_upa": 10
        ** (np.full(num_tonals_per_target, target_config["tonal_level_db"]) / 20),
        "frequencies_hz": list(frequencies),
        "phases_rad": list(rng.uniform(0.0, 2 * np.pi, num_tonals_per_target)),
        "position_mapping": [0, 2, 4],
        "velocity_mapping": [1, 3, 5],
        "tonal_bandwidth_hz": 2.0,
        "noise_amplitude_upa": 10 ** (target_config["noise_level_db"] / 20),
        "noise_spectral_exponent": -1.0,
    }

for target_config in target_configs:
    tonal_text = ", ".join(f"{f:.1f}" for f in target_config["frequencies_hz"])
    print(f"{target_config['name']}: tonals at {tonal_text} Hz")

target_ground_truths = []
relative_bearing_ground_truths = []

for target_config in target_configs:
    target_states = [
        GroundTruthState(
            target_config["start_vector"],
            timestamp=start_time,
            metadata=target_config["metadata"],
        )
    ]

    for i in range(1, num_steps):
        new_time = start_time + i * time_interval
        interval_now = new_time - target_states[-1].timestamp
        new_state_vector = constant_velocity_model.function(
            target_states[-1],
            noise=False,
            time_interval=interval_now,
        )
        target_states.append(
            GroundTruthState(
                new_state_vector,
                timestamp=new_time,
                metadata=target_states[-1].metadata,
            )
        )

    target_ground_truth = GroundTruthPath(target_states)
    target_ground_truths.append(target_ground_truth)

    bearing_states = []
    for target_state in target_ground_truth.states:
        platform_state = platform.get_platform_state_at(target_state.timestamp)
        assert platform_state is not None
        ref_sensor_position = np.mean(platform_state.array.state_vector, axis=1)
        target_position = np.array([target_state.state_vector[0], target_state.state_vector[2]])
        relative_position = target_position - ref_sensor_position[:2]
        bearing_rad = np.arctan2(relative_position[1], relative_position[0])
        bearing_states.append(
            GroundTruthState(
                state_vector=np.array([bearing_rad]),
                timestamp=target_state.timestamp,
            )
        )

    relative_bearing_ground_truths.append(GroundTruthPath(bearing_states))

fig_world = plot_world(truths=target_ground_truths, platform=platform).update_layout(
    title="World Picture: Target and Platform Trajectories",
    template="plotly_white",
    autosize=True,
    width=None,
    height=None,
)
# fig_world.show()

# %%
# Acoustic Environment and Signal Models
# --------------------------------------

ssp = Linear(surface_speed=sound_speed_ms, gradient=0.2)
bathymetry = FlatBathymetry(depth=-150.0)

propagation_model = CylindricalAcousticPropagationModel(
    ssp=ssp,
    attenuation_factor=10.0,
)

sampling_rate_hz = 500.0
frame_len = 1000
hop_factor = 2
fade_in_ms = 1000.0
ambient_noise_model = ColouredNoiseSignal(
    amplitude_upa=10 ** (55.0 / 20),
    spectral_exponent=-1.0,
    duration_s=time_interval.total_seconds(),
    sampling_rate_hz=sampling_rate_hz,
)


def _make_signal_models():
    """Build one signal model per target."""
    models = []
    for target_ground_truth in target_ground_truths:
        target_metadata = next(iter(target_ground_truth)).metadata
        models.append(
            SyntheticAnthropogenicSignal(
                duration_s=total_duration_s,
                sampling_rate_hz=sampling_rate_hz,
                frame_len=frame_len,
                hop_factor=hop_factor,
                tonal_bandwidth_hz=target_metadata["tonal_bandwidth_hz"],
                noise_amplitude_upa=target_metadata["noise_amplitude_upa"],
                noise_spectral_exponent=target_metadata["noise_spectral_exponent"],
                noise_freq_range_hz=(0.0, sampling_rate_hz / 2),
                tonal_noise_is_constant=True,
                noise_is_constant=True,
            )
        )
    return models


# %%
# Five Non-Overlapping Bands
# ---------------------------
#
# Band edges are offset by half a bin. :class:`~.FrequencyBand` is inclusive at *both*
# edges, so two bands meeting exactly on a bin frequency would both claim that bin and
# double-count it. With 1 Hz bins, half-integer edges guarantee no bin lands on a boundary.

band_count = 5
edges = np.linspace(band_fmin_hz, band_fmax_hz, band_count + 1) + 0.5
view_bands = [
    FrequencyBand(
        label=f"{edges[i]:.0f}-{edges[i + 1]:.0f} Hz",
        fmin=float(edges[i]),
        fmax=float(edges[i + 1]) - 1e-9,
    )
    for i in range(band_count)
]


def _tonals_in_band(band: FrequencyBand) -> str:
    """Describe which target tonals fall inside a band."""
    present = []
    for target_config in target_configs:
        inside = [f for f in target_config["frequencies_hz"] if band.fmin <= f <= band.fmax]
        if inside:
            label = target_config["name"].replace("Target ", "T")
            present.append(f"{label}: {', '.join(f'{f:.0f}' for f in inside)} Hz")
    return "<br>".join(present) if present else "no tonals"


# %%
# Sizing the Detector to the Band
# --------------------------------
#
# A CFAR window is a fixed number of beams, but a mainlobe is a fixed angle, so each band
# gets guard/training cells sized to its own beamwidth rather than settings borrowed from
# another band.

guard_scale = 2.0
train_scale = 4.0
threshold_factor = 1.75


def _band_detection_chain(band: FrequencyBand) -> list:
    """Build a CFAR + peak chain sized to a band's widest mainlobe."""
    mainlobe_beams = _beamwidth_deg(band.fmin) / beam_spacing_deg
    guard = max(1, math.ceil(guard_scale * mainlobe_beams / 2))
    training = max(2, math.ceil(train_scale * guard))
    return [
        CACFARDetector(
            num_guard_cells=guard,
            num_training_cells=training,
            threshold_factor=threshold_factor,
        ),
        PeakDetector(distance=max(1, math.ceil(mainlobe_beams))),
    ]


chains = {band.label: _band_detection_chain(band) for band in view_bands}

# %%
# Running the Simulation
# -----------------------
#
# Note that mirror half plane is set on the steering calculator. This mirrors the
# beamformer using half the beams which is faster.

simulator = ContinuousSTFTPassiveSonarArraySimulator(
    platform=platform,
    propagation_model=propagation_model,
    signal_models=_make_signal_models(),
    noise_model=ambient_noise_model,
    beamformer=DelayAndSumBeamformer(
        sampling_rate_hz=sampling_rate_hz,
        domain="broadband_power",
        nfft=frame_len,
        overlap=frame_len // hop_factor,
        bands=view_bands,
    ),
    steering_calculator=SteeringCalculator(
        ssp=ssp,
        steering_azimuths_rad=steering_azimuths_rad,
        mirror_half_plane=True,
        ),
    ground_truth_paths=target_ground_truths,
    fade_in_ms=fade_in_ms,
)

labels = [band.label for band in view_bands]

# %%
# Detections, Per Band and Collapsed
# -------------------------------------
#
# Each band's chain and normalisation live on its own :class:`~.BandDetector`.
# :class:`~.MultibandPassiveSonarDetector` consumes ``simulator.sensor_data_gen()`` directly,
# The same way ``PassiveSonarDetector`` does in every single-band example, and its
# ``detections_gen`` already yields the **union** of every band's detections per timestep,
# so a downstream tracker gets exactly this stream. Bucketing each detection by the ``band``
# already recorded in its metadata gives the per-band lists used for the panels below; a
# tonal picked up by two bands still counts as two detections, one per band, not merged into
# one. ``snr_history`` records the per-band SNR maps as a side effect of the same pass.

band_detectors = {label: BandDetector(detection_chain=chain) for label, chain in chains.items()}

detector = MultibandPassiveSonarDetector(
    band_detectors=band_detectors,
    sensor_data_gen=simulator.sensor_data_gen(),
    steering_azimuths_rad=steering_azimuths_rad,
)

detections_by_band: dict[str, list] = {label: [] for label in labels}
collapsed_detections = []
for _timestamp, batch in detector.detections_gen(progress_bar=True, total_timesteps=num_steps):
    for detection in batch:
        collapsed_detections.append(detection)
        detections_by_band[detection.metadata["band"]].append(detection)

snr_maps = detector.snr_history
map_rows = min(len(timesteps), next(iter(snr_maps.values())).shape[0])

total_per_band = sum(len(dets) for dets in detections_by_band.values())
print(f"{total_per_band} per-band detections, {len(collapsed_detections)} collapsed detections")

# %%
# Sub-band plots and collapsed detections
# ----------------------------------------
#
# Top row: each band's own SNR map. Bottom row: the detections that band's own chain
# produced. Panel titles list the target tonals that fall inside each band.
# Last column is the collapsed view: no single-band SNR map to show on top (it's a union
# across bands, not one band's own map), just the collapsed detections below.

subplot_titles = [f"{band.label}<br>{_tonals_in_band(band)}" for band in view_bands]
subplot_titles.append("Collapsed<br>Union of All Bands")

fig_bands = make_subplots(
    rows=2,
    cols=len(view_bands) + 1,
    shared_yaxes=True,
    vertical_spacing=0.12,
    horizontal_spacing=0.012,
    subplot_titles=subplot_titles,
)

for col, band in enumerate(view_bands, start=1):
    plot_btr(
        data=snr_maps[band.label][:map_rows],
        timesteps=np.array(timesteps[:map_rows]),
        steering_azimuths=np.rad2deg(steering_azimuths_rad),
        fig=fig_bands,
        row=1,
        col=col,
    )
    plot_btr(
        data=None,
        truths=relative_bearing_ground_truths,
        detections=detections_by_band[band.label],
        timesteps=np.array(timesteps[:map_rows]),
        steering_azimuths=np.rad2deg(steering_azimuths_rad),
        fig=fig_bands,
        row=2,
        col=col,
    )
    fig_bands.update_xaxes(title_text="Bearing (deg)", row=2, col=col)
    fig_bands.update_xaxes(title_text="", showticklabels=False, row=1, col=col)
    if col > 1:
        fig_bands.update_yaxes(title_text=None, row=1, col=col)
        fig_bands.update_yaxes(title_text=None, row=2, col=col)

plot_btr(
        data=None,
        timesteps=np.array(timesteps[:map_rows]),
        steering_azimuths=np.rad2deg(steering_azimuths_rad),
        fig=fig_bands,
        row=1,
        col=len(view_bands) + 1,
    )
plot_btr(
        data=None,
        truths=relative_bearing_ground_truths,
        detections=collapsed_detections,
        timesteps=np.array(timesteps[:map_rows]),
        steering_azimuths=np.rad2deg(steering_azimuths_rad),
        fig=fig_bands,
        row=2,
        col=len(view_bands) + 1,
    )

fig_bands.update_yaxes(title_text=None, row=1, col=len(view_bands) + 1)
fig_bands.update_yaxes(title_text=None, row=2, col=len(view_bands) + 1)
fig_bands.update_xaxes(title_text="Bearing (deg)", row=2, col=len(view_bands) + 1)
fig_bands.update_xaxes(title_text="", showticklabels=False, row=1, col=len(view_bands) + 1)
fig_bands.update_yaxes(title_text="Time (HH:MM)", row=1, col=1)
fig_bands.update_yaxes(title_text="Time (HH:MM)", row=2, col=1)
deduplicate_legend(fig_bands)
apply_shared_colourscale(
    fig_bands,
    colorbar=dict(
        title=dict(text="SNR (dB)", side="right"),
        x=1.02,
        xanchor="left",
        y=0.75,
        yanchor="middle",
        len=0.5,
        thickness=20,
    ),
)
fig_bands.for_each_annotation(lambda a: a.update(font=dict(size=10)))
fig_bands.update_layout(
    template="plotly_white",
    autosize=True,
    width=None,
    height=800,
    margin=dict(r=90, t=90),
    title=f"N={band_count} band beamformer, with per band detections",
    legend=dict(x=0.5, y=-0.2, xanchor="center", yanchor="top", orientation="h"),
)
# fig_bands.show()

# %%
# Key Takeaways
# -------------
#
# * **Each band gets its own SNR map and its own detector.** A tonal confined to one
#   narrow band competes only with the noise in that band, so it stands further above its
#   local noise floor than it would in a single wide band; at the cost of running one
#   detector per band, each taking its own trial against noise.
# * **Detectors must be sized to their band.** A CFAR window is a fixed number of beams,
#   but a mainlobe is a fixed angle whose width in beams changes across 50-240 Hz. Sizing
#   guard and training cells per band is the point of :class:`~.BandDetector`.
# * **Downstream tracking could be on detections from one band, or the union of all bands.**
