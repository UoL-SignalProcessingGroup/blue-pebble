"""
======================================
Comparing Theoretical and Empirical Pd
======================================

This example asks how far an analytic OS-CFAR model of false-alarm probability (Pfa) and
detection probability (Pd) can be trusted on beamformed data from Blue Pebble's own acoustic
simulator, and where it fails.

The scenario is stationary, so every scan sees the same target at the same signal-to-noise
ratio (SNR), and Pd can be measured as a hit rate over many scans. The measured ROC (Receiver
Operating Characteristic) is then compared with the analytic model twice: with its default
assumptions, and with the noise and signal statistics measured from the simulated data.
"""  # noqa: D205, D212, D400, D415

# %%
# Imports
# -------

from datetime import datetime, timedelta

# sphinx_gallery_thumbnail_path = "_static/thumbnails/comparing_theoretical_and_empirical_pd.png"
import numpy as np
import plotly.graph_objects as go
from scipy.stats import binomtest
from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel,
    ConstantVelocity,
)
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState

import bluepebble
from bluepebble.detector import (
    NonFluctuating,
    OSCFARDetector,
    RayleighFluctuation,
    estimate_effective_looks_per_frame,
    os_cfar_roc,
    snr_linear_from_ground_truth_bearing,
)
from bluepebble.detector._theory import solve_os_cfar_alpha
from bluepebble.models.environment import FlatBathymetry, Linear
from bluepebble.models.propagation import rtrsAcousticPropagationModel
from bluepebble.platform import TowedArrayPlatform
from bluepebble.signal.anthropogenic import SyntheticAnthropogenicSignal
from bluepebble.signal.random import ColouredNoiseSignal
from bluepebble.sigproc import MinimumVarianceDistortionlessResponseBeamformer, SteeringCalculator
from bluepebble.simulator import ContinuousSTFTPassiveSonarArraySimulator

# %%
# Stationary Scenario
# -------------------
#
# Platform and target do not move, so the geometry, and with it the received SNR, is the same in
# every scan. Each scan is then an independent trial of the same detection problem, which is
# what a Pd estimate needs. The acoustic channel is still ray-traced (``rtrs``), so multipath
# remains part of the problem.

seed = 42
bluepebble.set_seed(seed)

num_scans = 300
time_interval = timedelta(seconds=5.0)
start_time = datetime(2026, 1, 1)
timesteps = [start_time + i * time_interval for i in range(num_scans)]
total_duration_s = num_scans * time_interval.total_seconds()

platform = TowedArrayPlatform(
    states=GroundTruthState(np.array([0.0, 0.0, 0.0, 0.0, -5.0, 0.0]), timestamp=start_time),
    position_mapping=[0, 2, 4],
    velocity_mapping=[1, 3, 5],
    transition_models=[
        CombinedLinearGaussianTransitionModel([ConstantVelocity(0.0) for _ in range(3)])
    ],
    transition_times=[timedelta(seconds=total_duration_s)],
    num_sensors=128,
    cable_length_m=100.0,
    sensor_spacing_m=2.5,
    array_depth_m=-50.0,
)
for timestamp in timesteps[1:]:
    platform.move(timestamp)

target_start_vector = np.array([-2000.0, 0.0, 10000.0, 0.0, -5.0, 0.0])
target_metadata = {
    "amplitudes_upa": [10 ** (120 / 20)],
    "frequencies_hz": [200.0],
    "phases_rad": [np.pi],
    "tonal_bandwidth_hz": 1.0,
    "noise_amplitude_upa": 10 ** (90 / 20),
    "noise_spectral_exponent": -1.0,
    "position_mapping": [0, 2, 4],
}
target_states = [
    GroundTruthState(target_start_vector, timestamp=timestamp, metadata=target_metadata)
    for timestamp in timesteps
]
target_truth = GroundTruthPath(target_states)

ssp = Linear(surface_speed=1500.0, gradient=0.2)
propagation_model = rtrsAcousticPropagationModel(
    ssp=ssp,
    bathymetry=FlatBathymetry(depth=-150.0),
    use_all_frequencies=False,
    step_m=20.0,
    azimuth_search_width=2.0,
    azimuth_resolution=0.5,
    elevation_range=(-25.0, 25.0),
    elevation_resolution=1.0,
)

sampling_rate_hz = 500.0
fmin, fmax = 150.0, 250.0


def make_signal_model():
    """Build the target's source model (one instance per use: its STFT cache is single-use)."""
    return SyntheticAnthropogenicSignal(
        duration_s=total_duration_s,
        sampling_rate_hz=sampling_rate_hz,
        frame_len=500,
        hop_factor=2,
        noise_freq_range_hz=(0.0, sampling_rate_hz / 2),
        tonal_noise_is_constant=True,
        noise_is_constant=True,
    )


# The ambient level puts the target near an SNR where detection is genuinely uncertain.
ambient_noise_model = ColouredNoiseSignal(
    amplitude_upa=10 ** (75 / 20),
    spectral_exponent=-1,
    duration_s=time_interval.total_seconds(),
    sampling_rate_hz=sampling_rate_hz,
)

# The target sits on one side of the array axis, so only that half-plane is steered.
platform_state = platform.get_platform_state_at(timesteps[0])
assert platform_state is not None
array_endpoints_xy = platform_state.array.state_vector[:2, [0, -1]]
array_axis_rad = float(np.arctan2(*(array_endpoints_xy[:, 1] - array_endpoints_xy[:, 0])[::-1]))
steering_azimuths_rad = np.linspace(array_axis_rad - np.pi, array_axis_rad, 360, endpoint=False)

simulator = ContinuousSTFTPassiveSonarArraySimulator(
    platform=platform,
    propagation_model=propagation_model,
    signal_models=[make_signal_model()],
    noise_model=ambient_noise_model,
    beamformer=MinimumVarianceDistortionlessResponseBeamformer(
        sampling_rate_hz=sampling_rate_hz, fmin=fmin, fmax=fmax
    ),
    steering_calculator=SteeringCalculator(ssp=ssp, steering_azimuths_rad=steering_azimuths_rad),
    ground_truth_paths=[target_truth],
    fade_in_ms=1000.0,
)

# %%
# Collecting Beamformed Scans
# ---------------------------
#
# Only scans with the modal frame count are kept. The first scan is also dropped, because the
# simulator's fade-in depresses it.

all_scans = []
for _, sensor_data_set in simulator.sensor_data_gen(progress_bar=True):
    (sensor_data,) = sensor_data_set
    assert sensor_data.beamformed_data is not None
    all_scans.append(sensor_data.beamformed_data)
frame_counts = np.array([scan.shape[1] for scan in all_scans])
num_frames = int(np.bincount(frame_counts).argmax())
scans = [scan for i, scan in enumerate(all_scans) if i > 0 and scan.shape[1] == num_frames]
print(f"{len(scans)} of {num_scans} scans kept, each with {num_frames} frames")

# %%
# The Scenario's SNR
# ------------------
#
# ``snr_linear`` is the input the analytic model needs. Because the scenario is stationary,
# averaging the per-scan readings gives one stable value.

array_centre_xy = np.mean(platform_state.array.state_vector, axis=1)[:2]
target_bearing_rad = float(
    np.arctan2(
        target_start_vector[2] - array_centre_xy[1], target_start_vector[0] - array_centre_xy[0]
    )
)
guard_bins = 6
readings = [
    snr_linear_from_ground_truth_bearing(
        scan, target_bearing_rad, steering_azimuths_rad, guard_bins
    )
    for scan in scans
]
snr_per_scan = np.array([snr for snr, _ in readings])
target_bin = readings[0][1]
snr_linear = float(snr_per_scan.mean())
print(
    f"snr_linear: mean {snr_linear:.3f} ({10 * np.log10(snr_linear):.1f} dB), "
    f"per-scan std {snr_per_scan.std(ddof=1):.3f}"
)

# %%
# Statistics the Model Needs
# --------------------------
#
# By default the analytic model treats each per-frame power sample as one exponential draw
# (``effective_looks_per_frame`` K = 1) and the target as filling the whole processed band
# (``signal_looks_per_frame`` K_s = K). A broadband beamformer integrates many correlated
# frequency bins, so K is measured from the noise-only cells. K_s is the participation ratio of
# the source spectrum within the band, which for a tonal is far smaller than K.

noise_only_mask = np.ones((len(scans), len(steering_azimuths_rad)), dtype=bool)
noise_only_mask[:, max(0, target_bin - guard_bins) : target_bin + guard_bins + 1] = False
effective_looks = estimate_effective_looks_per_frame(scans, noise_only_mask)

source_stft, source_frequencies_hz, _, _ = make_signal_model().compute_stft(target_states[0])
in_band = (source_frequencies_hz >= fmin) & (source_frequencies_hz <= fmax)
band_power = np.mean(np.abs(source_stft[:, in_band]) ** 2, axis=0)
signal_looks = float(1.0 / np.sum((band_power / band_power.sum()) ** 2))
print(f"K = {effective_looks:.1f} effective looks per frame; K_s = {signal_looks:.2f}")

# %%
# Measured ROC
# ------------
#
# The detector's own statistic, power over its local noise estimate, is read from
# ``detection_snr_map`` for every cell. A threshold is set at each Pfa from the noise-only cells,
# and Pd is the fraction of scans in which the target cell exceeds it. The interval is a Wilson
# 95% interval, which treats scans as independent trials.

detector = OSCFARDetector(
    num_guard_cells=2, num_training_cells=10, rank=15, target_pfa=0.01, circular=False
)
snr_maps_db = np.array([detector.detection_snr_map(scan) for scan in scans])
target_statistic_db = snr_maps_db[:, target_bin]
noise_cells_db = snr_maps_db[noise_only_mask]

deviation = target_statistic_db - target_statistic_db.mean()
lag1 = float(np.sum(deviation[:-1] * deviation[1:]) / np.sum(deviation**2))
print(f"lag-1 autocorrelation of the target cell across scans: {lag1:+.2f}")

pfa_grid = np.geomspace(1e-3, 0.9, num=15)
thresholds_db = np.quantile(noise_cells_db, 1.0 - pfa_grid)
hits = np.array([np.sum(target_statistic_db > threshold) for threshold in thresholds_db])
intervals = [binomtest(int(k), len(scans)).proportion_ci(method="wilson") for k in hits]
pd_empirical = hits / len(scans)
pd_low = np.array([interval.low for interval in intervals])
pd_high = np.array([interval.high for interval in intervals])

# %%
# Analytic ROC
# ------------
#
# The same ROC from ``os_cfar_roc``, at the measured ``snr_linear``, for two target fluctuation
# models and two sets of assumptions: defaults, and the measured statistics above.

num_training_total = 20
rank = 15
models = {"Rayleigh": RayleighFluctuation(), "Non-fluctuating": NonFluctuating()}

analytic_pd = {}
for name, model in models.items():
    for label, looks, signal_looks_per_frame in (
        ("defaults", 1.0, None),
        ("measured", effective_looks, signal_looks),
    ):
        analytic_pd[(name, label)] = os_cfar_roc(
            pfa_grid,
            num_training_total=num_training_total,
            rank=rank,
            num_frames=num_frames,
            snr_linear=snr_linear,
            model=model,
            num_trials=1_000_000,
            rng=np.random.default_rng(seed),
            effective_looks_per_frame=looks,
            signal_looks_per_frame=signal_looks_per_frame,
        )

# %%
# What ``target_pfa`` Delivers
# ----------------------------
#
# The analytic model also sets the detector's threshold. Applying that threshold to the
# noise-only cells gives the false-alarm rate it actually produces.

achieved_pfa = {}
for label, looks in (("defaults", 1.0), ("measured K", effective_looks)):
    achieved_pfa[label] = np.array(
        [
            np.mean(
                noise_cells_db
                > 10
                * np.log10(
                    solve_os_cfar_alpha(
                        pfa, num_training_total, rank, num_frames, effective_looks_per_frame=looks
                    )
                )
            )
            for pfa in pfa_grid
        ]
    )

# %%
# Comparison
# ----------

print(" | ".join(["Pfa", "Pd measured [95% CI]", *(f"Pd {n} ({lb})" for n, lb in analytic_pd)]))
for i, pfa in enumerate(pfa_grid):
    cells = [f"{pfa:.4f}", f"{pd_empirical[i]:.3f} [{pd_low[i]:.3f}, {pd_high[i]:.3f}]"]
    cells += [f"{pd[i]:.3f}" for pd in analytic_pd.values()]
    print(" | ".join(cells))
print()
print("Pfa requested | " + " | ".join(f"achieved ({label})" for label in achieved_pfa))
for i, pfa in enumerate(pfa_grid):
    print(f"{pfa:.4f} | " + " | ".join(f"{a[i]:.5f}" for a in achieved_pfa.values()))

# %%
# ROC: Measured Against the Analytic Model
# ----------------------------------------
#
# Dotted lines use the model's defaults, solid lines the measured statistics. Error bars are the
# 95% intervals on the measured Pd.

fig_roc = go.Figure()
fig_roc.add_trace(
    go.Scatter(
        x=pfa_grid,
        y=pd_empirical,
        mode="markers",
        name="Measured",
        marker=dict(color="black", size=7),
        error_y=dict(
            type="data",
            symmetric=False,
            array=pd_high - pd_empirical,
            arrayminus=pd_empirical - pd_low,
        ),
    )
)
line_dash = {"defaults": "dot", "measured": "solid"}
colours = {"Rayleigh": "#0072b2", "Non-fluctuating": "#d55e00"}
for (name, label), pd in analytic_pd.items():
    fig_roc.add_trace(
        go.Scatter(
            x=pfa_grid,
            y=pd,
            mode="lines",
            name=f"{name} ({label})",
            line=dict(color=colours[name], dash=line_dash[label]),
        )
    )
fig_roc.update_xaxes(type="log", dtick=1, exponentformat="power", title_text="Pfa")
fig_roc.update_yaxes(title_text="Pd", range=[0, 1.05])
fig_roc.update_layout(
    title="Measured and Analytic ROC",
    template="plotly_white",
    autosize=True,
    width=None,
    height=500,
)

# %%
# Requested Against Achieved Pfa
# ------------------------------
#
# The false-alarm rate that each model's threshold produces on the noise-only cells. Points on
# the dashed line deliver the requested Pfa. Zero counts are drawn at one over the number of
# noise-only cells.

fig_pfa = go.Figure()
fig_pfa.add_trace(
    go.Scatter(
        x=pfa_grid,
        y=pfa_grid,
        mode="lines",
        name="Requested = achieved",
        line=dict(color="black", dash="dash"),
    )
)
for label, achieved in achieved_pfa.items():
    fig_pfa.add_trace(
        go.Scatter(
            x=pfa_grid,
            y=np.maximum(achieved, 1.0 / noise_cells_db.size),
            mode="lines+markers",
            name=f"Model threshold ({label})",
        )
    )
fig_pfa.update_xaxes(type="log", dtick=1, exponentformat="power", title_text="Requested Pfa")
fig_pfa.update_yaxes(type="log", dtick=1, exponentformat="power", title_text="Achieved Pfa")
fig_pfa.update_layout(
    title="Requested and Achieved Pfa",
    template="plotly_white",
    autosize=True,
    width=None,
    height=500,
)

# %%
# Reading the Results
# -------------------
#
# **With its defaults the analytic model is wrong by an order of magnitude.** Below a Pfa of
# about 1% it predicts a Pd of roughly 0.01 to 0.06 where the measured Pd is about 0.66 to 0.83.
# It also does not deliver the requested Pfa: its threshold produces essentially no false alarms
# until the requested Pfa passes about 0.1, and then admits almost every cell by 0.55. The
# beamformer integrates many correlated frequency bins, so the noise-only statistic is far
# smoother than the model's single exponential draw per frame (K is measured at about 36, not 1)
# and sits in a narrow band, which any threshold from the wrong spread misses.
#
# **With the measured statistics the model is a reasonable but biased predictor.** The threshold
# it sets delivers a Pfa within about 20% of the request across the whole range, though K was
# estimated from these same noise cells, so that check is in-sample. The Rayleigh model
# overpredicts Pd by up to about 7 points, mostly at or just above the upper edge of the 95%
# intervals. The non-fluctuating model, which one might have expected to suit a fixed tone,
# overpredicts by up to about 13 points. The choice of fluctuation model therefore matters about
# as much as the error that remains, and this example does not establish why both overpredict.
#
# The target cell's lag-1 autocorrelation across scans is close to zero, so treating scans as
# independent trials in the intervals is reasonable. The thresholds here come from the measured
# noise cells rather than from a model. For setting thresholds in operation see
# :doc:`/auto_examples/calibrating_cfar_from_noise`.
