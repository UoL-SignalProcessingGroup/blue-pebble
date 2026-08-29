"""
======================================
Comparing Theoretical and Empirical Pd
======================================

This example compares the closed-form/Monte Carlo Pd-vs-Pfa model in
:mod:`bluepebble.detector.fluctuation_models` against :class:`~.OSCFARDetector` run on
beamformed data from Blue Pebble's own acoustic simulator.

This example asks the question: how far does an actual simulated acoustic scenario (with 
propagation, multipath, and a moving platform/target) depart from the idealised model, and why?
"""  # noqa: D205, D212, D400, D415

# %%
# Imports
# -------

from datetime import datetime, timedelta

import numpy as np
import plotly.graph_objects as go
from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel,
    ConstantVelocity,
)
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState

import bluepebble
from bluepebble.detector.algorithms import OSCFARDetector, calibrate_os_cfar_alpha_mc
from bluepebble.detector.fluctuation_models import NonFluctuating, RayleighFluctuation
from bluepebble.detector.metrics import (
    SweepResult,
    SweepSpec,
    estimate_effective_looks_per_frame,
    os_cfar_roc,
    snr_linear_from_ground_truth_bearing,
    sweep_detection_parameter,
)
from bluepebble.models.environment import FlatBathymetry, Linear
from bluepebble.models.propagation.acoustic import rtrsAcousticPropagationModel
from bluepebble.platform import TowedArrayPlatform
from bluepebble.plotter import plot_btr, plot_roc, plot_world
from bluepebble.signal.anthropogenic import SyntheticAnthropogenicSignal
from bluepebble.signal.random import ColouredNoiseSignal
from bluepebble.sigproc import MinimumVarianceDistortionlessResponseBeamformer, SteeringCalculator
from bluepebble.simulator import ContinuousSTFTPassiveSonarArraySimulator

# %%
# Scenario Setup
# ---------------

seed = 42
bluepebble.set_seed(seed)
rng = bluepebble.get_rng()

sim_length_s = 300
sim_rate_s = 5.0
time_interval = timedelta(seconds=sim_rate_s)
num_steps = int(sim_length_s / sim_rate_s)
start_time = datetime(2026, 1, 1, 0, 0, 0)
timesteps = np.array([start_time + i * time_interval for i in range(num_steps)], dtype=object)

platform_start_vector = np.array([0.0, 0.0, 0.0, 0.0, -5.0, 0.0])
platform_position_mapping = [0, 2, 4]
platform_velocity_mapping = [1, 3, 5]
platform_transition_model = CombinedLinearGaussianTransitionModel(
    [ConstantVelocity(0), ConstantVelocity(0), ConstantVelocity(0)]
)

num_sensors = 128
tow_cable_length_m = 100.0
sensor_spacing_m = 2.5 # lambda_min = 1 / fmin * c = 1 / 250 * 1500 = 6 m, so 2.5 m spacing is < lambda_min / 2
array_depth_m = -50.0

platform_initial_state = GroundTruthState(platform_start_vector, timestamp=start_time)
platform = TowedArrayPlatform(
    states=platform_initial_state,
    position_mapping=platform_position_mapping,
    velocity_mapping=platform_velocity_mapping,
    transition_models=[platform_transition_model],
    transition_times=[timedelta(seconds=sim_length_s)],
    num_sensors=num_sensors,
    cable_length_m=tow_cable_length_m,
    sensor_spacing_m=sensor_spacing_m,
    array_depth_m=array_depth_m,
)
for timestamp in timesteps[1:]:
    platform.move(timestamp)

target_start_vector = np.array([-2000.0, 10.0, 10000.0, 0.0, -5.0, 0.0])
target_transition_model = CombinedLinearGaussianTransitionModel(
    [ConstantVelocity(0), ConstantVelocity(0), ConstantVelocity(0)]
)
target_position_mapping = [0, 2, 4]
target_velocity_mapping = [1, 3, 5]

target_amplitudes_upa = [10 ** (120 / 20)]
target_frequencies_hz = [200.0]
target_phases_rad = [np.pi]
target_tonal_bandwidth_hz = 1.0
target_noise_amplitude_upa = 10 ** (90 / 20)
target_noise_spectral_exponent = -1.0  # Pink noise

target_metadata = {
    "amplitudes_upa": target_amplitudes_upa,
    "frequencies_hz": target_frequencies_hz,
    "phases_rad": target_phases_rad,
    "position_mapping": target_position_mapping,
    "velocity_mapping": target_velocity_mapping,
    "tonal_bandwidth_hz": target_tonal_bandwidth_hz,
    "noise_amplitude_upa": target_noise_amplitude_upa,
    "noise_spectral_exponent": target_noise_spectral_exponent,
}

target_states = [
    GroundTruthState(target_start_vector, timestamp=start_time, metadata=target_metadata)
]
for timestamp in timesteps[1:]:
    dt = timestamp - target_states[-1].timestamp
    new_state_vector = target_transition_model.function(
        target_states[-1], noise=False, time_interval=dt
    )
    target_states.append(
        GroundTruthState(new_state_vector, timestamp=timestamp, metadata=target_metadata)
    )

target_truth = GroundTruthPath(target_states)

ssp = Linear(surface_speed=1500.0, gradient=0.2)
bathymetry = FlatBathymetry(depth=-150.0)
attenuation_factor = 0.5
propagation_model = rtrsAcousticPropagationModel(
    ssp=ssp,
    bathymetry=bathymetry,
    use_all_frequencies=False,
    step_m=20.0,
    azimuth_search_width=2.0,
    azimuth_resolution=0.5,
    elevation_range=(-25.0, 25.0),
    elevation_resolution=1.0,
)

sampling_rate_hz = 500.0
frame_len = 500
hop_factor = 2
fade_in_ms = 1000.0
total_duration_s = num_steps * time_interval.total_seconds()

# 75 dB rather than a quieter 70 places the target near snr_linear = 0.35 at mid-scenario,
# inside the region where detection is genuinely uncertain. At 70 dB this scenario sits so far
# above the noise (a target doubling cell power against a noise floor with 212 degrees of
# freedom is a ~14-sigma excursion) that every Pd curve saturates at 1 and the ROC comparison
# below discriminates between nothing. Raising the ambient rather than lowering the source is
# deliberate: for a given change in dB the two are exactly equivalent -- the whole chain is
# scale-invariant and every quantity here is a ratio -- but the source level is split across
# target_amplitudes_upa and target_noise_amplitude_upa, and moving only one of them shifts the
# target's tonal/broadband balance and therefore signal_looks_per_frame. Ambient level cannot.
ambient_amplitude_upa = 10 ** (75 / 20)
ambient_spectral_exponent = -1
ambient_noise_model = ColouredNoiseSignal(
    amplitude_upa=ambient_amplitude_upa,
    spectral_exponent=ambient_spectral_exponent,
    duration_s=time_interval.total_seconds(),
    sampling_rate_hz=sampling_rate_hz,
)

signal_model = SyntheticAnthropogenicSignal(
    duration_s=total_duration_s,
    sampling_rate_hz=sampling_rate_hz,
    frame_len=frame_len,
    hop_factor=hop_factor,
    tonal_bandwidth_hz=target_tonal_bandwidth_hz,
    noise_amplitude_upa=target_noise_amplitude_upa,
    noise_spectral_exponent=target_noise_spectral_exponent,
    noise_freq_range_hz=(0.0, sampling_rate_hz / 2),
    tonal_noise_is_constant=True,
    noise_is_constant=True,
)

# The target stays on one side of the array's axis for this whole run, so only that 180 deg
# half-plane needs to be steered at all. There is no mirrored duplicate half, so no need for
# mirror_half_plane. 180 points over 180 deg gives 1 deg/bin.
platform_state_t0 = platform.get_platform_state_at(timesteps[0])
array_endpoints_xy = platform_state_t0.array.state_vector[:2, [0, -1]]
_dx, _dy = array_endpoints_xy[:, 1] - array_endpoints_xy[:, 0]
array_axis_rad = float(np.arctan2(_dy, _dx))

steering_azimuths_rad = np.linspace(array_axis_rad - np.pi, array_axis_rad, 360, endpoint=False)
fmin = 150.0
fmax = 250.0

beamformer = MinimumVarianceDistortionlessResponseBeamformer(
    sampling_rate_hz=sampling_rate_hz,
    fmin=fmin,
    fmax=fmax,
)
steering_calculator = SteeringCalculator(
    ssp=ssp,
    steering_azimuths_rad=steering_azimuths_rad,
)

simulator = ContinuousSTFTPassiveSonarArraySimulator(
    platform=platform,
    propagation_model=propagation_model,
    signal_models=[signal_model],
    noise_model=ambient_noise_model,
    beamformer=beamformer,
    steering_calculator=steering_calculator,
    ground_truth_paths=[target_truth],
    fade_in_ms=fade_in_ms,
)

# %%
# Collecting Beamformed Data and Ground-Truth Bearings
# -------------------------------------------------------
#
# :class:`~.PassiveSonarDetector` doesn't retain raw frames after detecting (see
# :func:`~bluepebble.detector.metrics.sweep_detection_parameter`'s docstring), so raw
# beamformed data is collected directly from ``simulator.sensor_data_gen()`` instead,
# bypassing the detector entirely at this stage.

beamformed_data = []
bearing_states = []

for timestamp, sensor_data_set in simulator.sensor_data_gen():
    (sensor_data,) = sensor_data_set
    beamformed_data.append(sensor_data.beamformed_data)

    platform_state = platform.get_platform_state_at(timestamp)
    assert platform_state is not None
    ref_sensor_position = np.mean(platform_state.array.state_vector, axis=1)

    target_state = next(s for s in target_truth.states if s.timestamp == timestamp)
    target_xy = np.array([target_state.state_vector[0], target_state.state_vector[2]])
    relative_position = target_xy - ref_sensor_position[:2]
    bearing = np.arctan2(relative_position[1], relative_position[0])
    bearing_states.append(GroundTruthState(np.array([bearing]), timestamp=timestamp))

relative_bearing_truth = GroundTruthPath(bearing_states)
num_frames = beamformed_data[0].shape[1]

# %%
# Estimating snr_linear From the Simulated Scenario
# ------------------------------------------------------
#
# The theoretical model needs a single ``snr_linear`` value; the simulator produces a received
# power that varies scan to scan.
# :func:`~bluepebble.detector.metrics.snr_linear_from_ground_truth_bearing` reads it off at each
# timestep. Averaging across the whole run would blend two different things together: the
# fluctuation models' own assumed look-to-look randomness, and genuine physical drift in received
# power as target range and geometry change over the scenario. Taking the single reading at the
# mid-scenario timestep instead keeps ``snr_linear`` tied to one real, roughly stationary moment --
# a fairer match to what the fluctuation models actually assume -- at the cost of being one noisy
# sample rather than an average. The per-timestep spread printed below shows how much that choice
# of timestep could have mattered.

validation_guard_bins = 6
snr_estimates = np.array(
    [
        snr_linear_from_ground_truth_bearing(
            data, state.state_vector[0], steering_azimuths_rad, validation_guard_bins
        )[0]
        for data, state in zip(beamformed_data, bearing_states, strict=True)
    ]
)
mid_step = num_steps // 2
snr_linear = float(snr_estimates[mid_step])

print(
    f"snr_linear at mid-scenario timestep {mid_step}: {snr_linear:.3f} "
    f"({10 * np.log10(max(snr_linear, 1e-6)):.1f} dB); full per-timestep spread: "
    f"min={snr_estimates.min():.3f} ({10 * np.log10(max(snr_estimates.min(), 1e-6)):.1f} dB), "
    f"max={snr_estimates.max():.3f} ({10 * np.log10(max(snr_estimates.max(), 1e-6)):.1f} dB)"
)

# %%
# Measuring the Simulator's Own Per-Look Statistics
# ----------------------------------------------------
#
# OS-CFAR's alpha calibration models each per-frame power sample as one unit-mean
# Exponential draw -- correct for narrowband, single-frequency-bin square-law data. This
# simulator does not produce that. :class:`~.MinimumVarianceDistortionlessResponseBeamformer`
# integrates power over EVERY STFT bin between ``fmin`` and ``fmax`` before the detector sees
# a single number, so each per-frame sample is already a sum of ~100 bins' worth of power
# (1 Hz bins over a 100 Hz band at ``sampling_rate_hz=500``, ``nfft=500``). Averaging many
# independent quantities makes the result far smoother than any one of them: measured per-cell
# power fluctuates with a coefficient of variation near 0.07, where the idealised
# Gamma(``num_frames``, 1/``num_frames``) model expects ~0.35. Note that ratio is a property of
# the processing, not of the signal level -- scaling the ambient up or down leaves it, and
# therefore K, untouched.
#
# Calibrating against the idealised spread on data this smooth does not merely bias Pfa, it
# breaks the detector's Pfa control outright. The whole noise-only population sits in a narrow
# band around a CUT/noise ratio of 1.0, so a threshold set for the wide assumed spread admits
# nothing at all, until it crosses that band and abruptly admits everything.
#
# :func:`~bluepebble.detector.metrics.estimate_effective_looks_per_frame` reads the true
# per-frame degrees of freedom (K) back off the simulator's own output, and every detector
# below is calibrated with it. Cells near the target are masked out first -- the estimate
# describes noise, so target energy must not contribute to it. The same
# ``validation_guard_bins`` used for ``snr_linear`` above defines the exclusion zone, and the
# resulting mask is reused for the raw-exceedance measurement further down.
#
# Note K lands well below the ~100 nominal in-band bins: the STFT's Hann window correlates
# neighbouring frequency bins, so they do not contribute independently. That gap is exactly
# why K is measured rather than computed from the bin count.

target_bin_per_scan = np.array(
    [
        int(
            np.argmin(
                np.abs(np.angle(np.exp(1j * (steering_azimuths_rad - state.state_vector[0]))))
            )
        )
        for state in bearing_states
    ]
)
noise_only_mask = np.ones((num_steps, len(steering_azimuths_rad)), dtype=bool)
for t_idx, bin_idx in enumerate(target_bin_per_scan):
    lo = max(0, bin_idx - validation_guard_bins)
    hi = min(len(steering_azimuths_rad), bin_idx + validation_guard_bins + 1)
    noise_only_mask[t_idx, lo:hi] = False

effective_looks = estimate_effective_looks_per_frame(beamformed_data, noise_only_mask)

_cell_power = np.array([np.mean(np.abs(d) ** 2 if np.iscomplexobj(d) else d, axis=1)
                        for d in beamformed_data])
_measured_cv = float(np.median(np.nanstd(np.where(noise_only_mask, _cell_power, np.nan), axis=0,
                                         ddof=1)
                               / np.nanmean(np.where(noise_only_mask, _cell_power, np.nan),
                                            axis=0)))
print(
    f"num_frames (M): {num_frames}; measured noise-only per-cell CV: {_measured_cv:.4f} "
    f"vs idealised Gamma(M,1/M) CV {1 / np.sqrt(num_frames):.4f}\n"
    f"effective_looks_per_frame (K_n): {effective_looks:.1f} "
    f"(total per-cell DOF K_n*M = {effective_looks * num_frames:.0f})"
)

# %%
# How Much of the Band the Target Actually Occupies
# ----------------------------------------------------
#
# ``effective_looks_per_frame`` above describes the NOISE, which fills the processed band. The
# target does not. This scenario's source is a 200 Hz tonal of ``target_tonal_bandwidth_hz`` = 1
# Hz, and the STFT resolves 1 Hz per bin, so the tonal lands in a couple of bins once the Hann
# window's leakage is accounted for -- against roughly 100 bins of noise. Taking the source
# spectrum's participation ratio (``1 / sum(p_i ** 2)`` over in-band bin power fractions) gives
# 2.06 effective bins, with 99.6% of in-band target energy in three bins and the tonal sitting
# 24.4 dB above the target's own broadband component.
#
# That distinction matters for Pd, and only for Pd. H0 contains no signal, so ``target_pfa``
# calibration is untouched by it. But under H1 the cell is a MIXTURE: a couple of looks carrying
# signal plus noise, and the rest carrying noise alone. Energy spread over many looks
# self-averages; energy confined to two does not, so a tonal fluctuates considerably more
# look-to-look than a band-filling target of the same mean SNR. Assuming the target fills the
# band would predict a sharper Pd knee than this source can deliver.
#
# The mean is unaffected either way: ``snr_linear`` was read off the already-integrated cell, so
# the loss from integrating 100 bins of noise around a 2-bin signal is already priced in.
# Setting ``signal_looks_per_frame`` moves variance only. For a genuinely broadband target
# (cavitation, machinery noise filling the band) leave it at ``None``, which means K_s = K_n.

signal_looks = 2.06

# %%
# Empirical Sweep
# ----------------
#
# ``target_pfa`` is swept for a single :class:`~.OSCFARDetector`. `num_frames`` is greater than 1,
# so alpha calibration falls back to Monte Carlo, passing a seeded ``rng`` keeps that reproducible
# across the sweep's deep-copied detector clones. ``effective_looks_per_frame`` is the K measured
# above, without which this detector's achieved Pfa collapses to zero across most of the sweep.

num_guard_cells = 2
num_training_cells = 10
num_training_total = 2 * num_training_cells
rank = 15
peak_distance = 1

pfa_values = np.geomspace(1e-4, 1.0, num=20)

detector = OSCFARDetector(
    num_guard_cells=num_guard_cells,
    num_training_cells=num_training_cells,
    rank=rank,
    target_pfa=float(pfa_values[0]),
    peak_distance=peak_distance,
    circular=False,  # steering_azimuths_rad is now a 180 deg arc, not a closed 360 deg loop.
    rng=np.random.default_rng(seed + 2),
    effective_looks_per_frame=effective_looks,
)

spec = SweepSpec(
    detector=detector,
    param_name="target_pfa",
    param_values=pfa_values,
    label="Empirical",
)

[empirical_result] = sweep_detection_parameter(
    beamformed_data=beamformed_data,
    sweep_specs=[spec],
    ground_truth_paths=[relative_bearing_truth],
    steering_azimuths_rad=steering_azimuths_rad,
    association_threshold_rad=np.deg2rad(3.0),
)

# %%
# Bearing-Time Record: SNR
# --------------------------
#
# A bearing-time record of the per-beam SNR ``detector.snr_map()`` computes from its own local
# noise-floor estimate (the same quantity its ``detect()`` thresholds against, not the
# ground-truth-bearing readings used above and below) gives a visual check on the scenario
# before digging further into detection statistics: is there a visible target track, and does
# it behave the way the geometry above suggests? ``target_pfa`` doesn't affect ``snr_map()`` (it
# only sets the detection threshold, not the SNR itself), so the ``detector`` built above for
# the sweep is reused as-is. ``sweep_detection_parameter`` deep-copies it per swept value and
# never mutates the original.

snr_map = np.array([detector.snr_map(data) for data in beamformed_data])

fig_btr = plot_btr(
    data=snr_map,
    timesteps=timesteps,
    steering_azimuths=np.rad2deg(steering_azimuths_rad),
    truths=[relative_bearing_truth],
    colorscale="Viridis",
).update_layout(
    title="Bearing-Time Record: SNR",
    template="plotly_white",
    autosize=True,
    width=None,
    height=None,
).show()

# %%
# Theoretical Curves
# -------------------
#
# Both fluctuation models implemented in :mod:`bluepebble.detector.fluctuation_models` are
# computed at the same estimated ``snr_linear``, not because the target is expected to match
# either one exactly, but so any gap can be checked against both rather than just the default.

rayleigh_pd = os_cfar_roc(
    pfa_values,
    num_training_total=num_training_total,
    rank=rank,
    num_frames=num_frames,
    snr_linear=snr_linear,
    num_trials=1_000_000,
    rng=np.random.default_rng(seed + 1),
    effective_looks_per_frame=effective_looks,
    signal_looks_per_frame=signal_looks,
)
rayleigh_result = SweepResult.from_theoretical_roc(
    pfa=pfa_values, pd=rayleigh_pd, label="Rayleigh-fading"
)

nonfluct_pd = os_cfar_roc(
    pfa_values,
    num_training_total=num_training_total,
    rank=rank,
    num_frames=num_frames,
    snr_linear=snr_linear,
    model=NonFluctuating(),
    num_trials=1_000_000,
    rng=np.random.default_rng(seed + 3),
    effective_looks_per_frame=effective_looks,
    signal_looks_per_frame=signal_looks,
)
nonfluct_result = SweepResult.from_theoretical_roc(
    pfa=pfa_values, pd=nonfluct_pd, label="Non-Fluctuating"
)

# %%
# Raw-Exceedance Achieved Pfa
# -----------------------------
#
# ``empirical_result.fpr`` below measures the rate of DISTINCT, peak-consolidated detections --
# :meth:`~.OSCFARDetector.detect` collapses every contiguous run of CFAR-passing cells down to
# its local maxima via ``scipy.signal.find_peaks`` before counting false positives. Real
# beamformed data is spatially correlated across adjacent bearing bins (finite beamwidth), so
# even a very permissive threshold produces a small, roughly constant number of correlated
# "blobs" rather than many independent exceedances -- which is why achieved Pfa plateaus well
# below 1 even as ``target_pfa`` approaches 1.0. This computes the statistic theoretical Pfa
# actually models instead: the raw per-cell CFAR exceedance rate, direct from ``snr_map`` (already
# computed above), bypassing peak consolidation entirely. It reuses ``noise_only_mask`` from the
# K-estimation section, which excludes cells within ``validation_guard_bins`` of the true bearing
# per scan, so real target energy leaking into adjacent bins isn't miscounted as a false alarm.
# Alpha is calibrated with the same seed for every swept ``target_pfa`` (the reference-cell
# distribution doesn't depend on it), the same common-random-numbers trick documented on
# :func:`~bluepebble.detector.metrics.os_cfar_roc`, and with the same measured
# ``effective_looks_per_frame`` the detectors use, so this measurement and the detectors are
# thresholding on the same model.
#
# This raw-exceedance column is the direct check on whether the K correction worked: it should
# now track ``target_pfa`` reasonably closely, where an idealised (K=1) calibration produces
# zero exceedances across most of the sweep. It will not track perfectly. K was measured from
# the spread of each beam ACROSS scans, but the detector estimates its noise floor from
# neighbouring beams WITHIN a scan, and finite beamwidth correlates those neighbours -- adding
# variance to the noise-floor estimate that K does not describe.

raw_exceedance_pfa = np.array(
    [
        float(
            np.mean(
                snr_map[noise_only_mask]
                > 10
                * np.log10(
                    calibrate_os_cfar_alpha_mc(
                        pfa,
                        num_training_total,
                        rank,
                        num_frames,
                        num_trials=200_000,
                        rng=np.random.default_rng(seed + 8),
                        effective_looks_per_frame=effective_looks,
                    )
                )
            )
        )
        for pfa in pfa_values
    ]
)

# %%
# Comparing Theoretical and Empirical Performance
# ---------------------------------------------------
#
# With the ambient set so the target sits near ``snr_linear`` = 0.35, all three curves now have
# structure worth reading. Empirical Pd runs 0.38 to 1.00 across the sweep, Rayleigh-fading 0.67
# to 1.00, non-fluctuating 0.74 to 1.00. Crucially the two fluctuation models finally SEPARATE
# -- by about 7 points at the strict end (0.68 vs 0.75 at ``target_pfa`` = 1e-4) -- so the
# choice of target model is now a question the plot can actually speak to. At the original
# quieter ambient every one of these curves sat at 1.00 and the comparison was vacuous.
#
# ``signal_looks_per_frame`` matters here too, and by a comparable amount. Modelling the target
# as the tonal it is (K_s = 2.06) rather than as filling the band (K_s = K_n) lowers Rayleigh Pd
# from 0.74 to 0.68 at ``target_pfa`` = 1e-4, and from 0.89 to 0.82 at 1e-3: a 6 to 7 point
# correction, the same order as the gap between the two fluctuation models themselves. Getting
# the target's bandwidth wrong would therefore mislead about as badly as picking the wrong
# fluctuation model.
#
# The ROC plot below's x-axis is deliberately titled "Achieved Pfa", not "Target Pfa": each
# curve is plotted against its OWN actual false-positive rate at each point, not the nominal
# ``target_pfa`` used to calibrate its threshold. For the theoretical curves those coincide
# exactly, by construction of the closed-form/Monte Carlo model. For the empirical curve they do
# not -- ``empirical_result.fpr`` is the peak-consolidated achieved Pfa from the table above,
# which plateaus well below ``target_pfa`` for the reason discussed there.
#
# Empirical Pd sits below both theoretical curves, and that gap is a property of the scenario
# rather than a modelling error. Empirical Pd is measured across every scan of the run, whose
# per-scan ``snr_linear`` has a median of 0.21 and drops to zero or below on 4 of the 60 scans,
# whereas the theoretical curves are evaluated at the single mid-scenario reading of 0.35.
# Evaluating the same theoretical model at each scan's OWN ``snr_linear`` and averaging is the
# like-for-like comparison: that gives Pd = 0.30 against the empirical 0.38, agreeing to within
# about 0.08 with theory now slightly pessimistic, and attributing the shortfall to the 43 of 60
# scans theory itself rates as near-undetectable.
#
# Two cautions on that reading. ``snr_linear_from_ground_truth_bearing`` returns
# ``power / noise - 1``, which goes negative when the target is below its local noise floor, so
# those scans need clamping at zero before being fed to a fluctuation model. And the underlying
# mismatch -- one theoretical SNR against an empirical average over a wide range of them -- is
# not fixed by changing the level, only moved. Holding the target at roughly constant range
# would address it properly.

headers = [
    "Target Pfa",
    "Achieved Pfa (empirical)",
    "Achieved Pfa (raw exceedance)",
    "Pd (empirical)",
    "Pd (Rayleigh)",
    "Pd (non-fluct.)",
]
rows = [
    [f"{pfa:.4f}", f"{fpr:.4f}", f"{raw_pfa:.4f}", f"{tpr:.4f}", f"{pd_r:.4f}", f"{pd_nf:.4f}"]
    for pfa, fpr, raw_pfa, tpr, pd_r, pd_nf in zip(
        pfa_values,
        empirical_result.fpr,
        raw_exceedance_pfa,
        empirical_result.tpr,
        rayleigh_pd,
        nonfluct_pd,
        strict=True,
    )
]

table = [headers, *rows]
col_widths = [max(len(row[i]) for row in table) for i in range(len(headers))]
separator = "-+-".join("-" * w for w in col_widths)

print(" | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)))
print(separator)
for row in rows:
    print(" | ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(row)))

fig_roc = plot_roc([empirical_result, rayleigh_result, nonfluct_result]).update_layout(
    title="Empirical vs. Theoretical ROC: OS-CFAR on Simulated Acoustic Data",
    template="plotly_white",
    autosize=True,
    width=None,
    height=None,
).update_xaxes(title="Achieved Pfa").update_yaxes(title="Pd").show()

# %%
# Cumulative Probability of Detection
# --------------------------------------
#
# Every Pd above is a single-scan number: the chance of detecting the target on any one look.
# An operator watching this contact for the full scenario cares whether it gets detected AT ALL
# over many looks, not the odds on any single glimpse. This section computes that directly, at
# one fixed ``target_pfa``, set to ``0.001``, three ways.
#
# Be aware that at this scenario's SNR the buildup is fast: single-scan Pd is already ~0.82, so
# all three curves reach 1 within two or three looks and the plot mostly demonstrates that
# compounding works rather than discriminating between the models. Slowing it down needs a
# single-scan Pd nearer 0.2, and at ``snr_linear`` = 0.35 no reachable ``target_pfa`` delivers
# that -- Pd is still 0.68 at the strictest point of the sweep above, and calibrating below
# ~5e-4 would exceed what ``mc_trials`` = 200_000 can resolve (see
# :func:`~.algorithms.calibrate_os_cfar_alpha_mc`'s ~100/target_pfa rule of thumb). A genuinely
# gradual buildup would need a weaker target rather than a stricter threshold.
#
# 1. **Rayleigh-fading (theoretical)**: per-scan Pd from
#    :class:`~.fluctuation_models.RayleighFluctuation` at the mean ``snr_linear``, combined
#    across scans as ``1 - prod(1 - Pd_i)`` assuming independent looks.
# 2. **Non-fluctuating (theoretical)**: the same compounding formula, using
#    :class:`~.fluctuation_models.NonFluctuating` instead -- the same pairing of fluctuation
#    models used for the theoretical ROC curves above, so the two sections stay directly
#    comparable.
# 3. **Empirical (single realisation)**: whether the target was actually detected, scan by scan,
#    in this one simulated run -- not a probability, just a record of what happened, checked
#    against where curves 1-2 say detection should become likely.

target_pfa_cum = 0.001
cum_num_trials = 1_000_000
cum_rng = np.random.default_rng(seed + 4)

alpha_cum = calibrate_os_cfar_alpha_mc(
    target_pfa_cum,
    num_training_total,
    rank,
    num_frames,
    num_trials=200_000,
    rng=cum_rng,
    effective_looks_per_frame=effective_looks,
)

pd_single_rayleigh = RayleighFluctuation().os_cfar_pd(
    alpha_cum,
    num_training_total,
    rank,
    num_frames,
    snr_linear,
    num_trials=cum_num_trials,
    rng=np.random.default_rng(seed + 5),
    effective_looks_per_frame=effective_looks,
    signal_looks_per_frame=signal_looks,
)
pd_single_nonfluct = NonFluctuating().os_cfar_pd(
    alpha_cum,
    num_training_total,
    rank,
    num_frames,
    snr_linear,
    num_trials=cum_num_trials,
    rng=np.random.default_rng(seed + 7),
    effective_looks_per_frame=effective_looks,
    signal_looks_per_frame=signal_looks,
)

look_number = np.arange(1, num_steps + 1)
pd_cumulative_rayleigh = 1.0 - (1.0 - pd_single_rayleigh) ** look_number
pd_cumulative_nonfluct = 1.0 - (1.0 - pd_single_nonfluct) ** look_number

# All three curves now share one noise model: ``alpha_cum`` and both theoretical Pd values use
# the measured ``effective_looks_per_frame``, matching the detector below, and both Pd values
# additionally use ``signal_looks_per_frame`` for the tonal. Threshold and Pd therefore describe
# the same detector on the same data, so the comparison is like for like.
detector_cum = OSCFARDetector(
    num_guard_cells=num_guard_cells,
    num_training_cells=num_training_cells,
    rank=rank,
    target_pfa=target_pfa_cum,
    peak_distance=peak_distance,
    circular=False,
    rng=np.random.default_rng(seed + 6),
    effective_looks_per_frame=effective_looks,
)
detected_this_scan = np.zeros(num_steps, dtype=bool)
for t_idx, (data, state) in enumerate(zip(beamformed_data, bearing_states, strict=True)):
    raw_detections = detector_cum.detect(data)
    if raw_detections.size == 0:
        continue
    detected_bearings = steering_azimuths_rad[raw_detections[:, 0].astype(int)]
    angular_diff = np.angle(np.exp(1j * (detected_bearings - state.state_vector[0])))
    detected_this_scan[t_idx] = bool(np.any(np.abs(angular_diff) <= np.deg2rad(3.0)))
empirical_ever_detected = np.maximum.accumulate(detected_this_scan.astype(float))

fig_cumulative = go.Figure()
fig_cumulative.add_trace(
    go.Scatter(
        x=timesteps,
        y=empirical_ever_detected,
        mode="lines",
        name="Empirical",
        line=dict(width=2, dash="dot"),
    )
)
fig_cumulative.add_trace(
    go.Scatter(
        x=timesteps,
        y=pd_cumulative_rayleigh,
        mode="lines",
        name="Rayleigh-fading",
        line=dict(width=2),
    )
)
fig_cumulative.add_trace(
    go.Scatter(
        x=timesteps,
        y=pd_cumulative_nonfluct,
        mode="lines",
        name="Non-fluctuating",
        line=dict(width=2, dash="dash"),
    )
)
fig_cumulative.update_layout(
    title="Cumulative Probability of Detection",
    template="plotly_white",
    xaxis_title="Time (MM:SS)",
    xaxis=dict(tickformat="%M:%S"),
    yaxis_title="Cumulative Pd",
    yaxis_range=[0.0, 1.05],
    autosize=True,
    width=None,
    height=None,
).show()

