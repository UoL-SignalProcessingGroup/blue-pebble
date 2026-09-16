"""
===================================
Calibrating CFAR Detection to Noise
===================================

A CFAR detector's ``target_pfa`` is only as accurate as its noise model. The closed-form
calibration assumes the cell under test and its reference cells are independent Gamma
variables with a known number of looks. Beamformer output is neither independent (neighbouring
beams share noise) nor Gamma-tailed, so the false-alarm rate a detector achieves can differ from
the one requested by orders of magnitude.

This example measures that gap on noise-only delay-and-sum output, then closes it with
:func:`~bluepebble.detector.calibrate_from_noise`: the detector's own cell-to-noise ratio is
measured on noise-only scans, alpha is read from its empirical distribution, and a generalised
Pareto tail carries it below the rates the scans can observe. Every threshold is judged on
held-out scans, so the comparison is between what each method promises and what it delivers.

Three thresholds are compared for each beamformer domain and ambient spectrum:

- **Nominal**: the Gamma model with one look per frame.
- **Effective looks**: the Gamma model with looks estimated from the same noise by
  :func:`~bluepebble.detector.estimate_effective_looks_per_frame`.
- **Noise calibration**: :func:`~bluepebble.detector.calibrate_from_noise`.
"""  # noqa: D205, D212, D400, D415
# sphinx_gallery_skip_execution = True

# %%
# Imports
# -------

import pickle
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import bluepebble
from bluepebble.detector import (
    CACFARDetector,
    calibrate_from_noise,
    cell_noise_ratios,
    estimate_effective_looks_per_frame,
)
from bluepebble.signal.random import ColouredNoiseSignal, WhiteNoiseSignal
from bluepebble.sigproc import DelayAndSumBeamformer

# %%
# Configuration
# -------------
#
# A 100-element line array at 1 m spacing is steered to 360 bearings. Each scan is one second of
# ambient noise. Half of each case's scans calibrate; the other half are held out to measure the
# false-alarm rate each threshold actually achieves.

bluepebble.set_seed(7)
sampling_rate_hz = 1500.0
scan_duration_s = 1.0
num_sensors = 100
sensor_spacing_m = 1.0
sound_speed_ms = 1500.0
num_scans_per_half = 1500
steering_azimuths_rad = np.linspace(-np.pi, np.pi, 360, endpoint=False)
sensor_positions_m = np.arange(num_sensors) * sensor_spacing_m
steering_delays_s = np.outer(np.cos(steering_azimuths_rad), sensor_positions_m) / sound_speed_ms
target_pfas = np.logspace(-1, -5, 13)

cases = {
    ("frequency", "white"): DelayAndSumBeamformer(
        sampling_rate_hz=sampling_rate_hz, domain="frequency"
    ),
    ("frequency", "pink"): DelayAndSumBeamformer(
        sampling_rate_hz=sampling_rate_hz, domain="frequency"
    ),
    ("broadband_power", "white"): DelayAndSumBeamformer(
        sampling_rate_hz=sampling_rate_hz,
        domain="broadband_power",
        nfft=256,
        overlap=128,
        fmin=50.0,
        fmax=700.0,
    ),
    ("broadband_power", "pink"): DelayAndSumBeamformer(
        sampling_rate_hz=sampling_rate_hz,
        domain="broadband_power",
        nfft=256,
        overlap=128,
        fmin=50.0,
        fmax=700.0,
    ),
}


def make_detector(**kwargs):
    """CA-CFAR detector used throughout; consolidation off so every crossing counts."""
    return CACFARDetector(
        num_guard_cells=3,
        num_training_cells=10,
        target_pfa=1e-2,
        consolidate_peaks=False,
        **kwargs,
    )


# %%
# Noise-only scans
# ----------------
#
# In an operational set-up the scans come from the same simulator as the scenario, built without
# ``ground_truth_paths``, via :func:`~bluepebble.detector.beamformed_scans_from_sensor_data`. Here
# the array and beamformer are driven directly with ambient noise to keep the example fast.
# Beamforming is the slow part (about 0.2 s per scan in the broadband-power domain), so each
# case's calibration and held-out ratios are cached.

cache_dir = Path(tempfile.gettempdir()) / "bluepebble_cfar_noise_calibration"
cache_dir.mkdir(exist_ok=True)


def noise_signal(colour):
    """White or pink (1/f) ambient noise with unit amplitude."""
    if colour == "white":
        return WhiteNoiseSignal(
            amplitude_upa=1.0, duration_s=scan_duration_s, sampling_rate_hz=sampling_rate_hz
        )
    return ColouredNoiseSignal(
        amplitude_upa=1.0,
        spectral_exponent=-1.0,
        duration_s=scan_duration_s,
        sampling_rate_hz=sampling_rate_hz,
    )


def noise_scans(label, colour, beamformer, beam_power):
    """Yield noise-only beamformer scans, recording each scan's per-beam power."""
    noise = noise_signal(colour)
    num_samples = int(sampling_rate_hz * scan_duration_s)
    tic = time.perf_counter()
    for scan_index in range(num_scans_per_half):
        scan = beamformer.beamform(noise.generate(num_sensors, num_samples), steering_delays_s)
        power = np.abs(scan) ** 2 if np.iscomplexobj(scan) else scan
        beam_power.append(power.mean(axis=1))
        if (scan_index + 1) % 500 == 0:
            print(f"  {label}: {scan_index + 1} scans ({time.perf_counter() - tic:.0f} s)")
        yield scan


def simulate_case(domain, colour, beamformer):
    """Return the calibration and per-beam powers from one half, held-out ratios from the other."""
    path = cache_dir / f"{domain}_{colour}_{num_scans_per_half}.pkl"
    if path.exists():
        with path.open("rb") as file:
            return pickle.load(file)

    beam_power = []
    calibration = calibrate_from_noise(
        make_detector(),
        noise_scans(f"{domain}/{colour} calibration", colour, beamformer, beam_power),
    )
    held_out_ratios = cell_noise_ratios(
        make_detector(), noise_scans(f"{domain}/{colour} held out", colour, beamformer, [])
    )
    result = (calibration, np.asarray(beam_power), held_out_ratios)
    with path.open("wb") as file:
        pickle.dump(result, file)
    return result


# %%
# Thresholds and achieved false-alarm rates
# -----------------------------------------
#
# Achieved Pfa is the fraction of held-out cells whose power-to-noise ratio exceeds each
# threshold's alpha, which is exactly what unconsolidated ``detect()`` counts.

results = {}
for (domain, colour), beamformer in cases.items():
    print(f"{domain} / {colour} ambient noise")
    calibration, calibration_power, held_out_ratios = simulate_case(domain, colour, beamformer)
    num_frames = calibration.num_frames

    looks = estimate_effective_looks_per_frame([power[:, None] for power in calibration_power])
    effective_looks_per_frame = looks / num_frames

    thresholds = {
        "Nominal": make_detector(),
        "Effective looks": make_detector(effective_looks_per_frame=effective_looks_per_frame),
        "Noise calibration": make_detector(noise_calibration=calibration),
    }
    achieved = {}
    for name, detector in thresholds.items():
        rates = []
        for pfa in target_pfas:
            detector.target_pfa = float(pfa)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                alpha = detector._alpha_for(num_frames)
            rates.append(np.mean(held_out_ratios > alpha))
        achieved[name] = np.asarray(rates)
    results[(domain, colour)] = (achieved, held_out_ratios.size, effective_looks_per_frame)

# %%
# Results
# -------
#
# Points on the dashed diagonal are thresholds that deliver the requested rate. Held-out cells
# resolve rates down to roughly ``1 / num_cells``; below that an achieved rate of zero is plotted
# at the resolution floor.

figure = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=[f"{domain} DAS, {colour} noise" for domain, colour in cases],
    horizontal_spacing=0.1,
    vertical_spacing=0.14,
)
colours = {"Nominal": "#9e9e9e", "Effective looks": "#e69f00", "Noise calibration": "#0072b2"}
for index, (achieved, num_cells, _) in enumerate(results.values()):
    row, col = divmod(index, 2)
    floor = 1.0 / num_cells
    figure.add_trace(
        go.Scatter(
            x=target_pfas,
            y=target_pfas,
            mode="lines",
            line={"color": "black", "dash": "dash", "width": 1},
            name="Achieved = target",
            showlegend=index == 0,
        ),
        row=row + 1,
        col=col + 1,
    )
    for name, rates in achieved.items():
        figure.add_trace(
            go.Scatter(
                x=target_pfas,
                y=np.maximum(rates, floor),
                mode="lines+markers",
                line={"color": colours[name]},
                name=name,
                showlegend=index == 0,
            ),
            row=row + 1,
            col=col + 1,
        )
figure.update_xaxes(type="log", title="Target Pfa", autorange="reversed")
figure.update_yaxes(type="log", title="Achieved Pfa (held out)")
figure.update_layout(width=950, height=800, template="plotly_white")
figure.write_html(cache_dir / "achieved_vs_target_pfa.html")

# %%
# Summary table
# -------------
#
# Ratio of achieved to target Pfa. The calibrated threshold should stay within 0.8-1.25x down to
# 1e-3 and 0.67-1.5x at 1e-4.

for (domain, colour), (achieved, num_cells, effective_looks) in results.items():
    print(f"\n{domain} / {colour}  ({num_cells} held-out cells, fitted K = {effective_looks:.3g})")
    print("  target   " + "  ".join(f"{name:>17s}" for name in achieved))
    for pfa_index, pfa in enumerate(target_pfas):
        if pfa < 5.0 / num_cells:
            continue
        cells = "  ".join(f"{achieved[name][pfa_index] / pfa:17.2f}" for name in achieved)
        print(f"  {pfa:7.1e}  {cells}")
