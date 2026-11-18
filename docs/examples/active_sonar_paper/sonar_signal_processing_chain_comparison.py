r"""
========================================
Comparing Sonar Signal Processing Chains
========================================

This script runs the paper's bistatic scenario (``paper_scenario.py``) over many pings through
the whole chain, from the propagation model to Stone Soup tracks, and measures how each choice
in the chain changes the tracks. Each experiment changes one stage of a baseline chain and
holds the rest fixed, so any change in the tracks comes from that stage. Every experiment runs
over both seabeds: flat, the control, and the wedge, which slopes across the line of sight.

- **E1, propagation**: every Bellhop configuration, at the settings
  ``bellhop_propagation_model_configuration.py`` saved.
- **E4, beamformer**: delay-and-sum against MVDR, with two amounts of diagonal loading.
- **E5, detector**: CA-CFAR against OS-CFAR, over a range of false alarm probabilities, with
  calibrated and nominal (textbook) thresholds.
- **E6, arrays and post-detection**: which arrays feed the tracker, and the flank ghost
  removal and towed-array ambiguity resolution on and off.

The baseline chain is the configuration script's reference, 3D all, with frequency-domain
delay-and-sum and a calibrated OS-CFAR at a false alarm probability of :math:`10^{-7}`, every
array and every post-detection step.

Each ping is simulated once per propagation configuration, without beamforming, and every
chain that shares that configuration processes the same sensor data. So E4 to E6 cost one
simulation between them; only E1 pays for new propagation runs.

The chain for each array, then for all of them:

1. **Beamform** the sensors' data (delay-and-sum or MVDR, both keeping every sample).
2. **Detect**: matched-filter each beam and run the CFAR detector along range
   (:class:`~bluepebble.detector.ActiveSonarDetector`).
3. **Group** detections across beams into one per echo, with the bearing refined between beams.
4. **Remove flank ghosts**: a flank detection that the other flank hears 10 dB louder at the
   mirrored bearing is dropped.
5. **Convert** each bistatic range sum to a range from the receiving array.
6. **Resolve the towed array's ambiguity** against the other arrays' detections.
7. **Track**: one extended Kalman filter tracker fuses every array's measurements.

Each chain is scored against the truth: GOSPA (localisation, missed and false), false and
missed tracks, and each array's probability of detection and false measurements.

Assumptions
-----------
- The scenario's assumptions (``paper_scenario.py``): Munk water, a lossless reflective
  seabed, point targets, white noise, no reverberation, direct blast or Doppler.
- The steering, the delay-to-range conversion and the bistatic conversion use one sound speed
  and straight-line, horizontal paths. In refracting water over a sloping seabed that biases
  the measurements, as it would a real sonar's.
- The post-detection steps are the bistatic tracking example's, with its thresholds.
"""  # noqa: D205, D212, D400, D415

# %%
# Imports
# -------

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import paper_scenario as scenario
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from bluepebble.detector import ActiveSonarDetector, NoiseCalibrator, matched_filter_power
from bluepebble.sigproc import (
    DelayAndSumBeamformer,
    MinimumVarianceDistortionlessResponseBeamformer,
)
from bluepebble.types import ActiveSonarSensorData

# %%
# Configuration
# -------------

# Set to "plotly" to save interactive HTML figures, or "pdf" to save static PDF figures. Either
# way, figures are written to a folder named after this script.
PLOTTING_MODE = "pdf"
FIGURE_DIR = Path(__file__).with_name(f"{Path(__file__).stem}_figures")
RESULTS_PATH = Path(__file__).with_name(f"{Path(__file__).stem}_results.json")
MEASUREMENTS_PATH = Path(__file__).with_name(f"{Path(__file__).stem}_measurements.json")
SETTINGS_PATH = Path(__file__).with_name("bellhop_propagation_model_configuration_settings.json")

# Each run is the whole scenario again with its own noise, set by a seed: every array's noise
# seed is its entry in paper_scenario.NOISE_SEEDS plus the run's seed times RUN_SEED_STEP.
# The tracker tuning script tunes on the tuning seeds' runs and reports on the holdout seeds'
# runs, which the tuning never sees. Fewer pings or seeds make a quick check.
NUM_PINGS = scenario.NUM_PINGS
PING_TIMES = scenario.PING_TIMES[:NUM_PINGS]
TUNING_SEEDS = (1, 2)
HOLDOUT_SEEDS = (3, 4)
RUN_SEED_STEP = 100
RUNS = [{"seed": seed, "role": "tuning"} for seed in TUNING_SEEDS] + [
    {"seed": seed, "role": "holdout"} for seed in HOLDOUT_SEEDS
]
NUM_RUNS = len(RUNS)

# The experiments to run, by name (see EXPERIMENTS below).
RUN_EXPERIMENTS = ("E1", "E4", "E5", "E6")

# Scoring (GOSPA's cut-off and the settling pings are in paper_scenario.py).
GOSPA_C_M = scenario.GOSPA_C_M
GOSPA_P = scenario.GOSPA_P
SETTLE_PINGS = scenario.SETTLE_PINGS
DETECTION_GATE_M = 100.0  # a measurement within this of a target counts as detecting it


def save_figure(figure: go.Figure, name: str) -> None:
    """Write ``figure`` to ``FIGURE_DIR`` in the format set by ``PLOTTING_MODE``."""
    FIGURE_DIR.mkdir(exist_ok=True)
    if PLOTTING_MODE == "plotly":
        figure.write_html(FIGURE_DIR / f"{name}.html", include_plotlyjs="directory")
    elif PLOTTING_MODE == "pdf":
        figure.write_image(FIGURE_DIR / f"{name}.pdf")
    else:
        raise ValueError(f'PLOTTING_MODE must be "plotly" or "pdf", not {PLOTTING_MODE!r}')


# %%
# Propagation settings
# --------------------
#
# The converged settings of every propagation configuration over every seabed, from the
# configuration script. The baseline detector must be the one those settings converged with,
# so the two scripts are checked against each other.

settings = json.loads(SETTINGS_PATH.read_text())
saved_detector = settings["detector"]
baseline_detector = scenario.cfar_detector()
expected = {
    "class": type(baseline_detector).__name__,
    "target_pfa": baseline_detector.target_pfa,
    "num_guard_cells": baseline_detector.num_guard_cells,
    "num_training_cells": baseline_detector.num_training_cells,
    "rank": getattr(baseline_detector, "rank", None),
    "calibrated": baseline_detector.threshold_factor is None,
}
if saved_detector != expected:
    raise ValueError(
        f"The propagation settings in {SETTINGS_PATH.name} converged with detector "
        f"{saved_detector}, but paper_scenario.cfar_detector() is now {expected}. Rerun "
        "bellhop_propagation_model_configuration.py."
    )
print(f"Propagation settings from {SETTINGS_PATH.name}, created {settings['created']}")

# %%
# Chains and experiments
# ----------------------
#
# A chain is a dictionary of choices. Each experiment is a list of changes to the baseline,
# one stage at a time, and runs over every seabed: flat, the control, over which the
# propagation configurations should agree, and the wedge. A chain that appears in several
# experiments (the baseline appears in all of them) is run once.
#
# Every seabed's 3D all pings are simulated for E1 anyway, so running E4 to E6 over both
# seabeds costs no more propagation, only processing and tracking.

SEABEDS = tuple(settings["scenario"]["seabeds"])
BASELINE = {
    "seabed": None,  # every experiment runs over each of SEABEDS
    "propagation": settings["reference"],
    "beamformer": "DAS",
    "diagonal_loading": None,  # MVDR only
    "detector": "OS",
    "pfa": scenario.TARGET_PFA,
    "calibrated": True,
    "arrays": scenario.ARRAY_NAMES,
    "ghost_removal": True,
    "towed_ambiguity": True,
}
PFAS = (1e-4, 1e-5, 1e-6, 1e-7, 1e-8)
EXPERIMENTS = {
    "E1": [{"propagation": mode} for mode in settings["selected"]],
    "E4": [
        {"beamformer": "DAS"},
        {"beamformer": "MVDR", "diagonal_loading": 1e-3},
        {"beamformer": "MVDR", "diagonal_loading": 1e-1},
    ],
    "E5": [
        {"detector": kind, "pfa": pfa, "calibrated": calibrated}
        for kind in ("OS", "CA")
        for calibrated in (True, False)
        for pfa in PFAS
    ],
    "E6": [
        {"arrays": ("Bow",)},
        {"arrays": ("Bow", "Port", "Starboard")},
        {"arrays": ("Bow", "Towed")},
        {"arrays": scenario.ARRAY_NAMES},
        {"ghost_removal": False},
        {"towed_ambiguity": False},
    ],
}
# MVDR's STFT: 256 samples (32 ms) is longer than any array's travel time across it, and gives
# 249 snapshots in a ping, more than twice the 97 sensors of a flank array.
MVDR_NFFT = 256


def chain_key(chain):
    """Return a hashable key for a chain."""
    return tuple(sorted((k, tuple(v) if isinstance(v, list) else v) for k, v in chain.items()))


def chain_label(chain):
    """Return a short label for a chain: its seabed, then how it differs from the baseline."""
    parts = []
    for key, value in chain.items():
        if value == BASELINE[key] or key in ("seabed", "diagonal_loading"):
            continue
        if key == "arrays":
            parts.append("+".join(value))
        elif key == "pfa":
            parts.append(f"Pfa {value:g}")
        elif key == "calibrated":
            parts.append("nominal")
        elif key in ("ghost_removal", "towed_ambiguity"):
            parts.append(f"no {key.replace('_', ' ')}")
        elif key == "beamformer" and value == "MVDR":
            parts.append(f"MVDR (loading {chain['diagonal_loading']:g})")
        else:
            parts.append(str(value))
    return f"{chain['seabed']}: {', '.join(parts) or 'baseline'}"


chains = {}  # key: chain
experiment_chains = {}  # experiment: [key]
for experiment in RUN_EXPERIMENTS:
    experiment_chains[experiment] = []
    for seabed in SEABEDS:
        for change in EXPERIMENTS[experiment]:
            chain = {**BASELINE, "seabed": seabed, **change}
            if chain["beamformer"] == "DAS":
                chain["diagonal_loading"] = None
            key = chain_key(chain)
            chains.setdefault(key, chain)
            if key not in experiment_chains[experiment]:
                experiment_chains[experiment].append(key)

# Chains that share a propagation configuration and seabed share their simulated pings.
by_propagation = defaultdict(list)
for key, chain in chains.items():
    by_propagation[chain["seabed"], chain["propagation"]].append(key)
print(
    f"{len(chains)} chains over {len(by_propagation)} propagation configurations, "
    f"{NUM_PINGS} pings, {NUM_RUNS} run(s)"
)

# %%
# Scenario
# --------

platform, arrays = scenario.ship(PING_TIMES)
target_paths = scenario.targets(PING_TIMES)
steering = {name: scenario.steering_calculator(name) for name in scenario.ARRAY_NAMES}

# %%
# Beamformers and detectors
# -------------------------
#
# Each beamformer keeps every sample, so a beam's sample :math:`k` is still delay
# :math:`k / f_s`. Each detector is calibrated, when it is in calibrated mode, on a noise-only
# ping processed by its own beamformer: the calibration is tied to everything that shapes the
# noise. One noise-only ping per array and beamformer calibrates every detector that uses it.


def beamformer_key(chain):
    """Return what identifies a chain's beamformer."""
    return chain["beamformer"], chain["diagonal_loading"]


def make_beamformer(key):
    """Return the beamformer a ``beamformer_key`` names."""
    kind, loading = key
    if kind == "DAS":
        return DelayAndSumBeamformer(
            domain="frequency", sampling_rate_hz=scenario.SAMPLING_RATE_HZ
        )
    if kind == "MVDR":
        return MinimumVarianceDistortionlessResponseBeamformer(
            sampling_rate_hz=scenario.SAMPLING_RATE_HZ,
            domain="frequency",
            nfft=MVDR_NFFT,
            overlap=MVDR_NFFT // 2,
            fmin=scenario.FREQ_MIN_HZ,
            fmax=scenario.FREQ_MAX_HZ,
            diagonal_loading=loading,
        )
    raise ValueError(f"Unknown beamformer {kind!r}")


def detector_key(chain):
    """Return what identifies a chain's detector, including the beamformer it is set for."""
    return beamformer_key(chain), chain["detector"], chain["pfa"], chain["calibrated"]


beamformers = {beamformer_key(chain): None for chain in chains.values()}
beamformers = {key: make_beamformer(key) for key in beamformers}

start = time.perf_counter()
detectors = {}  # (detector key, array): calibrated detector
for bf_key, beamformer in beamformers.items():
    keys = {detector_key(c) for c in chains.values() if beamformer_key(c) == bf_key}
    for name in scenario.ARRAY_NAMES:
        scans = None
        for key in keys:
            _, kind, pfa, calibrated = key
            cfar = scenario.cfar_detector(kind, pfa, calibrated)
            if calibrated:
                if scans is None:
                    scans = scenario.noise_scans(arrays, name, beamformer=beamformer)
                NoiseCalibrator(cfar).calibrate_from_noise(scans)
            detectors[key, name] = cfar
calibration_s = time.perf_counter() - start
print(f"{len(detectors)} detectors calibrated in {calibration_s:.1f} s")

# %%
# Post-detection
# --------------
#
# The bistatic tracking example's steps, with its thresholds.
#
# - **Grouping**: an echo crosses the threshold on many beams. Strongest first, a detection is
#   dropped if a stronger one is within 30 m of range (or range sum) and within 30 degrees or
#   6 dB of it. Detections within 5 degrees of a fan's edge are dropped (a target just outside
#   the fan shows up there), and so is any more than 20 dB below the array's strongest on the
#   ping (far-sidelobe leakage). Each kept detection's bearing is refined between beams.
# - **Flank ghosts**: a flank detection is dropped when the other flank has one at the same
#   range sum (within 30 m), the mirrored bearing (within 10 degrees) and 10 dB louder.
# - **Bistatic conversion**: a range sum becomes the range from the receiving array, on the
#   bistatic ellipse with the receiver's bearing.
# - **Towed ambiguity**: a towed detection's bearing and its mirror image about the array axis
#   are both possible; an image within 150 m of another array's detection on the same ping is
#   kept, and if neither is, both are.

EDGE_RAD = scenario.EDGE_RAD
GROUP_RANGE_M = scenario.GROUP_RANGE_M
AMBIGUITY_DISTANCE_M = 150.0
wrap = scenario.wrap
group_detections = scenario.group_detections
is_mirror_ghost = scenario.is_mirror_ghost


def position(name, ping_time, bearing, range_m):
    """Return the horizontal position at a bearing and range from an array's reference point."""
    return scenario.reference_xyz(arrays[name], ping_time)[:2] + range_m * np.array(
        [np.cos(bearing), np.sin(bearing)]
    )


def post_process(grouped, chain, ping_time):
    """Return ``{array: [(bearing_rad, range_m, power_db)]}``, ready for the tracker.

    ``grouped`` holds every array's grouped detections on one ping; only the chain's arrays are
    kept, and its ghost removal and towed ambiguity steps applied if it has them.
    """
    used = chain["arrays"]
    measured = {name: list(grouped[name]) for name in used}
    if chain["ghost_removal"] and "Port" in used and "Starboard" in used:
        port, starboard = measured["Port"], measured["Starboard"]
        measured["Port"] = [d for d in port if not is_mirror_ghost(d, starboard)]
        measured["Starboard"] = [d for d in starboard if not is_mirror_ghost(d, port)]
    transmitter_xy = scenario.reference_xyz(arrays["Bow"], ping_time)[:2]
    for name in used:
        if name == "Bow":
            continue
        receiver_xy = scenario.reference_xyz(arrays[name], ping_time)[:2]
        measured[name] = [
            (b, scenario.bistatic_range_m(r, b, receiver_xy, transmitter_xy), p)
            for b, r, p in measured[name]
        ]
    if "Towed" in used:
        alpha = scenario.array_axis_rad(arrays["Towed"], ping_time)
        others = [
            position(name, ping_time, b, r)
            for name in used
            if name != "Towed"
            for b, r, _ in measured[name]
        ]
        resolved = []
        for bearing, range_m, power_db in measured["Towed"]:
            images = [bearing, wrap(2 * alpha - bearing)]
            near = [
                image
                for image in images
                if chain["towed_ambiguity"]
                and any(
                    np.linalg.norm(position("Towed", ping_time, image, range_m) - xy)
                    < AMBIGUITY_DISTANCE_M
                    for xy in others
                )
            ]
            resolved += [(image, range_m, power_db) for image in (near or images)]
        measured["Towed"] = resolved
    return measured


# %%
# Simulating and processing
# -------------------------
#
# For each propagation configuration and run, every array's pings are simulated without
# beamforming, one ping at a time. Each ping is beamformed once per beamformer, matched-filtered
# and detected once per detector, and post-processed once per chain; only the measurements are
# kept, so a ping's sensor data is dropped before the next is simulated.


def beamform(raw, name, beamformer, ping_time):
    """Return ``raw`` (sensor data alone) with its beams added by ``beamformer``."""
    state = arrays[name].get_platform_state_at(ping_time)
    return ActiveSonarSensorData(
        received_waveform=raw.received_waveform,
        transmit_pulse=raw.transmit_pulse,
        timestamp=raw.timestamp,
        beamformed_data=beamformer.beamform(
            raw.received_waveform, steering[name].calculate(state)
        ),
        steering_bearings_rad=steering[name].steering_bearings(state),
        transmitter_position=raw.transmitter_position,
        receiver_reference_position=raw.receiver_reference_position,
    )


def detect(ping, cfar):
    """Return the CFAR detections on one array's ping."""
    detector = ActiveSonarDetector(
        detector=cfar,
        sensor_data_gen=[(ping.timestamp, {ping})],
        sampling_rate_hz=scenario.SAMPLING_RATE_HZ,
        sound_speed_ms=scenario.SOUND_SPEED_MPS,
        min_range_m=scenario.MIN_RANGE_M,
    )
    _, detections = next(iter(detector.detections_gen()))
    return detections


measurements = {key: [[] for _ in range(NUM_RUNS)] for key in chains}  # [run][ping] -> dict
simulation_s, processing_s = defaultdict(float), defaultdict(float)
for (seabed, mode), keys in by_propagation.items():
    record = {**settings["configurations"][seabed][mode]}
    for run in range(NUM_RUNS):
        model = scenario.propagation_model_from_record(record)
        simulators = {
            name: scenario.simulator(
                arrays,
                name,
                model,
                target_paths.values(),
                PING_TIMES,
                seed=scenario.NOISE_SEEDS[name] + RUNS[run]["seed"] * RUN_SEED_STEP,
                beamformer=None,
            ).sensor_data_gen()
            for name in scenario.ARRAY_NAMES
        }
        for index, ping_time in enumerate(PING_TIMES):
            start = time.perf_counter()
            raw = {name: next(iter(next(gen)[1])) for name, gen in simulators.items()}
            simulation_s[seabed, mode] += time.perf_counter() - start

            # Group each array's detections once per detector, shared by every chain using it.
            grouped = {}
            for key in keys:
                chain = chains[key]
                d_key = detector_key(chain)
                if d_key in grouped:
                    continue
                start = time.perf_counter()
                grouped[d_key] = {}
                for name in scenario.ARRAY_NAMES:
                    beamformed = beamform(
                        raw[name], name, beamformers[beamformer_key(chain)], ping_time
                    )
                    power = matched_filter_power(beamformed)
                    found = detect(beamformed, detectors[d_key, name])
                    grouped[d_key][name] = group_detections(found, beamformed, power)
                processing_s[seabed, mode, d_key] += time.perf_counter() - start
            for key in keys:
                chain = chains[key]
                measurements[key][run].append(
                    post_process(grouped[detector_key(chain)], chain, ping_time)
                )
            print(
                f"{seabed:>8} {mode:>12} run {run} ping {index + 1}/{NUM_PINGS}:"
                f" {len(keys)} chains",
                flush=True,
            )
    print(f"{seabed} {mode}: simulated in {simulation_s[seabed, mode]:.0f} s")

# %%
# Tracking and scoring
# --------------------
#
# One tracker per chain and run (``paper_scenario.run_tracker``), at the untuned settings set
# from the sensors' specification (``paper_scenario.TRACKER_DEFAULTS``): the tracker fuses the
# chain's arrays by centralised, sequential measurement fusion through an extended Kalman
# filter, as in the bistatic tracking example. ``stone_soup_tracker_tuning.py`` tunes these
# settings on the measurements saved below.
#
# At every ping the tracks are scored against the truth with GOSPA (generalised optimal
# sub-pattern assignment), with a cut-off of 200 m and :math:`p = 1`: the localisation error
# of the tracks assigned to targets plus half the cut-off for every missed target and every
# false track (``paper_scenario.score_tracks``). Each array's measurements are also scored: a
# target counts as detected when a measurement is within 100 m of it, and a measurement near
# no target is false.


def score_measurements(chain_measurements):
    """Return each array's probability of detection and false measurements per ping."""
    detected, false = defaultdict(int), defaultdict(int)
    for ping_time, measured in zip(PING_TIMES, chain_measurements, strict=True):
        truths = [scenario.target_xyz(path, ping_time)[:2] for path in target_paths.values()]
        for name, ms in measured.items():
            xy = [position(name, ping_time, b, r) for b, r, _ in ms]
            detected[name] += sum(
                any(np.linalg.norm(p - t) < DETECTION_GATE_M for p in xy) for t in truths
            )
            false[name] += sum(
                all(np.linalg.norm(p - t) >= DETECTION_GATE_M for t in truths) for p in xy
            )
    num_truths = len(target_paths) * len(PING_TIMES)
    return {
        "pd": {name: detected[name] / num_truths for name in detected},
        "false_per_ping": {name: false[name] / len(PING_TIMES) for name in detected},
    }


results, example_tracks, tracking_s = {}, {}, defaultdict(float)
for key, chain in chains.items():
    runs = []
    for run in range(NUM_RUNS):
        start = time.perf_counter()
        tracks, states_at = scenario.run_tracker(
            arrays, PING_TIMES, measurements[key][run], chain["arrays"]
        )
        tracking_s[key] += time.perf_counter() - start
        runs.append(
            {
                **scenario.score_tracks(target_paths, PING_TIMES, states_at),
                **score_measurements(measurements[key][run]),
            }
        )
        if run == 0:
            example_tracks[key] = tracks
    results[key] = runs


def settled_mean(key, measure):
    """Return a chain's mean of a per-ping measure over its runs, after the settling pings."""
    return float(np.mean([r[measure][SETTLE_PINGS:] for r in results[key]]))


# %%
# Results
# -------

summary = {}
for experiment, keys in experiment_chains.items():
    print(f"\n{experiment}\n{'=' * len(experiment)}")
    print(
        f"{'Chain':>40} {'GOSPA (m)':>10} {'Localisation (m)':>17} {'False tracks':>13}"
        f" {'Missed':>7} {'Pd (all arrays)':>16} {'False meas/ping':>16}"
    )
    summary[experiment] = []
    for key in keys:
        chain = chains[key]
        pd = np.mean([np.mean(list(r["pd"].values())) for r in results[key]])
        false = np.mean([sum(r["false_per_ping"].values()) for r in results[key]])
        row = {
            "chain": chain_label(chain),
            "config": {k: list(v) if isinstance(v, tuple) else v for k, v in chain.items()},
            "gospa_m": settled_mean(key, "gospa_m"),
            "localisation_m": settled_mean(key, "localisation_m"),
            "false_tracks": settled_mean(key, "false_tracks"),
            "missed_targets": settled_mean(key, "missed_targets"),
            "pd": float(pd),
            "false_measurements_per_ping": float(false),
            "pd_by_array": {
                name: float(np.mean([r["pd"][name] for r in results[key]]))
                for name in chain["arrays"]
            },
            "gospa_by_ping_m": np.mean([r["gospa_m"] for r in results[key]], axis=0).tolist(),
            "tracking_s": tracking_s[key],
        }
        summary[experiment].append(row)
        print(
            f"{row['chain']:>40} {row['gospa_m']:>10.1f} {row['localisation_m']:>17.1f}"
            f" {row['false_tracks']:>13.2f} {row['missed_targets']:>7.2f} {row['pd']:>16.2f}"
            f" {row['false_measurements_per_ping']:>16.2f}"
        )

# %%
# Saved measurements and compute cost
# -----------------------------------
#
# Every chain's measurements, for every run and ping, are saved for the tracker tuning script,
# which re-runs only the tracker. Each chain's compute cost is what its propagation and signal
# processing configuration would take to run on its own, over every run: its propagation
# configuration's whole simulation, plus its beamforming, matched filtering, detection and
# grouping. Tracking is left out of the cost, since it depends on the tracker's settings, which
# the tuning script changes; it is saved separately.

chain_cost_s = {
    key: simulation_s[chain["seabed"], chain["propagation"]]
    + processing_s[chain["seabed"], chain["propagation"], detector_key(chain)]
    for key, chain in chains.items()
}


def plain(value):
    """Return ``value`` with tuples as lists and NumPy scalars as Python numbers, for JSON."""
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


MEASUREMENTS_PATH.write_text(
    json.dumps(
        {
            "settings_file": SETTINGS_PATH.name,
            "ping_times": [t.isoformat() for t in PING_TIMES],
            "run_seed_step": RUN_SEED_STEP,
            "runs": RUNS,
            "chains": [
                {
                    "label": chain_label(chain),
                    "config": plain(chain),
                    "cost_s": chain_cost_s[key],
                    "simulation_s": simulation_s[chain["seabed"], chain["propagation"]],
                    "processing_s": processing_s[
                        chain["seabed"], chain["propagation"], detector_key(chain)
                    ],
                    "tracking_s": tracking_s[key],
                    "measurements": plain(measurements[key]),
                }
                for key, chain in chains.items()
            ],
        }
    )
)
print(f"Measurements saved to {MEASUREMENTS_PATH.name}")

RESULTS_PATH.write_text(
    json.dumps(
        {
            "settings_file": SETTINGS_PATH.name,
            "pings": NUM_PINGS,
            "runs": RUNS,
            "settle_pings": SETTLE_PINGS,
            "gospa": {"c_m": GOSPA_C_M, "p": GOSPA_P},
            "experiments": summary,
        },
        indent=2,
    )
)
print(f"\nResults saved to {RESULTS_PATH.name}")

# %%
# GOSPA over time
# ---------------
#
# One panel per experiment: each chain's GOSPA at every ping, averaged over the runs.

figure = make_subplots(
    rows=len(summary), cols=1, subplot_titles=list(summary), vertical_spacing=0.08
)
ping_numbers = np.arange(1, NUM_PINGS + 1)
for row, (experiment, rows) in enumerate(summary.items(), start=1):
    for entry in rows:
        figure.add_trace(
            go.Scatter(
                x=ping_numbers,
                y=entry["gospa_by_ping_m"],
                mode="lines",
                name=f"{experiment}: {entry['chain']}",
                legendgroup=experiment,
            ),
            row=row,
            col=1,
        )
    figure.update_yaxes(title_text="GOSPA (m)", row=row, col=1)
figure.update_xaxes(title_text="Ping", row=len(summary), col=1)
figure.update_layout(
    template="plotly_white", title="GOSPA over Time", height=380 * len(summary), width=1300
)
save_figure(figure, "gospa_over_time")

# %%
# False tracks against missed targets
# -----------------------------------
#
# Each chain's mean false tracks against its mean missed targets per ping, one panel per
# experiment: the trade a detection threshold sets, and how other choices shift it.

figure = make_subplots(rows=1, cols=len(summary), subplot_titles=list(summary))
for col, rows in enumerate(summary.values(), start=1):
    figure.add_trace(
        go.Scatter(
            x=[entry["missed_targets"] for entry in rows],
            y=[entry["false_tracks"] for entry in rows],
            mode="markers+text",
            text=[entry["chain"] for entry in rows],
            textposition="top center",
            textfont={"size": 9},
            marker={"size": 10},
            showlegend=False,
        ),
        row=1,
        col=col,
    )
    figure.update_xaxes(title_text="Missed targets per ping", row=1, col=col)
    figure.update_yaxes(title_text="False tracks per ping" if col == 1 else None, row=1, col=col)
figure.update_layout(
    template="plotly_white",
    title="False Tracks against Missed Targets",
    height=550,
    width=480 * len(summary),
)
save_figure(figure, "false_tracks_against_missed_targets")

# %%
# Tracks
# ------
#
# For each experiment, one panel per chain from the first run: the true target tracks
# (dashed), every array's measurements, and the chain's final tracks.

ARRAY_COLOURS = {"Bow": "black", "Port": "crimson", "Starboard": "seagreen", "Towed": "purple"}
for experiment, keys in experiment_chains.items():
    cols = min(len(keys), 4)
    rows = -(-len(keys) // cols)
    figure = make_subplots(
        rows=rows, cols=cols, subplot_titles=[chain_label(chains[k]) for k in keys]
    )
    for index, key in enumerate(keys):
        row, col = index // cols + 1, index % cols + 1
        first = index == 0
        for target, path in target_paths.items():
            xy = np.array([scenario.target_xyz(path, t)[:2] for t in PING_TIMES])
            figure.add_trace(
                go.Scatter(
                    x=xy[:, 0],
                    y=xy[:, 1],
                    mode="lines",
                    line={"color": "black", "dash": "dash"},
                    name="Truth",
                    showlegend=first and target == "T1",
                ),
                row=row,
                col=col,
            )
        for name in chains[key]["arrays"]:
            xy = np.array(
                [
                    position(name, t, b, r)
                    for t, measured in zip(PING_TIMES, measurements[key][0], strict=True)
                    for b, r, _ in measured[name]
                ]
            )
            if xy.size:
                figure.add_trace(
                    go.Scatter(
                        x=xy[:, 0],
                        y=xy[:, 1],
                        mode="markers",
                        marker={"size": 3, "color": ARRAY_COLOURS[name]},
                        name=name,
                        legendgroup=name,
                        showlegend=first,
                    ),
                    row=row,
                    col=col,
                )
        for number, track in enumerate(example_tracks[key]):
            xy = np.array(
                [[float(s.state_vector[0, 0]), float(s.state_vector[2, 0])] for s in track]
            )
            figure.add_trace(
                go.Scatter(
                    x=xy[:, 0],
                    y=xy[:, 1],
                    mode="lines",
                    line={"color": "gold", "width": 3},
                    name="Tracks",
                    legendgroup="Tracks",
                    showlegend=first and number == 0,
                ),
                row=row,
                col=col,
            )
        figure.update_xaxes(range=[-500.0, 4000.0], row=row, col=col)
        figure.update_yaxes(range=[-2500.0, 2500.0], row=row, col=col)
    figure.update_layout(
        template="plotly_white",
        title=f"{experiment}: Measurements and Tracks (First Run)",
        height=420 * rows,
        width=420 * cols,
    )
    save_figure(figure, f"{experiment}_tracks")

# %%
# Compute time
# ------------

print("\nSimulation time per propagation configuration (all runs, all arrays):")
for (seabed, mode), seconds in simulation_s.items():
    print(f"{seabed:>8} {mode:>12}: {seconds:>8.1f} s")
print("\nBeamforming, detection and grouping time per detector (all runs, all arrays):")
for (seabed, mode, key), seconds in processing_s.items():
    (bf, loading), kind, pfa, calibrated = key
    bf_label = bf if loading is None else f"{bf} ({loading:g})"
    print(
        f"{seabed:>8} {mode:>12} {bf_label:>18} {kind:>3} Pfa {pfa:<6g}"
        f" {'cal' if calibrated else 'nom'}: {seconds:>8.1f} s"
    )
