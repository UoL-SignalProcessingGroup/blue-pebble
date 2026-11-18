r"""
==========================================
Configuring Bellhop for the Bistatic Sonar
==========================================

This script settles the propagation model for the paper's bistatic scenario, set up in
``paper_scenario.py``: a ship's bow array pings, and the bow, both flank arrays and a towed
array listen. It compares Blue Pebble's Bellhop configurations over a flat and a sloping
seabed, finds each one's converged ray and grid settings, and saves them for the scripts that
follow, which run the whole processing chain over many pings.

Five configurations are compared:

- **2D all**: :class:`~bluepebble.models.propagation.BellhopAcousticPropagationModel`, every
  element in one run, in the vertical plane towards the array's reference point.
- **2D far field**: one 2D run to the reference point, with plane-wave delays across the array.
- **3D all**: :class:`~bluepebble.models.propagation.Bellhop3DAcousticPropagationModel`, fully
  3D rays to every element. It is the most complete ray model here, so it is the reference.
- **3D far field**: one 3D run to the reference point, with plane-wave delays across the array.
- **Nx2D**: Bellhop3D's standard N x 2D mode: each element's vertical plane traced in 2D over
  the seabed along it.

A ray that reflects off a seabed sloping across its path is turned out of its vertical plane, so
only 3D can follow it. The water refracts (Munk profile), so there is no exact answer to score
against; every configuration is instead compared with 3D all.

1. **Detection**: a CFAR detector finds the echoes in each array's beamformed, matched-filtered
   data, near the targets.
2. **Convergence check**: each configuration's ray and grid settings are refined, level by
   level, until no array's detections move; it uses that level from then on.
3. **Pings**: one ping per configuration and seabed, on all four arrays.
4. **Difference from 3D all**: each configuration's detections against 3D all's, array by
   array, in range, bearing and position, set against its compute time.
5. **Saved settings**: every configuration's converged settings, for the later scripts.

Assumptions
-----------
- Everything about the scenario is in ``paper_scenario.py``: Munk water 300 m deep at the
  origin, a lossless fluid seabed (1700 m/s, 1.5 g/cm³), point targets with a 15 dB target
  strength, 72 dB white noise, and no reverberation or direct blast.
- Only the first ping's geometry is used here. The settings are assumed to stay converged as
  the ship and targets move over the later scripts' pings.
- The bow array measures slant range, the other arrays range sum (transmitter to target to
  receiver). Both are compared as measured.
"""  # noqa: D205, D212, D400, D415

# %%
# Imports
# -------

import json
import textwrap
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import paper_scenario as scenario
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from bluepebble.detector import ActiveSonarDetector, matched_filter_power
from bluepebble.models.propagation import (
    Bellhop3DAcousticPropagationModel,
    BellhopAcousticPropagationModel,
)

# %%
# Configuration
# -------------

# Set to "plotly" to save interactive HTML figures, or "pdf" to save static PDF figures. Either
# way, figures are written to a folder named after this script.
PLOTTING_MODE = "pdf"
FIGURE_DIR = Path(__file__).with_name(f"{Path(__file__).stem}_figures")
SETTINGS_PATH = Path(__file__).with_name(f"{Path(__file__).stem}_settings.json")

# The seabeds to compare. Flat is the control, over which every configuration should agree;
# the wedge slopes across the line of sight to most of the targets. ("Seamount" is also
# defined in paper_scenario.py, but left out of the paper.)
SEABEDS = ("Flat", "Wedge")
SLOPED_SEABEDS = [seabed for seabed in SEABEDS if seabed != "Flat"]

MODES = scenario.MODES
REFERENCE_MODE = scenario.REFERENCE_MODE
OTHER_MODES = [mode for mode in MODES if mode != REFERENCE_MODE]
ARRAYS = scenario.ARRAY_NAMES
PING_TIME = scenario.START

# Tolerances are set as delays and converted to each array's measure: the bow array's range
# is c * tau / 2, the others' range sum c * tau, so the same delay is twice the distance.
CONVERGED_DELAY_S = 2 * 0.05 / scenario.SOUND_SPEED_MPS  # 0.05 m of bow range, 0.1 m of sum
CONVERGED_BEARING_DEG = 0.05
MATCH_M = 15.0  # pairs measurements across configurations: half the grouping distance
MATCH_BEARING_DEG = 5.0
NEAR_TARGET_DEG = 15.0  # measurements further than this from every target are ignored
MULTIPATH_EXTRA_PATH_M = 400.0  # bottom echoes arrive up to this much path after the direct


def save_figure(figure: go.Figure, name: str) -> None:
    """Write ``figure`` to ``FIGURE_DIR`` in the format set by ``PLOTTING_MODE``."""
    FIGURE_DIR.mkdir(exist_ok=True)
    if PLOTTING_MODE == "plotly":
        figure.write_html(FIGURE_DIR / f"{name}.html", include_plotlyjs="directory")
    elif PLOTTING_MODE == "pdf":
        figure.write_image(FIGURE_DIR / f"{name}.pdf")
    else:
        raise ValueError(f'PLOTTING_MODE must be "plotly" or "pdf", not {PLOTTING_MODE!r}')


def report(title: str, text: str) -> None:
    """Print a heading and a short explanation before a set of results."""
    print(f"\n{title}\n{'=' * len(title)}")
    print(textwrap.fill(" ".join(text.split()), width=95))


def metres_per_second_of_delay(name):
    """Return how far one second of delay moves array ``name``'s range measure."""
    return scenario.SOUND_SPEED_MPS / (2 if name == "Bow" else 1)


def wrap_deg(angle_deg):
    """Wrap an angle in degrees to [-180, 180)."""
    return (np.asarray(angle_deg) + 180.0) % 360.0 - 180.0


SCRIPT_START_S = time.perf_counter()

# Ray and grid settings for each model class at refinement level 0, the coarsest. Each level
# above refines every one of them at once: twice the seabed points and elevation beams and, in
# 3D, half the bearing spacing, so level 2 has the models' default beams and spacing.
BASE_SETTINGS = {
    BellhopAcousticPropagationModel: {"n_bathy_samples": 50, "num_beams": 500},
    Bellhop3DAcousticPropagationModel: {
        "n_bathy_samples": 50,
        "num_beams": 500,
        "bearing_beam_spacing_deg": 0.4,
    },
}
MAX_LEVEL = 4  # 800 seabed points, 8000 beams, 0.025 deg: 256 times level 0's rays in 3D
SHORT_NAMES = {
    "n_bathy_samples": "seabed",
    "num_beams": "beams",
    "bearing_beam_spacing_deg": "bearing",
}
# {(seabed, mode): {setting: level}}, from the convergence check
LEVELS = {}


def settings_at(model_class, level):
    """Return ``model_class``'s settings at refinement ``level`` (one level, or per setting)."""
    base = BASE_SETTINGS[model_class]
    levels = level if isinstance(level, dict) else dict.fromkeys(base, level)
    return {
        name: value / 2 ** levels[name]
        if name.endswith("_deg")
        else int(round(value * 2 ** levels[name]))
        for name, value in base.items()
    }


def model(mode, seabed, level=None):
    """Return configuration ``mode`` over ``seabed`` at ``level`` (default: its converged one)."""
    level = LEVELS[seabed, mode] if level is None else level
    return scenario.propagation_model(mode, seabed, settings_at(MODES[mode][0], level))


# %%
# Scenario
# --------
#
# The ship, arrays and targets at the first ping, from ``paper_scenario.py``. The target tracks
# over all the later scripts' pings are drawn too, to show where the scenario goes.

platform, arrays = scenario.ship(ping_times=[PING_TIME])
target_paths = scenario.targets(ping_times=[PING_TIME])
target_tracks = scenario.targets()
_, arrays_over_time = scenario.ship()
report(
    "Scenario",
    "At the first ping: "
    + "; ".join(f"{name} {array.num_sensors} sensors" for name, array in arrays.items())
    + f". Delays are converted to range at {scenario.SOUND_SPEED_MPS:.1f} m/s, the sound speed "
    "at the bow array.",
)
print(f"\n{'Target':>6} {'Array':>10} {'Range or range sum (m)':>23} {'Bearing (deg)':>14}")
for target, path in target_paths.items():
    xyz = scenario.target_xyz(path, PING_TIME)
    for name in ARRAYS:
        r, b = scenario.true_range_and_bearing(arrays, name, xyz, PING_TIME)
        print(f"{target:>6} {name:>10} {r:>23.1f} {np.degrees(b):>14.1f}")

# %%
# Plan view
# ---------
#
# Each panel shows one sloped seabed's water depth, in 50 m steps, with the ship's track, the
# arrays at the first ping and the targets' tracks over the later scripts' pings (the marker is
# the first ping).

grid_m = np.linspace(-1000.0, 4000.0, 51)
figure = make_subplots(
    rows=1, cols=len(SLOPED_SEABEDS), subplot_titles=SLOPED_SEABEDS, horizontal_spacing=0.08
)
for col, seabed in enumerate(SLOPED_SEABEDS, start=1):
    bathymetry = scenario.seabed(seabed)
    depth_m = np.array([[-bathymetry.get_depth(x, y) for x in grid_m] for y in grid_m - 1500.0])
    figure.add_trace(
        go.Contour(
            x=grid_m,
            y=grid_m - 1500.0,
            z=depth_m,
            colorscale="Blues",
            opacity=0.6,
            contours={"start": 150.0, "end": 500.0, "size": 50.0},
            line={"color": "white", "width": 1},
            colorbar={"title": {"text": "Depth (m)", "side": "right"}, "len": 0.8},
            showscale=col == 1,
            hoverinfo="skip",
        ),
        row=1,
        col=col,
    )
    bow_track = np.array(
        [scenario.reference_xyz(arrays_over_time["Bow"], t) for t in scenario.PING_TIMES]
    )
    figure.add_trace(
        go.Scatter(
            x=bow_track[:, 0],
            y=bow_track[:, 1],
            mode="lines",
            line={"color": "black", "width": 2},
            name="Ship track",
            showlegend=col == 1,
        ),
        row=1,
        col=col,
    )
    for name, array in arrays.items():
        xyz = np.asarray(array.get_platform_state_at(PING_TIME).array.state_vector, dtype=float)
        figure.add_trace(
            go.Scatter(
                x=xyz[0],
                y=xyz[1],
                mode="markers",
                marker={"size": 4},
                name=name,
                legendgroup=name,
                showlegend=col == 1,
            ),
            row=1,
            col=col,
        )
    for target, path in target_tracks.items():
        track = np.array([scenario.target_xyz(path, t) for t in scenario.PING_TIMES])
        figure.add_trace(
            go.Scatter(
                x=track[:, 0],
                y=track[:, 1],
                mode="lines+markers+text",
                marker={"size": [10] + [0] * (len(track) - 1), "color": "crimson"},
                line={"color": "crimson"},
                text=[f"<b>{target}</b>"] + [""] * (len(track) - 1),
                textposition="middle right",
                name=target,
                showlegend=False,
            ),
            row=1,
            col=col,
        )
    figure.update_xaxes(title_text="x (m)", range=[-1000.0, 4000.0], row=1, col=col)
    figure.update_yaxes(
        title_text="y (m)" if col == 1 else None,
        range=[-2500.0, 2500.0],
        scaleanchor=f"x{col if col > 1 else ''}",
        row=1,
        col=col,
    )
figure.update_layout(
    template="plotly_white",
    title="Plan View: Seabeds, Ship and Target Tracks",
    legend={"orientation": "h", "yanchor": "top", "y": -0.15, "xanchor": "center", "x": 0.5},
    height=650,
    width=650 * len(SLOPED_SEABEDS),
)
save_figure(figure, "plan_view")

# %%
# Detection
# ---------
#
# Each array has the paper's baseline detector (``paper_scenario.cfar_detector``), an
# ordered-statistic CFAR calibrated on a noise-only ping of its own for a false alarm
# probability of :math:`10^{-7}`. :class:`~bluepebble.detector.ActiveSonarDetector`
# matched-filters each beam and runs it along range.
#
# Detections are then turned into the measurements a tracker would get, by the bistatic
# tracking example's grouping (``paper_scenario.group_detections``): an echo crosses the
# threshold on many beams and, spread by multipath, over several metres of range, so strongest
# first a detection is dropped if a stronger one is within 30 m of range (or range sum) and
# within 30 degrees or 6 dB of it. Detections at a fan's edge, and any more than 20 dB below the
# array's strongest, are dropped too, and each kept one's bearing is refined between beams.
# Only measurements near a target are compared: within 15 degrees of its bearing (or, for the
# towed array, its mirror image about the array axis) and from its straight-line range (or
# range sum) to 400 m of path beyond it, which covers its bottom echoes. That leaves out noise
# false alarms, which are the same in every configuration since the noise is.

start = time.perf_counter()
cfars = scenario.calibrated_cfar_detectors(arrays)
calibration_s = time.perf_counter() - start


def near_a_target(name, range_m, bearing_deg):
    """Return whether a detection on array ``name`` could be an echo of one of the targets."""
    to_range = metres_per_second_of_delay(name) / scenario.SOUND_SPEED_MPS  # 1/2 bow, 1 others
    for path in target_paths.values():
        xyz = scenario.target_xyz(path, PING_TIME)
        true_range_m, true_bearing = scenario.true_range_and_bearing(arrays, name, xyz, PING_TIME)
        bearings = [np.degrees(true_bearing)]
        if name == "Towed":
            alpha = scenario.array_axis_rad(arrays["Towed"], PING_TIME)
            bearings.append(np.degrees(2 * alpha - true_bearing))
        in_range = true_range_m - 5.0 < range_m < true_range_m + MULTIPATH_EXTRA_PATH_M * to_range
        if in_range and min(abs(wrap_deg(bearing_deg - b)) for b in bearings) < NEAR_TARGET_DEG:
            return True
    return False


def detect(ping, name):
    """Return array ``name``'s measurements near the targets: ``[(range_m, bearing_deg)]``.

    The range is the bow array's slant range, or the other arrays' range sum.
    """
    detector = ActiveSonarDetector(
        detector=cfars[name],
        sensor_data_gen=[(ping.timestamp, {ping})],
        sampling_rate_hz=scenario.SAMPLING_RATE_HZ,
        sound_speed_ms=scenario.SOUND_SPEED_MPS,
        min_range_m=scenario.MIN_RANGE_M,
    )
    _, detections = next(iter(detector.detections_gen()))
    grouped = scenario.group_detections(detections, ping, matched_filter_power(ping))
    measured = [(range_m, float(np.degrees(bearing))) for bearing, range_m, _ in grouped]
    return sorted(m for m in measured if near_a_target(name, *m))


# %%
# Pinging and timing
# ------------------
#
# A ping simulates every array with one propagation model, each array with its own noise. The
# propagation model's time is measured on its own, since the echo synthesis, beamforming and
# detection are the same work for every configuration, and is split in two:
#
# - **Bellhop runs**: running ``bellhopcxx`` and reading its arrivals.
# - **Model**: the whole propagation model, including turning arrivals into transfer functions.

propagation_s, bellhop_s = {}, {}


def add_timer(propagation_model, method, times, key):
    """Wrap ``propagation_model.method`` so that its wall-clock time is added to ``times[key]``.

    Overlapping calls (a model running several Bellhop processes at once) are counted once.
    """
    call = getattr(propagation_model, method)
    lock, state = threading.Lock(), {"running": 0, "since": 0.0}

    def timed_call(*args, **kwargs):
        with lock:
            if state["running"] == 0:
                state["since"] = time.perf_counter()
            state["running"] += 1
        try:
            return call(*args, **kwargs)
        finally:
            with lock:
                state["running"] -= 1
                if state["running"] == 0:
                    times[key] = times.get(key, 0.0) + time.perf_counter() - state["since"]

    setattr(propagation_model, method, timed_call)


def timed(propagation_model, key):
    """Return ``propagation_model``, timing its whole calls and its Bellhop runs within them."""
    add_timer(propagation_model, "propagate_spectrum", propagation_s, key)
    if hasattr(propagation_model, "_run_arrivals"):
        add_timer(propagation_model, "_run_arrivals", bellhop_s, key)
    return propagation_model


def ping(propagation_model):
    """Return ``{array: detections}`` from one ping of every array with ``propagation_model``."""
    found = {}
    for name in ARRAYS:
        simulator = scenario.simulator(
            arrays, name, propagation_model, target_paths.values(), [PING_TIME]
        )
        _, data = next(iter(simulator.sensor_data_gen()))
        found[name] = detect(next(iter(data)), name)
    return found


# %%
# Convergence check
# -----------------
#
# Each configuration's settings are checked over each seabed before it pings for the
# comparison, on what the comparison measures: every array's detections. Each check ping goes
# through the same simulator, noise, beamformer and detector, so the only difference between two
# pings is the settings.
#
# Two pings agree when, on every array, each measurement in one is paired with one in the other
# (the nearest in range within 15 m and within 5 degrees), and none moves by more than 0.05 m of
# bow range (0.1 m of range sum) or 0.05 degrees.
#
# The check has two passes:
#
# 1. **Together.** Every setting is refined at once, level by level from level 0 (the
#    coarsest), until a level agrees with the next. That level has converged.
# 2. **One at a time.** Each setting in turn is then coarsened on its own, and keeps the
#    coarsest level that still agrees with the converged ping. This gives each configuration
#    its cheapest converged settings and shows which settings the detections depend on.
#
# A configuration still changing at ``MAX_LEVEL`` pings there, and is reported as not converged.


def paired(one, other, name):
    """Pair two lists of ``(range_m, bearing_deg)`` detections on array ``name``.

    Returns ``(pairs, only_one, only_other)``: the pairs, nearest in range first, and the
    detections left unpaired in each list.
    """
    match_m = MATCH_M
    candidates = sorted(
        (abs(r1 - r2), i, j)
        for i, (r1, b1) in enumerate(one)
        for j, (r2, b2) in enumerate(other)
        if abs(r1 - r2) < match_m and abs(wrap_deg(b1 - b2)) < MATCH_BEARING_DEG
    )
    pairs, used_one, used_other = [], set(), set()
    for _, i, j in candidates:
        if i not in used_one and j not in used_other:
            used_one.add(i)
            used_other.add(j)
            pairs.append((one[i], other[j]))
    only_one = [d for i, d in enumerate(one) if i not in used_one]
    only_other = [d for j, d in enumerate(other) if j not in used_other]
    return pairs, only_one, only_other


def detections_agree(seabed, mode, test, reference, trial):
    """Print how far each array's detections moved from ``reference`` to ``trial``.

    Returns whether every array's detections held.
    """
    agree = True
    for name in ARRAYS:
        pairs, lost, gained = paired(reference[name], trial[name], name)
        tolerance_m = CONVERGED_DELAY_S * metres_per_second_of_delay(name)
        moved_m = max((abs(new[0] - ref[0]) for ref, new in pairs), default=0.0)
        moved_deg = max((abs(wrap_deg(new[1] - ref[1])) for ref, new in pairs), default=0.0)
        ok = (
            not lost
            and not gained
            and moved_m <= tolerance_m
            and moved_deg <= CONVERGED_BEARING_DEG
        )
        agree &= ok
        print(
            f"{seabed:>8} {mode:>12} {test:>10} {name:>10} {len(pairs):>6} {len(lost):>5}"
            f" {len(gained):>7} {moved_m:>11.3f} {moved_deg:>14.3f}"
            + ("" if ok else "  not converged"),
            flush=True,
        )
    return agree


report(
    "Convergence check",
    "Over each seabed and for each array, how far the detections move between two pings. Test "
    "'a-b' refines every setting together from level a to b; 'beams 1' and the like coarsen one "
    "setting to that level, against the converged ping. Paired counts the detections in both "
    "pings, Lost and Gained those in only one. Moved is the largest change in range (bow) or "
    "range sum (others), in metres, and in bearing, in degrees.",
)
print(
    f"\n{'Seabed':>8} {'Config':>12} {'Test':>10} {'Array':>10} {'Paired':>6} {'Lost':>5}"
    f" {'Gained':>7} {'Moved (m)':>11} {'Moved (deg)':>14}"
)
start = time.perf_counter()
converged = {}
for seabed in SEABEDS:
    for mode in MODES:
        names = list(BASE_SETTINGS[MODES[mode][0]])
        # Pass 1: every setting together.
        positions = ping(model(mode, seabed, 0))
        joint, converged[seabed, mode] = MAX_LEVEL, False
        for level in range(MAX_LEVEL):
            finer = ping(model(mode, seabed, level + 1))
            if detections_agree(seabed, mode, f"{level}-{level + 1}", positions, finer):
                joint, converged[seabed, mode] = level, True
                break
            positions = finer
        levels = dict.fromkeys(names, joint)
        # Pass 2: each setting on its own, against the converged ping.
        if converged[seabed, mode]:
            for setting in names:
                while levels[setting] > 0:
                    trial = {**levels, setting: levels[setting] - 1}
                    test = f"{SHORT_NAMES[setting]} {trial[setting]}"
                    if not detections_agree(
                        seabed, mode, test, positions, ping(model(mode, seabed, trial))
                    ):
                        break
                    levels = trial
        LEVELS[seabed, mode] = levels
convergence_check_s = time.perf_counter() - start
print()
for (seabed, mode), ok in converged.items():
    levels = LEVELS[seabed, mode]
    settings = settings_at(MODES[mode][0], levels)
    described = ", ".join(f"{SHORT_NAMES[k]} {levels[k]} ({settings[k]:g})" for k in levels)
    verdict = "converged" if ok else f"NOT converged by level {MAX_LEVEL}"
    print(f"{seabed:>8} {mode:>12}: {described}; {verdict}")

# %%
# Pings
# -----
#
# Every configuration now pings over every seabed at the levels the convergence check chose,
# with its propagation model timed.

detections, ping_s = {}, {}
report(
    "One ping per configuration",
    "Each line is one ping of all four arrays: Bellhop is the time running bellhopcxx, Model the "
    "whole propagation model, and Ping everything, including the echo synthesis, beamforming "
    "and detection that every configuration shares.",
)
print(f"\n{'Seabed':>8} {'Config':>12} {'Bellhop (s)':>12} {'Model (s)':>10} {'Ping (s)':>9}")
for seabed in SEABEDS:
    for mode in MODES:
        start = time.perf_counter()
        detections[seabed, mode] = ping(timed(model(mode, seabed), (seabed, mode)))
        ping_s[seabed, mode] = time.perf_counter() - start
        print(
            f"{seabed:>8} {mode:>12} {bellhop_s.get((seabed, mode), 0.0):>12.1f}"
            f" {propagation_s[seabed, mode]:>10.1f} {ping_s[seabed, mode]:>9.1f}"
        )

# %%
# Difference from 3D all
# ----------------------
#
# Each configuration's detections are paired with 3D all's on the same array, as in the
# convergence check, and compared:
#
# - **Range**: the root mean square difference, in bow range or range sum.
# - **Bearing**: the root mean square difference.
# - **Position**: range and bearing combined, as the distance between the two detections. A
#   bearing difference becomes a cross-range one, the target's range from the array times the
#   difference in radians; for the bistatic arrays that range comes from the range sum by the
#   bistatic ellipse. At 2 km, 0.1 degrees is 3.5 m across range, so bearing dominates.
# - **Echoes**: how many both detected, and how many only it or only 3D all detected. An echo
#   one detects and the other doesn't is a different input to the tracker.


def receiver_range_m(name, range_m, bearing_deg):
    """Return the horizontal range from array ``name`` of a detection, for cross-range."""
    if name == "Bow":
        return range_m
    receiver_xy = scenario.reference_xyz(arrays[name], PING_TIME)[:2]
    transmitter_xy = scenario.reference_xyz(arrays["Bow"], PING_TIME)[:2]
    return scenario.bistatic_range_m(range_m, np.radians(bearing_deg), receiver_xy, transmitter_xy)


def differences(seabed, mode, names):
    """Return a configuration's detection differences from 3D all's, over arrays ``names``."""
    range_diffs, bearing_diffs, position_diffs = [], [], []
    both = only_mine = only_reference = 0
    for name in names:
        pairs, mine, theirs = paired(
            detections[seabed, mode][name], detections[seabed, REFERENCE_MODE][name], name
        )
        both += len(pairs)
        only_mine += len(mine)
        only_reference += len(theirs)
        for (r, b), (ref_r, ref_b) in pairs:
            range_diff = r - ref_r
            bearing_diff = float(wrap_deg(b - ref_b))
            cross_m = receiver_range_m(name, ref_r, ref_b) * np.radians(bearing_diff)
            range_diffs.append(range_diff)
            bearing_diffs.append(bearing_diff)
            position_diffs.append(float(np.hypot(range_diff, cross_m)))

    def rms(values):
        return float(np.sqrt(np.mean(np.square(values)))) if values else np.nan

    return {
        "range_m": rms(range_diffs),
        "bearing_deg": rms(bearing_diffs),
        "position_m": rms(position_diffs),
        "both": both,
        "only_mine": only_mine,
        "only_reference": only_reference,
    }


GROUPS = {**{name: (name,) for name in ARRAYS}, "All arrays": ARRAYS}
difference = {
    (seabed, mode, group): differences(seabed, mode, names)
    for seabed in SEABEDS
    for mode in OTHER_MODES
    for group, names in GROUPS.items()
}

report(
    "Detections: difference from 3D all",
    "Per seabed and array, the RMS difference of each configuration's detections from 3D "
    "all's, over the echoes both detected, with how many both detected, only it and only 3D "
    "all, and its model time. Range is bow range or range sum.",
)
print(
    f"\n{'Seabed':>8} {'Array':>10} {'Config':>12} {'Range (m)':>10} {'Bearing (deg)':>14}"
    f" {'Position (m)':>13} {'Both':>5} {'Only it':>8} {'Only 3D all':>12} {'Model (s)':>10}"
)
for seabed in SEABEDS:
    for group in GROUPS:
        for mode in sorted(OTHER_MODES, key=lambda m: propagation_s[seabed, m]):
            d = difference[seabed, mode, group]
            print(
                f"{seabed:>8} {group:>10} {mode:>12} {d['range_m']:>10.2f}"
                f" {d['bearing_deg']:>14.2f} {d['position_m']:>13.2f} {d['both']:>5}"
                f" {d['only_mine']:>8} {d['only_reference']:>12}"
                f" {propagation_s[seabed, mode]:>10.1f}"
            )

# One row per array and one for all of them, one column per measure; in each panel the
# configurations run cheapest first (over the first sloped seabed), with a marker per seabed.

SEABED_STYLES = {
    "Flat": ("#eb6834", "diamond"),
    "Wedge": ("#2a78d6", "circle"),
    "Seamount": ("#1b9e77", "square"),
}
mode_order = sorted(OTHER_MODES, key=lambda m: propagation_s[SLOPED_SEABEDS[0], m])
MEASURES = (
    ("Range difference (m)", "range_m"),
    ("Bearing difference (deg)", "bearing_deg"),
    ("Position difference (m)", "position_m"),
)
figure = make_subplots(
    rows=len(GROUPS),
    cols=len(MEASURES),
    subplot_titles=[
        f"{group}: {label.rsplit(' (', 1)[0]}" for group in GROUPS for label, _ in MEASURES
    ],
    horizontal_spacing=0.07,
    vertical_spacing=0.06,
)
for row, group in enumerate(GROUPS, start=1):
    for col, (label, key) in enumerate(MEASURES, start=1):
        for seabed in SEABEDS:
            colour, symbol = SEABED_STYLES[seabed]
            figure.add_trace(
                go.Scatter(
                    x=mode_order,
                    y=[difference[seabed, mode, group][key] for mode in mode_order],
                    mode="markers",
                    marker={
                        "symbol": symbol,
                        "size": 11,
                        "color": colour,
                        "line": {"width": 1.5, "color": "white"},
                    },
                    name=seabed,
                    legendgroup=seabed,
                    offsetgroup=seabed,
                    showlegend=row == 1 and col == 1,
                    hovertemplate=f"{seabed}, {group}, %{{x}}<br>{label}: %{{y:.3g}}"
                    "<extra></extra>",
                ),
                row=row,
                col=col,
            )
        figure.update_yaxes(title_text=label if col == 1 else None, row=row, col=col)
        figure.update_xaxes(tickangle=-30, showticklabels=row == len(GROUPS), row=row, col=col)
figure.update_layout(
    template="plotly_white",
    title="Detections: RMS Difference from 3D All, by Receiving Array",
    scattermode="group",
    legend={"title": {"text": "Seabed"}},
    height=260 * len(GROUPS),
    width=1500,
)
save_figure(figure, "detection_difference_from_3d_all")

# %%
# Saved settings
# --------------
#
# Every configuration's converged settings over every seabed are saved to a JSON file next to
# this script, so the later scripts can rebuild any of them without repeating the convergence
# check (``paper_scenario.propagation_model_from_record``). Each entry holds the model class,
# its options, the refinement levels and the settings they give, whether they converged, and
# the model time. The detector the settings converged with is saved too, so a later script can
# check that its baseline detector is the same one.
#
# Every configuration is carried on to the processing-chain comparison, 3D all first, then the
# others ranked by how close their detections come to 3D all's over the sloped seabed, on all
# arrays together: first by the echoes detected differently (missed or extra), then by RMS
# position difference, then by model time.


def finite_or_none(value):
    """Return ``value`` as a float, or None if it is NaN (which JSON can't hold)."""
    return None if value is None or not np.isfinite(value) else float(value)


def selection_key(mode):
    """Rank a configuration over the sloped seabeds: echoes found differently, position, time."""
    rows = [difference[seabed, mode, "All arrays"] for seabed in SLOPED_SEABEDS]
    differs = sum(d["only_mine"] + d["only_reference"] for d in rows)
    positions = [d["position_m"] for d in rows if np.isfinite(d["position_m"])]
    position = float(np.mean(positions)) if positions else np.inf
    return differs, position, sum(propagation_s[seabed, mode] for seabed in SLOPED_SEABEDS)


ranking = sorted(OTHER_MODES, key=selection_key)
record = {
    "script": Path(__file__).name,
    "created": datetime.now().isoformat(timespec="seconds"),
    "scenario": {
        "module": "paper_scenario.py",
        "ping_time": PING_TIME.isoformat(),
        "arrays": list(ARRAYS),
        "seabeds": list(SEABEDS),
        "ssp": type(scenario.sound_speed_profile()).__name__,
        "bottom": scenario.BOTTOM,
        "freq_min_hz": scenario.FREQ_MIN_HZ,
        "freq_max_hz": scenario.FREQ_MAX_HZ,
    },
    "detector": {
        "class": type(cfars["Bow"]).__name__,
        "target_pfa": cfars["Bow"].target_pfa,
        "num_guard_cells": cfars["Bow"].num_guard_cells,
        "num_training_cells": cfars["Bow"].num_training_cells,
        "rank": getattr(cfars["Bow"], "rank", None),
        "calibrated": cfars["Bow"].threshold_factor is None,
    },
    "convergence": {
        "bow_range_m": CONVERGED_DELAY_S * metres_per_second_of_delay("Bow"),
        "range_sum_m": CONVERGED_DELAY_S * scenario.SOUND_SPEED_MPS,
        "bearing_deg": CONVERGED_BEARING_DEG,
        "max_level": MAX_LEVEL,
    },
    "reference": REFERENCE_MODE,
    "selected": [REFERENCE_MODE, *ranking],
    "ranking": [
        {
            "configuration": mode,
            **{
                seabed: {
                    key: finite_or_none(value) if isinstance(value, float) else value
                    for key, value in difference[seabed, mode, "All arrays"].items()
                }
                for seabed in SEABEDS
            },
        }
        for mode in ranking
    ],
    "configurations": {
        seabed: {
            mode: {
                "seabed": seabed,
                "model_class": MODES[mode][0].__name__,
                "options": MODES[mode][1],
                "levels": LEVELS[seabed, mode],
                "settings": settings_at(MODES[mode][0], LEVELS[seabed, mode]),
                "converged": converged[seabed, mode],
                "bellhop_s": finite_or_none(bellhop_s.get((seabed, mode))),
                "model_s": float(propagation_s[seabed, mode]),
            }
            for mode in MODES
        }
        for seabed in SEABEDS
    },
}
SETTINGS_PATH.write_text(json.dumps(record, indent=2))

report(
    "Ranking",
    f"Over the {' and '.join(s.lower() for s in SLOPED_SEABEDS)} seabeds, on all arrays: echoes "
    "detected differently from 3D all (missed plus extra), then mean RMS position difference. "
    f"Every configuration's settings are saved to {SETTINGS_PATH.name}.",
)
print(f"\n{'Rank':>4} {'Config':>12} {'Echoes different':>17} {'Position (m)':>13}")
for rank, mode in enumerate(ranking, start=1):
    differs, position, _ = selection_key(mode)
    print(f"{rank:>4} {mode:>12} {differs:>17} {position:>13.2f}")

# %%
# Compute time
# ------------

print(f"\nCFAR calibration (noise-only pings): {calibration_s:.1f} s")
print(f"Convergence check: {convergence_check_s:.1f} s")
print(f"Whole script, including figures: {time.perf_counter() - SCRIPT_START_S:.1f} s")

# %%
# Summary
# -------
#
# - 3D all is the reference: the most complete ray model here. The others are judged by how
#   far their detections are from its detections, array by array, not from an exact answer,
#   which refracting water doesn't have.
# - Over the flat seabed every configuration should agree; over the wedge, the bottom echoes
#   show what tracing in vertical planes, or treating the array as a point, misses.
#   Differences small next to the pulse's 0.77 m range resolution and the beams' width don't
#   matter to a tracker; larger ones, or echoes one detects and the other doesn't, do.
# - The saved settings and ranking feed the processing-chain comparison, which runs every
#   configuration through the whole chain and the tracker over many pings.
