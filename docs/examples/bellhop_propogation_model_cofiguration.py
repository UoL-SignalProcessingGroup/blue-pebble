r"""
===========================
Comparing 2D and 3D Bellhop
===========================

This example compares Blue Pebble's two Bellhop propagation models, for accuracy and compute
time, over a seabed that slopes across the line of sight. A ship's bow array pings and listens
for the echoes (monostatic), as the bow array does in
:ref:`sphx_glr_auto_examples_tracking_targets_with_bistatic_sonar.py`.

Each Bellhop model traces to every element by default, or with ``far_field=True`` only to the
array's reference point, with plane-wave delays across the array. The 2D model also has
``separate_planes``: by default every element shares one vertical plane, and with
``separate_planes=True`` each element gets its own. The 3D model can also run Bellhop's
built-in N x 2D mode, ``nx2d=True``. That gives six Bellhop configurations, compared here with
each other.

- **2D** (:class:`~bluepebble.models.propagation.BellhopAcousticPropagationModel`) traces rays
  in vertical planes.

  - **2D all**: the model's defaults, every element in one run, in the plane towards the
    array's reference point, with each element at its own range and depth.
  - **2D separate planes**: every element too, but one run per element, each in the plane
    through that element.
  - **2D far field**: one run to the reference point, with plane-wave delays across the
    array. A single receiver has a single plane, so ``separate_planes`` doesn't apply.

- **3D** (:class:`~bluepebble.models.propagation.Bellhop3DAcousticPropagationModel`) traces
  fully 3D rays, which can leave their launch bearing. ``separate_planes`` doesn't apply: every
  element is traced at its own bearing.

  - **3D all**: one run, with every element as a receiver at its own position.
  - **3D far field**: one run to the reference point, with plane-wave delays across the array.
  - **Nx2D**: Bellhop3D's standard N x 2D mode, one run in which every element's vertical
    plane is traced in 2D over the seabed along it. It is the same approximation as 2D
    separate planes, from Bellhop's own implementation: rays stay in their planes. Its
    receivers have to be a full grid of bearings, ranges and depths, so it traces far more
    points than there are elements.

- **RTRS** (:class:`~bluepebble.models.propagation.rtrsAcousticPropagationModel`), a separate 3D
  ray tracer, is set up below as an independent check on 3D all but left out for now: over the
  seamount its answer kept changing as its settings were refined.

A ray that reflects off a seabed sloping across its path is turned out of its vertical plane, so
only 3D can follow it. The water has a realistic, refracting sound-speed profile (Munk), so the
rays curve and there is no exact answer to score against. Every configuration is instead
compared with **3D all**, the most complete ray model here: it traces fully 3D rays to every
element. A run of 3D all with half its bearing beam spacing checks that the reference itself
has converged.

1. **Reference echoes**: 3D all's eigenrays from the bow array to each target name every echo
   (direct, surface- or bottom-reflected on each leg) and give its delay and arrival bearing.
2. **Detection**: a CFAR detector finds the echoes in each ping's beamformed, matched-filtered
   data, and each detection is matched to a reference echo.
3. **Convergence check**: each configuration's ray and grid settings are refined, level by
   level, until its detected echo positions stop changing; it uses that level from then on.
4. **Echoes**: one ping per configuration and seabed through
   :class:`~bluepebble.simulator.ActiveSonarSimulator`, detected as above.
5. **Difference from 3D all**, at two levels:

   - *Propagation*: each configuration's echo transfer functions on every element, against 3D
     all's. This is the model on its own, with no noise, beamformer or detector.
   - *Detections*: each configuration's detections against 3D all's, in range and bearing.

   Both are set against the time each configuration's propagation model took.

Assumptions
-----------
- Munk sound-speed profile, about 1547 m/s at the bow array and 1523 m/s at 300 m. Only the
  bathymetry varies with position.
- The seabed is a plane sloping down at 8% towards +y, and a lossless fluid half-space
  (1700 m/s, 1.5 g/cm³). Below the critical angle it reflects totally, so the bottom paths are
  as strong as the direct path.
- Each target is a point scatterer with a 15 dB target strength. There is no reverberation.
  White noise at 60 dB re 1 µPa is added for the detector, with the same realisation in every
  configuration, so every difference between them comes from the propagation model.
- Geometry is frozen during the ping, so there is no Doppler.
- The full script runs 12 pings (6 configurations over 2 seabeds), after the convergence
  check's noise-free pings; most of each ping's time is echo synthesis and beamforming, which
  every configuration shares.
"""  # noqa: D205, D212, D400, D415

# %%
# Imports
# -------

import textwrap
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import hilbert
from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel,
    ConstantVelocity,
)
from stonesoup.types.array import StateVector
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState

from bluepebble.arrays import ArrayState, BowArray, HostState, PlatformState
from bluepebble.detector import (
    ActiveSonarDetector,
    NoiseCalibrator,
    OSCFARDetector,
    matched_filter_scans_from_sensor_data,
)
from bluepebble.models import ConstantTargetStrength
from bluepebble.models.environment import (
    FlatBathymetry,
    Munk,
    SeamountBathymetry,
    WedgeBathymetry,
)
from bluepebble.models.propagation import (
    Bellhop3DAcousticPropagationModel,
    BellhopAcousticPropagationModel,
    rtrsAcousticPropagationModel,  # noqa: F401  # for RTRS, left out for now (see MODES)
)
from bluepebble.platform import SonarPlatform
from bluepebble.signal import LFMSignal
from bluepebble.signal.random import WhiteNoiseSignal
from bluepebble.sigproc import (
    DelayAndSumBeamformer,
    MatchedFilter,
    SteeringCalculator,
    parabolic_peak_offset,
)
from bluepebble.simulator import ActiveSonarSimulator

# %%
# Configuration
# -------------
#
# The pulse is a 0.5 s linear frequency-modulated (LFM) sweep over 1-3 kHz. With sound speed
# :math:`c` and bandwidth :math:`B`, it resolves path lengths :math:`c / B`, about 0.77 m, apart.

# Set to "plotly" to save interactive HTML figures, or "pdf" to save static PDF figures. Either
# way, figures are written to a folder named after this script.
PLOTTING_MODE = "pdf"
FIGURE_DIR = Path(__file__).with_name(f"{Path(__file__).stem}_figures")


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


SCRIPT_START_S = time.perf_counter()

START = datetime(2026, 1, 1)

DEPTH_AT_ORIGIN_M = 300.0
Y_GRADIENT = -0.08
TARGET_DEPTH_M = 80.0

SAMPLING_RATE_HZ = 8000
FREQ_MIN_HZ, FREQ_MAX_HZ = 1000.0, 3000.0
SOURCE_LEVEL_DB = 200.0
TARGET_STRENGTH_DB = 15.0
LISTEN_DURATION_S = 4.0
NOISE_LEVEL_DB = 60.0
NOISE_SEED = 7  # the same noise in every configuration's run
CALIBRATION_SEED = 11  # independent noise for the CFAR calibration ping
TARGET_PFA = 1e-7
CENTRE_HZ = (FREQ_MIN_HZ + FREQ_MAX_HZ) / 2

REFERENCE_MODE = "3D all"

# Convergence check: how far a detection may move from one refinement level to the next.
CONVERGED_RANGE_M = 0.05  # an eighth of the pulse's 0.39 m range resolution
CONVERGED_BEARING_DEG = 0.05

# %%
# Ship and bow array
# ------------------
#
# The ship is at the origin heading +x, across the slope. Its bow array is the dome of the
# bistatic tracking example: 6 m below the 5 m deep reference point, facing ahead, with sound
# from behind an element's face 20 dB down. It transmits from its reference point and receives
# on every element.

platform = SonarPlatform(
    states=[GroundTruthState(StateVector([0.0, 3.0, 0.0, 0.0, -5.0, 0.0]), timestamp=START)],
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
    baffle_attenuation_db=20.0,
    mounting_offset=StateVector([0.0, 0.0, -6.0]),
)
platform.add_sensor(bow)
bow_state = bow.get_platform_state_at(START)
bow_xyz = np.ravel(bow_state.array.ref_state_vector).astype(float)
report(
    "Bow array",
    f"{bow.num_sensors} elements on a forward-facing dome. Bellhop runs and ranges are measured "
    f"from its reference point at {np.round(bow_xyz, 1)} m.",
)

# %%
# Targets
# -------
#
# Three static targets sit 80 m deep ahead of the ship, at ranges 500 m apart so that their
# echo trains don't overlap. Ranges and bearings are from the bow array's reference point, with
# bearings anticlockwise from +x.
#
# - **T1**: 1.5 km dead ahead, looking straight across the slope.
# - **T2**: 2 km at 30 degrees to port, towards deeper water.
# - **T3**: 2.5 km at 30 degrees to starboard, towards shallower water.

TARGETS = {"T1": (1500.0, 0.0), "T2": (2000.0, 30.0), "T3": (2500.0, -30.0)}


def point_at(origin_xyz, range_m, bearing_deg, depth_m=TARGET_DEPTH_M):
    """Return the point at a horizontal range and bearing from ``origin_xyz``."""
    bearing = np.radians(bearing_deg)
    return np.array(
        [
            origin_xyz[0] + range_m * np.cos(bearing),
            origin_xyz[1] + range_m * np.sin(bearing),
            -depth_m,
        ]
    )


targets_xyz = {name: point_at(bow_xyz, r, b) for name, (r, b) in TARGETS.items()}
targets = {
    name: GroundTruthPath(
        [
            GroundTruthState(
                StateVector([xyz[0], 0.0, xyz[1], 0.0, xyz[2], 0.0]),
                timestamp=START,
                metadata={"position_mapping": [0, 2, 4]},
            )
        ]
    )
    for name, xyz in targets_xyz.items()
}

# %%
# Water, seabed and models
# ------------------------
#
# The sound speed follows the Munk profile: 1548 m/s at the surface, falling to 1523 m/s at
# 300 m. Sound bends towards lower speed, so the rays curve downwards, and paths near the
# surface run out with range: beyond a couple of kilometres a shallow target can be in the
# direct path's shadow, reached only off the seabed.
#
# Two seabeds are compared: a sloped one, chosen with ``SLOPED``, and a flat one at 300 m as
# the control.
#
# - **Wedge**: 300 m deep at the origin, deepening by 8 m per 100 m towards +y. It slopes
#   *across* any bearing near ±x and *along* the bearings ±y, the same everywhere.
# - **Seamount**: a rounded seamount rising from a 300 m plateau to 150 m. Its slope points
#   away from the summit, so it changes direction across the scene, and a bottom echo can
#   bounce off sides facing different ways.
#
# ``MODES`` lists the six configurations, and ``model`` builds one over a given seabed, all
# with the same lossless bottom.

ssp = Munk()
# The seabed that isn't flat. Every figure and the convergence check use it; set it to
# "Wedge" or "Seamount".
SLOPED = "Wedge"
SLOPED_SEABEDS = {
    "Wedge": WedgeBathymetry(
        depth_at_origin=-DEPTH_AT_ORIGIN_M, x_gradient=0.0, y_gradient=Y_GRADIENT
    ),
    # A rounded (Gaussian) seamount rising from the 300 m plateau to 150 m, merging into the
    # plateau by 1.5 km from its summit; its steepest slope, 500 m out, is 18%. A cone of the
    # same size would have a sharp summit and a crease at its base, where the slope jumps and
    # ray models don't converge. Its summit sits between the paths to T1 and T2, so their
    # bottom echoes bounce off sides that slope in different directions, while T3 is mostly
    # over the plateau. The finer grid is for RTRS, which takes the seabed as a grid; at the
    # default 1 km it would barely see the seamount.
    "Seamount": SeamountBathymetry(
        summit_position=(1200.0, 700.0, -150.0),
        radius=1500.0,
        plateau_depth=-DEPTH_AT_ORIGIN_M,
        shape="gaussian",
        resolution=50.0,
    ),
}
seabeds = {SLOPED: SLOPED_SEABEDS[SLOPED], "Flat": FlatBathymetry(depth=-DEPTH_AT_ORIGIN_M)}
BOTTOM = dict(
    bottom_sound_speed_mps=1700.0,
    bottom_density_g_cm3=1.5,
    bottom_attenuation_db_per_wavelength=0.0,
)
# RTRS takes the same lossless bottom in its own form. Its defaults are set for long ranges: here
# its azimuth search spans 6 degrees in 0.1 degree steps, so it catches bottom echoes turned
# sideways by the slope; its elevations are 0.25 degrees apart; and its sound-speed table has a
# sample every 5 m of depth, to follow the Munk profile.
RTRS = dict(
    bottom_model={
        "model": "acoustic",
        "compressional_speed_m_s": 1700.0,
        "density_g_cm3": 1.5,
        "compressional_attenuation_db_per_wavelength": 0.0,
    },
    water_density_g_cm3=1.0,
    ssp_resolution=(5000.0, 5000.0, 5.0),
    azimuth_search_width=6.0,
    azimuth_resolution=0.1,
    elevation_range=(-70.0, 70.0),
    elevation_resolution=0.25,
    step_m=5.0,
    integration_method="rk2",
)
MODES = {
    "2D all": (BellhopAcousticPropagationModel, {}),
    "2D separate planes": (BellhopAcousticPropagationModel, {"separate_planes": True}),
    "2D far field": (BellhopAcousticPropagationModel, {"far_field": True}),
    "3D all": (Bellhop3DAcousticPropagationModel, {}),
    "3D far field": (Bellhop3DAcousticPropagationModel, {"far_field": True}),
    "Nx2D": (Bellhop3DAcousticPropagationModel, {"nx2d": True}),
    # "RTRS": (rtrsAcousticPropagationModel, RTRS),  # left out for now: not converged here
}
OTHER_MODES = [mode for mode in MODES if mode != REFERENCE_MODE]

# Echo delays are shown as range, as a sonar would show them: with the sound speed at the bow
# array. The reference echoes and the detections are converted with the same speed, so it
# cancels when they are compared.
SOUND_SPEED_MPS = float(ssp.calculate(bow_xyz[2]))


# Ray and grid settings for each model class at refinement level 0, the coarsest. Both Bellhop
# models join the seabed points curvilinearly (their default), which a curved seabed needs, and
# take their default 50 seabed points, a quarter of the default elevation beams and four times
# 3D's default bearing spacing. Each level above refines every one of them at once: twice the
# seabed points and elevation beams and, in 3D, half the bearing spacing, so level 2 has the
# default beams and spacing. "Convergence check" picks each configuration's level over each
# seabed, and every ping after it uses that level.
BASE_SETTINGS = {
    BellhopAcousticPropagationModel: {"n_bathy_samples": 50, "num_beams": 500},
    Bellhop3DAcousticPropagationModel: {
        "n_bathy_samples": 50,
        "num_beams": 500,
        "bearing_beam_spacing_deg": 0.4,
    },
}
MAX_LEVEL = 4  # 800 seabed points, 8000 beams, 0.025 deg: 256 times level 0's rays in 3D
# Before the convergence check (for the reference echoes): 200 seabed points, 2000 beams, 0.1 deg
DEFAULT_LEVEL = 2
# {(seabed, mode): {setting: level}}, from the convergence check; DEFAULT_LEVEL until then
LEVELS = {}
SHORT_NAMES = {
    "n_bathy_samples": "seabed",
    "num_beams": "beams",
    "bearing_beam_spacing_deg": "bearing",
}


def settings_at(model_class, level):
    """Return ``model_class``'s settings at refinement ``level``.

    ``level`` is one level for every setting, or ``{setting: level}``.
    """
    base = BASE_SETTINGS.get(model_class, {})
    levels = level if isinstance(level, dict) else dict.fromkeys(base, level)
    return {
        name: value / 2 ** levels[name]
        if name.endswith("_deg")
        else int(round(value * 2 ** levels[name]))
        for name, value in base.items()
    }


def model(mode, seabed, level=None):
    """Return configuration ``mode`` over ``seabed``, at refinement ``level``.

    ``level`` is one level for every setting, or ``{setting: level}``. Without it, the levels
    over ``seabed`` from the convergence check (``DEFAULT_LEVEL`` before it).
    """
    model_class, settings = MODES[mode]
    bottom = BOTTOM if issubclass(model_class, BellhopAcousticPropagationModel) else {}
    level = LEVELS.get((seabed, mode), DEFAULT_LEVEL) if level is None else level
    return model_class(
        ssp=ssp,
        bathymetry=seabeds[seabed],
        **bottom,
        **{**settings, **settings_at(model_class, level)},
    )


RUNS = [(seabed, mode) for seabed in seabeds for mode in MODES]


def platform_state(xyz):
    """Return a platform state whose array is the points ``xyz``, shape ``(3, n)``."""
    reference = StateVector(xyz.mean(axis=1))
    return PlatformState(
        timestamp=START,
        host=HostState(
            state=GroundTruthState(reference, timestamp=START),
            heading_rad=0.0,
            position_mapping=(0, 1, 2),
        ),
        array=ArrayState(num_sensors=xyz.shape[1], state_vector=xyz, ref_state_vector=reference),
    )


def point_source(xyz):
    """Return a source state at ``xyz``, for a propagation model."""
    return GroundTruthState(
        StateVector(xyz), timestamp=START, metadata={"position_mapping": [0, 1, 2]}
    )


# %%
# Plan view
# ---------
#
# The shading shows the water depth of the sloped seabed, in 50 m steps; for the wedge, the
# arrow points down the slope. The markers are the bow array and the targets.

grid_m = np.linspace(-3000.0, 3000.0, 61)
depth_m = np.array([[-seabeds[SLOPED].get_depth(x, y) for x in grid_m] for y in grid_m])
figure = go.Figure(
    go.Contour(
        x=grid_m,
        y=grid_m,
        z=depth_m,
        colorscale="Blues",
        opacity=0.6,
        contours={
            "start": 50.0 * np.floor(depth_m.min() / 50.0),
            "end": 50.0 * np.ceil(depth_m.max() / 50.0),
            "size": 50.0,
        },
        line={"color": "white", "width": 1},
        colorbar={"title": {"text": "Depth (m)", "side": "right"}, "len": 0.8},
        showlegend=False,
        hoverinfo="skip",
    )
)
figure.add_trace(
    go.Scatter(
        x=[xyz[0] for xyz in targets_xyz.values()],
        y=[xyz[1] for xyz in targets_xyz.values()],
        mode="markers+text",
        text=[f"<b>{name}</b>" for name in targets_xyz],
        textposition="middle right",
        textfont={"size": 14, "color": "black"},
        marker={
            "symbol": "diamond",
            "size": 14,
            "color": "crimson",
            "line": {"width": 1.5, "color": "white"},
        },
        name="Targets",
    )
)
figure.add_trace(
    go.Scatter(
        x=[bow_xyz[0]],
        y=[bow_xyz[1]],
        mode="markers",
        marker={
            "symbol": "star",
            "size": 18,
            "color": "gold",
            "line": {"width": 2, "color": "black"},
        },
        name="Bow array (transmits and receives)",
    )
)
if SLOPED == "Wedge":
    figure.add_annotation(
        x=-2400.0,
        y=2300.0,
        ax=-2400.0,
        ay=-2300.0,
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        showarrow=True,
        arrowhead=2,
        arrowwidth=2,
        arrowcolor="black",
        text="",
    )
    figure.add_annotation(
        x=-2400.0,
        y=0.0,
        text="deeper",
        textangle=-90,
        showarrow=False,
        xshift=-14,
        font={"size": 13, "color": "black"},
    )
figure.update_layout(
    template="plotly_white",
    title=f"Plan View of the {SLOPED}",
    xaxis={"title": "x (m)", "range": [-3000.0, 3000.0]},
    yaxis={"title": "y (m)", "range": [-3000.0, 3000.0], "scaleanchor": "x"},
    legend={"orientation": "h", "yanchor": "top", "y": -0.12, "xanchor": "center", "x": 0.5},
    height=750,
)
save_figure(figure, "plan_view")

# %%
# Reference echoes
# ----------------
#
# 3D all traces the eigenrays from the bow array's reference point to each target. Each ray
# leg is direct (D), off the surface (S) or off the seabed (B), from its bounce counts; rays
# with more bounces are weaker and later, and aren't named. Several neighbouring beams usually
# report parts of one path, so each leg's delay and bearing are their amplitude-weighted mean.
#
# An echo goes out on one leg and back on another. Out and back are interchangeable for a
# monostatic sonar, so DS and SD arrive together, as do DB and BD, and SB and BS; each pair is
# named by one of them. The bow array is only 11 m deep, so a surface bounce adds under about
# 1.2 m of path: DD, DS and SS arrive as one cluster, as do DB and SB. A surface bounce flips the
# sign, and DS and SD add, so the cluster's strongest member is usually DS, not DD. An echo
# that needs a leg with no path, such as the direct path into a shadow zone, isn't there.
#
# The beams are time-aligned to the reference point, so an echo's reference delay is the sum
# of its two legs' delays from the reference point, and its bearing is the bearing its ray
# leaves the reference point at (by reciprocity, the bearing it arrives back from). DB and SB
# arrive from two directions at once, so they have no single bearing.

ECHOES = ("DD", "DS", "SS", "DB", "SB", "BB")
CLUSTERS = {"DD + DS + SS": ("DD", "DS", "SS"), "DB + SB": ("DB", "SB"), "BB": ("BB",)}
SINGLE_BEARING_ECHOES = ("DD", "DS", "SS", "BB")
LEGS = {(0, 0): "D", (1, 0): "S", (0, 1): "B"}  # (surface, bottom) bounces
SAME_PATH_S = 0.5e-3  # arrivals within this of a leg's strongest are parts of the same path


def reference_legs(name, seabed):
    """Return ``{leg: (delay_s, bearing_deg)}`` from 3D all's eigenrays to a target."""
    rays = model(REFERENCE_MODE, seabed).compute_eigenrays(bow_xyz, targets_xyz[name], CENTRE_HZ)
    legs = {}
    for bounces, leg in LEGS.items():
        group = [r for r in rays if (r.n_surface_bounces, r.n_bottom_bounces) == bounces]
        if not group:
            continue
        strongest = max(group, key=lambda r: abs(r.amplitude))
        group = [r for r in group if abs(r.delay_s - strongest.delay_s) < SAME_PATH_S]
        weights = np.abs([r.amplitude for r in group])
        legs[leg] = (
            float(weights @ [r.delay_s for r in group] / weights.sum()),
            float(weights @ [r.src_azimuth_deg for r in group] / weights.sum()),
        )
    return legs


start = time.perf_counter()
legs = {(name, seabed): reference_legs(name, seabed) for seabed in seabeds for name in targets}
reference_s = time.perf_counter() - start


def reference_delays_s(name, seabed):
    """Return each echo's reference delay, out and back, for the echoes that have a path."""
    leg = legs[name, seabed]
    return {
        echo: leg[echo[0]][0] + leg[echo[1]][0]
        for echo in ECHOES
        if echo[0] in leg and echo[1] in leg
    }


def reference_ranges_m(name, seabed):
    """Return each echo's reference range, its delay at the sonar's sound speed."""
    return {
        echo: SOUND_SPEED_MPS * delay / 2
        for echo, delay in reference_delays_s(name, seabed).items()
    }


def reference_bearings_deg(name, seabed):
    """Return the reference bearing of each echo that arrives from a single direction."""
    leg = legs[name, seabed]
    return {
        echo: (leg[echo[0]][1] + leg[echo[1]][1]) / 2
        for echo in reference_delays_s(name, seabed)
        if echo in SINGLE_BEARING_ECHOES
    }


def clusters_of(name, seabed):
    """Return ``{cluster: echoes}`` for the echoes a target has, leaving out empty clusters."""
    present = reference_delays_s(name, seabed)
    clusters = {c: tuple(e for e in echoes if e in present) for c, echoes in CLUSTERS.items()}
    return {cluster: echoes for cluster, echoes in clusters.items() if echoes}


def group_range_m(name, seabed, echoes):
    """Return the mean reference range of an echo group."""
    ranges = reference_ranges_m(name, seabed)
    return float(np.mean([ranges[e] for e in echoes]))


report(
    "Reference echoes (3D all)",
    "Each echo's reference range in metres, relative to the target's earliest echo, with its "
    "reference bearing relative to the target's true bearing in brackets where it has one. A "
    "dash means one of its legs has no path. A target with no named echoes is reached only by "
    "paths with more bounces, and is left out of the comparisons.",
)
print(f"\n{'Seabed':>6} {'Target':>6}", *(f"{echo:>16}" for echo in ECHOES))
for seabed in seabeds:
    for name in targets:
        ranges = reference_ranges_m(name, seabed)
        bearings = reference_bearings_deg(name, seabed)
        if not ranges:
            print(f"{seabed:>6} {name:>6}  no named echoes")
            continue
        first = min(ranges.values())
        cells = []
        for echo in ECHOES:
            if echo not in ranges:
                cells.append(f"{'-':>16}")
                continue
            cell = f"{ranges[echo] - first:.2f}"
            if echo in bearings:
                cell += f" ({bearings[echo] - TARGETS[name][1]:+.2f})"
            cells.append(f"{cell:>16}")
        print(f"{seabed:>6} {name:>6}", *cells)

# %%
# Echoes
# ------
#
# The bow array pings from its reference point and listens on every element, with
# delay-and-sum beams every degree from 60 degrees to port to 60 degrees to starboard. One ping
# is simulated with every configuration over both seabeds. The propagation model's time is
# measured separately from the rest of the ping, since the echo synthesis and beamforming are
# the same for all of them, and is split in two:
#
# - **Bellhop runs**: running ``bellhopcxx`` and reading its arrivals.
# - **Transfer functions**: turning each element's arrivals into a transfer function over the
#   simulator's frequency grid. With every element traced, that is one complex exponential per
#   arrival, frequency and element, and it takes far longer than the Bellhop runs. Far field
#   builds the reference point's transfer function once and shifts it to each element.
#
# The beams are kept as matched-filter envelopes: matched-filtering a signal's analytic version
# gives the echo's smooth outline, free of the oscillation at the pulse's carrier frequency.
# Delays are shown as range, :math:`r = c\tau / 2` for a monostatic sonar, where :math:`\tau`
# is the echo delay and :math:`c` the sound speed at the bow array.
#
# Each ping also keeps what its propagation model returned for the comparison with 3D all:
# every target's transfer function out to the target and back to each element. Their product
# is the echo's round-trip transfer function on each element. Turned back into time over the
# pulse's band, it is each element's echo response; a short window of it is kept around each
# echo cluster's reference delay.

pulse = LFMSignal(
    freq_min_hz=FREQ_MIN_HZ,
    freq_max_hz=FREQ_MAX_HZ,
    rise_time_s=0.03,
    duration_s=0.5,
    sampling_rate_hz=SAMPLING_RATE_HZ,
    source_level_db=SOURCE_LEVEL_DB,
)


def simulator(propagation_model, ground_truth_paths=None, seed=NOISE_SEED, noise=True):
    """Return a monostatic simulator: the bow array pings and listens, with seeded noise.

    With ``noise=False`` the ping holds the echoes alone.
    """
    return ActiveSonarSimulator(
        array=bow,
        propagation_model=propagation_model,
        signal=pulse,
        scattering_models=ConstantTargetStrength(target_strength_db=TARGET_STRENGTH_DB),
        ground_truth_paths=list(targets.values())
        if ground_truth_paths is None
        else ground_truth_paths,
        listen_duration_s=LISTEN_DURATION_S,
        noise_model=WhiteNoiseSignal(
            amplitude_upa=10 ** (NOISE_LEVEL_DB / 20),
            duration_s=LISTEN_DURATION_S,
            sampling_rate_hz=SAMPLING_RATE_HZ,
            seed=seed,
        )
        if noise
        else None,
        beamformer=DelayAndSumBeamformer(domain="frequency", sampling_rate_hz=SAMPLING_RATE_HZ),
        steering_calculator=SteeringCalculator(
            ssp=ssp, steering_sector_rad=(np.radians(-60.0), np.radians(60.0)), num_beams=121
        ),
        ping_timestamps=[START],
    )


def envelope_power(signals, replica):
    """Return the matched-filter envelope power of each row of ``signals``."""
    analytic = hilbert(np.real(np.atleast_2d(signals)), axis=1)
    return MatchedFilter().filter(analytic, replica) ** 2


def add_timer(propagation_model, method, times, key):
    """Wrap ``propagation_model.method`` so that its wall-clock time is added to ``times[key]``.

    2D separate planes makes its Bellhop runs several at a time, so the clock runs while any
    call is in progress, and overlapping calls are counted once.
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
    """Return ``propagation_model``, timing its whole calls and its Bellhop runs within them.

    RTRS has no separate ray-tracing step to time: it traces and builds its transfer functions
    in one call.
    """
    add_timer(propagation_model, "propagate_spectrum", propagation_s, key)
    if hasattr(propagation_model, "_run_arrivals"):
        add_timer(propagation_model, "_run_arrivals", bellhop_s, key)
    return propagation_model


def seconds(value, width):
    """Return ``value`` in seconds, right-aligned to ``width``, or a dash if it is None."""
    return f"{'-':>{width}}" if value is None else f"{value:>{width}.1f}"


WINDOW_MARGIN_S = 1.5e-3  # three times the pulse's 0.5 ms resolution
WINDOW_STEP_S = 0.1e-3


def record_transfer_functions(propagation_model, calls):
    """Wrap ``propagation_model.propagate_spectrum`` to append each call's result to ``calls``."""
    call = propagation_model.propagate_spectrum

    def recording_call(platform, source, frequencies_hz):
        transfer_functions, delays_s = call(platform, source, frequencies_hz)
        calls.append((np.asarray(frequencies_hz, dtype=float), np.asarray(transfer_functions)))
        return transfer_functions, delays_s

    propagation_model.propagate_spectrum = recording_call


def echo_responses(calls, seabed):
    """Return ``{target: {cluster: response}}``, each element's echo response near the cluster.

    ``calls`` are the propagation model's results in the simulator's order: for each target,
    the leg out to it (one receiver) and then the leg back to every element. Each response
    has shape ``(num_elements, num_times)``, sampled every ``WINDOW_STEP_S`` from just before
    the cluster's earliest reference delay to just after its latest.
    """
    responses = {}
    for name, (out, back) in zip(targets, zip(calls[::2], calls[1::2], strict=True), strict=True):
        frequencies_hz, transfer_out = out
        positive = frequencies_hz > 0
        round_trip = back[1][:, positive] * transfer_out.reshape(-1)[positive]
        delays = reference_delays_s(name, seabed)
        responses[name] = {}
        for cluster, echoes in clusters_of(name, seabed).items():
            times_s = np.arange(
                min(delays[e] for e in echoes) - WINDOW_MARGIN_S,
                max(delays[e] for e in echoes) + WINDOW_MARGIN_S,
                WINDOW_STEP_S,
            )
            basis = np.exp(2j * np.pi * np.outer(frequencies_hz[positive], times_s))
            responses[name][cluster] = round_trip @ basis
    return responses


# %%
# Detection
# ---------
#
# :class:`~bluepebble.detector.ActiveSonarDetector` matched-filters each beam and runs an
# ordered-statistic CFAR (Constant False Alarm Rate) detector along range, calibrated for a
# false alarm probability of :math:`10^{-7}` on a noise-only ping with its own, independent
# noise. Its noise estimate is the 300th smallest of the 400 training cells around each cell
# (200 each side, beyond 40 guard cells, so from about 4 m to 23 m away). A cell-averaging CFAR
# would average in any echo that falls in those cells: T3, in the shallower water, has its
# echoes only about 11 to 15 m apart, so each would raise the threshold for its neighbours.
# Ranking the cells ignores the few dozen an echo occupies.
# Each detection is a bearing (its beam) and a range. A target is detected on several
# neighbouring beams at once, so detections within 1 m of range of each other are grouped,
# keeping the beam with the strongest envelope and refining its bearing with a parabola through
# the neighbouring beams, as in the bistatic tracking example.
#
# The detector's matched filter works on the real signal, so its output still oscillates at the
# carrier frequency, and each detection's range sits on one of those oscillations rather than
# on the envelope's peak.

cfar = OSCFARDetector(
    num_guard_cells=40,
    num_training_cells=200,
    rank=300,
    target_pfa=TARGET_PFA,
    circular=False,
    peak_distance=20,
)
start = time.perf_counter()
noise_only = simulator(model("2D all", "Flat"), ground_truth_paths=[], seed=CALIBRATION_SEED)
NoiseCalibrator(cfar).calibrate_from_noise(
    matched_filter_scans_from_sensor_data(noise_only.sensor_data_gen())
)
calibration_s = time.perf_counter() - start


REFINE_BEAMS = 5


def detect(ping, seabed):
    """Return each target's grouped detections in ``ping``: ``[(range_m, bearing_deg)]``."""
    power = envelope_power(ping.beamformed_data, ping.transmit_pulse)
    beam_bearings_deg = np.degrees(np.asarray(ping.steering_bearings_rad))
    range_step_m = SOUND_SPEED_MPS / (2 * SAMPLING_RATE_HZ)
    detector = ActiveSonarDetector(
        detector=cfar,
        sensor_data_gen=[(ping.timestamp, {ping})],
        sampling_rate_hz=SAMPLING_RATE_HZ,
        sound_speed_ms=SOUND_SPEED_MPS,
        min_range_m=200.0,
    )
    _, detections = next(iter(detector.detections_gen()))
    found = {name: [] for name in targets}
    for detection in detections:
        bearing_deg = np.degrees(float(detection.state_vector[0, 0]))
        detected_range_m = float(detection.state_vector[1, 0])
        for name in targets:
            if not reference_ranges_m(name, seabed):
                continue
            low = min(reference_ranges_m(name, seabed).values()) - 2.0
            high = max(reference_ranges_m(name, seabed).values()) + 3.0
            if low < detected_range_m < high and abs(bearing_deg - TARGETS[name][1]) < 15.0:
                beam = int(np.argmin(np.abs(beam_bearings_deg - bearing_deg)))
                sample = int(round(detected_range_m / range_step_m))
                found[name].append((float(power[beam, sample]), detected_range_m, sample, beam))
    grouped = {}
    for name, candidates in found.items():
        kept = []
        for _, detected_range_m, sample, detected_beam in sorted(candidates, reverse=True):
            if any(abs(detected_range_m - r) < 1.0 for r, _ in kept):
                continue
            # The strongest beam near the detection's own: across the whole sector, an edge
            # beam can briefly outshine the target's beams between the echo's lobes.
            across = power[:, sample]
            near = slice(max(detected_beam - REFINE_BEAMS, 0), detected_beam + REFINE_BEAMS + 1)
            beam = near.start + int(np.argmax(across[near]))
            offset = (
                parabolic_peak_offset(10 * np.log10(across), beam)
                if 0 < beam < across.size - 1
                else 0.0
            )
            step_deg = beam_bearings_deg[1] - beam_bearings_deg[0]
            kept.append((detected_range_m, beam_bearings_deg[beam] + offset * step_deg))
        grouped[name] = sorted(kept)
    return grouped


MATCH_M = 0.75


def matched_detections(found, seabed, name):
    """Return ``{echo: (range_m, bearing_deg)}``, each of a target's echoes' detection.

    Each detection goes to the target's reference echo nearest it in range, if one is within
    ``MATCH_M``, and each echo keeps the detection nearest it.
    """
    ranges = reference_ranges_m(name, seabed)
    best = {}
    for r, b in found if ranges else []:
        echo = min(ranges, key=lambda e: abs(r - ranges[e]))
        error = abs(r - ranges[echo])
        if error < MATCH_M and (echo not in best or error < abs(best[echo][0] - ranges[echo])):
            best[echo] = (r, b)
    return best


# %%
# Convergence check
# -----------------
#
# Each configuration's settings are checked before it pings for the comparison, over each
# seabed in turn, on what the comparison measures: the detections' ranges and bearings. Each
# ping goes through the same simulator, noise, beamformer and CFAR detector as the comparison,
# with the same noise every time, so the only difference between two pings is the settings.
#
# Two pings agree when every detection in one has a partner in the other, the detection of the
# same target nearest it in range (within ``MATCH_M``), and none moves by more than
# ``CONVERGED_RANGE_M`` in range or ``CONVERGED_BEARING_DEG`` in bearing. Each row is labelled
# with the reference echo the detection matches, but the pairing is by range, so a detection
# that moves from nearer one echo to nearer its neighbour is still the same detection.
#
# The check has two passes:
#
# 1. **Together.** Every setting is refined at once, level by level from level 0 (the
#    coarsest), until a level agrees with the next. That level has converged.
# 2. **One at a time.** Each setting in turn is then coarsened on its own, one level at a time,
#    and its ping compared with the converged level's. It keeps the coarsest level that still
#    agrees, and the next setting is tested with it in place, so the final combination is the
#    last ping that agreed. The pass shows which settings the detections depend on, and gives
#    each configuration its cheapest converged settings.
#
# The configuration pings at the levels found from then on over that seabed. One still
# changing at ``MAX_LEVEL`` pings there, is reported as not converged and skips the second
# pass.
#
# The reference echoes above come from 3D all at ``DEFAULT_LEVEL``: they only name the
# echoes, which a refinement doesn't change.


def detected_positions(mode, seabed, level):
    """Return ``{target: {echo: (range_m, bearing_deg)}}`` from one ping at ``level``."""
    _, data = next(iter(simulator(model(mode, seabed, level=level)).sensor_data_gen()))
    found = detect(next(iter(data)), seabed)
    return {name: matched_detections(found[name], seabed, name) for name in targets}


def paired(one, other):
    """Return ``[(echo, one's detection or None, other's or None)]``, paired by range."""
    pairs = sorted(
        (abs(r - other_r), echo, other_echo)
        for echo, (r, _) in one.items()
        for other_echo, (other_r, _) in other.items()
        if abs(r - other_r) < MATCH_M
    )
    out, used, other_used = [], set(), set()
    for _, echo, other_echo in pairs:
        if echo not in used and other_echo not in other_used:
            used.add(echo)
            other_used.add(other_echo)
            out.append((echo, one[echo], other[other_echo]))
    out += [(echo, one[echo], None) for echo in one.keys() - used]
    out += [(echo, None, other[echo]) for echo in other.keys() - other_used]
    return sorted(out, key=lambda row: ECHOES.index(row[0]))


def detections_agree(seabed, mode, test, reference, trial):
    """Print how far each detection moved from ``reference`` to ``trial``; return if all held."""
    agree = True
    for name in targets:
        for echo, ref, new in paired(reference[name], trial[name]):
            row = f"{seabed:>6} {mode:>18} {test:>10} {name:>6} {echo:>5}"
            if ref is None or new is None:
                agree = False
                print(f"{row}  {'gained' if ref is None else 'lost'} a detection", flush=True)
                continue
            moved = (new[0] - ref[0], new[1] - ref[1])
            ok = abs(moved[0]) <= CONVERGED_RANGE_M and abs(moved[1]) <= CONVERGED_BEARING_DEG
            agree &= ok
            print(
                f"{row} {moved[0]:>+10.3f} {moved[1]:>+14.3f}" + ("" if ok else "  not converged"),
                flush=True,
            )
    return agree


report(
    "Convergence check",
    "Over each seabed, how far each detection moves between two pings, in range (m) and "
    "bearing (deg). Test 'a-b' refines every setting together from level a to b, finer minus "
    "coarser; 'beams 1' and the like coarsen one setting to that level, minus the converged "
    "ping. Each row is labelled with the reference echo its detection matches. Two pings agree "
    f"when every detection is in both and moves by no more than {CONVERGED_RANGE_M:g} m and "
    f"{CONVERGED_BEARING_DEG:g} deg.",
)
print(
    f"\n{'Seabed':>6} {'Configuration':>18} {'Test':>10} {'Target':>6} {'Echo':>5}"
    f" {'Range (m)':>10} {'Bearing (deg)':>14}"
)
start = time.perf_counter()
converged = {}
for seabed, mode in RUNS:
    names = list(BASE_SETTINGS.get(MODES[mode][0], {}))
    # Pass 1: every setting together.
    positions = detected_positions(mode, seabed, 0)
    joint, converged[seabed, mode] = MAX_LEVEL, False
    for level in range(MAX_LEVEL):
        finer = detected_positions(mode, seabed, level + 1)
        if detections_agree(seabed, mode, f"{level}-{level + 1}", positions, finer):
            joint, converged[seabed, mode] = level, True
            break
        positions = finer
    levels = dict.fromkeys(names, joint)
    # Pass 2: each setting on its own, against the converged ping.
    if converged[seabed, mode]:
        for name in names:
            while levels[name] > 0:
                trial = {**levels, name: levels[name] - 1}
                test = f"{SHORT_NAMES[name]} {trial[name]}"
                trial_positions = detected_positions(mode, seabed, trial)
                if not detections_agree(seabed, mode, test, positions, trial_positions):
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
    print(f"{seabed:>6} {mode:>18}: {described or 'no settings'}; {verdict}")

# %%
# Pings
# -----
#
# Every configuration now pings over both seabeds, with noise, each at the level the
# convergence check chose for it over that seabed.


pings, beam_power, propagation_s, bellhop_s, ping_s = {}, {}, {}, {}, {}
responses = {}
report(
    "One ping per configuration",
    "Each line is one simulated ping: the propagation model out to the targets and back, echo "
    f"synthesis on all {bow.num_sensors} elements, and beamforming. Bellhop runs and Transfer "
    "functions are the propagation model's two parts, Model their sum, and Ping the whole "
    "ping; the rest of the ping is the same work for every configuration. RTRS does both "
    "parts in one call, so only its Model time is given.",
)
print(
    f"\n{'Seabed':>6} {'Configuration':>18} {'Bellhop runs (s)':>17}"
    f" {'Transfer functions (s)':>23} {'Model (s)':>10} {'Ping (s)':>9}"
)
for seabed, mode in RUNS:
    calls = []
    propagation_model = timed(model(mode, seabed), (seabed, mode))
    record_transfer_functions(propagation_model, calls)
    start = time.perf_counter()
    _, data = next(iter(simulator(propagation_model).sensor_data_gen()))
    ping = next(iter(data))
    ping_s[seabed, mode] = time.perf_counter() - start
    responses[seabed, mode] = echo_responses(calls, seabed)
    band_frequencies_hz = calls[0][0]
    runs_s, model_s = bellhop_s.get((seabed, mode)), propagation_s[seabed, mode]
    print(
        f"{seabed:>6} {mode:>18} {seconds(runs_s, 17)}"
        f" {seconds(None if runs_s is None else model_s - runs_s, 23)} {model_s:>10.1f}"
        f" {ping_s[seabed, mode]:>9.1f}"
    )
    pings[seabed, mode] = ping
    beam_power[seabed, mode] = envelope_power(ping.beamformed_data, ping.transmit_pulse)
beam_bearings_deg = np.degrees(np.asarray(ping.steering_bearings_rad))
range_m = (
    np.arange(beam_power[SLOPED, "3D all"].shape[1]) * SOUND_SPEED_MPS / (2 * SAMPLING_RATE_HZ)
)
RANGE_STEP_M = range_m[1] - range_m[0]


detections = {run: detect(pings[run], run[0]) for run in RUNS}

# %%
# Detections against the reference echoes
# ---------------------------------------
#
# Each detection is assigned to the target's reference echo nearest it in range, if one is
# within 0.75 m, and each echo keeps the detection nearest it. The table gives each echo's
# range offset from its reference range, positive for late; a dash means no detection was
# assigned to it. Not every echo is expected to be found: in each cluster the weaker members
# merge into the strongest. The two bearing columns are the bearing offsets of the
# direct-cluster echo (DD, DS or SS) found nearest its reference range, and of BB, from their
# reference bearings.
#
# 3D all's own row isn't zero. Its offsets are what the beamformer and detector add: the
# detector's range lands on a carrier oscillation, and the beams look horizontally, so a steep
# echo off the bottom appears slightly nearer dead ahead than it is. The other configurations
# are compared with 3D all's detections, not with its reference echoes, so that the processing
# cancels.


def matched_errors(seabed, mode, name):
    """Return ``{echo: (range_offset_m, bearing_offset_deg or None)}`` for a target's echoes."""
    ranges = reference_ranges_m(name, seabed)
    bearings = reference_bearings_deg(name, seabed)
    matched = matched_detections(detections[seabed, mode][name], seabed, name)
    return {
        echo: (r - ranges[echo], b - bearings[echo] if echo in bearings else None)
        for echo, (r, b) in matched.items()
    }


report(
    "Detections against the reference echoes",
    "Each echo's range offset in metres from its reference range (positive is late); a dash "
    "means no detection was assigned to it. DS includes SD, DB includes BD and SB includes BS. "
    "Direct bearing is the bearing offset, in degrees, of the direct-cluster echo (DD, DS or SS) "
    "found nearest its reference range, and BB bearing that of BB.",
)
print(
    f"\n{'Seabed':>6} {'Target':>6} {'Configuration':>18}",
    *(f"{echo + ' (m)':>8}" for echo in ECHOES),
    f"{'Direct bearing (deg)':>21} {'BB bearing (deg)':>17}",
)
for seabed, mode in RUNS:
    for name in targets:
        errors = matched_errors(seabed, mode, name)
        ranges = [f"{errors[e][0]:>8.2f}" if e in errors else f"{'-':>8}" for e in ECHOES]
        direct = [errors[e] for e in CLUSTERS["DD + DS + SS"] if e in errors]
        direct_bearing = min(direct, key=lambda x: abs(x[0]))[1] if direct else None
        bearing_cells = [
            f"{'-':>21}" if direct_bearing is None else f"{direct_bearing:>21.2f}",
            f"{errors['BB'][1]:>17.2f}" if "BB" in errors else f"{'-':>17}",
        ]
        print(f"{seabed:>6} {name:>6} {mode:>18}", *ranges, *bearing_cells)

# %%
# Difference from 3D all
# ----------------------
#
# Each other configuration is compared with 3D all at two levels, per seabed.
#
# **Propagation.** Each element's echo response near each echo cluster, from the transfer
# functions the ping's propagation model returned, against 3D all's. The difference is the
# energy of the difference over the energy of 3D all's response, in dB, summed over the
# elements, the window and the three targets. Below about -20 dB the two agree closely; 0 dB
# means the difference is as large as the echo itself, as when an echo arrives a fraction of a
# wavelength (0.5-1.5 m here) off. This is the propagation model alone, with no noise,
# beamformer or detector.
#
# **Detections.** Each configuration's detections against 3D all's, echo by echo, for the
# echoes both found:
#
# - **Range** and **bearing**: the root mean square difference from 3D all's detection. Bearing
#   is compared only for the echoes that arrive from a single direction (DD, DS, SS and BB).
# - **Position**: range and bearing combined, as the distance in metres between the two
#   detections. A bearing difference becomes a cross-range one, the echo's range times the
#   difference in radians. At 2 km, 0.1 degrees is 3.5 m across range, so this is dominated by
#   bearing.
# - **Echoes**: how many echoes both found, how many only this configuration found, and how
#   many only 3D all found.
#
# **Convergence.** The convergence check above measures 3D all's precision in echo positions;
# here it is measured on the same scale as the differences, so that differences smaller than
# the reference's own can be recognised. Over the sloped seabed 3D all is rerun with half the
# bearing beam spacing of its chosen level there, and its echo responses compared with the
# normal run's.

REFERENCE_SETTINGS = settings_at(Bellhop3DAcousticPropagationModel, LEVELS[SLOPED, REFERENCE_MODE])
CONVERGENCE_SPACING_DEG = REFERENCE_SETTINGS["bearing_beam_spacing_deg"] / 2
start = time.perf_counter()
fine_model = Bellhop3DAcousticPropagationModel(
    ssp=ssp,
    bathymetry=seabeds[SLOPED],
    **BOTTOM,
    **{**REFERENCE_SETTINGS, "bearing_beam_spacing_deg": CONVERGENCE_SPACING_DEG},
)
calls = []
for xyz in targets_xyz.values():
    for receiver, source in (
        (platform_state(xyz.reshape(3, 1)), point_source(bow_xyz)),
        (bow_state, point_source(xyz)),
    ):
        transfer_functions, _ = fine_model.propagate_spectrum(
            receiver, source, band_frequencies_hz
        )
        calls.append((band_frequencies_hz, np.asarray(transfer_functions)))
fine_responses = echo_responses(calls, SLOPED)
convergence_s = time.perf_counter() - start
CONVERGENCE_LABEL = f"3D all, {CONVERGENCE_SPACING_DEG:g} deg beams"


def response_difference_db(mine, theirs, clusters=tuple(CLUSTERS)):
    """Return the energy of ``mine - theirs`` over that of ``theirs``, in dB, over ``clusters``."""
    difference = reference = 0.0
    for name in targets:
        for cluster, response in theirs[name].items():
            if cluster in clusters:
                difference += float(np.sum(np.abs(mine[name][cluster] - response) ** 2))
                reference += float(np.sum(np.abs(response) ** 2))
    if reference == 0.0:
        return np.nan  # no target has an echo in these clusters
    return 10 * np.log10(difference / reference)


def detection_difference(seabed, mode):
    """Return a configuration's detection differences from 3D all's, as a dict."""
    range_diffs, bearing_diffs, position_diffs = [], [], []
    both = only_mine = only_reference = 0
    for name in targets:
        mine = matched_errors(seabed, mode, name)
        theirs = matched_errors(seabed, REFERENCE_MODE, name)
        ranges = reference_ranges_m(name, seabed)
        both += len(mine.keys() & theirs.keys())
        only_mine += len(mine.keys() - theirs.keys())
        only_reference += len(theirs.keys() - mine.keys())
        for echo in mine.keys() & theirs.keys():
            range_diff = mine[echo][0] - theirs[echo][0]
            range_diffs.append(range_diff)
            if mine[echo][1] is not None:
                bearing_diff = mine[echo][1] - theirs[echo][1]
                bearing_diffs.append(bearing_diff)
                position_diffs.append(
                    np.hypot(range_diff, ranges[echo] * np.radians(bearing_diff))
                )

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


propagation_difference_db = {
    (seabed, mode): {
        cluster: response_difference_db(
            responses[seabed, mode], responses[seabed, REFERENCE_MODE], (cluster,)
        )
        for cluster in CLUSTERS
    }
    | {"All": response_difference_db(responses[seabed, mode], responses[seabed, REFERENCE_MODE])}
    for seabed in seabeds
    for mode in OTHER_MODES
}
propagation_difference_db[SLOPED, CONVERGENCE_LABEL] = {
    cluster: response_difference_db(fine_responses, responses[SLOPED, REFERENCE_MODE], (cluster,))
    for cluster in CLUSTERS
} | {"All": response_difference_db(fine_responses, responses[SLOPED, REFERENCE_MODE])}
detection_differences = {
    (seabed, mode): detection_difference(seabed, mode)
    for seabed in seabeds
    for mode in OTHER_MODES
}
model_s = {(seabed, mode): propagation_s[seabed, mode] for seabed, mode in RUNS}
model_s[SLOPED, CONVERGENCE_LABEL] = convergence_s

report(
    "Propagation: difference from 3D all",
    "Per seabed, each configuration's echo-response difference from 3D all's, in dB (lower is "
    "closer), for each echo cluster and all of them together, with its model time, sorted by "
    f"time. The last {SLOPED.lower()} row is 3D all traced with half its bearing beam spacing: "
    "how far the reference itself is from converged.",
)
print(
    f"\n{'Seabed':>6} {'Configuration':>26}",
    *(f"{cluster + ' (dB)':>17}" for cluster in (*CLUSTERS, "All")),
    f"{'Model (s)':>10}",
)
for seabed, mode in sorted(propagation_difference_db, key=lambda run: (run[0], model_s[run])):
    differences = propagation_difference_db[seabed, mode]
    print(
        f"{seabed:>6} {mode:>26}",
        *(f"{differences[cluster]:>17.1f}" for cluster in (*CLUSTERS, "All")),
        f"{model_s[seabed, mode]:>10.1f}",
    )

report(
    "Detections: difference from 3D all",
    "Per seabed, the root mean square difference of each configuration's detections from 3D "
    "all's, over the echoes both found, with how many echoes both found, only it found and only "
    "3D all found, and its model time, sorted by time.",
)
print(
    f"\n{'Seabed':>6} {'Configuration':>18} {'Range (m)':>10} {'Bearing (deg)':>14}"
    f" {'Position (m)':>13} {'Both':>5} {'Only it':>8} {'Only 3D all':>12} {'Model (s)':>10}"
)
for seabed in seabeds:
    for mode in sorted(OTHER_MODES, key=lambda m: model_s[seabed, m]):
        d = detection_differences[seabed, mode]
        print(
            f"{seabed:>6} {mode:>18} {d['range_m']:>10.2f} {d['bearing_deg']:>14.2f}"
            f" {d['position_m']:>13.2f} {d['both']:>5} {d['only_mine']:>8}"
            f" {d['only_reference']:>12} {model_s[seabed, mode]:>10.1f}"
        )

# Both figures put one measure in each panel against configuration, cheapest first over the
# sloped seabed, with the two seabeds side by side. The convergence run appears only over the
# sloped seabed.

SEABED_STYLES = {SLOPED: ("#2a78d6", "circle"), "Flat": ("#eb6834", "diamond")}
mode_order = sorted(OTHER_MODES, key=lambda m: model_s[SLOPED, m])


def configuration_figure(panels, categories, title):
    """Return a row of panels, each ``(label, values(seabed, category), log_y)``."""
    figure = make_subplots(
        rows=1,
        cols=len(panels),
        subplot_titles=[label.rsplit(" (", 1)[0] for label, _, _ in panels],
        horizontal_spacing=0.06,
    )
    for col, (label, values, log_y) in enumerate(panels, start=1):
        for seabed, (colour, symbol) in SEABED_STYLES.items():
            figure.add_trace(
                go.Scatter(
                    x=categories,
                    y=[values(seabed, category) for category in categories],
                    mode="markers",
                    marker={
                        "symbol": symbol,
                        "size": 12,
                        "color": colour,
                        "line": {"width": 2, "color": "white"},
                    },
                    name=seabed,
                    legendgroup=seabed,
                    offsetgroup=seabed,
                    showlegend=col == 1,
                    hovertemplate=f"{seabed}, %{{x}}<br>{label}: %{{y:.3g}}<extra></extra>",
                ),
                row=1,
                col=col,
            )
        figure.update_yaxes(title_text=label, type="log" if log_y else None, row=1, col=col)
        figure.update_xaxes(tickangle=-30, row=1, col=col)
    figure.update_layout(
        template="plotly_white",
        title=title,
        scattermode="group",
        legend={"title": {"text": "Seabed"}},
        height=520,
        width=1500,
    )
    return figure


def propagation_value(cluster):
    """Return a function giving a run's echo-response difference for ``cluster``, or None."""

    def value(seabed, category):
        run = propagation_difference_db.get((seabed, category))
        return None if run is None else run[cluster]

    return value


figure = configuration_figure(
    [
        *(
            (f"{cluster} difference (dB)", propagation_value(cluster), False)
            for cluster in CLUSTERS
        ),
        ("Model time (s)", lambda seabed, category: model_s.get((seabed, category)), True),
    ],
    [*mode_order, CONVERGENCE_LABEL],
    "Propagation: Echo-Response Difference from 3D All, and Model Time",
)
save_figure(figure, "propagation_difference_from_3d_all")

figure = configuration_figure(
    [
        (
            f"{label} difference ({unit})",
            lambda seabed, category, key=key: detection_differences[seabed, category][key],
            False,
        )
        for label, unit, key in (
            ("Range", "m", "range_m"),
            ("Bearing", "deg", "bearing_deg"),
            ("Position", "m", "position_m"),
        )
    ],
    mode_order,
    "Detections: RMS Difference from 3D All",
)
save_figure(figure, "detection_difference_from_3d_all")

# %%
# Beamformer output and detections
# --------------------------------
#
# Each panel shows the matched-filtered beams around one target's echoes, as bearing against range:
# what the sonar's display would show, with the grouped detections marked (red crosses). The top
# row is 2D separate planes and the bottom row 3D all, both over the sloped seabed; the tables
# above cover every configuration. The dashed line marks the target's true bearing and the dotted
# lines the reference range of every named echo, labelled by cluster. The members of a cluster are
# under 1.2 m apart, so at this scale their lines nearly coincide; they overlap in phase-inverted
# copies that split each cluster into lobes in range, and the detector often finds more than one.

PLOT_MODES = {"2D separate planes": "steelblue", "3D all": "crimson"}


def bearing_window(name):
    """Return the beams within 10 degrees of a target's true bearing."""
    return np.abs(beam_bearings_deg - TARGETS[name][1]) <= 10.0 + 1e-6


figure = make_subplots(
    rows=2,
    cols=len(targets),
    subplot_titles=[f"{name}, {mode}" for mode in PLOT_MODES for name in targets],
    shared_yaxes=True,
    horizontal_spacing=0.04,
    vertical_spacing=0.14,
)
for row, mode in enumerate(PLOT_MODES, start=1):
    for col, name in enumerate(targets, start=1):
        ranges = reference_ranges_m(name, SLOPED)
        if not ranges:
            figure.add_annotation(
                text="no named echoes",
                x=0.5,
                y=0.5,
                xref="x domain",
                yref="y domain",
                showarrow=False,
                row=row,
                col=col,
            )
            continue
        span = (range_m > min(ranges.values()) - 2) & (range_m < max(ranges.values()) + 3)
        beams = bearing_window(name)
        image = beam_power[SLOPED, mode][np.ix_(beams, span)]
        figure.add_trace(
            go.Heatmap(
                x=range_m[span],
                y=beam_bearings_deg[beams] - TARGETS[name][1],
                z=10 * np.log10(image / image.max()),
                zmin=-30,
                zmax=0,
                colorscale="Viridis",
                colorbar={"title": {"text": "dB re peak", "side": "right"}, "len": 0.8},
                showscale=row == col == 1,
            ),
            row=row,
            col=col,
        )
        found = detections[SLOPED, mode][name]
        figure.add_trace(
            go.Scatter(
                x=[r for r, _ in found],
                y=[b - TARGETS[name][1] for _, b in found],
                mode="markers",
                marker={"symbol": "x", "size": 10, "color": "red"},
                name="Detections",
                showlegend=row == col == 1,
            ),
            row=row,
            col=col,
        )
        for cluster, echoes in clusters_of(name, SLOPED).items():
            for echo in echoes:
                figure.add_vline(
                    x=ranges[echo],
                    line_dash="dot",
                    line_color="white",
                    line_width=1,
                    row=row,
                    col=col,
                )
            # One label per cluster, at the top of the panel. BB is near the right edge, so its
            # label hangs to the left of its line.
            figure.add_annotation(
                x=ranges[echoes[0]],
                y=1.0,
                yref="y domain",
                yanchor="top",
                xanchor="right" if cluster == "BB" else "left",
                text="/".join(echoes),
                showarrow=False,
                font={"color": "white", "size": 11},
                bgcolor="rgba(0, 0, 0, 0.6)",
                borderpad=2,
                row=row,
                col=col,
            )
        figure.add_hline(y=0.0, line_dash="dash", line_color="white", row=row, col=col)
        figure.update_xaxes(title_text="Range (m)", row=row, col=col)
    figure.update_yaxes(title_text="Bearing − true (deg)", row=row, col=1)
figure.update_yaxes(range=[-10.5, 10.5])
figure.update_layout(
    template="plotly_white",
    title=f"Beamformer Output (matched-filtered beams) and Detections over the {SLOPED}",
    legend={"orientation": "h", "yanchor": "bottom", "y": 1.04, "xanchor": "right", "x": 1.0},
    height=750,
    width=1150,
)
save_figure(figure, "beamformer_output")

# %%
# Beam response against bearing
# -----------------------------
#
# The heatmap shows where the echoes are; this figure shows the bearing estimate. For each echo
# cluster, every beam's strongest matched-filter output within 1.5 m of the cluster's mean
# reference range is plotted against the beam's bearing, in dB relative to the stronger of the two
# configurations drawn. The mainlobe's peak is the beamformer's estimate of the bearing the
# echoes arrive from (markers, refined between beams with a parabola), and its width is the
# bearing resolution: two targets closer than that at the same range merge. The dashed line is
# the target's true bearing. A panel is empty when the target has no echo in that cluster.
#
# The beams look horizontally, so an echo arriving steeply off the bottom appears slightly
# closer to dead ahead than it is; and over a sloped seabed a bottom reflection can arrive from off
# the target's bearing, which only 3D models.

ECHO_HALF_WINDOW_M = 1.5
BEARING_HALF_SPAN_DEG = 25.0
figure = make_subplots(
    rows=len(CLUSTERS),
    cols=len(targets),
    subplot_titles=[f"{name} {cluster}" for cluster in CLUSTERS for name in targets],
    shared_yaxes=True,
    horizontal_spacing=0.05,
    vertical_spacing=0.1,
)
for row, cluster in enumerate(CLUSTERS, start=1):
    for col, name in enumerate(targets, start=1):
        true_deg = TARGETS[name][1]
        echoes = clusters_of(name, SLOPED).get(cluster)
        if echoes is None:
            continue
        window = np.abs(range_m - group_range_m(name, SLOPED, echoes)) < ECHO_HALF_WINDOW_M
        beams = np.abs(beam_bearings_deg - true_deg) <= BEARING_HALF_SPAN_DEG + 1e-6
        response = {mode: beam_power[SLOPED, mode][:, window].max(axis=1) for mode in PLOT_MODES}
        peak = max(float(r[beams].max()) for r in response.values())
        for mode, colour in PLOT_MODES.items():
            response_db = 10 * np.log10(response[mode] / peak)
            figure.add_trace(
                go.Scatter(
                    x=beam_bearings_deg[beams],
                    y=response_db[beams],
                    mode="lines",
                    name=mode,
                    showlegend=row == col == 1,
                    line={"color": colour, "width": 2},
                ),
                row=row,
                col=col,
            )
            beam = int(np.argmax(np.where(beams, response_db, -np.inf)))
            offset = (
                parabolic_peak_offset(response_db, beam)
                if 0 < beam < response_db.size - 1
                else 0.0
            )
            step_deg = beam_bearings_deg[1] - beam_bearings_deg[0]
            figure.add_trace(
                go.Scatter(
                    x=[beam_bearings_deg[beam] + offset * step_deg],
                    y=[response_db[beam]],
                    mode="markers",
                    marker={"symbol": "diamond", "size": 9, "color": colour},
                    name=f"{mode} estimate",
                    showlegend=row == col == 1,
                ),
                row=row,
                col=col,
            )
        figure.add_vline(x=true_deg, line_dash="dash", line_color="black", row=row, col=col)
        if row == len(CLUSTERS):
            figure.update_xaxes(title_text="Bearing (deg)", row=row, col=col)
    figure.update_yaxes(title_text="dB re peak", row=row, col=1)
figure.update_yaxes(range=[-30.0, 1.0])
figure.update_layout(
    template="plotly_white",
    title=f"Beam Response against Bearing for Each Echo over the {SLOPED} (dashed: true bearing)",
    legend={"orientation": "h", "yanchor": "bottom", "y": 1.04, "xanchor": "right", "x": 1.0},
    height=900,
    width=1150,
)
save_figure(figure, "beam_response")

# %%
# Compute time
# ------------
#
# The compute time of every configuration, on the machine that ran the script. For each ping,
# Bellhop is the time running
# ``bellhopcxx``, Model the whole propagation model including the transfer functions, and Ping
# the whole ping, including the echo synthesis and beamforming that every configuration shares.

report(
    "Compute time",
    "Seconds per configuration on this machine: for each seabed's ping, the Bellhop runs, the "
    "whole propagation model and the whole ping. A dash means the configuration has no "
    "separate Bellhop step (RTRS).",
)
print(
    f"\n{'Configuration':>18}",
    *(
        f"{seabed + ' ' + part + ' (s)':>18}"
        for seabed in seabeds
        for part in ("Bellhop", "model", "ping")
    ),
)
for mode in MODES:
    cells = [
        seconds(times.get((seabed, mode)), 18)
        for seabed in seabeds
        for times in (bellhop_s, propagation_s, ping_s)
    ]
    print(f"{mode:>18}", *cells)
print(f"\nConvergence check (detections, level by level): {convergence_check_s:.1f} s")
print(f"CFAR calibration (noise-only ping): {calibration_s:.1f} s")
print(f"Reference eigenrays (3D all, 6 runs): {reference_s:.1f} s")
print(
    f"Convergence check (3D all, {CONVERGENCE_SPACING_DEG:g} deg beams, {SLOPED.lower()}):"
    f" {convergence_s:.1f} s"
)
print(f"Whole script, including figures: {time.perf_counter() - SCRIPT_START_S:.1f} s")

# %%
# Summary
# -------
#
# - 3D all is the reference: the most complete ray model, checked against a finer-traced run
#   of itself. The other configurations are judged by how far they are from it, not from an
#   exact answer, which a refracting profile doesn't have.
# - The propagation comparison is the model alone. It shows which echoes each cheaper
#   configuration gets wrong and by how much; the direct cluster should agree everywhere except
#   far field, and the bottom echoes over the sloped seabed show what tracing in vertical planes
#   misses.
# - The detection comparison is what that does to the sonar's output, after the beamformer
#   and detector: differences small next to the pulse's 0.4 m range resolution and the beams'
#   width don't matter to a tracker; larger ones, or echoes found by one and not the other, do.
# - The far-field configurations are the cheapest for each model; 3D's cost is set mostly by
#   its bearing beam spacing and the width of its bearing fan.
