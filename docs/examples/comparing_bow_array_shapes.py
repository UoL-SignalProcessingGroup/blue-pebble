r"""
=========================
Comparing Bow Array Shapes
=========================

A :class:`~bluepebble.arrays.BowArray` lays its sensors on one of two surfaces, and the shape
decides which way each sensor faces.

- **Spherical**: a forward-facing patch of a sphere. The hull behind the dome baffles every
  sensor, so all of them face forward.
- **Cylindrical**: a ring of vertical staves (a stave is a vertical column of sensors) around a
  vertical axis, as on most surface-ship hull sonars. Each stave faces outwards, so the staves on
  the far side of the cylinder from a contact are the baffled ones.

Both arrays here have a 0.82 m radius and sensors at most 0.131 m apart, which is within half a
wavelength up to 5.7 kHz. The cylinder is 1.44 m high. Its full ring has

.. math::

    n = \left\lceil \frac{2 \pi r}{d} \right\rceil = 40

staves at azimuths :math:`\theta_k = 2 \pi k / n` from boresight, each with
:math:`\lceil h / d \rceil = 11` sensors, 440 in all. An ``azimuth_extent_rad`` below
:math:`2 \pi` keeps the staves within half of it of boresight, at the same positions.

Assumptions
-----------
- Staves are straight, vertical and identical, with evenly spaced rows.
- A sensor hears a direction at full gain within 90 degrees of its normal, and at the baffle
  level beyond it. There is no other element directivity.
"""  # noqa: D205, D212, D400, D415
# sphinx_gallery_thumbnail_path = "_static/thumbnails/comparing_bow_array_shapes.png"

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
from stonesoup.types.groundtruth import GroundTruthState

from bluepebble.arrays import BowArray
from bluepebble.platform import SonarPlatform

# %%
# Arrays
# ------
#
# The ship steams along +x, so the array frame (+x forward, +y to port, +z up) lines up with the
# plot axes. Both arrays are mounted at the ship's reference point.

START = datetime(2026, 1, 1)
RADIUS_M = 0.82
SPACING_M = 0.131
SENSOR_SIZE_M = 0.1
FREQ_MAX_HZ = 5000.0
BAFFLE_ATTENUATION_DB = 20.0

platform = SonarPlatform(
    states=[GroundTruthState(StateVector([0.0, 3.0, 0.0, 0.0, -5.0, 0.0]), timestamp=START)],
    position_mapping=[0, 2, 4],
    transition_models=[CombinedLinearGaussianTransitionModel([ConstantVelocity(0.0)] * 3)],
    transition_times=[timedelta(hours=1)],
)
arrays = {
    "Spherical": BowArray(
        dome_radius_m=RADIUS_M,
        azimuth_extent_rad=np.radians(120.0),
        elevation_extent_rad=np.radians(60.0),
        sensor_spacing_m=SPACING_M,
        sensor_size_m=SENSOR_SIZE_M,
        freq_max_hz=FREQ_MAX_HZ,
        baffle_attenuation_db=BAFFLE_ATTENUATION_DB,
    ),
    "Cylindrical": BowArray(
        shape="cylindrical",
        dome_radius_m=RADIUS_M,
        azimuth_extent_rad=2 * np.pi,
        height_m=1.44,
        sensor_spacing_m=SPACING_M,
        sensor_size_m=SENSOR_SIZE_M,
        freq_max_hz=FREQ_MAX_HZ,
        baffle_attenuation_db=BAFFLE_ATTENUATION_DB,
    ),
}
for array in arrays.values():
    platform.add_sensor(array)

states = {name: array.get_platform_state_at(START).array for name, array in arrays.items()}
for name, state in states.items():
    print(f"{name}: {state.num_sensors} sensors")

# %%
# Layout and normals
# ------------------
#
# Each sensor is drawn with its normal, the direction it faces, with x forward, y to port and z
# up. The dome's normals are parallel. The cylinder's point outwards along the radius and have no
# vertical part.

SENSOR_COLOUR = "#636EFA"
NORMAL_COLOUR = "#7F7F7F"
NORMAL_LENGTH_M = 0.25


def sensor_offsets(state):
    """Return sensor positions relative to the array's reference point, shape (3, N)."""
    return state.state_vector - np.asarray(state.ref_state_vector)


def normal_segments(state):
    """Return one line segment per sensor along its normal, separated by gaps."""
    start = sensor_offsets(state)
    end = start + NORMAL_LENGTH_M * np.asarray(state.sensor_normals)
    return np.stack([start, end, np.full_like(start, np.nan)], axis=2).reshape(3, -1)


figure = make_subplots(
    rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]], subplot_titles=list(states)
)
for col, state in enumerate(states.values(), start=1):
    x, y, z = sensor_offsets(state)
    nx, ny, nz = normal_segments(state)
    figure.add_trace(
        go.Scatter3d(
            x=nx,
            y=ny,
            z=nz,
            mode="lines",
            line={"color": NORMAL_COLOUR, "width": 2},
            name="Normal",
            legendgroup="normal",
            showlegend=col == 1,
            hoverinfo="skip",
        ),
        row=1,
        col=col,
    )
    figure.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers",
            marker={"color": SENSOR_COLOUR, "size": 3},
            name="Sensor",
            legendgroup="sensor",
            showlegend=col == 1,
        ),
        row=1,
        col=col,
    )
scene = {
    "aspectmode": "data",
    "xaxis_title": "x (m)",
    "yaxis_title": "y (m)",
    "zaxis_title": "z (m)",
    "camera": {"eye": {"x": 1.6, "y": -1.6, "z": 0.9}},
}
figure.update_layout(
    template="plotly_white", title="Sensor Layout and Normals", scene=scene, scene2=scene
).show()

# %%
# Seen from above
# ---------------
#
# From above, each stave of the cylinder is one point, and the dome's rows sit at different
# radii. The dome covers a 120 degree arc ahead of the ship. The cylinder's staves go all the way
# round, 9 degrees apart.

figure = make_subplots(rows=1, cols=2, subplot_titles=list(states))
for col, state in enumerate(states.values(), start=1):
    x, y, _ = sensor_offsets(state)
    nx, ny, _ = normal_segments(state)
    figure.add_trace(
        go.Scatter(
            x=nx,
            y=ny,
            mode="lines",
            line={"color": NORMAL_COLOUR, "width": 1},
            name="Normal",
            legendgroup="normal",
            showlegend=col == 1,
            hoverinfo="skip",
        ),
        row=1,
        col=col,
    )
    figure.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker={"color": SENSOR_COLOUR, "size": 8},
            name="Sensor",
            legendgroup="sensor",
            showlegend=col == 1,
        ),
        row=1,
        col=col,
    )
    figure.update_xaxes(title_text="x (m), forward", range=[-1.2, 1.2], row=1, col=col)
    figure.update_yaxes(title_text="y (m), to port", range=[-1.2, 1.2], row=1, col=col)
figure.update_yaxes(scaleanchor="x", row=1, col=1)
figure.update_yaxes(scaleanchor="x2", row=1, col=2)
figure.update_layout(template="plotly_white", title="Sensor Layout Seen from Above").show()

# %%
# Baffling
# --------
#
# A sensor's gain towards a contact is 1 when the contact is within 90 degrees of its normal, and
# the baffle level, here 20 dB down, otherwise. For a contact dead astern, the whole dome is
# baffled. The cylinder hears the contact on its aft-facing staves only.

astern = [-1.0, 0.0, 0.0]
for name, state in states.items():
    hearing = int(np.sum(state.sensor_gains(astern) == 1.0))
    print(f"{name}: {hearing} of {state.num_sensors} sensors face a contact dead astern")
