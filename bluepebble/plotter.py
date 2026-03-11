"""Defines plotting utilities."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

import cmocean
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from numpy.typing import ArrayLike
from plotly.subplots import make_subplots
from scipy import signal as scipy_signal
from stonesoup.platform.base import Platform
from stonesoup.types.detection import Detection
from stonesoup.types.groundtruth import GroundTruthPath
from stonesoup.types.track import Track

from .detector.metrics import SweepResult


def _distance_axis_scale(min_val: float, max_val: float) -> tuple[float, str]:
    """Return distance scale factor and unit label from axis limits.

    Parameters
    ----------
    min_val : float
        Minimum value of the axis (e.g., minimum x or y coordinate in metres).
    max_val : float
        Maximum value of the axis (e.g., maximum x or y coordinate in metres).

    Notes
    -----
    Input coordinates are assumed to be metres.

    """
    max_abs = max(abs(min_val), abs(max_val))
    if max_abs >= 1e3:
        return 1e-3, "km"
    return 1.0, "m"


def _range_padding_for_scale(scale: float, span: float) -> float:
    """Return axis padding in native units from display scale and scene span.

    Padding is 5% of span in display units with a floor of 0.5 display units.

    Parameters
    ----------
    scale : float
        Display scale factor (e.g., 1e-3 for km if input is in m).
    span : float
        Span of the scene in native units (e.g., metres).

    Returns
    -------
    float
        Padding in native units to add to axis limits for display purposes.

    """
    span_display_units = span * scale
    pad_display_units = max(0.5, 0.05 * span_display_units)
    return pad_display_units / scale


def _expand_heatmap_coords(coords: ArrayLike) -> np.ndarray:
    """Expand outer heatmap coordinates by half a cell width.

    Plotly heatmaps render against the supplied coordinate centres. Nudging the first and
    last centres outward by half a cell helps the rendered bathymetry visually fill the
    intended scene bounds.
    """
    array = np.asarray(coords, dtype=float)
    if array.ndim != 1 or array.size < 2:
        return array

    expanded = array.copy()
    expanded[0] -= (array[1] - array[0]) / 2.0
    expanded[-1] += (array[-1] - array[-2]) / 2.0
    return expanded


def _mpl_cmap_to_plotly(cmap: object, n: int = 256) -> list[list[float | str]]:
    """Convert a Matplotlib colormap to Plotly colorscale format."""
    vals = np.linspace(0.0, 1.0, n)
    rgba = cmap(vals)
    return [
        [float(v), f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})"]
        for v, (r, g, b, _) in zip(vals, rgba, strict=False)
    ]


def _two_slope_colorscale(
    cmap: object,
    zmin: float,
    zmax: float,
    vcenter: float = 0.0,
    n: int = 256,
) -> list[list[float | str]]:
    """Create a Plotly colorscale with a fixed midpoint in data space."""
    if not np.isfinite(zmin) or not np.isfinite(zmax) or zmax <= zmin:
        return _mpl_cmap_to_plotly(cmap, n=n)

    t0 = float(np.clip((vcenter - zmin) / (zmax - zmin), 0.001, 0.999))
    n_lo = max(2, int(n * t0))
    n_hi = max(2, n - n_lo)
    colorscale: list[list[float | str]] = []

    for i, c in enumerate(np.linspace(0.0, 0.5, n_lo)):
        pos = t0 * i / (n_lo - 1)
        r, g, b, _ = cmap(c)
        colorscale.append(
            [round(float(pos), 6), f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})"]
        )

    for i, c in enumerate(np.linspace(0.5, 1.0, n_hi)):
        if i == 0:
            continue
        pos = t0 + (1.0 - t0) * i / (n_hi - 1)
        r, g, b, _ = cmap(c)
        colorscale.append(
            [round(float(pos), 6), f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})"]
        )

    return colorscale


def _validate_non_empty_1d(array_like: ArrayLike, name: str) -> np.ndarray:
    """Validate that input is a non-empty one-dimensional sequence."""
    array = np.asarray(array_like)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D sequence")
    if array.size == 0:
        raise ValueError(f"{name} is empty")
    return array


def _validate_btr_shapes(
    timesteps: ArrayLike,
    steering_azimuths: ArrayLike,
    data: ArrayLike | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Validate BTR array dimensions and compatibility."""
    timesteps_array = _validate_non_empty_1d(timesteps, "timesteps")
    steering_array = _validate_non_empty_1d(steering_azimuths, "steering_azimuths")

    try:
        steering_array = steering_array.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("steering_azimuths must contain numeric values") from exc

    if data is None:
        return timesteps_array, steering_array, None

    data_array = np.asarray(data)
    if data_array.ndim != 2:
        raise ValueError("data must be a 2D array")

    expected_shape = (timesteps_array.size, steering_array.size)
    if data_array.shape != expected_shape:
        raise ValueError(f"data shape mismatch: expected {expected_shape}, got {data_array.shape}")
    return timesteps_array, steering_array, data_array


def _validate_spectrogram_params(
    sr: int,
    n_fft: int,
    hop_length: int,
    y_lim: tuple[float, float] | None,
    yaxis_format: str,
) -> tuple[str, tuple[float, float] | None]:
    """Validate spectrogram parameters and normalise axis-format input."""
    if sr <= 0:
        raise ValueError("sr must be positive")
    if n_fft <= 0:
        raise ValueError("n_fft must be positive")
    if hop_length <= 0:
        raise ValueError("hop_length must be positive")
    if hop_length > n_fft:
        raise ValueError("hop_length must be less than or equal to n_fft")

    if not isinstance(yaxis_format, str):
        raise ValueError("yaxis_format must be one of {'kHz', 'Hz'}")

    normalised_format = yaxis_format.strip().lower()
    if normalised_format == "khz":
        canonical_format = "kHz"
    elif normalised_format == "hz":
        canonical_format = "Hz"
    else:
        raise ValueError("yaxis_format must be one of {'kHz', 'Hz'}")

    if y_lim is None:
        normalised_y_lim = None
    else:
        if not isinstance(y_lim, (tuple, list, np.ndarray)) or len(y_lim) != 2:
            raise ValueError("y_lim must be a (low, high) pair")
        low, high = float(y_lim[0]), float(y_lim[1])
        if not np.isfinite(low) or not np.isfinite(high):
            raise ValueError("y_lim values must be finite")
        if low >= high:
            raise ValueError("y_lim must satisfy low < high")
        normalised_y_lim = (low, high)

    return canonical_format, normalised_y_lim


def _normalise_plotly_figsize(figsize: tuple[float, float]) -> tuple[int, int]:
    """Normalise a requested figure size to Plotly pixel dimensions.

    For historical compatibility, small values are interpreted as inches and converted
    using 100 px/in. Larger values are assumed to already be pixels.
    """
    if len(figsize) != 2:
        raise ValueError("figsize must be a (width, height) pair")

    width_raw = float(figsize[0])
    height_raw = float(figsize[1])
    if width_raw <= 0 or height_raw <= 0:
        raise ValueError("figsize values must be positive")

    # Matplotlib-style defaults like (12, 6) should map to sensible Plotly pixels.
    if max(width_raw, height_raw) <= 40:
        width_raw *= 100.0
        height_raw *= 100.0

    width_px = max(10, int(round(width_raw)))
    height_px = max(10, int(round(height_raw)))
    return width_px, height_px


def launch_bathymetry_and_sound_speed_viewer(
    bathymetry,
    ssp,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    z_res_m: float = 10.0,
    host: str = "127.0.0.1",
    port: int = 8050,
    debug: bool = False,
    jupyter_mode: str | None = None,
) -> None:
    """Launch an interactive bathymetry/profile dashboard for measured environments.

    The dashboard provides:
    - A bathymetry map (Blue Pebble ``-z`` convention) used as a profile selector.
    - A selected-point sound-speed profile plot (depth shown as ``+z`` downward).

    This viewer is designed for measured-data models where bathymetry and SSP vary in
    both horizontal and vertical dimensions. It expects model objects compatible with
    ``GEBCOBathymetry`` and ``LeroyCopernicusSoundSpeedProfile``.

    Parameters
    ----------
    bathymetry : object
        Bathymetry model instance supporting ``get_grid(x_range, y_range)`` and exposing
        local x/y coverage arrays.
    ssp : object
        Sound-speed model instance supporting Leroy/Copernicus cached fields and
        interpolation helpers.
    x_range : tuple[float, float] | None
        Optional x-range in meters for the viewer. If ``None``, the overlap between
        bathymetry and SSP coverage is used.
    y_range : tuple[float, float] | None
        Optional y-range in meters for the viewer. If ``None``, the overlap between
        bathymetry and SSP coverage is used.
    z_res_m : float
        Vertical resolution in meters used to resample SSP profiles.
    host : str
        Dash server host.
    port : int
        Dash server port.
    debug : bool
        Dash debug flag.
    jupyter_mode : str | None
        Optional Dash notebook display mode. Supported values are
        ``{"inline", "tab", "external", "jupyterlab"}``.
        If ``None`` (default), the app runs as a standard local web server.

    Returns
    -------
    None
        Runs the Dash application until the server is stopped.

    """
    if z_res_m <= 0.0:
        raise ValueError("z_res_m must be positive")

    if jupyter_mode is not None:
        if not isinstance(jupyter_mode, str):
            raise ValueError(
                "jupyter_mode must be one of {'inline', 'tab', 'external', 'jupyterlab'} or None"
            )
        jupyter_mode = jupyter_mode.strip().lower()
        allowed_jupyter_modes = {"inline", "tab", "external", "jupyterlab"}
        if jupyter_mode not in allowed_jupyter_modes:
            raise ValueError(
                "jupyter_mode must be one of {'inline', 'tab', 'external', 'jupyterlab'} or None"
            )

    try:
        import importlib

        dash_module = importlib.import_module("dash")
        Dash = dash_module.Dash
        dcc = dash_module.dcc
        html = dash_module.html
        Input = dash_module.Input
        Output = dash_module.Output
        State = dash_module.State
    except ImportError as exc:
        raise ImportError(
            "dash is required for launch_bathymetry_and_sound_speed_viewer. "
            "Install with `pip install dash`."
        ) from exc

    if not hasattr(bathymetry, "_ensure_loaded") or not hasattr(bathymetry, "get_grid"):
        raise TypeError("bathymetry must provide _ensure_loaded() and get_grid(x_range, y_range).")
    if not hasattr(ssp, "_ensure_loaded"):
        raise TypeError("ssp must provide _ensure_loaded().")

    bathymetry._ensure_loaded()
    ssp._ensure_loaded()

    required_ssp_attrs = [
        "_x_m",
        "_y_m",
        "_z_m",
        "_c_zyx",
        "_interp_3d_horizontal",
        "_extrapolate_columns_to_depth",
        "fill_speed_m_s",
    ]
    missing_ssp_attrs = [name for name in required_ssp_attrs if not hasattr(ssp, name)]
    if missing_ssp_attrs:
        raise TypeError(
            "ssp is missing required attributes/methods for measured-data viewing: "
            f"{missing_ssp_attrs}"
        )

    if not hasattr(bathymetry, "_x_m") or not hasattr(bathymetry, "_y_m"):
        raise TypeError("bathymetry must expose _x_m and _y_m coverage arrays.")

    overlap_x_min = max(float(np.min(bathymetry._x_m)), float(np.min(ssp._x_m)))
    overlap_x_max = min(float(np.max(bathymetry._x_m)), float(np.max(ssp._x_m)))
    overlap_y_min = max(float(np.min(bathymetry._y_m)), float(np.min(ssp._y_m)))
    overlap_y_max = min(float(np.max(bathymetry._y_m)), float(np.max(ssp._y_m)))

    if overlap_x_min >= overlap_x_max or overlap_y_min >= overlap_y_max:
        raise ValueError("Bathymetry and SSP domains do not overlap in x/y.")

    if x_range is None:
        x_range = (overlap_x_min, overlap_x_max)
    if y_range is None:
        y_range = (overlap_y_min, overlap_y_max)

    x_range = (float(x_range[0]), float(x_range[1]))
    y_range = (float(y_range[0]), float(y_range[1]))
    if x_range[0] >= x_range[1] or y_range[0] >= y_range[1]:
        raise ValueError("x_range and y_range must be strictly increasing")

    if x_range[0] < overlap_x_min or x_range[1] > overlap_x_max:
        raise ValueError("x_range must lie within overlapping bathymetry/SSP x-domain")
    if y_range[0] < overlap_y_min or y_range[1] > overlap_y_max:
        raise ValueError("y_range must lie within overlapping bathymetry/SSP y-domain")

    x_bty_m, y_bty_m, z_bty_xy_m = bathymetry.get_grid(x_range=x_range, y_range=y_range)
    x_bty_m = np.asarray(x_bty_m, dtype=float)
    y_bty_m = np.asarray(y_bty_m, dtype=float)
    z_bty_xy_m = np.asarray(z_bty_xy_m, dtype=float)

    if z_bty_xy_m.shape != (len(x_bty_m), len(y_bty_m)):
        raise ValueError("bathymetry.get_grid returned unexpected z-grid shape")

    z_bty_xy_m = np.minimum(z_bty_xy_m, 0.0)
    depth_limit_yx_m = np.abs(z_bty_xy_m.T)
    max_depth_m = float(np.nanmax(depth_limit_yx_m))
    if not np.isfinite(max_depth_m) or max_depth_m <= 0.0:
        raise ValueError("Unable to infer positive seabed depths from bathymetry grid")

    num_depth_points = int(np.ceil(max_depth_m / z_res_m)) + 1
    z_profile_m = np.linspace(0.0, max_depth_m, max(2, num_depth_points))

    c_horiz_zyx = ssp._interp_3d_horizontal(
        np.asarray(ssp._c_zyx, dtype=float),
        np.asarray(ssp._x_m, dtype=float),
        np.asarray(ssp._y_m, dtype=float),
        x_bty_m,
        y_bty_m,
    )
    c_profile_zyx = ssp._extrapolate_columns_to_depth(
        c_horiz_zyx,
        np.asarray(ssp._z_m, dtype=float),
        z_profile_m,
        c_fill=float(ssp.fill_speed_m_s),
    )

    water_mask_zyx = z_profile_m[:, None, None] <= depth_limit_yx_m[None, :, :]
    c_profile_zyx = np.where(water_mask_zyx, c_profile_zyx, np.nan)

    z_bty_min = float(np.nanmin(z_bty_xy_m))
    z_bty_max = float(np.nanmax(z_bty_xy_m))
    z_eps = max(1e-9, 1e-6 * max(abs(z_bty_min), abs(z_bty_max), 1.0))
    z_bty_display_min = z_bty_min if z_bty_min < 0.0 else -z_eps
    z_bty_display_max = z_bty_max if z_bty_max > 0.0 else z_eps

    bathymetry_colorscale = _two_slope_colorscale(
        cmocean.cm.topo,
        z_bty_display_min,
        z_bty_display_max,
        vcenter=0.0,
    )

    ix0 = len(x_bty_m) // 2
    iy0 = len(y_bty_m) // 2

    def _nearest_index(values: np.ndarray, target: float) -> int:
        return int(np.argmin(np.abs(values - float(target))))

    def _map_figure(ix: int, iy: int) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(
            go.Heatmap(
                x=x_bty_m,
                y=y_bty_m,
                z=z_bty_xy_m.T,
                colorscale=bathymetry_colorscale,
                zmin=z_bty_display_min,
                zmax=z_bty_display_max,
                colorbar=dict(title=dict(text="Bathymetry z (m)"), thickness=20),
                hovertemplate=(
                    "x=%{x:.1f} m<br>y=%{y:.1f} m<br>Bathymetry z=%{z:.1f} m<extra></extra>"
                ),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[x_bty_m[ix]],
                y=[y_bty_m[iy]],
                mode="markers",
                marker=dict(
                    symbol="star", color="yellow", size=12, line=dict(color="black", width=1)
                ),
                name="Selected",
                hovertemplate="Selected<br>x=%{x:.1f} m<br>y=%{y:.1f} m<extra></extra>",
            )
        )
        fig.update_layout(
            template="plotly_white",
            title=dict(text="Bathymetry Selector", x=0.5),
            xaxis=dict(title="x (m)"),
            yaxis=dict(title="y (m)", scaleanchor="x", scaleratio=1),
            margin=dict(l=40, r=20, t=44, b=40),
            legend=dict(x=0.01, y=0.99),
        )
        return fig

    def _profile_figure(ix: int, iy: int) -> go.Figure:
        profile = c_profile_zyx[:, iy, ix]
        finite_profile = profile[np.isfinite(profile)]
        if finite_profile.size:
            c_min = float(np.nanmin(finite_profile)) - 5.0
            c_max = float(np.nanmax(finite_profile)) + 5.0
        else:
            c_min, c_max = 1450.0, 1550.0

        seabed_depth_m = float(depth_limit_yx_m[iy, ix])

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=profile,
                y=z_profile_m,
                mode="lines",
                line=dict(width=2),
                name="c(z)",
                hovertemplate="c=%{x:.2f} m/s<br>depth=%{y:.1f} m<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[c_min, c_max],
                y=[seabed_depth_m, seabed_depth_m],
                mode="lines",
                line=dict(color="saddlebrown", width=1.5, dash="dot"),
                name="Seabed",
                hovertemplate=f"Seabed depth={seabed_depth_m:.1f} m<extra></extra>",
                showlegend=False,
            )
        )
        fig.update_layout(
            template="plotly_white",
            title=dict(text="Sound Speed Profile", x=0.5),
            xaxis=dict(title="c (m/s)", range=[c_min, c_max]),
            yaxis=dict(title="Depth (+z, m)", autorange="reversed", range=[seabed_depth_m, 0.0]),
            margin=dict(l=40, r=20, t=44, b=40),
            showlegend=False,
        )
        return fig

    app = Dash(__name__)
    app.title = "Bathymetry and Sound Speed Viewer"

    app.layout = html.Div(
        style={"fontFamily": "Arial, sans-serif", "padding": "12px"},
        children=[
            html.H2("Bathymetry and Sound Speed Viewer", style={"textAlign": "center"}),
            html.Div(
                id="info-bar",
                style={"textAlign": "center", "marginBottom": "10px", "fontSize": "13px"},
                children=(
                    "Click any bathymetry point to inspect the local sound speed profile. "
                    "Bathymetry uses Blue Pebble -z; profile depth is shown as +z downward."
                ),
            ),
            dcc.Store(id="selected-indices", data={"ix": ix0, "iy": iy0}),
            html.Div(
                style={"display": "flex", "gap": "10px"},
                children=[
                    dcc.Graph(
                        id="bathymetry-map",
                        style={"flex": "2", "minWidth": "0", "height": "64vh"},
                        config={"scrollZoom": True},
                    ),
                    dcc.Graph(
                        id="ssp-profile",
                        style={"flex": "1", "minWidth": "0", "height": "64vh"},
                    ),
                ],
            ),
            html.Div(
                id="selected-point-label",
                style={"textAlign": "center", "marginTop": "10px", "fontSize": "13px"},
            ),
        ],
    )

    @app.callback(
        Output("selected-indices", "data"),
        Input("bathymetry-map", "clickData"),
        State("selected-indices", "data"),
    )
    def _update_selected_indices(click_data, selected_data):
        if not click_data or "points" not in click_data or len(click_data["points"]) == 0:
            return selected_data
        point = click_data["points"][0]
        if "x" not in point or "y" not in point:
            return selected_data

        ix = _nearest_index(x_bty_m, point["x"])
        iy = _nearest_index(y_bty_m, point["y"])
        return {"ix": ix, "iy": iy}

    @app.callback(Output("bathymetry-map", "figure"), Input("selected-indices", "data"))
    def _update_bathymetry_map(selected_data):
        ix = int(selected_data["ix"])
        iy = int(selected_data["iy"])
        return _map_figure(ix, iy)

    @app.callback(
        Output("ssp-profile", "figure"),
        Output("selected-point-label", "children"),
        Input("selected-indices", "data"),
    )
    def _update_profile(selected_data):
        ix = int(selected_data["ix"])
        iy = int(selected_data["iy"])
        seabed_z_m = float(z_bty_xy_m[ix, iy])
        label = (
            f"Selected point: x={x_bty_m[ix]:.1f} m, y={y_bty_m[iy]:.1f} m, "
            f"bathymetry z={seabed_z_m:.1f} m"
        )
        return _profile_figure(ix, iy), label

    if jupyter_mode is None:
        print(f"\n  Bathymetry and Sound Speed Viewer running -> http://{host}:{port}/\n")
        app.run(host=host, port=int(port), debug=bool(debug))
    else:
        print(
            "\n  Bathymetry and Sound Speed Viewer running in notebook mode "
            f"'{jupyter_mode}' -> http://{host}:{port}/\n"
        )
        app.run(host=host, port=int(port), debug=bool(debug), jupyter_mode=jupyter_mode)


def plot_world(
    truths: Sequence[GroundTruthPath],
    platform: Platform,
    bathymetry: object | None = None,
    figsize: tuple[float, float] = (600, 500),
) -> go.Figure:
    """Plot the world picture of the platform and target trajectories.

    Parameters
    ----------
    truths : Sequence[GroundTruthPath]
        Ground-truth paths representing target trajectories.
    platform : Platform
        The platform whose trajectory is to be plotted.
    bathymetry : object | None
        Optional bathymetry model implementing ``get_grid(x_range, y_range)``.
        If provided, bathymetry is rendered as a background heatmap.
    figsize : tuple[float, float]
        Figure size. Values that look like inches (for example ``(12, 6)``) are
        converted to pixels using 100 px/in; larger values are treated as pixels.
        Default is ``(600, 500)``.

    Returns
    -------
    go.Figure
        A Plotly figure object containing the world picture plot.

    """
    num_truths = len(truths)

    fig = go.Figure()
    fig.update_layout(colorway=px.colors.qualitative.Plotly)
    colorway = list(fig.layout.colorway or px.colors.qualitative.Plotly)
    group_counts = {"platform": 1, "truths": num_truths}
    added_group_titles: set[str] = set()

    def _legend_group_kwargs(group_name: str, group_title: str) -> dict[str, str]:
        """Return legend-group kwargs and show group title only when group has multiple entries."""
        kwargs = {"legendgroup": group_name}
        if group_counts.get(group_name, 0) > 1 and group_name not in added_group_titles:
            kwargs["legendgrouptitle_text"] = group_title
            added_group_titles.add(group_name)
        return kwargs

    if len(platform.platform_history) == 0:
        raise ValueError("platform.platform_history is empty")

    plat_x = [float(entry.host.state.state_vector[0]) for entry in platform.platform_history]
    plat_y = [float(entry.host.state.state_vector[2]) for entry in platform.platform_history]

    gt_x = [[] for _ in range(num_truths)]
    gt_y = [[] for _ in range(num_truths)]
    for idx, truth in enumerate(truths):
        gt_x[idx] = [float(state.state_vector[0]) for state in truth]
        gt_y[idx] = [float(state.state_vector[2]) for state in truth]

    all_x = plat_x + [x for sublist in gt_x for x in sublist]
    all_y = plat_y + [y for sublist in gt_y for y in sublist]
    if not all_x or not all_y:
        raise ValueError("no coordinates available to plot")

    raw_min_x, raw_max_x = min(all_x), max(all_x)
    raw_min_y, raw_max_y = min(all_y), max(all_y)
    scale, unit = _distance_axis_scale(min(raw_min_x, raw_min_y), max(raw_max_x, raw_max_y))
    raw_span = max(raw_max_x - raw_min_x, raw_max_y - raw_min_y)
    pad = _range_padding_for_scale(scale, raw_span)

    min_x, max_x = raw_min_x - pad, raw_max_x + pad
    min_y, max_y = raw_min_y - pad, raw_max_y + pad

    mid_x = (max_x + min_x) / 2
    mid_y = (max_y + min_y) / 2
    # Keep a 1:1 spatial aspect by expanding the smaller axis to match the larger span.
    max_span = max(max_x - min_x, max_y - min_y)

    x_range_native = [mid_x - max_span / 2, mid_x + max_span / 2]
    y_range_native = [mid_y - max_span / 2, mid_y + max_span / 2]

    if bathymetry is not None:
        if not hasattr(bathymetry, "get_grid"):
            raise ValueError("bathymetry must provide get_grid(x_range, y_range)")

        bty_x, bty_y, bty_z = bathymetry.get_grid(
            x_range=(x_range_native[0], x_range_native[1]),
            y_range=(y_range_native[0], y_range_native[1]),
        )
        bty_x = _expand_heatmap_coords(np.asarray(bty_x, dtype=float) * scale)
        bty_y = _expand_heatmap_coords(np.asarray(bty_y, dtype=float) * scale)
        bty_depth = np.asarray(bty_z, dtype=float)

        zmin_raw = float(np.nanmin(bty_depth))
        zmax_raw = float(np.nanmax(bty_depth))
        eps = max(1e-9, 1e-6 * max(abs(zmin_raw), abs(zmax_raw), 1.0))
        zmin = zmin_raw if zmin_raw < 0.0 else -eps
        zmax = zmax_raw if zmax_raw > 0.0 else eps
        colorscale = _two_slope_colorscale(cmocean.cm.topo, zmin, zmax, vcenter=0.0)

        hovertemplate = (
            "X: %{x:.2f} {unit}<br>Y: %{y:.2f} {unit}<br>Bathymetry z: %{z:.2f} m<extra></extra>"
        )
        fig.add_trace(
            go.Heatmap(
                x=bty_x,
                y=bty_y,
                z=bty_depth.T,
                colorscale=colorscale,
                zmin=zmin,
                zmax=zmax,
                opacity=0.8,
                colorbar=dict(
                    title=dict(text="Bathymetry z (m)"),
                    thickness=24,
                    len=0.85,
                    y=0.5,
                    yanchor="middle",
                    x=1.1,
                    xanchor="left",
                    xpad=0,
                ),
                hovertemplate=hovertemplate,
            )
        )

    # Convert coordinates and precomputed ranges from native units to display units.
    plat_x = [x * scale for x in plat_x]
    plat_y = [y * scale for y in plat_y]
    gt_x = [[x * scale for x in x_coords] for x_coords in gt_x]
    gt_y = [[y * scale for y in y_coords] for y_coords in gt_y]
    x_range = [value * scale for value in x_range_native]
    y_range = [value * scale for value in y_range_native]

    # Plot a single marker for stationary platforms to avoid a degenerate line trace.
    platform_is_stationary = len(plat_x) <= 1 or (
        np.allclose(plat_x, plat_x[0]) and np.allclose(plat_y, plat_y[0])
    )
    if platform_is_stationary:
        fig.add_trace(
            go.Scatter(
                x=[plat_x[0]],
                y=[plat_y[0]],
                mode="markers",
                marker=dict(color="black", size=10),
                name="Platform",
                **_legend_group_kwargs("platform", "Platform"),
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=plat_x,
                y=plat_y,
                mode="lines",
                line=dict(color="black", width=3),
                name="Platform",
                **_legend_group_kwargs("platform", "Platform"),
            )
        )

    names = [f"Truth {i + 1}" if num_truths > 1 else "Truth" for i in range(num_truths)]
    for i in range(num_truths):
        fig.add_trace(
            go.Scatter(
                x=gt_x[i],
                y=gt_y[i],
                mode="lines",
                line=dict(color=colorway[i % len(colorway)], width=3, dash="5px,2px"),
                name=names[i],
                **_legend_group_kwargs("truths", "Ground Truths"),
            )
        )

    width_px, height_px = _normalise_plotly_figsize(figsize)

    fig.update_layout(
        autosize=False,
        width=width_px,
        height=height_px,
        showlegend=True,
        template="plotly_white",
        xaxis=dict(
            title=f"X Position ({unit})",
            range=x_range,
        ),
        yaxis=dict(
            title=f"Y Position ({unit})",
            range=y_range,
            scaleanchor="x",
            scaleratio=1,
        ),
    )

    if bathymetry is not None:
        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.16,
                xanchor="left",
                x=0.0,
            ),
            margin=dict(b=120),
        )

    return fig


def plot_btr(
    timesteps: ArrayLike,
    steering_azimuths: ArrayLike,
    data: ArrayLike | None = None,
    truths: Sequence[GroundTruthPath] | None = None,
    detections: Sequence[Detection] | None = None,
    tracks: Sequence[Track] | None = None,
    data_type: str = "SNR (dB)",
    cmin: float | None = None,
    cmax: float | None = None,
    figsize: tuple[float, float] = (800, 600),
    fig: go.Figure | None = None,
    row: int | None = None,
    col: int | None = None,
) -> go.Figure:
    """Plot the bearing-time record (BTR) of the beamformed data.

    Optionally overlays truth trajectories and detections on the BTR plot.

    Parameters
    ----------
    timesteps : ArrayLike
        Timesteps corresponding to the first dimension of ``data``.
    steering_azimuths : ArrayLike
        Steering azimuth angles corresponding to the second dimension of ``data``.
    data : ArrayLike | None
        Beamformed data to plot as a heatmap. If ``None``, no heatmap is drawn and
        only overlays are rendered.
    truths : Sequence[GroundTruthPath] | None
        Ground-truth paths representing target trajectories. Default is ``None``.
    detections : Sequence[Detection] | None
        Detection objects to overlay. Default is ``None``.
    tracks : Sequence[Track] | None
        Track objects to overlay. Default is ``None``.
    data_type : str
        Label for the plotted heatmap quantity (for example ``"SNR (dB)"``).
        Used as the colorbar title. Default is ``"SNR (dB)"``.
    cmin : float | None
        Optional lower bound of the heatmap color scale. If ``None`` (default),
        Plotly automatically chooses the lower bound from the data.
    cmax : float | None
        Optional upper bound of the heatmap color scale. If ``None`` (default),
        Plotly automatically chooses the upper bound from the data.
    figsize : tuple[float, float]
        Figure size for standalone plots. Values that look like inches (for example
        ``(12, 6)``) are converted to pixels using 100 px/in; larger values are
        treated as pixels. Ignored when ``fig`` is provided.
    fig : go.Figure | None
        Optional target figure. Provide a subplot figure from
        :func:`plotly.subplots.make_subplots` to draw directly into a cell.
        If None, a new standalone figure is created.
    row : int | None
        Subplot row when ``fig`` is provided.
    col : int | None
        Subplot column when ``fig`` is provided.

    Returns
    -------
    go.Figure
        A Plotly figure object containing the BTR plot.

    """

    def _wrap_bearing_deg(angle_deg: float) -> float:
        """Wrap degrees to the interval [-180, 180)."""
        return (angle_deg + 180.0) % 360.0 - 180.0

    def _split_wrapped_line(
        bearings_deg: list[float], times: list[datetime], jump_threshold_deg: float = 180.0
    ) -> tuple[list[float | None], list[datetime | None]]:
        """Insert gaps when bearings jump across wrap boundaries."""
        if not bearings_deg or not times:
            return [], []

        split_bearings: list[float | None] = [bearings_deg[0]]
        split_times: list[datetime | None] = [times[0]]
        prev_bearing = bearings_deg[0]

        for bearing, timestamp in zip(bearings_deg[1:], times[1:], strict=False):
            if abs(bearing - prev_bearing) > jump_threshold_deg:
                # None inserts a gap so wrapped bearings do not draw cross-plot lines.
                split_bearings.append(None)
                split_times.append(None)
            split_bearings.append(bearing)
            split_times.append(timestamp)
            prev_bearing = bearing

        return split_bearings, split_times

    if fig is not None:
        if (row is None) != (col is None):
            raise ValueError("row and col must both be provided when fig is supplied")
        if row is None or col is None:
            raise ValueError("row and col must both be provided when fig is supplied")
        if row <= 0 or col <= 0:
            raise ValueError("row and col must be positive")

    if fig is None and (row is not None or col is not None):
        raise ValueError("row and col can only be used when fig is supplied")

    if cmin is not None:
        cmin = float(cmin)
        if not np.isfinite(cmin):
            raise ValueError("cmin must be finite when provided")
    if cmax is not None:
        cmax = float(cmax)
        if not np.isfinite(cmax):
            raise ValueError("cmax must be finite when provided")
    if cmin is not None and cmax is not None and cmin >= cmax:
        raise ValueError("cmin must be less than cmax")

    timesteps_array, steering_array, data_array = _validate_btr_shapes(
        timesteps=timesteps,
        steering_azimuths=steering_azimuths,
        data=data,
    )

    target_fig = go.Figure() if fig is None else fig
    using_subplot_target = fig is not None
    existing_group_counts: dict[str, int] = {}
    groups_with_titles: set[str] = set()
    for trace in target_fig.data:
        group = getattr(trace, "legendgroup", None)
        if not group:
            continue
        group_name = str(group)
        existing_group_counts[group_name] = existing_group_counts.get(group_name, 0) + 1
        group_title = getattr(getattr(trace, "legendgrouptitle", None), "text", None)
        if group_title:
            groups_with_titles.add(group_name)

    planned_group_counts = {
        "detections": 1 if detections is not None else 0,
        "tracks": len(tracks) if tracks is not None else 0,
        "truths": len(truths) if truths is not None else 0,
    }
    total_group_counts = existing_group_counts.copy()
    for group_name, count in planned_group_counts.items():
        total_group_counts[group_name] = total_group_counts.get(group_name, 0) + count

    added_legend_groups: set[str] = set()

    def _legend_group_kwargs(group_name: str, group_title: str) -> dict[str, str]:
        """Return legend-group kwargs and add a title once per group per figure."""
        kwargs = {"legendgroup": group_name}
        should_show_title = total_group_counts.get(group_name, 0) > 1
        if (
            should_show_title
            and group_name not in groups_with_titles
            and group_name not in added_legend_groups
        ):
            kwargs["legendgrouptitle_text"] = group_title
            added_legend_groups.add(group_name)
        return kwargs

    colorway = list(target_fig.layout.colorway or px.colors.qualitative.Plotly)
    track_colorway = list(reversed(colorway))

    if data_array is not None:
        heatmap = go.Heatmap(
            z=data_array,
            y=timesteps_array,
            x=steering_array,
            colorscale="Viridis",
            zmin=cmin,
            zmax=cmax,
            colorbar=dict(
                title=dict(text=data_type),
                thickness=24,
                len=1.0,
                x=1.02,
                xanchor="left",
                xpad=0,
            ),
        )
        if using_subplot_target:
            target_fig.add_trace(heatmap, row=row, col=col)
        else:
            target_fig.add_trace(heatmap)

    if detections is not None:
        det_x = [_wrap_bearing_deg(float(np.rad2deg(det.state_vector[0]))) for det in detections]
        det_y = [det.timestamp for det in detections]
        detection_trace = go.Scatter(
            x=det_x,
            y=det_y,
            mode="markers",
            marker=dict(size=5, line=dict(width=1), color="white", opacity=0.8),
            name="Detection",
            **_legend_group_kwargs("detections", "Detections"),
        )
        if using_subplot_target:
            target_fig.add_trace(detection_trace, row=row, col=col)
        else:
            target_fig.add_trace(detection_trace)

    if tracks is not None:
        # Key by object identity so each track keeps a stable color across traces.
        track_color_map: dict[int, str] = {}
        for idx, track in enumerate(tracks):
            track_key = id(track)
            if track_key not in track_color_map:
                track_color_map[track_key] = track_colorway[
                    len(track_color_map) % len(track_colorway)
                ]
            track_color = track_color_map[track_key]
            track_x = [
                _wrap_bearing_deg(float(np.rad2deg(state.state_vector[0]))) for state in track
            ]
            track_y = [state.timestamp for state in track]
            track_x, track_y = _split_wrapped_line(track_x, track_y)
            track_trace = go.Scatter(
                x=track_x,
                y=track_y,
                mode="lines",
                connectgaps=False,
                line=dict(color=track_color, width=4),
                name=f"Track {idx + 1}" if len(tracks) > 1 else "Track",
                **_legend_group_kwargs("tracks", "Tracks"),
            )
            if using_subplot_target:
                target_fig.add_trace(track_trace, row=row, col=col)
            else:
                target_fig.add_trace(track_trace)

    if truths is not None:
        # Reuse colors for repeated truth objects when multiple overlays are added.
        truth_color_map: dict[int, str] = {}
        gt_x = [
            [_wrap_bearing_deg(float(np.rad2deg(state.state_vector[0]))) for state in truth]
            for truth in truths
        ]
        gt_y = [[state.timestamp for state in truth] for truth in truths]
        for idx, truth in enumerate(truths):
            truth_key = id(truth)
            if truth_key not in truth_color_map:
                truth_color_map[truth_key] = colorway[len(truth_color_map) % len(colorway)]
            truth_color = truth_color_map[truth_key]
            truth_x, truth_y = _split_wrapped_line(gt_x[idx], gt_y[idx])
            truth_trace = go.Scatter(
                x=truth_x,
                y=truth_y,
                mode="lines",
                connectgaps=False,
                line=dict(color=truth_color, width=3, dash="dash"),
                name=f"Truth {idx + 1}" if len(truths) > 1 else "Truth",
                **_legend_group_kwargs("truths", "Ground Truths"),
            )
            if using_subplot_target:
                target_fig.add_trace(truth_trace, row=row, col=col)
            else:
                target_fig.add_trace(truth_trace)

    bearing_min = float(np.min(steering_array))
    bearing_max = float(np.max(steering_array))
    bearing_span = bearing_max - bearing_min
    bearing_start = float(steering_array[0])
    bearing_end = float(steering_array[-1])

    xaxis_config = dict(
        range=[bearing_start, bearing_end],
        tickmode="linear",
        tick0=bearing_start,
        dtick=bearing_span / 6.0 if bearing_span > 0 else 1.0,
        tickangle=-45,
        title="Bearing (°)",
        showline=True,
    )
    if not using_subplot_target:
        xaxis_config["domain"] = [0.0, 0.9]

    if using_subplot_target:
        target_fig.update_xaxes(**xaxis_config, row=row, col=col)
    else:
        target_fig.update_xaxes(**xaxis_config)

    yaxis_config = dict(
        # Set explicit descending bounds to keep a consistent BTR orientation.
        range=[np.max(timesteps_array), np.min(timesteps_array)],
        tickformat="%H:%M",
        autorange=False,
        title="Time (HH:MM)",
        showline=True,
    )
    if using_subplot_target:
        target_fig.update_yaxes(**yaxis_config, row=row, col=col)
    else:
        target_fig.update_yaxes(**yaxis_config)

    if not using_subplot_target:
        target_fig.update_layout(
            template="plotly_white",
            width=figsize[0],
            height=figsize[1],
            showlegend=True,
            plot_bgcolor="white",
            paper_bgcolor="white",
        )

    return target_fig


def plot_spectrogram(
    signal: ArrayLike,
    sr: int,
    n_fft: int = 4096,
    hop_length: int = 1024,
    y_lim: tuple[float, float] | None = None,
    yaxis_format: str = "kHz",
    figsize: tuple[float, float] = (12, 6),
) -> go.Figure:
    """Generate and display a formatted spectrogram with Plotly.

    Parameters
    ----------
    signal : ArrayLike
        1D array-like audio signal.
    sr : int
        Sampling rate in Hz.
    n_fft : int
        FFT window size.
    hop_length : int
        STFT hop length.
    y_lim : tuple[float, float] | None
        Optional y-axis limits in Hz as ``(min, max)``.
    yaxis_format : str
        ``"kHz"`` to label y-axis in kHz or ``"Hz"`` for Hz.
    figsize : tuple[float, float]
        Figure size. Values that look like inches (for example ``(12, 6)``) are
        converted to pixels using 100 px/in; larger values are treated as pixels.

    Returns
    -------
    go.Figure
        A Plotly figure containing the spectrogram.

    """
    yaxis_format, y_lim = _validate_spectrogram_params(
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        y_lim=y_lim,
        yaxis_format=yaxis_format,
    )

    signal = np.asarray(signal)
    if signal.size == 0:
        raise ValueError("signal is empty")
    if signal.ndim > 1:
        signal = signal.flatten()

    freqs_hz, times, zxx = scipy_signal.stft(
        signal,
        fs=sr,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        nfft=n_fft,
        boundary=None,
        padded=False,
        return_onesided=True,
    )

    magnitude = np.abs(zxx)
    ref = np.max(magnitude)
    if ref <= 0:
        ref = 1.0
    amin = 1e-10
    # Convert to dB relative to peak magnitude while guarding against log(0).
    s_db = 20.0 * np.log10(np.maximum(amin, magnitude)) - 20.0 * np.log10(ref)

    vmax = float(np.max(s_db))
    # Display a fixed 60 dB window to keep low-energy detail visible.
    vmin = vmax - 60.0

    if yaxis_format == "kHz":
        y_values = freqs_hz / 1000.0
        y_title = "Frequency (kHz)"
        y_range = [y_lim[0] / 1000.0, y_lim[1] / 1000.0] if y_lim else None
    else:
        y_values = freqs_hz
        y_title = "Frequency (Hz)"
        y_range = list(y_lim) if y_lim else None

    fig = go.Figure(
        data=go.Heatmap(
            z=s_db,
            x=times,
            y=y_values,
            colorscale="Viridis",
            zmin=vmin,
            zmax=vmax,
            colorbar=dict(
                title=dict(text="Intensity (dB)"),
                thickness=24,
                len=1.0,
            ),
        )
    )

    width_px, height_px = _normalise_plotly_figsize(figsize)
    fig.update_layout(width=width_px, height=height_px, template="plotly_white")

    fig.update_xaxes(
        title_text="Time (s)",
        range=[0, len(signal) / float(sr)],
        showgrid=False,
    )
    fig.update_yaxes(
        title_text=y_title,
        range=y_range,
        showgrid=False,
    )

    return fig


def plot_roc(
    results: Sequence[SweepResult],
    show_diagonal: bool = True,
    figsize: tuple[float, float] = (600, 500),
) -> go.Figure:
    """Plot Receiver Operating Characteristic (ROC) curves for one or more sweep results.

    Parameters
    ----------
    results : Sequence[SweepResult]
        Sweep results produced by :func:`~bluepebble.detector.metrics.sweep_detection_parameter`.
        Each result is drawn as a separate trace using its ``label`` attribute.
    show_diagonal : bool
        If ``True`` (default), overlay the random-classifier diagonal.
    figsize : tuple[float, float]
        Figure dimensions in pixels.  Default is ``(600, 500)``.

    Returns
    -------
    go.Figure
        Plotly figure containing the ROC curves.

    """
    colorway = px.colors.qualitative.Plotly
    grid_color = "rgba(200, 200, 200, 0.5)"
    axis_line = "rgba(160, 160, 160, 1.0)"

    fig = go.Figure()

    for i, result in enumerate(results):
        # Sort by x-axis metric so connected lines are drawn in curve order.
        order = np.argsort(result.fpr)
        name = f"{result.label} (AUC={result.auc_roc:.3f})"
        fig.add_trace(
            go.Scatter(
                x=result.fpr[order],
                y=result.tpr[order],
                mode="lines",
                line=dict(width=2, color=colorway[i % len(colorway)]),
                name=name,
            )
        )

    if show_diagonal:
        fig.add_trace(
            go.Scatter(
                x=[0.0, 1.0],
                y=[0.0, 1.0],
                mode="lines",
                line=dict(width=1, color="grey", dash="dash"),
                name="Random",
                showlegend=True,
            )
        )

    fig.update_layout(
        template="plotly_white",
        width=figsize[0],
        height=figsize[1],
        showlegend=True,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(
        title_text="False Positive Rate",
        range=[0.0, 1.0],
        showgrid=True,
        gridcolor=grid_color,
        showline=True,
        linewidth=1,
        linecolor=axis_line,
    )
    fig.update_yaxes(
        title_text="True Positive Rate",
        range=[0.0, 1.05],
        showgrid=True,
        gridcolor=grid_color,
        showline=True,
        linewidth=1,
        linecolor=axis_line,
    )

    return fig


def plot_pr(
    results: Sequence[SweepResult],
    figsize: tuple[float, float] = (600, 500),
) -> go.Figure:
    """Plot Precision-Recall (PR) curves for one or more sweep results.

    Parameters
    ----------
    results : Sequence[SweepResult]
        Sweep results produced by :func:`~bluepebble.detector.metrics.sweep_detection_parameter`.
        Each result is drawn as a separate trace using its ``label`` attribute.
    figsize : tuple[float, float]
        Figure dimensions in pixels.  Default is ``(600, 500)``.

    Returns
    -------
    go.Figure
        Plotly figure containing the PR curves.

    """
    colorway = px.colors.qualitative.Plotly
    grid_color = "rgba(200, 200, 200, 0.5)"
    axis_line = "rgba(160, 160, 160, 1.0)"

    fig = go.Figure()

    for i, result in enumerate(results):
        # Sort by x-axis metric so connected lines are drawn in curve order.
        order = np.argsort(result.recall)
        name = f"{result.label} (AUC={result.auc_pr:.3f})"
        fig.add_trace(
            go.Scatter(
                x=result.recall[order],
                y=result.precision[order],
                mode="lines",
                line=dict(width=2, color=colorway[i % len(colorway)]),
                name=name,
            )
        )

    fig.update_layout(
        template="plotly_white",
        width=figsize[0],
        height=figsize[1],
        showlegend=True,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(
        title_text="Recall",
        range=[0.0, 1.0],
        showgrid=True,
        gridcolor=grid_color,
        showline=True,
        linewidth=1,
        linecolor=axis_line,
    )
    fig.update_yaxes(
        title_text="Precision",
        range=[0.0, 1.05],
        showgrid=True,
        gridcolor=grid_color,
        showline=True,
        linewidth=1,
        linecolor=axis_line,
    )

    return fig


def plot_roc_pr(
    results: Sequence[SweepResult],
    show_diagonal: bool = True,
    figsize: tuple[float, float] = (1100, 500),
) -> go.Figure:
    """Plot ROC and Precision-Recall curves side-by-side for one or more sweep results.

    Parameters
    ----------
    results : Sequence[SweepResult]
        Sweep results produced by :func:`~bluepebble.detector.metrics.sweep_detection_parameter`.
        Each result is drawn as a separate trace pair (same colour in both subplots)
        using its ``label`` attribute.
    show_diagonal : bool
        If ``True`` (default), overlay the random-classifier diagonal on the ROC subplot.
    figsize : tuple[float, float]
        Figure dimensions in pixels.  Default is ``(1100, 500)``.

    Returns
    -------
    go.Figure
        Plotly figure with ROC (left) and PR (right) subplots.

    """
    colorway = px.colors.qualitative.Plotly

    fig = make_subplots(rows=1, cols=2, subplot_titles=("ROC Curve", "Precision-Recall Curve"))

    for i, result in enumerate(results):
        color = colorway[i % len(colorway)]
        # Sort each subplot independently to avoid zig-zag line artifacts.
        roc_order = np.argsort(result.fpr)
        pr_order = np.argsort(result.recall)

        fig.add_trace(
            go.Scatter(
                x=result.fpr[roc_order],
                y=result.tpr[roc_order],
                mode="lines",
                line=dict(width=2, color=color),
                name=f"{result.label} (AUC={result.auc_roc:.3f})",
                legendgroup=result.label,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=result.recall[pr_order],
                y=result.precision[pr_order],
                mode="lines",
                line=dict(width=2, color=color),
                name=f"{result.label} (AUC={result.auc_pr:.3f})",
                legendgroup=result.label,
                showlegend=False,  # suppress duplicate; ROC trace represents this group
            ),
            row=1,
            col=2,
        )

    if show_diagonal:
        fig.add_trace(
            go.Scatter(
                x=[0.0, 1.0],
                y=[0.0, 1.0],
                mode="lines",
                line=dict(width=1, color="grey", dash="dash"),
                name="Random",
                showlegend=True,
            ),
            row=1,
            col=1,
        )

    fig.update_xaxes(title_text="False Positive Rate", range=[0.0, 1.0], col=1)
    fig.update_yaxes(title_text="True Positive Rate", range=[0.0, 1.05], col=1)

    fig.update_xaxes(title_text="Recall", range=[0.0, 1.0], col=2)
    fig.update_yaxes(title_text="Precision", range=[0.0, 1.05], col=2)

    fig.update_layout(
        template="plotly_white",
        width=figsize[0],
        height=figsize[1],
        showlegend=True,
    )

    return fig
