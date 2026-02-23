"""Defines plotting utilities."""

import copy
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from itertools import cycle
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.patches import Ellipse, Patch
from scipy import signal as scipy_signal
from stonesoup.platform.base import Platform
from stonesoup.types.detection import Clutter, Detection, MissedDetection, TrueDetection
from stonesoup.types.groundtruth import GroundTruthPath
from stonesoup.types.track import Track
from tqdm.auto import tqdm

# A default style guide for all plot appearances
DEFAULT_STYLE_GUIDE = {
    "figure": {
        "sns_style": "whitegrid",
        "figsize": (10, 8),
        "font_size": 12,
    },
    "title": {
        "text": None,
        "fontsize_offset": 2,
        "pad": 15,
    },
    "legend": {
        "display": True,
        # "location": "upper left",
        # "anchor": (1.02, 1),
        "location": "upper center",
        "anchor": (0.5, -0.15),
        "ncol": 3,
        "fancybox": True,
        "shadow": False,
        "frameon": True,
        "framealpha": 0.95,
    },
    "axes": {
        "grid_color": "#e0e0e0",
        "grid_linestyle": "--",
        "x_label": "X",
        "y_label": "Y",
    },
    "elements": {
        "platform": {"linestyle": "-.", "linewidth": 2, "alpha": 0.8, "zorder": 4},
        "truth": {"linestyle": "--", "linewidth": 3, "alpha": 0.6, "zorder": 4},
        "track": {"linewidth": 3, "zorder": 3},
        "uncertainty": {"alpha": 0.2, "zorder": 1},
        "particle": {
            "linestyle": "None",
            "marker": ".",
            "markersize": 3,
            "alpha": 0.3,
            "zorder": 2,
        },
        "detection": {
            "marker": "o",
            "markersize": 4,
            "markerfacecolor": "none",
            "markeredgewidth": 1,
            "color": "#9467bd",
            "alpha": 0.5,
            "linewidth": 0,
            "zorder": 6,
        },
        "clutter": {
            "marker": "x",
            "markersize": 4,
            "color": "grey",
            "linewidth": 0,
            "alpha": 0.5,
            "zorder": 5,
        },
    },
}


def _get_time_axis_formatter(time_span_seconds: float) -> tuple:
    """Calculate the optimal matplotlib locator and formatter for a time axis.

    Parameters
    ----------
    time_span_seconds  : float
        The total duration of the time span in seconds.

    Returns
    -------
    tuple
        A tuple containing:
        - locator (matplotlib.dates.DateLocator)
            The locator for the x-axis.
        - formatter (matplotlib.dates.DateFormatter)
            The formatter for the x-axis.
        - label (str)
            A label for the x-axis.

    """
    target_ticks = 6
    axis_configs = [
        # (threshold, converter, intervals, locator, format, label)
        (300, 1, [1, 5, 10, 15, 30], mdates.SecondLocator, "%M:%S", "Time (MM:SS)"),
        (7200, 60, [1, 2, 5, 10, 15], mdates.MinuteLocator, "%H:%M", "Time (HH:MM)"),
        (
            172800,
            3600,
            [1, 2, 3, 4, 6],
            mdates.HourLocator,
            "%d %H:%M",
            "Time (Day HH:MM)",
        ),
    ]
    # Default for very long durations
    default_config = (mdates.DayLocator, [1, 2, 3, 5, 7], "%b %d", "Date", 86400)
    locator_cls, intervals, fmt, label, converter = default_config

    for threshold, conv, ivals, loc, f, l in axis_configs:  # noqa: E741
        if time_span_seconds < threshold:
            locator_cls, intervals, fmt, label, converter = loc, ivals, f, l, conv
            break

    best_interval = min(
        intervals, key=lambda x: abs((time_span_seconds / converter) / x - target_ticks)
    )
    return locator_cls(interval=best_interval), mdates.DateFormatter(fmt), label


# ==============================================================================
# --- Base Plotter Class ---
# ==============================================================================


class BasePlotter(ABC):
    """An abstract base class for plotting trackers.

    This class provides a template for creating plots and animations of tracking
    scenarios. It handles common logic for figure setup, animation loops, and
    saving, while deferring plot-specific implementations (like data preparation
    and axis configuration) to subclasses.
    """

    def __init__(self, defaults: dict | None = None, style_guide: dict | None = None) -> None:
        """Initialise the plotter using a comprehensive style guide.

        Parameters
        ----------
            defaults  : dict
                A dictionary of default styles to apply to the plotter.
                This is merged with the class-specific defaults.
            style_guide  : dict
                A dictionary to override the default styles for this
                plotter. If None, the class defaults are used.

        """
        # Deep copy ensures that default styles are not modified globally.
        self.style_guide = copy.deepcopy(DEFAULT_STYLE_GUIDE)

        # Merge in the class-specific defaults (e.g., from CartesianPlotter)
        if defaults:
            for key, value in defaults.items():
                if key in self.style_guide and isinstance(self.style_guide[key], dict):
                    self.style_guide[key].update(value)
                else:
                    self.style_guide[key] = value

        # Finally, merge in the user-provided overrides
        if style_guide:
            for key, value in style_guide.items():
                if key in self.style_guide and isinstance(self.style_guide[key], dict):
                    self.style_guide[key].update(value)
                else:
                    self.style_guide[key] = value

        # Apply global seaborn and matplotlib styles
        style_cfg = self.style_guide["figure"]
        grid_cfg = self.style_guide["axes"]
        sns.set_style(style_cfg["sns_style"])
        plt.rcParams.update(
            {
                "font.size": style_cfg["font_size"],
                "grid.color": grid_cfg["grid_color"],
                "grid.linestyle": grid_cfg["grid_linestyle"],
            }
        )

        self.anim = None
        self.fig = None
        self.ax = None
        self.num_timesteps = 0
        self.color_cycler = cycle(sns.color_palette("colorblind"))

    def get_color(self) -> str:
        """Return the next distinct color from the internal color palette."""
        return next(self.color_cycler)

    def plot(
        self,
        truths: list[GroundTruthPath],
        tracks: list[Track],
        all_detections: list[set[Detection]],
        timesteps: list[datetime],
        mapping: list[int],
        platforms: list[Platform] | None = None,
        ax: Axes | None = None,
        **kwargs: Any,
    ) -> Axes:
        """Generate a static plot of the final state of a tracking scenario.

        This method plots all historical data up to the final timestep.

        Parameters
        ----------
            truths  : list[GroundTruthPath]
                A list of ground truth paths.
            tracks  : list[Track]
                A list of track objects.
            all_detections  : list[set[Detection]]
                A list of detection sets,
                one for each timestep.
            timesteps  : list[datetime]
                A list of timestamps for the x-axis.
            mapping  : list[int]
                Mapping used by some subclasses to access
                specific components of the state vector.
            platforms  : list[Platform] | None
                A list of platform objects.
            ax  : Axes | None
                An existing matplotlib Axes object to plot on.
                If None, a new figure and axes are created.
            **kwargs : Any
                Additional keyword arguments passed to helper methods.

        Returns
        -------
            Axes
                The matplotlib Axes object containing the plot.

        """
        data, plot_objects = self._setup_plot(
            truths, tracks, all_detections, timesteps, mapping, platforms, ax, **kwargs
        )

        for t in timesteps:
            self._update_plot_objects(t, data, plot_objects, mapping)

        self.fig.tight_layout()
        return self.ax

    def animate(
        self,
        truths: list[GroundTruthPath],
        tracks: list[Track],
        all_detections: list[set[Detection]],
        timesteps: list[datetime],
        mapping: list[int],
        platforms: list[Platform] | None = None,
        **kwargs: Any,
    ) -> tuple[Axes, FuncAnimation]:
        """Generate an animation of the tracking process over time.

        This method creates an animation that updates the plot for each timestep,
        showing the evolution of the tracking scenario over time.

        Parameters
        ----------
            truths  : list[GroundTruthPath]
                A list of ground truth paths.
            tracks  : list[Track]
                A list of track objects.
            all_detections  : list[set[Detection]]
                A list of detection sets.
            timesteps  : list[datetime]
                A list of timestamps for each frame.
            mapping  : list[int]
                A list of indices used by subclasses to access
                specific components of the state vector (e.g., `[0, 2]` for x and y).
            platforms  : list[Platform], optional
                A list of platform objects.
            **kwargs : Any
                Additional keyword arguments passed to helper methods.

        Returns
        -------
            tuple[Axes, FuncAnimation]
                The axes and the animation object. The
            animation object must be kept in scope to be displayed.

        """
        self.num_timesteps = len(timesteps)
        data, plot_objects = self._setup_plot(
            truths,
            tracks,
            all_detections,
            timesteps,
            mapping,
            platforms,
            None,
            **kwargs,
        )
        self.fig.tight_layout()

        def _update(k: int) -> list:
            """Update function for FuncAnimation.

            Parameters
            ----------
                k  : int
                    The current frame index.

            Returns
            -------
                list
                    The updated plot objects for the current frame.

            """
            return self._update_plot_objects(timesteps[k], data, plot_objects, mapping)

        self.anim = FuncAnimation(
            self.fig, _update, frames=range(len(timesteps)), repeat=True, interval=100
        )

        return self.ax, self.anim

    def show(self, tight_layout: bool = True) -> None:
        """Display the generated plot or animation.

        Parameters
        ----------
            tight_layout  : bool
                Whether to apply tight layout to the figure.
                Defaults to True. If False, the layout will not be adjusted.

        Raises
        ------
            ValueError
                If no plot or animation has been created.

        """
        if not self.fig and not self.anim:
            raise ValueError("No plot or animation to show. Call plot() or animate() first.")
        if tight_layout and self.fig:
            self.fig.tight_layout()
        plt.show()

    def save(self, filename: str, **kwargs: Any) -> None:
        """Save the plot or animation to a file using a tqdm progress bar.

        This method can save static plots (e.g., 'figure.png') or animations
        (e.g., 'animation.gif' or 'animation.mp4').

        Parameters
        ----------
            filename  : str
                The path and name of the file to save.
            **kwargs : Any
                Additional keyword arguments passed to the underlying
                matplotlib save function (e.g., `dpi`, `writer`).

        """
        if self.anim:
            # 1. Create a tqdm instance
            with tqdm(total=self.num_timesteps, desc=f"Saving {filename}") as pbar:
                # 2. Define a callback function for matplotlib
                def update_func(frame, total_frames):
                    pbar.update(1)

                # 3. Call anim.save with the callback
                self.anim.save(
                    filename,
                    progress_callback=update_func,
                    **kwargs,
                )
            self.anim = None
        elif self.fig:
            self.fig.savefig(filename, bbox_inches="tight", **kwargs)
            self.fig = None
        else:
            raise ValueError("No plot or animation to save. Call plot() or animate() first.")

    def close(self):
        """Close the current plot figure."""
        if self.fig:
            plt.close(self.fig)

    def _setup_figure(self, ax: Axes | None = None) -> None:
        """Initialise the matplotlib figure and axes objects.

        Parameters
        ----------
            ax  : Axes | None
                An existing matplotlib Axes object to use.

        """
        if ax is None:
            figsize = self.style_guide["figure"]["figsize"]
            self.fig, self.ax = plt.subplots(1, 1, figsize=figsize)
        else:
            self.ax = ax
            self.fig = ax.get_figure()

    def _setup_plot(
        self,
        truths: list[GroundTruthPath],
        tracks: list[Track],
        all_detections: list[Detection],
        timesteps: list[datetime],
        mapping: dict,
        platforms: list[Platform] | None,
        ax: Axes,
        **kwargs: Any,
    ) -> tuple[dict, dict]:
        """Handle the common setup sequence for both plots and animations.

        Parameters
        ----------
            truths  : list[GroundTruthPath]
                A list of ground truth paths.
            tracks  : list[Track]
                A list of track objects.
            all_detections  : list[Detection]
                A list of detection sets, one for each
                timestep.
            timesteps  : list[datetime]
                A list of timestamps for the x-axis.
            mapping  : dict
                Mapping used by some subclasses.
            platforms  : list[Platform] | None
                A list of platform objects.
            ax  : Axes
                An existing matplotlib Axes object to plot on.
            **kwargs
                Additional keyword arguments for customisation.

        Returns
        -------
            tuple
                A tuple containing:
            - data (dict)
                A structured dictionary with keys for truths,
                  tracks, uncertainty, particles, measurements, and clutter.
            - plot_kwargs (dict)
                A dictionary of plot-specific parameters
                  derived from the input arguments.

        """
        self._setup_figure(ax)

        # Configure title from the style guide
        title_cfg = self.style_guide["title"]
        if title_cfg.get("text"):
            self.ax.set_title(
                title_cfg["text"],
                fontsize=(self.style_guide["figure"]["font_size"] + title_cfg["fontsize_offset"]),
                pad=title_cfg["pad"],
            )

        data, plot_kwargs = self._prepare_data(
            truths, tracks, all_detections, timesteps, mapping, platforms, **kwargs
        )

        self._configure_axes(data, timesteps)

        if self.style_guide["legend"]["display"]:
            self._configure_legend(data, **plot_kwargs)

        plot_objects = self._init_plot_objects(data, **plot_kwargs)
        return data, plot_objects

    @abstractmethod
    def _configure_axes(self, data: dict, timesteps: list[datetime]) -> None:
        """Abstract method to configure the plot's axes.

        Subclasses must implement this to set appropriate limits, labels, and
        ticks for their specific coordinate system (e.g., Bearing vs. Time).

        Parameters
        ----------
            data  : dict
                The structured data containing truths, tracks, and detections.
            timesteps  : list[datetime]
                The list of timesteps for the plot.

        """
        raise NotImplementedError

    def _configure_legend(self, data: dict, **kwargs: Any) -> None:
        """Configure the legend dynamically based on the plotted data.

        This method creates legend handles based on the presence of truths, tracks,
        and detections in the data.

        Parameters
        ----------
            data  : dict
                The structured data containing truths, tracks, and detections.
            **kwargs
                Additional keyword arguments for customisation.

        """
        handles = []
        element_styles = self.style_guide["elements"]

        if data["history"]["platforms"]:
            style = {
                k: v for k, v in element_styles["platform"].items() if k not in ["alpha", "zorder"]
            }
            handles.append(plt.Line2D([0], [0], color="k", label="Platform", **style))

        if data["history"]["truths"]:
            # Create a representative line for the legend from the style guide
            style = {
                k: v for k, v in element_styles["truth"].items() if k not in ["alpha", "zorder"]
            }
            handles.append(plt.Line2D([0], [0], color="k", label="Truth", **style))

        if data["history"]["tracks"]:
            style = {
                k: v for k, v in element_styles["track"].items() if k not in ["alpha", "zorder"]
            }
            handles.append(plt.Line2D([0], [0], color="k", label="Track", **style))
            if kwargs.get("plot_uncertainty", False):
                handles.append(Patch(facecolor="k", alpha=0.2, label="Uncertainty"))
            if kwargs.get("plot_particles", False):
                style = {
                    k: v
                    for k, v in element_styles["particle"].items()
                    if k not in ["alpha", "zorder"]
                }
                handles.append(plt.Line2D([0], [0], color="k", label="Particle", **style))

        distinguish = kwargs.get("distinguish_detections", False)
        has_detections = kwargs.get("has_true_detections", False)
        has_clutter = kwargs.get("has_clutter", False)

        if has_detections or (has_clutter and not distinguish):
            style = {
                k: v
                for k, v in element_styles["detection"].items()
                if k not in ["alpha", "zorder"]
            }
            handles.append(plt.Line2D([0], [0], label="Detection", **style))

        if has_clutter and distinguish:
            style = {
                k: v for k, v in element_styles["clutter"].items() if k not in ["alpha", "zorder"]
            }
            handles.append(plt.Line2D([0], [0], label="Clutter", **style))

        if handles:
            legend_cfg = self.style_guide["legend"]
            self.ax.legend(
                handles=handles,
                loc=legend_cfg["location"],
                ncol=legend_cfg["ncol"],
                bbox_to_anchor=legend_cfg["anchor"],
                fancybox=legend_cfg["fancybox"],
                shadow=legend_cfg["shadow"],
                frameon=legend_cfg["frameon"],
                framealpha=legend_cfg["framealpha"],
            )

    @abstractmethod
    def _update_plot_objects(
        self,
        timestep: datetime,
        data: dict,
        plot_objects: dict,
        mapping: list[int],
    ) -> list:
        """Abstract method to update plot objects with data for a new frame.

        This method is called for each timestep to update the data of the
        artists created by `_init_plot_objects`.

        Parameters
        ----------
            timestep  : datetime
                The current timestep to update the plot for.
            data  : dict
                The structured data containing truths, tracks, and detections.
            plot_objects  : dict
                The dictionary of plot objects created by
                `_init_plot_objects`.
            mapping  : list[int]
                A mapping used to access specific components of the
                state vector.

        Returns
        -------
            list
                A list of the updated artists, required by FuncAnimation.

        """
        raise NotImplementedError

    @property
    @abstractmethod
    def coord_names(self) -> list[str]:
        """A list of two strings defining the coordinate names (e.g., ['x', 'y'])."""
        raise NotImplementedError

    def _prepare_data(
        self,
        truths: list[GroundTruthPath],
        tracks: list[Track],
        all_detections: list[set[Detection]],
        timesteps: list[datetime],
        mapping: list | tuple,
        platforms: list[Platform] | None = None,
        **kwargs,
    ) -> tuple[dict, dict]:
        """Structure input data for plotting.

        This method prepares the data for plotting by organising truths, tracks,
        and detections into a structured format.

        Parameters
        ----------
            truths  : list[GroundTruthPath]
                A list of ground truth paths.
            tracks  : list[Track]
                A list of track objects.
            all_detections  : list[set[Detection]]
                A list of detection sets,
                one for each timestep.
            timesteps  : list[datetime]
                A list of timestamps for the x-axis.
            mapping  : list | tuple, optional
                Mapping used by some subclasses.
            platforms  : list[Platform] | None, optional
                A list of platform objects.
            **kwargs
                Additional keyword arguments for customisation.

        Returns
        -------
            tuple
                A tuple containing:
            - data (dict)
                A structured dictionary with keys for truths,
                  tracks, uncertainty, particles, measurements, and clutter.
            - plot_kwargs (dict)
                A dictionary of plot-specific parameters
                  derived from the input arguments.

        """
        # Check for different detection types
        has_true_detections = any(isinstance(d, TrueDetection) for s in all_detections for d in s)
        has_clutter = any(isinstance(d, Clutter) for s in all_detections for d in s)
        plot_kwargs = {
            "plot_particles": kwargs.get("plot_particles", False),
            "plot_uncertainty": kwargs.get("plot_uncertainty", False),
            "distinguish_detections": has_true_detections and has_clutter,
            "has_true_detections": has_true_detections,
            "has_clutter": has_clutter,
        }

        # Initialise data structure
        data = {
            t: {
                "truths": {},
                "tracks": {},
                "uncertainty": {},
                "particles": {},
                "measurements": [],
                "clutter": [],
                "platforms": {},
            }
            for t in timesteps
        }

        # Populate with truths, tracks, and detections
        for truth in truths:
            for state in truth:
                data[state.timestamp]["truths"][truth.id] = state
        for track in tracks:
            for state in track:
                # This check makes the logic permissive and fixes the crash
                if state.timestamp in data:
                    data[state.timestamp]["tracks"][track.id] = state
                    if plot_kwargs["plot_uncertainty"] and hasattr(state, "covar"):
                        data[state.timestamp]["uncertainty"][track.id] = state
                    if plot_kwargs["plot_particles"] and hasattr(state, "state_vector"):
                        data[state.timestamp]["particles"][track.id] = state
        if platforms:
            for i, platform in enumerate(platforms):
                for state in platform.states:
                    if state.timestamp in data:
                        data[state.timestamp]["platforms"][f"platform_{i}"] = state
        for det_set in all_detections:
            for det in det_set:
                if isinstance(det, MissedDetection):
                    continue
                category = (
                    "clutter"
                    if (isinstance(det, Clutter) and plot_kwargs["distinguish_detections"])
                    else "measurements"
                )
                data[det.timestamp][category].append(det)

        # Initialize history using the abstract coord_names property
        coord1, coord2 = self.coord_names
        data["history"] = {
            "truths": {t.id: {coord1: [], coord2: []} for t in truths},
            "tracks": {t.id: {coord1: [], coord2: [], "std_dev": []} for t in tracks},
            "measurements": {coord1: [], coord2: []},
            "clutter": {coord1: [], coord2: []},
            "platforms": {
                f"platform_{i}": {coord1: [], coord2: []} for i in range(len(platforms or []))
            },
        }

        return data, plot_kwargs

    def _init_plot_objects(self, data: dict, **kwargs: Any) -> dict:
        """Create initial, empty plot artists for common elements.

        Parameters
        ----------
            data  : dict
                The structured data prepared for plotting.
            **kwargs : Any
                Additional keyword arguments for specific plot styles.

        Returns
        -------
            dict
                A dictionary of plot objects, keyed by their type (e.g., "truths",
            "tracks", "measurements", "clutter", "platforms").

        """
        plots = {"truths": {}, "tracks": {}, "platforms": {}}
        element_styles = self.style_guide["elements"]

        # Platforms
        for platform_id in data["history"]["platforms"]:
            style = element_styles["platform"].copy()
            (plots["platforms"][platform_id],) = self.ax.plot(
                [], [], color=self.get_color(), **style
            )

        # Truths
        for truth_id in data["history"]["truths"]:
            style = element_styles["truth"].copy()
            (plots["truths"][truth_id],) = self.ax.plot([], [], color=self.get_color(), **style)
        # Tracks
        for track_id in data["history"]["tracks"]:
            style = element_styles["track"].copy()
            (plots["tracks"][track_id],) = self.ax.plot([], [], color=self.get_color(), **style)

        # Detections and Clutter
        (plots["meas"],) = self.ax.plot([], [], **element_styles["detection"])
        (plots["clutter"],) = self.ax.plot([], [], **element_styles["clutter"])

        return plots


# ==============================================================================
# --- Bearings Plotter ---
# ==============================================================================


class BearingsPlotter(BasePlotter):
    """A plotter for visualising tracks in a bearing vs. time coordinate system.

    This class provides an implementation of the BasePlotter, specifically
    for creating plots where the y-axis represents time and the x-axis
    represents the bearing from the sensor platform.
    """

    coord_names = ["bearing", "time"]  # Define coordinate names for the base class

    def __init__(
        self,
        style_guide: dict | None = None,
        bearing_range_deg: tuple[float, float] = (0, 180),
    ) -> None:
        """Initialise the plotter using a comprehensive style guide.

        Parameters
        ----------
            style_guide  : dict | None
                A dictionary to override the
                default styles for this plotter. If None, the default styles are used.
            bearing_range_deg  : tuple[float, float]
                The range of bearing angles to
                plot.

        """
        # Define the default styles specific to a bearings plot
        bearings_defaults = {
            "axes": {"x_label": "Bearing (°)", "y_label": "Time"},
        }

        super().__init__(defaults=bearings_defaults, style_guide=style_guide)
        self.bearing_range_deg = bearing_range_deg

    def plot_snr(
        self,
        snr_array: np.ndarray,
        timesteps: list[datetime],
        all_detections: list[set[Detection]],
        truths: list[GroundTruthPath] = None,
        ax: Axes = None,
        add_colorbar: bool = True,
        colorbar_ax: list[Axes] = None,
        add_legend: bool = True,
    ) -> tuple[Axes, Any]:
        """Plot a heatmap of SNR vs. time, with detections overlaid.

        This method generates a "waterfall" plot to visualise the raw SNR
        data from the beamformer, which is useful for diagnosing the detection
        process.

        Parameters
        ----------
            snr_array  : np.ndarray
                A 2D array of SNR values from the
                simulation loop.
            timesteps  : list[datetime]
                A list of all simulation timestamps.
            all_detections  : list[set[Detection]]
                A list of detection sets
                to overlay on the heatmap.
            truths  : list[GroundTruthPath], optional
                A list of ground truth
                paths to overlay for context. Defaults to None.
            ax  : Axes, optional
                An existing matplotlib Axes object to plot on.
                If None, a new figure and axes are created.
            add_colorbar  : bool
                Whether to add a colorbar to the plot.
                Defaults to True.
            colorbar_ax  : list[Axes], optional
                If provided, the colorbar will
                be added to this Axes object instead of the main Axes.
            add_legend  : bool
                Whether to add a legend to the plot.
                Defaults to True.

        Returns
        -------
            tuple[Axes, Any]
                The matplotlib Axes object containing the plot and
            the image object created by `imshow`.

        """
        self._setup_figure(ax)
        element_styles = self.style_guide["elements"]

        # --- Extract Detections for Plotting ---
        detection_times = []
        detection_bearings = []
        for detection_set in all_detections:
            for detection in detection_set:
                detection_times.append(mdates.date2num(detection.timestamp))
                detection_bearings.append(np.rad2deg(detection.state_vector[0]))

        # 1. Plot the SNR data as a heatmap
        im = self.ax.imshow(
            snr_array,
            aspect="auto",
            origin="lower",
            extent=[
                self.bearing_range_deg[0],
                self.bearing_range_deg[1],
                mdates.date2num(timesteps[0]),
                mdates.date2num(timesteps[-1]),
            ],
            cmap="cividis",
            vmin=np.percentile(snr_array, 20),
            vmax=np.percentile(snr_array, 99.8),
        )

        legend_handles, legend_labels = [], []

        # 2. Overlay the ground truth paths (if provided)
        if truths:
            truth_style = element_styles["truth"].copy()
            truth_style["color"] = "white"  # Override color for visibility on heatmap
            for truth in truths:
                times = [mdates.date2num(state.timestamp) for state in truth]
                bearings = [np.rad2deg(state.state_vector[0]) for state in truth]
                (line,) = self.ax.plot(bearings, times, label="Ground Truth", **truth_style)
            legend_handles.append(line)
            legend_labels.append("Ground Truth")

        # 3. Overlay the detections (if any)
        if detection_bearings and detection_times:
            detection_style = element_styles["detection"].copy()
            # Use hollow circles with edge color for consistency with main plot
            scatter = self.ax.scatter(
                detection_bearings,
                detection_times,
                s=detection_style.get("markersize", 4) ** 2,  # scatter uses area
                label="Detection",
                marker=detection_style["marker"],
                facecolors="none",
                edgecolors=detection_style.get("color"),
                linewidths=detection_style.get("markeredgewidth", 1),
                alpha=detection_style.get("alpha", 0.5),
            )
            legend_handles.append(scatter)
            legend_labels.append("Detection")

        self.ax.set_xlim(self.bearing_range_deg)
        self.ax.set_ylim([timesteps[0], timesteps[-1] + timedelta(seconds=10)])
        self.ax.invert_yaxis()
        self.ax.set_xticks(np.arange(self.bearing_range_deg[0], self.bearing_range_deg[1] + 1, 60))
        self.ax.set_xlabel("Bearing (°)")

        time_span = (timesteps[-1] - timesteps[0]).total_seconds()
        locator, formatter, label = _get_time_axis_formatter(time_span)
        self.ax.yaxis.set_major_locator(locator)
        self.ax.yaxis.set_major_formatter(formatter)
        self.ax.set_ylabel(label)

        # --- Formatting the Axes ---
        self.ax.set_xlabel("Bearing (°)")
        self.ax.yaxis_date()

        if add_colorbar:
            cax = colorbar_ax if colorbar_ax is not None else self.ax
            self.fig.colorbar(im, ax=cax, label="SNR (dB)")

        # --- Tidy up the Legend ---
        if add_legend:
            if "Ground Truth" in legend_labels:
                idx = legend_labels.index("Ground Truth")
                legend_handles[idx] = plt.Line2D(
                    [0], [0], color="black", linestyle="--", label="Ground Truth"
                )

            if legend_handles:
                self.ax.legend(
                    handles=legend_handles,
                    labels=legend_labels,
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.12),
                    ncol=len(legend_handles),
                    fancybox=True,
                    shadow=False,
                    frameon=True,
                    framealpha=0.95,
                )

        self.ax.grid(False)
        self.fig.tight_layout()
        return self.ax, im

    def _configure_axes(self, _data: dict, timesteps: list[datetime]) -> None:
        """Configure the axes for a bearing vs. time plot.

        Parameters
        ----------
            _data  : dict
                The structured data prepared for plotting. Unused here,
                but kept for consistency with the base class.
            timesteps  : list[datetime]
                The list of timestamps for the x-axis.

        """
        if not timesteps:
            return

        self.ax.set_xlim(self.bearing_range_deg)
        self.ax.set_ylim([timesteps[0], timesteps[-1] + timedelta(seconds=10)])
        self.ax.invert_yaxis()
        self.ax.set_xticks(np.arange(self.bearing_range_deg[0], self.bearing_range_deg[1] + 1, 60))
        self.ax.set_xlabel(self.style_guide["axes"]["x_label"])

        # Call the new utility function for time axis formatting
        time_span = (timesteps[-1] - timesteps[0]).total_seconds()
        locator, formatter, label = _get_time_axis_formatter(time_span)
        self.ax.yaxis.set_major_locator(locator)
        self.ax.yaxis.set_major_formatter(formatter)
        self.ax.set_ylabel(label)

    def _init_plot_objects(self, data: dict, **kwargs) -> dict:
        """Initialise artists, adding bearing-specific ones to the base artists.

        Parameters
        ----------
            data  : dict
                The structured data prepared for plotting.
            **kwargs
                Additional keyword arguments for customisation.

        Returns
        -------
            dict
                A dictionary of matplotlib artists for the plot.

        """
        # Get the common plot objects (truths, tracks, detections) from the parent
        plots = super()._init_plot_objects(data, **kwargs)
        element_styles = self.style_guide["elements"]

        # Add bearing-specific artists for uncertainty and particles
        plots["uncertainty"], plots["particles"] = {}, {}
        for track_id, track_artist in plots["tracks"].items():
            color = track_artist.get_color()
            plots["uncertainty"][track_id] = self.ax.fill_betweenx(
                [], [], [], color=color, **element_styles["uncertainty"]
            )
            plots["particles"][track_id] = self.ax.plot(
                [], [], color=color, **element_styles["particle"]
            )[0]

        return plots

    def _update_plot_objects(
        self, timestep: datetime, data: dict, plot_objects: dict, mapping: list[int]
    ) -> list:
        """Update the plot artists with data for a new frame.

        For a given timestep, this method appends the new data points to the
        history and then updates the 'data' property of the corresponding
        matplotlib artist to redraw it on the canvas.

        Parameters
        ----------
            timestep  : datetime
                The timestamp for the current animation frame.
            data  : dict
                The dictionary of all prepared simulation data.
            plot_objects  : dict
                The dictionary of matplotlib artists to update.
            mapping  : list[int]
                A mapping used to access specific components of the
                state vector.

        Returns
        -------
            list
                A list of all updated artists, required by FuncAnimation
            for efficient rendering.

        """
        current_data = data[timestep]
        history = data["history"]

        for truth_id, state in current_data["truths"].items():
            history["truths"][truth_id]["bearing"].append(
                np.rad2deg(state.state_vector[mapping[0], 0])
            )
            history["truths"][truth_id]["time"].append(state.timestamp)
            plot_objects["truths"][truth_id].set_data(
                history["truths"][truth_id]["bearing"],
                history["truths"][truth_id]["time"],
            )

        for track_id, state in current_data["tracks"].items():
            history["tracks"][track_id]["bearing"].append(
                np.rad2deg(state.state_vector[mapping[0], 0])
            )
            history["tracks"][track_id]["time"].append(state.timestamp)
            std_dev = 0
            if track_id in current_data["uncertainty"]:
                std_dev = np.rad2deg(np.sqrt(current_data["uncertainty"][track_id].covar[0, 0]))
            history["tracks"][track_id]["std_dev"].append(std_dev)
            plot_objects["tracks"][track_id].set_data(
                history["tracks"][track_id]["bearing"],
                history["tracks"][track_id]["time"],
            )

            if track_id in plot_objects["uncertainty"]:
                time_hist, bearing_hist, std_hist = (
                    history["tracks"][track_id]["time"],
                    np.array(history["tracks"][track_id]["bearing"]),
                    np.array(history["tracks"][track_id]["std_dev"]),
                )
                plot_objects["uncertainty"][track_id].set_paths(
                    [
                        np.vstack(
                            [
                                np.append(
                                    bearing_hist - std_hist,
                                    (bearing_hist + std_hist)[::-1],
                                ),
                                np.append(
                                    mdates.date2num(time_hist),
                                    mdates.date2num(time_hist)[::-1],
                                ),
                            ]
                        ).T
                    ]
                )

            if track_id in current_data["particles"]:
                part_state = current_data["particles"][track_id]
                bearings, times = (
                    np.rad2deg(part_state.state_vector[mapping[0], :]),
                    [part_state.timestamp] * len(part_state.state_vector[mapping[0], :]),
                )
                plot_objects["particles"][track_id].set_data(bearings, times)

        self._update_detection_history(
            current_data["measurements"],
            history["measurements"],
            plot_objects["meas"],
            mapping,
        )
        self._update_detection_history(
            current_data["clutter"],
            history["clutter"],
            plot_objects["clutter"],
            mapping,
        )

        return [
            artist
            for group in plot_objects.values()
            for artist in (group.values() if isinstance(group, dict) else [group])
        ]

    def _update_detection_history(
        self,
        new_detections: list[Detection],
        history_entry: dict,
        plot_artist: plt.Line2D,
        mapping: list[int],
    ) -> None:
        """Update history and plot data for detection types.

        Parameters
        ----------
            new_detections  : list[Detection]
                The list of new detections for
                the current timestep.
            history_entry  : dict
                The corresponding history dictionary to append
                data to.
            plot_artist  : plt.Line2D
                The matplotlib artist to update.
            mapping  : list[int]
                A mapping used to access specific components of the
                state vector.

        """
        if not new_detections:
            return
        history_entry["bearing"].extend(
            [np.rad2deg(d.state_vector[mapping[0], 0]) for d in new_detections]
        )
        history_entry["time"].extend([d.timestamp for d in new_detections])
        plot_artist.set_data(history_entry["bearing"], history_entry["time"])


class CartesianPlotter(BasePlotter):
    """A plotter for visualising tracks in Cartesian (x, y) space.

    This class provides a concrete implementation of the BasePlotter for creating
    plots in a 2D Cartesian coordinate system.
    """

    coord_names = ["x", "y"]  # Define coordinate names for the base class

    def __init__(self, style_guide: dict = None, offset: int = 100) -> None:
        """Initialise the Cartesian plotter with a style guide and offset.

        Parameters
        ----------
            style_guide  : dict, optional
                A dictionary to override the default
                styles for this plotter. Defaults to None.
            offset  : int, optional
                The padding in meters to add around the
                data when setting axis limits. Defaults to 100.

        """
        cartesian_defaults = {
            "axes": {"x_label": "X (metres)", "y_label": "Y (metres)"},
            "figure": {"figsize": (10, 10)},
        }
        super().__init__(defaults=cartesian_defaults, style_guide=style_guide)
        self.offset = offset

    def _convert_measurement(self, measurement: Detection) -> np.ndarray:
        """Convert a measurement's state to Cartesian coordinates.

        If the measurement has an associated measurement model, this method
        uses the model's `inverse_function` to perform the conversion.
        Otherwise, it assumes the measurement is already Cartesian.

        Parameters
        ----------
            measurement  : Detection
                The detection object to convert.

        Returns
        -------
            np.ndarray
                The state vector in Cartesian coordinates [x, y, ...].

        """
        if measurement.measurement_model:
            # This assumes the inverse function returns a Cartesian state
            return measurement.measurement_model.inverse_function(measurement)
        return measurement.state_vector

    def _configure_axes(self, data: dict, timesteps: list[datetime]):
        """Configure the plot's axes based on the range of the data.

        This method iterates through all data points to find the minimum and
        maximum x and y values, then sets the axes limits with a defined
        offset.

        Parameters
        ----------
            data  : dict
                The prepared data dictionary.
            timesteps  : list[datetime]
                A list of all simulation timestamps.

        """
        all_x, all_y = [], []
        for t in timesteps:
            for state in data[t]["platforms"].values():
                all_x.append(state.state_vector[0, 0])
                all_y.append(state.state_vector[2, 0])
            for state in data[t]["truths"].values():
                all_x.append(state.state_vector[0, 0])
                all_y.append(state.state_vector[2, 0])
            for state in data[t]["tracks"].values():
                all_x.append(state.state_vector[0, 0])
                all_y.append(state.state_vector[2, 0])
            for detection in data[t]["measurements"] + data[t]["clutter"]:
                state_vec = self._convert_measurement(detection)
                all_x.append(state_vec[0])
                all_y.append(state_vec[1])

        if not all_x or not all_y:
            return

        self.ax.set_xlim([min(all_x) - self.offset, max(all_x) + self.offset])
        self.ax.set_ylim([min(all_y) - self.offset, max(all_y) + self.offset])

        # Set labels from the style guide
        self.ax.set_xlabel(self.style_guide["axes"]["x_label"])
        self.ax.set_ylabel(self.style_guide["axes"]["y_label"])
        self.ax.set_aspect("equal", adjustable="box")

    def _update_plot_objects(
        self, timestep: datetime, data: dict, plot_objects: dict, mapping: list[int]
    ) -> list:
        """Update the plot artists with data for a new frame.

        For a given timestep, this method appends new x/y coordinates to the
        history and updates the data of the corresponding matplotlib artists.

        Parameters
        ----------
            timestep  : datetime
                The timestamp for the current animation frame.
            data  : dict
                The dictionary of all prepared simulation data.
            plot_objects  : dict
                The dictionary of matplotlib artists to update.
            mapping  : list[int]
                A mapping used to access specific components of the
                state vector.

        Returns
        -------
            List
                A list of all updated artists, required by FuncAnimation.

        """
        current_data = data[timestep]
        history = data["history"]

        for platform_id, state in current_data["platforms"].items():
            history["platforms"][platform_id]["x"].append(state.state_vector[mapping[0], 0])
            history["platforms"][platform_id]["y"].append(state.state_vector[mapping[1], 0])
            plot_objects["platforms"][platform_id].set_data(
                history["platforms"][platform_id]["x"],
                history["platforms"][platform_id]["y"],
            )

        for truth_id, state in current_data["truths"].items():
            history["truths"][truth_id]["x"].append(state.state_vector[mapping[0], 0])
            history["truths"][truth_id]["y"].append(state.state_vector[mapping[1], 0])
            plot_objects["truths"][truth_id].set_data(
                history["truths"][truth_id]["x"], history["truths"][truth_id]["y"]
            )

        for track_id, state in current_data["tracks"].items():
            history["tracks"][track_id]["x"].append(state.state_vector[mapping[0], 0])
            history["tracks"][track_id]["y"].append(state.state_vector[mapping[1], 0])
            plot_objects["tracks"][track_id].set_data(
                history["tracks"][track_id]["x"], history["tracks"][track_id]["y"]
            )

            # Check if an uncertainty artist exists for this track
            if "uncertainty" in plot_objects and track_id in plot_objects["uncertainty"]:
                ellipse = plot_objects["uncertainty"][track_id]

                xy_indices = [mapping[0], mapping[1]]
                covar_2d = state.covar[np.ix_(xy_indices, xy_indices)]

                # Proceed only if the 2x2 covariance is valid
                if np.all(np.linalg.eigvals(covar_2d) >= 0):
                    vals, vecs = np.linalg.eigh(covar_2d)
                    order = vals.argsort()[::-1]
                    vals, vecs = vals[order], vecs[:, order]

                    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
                    # Unpacking is now safe as 'vals' has 2 elements
                    width, height = 2 * 2.0 * np.sqrt(vals)

                    ellipse.center = state.state_vector[xy_indices, 0]
                    ellipse.width = width
                    ellipse.height = height
                    ellipse.angle = angle

        # Update detection history
        meas_xy = [self._convert_measurement(d) for d in current_data["measurements"]]
        history["measurements"]["x"].extend([p[mapping[0]] for p in meas_xy])
        history["measurements"]["y"].extend([p[mapping[1]] for p in meas_xy])
        plot_objects["meas"].set_data(history["measurements"]["x"], history["measurements"]["y"])

        clutter_xy = [self._convert_measurement(d) for d in current_data["clutter"]]
        history["clutter"]["x"].extend([p[mapping[0]] for p in clutter_xy])
        history["clutter"]["y"].extend([p[mapping[1]] for p in clutter_xy])
        plot_objects["clutter"].set_data(history["clutter"]["x"], history["clutter"]["y"])

        return [
            artist
            for group in plot_objects.values()
            for artist in (group.values() if isinstance(group, dict) else [group])
        ]

    def _init_plot_objects(self, data: dict, **kwargs) -> dict:
        """Initialise artists, adding Cartesian-specific ones for uncertainty."""
        plots = super()._init_plot_objects(data, **kwargs)
        element_styles = self.style_guide["elements"]

        plot_uncertainty = kwargs.get("plot_uncertainty", False)

        # Only create the uncertainty artists if they are going to be plotted.
        if plot_uncertainty:
            plots["uncertainty"] = {}
            for track_id, track_artist in plots["tracks"].items():
                color = track_artist.get_color()
                style = element_styles["uncertainty"].copy()  # Or "uncertainty"

                # Create an initial, invisible ellipse for each track
                ellipse = Ellipse(xy=(0, 0), width=0, height=0, color=color, **style)
                plots["uncertainty"][track_id] = self.ax.add_patch(ellipse)

        return plots


def _distance_axis_scale(min_val: float, max_val: float) -> tuple[float, str]:
    """Return distance scale factor and unit label from axis limits.

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
    """
    span_display_units = span * scale
    pad_display_units = max(0.5, 0.05 * span_display_units)
    return pad_display_units / scale


def plot_world(truths: list[GroundTruthPath], platform: Platform) -> go.Figure:
    """Plot the world picture of the platform and target trajectories.

    Parameters
    ----------
    truths : list[GroundTruthPath]
        A list of GroundTruthPath objects representing the trajectories of the targets.
    platform : Platform
        The platform whose trajectory is to be plotted.

    Returns
    -------
    go.Figure
        A Plotly figure object containing the world picture plot.

    """
    num_truths = len(truths)

    fig = go.Figure()
    fig.update_layout(colorway=px.colors.qualitative.Plotly)
    colorway = list(fig.layout.colorway or px.colors.qualitative.Plotly)

    plat_x = [float(entry.host.state.state_vector[0]) for entry in platform.platform_history]
    plat_y = [float(entry.host.state.state_vector[2]) for entry in platform.platform_history]

    gt_x = [[] for _ in range(num_truths)]
    gt_y = [[] for _ in range(num_truths)]
    for idx, truth in enumerate(truths):
        gt_x[idx] = [float(state.state_vector[0]) for state in truth]
        gt_y[idx] = [float(state.state_vector[2]) for state in truth]

    all_x = plat_x + [x for sublist in gt_x for x in sublist]
    all_y = plat_y + [y for sublist in gt_y for y in sublist]

    raw_min_x, raw_max_x = min(all_x), max(all_x)
    raw_min_y, raw_max_y = min(all_y), max(all_y)
    scale, unit = _distance_axis_scale(min(raw_min_x, raw_min_y), max(raw_max_x, raw_max_y))
    raw_span = max(raw_max_x - raw_min_x, raw_max_y - raw_min_y)
    pad = _range_padding_for_scale(scale, raw_span)

    min_x, max_x = raw_min_x - pad, raw_max_x + pad
    min_y, max_y = raw_min_y - pad, raw_max_y + pad

    mid_x = (max_x + min_x) / 2
    mid_y = (max_y + min_y) / 2
    max_span = max(max_x - min_x, max_y - min_y)

    x_range = [mid_x - max_span / 2, mid_x + max_span / 2]
    y_range = [mid_y - max_span / 2, mid_y + max_span / 2]

    plat_x = [x * scale for x in plat_x]
    plat_y = [y * scale for y in plat_y]
    gt_x = [[x * scale for x in x_coords] for x_coords in gt_x]
    gt_y = [[y * scale for y in y_coords] for y_coords in gt_y]
    x_range = [value * scale for value in x_range]
    y_range = [value * scale for value in y_range]

    fig.add_trace(
        go.Scatter(
            x=plat_x,
            y=plat_y,
            mode="lines",
            line=dict(color="black", width=3),
            name="Platform",
        )
    )

    names = [f"Target {i + 1}" if num_truths > 1 else "Target" for i in range(num_truths)]
    for i in range(num_truths):
        fig.add_trace(
            go.Scatter(
                x=gt_x[i],
                y=gt_y[i],
                mode="lines",
                line=dict(color=colorway[i % len(colorway)], width=3, dash="5px,2px"),
                name=names[i],
            )
        )

    fig.update_layout(
        font=dict(size=16, color="black"),
        showlegend=True,
        legend=dict(x=0.5, y=1.2, xanchor="center", orientation="h"),
        plot_bgcolor="white",
        xaxis=dict(
            title=f"X Position ({unit})",
            range=x_range,
            constrain="range",
            showgrid=True,
            gridcolor="rgba(200,200,200,0.5)",
            linecolor="black",
            zeroline=True,
            zerolinecolor="rgba(200, 200, 200, 0.5)",
            zerolinewidth=0.5,
        ),
        yaxis=dict(
            title=f"Y Position ({unit})",
            range=y_range,
            constrain="range",
            scaleanchor="x",
            scaleratio=1,
            showgrid=True,
            gridcolor="rgba(200,200,200,0.5)",
            linecolor="black",
            zeroline=True,
            zerolinecolor="rgba(200, 200, 200, 0.5)",
            zerolinewidth=0.5,
        ),
    )

    return fig


def plot_btr(
    timesteps: np.ndarray,
    steering_azimuths: np.ndarray,
    data: np.ndarray | None = None,
    truths: list[GroundTruthPath] | None = None,
    detections: list[Detection] | None = None,
    tracks: list[Track] | None = None,
    data_type: str = "SNR (dB)",
    width_height_px: tuple[int, int] = (800, 600),
) -> go.Figure:
    """Plot the bearing-time record (BTR) of the beamformed data.

    Optionally overlays truth trajectories and detections on the BTR plot.

    Parameters
    ----------
    data : np.ndarray | None
        The beamformed data to be plotted (in dB) as a heatmap. If None, no data is
        plotted.
    timesteps : np.ndarray
        The timesteps corresponding to the beamformed data.
    steering_azimuths : np.ndarray
        The steering azimuth angles corresponding to the beamformed data.
    truths : list[GroundTruthPath] | None
        A list of GroundTruthPath objects representing the trajectories of the targets.
        Default is None, in which case no truth trajectories will be plotted.
    detections : list[Detection] | None
        A list of Detection objects representing the detections to be plotted.
        Default is None, in which case no detections will be plotted.
    tracks : list[Track] | None
        A list of Track objects representing the tracks to be plotted. Default is None,
        in which case no tracks will be plotted.
    data_type : str
        A string label for the type of data being plotted (e.g., "SNR (dB)").
        This is used for the colorbar title. Default is "SNR (dB)".

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
                split_bearings.append(None)
                split_times.append(None)
            split_bearings.append(bearing)
            split_times.append(timestamp)
            prev_bearing = bearing

        return split_bearings, split_times

    fig = go.Figure()
    fig.update_layout(colorway=px.colors.qualitative.Plotly)
    colorway = list(fig.layout.colorway or px.colors.qualitative.Plotly)
    track_colorway = list(reversed(colorway))

    if data is not None:
        fig.add_trace(
            go.Heatmap(
                z=data,
                y=timesteps,
                x=steering_azimuths,
                colorscale="Viridis",
                colorbar=dict(
                    title=dict(text=data_type, side="right", font=dict(size=16)),
                    thickness=24,
                    len=1.0,
                    tickfont=dict(size=14),
                    x=0.92,
                    xpad=0,
                ),
            )
        )

    if detections is not None:
        det_x = [_wrap_bearing_deg(float(np.rad2deg(det.state_vector[0]))) for det in detections]
        det_y = [det.timestamp for det in detections]
        fig.add_trace(
            go.Scatter(
                x=det_x,
                y=det_y,
                mode="markers",
                marker=dict(size=5, line=dict(width=1), color="white", opacity=0.8),
                name="Detection",
            )
        )

    if tracks is not None:
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
            fig.add_trace(
                go.Scatter(
                    x=track_x,
                    y=track_y,
                    mode="lines",
                    connectgaps=False,
                    line=dict(color=track_color, width=4),
                    name=f"Track {idx + 1}" if len(tracks) > 1 else "Track",
                )
            )

    if truths is not None:
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
            fig.add_trace(
                go.Scatter(
                    x=truth_x,
                    y=truth_y,
                    mode="lines",
                    connectgaps=False,
                    line=dict(color=truth_color, width=3, dash="dash"),
                    name=f"Target {idx + 1}" if len(truths) > 1 else "Target",
                )
            )

    fig.update_xaxes(
        range=[steering_azimuths[0], steering_azimuths[-1]],
        domain=[0.0, 0.9],
        tickmode="linear",
        tick0=steering_azimuths[0],
        dtick=steering_azimuths[-1] // 3,
        tickangle=-45,
        tickfont=dict(size=14),
        showgrid=True,
        gridcolor="rgba(200, 200, 200, 0.5)",
        title="Bearing (°)",
        ticks="outside",
        tickcolor="rgba(160, 160, 160, 1.0)",
        showline=True,
        linewidth=1,
        linecolor="rgba(160, 160, 160, 1.0)",
    )

    fig.update_yaxes(
        range=[timesteps[-1], timesteps[0]],
        showgrid=True,
        gridcolor="rgba(200, 200, 200, 0.5)",
        tickformat="%H:%M",
        tickfont=dict(size=14),
        autorange=False,
        title="Time (HH:MM)",
        tickcolor="rgba(160, 160, 160, 1.0)",
        showline=True,
        linewidth=1,
        linecolor="rgba(160, 160, 160, 1.0)",
    )

    fig.update_layout(
        width=width_height_px[0],
        height=width_height_px[1],
        font=dict(size=16, color="black"),
        showlegend=True,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    return fig


def plot_spectrogram(
    signal: np.ndarray,
    sr: int,
    n_fft: int = 4096,
    hop_length: int = 1024,
    y_lim: tuple[float, float] | None = None,
    yaxis_format: str = "kHz",
    fig_size: tuple[int, int] = (12, 6),
    font_size_label: int = 16,
    font_size_tick: int = 14,
) -> None:
    """Generate and display a formatted spectrogram with Plotly.

    Parameters
    ----------
    signal : np.ndarray
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
        ``"kHz"`` to label y-axis in kHz or ``"hz"`` for Hz.
    fig_size : tuple[int, int]
        Figure size as ``(width, height)`` in notebook-style inches.
    font_size_label : int
        Axis label font size.
    font_size_tick : int
        Axis tick font size.

    """
    signal = np.asarray(signal)
    if signal.size == 0:
        raise ValueError("signal is empty")
    if signal.ndim > 1:
        signal = signal.flatten()
    if sr <= 0:
        raise ValueError("sr must be positive")

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
    s_db = 20.0 * np.log10(np.maximum(amin, magnitude)) - 20.0 * np.log10(ref)

    vmax = float(np.max(s_db))
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
            colorbar=dict(title="Intensity (dB)", ticksuffix=" dB"),
        )
    )

    fig.update_layout(
        width=int(fig_size[0] * 100),
        height=int(fig_size[1] * 100),
        margin=dict(l=80, r=80, t=30, b=60),
    )

    fig.update_xaxes(
        title_text="Time (s)",
        title_font=dict(size=font_size_label),
        tickfont=dict(size=font_size_tick),
        range=[0, len(signal) / float(sr)],
    )
    fig.update_yaxes(
        title_text=y_title,
        title_font=dict(size=font_size_label),
        tickfont=dict(size=font_size_tick),
        range=y_range,
    )

    return fig
