"""Defines plotting utilities.

© Copyright 2025 Joshua J. Wakefield.
Licensed under the MIT License.
"""

import copy
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from itertools import cycle
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.patches import Ellipse, Patch
from tqdm.auto import tqdm

from nereus.types.angles import Bearing
from nereus.types.detections import Clutter, Detection, MissedDetection, TrueDetection
from nereus.types.states import GroundTruthPath, Track

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
        "location": "upper left",
        "anchor": (1.02, 1),
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
            "alpha": 0.7,
            "linewidth": 0,
            "zorder": 6,
        },
        "clutter": {
            "marker": "x",
            "markersize": 4,
            "color": "grey",
            "linewidth": 0,
            "alpha": 0.6,
            "zorder": 5,
        },
    },
}


def _get_time_axis_formatter(time_span_seconds: float) -> tuple:
    """Calculate the optimal matplotlib locator and formatter for a time axis.

    Args:
        time_span_seconds (float): The total duration of the time span in seconds.

    Returns:
        tuple: A tuple containing:
            - locator (matplotlib.dates.DateLocator): The locator for the x-axis.
            - formatter (matplotlib.dates.DateFormatter): The formatter for the x-axis.
            - label (str): A label for the x-axis.

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

    def __init__(
        self, defaults: dict | None = None, style_guide: dict | None = None
    ) -> None:
        """Initialise the plotter using a comprehensive style guide.

        Args:
            defaults (dict): A dictionary of default styles to apply to the plotter.
                This is merged with the class-specific defaults.
            style_guide (dict): A dictionary to override the default styles for this
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
        ax: Axes | None = None,
        **kwargs: Any,
    ) -> Axes:
        """Generate a static plot of the final state of a tracking scenario.

        This method plots all historical data up to the final timestep.

        Args:
            truths (list[GroundTruthPath]): A list of ground truth paths.
            tracks (list[Track]): A list of track objects.
            all_detections (list[set[Detection]]): A list of detection sets,
                one for each timestep.
            timesteps (list[datetime]): A list of timestamps for the x-axis.
            mapping (list[int]): Mapping used by some subclasses to access
                specific components of the state vector.
            ax (Axes | None): An existing matplotlib Axes object to plot on.
                If None, a new figure and axes are created.
            **kwargs (Any): Additional keyword arguments passed to helper methods.

        Returns:
            Axes: The matplotlib Axes object containing the plot.

        """
        data, plot_objects = self._setup_plot(
            truths, tracks, all_detections, timesteps, mapping, ax, **kwargs
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
        **kwargs: Any,
    ) -> tuple[Axes, FuncAnimation]:
        """Generate an animation of the tracking process over time.

        This method creates an animation that updates the plot for each timestep,
        showing the evolution of the tracking scenario over time.

        Args:
            truths (List[GroundTruthPath]): A list of ground truth paths.
            tracks (List[Track]): A list of track objects.
            all_detections (List[Set[Detection]]): A list of detection sets.
            timesteps (List[datetime]): A list of timestamps for each frame.
            mapping (List[int], optional): Mapping used by some subclasses.
            **kwargs (Any): Additional keyword arguments passed to helper methods.

        Returns:
            Tuple[Axes, FuncAnimation]: The axes and the animation object. The
            animation object must be kept in scope to be displayed.

        """
        self.num_timesteps = len(timesteps)
        data, plot_objects = self._setup_plot(
            truths, tracks, all_detections, timesteps, mapping, None, **kwargs
        )
        self.fig.tight_layout()

        def _update(k: int) -> list:
            """Update function for FuncAnimation.

            Args:
                k (int): The current frame index.

            Returns:
                list: The updated plot objects for the current frame.

            """
            return self._update_plot_objects(timesteps[k], data, plot_objects, mapping)

        self.anim = FuncAnimation(
            self.fig, _update, frames=range(len(timesteps)), repeat=True, interval=100
        )

        return self.ax, self.anim

    def show(self, tight_layout: bool = True) -> None:
        """Display the generated plot or animation.

        Args:
            tight_layout (bool): Whether to apply tight layout to the figure.
                Defaults to True. If False, the layout will not be adjusted.

        Raises:
            ValueError: If no plot or animation has been created.

        """
        if not self.fig and not self.anim:
            raise ValueError(
                "No plot or animation to show. Call plot() or animate() first."
            )
        if tight_layout and self.fig:
            self.fig.tight_layout()
        plt.show()

    def save(self, filename: str, **kwargs: Any) -> None:
        """Save the plot or animation to a file using a tqdm progress bar.

        This method can save static plots (e.g., 'figure.png') or animations
        (e.g., 'animation.gif' or 'animation.mp4').

        Args:
            filename (str): The path and name of the file to save.
            **kwargs (Any): Additional keyword arguments passed to the underlying
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
            raise ValueError(
                "No plot or animation to save. Call plot() or animate() first."
            )

    def close(self):
        """Close the current plot figure."""
        if self.fig:
            plt.close(self.fig)

    def _setup_figure(self, ax: Axes | None = None) -> None:
        """Initialise the matplotlib figure and axes objects.

        Args:
            ax (Axes | None): An existing matplotlib Axes object to use.

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
        ax: Axes,
        **kwargs: Any,
    ) -> tuple[dict, dict]:
        """Handle the common setup sequence for both plots and animations.

        Args:
            truths (list[GroundTruthPath]): A list of ground truth paths.
            tracks (list[Track]): A list of track objects.
            all_detections (list[Detection]): A list of detection sets, one for each
                timestep.
            timesteps (list[datetime]): A list of timestamps for the x-axis.
            mapping (dict): Mapping used by some subclasses.
            ax (Axes): An existing matplotlib Axes object to plot on.
            **kwargs: Additional keyword arguments for customisation.

        Returns:
            tuple: A tuple containing:
                - data (dict): A structured dictionary with keys for truths,
                  tracks, uncertainty, particles, measurements, and clutter.
                - plot_kwargs (dict): A dictionary of plot-specific parameters
                  derived from the input arguments.

        """
        self._setup_figure(ax)

        # Configure title from the style guide
        title_cfg = self.style_guide["title"]
        if title_cfg.get("text"):
            self.ax.set_title(
                title_cfg["text"],
                fontsize=(
                    self.style_guide["figure"]["font_size"]
                    + title_cfg["fontsize_offset"]
                ),
                pad=title_cfg["pad"],
            )

        data, plot_kwargs = self._prepare_data(
            truths, tracks, all_detections, timesteps, mapping, **kwargs
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

        Args:
            data (dict): The structured data containing truths, tracks, and detections.
            timesteps (list[datetime]): The list of timesteps for the plot.

        """
        raise NotImplementedError

    def _configure_legend(self, data: dict, **kwargs: Any) -> None:
        """Configure the legend dynamically based on the plotted data.

        This method creates legend handles based on the presence of truths, tracks,
        and detections in the data.

        Args:
            data (dict): The structured data containing truths, tracks, and detections.
            **kwargs: Additional keyword arguments for customisation.

        """
        handles = []
        element_styles = self.style_guide["elements"]

        if data["history"]["truths"]:
            # Create a representative line for the legend from the style guide
            style = {
                k: v
                for k, v in element_styles["truth"].items()
                if k not in ["alpha", "zorder"]
            }
            handles.append(plt.Line2D([0], [0], color="k", label="Truth", **style))

        if data["history"]["tracks"]:
            style = {
                k: v
                for k, v in element_styles["track"].items()
                if k not in ["alpha", "zorder"]
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
                handles.append(
                    plt.Line2D([0], [0], color="k", label="Particle", **style)
                )

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
                k: v
                for k, v in element_styles["clutter"].items()
                if k not in ["alpha", "zorder"]
            }
            handles.append(plt.Line2D([0], [0], label="Clutter", **style))

        if handles:
            legend_cfg = self.style_guide["legend"]
            self.ax.legend(
                handles=handles,
                loc=legend_cfg["location"],
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

        Args:
            timestep (datetime): The current timestep to update the plot for.
            data (dict): The structured data containing truths, tracks, and detections.
            plot_objects (dict): The dictionary of plot objects created by
                `_init_plot_objects`.
            mapping (list[int]): A mapping used to access specific components of the
                state vector.

        Returns:
            list: A list of the updated artists, required by FuncAnimation.

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
        **kwargs,
    ) -> tuple[dict, dict]:
        """Structure input data for plotting.

        This method prepares the data for plotting by organising truths, tracks,
        and detections into a structured format.

        Args:
            truths (list[GroundTruthPath]): A list of ground truth paths.
            tracks (list[Track]): A list of track objects.
            all_detections (list[set[Detection]]): A list of detection sets,
                one for each timestep.
            timesteps (list[datetime]): A list of timestamps for the x-axis.
            mapping (list | tuple, optional): Mapping used by some subclasses.
            **kwargs: Additional keyword arguments for customisation.

        Returns:
            tuple: A tuple containing:
                - data (dict): A structured dictionary with keys for truths,
                  tracks, uncertainty, particles, measurements, and clutter.
                - plot_kwargs (dict): A dictionary of plot-specific parameters
                  derived from the input arguments.

        """
        # Check for different detection types
        has_true_detections = any(
            isinstance(d, TrueDetection) for s in all_detections for d in s
        )
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
        for det_set in all_detections:
            for det in det_set:
                if isinstance(det, MissedDetection):
                    continue
                category = (
                    "clutter"
                    if (
                        isinstance(det, Clutter)
                        and plot_kwargs["distinguish_detections"]
                    )
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
        }

        return data, plot_kwargs

    def _init_plot_objects(self, data: dict, **kwargs: Any) -> dict:
        """Create initial, empty plot artists for common elements.

        Args:
            data (dict): The structured data prepared for plotting.
            **kwargs (Any): Additional keyword arguments for specific plot styles.

        Returns:
            dict: A dictionary of plot objects, keyed by their type (e.g., "truths",
            "tracks", "measurements", "clutter").

        """
        plots = {"truths": {}, "tracks": {}}
        element_styles = self.style_guide["elements"]

        # Truths
        for truth_id in data["history"]["truths"]:
            style = element_styles["truth"].copy()
            (plots["truths"][truth_id],) = self.ax.plot(
                [], [], color=self.get_color(), **style
            )
        # Tracks
        for track_id in data["history"]["tracks"]:
            style = element_styles["track"].copy()
            (plots["tracks"][track_id],) = self.ax.plot(
                [], [], color=self.get_color(), **style
            )

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

    def __init__(self, style_guide: dict | None = None) -> None:
        """Initialise the plotter using a comprehensive style guide.

        Args:
            style_guide (dict | None): A dictionary to override the
                default styles for this plotter.

        """
        # Define the default styles specific to a bearings plot
        bearings_defaults = {
            "axes": {"x_label": "Bearing (°)", "y_label": "Time"},
        }

        super().__init__(defaults=bearings_defaults, style_guide=style_guide)

    def plot_snr(
        self,
        snr_array: np.ndarray,
        timesteps: list[datetime],
        all_detections: list[set[Detection]],
        truths: list[GroundTruthPath] = None,
        ax: Axes = None,
        add_colorbar: bool = True,
        colorbar_ax: list[Axes] = None,
    ) -> Axes:
        """Plot a heatmap of SNR vs. time, with detections overlaid.

        This method generates a "waterfall" plot to visualise the raw SNR
        data from the beamformer, which is useful for diagnosing the detection
        process.

        Args:
            snr_array (np.ndarray): A 2D array of SNR values from the
                simulation loop.
            timesteps (List[datetime]): A list of all simulation timestamps.
            all_detections (List[Set[Detection]]): A list of detection sets
                to overlay on the heatmap.
            truths (List[GroundTruthPath], optional): A list of ground truth
                paths to overlay for context. Defaults to None.
            ax (Axes, optional): An existing matplotlib Axes object to plot on.
                If None, a new figure and axes are created.
            add_colorbar (bool): Whether to add a colorbar to the plot.
                Defaults to True.
            colorbar_ax (List[Axes], optional): If provided, the colorbar will
                be added to this Axes object instead of the main Axes.

        Returns:
            Axes: The matplotlib Axes object containing the plot.

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
                0,
                180,
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
                (line,) = self.ax.plot(
                    bearings, times, label="Ground Truth", **truth_style
                )
            legend_handles.append(line)
            legend_labels.append("Ground Truth")

        # 3. Overlay the detections (if any)
        if detection_bearings and detection_times:
            detection_style = element_styles["detection"].copy()
            detection_style["color"] = "red"  # Override color for visibility
            scatter = self.ax.scatter(
                detection_bearings,
                detection_times,
                s=20,  # Keep a fixed size for clarity
                label="Detection",
                marker=detection_style["marker"],
                color=detection_style["color"],
            )
            legend_handles.append(scatter)
            legend_labels.append("Detection")

        self.ax.set_xlim([0, 180])
        self.ax.set_ylim([timesteps[0], timesteps[-1] + timedelta(seconds=10)])
        self.ax.invert_yaxis()
        self.ax.set_xticks(np.arange(0, 181, 30))
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
        if "Ground Truth" in legend_labels:
            idx = legend_labels.index("Ground Truth")
            legend_handles[idx] = plt.Line2D(
                [0], [0], color="black", linestyle="--", label="Ground Truth"
            )

        if legend_handles:
            self.ax.legend(
                handles=legend_handles,
                labels=legend_labels,
                loc="upper left",
                bbox_to_anchor=(1.35, 1),
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

        Args:
            _data (dict): The structured data prepared for plotting. Unused here,
                but kept for consistency with the base class.
            timesteps (list[datetime]): The list of timestamps for the x-axis.

        """
        if not timesteps:
            return

        self.ax.set_xlim([0, 180])
        self.ax.set_ylim([timesteps[0], timesteps[-1] + timedelta(seconds=10)])
        self.ax.invert_yaxis()
        self.ax.set_xticks(np.arange(0, 181, 30))
        self.ax.set_xlabel(self.style_guide["axes"]["x_label"])

        # Call the new utility function for time axis formatting
        time_span = (timesteps[-1] - timesteps[0]).total_seconds()
        locator, formatter, label = _get_time_axis_formatter(time_span)
        self.ax.yaxis.set_major_locator(locator)
        self.ax.yaxis.set_major_formatter(formatter)
        self.ax.set_ylabel(label)

    def _init_plot_objects(self, data: dict, **kwargs) -> dict:
        """Initialise artists, adding bearing-specific ones to the base artists.

        Args:
            data (dict): The structured data prepared for plotting.
            **kwargs: Additional keyword arguments for customisation.

        Returns:
            dict: A dictionary of matplotlib artists for the plot.

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

        Args:
            timestep (datetime): The timestamp for the current animation frame.
            data (dict): The dictionary of all prepared simulation data.
            plot_objects (dict): The dictionary of matplotlib artists to update.
            mapping (list[int]): A mapping used to access specific components of the
                state vector.

        Returns:
            list: A list of all updated artists, required by FuncAnimation
            for efficient rendering.

        """
        current_data = data[timestep]
        history = data["history"]

        for truth_id, state in current_data["truths"].items():
            history["truths"][truth_id]["bearing"].append(
                np.rad2deg(Bearing(state.state_vector[mapping[0], 0]))
            )
            history["truths"][truth_id]["time"].append(state.timestamp)
            plot_objects["truths"][truth_id].set_data(
                history["truths"][truth_id]["bearing"],
                history["truths"][truth_id]["time"],
            )

        for track_id, state in current_data["tracks"].items():
            history["tracks"][track_id]["bearing"].append(
                np.rad2deg(Bearing(state.state_vector[mapping[0], 0]))
            )
            history["tracks"][track_id]["time"].append(state.timestamp)
            std_dev = 0
            if track_id in current_data["uncertainty"]:
                std_dev = np.rad2deg(
                    np.sqrt(current_data["uncertainty"][track_id].covar[0, 0])
                )
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
                    np.rad2deg(Bearing(part_state.state_vector[mapping[0], :])),
                    [part_state.timestamp]
                    * len(part_state.state_vector[mapping[0], :]),
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

        Args:
            new_detections (list[Detection]): The list of new detections for
                the current timestep.
            history_entry (dict): The corresponding history dictionary to append
                data to.
            plot_artist (plt.Line2D): The matplotlib artist to update.
            mapping (list[int]): A mapping used to access specific components of the
                state vector.

        """
        if not new_detections:
            return
        history_entry["bearing"].extend(
            [np.rad2deg(Bearing(d.state_vector[mapping[0], 0])) for d in new_detections]
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
        """Initialise the Cartesian plotter with a style guide and offset."""
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

        Args:
            measurement (Detection): The detection object to convert.

        Returns:
            np.ndarray: The state vector in Cartesian coordinates [x, y, ...].

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

        Args:
            data (dict): The prepared data dictionary.
            timesteps (list[datetime]): A list of all simulation timestamps.

        """
        all_x, all_y = [], []
        for t in timesteps:
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

        Args:
            timestep (datetime): The timestamp for the current animation frame.
            data (dict): The dictionary of all prepared simulation data.
            plot_objects (dict): The dictionary of matplotlib artists to update.
            mapping (list[int]): A mapping used to access specific components of the
                state vector.

        Returns:
            List: A list of all updated artists, required by FuncAnimation.

        """
        current_data = data[timestep]
        history = data["history"]

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
            if (
                "uncertainty" in plot_objects
                and track_id in plot_objects["uncertainty"]
            ):
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
        plot_objects["meas"].set_data(
            history["measurements"]["x"], history["measurements"]["y"]
        )

        clutter_xy = [self._convert_measurement(d) for d in current_data["clutter"]]
        history["clutter"]["x"].extend([p[mapping[0]] for p in clutter_xy])
        history["clutter"]["y"].extend([p[mapping[1]] for p in clutter_xy])
        plot_objects["clutter"].set_data(
            history["clutter"]["x"], history["clutter"]["y"]
        )

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
                style = element_styles[
                    "uncertainty_cartesian"
                ].copy()  # Or "uncertainty"

                # Create an initial, invisible ellipse for each track
                ellipse = Ellipse(xy=(0, 0), width=0, height=0, color=color, **style)
                plots["uncertainty"][track_id] = self.ax.add_patch(ellipse)

        return plots
        return plots
        return plots
        return plots
        return plots
