"""Writing KRAKEN / KRAKENC environment files for a range-independent waveguide."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from ...environment.seabed import LayeredSeabed, SeabedLayer


@dataclass(frozen=True)
class KrakenEnvironment:
    """Everything KRAKEN needs to describe a range-independent waveguide.

    Attributes
    ----------
    ssp_depths_m, ssp_speeds_m_s : tuple of float
        Water sound speed profile, starting at 0 m and ending at ``water_depth_m``.
    water_depth_m : float
        Water depth in metres.
    seabed : LayeredSeabed
        Sediment layers and halfspace.
    thorp_attenuation : bool
        Whether to add Thorp volume attenuation (KRAKEN top option ``'T'``).
    ssp_interpolation : str
        KRAKEN SSP option: ``'C'`` (c-linear), ``'N'`` (n^2-linear) or ``'S'`` (spline).

    """

    ssp_depths_m: tuple[float, ...]
    ssp_speeds_m_s: tuple[float, ...]
    water_depth_m: float
    seabed: LayeredSeabed
    thorp_attenuation: bool = True
    ssp_interpolation: str = "C"

    def __post_init__(self) -> None:
        """Validate the profile against the water depth."""
        depths = np.asarray(self.ssp_depths_m, dtype=float)
        speeds = np.asarray(self.ssp_speeds_m_s, dtype=float)
        if depths.size < 2 or depths.shape != speeds.shape:
            raise ValueError("The SSP needs at least two points with matching depths and speeds")
        if depths[0] != 0.0 or not np.isclose(depths[-1], self.water_depth_m):
            raise ValueError("The SSP must start at 0 m and end at the water depth")
        if np.any(np.diff(depths) <= 0):
            raise ValueError("SSP depths must be strictly increasing")
        if self.ssp_interpolation not in {"C", "N", "S"}:
            raise ValueError(f"Unknown SSP interpolation {self.ssp_interpolation!r}")

    @property
    def min_speed_m_s(self) -> float:
        """Slowest water sound speed, in m/s."""
        return float(np.min(self.ssp_speeds_m_s))

    @property
    def bottom_depth_m(self) -> float:
        """Depth of the top of the halfspace, in metres."""
        return float(self.water_depth_m + self.seabed.total_thickness_m)


def _fmt(value: float) -> str:
    """Format a number for a Fortran list-directed read without losing precision."""
    return f"{float(value):.15g}"


def _layer_line(depth_m: float, speed_m_s: float, layer: SeabedLayer) -> str:
    return (
        f"{_fmt(depth_m)} {_fmt(speed_m_s)} {_fmt(layer.shear_speed_m_s)} "
        f"{_fmt(layer.density_g_cm3)} {_fmt(layer.attenuation_db_per_m_khz)} "
        f"{_fmt(layer.shear_attenuation_db_per_m_khz)} /"
    )


def environment_text(
    environment: KrakenEnvironment,
    frequencies_hz: ArrayLike,
    source_depths_m: ArrayLike,
    receiver_depths_m: ArrayLike,
    c_low_m_s: float = 0.0,
    c_high_m_s: float | None = None,
    rmax_km: float = 0.0,
    title: str = "bluepebble",
    nominal_frequency_hz: float | None = None,
    root_restarts: bool = True,
    mesh_points_per_wavelength: float | None = None,
) -> str:
    """Return the text of a broadband KRAKEN environment file.

    The modes are tabulated at the merged source and receiver depths. Attenuation is
    written in dB/(m kHz) (option ``'F'``). Every frequency goes in one file with the
    broadband option ``'B'``, so KRAKEN writes one mode set per frequency.

    Parameters
    ----------
    environment : KrakenEnvironment
        The waveguide.
    frequencies_hz : sequence of float
        Frequencies to compute modes at, in Hz.
    source_depths_m, receiver_depths_m : sequence of float
        Tabulation depths in metres, positive down. A receiver list of two values with
        more requested points is not used: depths are always written out in full.
    c_low_m_s : float, optional
        Lowest phase speed to search, in m/s. 0 lets KRAKEN choose.
    c_high_m_s : float, optional
        Highest phase speed kept, in m/s. Defaults to the halfspace speed, which keeps
        the trapped modes only. KRAKENC finds leaky modes above it.
    rmax_km : float, optional
        Maximum range of interest, in km. KRAKEN refines its mesh until the phase error at
        this range is small, so set it to the furthest range to be computed.
    title : str, optional
        Title written in the file.
    nominal_frequency_hz : float, optional
        KRAKEN's reference frequency ``freq0``; defaults to the highest frequency. KRAKEN
        sizes its mesh at ``freq0`` and scales it by ``freq / freq0``, so files that share
        ``freq0`` share meshes, and splitting a grid over several files changes nothing.
    mesh_points_per_wavelength : float, optional
        Finite-difference mesh density at ``freq0`` in each medium, from its slowest
        speed. ``None`` lets KRAKEN choose 20 points per wavelength. KRAKEN scales the
        mesh with frequency, so the density holds at every frequency.
    root_restarts : bool, optional
        Set KRAKENC's option ``'.'`` (fifth top-option character), which restarts its
        secant root finder from random points before concluding that every mode has been
        found. Without it KRAKENC can miss modes, typically near-degenerate ones that
        live in a sediment layer. KRAKEN ignores the option.

    Returns
    -------
    str
        The file's contents.

    """
    frequencies = np.asarray(frequencies_hz, dtype=float).ravel()
    if frequencies.size == 0 or np.any(frequencies <= 0):
        raise ValueError("Frequencies must be positive and non-empty")
    seabed = environment.seabed
    halfspace = seabed.halfspace
    if c_high_m_s is None:
        c_high_m_s = halfspace.top_speed_m_s
    if not c_high_m_s > c_low_m_s:
        raise ValueError("c_high_m_s must exceed c_low_m_s")

    num_media = 1 + len(seabed.layers)
    thorp = "T" if environment.thorp_attenuation else " "
    restart = "." if root_restarts else " "
    options = f"{environment.ssp_interpolation}VF{thorp}{restart}B"

    freq0 = float(np.max(frequencies)) if nominal_frequency_hz is None else nominal_frequency_hz
    lines = [f"'{title}'", _fmt(freq0), str(num_media), f"'{options}'"]

    def mesh(thickness: float, slowest: float) -> int:
        if mesh_points_per_wavelength is None:
            return 0
        wavelength = slowest / freq0
        return max(10, int(np.ceil(thickness / wavelength * mesh_points_per_wavelength)))

    # Water column: zero density contrast at the surface; the vacuum top is option 'V'.
    depth = float(environment.water_depth_m)
    lines.append(f"{mesh(depth, environment.min_speed_m_s)} 0.0 {_fmt(depth)}")
    for index, (z, c) in enumerate(
        zip(environment.ssp_depths_m, environment.ssp_speeds_m_s, strict=True)
    ):
        if index == 0:
            lines.append(f"{_fmt(z)} {_fmt(c)} 0.0 1.0 0.0 0.0 /")
        else:
            lines.append(f"{_fmt(z)} {_fmt(c)} /")

    # Each sediment layer is its own medium, with a linear speed gradient.
    for layer in seabed.layers:
        assert layer.thickness_m is not None  # LayeredSeabed rejects layers without one
        top = depth
        depth = top + float(layer.thickness_m)
        slowest = min(layer.top_speed_m_s, layer.bottom_speed_m_s)
        if layer.shear_speed_m_s > 0:
            slowest = min(slowest, layer.shear_speed_m_s)
        lines.append(f"{mesh(depth - top, slowest)} 0.0 {_fmt(depth)}")
        lines.append(_layer_line(top, layer.top_speed_m_s, layer))
        lines.append(_layer_line(depth, layer.bottom_speed_m_s, layer))

    lines.append("'A' 0.0")
    lines.append(_layer_line(depth, halfspace.top_speed_m_s, halfspace))
    lines.append(f"{_fmt(c_low_m_s)} {_fmt(c_high_m_s)}")
    lines.append(_fmt(rmax_km))

    for depths in (source_depths_m, receiver_depths_m):
        values = np.asarray(depths, dtype=float).ravel()
        if values.size == 0:
            raise ValueError("Source and receiver depth lists must be non-empty")
        lines.append(str(values.size))
        lines.extend(_wrap(values))

    lines.append(str(frequencies.size))
    lines.extend(_wrap(frequencies))
    return "\n".join(lines) + "\n"


def _wrap(values: np.ndarray, per_line: int = 8) -> list[str]:
    """Write a vector over several lines, terminated by a slash."""
    chunks = [values[i : i + per_line] for i in range(0, values.size, per_line)]
    out = [" ".join(_fmt(v) for v in chunk) for chunk in chunks]
    out[-1] += " /"
    return out


def field_parameters_text(
    source_depths_m: ArrayLike,
    receiver_depths_m: ArrayLike,
    ranges_km: ArrayLike,
    max_modes: int = 20000,
    title: str = "bluepebble",
) -> str:
    """Return a FIELD ``.flp`` file for a range-independent point source.

    Parameters
    ----------
    source_depths_m, receiver_depths_m : sequence of float
        Depths in metres.
    ranges_km : sequence of float
        Receiver ranges in km.
    max_modes : int, optional
        Most modes FIELD may sum.
    title : str, optional
        Title written to the shade file.

    Returns
    -------
    str
        The file's contents.

    """
    lines = [f"'{title}'", "'RA'", str(int(max_modes)), "1", "0.0 /"]
    for values in (ranges_km, source_depths_m, receiver_depths_m):
        array = np.asarray(values, dtype=float).ravel()
        lines.append(str(array.size))
        lines.extend(_wrap(array))
    # FIELD needs one range offset (array tilt) per receiver depth; all zero here.
    num_receivers = np.asarray(receiver_depths_m).size
    lines.append(str(num_receivers))
    lines.extend(_wrap(np.zeros(num_receivers)))
    return "\n".join(lines) + "\n"


def write_text(path: Path, text: str) -> Path:
    """Write ``text`` to ``path``, creating its directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path
