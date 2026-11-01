"""Eigenray propagation with Bellhop (``bellhopcxx``) in arrivals mode.

Bellhop is an external program, not a Python dependency: install ``bellhopcxx`` from the
bellhopcuda project (https://github.com/A-New-BellHope/bellhopcuda) and put it on ``PATH``, or
give its full path as ``exe_path``.
"""

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from shutil import which
from typing import TYPE_CHECKING, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from stonesoup.base import Base, Property

if TYPE_CHECKING:
    from ..environment import Bathymetry, SoundSpeedProfile

FloatArray: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True)
class Eigenray:
    """One arrival returned by Bellhop in arrivals mode.

    A single geometric path usually comes back as two or more arrivals from neighbouring
    beams, whose amplitudes add up to the path's amplitude, so sum every arrival rather than
    picking one per path.

    Attributes
    ----------
    amplitude : float
        Linear pressure amplitude (not dB), relative to the pressure 1 m from the source.
    phase_rad : float
        Phase of the arrival in radians, as Bellhop reports it: for example, pi for a path
        with one reflection off the pressure-release surface.
    delay_s : float
        One-way travel time in seconds.
    src_angle_deg : float
        Ray angle at the source in degrees, as Bellhop reports it (positive is downward).
    rcv_angle_deg : float
        Ray angle at the receiver in degrees.
    n_surface_bounces : int
        Number of surface reflections along the path.
    n_bottom_bounces : int
        Number of bottom reflections along the path.

    """

    amplitude: float
    phase_rad: float
    delay_s: float
    src_angle_deg: float
    rcv_angle_deg: float
    n_surface_bounces: int
    n_bottom_bounces: int


class BellhopAcousticPropagationModel(Base):
    """One-way eigenray propagation between two points, computed by Bellhop.

    Each call writes a Bellhop environment (``.env``) and range-dependent bathymetry (``.bty``)
    file for the vertical plane through the source and receiver, runs ``bellhopcxx`` in
    arrivals mode with geometric beams, and parses the arrivals (``.arr``) file. An active
    sonar round trip is two calls: sonar to target, then target to sonar.

    This is not a :class:`~bluepebble.models.propagation.SpectrumPropagationModel`: it returns
    discrete time-domain arrivals rather than transfer functions.

    Parameters
    ----------
    ssp : SoundSpeedProfile
        Sound speed profile, used as a function of depth only.
    bathymetry : Bathymetry
        Water depth, sampled along the source-receiver line.
    exe_path : str, optional
        Name or full path of the ``bellhopcxx`` executable.
    n_ssp_samples : int, optional
        Number of depths in the sound speed table written to the environment file.
    n_bathy_samples : int, optional
        Number of ranges in the bathymetry profile.
    bottom_sound_speed_mps : float, optional
        Compressional sound speed of the sediment half-space, in m/s.
    bottom_density_g_cm3 : float, optional
        Density of the sediment half-space, in g/cm³.
    bottom_attenuation_db_per_wavelength : float, optional
        Compressional attenuation of the sediment half-space, in dB per wavelength.
    beam_angle_limits_deg : tuple of float, optional
        Lowest and highest launch angles of the beam fan, in degrees.

    Assumptions
    -----------
    - Propagation is in the vertical plane through the source and receiver (2D, Nx2D is not
      used), so out-of-plane bathymetry and refraction are ignored.
    - The sound speed varies with depth only: the same profile applies at every range.
    - The surface is a flat pressure-release (vacuum) boundary.
    - The bottom is a fluid half-space (no shear) below the bathymetry, with a flat-interface
      reflection coefficient and no roughness.

    """

    ssp: "SoundSpeedProfile" = Property(doc="Sound speed profile for the water column")
    bathymetry: "Bathymetry" = Property(doc="Bathymetry model (water depth)")
    exe_path: str = Property(
        default="bellhopcxx", doc="Name or full path of the bellhopcxx executable"
    )
    n_ssp_samples: int = Property(
        default=50, doc="Number of depths in the sound speed table written to the env file"
    )
    n_bathy_samples: int = Property(
        default=50, doc="Number of ranges in the bathymetry profile written to the bty file"
    )
    bottom_sound_speed_mps: float = Property(
        default=1700.0, doc="Compressional sound speed of the sediment half-space, in m/s"
    )
    bottom_density_g_cm3: float = Property(
        default=1.5, doc="Density of the sediment half-space, in g/cm^3"
    )
    bottom_attenuation_db_per_wavelength: float = Property(
        default=0.5, doc="Compressional attenuation of the sediment half-space, in dB/wavelength"
    )
    beam_angle_limits_deg: tuple[float, float] = Property(
        default=(-89.0, 89.0), doc="Lowest and highest beam launch angles, in degrees"
    )

    def compute_eigenrays(
        self,
        source_position: ArrayLike,
        receiver_position: ArrayLike,
        frequency_hz: float,
    ) -> list[Eigenray]:
        """Run Bellhop for one source and receiver and return every arrival.

        Parameters
        ----------
        source_position : ArrayLike
            World-frame ``(x, y, z)`` of the source in metres, with z negative below the
            surface (Blue Pebble's +z up convention).
        receiver_position : ArrayLike
            World-frame ``(x, y, z)`` of the receiver in metres, with the same convention.
        frequency_hz : float
            Frequency in Hz, which sets the bottom loss and beam widths.

        Returns
        -------
        list[Eigenray]
            Every arrival Bellhop found; empty if it found no paths.

        Raises
        ------
        ValueError
            If either point is not below the surface, or the two are at the same horizontal
            position.
        FileNotFoundError
            If ``bellhopcxx`` cannot be found.
        RuntimeError
            If ``bellhopcxx`` fails.

        """
        source = np.asarray(source_position, dtype=float).ravel()
        receiver = np.asarray(receiver_position, dtype=float).ravel()
        source_depth_m, receiver_depth_m = -source[2], -receiver[2]
        if source_depth_m <= 0 or receiver_depth_m <= 0:
            raise ValueError(
                "Source and receiver must be below the surface (z < 0); got z = "
                f"{source[2]} and {receiver[2]}."
            )
        range_m = float(np.hypot(*(receiver[:2] - source[:2])))
        if range_m <= 0:
            raise ValueError("Source and receiver must be at different horizontal positions.")

        exe = self._resolve_exe()
        range_km, bottom_depth_m = self._bathymetry_profile(source[:2], receiver[:2], range_m)
        water_depth_m = float(bottom_depth_m.max())
        with tempfile.TemporaryDirectory() as tmp:
            stem = Path(tmp) / "bp"
            self._write_env(
                stem, source_depth_m, receiver_depth_m, range_m, water_depth_m, frequency_hz
            )
            self._write_bty(stem, range_km, bottom_depth_m)
            self._run_bellhopcxx(exe, stem)
            return self._parse_arr(stem.with_suffix(".arr"))

    def _resolve_exe(self) -> str:
        """Return the full path of ``bellhopcxx``, looking it up once."""
        cached = getattr(self, "_exe_resolved", None)
        if cached is None:
            cached = which(self.exe_path)
            if cached is None:
                raise FileNotFoundError(
                    f"bellhopcxx executable '{self.exe_path}' not found. Build bellhopcxx from "
                    "https://github.com/A-New-BellHope/bellhopcuda and put it on PATH, or "
                    "pass its full path as exe_path."
                )
            self._exe_resolved = cached
        return cached

    def _bathymetry_profile(
        self, source_xy_m: FloatArray, receiver_xy_m: FloatArray, range_m: float
    ) -> tuple[FloatArray, FloatArray]:
        """Sample the water depth along the source-receiver line.

        Returns
        -------
        tuple of FloatArray
            Ranges from the source in km, and positive water depths in metres.

        """
        t = np.linspace(0.0, 1.0, self.n_bathy_samples)
        points = source_xy_m[None, :] + t[:, None] * (receiver_xy_m - source_xy_m)[None, :]
        # Bathymetry models give the seabed as z below the surface (negative); Bellhop wants
        # positive depths.
        depth_m = np.array(
            [abs(float(self.bathymetry.get_depth(float(x), float(y)))) for x, y in points]
        )
        return t * range_m / 1000.0, depth_m

    def _write_env(
        self,
        stem: Path,
        source_depth_m: float,
        receiver_depth_m: float,
        range_m: float,
        water_depth_m: float,
        frequency_hz: float,
    ) -> None:
        """Write a Bellhop arrivals-mode environment file."""
        depths = np.linspace(0.0, water_depth_m, self.n_ssp_samples)
        speeds = [float(self.ssp.calculate(d)) for d in depths]
        ssp_lines = "\n".join(
            f"  {d:.2f}  {c:.4f}  /" for d, c in zip(depths, speeds, strict=True)
        )
        range_km = range_m / 1000.0
        low_angle, high_angle = self.beam_angle_limits_deg
        env = "\n".join(
            [
                "'BellhopArrivals'",
                f"{frequency_hz:.4f}",
                "1",  # number of media
                "'SVW'",  # spline SSP, vacuum above the surface, attenuation in dB/wavelength
                f"0 0.0 {water_depth_m:.2f}",
                ssp_lines,
                "'A~' 0.0",  # acousto-elastic half-space below a .bty bathymetry, no roughness
                f"{water_depth_m:.2f}  {self.bottom_sound_speed_mps}  0.0  "
                f"{self.bottom_density_g_cm3}  {self.bottom_attenuation_db_per_wavelength}  /",
                "1",
                f"{source_depth_m:.4f} /",
                "1",
                f"{receiver_depth_m:.4f} /",
                "1",
                f"{range_km:.6f} /",
                "'AG'",  # ASCII arrivals, geometric (hat) beams in Cartesian coordinates
                "0",  # number of beams: let Bellhop choose
                f"{low_angle} {high_angle} /",
                f"0.0 {water_depth_m * 1.05:.2f} {range_km * 1.05:.6f}",
            ]
        )
        stem.with_suffix(".env").write_text(env)

    @staticmethod
    def _write_bty(stem: Path, range_km: FloatArray, depth_m: FloatArray) -> None:
        """Write a piecewise-linear range-dependent bathymetry file."""
        lines = ["'L'", str(len(range_km))]
        lines += [f"{r:.6f} {d:.2f}" for r, d in zip(range_km, depth_m, strict=True)]
        stem.with_suffix(".bty").write_text("\n".join(lines))

    @staticmethod
    def _run_bellhopcxx(exe: str, stem: Path) -> None:
        """Run ``bellhopcxx`` on ``stem``, raising with its output if it fails."""
        try:
            subprocess.run([exe, str(stem)], capture_output=True, check=True, text=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"bellhopcxx failed (exit {e.returncode}).\n"
                f"env file: {stem}.env\nstdout: {e.stdout}\nstderr: {e.stderr}"
            ) from e

    @staticmethod
    def _parse_arr(arr_path: Path) -> list[Eigenray]:
        """Parse a 2D ASCII arrivals file into eigenrays.

        The file holds a header (``'2D'``, the frequency, the source depths, receiver depths
        and ranges, each count first, and a maximum-arrivals hint), then for every source,
        receiver and range a count of arrivals followed by one line per arrival::

            amplitude  phase_deg  delay_s  delay_imag_s  src_angle  rcv_angle  n_top  n_bottom
        """
        eigenrays: list[Eigenray] = []
        with open(arr_path) as f:
            f.readline()  # '2D'
            f.readline()  # frequency
            num_sources = int(f.readline().split()[0])
            num_receiver_depths = int(f.readline().split()[0])
            num_ranges = int(f.readline().split()[0])
            f.readline()  # maximum number of arrivals, a pre-allocation hint
            for _ in range(num_sources * num_receiver_depths * num_ranges):
                for _ in range(int(f.readline())):
                    parts = f.readline().split()
                    eigenrays.append(
                        Eigenray(
                            amplitude=float(parts[0]),
                            phase_rad=float(np.radians(float(parts[1]))),
                            delay_s=float(parts[2]),
                            src_angle_deg=float(parts[4]),
                            rcv_angle_deg=float(parts[5]),
                            n_surface_bounces=int(parts[6]),
                            n_bottom_bounces=int(parts[7]),
                        )
                    )
        return eigenrays
