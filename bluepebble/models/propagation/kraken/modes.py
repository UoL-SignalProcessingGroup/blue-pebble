"""Normal modes on a frequency grid, and the mode sum that turns them into pressure.

The mode sum reproduces FIELD's point-source evaluation (``KrakenField/EvaluateMod.f90``)::

    p(r, z) = i sqrt(2 pi) e^{i pi/4} / sqrt(r) sum_m phi_m(z_s) phi_m(z) e^{-i k_m r} / sqrt(k_m)

Pressure is relative to the free-field pressure 1 m from the source, so spherical
spreading gives ``|p| = 1/r``. The time dependence is ``e^{+i omega t}``, the same as Blue
Pebble's delay convention ``tau -> exp(-2j pi f tau)``, so the result is used as a transfer
function directly. Mode shapes are interpolated linearly in depth and continued into an
acoustic halfspace with FIELD's exponential tail (``KrakenField/ReadModes.f90``).
"""

from __future__ import annotations

import hashlib
import math
import warnings
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from numba import njit, prange
from numpy.typing import ArrayLike

from .env import KrakenEnvironment, environment_text, write_text
from .files import ModeFile, read_mode_file
from .runner import AcousticsToolboxBinaries, run_program

#: FIELD's point-source constant, i sqrt(2 pi) e^{i pi / 4}.
PREFACTOR = 1j * math.sqrt(2.0 * math.pi) * complex(math.cos(math.pi / 4), math.sin(math.pi / 4))


def _pekeris_root(value: np.ndarray) -> np.ndarray:
    """Branch of the square root used by the Acoustics Toolbox (``misc/PekRoot.f90``)."""
    value = np.asarray(value, dtype=np.complex128)
    return np.where(value.real >= 0.0, np.sqrt(value), 1j * np.sqrt(-value))


@dataclass
class ModeSet:
    """Normal modes of one range-independent waveguide over a frequency grid.

    Frequencies with fewer modes are padded: ``phi`` is zero there, so padded modes add
    nothing to a sum, and ``k`` is one so nothing divides by zero.

    Attributes
    ----------
    frequencies_hz : numpy.ndarray
        Shape ``(F,)``.
    k : numpy.ndarray
        Horizontal wavenumbers, shape ``(F, M)`` complex128, ``Im k <= 0``.
    num_modes : numpy.ndarray
        Number of valid modes at each frequency, shape ``(F,)`` int.
    phi : numpy.ndarray
        Mode shapes at ``depths_m``, shape ``(Nz, F, M)`` complex64.
    depths_m : numpy.ndarray
        Tabulation depths, increasing, shape ``(Nz,)``.
    top_depth_m, bottom_depth_m : float
        Depth of the surface and of the top of the bottom halfspace.
    bottom_boundary : str
        Bottom boundary letter; ``'A'`` gives an exponential tail below ``bottom_depth_m``.
    bottom_cp : numpy.ndarray
        Complex halfspace compressional speed at each frequency, shape ``(F,)``.
    is_krakenc : bool
        Whether the modes came from KRAKENC, which uses complex ``k`` in the tail.

    """

    frequencies_hz: np.ndarray
    k: np.ndarray
    num_modes: np.ndarray
    phi: np.ndarray
    depths_m: np.ndarray
    top_depth_m: float = 0.0
    bottom_depth_m: float = np.inf
    bottom_boundary: str = "R"
    bottom_cp: np.ndarray | None = None
    is_krakenc: bool = True

    def __post_init__(self) -> None:
        """Coerce arrays to the layout the mode-sum kernel needs."""
        self.frequencies_hz = np.ascontiguousarray(self.frequencies_hz, dtype=np.float64)
        self.k = np.ascontiguousarray(self.k, dtype=np.complex128)
        self.num_modes = np.ascontiguousarray(self.num_modes, dtype=np.int64)
        self.phi = np.ascontiguousarray(self.phi, dtype=np.complex64)
        self.depths_m = np.ascontiguousarray(self.depths_m, dtype=np.float64)
        num_freq, max_modes = self.k.shape
        if self.frequencies_hz.shape != (num_freq,) or self.num_modes.shape != (num_freq,):
            raise ValueError("frequencies_hz and num_modes must have one entry per row of k")
        if self.phi.shape != (self.depths_m.size, num_freq, max_modes):
            raise ValueError(
                f"phi has shape {self.phi.shape}, expected "
                f"{(self.depths_m.size, num_freq, max_modes)}"
            )
        if np.any(np.diff(self.depths_m) <= 0):
            raise ValueError("depths_m must be strictly increasing")
        if np.any(self.num_modes > max_modes) or np.any(self.num_modes < 0):
            raise ValueError("num_modes is out of range")
        if self.bottom_cp is not None:
            self.bottom_cp = np.asarray(self.bottom_cp, dtype=np.complex128).reshape(num_freq)
        # Padded entries must not contribute or divide by zero.
        valid = self.mode_mask
        self.k = np.where(valid, self.k, 1.0 + 0j)
        self.phi = np.where(valid[np.newaxis], self.phi, 0).astype(np.complex64)
        self._clear_caches()

    #: How many source depths and sensor layouts :meth:`transfer` remembers.
    _CACHE_SIZE = 8

    def _clear_caches(self) -> None:
        """Forget the per-call quantities that depend only on the modes and depths."""
        self._sqrt_k: np.ndarray | None = None
        self._gamma: np.ndarray | None = None
        self._excitations: dict[float, np.ndarray] = {}
        self._weights: dict[bytes, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def _excitation(self, source_depth_m: float) -> np.ndarray:
        """``phi_m(z_s) / sqrt(k_m)`` on the whole grid, ``(F, M)``, cached by depth.

        A moving source usually keeps its depth, so successive knots share this.
        """
        key = float(source_depth_m)
        cached = self._excitations.get(key)
        if cached is None:
            if self._sqrt_k is None:
                self._sqrt_k = np.sqrt(self.k)
            cached = self.mode_values(np.array([key]))[0] / self._sqrt_k
            if len(self._excitations) >= self._CACHE_SIZE:
                self._excitations.pop(next(iter(self._excitations)))
            self._excitations[key] = cached
        return cached

    def _cached_depth_weights(
        self, depths_m: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """:meth:`depth_weights` for exact depths seen before; an array keeps its layout."""
        key = depths_m.tobytes()
        cached = self._weights.get(key)
        if cached is None:
            cached = self.depth_weights(depths_m)
            if len(self._weights) >= self._CACHE_SIZE:
                self._weights.pop(next(iter(self._weights)))
            self._weights[key] = cached
        return cached

    @property
    def max_modes(self) -> int:
        """Width of the padded mode axis."""
        return int(self.k.shape[1])

    @property
    def mode_mask(self) -> np.ndarray:
        """``(F, M)`` boolean array of valid modes."""
        return np.arange(self.k.shape[1])[np.newaxis, :] < self.num_modes[:, np.newaxis]

    @property
    def nbytes(self) -> int:
        """Memory held by the mode arrays, in bytes."""
        return int(self.phi.nbytes + self.k.nbytes)

    # ----------------------------------------------------------------------------------
    # Depth interpolation
    # ----------------------------------------------------------------------------------
    def depth_weights(self, depths_m: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Interpolation indices, weights and halfspace distances for some depths.

        Inside the table, mode shapes are linear between the bracketing tabulated depths,
        as FIELD's ``Weight`` routine does. Below the top of an acoustic halfspace the last
        tabulated value is continued with an exponential tail.

        Parameters
        ----------
        depths_m : numpy.ndarray
            Depths in metres; the sign is ignored.

        Returns
        -------
        index : numpy.ndarray
            Lower bracketing table index, int64.
        weight : numpy.ndarray
            Weight of the upper bracketing value.
        tail_m : numpy.ndarray
            Distance below the top of the halfspace; 0 outside it.

        Raises
        ------
        ValueError
            If a depth lies outside the table but above the halfspace.

        """
        depths = np.abs(np.asarray(depths_m, dtype=np.float64)).reshape(-1)
        table = self.depths_m
        tail = np.zeros_like(depths)
        below = depths > self.bottom_depth_m
        if np.any(below):
            if self.bottom_boundary != "A":
                raise ValueError(
                    f"Depth {depths[below].max():.2f} m is below the bottom at "
                    f"{self.bottom_depth_m:.2f} m"
                )
            tail[below] = depths[below] - self.bottom_depth_m
        # One tabulated depth's worth of slack, as FIELD warns rather than fails.
        span = float(table[-1] - table[0]) if table.size > 1 else 0.0
        tolerance = 1e-6 * max(span, 1.0)
        inside = ~below
        outside = inside & ((depths < table[0] - tolerance) | (depths > table[-1] + tolerance))
        if np.any(outside):
            bad = depths[outside]
            raise ValueError(
                f"Depths {bad.min():.3f}-{bad.max():.3f} m fall outside the mode table "
                f"{table[0]:.3f}-{table[-1]:.3f} m; recompute the modes over a wider span"
            )
        if table.size == 1:
            return np.zeros(depths.size, np.int64), np.zeros(depths.size), tail
        query = np.where(below, table[-1], depths)
        index = np.clip(np.searchsorted(table, query, side="left") - 1, 0, table.size - 2)
        weight = (query - table[index]) / (table[index + 1] - table[index])
        return index.astype(np.int64), weight.astype(np.float64), tail

    def bottom_gamma(self) -> np.ndarray:
        """Vertical decay rate of each mode in the bottom halfspace, shape ``(F, M)``."""
        if self.bottom_boundary != "A" or self.bottom_cp is None:
            return np.zeros(self.k.shape, dtype=np.complex128)
        omega = 2.0 * np.pi * self.frequencies_hz
        k_bottom2 = (omega / self.bottom_cp) ** 2
        k = self.k if self.is_krakenc else self.k.real.astype(np.complex128)
        return np.ascontiguousarray(_pekeris_root(k**2 - k_bottom2[:, np.newaxis]))

    def mode_values(self, depths_m: np.ndarray) -> np.ndarray:
        """Mode shapes at arbitrary depths, shape ``(n, F, M)``.

        Parameters
        ----------
        depths_m : numpy.ndarray
            Depths in metres.

        Returns
        -------
        numpy.ndarray
            Interpolated mode shapes, complex128.

        """
        index, weight, tail = self.depth_weights(depths_m)
        upper = np.minimum(index + 1, self.depths_m.size - 1)
        lower_values = self.phi[index].astype(np.complex128)
        upper_values = self.phi[upper].astype(np.complex128)
        values = lower_values + weight[:, None, None] * (upper_values - lower_values)
        if np.any(tail > 0):
            gamma = self.bottom_gamma()
            values = values * np.exp(-gamma[np.newaxis] * tail[:, None, None])
        return values

    # ----------------------------------------------------------------------------------
    # Mode sum
    # ----------------------------------------------------------------------------------
    def transfer(
        self,
        source_depth_m: float,
        receiver_depths_m: np.ndarray,
        ranges_m: np.ndarray,
        frequency_slice: slice | None = None,
    ) -> np.ndarray:
        """Pressure at receiver-range pairs, shape ``(n, F)`` complex128.

        Each receiver ``i`` is at depth ``receiver_depths_m[i]`` and horizontal range
        ``ranges_m[i]``, which is the layout of a sensor array.

        Parameters
        ----------
        source_depth_m : float
            Source depth in metres.
        receiver_depths_m, ranges_m : numpy.ndarray
            Receiver depths and horizontal ranges in metres, the same length. Ranges must
            be positive.
        frequency_slice : slice, optional
            Restrict to a contiguous part of the frequency grid.

        Returns
        -------
        numpy.ndarray
            Complex pressure re 1 m.

        """
        depths = np.abs(np.asarray(receiver_depths_m, dtype=np.float64)).reshape(-1)
        ranges = np.asarray(ranges_m, dtype=np.float64).reshape(-1)
        if depths.shape != ranges.shape:
            raise ValueError("receiver_depths_m and ranges_m must be the same length")
        if np.any(ranges <= 0):
            raise ValueError("Ranges must be positive")
        sl = frequency_slice if frequency_slice is not None else slice(None)
        excitation = self._excitation(abs(float(source_depth_m)))[sl]
        index, weight, tail = self._cached_depth_weights(depths)
        if np.any(tail > 0):
            if self._gamma is None:
                self._gamma = self.bottom_gamma()
            gamma = self._gamma[sl]
        else:
            gamma = np.zeros((1, 1), np.complex128)
        k = np.ascontiguousarray(self.k[sl])
        out = np.empty((depths.size, k.shape[0]), dtype=np.complex128)
        with warnings.catch_warnings():
            # numba probes for TBB before falling back to OpenMP and warns if it is old.
            warnings.filterwarnings("ignore", message=".*TBB.*")
            _mode_sum(
                np.ascontiguousarray(self.phi[:, sl, :]),
                k,
                np.ascontiguousarray(self.num_modes[sl]),
                np.ascontiguousarray(excitation),
                index,
                weight,
                tail,
                np.ascontiguousarray(gamma),
                ranges,
                out,
            )
        return out

    def pressure_grid(
        self, source_depth_m: float, receiver_depths_m: ArrayLike, ranges_m: ArrayLike
    ) -> np.ndarray:
        """Pressure on a depth-by-range grid, shape ``(F, Nz, Nr)``, as FIELD computes it.

        Parameters
        ----------
        source_depth_m : float
            Source depth in metres.
        receiver_depths_m, ranges_m : numpy.ndarray
            Grid axes in metres.

        Returns
        -------
        numpy.ndarray
            Complex pressure re 1 m.

        """
        depths = np.asarray(receiver_depths_m, dtype=float).reshape(-1)
        ranges = np.asarray(ranges_m, dtype=float).reshape(-1)
        zz, rr = np.meshgrid(depths, ranges, indexing="ij")
        values = self.transfer(source_depth_m, zz.ravel(), rr.ravel())
        return values.T.reshape(-1, depths.size, ranges.size)

    # ----------------------------------------------------------------------------------
    # Dispersion
    # ----------------------------------------------------------------------------------
    def group_speeds(self) -> np.ndarray:
        """Group speed of each mode, ``d omega / d Re k``, shape ``(F, M)``.

        Modes are tracked between neighbouring frequencies rather than matched by index,
        so a mode the solver found at one frequency but not the next cannot shift the
        others: each mode is paired with the mode nearest its wavenumber scaled to the
        next frequency, and only if that is well inside the spacing of its neighbours.
        The slope is the mean of the forward and backward differences, a central
        difference where both exist. Unpaired modes are NaN. With a single frequency the
        phase speed is returned instead.
        """
        k = np.where(self.mode_mask, self.k.real, np.nan)
        omega = 2.0 * np.pi * self.frequencies_hz
        if omega.size < 2:
            return omega[:, np.newaxis] / k
        forward = np.full(k.shape, np.nan)
        backward = np.full(k.shape, np.nan)
        for i in range(omega.size - 1):
            pairs = self._pair(i, i + 1)
            modes = np.nonzero(pairs >= 0)[0]
            slope = (k[i + 1, pairs[modes]] - k[i, modes]) / (omega[i + 1] - omega[i])
            forward[i, modes] = slope
            backward[i + 1, pairs[modes]] = slope
        with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # unpaired modes: all-NaN means
            slope = np.nanmean(np.stack((forward, backward)), axis=0)
            speeds = 1.0 / slope
        speeds[~np.isfinite(speeds) | (speeds <= 0)] = np.nan
        return speeds

    def _pair(self, i: int, j: int) -> np.ndarray:
        """Pair modes at frequency ``i`` with modes at ``j``; -1 where none is close enough."""
        omega = 2.0 * np.pi * self.frequencies_hz
        here = self.k[i, : self.num_modes[i]].real
        there = self.k[j, : self.num_modes[j]].real
        pairs = np.full(here.size, -1, dtype=np.int64)
        if here.size == 0 or there.size == 0:
            return pairs
        predicted = here * omega[j] / omega[i]
        nearest = np.argmin(np.abs(there[np.newaxis, :] - predicted[:, np.newaxis]), axis=1)
        miss = np.abs(there[nearest] - predicted)
        if here.size > 1:
            gaps = np.abs(here[:, np.newaxis] - here[np.newaxis, :])
            np.fill_diagonal(gaps, np.inf)
            spacing = gaps.min(axis=1)
        else:
            spacing = np.full(1, np.inf)
        counts = np.bincount(nearest, minlength=there.size)
        paired = (miss < 0.25 * spacing) & (counts[nearest] == 1)
        pairs[paired] = nearest[paired]
        return pairs

    def mode_count_drops(self) -> np.ndarray:
        """Frequencies where fewer modes were found than at both neighbours.

        A solver that misses a mode at one frequency leaves a one-bin glitch in the
        transfer function. Modes appear one at a time as frequency rises, so a dip below
        both neighbours means a mode was missed (or flickered at cutoff).

        Returns
        -------
        numpy.ndarray
            Indices into ``frequencies_hz``.

        """
        n = self.num_modes
        if n.size < 3:
            return np.zeros(0, dtype=int)
        interior = (n[1:-1] < n[:-2]) & (n[1:-1] < n[2:])
        return np.nonzero(interior)[0] + 1

    def significant_modes(
        self, source_depth_m: float, range_m: float, threshold_db: float = 40.0
    ) -> np.ndarray:
        """Modes within ``threshold_db`` of the strongest at a range, shape ``(F, M)`` bool.

        A mode's strength is its excitation at the source depth, attenuated over
        ``range_m``: ``|phi_m(z_s)| exp(Im k_m r) / sqrt(|k_m|)``.

        Parameters
        ----------
        source_depth_m : float
            Source depth in metres.
        range_m : float
            Range in metres.
        threshold_db : float, optional
            How far below the strongest mode at each frequency to keep.

        Returns
        -------
        numpy.ndarray
            Mask of modes that matter.

        """
        excitation = np.abs(self.mode_values(np.array([source_depth_m]))[0])
        strength = excitation * np.exp(self.k.imag * range_m) / np.sqrt(np.abs(self.k))
        strength = np.where(self.mode_mask, strength, 0.0)
        peak = np.max(strength, axis=1, keepdims=True)
        with np.errstate(divide="ignore"):
            return (
                self.mode_mask & (strength >= peak * 10.0 ** (-threshold_db / 20.0)) & (peak > 0)
            )

    # ----------------------------------------------------------------------------------
    # Persistence
    # ----------------------------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        """Save to a compressed ``.npz`` file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            frequencies_hz=self.frequencies_hz,
            k=self.k,
            num_modes=self.num_modes,
            phi=self.phi,
            depths_m=self.depths_m,
            top_depth_m=self.top_depth_m,
            bottom_depth_m=self.bottom_depth_m,
            bottom_boundary=self.bottom_boundary,
            bottom_cp=self.bottom_cp if self.bottom_cp is not None else np.array([]),
            is_krakenc=self.is_krakenc,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> ModeSet:
        """Load a mode set written by :meth:`save`."""
        with np.load(Path(path), allow_pickle=False) as data:
            bottom_cp = data["bottom_cp"]
            return cls(
                frequencies_hz=data["frequencies_hz"],
                k=data["k"],
                num_modes=data["num_modes"],
                phi=data["phi"],
                depths_m=data["depths_m"],
                top_depth_m=float(data["top_depth_m"]),
                bottom_depth_m=float(data["bottom_depth_m"]),
                bottom_boundary=str(data["bottom_boundary"]),
                bottom_cp=bottom_cp if bottom_cp.size else None,
                is_krakenc=bool(data["is_krakenc"]),
            )

    @classmethod
    def from_mode_files(cls, files: Sequence[ModeFile]) -> ModeSet:
        """Combine mode files that share depths, e.g. chunks of one frequency grid.

        Parameters
        ----------
        files : sequence of ModeFile
            Files in frequency order.

        Returns
        -------
        ModeSet
            Every frequency of every file.

        """
        sets = [s for f in files for s in f.sets]
        depths = files[0].depths_m
        for f in files[1:]:
            if not np.array_equal(f.depths_m, depths):
                raise ValueError("Mode files tabulate different depths")
        max_modes = max(1, max(s.k.size for s in sets))
        num_freq = len(sets)
        k = np.ones((num_freq, max_modes), dtype=np.complex128)
        phi = np.zeros((depths.size, num_freq, max_modes), dtype=np.complex64)
        for i, s in enumerate(sets):
            k[i, : s.k.size] = s.k
            phi[:, i, : s.k.size] = s.phi.T
        bottom = sets[0].bottom
        return cls(
            frequencies_hz=np.array([s.frequency_hz for s in sets]),
            k=k,
            num_modes=np.array([s.k.size for s in sets]),
            phi=phi,
            depths_m=depths,
            top_depth_m=sets[0].top.depth_m,
            bottom_depth_m=bottom.depth_m,
            bottom_boundary=bottom.boundary,
            bottom_cp=np.array([s.bottom.cp for s in sets]),
            is_krakenc=files[0].is_krakenc,
        )


@njit(parallel=True, cache=True, fastmath=False)
def _mode_sum(phi, k, num_modes, excitation, index, weight, tail, gamma, ranges, out):  # noqa: D103
    # out[i, f] = PREFACTOR / sqrt(r_i) * sum_m excitation[f, m] phi_m(z_i) e^{-i k_m r_i}
    # with excitation = phi_m(z_s) / sqrt(k_m). The exponential is evaluated in real
    # arithmetic: e^{-i k r} = e^{Im(k) r} (cos(Re(k) r) - i sin(Re(k) r)).
    num_receivers = index.size
    num_freq = k.shape[0]
    last = phi.shape[0] - 1
    root_i = complex(math.cos(math.pi / 4), math.sin(math.pi / 4))
    prefactor = 1j * math.sqrt(2.0 * math.pi) * root_i
    for i in prange(num_receivers):
        lower = index[i]
        upper = min(lower + 1, last)
        w = weight[i]
        depth_below = tail[i]
        r = ranges[i]
        scale = prefactor / math.sqrt(r)
        for f in range(num_freq):
            total_re = 0.0
            total_im = 0.0
            for m in range(num_modes[f]):
                a = phi[lower, f, m]
                b = phi[upper, f, m]
                v_re = a.real + w * (b.real - a.real)
                v_im = a.imag + w * (b.imag - a.imag)
                if depth_below > 0.0:
                    t = np.exp(-gamma[f, m] * depth_below)
                    v_re, v_im = v_re * t.real - v_im * t.imag, v_re * t.imag + v_im * t.real
                e = excitation[f, m]
                c_re = e.real * v_re - e.imag * v_im
                c_im = e.real * v_im + e.imag * v_re
                km = k[f, m]
                decay = math.exp(km.imag * r)
                phase = km.real * r
                h_re = decay * math.cos(phase)
                h_im = -decay * math.sin(phase)
                total_re += c_re * h_re - c_im * h_im
                total_im += c_re * h_im + c_im * h_re
            out[i, f] = scale * (total_re + 1j * total_im)


# --------------------------------------------------------------------------------------
def environment_cache_key(text_parts: Sequence[str], program: str) -> str:
    """Hash of everything that determines a mode set."""
    digest = hashlib.sha256(program.encode())
    for part in text_parts:
        digest.update(part.encode())
    return digest.hexdigest()[:24]


def compute_mode_set(
    environment: KrakenEnvironment,
    frequencies_hz: ArrayLike,
    depths_m: ArrayLike,
    program: str = "krakenc",
    binaries: AcousticsToolboxBinaries | None = None,
    c_low_m_s: float = 0.0,
    c_high_m_s: float | None = None,
    rmax_km: float = 0.0,
    num_workers: int | None = None,
    frequencies_per_run: int = 16,
    cache_dir: str | Path | None = None,
    work_dir: str | Path | None = None,
    mesh_points_per_wavelength: float | None = None,
    richardson: bool | None = None,
) -> ModeSet:
    """Run KRAKEN or KRAKENC over a frequency grid and collect the modes.

    The grid is split into chunks of ``frequencies_per_run``; each chunk is one broadband
    run in its own directory, and chunks run in parallel.

    Parameters
    ----------
    environment : KrakenEnvironment
        The waveguide.
    frequencies_hz : sequence of float
        Frequency grid in Hz.
    depths_m : sequence of float
        Depths to tabulate the modes at, in metres.
    program : {"krakenc", "kraken"}, optional
        Which solver to run. KRAKENC handles attenuation exactly and finds leaky modes.
    binaries : AcousticsToolboxBinaries, optional
        Located programs; found automatically if omitted.
    c_low_m_s, c_high_m_s : float, optional
        Phase-speed window; see :func:`~.env.environment_text`.
    rmax_km : float, optional
        Furthest range of interest, in km, which sets KRAKEN's mesh convergence test.
    num_workers : int, optional
        Parallel runs; defaults to the number of CPUs.
    frequencies_per_run : int, optional
        Frequencies per broadband run.
    cache_dir : path-like, optional
        Directory to cache mode sets in, keyed on a hash of their inputs.
    work_dir : path-like, optional
        Directory to run in; a temporary one if omitted. Kept for inspection if given.
    mesh_points_per_wavelength : float, optional
        Finite-difference mesh density. Defaults to 80 for KRAKENC, which then solves on
        that single mesh, and to KRAKEN's own choice of 20 for KRAKEN, which refines it.
    richardson : bool, optional
        Whether to let the solver refine its mesh and Richardson-extrapolate the
        wavenumbers until the phase error at ``rmax_km`` is small. Defaults to on for
        KRAKEN and off for KRAKENC. KRAKENC pairs modes by index between meshes, and when
        its root finder finds different modes on different meshes the extrapolation
        mixes them, leaving wrong mode sets at scattered frequencies and one-bin glitches
        in the transfer function. A single fine mesh avoids that: the error falls as the
        square of the spacing, to about 0.3 % of the field at 80 points per wavelength.
        KRAKEN counts its modes with Sturm sequences, so its extrapolation is safe and
        converges to about 0.1 %.

    Notes
    -----
    KRAKEN treats attenuation as a perturbation, KRAKENC exactly. Over a lossy
    sediment the two can differ by several percent in the field, more at short range
    where steep, lossy modes matter, which is why KRAKENC is the default.

    Returns
    -------
    ModeSet
        The modes on the grid, tabulated at ``depths_m``.

    """
    if program not in {"krakenc", "kraken"}:
        raise ValueError(f"program must be 'krakenc' or 'kraken', got {program!r}")
    frequencies = np.asarray(frequencies_hz, dtype=float).ravel()
    depths = np.unique(np.abs(np.asarray(depths_m, dtype=float).ravel()))
    if richardson is None:
        richardson = program == "kraken"
    if mesh_points_per_wavelength is None and program == "krakenc":
        mesh_points_per_wavelength = 80.0
    # KRAKEN stops refining once the phase error at RMax is small; RMax = 0 stops at once.
    rmax_km = float(rmax_km) if richardson else 0.0
    chunk = max(1, int(frequencies_per_run))
    chunks = [frequencies[i : i + chunk] for i in range(0, frequencies.size, chunk)]
    texts = [
        environment_text(
            environment,
            part,
            source_depths_m=depths[:1],
            receiver_depths_m=depths,
            c_low_m_s=c_low_m_s,
            c_high_m_s=c_high_m_s,
            rmax_km=rmax_km,
            nominal_frequency_hz=float(frequencies.max()),
            mesh_points_per_wavelength=mesh_points_per_wavelength,
        )
        for part in chunks
    ]

    cache_path = None
    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"modes_{environment_cache_key(texts, program)}.npz"
        if cache_path.is_file():
            return ModeSet.load(cache_path)

    binaries = binaries or AcousticsToolboxBinaries.find()
    executable = binaries.require(program)

    def run(work_root: Path) -> ModeSet:
        def one(i: int) -> ModeFile:
            directory = work_root / f"chunk_{i:04d}"
            write_text(directory / "modes.env", texts[i])
            return read_mode_file(run_program(executable, directory, "modes", ".mod"))

        workers = max(1, min(num_workers or (len(texts) and _cpu_count()), len(texts)))
        with ThreadPoolExecutor(workers) as pool:
            files = list(pool.map(one, range(len(texts))))
        return ModeSet.from_mode_files(files)

    if work_dir is not None:
        modes = run(Path(work_dir))
    else:
        with TemporaryDirectory(prefix="bluepebble_kraken_") as tmp:
            modes = run(Path(tmp))
    drops = modes.mode_count_drops()
    if drops.size > max(1, 0.01 * frequencies.size):
        warnings.warn(
            f"{program} found fewer modes at {drops.size} frequencies than at both of their "
            f"neighbours, e.g. {modes.frequencies_hz[drops[0]]:.2f} Hz; modes may be missing. "
            "Check the solver's print file, or try program='kraken'",
            stacklevel=2,
        )
    if cache_path is not None:
        modes.save(cache_path)
    return modes


def _cpu_count() -> int:
    import os

    return os.cpu_count() or 1
