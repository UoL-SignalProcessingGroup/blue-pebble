"""Shared types, STFT helpers, and base classes for beamforming algorithms."""

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal, TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from stonesoup.base import Base, Property

ComplexArray: TypeAlias = NDArray[np.complex128]
FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.intp]
BoolArray: TypeAlias = NDArray[np.bool_]
BeamformerOutput: TypeAlias = ComplexArray | FloatArray
DomainType: TypeAlias = Literal["time", "frequency", "broadband_power"]


@dataclass(frozen=True)
class MirrorPlan:
    """Bookkeeping to expand a half-plane beamformer output to the full steering grid.

    Produced by :meth:`SteeringCalculator.mirror_plan` from the array's instantaneous axis,
    and consumed by :meth:`Beamformer.expand_mirrored`. See
    ``SteeringCalculator.mirror_half_plane`` for the symmetry this exploits.
    """

    primary_mask: BoolArray
    """``(num_beams,)``. ``True`` where this beam was steered directly (the "primary" half)."""

    mirror_idx: IntArray
    """``(num_beams,)``. Index of each beam's mirror partner across the array axis; an
    involution, so ``mirror_idx[mirror_idx[k]] == k`` for every ``k``."""

    roll_shift: int
    """Bins to roll the axis-centred reconstruction back onto the original steering grid."""


def _stft(x: ArrayLike, nfft: int, overlap: int) -> ComplexArray:
    """Compute a per-sensor STFT using a Hann window and fixed overlap.

    Parameters
    ----------
    x : ArrayLike
        Sensor data with shape ``(num_sensors, num_samples)``.
    nfft : int
        STFT window length (FFT size).
    overlap : int
        Overlap between adjacent windows in samples. Values greater than or equal to
        ``nfft`` fall back to a hop of one sample.

    Returns
    -------
    ComplexArray
        STFT tensor with shape ``(num_sensors, num_frames, nfft)``.

    Raises
    ------
    ValueError
        If ``num_samples < nfft``.

    """
    x_array = np.asarray(x)
    M, T = x_array.shape
    if T < nfft:
        raise ValueError(f"Input signal length T={T} is less than window size nfft={nfft}.")
    hop = max(1, nfft - overlap)
    # pad to fit last frame exactly
    n_frames = 1 + (max(0, T - nfft) // hop)
    pad = (n_frames - 1) * hop + nfft - T
    if pad > 0:
        x_array = np.pad(x_array, ((0, 0), (0, pad)), mode="constant")

    window = np.hanning(nfft).astype(np.float64)
    # Make a 3D view: (M, n_frames, nfft)
    stride_t = x_array.strides[1]
    frames = np.lib.stride_tricks.as_strided(
        x_array,
        shape=(M, n_frames, nfft),
        strides=(x_array.strides[0], hop * stride_t, stride_t),
        writeable=False,
    )
    frames = frames * window  # broadcasts over last axis
    return np.fft.fft(frames, axis=2)


def _stft_bin_frequencies(nfft: int, fs: float, f0: float = 0.0) -> FloatArray:
    """Map STFT bin indices to analog frequencies.

    Bin indices are mapped to centered FFT frequencies (``[0 ... nfft/2-1, -nfft/2 ... -1]``)
    and offset by the carrier frequency, so the result is valid for baseband data.

    Parameters
    ----------
    nfft : int
        Number of FFT points.
    fs : float
        Sampling frequency in Hz.
    f0 : float, optional
        Carrier frequency offset in Hz for baseband data. Defaults to ``0.0``.

    Returns
    -------
    FloatArray
        Frequency of each STFT bin in Hz, with shape ``(nfft,)``.

    """
    k = np.arange(nfft)
    k_centered = np.where(k <= nfft // 2, k, k - nfft)
    return f0 + (fs / nfft) * k_centered


def _active_bin_indices(
    f_bins: FloatArray,
    fmin: float | None = None,
    fmax: float | None = None,
) -> IntArray:
    """Select the STFT bins lying within a frequency band.

    Parameters
    ----------
    f_bins : FloatArray
        Bin frequencies in Hz, as returned by :func:`_stft_bin_frequencies`.
    fmin : float | None, optional
        Lower band edge in Hz, inclusive. If ``None``, the minimum bin frequency is used.
    fmax : float | None, optional
        Upper band edge in Hz, inclusive. If ``None``, the maximum bin frequency is used.

    Returns
    -------
    IntArray
        Indices of the bins inside the band. Empty when no bin falls within it.

    """
    low = float(f_bins.min()) if fmin is None else fmin
    high = float(f_bins.max()) if fmax is None else fmax
    active = (f_bins >= low) & (f_bins <= high)
    return np.nonzero(active)[0]


class Beamformer(Base, ABC):
    """Abstract interface for beamforming algorithms."""

    @abstractmethod
    def beamform(
        self,
        sensor_signals: ArrayLike,
        steering_delays_s: ArrayLike,
        mirror_plan: MirrorPlan | None = None,
    ) -> BeamformerOutput:
        """Form directional beams from multi-sensor input data.

        Parameters
        ----------
        sensor_signals : ArrayLike
            Sensor signal matrix with shape ``(num_sensors, num_samples)``.
        steering_delays_s : ArrayLike
            Steering-delay matrix (seconds) with shape ``(num_directions, num_sensors)``.
        mirror_plan : MirrorPlan | None, optional
            When provided (from ``SteeringCalculator.mirror_plan``), ``steering_delays_s``
            is expected to cover only the primary half of the steering grid -- as returned
            by a ``SteeringCalculator`` configured with ``mirror_half_plane=True`` and the
            output is expanded to the full grid via :meth:`expand_mirrored` before returning.

        Returns
        -------
        BeamformerOutput
            Beamformer output matrix. Concrete implementations define whether this
            contains complex beamformed time-series or real-valued beam power. When
            ``mirror_plan`` is given, the direction axis always covers the full steering
            grid, never just the primary half.

        """
        pass

    @staticmethod
    def expand_mirrored(
        output: BeamformerOutput,
        mirror_plan: MirrorPlan,
        direction_axis: int = 0,
    ) -> BeamformerOutput:
        """Expand a half-plane beamformer output to the full steering grid.

        The reconstruction is an exact index lookup, not an interpolation: each beam in
        the mirrored half is a copy of its exact mirror partner from the primary half (see
        ``SteeringCalculator.mirror_half_plane``), rolled back onto the original steering
        grid. It applies uniformly to any beamformer output shape, complex time/frequency
        signals or real-valued power maps, single-band or multiband. Mirroring is a
        property of the steering delays themselves, not of what's done with them.

        Parameters
        ----------
        output : BeamformerOutput
            Beamformer output computed for only the primary-half directions, ordered to
            match ``steering_azimuths_rad[mirror_plan.primary_mask]``.
        mirror_plan : MirrorPlan
            Mirror bookkeeping from ``SteeringCalculator.mirror_plan``.
        direction_axis : int, optional
            Axis of ``output`` that indexes steering direction. Defaults to ``0``;
            multiband beamformers pass ``1`` for their
            ``(num_bands, num_directions, num_frames)`` output.

        Returns
        -------
        BeamformerOutput
            ``output`` expanded to all ``len(mirror_plan.mirror_idx)`` directions, on the
            original (non-rotated) steering grid.

        """
        output_array = np.asarray(output)
        num_beams = len(mirror_plan.mirror_idx)

        full_shape = list(output_array.shape)
        full_shape[direction_axis] = num_beams
        full_output = np.empty(full_shape, dtype=output_array.dtype)

        primary_mask = mirror_plan.primary_mask
        primary_slice: list[slice | BoolArray] = [slice(None)] * output_array.ndim
        primary_slice[direction_axis] = primary_mask
        full_output[tuple(primary_slice)] = output_array

        rank = np.full(num_beams, -1, dtype=np.int64)
        rank[primary_mask] = np.arange(output_array.shape[direction_axis])

        secondary_mask = ~primary_mask
        secondary_slice: list[slice | BoolArray] = [slice(None)] * output_array.ndim
        secondary_slice[direction_axis] = secondary_mask

        gather_slice: list[slice | IntArray] = [slice(None)] * output_array.ndim
        gather_slice[direction_axis] = rank[mirror_plan.mirror_idx[secondary_mask]]

        full_output[tuple(secondary_slice)] = output_array[tuple(gather_slice)]

        return cast(
            BeamformerOutput, np.roll(full_output, mirror_plan.roll_shift, axis=direction_axis)
        )


class FrequencyBand(Base):
    """A named frequency band for multiband beamforming.

    Bands are passed to an STFT-based beamformer via its ``bands`` property to integrate
    several sub-bands from a single pass over the data. Bands may overlap freely: a bin
    shared between bands is processed once and accumulated into each band that contains it.

    The ``label`` is the join key between a band and its downstream detector, so labels must
    be unique within one beamformer.
    """

    label: str = Property(doc="Human-readable band name, used to key detectors and results")
    fmin: float = Property(doc="Lower band edge, in Hz (inclusive)")
    fmax: float = Property(doc="Upper band edge, in Hz (inclusive)")

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialise the frequency band.

        Raises
        ------
        ValueError
            If ``fmax`` is not greater than ``fmin``.

        """
        super().__init__(*args, **kwargs)
        if self.fmax <= self.fmin:
            raise ValueError(
                f"Band {self.label!r} has fmax ({self.fmax}) <= fmin ({self.fmin}); "
                "bands must span a positive frequency range"
            )


class _STFTBeamformer(Beamformer):
    """Shared configuration and band bookkeeping for STFT-based beamformers.

    Concrete subclasses transform sensor data into short-time frequency bins, do per-bin
    work, and integrate the result over frequency. This base holds the STFT geometry and
    the band selection that governs which bins are integrated into which output map.
    """

    sampling_rate_hz: float = Property(
        doc="The sampling frequency of the sensor signals, in Hz",
    )
    nfft: int = Property(default=500, doc="STFT window size (samples)")
    overlap: int = Property(default=250, doc="STFT overlap (samples)")
    f0: float = Property(default=0.0, doc="Carrier frequency for baseband data (Hz)")
    fmin: float | None = Property(default=None, doc="Minimum frequency to integrate (Hz)")
    fmax: float | None = Property(default=None, doc="Maximum frequency to integrate (Hz)")
    bands: list[FrequencyBand] | None = Property(
        default=None,
        doc="Frequency bands to integrate separately, producing one power map per band. "
        "When None, the scalar 'fmin'/'fmax' single-band behaviour is used and the "
        "output is two-dimensional.",
    )
    normalise_by_bandwidth: bool = Property(
        default=False,
        doc="Divide each band's power by its number of active frequency bins, making "
        "bands of differing width directly comparable in level.",
    )
    parallelise: bool = Property(
        default=True,
        doc="Whether the numba-accelerated inner kernels use multiple threads. Numba's "
        "parallel=True is fixed at compile time, so each kernel affected by this flag is "
        "compiled once in a multi-threaded form and once in a single-threaded form, and "
        "this property selects between the two per instance -- no global numba or thread "
        "pool state is touched. Set to False when an outer parallel loop (for example a "
        "sensitivity-analysis harness distributing work across processes) already "
        "saturates the available cores, to avoid oversubscription from nested "
        "parallelism.",
    )

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialise the beamformer and validate any configured bands.

        Raises
        ------
        ValueError
            If ``bands`` is an empty list, or if two bands share a label.

        """
        super().__init__(*args, **kwargs)

        if self.bands is None:
            return

        if not self.bands:
            raise ValueError("bands must contain at least one FrequencyBand, or be None")

        labels = [band.label for band in self.bands]
        duplicates = sorted({label for label in labels if labels.count(label) > 1})
        if duplicates:
            raise ValueError(
                f"Band labels must be unique; got duplicates: {duplicates}. "
                "Labels are the join key to downstream detectors."
            )

    def _band_bin_indices(
        self,
        f_bins: FloatArray,
        fmin: float | None = None,
        fmax: float | None = None,
    ) -> list[IntArray]:
        """Select the active STFT bins for each configured band.

        Parameters
        ----------
        f_bins : FloatArray
            Bin frequencies in Hz, as returned by :func:`_stft_bin_frequencies`.
        fmin : float | None, optional
            Lower band edge used only in single-band mode (when ``bands`` is None).
        fmax : float | None, optional
            Upper band edge used only in single-band mode (when ``bands`` is None).

        Returns
        -------
        list of IntArray
            Active bin indices per band, ordered as ``bands``. In single-band mode this is
            a one-element list covering ``[fmin, fmax]``.

        """
        if self.bands is not None:
            return [_active_bin_indices(f_bins, band.fmin, band.fmax) for band in self.bands]
        return [_active_bin_indices(f_bins, fmin, fmax)]

    @staticmethod
    def _bins_to_bands(per_band_bins: list[IntArray]) -> dict[int, list[int]]:
        """Invert per-band bin lists into a bin-index to band-indices map.

        Iterating this map visits each bin in the union of all bands exactly once, which is
        what makes K overlapping bands cost one pass rather than K.

        Parameters
        ----------
        per_band_bins : list of IntArray
            Active bin indices per band.

        Returns
        -------
        dict
            Maps each active bin index to the indices of every band containing it.

        """
        bins_to_bands: dict[int, list[int]] = defaultdict(list)
        for band_idx, bin_indices in enumerate(per_band_bins):
            for bin_idx in bin_indices:
                bins_to_bands[int(bin_idx)].append(band_idx)
        return bins_to_bands

    def _finalise_band_power(
        self,
        power: FloatArray,
        per_band_bins: list[IntArray],
    ) -> FloatArray:
        """Apply bandwidth normalisation and collapse the band axis in single-band mode.

        Parameters
        ----------
        power : FloatArray
            Accumulated power with shape ``(num_bands, num_directions, num_frames)``.
        per_band_bins : list of IntArray
            Active bin indices per band, used for the bin counts.

        Returns
        -------
        FloatArray
            Shape ``(num_bands, num_directions, num_frames)`` when ``bands`` is set,
            otherwise ``(num_directions, num_frames)``.

        """
        if self.normalise_by_bandwidth:
            for band_idx, bin_indices in enumerate(per_band_bins):
                if len(bin_indices) > 0:
                    power[band_idx] /= len(bin_indices)

        if self.bands is None:
            return power[0]
        return power
