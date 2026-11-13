"""Base signal properties and methods for signal models."""

import warnings
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray
from stonesoup.base import Base, Property

from .utils import WindowType, compute_stft

if TYPE_CHECKING:
    from stonesoup.types.state import State

ComplexArray: TypeAlias = NDArray[np.complex128]
FloatArray: TypeAlias = NDArray[np.float64]
CachedStftResult: TypeAlias = tuple[NDArray[np.complex64], FloatArray, int, FloatArray]


def _get_source_metadata(source: "State") -> Mapping[str, object]:
    """Validate and return the metadata mapping from a source state.

    Parameters
    ----------
    source : State
        Source state exposing a ``metadata`` attribute.

    Returns
    -------
    Mapping[str, object]
        The validated metadata mapping.

    Raises
    ------
    ValueError
        If ``metadata`` is absent or not mapping-like.

    """
    metadata = getattr(source, "metadata", None)
    if not isinstance(metadata, Mapping):
        msg = "Source state metadata must be mapping-like"
        raise ValueError(msg)
    return metadata


class Signal(Base):
    """Shared sampling parameter contract for all signal and noise models.

    ``Signal`` is the public unified root of the signal hierarchy. Its branches are
    :class:`SourceSignal` (the sources the passive simulators propagate:
    :class:`~bluepebble.signal.anthropogenic.AnthropogenicSignal` and
    :class:`~bluepebble.signal.biological.BiologicalSignal`),
    :class:`~bluepebble.signal.random.RandomSignal` and
    :class:`~bluepebble.signal.active.ActiveSignal`.

    This class carries only the shared ``duration_s`` / ``sampling_rate_hz``
    parameter contract and the derived ``num_samples`` property.  Each branch
    defines its own generation interface.

    Parameters
    ----------
    duration_s : float
        Duration of the signal in seconds.
    sampling_rate_hz : int
        Sampling rate in Hertz.

    """

    duration_s: float = Property(doc="Duration of the signal in seconds")
    sampling_rate_hz: int = Property(doc="Sampling rate in Hertz")

    @property
    def num_samples(self) -> int:
        """Calculate the number of samples based on duration and sampling rate.

        Returns
        -------
        int
            The number of samples in the signal snapshot.

        """
        return int(self.duration_s * self.sampling_rate_hz)


class SourceSignal(Signal, ABC):
    """Base class for source signal models that the passive simulators propagate.

    A source model generates its full-duration source waveform once and caches it together
    with its STFT (Short-Time Fourier Transform). Subclasses implement
    :meth:`_generate_base_signal`.

    Lifecycle
    ---------
    The simulators fill the cache themselves, so pass them a model that has not been computed:

    - :class:`~bluepebble.simulator.ContinuousSTFTPassiveSonarArraySimulator` calls
      :meth:`compute_stft` once per target. That raises :exc:`RuntimeError` if the cache is
      already filled, so call :meth:`reset` before using the same instance in another run.
    - :class:`~bluepebble.simulator.ContinuousFractionalDelayPassiveSonarArraySimulator` and
      :class:`~bluepebble.simulator.DiscretePassiveSonarArraySimulator` call
      :meth:`get_source_waveform`, which fills the cache on first use and returns the cached
      waveform afterwards. A second run therefore repeats the first run's waveform unless
      :meth:`reset` is called in between.
    """

    frame_len: int = Property(default=1024, doc="STFT frame length in samples")
    hop_factor: int = Property(
        default=4,
        doc="Hop factor (hop = frame_len // hop_factor)",
    )
    window_type: WindowType = Property(default="hann", doc="Window type for STFT")

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialise shared STFT caches."""
        super().__init__(*args, **kwargs)
        self._stft_cache: NDArray[np.complex64] | None = None
        self._frequencies: FloatArray | None = None
        self._hop: int | None = None
        self._window: FloatArray | None = None
        self._source_signal: ComplexArray | None = None
        self._cached_source: "State | None" = None  # noqa: UP037

    @abstractmethod
    def _generate_base_signal(self, source: "State") -> ComplexArray:
        """Generate full-duration source waveform for STFT processing."""

    def _prepare_waveform(self, source: "State") -> ComplexArray:
        """Return the waveform that :meth:`compute_stft` caches.

        Subclasses override this to validate or reshape what :meth:`_generate_base_signal`
        returns.
        """
        return self._generate_base_signal(source)

    def compute_stft(self, source: "State") -> CachedStftResult:
        """Compute and cache STFT outputs for the source signal.

        This method may only be called once per simulation run.  If the cache
        is already populated, :exc:`RuntimeError` is raised — call
        :meth:`reset` first to clear it before recomputing.

        Parameters
        ----------
        source : State
            Source state used by concrete implementations to build the waveform.

        Returns
        -------
        CachedStftResult
            Cached STFT tuple ``(stft, frequencies_hz, hop_samples, window)``.

        Raises
        ------
        RuntimeError
            If :meth:`compute_stft` has already been called on this instance.
            Call :meth:`reset` to clear the cache before recomputing.

        """
        if self._stft_cache is not None:
            msg = (
                f"{type(self).__name__}.compute_stft() has already been called. "
                "Call reset() first to clear the cache before recomputing."
            )
            raise RuntimeError(msg)

        self._source_signal = self._prepare_waveform(source)

        stft, freq_normalized, hop, window = compute_stft(
            self._source_signal, self.frame_len, self.hop_factor, self.window_type
        )
        stft = np.asarray(stft, dtype=np.complex64)
        frequencies = cast(FloatArray, freq_normalized * self.sampling_rate_hz)
        window_float = np.asarray(window, dtype=np.float64)

        self._stft_cache = stft
        self._frequencies = frequencies
        self._hop = hop
        self._window = window_float
        self._cached_source = source

        return stft, frequencies, hop, window_float

    def get_stft(self) -> CachedStftResult:
        """Return cached STFT data.

        Returns
        -------
        CachedStftResult
            Cached STFT tuple ``(stft, frequencies_hz, hop_samples, window)``.

        Raises
        ------
        RuntimeError
            If :meth:`compute_stft` has not been called yet.

        """
        if self._stft_cache is None:
            msg = "STFT not computed yet. Call compute_stft() first."
            raise RuntimeError(msg)

        return (
            self._stft_cache,
            cast(FloatArray, self._frequencies),
            cast(int, self._hop),
            cast(FloatArray, self._window),
        )

    def get_source_signal(self) -> ComplexArray:
        """Return the cached full-duration source signal.

        Returns
        -------
        ComplexArray
            Cached source waveform.

        Raises
        ------
        RuntimeError
            If :meth:`compute_stft` has not been called yet.

        """
        if self._source_signal is None:
            msg = "Source signal not generated yet. Call compute_stft() first."
            raise RuntimeError(msg)

        return self._source_signal

    def get_source_waveform(self, source: "State") -> ComplexArray:
        """Return the cached source waveform, computing it on first call.

        If the cache is already populated but ``source`` is not the same object
        that was passed to :meth:`compute_stft`, a :exc:`UserWarning` is raised.
        This guards against stale-cache bugs in multi-source simulations where
        the same model instance is accidentally reused across different targets.
        Call :meth:`reset` then :meth:`compute_stft` to update the cache.

        Parameters
        ----------
        source : State
            Source state used to generate the waveform if not yet cached.

        Returns
        -------
        ComplexArray
            Full-duration source waveform.

        """
        if self._source_signal is None:
            self.compute_stft(source)
        elif source is not self._cached_source:
            warnings.warn(
                f"{type(self).__name__}.get_source_waveform() was called with a different "
                "source state than the one used to build the cached STFT. "
                "The cached result will be returned unchanged. "
                "Call reset() then compute_stft() to recompute for the new source.",
                UserWarning,
                stacklevel=2,
            )
        return cast(ComplexArray, self._source_signal)

    def stft_geometry(self) -> tuple[int, FloatArray, int, FloatArray, int]:
        """Return STFT geometry derived purely from signal model properties.

        Returns
        -------
        tuple of (int, FloatArray, int, FloatArray, int)
            ``(num_freq_bins, frequencies_hz, hop, window, num_frames)``.
            No source State is required.

        """
        stft, freq_normalized, hop, window = compute_stft(
            np.zeros(self.num_samples, dtype=np.complex64),
            self.frame_len,
            self.hop_factor,
            self.window_type,
        )
        num_frames, num_freq_bins = stft.shape
        freqs = np.asarray(freq_normalized * self.sampling_rate_hz, dtype=np.float64)
        return num_freq_bins, freqs, int(hop), np.asarray(window, dtype=np.float64), num_frames

    def reset(self) -> None:
        """Clear the cached STFT and source-signal state.

        Call this before reusing a signal model instance across multiple
        simulation runs, or after changing signal parameters, so that the next
        call to :meth:`compute_stft` generates a fresh waveform and STFT.

        This method is safe to call even if :meth:`compute_stft` has never been
        called — it is a no-op in that case.

        Subclasses that maintain additional caches (e.g. noise realisations in
        :class:`~bluepebble.signal.anthropogenic.SyntheticAnthropogenicSignal`) override this
        method and call ``super().reset()`` to ensure all state is cleared.
        """
        self._stft_cache = None
        self._frequencies = None
        self._hop = None
        self._window = None
        self._source_signal = None
        self._cached_source = None
