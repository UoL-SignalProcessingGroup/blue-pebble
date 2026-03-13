"""Shared abstractions for anthropogenic signal models."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from stonesoup.base import Property

from ..base import ContinuousTimestepSignal
from ..utils import compute_stft

if TYPE_CHECKING:
    from stonesoup.types.state import State

FloatArray: TypeAlias = NDArray[np.float64]
ComplexArray: TypeAlias = NDArray[np.complexfloating[Any, Any]]
Complex128Array: TypeAlias = NDArray[np.complex128]
CachedStftResult: TypeAlias = tuple[NDArray[np.complex64], FloatArray, int, FloatArray]


class AnthropogenicSignalBase(ContinuousTimestepSignal, ABC):
    """Base class for STFT-first anthropogenic signal models."""

    frame_len: int = Property(default=1024, doc="STFT frame length in samples")
    hop_factor: int = Property(
        default=4,
        doc="Hop factor (hop = frame_len // hop_factor)",
    )
    window_type: str = Property(default="hann", doc="Window type for STFT")

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialise shared STFT caches."""
        super().__init__(*args, **kwargs)
        self._stft_cache: NDArray[np.complex64] | None = None
        self._frequencies: FloatArray | None = None
        self._hop: int | None = None
        self._window: FloatArray | None = None
        self._source_signal: ComplexArray | None = None

    @abstractmethod
    def _generate_base_signal(self, source: "State") -> ComplexArray:
        """Generate full-duration source waveform for STFT processing."""

    def compute_stft(self, source: "State") -> CachedStftResult:
        """Compute and cache STFT outputs for the source signal.

        Parameters
        ----------
        source : State
            Source state used by concrete implementations to build the waveform.

        Returns
        -------
        CachedStftResult
            Cached STFT tuple ``(stft, frequencies_hz, hop_samples, window)``.

        """
        if self._stft_cache is not None:
            return (
                self._stft_cache,
                cast(FloatArray, self._frequencies),
                cast(int, self._hop),
                cast(FloatArray, self._window),
            )

        self._source_signal = self._generate_base_signal(source)

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
        return cast(ComplexArray, self._source_signal)

    def generate(
        self,
        source: "State",
        sensor_delays_s: ArrayLike,
        tloss_db: ArrayLike | float,
        propagation_time_s: float,
    ) -> Complex128Array:
        """Reject per-timestep generation for STFT-first models.

        Anthropogenic models are designed for frequency-domain
        processing via :meth:`compute_stft`.

        Raises
        ------
        NotImplementedError
            Always raised for this base implementation.

        """
        _ = source, sensor_delays_s, tloss_db, propagation_time_s
        msg = (
            f"{type(self).__name__} does not support per-timestep generation. "
            "Use compute_stft() and process in frequency domain via "
            "ContinuousPassiveSonarArraySimulator."
        )
        raise NotImplementedError(msg)

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
        """Clear cached STFT and source-signal state."""
        self._stft_cache = None
        self._frequencies = None
        self._hop = None
        self._window = None
        self._source_signal = None
