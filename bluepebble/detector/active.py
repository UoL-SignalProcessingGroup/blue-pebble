"""Active sonar detector: matched filtering and range estimation from echo delay."""

from collections.abc import Generator, Iterable
from datetime import datetime
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray, ArrayLike

from stonesoup.base import Property
from stonesoup.buffered_generator import BufferedGenerator
from stonesoup.reader.base import DetectionReader
from stonesoup.types.detection import Detection

from .algorithms import DetectionAlgorithm
from ..types.sensordata import ActiveSonarSensorData

ComplexArray: TypeAlias = NDArray[np.complexfloating[Any, Any]]
FloatArray: TypeAlias = NDArray[np.float64]
DetectionArray: TypeAlias = NDArray[np.float64]
SensorDataStep: TypeAlias = tuple[datetime, Iterable[ActiveSonarSensorData]]
DetectionBatch: TypeAlias = tuple[datetime, set[Detection]]



class ActiveSonarDetectorOmni(DetectionReader):
    """Matched-filter range detector for monostatic active sonar.

    Processes ``ActiveSonarSensorData`` produced by :class:`.BellhopActiveSonarSimulatorOmni`.
    For each ping it:

    1. Computes the matched filter envelope via FFT-based cross-correlation.
    2. Runs the envelope through a configurable ``detection_chain`` (e.g. CFAR
       followed by a peak-picker) to find candidate detections.
    3. Converts each surviving sample index to a one-way delay and then to
       slant range.
    4. Yields Stone Soup ``Detection`` objects with ``state_vector = [[range_m]]``.

    Parameters
    ----------
    sensor_data_gen : Generator
        Generator yielding ``(datetime, set[ActiveSonarSensorData])`` tuples,
        as produced by :class:`.BellhopActiveSonarSimulatorOmni`.
    sampling_rate_hz : int
        Sampling rate of the received waveform in Hz.
    sound_speed_ms : float
        Assumed sound speed in m/s used to convert round-trip delay to slant
        range. Default ``1500.0``.
    min_range_m : float
        Minimum slant range in metres. Detections at shorter ranges (e.g.
        direct-blast leakage) are suppressed. Default ``10.0``.
    detection_chain : list[DetectionAlgorithm]
        Sequential list of detection algorithms applied to the MF envelope.
        A typical chain is ``[CACFARDetector(...), PeakDetector(...)]``.

    """

    sensor_data_gen: Generator[SensorDataStep, None, None] = Property(
        doc="Generator yielding (datetime, set[ActiveSonarSensorData]) tuples"
    )
    sampling_rate_hz: int = Property(
        doc="Sampling rate of the received waveform in Hz"
    )
    sound_speed_ms: float = Property(
        default=1500.0,
        doc="Sound speed in m/s used to convert round-trip delay to slant range",
    )
    min_range_m: float = Property(
        default=10.0,
        doc="Minimum slant range in metres; shorter-range peaks are suppressed",
    )
    detection_chain: list[DetectionAlgorithm] = Property(
        doc="A list of detection algorithms to apply sequentially.",
    )

    @BufferedGenerator.generator_method
    def detections_gen(self) -> Generator[DetectionBatch, None, None]:
        """Yield range detections from matched-filtered ping data.

        Yields
        ------
        tuple[datetime, set[Detection]]
            Ping timestamp and detections found in that ping.  Each
            ``Detection`` carries:

            - ``state_vector`` : ``[[range_m]]`` — slant range in metres.
            - ``metadata["mf_amplitude"]`` : normalised MF peak amplitude.
            - ``metadata["delay_s"]`` : estimated round-trip delay in seconds.

        """
        fs = self.sampling_rate_hz
        c = self.sound_speed_ms

        min_idx = max(1, int(2.0 * self.min_range_m / c * fs))

        for timestamp, sensor_data_set in self.sensor_data_gen:
            detections: set[Detection] = set()

            for sensor_data in sensor_data_set:
                envelope = matched_filter(
                    sensor_data.received_waveform, sensor_data.transmit_pulse
                )

                # The detection chain (e.g. CFAR, ThresholdDetector) operates on a
                # dB-scale map, but the MF envelope above is a linear [0, 1]
                # amplitude. Convert before running the chain, which takes the
                # place of an SNR map in the active sonar case.
                envelope_db = 20.0 * np.log10(np.maximum(envelope, 1e-12))
                raw_detections: DetectionArray = self._run_detection_chain(envelope_db)

                # Create Stone Soup Detections from the raw results
                if raw_detections.size > 0:
                    for row in raw_detections:

                        # Take the sample index from this detection row; look up the
                        # linear MF amplitude at that index for the metadata below.
                        idx = int(row[0])
                        amp = float(envelope[idx])

                        if idx < min_idx:
                            continue

                        delay_s = idx / fs
                        range_m = delay_s * c / 2.0

                        detections.add(
                            Detection(
                                state_vector=[[range_m]],
                                timestamp=timestamp,
                                metadata={
                                    "mf_amplitude": float(amp),
                                    "delay_s": float(delay_s),
                                },
                            )
                        )

            yield timestamp, detections

    def _run_detection_chain(self, initial_snr_map: ArrayLike) -> DetectionArray:
        """Process a data map through a sequential chain of detection algorithms.

        Each algorithm in ``detection_chain`` is applied in sequence; the set of detections
        produced by one stage is converted to a sparse input map for the next stage (non-detected
        indices set to -inf).

        Parameters
        ----------
        initial_snr_map : ArrayLike
            The initial 1D data map (for example, SNR in dB) to be processed.

        Returns
        -------
        DetectionArray
            A 2D array of final detections where each row is ``[index, value]``. Returns an empty
            array if no detections are found at any stage.

        """
        initial_snr_map_array = np.asarray(initial_snr_map, dtype=np.float64)
        if not self.detection_chain:
            return np.empty((0, 2), dtype=np.float64)

        input_data_map = initial_snr_map_array
        final_detections = np.empty((0, 2), dtype=np.float64)

        for algorithm in self.detection_chain:
            current_detections = algorithm.detect(input_data_map)

            if current_detections.size == 0:
                return np.empty((0, 2), dtype=np.float64)

            final_detections = current_detections

            input_data_map = np.full(len(initial_snr_map_array), -np.inf, dtype=np.float64)
            indices = final_detections[:, 0].astype(int)
            values = final_detections[:, 1]
            input_data_map[indices] = values

        return np.asarray(final_detections, dtype=np.float64)


def matched_filter(received: ComplexArray, pulse: ComplexArray) -> FloatArray:
    """Compute the matched filter envelope of a received waveform.

    Cross-correlates ``received`` with ``pulse`` in the frequency domain and
    returns the normalised real envelope. The output index ``k`` corresponds to
    a round-trip delay of ``k / sampling_rate_hz`` seconds.

    Parameters
    ----------
    received : ComplexArray
        Complex analytic received waveform, shape ``(n_receive,)``.
    pulse : ComplexArray
        Complex analytic transmit pulse, shape ``(n_pulse,)``.

    Returns
    -------
    FloatArray
        Normalised matched filter envelope, shape ``(n_receive,)``, with values
        in ``[0, 1]``. A value of 1.0 at index ``k`` means a perfect match at
        delay ``k / fs``.

    """
    n = len(received) + len(pulse) - 1
    R = np.fft.fft(received, n)
    H = np.fft.fft(pulse, n)
    mf_full = np.fft.ifft(R * np.conj(H), n)
    envelope = np.abs(mf_full[: len(received)])

    peak = envelope.max()
    if peak > 0.0:
        envelope = envelope / peak
    return np.asarray(envelope, dtype=np.float64)
