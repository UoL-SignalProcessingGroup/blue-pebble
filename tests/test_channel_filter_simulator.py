"""Tests for the channel-filter simulator and the source signal models built for it."""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import signal as scipy_signal
from stonesoup.types.array import StateVector
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState

FS = 10000
SOUND_SPEED = 1500.0
START = datetime(2014, 2, 19, 12, 0, tzinfo=UTC)
METADATA = {"position_mapping": [0, 2, 4], "velocity_mapping": [1, 3, 5]}

# Real bluepebble classes, populated lazily by the isolation fixture.
BP: SimpleNamespace | None = None


@pytest.fixture(autouse=True, scope="module")
def _isolate_bluepebble_import_state():
    """Import the real package for this module, then restore ``sys.modules``.

    Sibling mock-based tests reload simulator submodules against a fake Stone Soup and rely
    on them not being cached, so this follows the pattern in ``test_doppler.py``.
    """
    global BP
    saved_modules = dict(sys.modules)
    bluepebble_module = sys.modules.get("bluepebble")
    saved_package_attrs = set(vars(bluepebble_module)) if bluepebble_module is not None else None
    for name in list(sys.modules):
        if name == "bluepebble.simulator" or name.startswith("bluepebble.simulator."):
            del sys.modules[name]

    from bluepebble.models.environment import Constant
    from bluepebble.models.propagation import (
        SpectrumPropagationModel,
        SphericalAcousticPropagationModel,
    )
    from bluepebble.signal import ContinuousWaveSignal, SpectrumLevelSignal, WhiteNoiseSignal
    from bluepebble.simulator import ContinuousChannelFilterPassiveSonarArraySimulator

    BP = SimpleNamespace(
        ContinuousChannelFilterPassiveSonarArraySimulator=ContinuousChannelFilterPassiveSonarArraySimulator,
        SphericalAcousticPropagationModel=SphericalAcousticPropagationModel,
        SpectrumPropagationModel=SpectrumPropagationModel,
        Constant=Constant,
        SpectrumLevelSignal=SpectrumLevelSignal,
        ContinuousWaveSignal=ContinuousWaveSignal,
        WhiteNoiseSignal=WhiteNoiseSignal,
    )
    yield
    BP = None
    # A compiled rtrs can only be initialised once per process, so it has to stay imported.
    compiled = {
        name: module
        for name, module in sys.modules.items()
        if name == "rtrs" or name.startswith("rtrs.")
    }
    sys.modules.clear()
    sys.modules.update(saved_modules)
    sys.modules.update(compiled)
    if bluepebble_module is not None:
        package = sys.modules["bluepebble"]
        for attribute in set(vars(package)) - saved_package_attrs:
            delattr(package, attribute)


class _StaticArray:
    """A fixed line array along the x axis."""

    def __init__(self, num_sensors: int = 4, spacing: float = 0.45, depth: float = 3.0) -> None:
        """Lay the elements out."""
        self.num_sensors = num_sensors
        positions = np.vstack(
            [spacing * np.arange(num_sensors), np.zeros(num_sensors), np.full(num_sensors, depth)]
        )
        self.state = SimpleNamespace(
            array=SimpleNamespace(state_vector=positions, ref_state_vector=positions[:, [0]]),
            num_sensors=num_sensors,
        )

    def get_platform_state_at(self, timestamp: datetime) -> SimpleNamespace:
        """Return the unchanging geometry."""
        return self.state


def _path(position_at, duration_s: float, interval_s: float = 0.25) -> GroundTruthPath:
    """Sample a position function onto knots spanning the run and a margin either side."""
    states = []
    for time_s in np.arange(-1.0, duration_s + 1.0 + interval_s, interval_s):
        x, y, z = position_at(float(time_s))
        states.append(
            GroundTruthState(
                StateVector([x, 0.0, y, 0.0, z, 0.0]),
                timestamp=START + timedelta(seconds=float(time_s)),
                metadata=METADATA,
            )
        )
    return GroundTruthPath(states)


def _simulator(position_at, duration_s=8.0, level_db=100.0, seed=1, **kwargs):
    preroll_s = kwargs.pop("source_preroll_s", 4.0)
    return BP.ContinuousChannelFilterPassiveSonarArraySimulator(
        array=kwargs.pop("array", _StaticArray()),
        propagation_model=BP.SphericalAcousticPropagationModel(
            ssp=BP.Constant(speed=SOUND_SPEED), attenuation_factor=0.0
        ),
        signal_models=[
            BP.SpectrumLevelSignal(
                duration_s=duration_s + preroll_s,
                sampling_rate_hz=FS,
                level_db=level_db,
                seed=seed,
            )
        ],
        ground_truth_paths=[_path(position_at, duration_s)],
        start_time=START,
        duration_s=duration_s,
        source_preroll_s=preroll_s,
        **kwargs,
    )


def _density(signal: np.ndarray, nperseg: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """Return the two-sided Welch density, which for a one-sided signal is the real one's."""
    return scipy_signal.welch(
        signal, fs=FS, nperseg=nperseg, noverlap=nperseg // 2, return_onesided=False
    )


def _in_band_level_db(signal: np.ndarray, low=900.0, high=1600.0) -> float:
    frequencies, density = _density(signal)
    return float(10.0 * np.log10(np.mean(density[(frequencies >= low) & (frequencies <= high)])))


class TestLevel:
    """Output level follows the channel and nothing else."""

    @pytest.mark.parametrize("range_m", [100.0, 307.2, 614.4, 921.6, 1500.0, 2000.0])
    def test_level_does_not_depend_on_delay_modulo_frame(self, range_m):
        """Received level follows spreading at any delay."""
        # These ranges put the delay at various fractions of a 4096-sample frame.
        output = _simulator(lambda t: (range_m, 0.0, 3.0)).render()
        received = _in_band_level_db(output[0][FS:-FS])
        assert received == pytest.approx(100.0 - 20.0 * np.log10(range_m), abs=0.5)

    def test_sensor_gains_apply(self):
        """Per-sensor gains scale the source signal."""
        gains = np.array([1.0, 0.5, 1.0, 1.0])
        output = _simulator(lambda t: (500.0, 0.0, 3.0), sensor_gains=gains).render()
        difference = _in_band_level_db(output[0][FS:-FS]) - _in_band_level_db(output[1][FS:-FS])
        assert difference == pytest.approx(6.02, abs=0.5)

    def test_last_snapshot_is_not_silent(self):
        """The end of the run holds its level."""
        output = _simulator(lambda t: (500.0, 0.0, 3.0)).render()
        assert _in_band_level_db(output[0][-2 * FS :]) == pytest.approx(
            _in_band_level_db(output[0][2 * FS : 4 * FS]), abs=1.0
        )

    def test_first_samples_are_not_silent(self):
        """Sound already in the water at the start of the run arrives there."""
        output = _simulator(lambda t: (3000.0, 0.0, 3.0)).render()
        assert np.std(output[0][:FS]) == pytest.approx(np.std(output[0][FS : 2 * FS]), rel=0.2)

    def test_source_is_silent_outside_its_path(self):
        """A source only sounds while its path has knots."""
        simulator = _simulator(lambda t: (500.0, 0.0, 3.0))
        states = [s for s in simulator.ground_truth_paths[0] if (s.timestamp - START).seconds < 3]
        simulator.ground_truth_paths[0] = GroundTruthPath(states)
        output = simulator.render()
        assert np.std(output[0][5 * FS :]) == 0.0
        assert np.std(output[0][FS : 2 * FS]) > 0.0

    def test_short_preroll_is_rejected(self):
        """The run must have waveform for the longest delay."""
        with pytest.raises(ValueError, match="source_preroll_s"):
            _simulator(lambda t: (6000.0, 0.0, 3.0), source_preroll_s=1.0).render()


class TestDelayAndDoppler:
    """The bulk delay and inter-sensor delays are real delays."""

    def test_intersensor_delay_matches_geometry(self):
        """Endfire along +x: the last sensor hears it first by 3 * 0.45 / c."""
        output = _simulator(lambda t: (800.0, 0.0, 3.0)).render()
        first, last = output[0][FS:-FS].real, output[3][FS:-FS].real
        lags = scipy_signal.correlation_lags(first.size, last.size)
        peak = lags[np.argmax(scipy_signal.correlate(first, last))]
        assert peak / FS == pytest.approx(3 * 0.45 / SOUND_SPEED, abs=1.5 / FS)

    def test_absolute_delay_against_source(self):
        """The received waveform is the source, delayed by range / c."""
        range_m = 750.0
        simulator = _simulator(lambda t: (range_m, 0.0, 3.0), array=_StaticArray(1))
        output = simulator.render()[0].real
        waveform = np.real(simulator.signal_models[0].get_source_waveform(None))
        preroll = int(4.0 * FS)
        expected = int(round(range_m / SOUND_SPEED * FS))
        segment = output[2 * FS : 5 * FS]

        def correlation_at(lag: int) -> float:
            start = preroll + 2 * FS - expected + lag
            return float(np.corrcoef(segment, waveform[start : start + segment.size])[0, 1])

        # The taper at the band edges removes a little of the source, so not quite 1.
        assert correlation_at(0) > 0.95
        assert correlation_at(0) > correlation_at(-2) and correlation_at(0) > correlation_at(2)

    def test_doppler_matches_radial_speed(self):
        """The delay rate matches the radial speed."""
        # Closing at 15 m/s compresses time by 1 %. Without a tone to read a shift from, the
        # same thing shows as the lag between source and received waveform shrinking at
        # that rate. Segments stay short so that compression barely decorrelates them.
        speed = 15.0
        simulator = _simulator(
            lambda t: (2000.0 - speed * t, 0.0, 3.0),
            duration_s=10.0,
            array=_StaticArray(1),
        )
        output = simulator.render()[0].real
        waveform = np.real(simulator.signal_models[0].get_source_waveform(None))
        delays = []
        for start_s in (2, 8):
            segment = output[start_s * FS : start_s * FS + 400]
            expected_delay_s = (2000.0 - speed * start_s) / SOUND_SPEED
            search_from = int((start_s + 4.0 - expected_delay_s) * FS) - 100
            correlation = scipy_signal.correlate(
                waveform[search_from : search_from + 600], segment, mode="valid"
            )
            position = search_from + int(np.argmax(np.abs(correlation)))
            delays.append(start_s + 4.0 - position / FS)
        assert (delays[0] - delays[1]) / 6.0 == pytest.approx(speed / SOUND_SPEED, abs=1e-3)


def test_multipath_colour_survives_rendering():
    """A notched channel stays notched: the fractional-delay backend would flatten it."""

    class _TwoPathModel(BP.SphericalAcousticPropagationModel, BP.SpectrumPropagationModel):
        """Direct path plus an equal, inverted path 5 ms later: nulls every 200 Hz."""

        def propagate_spectrum(self, platform, source, frequencies_hz):
            """Add the inverted path."""
            transfer, delays = super().propagate_spectrum(platform, source, frequencies_hz)
            frequencies = np.asarray(frequencies_hz)
            return transfer * (1.0 - np.exp(-2j * np.pi * frequencies * 5e-3)), delays

    simulator = _simulator(lambda t: (500.0, 0.0, 3.0), duration_s=20.0)
    simulator.propagation_model = _TwoPathModel(
        ssp=BP.Constant(speed=SOUND_SPEED), attenuation_factor=0.0
    )
    output = simulator.render()[0][FS:-FS]
    frequencies, density = _density(output)
    level = 10.0 * np.log10(density)
    # 1 - exp(-j 2 pi f 5 ms) is zero at multiples of 200 Hz and doubles midway between.
    peak = level[np.argmin(np.abs(frequencies - 1100.0))]
    null = level[np.argmin(np.abs(frequencies - 1200.0))]
    assert peak - null > 15.0


class TestSpectrumLevelSignal:
    """Source waveforms have the level and statistics they were asked for."""

    def test_level_and_exponent(self):
        """The source spectrum matches its level and exponent."""
        signal = BP.SpectrumLevelSignal(
            duration_s=30.0, sampling_rate_hz=FS, level_db=120.0, spectral_exponent=-2.0, seed=1
        ).get_source_waveform(None)
        assert np.iscomplexobj(signal)
        grid, density = _density(signal, nperseg=2048)
        level = 10.0 * np.log10(density)
        assert level[np.argmin(np.abs(grid - 1000.0))] == pytest.approx(120.0, abs=1.5)
        low = level[np.argmin(np.abs(grid - 850.0))]
        difference = low - level[np.argmin(np.abs(grid - 1600.0))]
        assert difference == pytest.approx(20.0 * np.log10(1600.0 / 850.0), abs=2.0)

    def test_different_seeds_are_uncorrelated(self):
        """Separate seeds give independent waveforms."""
        first, second = (
            BP.SpectrumLevelSignal(
                duration_s=5.0, sampling_rate_hz=FS, level_db=100.0, seed=seed
            ).get_source_waveform(None)
            for seed in (1, 2)
        )
        assert abs(np.corrcoef(first.real, second.real)[0, 1]) < 0.05


class TestContinuousWaveSignal:
    """A CW projector's tones hold exactly their levels, at the source and the array."""

    @staticmethod
    def _tones(levels_db, frequencies_hz):
        levels = np.asarray(levels_db, dtype=float)
        return GroundTruthState(
            StateVector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            timestamp=START,
            metadata={
                "frequencies_hz": np.asarray(frequencies_hz, dtype=float),
                "amplitudes_upa": np.sqrt(2.0) * 10.0 ** (levels / 20.0),
                "phases_rad": np.zeros(levels.size),
            },
        )

    def test_tones_have_their_rms_levels(self):
        """Each tone's RMS level is its metadata level, all at positive frequencies."""
        signal = BP.ContinuousWaveSignal(duration_s=10.0, sampling_rate_hz=FS)
        waveform = signal.get_source_waveform(self._tones([120.0, 110.0], [900.0, 1300.0]))
        assert waveform.size == int(10.0 * FS)
        spectrum = np.abs(np.fft.fft(waveform)) ** 2 / waveform.size**2
        grid = np.fft.fftfreq(waveform.size, 1.0 / FS)
        assert np.sum(spectrum[grid < 0]) < 1e-12 * np.sum(spectrum)
        for frequency, level in ((900.0, 120.0), (1300.0, 110.0)):
            near = np.abs(grid - frequency) < 0.5
            assert 10.0 * np.log10(spectrum[near].sum()) == pytest.approx(level, abs=0.01)

    def test_spreading_loss_through_the_simulator(self):
        """A static CW source arrives at its level less spherical spreading."""
        range_m, duration_s, preroll_s = 200.0, 4.0, 4.0
        tones = self._tones([120.0], [1000.0])
        path = _path(lambda t: (range_m, 0.0, 3.0), duration_s)
        for state in path.states:
            state.metadata = {**METADATA, **tones.metadata}
        simulator = BP.ContinuousChannelFilterPassiveSonarArraySimulator(
            array=_StaticArray(),
            propagation_model=BP.SphericalAcousticPropagationModel(
                ssp=BP.Constant(speed=SOUND_SPEED), attenuation_factor=0.0
            ),
            signal_models=[
                BP.ContinuousWaveSignal(duration_s=duration_s + preroll_s, sampling_rate_hz=FS)
            ],
            ground_truth_paths=[path],
            start_time=START,
            duration_s=duration_s,
            source_preroll_s=preroll_s,
        )
        received = simulator.render(include_noise=False)[:, int(FS) : -int(FS)]
        level = 20.0 * np.log10(np.std(received, axis=1))
        expected = 120.0 - 20.0 * np.log10(np.hypot(range_m, 0.45 * np.arange(4)))
        np.testing.assert_allclose(level, expected, atol=0.3)


    def test_tones_must_lie_below_nyquist(self):
        """A tone at or above fs/2 has no one-sided form."""
        signal = BP.ContinuousWaveSignal(duration_s=1.0, sampling_rate_hz=FS)
        with pytest.raises(ValueError, match="fs/2"):
            signal.get_source_waveform(self._tones([100.0], [FS / 2]))


class TestOutput:
    """The output follows the package's one-sided convention."""

    def test_output_is_one_sided(self):
        """No power at negative frequencies, as for the sources and the other simulators."""
        output = _simulator(lambda t: (500.0, 0.0, 3.0)).render(include_noise=False)
        assert np.iscomplexobj(output)
        frequencies, density = _density(output[0][FS:-FS])
        assert np.sum(density[frequencies < 0]) < 1e-6 * np.sum(density)

    def test_one_shared_signal_model_is_copied_per_path(self):
        """One model for two paths gives each path its own waveform, as the other simulators do."""
        simulator = _simulator(lambda t: (500.0, 0.0, 3.0))
        simulator.ground_truth_paths.append(_path(lambda t: (-500.0, 0.0, 3.0), 8.0))
        simulator.signal_models[0] = BP.SpectrumLevelSignal(
            duration_s=12.0, sampling_rate_hz=FS, level_db=100.0
        )
        output = simulator.render(include_noise=False)
        assert np.all(np.isfinite(output))
        assert simulator.signal_models[0]._source_signal is None

    def test_snapshots_carry_the_noise_and_timestamps(self):
        """Snapshots are contiguous, stamped at their centres, with the noise added."""
        simulator = _simulator(lambda t: (500.0, 0.0, 3.0), snapshot_duration_s=2.0)
        simulator.noise_model = BP.WhiteNoiseSignal(
            duration_s=8.0, sampling_rate_hz=FS, amplitude_upa=1.0, seed=2
        )
        batches = list(simulator.sensor_data_gen())
        assert len(batches) == 4
        timestamp, data = batches[0]
        assert timestamp == START + timedelta(seconds=1.0)
        assert next(iter(data)).raw_signals.shape == (4, 2 * FS)
