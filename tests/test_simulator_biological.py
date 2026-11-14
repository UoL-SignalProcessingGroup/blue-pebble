"""Tests for biological sources in the passive simulators, through the shared source lifecycle.

The oracles are geometric and physical: a beamformed peak at the source's true bearing,
spherical spreading (ten times the range, a tenth of the pressure), and arrival order across
the array (the nearer sensor hears the source first).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import signal as sp_signal
from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel,
    ConstantVelocity,
)
from stonesoup.types.array import StateVector
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState

from .support import isolate_real_bluepebble_imports, restore_real_bluepebble_imports

START = datetime(2026, 1, 1)
C = 1500.0
FS = 4000
NUM_STEPS = 8
STEP_S = 1.0
BAND_HZ = (300.0, 700.0)
NUM_SENSORS = 32
SPACING_M = 0.75
SIMULATORS = ["stft", "fractional_delay", "discrete"]
COLONY_LEVEL_DB = 150.0


@pytest.fixture
def bp():
    """Import the real classes without leaking caching into other tests."""
    snapshot = isolate_real_bluepebble_imports()
    import bluepebble.signal as signal_package
    from bluepebble.arrays import RigidArray
    from bluepebble.models.environment import Constant
    from bluepebble.models.propagation import SphericalAcousticPropagationModel
    from bluepebble.platform import SonarPlatform
    from bluepebble.signal import (
        AnthropogenicSignal,
        BiologicalSignal,
        SnappingShrimpColonySignal,
        SourceSignal,
        SyntheticAnthropogenicSignal,
        analytic_signal,
    )
    from bluepebble.signal.effects import Effect, Reverb
    from bluepebble.sigproc import DelayAndSumBeamformer, SteeringCalculator
    from bluepebble.simulator import (
        ContinuousFractionalDelayPassiveSonarArraySimulator,
        ContinuousSTFTPassiveSonarArraySimulator,
        DiscretePassiveSonarArraySimulator,
    )

    class Line(RigidArray):
        """Sensors ``SPACING_M`` apart along the platform's x axis, the first at the origin."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            zeros = np.zeros(NUM_SENSORS)
            self._local_offsets = np.vstack([np.arange(NUM_SENSORS) * SPACING_M, zeros, zeros])
            self._local_axis = np.array([1.0, 0.0, 0.0])

    class BandNoiseSource(BiologicalSignal):
        """Seeded noise in ``BAND_HZ``, one-sided, standing in for a biological source."""

        def _generate_base_signal(self, source):
            sos = sp_signal.butter(6, BAND_HZ, btype="bandpass", fs=FS, output="sos")
            noise = sp_signal.sosfiltfilt(sos, self._rng.standard_normal(self.num_samples))
            return analytic_signal(noise)

    yield SimpleNamespace(
        AnthropogenicSignal=AnthropogenicSignal,
        BandNoiseSource=BandNoiseSource,
        BiologicalSignal=BiologicalSignal,
        Constant=Constant,
        DelayAndSumBeamformer=DelayAndSumBeamformer,
        Effect=Effect,
        Line=Line,
        Reverb=Reverb,
        SnappingShrimpColonySignal=SnappingShrimpColonySignal,
        SonarPlatform=SonarPlatform,
        SourceSignal=SourceSignal,
        SphericalAcousticPropagationModel=SphericalAcousticPropagationModel,
        SteeringCalculator=SteeringCalculator,
        SyntheticAnthropogenicSignal=SyntheticAnthropogenicSignal,
        signal_package=signal_package,
        simulators={
            "stft": ContinuousSTFTPassiveSonarArraySimulator,
            "fractional_delay": ContinuousFractionalDelayPassiveSonarArraySimulator,
            "discrete": DiscretePassiveSonarArraySimulator,
        },
    )
    restore_real_bluepebble_imports(snapshot)


def _timestamps(num_steps=NUM_STEPS):
    return [START + timedelta(seconds=STEP_S * step) for step in range(num_steps)]


def _array(bp, num_steps=NUM_STEPS):
    platform = bp.SonarPlatform(
        states=[GroundTruthState(StateVector([0.0, 0.0, 0.0, 0.0, -50.0, 0.0]), timestamp=START)],
        position_mapping=[0, 2, 4],
        transition_models=[CombinedLinearGaussianTransitionModel([ConstantVelocity(0.0)] * 3)],
        transition_times=[timedelta(hours=1)],
    )
    for timestamp in _timestamps(num_steps)[1:]:
        platform.move(timestamp)
    array = bp.Line()
    platform.add_sensor(array)
    return array


def _stationary_path(range_m, bearing_deg, num_steps=NUM_STEPS, **metadata):
    """Return a source held at ``range_m`` and ``bearing_deg`` from the array's first sensor."""
    bearing = np.deg2rad(bearing_deg)
    state_vector = StateVector(
        [range_m * np.cos(bearing), 0.0, range_m * np.sin(bearing), 0.0, -50.0, 0.0]
    )
    return GroundTruthPath(
        [
            GroundTruthState(
                state_vector, timestamp=ts, metadata={"position_mapping": [0, 2, 4], **metadata}
            )
            for ts in _timestamps(num_steps)
        ]
    )


def _source(bp, seed=1, **kwargs):
    return bp.BandNoiseSource(
        duration_s=NUM_STEPS * STEP_S, sampling_rate_hz=FS, seed=seed, **kwargs
    )


def _run(bp, simulator, models, paths, beamform=False, num_steps=NUM_STEPS):
    """Run noise-free sources through ``simulator``; return its sensor data per scan."""
    beamforming = {}
    if beamform:
        beamforming = {
            "beamformer": bp.DelayAndSumBeamformer(domain="frequency", sampling_rate_hz=FS),
            "steering_calculator": bp.SteeringCalculator(
                ssp=bp.Constant(speed=C), steering_sector_rad=(0.0, np.pi), num_beams=181
            ),
        }
    generator = bp.simulators[simulator](
        array=_array(bp, num_steps),
        propagation_model=bp.SphericalAcousticPropagationModel(
            ssp=bp.Constant(speed=C), attenuation_factor=0.0
        ),
        signal_models=models,
        ground_truth_paths=paths,
        **beamforming,
    ).sensor_data_gen()
    return [next(iter(data)) for _, data in generator]


def _scans(bp, simulator, range_m, bearing_deg, beamform=False):
    """Run one band-noise source through ``simulator``; return its sensor data per scan."""
    return _run(bp, simulator, [_source(bp)], [_stationary_path(range_m, bearing_deg)], beamform)


def _beam_power(scans, first_step=3):
    """Return delay-and-sum power per beam, summed from ``first_step`` on, and the bearings."""
    power = sum(
        np.sum(np.abs(scan.beamformed_data) ** 2, axis=tuple(range(1, scan.beamformed_data.ndim)))
        for scan in scans[first_step:]
    )
    return power, np.rad2deg(scans[-1].steering_bearings_rad)


def _steady_signals(scans, first_step=3):
    """Return the sensor signals from ``first_step`` on, once every source has arrived."""
    return np.concatenate([scan.raw_signals for scan in scans[first_step:]], axis=1)


def test_both_source_families_share_the_source_signal_base(bp):
    """Anthropogenic and biological sources are both ``SourceSignal``, which is exported."""
    assert issubclass(bp.BiologicalSignal, bp.SourceSignal)
    assert issubclass(bp.AnthropogenicSignal, bp.SourceSignal)
    assert "SourceSignal" in bp.signal_package.__all__


@pytest.mark.parametrize("simulator", SIMULATORS)
def test_stationary_biological_source_peaks_at_its_bearing(bp, simulator):
    """Delay-and-sum power summed over the run peaks at the source's true bearing.

    The half-power beam width is about lambda / aperture = 3 m / 23 m, or 7 degrees, at the
    band centre, so 2 degrees is well inside the main lobe.
    """
    true_bearing_deg = 60.0
    scans = _scans(bp, simulator, range_m=2000.0, bearing_deg=true_bearing_deg, beamform=True)

    assert len(scans) == NUM_STEPS
    for scan in scans:
        assert scan.raw_signals.shape[0] == NUM_SENSORS
        assert np.all(np.isfinite(scan.raw_signals))

    power, bearings_deg = _beam_power(scans)
    assert bearings_deg[np.argmax(power)] == pytest.approx(true_bearing_deg, abs=2.0)


@pytest.mark.parametrize("simulator", SIMULATORS)
def test_ten_times_the_range_gives_a_tenth_of_the_pressure(bp, simulator):
    """Spherical spreading: 20 dB more loss at ten times the range, a factor 10 in RMS.

    The two windows hold different stretches of the same stationary noise, so their RMS
    values differ by a few per cent from sampling alone.
    """
    near = _steady_signals(_scans(bp, simulator, range_m=200.0, bearing_deg=90.0))
    far = _steady_signals(_scans(bp, simulator, range_m=2000.0, bearing_deg=90.0))

    def rms(signals):
        return np.sqrt(np.mean(np.abs(signals) ** 2))

    assert rms(near) / rms(far) == pytest.approx(10.0, rel=0.05)


@pytest.mark.parametrize("simulator", SIMULATORS)
def test_the_nearer_sensor_hears_the_source_first(bp, simulator):
    """A source off the array's far end reaches the last sensor one aperture's travel early."""
    scans = _scans(bp, simulator, range_m=2000.0, bearing_deg=0.0)
    signals = np.real(_steady_signals(scans))
    first_sensor, last_sensor = signals[0], signals[-1]

    correlation = sp_signal.correlate(first_sensor, last_sensor, mode="full")
    lags = sp_signal.correlation_lags(first_sensor.size, last_sensor.size, mode="full")
    lag_of_first_behind_last = lags[np.argmax(correlation)]

    aperture_m = (NUM_SENSORS - 1) * SPACING_M
    assert lag_of_first_behind_last == pytest.approx(aperture_m / C * FS, abs=1.5)


def test_compute_stft_needs_a_reset_and_then_draws_a_new_realisation(bp):
    """The cached lifecycle: one ``compute_stft`` per ``reset``, each a fresh waveform."""
    model = _source(bp)
    state = _stationary_path(2000.0, 90.0)[0]

    model.compute_stft(state)
    first = model.get_source_signal().copy()
    np.testing.assert_array_equal(model.get_source_waveform(state), first)
    with pytest.raises(RuntimeError, match="reset"):
        model.compute_stft(state)

    model.reset()
    model.compute_stft(state)
    assert not np.array_equal(model.get_source_signal(), first)


def test_generate_and_apply_propagation_are_deprecated_but_still_work(bp):
    """Both warn and name the simulators; 20 dB of loss still scales the waveform by 0.1."""
    model = _source(bp)
    state = _stationary_path(2000.0, 90.0)[0]
    waveform = np.arange(8, dtype=np.complex128)

    with pytest.warns(DeprecationWarning, match="simulator's signal_models"):
        propagated = model._apply_propagation(waveform, np.zeros(2), 20.0, 0.0)
    np.testing.assert_allclose(propagated, np.tile(0.1 * waveform, (2, 1)), atol=1e-12)

    with pytest.warns(DeprecationWarning, match=r"generate\(\) is deprecated") as record:
        generated = model.generate(state, np.zeros(2), 0.0, 0.0)
    assert len(record) == 1
    assert generated.shape == (2, model.num_samples)


def test_effects_are_deprecated(bp):
    """Building an effect warns, whether it is ``Reverb`` or a user's own subclass."""

    class Doubler(bp.Effect):
        def apply(self, signals, sampling_rate_hz):
            return 2.0 * signals

    with pytest.warns(DeprecationWarning, match="Reverb is deprecated"):
        bp.Reverb(duration_s=0.1)
    with pytest.warns(DeprecationWarning, match="Doubler is deprecated"):
        Doubler()


def _colony(bp, num_steps=NUM_STEPS, seed=2):
    """Return a colony whose calibration window is one scan, so its level is per scan."""
    return bp.SnappingShrimpColonySignal(
        duration_s=num_steps * STEP_S,
        sampling_rate_hz=FS,
        snap_rate_hz=200.0,
        median_source_level_db=COLONY_LEVEL_DB,
        band_hz=BAND_HZ,
        calibration_window_s=STEP_S,
        amplitude_cap=31.6,
        seed=seed,
    )


def test_colony_peaks_at_its_bearing_at_its_level_less_the_spreading_loss(bp):
    """A colony's median scan level at a sensor is ``L - 20 log10(r)`` under spherical spreading.

    Over eight seeds the median of these 61 scans scattered by 0.23 dB (standard deviation)
    about that level, so 1 dB is about four standard deviations.
    """
    num_steps, range_m, bearing_deg = 64, 2000.0, 60.0
    scans = _run(
        bp,
        "stft",
        [_colony(bp, num_steps)],
        [_stationary_path(range_m, bearing_deg, num_steps)],
        beamform=True,
        num_steps=num_steps,
    )

    power, bearings_deg = _beam_power(scans)
    assert bearings_deg[np.argmax(power)] == pytest.approx(bearing_deg, abs=2.0)

    scan_powers = [np.mean(np.abs(scan.raw_signals[0]) ** 2) for scan in scans[3:]]
    median_level_db = 10 * np.log10(np.median(scan_powers))
    assert median_level_db == pytest.approx(COLONY_LEVEL_DB - 20 * np.log10(range_m), abs=1.0)


def _ship_and_path(bp, bearing_deg, seed=3):
    """Return a two-tonal ship at the colony's level and its path, 2 km away."""
    path = _stationary_path(
        2000.0,
        bearing_deg,
        frequencies_hz=np.array([400.0, 550.0]),
        amplitudes_upa=np.full(2, 10 ** (COLONY_LEVEL_DB / 20)),
        phases_rad=np.zeros(2),
        tonal_bandwidth_hz=2.0,
        noise_amplitude_upa=0.0,
    )
    ship = bp.SyntheticAnthropogenicSignal(
        duration_s=NUM_STEPS * STEP_S, sampling_rate_hz=FS, seed=seed
    )
    return ship, path


def test_colony_and_ship_share_a_run_and_each_peaks_at_its_own_bearing(bp):
    """A colony and a ship go through one STFT run together."""
    colony_bearing_deg, ship_bearing_deg = 60.0, 120.0
    ship, ship_path = _ship_and_path(bp, ship_bearing_deg)

    scans = _run(
        bp,
        "stft",
        [_colony(bp), ship],
        [_stationary_path(2000.0, colony_bearing_deg), ship_path],
        beamform=True,
    )

    power, bearings_deg = _beam_power(scans)
    peaks, _ = sp_signal.find_peaks(power)
    two_strongest = peaks[np.argsort(power[peaks])[-2:]]
    assert sorted(bearings_deg[two_strongest]) == pytest.approx(
        [colony_bearing_deg, ship_bearing_deg], abs=2.0
    )


def test_colony_and_ship_together_are_the_sum_of_each_alone(bp):
    """The sensor signals add exactly, and the powers add up to the sources' correlation.

    Both sources are one-sided and band-limited, so a sensor's power is its in-band power.
    The sources are independent, which leaves a cross term: over eight ship seeds it was
    within 0.1% of the summed power, so 2% is well clear of it.
    """
    colony_path = _stationary_path(2000.0, 60.0)

    def steady(models, paths):
        return _steady_signals(_run(bp, "stft", models, paths))

    colony_alone = steady([_colony(bp)], [colony_path])
    ship, ship_path = _ship_and_path(bp, 120.0)
    ship_alone = steady([ship], [ship_path])
    ship, ship_path = _ship_and_path(bp, 120.0)
    together = steady([_colony(bp), ship], [colony_path, ship_path])

    # Runs agree to complex64 rounding.
    peak = np.max(np.abs(together))
    np.testing.assert_allclose(together, colony_alone + ship_alone, atol=1e-5 * peak)

    def power(signals):
        return np.mean(np.abs(signals) ** 2)

    assert power(colony_alone) > 0.1 * power(ship_alone)
    assert power(together) == pytest.approx(power(colony_alone) + power(ship_alone), rel=0.02)
