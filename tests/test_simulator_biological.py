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


@pytest.fixture
def bp():
    """Import the real classes without leaking caching into other tests."""
    snapshot = isolate_real_bluepebble_imports()
    import bluepebble.signal as signal_package
    from bluepebble.arrays import RigidArray
    from bluepebble.models.environment import Constant
    from bluepebble.models.propagation import SphericalAcousticPropagationModel
    from bluepebble.platform import SonarPlatform
    from bluepebble.signal import AnthropogenicSignal, BiologicalSignal, SourceSignal
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
        """Seeded real noise in ``BAND_HZ``, standing in for a biological source."""

        def _generate_base_signal(self, source):
            sos = sp_signal.butter(6, BAND_HZ, btype="bandpass", fs=FS, output="sos")
            return sp_signal.sosfiltfilt(sos, self._rng.standard_normal(self.num_samples))

    yield SimpleNamespace(
        AnthropogenicSignal=AnthropogenicSignal,
        BandNoiseSource=BandNoiseSource,
        BiologicalSignal=BiologicalSignal,
        Constant=Constant,
        DelayAndSumBeamformer=DelayAndSumBeamformer,
        Effect=Effect,
        Line=Line,
        Reverb=Reverb,
        SonarPlatform=SonarPlatform,
        SourceSignal=SourceSignal,
        SphericalAcousticPropagationModel=SphericalAcousticPropagationModel,
        SteeringCalculator=SteeringCalculator,
        signal_package=signal_package,
        simulators={
            "stft": ContinuousSTFTPassiveSonarArraySimulator,
            "fractional_delay": ContinuousFractionalDelayPassiveSonarArraySimulator,
            "discrete": DiscretePassiveSonarArraySimulator,
        },
    )
    restore_real_bluepebble_imports(snapshot)


def _timestamps():
    return [START + timedelta(seconds=STEP_S * step) for step in range(NUM_STEPS)]


def _array(bp):
    platform = bp.SonarPlatform(
        states=[GroundTruthState(StateVector([0.0, 0.0, 0.0, 0.0, -50.0, 0.0]), timestamp=START)],
        position_mapping=[0, 2, 4],
        transition_models=[CombinedLinearGaussianTransitionModel([ConstantVelocity(0.0)] * 3)],
        transition_times=[timedelta(hours=1)],
    )
    for timestamp in _timestamps()[1:]:
        platform.move(timestamp)
    array = bp.Line()
    platform.add_sensor(array)
    return array


def _stationary_path(range_m, bearing_deg):
    """Return a source held at ``range_m`` and ``bearing_deg`` from the array's first sensor."""
    bearing = np.deg2rad(bearing_deg)
    state_vector = StateVector(
        [range_m * np.cos(bearing), 0.0, range_m * np.sin(bearing), 0.0, -50.0, 0.0]
    )
    return GroundTruthPath(
        [
            GroundTruthState(state_vector, timestamp=ts, metadata={"position_mapping": [0, 2, 4]})
            for ts in _timestamps()
        ]
    )


def _source(bp, seed=1, **kwargs):
    return bp.BandNoiseSource(
        duration_s=NUM_STEPS * STEP_S, sampling_rate_hz=FS, seed=seed, **kwargs
    )


def _scans(bp, simulator, range_m, bearing_deg, beamform=False):
    """Run one noise-free source through ``simulator``; return its sensor data per scan."""
    beamforming = {}
    if beamform:
        beamforming = {
            "beamformer": bp.DelayAndSumBeamformer(domain="frequency", sampling_rate_hz=FS),
            "steering_calculator": bp.SteeringCalculator(
                ssp=bp.Constant(speed=C), steering_sector_rad=(0.0, np.pi), num_beams=181
            ),
        }
    generator = bp.simulators[simulator](
        array=_array(bp),
        propagation_model=bp.SphericalAcousticPropagationModel(
            ssp=bp.Constant(speed=C), attenuation_factor=0.0
        ),
        signal_models=[_source(bp)],
        ground_truth_paths=[_stationary_path(range_m, bearing_deg)],
        **beamforming,
    ).sensor_data_gen()
    return [next(iter(data)) for _, data in generator]


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

    power = sum(
        np.sum(np.abs(scan.beamformed_data) ** 2, axis=tuple(range(1, scan.beamformed_data.ndim)))
        for scan in scans[3:]
    )
    peak_bearing_deg = np.rad2deg(scans[-1].steering_bearings_rad[np.argmax(power)])
    assert peak_bearing_deg == pytest.approx(true_bearing_deg, abs=2.0)


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
