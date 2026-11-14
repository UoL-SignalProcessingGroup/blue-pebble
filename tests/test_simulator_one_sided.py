"""Tests that a one-sided source stays one-sided, delayed and at its level in every simulator.

The channel is spherical spreading with no absorption, ``H(f) = exp(-2j * pi * f * R / c) / R``,
so the oracle at a sensor ``R`` away is the source waveform delayed by ``R / c`` and divided by
``R``. The source sits a whole number of samples (and of STFT hops) from the first sensor, which
makes that delay exact in all three simulators.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest
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
NUM_SENSORS = 32
SPACING_M = 0.75
FRAME_LEN = 1024
HOP = FRAME_LEN // 4
SIMULATORS = ["stft", "fractional_delay", "discrete"]
BAND_HZ = (300.0, 700.0)
AMPLITUDE_UPA = 50.0
# Three hops: 768 samples, or 288 m at 1500 m/s.
DELAY_SAMPLES = 3 * HOP
RANGE_M = DELAY_SAMPLES / FS * C


@pytest.fixture
def bp():
    """Import the real classes without leaking caching into other tests."""
    snapshot = isolate_real_bluepebble_imports()
    from bluepebble.arrays import RigidArray
    from bluepebble.models.environment import Constant
    from bluepebble.models.propagation import SphericalAcousticPropagationModel
    from bluepebble.platform import SonarPlatform
    from bluepebble.signal import SyntheticAnthropogenicSignal, WhiteNoiseSignal
    from bluepebble.sigproc import (
        DelayAndSumBeamformer,
        MinimumVarianceDistortionlessResponseBeamformer,
        SteeringCalculator,
    )
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

    yield SimpleNamespace(
        Constant=Constant,
        DelayAndSumBeamformer=DelayAndSumBeamformer,
        Line=Line,
        MinimumVarianceDistortionlessResponseBeamformer=(
            MinimumVarianceDistortionlessResponseBeamformer
        ),
        SonarPlatform=SonarPlatform,
        SphericalAcousticPropagationModel=SphericalAcousticPropagationModel,
        SteeringCalculator=SteeringCalculator,
        SyntheticAnthropogenicSignal=SyntheticAnthropogenicSignal,
        WhiteNoiseSignal=WhiteNoiseSignal,
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


def _noise_ship_path(range_m, bearing_deg):
    """Return a ship of white noise in ``BAND_HZ``, held at a range and bearing of sensor 0."""
    bearing = np.deg2rad(bearing_deg)
    state_vector = StateVector(
        [range_m * np.cos(bearing), 0.0, range_m * np.sin(bearing), 0.0, -50.0, 0.0]
    )
    empty = np.array([])
    metadata = {
        "position_mapping": [0, 2, 4],
        "amplitudes_upa": empty,
        "frequencies_hz": empty,
        "phases_rad": empty,
        "tonal_bandwidth_hz": 1.0,
        "noise_amplitude_upa": AMPLITUDE_UPA,
        "noise_spectral_exponent": 0.0,
    }
    return GroundTruthPath(
        [GroundTruthState(state_vector, timestamp=ts, metadata=metadata) for ts in _timestamps()]
    )


def _run(bp, simulator, range_m=RANGE_M, bearing_deg=90.0, beamformer=None, ambient_upa=None):
    """Run the noise ship; return the source waveform and the scans.

    There is no ambient noise unless ``ambient_upa`` gives its amplitude.
    """
    ship = bp.SyntheticAnthropogenicSignal(
        duration_s=NUM_STEPS * STEP_S,
        sampling_rate_hz=FS,
        frame_len=FRAME_LEN,
        noise_freq_range_hz=BAND_HZ,
        seed=1,
    )
    options = {}
    if simulator != "discrete":
        options = {"fade_in_ms": 0.0, "fade_out_ms": 0.0}
    if ambient_upa is not None:
        options["noise_model"] = bp.WhiteNoiseSignal(
            duration_s=STEP_S, sampling_rate_hz=FS, amplitude_upa=ambient_upa, seed=2
        )
    if beamformer is not None:
        options["beamformer"] = beamformer
        options["steering_calculator"] = bp.SteeringCalculator(
            ssp=bp.Constant(speed=C), steering_sector_rad=(0.0, np.pi), num_beams=181
        )
    generator = bp.simulators[simulator](
        array=_array(bp),
        propagation_model=bp.SphericalAcousticPropagationModel(
            ssp=bp.Constant(speed=C), attenuation_factor=0.0
        ),
        signal_models=[ship],
        ground_truth_paths=[_noise_ship_path(range_m, bearing_deg)],
        **options,
    ).sensor_data_gen()
    scans = [next(iter(data)) for _, data in generator]
    return ship.get_source_signal(), scans


def _record_lag(simulator):
    """Return the sample of a simulator's record at which the source's first sample arrives.

    ``stft`` drops the first hop of its overlap-add, so its record starts a hop late.
    """
    return DELAY_SAMPLES - (HOP if simulator == "stft" else 0)


def _steady(simulator, received_len):
    """Return the samples of a simulator's record that hold the fully rendered source.

    - ``stft``: its first and last frames are only partly overlapped.
    - ``fractional_delay``: it renders nothing after its last timestamp, so its final scan
      is silent. That is how it behaved before the sources became one-sided.
    - ``discrete``: every sample, since each scan holds its whole stretch of the source.
    """
    if simulator == "discrete":
        return slice(None)
    lag = _record_lag(simulator)
    if simulator == "stft":
        return slice(lag + FRAME_LEN, received_len - FRAME_LEN)
    return slice(lag, received_len - int(STEP_S * FS))


def _expected_at_first_sensor(simulator, source, received_len):
    """Return what sensor 0 should hold: the source delayed by ``R / c`` and divided by ``R``.

    ``discrete`` transforms each scan on its own, so its delay is circular within the scan.
    """
    if simulator == "discrete":
        chunks = source.reshape(NUM_STEPS, -1)
        return np.roll(chunks, DELAY_SAMPLES, axis=1).ravel() / RANGE_M
    delayed = np.concatenate([np.zeros(_record_lag(simulator)), source])
    return delayed[:received_len] / RANGE_M


@pytest.mark.parametrize("simulator", SIMULATORS)
def test_first_sensor_holds_the_source_delayed_and_spread(bp, simulator):
    """Sensor 0 holds ``x(t - R / c) / R``, in amplitude and in phase.

    The error was measured at 3e-14 of the power or below in all three simulators, the level
    of single-precision rounding.
    """
    source, scans = _run(bp, simulator)
    received = np.concatenate([scan.raw_signals[0] for scan in scans])

    expected = _expected_at_first_sensor(simulator, source, received.size)
    steady = _steady(simulator, received.size)

    error = np.sum(np.abs(received[steady] - expected[steady]) ** 2)
    assert error < 1e-10 * np.sum(np.abs(expected[steady]) ** 2)


@pytest.mark.parametrize("simulator", SIMULATORS)
def test_power_at_a_sensor_is_the_source_power_over_range_squared(bp, simulator):
    """A source of RMS amplitude ``A`` gives a sensor ``R`` away a power of ``(A / R) ** 2``.

    The ship's power is all in ``BAND_HZ`` at positive frequencies, so this is its in-band
    power. The continuous simulators' records leave out the source's last ``R / c`` seconds
    and the discrete one's holds all of it, so the first two agree with ``(A / R) ** 2`` to
    the scatter of that stretch of noise (under 1% here) and the third to rounding.
    """
    _, scans = _run(bp, simulator)
    received = np.concatenate([scan.raw_signals[0] for scan in scans])

    power = np.mean(np.abs(received[_steady(simulator, received.size)]) ** 2)

    tolerance = 1e-5 if simulator == "discrete" else 0.05
    assert power == pytest.approx((AMPLITUDE_UPA / RANGE_M) ** 2, rel=tolerance)


def _negative_bin_fraction(waveform, taper=False):
    """Return the share of a waveform's power in the bins ``fftfreq`` labels negative."""
    if taper:
        waveform = waveform * np.hanning(waveform.size)
    power = np.abs(np.fft.fft(waveform)) ** 2
    return power[np.fft.fftfreq(waveform.size) < 0].sum() / power.sum()


@pytest.mark.parametrize("simulator", ["stft", "fractional_delay"])
def test_continuous_sensor_data_has_no_power_at_negative_frequencies(bp, simulator):
    """With no noise model, what reaches the sensors is as one-sided as the source.

    The record is not periodic, so it is tapered before the transform; a rectangular
    transform leaks about 5e-5 of the power into the negative bins on its own. The share was
    measured at 4e-15 or below for every sensor, the level of single-precision rounding.
    """
    _, scans = _run(bp, simulator, bearing_deg=60.0)
    received = np.concatenate([scan.raw_signals for scan in scans], axis=1)
    arrived = received[:, _steady(simulator, received.shape[1])]

    fractions = [_negative_bin_fraction(sensor, taper=True) for sensor in arrived]

    assert max(fractions) < 1e-12


def test_discrete_sensor_data_keeps_each_scan_spectrum(bp):
    """The discrete simulator leaves each scan's power spectrum as the source's, over R squared.

    It transforms each scan alone, and a scan cut from the one-sided source already has
    between 6e-5 and 5e-4 of its power in the negative bins from the cut. A pure delay only
    turns each bin's phase, so the sensor's spectrum must be the source scan's, bin for bin.
    """
    source, scans = _run(bp, "discrete", bearing_deg=60.0)

    for scan, source_scan in zip(scans, source.reshape(NUM_STEPS, -1), strict=True):
        source_power = np.abs(np.fft.fft(source_scan)) ** 2
        assert _negative_bin_fraction(source_scan) > 1e-5
        np.testing.assert_allclose(
            np.abs(np.fft.fft(scan.raw_signals[0])) ** 2,
            source_power / RANGE_M**2,
            atol=1e-5 * source_power.max() / RANGE_M**2,
        )


@pytest.mark.parametrize("kind", ["delay_and_sum", "mvdr"])
def test_one_sided_source_peaks_at_its_bearing_with_the_default_band(bp, kind):
    """STFT beam power over the default band, 0 Hz and up, peaks at the source's bearing.

    The half-power beam width is about lambda / aperture = 3 m / 23 m, or 7 degrees, at the
    band centre, so 2 degrees is well inside the main lobe.
    """
    shared = dict(sampling_rate_hz=FS, nfft=256, overlap=128)
    if kind == "mvdr":
        beamformer = bp.MinimumVarianceDistortionlessResponseBeamformer(**shared)
    else:
        beamformer = bp.DelayAndSumBeamformer(domain="broadband_power", **shared)

    # MVDR cannot factorise the covariance of a silent scan, so the sensors get ambient
    # noise 20 dB below the source's level at them.
    _, scans = _run(
        bp,
        "stft",
        range_m=2000.0,
        bearing_deg=60.0,
        beamformer=beamformer,
        ambient_upa=0.1 * AMPLITUDE_UPA / 2000.0,
    )

    power = sum(scan.beamformed_data.sum(axis=1) for scan in scans[3:])
    bearings_deg = np.rad2deg(scans[-1].steering_bearings_rad)
    assert bearings_deg[np.argmax(power)] == pytest.approx(60.0, abs=2.0)
