"""Tests for ActiveSonarSimulator."""

from __future__ import annotations

from datetime import datetime, timedelta
from shutil import which
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

try:
    import rtrs  # noqa: F401

    HAVE_RTRS = True
except ImportError:
    HAVE_RTRS = False

needs_rtrs = pytest.mark.skipif(not HAVE_RTRS, reason="rtrs not installed")
needs_bellhop = pytest.mark.skipif(which("bellhopcxx") is None, reason="bellhopcxx not on PATH")

START = datetime(2026, 1, 1)
C = 1500.0
FS = 8000


@pytest.fixture
def bp():
    """Import the real classes without leaking caching into other tests."""
    snapshot = isolate_real_bluepebble_imports()
    from bluepebble.arrays import RigidArray, SingleSensorArray
    from bluepebble.models import ConstantTargetStrength
    from bluepebble.models.environment import Constant, FlatBathymetry
    from bluepebble.models.propagation import (
        BellhopAcousticPropagationModel,
        SpectrumPropagationModel,
        SphericalAcousticPropagationModel,
        rtrsAcousticPropagationModel,
    )
    from bluepebble.platform import SonarPlatform, Sonobuoy
    from bluepebble.signal import CWSignal, LFMSignal
    from bluepebble.signal.random import WhiteNoiseSignal
    from bluepebble.sigproc import (
        DelayAndSumBeamformer,
        MatchedFilter,
        SteeringCalculator,
        parabolic_peak_offset,
    )
    from bluepebble.simulator import ActiveSonarSimulator

    class StraightLine(SpectrumPropagationModel):
        """Unit-amplitude straight-line propagation at 1500 m/s, or a fixed delay."""

        def __init__(self, delay_s=None, nan_delay=False):
            self.delay_s = delay_s
            self.nan_delay = nan_delay
            self.calls = []

        def propagate_spectrum(self, platform, source, frequencies_hz):
            source_xyz = np.asarray(
                source.state_vector[source.metadata["position_mapping"]], dtype=float
            )
            distances = np.linalg.norm(platform.array.state_vector - source_xyz, axis=0)
            delays = (
                distances / C if self.delay_s is None else np.full(distances.size, self.delay_s)
            )
            self.calls.append((platform, source))
            if self.nan_delay:
                delays = np.full(distances.size, np.nan)
            f = np.asarray(frequencies_hz, dtype=float)
            transfer = np.exp(-2j * np.pi * f[np.newaxis, :] * np.nan_to_num(delays)[:, None])
            return transfer, delays

    class Pair(RigidArray):
        """Two sensors 0.5 m apart along y, facing -y and +y."""

        def __init__(self, *args, baffle_db=20.0, **kwargs):
            super().__init__(*args, **kwargs)
            self.baffle_attenuation_db = baffle_db
            self._local_offsets = np.array([[0.0, 0.0], [-0.25, 0.25], [0.0, 0.0]])
            self._local_normals = np.array([[0.0, 0.0], [-1.0, 1.0], [0.0, 0.0]])

    class Facing(RigidArray):
        """One sensor facing ``normal``, with a 40 dB baffle."""

        def __init__(self, *args, normal=(-1.0, 0.0, 0.0), **kwargs):
            super().__init__(*args, **kwargs)
            self.baffle_attenuation_db = 40.0
            self._local_offsets = np.zeros((3, 1))
            self._local_normals = np.asarray(normal, dtype=float).reshape(3, 1)

    class Line(RigidArray):
        """Five sensors 2 m apart along x."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._local_offsets = np.vstack([np.arange(5) * 2.0, np.zeros(5), np.zeros(5)])

    yield SimpleNamespace(
        ActiveSonarSimulator=ActiveSonarSimulator,
        BellhopAcousticPropagationModel=BellhopAcousticPropagationModel,
        Constant=Constant,
        FlatBathymetry=FlatBathymetry,
        ConstantTargetStrength=ConstantTargetStrength,
        CWSignal=CWSignal,
        DelayAndSumBeamformer=DelayAndSumBeamformer,
        Facing=Facing,
        LFMSignal=LFMSignal,
        Line=Line,
        MatchedFilter=MatchedFilter,
        Pair=Pair,
        SingleSensorArray=SingleSensorArray,
        SonarPlatform=SonarPlatform,
        Sonobuoy=Sonobuoy,
        SphericalAcousticPropagationModel=SphericalAcousticPropagationModel,
        SteeringCalculator=SteeringCalculator,
        StraightLine=StraightLine,
        WhiteNoiseSignal=WhiteNoiseSignal,
        rtrsAcousticPropagationModel=rtrsAcousticPropagationModel,
        parabolic_peak_offset=parabolic_peak_offset,
    )
    restore_real_bluepebble_imports(snapshot)


def _platform(bp, x=0.0, y=0.0, z=-10.0, vx=0.0):
    return bp.SonarPlatform(
        states=[GroundTruthState(StateVector([x, vx, y, 0.0, z, 0.0]), timestamp=START)],
        position_mapping=[0, 2, 4],
        transition_models=[CombinedLinearGaussianTransitionModel([ConstantVelocity(0.0)] * 3)],
        transition_times=[timedelta(hours=1)],
    )


def _mounted(platform, array):
    platform.add_sensor(array)
    return array


def _target(x, y=0.0, z=-10.0, timestamps=(START,)):
    return GroundTruthPath(
        [
            GroundTruthState(
                StateVector([x, 0.0, y, 0.0, z, 0.0]),
                timestamp=t,
                metadata={"position_mapping": [0, 2, 4]},
            )
            for t in timestamps
        ]
    )


def _cw(bp, level_db=0.0):
    return bp.CWSignal(
        frequency_hz=1000.0, duration_s=0.01, sampling_rate_hz=FS, source_level_db=level_db
    )


def _simulate(bp, array, targets, model=None, signal=None, ts_db=0.0, **kwargs):
    simulator = bp.ActiveSonarSimulator(
        array=array,
        propagation_model=model or bp.StraightLine(),
        signal=signal or _cw(bp),
        scattering_models=bp.ConstantTargetStrength(target_strength_db=ts_db),
        ground_truth_paths=targets,
        listen_duration_s=kwargs.pop("listen_duration_s", 2.0),
        **kwargs,
    )
    return [next(iter(data)) for _, data in simulator.sensor_data_gen()]


def _level_db(data, delay_s, row=0):
    """Return the waveform level 5 ms into an echo arriving at ``delay_s``, in dB."""
    return 20 * np.log10(np.abs(data.received_waveform[row, int(round((delay_s + 0.005) * FS))]))


def _peak_delay_s(bp, data, row=0, fs=FS):
    """Return the matched-filter peak delay of one row, refined to a fraction of a sample."""
    mf = bp.MatchedFilter().filter(data.received_waveform[row], data.transmit_pulse)
    k = int(np.argmax(mf))
    return (k + bp.parabolic_peak_offset(mf, k)) / fs


def test_monostatic_echo_arrives_at_the_round_trip_delay(bp) -> None:
    """A target 1 km away echoes at 2000/1500 s (Tom's monostatic reference)."""
    array = _mounted(_platform(bp), bp.SingleSensorArray())

    (data,) = _simulate(bp, array, [_target(1000.0)])

    assert data.received_waveform.shape == (1, 2 * FS)
    assert _peak_delay_s(bp, data) == pytest.approx(2000 / C, abs=1 / FS)


def test_bistatic_echo_arrives_at_the_range_sum(bp) -> None:
    """A bistatic echo arrives at the range sum (Tom's 2200/1500 s reference)."""
    receiver = _mounted(_platform(bp, x=-200.0), bp.SingleSensorArray())
    transmitter = _mounted(_platform(bp), bp.SingleSensorArray())

    (data,) = _simulate(bp, receiver, [_target(1000.0)], transmitter=transmitter)

    assert _peak_delay_s(bp, data) == pytest.approx(2200 / C, abs=1 / FS)
    np.testing.assert_allclose(data.transmitter_position, [0.0, 0.0, -10.0])
    np.testing.assert_allclose(data.receiver_reference_position, [-200.0, 0.0, -10.0])


def test_sonobuoy_receives_a_hull_array_ping(bp) -> None:
    """A Sonobuoy's hydrophone at 50 m depth hears a ship's ping at the bistatic delay."""
    buoy = bp.Sonobuoy(
        states=GroundTruthState(StateVector([0.0, 800.0, 0.0]), timestamp=START),
        position_mapping=[0, 1, 2],
        hydrophone_depth_m=50.0,
    )
    transmitter = _mounted(_platform(bp), bp.SingleSensorArray())
    target = np.array([1000.0, 0.0, -50.0])

    (data,) = _simulate(
        bp, buoy.hydrophone, [_target(*target)], transmitter=transmitter, listen_duration_s=2.5
    )

    expected_s = (
        np.linalg.norm(target - [0.0, 0.0, -10.0]) + np.linalg.norm(target - [0.0, 800.0, -50.0])
    ) / C
    assert _peak_delay_s(bp, data) == pytest.approx(expected_s, abs=1 / FS)


def test_out_leg_propagates_from_the_transmitter_to_the_target(bp) -> None:
    """The out leg runs from the transmitter to the target, not relying on reciprocity."""
    model = bp.StraightLine()
    array = _mounted(_platform(bp), bp.SingleSensorArray())

    _simulate(bp, array, [_target(1000.0, z=-30.0)], model=model)

    (out_receiver, out_source), (in_receiver, in_source) = model.calls
    np.testing.assert_allclose(out_receiver.array.state_vector[:, 0], [1000.0, 0.0, -30.0])
    np.testing.assert_allclose(np.ravel(out_source.state_vector), [0.0, 0.0, -10.0])
    assert in_receiver.array.num_sensors == 1
    np.testing.assert_allclose(np.ravel(in_source.state_vector)[[0, 2, 4]], [1000, 0, -30])


def test_echo_level_follows_the_sonar_equation(bp) -> None:
    """With spherical spreading, the echo level is SL - 40 log10 R + TS."""
    model = bp.SphericalAcousticPropagationModel(ssp=bp.Constant(speed=C), attenuation_factor=0.0)
    array = _mounted(_platform(bp), bp.SingleSensorArray())

    (data,) = _simulate(
        bp, array, [_target(1000.0)], model=model, signal=_cw(bp, 200.0), ts_db=10.0
    )

    assert _level_db(data, 2000 / C) == pytest.approx(200 - 40 * np.log10(1000.0) + 10, abs=0.1)


def test_baffles_attenuate_on_transmit_and_receive(bp) -> None:
    """A 40 dB baffle facing away costs 80 dB monostatic and 40 dB on transmit only.

    These are Tom's references.
    """
    mono = _mounted(_platform(bp), bp.Facing())
    (data,) = _simulate(bp, mono, [_target(1000.0)])
    mid_echo = int(round((2000 / C + 0.005) * FS))
    assert 20 * np.log10(np.abs(data.received_waveform[0, mid_echo])) == pytest.approx(
        -80, abs=0.1
    )

    receiver = _mounted(_platform(bp, x=-200.0), bp.SingleSensorArray())
    transmitter = _mounted(_platform(bp), bp.Facing())
    (data,) = _simulate(bp, receiver, [_target(1000.0)], transmitter=transmitter)
    mid_echo = int(round((2200 / C + 0.005) * FS))
    assert 20 * np.log10(np.abs(data.received_waveform[0, mid_echo])) == pytest.approx(
        -40, abs=0.1
    )


def test_each_sensor_gets_its_own_baffle_gain(bp) -> None:
    """Sensors facing towards and away get gains 1 and 0.1; transmit gain is their mean.

    With a 20 dB baffle the transmit gain is 0.55.
    """
    array = _mounted(_platform(bp), bp.Pair())

    (data,) = _simulate(bp, array, [_target(0.0, y=1000.0)])

    mid_echo = int(round((2000 / C + 0.005) * FS))
    np.testing.assert_allclose(
        np.abs(data.received_waveform[:, mid_echo]), [0.55 * 0.1, 0.55 * 1.0], rtol=1e-2
    )


def test_lfm_matched_filter_peaks_at_the_round_trip_delay(bp) -> None:
    """A 10 ms one-way delay puts the LFM peak at sample 200 (Tom's rtrs reference)."""
    signal = bp.LFMSignal(
        freq_min_hz=800.0,
        freq_max_hz=1200.0,
        rise_time_s=0.005,
        duration_s=0.05,
        sampling_rate_hz=10_000,
        source_level_db=0.0,
    )
    array = _mounted(_platform(bp), bp.SingleSensorArray())

    (data,) = _simulate(
        bp,
        array,
        [_target(2000.0)],
        model=bp.StraightLine(delay_s=0.01),
        signal=signal,
        listen_duration_s=0.1,
    )

    mf = bp.MatchedFilter().filter(data.received_waveform[0], data.transmit_pulse)
    assert int(np.argmax(mf)) == 200


def test_each_sensor_hears_its_own_delay(bp) -> None:
    """Every sensor of a line array hears the echo at its own out-and-back path length."""
    array = _mounted(_platform(bp), bp.Line())
    target = np.array([300.0, 400.0, -10.0])

    (data,) = _simulate(bp, array, [_target(*target)])

    sensors = np.vstack([np.arange(5) * 2.0, np.zeros(5), np.full(5, -10.0)])
    out_m = np.linalg.norm(target - sensors[:, 0])
    expected_s = (out_m + np.linalg.norm(sensors - target[:, None], axis=0)) / C
    measured_s = [_peak_delay_s(bp, data, row) for row in range(5)]
    # Parabolic refinement of a CW correlation peak is biased by up to ~0.1 sample.
    np.testing.assert_allclose(measured_s, expected_s, atol=0.2 / FS)


def test_late_echo_does_not_wrap_to_the_start(bp) -> None:
    """An echo cut off by the end of the window leaves the start of the window silent."""
    array = _mounted(_platform(bp), bp.SingleSensorArray())
    listen_s = 2000 / C + 0.005  # the echo is half inside the window

    (data,) = _simulate(bp, array, [_target(1000.0)], listen_duration_s=listen_s)

    waveform = np.abs(data.received_waveform[0])
    assert waveform[-10:].min() > 0.9
    assert waveform[: FS // 2].max() < 1e-3


@pytest.mark.parametrize(
    "case",
    ["no_state", "beyond_window", "overhead", "no_path"],
)
def test_unheard_targets_leave_the_waveform_silent(bp, case) -> None:
    """Targets that cannot be heard add nothing to the waveform.

    These are targets with no state at the ping, echoes after the window, targets overhead
    and targets with no propagation path.
    """
    array = _mounted(_platform(bp), bp.SingleSensorArray())
    model = bp.StraightLine(nan_delay=case == "no_path")
    target = {
        "no_state": _target(1000.0, timestamps=(START + timedelta(seconds=1),)),
        "beyond_window": _target(5000.0),
        "overhead": _target(0.5, z=-100.0),
        "no_path": _target(1000.0),
    }[case]

    (data,) = _simulate(bp, array, [target], model=model)

    assert np.all(data.received_waveform == 0)


def test_pings_at_every_platform_timestamp_by_default(bp) -> None:
    """Without ping_timestamps the simulator pings at each recorded platform time."""
    platform = _platform(bp, vx=5.0)
    array = _mounted(platform, bp.SingleSensorArray())
    times = [START + timedelta(seconds=s) for s in (0, 10, 20)]
    for t in times[1:]:
        platform.move(t)
    simulator = bp.ActiveSonarSimulator(
        array=array,
        propagation_model=bp.StraightLine(),
        signal=_cw(bp),
        scattering_models=[bp.ConstantTargetStrength(target_strength_db=0.0)],
        ground_truth_paths=[_target(1000.0, timestamps=times)],
        listen_duration_s=2.0,
    )

    pings = list(simulator.sensor_data_gen())

    assert [t for t, _ in pings] == times
    delays = [_peak_delay_s(bp, next(iter(d))) for _, d in pings]
    np.testing.assert_allclose(delays, [2 * (1000 - 5 * s) / C for s in (0, 10, 20)], atol=1 / FS)


def test_missing_transmitter_state_raises(bp) -> None:
    """A ping time the transmitter's platform has not been moved to is an error."""
    receiver = _mounted(_platform(bp, x=-200.0), bp.SingleSensorArray())
    transmitter = _mounted(_platform(bp), bp.SingleSensorArray())
    later = START + timedelta(seconds=5)
    receiver.movement_controller.move(later)

    with pytest.raises(ValueError, match="transmitter"):
        _simulate(
            bp, receiver, [_target(1000.0)], transmitter=transmitter, ping_timestamps=[later]
        )


def test_seeded_noise_is_repeatable(bp) -> None:
    """The same noise seed gives the same waveform, at the configured RMS."""
    array = _mounted(_platform(bp), bp.SingleSensorArray())

    def run():
        noise = bp.WhiteNoiseSignal(amplitude_upa=0.5, seed=3, duration_s=2.0, sampling_rate_hz=FS)
        return _simulate(bp, array, [], noise_model=noise)[0].received_waveform

    first, second = run(), run()
    np.testing.assert_array_equal(first, second)
    assert np.sqrt(np.mean(np.abs(first) ** 2)) == pytest.approx(0.5, rel=0.05)


def test_settings_are_checked(bp) -> None:
    """A model without spectra and a non-positive listen window are rejected."""
    array = _mounted(_platform(bp), bp.SingleSensorArray())

    with pytest.raises(TypeError, match="SpectrumPropagationModel"):
        _simulate(bp, array, [], model=object())
    with pytest.raises(ValueError, match="listen_duration_s"):
        _simulate(bp, array, [], listen_duration_s=0.0)


@needs_bellhop
def test_bellhop_direct_echo_matches_spherical_spreading(bp) -> None:
    """In deep isovelocity water the Bellhop direct-path echo matches spherical spreading.

    Platform and target are 2 km deep in 4 km of water, so every surface or bottom path
    arrives after the 2 s window and only the direct echo is heard.
    """
    bellhop = bp.BellhopAcousticPropagationModel(
        ssp=bp.Constant(speed=C), bathymetry=bp.FlatBathymetry(depth=-4000.0)
    )
    spherical = bp.SphericalAcousticPropagationModel(
        ssp=bp.Constant(speed=C), attenuation_factor=0.0
    )
    signal = _cw(bp, 200.0)

    def run(model):
        array = _mounted(_platform(bp, z=-2000.0), bp.SingleSensorArray())
        (data,) = _simulate(bp, array, [_target(1000.0, z=-2000.0)], model=model, signal=signal)
        return data

    with_bellhop, with_spherical = run(bellhop), run(spherical)

    assert _peak_delay_s(bp, with_bellhop) == pytest.approx(
        _peak_delay_s(bp, with_spherical), abs=1 / FS
    )
    assert _level_db(with_bellhop, 2000 / C) == pytest.approx(
        _level_db(with_spherical, 2000 / C), abs=1.0
    )


@needs_bellhop
def test_bellhop_sonobuoy_hears_the_bistatic_echo(bp) -> None:
    """A sonobuoy hears a ship's ping through Bellhop at the bistatic path length."""
    bellhop = bp.BellhopAcousticPropagationModel(
        ssp=bp.Constant(speed=C), bathymetry=bp.FlatBathymetry(depth=-4000.0)
    )
    buoy = bp.Sonobuoy(
        states=GroundTruthState(StateVector([0.0, 800.0, 0.0]), timestamp=START),
        position_mapping=[0, 1, 2],
        hydrophone_depth_m=1500.0,
    )
    transmitter = _mounted(_platform(bp, z=-1500.0), bp.SingleSensorArray())
    target = np.array([1000.0, 0.0, -1500.0])

    (data,) = _simulate(
        bp, buoy.hydrophone, [_target(*target)], model=bellhop, transmitter=transmitter
    )

    expected_s = (
        np.linalg.norm(target - [0.0, 0.0, -1500.0])
        + np.linalg.norm(target - [0.0, 800.0, -1500.0])
    ) / C
    assert _peak_delay_s(bp, data) == pytest.approx(expected_s, abs=1 / FS)


@needs_rtrs
def test_rtrs_echo_matches_spherical_spreading(bp) -> None:
    """In deep isovelocity water the rtrs echo matches spherical spreading.

    rtrs needs a fine launch-angle grid here: at its 1 degree default this echo is about
    6.5 dB low and a sample early.
    """
    rtrs = bp.rtrsAcousticPropagationModel(
        ssp=bp.Constant(speed=C),
        bathymetry=bp.FlatBathymetry(depth=-4000.0),
        step_m=5.0,
        elevation_resolution=0.05,
        azimuth_resolution=0.05,
    )
    spherical = bp.SphericalAcousticPropagationModel(
        ssp=bp.Constant(speed=C), attenuation_factor=0.0
    )

    def run(model):
        array = _mounted(_platform(bp, z=-2000.0), bp.SingleSensorArray())
        (data,) = _simulate(
            bp, array, [_target(1000.0, z=-2000.0)], model=model, signal=_cw(bp, 200.0)
        )
        return data

    with_rtrs, with_spherical = run(rtrs), run(spherical)

    assert _peak_delay_s(bp, with_rtrs) == pytest.approx(
        _peak_delay_s(bp, with_spherical), abs=0.1 / FS
    )
    assert _level_db(with_rtrs, 2000 / C) == pytest.approx(
        _level_db(with_spherical, 2000 / C), abs=0.5
    )


# Baseband: the pulse is its complex envelope about a carrier, simulated at a lower rate.
BASEBAND_FS = FS // 8
CARRIER_HZ = 1000.0


def _spherical(bp):
    return bp.SphericalAcousticPropagationModel(ssp=bp.Constant(speed=C), attenuation_factor=0.0)


def _lfm(bp, sampling_rate_hz, baseband, bandwidth_hz=400.0, duration_s=0.05, rise_time_s=0.005):
    """Return an LFM pulse centred on ``CARRIER_HZ``."""
    return bp.LFMSignal(
        freq_min_hz=CARRIER_HZ - bandwidth_hz / 2,
        freq_max_hz=CARRIER_HZ + bandwidth_hz / 2,
        rise_time_s=rise_time_s,
        duration_s=duration_s,
        sampling_rate_hz=sampling_rate_hz,
        source_level_db=200.0,
        baseband=baseband,
    )


def _gaussian_pulse(bp, sampling_rate_hz, baseband):
    """Return a Gaussian-envelope tone at ``CARRIER_HZ``.

    Its spectrum is a Gaussian 40 Hz wide (one standard deviation), so sampling its envelope
    at ``BASEBAND_FS`` aliases nothing, unlike the skirts of an LFM's taper.
    """

    class GaussianPulse(bp.CWSignal):
        def generate(self):
            t = np.arange(self.num_samples) / self.sampling_rate_hz
            width_s = self.duration_s / 16
            return super().generate() * np.exp(-0.5 * ((t - self.duration_s / 2) / width_s) ** 2)

    return GaussianPulse(
        frequency_hz=CARRIER_HZ,
        duration_s=0.064,
        sampling_rate_hz=sampling_rate_hz,
        source_level_db=200.0,
        baseband=baseband,
    )


def _demodulated(waveform, decimation=FS // BASEBAND_FS):
    """Mix a passband waveform down by ``CARRIER_HZ``, low-pass it and decimate it."""
    num_samples = waveform.shape[-1]
    mixed = waveform * np.exp(-2j * np.pi * CARRIER_HZ * np.arange(num_samples) / FS)
    spectrum = np.fft.fft(mixed)
    spectrum[..., np.abs(np.fft.fftfreq(num_samples, 1 / FS)) >= FS / decimation / 2] = 0
    return np.fft.ifft(spectrum)[..., ::decimation]


def _relative_error(actual, expected):
    return np.linalg.norm(actual - expected) / np.linalg.norm(expected)


def _baseband_and_demodulated_passband(bp, make_signal, model, z=-10.0, array=None):
    """Simulate one scene at both rates; return the baseband and demodulated passband data.

    ``band_threshold_db`` is raised so that the band mask, which falls on different bins at
    the two rates, cuts nothing that matters.
    """

    def run(signal):
        mounted = _mounted(_platform(bp, z=z), (array or bp.SingleSensorArray)())
        (data,) = _simulate(
            bp,
            mounted,
            [_target(600.0, y=800.0, z=z)],
            model=model,
            signal=signal,
            band_threshold_db=200.0,
        )
        return data

    baseband = run(make_signal(bp, BASEBAND_FS, True))
    passband = run(make_signal(bp, FS, False))
    return baseband, passband


def test_baseband_waveform_is_the_demodulated_passband_waveform(bp) -> None:
    """A baseband run equals the passband run mixed down by the carrier and decimated.

    Each of five sensors hears the echo at its own delay, so each row checks the carrier
    phase of a different delay.
    """
    baseband, passband = _baseband_and_demodulated_passband(
        bp, _gaussian_pulse, _spherical(bp), array=bp.Line
    )

    assert baseband.received_waveform.shape == (5, 2 * BASEBAND_FS)
    assert _relative_error(
        baseband.received_waveform, _demodulated(passband.received_waveform)
    ) < 1e-6
    assert baseband.demodulation_frequency_hz == CARRIER_HZ
    assert passband.demodulation_frequency_hz == 0.0


def test_baseband_lfm_differs_from_passband_only_by_its_own_aliasing(bp) -> None:
    """An LFM echo misses the demodulated passband echo by the aliasing of its pulse.

    The taper's spectral skirts lie beyond the baseband Nyquist frequency. The passband run's
    low-pass removes them, and sampling the envelope directly folds them back in.
    """
    baseband, passband = _baseband_and_demodulated_passband(bp, _lfm, _spherical(bp))

    def padded(pulse, sampling_rate_hz):
        return np.pad(pulse, (0, 2 * sampling_rate_hz - pulse.size))

    pulse_aliasing = _relative_error(
        padded(baseband.transmit_pulse, BASEBAND_FS),
        _demodulated(padded(passband.transmit_pulse, FS)),
    )
    echo_error = _relative_error(
        baseband.received_waveform, _demodulated(passband.received_waveform)
    )

    assert 1e-3 < pulse_aliasing < 1e-2
    assert echo_error == pytest.approx(pulse_aliasing, rel=0.01)


@needs_bellhop
@pytest.mark.parametrize("volume_attenuation", ["none", "thorp"])
def test_bellhop_baseband_waveform_is_the_demodulated_passband_waveform(
    bp, volume_attenuation
) -> None:
    """The baseband run matches the demodulated passband run through Bellhop too.

    Thorp absorption depends on frequency, so it also checks that each bin is evaluated at
    its physical frequency. The beam fan is narrow so that only the direct path is traced:
    surface and bottom echoes arrive after the FFT length and wrap to a different place at
    each rate.
    """
    bellhop = bp.BellhopAcousticPropagationModel(
        ssp=bp.Constant(speed=C),
        bathymetry=bp.FlatBathymetry(depth=-4000.0),
        beam_angle_limits_deg=(-20.0, 20.0),
        volume_attenuation=volume_attenuation,
    )

    baseband, passband = _baseband_and_demodulated_passband(
        bp, _gaussian_pulse, bellhop, z=-2000.0
    )

    assert np.abs(baseband.received_waveform).max() > 0
    assert _relative_error(
        baseband.received_waveform, _demodulated(passband.received_waveform)
    ) < 1e-6


def _monostatic_baseband_echo(bp, range_m):
    array = _mounted(_platform(bp), bp.SingleSensorArray())
    (data,) = _simulate(
        bp,
        array,
        [_target(range_m)],
        model=_spherical(bp),
        signal=_lfm(bp, BASEBAND_FS, baseband=True),
    )
    return data


def test_baseband_echo_arrives_at_the_round_trip_delay(bp) -> None:
    """A baseband echo's matched-filter peak is at 2R/c, here 0.4 of the way between samples."""
    range_m = 752.55

    data = _monostatic_baseband_echo(bp, range_m)

    assert 2 * range_m / C * BASEBAND_FS == pytest.approx(1003.4)
    assert _peak_delay_s(bp, data, fs=BASEBAND_FS) == pytest.approx(
        2 * range_m / C, abs=0.1 / BASEBAND_FS
    )


def test_baseband_echo_carries_the_carrier_phase_of_its_delay(bp) -> None:
    """A baseband echo is the pulse rotated by exp(-j 2 pi f_c tau), tau its round trip.

    The 1.003 s delay is a whole number of samples and 1003 carrier cycles, so the same test
    at a 1.0034 s delay, 0.4 of a cycle more, shows the phase is the carrier's and not zero.
    """
    for range_m, cycles in ((752.25, 0.0), (752.55, 0.4)):
        data = _monostatic_baseband_echo(bp, range_m)
        start = int(round(2 * range_m / C * BASEBAND_FS - cycles))
        echo = data.received_waveform[0, start : start + data.transmit_pulse.size]

        rotation = np.vdot(data.transmit_pulse, echo)

        assert np.angle(rotation * np.exp(2j * np.pi * cycles)) == pytest.approx(0.0, abs=1e-3)


@pytest.mark.parametrize(
    ("baseband", "f0", "match"),
    [
        (True, 0.0, "must be 1000.0Hz to steer this simulator's baseband data"),
        (True, 900.0, "must be 1000.0Hz"),
        (False, 1000.0, "must be 0.0Hz to steer this simulator's passband data"),
        (True, 1000.0, None),
        (False, 0.0, None),
    ],
)
def test_beamformer_carrier_must_match_the_signal(bp, baseband, f0, match) -> None:
    """A beamformer whose f0 is not what the data is demodulated by is rejected."""
    sampling_rate_hz = BASEBAND_FS if baseband else FS
    array = _mounted(_platform(bp), bp.Line())

    def simulate():
        return _simulate(
            bp,
            array,
            [_target(600.0, y=800.0)],
            signal=_lfm(bp, sampling_rate_hz, baseband),
            beamformer=bp.DelayAndSumBeamformer(
                domain="frequency", sampling_rate_hz=sampling_rate_hz, f0=f0
            ),
            steering_calculator=bp.SteeringCalculator(
                ssp=bp.Constant(speed=C), steering_sector_rad=(0.0, np.pi), num_beams=19
            ),
        )

    if match is None:
        (data,) = simulate()
        assert data.beamformed_data.shape == (19, 2 * sampling_rate_hz)
    else:
        with pytest.raises(ValueError, match=match):
            simulate()


@pytest.mark.parametrize(
    ("rise_fraction", "rate_in_bandwidths", "loss_db"),
    [
        # Sharp edges: -20 log10 sinc(B / 2 fs).
        (0.004, 1.25, 2.42),
        (0.004, 2.0, 0.91),
        (0.004, 20.0, 0.01),
        # A 5% taper at each edge narrows the spectrum.
        (0.05, 1.25, 2.17),
        (0.05, 2.0, 0.82),
    ],
)
def test_matched_filter_straddle_loss_at_low_sampling_rates(
    bp, rise_fraction, rate_in_bandwidths, loss_db
) -> None:
    """An echo half way between samples loses the peak level the simulator's docs give."""
    bandwidth_hz, duration_s = 400.0, 0.5
    sampling_rate_hz = int(rate_in_bandwidths * bandwidth_hz)
    signal = _lfm(
        bp,
        sampling_rate_hz,
        baseband=sampling_rate_hz < CARRIER_HZ,
        duration_s=duration_s,
        rise_time_s=rise_fraction * duration_s,
    )
    array = _mounted(_platform(bp), bp.SingleSensorArray())

    def peak(delay_samples):
        (data,) = _simulate(
            bp,
            array,
            [_target(1000.0)],
            model=bp.StraightLine(delay_s=delay_samples / sampling_rate_hz / 2),
            signal=signal,
            listen_duration_s=1.5,
        )
        return bp.MatchedFilter().filter(data.received_waveform[0], data.transmit_pulse).max()

    on_sample = peak(200.0)
    worst = min(peak(200.0 + fraction) for fraction in np.linspace(0.1, 0.9, 9))

    assert -20 * np.log10(worst / on_sample) == pytest.approx(loss_db, abs=0.04)
    assert -20 * np.log10(peak(200.5) / on_sample) == pytest.approx(loss_db, abs=0.04)


def test_passband_results_are_those_of_v0_8_0(bp) -> None:
    """The default passband waveform is unchanged by the baseband option.

    The values were saved from the code before the option existed (commit 0d00482). They are
    compared to rounding and not bit for bit, since FFT results can differ between platforms.
    """
    array = _mounted(_platform(bp), bp.Line())

    (data,) = _simulate(
        bp,
        array,
        [_target(600.0, y=800.0)],
        model=_spherical(bp),
        signal=_lfm(bp, FS, baseband=False),
    )

    np.testing.assert_allclose(
        data.received_waveform[[0, 2, 4, 4], [10700, 10800, 10900, 11000]],
        [
            -7913.202306404423 + 5541.250245355758j,
            9426.721564040488 - 3408.2653519194164j,
            9112.791424442228 + 4233.043504694662j,
            9080.028049733899 - 4305.990547246536j,
        ],
        rtol=1e-13,
    )
