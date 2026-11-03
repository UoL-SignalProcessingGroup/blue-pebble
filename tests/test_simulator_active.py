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
    from bluepebble.sigproc import MatchedFilter, parabolic_peak_offset
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
        Facing=Facing,
        LFMSignal=LFMSignal,
        Line=Line,
        MatchedFilter=MatchedFilter,
        Pair=Pair,
        SingleSensorArray=SingleSensorArray,
        SonarPlatform=SonarPlatform,
        Sonobuoy=Sonobuoy,
        SphericalAcousticPropagationModel=SphericalAcousticPropagationModel,
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
