"""Tests for ActiveSonarDetector, run on ActiveSonarSimulator output."""

from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest
from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel,
    ConstantVelocity,
)
from stonesoup.types.angle import Bearing
from stonesoup.types.array import StateVector
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState

from .support import isolate_real_bluepebble_imports, restore_real_bluepebble_imports

START = datetime(2026, 1, 1)
C = 1500.0
FS = 8000
BANDWIDTH_HZ = 400.0
RANGE_RESOLUTION_M = C / (2 * BANDWIDTH_HZ)


@pytest.fixture
def bp():
    """Import the real classes without leaking caching into other tests."""
    snapshot = isolate_real_bluepebble_imports()
    from bluepebble.arrays import BowArray, RigidArray, SingleSensorArray
    from bluepebble.detector import (
        ActiveSonarDetector,
        CACFARDetector,
        NoiseCalibrator,
        matched_filter_power,
        matched_filter_scans_from_sensor_data,
    )
    from bluepebble.models import ConstantTargetStrength
    from bluepebble.models.environment import Constant
    from bluepebble.models.propagation import SphericalAcousticPropagationModel
    from bluepebble.platform import SonarPlatform
    from bluepebble.signal import LFMSignal
    from bluepebble.signal.random import WhiteNoiseSignal
    from bluepebble.sigproc import DelayAndSumBeamformer, SteeringCalculator
    from bluepebble.simulator import ActiveSonarSimulator

    class Line(RigidArray):
        """Sixteen sensors 0.5 m apart along the platform's x axis."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._local_offsets = np.vstack([np.arange(16) * 0.5, np.zeros(16), np.zeros(16)])
            self._local_axis = np.array([1.0, 0.0, 0.0])

    yield SimpleNamespace(
        ActiveSonarDetector=ActiveSonarDetector,
        ActiveSonarSimulator=ActiveSonarSimulator,
        BowArray=BowArray,
        CACFARDetector=CACFARDetector,
        Constant=Constant,
        ConstantTargetStrength=ConstantTargetStrength,
        DelayAndSumBeamformer=DelayAndSumBeamformer,
        LFMSignal=LFMSignal,
        Line=Line,
        NoiseCalibrator=NoiseCalibrator,
        SingleSensorArray=SingleSensorArray,
        SonarPlatform=SonarPlatform,
        SphericalAcousticPropagationModel=SphericalAcousticPropagationModel,
        SteeringCalculator=SteeringCalculator,
        WhiteNoiseSignal=WhiteNoiseSignal,
        matched_filter_power=matched_filter_power,
        matched_filter_scans_from_sensor_data=matched_filter_scans_from_sensor_data,
    )
    restore_real_bluepebble_imports(snapshot)


def _platform(bp, x=0.0, num_pings=1):
    platform = bp.SonarPlatform(
        states=[GroundTruthState(StateVector([x, 0.0, 0.0, 0.0, -50.0, 0.0]), timestamp=START)],
        position_mapping=[0, 2, 4],
        transition_models=[CombinedLinearGaussianTransitionModel([ConstantVelocity(0.0)] * 3)],
        transition_times=[timedelta(hours=1)],
    )
    for ping in range(1, num_pings):
        platform.move(START + timedelta(seconds=10 * ping))
    return platform


def _mounted(platform, array):
    platform.add_sensor(array)
    return array


def _target(x, y=0.0, z=-50.0):
    return GroundTruthPath(
        [
            GroundTruthState(
                StateVector([x, 0.0, y, 0.0, z, 0.0]),
                timestamp=START,
                metadata={"position_mapping": [0, 2, 4]},
            )
        ]
    )


def _sensor_data(bp, array, targets, noise_upa=1.0, seed=1, **kwargs):
    """Return the simulator's sensor data: an LFM ping, spherical spreading, white noise."""
    signal = bp.LFMSignal(
        freq_min_hz=1000.0 - BANDWIDTH_HZ / 2,
        freq_max_hz=1000.0 + BANDWIDTH_HZ / 2,
        rise_time_s=0.005,
        duration_s=0.05,
        sampling_rate_hz=FS,
        source_level_db=200.0,
    )
    noise = bp.WhiteNoiseSignal(
        amplitude_upa=noise_upa, seed=seed, duration_s=1.0, sampling_rate_hz=FS
    )
    return bp.ActiveSonarSimulator(
        array=array,
        propagation_model=bp.SphericalAcousticPropagationModel(
            ssp=bp.Constant(speed=C), attenuation_factor=0.0
        ),
        signal=signal,
        scattering_models=bp.ConstantTargetStrength(target_strength_db=-40.0),
        ground_truth_paths=targets,
        listen_duration_s=2.0,
        noise_model=noise,
        **kwargs,
    ).sensor_data_gen()


def _cfar(bp, **kwargs):
    settings = dict(
        num_guard_cells=40,
        num_training_cells=200,
        circular=False,
        peak_distance=200,
    )
    settings.update(kwargs)
    if "target_pfa" not in settings:
        settings.setdefault("threshold_factor", 20.0)
    return bp.CACFARDetector(**settings)


def _detections(bp, sensor_data, detector=None, **kwargs):
    reader = bp.ActiveSonarDetector(
        detector=detector or _cfar(bp),
        sensor_data_gen=sensor_data,
        sampling_rate_hz=FS,
        sound_speed_ms=C,
        **kwargs,
    )
    return [detections for _, detections in reader.detections_gen()]


def _strongest(detections):
    return max(detections, key=lambda d: d.metadata["snr_db"])


def test_monostatic_range_is_within_the_resolution(bp) -> None:
    """A single hydrophone's strongest detection lies within c/(2B) of the true range."""
    array = _mounted(_platform(bp), bp.SingleSensorArray())

    (detections,) = _detections(bp, _sensor_data(bp, array, [_target(1000.0)]))

    best = _strongest(detections)
    assert best.state_vector.shape == (1, 1)
    assert float(best.state_vector[0, 0]) == pytest.approx(1000.0, abs=RANGE_RESOLUTION_M)
    assert best.metadata["delay_s"] == pytest.approx(2000.0 / C, abs=RANGE_RESOLUTION_M / C)
    assert best.metadata["bistatic"] is False


def test_beamformed_detection_has_bearing_and_range(bp) -> None:
    """With beams, the strongest detection carries the target's bearing and range."""
    array = _mounted(_platform(bp), bp.Line())
    steering = bp.SteeringCalculator(
        ssp=bp.Constant(speed=C), steering_sector_rad=(0.0, np.pi), num_beams=91
    )
    beamformer = bp.DelayAndSumBeamformer(domain="frequency", sampling_rate_hz=FS)
    true_bearing = np.deg2rad(60.0)
    target = _target(1000.0 * np.cos(true_bearing), 1000.0 * np.sin(true_bearing))

    (detections,) = _detections(
        bp,
        _sensor_data(
            bp, array, [target], beamformer=beamformer, steering_calculator=steering
        ),
    )

    best = _strongest(detections)
    bearing, range_m = best.state_vector[:, 0]
    assert isinstance(bearing, Bearing)
    # A line array cannot tell a bearing from its mirror about the array axis (here x), and
    # the beam width is about lambda / aperture = 1.5 m / 8 m.
    assert abs(float(bearing)) == pytest.approx(true_bearing, abs=0.2)
    assert float(range_m) == pytest.approx(1000.0, abs=RANGE_RESOLUTION_M + 4.0)


@pytest.mark.parametrize("true_bearing_deg", [20.0, -20.0])
def test_cylindrical_bow_sector_puts_an_echo_at_its_bearing_and_range(
    bp, true_bearing_deg
) -> None:
    """A cylindrical sector's beams peak at the target's bearing, to one beam, and its range.

    The peak is read off the matched-filter power, not the detections: each beam's CFAR ratio
    is taken against that beam's own range cells, so it does not rank the beams.
    """
    # 11 staves of 2 sensors: a 0.59 m chord and a 0.5 m row pitch, within lambda/2 at 1.2 kHz.
    array = _mounted(
        _platform(bp),
        bp.BowArray(
            dome_radius_m=3.0,
            azimuth_extent_rad=np.radians(120.0),
            sensor_spacing_m=0.6,
            sensor_size_m=0.3,
            freq_max_hz=1200.0,
            shape="cylindrical",
            height_m=1.0,
        ),
    )
    assert array.num_sensors == 22
    beam_spacing_rad = np.radians(1.0)
    steering = bp.SteeringCalculator(
        ssp=bp.Constant(speed=C), steering_sector_rad=(-np.pi / 3, np.pi / 3), num_beams=121
    )
    beamformer = bp.DelayAndSumBeamformer(domain="frequency", sampling_rate_hz=FS)
    true_bearing = np.radians(true_bearing_deg)
    target = _target(1000.0 * np.cos(true_bearing), 1000.0 * np.sin(true_bearing))

    ((_, (sensor_data,)),) = _sensor_data(
        bp, array, [target], beamformer=beamformer, steering_calculator=steering
    )

    power = bp.matched_filter_power(sensor_data)
    beam, sample = np.unravel_index(np.argmax(power), power.shape)
    assert sensor_data.steering_bearings_rad[beam] == pytest.approx(
        true_bearing, abs=beam_spacing_rad
    )
    assert C / 2 * sample / FS == pytest.approx(1000.0, abs=RANGE_RESOLUTION_M)


def test_bistatic_detection_reports_the_range_sum(bp) -> None:
    """A bistatic echo's range is the sum of the transmitter and receiver ranges."""
    receiver = _mounted(_platform(bp, x=-200.0), bp.SingleSensorArray())
    transmitter = _mounted(_platform(bp), bp.SingleSensorArray())

    (detections,) = _detections(
        bp, _sensor_data(bp, receiver, [_target(1000.0)], transmitter=transmitter)
    )

    best = _strongest(detections)
    assert float(best.state_vector[0, 0]) == pytest.approx(2200.0, abs=2 * RANGE_RESOLUTION_M)
    assert best.metadata["bistatic"] is True
    np.testing.assert_allclose(best.metadata["transmitter_position"], [0.0, 0.0, -50.0])


def test_min_range_drops_close_detections(bp) -> None:
    """Detections closer than min_range_m are dropped."""
    array = _mounted(_platform(bp), bp.SingleSensorArray())

    (detections,) = _detections(
        bp, _sensor_data(bp, array, [_target(1000.0)]), min_range_m=1200.0
    )

    assert all(float(d.state_vector[0, 0]) >= 1200.0 for d in detections)


def test_calibrated_false_alarm_rate(bp) -> None:
    """After calibration on noise-only pings, fresh noise false-alarms near target_pfa.

    Matched-filter noise is correlated over the pulse's resolution cell, so false alarms come
    in clumps and the measured rate scatters by about 25% between runs of this size; the
    factor-of-2 bounds leave room for that.
    """
    array = _mounted(_platform(bp, num_pings=24), bp.SingleSensorArray())
    target_pfa = 1e-3
    detector = _cfar(bp, target_pfa=target_pfa, consolidate_peaks=False)
    calibration_pings = _sensor_data(bp, array, [], seed=11)
    bp.NoiseCalibrator(detector).calibrate_from_noise(
        bp.matched_filter_scans_from_sensor_data(calibration_pings)
    )

    pings = _detections(bp, _sensor_data(bp, array, [], seed=12), detector=detector)

    false_alarm_rate = sum(len(d) for d in pings) / (len(pings) * 2 * FS)
    assert target_pfa / 2 < false_alarm_rate < 2 * target_pfa


def test_delay_is_refined_only_at_a_peak(bp) -> None:
    """A CFAR cell on a slope keeps its own delay; at a peak it moves by under half a sample."""
    from bluepebble.detector.active import _peak_offset

    slope = np.array([1.0, 2.0, 3.01, 4.03, 5.06])
    peak = np.array([1.0, 3.0, 4.0, 3.5, 1.0])

    assert _peak_offset(slope, 2) == 0.0
    assert 0.0 < _peak_offset(peak, 2) < 0.5


def test_misconfigured_inputs_raise(bp) -> None:
    """Wrapping CFAR, several channels without beams, and resized beams are rejected."""
    single = _mounted(_platform(bp), bp.SingleSensorArray())
    with pytest.raises(ValueError, match="circular=False"):
        _detections(bp, _sensor_data(bp, single, []), detector=_cfar(bp, circular=True))

    line = _mounted(_platform(bp), bp.Line())
    with pytest.raises(ValueError, match="beamformer"):
        _detections(bp, _sensor_data(bp, line, []))

    (_, (data,)) = next(iter(_sensor_data(bp, single, [])))
    data.beamformed_data = data.received_waveform[:, :-10]
    with pytest.raises(ValueError, match="domain='frequency'"):
        _detections(bp, [(data.timestamp, {data})])
