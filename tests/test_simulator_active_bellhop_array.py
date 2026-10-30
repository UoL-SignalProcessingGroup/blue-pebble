"""Tests for BellhopActiveSonarSimulatorArray's bistatic transmitter and hull baffling."""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest
from stonesoup.types.array import StateVector, StateVectors
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState

from .support import isolate_real_bluepebble_imports, restore_real_bluepebble_imports

T0 = datetime(2024, 1, 1)
C = 1500.0
FS = 8000.0


@pytest.fixture(scope="module")
def bp():
    """Import the simulator without leaking real caching into other tests."""
    snapshot = isolate_real_bluepebble_imports()
    from bluepebble.sensors.base import ArrayState
    from bluepebble.signal.active import CWSignal
    from bluepebble.simulator.active_bellhop import (
        BellhopActiveSonarSimulatorArray,
        BellhopActiveSonarSimulatorArrayPerElement,
    )

    yield SimpleNamespace(
        ArrayState=ArrayState,
        CWSignal=CWSignal,
        Simulator=BellhopActiveSonarSimulatorArray,
        PerElementSimulator=BellhopActiveSonarSimulatorArrayPerElement,
    )
    restore_real_bluepebble_imports(snapshot)


class _StraightLineModel:
    """One unit-amplitude eigenray per call, delayed by horizontal distance / C."""

    def compute_eigenrays(self, src_depth, rcv_depth, src_xy, rcv_xy, freq_hz):
        distance = float(np.hypot(rcv_xy[0] - src_xy[0], rcv_xy[1] - src_xy[1]))
        return [SimpleNamespace(amplitude=1.0, delay_s=distance / C)]


def _sensor(bp, ref_xy, normals=None):
    """Return a one-element fake sensor at ``ref_xy`` (z = -10 m), optionally with a normal."""
    ref = StateVector([ref_xy[0], ref_xy[1], -10.0])
    array = bp.ArrayState(
        num_sensors=1,
        state_vector=StateVectors(np.asarray(ref)),
        ref_state_vector=ref,
        element_normals=None if normals is None else StateVectors(np.array(normals).T),
        baffle_attenuation_db=40.0,
    )
    return SimpleNamespace(
        NOMINAL_SOUND_SPEED=C, get_platform_state_at=lambda ts: SimpleNamespace(array=array)
    )


def _run(bp, receiver, transmitter=None, target_xy=(1000.0, 0.0), per_element=False):
    target = GroundTruthPath([GroundTruthState(
        StateVector([target_xy[0], 0.0, target_xy[1], 0.0, -10.0, 0.0]), timestamp=T0
    )])
    extra = {} if per_element else {"transmitter": transmitter}
    simulator = (bp.PerElementSimulator if per_element else bp.Simulator)(
        platform=receiver,
        **extra,
        propagation_model=_StraightLineModel(),
        signal=bp.CWSignal(
            frequency_hz=1000.0, duration_s=0.01, sampling_rate_hz=FS, source_level_db=0.0
        ),
        ground_truth_paths=[target],
        target_strength_db=0.0,
        ping_timestamps=[T0],
        target_position_mapping=[0, 2, 4],
        receive_duration_s=2.0,
        amplitude_cutoff=0.0,
    )
    _, data = next(simulator.sensor_data_gen())
    return next(iter(data)).received_waveform[0]


def _first_arrival_s(waveform: np.ndarray) -> float:
    return np.flatnonzero(np.abs(waveform) > 0)[0] / FS


def test_monostatic_delay_is_round_trip(bp) -> None:
    """Without a transmitter the echo takes twice the one-way delay."""
    waveform = _run(bp, _sensor(bp, (0.0, 0.0)))

    assert _first_arrival_s(waveform) == pytest.approx(2000.0 / C, abs=1 / FS)


def test_bistatic_delay_uses_transmitter_out_path(bp) -> None:
    """With a transmitter the out-path starts there and only the back-path ends at the rx."""
    waveform = _run(bp, _sensor(bp, (-200.0, 0.0)), transmitter=_sensor(bp, (0.0, 0.0)))

    assert _first_arrival_s(waveform) == pytest.approx((1000.0 + 1200.0) / C, abs=1 / FS)


def _ratio_db(quiet: np.ndarray, loud: np.ndarray) -> float:
    return float(20.0 * np.log10(np.abs(quiet).max() / np.abs(loud).max()))


def test_monostatic_baffle_attenuates_both_legs(bp) -> None:
    """A monostatic array facing away from the target loses the baffle on transmit and receive."""
    facing = _run(bp, _sensor(bp, (0.0, 0.0), normals=[[1.0, 0.0, 0.0]]))
    behind = _run(bp, _sensor(bp, (0.0, 0.0), normals=[[-1.0, 0.0, 0.0]]))

    assert _ratio_db(behind, facing) == pytest.approx(-80.0, abs=1e-6)


def test_per_element_simulator_baffles_both_legs(bp) -> None:
    """The per-element simulator applies the same two-leg baffle as the shared-eigenray one."""
    facing = _run(bp, _sensor(bp, (0.0, 0.0), normals=[[1.0, 0.0, 0.0]]), per_element=True)
    behind = _run(bp, _sensor(bp, (0.0, 0.0), normals=[[-1.0, 0.0, 0.0]]), per_element=True)

    assert _ratio_db(behind, facing) == pytest.approx(-80.0, abs=1e-6)


def test_transmitter_facing_away_attenuates_out_leg_only(bp) -> None:
    """A bistatic transmitter facing away from the target pings it at the baffle level."""
    receiver = _sensor(bp, (-200.0, 0.0), normals=[[1.0, 0.0, 0.0]])
    facing = _run(bp, receiver, transmitter=_sensor(bp, (0.0, 0.0), normals=[[1.0, 0.0, 0.0]]))
    behind = _run(bp, receiver, transmitter=_sensor(bp, (0.0, 0.0), normals=[[-1.0, 0.0, 0.0]]))

    assert _ratio_db(behind, facing) == pytest.approx(-40.0, abs=1e-6)


def test_transmit_gain_is_mean_of_element_gains(bp) -> None:
    """A two-sided array transmits from its facing half only; no normals means omni."""
    array = bp.ArrayState(
        num_sensors=2,
        state_vector=StateVectors(np.zeros((3, 2))),
        ref_state_vector=StateVector([0.0, 0.0, 0.0]),
        element_normals=StateVectors(np.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]).T),
        baffle_attenuation_db=20.0,
    )
    omni = bp.ArrayState(
        num_sensors=2,
        state_vector=StateVectors(np.zeros((3, 2))),
        ref_state_vector=StateVector([0.0, 0.0, 0.0]),
    )

    np.testing.assert_allclose(array.element_gains([0.0, 5.0, 0.0]), [1.0, 0.1])
    assert array.transmit_gain([0.0, 5.0, 0.0]) == pytest.approx(0.55)
    assert omni.transmit_gain([0.0, -5.0, 0.0]) == 1.0
