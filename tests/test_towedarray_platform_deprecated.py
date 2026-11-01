"""Tests for the deprecated TowedArrayPlatform and the simulator's deprecated platform argument."""

from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest
from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel,
    ConstantVelocity,
    KnownTurnRate,
)
from stonesoup.types.groundtruth import GroundTruthState

from .support import isolate_real_bluepebble_imports, restore_real_bluepebble_imports

START = datetime(2026, 1, 1)
ARRAY = dict(
    num_sensors=6,
    cable_length_m=80.0,
    sensor_spacing_m=2.0,
    array_depth_m=40.0,
    reference_sensor_idx=2,
)


@pytest.fixture
def bp():
    """Import the real platform, array and simulator classes without leaking caching."""
    snapshot = isolate_real_bluepebble_imports()
    from bluepebble.arrays import TowedArray
    from bluepebble.platform import SonarPlatform, TowedArrayPlatform
    from bluepebble.simulator import DiscretePassiveSonarArraySimulator

    yield SimpleNamespace(
        SonarPlatform=SonarPlatform,
        TowedArrayPlatform=TowedArrayPlatform,
        TowedArray=TowedArray,
        Simulator=DiscretePassiveSonarArraySimulator,
    )
    restore_real_bluepebble_imports(snapshot)


def _platform_kwargs() -> dict:
    straight = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.0)] * 3)
    turn = CombinedLinearGaussianTransitionModel(
        [KnownTurnRate([0.0, 0.0], np.radians(4.0)), ConstantVelocity(0.0)]
    )
    return dict(
        states=GroundTruthState([0.0, 4.0, 0.0, 0.0, -5.0, 0.0], timestamp=START),
        position_mapping=[0, 2, 4],
        velocity_mapping=[1, 3, 5],
        transition_models=[straight, turn],
        transition_times=[timedelta(seconds=10), timedelta(seconds=30)],
    )


def _old_platform(bp):
    with pytest.warns(DeprecationWarning, match="TowedArray"):
        return bp.TowedArrayPlatform(**_platform_kwargs(), **ARRAY)


def test_matches_sonar_platform_with_mounted_towed_array(bp) -> None:
    """The wrapper should reproduce a SonarPlatform + TowedArray exactly, through a turn."""
    old = _old_platform(bp)
    platform = bp.SonarPlatform(**_platform_kwargs())
    new = bp.TowedArray(**ARRAY)
    platform.add_sensor(new)

    timestamps = [START + timedelta(seconds=k) for k in range(1, 31)]
    for timestamp in timestamps:
        old.move(timestamp)
        platform.move(timestamp)

    for timestamp in timestamps:
        old_state = old.get_platform_state_at(timestamp)
        new_state = new.get_platform_state_at(timestamp)
        assert old_state is not None and new_state is not None
        np.testing.assert_array_equal(old_state.array.state_vector, new_state.array.state_vector)
        np.testing.assert_array_equal(
            old_state.array.ref_state_vector, new_state.array.ref_state_vector
        )
        assert old_state.host.heading_rad == new_state.host.heading_rad
    np.testing.assert_array_equal(old.host_path, new.host_path)
    assert len(old.sensor_paths) == len(new.sensor_paths) == ARRAY["num_sensors"]


def test_forwards_the_old_array_queries(bp) -> None:
    """The wrapper still answers the queries TowedArrayPlatform used to."""
    platform = _old_platform(bp)
    platform.move(START + timedelta(seconds=1))

    assert platform.arrays == (platform.towed_array,)
    assert platform.array.num_sensors == ARRAY["num_sensors"]
    assert platform.get_host_state_at(START) is platform.states[0]
    assert len(platform.get_sensor_states_at(START)) == ARRAY["num_sensors"]
    assert len(platform.towed_sensors) == ARRAY["num_sensors"]
    assert "TowedArrayPlatform(num_sensors=6" in repr(platform)


def test_simulator_platform_argument_is_deprecated_and_resolves_to_the_array(bp) -> None:
    """Passing the old combined platform as ``platform=`` simulates its towed array."""
    platform = _old_platform(bp)

    with pytest.warns(DeprecationWarning, match="array="):
        simulator = bp.Simulator(
            platform=platform, propagation_model=object(), signal_models=[object()]
        )

    assert simulator.array is platform.towed_array


def test_simulator_requires_an_array(bp) -> None:
    """Without array= (or the deprecated platform=) the simulator has nothing to simulate."""
    simulator = bp.Simulator(propagation_model=object(), signal_models=[object()])

    with pytest.raises(TypeError, match="array="):
        simulator._sorted_timestamps()
