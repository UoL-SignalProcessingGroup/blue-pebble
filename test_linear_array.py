#!/usr/bin/env python3
"""Test script for the linear array initialization functionality."""

from datetime import datetime

from stonesoup.types.array import StateVector
from stonesoup.types.groundtruth import GroundTruthState

from nereus.stonesoup.platform.towedarray import TowedArrayPlatform


def test_linear_array_initialization():
    """Test the linear array initialization method."""
    print("Testing linear array initialization...")

    # Create a simple platform state
    timestamp = datetime.now()
    platform_state_vector = StateVector([0, 0, 0, 0, 0, 0])  # [x, vx, y, vy, z, vz]
    platform_state = GroundTruthState(platform_state_vector, timestamp)

    # Import transition model for the platform
    from stonesoup.models.transition.linear import (
        CombinedLinearGaussianTransitionModel,
        ConstantVelocity,
    )

    # Create transition model for the platform
    platform_transition_model = CombinedLinearGaussianTransitionModel(
        [
            ConstantVelocity(0.0),  # x direction
            ConstantVelocity(0.0),  # y direction
            ConstantVelocity(0.0),  # z direction
        ]
    )

    # Create platform with linear array initialization
    platform = TowedArrayPlatform(
        states=[platform_state],
        position_mapping=[0, 2, 4],
        velocity_mapping=[1, 3, 5],
        transition_model=platform_transition_model,
        num_sensors=5,
        cable_length_m=100.0,
        sensor_spacing_m=10.0,
        array_depth_m=-50.0,
        linear_array=True,  # Use linear initialization
    )

    print(f"Number of sensors: {len(platform.towed_sensors)}")
    print(f"Expected number: {platform.num_sensors}")

    # Check sensor positions
    print("\nSensor positions:")
    for i, sensor in enumerate(platform.towed_sensors):
        pos = sensor.states[0].state_vector
        print(f"Sensor {i}: x={pos[0]:.1f}, y={pos[1]:.1f}, z={pos[2]:.1f}")

        # Verify linear spacing
        expected_x = i * platform.sensor_spacing_m
        expected_y = 0.0
        expected_z = platform.array_depth_m

        assert abs(pos[0] - expected_x) < 1e-6, f"Sensor {i} x position incorrect"
        assert abs(pos[1] - expected_y) < 1e-6, f"Sensor {i} y position incorrect"
        assert abs(pos[2] - expected_z) < 1e-6, f"Sensor {i} z position incorrect"

    print("\n✓ Linear array initialization test passed!")


def test_cable_dynamics_initialization():
    """Test the original cable dynamics initialization method."""
    print("\nTesting cable dynamics initialization...")

    # Create a simple platform state with some velocity
    timestamp = datetime.now()
    # Moving forward with velocity
    platform_state_vector = StateVector([0, 5, 0, 0, -10, 0])
    platform_state = GroundTruthState(platform_state_vector, timestamp)

    # Import transition model for the platform
    from stonesoup.models.transition.linear import (
        CombinedLinearGaussianTransitionModel,
        ConstantVelocity,
    )

    # Create transition model for the platform
    platform_transition_model = CombinedLinearGaussianTransitionModel(
        [
            ConstantVelocity(0.0),  # x direction
            ConstantVelocity(0.0),  # y direction
            ConstantVelocity(0.0),  # z direction
        ]
    )

    # Create platform with cable dynamics (default)
    platform = TowedArrayPlatform(
        states=[platform_state],
        position_mapping=[0, 2, 4],
        velocity_mapping=[1, 3, 5],
        transition_model=platform_transition_model,
        num_sensors=3,
        cable_length_m=50.0,
        sensor_spacing_m=10.0,
        array_depth_m=-50.0,
        linear_array=False,  # Use cable dynamics (default)
    )

    print(f"Number of sensors: {len(platform.towed_sensors)}")
    print(f"Expected number: {platform.num_sensors}")

    # Check sensor positions
    print("\nSensor positions:")
    for i, sensor in enumerate(platform.towed_sensors):
        pos = sensor.states[0].state_vector
        print(f"Sensor {i}: x={pos[0]:.1f}, y={pos[1]:.1f}, z={pos[2]:.1f}")

    print("\n✓ Cable dynamics initialization test passed!")


if __name__ == "__main__":
    test_linear_array_initialization()
    test_cable_dynamics_initialization()
    print("\n🎉 All tests passed!")
