"""Tests for acoustic simulator entry points."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from .support import (
    install_fake_stonesoup,
    install_fake_stonesoup_simulator_modules,
    install_repo_package,
    load_package_module_from_repo,
)


def _install_fake_acoustic_dependencies(monkeypatch) -> None:
    """Install minimal Blue Pebble dependency modules for importing ``acoustic.py``."""
    propagation_module = ModuleType("bluepebble.models.propagation")
    propagation_module.AcousticPropagationModel = type("AcousticPropagationModel", (), {})
    propagation_module.SpectrumPropagationModel = type("SpectrumPropagationModel", (), {})

    platform_module = ModuleType("bluepebble.platform")
    platform_module.TowedArrayPlatform = type("TowedArrayPlatform", (), {})

    signal_package = ModuleType("bluepebble.signal")
    signal_package.__path__ = []
    ambient_module = ModuleType("bluepebble.signal.ambient")
    ambient_module.AmbientNoise = type("AmbientNoise", (), {})
    signal_base_module = ModuleType("bluepebble.signal.base")
    signal_base_module.Signal = type("Signal", (), {})
    signal_base_module.DiscreteTimestepSignal = type("DiscreteTimestepSignal", (), {})
    signal_base_module.ContinuousTimestepSignal = type("ContinuousTimestepSignal", (), {})
    signal_utils_module = ModuleType("bluepebble.signal.utils")
    signal_utils_module.apply_fade_in = lambda signal, fade_samples: signal
    signal_utils_module.apply_fade_out = lambda signal, fade_samples: signal
    signal_utils_module.inverse_stft = lambda stft, frame_len, hop, window: np.zeros(
        stft.shape[0], dtype=np.complex64
    )
    signal_package.ambient = ambient_module
    signal_package.base = signal_base_module
    signal_package.utils = signal_utils_module

    beamformer_module = ModuleType("bluepebble.sigproc.beamformer")
    beamformer_module.Beamformer = type("Beamformer", (), {})
    beamformer_module.SteeringCalculator = type("SteeringCalculator", (), {})

    anthropogenic_package = ModuleType("bluepebble.signal.anthropogenic")
    anthropogenic_package.__path__ = []
    anthropogenic_base_module = ModuleType("bluepebble.signal.anthropogenic.base")
    anthropogenic_base_module.AnthropogenicSignalBase = type("AnthropogenicSignalBase", (), {})
    signal_package.anthropogenic = anthropogenic_package

    monkeypatch.setitem(sys.modules, "bluepebble.models.propagation", propagation_module)
    monkeypatch.setitem(sys.modules, "bluepebble.platform", platform_module)
    monkeypatch.setitem(sys.modules, "bluepebble.signal", signal_package)
    monkeypatch.setitem(sys.modules, "bluepebble.signal.ambient", ambient_module)
    monkeypatch.setitem(sys.modules, "bluepebble.signal.anthropogenic", anthropogenic_package)
    monkeypatch.setitem(
        sys.modules, "bluepebble.signal.anthropogenic.base", anthropogenic_base_module
    )
    monkeypatch.setitem(sys.modules, "bluepebble.signal.base", signal_base_module)
    monkeypatch.setitem(sys.modules, "bluepebble.signal.utils", signal_utils_module)
    monkeypatch.setitem(sys.modules, "bluepebble.sigproc.beamformer", beamformer_module)


def _spectrum_propagation_base():
    """Return the fake SpectrumPropagationModel base registered for the current test."""
    return sys.modules["bluepebble.models.propagation"].SpectrumPropagationModel


def _load_acoustic_module(monkeypatch):
    """Load simulator modules and expose compatibility aliases used by these tests."""
    install_fake_stonesoup(monkeypatch)
    install_fake_stonesoup_simulator_modules(monkeypatch)
    install_repo_package(monkeypatch, "bluepebble", "bluepebble")
    install_repo_package(monkeypatch, "bluepebble.simulator", "bluepebble/simulator")
    install_repo_package(monkeypatch, "bluepebble.types", "bluepebble/types")
    _install_fake_acoustic_dependencies(monkeypatch)
    discrete = load_package_module_from_repo(
        "bluepebble/simulator/discrete.py",
        "bluepebble.simulator.discrete",
    )
    continuous = load_package_module_from_repo(
        "bluepebble/simulator/continuous.py",
        "bluepebble.simulator.continuous",
    )
    return SimpleNamespace(
        PassiveSonarArraySimulator=discrete.DiscretePassiveSonarArraySimulator,
        BroadbandPassiveSonarArraySimulator=continuous.ContinuousSTFTPassiveSonarArraySimulator,
        FractionalDelayPassiveSonarArraySimulator=continuous.ContinuousFractionalDelayPassiveSonarArraySimulator,
    )


def _install_fake_signal_utils(
    monkeypatch,
    reconstructed_signal: np.ndarray | None = None,
    inverse_stft_fn=None,
) -> None:
    """Install a minimal ``bluepebble.signal.utils`` module for broadband tests."""
    signal_package = ModuleType("bluepebble.signal")
    signal_package.__path__ = []

    utils_module = ModuleType("bluepebble.signal.utils")
    utils_module.apply_fade_in = lambda signal, fade_samples: signal
    utils_module.apply_fade_out = lambda signal, fade_samples: signal
    if inverse_stft_fn is None:
        utils_module.inverse_stft = lambda stft, frame_len, hop, window: np.asarray(
            reconstructed_signal,
            dtype=np.complex64,
        )
    else:
        utils_module.inverse_stft = inverse_stft_fn
    signal_package.utils = utils_module

    monkeypatch.setitem(sys.modules, "bluepebble.signal", signal_package)
    monkeypatch.setitem(sys.modules, "bluepebble.signal.utils", utils_module)

    # continuous.py imports these callables directly; patch bound names when already loaded.
    continuous_module = sys.modules.get("bluepebble.simulator.continuous")
    if continuous_module is not None:
        continuous_module.apply_fade_in = utils_module.apply_fade_in
        continuous_module.apply_fade_out = utils_module.apply_fade_out
        continuous_module.inverse_stft = utils_module.inverse_stft


@dataclass
class FakeState:
    """Simple timestamped state."""

    timestamp: datetime
    state_vector: np.ndarray | None = None


@dataclass
class FakePath:
    """Simple iterable container of states."""

    states: list[FakeState]

    def __iter__(self):
        """Iterate over the contained states."""
        return iter(self.states)


class FakePlatform:
    """Minimal platform exposing simulator-facing methods."""

    def __init__(self, timestamps: list[datetime], num_sensors: int = 2):
        """Store platform timestamps and expose them through the expected API."""
        self.num_sensors = num_sensors
        self.movement_controller = SimpleNamespace(
            states=[FakeState(timestamp=t) for t in timestamps]
        )
        self._platform_states = {t: SimpleNamespace(timestamp=t) for t in timestamps}

    def get_platform_state_at(self, timestamp: datetime):
        """Return the platform state for the requested timestamp."""
        return self._platform_states[timestamp]


class FakeBroadbandSignalModel:
    """Minimal broadband signal model for simulator edge-case tests."""

    sampling_rate_hz = 1000.0
    frame_len = 4
    num_samples = 8

    def compute_stft(self, state):
        """Return a tiny deterministic STFT and metadata."""
        return (
            np.ones((2, 2), dtype=np.complex64),
            np.array([100.0, 200.0]),
            2,
            np.ones(4, dtype=np.float32),
        )

    def stft_geometry(self):
        """Return STFT geometry without a source state."""
        return (2, np.array([100.0, 200.0], dtype=np.float64), 2, np.ones(4, dtype=np.float64), 2)

    def get_source_signal(self):
        """Return a small fixed-length source signal."""
        return np.ones(8, dtype=np.float32)


class TinyBroadbandSignalModel:
    """Broadband signal model with a single STFT frame."""

    sampling_rate_hz = 1000.0
    frame_len = 2

    def compute_stft(self, state):
        """Return a one-frame STFT for missing-target interpolation tests."""
        return (
            np.ones((1, 1), dtype=np.complex64),
            np.array([100.0], dtype=np.float32),
            2,
            np.ones(2, dtype=np.float32),
        )

    def stft_geometry(self):
        """Return STFT geometry without a source state."""
        return (1, np.array([100.0], dtype=np.float64), 2, np.ones(2, dtype=np.float64), 1)

    def get_source_signal(self):
        """Return a minimal source signal."""
        return np.ones(4, dtype=np.float32)


def test_passive_generate_sensor_data_combines_targets_noise_and_beamforming(monkeypatch) -> None:
    """Passive simulation should sum target signals, add noise, and beamform once."""
    acoustic = _load_acoustic_module(monkeypatch)
    timestamp = datetime(2026, 1, 1, 12, 0, 0)
    platform = FakePlatform([timestamp], num_sensors=2)

    class FakePropagationModel(_spectrum_propagation_base()):
        def propagate_spectrum(self, platform_state, target_state, frequencies):
            _ = platform_state, target_state, frequencies
            return np.array(
                [[1.0, 1.0, 1.0], [10.0, 10.0, 10.0]],
                dtype=np.complex128,
            ), 0.0

    class FakeSignalModel:
        sampling_rate_hz = 1.0
        num_samples = 3

        def _generate_base_signal(self, target_state):
            _ = target_state
            return np.array([1.0, 2.0, 3.0], dtype=np.complex128)

        def get_source_waveform(self, source):
            return self._generate_base_signal(source)

    class FakeNoiseModel:
        def generate(self, num_sensors, num_samples=None):
            assert num_sensors == 2
            return np.ones((2, 3), dtype=np.complex128)

    class FakeSteeringCalculator:
        def calculate(self, platform_state):
            return np.array([0.0, 0.05])

    class FakeBeamformer:
        def __init__(self):
            self.calls = []

        def beamform(self, sensor_signals, steering_delays_s):
            self.calls.append((sensor_signals.copy(), steering_delays_s.copy()))
            return sensor_signals.sum(axis=0)

    beamformer = FakeBeamformer()
    target_path = FakePath(states=[FakeState(timestamp=timestamp, state_vector=np.array([1.0]))])
    simulator = acoustic.PassiveSonarArraySimulator(
        platform=platform,
        propagation_model=FakePropagationModel(),
        signal_models=[FakeSignalModel()],
        noise_model=FakeNoiseModel(),
        beamformer=beamformer,
        steering_calculator=FakeSteeringCalculator(),
        ground_truth_paths=[target_path],
    )

    generated = list(simulator.sensor_data_gen())
    assert [generated_timestamp for generated_timestamp, _ in generated] == [timestamp]
    sensor_data = next(iter(generated[0][1]))

    expected_raw = np.array([[2.0, 3.0, 4.0], [11.0, 21.0, 31.0]], dtype=np.complex128)
    np.testing.assert_array_equal(sensor_data.raw_signals, expected_raw)
    np.testing.assert_array_equal(sensor_data.beamformed_data, expected_raw.sum(axis=0))
    assert sensor_data.timestamp == timestamp
    assert len(beamformer.calls) == 1


def test_passive_sensor_data_gen_yields_sorted_unique_timestamps(monkeypatch) -> None:
    """The passive simulator generator should sort and de-duplicate platform timestamps."""
    acoustic = _load_acoustic_module(monkeypatch)
    t0 = datetime(2026, 1, 1, 12, 0, 0)
    t1 = t0 + timedelta(seconds=1)
    platform = FakePlatform([t1, t0, t1], num_sensors=1)

    class FakePropagationModel(_spectrum_propagation_base()):
        def propagate_spectrum(self, platform_state, target_state, frequencies):
            _ = platform_state, target_state, frequencies
            return np.ones((1, 1), dtype=np.complex128), 0.0

    class FakeSignalModel:
        sampling_rate_hz = 1.0
        num_samples = 1

        def _generate_base_signal(self, target_state):
            _ = target_state
            return np.array([1.0], dtype=np.complex128)

        def get_source_waveform(self, source):
            return self._generate_base_signal(source)

    class FakeBeamformer:
        def beamform(self, sensor_signals, steering_delays_s):
            return sensor_signals.copy()

    class FakeSteeringCalculator:
        def calculate(self, platform_state):
            return np.array([0.0])

    path = FakePath(
        states=[
            FakeState(timestamp=t0, state_vector=np.array([0.0])),
            FakeState(timestamp=t1, state_vector=np.array([0.0])),
        ]
    )
    simulator = acoustic.PassiveSonarArraySimulator(
        platform=platform,
        propagation_model=FakePropagationModel(),
        signal_models=[FakeSignalModel()],
        noise_model=None,
        beamformer=FakeBeamformer(),
        steering_calculator=FakeSteeringCalculator(),
        ground_truth_paths=[path],
    )

    generated = list(simulator.sensor_data_gen())

    assert [timestamp for timestamp, _ in generated] == [t0, t1]
    assert all(len(sensor_data_set) == 1 for _, sensor_data_set in generated)


def test_passive_sensor_data_gen_raises_when_platform_has_no_states(monkeypatch) -> None:
    """sensor_data_gen should raise ValueError when the platform has no movement states."""
    install_fake_stonesoup(monkeypatch)
    install_fake_stonesoup_simulator_modules(monkeypatch)
    install_repo_package(monkeypatch, "bluepebble", "bluepebble")
    install_repo_package(monkeypatch, "bluepebble.simulator", "bluepebble/simulator")
    install_repo_package(monkeypatch, "bluepebble.types", "bluepebble/types")
    _install_fake_acoustic_dependencies(monkeypatch)
    discrete = load_package_module_from_repo(
        "bluepebble/simulator/discrete.py",
        "bluepebble.simulator.discrete",
    )
    platform = FakePlatform([], num_sensors=2)

    class FakePropagationModel(_spectrum_propagation_base()):
        def propagate_spectrum(self, platform_state, target_state, frequencies):
            raise AssertionError("should not be reached")

    simulator = discrete.DiscretePassiveSonarArraySimulator(
        platform=platform,
        propagation_model=FakePropagationModel(),
        signal_models=[],
        noise_model=None,
        beamformer=None,
        steering_calculator=None,
        ground_truth_paths=[],
    )

    with pytest.raises(ValueError, match="platform has no movement states"):
        list(simulator.sensor_data_gen())


def test_passive_generate_sensor_data_uses_zero_signal_when_target_absent(monkeypatch) -> None:
    """A timestep with no matching target state should still produce zero-filled sensor data."""
    acoustic = _load_acoustic_module(monkeypatch)
    timestamp = datetime(2026, 1, 1, 12, 0, 0)
    other_timestamp = timestamp + timedelta(seconds=1)
    platform = FakePlatform([timestamp], num_sensors=2)

    class FakeSignalModel:
        sampling_rate_hz = 1.0
        num_samples = 4

        def _generate_base_signal(self, target_state):
            _ = target_state
            return np.array([1.0, 2.0, 3.0, 4.0], dtype=np.complex128)

        def get_source_waveform(self, source):
            return self._generate_base_signal(source)

    class FakeBeamformer:
        def beamform(self, sensor_signals, steering_delays_s):
            return sensor_signals.sum(axis=0)

    class FakeSteeringCalculator:
        def calculate(self, platform_state):
            return np.array([0.0, 0.0])

    class FakePropagationModel(_spectrum_propagation_base()):
        def propagate_spectrum(self, platform_state, target_state, frequencies):
            return np.ones((2, len(frequencies)), dtype=np.complex128), 0.0

    simulator = acoustic.PassiveSonarArraySimulator(
        platform=platform,
        propagation_model=FakePropagationModel(),
        signal_models=[FakeSignalModel()],
        noise_model=None,
        beamformer=FakeBeamformer(),
        steering_calculator=FakeSteeringCalculator(),
        ground_truth_paths=[
            FakePath(states=[FakeState(timestamp=other_timestamp, state_vector=np.array([0.0]))])
        ],
    )

    generated = list(simulator.sensor_data_gen())
    assert [generated_timestamp for generated_timestamp, _ in generated] == [timestamp]
    sensor_data = next(iter(generated[0][1]))

    np.testing.assert_array_equal(sensor_data.raw_signals, np.zeros((2, 4), dtype=np.complex128))
    np.testing.assert_array_equal(sensor_data.beamformed_data, np.zeros(4, dtype=np.complex128))


def test_passive_sensor_data_gen_emits_zero_snapshots_when_no_targets(monkeypatch) -> None:
    """Passive simulation should emit zeroed snapshots when there are no targets."""
    acoustic = _load_acoustic_module(monkeypatch)
    t0 = datetime(2026, 1, 1, 12, 0, 0)
    t1 = t0 + timedelta(seconds=1)
    platform = FakePlatform([t0, t1], num_sensors=1)

    class FakeSignalModel:
        sampling_rate_hz = 1.0
        num_samples = 1

    class FakePropagationModel(_spectrum_propagation_base()):
        def propagate_spectrum(self, platform_state, target_state, frequencies):
            return np.ones((1, len(frequencies)), dtype=np.complex128), 0.0

    simulator = acoustic.PassiveSonarArraySimulator(
        platform=platform,
        propagation_model=FakePropagationModel(),
        signal_models=[FakeSignalModel()],
        noise_model=None,
        beamformer=None,
        steering_calculator=None,
        ground_truth_paths=[],
    )

    generated = list(simulator.sensor_data_gen())

    assert [timestamp for timestamp, _ in generated] == [t0, t1]
    assert all(len(sensor_data_set) == 1 for _, sensor_data_set in generated)
    first_data = next(iter(generated[0][1]))
    second_data = next(iter(generated[1][1]))
    np.testing.assert_array_equal(first_data.raw_signals, np.array([[0.0 + 0.0j]]))
    np.testing.assert_array_equal(second_data.raw_signals, np.array([[0.0 + 0.0j]]))


def test_broadband_requires_at_least_two_timesteps(monkeypatch) -> None:
    """Broadband processing needs at least two timestamps for interpolation."""
    acoustic = _load_acoustic_module(monkeypatch)
    timestamp = datetime(2026, 1, 1, 12, 0, 0)
    platform = FakePlatform([timestamp], num_sensors=1)

    simulator = acoustic.BroadbandPassiveSonarArraySimulator(
        platform=platform,
        propagation_model=SimpleNamespace(),
        signal_models=[SimpleNamespace()],
        noise_model=None,
        beamformer=None,
        steering_calculator=None,
        ground_truth_paths=[],
    )

    with pytest.raises(ValueError, match="Need at least 2 timesteps"):
        list(simulator.sensor_data_gen())


def test_stft_simulator_zero_targets_emits_noise_only(monkeypatch) -> None:
    """STFT simulator with no targets should yield noise-only snapshots without error."""
    acoustic = _load_acoustic_module(monkeypatch)
    t0 = datetime(2026, 1, 1, 12, 0, 0)
    t1 = t0 + timedelta(seconds=1)
    platform = FakePlatform([t0, t1], num_sensors=2)

    class ConstantNoise:
        def generate(self, num_sensors, num_samples=None):
            return np.ones((num_sensors, num_samples or 1), dtype=np.complex64) * 5.0

    simulator = acoustic.BroadbandPassiveSonarArraySimulator(
        platform=platform,
        propagation_model=SimpleNamespace(),
        signal_models=[FakeBroadbandSignalModel()],
        noise_model=ConstantNoise(),
        beamformer=None,
        steering_calculator=None,
        ground_truth_paths=[],
        fade_in_ms=0.0,
    )

    generated = list(simulator.sensor_data_gen())

    assert [ts for ts, _ in generated] == [t0, t1]
    assert all(len(sensor_data_set) == 1 for _, sensor_data_set in generated)
    for _, sensor_data_set in generated:
        data = next(iter(sensor_data_set))
        assert np.all(data.raw_signals == 5.0)


def test_fractional_delay_simulator_zero_targets_emits_noise_only(monkeypatch) -> None:
    """Fractional-delay simulator with no targets should yield noise-only snapshots."""
    acoustic = _load_acoustic_module(monkeypatch)
    t0 = datetime(2026, 1, 1, 12, 0, 0)
    t1 = t0 + timedelta(seconds=1)
    platform = FakePlatform([t0, t1], num_sensors=2)

    class ConstantNoise:
        def generate(self, num_sensors, num_samples=None):
            return np.ones((num_sensors, num_samples or 1), dtype=np.complex64) * 7.0

    simulator = acoustic.FractionalDelayPassiveSonarArraySimulator(
        platform=platform,
        propagation_model=SimpleNamespace(),
        signal_models=[FakeBroadbandSignalModel()],
        noise_model=ConstantNoise(),
        beamformer=None,
        steering_calculator=None,
        ground_truth_paths=[],
        fade_in_ms=0.0,
        fade_out_ms=0.0,
    )

    generated = list(simulator.sensor_data_gen())

    assert [ts for ts, _ in generated] == [t0, t1]
    assert all(len(sensor_data_set) == 1 for _, sensor_data_set in generated)
    for _, sensor_data_set in generated:
        data = next(iter(sensor_data_set))
        assert np.all(data.raw_signals == 7.0)


def test_broadband_rejects_mismatched_signal_model_count(monkeypatch) -> None:
    """Per-target signal models must match the number of targets."""
    acoustic = _load_acoustic_module(monkeypatch)
    t0 = datetime(2026, 1, 1, 12, 0, 0)
    t1 = t0 + timedelta(seconds=1)
    platform = FakePlatform([t0, t1], num_sensors=1)
    target_paths = [
        FakePath(states=[FakeState(timestamp=t0), FakeState(timestamp=t1)]),
        FakePath(states=[FakeState(timestamp=t0), FakeState(timestamp=t1)]),
    ]

    class FakeBroadbandSignalModel:
        sampling_rate_hz = 1000.0
        frame_len = 8

        def compute_stft(self, state):
            return np.ones((2, 3), dtype=np.complex64), np.array([0.0, 0.5, 1.0]), 2, np.ones(8)

        def get_source_signal(self):
            return np.ones(8, dtype=np.float32)

    simulator = acoustic.BroadbandPassiveSonarArraySimulator(
        platform=platform,
        propagation_model=SimpleNamespace(),
        signal_models=[FakeBroadbandSignalModel()],
        noise_model=None,
        beamformer=None,
        steering_calculator=None,
        ground_truth_paths=target_paths,
    )

    # Single-model lists are explicitly supported, so force the mismatch case with >1 models.
    simulator.signal_models = [
        FakeBroadbandSignalModel(),
        FakeBroadbandSignalModel(),
        FakeBroadbandSignalModel(),
    ]

    with pytest.raises(ValueError, match="Number of signal models"):
        list(simulator.sensor_data_gen())


def test_broadband_truncates_long_noise(monkeypatch) -> None:
    """Noise longer than a timestep slice should be truncated to the snapshot length."""
    acoustic = _load_acoustic_module(monkeypatch)
    _install_fake_signal_utils(monkeypatch, reconstructed_signal=np.array([1, 2, 3, 4, 5]))
    t0 = datetime(2026, 1, 1, 12, 0, 0)
    t1 = t0 + timedelta(seconds=1)
    platform = FakePlatform([t0, t1], num_sensors=1)
    path = FakePath(states=[FakeState(timestamp=t0), FakeState(timestamp=t1)])

    class FakePropagationModel(_spectrum_propagation_base()):
        def propagate_spectrum(self, platform_state, target_state, frequencies):
            return np.ones((1, len(frequencies)), dtype=np.complex64), 0.0

        def compute_sensor_delays(self, platform_state, target_state):
            return np.array([0.0], dtype=float)

    class LongNoiseModel:
        def __init__(self):
            self.seen_num_samples = []

        def generate(self, num_sensors, num_samples=None):
            self.seen_num_samples.append(num_samples)
            return np.full((num_sensors, (num_samples or 1) + 1), 10.0, dtype=np.complex64)

    noise_model = LongNoiseModel()
    simulator = acoustic.BroadbandPassiveSonarArraySimulator(
        platform=platform,
        propagation_model=FakePropagationModel(),
        signal_models=[FakeBroadbandSignalModel()],
        noise_model=noise_model,
        beamformer=None,
        steering_calculator=None,
        ground_truth_paths=[path],
        fade_in_ms=0.0,
    )

    generated = list(simulator.sensor_data_gen())

    assert noise_model.seen_num_samples == [2, 3]
    first_data = next(iter(generated[0][1]))
    second_data = next(iter(generated[1][1]))
    np.testing.assert_array_equal(
        first_data.raw_signals,
        np.array([[11.0, 12.0]], dtype=np.complex64),
    )
    np.testing.assert_array_equal(
        second_data.raw_signals,
        np.array([[13.0, 14.0, 15.0]], dtype=np.complex64),
    )
    assert first_data.beamformed_data is None
    assert second_data.beamformed_data is None


def test_broadband_pads_short_noise_and_beamforms_real_part(monkeypatch) -> None:
    """Short noise should be zero-padded and beamforming should see real-valued signals."""
    acoustic = _load_acoustic_module(monkeypatch)
    _install_fake_signal_utils(monkeypatch, reconstructed_signal=np.array([1, 2, 3, 4, 5]))
    t0 = datetime(2026, 1, 1, 12, 0, 0)
    t1 = t0 + timedelta(seconds=1)
    platform = FakePlatform([t0, t1], num_sensors=1)
    path = FakePath(states=[FakeState(timestamp=t0), FakeState(timestamp=t1)])

    class FakePropagationModel(_spectrum_propagation_base()):
        def propagate_spectrum(self, platform_state, target_state, frequencies):
            return np.ones((1, len(frequencies)), dtype=np.complex64), 0.0

        def compute_sensor_delays(self, platform_state, target_state):
            return np.array([0.0], dtype=float)

    class ShortNoiseModel:
        def generate(self, num_sensors, num_samples=None):
            short_samples = max((num_samples or 1) - 1, 1)
            return np.full((num_sensors, short_samples), 10.0, dtype=np.complex64)

    class FakeSteeringCalculator:
        def __init__(self):
            self.calls = []

        def calculate(self, platform_state):
            self.calls.append(platform_state.timestamp)
            return np.array([0.0])

    class RecordingBeamformer:
        def __init__(self):
            self.calls = []

        def beamform(self, sensor_signals, steering_delays_s):
            self.calls.append((sensor_signals.copy(), steering_delays_s.copy()))
            return sensor_signals.copy()

    beamformer = RecordingBeamformer()
    steering_calculator = FakeSteeringCalculator()
    simulator = acoustic.BroadbandPassiveSonarArraySimulator(
        platform=platform,
        propagation_model=FakePropagationModel(),
        signal_models=[FakeBroadbandSignalModel()],
        noise_model=ShortNoiseModel(),
        beamformer=beamformer,
        steering_calculator=steering_calculator,
        ground_truth_paths=[path],
        fade_in_ms=0.0,
    )

    generated = list(simulator.sensor_data_gen())

    assert steering_calculator.calls == [t0, t1]
    assert len(beamformer.calls) == 2
    first_signals, first_delays = beamformer.calls[0]
    second_signals, second_delays = beamformer.calls[1]
    np.testing.assert_array_equal(first_delays, np.array([0.0]))
    np.testing.assert_array_equal(second_delays, np.array([0.0]))
    np.testing.assert_array_equal(first_signals, np.array([[11.0 + 0.0j, 2.0 + 0.0j]]))
    np.testing.assert_array_equal(
        second_signals,
        np.array([[13.0 + 0.0j, 14.0 + 0.0j, 5.0 + 0.0j]]),
    )
    first_data = next(iter(generated[0][1]))
    second_data = next(iter(generated[1][1]))
    np.testing.assert_array_equal(first_data.beamformed_data, first_signals)
    np.testing.assert_array_equal(second_data.beamformed_data, second_signals)


def test_broadband_sums_multiple_targets_with_shared_signal_model(monkeypatch) -> None:
    """A single signal model should be replicated and summed across multiple targets."""
    acoustic = _load_acoustic_module(monkeypatch)
    _install_fake_signal_utils(
        monkeypatch,
        inverse_stft_fn=lambda stft, frame_len, hop, window: np.asarray(
            np.sum(stft, axis=1),
            dtype=np.complex64,
        ),
    )
    t0 = datetime(2026, 1, 1, 12, 0, 0)
    t1 = t0 + timedelta(seconds=1)
    platform = FakePlatform([t0, t1], num_sensors=1)
    path_one = FakePath(
        states=[
            FakeState(timestamp=t0, state_vector=np.array([1.0])),
            FakeState(timestamp=t1, state_vector=np.array([1.0])),
        ]
    )
    path_two = FakePath(
        states=[
            FakeState(timestamp=t0, state_vector=np.array([2.0])),
            FakeState(timestamp=t1, state_vector=np.array([2.0])),
        ]
    )

    class FakePropagationModel(_spectrum_propagation_base()):
        def propagate_spectrum(self, platform_state, target_state, frequencies):
            gain = float(target_state.state_vector[0])
            return np.full((1, len(frequencies)), gain, dtype=np.complex64), 0.0

        def compute_sensor_delays(self, platform_state, target_state):
            return np.array([0.0], dtype=float)

    simulator = acoustic.BroadbandPassiveSonarArraySimulator(
        platform=platform,
        propagation_model=FakePropagationModel(),
        signal_models=[FakeBroadbandSignalModel()],
        noise_model=None,
        beamformer=None,
        steering_calculator=None,
        ground_truth_paths=[path_one, path_two],
        fade_in_ms=0.0,
    )

    generated = list(simulator.sensor_data_gen())

    first_data = next(iter(generated[0][1]))
    second_data = next(iter(generated[1][1]))
    np.testing.assert_array_equal(first_data.raw_signals, np.array([[6.0]], dtype=np.complex64))
    np.testing.assert_array_equal(second_data.raw_signals, np.array([[6.0]], dtype=np.complex64))


def test_broadband_handles_target_missing_at_a_timestep(monkeypatch) -> None:
    """A target absent at one timestep should not break broadband interpolation."""
    acoustic = _load_acoustic_module(monkeypatch)
    _install_fake_signal_utils(
        monkeypatch,
        inverse_stft_fn=lambda stft, frame_len, hop, window: np.array(
            [stft[0, 0], 0.0],
            dtype=np.complex64,
        ),
    )
    t0 = datetime(2026, 1, 1, 12, 0, 0)
    t1 = t0 + timedelta(seconds=1)
    platform = FakePlatform([t0, t1], num_sensors=1)
    path = FakePath(states=[FakeState(timestamp=t0, state_vector=np.array([1.0]))])

    class FakePropagationModel(_spectrum_propagation_base()):
        def propagate_spectrum(self, platform_state, target_state, frequencies):
            return np.ones((1, len(frequencies)), dtype=np.complex64), 0.0

        def compute_sensor_delays(self, platform_state, target_state):
            return np.array([0.0], dtype=float)

    simulator = acoustic.BroadbandPassiveSonarArraySimulator(
        platform=platform,
        propagation_model=FakePropagationModel(),
        signal_models=[TinyBroadbandSignalModel()],
        noise_model=None,
        beamformer=None,
        steering_calculator=None,
        ground_truth_paths=[path],
        fade_in_ms=0.0,
    )

    generated = list(simulator.sensor_data_gen())

    assert [timestamp for timestamp, _ in generated] == [t0, t1]
    assert all(len(sensor_data_set) == 1 for _, sensor_data_set in generated)
    first_data = next(iter(generated[0][1]))
    second_data = next(iter(generated[1][1]))
    assert first_data.raw_signals.shape == (1, 1)
    assert second_data.raw_signals.shape == (1, 1)
    assert np.isfinite(first_data.raw_signals).all()
    assert np.isfinite(second_data.raw_signals).all()


def test_broadband_beamformer_receives_sensors_in_native_array_order(monkeypatch) -> None:
    """Broadband reconstruction should preserve simulator sensor ordering for beamforming."""
    acoustic = _load_acoustic_module(monkeypatch)
    _install_fake_signal_utils(
        monkeypatch,
        inverse_stft_fn=lambda stft, frame_len, hop, window: np.array(
            [stft[0, 0], stft[0, 0]],
            dtype=np.complex64,
        ),
    )
    t0 = datetime(2026, 1, 1, 12, 0, 0)
    t1 = t0 + timedelta(seconds=1)
    platform = FakePlatform([t0, t1], num_sensors=2)
    path = FakePath(states=[FakeState(timestamp=t0), FakeState(timestamp=t1)])

    class FakePropagationModel(_spectrum_propagation_base()):
        def propagate_spectrum(self, platform_state, target_state, frequencies):
            return np.array([[1.0], [2.0]], dtype=np.complex64), 0.0

        def compute_sensor_delays(self, platform_state, target_state):
            return np.array([0.0, 0.0], dtype=float)

    class FakeSteeringCalculator:
        def calculate(self, platform_state):
            return np.array([0.0, 0.1])

    class RecordingBeamformer:
        def __init__(self):
            self.calls = []

        def beamform(self, sensor_signals, steering_delays_s):
            self.calls.append((sensor_signals.copy(), steering_delays_s.copy()))
            return sensor_signals.sum(axis=0)

    beamformer = RecordingBeamformer()
    simulator = acoustic.BroadbandPassiveSonarArraySimulator(
        platform=platform,
        propagation_model=FakePropagationModel(),
        signal_models=[TinyBroadbandSignalModel()],
        noise_model=None,
        beamformer=beamformer,
        steering_calculator=FakeSteeringCalculator(),
        ground_truth_paths=[path],
        fade_in_ms=0.0,
    )

    generated = list(simulator.sensor_data_gen())

    assert len(beamformer.calls) == 2
    first_signals, first_delays = beamformer.calls[0]
    np.testing.assert_array_equal(first_delays, np.array([0.0, 0.1]))
    np.testing.assert_array_equal(first_signals, np.array([[1.0 + 0.0j], [2.0 + 0.0j]]))
    first_data = next(iter(generated[0][1]))
    np.testing.assert_array_equal(first_data.raw_signals, first_signals)
