"""Focused tests for simulator module helpers and edge-case branches."""

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


def _install_fake_simulator_dependencies(monkeypatch) -> None:
    """Install lightweight dependency modules required by simulator imports."""
    propagation_module = ModuleType("bluepebble.models.propagation")
    propagation_module.AcousticPropagationModel = type("AcousticPropagationModel", (), {})

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

    monkeypatch.setitem(sys.modules, "bluepebble.models.propagation", propagation_module)
    monkeypatch.setitem(sys.modules, "bluepebble.platform", platform_module)
    monkeypatch.setitem(sys.modules, "bluepebble.signal", signal_package)
    monkeypatch.setitem(sys.modules, "bluepebble.signal.ambient", ambient_module)
    monkeypatch.setitem(sys.modules, "bluepebble.signal.base", signal_base_module)
    monkeypatch.setitem(sys.modules, "bluepebble.signal.utils", signal_utils_module)
    monkeypatch.setitem(sys.modules, "bluepebble.sigproc.beamformer", beamformer_module)


def _load_simulator_modules(monkeypatch):
    """Load simulator modules with fake dependency scaffolding."""
    install_fake_stonesoup(monkeypatch)
    install_fake_stonesoup_simulator_modules(monkeypatch)
    install_repo_package(monkeypatch, "bluepebble", "bluepebble")
    install_repo_package(monkeypatch, "bluepebble.simulator", "bluepebble/simulator")
    _install_fake_simulator_dependencies(monkeypatch)
    base = load_package_module_from_repo(
        "bluepebble/simulator/base.py",
        "bluepebble.simulator.base",
    )
    discrete = load_package_module_from_repo(
        "bluepebble/simulator/discrete.py",
        "bluepebble.simulator.discrete",
    )
    continuous = load_package_module_from_repo(
        "bluepebble/simulator/continuous.py",
        "bluepebble.simulator.continuous",
    )
    return base, discrete, continuous


@dataclass
class _FakeState:
    timestamp: datetime
    state_vector: np.ndarray | None = None


@dataclass
class _FakePath:
    states: list[_FakeState]

    def __iter__(self):
        return iter(self.states)


class _FakePlatform:
    def __init__(self, timestamps: list[datetime], num_sensors: int):
        self.num_sensors = num_sensors
        self.movement_controller = SimpleNamespace(states=[_FakeState(t) for t in timestamps])
        self._states = {t: SimpleNamespace(timestamp=t) for t in timestamps}

    def get_platform_state_at(self, timestamp: datetime):
        return self._states[timestamp]


def test_base_resolve_models_and_target_lookup(monkeypatch) -> None:
    """Base helpers should resolve model lists and find timestamped target states."""
    _base, discrete, _continuous = _load_simulator_modules(monkeypatch)

    timestamp = datetime(2026, 1, 1, 12, 0, 0)
    target_path = _FakePath(states=[_FakeState(timestamp)])
    found_state = discrete.DiscretePassiveSonarArraySimulator._target_state_at(
        target_path, timestamp
    )
    assert found_state.timestamp == timestamp
    assert (
        discrete.DiscretePassiveSonarArraySimulator._target_state_at(
            target_path, timestamp + timedelta(seconds=1)
        )
        is None
    )

    with pytest.raises(ValueError, match="must contain at least one model"):
        discrete.DiscretePassiveSonarArraySimulator._resolve_models([], 1, "models")

    assert discrete.DiscretePassiveSonarArraySimulator._resolve_models([1], 1, "models") == [1]
    assert (
        discrete.DiscretePassiveSonarArraySimulator._resolve_models([1], 3, "models")
        == [1, 1, 1]
    )

    with pytest.raises(ValueError, match="must match number of targets"):
        discrete.DiscretePassiveSonarArraySimulator._resolve_models([1, 2], 3, "models")


def test_base_generate_noise_handles_shapes_and_duration_restore(monkeypatch) -> None:
    """Noise generation should truncate/pad outputs and restore temporary duration overrides."""
    _base, discrete, _continuous = _load_simulator_modules(monkeypatch)
    simulator = discrete.DiscretePassiveSonarArraySimulator()

    assert simulator._generate_noise(num_sensors=1, num_samples=2) is None

    class LongNoise:
        def __init__(self):
            self.duration_s = 9.0
            self.seen = []

        def generate(self, num_sensors):
            self.seen.append(self.duration_s)
            return np.ones((num_sensors, 5), dtype=np.complex64)

    long_noise = LongNoise()
    simulator.noise_model = long_noise
    truncated = simulator._generate_noise(num_sensors=2, num_samples=3, sampling_rate_hz=1000.0)
    assert long_noise.duration_s == pytest.approx(9.0)
    assert long_noise.seen == [pytest.approx(0.003)]
    assert truncated.shape == (2, 3)

    class ShortNoise:
        def generate(self, num_sensors):
            return np.full((num_sensors, 1), 5.0, dtype=np.complex64)

    simulator.noise_model = ShortNoise()
    padded = simulator._generate_noise(num_sensors=2, num_samples=3, sampling_rate_hz=1000.0)
    np.testing.assert_array_equal(
        padded,
        np.array([[5.0, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=np.complex64),
    )

    class FailingNoise:
        def __init__(self):
            self.duration_s = 4.0

        def generate(self, num_sensors):
            _ = num_sensors
            raise RuntimeError("boom")

    failing_noise = FailingNoise()
    simulator.noise_model = failing_noise
    with pytest.raises(RuntimeError, match="boom"):
        simulator._generate_noise(num_sensors=1, num_samples=2, sampling_rate_hz=1000.0)
    assert failing_noise.duration_s == pytest.approx(4.0)


def test_base_beamform_if_configured_and_make_sensor_data(monkeypatch) -> None:
    """Beamforming helper should no-op without dependencies and build sensor payloads when set."""
    _base, discrete, _continuous = _load_simulator_modules(monkeypatch)

    timestamp = datetime(2026, 1, 1, 12, 0, 0)
    simulator = discrete.DiscretePassiveSonarArraySimulator(
        platform=_FakePlatform([timestamp], num_sensors=1)
    )

    sensor_signals = np.array([[1.0 + 0.0j]], dtype=np.complex64)
    assert simulator._beamform_if_configured(timestamp, sensor_signals) is None

    class Steering:
        def __init__(self):
            self.seen = None

        def calculate(self, platform_state):
            self.seen = platform_state
            return np.array([0.1])

    class Beamformer:
        def __init__(self):
            self.calls = []

        def beamform(self, signals, delays):
            self.calls.append((signals.copy(), delays.copy()))
            return np.array([42.0 + 0.0j], dtype=np.complex64)

    steering = Steering()
    beamformer = Beamformer()
    simulator.steering_calculator = steering
    simulator.beamformer = beamformer

    beamformed = simulator._beamform_if_configured(timestamp, sensor_signals)
    np.testing.assert_array_equal(beamformed, np.array([42.0 + 0.0j], dtype=np.complex64))
    assert steering.seen.timestamp == timestamp
    np.testing.assert_array_equal(beamformer.calls[0][0], sensor_signals)
    np.testing.assert_array_equal(beamformer.calls[0][1], np.array([0.1]))

    payload = simulator._make_sensor_data(timestamp, sensor_signals, beamformed)
    np.testing.assert_array_equal(payload.raw_signals, sensor_signals)
    np.testing.assert_array_equal(payload.beamformed_data, beamformed)
    assert payload.timestamp == timestamp


def test_discrete_source_signal_resolution_and_validation_errors(monkeypatch) -> None:
    """Discrete simulator source-resolution helper should raise clear errors on invalid inputs."""
    _base, discrete, _continuous = _load_simulator_modules(monkeypatch)

    class RuntimeSignalModel:
        def __init__(self):
            self.calls = 0
            self.stft_calls = 0

        def get_source_signal(self):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("not ready")
            return np.array([1.0, 2.0], dtype=np.float32)

        def compute_stft(self, state):
            _ = state
            self.stft_calls += 1

    with pytest.raises(ValueError, match="empty target path"):
        discrete.DiscretePassiveSonarArraySimulator._get_broadband_source_signal(
            RuntimeSignalModel(),
            first_state=None,
        )

    runtime_model = RuntimeSignalModel()
    signal = discrete.DiscretePassiveSonarArraySimulator._get_broadband_source_signal(
        runtime_model,
        first_state=SimpleNamespace(),
    )
    assert runtime_model.stft_calls == 1
    np.testing.assert_array_equal(signal, np.array([1.0 + 0.0j, 2.0 + 0.0j]))

    class BaseSignalModel:
        def _generate_base_signal(self, state):
            _ = state
            return np.array([3.0, 4.0], dtype=np.float32)

    with pytest.raises(ValueError, match="empty target path"):
        discrete.DiscretePassiveSonarArraySimulator._get_broadband_source_signal(
            BaseSignalModel(),
            first_state=None,
        )

    with pytest.raises(TypeError, match="must implement either"):
        discrete.DiscretePassiveSonarArraySimulator._get_broadband_source_signal(
            object(),
            first_state=SimpleNamespace(),
        )

    assert (
        discrete.DiscretePassiveSonarArraySimulator._get_target_first_state(_FakePath(states=[]))
        is None
    )


def test_discrete_sensor_data_gen_validates_inputs_and_clamps_empty_chunks(monkeypatch) -> None:
    """Discrete simulator should validate required interfaces and clamp empty chunk boundaries."""
    _base, discrete, _continuous = _load_simulator_modules(monkeypatch)
    timestamp = datetime(2026, 1, 1, 12, 0, 0)

    simulator = discrete.DiscretePassiveSonarArraySimulator(
        platform=_FakePlatform([timestamp], num_sensors=1),
        propagation_model=SimpleNamespace(),
        signal_models=[SimpleNamespace(sampling_rate_hz=1.0, num_samples=1)],
        ground_truth_paths=[],
    )
    with pytest.raises(AttributeError, match="propagate_spectrum"):
        list(simulator.sensor_data_gen())

    simulator = discrete.DiscretePassiveSonarArraySimulator(
        platform=_FakePlatform([timestamp], num_sensors=1),
        propagation_model=SimpleNamespace(
            propagate_spectrum=lambda platform_state, target_state, frequencies: (
                np.ones((1, len(frequencies)), dtype=np.complex64),
                0.0,
            )
        ),
        signal_models=[SimpleNamespace(sampling_rate_hz=1.0, num_samples=0)],
        ground_truth_paths=[],
    )
    with pytest.raises(ValueError, match="num_samples must be greater than zero"):
        list(simulator.sensor_data_gen())

    t1 = timestamp + timedelta(seconds=1)
    path = _FakePath(states=[_FakeState(timestamp), _FakeState(t1)])

    class TinySignal:
        sampling_rate_hz = 0.2
        num_samples = 1

        def _generate_base_signal(self, state):
            _ = state
            return np.array([2.0 + 0.0j], dtype=np.complex64)

    simulator = discrete.DiscretePassiveSonarArraySimulator(
        platform=_FakePlatform([timestamp, t1], num_sensors=1),
        propagation_model=SimpleNamespace(
            propagate_spectrum=lambda platform_state, target_state, frequencies: (
                np.ones((1, len(frequencies)), dtype=np.complex64),
                0.0,
            )
        ),
        signal_models=[TinySignal()],
        ground_truth_paths=[path],
    )

    generated = list(simulator.sensor_data_gen())
    assert [ts for ts, _ in generated] == [timestamp, t1]
    assert all(next(iter(data)).raw_signals.shape[1] == 1 for _, data in generated)


def test_discrete_sensor_data_gen_validates_model_consistency(monkeypatch) -> None:
    """Discrete simulator should reject inconsistent per-target model metadata."""
    _base, discrete, _continuous = _load_simulator_modules(monkeypatch)
    timestamp = datetime(2026, 1, 1, 12, 0, 0)
    path_a = _FakePath(states=[_FakeState(timestamp)])
    path_b = _FakePath(states=[_FakeState(timestamp)])

    propagation = SimpleNamespace(
        propagate_spectrum=lambda platform_state, target_state, frequencies: (
            np.ones((1, len(frequencies)), dtype=np.complex64),
            0.0,
        )
    )

    class SignalA:
        sampling_rate_hz = 10.0
        num_samples = 4

        def _generate_base_signal(self, state):
            _ = state
            return np.ones(4, dtype=np.complex64)

    class SignalBadSamples(SignalA):
        num_samples = 3

    class SignalBadRate(SignalA):
        sampling_rate_hz = 20.0

    simulator = discrete.DiscretePassiveSonarArraySimulator(
        platform=_FakePlatform([timestamp], num_sensors=1),
        propagation_model=propagation,
        signal_models=[SignalA(), SignalBadSamples()],
        ground_truth_paths=[path_a, path_b],
    )
    with pytest.raises(ValueError, match="same num_samples"):
        list(simulator.sensor_data_gen())

    simulator = discrete.DiscretePassiveSonarArraySimulator(
        platform=_FakePlatform([timestamp], num_sensors=1),
        propagation_model=propagation,
        signal_models=[SignalA(), SignalBadRate()],
        ground_truth_paths=[path_a, path_b],
    )
    with pytest.raises(ValueError, match="same sampling_rate_hz"):
        list(simulator.sensor_data_gen())

    simulator = discrete.DiscretePassiveSonarArraySimulator(
        platform=_FakePlatform([timestamp], num_sensors=1),
        propagation_model=propagation,
        signal_models=[SignalA()],
        ground_truth_paths=[],
    )
    monkeypatch.setattr(simulator, "_resolve_signal_models", lambda num_targets: [])
    with pytest.raises(ValueError, match="must contain at least one model"):
        list(simulator.sensor_data_gen())


def test_discrete_sensor_data_gen_pads_truncates_and_skips_absent_targets(monkeypatch) -> None:
    """Discrete simulator should pad/truncate source waveforms and ignore missing target states."""
    _base, discrete, _continuous = _load_simulator_modules(monkeypatch)
    t0 = datetime(2026, 1, 1, 12, 0, 0)
    t1 = t0 + timedelta(seconds=1)
    platform = _FakePlatform([t0, t1], num_sensors=1)

    path_short = _FakePath(states=[_FakeState(t0), _FakeState(t1)])
    # Missing t1 to exercise the "target_state is None" branch during generation.
    path_long = _FakePath(states=[_FakeState(t0)])

    class ShortSignal:
        sampling_rate_hz = 4.0
        num_samples = 4

        def _generate_base_signal(self, state):
            _ = state
            return np.array([1.0, 0.0], dtype=np.complex64)

    class LongSignal(ShortSignal):
        def _generate_base_signal(self, state):
            _ = state
            return np.array([2.0, 0.0, 0.0, 0.0, 9.0], dtype=np.complex64)

    simulator = discrete.DiscretePassiveSonarArraySimulator(
        platform=platform,
        propagation_model=SimpleNamespace(
            propagate_spectrum=lambda platform_state, target_state, frequencies: (
                np.ones((1, len(frequencies)), dtype=np.complex64),
                0.0,
            )
        ),
        signal_models=[ShortSignal(), LongSignal()],
        ground_truth_paths=[path_short, path_long],
    )
    generated = list(simulator.sensor_data_gen())
    assert [ts for ts, _ in generated] == [t0, t1]
    first = next(iter(generated[0][1]))
    second = next(iter(generated[1][1]))
    assert first.raw_signals.shape == (1, 4)
    assert second.raw_signals.shape == (1, 1)
    assert np.isfinite(first.raw_signals).all()
    assert np.isfinite(second.raw_signals).all()


def test_continuous_helpers_cover_validation_and_interpolation_branches(monkeypatch) -> None:
    """Continuous helpers should validate modes and support scalar/vector interpolation paths."""
    _base, _discrete, continuous = _load_simulator_modules(monkeypatch)

    simulator = continuous.ContinuousSTFTPassiveSonarArraySimulator(mode="invalid")
    with pytest.raises(ValueError, match="Unsupported mode"):
        simulator._validate_mode()

    assert continuous.ContinuousSTFTPassiveSonarArraySimulator._interp_index_alpha(
        1.0, np.array([0.0])
    ) == (0, 0.0)
    assert continuous.ContinuousSTFTPassiveSonarArraySimulator._interp_index_alpha(
        12.0, np.array([0.0, 10.0])
    ) == (0, 1.0)

    with pytest.raises(ValueError, match="Invalid STFT bin count"):
        continuous.ContinuousSTFTPassiveSonarArraySimulator._ifft_frame(
            np.ones(2, dtype=np.complex64),
            frame_len=4,
            num_freq_bins=2,
        )

    H_mag_hist = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    H_phase_hist = np.zeros_like(H_mag_hist)
    tau_hist = np.array([0.0, 0.0], dtype=np.float64)
    freqs = np.array([10.0, 20.0], dtype=np.float64)

    scalar_interp = simulator._interpolate_channel_from_residual_histories(
        H_mag_hist=H_mag_hist,
        H_phase_hist=H_phase_hist,
        tau_hist=tau_hist,
        step_idx=0,
        alpha=0.5,
        frequencies_hz=freqs,
    )
    np.testing.assert_allclose(scalar_interp.real, np.array([2.0, 3.0]))
    np.testing.assert_allclose(scalar_interp.imag, np.array([0.0, 0.0]))

    H_mag_hist_vec = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
    H_phase_hist_vec = np.zeros_like(H_mag_hist_vec)
    tau_hist_vec = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    vector_interp = simulator._interpolate_channel_from_residual_histories(
        H_mag_hist=H_mag_hist_vec,
        H_phase_hist=H_phase_hist_vec,
        tau_hist=tau_hist_vec,
        step_idx=np.array([0, 1], dtype=np.int32),
        alpha=np.array([0.5, 0.0], dtype=np.float64),
        frequencies_hz=freqs,
    )
    assert vector_interp.shape == (2, 2)
    assert np.isfinite(vector_interp).all()


def test_continuous_fade_and_slice_helpers_and_fractional_resample(monkeypatch) -> None:
    """Continuous helper utilities should call fade functions and produce stable sample slicing."""
    _base, _discrete, continuous = _load_simulator_modules(monkeypatch)

    calls = {"in": 0, "out": 0}

    def fake_fade_in(signal, fade_samples):
        calls["in"] += 1
        _ = fade_samples
        return signal + 1.0

    def fake_fade_out(signal, fade_samples):
        calls["out"] += 1
        _ = fade_samples
        return signal * 2.0

    monkeypatch.setattr(continuous, "apply_fade_in", fake_fade_in)
    monkeypatch.setattr(continuous, "apply_fade_out", fake_fade_out)

    simulator = continuous.ContinuousSTFTPassiveSonarArraySimulator(
        fade_in_ms=10.0,
        fade_out_ms=10.0,
    )
    receiver = np.ones((2, 4), dtype=np.complex64)
    faded = simulator._apply_fades(receiver.copy(), fs=100.0, do_fade_out=True)
    assert calls == {"in": 2, "out": 2}
    np.testing.assert_array_equal(faded, np.full((2, 4), 4.0 + 0.0j, dtype=np.complex64))

    sample_idx = simulator._slice_uniform_step_samples(
        np.zeros((2, 5), dtype=np.complex64),
        n_steps=2,
    )
    np.testing.assert_array_equal(sample_idx, np.array([0, 2, 5], dtype=np.int64))

    ctx = continuous._STFTCommonContext(
        all_timestamps=[],
        step_times_s=np.array([0.0, 9.0, 20.0]),
        n_steps=3,
        num_sensors=1,
        num_frames=1,
        num_freq_bins=1,
        frame_len=2,
        fs=1.0,
        frequencies=np.array([0.0]),
        hop=1,
        window=np.ones(2, dtype=np.float32),
    )
    np.testing.assert_array_equal(
        simulator._slice_knot_step_samples(ctx, out_len=8),
        np.array([0, 8, 8]),
    )

    real_resampled = (
        continuous.ContinuousFractionalDelayPassiveSonarArraySimulator._fractional_delay_resample(
            source_signal=np.array([0.0, 1.0, 2.0], dtype=np.float32),
            delay_s=np.zeros(3, dtype=np.float64),
            fs=1.0,
        )
    )
    np.testing.assert_array_equal(real_resampled, np.array([0.0, 1.0, 2.0], dtype=np.complex64))

    complex_resampled = (
        continuous.ContinuousFractionalDelayPassiveSonarArraySimulator._fractional_delay_resample(
            source_signal=np.array([0.0 + 0.0j, 1.0 + 1.0j, 2.0 + 2.0j], dtype=np.complex64),
            delay_s=np.zeros(3, dtype=np.float64),
            fs=1.0,
        )
    )
    np.testing.assert_array_equal(
        complex_resampled,
        np.array([0.0 + 0.0j, 1.0 + 1.0j, 2.0 + 2.0j], dtype=np.complex64),
    )


def test_deprecated_discrete_simulator_warns_and_yields(monkeypatch) -> None:
    """Deprecated discrete simulator should warn and yield one payload per timestamp."""
    _base, discrete, _continuous = _load_simulator_modules(monkeypatch)
    t0 = datetime(2026, 1, 1, 12, 0, 0)
    t1 = t0 + timedelta(seconds=1)

    with pytest.warns(DeprecationWarning, match="deprecated"):
        simulator = discrete.DepreciatedDiscretePassiveSonarArraySimulator(
            platform=_FakePlatform([t1, t0], num_sensors=1)
        )

    marker = object()
    monkeypatch.setattr(simulator, "_generate_sensor_data_at", lambda timestamp: marker)
    generated = list(simulator.sensor_data_gen())
    assert [ts for ts, _ in generated] == [t0, t1]
    assert all(data_set == {marker} for _, data_set in generated)


def test_deprecated_discrete_generate_sensor_data_validates_and_covers_modes(monkeypatch) -> None:
    """Deprecated generator should validate method names and support spectrum/TL branches."""
    _base, discrete, _continuous = _load_simulator_modules(monkeypatch)
    timestamp = datetime(2026, 1, 1, 12, 0, 0)
    platform = _FakePlatform([timestamp], num_sensors=1)
    path = _FakePath(states=[_FakeState(timestamp, state_vector=np.array([0.0]))])

    class FallbackSignalModel:
        sampling_rate_hz = 2.0
        num_samples = 2

        def generate(self, target_state, sensor_delays_s, tloss_db, prop_time_s):
            _ = target_state
            assert tloss_db == pytest.approx(3.0)
            assert prop_time_s == pytest.approx(0.5)
            np.testing.assert_array_equal(sensor_delays_s, np.array([0.0]))
            return np.array([[7.0 + 0.0j, 8.0 + 0.0j]], dtype=np.complex128)

    with pytest.warns(DeprecationWarning, match="deprecated"):
        simulator = discrete.DepreciatedDiscretePassiveSonarArraySimulator(
            platform=platform,
            propagation_model=SimpleNamespace(),
            signal_models=[FallbackSignalModel()],
            ground_truth_paths=[path],
            propagation_method="not-a-mode",
        )
    with pytest.raises(ValueError, match="Unsupported propagation_method"):
        simulator._generate_sensor_data_at(timestamp)

    with pytest.warns(DeprecationWarning, match="deprecated"):
        simulator = discrete.DepreciatedDiscretePassiveSonarArraySimulator(
            platform=platform,
            propagation_model=SimpleNamespace(),
            signal_models=[FallbackSignalModel()],
            ground_truth_paths=[path],
            propagation_method="spectrum",
        )
    with pytest.raises(AttributeError, match="requires propagation_model"):
        simulator._generate_sensor_data_at(timestamp)

    propagation = SimpleNamespace(
        propagate_spectrum=lambda platform_state, target_state, frequencies: (
            np.ones((1, len(frequencies)), dtype=np.complex128),
            0.0,
        ),
        propagate=lambda platform_state, target_state: (3.0, 0.5),
        compute_sensor_delays=lambda platform_state, target_state: np.array([0.0]),
    )
    with pytest.warns(DeprecationWarning, match="deprecated"):
        simulator = discrete.DepreciatedDiscretePassiveSonarArraySimulator(
            platform=platform,
            propagation_model=propagation,
            signal_models=[FallbackSignalModel()],
            ground_truth_paths=[path],
            propagation_method="spectrum",
        )
    spectrum_data = simulator._generate_sensor_data_at(timestamp)
    np.testing.assert_array_equal(
        spectrum_data.raw_signals,
        np.array([[7.0 + 0.0j, 8.0 + 0.0j]], dtype=np.complex128),
    )

    class TlSignalModel:
        num_samples = 2

        def generate(self, target_state, sensor_delays_s, tloss_db, prop_time_s):
            _ = target_state
            assert tloss_db == pytest.approx(9.0)
            assert prop_time_s == pytest.approx(0.25)
            np.testing.assert_array_equal(sensor_delays_s, np.array([0.0]))
            return np.array([[1.0 + 0.0j, 2.0 + 0.0j]], dtype=np.complex128)

    class Noise:
        def generate(self, num_sensors):
            return np.array([[10.0 + 0.0j, 10.0 + 0.0j]], dtype=np.complex128)

    class Steering:
        def calculate(self, platform_state):
            return np.array([0.0])

    class Beamformer:
        def beamform(self, sensor_signals, steering_delays_s):
            _ = steering_delays_s
            return sensor_signals.copy()

    propagation_tl = SimpleNamespace(
        propagate=lambda platform_state, target_state: (9.0, 0.25),
        compute_sensor_delays=lambda platform_state, target_state: np.array([0.0]),
    )
    with pytest.warns(DeprecationWarning, match="deprecated"):
        simulator = discrete.DepreciatedDiscretePassiveSonarArraySimulator(
            platform=platform,
            propagation_model=propagation_tl,
            signal_models=[TlSignalModel()],
            noise_model=Noise(),
            beamformer=Beamformer(),
            steering_calculator=Steering(),
            ground_truth_paths=[path],
            propagation_method="transmission_loss",
        )
    tl_data = simulator._generate_sensor_data_at(timestamp)
    np.testing.assert_array_equal(
        tl_data.raw_signals,
        np.array([[11.0 + 0.0j, 12.0 + 0.0j]], dtype=np.complex128),
    )
    np.testing.assert_array_equal(tl_data.beamformed_data, tl_data.raw_signals)

    class SpectrumBaseSignal:
        sampling_rate_hz = 2.0
        num_samples = 4

        def _generate_base_signal(self, state):
            _ = state
            return np.array([1.0 + 0.0j, 2.0 + 0.0j], dtype=np.complex128)

    propagation_spectrum = SimpleNamespace(
        propagate_spectrum=lambda platform_state, target_state, frequencies: (
            np.ones((1, len(frequencies)), dtype=np.complex128),
            0.0,
        ),
        propagate=lambda platform_state, target_state: (0.0, 0.0),
        compute_sensor_delays=lambda platform_state, target_state: np.array([0.0]),
    )
    with pytest.warns(DeprecationWarning, match="deprecated"):
        simulator = discrete.DepreciatedDiscretePassiveSonarArraySimulator(
            platform=platform,
            propagation_model=propagation_spectrum,
            signal_models=[SpectrumBaseSignal()],
            ground_truth_paths=[path],
            propagation_method="spectrum",
        )
    spectrum_base_data = simulator._generate_sensor_data_at(timestamp)
    assert spectrum_base_data.raw_signals.shape == (1, 4)

    class SpectrumLongSignal(SpectrumBaseSignal):
        def _generate_base_signal(self, state):
            _ = state
            return np.array([1.0, 2.0, 3.0, 4.0, 9.0], dtype=np.complex128)

    with pytest.warns(DeprecationWarning, match="deprecated"):
        simulator = discrete.DepreciatedDiscretePassiveSonarArraySimulator(
            platform=platform,
            propagation_model=propagation_spectrum,
            signal_models=[SpectrumLongSignal()],
            ground_truth_paths=[path],
            propagation_method="spectrum",
        )
    spectrum_long_data = simulator._generate_sensor_data_at(timestamp)
    assert spectrum_long_data.raw_signals.shape == (1, 4)

    class SpectrumExactSignal(SpectrumBaseSignal):
        def _generate_base_signal(self, state):
            _ = state
            return np.array([1.0, 2.0, 3.0, 4.0], dtype=np.complex128)

    with pytest.warns(DeprecationWarning, match="deprecated"):
        simulator = discrete.DepreciatedDiscretePassiveSonarArraySimulator(
            platform=platform,
            propagation_model=propagation_spectrum,
            signal_models=[SpectrumExactSignal()],
            ground_truth_paths=[path],
            propagation_method="spectrum",
        )
    spectrum_exact_data = simulator._generate_sensor_data_at(timestamp)
    assert spectrum_exact_data.raw_signals.shape == (1, 4)

    missing_path = _FakePath(states=[_FakeState(timestamp + timedelta(seconds=1))])
    with pytest.warns(DeprecationWarning, match="deprecated"):
        simulator = discrete.DepreciatedDiscretePassiveSonarArraySimulator(
            platform=platform,
            propagation_model=propagation_tl,
            signal_models=[TlSignalModel()],
            ground_truth_paths=[missing_path],
            propagation_method="transmission_loss",
        )
    missing_target_data = simulator._generate_sensor_data_at(timestamp)
    np.testing.assert_array_equal(
        missing_target_data.raw_signals,
        np.zeros((1, 2), dtype=np.complex128),
    )


def test_continuous_build_target_histories_and_modes_cover_wola_and_cola(monkeypatch) -> None:
    """Continuous simulator should validate STFT shape and run both WOLA/COLA synthesis modes."""
    _base, _discrete, continuous = _load_simulator_modules(monkeypatch)
    t0 = datetime(2026, 1, 1, 12, 0, 0)
    t1 = t0 + timedelta(seconds=1)
    platform = _FakePlatform([t0, t1], num_sensors=1)

    class GoodModel:
        sampling_rate_hz = 2.0
        frame_len = 4

        def compute_stft(self, state):
            _ = state
            stft = np.array(
                [[1.0 + 0.0j, 0.5 + 0.0j, 0.2 + 0.0j], [1.0 + 0.0j, 0.5 + 0.0j, 0.2 + 0.0j]],
                dtype=np.complex64,
            )
            return stft, np.array([0.0, 0.5, 1.0]), 2, np.ones(4, dtype=np.float32)

    class BadShapeModel(GoodModel):
        def compute_stft(self, state):
            _ = state
            stft = np.array([[1.0 + 0.0j, 0.5 + 0.0j, 0.2 + 0.0j]], dtype=np.complex64)
            return stft, np.array([0.0, 0.5, 1.0]), 2, np.ones(4, dtype=np.float32)

    path_a = _FakePath(states=[_FakeState(t0), _FakeState(t1)])
    path_b = _FakePath(states=[_FakeState(t0), _FakeState(t1)])

    propagation_stub = SimpleNamespace(
        propagate_spectrum=lambda platform_state, target_state, freqs: (
            np.ones((1, len(freqs)), dtype=np.complex64),
            0.0,
        ),
        compute_sensor_delays=lambda platform_state, target_state: np.array([0.0]),
    )
    simulator = continuous.ContinuousSTFTPassiveSonarArraySimulator(
        platform=platform,
        propagation_model=propagation_stub,
        signal_models=[GoodModel()],
    )
    ctx = simulator._build_common_context([t0, t1], [GoodModel()], first_state=path_a.states[0])
    with pytest.raises(RuntimeError, match="must share the same shape"):
        simulator._build_target_histories(ctx, [path_a, path_b], [GoodModel(), BadShapeModel()])

    propagation = SimpleNamespace(
        propagate_spectrum=lambda platform_state, target_state, freqs: (
            np.ones((1, len(freqs)), dtype=np.complex64),
            0.0,
        ),
        compute_sensor_delays=lambda platform_state, target_state: np.array([0.0]),
    )

    for mode in ("wola_interp", "cola"):
        sim = continuous.ContinuousSTFTPassiveSonarArraySimulator(
            platform=platform,
            propagation_model=propagation,
            signal_models=[GoodModel()],
            ground_truth_paths=[path_a],
            mode=mode,
            fade_in_ms=0.0,
            fade_out_ms=0.0,
        )
        generated = list(sim.sensor_data_gen())
        assert [ts for ts, _ in generated] == [t0, t1]
        assert all(len(next(iter(payload)).raw_signals.shape) == 2 for _, payload in generated)

    two_sided = continuous.ContinuousSTFTPassiveSonarArraySimulator._ifft_frame(
        frame_spec=np.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex64),
        frame_len=4,
        num_freq_bins=4,
    )
    one_sided = continuous.ContinuousSTFTPassiveSonarArraySimulator._ifft_frame(
        frame_spec=np.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex64),
        frame_len=4,
        num_freq_bins=3,
    )
    assert two_sided.shape == (4,)
    assert one_sided.shape == (4,)


def test_fractional_delay_simulator_covers_errors_fallback_and_outputs(monkeypatch) -> None:
    """Fractional-delay simulator should exercise validation, fallback, and output branches."""
    _base, _discrete, continuous = _load_simulator_modules(monkeypatch)
    t0 = datetime(2026, 1, 1, 12, 0, 0)
    t1 = t0 + timedelta(seconds=1)

    class ToggleSourceModel:
        sampling_rate_hz = 2.0
        frame_len = 4

        def __init__(self, source_signal: np.ndarray, raise_calls: set[int]):
            self._source_signal = source_signal
            self._raise_calls = set(raise_calls)
            self._calls = 0
            self.stft_calls = 0

        def get_source_signal(self):
            self._calls += 1
            if self._calls in self._raise_calls:
                raise RuntimeError("not ready")
            return self._source_signal

        def compute_stft(self, state):
            _ = state
            self.stft_calls += 1
            return (
                np.ones((2, 3), dtype=np.complex64),
                np.array([0.0, 0.5, 1.0]),
                2,
                np.ones(4, dtype=np.float32),
            )

    short_sim = continuous.ContinuousFractionalDelayPassiveSonarArraySimulator(
        platform=_FakePlatform([t0], num_sensors=1),
        signal_models=[ToggleSourceModel(np.array([1.0, 2.0], dtype=np.complex64), set())],
        ground_truth_paths=[_FakePath(states=[_FakeState(t0)])],
    )
    with pytest.raises(ValueError, match="Need at least 2 timesteps"):
        list(short_sim.sensor_data_gen())

    no_target_sim = continuous.ContinuousFractionalDelayPassiveSonarArraySimulator(
        platform=_FakePlatform([t0, t1], num_sensors=1),
        signal_models=[ToggleSourceModel(np.array([1.0, 2.0], dtype=np.complex64), set())],
        ground_truth_paths=[],
    )
    with pytest.raises(ValueError, match="requires at least one target"):
        list(no_target_sim.sensor_data_gen())

    path_a = _FakePath(states=[_FakeState(t0), _FakeState(t1)])
    path_b = _FakePath(states=[_FakeState(t0), _FakeState(t1)])
    mismatch_sim = continuous.ContinuousFractionalDelayPassiveSonarArraySimulator(
        platform=_FakePlatform([t0, t1], num_sensors=1),
        propagation_model=SimpleNamespace(
            propagate_spectrum=lambda platform_state, target_state, freqs: (
                np.ones((1, len(freqs)), dtype=np.complex64),
                0.0,
            ),
            compute_sensor_delays=lambda platform_state, target_state: np.array([0.0]),
        ),
        signal_models=[
            ToggleSourceModel(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.complex64), set()),
            ToggleSourceModel(np.array([1.0, 2.0, 3.0], dtype=np.complex64), set()),
        ],
        ground_truth_paths=[path_a, path_b],
    )
    with pytest.raises(RuntimeError, match="same sample length"):
        list(mismatch_sim.sensor_data_gen())

    fade_calls = {"in": 0, "out": 0}

    def fake_fade_in(signal, fade_samples):
        fade_calls["in"] += 1
        _ = fade_samples
        return signal

    def fake_fade_out(signal, fade_samples):
        fade_calls["out"] += 1
        _ = fade_samples
        return signal

    monkeypatch.setattr(continuous, "apply_fade_in", fake_fade_in)
    monkeypatch.setattr(continuous, "apply_fade_out", fake_fade_out)

    model = ToggleSourceModel(
        np.array([1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j, 4.0 + 0.0j], dtype=np.complex64),
        raise_calls={1, 3},
    )

    class Noise:
        def generate(self, num_sensors):
            return np.ones((num_sensors, 2), dtype=np.complex64)

    class Steering:
        def calculate(self, platform_state):
            return np.array([0.0])

    class Beamformer:
        def beamform(self, sensor_signals, steering_delays_s):
            _ = steering_delays_s
            return sensor_signals.copy()

    success_sim = continuous.ContinuousFractionalDelayPassiveSonarArraySimulator(
        platform=_FakePlatform([t0, t1], num_sensors=1),
        propagation_model=SimpleNamespace(
            propagate_spectrum=lambda platform_state, target_state, freqs: (
                np.ones((1, len(freqs)), dtype=np.complex64),
                0.0,
            ),
            compute_sensor_delays=lambda platform_state, target_state: np.array([0.0]),
        ),
        signal_models=[model],
        noise_model=Noise(),
        beamformer=Beamformer(),
        steering_calculator=Steering(),
        ground_truth_paths=[path_a],
        fade_in_ms=1.0,
        fade_out_ms=1.0,
    )
    generated = list(success_sim.sensor_data_gen())
    assert [ts for ts, _ in generated] == [t0, t1]
    assert model.stft_calls >= 2
    assert fade_calls["in"] >= 1
    assert fade_calls["out"] >= 1
    assert all(np.isfinite(next(iter(payload)).raw_signals).all() for _, payload in generated)


def test_continuous_stft_interp_pads_sensor_lengths_and_fractional_paths_without_options(
    monkeypatch,
) -> None:
    """Cover STFT padding and fractional-delay branches with no fades/no noise/no beamforming."""
    _base, _discrete, continuous = _load_simulator_modules(monkeypatch)
    simulator = continuous.ContinuousSTFTPassiveSonarArraySimulator(
        fade_in_ms=0.0,
        fade_out_ms=0.0,
    )

    returns = [
        np.array([1.0 + 0.0j, 2.0 + 0.0j], dtype=np.complex64),
        np.array([3.0 + 0.0j], dtype=np.complex64),
    ]

    def fake_inverse_stft(stft, frame_len, hop, window):
        _ = stft, frame_len, hop, window
        return returns.pop(0)

    monkeypatch.setattr(continuous, "inverse_stft", fake_inverse_stft)

    ctx = continuous._STFTCommonContext(
        all_timestamps=[datetime(2026, 1, 1, 12, 0, 0), datetime(2026, 1, 1, 12, 0, 1)],
        step_times_s=np.array([0.0, 1.0], dtype=np.float64),
        n_steps=2,
        num_sensors=2,
        num_frames=1,
        num_freq_bins=1,
        frame_len=2,
        fs=1.0,
        frequencies=np.array([0.0], dtype=np.float64),
        hop=1,
        window=np.ones(2, dtype=np.float32),
    )
    targets_data = [
        continuous._STFTTargetHistory(
            source_stft=np.ones((1, 1), dtype=np.complex64),
            H_hist=np.ones((2, 2, 1), dtype=np.complex64),
            tau_hist=np.zeros((2, 2), dtype=np.float64),
        )
    ]
    receiver, step_idx = simulator._synthesise_stft_interp(ctx, targets_data)
    assert receiver.shape == (2, 2)
    np.testing.assert_array_equal(step_idx, np.array([0, 1, 2], dtype=np.int64))

    t0 = datetime(2026, 1, 1, 12, 0, 0)
    t1 = t0 + timedelta(seconds=1)
    path_missing = _FakePath(states=[_FakeState(t0)])

    class SimpleSource:
        sampling_rate_hz = 2.0
        frame_len = 4

        def get_source_signal(self):
            return np.array([1.0, 2.0, 3.0, 4.0], dtype=np.complex64)

        def compute_stft(self, state):
            _ = state
            return (
                np.ones((2, 3), dtype=np.complex64),
                np.array([0.0, 0.5, 1.0]),
                2,
                np.ones(4, dtype=np.float32),
            )

    frac = continuous.ContinuousFractionalDelayPassiveSonarArraySimulator(
        platform=_FakePlatform([t0, t1], num_sensors=1),
        propagation_model=SimpleNamespace(
            propagate_spectrum=lambda platform_state, target_state, freqs: (
                np.ones((1, len(freqs)), dtype=np.complex64),
                0.0,
            ),
            compute_sensor_delays=lambda platform_state, target_state: np.array([0.0]),
        ),
        signal_models=[SimpleSource()],
        noise_model=None,
        beamformer=None,
        steering_calculator=None,
        ground_truth_paths=[path_missing],
        fade_in_ms=0.0,
        fade_out_ms=0.0,
    )
    generated = list(frac.sensor_data_gen())
    assert [ts for ts, _ in generated] == [t0, t1]
    assert all(len(next(iter(payload)).raw_signals.shape) == 2 for _, payload in generated)
