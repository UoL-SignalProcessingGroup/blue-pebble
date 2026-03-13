"""Tests for anthropogenic signal abstractions and public classes."""

from __future__ import annotations

import wave
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from .support import (
    install_fake_stonesoup,
    install_repo_package,
    load_package_module_from_repo,
)


@dataclass
class _FakeSourceState:
    """Lightweight source state for signal model tests."""

    metadata: dict
    timestamp: datetime


def _load_anthropogenic_modules(monkeypatch):
    """Load anthropogenic modules with lightweight Stone Soup scaffolding."""
    install_fake_stonesoup(monkeypatch)
    install_repo_package(monkeypatch, "bluepebble", "bluepebble")
    install_repo_package(monkeypatch, "bluepebble.signal", "bluepebble/signal")
    install_repo_package(
        monkeypatch,
        "bluepebble.signal.anthropogenic",
        "bluepebble/signal/anthropogenic",
    )

    load_package_module_from_repo("bluepebble/signal/base.py", "bluepebble.signal.base")
    load_package_module_from_repo("bluepebble/signal/utils.py", "bluepebble.signal.utils")
    anthropogenic_base = load_package_module_from_repo(
        "bluepebble/signal/anthropogenic/base.py",
        "bluepebble.signal.anthropogenic.base",
    )
    anthropogenic_discrete = load_package_module_from_repo(
        "bluepebble/signal/anthropogenic/discrete.py",
        "bluepebble.signal.anthropogenic.discrete",
    )
    anthropogenic_continuous = load_package_module_from_repo(
        "bluepebble/signal/anthropogenic/continuous.py",
        "bluepebble.signal.anthropogenic.continuous",
    )
    anthropogenic_api = load_package_module_from_repo(
        "bluepebble/signal/anthropogenic/__init__.py",
        "bluepebble.signal.anthropogenic",
    )

    return anthropogenic_base, anthropogenic_discrete, anthropogenic_continuous, anthropogenic_api


def _default_tonal_source(timestamp: datetime) -> _FakeSourceState:
    """Create a deterministic tonal source state."""
    return _FakeSourceState(
        metadata={
            "amplitudes_upa": np.array([1.0, 0.5]),
            "frequencies_hz": np.array([2.0, 3.0]),
            "phases_rad": np.array([0.0, np.pi / 4]),
        },
        timestamp=timestamp,
    )


def _write_pcm_wav(path: Path, fs_hz: int, samples: np.ndarray) -> None:
    """Write a mono 16-bit PCM waveform for measured-signal tests."""
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = np.asarray(np.round(clipped * 32767.0), dtype=np.int16)

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(fs_hz)
        wav_file.writeframes(pcm.tobytes())


def test_narrowband_tonal_generate_validates_tloss_length(monkeypatch) -> None:
    """Per-tonal transmission loss vectors must match tonal count."""
    _, discrete, _, _ = _load_anthropogenic_modules(monkeypatch)

    model = discrete.NarrowbandTonalSignal(duration_s=1.0, sampling_rate_hz=8)
    source = _default_tonal_source(datetime(2026, 1, 1, 12, 0, 0))

    with pytest.raises(ValueError, match="Length of tloss_db array"):
        model.generate(
            source=source,
            sensor_delays_s=np.array([0.0, 0.1]),
            tloss_db=np.array([3.0]),
            propagation_time_s=0.0,
        )


def test_narrowband_tonal_generate_validates_metadata_keys(monkeypatch) -> None:
    """Source metadata must include amplitudes, frequencies, and phases."""
    _, discrete, _, _ = _load_anthropogenic_modules(monkeypatch)

    model = discrete.NarrowbandTonalSignal(duration_s=1.0, sampling_rate_hz=8)
    invalid_source = _FakeSourceState(
        metadata={
            "amplitudes_upa": np.array([1.0]),
            "frequencies_hz": np.array([2.0]),
        },
        timestamp=datetime(2026, 1, 1, 12, 0, 0),
    )

    with pytest.raises(ValueError, match="Source metadata missing required keys"):
        model.generate(
            source=invalid_source,
            sensor_delays_s=np.array([0.0]),
            tloss_db=0.0,
            propagation_time_s=0.0,
        )


def test_narrowband_tonal_generate_rejects_non_mapping_metadata(monkeypatch) -> None:
    """Source metadata must be mapping-like for tonal generation."""
    _, discrete, _, _ = _load_anthropogenic_modules(monkeypatch)

    model = discrete.NarrowbandTonalSignal(duration_s=1.0, sampling_rate_hz=8)
    invalid_source = SimpleNamespace(
        metadata=[("amplitudes_upa", [1.0])],
        timestamp=datetime(2026, 1, 1, 12, 0, 0),
    )

    with pytest.raises(ValueError, match="mapping-like"):
        model.generate(
            source=invalid_source,
            sensor_delays_s=np.array([0.0]),
            tloss_db=0.0,
            propagation_time_s=0.0,
        )


def test_narrowband_tonal_generate_validates_metadata_array_shape(monkeypatch) -> None:
    """Tonal metadata arrays must be one-dimensional."""
    _, discrete, _, _ = _load_anthropogenic_modules(monkeypatch)

    model = discrete.NarrowbandTonalSignal(duration_s=1.0, sampling_rate_hz=8)
    invalid_source = _FakeSourceState(
        metadata={
            "amplitudes_upa": np.array([[1.0, 0.5]]),
            "frequencies_hz": np.array([2.0, 3.0]),
            "phases_rad": np.array([0.0, np.pi / 4]),
        },
        timestamp=datetime(2026, 1, 1, 12, 0, 0),
    )

    with pytest.raises(ValueError, match="one-dimensional"):
        model.generate(
            source=invalid_source,
            sensor_delays_s=np.array([0.0]),
            tloss_db=0.0,
            propagation_time_s=0.0,
        )


def test_narrowband_tonal_generate_validates_metadata_lengths(monkeypatch) -> None:
    """Tonal metadata arrays must have matching lengths."""
    _, discrete, _, _ = _load_anthropogenic_modules(monkeypatch)

    model = discrete.NarrowbandTonalSignal(duration_s=1.0, sampling_rate_hz=8)
    invalid_source = _FakeSourceState(
        metadata={
            "amplitudes_upa": np.array([1.0, 0.5]),
            "frequencies_hz": np.array([2.0]),
            "phases_rad": np.array([0.0, np.pi / 4]),
        },
        timestamp=datetime(2026, 1, 1, 12, 0, 0),
    )

    with pytest.raises(ValueError, match="matching lengths"):
        model.generate(
            source=invalid_source,
            sensor_delays_s=np.array([0.0]),
            tloss_db=0.0,
            propagation_time_s=0.0,
        )


def test_narrowband_tonal_generate_validates_tloss_dimensions(monkeypatch) -> None:
    """Transmission loss input must be scalar-like or one-dimensional."""
    _, discrete, _, _ = _load_anthropogenic_modules(monkeypatch)

    model = discrete.NarrowbandTonalSignal(duration_s=1.0, sampling_rate_hz=8)
    source = _default_tonal_source(datetime(2026, 1, 1, 12, 0, 0))

    with pytest.raises(ValueError, match="scalar-like or one-dimensional"):
        model.generate(
            source=source,
            sensor_delays_s=np.array([0.0]),
            tloss_db=np.array([[0.0, 3.0]]),
            propagation_time_s=0.0,
        )


def test_narrowband_tonal_generate_validates_sensor_delay_dimensions(monkeypatch) -> None:
    """Sensor delays must be one-dimensional."""
    _, discrete, _, _ = _load_anthropogenic_modules(monkeypatch)

    model = discrete.NarrowbandTonalSignal(duration_s=1.0, sampling_rate_hz=8)
    source = _default_tonal_source(datetime(2026, 1, 1, 12, 0, 0))

    with pytest.raises(ValueError, match="one-dimensional"):
        model.generate(
            source=source,
            sensor_delays_s=np.array([[0.0, 0.1]]),
            tloss_db=0.0,
            propagation_time_s=0.0,
        )


def test_narrowband_tonal_generate_returns_complex_sensor_matrix(monkeypatch) -> None:
    """Narrowband synthesis should return complex sensor snapshots."""
    _, discrete, _, _ = _load_anthropogenic_modules(monkeypatch)

    model = discrete.NarrowbandTonalSignal(duration_s=1.0, sampling_rate_hz=8)
    source = _default_tonal_source(datetime(2026, 1, 1, 12, 0, 0))

    sensor_signals = model.generate(
        source=source,
        sensor_delays_s=np.array([0.0, 0.1]),
        tloss_db=0.0,
        propagation_time_s=0.05,
    )

    assert sensor_signals.shape == (2, 8)
    assert np.iscomplexobj(sensor_signals)
    assert np.isfinite(sensor_signals).all()


def test_narrowband_blended_tracks_state_and_resets(monkeypatch) -> None:
    """Blended model should accumulate time and clear state on reset."""
    _, discrete, _, _ = _load_anthropogenic_modules(monkeypatch)

    model = discrete.NarrowbandBlendedTonalSignal(
        duration_s=1.0,
        sampling_rate_hz=8,
        blend_fraction=0.25,
    )

    t0 = datetime(2026, 1, 1, 12, 0, 0)
    source_1 = _default_tonal_source(t0)
    source_2 = _default_tonal_source(t0 + timedelta(seconds=0.5))

    _ = model.generate(source_1, np.array([0.0]), 0.0, 0.0)
    _ = model.generate(source_2, np.array([0.0]), 0.0, 0.0)

    assert len(model._source_states) == 1
    state = next(iter(model._source_states.values()))
    assert state["cumulative_time"] == pytest.approx(0.5)
    assert state["previous_signal"] is not None

    model.reset()
    assert model._source_states == {}


def test_narrowband_blended_requires_timestamp(monkeypatch) -> None:
    """Stateful narrowband models must receive timestamped source states."""
    _, discrete, _, _ = _load_anthropogenic_modules(monkeypatch)

    model = discrete.NarrowbandBlendedTonalSignal(
        duration_s=1.0,
        sampling_rate_hz=8,
        blend_fraction=0.25,
    )
    source_without_timestamp = SimpleNamespace(
        metadata={
            "amplitudes_upa": np.array([1.0]),
            "frequencies_hz": np.array([2.0]),
            "phases_rad": np.array([0.0]),
        }
    )

    with pytest.raises(ValueError, match="must define a timestamp"):
        model.generate(source_without_timestamp, np.array([0.0]), 0.0, 0.0)


def test_narrowband_overlap_add_uses_buffer_across_timesteps(monkeypatch) -> None:
    """Overlap-add model should use prior buffered content on subsequent calls."""
    _, discrete, _, _ = _load_anthropogenic_modules(monkeypatch)

    model = discrete.NarrowbandOverlapAddTonalSignal(duration_s=1.0, sampling_rate_hz=8)
    t0 = datetime(2026, 1, 1, 12, 0, 0)
    source_1 = _default_tonal_source(t0)
    source_2 = _default_tonal_source(t0 + timedelta(seconds=1.0))

    first = model.generate(source_1, np.array([0.0]), 0.0, 0.0)
    second = model.generate(source_2, np.array([0.0]), 0.0, 0.0)

    assert first.shape == second.shape == (1, 8)
    assert np.any(np.abs(second - first) > 1e-9)

    model.reset()
    assert model._source_states == {}
    assert model._synthesis_window is None


def test_broadband_stft_base_caches_and_resets(monkeypatch) -> None:
    """Shared broadband STFT base should cache generation and clear on reset."""
    anthropogenic_base, _, _, _ = _load_anthropogenic_modules(monkeypatch)

    class DummyBroadbandModel(anthropogenic_base.BroadbandStftSignalBase):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.calls = 0

        def _generate_base_signal(self, source) -> np.ndarray:
            _ = source
            self.calls += 1
            return np.ones(self.num_samples, dtype=np.float64)

    model = DummyBroadbandModel(
        duration_s=1.0,
        sampling_rate_hz=16,
        frame_len=8,
        hop_factor=2,
    )

    source = SimpleNamespace()
    stft_1, frequencies_1, hop_1, window_1 = model.compute_stft(source)
    stft_2, frequencies_2, hop_2, window_2 = model.compute_stft(source)

    assert model.calls == 1
    assert stft_1 is stft_2
    np.testing.assert_array_equal(frequencies_1, frequencies_2)
    assert hop_1 == hop_2 == 4
    np.testing.assert_array_equal(window_1, window_2)

    model.reset()
    with pytest.raises(RuntimeError, match="STFT not computed yet"):
        model.get_stft()


def test_broadband_stft_base_generate_always_raises(monkeypatch) -> None:
    """Broadband base class should reject per-timestep generation API usage."""
    anthropogenic_base, _, _, _ = _load_anthropogenic_modules(monkeypatch)

    class DummyBroadbandModel(anthropogenic_base.BroadbandStftSignalBase):
        def _generate_base_signal(self, source) -> np.ndarray:
            _ = source
            return np.ones(self.num_samples, dtype=np.float64)

    model = DummyBroadbandModel(duration_s=1.0, sampling_rate_hz=16)

    with pytest.raises(NotImplementedError, match="does not support per-timestep generation"):
        model.generate(SimpleNamespace(), np.array([0.0]), 0.0, 0.0)


def test_broadband_synthetic_compute_stft_contract(monkeypatch) -> None:
    """Synthetic broadband model should expose STFT outputs and cache source signal."""
    _, _, continuous, _ = _load_anthropogenic_modules(monkeypatch)

    source = _default_tonal_source(datetime(2026, 1, 1, 12, 0, 0))
    model = continuous.BroadbandSyntheticSignal(
        duration_s=1.0,
        sampling_rate_hz=16,
        frame_len=8,
        hop_factor=2,
        noise_amplitude_upa=0.0,
    )

    stft, frequencies, hop, window = model.compute_stft(source)

    assert stft.ndim == 2
    assert stft.shape[1] == len(frequencies)
    assert hop == 4
    assert len(window) == 8
    assert model.get_source_signal().shape == (16,)


def test_broadband_recorded_compute_stft_contract(monkeypatch, tmp_path: Path) -> None:
    """Recorded broadband model should compute STFT from WAV input."""
    _, _, continuous, _ = _load_anthropogenic_modules(monkeypatch)

    wav_path = tmp_path / "example.wav"
    fs_hz = 16
    t = np.arange(16, dtype=float) / fs_hz
    waveform = 0.5 * np.sin(2 * np.pi * 2.0 * t)
    _write_pcm_wav(wav_path, fs_hz=fs_hz, samples=waveform)

    model = continuous.BroadbandRecordedSignal(
        wav_path=str(wav_path),
        duration_s=1.0,
        sampling_rate_hz=16,
        frame_len=8,
        hop_factor=2,
        level_db_re_1upa=85.0,
    )

    stft, frequencies, hop, window = model.compute_stft(SimpleNamespace())

    assert stft.ndim == 2
    assert stft.shape[1] == len(frequencies)
    assert hop == 4
    assert len(window) == 8
    assert model.get_source_signal().shape == (16,)


def test_broadband_recorded_missing_wav_raises(monkeypatch, tmp_path: Path) -> None:
    """Recorded signal model should raise for missing WAV paths."""
    _, _, continuous, _ = _load_anthropogenic_modules(monkeypatch)

    model = continuous.BroadbandRecordedSignal(
        wav_path=str(tmp_path / "missing.wav"),
        duration_s=1.0,
        sampling_rate_hz=16,
        frame_len=8,
        hop_factor=2,
    )

    with pytest.raises(FileNotFoundError, match="Measured WAV file not found"):
        model.compute_stft(SimpleNamespace())


def test_broadband_recorded_empty_segment_raises(monkeypatch, tmp_path: Path) -> None:
    """Selecting an empty WAV segment should fail clearly."""
    _, _, continuous, _ = _load_anthropogenic_modules(monkeypatch)

    wav_path = tmp_path / "short.wav"
    _write_pcm_wav(wav_path, fs_hz=16, samples=np.zeros(16, dtype=float))

    model = continuous.BroadbandRecordedSignal(
        wav_path=str(wav_path),
        duration_s=1.0,
        sampling_rate_hz=16,
        frame_len=8,
        hop_factor=2,
        segment_start_s=5.0,
        segment_duration_s=1.0,
    )

    with pytest.raises(ValueError, match="Selected WAV segment is empty"):
        model.compute_stft(SimpleNamespace())


def test_broadband_recorded_invalid_duration_mode_raises(monkeypatch, tmp_path: Path) -> None:
    """Unknown duration matching modes should raise ValueError."""
    _, _, continuous, _ = _load_anthropogenic_modules(monkeypatch)

    wav_path = tmp_path / "short.wav"
    _write_pcm_wav(wav_path, fs_hz=16, samples=np.zeros(4, dtype=float))

    model = continuous.BroadbandRecordedSignal(
        wav_path=str(wav_path),
        duration_s=1.0,
        sampling_rate_hz=16,
        frame_len=8,
        hop_factor=2,
        duration_match_mode="invalid_mode",
    )

    with pytest.raises(ValueError, match="Unknown duration_match_mode"):
        model.compute_stft(SimpleNamespace())


def test_broadband_recorded_to_float_mono_accepts_array_like(monkeypatch) -> None:
    """Float-mono conversion should accept array-like (duck-typed) input."""
    _, _, continuous, _ = _load_anthropogenic_modules(monkeypatch)

    mono = continuous.BroadbandRecordedSignal._to_float_mono([[0, 2], [4, 6]])

    assert mono.dtype == np.float64
    np.testing.assert_allclose(mono, np.array([1.0, 5.0], dtype=np.float64))


def test_anthropogenic_public_api_exports_new_classes(monkeypatch) -> None:
    """Anthropogenic package should expose renamed classes and new base abstractions."""
    _, _, _, anthropogenic_api = _load_anthropogenic_modules(monkeypatch)

    expected_exports = {
        "NarrowbandSignalBase",
        "NarrowbandStatefulSignalBase",
        "BroadbandStftSignalBase",
        "NarrowbandTonalSignal",
        "NarrowbandBlendedTonalSignal",
        "NarrowbandOverlapAddTonalSignal",
        "BroadbandSyntheticSignal",
        "BroadbandRecordedSignal",
    }

    assert set(anthropogenic_api.__all__) == expected_exports

    for export_name in expected_exports:
        assert hasattr(anthropogenic_api, export_name)

    assert not hasattr(anthropogenic_api, "BroadbandShipSignal")
    assert not hasattr(anthropogenic_api, "BroadbandMeasuredSignal")
