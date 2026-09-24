"""Tests for anthropogenic signal abstractions and public classes."""

from __future__ import annotations

import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from .support import (
    install_fake_stonesoup,
    install_repo_package,
    load_package_module_from_repo,
)


def _load_anthropogenic_modules(monkeypatch):
    """Load anthropogenic modules with lightweight Stone Soup scaffolding."""
    install_fake_stonesoup(monkeypatch)
    install_repo_package(monkeypatch, "bluepebble", "bluepebble")
    install_repo_package(monkeypatch, "bluepebble.signal", "bluepebble/signal")

    load_package_module_from_repo("bluepebble/signal/base.py", "bluepebble.signal.base")
    load_package_module_from_repo("bluepebble/signal/utils.py", "bluepebble.signal.utils")
    anthropogenic = load_package_module_from_repo(
        "bluepebble/signal/anthropogenic.py",
        "bluepebble.signal.anthropogenic",
    )

    return anthropogenic, anthropogenic, anthropogenic


def _write_pcm_wav(path: Path, fs_hz: int, samples: np.ndarray) -> None:
    """Write a mono 16-bit PCM waveform for measured-signal tests."""
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = np.asarray(np.round(clipped * 32767.0), dtype=np.int16)

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(fs_hz)
        wav_file.writeframes(pcm.tobytes())


def test_anthropogenic_base_caches_and_resets(monkeypatch) -> None:
    """AnthropogenicSignal should cache STFT generation and clear on reset."""
    anthropogenic_base, _, _ = _load_anthropogenic_modules(monkeypatch)

    class DummyModel(anthropogenic_base.AnthropogenicSignal):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.calls = 0

        def _generate_base_signal(self, source) -> np.ndarray:
            _ = source
            self.calls += 1
            return np.ones(self.num_samples, dtype=np.float64)

    model = DummyModel(
        duration_s=1.0,
        sampling_rate_hz=16,
        frame_len=8,
        hop_factor=2,
    )

    source = SimpleNamespace()
    stft, frequencies, hop, window = model.compute_stft(source)

    assert model.calls == 1
    assert stft.shape[1] == 5  # frame_len // 2 + 1 frequency bins
    assert hop == 4

    with pytest.raises(RuntimeError, match="reset\\(\\)"):
        model.compute_stft(source)

    model.reset()
    with pytest.raises(RuntimeError, match="STFT not computed yet"):
        model.get_stft()

    model.compute_stft(source)
    assert model.calls == 2


def _source(**overrides) -> SimpleNamespace:
    """Return a one-tonal source whose metadata fully describes what it emits."""
    metadata = {
        "amplitudes_upa": np.array([1.0]),
        "frequencies_hz": np.array([4.0]),
        "phases_rad": np.array([0.0]),
        "tonal_bandwidth_hz": 2.0,
        "noise_amplitude_upa": 0.0,
        "noise_spectral_exponent": -1.0,
    }
    metadata.update(overrides)
    return SimpleNamespace(metadata=metadata)


def _synthetic_model(anthropogenic_models, **kwargs):
    """Return a small SyntheticAnthropogenicSignal; kwargs add constructor arguments."""
    return anthropogenic_models.SyntheticAnthropogenicSignal(
        duration_s=1.0, sampling_rate_hz=16, frame_len=8, hop_factor=2, **kwargs
    )


def test_synthetic_signal_compute_stft_contract(monkeypatch) -> None:
    """SyntheticSignal should expose STFT outputs and cache source signal."""
    _, anthropogenic_models, _ = _load_anthropogenic_modules(monkeypatch)

    source = _source()
    model = _synthetic_model(anthropogenic_models)

    stft, frequencies, hop, window = model.compute_stft(source)

    assert stft.ndim == 2
    assert stft.shape[1] == len(frequencies)
    assert hop == 4
    assert len(window) == 8
    assert model.get_source_signal().shape == (16,)


def test_synthetic_signal_rejects_non_positive_tonal_bandwidth(monkeypatch) -> None:
    """A source whose metadata gives a non-positive tonal bandwidth is rejected."""
    _, anthropogenic_models, _ = _load_anthropogenic_modules(monkeypatch)
    model = _synthetic_model(anthropogenic_models)

    with pytest.raises(ValueError, match="tonal_bandwidth_hz must be finite and > 0"):
        model.compute_stft(_source(tonal_bandwidth_hz=0.0))


def test_synthetic_signal_rejects_negative_noise_variance(monkeypatch) -> None:
    """SyntheticAnthropogenicSignal should reject negative noise variance."""
    _, anthropogenic_models, _ = _load_anthropogenic_modules(monkeypatch)

    with pytest.raises(ValueError, match="noise_variance must be finite and >= 0"):
        anthropogenic_models.SyntheticAnthropogenicSignal(
            duration_s=1.0,
            sampling_rate_hz=16,
            frame_len=8,
            hop_factor=2,
            noise_variance=-1.0,
        )


def test_synthetic_signal_needs_its_source_characteristics_in_metadata(monkeypatch) -> None:
    """Missing tonal bandwidth or noise level is an error, naming each missing key."""
    _, anthropogenic_models, _ = _load_anthropogenic_modules(monkeypatch)
    source = _source()
    del source.metadata["tonal_bandwidth_hz"]
    del source.metadata["noise_amplitude_upa"]

    with pytest.raises(ValueError, match="tonal_bandwidth_hz, noise_amplitude_upa"):
        _synthetic_model(anthropogenic_models).compute_stft(source)


def test_noise_spectral_exponent_is_only_needed_when_there_is_noise(monkeypatch) -> None:
    """With no broadband noise there is nothing to shape, so the exponent may be omitted."""
    _, anthropogenic_models, _ = _load_anthropogenic_modules(monkeypatch)

    quiet = _source(noise_amplitude_upa=0.0)
    del quiet.metadata["noise_spectral_exponent"]
    _synthetic_model(anthropogenic_models).compute_stft(quiet)

    noisy = _source(noise_amplitude_upa=1.0)
    del noisy.metadata["noise_spectral_exponent"]
    with pytest.raises(ValueError, match="noise_spectral_exponent"):
        _synthetic_model(anthropogenic_models).compute_stft(noisy)


def test_synthetic_signal_rejects_negative_noise_amplitude(monkeypatch) -> None:
    """A negative broadband noise amplitude in metadata is rejected."""
    _, anthropogenic_models, _ = _load_anthropogenic_modules(monkeypatch)

    with pytest.raises(ValueError, match="noise_amplitude_upa must be finite and >= 0"):
        _synthetic_model(anthropogenic_models).compute_stft(_source(noise_amplitude_upa=-1.0))


def test_deprecated_constructor_characteristics_warn_and_fill_only_missing_keys(
    monkeypatch,
) -> None:
    """The old constructor arguments still work for one release, but metadata takes priority."""
    _, anthropogenic_models, _ = _load_anthropogenic_modules(monkeypatch)

    with pytest.warns(DeprecationWarning, match="noise_amplitude_upa"):
        model = _synthetic_model(anthropogenic_models, noise_amplitude_upa=5.0)
    source = _source(noise_amplitude_upa=0.0)
    del source.metadata["tonal_bandwidth_hz"]  # neither given: still an error
    with pytest.raises(ValueError, match="tonal_bandwidth_hz"):
        model.compute_stft(source)

    with pytest.warns(DeprecationWarning, match="tonal_bandwidth_hz"):
        bridged = _synthetic_model(anthropogenic_models, tonal_bandwidth_hz=2.0)
    source = _source(noise_amplitude_upa=0.0)
    del source.metadata["tonal_bandwidth_hz"]  # filled from the deprecated argument
    bridged.compute_stft(source)

    # When both are given, the metadata wins: a zero-noise source gets no broadband noise even
    # though the deprecated argument asks for some. The same seed makes the tonals identical.
    with pytest.warns(DeprecationWarning):
        both = _synthetic_model(anthropogenic_models, noise_amplitude_upa=5.0, seed=0)
    both.compute_stft(_source(noise_amplitude_upa=0.0))
    reference = _synthetic_model(anthropogenic_models, seed=0)
    reference.compute_stft(_source(noise_amplitude_upa=0.0))
    np.testing.assert_allclose(both.get_source_signal(), reference.get_source_signal())


def test_synthetic_signal_rejects_invalid_noise_variance_after_init(monkeypatch) -> None:
    """Generation should fail if noise variance becomes invalid after init."""
    _, anthropogenic_models, _ = _load_anthropogenic_modules(monkeypatch)

    source = _source(noise_amplitude_upa=1.0)
    model = _synthetic_model(anthropogenic_models, noise_variance=1.0)
    model.noise_variance = -1.0

    with pytest.raises(ValueError, match="noise_variance must be finite and >= 0"):
        model.compute_stft(source)


def test_recorded_signal_compute_stft_contract(monkeypatch, tmp_path: Path) -> None:
    """RecordedSignal should compute STFT from WAV input."""
    _, anthropogenic_models, _ = _load_anthropogenic_modules(monkeypatch)

    wav_path = tmp_path / "example.wav"
    fs_hz = 16
    t = np.arange(16, dtype=float) / fs_hz
    waveform = 0.5 * np.sin(2 * np.pi * 2.0 * t)
    _write_pcm_wav(wav_path, fs_hz=fs_hz, samples=waveform)

    model = anthropogenic_models.RecordedAnthropogenicSignal(
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


def test_recorded_signal_missing_wav_raises(monkeypatch, tmp_path: Path) -> None:
    """RecordedSignal should raise for missing WAV paths."""
    _, anthropogenic_models, _ = _load_anthropogenic_modules(monkeypatch)

    model = anthropogenic_models.RecordedAnthropogenicSignal(
        wav_path=str(tmp_path / "missing.wav"),
        duration_s=1.0,
        sampling_rate_hz=16,
        frame_len=8,
        hop_factor=2,
    )

    with pytest.raises(FileNotFoundError, match="Measured WAV file not found"):
        model.compute_stft(SimpleNamespace())


def test_recorded_signal_empty_segment_raises(monkeypatch, tmp_path: Path) -> None:
    """Selecting an empty WAV segment should fail clearly."""
    _, anthropogenic_models, _ = _load_anthropogenic_modules(monkeypatch)

    wav_path = tmp_path / "short.wav"
    _write_pcm_wav(wav_path, fs_hz=16, samples=np.zeros(16, dtype=float))

    model = anthropogenic_models.RecordedAnthropogenicSignal(
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


def test_recorded_signal_invalid_duration_mode_raises(monkeypatch, tmp_path: Path) -> None:
    """Unknown duration matching modes should raise ValueError."""
    _, anthropogenic_models, _ = _load_anthropogenic_modules(monkeypatch)

    wav_path = tmp_path / "short.wav"
    _write_pcm_wav(wav_path, fs_hz=16, samples=np.zeros(4, dtype=float))

    model = anthropogenic_models.RecordedAnthropogenicSignal(
        wav_path=str(wav_path),
        duration_s=1.0,
        sampling_rate_hz=16,
        frame_len=8,
        hop_factor=2,
        duration_match_mode="invalid_mode",
    )

    with pytest.raises(ValueError, match="Unknown duration_match_mode"):
        model.compute_stft(SimpleNamespace())


def test_recorded_signal_to_float_mono_accepts_array_like(monkeypatch) -> None:
    """Float-mono conversion should accept array-like (duck-typed) input."""
    _, anthropogenic_models, _ = _load_anthropogenic_modules(monkeypatch)

    mono = anthropogenic_models.RecordedAnthropogenicSignal._to_float_mono([[0, 2], [4, 6]])

    assert mono.dtype == np.float64
    np.testing.assert_allclose(mono, np.array([1.0, 5.0], dtype=np.float64))


def test_get_source_waveform_warns_on_different_source(monkeypatch) -> None:
    """get_source_waveform should warn when called with a different source object."""
    anthropogenic_base, _, _ = _load_anthropogenic_modules(monkeypatch)

    class DummyModel(anthropogenic_base.AnthropogenicSignal):
        def _generate_base_signal(self, source) -> np.ndarray:
            _ = source
            return np.ones(self.num_samples, dtype=np.float64)

    model = DummyModel(
        duration_s=1.0,
        sampling_rate_hz=16,
        frame_len=8,
        hop_factor=2,
    )

    source_a = SimpleNamespace()
    source_b = SimpleNamespace()
    model.compute_stft(source_a)

    with pytest.warns(UserWarning, match="different source state"):
        result = model.get_source_waveform(source_b)

    # Cached result is still returned unchanged
    assert result.shape == (16,)


def test_get_source_waveform_no_warn_same_source(monkeypatch) -> None:
    """get_source_waveform should not warn when called with the original source."""
    import warnings as _warnings

    anthropogenic_base, _, _ = _load_anthropogenic_modules(monkeypatch)

    class DummyModel(anthropogenic_base.AnthropogenicSignal):
        def _generate_base_signal(self, source) -> np.ndarray:
            _ = source
            return np.ones(self.num_samples, dtype=np.float64)

    model = DummyModel(
        duration_s=1.0,
        sampling_rate_hz=16,
        frame_len=8,
        hop_factor=2,
    )

    source = SimpleNamespace()
    model.compute_stft(source)

    with _warnings.catch_warnings():
        _warnings.simplefilter("error")
        model.get_source_waveform(source)  # must not raise


def test_reset_before_compute_stft_is_safe(monkeypatch) -> None:
    """reset() should be safe to call before compute_stft has ever been called."""
    anthropogenic_base, _, _ = _load_anthropogenic_modules(monkeypatch)

    class DummyModel(anthropogenic_base.AnthropogenicSignal):
        def _generate_base_signal(self, source) -> np.ndarray:
            return np.ones(self.num_samples, dtype=np.float64)

    model = DummyModel(
        duration_s=1.0,
        sampling_rate_hz=16,
        frame_len=8,
        hop_factor=2,
    )

    # Must not raise
    model.reset()


def test_anthropogenic_public_api_exports(monkeypatch) -> None:
    """Anthropogenic package should expose the correct public classes."""
    _, _, anthropogenic_api = _load_anthropogenic_modules(monkeypatch)

    expected_exports = {
        "AnthropogenicSignal",
        "SyntheticAnthropogenicSignal",
        "RecordedAnthropogenicSignal",
    }

    for export_name in expected_exports:
        assert hasattr(anthropogenic_api, export_name)
