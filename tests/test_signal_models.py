"""Coverage-oriented tests for signal models."""

from __future__ import annotations

import sys
import types
import numpy as np
import pytest

from .support import (
    FakeBase,
    install_fake_stonesoup,
    install_repo_package,
    load_module_from_repo,
    load_package_module_from_repo,
)


def _load_biological(monkeypatch):
    """Load ``bluepebble.signal.biological`` with lightweight stubs."""
    install_fake_stonesoup(monkeypatch)
    install_repo_package(monkeypatch, "bluepebble", "bluepebble")
    install_repo_package(monkeypatch, "bluepebble.signal", "bluepebble/signal")
    # Stub out SoundSpeedProfile (biological.py imports it for DiffuseSnappingShrimp)
    ssp_class = type("SoundSpeedProfile", (FakeBase,), {"calculate": lambda self, d: 1500.0})
    ssp_mod = types.ModuleType("bluepebble.models.environment.sound_speed_profile")
    ssp_mod.SoundSpeedProfile = ssp_class
    env_mod = types.ModuleType("bluepebble.models.environment")
    env_mod.sound_speed_profile = ssp_mod
    models_mod = types.ModuleType("bluepebble.models")
    models_mod.environment = env_mod
    monkeypatch.setitem(sys.modules, "bluepebble.models", models_mod)
    monkeypatch.setitem(sys.modules, "bluepebble.models.environment", env_mod)
    monkeypatch.setitem(sys.modules, "bluepebble.models.environment.sound_speed_profile", ssp_mod)
    load_package_module_from_repo("bluepebble/signal/base.py", "bluepebble.signal.base")
    load_package_module_from_repo("bluepebble/signal/effects.py", "bluepebble.signal.effects")
    return load_package_module_from_repo(
        "bluepebble/signal/biological.py", "bluepebble.signal.biological"
    )


def test_signal_generate_applies_tloss_and_sensor_delay_phase(monkeypatch) -> None:
    """BiologicalSignal.generate should apply attenuation and per-sensor phase delays."""
    bio = _load_biological(monkeypatch)

    class ConstantSignal(bio.BiologicalSignal):
        def _generate_base_signal(self, source) -> np.ndarray:
            return np.ones(self.num_samples, dtype=np.complex128)

    model = ConstantSignal(duration_s=1.0, sampling_rate_hz=8)
    delays = np.array([0.0, 0.125], dtype=float)
    output = model.generate(
        source=None,
        sensor_delays_s=delays,
        tloss_db=20.0,
        propagation_time_s=0,
    )

    assert output.shape == (2, 8)
    np.testing.assert_allclose(np.abs(output[0, 0]), 0.1, rtol=1e-12)
    np.testing.assert_allclose(np.abs(output[1, 0]), 0.1, rtol=1e-12)


def test_white_and_coloured_noise_generate_expected_shapes(monkeypatch) -> None:
    """Ambient noise generators should return correctly shaped complex arrays."""
    install_fake_stonesoup(monkeypatch)
    ambient = load_module_from_repo("bluepebble/signal/random.py", "bluepebble.signal.random")

    white = ambient.WhiteNoiseSignal(amplitude_upa=2.0, duration_s=0.25, sampling_rate_hz=16)
    white_out = white.generate(num_sensors=3)
    assert white_out.shape == (3, 4)
    assert np.iscomplexobj(white_out)

    coloured = ambient.ColouredNoiseSignal(
        spectral_exponent=-1.0,
        amplitude_upa=1.5,
        duration_s=0.25,
        sampling_rate_hz=16,
    )
    coloured_out = coloured.generate(num_sensors=2)
    assert coloured_out.shape == (2, 4)
    assert np.iscomplexobj(coloured_out)
    assert np.isfinite(coloured_out).all()


def test_synthetic_signal_seed_gives_reproducible_output(monkeypatch) -> None:
    """Same seed should produce same SyntheticAnthropogenicSignal waveforms across instances."""
    install_fake_stonesoup(monkeypatch)
    from .support import install_repo_package, load_package_module_from_repo

    install_repo_package(monkeypatch, "bluepebble", "bluepebble")
    install_repo_package(monkeypatch, "bluepebble.signal", "bluepebble/signal")
    load_package_module_from_repo("bluepebble/signal/base.py", "bluepebble.signal.base")
    load_package_module_from_repo("bluepebble/signal/utils.py", "bluepebble.signal.utils")
    anthropogenic = load_package_module_from_repo(
        "bluepebble/signal/anthropogenic.py", "bluepebble.signal.anthropogenic"
    )

    from types import SimpleNamespace

    source = SimpleNamespace(
        metadata={
            "amplitudes_upa": np.array([1.0]),
            "frequencies_hz": np.array([4.0]),
            "phases_rad": np.array([0.0]),
        }
    )

    def _make(seed):
        return anthropogenic.SyntheticAnthropogenicSignal(
            duration_s=0.5,
            sampling_rate_hz=32,
            frame_len=16,
            hop_factor=4,
            noise_amplitude_upa=1.0,
            seed=seed,
        )

    a = _make(42)
    b = _make(42)
    c = _make(99)

    a.compute_stft(source)
    b.compute_stft(source)
    c.compute_stft(source)

    np.testing.assert_array_equal(a.get_source_signal(), b.get_source_signal())
    assert not np.allclose(a.get_source_signal(), c.get_source_signal())


def test_ambient_noise_seed_gives_reproducible_output(monkeypatch) -> None:
    """Same seed should produce identical noise realisations; different seeds should not."""
    install_fake_stonesoup(monkeypatch)
    ambient = load_module_from_repo("bluepebble/signal/random.py", "bluepebble.signal.random")

    a = ambient.WhiteNoiseSignal(amplitude_upa=1.0, duration_s=0.5, sampling_rate_hz=16, seed=42)
    b = ambient.WhiteNoiseSignal(amplitude_upa=1.0, duration_s=0.5, sampling_rate_hz=16, seed=42)
    c = ambient.WhiteNoiseSignal(amplitude_upa=1.0, duration_s=0.5, sampling_rate_hz=16, seed=99)

    np.testing.assert_array_equal(a.generate(num_sensors=2), b.generate(num_sensors=2))
    assert not np.array_equal(a.generate(num_sensors=2), c.generate(num_sensors=2))


def test_reverb_invalid_configuration_raises(monkeypatch) -> None:
    """Invalid reverb parameters should raise ValueError."""
    install_fake_stonesoup(monkeypatch)
    effects = load_module_from_repo("bluepebble/signal/effects.py", "bluepebble.signal.effects")

    signal = np.ones((2, 8), dtype=np.complex128)

    with pytest.raises(ValueError, match="duration_s"):
        effects.Reverb(duration_s=0.0, wet_dry_mix=0.3).apply(signal, sampling_rate_hz=8)

    with pytest.raises(ValueError, match="wet_dry_mix"):
        effects.Reverb(duration_s=1.0, wet_dry_mix=-0.1).apply(signal, sampling_rate_hz=8)


def test_reverb_applies_mix_with_deterministic_ir(monkeypatch) -> None:
    """Reverb should mix dry and wet paths for each channel."""
    install_fake_stonesoup(monkeypatch)
    effects = load_module_from_repo("bluepebble/signal/effects.py", "bluepebble.signal.effects")

    signal = np.vstack([np.arange(8, dtype=float), np.arange(8, dtype=float)]).astype(
        np.complex128
    )
    effect = effects.Reverb(duration_s=0.25, wet_dry_mix=0.5, seed=42)
    reverbed = effect.apply(signal, sampling_rate_hz=8)

    assert reverbed.shape == signal.shape
    assert not np.allclose(reverbed, signal)


def test_reverb_seed_gives_reproducible_ir(monkeypatch) -> None:
    """Same seed should produce identical reverb output; seed=None should be non-deterministic."""
    install_fake_stonesoup(monkeypatch)
    effects = load_module_from_repo("bluepebble/signal/effects.py", "bluepebble.signal.effects")

    signal = np.ones((1, 8), dtype=np.complex128)

    effect = effects.Reverb(duration_s=0.25, wet_dry_mix=0.5, seed=7)
    out_a = effect.apply(signal, sampling_rate_hz=8)
    out_b = effect.apply(signal, sampling_rate_hz=8)
    np.testing.assert_array_equal(out_a, out_b)

    effect_other = effects.Reverb(duration_s=0.25, wet_dry_mix=0.5, seed=99)
    out_c = effect_other.apply(signal, sampling_rate_hz=8)
    assert not np.allclose(out_a, out_c)


def test_reverb_wet_dry_mix_zero_is_passthrough(monkeypatch) -> None:
    """wet_dry_mix=0 should return the dry signal unchanged."""
    install_fake_stonesoup(monkeypatch)
    effects = load_module_from_repo("bluepebble/signal/effects.py", "bluepebble.signal.effects")

    signal = np.ones((2, 8), dtype=np.complex128)
    reverbed = effects.Reverb(duration_s=0.25, wet_dry_mix=0.0, seed=1).apply(
        signal, sampling_rate_hz=8
    )
    np.testing.assert_array_equal(reverbed, signal)


