"""Tests for Signal root properties and Biological base generation helpers."""

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

# ---------------------------------------------------------------------------
# Loader helpers
# ---------------------------------------------------------------------------


def _load_signal_base(monkeypatch):
    """Load ``bluepebble.signal.base`` with lightweight Stone Soup stubs."""
    install_fake_stonesoup(monkeypatch)
    return load_module_from_repo("bluepebble/signal/base.py", "bluepebble.signal.base")


def _install_ssp_stub(monkeypatch) -> type:
    """Install a minimal SoundSpeedProfile stub."""
    ssp_class = type(
        "SoundSpeedProfile",
        (FakeBase,),
        {"calculate": lambda self, depth: 1500.0},
    )
    ssp_module = types.ModuleType("bluepebble.models.environment.sound_speed_profile")
    ssp_module.SoundSpeedProfile = ssp_class
    env_module = types.ModuleType("bluepebble.models.environment")
    env_module.sound_speed_profile = ssp_module
    models_module = types.ModuleType("bluepebble.models")
    models_module.environment = env_module

    monkeypatch.setitem(sys.modules, "bluepebble.models", models_module)
    monkeypatch.setitem(sys.modules, "bluepebble.models.environment", env_module)
    monkeypatch.setitem(
        sys.modules, "bluepebble.models.environment.sound_speed_profile", ssp_module
    )
    return ssp_class


def _load_biological(monkeypatch):
    """Load ``bluepebble.signal.biological`` with lightweight stubs."""
    install_fake_stonesoup(monkeypatch)
    install_repo_package(monkeypatch, "bluepebble", "bluepebble")
    install_repo_package(monkeypatch, "bluepebble.signal", "bluepebble/signal")
    _install_ssp_stub(monkeypatch)
    load_package_module_from_repo("bluepebble/signal/base.py", "bluepebble.signal.base")
    load_package_module_from_repo("bluepebble/signal/effects.py", "bluepebble.signal.effects")
    return load_package_module_from_repo(
        "bluepebble/signal/biological.py", "bluepebble.signal.biological"
    )


# ---------------------------------------------------------------------------
# Signal root class tests
# ---------------------------------------------------------------------------


def test_signal_num_samples(monkeypatch) -> None:
    """``num_samples`` should equal ``int(duration_s * sampling_rate_hz)``."""
    signal_base = _load_signal_base(monkeypatch)
    model = signal_base.Signal(duration_s=0.5, sampling_rate_hz=8)
    assert model.num_samples == 4


def test_signal_is_public_root(monkeypatch) -> None:
    """``Signal`` is the publicly exported root; ``_SignalBase`` is an alias for it."""
    signal_base = _load_signal_base(monkeypatch)
    assert signal_base._SignalBase is signal_base.Signal


# ---------------------------------------------------------------------------
# Biological base-class interface tests
# ---------------------------------------------------------------------------


def test_generate_rejects_non_1d_sensor_delays(monkeypatch) -> None:
    """Sensor delays must be one-dimensional."""
    bio = _load_biological(monkeypatch)

    class ConstantSignal(bio.BiologicalSignal):
        def _generate_base_signal(self, source):
            return np.ones(self.num_samples, dtype=np.complex128)

    model = ConstantSignal(duration_s=1.0, sampling_rate_hz=8)

    with pytest.raises(ValueError, match="sensor_delays_s must be one-dimensional"):
        model.generate(
            source=None,
            sensor_delays_s=np.array([[0.0, 0.1]]),
            tloss_db=0.0,
            propagation_time_s=0.0,
        )


def test_generate_rejects_non_1d_base_signal(monkeypatch) -> None:
    """Generated base signals must be one-dimensional arrays."""
    bio = _load_biological(monkeypatch)

    class MatrixSignal(bio.BiologicalSignal):
        def _generate_base_signal(self, source):
            return np.ones((2, 2), dtype=np.complex128)

    model = MatrixSignal(duration_s=1.0, sampling_rate_hz=8)

    with pytest.raises(ValueError, match="must return a one-dimensional array"):
        model.generate(
            source=None,
            sensor_delays_s=np.array([0.0]),
            tloss_db=0.0,
            propagation_time_s=0.0,
        )


def test_generate_pads_short_base_signal(monkeypatch) -> None:
    """Short base signals should be zero-padded to ``num_samples``."""
    bio = _load_biological(monkeypatch)

    class ShortSignal(bio.BiologicalSignal):
        def _generate_base_signal(self, source):
            return np.array([1.0 + 1.0j, 2.0 + 2.0j], dtype=np.complex128)

    model = ShortSignal(duration_s=0.5, sampling_rate_hz=8)

    with pytest.warns(UserWarning, match="zero-padding"):
        output = model.generate(
            source=None,
            sensor_delays_s=np.array([0.0]),
            tloss_db=0.0,
            propagation_time_s=0.0,
        )

    assert output.shape == (1, 4)
    np.testing.assert_allclose(
        output[0],
        np.array([1.0 + 1.0j, 2.0 + 2.0j, 0.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128),
        rtol=1e-12,
        atol=1e-12,
    )


def test_generate_truncates_long_base_signal(monkeypatch) -> None:
    """Long base signals should be truncated to ``num_samples``."""
    bio = _load_biological(monkeypatch)

    class LongSignal(bio.BiologicalSignal):
        def _generate_base_signal(self, source):
            return np.array(
                [1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j, 4.0 + 0.0j, 5.0 + 0.0j],
                dtype=np.complex128,
            )

    model = LongSignal(duration_s=0.5, sampling_rate_hz=8)

    with pytest.warns(UserWarning, match="truncating"):
        output = model.generate(
            source=None,
            sensor_delays_s=np.array([0.0]),
            tloss_db=0.0,
            propagation_time_s=0.0,
        )

    assert output.shape == (1, 4)
    np.testing.assert_allclose(
        output[0],
        np.array([1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j, 4.0 + 0.0j], dtype=np.complex128),
        rtol=1e-12,
        atol=1e-12,
    )


def test_generate_accepts_frequency_dependent_tloss_vector(monkeypatch) -> None:
    """Frequency-dependent transmission-loss arrays should be accepted when sized correctly."""
    bio = _load_biological(monkeypatch)

    class ConstantSignal(bio.BiologicalSignal):
        def _generate_base_signal(self, source):
            return np.array(
                [1.0 + 0.0j, 0.5 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128
            )

    model = ConstantSignal(duration_s=0.5, sampling_rate_hz=8)

    output = model.generate(
        source=None,
        sensor_delays_s=np.array([0.0]),
        tloss_db=np.zeros(model.num_samples, dtype=float),
        propagation_time_s=0.0,
    )

    np.testing.assert_allclose(
        output[0],
        np.array([1.0 + 0.0j, 0.5 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128),
        rtol=1e-12,
        atol=1e-12,
    )


def test_generate_rejects_tloss_vector_length_mismatch(monkeypatch) -> None:
    """Frequency-dependent transmission-loss arrays must match ``num_samples``."""
    bio = _load_biological(monkeypatch)

    class ConstantSignal(bio.BiologicalSignal):
        def _generate_base_signal(self, source):
            return np.ones(self.num_samples, dtype=np.complex128)

    model = ConstantSignal(duration_s=0.5, sampling_rate_hz=8)

    with pytest.raises(ValueError, match="must have length equal to num_samples"):
        model.generate(
            source=None,
            sensor_delays_s=np.array([0.0]),
            tloss_db=np.array([1.0, 2.0, 3.0]),
            propagation_time_s=0.0,
        )


def test_generate_rejects_multidimensional_tloss(monkeypatch) -> None:
    """Transmission loss must be scalar-like or one-dimensional."""
    bio = _load_biological(monkeypatch)

    class ConstantSignal(bio.BiologicalSignal):
        def _generate_base_signal(self, source):
            return np.ones(self.num_samples, dtype=np.complex128)

    model = ConstantSignal(duration_s=0.5, sampling_rate_hz=8)

    with pytest.raises(ValueError, match="tloss_db must be scalar-like or one-dimensional"):
        model.generate(
            source=None,
            sensor_delays_s=np.array([0.0]),
            tloss_db=np.zeros((1, model.num_samples), dtype=float),
            propagation_time_s=0.0,
        )


# ---------------------------------------------------------------------------
# Hierarchy verification tests
# ---------------------------------------------------------------------------


def test_hierarchy_biological_is_subclass_of_signal(monkeypatch) -> None:
    """``Biological`` must be a subtype of ``Signal``."""
    bio = _load_biological(monkeypatch)
    assert issubclass(bio.BiologicalSignal, bio.Signal)


def test_hierarchy_signal_root_has_no_generate(monkeypatch) -> None:
    """``Signal`` is the unified root — it does not expose a ``generate()`` method."""
    signal_base = _load_signal_base(monkeypatch)
    assert not hasattr(signal_base.Signal, "generate")


def test_hierarchy_biological_exposes_generate(monkeypatch) -> None:
    """``Biological`` exposes the per-timestep ``generate()`` interface."""
    bio = _load_biological(monkeypatch)
    assert hasattr(bio.BiologicalSignal, "generate")
