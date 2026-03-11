"""Tests for base signal generation helpers and validation branches."""

from __future__ import annotations

import numpy as np
import pytest

from .support import install_fake_stonesoup, load_module_from_repo


def _load_signal_base(monkeypatch):
    """Load ``bluepebble.signal.base`` with lightweight Stone Soup stubs."""
    install_fake_stonesoup(monkeypatch)
    return load_module_from_repo("bluepebble/signal/base.py", "bluepebble.signal.base")


def test_generate_rejects_non_1d_sensor_delays(monkeypatch) -> None:
    """Sensor delays must be one-dimensional."""
    signal_base = _load_signal_base(monkeypatch)

    class ConstantSignal(signal_base.Signal):
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


def test_generate_requires_subclass_signal_generator(monkeypatch) -> None:
    """Base class should require subclasses to implement a signal generator."""
    signal_base = _load_signal_base(monkeypatch)

    model = signal_base.Signal(duration_s=1.0, sampling_rate_hz=8)

    with pytest.raises(NotImplementedError, match=r"must implement either generate\(\) or"):
        model.generate(
            source=None,
            sensor_delays_s=np.array([0.0]),
            tloss_db=0.0,
            propagation_time_s=0.0,
        )


def test_generate_rejects_non_1d_base_signal(monkeypatch) -> None:
    """Generated base signals must be one-dimensional arrays."""
    signal_base = _load_signal_base(monkeypatch)

    class MatrixSignal(signal_base.Signal):
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
    signal_base = _load_signal_base(monkeypatch)

    class ShortSignal(signal_base.Signal):
        def _generate_base_signal(self, source):
            return np.array([1.0 + 1.0j, 2.0 + 2.0j], dtype=np.complex128)

    model = ShortSignal(duration_s=0.5, sampling_rate_hz=8)

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
    signal_base = _load_signal_base(monkeypatch)

    class LongSignal(signal_base.Signal):
        def _generate_base_signal(self, source):
            return np.array(
                [1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j, 4.0 + 0.0j, 5.0 + 0.0j],
                dtype=np.complex128,
            )

    model = LongSignal(duration_s=0.5, sampling_rate_hz=8)

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
    signal_base = _load_signal_base(monkeypatch)

    class ConstantSignal(signal_base.Signal):
        def _generate_base_signal(self, source):
            return np.array([1.0 + 0.0j, 0.5 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)

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
    signal_base = _load_signal_base(monkeypatch)

    class ConstantSignal(signal_base.Signal):
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
    signal_base = _load_signal_base(monkeypatch)

    class ConstantSignal(signal_base.Signal):
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
