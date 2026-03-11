"""Coverage-oriented tests for signal models and Bellhop utilities."""

from __future__ import annotations

import builtins
from pathlib import Path

import numpy as np

from .support import install_fake_stonesoup, load_module_from_repo


def test_signal_generate_applies_tloss_and_sensor_delay_phase(monkeypatch) -> None:
    """Signal generation should apply attenuation and per-sensor phase delays."""
    install_fake_stonesoup(monkeypatch)
    signal_base = load_module_from_repo("bluepebble/signal/base.py", "bluepebble.signal.base")

    class ConstantSignal(signal_base.Signal):
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
    ambient = load_module_from_repo("bluepebble/signal/ambient.py", "bluepebble.signal.ambient")

    white = ambient.WhiteNoise(amplitude_upa=2.0, duration_s=0.25, sampling_rate_hz=16)
    white_out = white.generate(num_sensors=3)
    assert white_out.shape == (3, 4)
    assert np.iscomplexobj(white_out)

    coloured = ambient.ColouredNoise(
        spectral_exponent=-1.0,
        amplitude_upa=1.5,
        duration_s=0.25,
        sampling_rate_hz=16,
    )
    coloured_out = coloured.generate(num_sensors=2)
    assert coloured_out.shape == (2, 4)
    assert np.iscomplexobj(coloured_out)
    assert np.isfinite(coloured_out).all()


def test_reverb_invalid_configuration_returns_input(monkeypatch) -> None:
    """Invalid reverb parameters should short-circuit and return the original signal."""
    install_fake_stonesoup(monkeypatch)
    effects = load_module_from_repo("bluepebble/signal/effects.py", "bluepebble.signal.effects")

    signal = np.ones((2, 8), dtype=np.complex128)
    effect = effects.Reverb(duration_s=0.0, wet_dry_mix=0.3)

    returned = effect.apply(signal, sampling_rate_hz=8)
    assert returned is signal


def test_reverb_applies_mix_with_deterministic_ir(monkeypatch) -> None:
    """Reverb should mix dry and wet paths for each channel."""
    install_fake_stonesoup(monkeypatch)
    effects = load_module_from_repo("bluepebble/signal/effects.py", "bluepebble.signal.effects")

    monkeypatch.setattr(effects.np.random, "randn", lambda n: np.ones(n, dtype=float))

    signal = np.vstack([np.arange(8, dtype=float), np.arange(8, dtype=float)]).astype(
        np.complex128
    )
    effect = effects.Reverb(duration_s=0.25, wet_dry_mix=0.5)
    reverbed = effect.apply(signal, sampling_rate_hz=8)

    assert reverbed.shape == signal.shape
    assert not np.allclose(reverbed, signal)


class _FakeShadeFile:
    """A tiny binary file stub supporting seek/read for test doubles."""

    def __init__(self, plot_type: str):
        self._pos = 0
        self._plot_type = plot_type

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def seek(self, offset: int, whence: int = 0):
        if whence == 0:
            self._pos = offset
        elif whence == 1:
            self._pos += offset
        else:
            raise ValueError("Only SEEK_SET/SEEK_CUR are supported in this test stub")

    def tell(self) -> int:
        return self._pos

    def read(self, size: int) -> bytes:
        _ = size
        return self._plot_type.encode("utf-8").ljust(10, b" ")


def test_read_shade_file_reads_standard_full_field(monkeypatch) -> None:
    """Standard shade files should parse geometry and pressure field."""
    bellhop = load_module_from_repo("bluepebble/utils/bellhop.py", "bluepebble.utils.bellhop")

    fake_file = _FakeShadeFile(plot_type="rect")
    monkeypatch.setattr(builtins, "open", lambda *_args, **_kwargs: fake_file)

    calls = iter(
        [
            np.array([16], dtype=np.int32),
            np.array([1, 1, 1, 1, 1, 1, 2], dtype=np.int32),
            np.array([100.0], dtype=np.float32),
            np.array([0.0], dtype=np.float32),
            np.array([1000.0], dtype=np.float32),
            np.array([2000.0], dtype=np.float32),
            np.array([50.0], dtype=np.float32),
            np.array([100.0], dtype=np.float32),
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([1 + 1j, 2 + 2j], dtype=np.complex64),
        ]
    )

    monkeypatch.setattr(bellhop.np, "fromfile", lambda *_args, **_kwargs: next(calls))
    pressure, geometry = bellhop.read_shade_file(Path("dummy.shd"))

    assert pressure.shape == (1, 1, 1, 2)
    np.testing.assert_allclose(pressure[0, 0, 0], np.array([1 + 1j, 2 + 2j], dtype=np.complex64))
    assert geometry["plot_type"] == "rect"
    np.testing.assert_allclose(geometry["source_x"], np.array([1000.0], dtype=np.float32))


def test_read_shade_file_reads_tl_slice_by_source_position(monkeypatch) -> None:
    """Compressed TL shade files should support nearest-source slice extraction."""
    bellhop = load_module_from_repo("bluepebble/utils/bellhop.py", "bluepebble.utils.bellhop")

    fake_file = _FakeShadeFile(plot_type="TL")
    monkeypatch.setattr(builtins, "open", lambda *_args, **_kwargs: fake_file)

    calls = iter(
        [
            np.array([16], dtype=np.int32),
            np.array([1, 1, 2, 2, 1, 1, 2], dtype=np.int32),
            np.array([200.0], dtype=np.float32),
            np.array([0.0], dtype=np.float32),
            np.array([0.0, 2000.0], dtype=np.float32),
            np.array([0.0, 2000.0], dtype=np.float32),
            np.array([20.0], dtype=np.float32),
            np.array([30.0], dtype=np.float32),
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([3 + 4j, 5 + 6j], dtype=np.complex64),
        ]
    )

    monkeypatch.setattr(bellhop.np, "fromfile", lambda *_args, **_kwargs: next(calls))
    pressure, geometry = bellhop.read_shade_file(Path("dummy_tl.shd"), xs=1.6, ys=0.1)

    assert pressure.shape == (1, 1, 1, 2)
    np.testing.assert_allclose(pressure[0, 0, 0], np.array([3 + 4j, 5 + 6j], dtype=np.complex64))
    np.testing.assert_allclose(geometry["source_x"], np.array([0.0, 2000.0]))
    np.testing.assert_allclose(geometry["source_y"], np.array([0.0, 2000.0]))


def test_read_shade_file_returns_none_pair_for_missing_file(monkeypatch) -> None:
    """Missing files should be handled gracefully by returning ``(None, None)``."""
    bellhop = load_module_from_repo("bluepebble/utils/bellhop.py", "bluepebble.utils.bellhop")

    def _raise_file_not_found(*_args, **_kwargs):
        raise FileNotFoundError("missing")

    monkeypatch.setattr(builtins, "open", _raise_file_not_found)

    pressure, geometry = bellhop.read_shade_file(Path("missing.shd"))
    assert pressure is None
    assert geometry is None
