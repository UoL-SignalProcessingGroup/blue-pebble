"""Tests for deterministic acoustic propagation model behaviour."""

from __future__ import annotations

import importlib.util
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from .support import (
    install_fake_stonesoup,
    install_repo_package,
    load_package_module_from_repo,
)


def _load_propagation_module(monkeypatch):
    """Load acoustic propagation models with minimal dependency scaffolding."""
    install_fake_stonesoup(monkeypatch)
    install_repo_package(monkeypatch, "bluepebble", "bluepebble")
    install_repo_package(monkeypatch, "bluepebble.models", "bluepebble/models")
    install_repo_package(
        monkeypatch,
        "bluepebble.models.propagation",
        "bluepebble/models/propagation",
    )

    utils_module = ModuleType("bluepebble.utils")
    utils_module.read_shade_file = lambda path: (np.array([1.0 + 0.0j]), None)

    environment_module = ModuleType("bluepebble.models.environment")
    environment_module.Bathymetry = type("Bathymetry", (), {})
    environment_module.SoundSpeedProfile = type("SoundSpeedProfile", (), {})

    monkeypatch.setitem(sys.modules, "bluepebble.utils", utils_module)
    monkeypatch.setitem(sys.modules, "bluepebble.models.environment", environment_module)

    return load_package_module_from_repo(
        "bluepebble/models/propagation/acoustic.py",
        "bluepebble.models.propagation.acoustic",
    )


class ConstantSSP:
    """Simple constant sound-speed profile for tests."""

    def __init__(self, speed: float):
        """Store the constant sound speed to return for any depth input."""
        self.speed = speed

    def calculate(self, depth):
        """Return a scalar or array filled with the configured speed."""
        depth_array = np.asarray(depth)
        if depth_array.ndim == 0:
            return float(self.speed)
        return np.full(depth_array.shape, self.speed, dtype=float)


class GridSSP(ConstantSSP):
    """Constant SSP with a simple gridded representation for rtrs tests."""

    def get_3d_grid(self, x_range, y_range, z_range, x_res, y_res, z_res):
        """Return a small constant grid and record the requested ranges if needed."""
        x_grid = np.array([x_range[0], x_range[1]], dtype=float)
        y_grid = np.array([y_range[0], y_range[1]], dtype=float)
        z_grid = np.array([z_range[0], z_range[1]], dtype=float)
        c_grid = np.full((len(x_grid) * len(y_grid) * len(z_grid),), self.speed, dtype=float)
        return x_grid, y_grid, z_grid, c_grid


class FlatNegativeBathymetry:
    """Minimal bathymetry model returning a constant negative depth."""

    def __init__(self, depth: float = -200.0):
        """Store the constant seafloor depth."""
        self.depth = depth

    def get_depth(self, x: float, y: float) -> float:
        """Return the same depth at every position."""
        return self.depth

    def get_grid(self, x_range, y_range):
        """Return a two-by-two constant bathymetry grid."""
        x_grid = np.array([x_range[0], x_range[1]], dtype=float)
        y_grid = np.array([y_range[0], y_range[1]], dtype=float)
        z_grid = np.full((2, 2), self.depth, dtype=float)
        return x_grid, y_grid, z_grid


def _make_platform_and_source():
    """Create a simple two-sensor geometry and a source state."""
    from datetime import datetime

    from stonesoup.types.array import StateVector
    from stonesoup.types.state import State

    ts = datetime(2024, 1, 1)

    array_state = np.array([[3.0, 6.0], [4.0, 8.0], [-10.0, -10.0]])
    el0 = State(state_vector=StateVector([3.0, 4.0, -10.0]), timestamp=ts)
    el1 = State(state_vector=StateVector([6.0, 8.0, -10.0]), timestamp=ts)

    sensor_array = SimpleNamespace(
        position_matrix_at=lambda t: array_state,
        element_states_at=lambda t: [el0, el1],
        reference_element_idx=0,
    )
    platform = SimpleNamespace(sensor_array=sensor_array)
    source = SimpleNamespace(
        state_vector=np.array([[0.0], [0.0], [-10.0]]),
        timestamp=ts,
        metadata={
            "position_mapping": [0, 1, 2],
            "frequencies_hz": np.array([100.0, 250.0]),
            "amplitudes_upa": np.array([0.1, 0.9]),
        },
    )
    return platform, source


def test_compute_sensor_delays_uses_reference_sensor_and_ssp(monkeypatch) -> None:
    """Sensor delays should be relative to the reference sensor travel time."""
    propagation = _load_propagation_module(monkeypatch)
    platform, source = _make_platform_and_source()
    model = propagation.CylindricalAcousticPropagationModel(ssp=ConstantSSP(1500.0))

    delays = model.compute_sensor_delays(platform, source)

    np.testing.assert_allclose(delays, np.array([0.0, (10.0 - 5.0) / 1500.0]))


def test_cylindrical_propagate_matches_analytic_formula(monkeypatch) -> None:
    """Cylindrical spreading loss and time should follow the documented formula."""
    propagation = _load_propagation_module(monkeypatch)
    platform, source = _make_platform_and_source()
    model = propagation.CylindricalAcousticPropagationModel(
        ssp=ConstantSSP(1500.0),
        attenuation_factor=0.5,
    )

    tloss, travel_time = model.propagate(platform, source)

    assert tloss == pytest.approx(10.0 * np.log10(5.0) + 0.5 * (5.0 / 1000.0))
    assert travel_time == pytest.approx(5.0 / 1500.0)


def test_cylindrical_propagate_spectrum_returns_expected_shape_and_amplitude(monkeypatch) -> None:
    """Spectrum propagation should return one transfer function per sensor and frequency."""
    propagation = _load_propagation_module(monkeypatch)
    platform, source = _make_platform_and_source()
    model = propagation.CylindricalAcousticPropagationModel(
        ssp=ConstantSSP(1500.0),
        attenuation_factor=0.0,
    )

    transfer, travel_time = model.propagate_spectrum(
        platform,
        source,
        frequencies_hz=np.array([0.0, 100.0]),
    )

    assert transfer.shape == (2, 2)
    np.testing.assert_allclose(
        transfer[:, 0].real,
        np.array([1 / np.sqrt(5.0), 1 / np.sqrt(10.0)]),
    )
    np.testing.assert_allclose(transfer[:, 0].imag, np.array([0.0, 0.0]))
    assert travel_time == pytest.approx(5.0 / 1500.0)


def test_spherical_propagate_matches_analytic_formula(monkeypatch) -> None:
    """Spherical spreading loss and time should follow the documented formula."""
    propagation = _load_propagation_module(monkeypatch)
    platform, source = _make_platform_and_source()
    model = propagation.SphericalAcousticPropagationModel(
        ssp=ConstantSSP(1500.0),
        attenuation_factor=0.001,
    )

    tloss, travel_time = model.propagate(platform, source)

    assert tloss == pytest.approx(20.0 * np.log10(5.0) + 0.001 * (5.0 / 1000.0))
    assert travel_time == pytest.approx(5.0 / 1500.0)


def test_spherical_propagate_spectrum_returns_inverse_distance_amplitude(monkeypatch) -> None:
    """Zero-frequency spherical transfer magnitude should equal inverse distance."""
    propagation = _load_propagation_module(monkeypatch)
    platform, source = _make_platform_and_source()
    model = propagation.SphericalAcousticPropagationModel(
        ssp=ConstantSSP(1500.0),
        attenuation_factor=0.0,
    )

    transfer, _travel_time = model.propagate_spectrum(
        platform,
        source,
        frequencies_hz=np.array([0.0]),
    )

    np.testing.assert_allclose(transfer[:, 0].real, np.array([1 / 5.0, 1 / 10.0]))
    np.testing.assert_allclose(transfer[:, 0].imag, np.array([0.0, 0.0]))


def test_negative_attenuation_is_rejected(monkeypatch) -> None:
    """Analytic propagation models should reject negative attenuation factors."""
    propagation = _load_propagation_module(monkeypatch)

    with pytest.raises(ValueError, match="non-negative"):
        propagation.CylindricalAcousticPropagationModel(
            ssp=ConstantSSP(1500.0),
            attenuation_factor=-0.1,
        )

    with pytest.raises(ValueError, match="non-negative"):
        propagation.SphericalAcousticPropagationModel(
            ssp=ConstantSSP(1500.0),
            attenuation_factor=-0.1,
        )


def test_rtrs_validation_and_helper_methods(monkeypatch) -> None:
    """Rtrs helper methods should validate options and compute derived values predictably."""
    propagation = _load_propagation_module(monkeypatch)
    bathymetry = SimpleNamespace()

    with pytest.raises(ValueError, match="integration_method"):
        propagation.rtrsAcousticPropagationModel(
            ssp=ConstantSSP(1500.0),
            bathymetry=bathymetry,
            integration_method="bad",
        )

    with pytest.raises(ValueError, match="bottom_model"):
        propagation.rtrsAcousticPropagationModel(
            ssp=ConstantSSP(1500.0),
            bathymetry=bathymetry,
            bottom_model="rigid",
        )

    model = propagation.rtrsAcousticPropagationModel(
        ssp=ConstantSSP(1500.0),
        bathymetry=bathymetry,
        step_m=20.0,
        azimuth_search_width=2.0,
        azimuth_resolution=1.0,
        bottom_model={"model": "acoustic", "density": 1.5},
    )

    assert model._resolved_bottom_model() == {"model": "acoustic", "density": 1.5}
    assert model._calculate_max_steps_and_range(100.0) == (1006, 120.0)
    assert model._calculate_launch_azimuths(
        source_position=np.array([0.0, 0.0, 0.0]),
        receiver_position=np.array([1.0, 0.0, 0.0]),
    ) == pytest.approx([91.0, 90.0, 89.0])


def test_rtrs_methods_raise_clear_error_when_package_missing(monkeypatch) -> None:
    """rtrs-backed methods should fail clearly when the optional dependency is absent."""
    if importlib.util.find_spec("rtrs") is not None:
        pytest.skip("rtrs is installed in this environment")

    propagation = _load_propagation_module(monkeypatch)
    platform, source = _make_platform_and_source()
    monkeypatch.delitem(sys.modules, "rtrs", raising=False)

    model = propagation.rtrsAcousticPropagationModel(
        ssp=GridSSP(1500.0),
        bathymetry=FlatNegativeBathymetry(-200.0),
    )

    with pytest.raises(ImportError, match="rtrs package is not installed"):
        model.propagate(platform, source)

    with pytest.raises(ImportError, match="rtrs package is not installed"):
        model.propagate_spectrum(platform, source, frequencies_hz=np.array([100.0]))


def test_rtrs_propagate_builds_single_receiver_config_and_computes_tloss(monkeypatch) -> None:
    """Mocked rtrs propagation should build a single-receiver config and decode pressure."""
    propagation = _load_propagation_module(monkeypatch)
    platform, source = _make_platform_and_source()
    captured = {}

    def fake_run_simulation(env_config):
        captured["env_config"] = env_config
        return {
            "pressure_field": {
                "shape": [1, 1, 1, 1],
                "pressure_re": [0.5],
                "pressure_im": [0.0],
            }
        }

    fake_rtrs = ModuleType("rtrs")
    fake_rtrs.run_simulation = fake_run_simulation
    monkeypatch.setitem(sys.modules, "rtrs", fake_rtrs)

    model = propagation.rtrsAcousticPropagationModel(
        ssp=GridSSP(1500.0),
        bathymetry=FlatNegativeBathymetry(-200.0),
        step_m=20.0,
    )

    tloss, travel_time = model.propagate(platform, source)

    assert tloss == pytest.approx(6.020599913279624)
    assert travel_time == pytest.approx(5.0 / 1500.0)
    env_config = captured["env_config"]
    assert env_config["source"]["freq_hz"] == [250.0]
    assert env_config["receivers"]["x_rcvr_m"] == [3.0]
    assert env_config["receivers"]["y_rcvr_m"] == [4.0]
    assert env_config["receivers"]["z_rcvr_m"] == [10.0]
    assert env_config["beam"]["integration_method"] == "euler"


def test_rtrs_propagate_returns_per_frequency_tloss_when_requested(monkeypatch) -> None:
    """Mocked rtrs propagation should return a TL value per frequency when enabled."""
    propagation = _load_propagation_module(monkeypatch)
    platform, source = _make_platform_and_source()

    fake_rtrs = ModuleType("rtrs")
    fake_rtrs.run_simulation = lambda env_config: {
        "pressure_field": {
            "shape": [2, 1, 1, 1],
            "pressure_re": [0.1, 0.0],
            "pressure_im": [0.0, 0.0],
        }
    }
    monkeypatch.setitem(sys.modules, "rtrs", fake_rtrs)

    model = propagation.rtrsAcousticPropagationModel(
        ssp=GridSSP(1500.0),
        bathymetry=FlatNegativeBathymetry(-200.0),
        use_all_frequencies=True,
    )

    tloss, travel_time = model.propagate(platform, source)

    np.testing.assert_allclose(tloss, np.array([20.0, 999.0]))
    assert travel_time == pytest.approx(5.0 / 1500.0)


def test_rtrs_propagate_returns_single_frequency_zero_pressure_fallback(monkeypatch) -> None:
    """Single-frequency rtrs propagation should also use the 999.0 fallback for zero pressure."""
    propagation = _load_propagation_module(monkeypatch)
    platform, source = _make_platform_and_source()

    fake_rtrs = ModuleType("rtrs")
    fake_rtrs.run_simulation = lambda env_config: {
        "pressure_field": {
            "shape": [1, 1, 1, 1],
            "pressure_re": [0.0],
            "pressure_im": [0.0],
        }
    }
    monkeypatch.setitem(sys.modules, "rtrs", fake_rtrs)

    model = propagation.rtrsAcousticPropagationModel(
        ssp=GridSSP(1500.0),
        bathymetry=FlatNegativeBathymetry(-200.0),
    )

    tloss, travel_time = model.propagate(platform, source)

    assert tloss == pytest.approx(999.0)
    assert travel_time == pytest.approx(5.0 / 1500.0)


def test_rtrs_propagate_spectrum_transposes_and_conjugates_transfer_functions(
    monkeypatch,
) -> None:
    """Mocked spectrum propagation should return transposed, conjugated transfer functions."""
    propagation = _load_propagation_module(monkeypatch)
    platform, source = _make_platform_and_source()
    captured = {}

    def fake_run_simulation(env_config):
        captured["env_config"] = env_config
        return {
            "pressure_field": {
                "shape": [2, 2, 1, 1],
                "pressure_re": [1.0, 2.0, 3.0, 4.0],
                "pressure_im": [10.0, 20.0, 30.0, 40.0],
            }
        }

    fake_rtrs = ModuleType("rtrs")
    fake_rtrs.run_simulation = fake_run_simulation
    monkeypatch.setitem(sys.modules, "rtrs", fake_rtrs)

    model = propagation.rtrsAcousticPropagationModel(
        ssp=GridSSP(1500.0),
        bathymetry=FlatNegativeBathymetry(-200.0),
    )

    transfer, travel_time = model.propagate_spectrum(
        platform,
        source,
        frequencies_hz=np.array([100.0, 250.0]),
    )

    assert transfer.shape == (2, 2)
    np.testing.assert_array_equal(
        transfer,
        np.array(
            [
                [1.0 - 10.0j, 3.0 - 30.0j],
                [2.0 - 20.0j, 4.0 - 40.0j],
            ]
        ),
    )
    assert travel_time == pytest.approx(5.0 / 1500.0)
    env_config = captured["env_config"]
    assert env_config["receivers"]["x_rcvr_m"] == [3.0, 6.0]
    assert env_config["receivers"]["z_rcvr_m"] == [10.0, 10.0]
    assert env_config["source"]["freq_hz"] == [100.0, 250.0]


def test_rtrs_propagate_runs_with_real_backend(monkeypatch) -> None:
    """The real rtrs backend should return a finite scalar TL and travel time."""
    pytest.importorskip("rtrs")
    propagation = _load_propagation_module(monkeypatch)
    platform, source = _make_platform_and_source()
    model = propagation.rtrsAcousticPropagationModel(
        ssp=GridSSP(1500.0),
        bathymetry=FlatNegativeBathymetry(-200.0),
        step_m=20.0,
    )

    tloss, travel_time = model.propagate(platform, source)

    assert np.isscalar(tloss)
    assert np.isfinite(float(tloss))
    assert float(tloss) > 0.0
    assert travel_time == pytest.approx(5.0 / 1500.0)


def test_rtrs_propagate_per_frequency_runs_with_real_backend(monkeypatch) -> None:
    """The real rtrs backend should return finite per-frequency TL values when requested."""
    pytest.importorskip("rtrs")
    propagation = _load_propagation_module(monkeypatch)
    platform, source = _make_platform_and_source()
    model = propagation.rtrsAcousticPropagationModel(
        ssp=GridSSP(1500.0),
        bathymetry=FlatNegativeBathymetry(-200.0),
        use_all_frequencies=True,
    )

    tloss, travel_time = model.propagate(platform, source)

    assert tloss.shape == (2,)
    assert np.isfinite(tloss).all()
    assert np.all(tloss > 0.0)
    assert travel_time == pytest.approx(5.0 / 1500.0)


def test_rtrs_propagate_spectrum_runs_with_real_backend(monkeypatch) -> None:
    """The real rtrs backend should return finite complex transfer functions."""
    pytest.importorskip("rtrs")
    propagation = _load_propagation_module(monkeypatch)
    platform, source = _make_platform_and_source()
    model = propagation.rtrsAcousticPropagationModel(
        ssp=GridSSP(1500.0),
        bathymetry=FlatNegativeBathymetry(-200.0),
    )

    transfer, travel_time = model.propagate_spectrum(
        platform,
        source,
        frequencies_hz=np.array([100.0, 250.0]),
    )

    assert transfer.shape == (2, 2)
    assert np.isfinite(transfer).all()
    assert np.any(np.abs(transfer) > 0.0)
    assert travel_time == pytest.approx(5.0 / 1500.0)
