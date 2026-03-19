"""Tests for deterministic acoustic propagation model behaviour."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from .support import (
    install_fake_stonesoup,
    install_repo_package,
    load_module_from_repo,
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
    array_state = np.array(
        [
            [3.0, 6.0],
            [4.0, 8.0],
            [-10.0, -10.0],
        ]
    )
    ref_state = array_state[:, [0]]
    platform = SimpleNamespace(
        array=SimpleNamespace(state_vector=array_state, ref_state_vector=ref_state)
    )
    source = SimpleNamespace(
        state_vector=np.array([[0.0], [0.0], [-10.0]]),
        metadata={
            "position_mapping": [0, 1, 2],
            "frequencies_hz": np.array([100.0, 250.0]),
            "amplitudes_upa": np.array([0.1, 0.9]),
        },
    )
    return platform, source


def _project_bellhop_executable() -> Path:
    """Return the repo-local Bellhop executable path or skip if it is unavailable."""
    exe_path = (
        Path(__file__).resolve().parents[1]
        / "external_tools"
        / "bellhopcuda"
        / "bin"
        / "bellhopcxx"
    )
    if not exe_path.exists() or not exe_path.is_file():
        pytest.skip("project-local bellhopcxx executable is not available")
    return exe_path


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


def test_bellhop_executable_resolution_uses_shutil_which(monkeypatch) -> None:
    """Bellhop path resolution should return the discovered executable path."""
    propagation = _load_propagation_module(monkeypatch)
    monkeypatch.setattr(propagation, "which", lambda exe_name: f"/usr/local/bin/{exe_name}")

    resolved = propagation.BellhopAcousticPropagationModel._resolve_bellhop_executable(
        "bellhopcxx"
    )

    assert resolved == "/usr/local/bin/bellhopcxx"


def test_bellhop_executable_resolution_raises_when_missing(monkeypatch) -> None:
    """Missing Bellhop executables should raise a clear file-not-found error."""
    propagation = _load_propagation_module(monkeypatch)
    monkeypatch.setattr(propagation, "which", lambda exe_name: None)

    with pytest.raises(FileNotFoundError, match="Bellhop executable"):
        propagation.BellhopAcousticPropagationModel._resolve_bellhop_executable("bellhopcxx")


def test_bellhop_create_env_file_writes_expected_default_content(monkeypatch, tmp_path) -> None:
    """Bellhop env-file generation should encode the default geometry and source settings."""
    propagation = _load_propagation_module(monkeypatch)
    monkeypatch.setattr(propagation, "which", lambda exe_name: f"/usr/local/bin/{exe_name}")

    platform, source = _make_platform_and_source()
    model = propagation.BellhopAcousticPropagationModel(
        env_depth=200.0,
        ssp=ConstantSSP(1500.0),
    )

    model._create_env_file(platform, source, output_dir=tmp_path)

    env_text = (tmp_path / "env.env").read_text(encoding="utf-8")

    assert "'env'" in env_text
    assert "250.0" in env_text  # loudest source frequency
    assert "'SVW'" in env_text
    assert "'Cb'" in env_text
    assert "10.0 /" in env_text  # source and receiver depths
    assert "0.0 0.005 /" in env_text  # max range in km for 5 m source-reference distance
    assert "0.0 240.0 0.01" in env_text  # plot-depth and range margin line


def test_bellhop_create_env_file_honours_custom_options(monkeypatch, tmp_path) -> None:
    """Bellhop env-file generation should respect caller-supplied run configuration."""
    propagation = _load_propagation_module(monkeypatch)
    monkeypatch.setattr(propagation, "which", lambda exe_name: f"/usr/local/bin/{exe_name}")

    platform, source = _make_platform_and_source()
    model = propagation.BellhopAcousticPropagationModel(
        env_depth=300.0,
        ssp=ConstantSSP(1480.0),
    )

    model._create_env_file(
        platform,
        source,
        output_dir=tmp_path,
        options="CVW",
        bottom_bc="R",
        runtype="Ab",
        nbeams=21,
        beam_angles=[-30.0, 45.0],
    )

    env_text = (tmp_path / "env.env").read_text(encoding="utf-8")

    assert "'CVW'" in env_text
    assert "'R' 0.0" in env_text
    assert "'Ab'" in env_text
    assert "21" in env_text
    assert "-30.0 45.0 /" in env_text
    assert "0.0 1480.0  /" in env_text
    assert "300.0 1700.0 0.0 1.5 0.5 /" in env_text


def test_bellhop_propagate_returns_large_loss_for_zero_pressure(monkeypatch) -> None:
    """Bellhop propagation should convert zero pressure into the finite fallback loss."""
    propagation = _load_propagation_module(monkeypatch)
    monkeypatch.setattr(propagation, "which", lambda exe_name: f"/usr/local/bin/{exe_name}")
    monkeypatch.setattr(propagation.subprocess, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        propagation,
        "read_shade_file",
        lambda path: (np.array([0.0 + 0.0j, 0.0 + 0.0j]), None),
    )

    platform = SimpleNamespace(array=SimpleNamespace(ref_state_vector=np.array([3.0, 4.0, -10.0])))
    source = SimpleNamespace(
        state_vector=np.array([0.0, 0.0, 0.0, 0.0, -10.0]),
        metadata={
            "position_mapping": [0, 2, 4],
            "frequencies_hz": np.array([100.0, 250.0]),
            "amplitudes_upa": np.array([0.1, 0.9]),
        },
    )
    model = propagation.BellhopAcousticPropagationModel(
        env_depth=200.0,
        ssp=ConstantSSP(1500.0),
    )

    tloss, travel_time = model.propagate(platform, source)

    assert tloss == pytest.approx(999.0)
    assert travel_time == pytest.approx(5.0 / 1500.0)


def test_bellhop_propagate_reraises_subprocess_errors(monkeypatch) -> None:
    """Bellhop subprocess failures should be surfaced to the caller unchanged."""
    propagation = _load_propagation_module(monkeypatch)
    monkeypatch.setattr(propagation, "which", lambda exe_name: f"/usr/local/bin/{exe_name}")

    def fake_run(*args, **kwargs):
        raise propagation.subprocess.CalledProcessError(
            returncode=1,
            cmd=args[0],
            output="stdout",
            stderr="stderr",
        )

    monkeypatch.setattr(propagation.subprocess, "run", fake_run)
    platform = SimpleNamespace(array=SimpleNamespace(ref_state_vector=np.array([3.0, 4.0, -10.0])))
    source = SimpleNamespace(
        state_vector=np.array([0.0, 0.0, 0.0, 0.0, -10.0]),
        metadata={
            "position_mapping": [0, 2, 4],
            "frequencies_hz": np.array([100.0, 250.0]),
            "amplitudes_upa": np.array([0.1, 0.9]),
        },
    )
    model = propagation.BellhopAcousticPropagationModel(
        env_depth=200.0,
        ssp=ConstantSSP(1500.0),
    )

    with pytest.raises(propagation.subprocess.CalledProcessError):
        model.propagate(platform, source)


def test_bellhop_propagate_invokes_executable_and_reads_shade_file(monkeypatch) -> None:
    """Bellhop propagation should call the executable and read the expected shade file."""
    propagation = _load_propagation_module(monkeypatch)
    monkeypatch.setattr(propagation, "which", lambda exe_name: f"/usr/local/bin/{exe_name}")
    captured = {}

    def fake_run(command, capture_output, check, text):
        captured["command"] = command
        captured["capture_output"] = capture_output
        captured["check"] = check
        captured["text"] = text
        return None

    def fake_read_shade_file(path):
        captured["shade_path"] = path
        return np.array([1.0 + 0.0j]), None

    monkeypatch.setattr(propagation.subprocess, "run", fake_run)
    monkeypatch.setattr(propagation, "read_shade_file", fake_read_shade_file)
    platform = SimpleNamespace(array=SimpleNamespace(ref_state_vector=np.array([3.0, 4.0, -10.0])))
    source = SimpleNamespace(
        state_vector=np.array([0.0, 0.0, 0.0, 0.0, -10.0]),
        metadata={
            "position_mapping": [0, 2, 4],
            "frequencies_hz": np.array([100.0, 250.0]),
            "amplitudes_upa": np.array([0.1, 0.9]),
        },
    )
    model = propagation.BellhopAcousticPropagationModel(
        env_depth=200.0,
        ssp=ConstantSSP(1500.0),
    )

    tloss, travel_time = model.propagate(platform, source)

    assert captured["command"][0] == "/usr/local/bin/bellhopcxx"
    assert captured["command"][1].endswith("/env")
    assert captured["capture_output"] is True
    assert captured["check"] is True
    assert captured["text"] is True
    assert str(captured["shade_path"]).endswith("/env.shd")
    assert tloss == pytest.approx(0.0)
    assert travel_time == pytest.approx(5.0 / 1500.0)


def test_bellhop_propagate_uses_last_pressure_value_from_shade_data(monkeypatch) -> None:
    """Bellhop propagation should report transmission loss from the last pressure sample."""
    propagation = _load_propagation_module(monkeypatch)
    monkeypatch.setattr(propagation, "which", lambda exe_name: f"/usr/local/bin/{exe_name}")
    monkeypatch.setattr(propagation.subprocess, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        propagation,
        "read_shade_file",
        lambda path: (np.array([0.25 + 0.0j, 0.5 + 0.0j]), None),
    )
    platform = SimpleNamespace(array=SimpleNamespace(ref_state_vector=np.array([3.0, 4.0, -10.0])))
    source = SimpleNamespace(
        state_vector=np.array([0.0, 0.0, 0.0, 0.0, -10.0]),
        metadata={
            "position_mapping": [0, 2, 4],
            "frequencies_hz": np.array([100.0, 250.0]),
            "amplitudes_upa": np.array([0.1, 0.9]),
        },
    )
    model = propagation.BellhopAcousticPropagationModel(
        env_depth=200.0,
        ssp=ConstantSSP(1500.0),
    )

    tloss, travel_time = model.propagate(platform, source)

    assert tloss == pytest.approx(6.020599913279624)
    assert travel_time == pytest.approx(5.0 / 1500.0)


def test_bellhop_propagate_returns_scalar_time_for_column_vector_states(monkeypatch) -> None:
    """Bellhop propagation should return scalar outputs for Stone Soup-style column vectors."""
    propagation = _load_propagation_module(monkeypatch)
    monkeypatch.setattr(propagation, "which", lambda exe_name: f"/usr/local/bin/{exe_name}")
    monkeypatch.setattr(propagation.subprocess, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        propagation,
        "read_shade_file",
        lambda path: (np.array([1.0 + 0.0j]), None),
    )
    platform = SimpleNamespace(
        array=SimpleNamespace(ref_state_vector=np.array([[3.0], [4.0], [-10.0]]))
    )
    source = SimpleNamespace(
        state_vector=np.array([[0.0], [0.0], [0.0], [0.0], [-10.0]]),
        metadata={
            "position_mapping": [0, 2, 4],
            "frequencies_hz": np.array([100.0, 250.0]),
            "amplitudes_upa": np.array([0.1, 0.9]),
        },
    )
    model = propagation.BellhopAcousticPropagationModel(
        env_depth=200.0,
        ssp=ConstantSSP(1500.0),
    )

    tloss, travel_time = model.propagate(platform, source)

    assert np.isscalar(tloss)
    assert np.isscalar(travel_time)
    assert tloss == pytest.approx(0.0)
    assert travel_time == pytest.approx(5.0 / 1500.0)


def test_bellhop_propagate_handles_higher_dimensional_shade_output(monkeypatch) -> None:
    """Bellhop propagation should squeeze multi-dimensional shade data before decoding TL."""
    propagation = _load_propagation_module(monkeypatch)
    monkeypatch.setattr(propagation, "which", lambda exe_name: f"/usr/local/bin/{exe_name}")
    monkeypatch.setattr(propagation.subprocess, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        propagation,
        "read_shade_file",
        lambda path: (np.array([[[[0.25 + 0.0j]], [[0.5 + 0.0j]]]]), None),
    )
    platform = SimpleNamespace(array=SimpleNamespace(ref_state_vector=np.array([3.0, 4.0, -10.0])))
    source = SimpleNamespace(
        state_vector=np.array([0.0, 0.0, 0.0, 0.0, -10.0]),
        metadata={
            "position_mapping": [0, 2, 4],
            "frequencies_hz": np.array([100.0, 250.0]),
            "amplitudes_upa": np.array([0.1, 0.9]),
        },
    )
    model = propagation.BellhopAcousticPropagationModel(
        env_depth=200.0,
        ssp=ConstantSSP(1500.0),
    )

    tloss, travel_time = model.propagate(platform, source)

    assert tloss == pytest.approx(6.020599913279624)
    assert travel_time == pytest.approx(5.0 / 1500.0)


def test_bellhop_create_env_file_handles_column_vector_geometry(monkeypatch, tmp_path) -> None:
    """Bellhop env-file generation should accept Stone Soup-style column-vector states."""
    propagation = _load_propagation_module(monkeypatch)
    monkeypatch.setattr(propagation, "which", lambda exe_name: f"/usr/local/bin/{exe_name}")
    platform = SimpleNamespace(
        array=SimpleNamespace(ref_state_vector=np.array([[3.0], [4.0], [-15.0]]))
    )
    source = SimpleNamespace(
        state_vector=np.array([[0.0], [0.0], [0.0], [0.0], [-5.0]]),
        metadata={
            "position_mapping": [0, 2, 4],
            "frequencies_hz": np.array([125.0, 250.0, 400.0]),
            "amplitudes_upa": np.array([0.9, 0.1, 0.2]),
        },
    )
    model = propagation.BellhopAcousticPropagationModel(
        env_depth=150.0,
        ssp=ConstantSSP(1490.0),
    )

    model._create_env_file(platform, source, output_dir=tmp_path)

    env_text = (tmp_path / "env.env").read_text(encoding="utf-8")

    assert "125.0" in env_text
    assert "5.0 /" in env_text
    assert "15.0 /" in env_text
    assert "0.0 0.011180339887498949 /" in env_text


def test_bellhop_propagate_runs_with_real_local_executable(monkeypatch) -> None:
    """Bellhop should run against the project-local executable and return finite outputs."""
    propagation = _load_propagation_module(monkeypatch)
    exe_path = _project_bellhop_executable()
    real_utils = load_package_module_from_repo(
        "bluepebble/utils/bellhop.py",
        "bluepebble.utils.bellhop_real_test",
    )
    monkeypatch.setattr(propagation, "read_shade_file", real_utils.read_shade_file)

    platform = SimpleNamespace(array=SimpleNamespace(ref_state_vector=np.array([3.0, 4.0, -10.0])))
    source = SimpleNamespace(
        state_vector=np.array([0.0, 0.0, 0.0, 0.0, -10.0]),
        metadata={
            "position_mapping": [0, 2, 4],
            "frequencies_hz": np.array([100.0, 250.0]),
            "amplitudes_upa": np.array([0.1, 0.9]),
        },
    )
    model = propagation.BellhopAcousticPropagationModel(
        env_depth=200.0,
        ssp=ConstantSSP(1500.0),
        exe_path=str(exe_path),
    )

    tloss, travel_time = model.propagate(platform, source)

    assert np.isscalar(tloss)
    assert np.isfinite(float(tloss))
    assert 0.0 < float(tloss) < 999.0
    assert travel_time == pytest.approx(5.0 / 1500.0)


def test_bellhop_propagate_runs_with_real_local_executable_and_constant_ssp(monkeypatch) -> None:
    """Bellhop should also run with the package's real constant sound-speed profile."""
    propagation = _load_propagation_module(monkeypatch)
    exe_path = _project_bellhop_executable()
    real_utils = load_package_module_from_repo(
        "bluepebble/utils/bellhop.py",
        "bluepebble.utils.bellhop_real_test_constant_ssp",
    )
    sound_speed_profile = load_module_from_repo(
        "bluepebble/models/environment/sound_speed_profile.py",
        "bluepebble_sound_speed_profile_real_bellhop_test",
    )
    monkeypatch.setattr(propagation, "read_shade_file", real_utils.read_shade_file)

    platform = SimpleNamespace(
        array=SimpleNamespace(ref_state_vector=np.array([[3.0], [4.0], [-10.0]]))
    )
    source = SimpleNamespace(
        state_vector=np.array([[0.0], [0.0], [0.0], [0.0], [-10.0]]),
        metadata={
            "position_mapping": [0, 2, 4],
            "frequencies_hz": np.array([100.0, 250.0]),
            "amplitudes_upa": np.array([0.1, 0.9]),
        },
    )
    model = propagation.BellhopAcousticPropagationModel(
        env_depth=200.0,
        ssp=sound_speed_profile.Constant(speed=1500.0),
        exe_path=str(exe_path),
    )

    tloss, travel_time = model.propagate(platform, source)

    assert np.isscalar(tloss)
    assert np.isscalar(travel_time)
    assert np.isfinite(float(tloss))
    assert 0.0 < float(tloss) < 999.0
    assert travel_time == pytest.approx(5.0 / 1500.0)


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
