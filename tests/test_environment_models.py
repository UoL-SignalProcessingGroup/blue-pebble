"""Tests for simple deterministic environment models."""

from __future__ import annotations

import numpy as np
import pytest

from .support import install_fake_stonesoup, load_module_from_repo


def test_sound_speed_profiles_handle_negative_depths(monkeypatch) -> None:
    """Profiles should treat negative z values as positive depth below the surface."""
    install_fake_stonesoup(monkeypatch)
    sound_speed_profile = load_module_from_repo(
        "bluepebble/models/environment/sound_speed_profile.py",
        "bluepebble_sound_speed_profile_test",
    )

    constant_profile = sound_speed_profile.Constant(speed=1480.0)
    linear_profile = sound_speed_profile.Linear(surface_speed=1500.0, gradient=0.01)

    assert constant_profile.calculate(-250.0) == pytest.approx(1480.0)
    assert linear_profile.calculate(-250.0) == pytest.approx(1502.5)


def test_sound_speed_profile_grid_tiles_depth_profile(monkeypatch) -> None:
    """A 1D SSP should be broadcast consistently across x and y dimensions."""
    install_fake_stonesoup(monkeypatch)
    sound_speed_profile = load_module_from_repo(
        "bluepebble/models/environment/sound_speed_profile.py",
        "bluepebble_sound_speed_profile_grid_test",
    )

    profile = sound_speed_profile.Constant(speed=1490.0)

    x_grid, y_grid, z_grid, c_grid = profile.get_3d_grid(
        x_range=(0.0, 1000.0),
        y_range=(0.0, 1000.0),
        z_range=(-200.0, 0.0),
        x_res=1000.0,
        y_res=1000.0,
        z_res=100.0,
    )

    assert x_grid.shape == (2,)
    assert y_grid.shape == (2,)
    assert z_grid.shape == (3,)
    assert c_grid.shape == (12,)
    np.testing.assert_allclose(c_grid, np.full(12, 1490.0))


@pytest.mark.parametrize(
    ("mid_depth", "steepness"), [(1000.0, 0.005), (300.0, 0.05), (2000.0, -0.002)]
)
def test_arctan_profile_starts_at_its_surface_speed(
    monkeypatch, mid_depth: float, steepness: float
) -> None:
    """The step is offset so c(0) is surface_speed, leaving its shape unchanged."""
    install_fake_stonesoup(monkeypatch)
    sound_speed_profile = load_module_from_repo(
        "bluepebble/models/environment/sound_speed_profile.py",
        f"bluepebble_arctan_surface_{mid_depth:g}_{steepness:g}",
    )
    profile = sound_speed_profile.Arctan(
        surface_speed=1490.0, mid_depth=mid_depth, steepness=steepness
    )
    depths = np.linspace(0.0, 5000.0, 51)

    assert profile.calculate(0.0) == pytest.approx(1490.0)
    # Same curve as the uncorrected step, shifted by a constant.
    uncorrected = 1490.0 + 50.0 * np.arctan(steepness * (depths - mid_depth))
    np.testing.assert_allclose(
        profile.calculate(depths) - uncorrected, 50.0 * np.arctan(steepness * mid_depth)
    )


@pytest.mark.parametrize("depth", [0.0, 10.0])
def test_flat_bathymetry_rejects_non_negative_depth(monkeypatch, depth) -> None:
    """Flat bathymetry should enforce the package's negative-depth convention."""
    install_fake_stonesoup(monkeypatch)
    bathymetry = load_module_from_repo(
        "bluepebble/models/environment/bathymetry.py",
        "bluepebble_bathymetry_flat_test",
    )

    with pytest.raises(ValueError, match="Depth must be negative"):
        bathymetry.FlatBathymetry(depth=depth)


def test_flat_bathymetry_positive_depth_error_hint(monkeypatch) -> None:
    """A positive depth's error should suggest the negated depth."""
    install_fake_stonesoup(monkeypatch)
    bathymetry = load_module_from_repo(
        "bluepebble/models/environment/bathymetry.py",
        "bluepebble_bathymetry_flat_test",
    )

    with pytest.raises(ValueError, match=r"Did you mean -10\.0\?"):
        bathymetry.FlatBathymetry(depth=10.0)


def test_flat_bathymetry_zero_depth_error_has_no_hint(monkeypatch) -> None:
    """A depth value of zero should not suggest a correction."""
    install_fake_stonesoup(monkeypatch)
    bathymetry = load_module_from_repo(
        "bluepebble/models/environment/bathymetry.py",
        "bluepebble_bathymetry_flat_test",
    )

    with pytest.raises(ValueError) as excinfo:
        bathymetry.FlatBathymetry(depth=0.0)
    assert "Did you mean" not in str(excinfo.value)


def test_wedge_bathymetry_clamps_depth_to_surface(monkeypatch) -> None:
    """Positive sloping depths should be capped at the sea surface."""
    install_fake_stonesoup(monkeypatch)
    bathymetry = load_module_from_repo(
        "bluepebble/models/environment/bathymetry.py",
        "bluepebble_bathymetry_wedge_test",
    )

    model = bathymetry.WedgeBathymetry(depth_at_origin=-100.0, x_gradient=1.0, y_gradient=0.0)

    assert model.get_depth(50.0, 0.0) == pytest.approx(-50.0)
    assert model.get_depth(200.0, 0.0) == pytest.approx(0.0)


def test_seamount_bathymetry_interpolates_inside_radius(monkeypatch) -> None:
    """Seamount depths should vary linearly between summit and plateau."""
    install_fake_stonesoup(monkeypatch)
    bathymetry = load_module_from_repo(
        "bluepebble/models/environment/bathymetry.py",
        "bluepebble_bathymetry_seamount_test",
    )

    model = bathymetry.SeamountBathymetry(
        summit_position=(0.0, 0.0, -100.0),
        radius=1000.0,
        plateau_depth=-500.0,
    )

    assert model.get_depth(0.0, 0.0) == pytest.approx(-100.0)
    assert model.get_depth(500.0, 0.0) == pytest.approx(-300.0)
    assert model.get_depth(2000.0, 0.0) == pytest.approx(-500.0)


@pytest.mark.parametrize("radius", [0.0, -10.0])
def test_seamount_bathymetry_rejects_non_positive_radius(monkeypatch, radius) -> None:
    """Seamount bathymetry should reject a non-positive radius."""
    install_fake_stonesoup(monkeypatch)
    bathymetry = load_module_from_repo(
        "bluepebble/models/environment/bathymetry.py", "bluepebble_bathymetry_seamount_test"
    )

    with pytest.raises(ValueError, match="Radius must be positive"):
        bathymetry.SeamountBathymetry(radius=radius)


@pytest.mark.parametrize("plateau_depth", [0.0, 10.0])
def test_seamount_bathymetry_rejects_non_negative_plateau(monkeypatch, plateau_depth) -> None:
    """Seamount bathymetry should reject a plateau at or above the surface."""
    install_fake_stonesoup(monkeypatch)
    bathymetry = load_module_from_repo(
        "bluepebble/models/environment/bathymetry.py", "bluepebble_bathymetry_seamount_test"
    )

    with pytest.raises(ValueError, match="Plateau depth must be negative"):
        bathymetry.SeamountBathymetry(plateau_depth=plateau_depth)


@pytest.mark.parametrize("summit_z", [10.0, -5000.0, -6000.0])
def test_seamount_bathymetry_rejects_summit_above_surface_or_below_plateau(
    monkeypatch, summit_z
) -> None:
    """Seamount bathymetry should reject a summit above the surface or not above the plateau."""
    install_fake_stonesoup(monkeypatch)
    bathymetry = load_module_from_repo(
        "bluepebble/models/environment/bathymetry.py", "bluepebble_bathymetry_seamount_test"
    )

    with pytest.raises(ValueError, match="Summit depth"):
        bathymetry.SeamountBathymetry(summit_position=(0.0, 0.0, summit_z), plateau_depth=-5000.0)


def test_gebco_bathymetry_missing_file_fails_at_construction(monkeypatch, tmp_path) -> None:
    """GEBCO bathymetry should load eagerly, so a missing file fails at construction."""
    install_fake_stonesoup(monkeypatch)
    bathymetry = load_module_from_repo(
        "bluepebble/models/environment/bathymetry.py", "bluepebble_bathymetry_gebco_test"
    )

    with pytest.raises(FileNotFoundError, match="GEBCO bathymetry file not found"):
        bathymetry.GEBCOBathymetry(file_path=str(tmp_path / "missing.nc"))
