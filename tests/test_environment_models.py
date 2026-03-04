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


def test_flat_bathymetry_rejects_non_negative_depth(monkeypatch) -> None:
    """Flat bathymetry should enforce the package's negative-depth convention."""
    install_fake_stonesoup(monkeypatch)
    bathymetry = load_module_from_repo(
        "bluepebble/models/environment/bathymetry.py",
        "bluepebble_bathymetry_flat_test",
    )

    with pytest.raises(ValueError, match="Depth must be negative"):
        bathymetry.FlatBathymetry(depth=10.0)


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
