"""Tests for target scattering models."""

from __future__ import annotations

import numpy as np
import pytest

from .support import isolate_real_bluepebble_imports, restore_real_bluepebble_imports


@pytest.fixture
def scattering():
    """Import bluepebble.models.scattering without leaking real caching into other tests.

    See ``isolate_real_bluepebble_imports`` for why this matters: a plain top-level
    ``from bluepebble... import ...`` would otherwise permanently cache real submodule
    attributes on the shared ``bluepebble`` package for the rest of the test session.
    """
    snapshot = isolate_real_bluepebble_imports()
    from bluepebble.models import scattering as module

    yield module
    restore_real_bluepebble_imports(snapshot)


def test_target_scattering_model_is_abstract(scattering) -> None:
    """TargetScatteringModel cannot be instantiated directly."""
    with pytest.raises(TypeError):
        scattering.TargetScatteringModel()  # type: ignore[abstract]


def test_constant_target_strength_matches_bellhop_path_convention(scattering) -> None:
    """ConstantTargetStrength should match active.py's sqrt(10**(TS_db/10)) convention."""
    target_strength_db = 15.0
    model = scattering.ConstantTargetStrength(target_strength_db=target_strength_db)

    expected_gain = np.sqrt(10.0 ** (target_strength_db / 10.0))
    result = model.scatter([800.0, 1000.0, 1200.0])

    np.testing.assert_allclose(result, [expected_gain] * 3)


def test_constant_target_strength_is_frequency_independent(scattering) -> None:
    """The same scalar gain should be returned regardless of the requested frequencies."""
    model = scattering.ConstantTargetStrength(target_strength_db=10.0)

    result = model.scatter(np.linspace(500.0, 5000.0, 50))

    assert np.allclose(result, result[0])


def test_constant_target_strength_output_shape_matches_input(scattering) -> None:
    """The returned gain array should have the same shape as the requested frequencies."""
    model = scattering.ConstantTargetStrength(target_strength_db=0.0)

    frequencies = np.array([100.0, 200.0, 300.0, 400.0])
    result = model.scatter(frequencies)

    assert result.shape == frequencies.shape


def test_constant_target_strength_zero_db_is_unity_gain(scattering) -> None:
    """0 dB target strength should give a linear gain of exactly 1.0."""
    model = scattering.ConstantTargetStrength(target_strength_db=0.0)

    result = model.scatter([1000.0])

    np.testing.assert_allclose(result, [1.0])
