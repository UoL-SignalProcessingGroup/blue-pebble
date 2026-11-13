"""Tests that the removed TowedArrayPlatform and simulator ``platform=`` name their replacement."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from .support import isolate_real_bluepebble_imports, restore_real_bluepebble_imports


@pytest.fixture
def bp():
    """Import the real platform and simulator packages without leaking caching."""
    snapshot = isolate_real_bluepebble_imports()
    import bluepebble.platform as platform_pkg
    from bluepebble.simulator import DiscretePassiveSonarArraySimulator

    yield SimpleNamespace(platform=platform_pkg, Simulator=DiscretePassiveSonarArraySimulator)
    restore_real_bluepebble_imports(snapshot)


def test_towed_array_platform_import_names_its_replacement(bp) -> None:
    """ImportError, not AttributeError, so ``from ... import`` keeps the guidance."""
    with pytest.raises(ImportError, match="v0.8.0.*TowedArray.*add_sensor"):
        _ = bp.platform.TowedArrayPlatform


def test_unknown_platform_attribute_still_raises_attributeerror(bp) -> None:
    """Only the removed name is special-cased."""
    with pytest.raises(AttributeError, match="has no attribute"):
        _ = bp.platform.NotAThing


def test_simulator_platform_argument_names_its_replacement(bp) -> None:
    """Passing the removed ``platform=`` says to pass the mounted array as ``array=``."""
    with pytest.raises(TypeError, match=r"platform=\.\.\.\) was removed in v0\.8\.0.*array="):
        bp.Simulator(platform=object(), propagation_model=object(), signal_models=[object()])


def test_simulator_requires_an_array(bp) -> None:
    """Without ``array=`` the simulator has nothing to simulate."""
    simulator = bp.Simulator(propagation_model=object(), signal_models=[object()])

    with pytest.raises(TypeError, match="array="):
        simulator._sorted_timestamps()
