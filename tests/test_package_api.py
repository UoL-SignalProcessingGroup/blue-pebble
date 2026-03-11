"""Tests for the package-level public API."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

import bluepebble


def test_dir_exposes_public_names() -> None:
    """The package should advertise its documented public API."""
    exported_names = dir(bluepebble)

    for name in bluepebble.__all__:
        assert name in exported_names


@pytest.mark.parametrize("submodule_name", ["signal", "types"])
def test_getattr_lazily_imports_and_caches_submodules(monkeypatch, submodule_name: str) -> None:
    """Lazy submodule access should import once and cache the result."""
    importlib.reload(bluepebble)
    sentinel_module = SimpleNamespace(__name__=f"bluepebble.{submodule_name}")
    import_calls: list[str] = []

    def fake_import_module(name: str):
        import_calls.append(name)
        return sentinel_module

    monkeypatch.setattr(bluepebble, "import_module", fake_import_module)

    assert submodule_name not in bluepebble.__dict__

    result = getattr(bluepebble, submodule_name)

    assert result is sentinel_module
    assert bluepebble.__dict__[submodule_name] is sentinel_module
    assert import_calls == [f"bluepebble.{submodule_name}"]


def test_getattr_rejects_unknown_attributes() -> None:
    """Unknown attributes should raise the standard package error."""
    with pytest.raises(AttributeError, match="has no attribute"):
        _ = bluepebble.not_a_real_attribute
