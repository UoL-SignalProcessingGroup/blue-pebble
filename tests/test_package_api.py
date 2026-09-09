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


@pytest.mark.parametrize(
    ("removed_name", "expected_guidance"),
    [
        ("PeakDetector", "OSCFARDetector"),
        # Not target_pfa: a CFAR detector is not a drop-in replacement for a fixed
        # global threshold, and the message must not imply that it is.
        ("ThresholdDetector", "beam_snr"),
        ("run_detection_chain", "detector.detect"),
    ],
)
def test_removed_chain_api_raises_importerror_naming_its_replacement(
    removed_name: str, expected_guidance: str
) -> None:
    """The chain API is gone; importing it should say what replaced it, not just fail.

    ImportError rather than AttributeError is load-bearing here: ``from bluepebble.detector
    import X`` discards an AttributeError's message and substitutes a generic one, so the
    migration guidance would never reach the caller.
    """
    import bluepebble.detector as detector_pkg

    with pytest.raises(ImportError, match=expected_guidance):
        getattr(detector_pkg, removed_name)


def test_unknown_detector_attribute_still_raises_attributeerror() -> None:
    """Only the known removed names are special-cased; anything else behaves normally."""
    import bluepebble.detector as detector_pkg

    unknown_name = "NotAThing"

    with pytest.raises(AttributeError, match="has no attribute"):
        getattr(detector_pkg, unknown_name)
