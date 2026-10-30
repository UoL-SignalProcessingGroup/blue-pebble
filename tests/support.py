"""Shared helpers for lightweight unit tests."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import bluepebble

REPO_ROOT = Path(__file__).resolve().parents[1]


def isolate_real_bluepebble_imports():
    """Evict any cached ``bluepebble`` modules so a following real import is guaranteed fresh.

    Most tests in this suite import ``bluepebble`` code through the fake-dependency
    ``install_fake_stonesoup*`` / ``load_package_module_from_repo`` helpers above, which never
    touch the real package tree. Two problems arise for a test that instead does a plain
    ``from bluepebble.x.y import Z``:

    1. ``load_package_module_from_repo`` writes straight to ``sys.modules`` (not via
       ``monkeypatch``), so a submodule it loads while a fake ``stonesoup.base`` is active
       (e.g. ``bluepebble.signal.base``, loaded by ``test_signal_anthropogenic.py``) stays
       cached under that fake ``Base``/``Property`` for the rest of the session. A later real
       import of anything built on top of it (e.g. ``bluepebble.signal.active``) then picks up
       that stale, fake-metaclass copy instead of re-executing the file, and blows up with
       ``AttributeError: Cannot use Property on this class type``.
    2. Python's import system also caches ``bluepebble.x`` (and ``y``) as real attributes on the
       live ``bluepebble`` package object -- permanently, for the rest of the session -- which
       breaks other tests that assert on ``bluepebble``'s lazy-import behaviour
       (``test_package_api.py``) or that load a submodule fresh under fake dependencies
       (``test_simulator_acoustic.py``).

    Call this immediately before a real ``bluepebble`` import and pass the result to
    :func:`restore_real_bluepebble_imports` (e.g. via a fixture) to put back whatever was there
    -- real or fake -- once the real import is no longer needed.
    """
    # Keep "bluepebble" itself in place (not just its submodules): this module holds its own
    # top-level `import bluepebble` reference, and re-creating that object here would leave
    # this function updating a different object than the one other code actually ends up using.
    removed_modules = {
        name: sys.modules.pop(name)
        for name in list(sys.modules)
        if name.startswith("bluepebble.")
    }
    # Only the specific names bluepebble.__getattr__ lazily caches via `globals()[name] =
    # module` (see bluepebble/__init__.py) -- never touch dunders (__name__, __all__, ...),
    # which live in the same __dict__ and must not be disturbed mid-cleanup.
    removed_attrs = {
        name: bluepebble.__dict__.pop(name)
        for name in _LAZY_BLUEPEBBLE_ATTRS
        if name in bluepebble.__dict__
    }
    return removed_modules, removed_attrs


def restore_real_bluepebble_imports(snapshot) -> None:
    """Discard modules imported since :func:`isolate_real_bluepebble_imports` and put back."""
    removed_modules, removed_attrs = snapshot
    for name in list(sys.modules):
        if name.startswith("bluepebble."):
            del sys.modules[name]
    sys.modules.update(removed_modules)
    for name in _LAZY_BLUEPEBBLE_ATTRS:
        bluepebble.__dict__.pop(name, None)
    bluepebble.__dict__.update(removed_attrs)


_LAZY_BLUEPEBBLE_ATTRS = frozenset({
    "detector", "models", "platform", "plotter", "signal", "sigproc", "simulator", "types",
    "get_rng", "set_seed",
})


class FakeProperty:
    """Minimal descriptor compatible with Stone Soup-style ``Property`` usage."""

    def __init__(self, *args, default=None, doc: str | None = None):
        """Store the optional default while tolerating Stone Soup-style signatures."""
        self.type_hint = args[0] if args else None
        self.default = default
        self.__doc__ = doc
        self.name: str | None = None

    def __set_name__(self, owner, name: str) -> None:
        """Record the attribute name when the descriptor is bound to a class."""
        self.name = name

    def __get__(self, instance, owner=None):
        """Return the stored value or the configured default."""
        if instance is None:
            return self
        return instance.__dict__.get(self.name, self.default)

    def __set__(self, instance, value) -> None:
        """Persist the assigned value on the instance."""
        instance.__dict__[self.name] = value


class FakeBase:
    """Minimal base class that applies descriptor defaults and runs ``__post_init__``."""

    def __init__(self, **kwargs):
        """Populate descriptor defaults, apply overrides, and run post-init hooks."""
        for name, attr in type(self).__dict__.items():
            if isinstance(attr, FakeProperty) and name not in kwargs:
                setattr(self, name, attr.default)

        for key, value in kwargs.items():
            setattr(self, key, value)

        post_init = getattr(self, "__post_init__", None)
        if callable(post_init):
            post_init()


def install_fake_stonesoup(monkeypatch) -> None:
    """Install a lightweight fake ``stonesoup.base`` module for unit tests."""
    stonesoup_module = types.ModuleType("stonesoup")
    stonesoup_base_module = types.ModuleType("stonesoup.base")
    stonesoup_base_module.Base = FakeBase
    stonesoup_base_module.Property = FakeProperty
    stonesoup_module.base = stonesoup_base_module

    monkeypatch.setitem(sys.modules, "stonesoup", stonesoup_module)
    monkeypatch.setitem(sys.modules, "stonesoup.base", stonesoup_base_module)


def install_fake_stonesoup_plotter_modules(monkeypatch) -> None:
    """Install minimal Stone Soup modules required for importing ``plotter.py``."""
    stonesoup_module = sys.modules.get("stonesoup", types.ModuleType("stonesoup"))

    platform_module = types.ModuleType("stonesoup.platform")
    platform_base_module = types.ModuleType("stonesoup.platform.base")
    platform_base_module.Platform = type("Platform", (), {})
    platform_module.base = platform_base_module

    types_module = types.ModuleType("stonesoup.types")
    detection_module = types.ModuleType("stonesoup.types.detection")
    detection_module.Detection = type("Detection", (), {})
    groundtruth_module = types.ModuleType("stonesoup.types.groundtruth")
    groundtruth_module.GroundTruthPath = type("GroundTruthPath", (), {})
    track_module = types.ModuleType("stonesoup.types.track")
    track_module.Track = type("Track", (), {})

    types_module.detection = detection_module
    types_module.groundtruth = groundtruth_module
    types_module.track = track_module

    stonesoup_module.platform = platform_module
    stonesoup_module.types = types_module

    monkeypatch.setitem(sys.modules, "stonesoup", stonesoup_module)
    monkeypatch.setitem(sys.modules, "stonesoup.platform", platform_module)
    monkeypatch.setitem(sys.modules, "stonesoup.platform.base", platform_base_module)
    monkeypatch.setitem(sys.modules, "stonesoup.types", types_module)
    monkeypatch.setitem(sys.modules, "stonesoup.types.detection", detection_module)
    monkeypatch.setitem(sys.modules, "stonesoup.types.groundtruth", groundtruth_module)
    monkeypatch.setitem(sys.modules, "stonesoup.types.track", track_module)


def install_fake_stonesoup_simulator_modules(monkeypatch) -> None:
    """Install minimal Stone Soup modules required for simulator imports."""
    stonesoup_module = sys.modules.get("stonesoup", types.ModuleType("stonesoup"))

    simulator_module = types.ModuleType("stonesoup.simulator")
    simulator_base_module = types.ModuleType("stonesoup.simulator.base")
    simulator_base_module.SensorSimulator = type("SensorSimulator", (FakeBase,), {})
    simulator_module.base = simulator_base_module

    types_module = getattr(stonesoup_module, "types", types.ModuleType("stonesoup.types"))
    sensordata_module = types.ModuleType("stonesoup.types.sensordata")
    sensordata_module.SensorData = type("SensorData", (FakeBase,), {})
    types_module.sensordata = sensordata_module

    stonesoup_module.simulator = simulator_module
    stonesoup_module.types = types_module

    monkeypatch.setitem(sys.modules, "stonesoup", stonesoup_module)
    monkeypatch.setitem(sys.modules, "stonesoup.simulator", simulator_module)
    monkeypatch.setitem(sys.modules, "stonesoup.simulator.base", simulator_base_module)
    monkeypatch.setitem(sys.modules, "stonesoup.types", types_module)
    monkeypatch.setitem(sys.modules, "stonesoup.types.sensordata", sensordata_module)


def load_module_from_repo(relative_path: str, module_name: str):
    """Load a module directly from a repository path."""
    module_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def install_repo_package(monkeypatch, package_name: str, relative_path: str) -> None:
    """Install a lightweight package module pointing at a repository directory."""
    package = types.ModuleType(package_name)
    package.__path__ = [str(REPO_ROOT / relative_path)]
    monkeypatch.setitem(sys.modules, package_name, package)


def load_package_module_from_repo(relative_path: str, module_name: str):
    """Load a package submodule directly from a repository path."""
    module_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
