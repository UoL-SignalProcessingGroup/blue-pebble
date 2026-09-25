"""Shared helpers for lightweight unit tests."""

from __future__ import annotations

import importlib.util
import sys
import types
from math import comb
from pathlib import Path

from scipy import integrate
from scipy.stats import gamma as gamma_dist

REPO_ROOT = Path(__file__).resolve().parents[1]


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


# CFAR Pfa references by numerical quadrature, independent of the library's closed forms.


def ca_pfa_by_quadrature(alpha, num_training_total, total_looks):
    """P(CUT > alpha * mean(refs)) with CUT, refs unit-mean Gamma(total_looks) cells.

    Integrates over the sum of reference cells, S ~ Gamma(N * L, scale 1/L), of the CUT
    survival function, CUT ~ Gamma(L, scale 1/L). Written without the Beta-distribution identity
    used by solve_ca_cfar_alpha.
    """
    looks = total_looks
    ref_sum = gamma_dist(a=num_training_total * looks, scale=1.0 / looks)
    cut = gamma_dist(a=looks, scale=1.0 / looks)
    upper = ref_sum.ppf(1 - 1e-14)

    def integrand(s):
        return ref_sum.pdf(s) * cut.sf(alpha * s / num_training_total)

    value, _ = integrate.quad(integrand, 0.0, upper, limit=500, epsabs=1e-14, epsrel=1e-10)
    return value


def os_pfa_by_quadrature(alpha, num_training_total, rank, total_looks):
    """P(CUT > alpha * X_(k)) for unit-mean Gamma(total_looks) cells, by quadrature.

    X_(k), the k-th smallest of N iid cells with CDF F and PDF f, has density
    ``k * C(N, k) * F^(k-1) * (1 - F)^(N-k) * f``.
    """
    n, k = num_training_total, rank
    cell = gamma_dist(a=total_looks, scale=1.0 / total_looks)

    def integrand(x):
        f_cdf = cell.cdf(x)
        density = k * comb(n, k) * f_cdf ** (k - 1) * (1 - f_cdf) ** (n - k) * cell.pdf(x)
        return density * cell.sf(alpha * x)

    upper = cell.ppf(1 - 1e-12)
    value, _ = integrate.quad(integrand, 0.0, upper, limit=500, epsabs=1e-13, epsrel=1e-9)
    return value
