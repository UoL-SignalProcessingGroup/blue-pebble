"""Blue Pebble package public API."""

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("blue-pebble")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    "__version__",
    "get_rng",
    "set_seed",
    "detector",
    "models",
    "platform",
    "plotter",
    "sensors",
    "signal",
    "sigproc",
    "simulator",
    "types",
]


def __getattr__(name: str):
    """Lazily import top-level subpackages and seed helpers on first access."""
    if name in {
        "detector",
        "models",
        "platform",
        "plotter",
        "sensors",
        "signal",
        "sigproc",
        "simulator",
        "types",
    }:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    if name in {"get_rng", "set_seed"}:
        from bluepebble._seed import get_rng, set_seed  # noqa: PLC0415

        globals()["get_rng"] = get_rng
        globals()["set_seed"] = set_seed
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Expose lazy subpackages through introspection."""
    return sorted(set(globals()) | set(__all__))
