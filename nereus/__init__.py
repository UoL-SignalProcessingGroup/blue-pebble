"""Nereus package public API."""

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("nereus")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    "__version__",
    "detector",
    "models",
    "platform",
    "signal",
    "sigproc",
    "simulator",
    "utils",
]


def __getattr__(name: str):
    """Lazily import top-level subpackages on first access."""
    if name in {"detector", "models", "platform", "signal", "sigproc", "simulator", "utils"}:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Expose lazy subpackages through introspection."""
    return sorted(set(globals()) | set(__all__))
