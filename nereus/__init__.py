"""Nereus package public API."""

from importlib.metadata import PackageNotFoundError, version

from . import detector, models, platform, signal, sigproc, simulator, utils

try:
    __version__ = version("nereus")
except PackageNotFoundError:  # pragma: no cover
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
