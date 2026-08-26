"""Platform models public API."""

from .base import HostPlatform
from .sonobuoy import OmniSonobuoyPlatform

__all__ = ["HostPlatform", "OmniSonobuoyPlatform"]
