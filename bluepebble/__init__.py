"""Blue Pebble package public API.

Coordinate frames
-----------------
Positions are in a local Cartesian frame in metres: +x east, +y north and +z up, so depths
are negative. Geographic data (bathymetry, sound-speed profiles) is projected onto it.

Directions in the horizontal plane are angles in radians, measured anticlockwise from +x
and wrapped to ``[-pi, pi)``, the convention of Stone Soup's
:class:`~stonesoup.types.angle.Bearing`. Steering azimuths, detected and ground-truth
bearings, and platform headings are all this same quantity. The navigation conventions,
true bearing (clockwise from north) and relative bearing (clockwise from the platform's
heading), are used only for display, through :func:`~bluepebble.plotter.plot_btr`'s
``bearing_convention``.

:class:`~bluepebble.sigproc.SteeringCalculator` builds its steering grid from a sector
``(start, end)`` that runs anticlockwise from start to end; ``(-np.pi, np.pi)`` is a full
circle, which does not repeat its start since -pi and pi point the same way. By default the
sector is in the array frame: angles are measured anticlockwise from the array's forward
direction (from its last sensor towards sensor 0), so port is ``(0, np.pi)``, starboard is
``(-np.pi, 0)``, and the grid turns with the array. The order matters, since ``(0, -np.pi)``
is port too; :data:`~bluepebble.sigproc.PORT`, :data:`~bluepebble.sigproc.STARBOARD` and
:data:`~bluepebble.sigproc.FULL_CIRCLE` spell the common sectors out. Each scan's steering
bearings are still world azimuths, as above, and travel with its data in
:attr:`~bluepebble.types.PassiveSonarSensorData.steering_bearings_rad`. With
``frame="world"`` the sector is measured from +x instead and the grid stays fixed.
"""

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
