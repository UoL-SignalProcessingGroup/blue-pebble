"""Global seed management for reproducible bluepebble simulations."""

import numpy as np

_seed_sequence: np.random.SeedSequence | None = None
_global_rng: np.random.Generator | None = None


def set_seed(seed: int) -> None:
    """Seed the global bluepebble RNG for reproducible simulations.

    Call this once at the top of a script before constructing any signal
    classes or drawing random values.  All signal classes that have
    ``seed=None`` (the default) will automatically draw independent,
    non-overlapping RNG streams derived from this seed.  Use
    :func:`get_rng` to obtain the same global generator for any additional
    random draws in the calling script.

    Parameters
    ----------
    seed : int
        Integer seed passed to :class:`numpy.random.SeedSequence`.

    Examples
    --------
    >>> import bluepebble
    >>> bluepebble.set_seed(42)
    >>> rng = bluepebble.get_rng()
    >>> rng.uniform(0, 1)  # deterministic draw

    """
    global _seed_sequence, _global_rng
    _seed_sequence = np.random.SeedSequence(seed)
    _global_rng = np.random.default_rng(_seed_sequence)


def get_rng() -> np.random.Generator:
    """Return the global :class:`numpy.random.Generator` for use in scripts.

    Returns the Generator created by the most recent :func:`set_seed` call.
    If :func:`set_seed` has not been called, returns a fresh non-deterministic
    Generator (equivalent to ``np.random.default_rng()``).

    Returns
    -------
    numpy.random.Generator
        The global Generator instance.

    """
    if _global_rng is None:
        return np.random.default_rng()
    return _global_rng


def _spawn_rng(seed: int | None) -> np.random.Generator:
    """Return a :class:`numpy.random.Generator` for a signal class instance.

    Resolution order:

    1. Explicit ``seed`` value → deterministic independent stream.
    2. ``seed=None`` and a global seed has been set via :func:`set_seed` →
       spawn a unique child from the global :class:`numpy.random.SeedSequence`,
       guaranteeing an independent, non-overlapping stream.
    3. ``seed=None`` and no global seed → non-deterministic (preserves the
       existing default behaviour).

    Parameters
    ----------
    seed : int or None
        Per-instance seed value from the calling signal class.

    Returns
    -------
    numpy.random.Generator

    """
    if seed is not None:
        return np.random.default_rng(seed)
    if _seed_sequence is not None:
        return np.random.default_rng(_seed_sequence.spawn(1)[0])
    return np.random.default_rng()
