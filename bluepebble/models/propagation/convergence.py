r"""Find the coarsest settings at which a propagation model's answer has converged.

Ray models have settings that trade accuracy for time: how many rays they launch, over what
spread of directions, and how finely they sample the environment. Too coarse and they miss
paths or misplace them; too fine and they waste time. :func:`converge_settings` tunes such
settings one at a time, each through its tuning field (a list of values from coarse to fine),
and keeps the coarsest value whose answer no longer changes when the setting is made finer.

"The answer" is the transfer function the model returns from
:meth:`~bluepebble.models.propagation.SpectrumPropagationModel.propagate_spectrum`, on a few
test legs (a source and some receivers) and frequencies chosen by the caller. Two answers
are compared by the energy of their difference over the energy of the finer one, in dB, summed
over every leg, receiver and frequency. That catches every way a setting can matter: a path
that appears or disappears, a delay that moves (a delay error :math:`\delta\tau` at
frequency :math:`f` is a relative error of about :math:`2 \pi f \delta\tau`, so -30 dB is
about 1.5 us at 3 kHz), and a change in amplitude (-30 dB is about 3%).

Each setting is tuned with the settings already tuned held at their converged values, and the
rest at the first value of their tuning field; so tune the settings in order of importance, and
start each tuning field at the value you would otherwise use. A model is only ever run once for a
given combination of settings.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from .acoustic import SpectrumPropagationModel


#: A fixed time for the test legs' states: the geometry is static, so any time will do.
_TEST_TIME = datetime(2000, 1, 1)


@dataclass(frozen=True)
class ConvergenceStep:
    """One comparison between a setting's value and the next finer one.

    Attributes
    ----------
    setting : str
        The setting being tuned.
    value, finer_value : Any
        The two values compared, the second the next one in the tuning field.
    difference_db : float
        Energy of the difference between their answers over the energy of the finer
        answer, in dB.
    finer_run_time_s : float
        Time taken to compute the finer answer, over every test leg.
    converged : bool
        Whether ``difference_db`` is at or below the tolerance, so ``value`` is kept.

    """

    setting: str
    value: Any
    finer_value: Any
    difference_db: float
    finer_run_time_s: float
    converged: bool


@dataclass
class ConvergenceResult:
    """The tuned settings and how they were reached.

    Attributes
    ----------
    settings : dict
        The value chosen for every tuned setting: the coarsest that converged, or the finest
        in its tuning field if none did.
    steps : list of ConvergenceStep
        Every comparison made, in order.
    unconverged : list of str
        Settings whose answer was still changing at the end of their tuning field.

    """

    settings: dict[str, Any]
    steps: list[ConvergenceStep] = field(default_factory=list)
    unconverged: list[str] = field(default_factory=list)

    def report(self) -> str:
        """Return the steps and chosen settings as readable text."""
        lines = [f"{'Setting':>26} {'Value':>20} {'Finer':>20} {'Difference':>11} {'Time':>8}"]
        for step in self.steps:
            lines.append(
                f"{step.setting:>26} {_short(step.value):>20} {_short(step.finer_value):>20}"
                f" {step.difference_db:>8.1f} dB {step.finer_run_time_s:>6.1f} s"
                + ("  converged" if step.converged else "")
            )
        lines.append("Chosen: " + ", ".join(f"{k}={_short(v)}" for k, v in self.settings.items()))
        if self.unconverged:
            lines.append(
                "Still changing at the end of its tuning field: " + ", ".join(self.unconverged)
            )
        return "\n".join(lines)


def converge_settings(
    make_model: Callable[[Mapping[str, Any]], SpectrumPropagationModel],
    tuning_fields: Mapping[str, Sequence[Any]],
    legs: Sequence[tuple[ArrayLike, ArrayLike]],
    frequencies_hz: ArrayLike,
    tolerance_db: float = -30.0,
    on_step: Callable[[ConvergenceStep], None] | None = None,
) -> ConvergenceResult:
    """Tune each setting in ``tuning_fields`` to the coarsest value whose answer has converged.

    Parameters
    ----------
    make_model : callable
        Builds the model from a mapping of the settings being tuned, each to one value.
    tuning_fields : mapping
        For each setting, its candidate values from coarsest to finest, tuned in this order.
        Values must be hashable.
    legs : sequence of (source, receivers)
        Test legs: a source position ``(3,)`` and receiver positions ``(3, n)`` (or ``(3,)``
        for one), world frame in metres with z negative below the surface.
    frequencies_hz : array-like
        Frequencies the transfer functions are compared at, in Hz.
    tolerance_db : float, optional
        Largest difference, in dB, at which a setting counts as converged. Default -30.
    on_step : callable, optional
        Called with each :class:`ConvergenceStep` as it is made, e.g. to print progress.

    Returns
    -------
    ConvergenceResult
        The chosen settings, every step, and any setting that didn't converge.

    Raises
    ------
    ValueError
        If ``tuning_fields`` or ``legs`` is empty, or a tuning field has no values.

    """
    if not tuning_fields:
        raise ValueError("tuning_fields must name at least one setting.")
    if not legs:
        raise ValueError("legs must hold at least one test leg.")
    for name, values in tuning_fields.items():
        if len(values) == 0:
            raise ValueError(f"The tuning field for {name!r} has no values.")
    frequencies = np.asarray(frequencies_hz, dtype=float).reshape(-1)
    platforms = [(_platform_state(receivers), _source_state(source)) for source, receivers in legs]
    cache: dict[tuple[tuple[str, Any], ...], tuple[list[NDArray[Any]], float]] = {}

    def answer(settings: Mapping[str, Any]) -> tuple[list[NDArray[Any]], float]:
        key = tuple(sorted(settings.items()))
        if key not in cache:
            model = make_model(dict(settings))
            start = time.perf_counter()
            transfer_functions = [
                np.asarray(model.propagate_spectrum(platform, source, frequencies)[0])
                for platform, source in platforms
            ]
            cache[key] = (transfer_functions, time.perf_counter() - start)
        return cache[key]

    settings = {name: values[0] for name, values in tuning_fields.items()}
    result = ConvergenceResult(settings=settings)
    for name, values in tuning_fields.items():
        for value, finer_value in zip(values[:-1], values[1:], strict=True):
            coarse, _ = answer({**settings, name: value})
            fine, fine_time_s = answer({**settings, name: finer_value})
            difference_db = _difference_db(coarse, fine)
            step = ConvergenceStep(
                setting=name,
                value=value,
                finer_value=finer_value,
                difference_db=difference_db,
                finer_run_time_s=fine_time_s,
                converged=difference_db <= tolerance_db,
            )
            result.steps.append(step)
            if on_step is not None:
                on_step(step)
            if step.converged:
                settings[name] = value
                break
        else:
            settings[name] = values[-1]
            if len(values) > 1:
                result.unconverged.append(name)
    return result


def _difference_db(coarse: list[NDArray[Any]], fine: list[NDArray[Any]]) -> float:
    """Return the energy of ``coarse - fine`` over that of ``fine``, in dB, over every leg."""
    difference = sum(float(np.sum(np.abs(c - f) ** 2)) for c, f in zip(coarse, fine, strict=True))
    reference = sum(float(np.sum(np.abs(f) ** 2)) for f in fine)
    if reference == 0.0:
        return 0.0 if difference == 0.0 else np.inf
    if difference == 0.0:
        return -np.inf
    return float(10 * np.log10(difference / reference))


def _platform_state(receivers: ArrayLike) -> Any:
    """Return a platform state whose array is the receiver positions, shape ``(3, n)``."""
    from stonesoup.types.array import StateVector
    from stonesoup.types.groundtruth import GroundTruthState

    from ...arrays import ArrayState, HostState, PlatformState

    xyz = np.asarray(receivers, dtype=float).reshape(3, -1)
    reference = StateVector(xyz.mean(axis=1))
    return PlatformState(
        timestamp=_TEST_TIME,
        host=HostState(
            state=GroundTruthState(reference, timestamp=_TEST_TIME),
            heading_rad=0.0,
            position_mapping=(0, 1, 2),
        ),
        array=ArrayState(num_sensors=xyz.shape[1], state_vector=xyz, ref_state_vector=reference),
    )


def _source_state(source: ArrayLike) -> Any:
    """Return a source state at ``source``, for a propagation model."""
    from stonesoup.types.array import StateVector
    from stonesoup.types.groundtruth import GroundTruthState

    return GroundTruthState(
        StateVector(np.asarray(source, dtype=float).reshape(3)),
        timestamp=_TEST_TIME,
        metadata={"position_mapping": [0, 1, 2]},
    )


def _short(value: Any) -> str:
    """Return ``value`` compactly, for the report."""
    if isinstance(value, tuple):
        return "(" + ", ".join(_short(v) for v in value) + ")"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)
