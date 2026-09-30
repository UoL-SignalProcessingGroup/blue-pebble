"""Time :class:`~bluepebble.sigproc.DelayAndSumBeamformer` on its own.

Only ``beamform`` is timed. The sensor signals, platform and steering delays are built beforehand,
so the figures exclude signal synthesis, propagation and steering-delay calculation. The first
call for each configuration is untimed; it absorbs Numba compilation in the ``time`` and
``frequency`` domains and supplies the output that ``--save`` and ``--check`` use.

Usage::

    python benchmarks/das_beamformer.py
    python benchmarks/das_beamformer.py --domains broadband_power frequency --sensors 64 200
    python benchmarks/das_beamformer.py --save before.npz
    python benchmarks/das_beamformer.py --check before.npz

Save a run before changing the beamformer and check against it afterwards, so a speed-up that
alters the output is caught.
"""

import argparse
import sys
import timeit
from datetime import datetime, timedelta
from functools import partial

import numpy as np
from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel,
    ConstantVelocity,
)
from stonesoup.types.groundtruth import GroundTruthState

from bluepebble.models.environment import Constant
from bluepebble.platform import TowedArrayPlatform
from bluepebble.sigproc import DelayAndSumBeamformer, SteeringCalculator

SAMPLING_RATE_HZ = 500.0
SENSOR_SPACING_M = 0.5
BAND_HZ = (120.0, 249.0)  # only the broadband_power domain integrates over a band
CHECK_RTOL = 1e-6


def _steering_delays(num_sensors, num_beams):
    """Return steering delays for a stationary array, steered over the half-plane east of it."""
    start = datetime(2026, 1, 1)
    platform = TowedArrayPlatform(
        states=[GroundTruthState(np.array([0.0, 0.0, 0.0, 0.01, -5.0, 0.0]), timestamp=start)],
        position_mapping=[0, 2, 4],
        velocity_mapping=[1, 3, 5],
        transition_models=[
            CombinedLinearGaussianTransitionModel([ConstantVelocity(0.0) for _ in range(3)])
        ],
        transition_times=[timedelta(seconds=1)],
        num_sensors=num_sensors,
        cable_length_m=400.0,
        sensor_spacing_m=SENSOR_SPACING_M,
        array_depth_m=-50.0,
    )
    calculator = SteeringCalculator(
        ssp=Constant(speed=1500.0),
        steering_sector_rad=(-np.pi / 2, np.pi / 2),
        num_beams=num_beams,
        frame="world",
    )
    return calculator.calculate(platform.get_platform_state_at(start))


def _sensor_signals(num_sensors, num_samples, seed=0):
    """Return reproducible complex Gaussian sensor data, complex64 as the simulators emit it."""
    rng = np.random.default_rng(seed)
    shape = (num_sensors, num_samples)
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex64)


def _check(output, reference):
    """Return a short verdict comparing ``output`` with a saved ``reference``."""
    if reference is None:
        return "missing from reference"
    if output.shape != reference.shape:
        return f"DIFFERS (shape {output.shape} vs {reference.shape})"
    max_rel = float(np.max(np.abs(output - reference)) / np.max(np.abs(reference)))
    verdict = "match" if np.allclose(output, reference, rtol=CHECK_RTOL, atol=0.0) else "DIFFERS"
    return f"{verdict} (max rel diff {max_rel:.1e})"


def main() -> None:
    """Time each configuration and optionally save or check the outputs."""
    parser = argparse.ArgumentParser(description="Time DelayAndSumBeamformer.beamform alone.")
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["broadband_power"],
        choices=["time", "frequency", "broadband_power"],
    )
    parser.add_argument("--sensors", nargs="+", type=int, default=[64, 128, 200])
    parser.add_argument("--beams", nargs="+", type=int, default=[181])
    parser.add_argument("--duration-s", type=float, default=5.0, help="scan length in seconds")
    parser.add_argument("--repeats", type=int, default=7, help="timed calls per configuration")
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument("--save", metavar="NPZ", help="save each configuration's output")
    output_mode.add_argument("--check", metavar="NPZ", help="compare outputs with a saved run")
    args = parser.parse_args()

    num_samples = int(round(args.duration_s * SAMPLING_RATE_HZ))
    reference = dict(np.load(args.check)) if args.check else None
    outputs = {}
    any_differs = False

    print(
        f"numpy {np.__version__}; {num_samples} samples per scan; best and median of "
        f"{args.repeats} calls"
    )
    print(
        f"{'domain':<16}{'sensors':>8}{'beams':>7}  {'output shape':<14}"
        f"{'best ms':>9}{'median ms':>11}"
    )
    for domain in args.domains:
        for num_sensors in args.sensors:
            signals = _sensor_signals(num_sensors, num_samples)
            beamformer = DelayAndSumBeamformer(
                sampling_rate_hz=SAMPLING_RATE_HZ,
                fmin=BAND_HZ[0],
                fmax=BAND_HZ[1],
                shading=np.hanning(num_sensors),
                domain=domain,
            )
            for num_beams in args.beams:
                delays = _steering_delays(num_sensors, num_beams)
                call = partial(beamformer.beamform, signals, delays)
                output = np.asarray(call())
                times = timeit.repeat(call, number=1, repeat=args.repeats)

                key = f"{domain}_{num_sensors}x{num_beams}"
                outputs[key] = output
                line = (
                    f"{domain:<16}{num_sensors:>8}{num_beams:>7}  {str(output.shape):<14}"
                    f"{1e3 * min(times):>9.1f}{1e3 * float(np.median(times)):>11.1f}"
                )
                if reference is not None:
                    verdict = _check(output, reference.get(key))
                    any_differs |= not verdict.startswith("match")
                    line += f"  {verdict}"
                print(line)

    if args.save:
        np.savez(args.save, **outputs)
        print(f"saved {len(outputs)} outputs to {args.save}")
    if any_differs:
        sys.exit(1)


if __name__ == "__main__":
    main()
