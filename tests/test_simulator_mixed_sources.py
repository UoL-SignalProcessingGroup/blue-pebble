"""Tests that the STFT simulator takes real and complex sources in the same run.

A real source waveform has a one-sided STFT and a complex one a two-sided STFT. The simulator
is linear in its sources, so superposition is the oracle: what a real source adds to a run
with a complex source must equal what it produces on its own.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import signal as sp_signal
from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel,
    ConstantVelocity,
)
from stonesoup.types.array import StateVector
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState

from .support import isolate_real_bluepebble_imports, restore_real_bluepebble_imports

START = datetime(2026, 1, 1)
C = 1500.0
FS = 2000
NUM_STEPS = 6
STEP_S = 1.0
BAND_HZ = (200.0, 600.0)
RANGE_M = 300.0
# Runs agree to complex64 rounding, measured at 3e-7 of the peak.
ROUNDING = 1e-5


@pytest.fixture
def bp():
    """Import the real classes without leaking caching into other tests."""
    snapshot = isolate_real_bluepebble_imports()
    from stonesoup.base import Property

    from bluepebble.arrays import RigidArray
    from bluepebble.models.environment import Constant
    from bluepebble.models.propagation import SphericalAcousticPropagationModel
    from bluepebble.platform import SonarPlatform
    from bluepebble.signal import AnthropogenicSignal
    from bluepebble.simulator import ContinuousSTFTPassiveSonarArraySimulator

    class Line(RigidArray):
        """Eight sensors 0.5 m apart along the platform's x axis."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._local_offsets = np.vstack([np.arange(8) * 0.5, np.zeros(8), np.zeros(8)])
            self._local_axis = np.array([1.0, 0.0, 0.0])

    class BandNoise(AnthropogenicSignal):
        """Seeded noise in ``BAND_HZ``, so the DC and Nyquist bins carry no energy."""

        complex_valued: bool = Property(default=False, doc="Add an independent imaginary part")
        noise_seed: int = Property(default=0, doc="Seed of this source's waveform")

        def _generate_base_signal(self, source):
            rng = np.random.default_rng(self.noise_seed)
            sos = sp_signal.butter(6, BAND_HZ, btype="bandpass", fs=FS, output="sos")
            waveform = sp_signal.sosfiltfilt(sos, rng.standard_normal(self.num_samples))
            if self.complex_valued:
                imaginary = sp_signal.sosfiltfilt(sos, rng.standard_normal(self.num_samples))
                return (waveform + 1j * imaginary).astype(np.complex128)
            return waveform.astype(np.complex128)

    yield SimpleNamespace(
        BandNoise=BandNoise,
        Constant=Constant,
        ContinuousSTFTPassiveSonarArraySimulator=ContinuousSTFTPassiveSonarArraySimulator,
        Line=Line,
        SonarPlatform=SonarPlatform,
        SphericalAcousticPropagationModel=SphericalAcousticPropagationModel,
    )
    restore_real_bluepebble_imports(snapshot)


REAL = {"complex_valued": False, "noise_seed": 1, "bearing_deg": 60.0}
COMPLEX = {"complex_valued": True, "noise_seed": 2, "bearing_deg": 120.0}


def _timestamps():
    return [START + timedelta(seconds=STEP_S * step) for step in range(NUM_STEPS)]


def _array(bp):
    platform = bp.SonarPlatform(
        states=[GroundTruthState(StateVector([0.0, 0.0, 0.0, 0.0, -50.0, 0.0]), timestamp=START)],
        position_mapping=[0, 2, 4],
        transition_models=[CombinedLinearGaussianTransitionModel([ConstantVelocity(0.0)] * 3)],
        transition_times=[timedelta(hours=1)],
    )
    for timestamp in _timestamps()[1:]:
        platform.move(timestamp)
    array = bp.Line()
    platform.add_sensor(array)
    return array


def _stationary_path(bearing_deg):
    bearing = np.deg2rad(bearing_deg)
    state_vector = StateVector(
        [RANGE_M * np.cos(bearing), 0.0, RANGE_M * np.sin(bearing), 0.0, -50.0, 0.0]
    )
    return GroundTruthPath(
        [
            GroundTruthState(state_vector, timestamp=ts, metadata={"position_mapping": [0, 2, 4]})
            for ts in _timestamps()
        ]
    )


def _received(bp, sources, mode="stft_interp", frame_len=256):
    """Return the noise-free sensor signals of one run, shape (num_sensors, num_samples)."""
    simulator = bp.ContinuousSTFTPassiveSonarArraySimulator(
        array=_array(bp),
        propagation_model=bp.SphericalAcousticPropagationModel(
            ssp=bp.Constant(speed=C), attenuation_factor=0.0
        ),
        signal_models=[
            bp.BandNoise(
                duration_s=NUM_STEPS * STEP_S,
                sampling_rate_hz=FS,
                frame_len=frame_len,
                complex_valued=source["complex_valued"],
                noise_seed=source["noise_seed"],
            )
            for source in sources
        ],
        ground_truth_paths=[_stationary_path(source["bearing_deg"]) for source in sources],
        mode=mode,
    )
    return np.concatenate(
        [next(iter(data)).raw_signals for _, data in simulator.sensor_data_gen()], axis=1
    )


@pytest.mark.parametrize("mode", ["stft_interp", "wola_interp", "cola"])
@pytest.mark.parametrize("frame_len", [256, 255])
def test_real_source_adds_the_same_signal_beside_a_complex_source(bp, mode, frame_len):
    """Superposition: (real + complex) - complex equals the real source's own run."""
    real_alone = _received(bp, [REAL], mode, frame_len)
    complex_alone = _received(bp, [COMPLEX], mode, frame_len)
    together = _received(bp, [REAL, COMPLEX], mode, frame_len)

    peak = np.max(np.abs(real_alone))
    assert peak > 0
    np.testing.assert_allclose(together - complex_alone, real_alone, atol=ROUNDING * peak)


def test_source_order_does_not_change_the_sum(bp):
    """Superposition is symmetric, so which source comes first cannot matter."""
    real_first = _received(bp, [REAL, COMPLEX])
    complex_first = _received(bp, [COMPLEX, REAL])

    peak = np.max(np.abs(real_first))
    np.testing.assert_allclose(complex_first, real_first, atol=ROUNDING * peak)
