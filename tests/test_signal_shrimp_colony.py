"""Tests for SnappingShrimpColonySignal against its documented equations.

Oracles are closed forms and published values, not the model's own output: the level
definition (median window level equals ``median_source_level_db``), the unit-energy pulse,
the pass band, Poisson counting statistics, and the snap-rate fit's value at 20 degrees.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from scipy import signal as sp_signal
from stonesoup.types.groundtruth import GroundTruthState

from .support import isolate_real_bluepebble_imports, restore_real_bluepebble_imports

FS = 2000
BAND_HZ = (300.0, 800.0)
LEVEL_DB = 130.0
WINDOW_S = 0.5
SOURCE = GroundTruthState([0.0], metadata={})


@pytest.fixture(scope="module")
def bp():
    """Import the real classes without leaking caching into other tests.

    Module-scoped so the colony's calibration, cached per set of parameters, is computed once
    for the file and not once per test.
    """
    snapshot = isolate_real_bluepebble_imports()
    from bluepebble.signal import SnappingShrimpColonySignal, snap_rate_from_temperature
    from bluepebble.signal.biological import (
        DiffuseSnappingShrimpSignal,
        PointSourceSnappingShrimpSignal,
    )

    yield SimpleNamespace(
        DiffuseSnappingShrimpSignal=DiffuseSnappingShrimpSignal,
        PointSourceSnappingShrimpSignal=PointSourceSnappingShrimpSignal,
        SnappingShrimpColonySignal=SnappingShrimpColonySignal,
        snap_rate_from_temperature=snap_rate_from_temperature,
    )
    restore_real_bluepebble_imports(snapshot)


def _colony(bp, **kwargs):
    settings = dict(
        duration_s=60.0,
        sampling_rate_hz=FS,
        snap_rate_hz=100.0,
        median_source_level_db=LEVEL_DB,
        band_hz=BAND_HZ,
        calibration_window_s=WINDOW_S,
        # Long frames keep the STFT, which is not under test here, to a few frames.
        frame_len=8192,
        seed=1,
    )
    settings.update(kwargs)
    return bp.SnappingShrimpColonySignal(**settings)


def _median_window_level_db(waveform):
    """Return the band level of the median ``WINDOW_S`` window, in dB re 1 µPa²."""
    samples = int(WINDOW_S * FS)
    windows = waveform[: waveform.size // samples * samples].reshape(-1, samples)
    return 10 * np.log10(np.median(np.mean(np.abs(windows) ** 2, axis=1)))


def _negative_bin_fraction(waveform):
    """Return the share of a waveform's power in the bins ``fftfreq`` labels negative."""
    power = np.abs(np.fft.fft(waveform)) ** 2
    return power[np.fft.fftfreq(waveform.size) < 0].sum() / power.sum()


def test_snap_rate_fit_gives_34_snaps_per_second_at_20_degrees(bp):
    """(137 * 20 - 685) / 60 = 34.25 snaps per second, and none at or below 5 degrees."""
    assert bp.snap_rate_from_temperature(20.0) == pytest.approx(34.25)
    assert bp.snap_rate_from_temperature(5.0) == 0.0
    assert bp.snap_rate_from_temperature(0.0) == 0.0


@pytest.mark.parametrize(
    ("snap_rate_hz", "amplitude_cap", "duration_s"),
    [
        (100.0, None, 7200.0),
        (400.0, None, 7200.0),
        (100.0, 31.6, 4800.0),
        (400.0, 31.6, 4800.0),
    ],
)
def test_median_window_level_is_the_requested_level(bp, snap_rate_hz, amplitude_cap, duration_s):
    """The median level over ``calibration_window_s`` windows is ``median_source_level_db``.

    The runs are long enough for the tolerance to mean something: over eight seeds each
    case's median scattered by 0.03 to 0.06 dB (standard deviation) about the requested
    level, so 0.2 dB is at least three standard deviations.
    """
    waveform = _colony(
        bp, snap_rate_hz=snap_rate_hz, amplitude_cap=amplitude_cap, duration_s=duration_s
    ).get_source_waveform(SOURCE)

    assert _median_window_level_db(waveform) == pytest.approx(LEVEL_DB, abs=0.2)


def test_one_snap_carries_the_energy_the_calibration_gives_it(bp):
    """With every amplitude clipped to the same size, a snap's energy has a closed form.

    A tiny cap makes ``|A_i|`` equal the cap for every snap, so a window's energy is its snap
    count times one snap's energy ``E``, and the calibration sets ``median(count) * E`` to
    ``10^(L / 10) * T``. The median of a Poisson count of mean 50 is 50.
    """

    class OneSnap(bp.SnappingShrimpColonySignal):
        def _snap_samples(self):
            return np.array([self.num_samples // 2])

    waveform = OneSnap(
        duration_s=4.0,
        sampling_rate_hz=FS,
        snap_rate_hz=100.0,
        median_source_level_db=LEVEL_DB,
        band_hz=BAND_HZ,
        calibration_window_s=WINDOW_S,
        amplitude_cap=1e-9,
    ).get_source_waveform(SOURCE)

    snap_energy = np.sum(np.abs(waveform) ** 2) / FS
    assert snap_energy == pytest.approx(10 ** (LEVEL_DB / 10) * WINDOW_S / 50, rel=0.01)


def test_real_part_is_the_snap_pressure_over_root_two(bp):
    """The real part of one snap is its zero-phase Butterworth pulse over ``sqrt(2)``.

    The pulse is built here from the band alone and scaled to the snap's energy, so the
    comparison covers the shape and the ``1 / sqrt(2)`` together. The snap's sign is random.
    """

    class OneSnap(bp.SnappingShrimpColonySignal):
        def _snap_samples(self):
            return np.array([self.num_samples // 2])

    model = OneSnap(
        duration_s=4.0,
        sampling_rate_hz=FS,
        snap_rate_hz=100.0,
        median_source_level_db=LEVEL_DB,
        band_hz=BAND_HZ,
        calibration_window_s=WINDOW_S,
        amplitude_cap=1e-9,
    )
    waveform = model.get_source_waveform(SOURCE)

    impulse = np.zeros(model.num_samples)
    impulse[model.num_samples // 2] = 1.0
    sos = sp_signal.butter(8, BAND_HZ, btype="bandpass", fs=FS, output="sos")
    pulse = sp_signal.sosfiltfilt(sos, impulse)
    pressure = pulse * np.sqrt(np.sum(np.abs(waveform) ** 2) / np.sum(pulse**2))
    sign = np.sign(waveform.real[np.argmax(np.abs(pulse))] * pulse[np.argmax(np.abs(pulse))])

    np.testing.assert_allclose(
        waveform.real, sign * pressure / np.sqrt(2.0), atol=1e-9 * np.abs(pressure).max()
    )


def test_power_lies_in_the_pass_band(bp):
    """At least 99% of the waveform's power is inside ``band_hz``."""
    waveform = np.real(_colony(bp, duration_s=120.0).get_source_waveform(SOURCE))

    frequencies, psd = sp_signal.welch(waveform, fs=FS, nperseg=4096)
    in_band = (frequencies >= BAND_HZ[0]) & (frequencies <= BAND_HZ[1])
    assert psd[in_band].sum() / psd.sum() >= 0.99


def test_snap_count_is_poisson_about_rate_times_duration(bp):
    """The number of snaps is within three standard deviations of ``rate * duration``."""
    model = _colony(bp, snap_rate_hz=100.0, duration_s=600.0)

    snap_samples = model._snap_samples()

    expected = 100.0 * 600.0
    assert abs(snap_samples.size - expected) < 3 * np.sqrt(expected)
    assert np.all(np.diff(snap_samples) >= 0)
    assert snap_samples.min() >= 0
    assert snap_samples.max() < model.num_samples


@pytest.mark.parametrize("duration_s", [60.0, 60.0005])
def test_output_is_a_one_sided_waveform_of_the_run_length(bp, duration_s):
    """The waveform is finite, ``num_samples`` long (even and odd) and has no negative bins."""
    model = _colony(bp, duration_s=duration_s)

    waveform = model.get_source_waveform(SOURCE)

    assert waveform.shape == (model.num_samples,)
    assert model.num_samples % 2 == (duration_s != 60.0)
    assert waveform.dtype == np.complex128
    assert np.all(np.isfinite(waveform))
    assert _negative_bin_fraction(waveform) < 1e-12


def test_a_colony_that_never_snaps_is_silent(bp):
    """A zero snap rate gives a zero waveform, not an error."""
    waveform = _colony(bp, snap_rate_hz=0.0).get_source_waveform(SOURCE)

    assert not np.any(waveform)


def test_a_window_that_is_sometimes_empty_still_calibrates(bp):
    """Three snaps per window on average leaves some windows empty, and still has a median.

    Over eight seeds the median scattered by 0.15 dB, so 0.5 dB is three standard deviations.
    """
    waveform = _colony(bp, snap_rate_hz=6.0, duration_s=1800.0).get_source_waveform(SOURCE)

    assert _median_window_level_db(waveform) == pytest.approx(LEVEL_DB, abs=0.5)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"band_hz": (300.0, 1200.0)}, "Nyquist"),
        ({"band_hz": (800.0, 300.0)}, "band_hz"),
        ({"band_hz": (0.0, 500.0)}, "band_hz"),
        ({"alpha": 0.0}, "alpha"),
        ({"alpha": 2.5}, "alpha"),
        ({"snap_rate_hz": -1.0}, "snap_rate_hz"),
        ({"calibration_window_s": 0.0}, "calibration_window_s"),
        ({"filter_order": 0}, "filter_order"),
        ({"amplitude_cap": 0.0}, "amplitude_cap"),
        ({"snap_rate_hz": 0.2}, "too few"),
    ],
)
def test_invalid_parameters_are_rejected_at_construction(bp, kwargs, message):
    """Each bad parameter raises a ValueError that names it."""
    with pytest.raises(ValueError, match=message):
        _colony(bp, **kwargs)


def test_old_shrimp_classes_are_deprecated_in_favour_of_the_colony(bp):
    """Both old classes warn on construction and name the replacement."""
    with pytest.warns(DeprecationWarning, match="use SnappingShrimpColonySignal instead"):
        bp.PointSourceSnappingShrimpSignal(
            temperature_celsius=20.0, duration_s=0.1, sampling_rate_hz=48_000
        )
    with pytest.warns(DeprecationWarning, match="not spatially diffuse"):
        bp.DiffuseSnappingShrimpSignal(
            temperature_celsius=20.0, duration_s=0.1, sampling_rate_hz=48_000, ssp=None
        )
