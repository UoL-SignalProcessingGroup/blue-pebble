"""Tests for SnappingShrimpColonySignal against its documented equations.

Oracles are closed forms and published values, not the model's own output: the level
definition (median window level equals ``median_source_level_db``), the unit-energy pulse,
the pass band, Poisson counting statistics, the doubly stochastic Fano factor, the lognormal
level spread, the pulse's spectral slope, and the snap-rate fit's value at 20 degrees.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from scipy import signal as sp_signal
from scipy.stats import levy_stable, lognorm
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
    from bluepebble.signal import (
        AlphaStableAmplitudes,
        LognormalAmplitudes,
        SnappingShrimpColonySignal,
        snap_rate_from_temperature,
    )
    from bluepebble.signal.biological import (
        DiffuseSnappingShrimpSignal,
        PointSourceSnappingShrimpSignal,
    )

    yield SimpleNamespace(
        DiffuseSnappingShrimpSignal=DiffuseSnappingShrimpSignal,
        PointSourceSnappingShrimpSignal=PointSourceSnappingShrimpSignal,
        AlphaStableAmplitudes=AlphaStableAmplitudes,
        LognormalAmplitudes=LognormalAmplitudes,
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
    with pytest.warns(DeprecationWarning, match="snap_rate_from_temperature"):
        assert bp.snap_rate_from_temperature(20.0) == pytest.approx(34.25)
    with pytest.warns(DeprecationWarning):
        assert bp.snap_rate_from_temperature(5.0) == 0.0
    with pytest.warns(DeprecationWarning):
        assert bp.snap_rate_from_temperature(0.0) == 0.0


def test_default_alpha_lies_in_the_measured_range(bp):
    """The default exponent is inside the 1.6-1.9 fitted to shrimp-dominated noise."""
    model = bp.SnappingShrimpColonySignal(
        duration_s=1.0,
        sampling_rate_hz=FS,
        snap_rate_hz=100.0,
        median_source_level_db=LEVEL_DB,
        band_hz=BAND_HZ,
        calibration_window_s=WINDOW_S,
    )
    assert 1.6 <= model.amplitude_distribution.alpha <= 1.9


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
    comparison covers the shape, the ``1 / sqrt(2)`` and the polarity together: snap
    amplitudes are magnitudes, so the snap has the pulse's own sign.
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

    np.testing.assert_allclose(
        waveform.real, pressure / np.sqrt(2.0), atol=1e-9 * np.abs(pressure).max()
    )


def test_every_snap_has_positive_polarity(bp):
    """Snap amplitudes are magnitudes, so every snap's central peak is positive.

    Snaps are spaced far apart so their pulses do not overlap; the real waveform at each snap
    sample is then that snap's central peak, positive whatever the draw.
    """
    spacing = 2_000

    class Sparse(bp.SnappingShrimpColonySignal):
        def _snap_samples(self):
            return np.arange(spacing, self.num_samples - spacing, spacing)

    model = Sparse(
        duration_s=20.0,
        sampling_rate_hz=FS,
        snap_rate_hz=100.0,
        median_source_level_db=LEVEL_DB,
        band_hz=BAND_HZ,
        calibration_window_s=WINDOW_S,
        seed=3,
    )
    waveform = model.get_source_waveform(SOURCE)
    peaks = waveform.real[model._snap_samples()]
    assert peaks.size > 10
    assert np.all(peaks > 0)


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


def test_default_colony_is_poisson_times_alpha_stable_amplitudes_and_a_flat_pulse(bp):
    """With no option set, the waveform is the 0.10.0 model, rebuilt here from its draws.

    From one generator state the model draws a Poisson count, uniform snap samples and
    alpha-stable amplitudes, in that order, and passes the train through the band-pass alone.
    """
    model = _colony(bp, duration_s=20.0)
    model._rng = np.random.default_rng(7)
    waveform = model.get_source_waveform(SOURCE)

    rng = np.random.default_rng(7)
    count = rng.poisson(100.0 * 20.0)
    samples = np.sort(rng.integers(0, model.num_samples, count))
    amplitudes = np.abs(levy_stable.rvs(1.7, 0.0, size=count, random_state=rng))
    train = np.zeros(model.num_samples)
    np.add.at(train, samples, model._amplitude_scale() * amplitudes)
    sos, gain = model._unit_energy_filter()
    pressure = gain * sp_signal.sosfiltfilt(sos, train)

    # The one-sided form treats the 0 Hz and Nyquist bins on their own, so compare the rest.
    spectrum = np.fft.rfft(np.sqrt(2.0) * waveform.real)[1:-1]
    expected = np.fft.rfft(pressure)[1:-1]
    np.testing.assert_allclose(spectrum, expected, atol=1e-9 * np.abs(expected).max())


def test_lognormal_snap_levels_have_the_requested_spread(bp):
    """Lognormal snap levels are normal in dB about the unit scale, with the set deviation.

    Snaps are spaced so their pulses do not overlap, so the waveform at each snap sample is
    that snap's amplitude times one constant. With 4999 snaps the standard error of the
    deviation is 0.06 dB and that of the median 0.1 dB.
    """
    spacing = 400

    class Sparse(bp.SnappingShrimpColonySignal):
        def _snap_samples(self):
            return np.arange(spacing, self.num_samples - spacing, spacing)

    model = Sparse(
        duration_s=1000.0,
        sampling_rate_hz=FS,
        snap_rate_hz=100.0,
        median_source_level_db=LEVEL_DB,
        band_hz=BAND_HZ,
        calibration_window_s=WINDOW_S,
        amplitude_distribution=bp.LognormalAmplitudes(6.0),
        seed=3,
    )
    waveform = model.get_source_waveform(SOURCE)
    sos, gain = model._unit_energy_filter()
    impulse = np.zeros(4096)
    impulse[2048] = 1.0
    unit_peak = model._amplitude_scale() * gain * sp_signal.sosfiltfilt(sos, impulse)[2048]

    levels_db = 20 * np.log10(np.sqrt(2.0) * waveform.real[model._snap_samples()] / unit_peak)

    assert levels_db.std() == pytest.approx(6.0, abs=0.25)
    assert np.median(levels_db) == pytest.approx(0.0, abs=0.4)


def _fano_factor_theory(rate_hz, log_sd, correlation_s, counting_s):
    """Fano factor of a Poisson process whose log-rate is an OU process (class docstring)."""
    u = np.linspace(0.0, counting_s, 20_001)
    covariance = np.exp(log_sd**2 * np.exp(-u / correlation_s)) - 1
    return 1 + 2 * rate_hz / counting_s * np.trapezoid((counting_s - u) * covariance, u)


@pytest.mark.parametrize("counting_s", [0.25, 1.0, 4.0])
def test_fluctuating_rate_keeps_the_mean_rate_and_gives_the_theoretical_fano_factor(
    bp, counting_s
):
    """Snap counts keep their mean and take the doubly stochastic variance.

    Two hours at 100 snaps per second, a log-rate deviation of 0.3 and a correlation time of
    0.5 s. Over eight seeds the Fano factor averaged within 1% of the closed form and
    scattered by 1.2%, 2.3% and 5.0% at the three counting times, so 16% is at least three
    standard deviations; the mean count scattered by 0.36%.
    """
    model = _colony(bp, duration_s=7200.0, rate_log_sd=0.3, rate_correlation_s=0.5)

    snap_samples = model._snap_samples()

    samples_per_window = int(counting_s * FS)
    num_windows = model.num_samples // samples_per_window
    counts = np.bincount(snap_samples // samples_per_window)[:num_windows]
    assert counts.mean() == pytest.approx(100.0 * counting_s, rel=0.015)
    assert counts.var(ddof=1) / counts.mean() == pytest.approx(
        _fano_factor_theory(100.0, 0.3, 0.5, counting_s), rel=0.16
    )
    assert np.all(np.diff(snap_samples) >= 0)
    assert snap_samples.min() >= 0
    assert snap_samples.max() < model.num_samples


@pytest.mark.parametrize(
    ("spread_db", "options"),
    [
        (10.0, {}),
        (None, {"rate_log_sd": 0.8, "rate_correlation_s": 2.0}),
        (None, {"spectral_slope_db_per_octave": -3.0}),
        (
            10.0,
            {"rate_log_sd": 0.8, "rate_correlation_s": 2.0, "spectral_slope_db_per_octave": -3.0},
        ),
    ],
)
def test_median_window_level_holds_under_each_option(bp, spread_db, options):
    """Each option, and all three together, leave the median window at the requested level.

    The first and last cases draw lognormal amplitudes of 10 dB spread.

    The fluctuating rate has a correlation time four times the window, so whole windows are
    busy or quiet; the calibration draws its window counts from the same process. Over eight
    seeds the median scattered by 0.02 to 0.11 dB about a mean within 0.05 dB of the requested
    level, so 0.35 dB is at least three standard deviations.
    """
    if spread_db is not None:
        options = options | {"amplitude_distribution": bp.LognormalAmplitudes(spread_db)}
    waveform = _colony(bp, duration_s=7200.0, **options).get_source_waveform(SOURCE)

    assert _median_window_level_db(waveform) == pytest.approx(LEVEL_DB, abs=0.35)


def test_alpha_is_not_an_argument_of_the_colony(bp):
    """The exponent belongs to ``AlphaStableAmplitudes``; the colony no longer takes it."""
    with pytest.raises(TypeError, match="alpha"):
        _colony(bp, alpha=1.9)


@pytest.mark.parametrize("distribution", ["lognormal", lognorm(s=1.0), 1.7])
def test_only_the_two_amplitude_distributions_are_accepted(bp, distribution):
    """A name, a SciPy distribution or a number is rejected, with the accepted two named."""
    with pytest.raises(ValueError, match="AlphaStableAmplitudes or LognormalAmplitudes"):
        _colony(bp, amplitude_distribution=distribution)


def test_amplitude_distributions_check_their_parameters(bp):
    """An exponent outside (0, 2] and a spread of zero or less are rejected."""
    with pytest.raises(ValueError, match="alpha"):
        bp.AlphaStableAmplitudes(2.5)
    with pytest.raises(ValueError, match="alpha"):
        bp.AlphaStableAmplitudes(0.0)
    with pytest.raises(ValueError, match="spread_db"):
        bp.LognormalAmplitudes(0.0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"band_hz": (300.0, 1200.0)}, "Nyquist"),
        ({"band_hz": (800.0, 300.0)}, "band_hz"),
        ({"band_hz": (0.0, 500.0)}, "band_hz"),
        ({"snap_rate_hz": -1.0}, "snap_rate_hz"),
        ({"calibration_window_s": 0.0}, "calibration_window_s"),
        ({"filter_order": 0}, "filter_order"),
        ({"amplitude_cap": 0.0}, "amplitude_cap"),
        ({"rate_log_sd": -0.1}, "rate_log_sd"),
        ({"rate_log_sd": 0.5}, "rate_correlation_s"),
        ({"rate_log_sd": 0.5, "rate_correlation_s": 0.0}, "rate_correlation_s"),
        ({"spectral_slope_db_per_octave": float("nan")}, "spectral_slope_db_per_octave"),
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
