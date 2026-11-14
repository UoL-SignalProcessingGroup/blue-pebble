"""Tests that the ship sources emit one-sided waveforms at their stated level.

A one-sided (analytic) waveform has no power in the bins ``numpy.fft.fftfreq`` labels negative,
and its complex RMS is the source's stated RMS amplitude. The oracles are that definition, the
closed form of the Gaussian tonal filter, the power-law slope, and for the recorded ship the
recording itself. The colony, whale and deprecated shrimp sources are covered beside their own
tests.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.io import wavfile
from scipy.special import erf
from stonesoup.types.groundtruth import GroundTruthState

from .support import isolate_real_bluepebble_imports, restore_real_bluepebble_imports

FS = 500
TONAL_HZ = 100.0
BANDWIDTH_HZ = 4.0


@pytest.fixture(scope="module")
def bp():
    """Import the real classes without leaking caching into other tests."""
    snapshot = isolate_real_bluepebble_imports()
    from bluepebble.signal import RecordedAnthropogenicSignal, SyntheticAnthropogenicSignal

    yield SimpleNamespace(
        RecordedAnthropogenicSignal=RecordedAnthropogenicSignal,
        SyntheticAnthropogenicSignal=SyntheticAnthropogenicSignal,
    )
    restore_real_bluepebble_imports(snapshot)


def _ship_source(**overrides):
    """Return a source with one tonal and no broadband noise; overrides change its metadata."""
    metadata = {
        "amplitudes_upa": np.array([3.0]),
        "frequencies_hz": np.array([TONAL_HZ]),
        "phases_rad": np.array([0.0]),
        "tonal_bandwidth_hz": BANDWIDTH_HZ,
        "noise_amplitude_upa": 0.0,
        "noise_spectral_exponent": -1.0,
    }
    metadata.update(overrides)
    return GroundTruthState([0.0], metadata=metadata)


def _noise_source(**overrides):
    """Return a source with broadband noise of RMS 5 µPa and no tonals."""
    empty = np.array([])
    return _ship_source(
        amplitudes_upa=empty,
        frequencies_hz=empty,
        phases_rad=empty,
        noise_amplitude_upa=5.0,
        **overrides,
    )


def _ship(bp, duration_s=8.0, seed=1, **kwargs):
    return bp.SyntheticAnthropogenicSignal(
        duration_s=duration_s, sampling_rate_hz=FS, frame_len=256, seed=seed, **kwargs
    )


def _rms(waveform):
    return np.sqrt(np.mean(np.abs(waveform) ** 2))


def _power_spectrum(waveform):
    """Return each bin's signed frequency in Hz and its power."""
    return np.fft.fftfreq(waveform.size, 1 / FS), np.abs(np.fft.fft(waveform)) ** 2


def _negative_bin_fraction(waveform):
    """Return the share of a waveform's power in the bins ``fftfreq`` labels negative."""
    frequencies, power = _power_spectrum(waveform)
    return power[frequencies < 0].sum() / power.sum()


# 8 s is an even number of samples at 500 Hz and 8.002 s an odd one.
@pytest.mark.parametrize("duration_s", [8.0, 8.002])
@pytest.mark.parametrize("use_powerlaw_noise", [False, True])
def test_synthetic_ship_is_one_sided(bp, duration_s, use_powerlaw_noise):
    """Tonals and noise together, over the whole band, leave the negative bins empty."""
    model = _ship(
        bp,
        duration_s,
        noise_freq_range_hz=(0.0, FS / 2),
        use_powerlaw_noise=use_powerlaw_noise,
    )
    source = _ship_source(
        amplitudes_upa=np.array([3.0, 2.0]),
        # A filter this wide at 2 Hz reaches below 0 Hz, and at 248 Hz reaches past fs/2.
        frequencies_hz=np.array([2.0, 248.0]),
        phases_rad=np.array([0.3, 1.1]),
        noise_amplitude_upa=5.0,
    )

    waveform = model.get_source_waveform(source)

    assert waveform.size % 2 == (duration_s != 8.0)
    assert _negative_bin_fraction(waveform) < 1e-12


def test_tonal_rms_is_its_stated_amplitude(bp):
    """A tonal of amplitude 3 µPa has a complex RMS of 3 µPa."""
    waveform = _ship(bp).get_source_waveform(_ship_source())

    assert _rms(waveform) == pytest.approx(3.0, rel=1e-12)


@pytest.mark.parametrize("use_powerlaw_noise", [False, True])
def test_noise_rms_is_its_stated_amplitude(bp, use_powerlaw_noise):
    """Broadband noise of amplitude 5 µPa has a complex RMS of 5 µPa."""
    model = _ship(bp, noise_freq_range_hz=(20.0, 200.0), use_powerlaw_noise=use_powerlaw_noise)

    assert _rms(model.get_source_waveform(_noise_source())) == pytest.approx(5.0, rel=1e-12)


def test_tonal_power_follows_the_gaussian_filter_about_its_frequency(bp):
    """The share of a tonal's power within ``bandwidth / 2`` of its frequency is erf(1.1775).

    The filter's amplitude response is a Gaussian of full width ``bandwidth`` at half
    maximum, ``sigma = bandwidth / 2.355``, so the power spectrum is a Gaussian of standard
    deviation ``sigma / sqrt(2)`` and the share inside ``+-bandwidth / 2`` is
    ``erf(2.355 / 2) = 0.904``. Were ``bandwidth`` the -3 dB width the share would be 0.761.
    Over eight seeds of this 400 s waveform the share scattered by 0.004 (standard
    deviation) and the centroid by 0.02 Hz.
    """
    waveform = _ship(bp, duration_s=400.0).get_source_waveform(_ship_source())

    frequencies, power = _power_spectrum(waveform)
    inside = np.abs(frequencies - TONAL_HZ) <= BANDWIDTH_HZ / 2
    assert power[inside].sum() / power.sum() == pytest.approx(erf(2.355 / 2), abs=0.02)
    assert np.sum(frequencies * power) / power.sum() == pytest.approx(TONAL_HZ, abs=0.1)


def test_tonal_phase_rotates_the_complex_envelope(bp):
    """Raising ``phases_rad`` by phi multiplies the tonal by ``exp(i * phi)``."""
    phi = 0.7
    reference = _ship(bp, seed=4).get_source_waveform(_ship_source())
    rotated = _ship(bp, seed=4).get_source_waveform(_ship_source(phases_rad=np.array([phi])))

    np.testing.assert_allclose(rotated, reference * np.exp(1j * phi), atol=1e-12)


def test_power_law_noise_has_the_stated_slope_and_band_edges(bp):
    """With ``use_powerlaw_noise`` the power at ``f`` is proportional to ``f ** exponent``."""
    model = _ship(bp, noise_freq_range_hz=(20.0, 200.0), use_powerlaw_noise=True)
    waveform = model.get_source_waveform(_noise_source(noise_spectral_exponent=-2.0))

    frequencies, power = _power_spectrum(waveform)
    in_band = (frequencies >= 20.0) & (frequencies <= 200.0)
    assert power[~in_band].sum() < 1e-12 * power.sum()
    at_40_hz = power[np.argmin(np.abs(frequencies - 40.0))]
    at_160_hz = power[np.argmin(np.abs(frequencies - 160.0))]
    assert at_160_hz / at_40_hz == pytest.approx(4.0**-2.0, rel=1e-9)


def test_random_noise_has_the_stated_slope_and_band_edges(bp):
    """Pink noise has equal power in each octave, and none outside the band.

    Over eight seeds of this 400 s waveform the ratio of the 100 to 200 Hz octave to the
    25 to 50 Hz octave scattered by 0.01 (standard deviation).
    """
    model = _ship(bp, duration_s=400.0, noise_freq_range_hz=(20.0, 220.0))
    waveform = model.get_source_waveform(_noise_source(noise_spectral_exponent=-1.0))

    frequencies, power = _power_spectrum(waveform)
    in_band = (frequencies >= 20.0) & (frequencies <= 220.0)
    assert power[~in_band].sum() < 1e-12 * power.sum()
    low_octave = power[(frequencies >= 25.0) & (frequencies < 50.0)].sum()
    high_octave = power[(frequencies >= 100.0) & (frequencies < 200.0)].sum()
    assert high_octave / low_octave == pytest.approx(1.0, abs=0.05)


@pytest.mark.parametrize("frequency_hz", [0.0, -50.0, FS / 2, 300.0])
def test_tonal_outside_the_sampled_band_is_rejected(bp, frequency_hz):
    """A tonal must lie above 0 Hz and below fs/2."""
    source = _ship_source(
        frequencies_hz=np.array([TONAL_HZ, frequency_hz]),
        amplitudes_upa=np.ones(2),
        phases_rad=np.zeros(2),
    )

    with pytest.raises(ValueError, match="frequencies_hz must lie inside"):
        _ship(bp).get_source_waveform(source)


@pytest.mark.parametrize("noise_range", [(20.0, 251.0), (-1.0, 200.0), (200.0, 20.0)])
def test_noise_range_outside_the_sampled_band_is_rejected(bp, noise_range):
    """The noise range must satisfy ``0 <= low < high <= fs/2``."""
    model = _ship(bp, noise_freq_range_hz=noise_range)

    with pytest.raises(ValueError, match="noise_freq_range_hz must satisfy"):
        model.get_source_waveform(_noise_source())


def test_noise_range_is_not_checked_for_a_source_without_noise(bp):
    """The default 20 to 200 Hz range does not stop a tonal-only source at a low rate."""
    model = bp.SyntheticAnthropogenicSignal(
        duration_s=8.0, sampling_rate_hz=100, frame_len=64, seed=1
    )

    waveform = model.get_source_waveform(_ship_source(frequencies_hz=np.array([20.0])))

    assert _rms(waveform) == pytest.approx(3.0)


def _recording(num_samples, offset=0.0):
    """Return seeded noise with no power at 0 Hz or fs/2, as float32, plus ``offset``."""
    spectrum = np.fft.rfft(np.random.default_rng(6).standard_normal(num_samples))
    spectrum[0] = 0.0
    if num_samples % 2 == 0:
        spectrum[-1] = 0.0
    waveform = np.fft.irfft(spectrum, num_samples)
    return (0.1 * waveform / np.abs(waveform).max() + offset).astype(np.float32)


def _recorded_ship(bp, tmp_path, recording, level_db=100.0):
    wav_path = tmp_path / "ship.wav"
    wavfile.write(wav_path, FS, recording)
    return bp.RecordedAnthropogenicSignal(
        wav_path=str(wav_path),
        duration_s=recording.size / FS,
        sampling_rate_hz=FS,
        frame_len=256,
        level_db_re_1upa=level_db,
    )


@pytest.mark.parametrize("num_samples", [4000, 4001])
def test_recorded_ship_is_one_sided_at_its_stated_level(bp, tmp_path, num_samples):
    """The complex RMS is ``10 ** (level / 20)`` µPa and the negative bins are empty."""
    model = _recorded_ship(bp, tmp_path, _recording(num_samples, offset=0.02))

    waveform = model.get_source_waveform(GroundTruthState([0.0], metadata={}))

    assert waveform.size == num_samples
    assert _negative_bin_fraction(waveform) < 1e-12
    assert _rms(waveform) == pytest.approx(10 ** (100.0 / 20), rel=1e-12)


@pytest.mark.parametrize("num_samples", [4000, 4001])
def test_recorded_ship_real_part_is_the_recording_over_root_two(bp, tmp_path, num_samples):
    """For a zero-mean recording the real part is the levelled recording over ``sqrt(2)``.

    The levelled recording is the recording scaled to an RMS of ``10 ** (level / 20)`` µPa,
    which is what this source returned before v0.9.0. The float32 samples keep 1e-9 of the
    power at 0 Hz and fs/2, hence the tolerance.
    """
    recording = _recording(num_samples)
    model = _recorded_ship(bp, tmp_path, recording)

    waveform = model.get_source_waveform(GroundTruthState([0.0], metadata={}))

    pressure = recording.astype(np.float64)
    pressure *= 10 ** (100.0 / 20) / _rms(pressure)
    np.testing.assert_allclose(waveform.real, pressure / np.sqrt(2.0), atol=1e-6 * _rms(pressure))
