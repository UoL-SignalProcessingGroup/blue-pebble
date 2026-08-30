"""Unit tests for array-resolution derived processing windows."""

from __future__ import annotations

import math

import numpy as np
import pytest

from bluepebble.sigproc import beams_per_mainlobe, cfar_window_for_mainlobe

# A 200-sensor, 0.5 m array over a 361-beam full circle -- the geometry the detector
# examples use.
APERTURE_M = (200 - 1) * 0.5
BEAM_SPACING_RAD = np.deg2rad(1.0)


def test_mainlobe_matches_the_textbook_beamwidth() -> None:
    """0.886 * lambda / L, converted to beams, is the whole formula."""
    mainlobe = beams_per_mainlobe(APERTURE_M, 50.0, BEAM_SPACING_RAD, sound_speed_ms=1500.0)

    expected_rad = 0.886 * (1500.0 / 50.0) / APERTURE_M
    assert mainlobe == pytest.approx(expected_rad / BEAM_SPACING_RAD)
    # ~15 beams at the low end of the examples' 50-200 Hz band
    assert mainlobe == pytest.approx(15.3, abs=0.1)


def test_mainlobe_narrows_with_frequency() -> None:
    """Halving the wavelength halves the mainlobe."""
    low = beams_per_mainlobe(APERTURE_M, 50.0, BEAM_SPACING_RAD)
    high = beams_per_mainlobe(APERTURE_M, 100.0, BEAM_SPACING_RAD)

    assert high == pytest.approx(low / 2)


def test_mainlobe_broadens_away_from_broadside() -> None:
    """A line array loses resolution as 1/cos towards endfire."""
    broadside = beams_per_mainlobe(APERTURE_M, 50.0, BEAM_SPACING_RAD)
    off = beams_per_mainlobe(APERTURE_M, 50.0, BEAM_SPACING_RAD, steering_rad=np.deg2rad(60.0))

    assert off == pytest.approx(broadside / math.cos(np.deg2rad(60.0)))
    assert off > broadside


def test_shading_widens_the_mainlobe() -> None:
    """A taper trades mainlobe width for sidelobe suppression."""
    uniform = beams_per_mainlobe(APERTURE_M, 50.0, BEAM_SPACING_RAD)
    hann = beams_per_mainlobe(APERTURE_M, 50.0, BEAM_SPACING_RAD, shading_factor=0.886 * 1.5)

    assert hann == pytest.approx(uniform * 1.5)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"aperture_m": 0.0}, "aperture_m"),
        ({"frequency_hz": 0.0}, "frequency_hz"),
        ({"beam_spacing_rad": -1.0}, "beam_spacing_rad"),
        ({"sound_speed_ms": 0.0}, "sound_speed_ms"),
        ({"steering_rad": np.pi / 2}, "endfire"),
    ],
)
def test_mainlobe_rejects_degenerate_geometry(kwargs, match) -> None:
    """Each of these makes the beamwidth meaningless rather than merely large."""
    args = {"aperture_m": APERTURE_M, "frequency_hz": 50.0, "beam_spacing_rad": BEAM_SPACING_RAD}
    args.update(kwargs)
    with pytest.raises(ValueError, match=match):
        beams_per_mainlobe(**args)


def test_guard_band_reaches_past_the_mainlobe() -> None:
    """The point of the derivation: training cells must fall outside the lobe.

    A target spans roughly +/- mainlobe/2 beams about its peak, so guard cells have to
    cover at least that half-width or the training cells measure the target and the
    reported SNR collapses.
    """
    mainlobe = beams_per_mainlobe(APERTURE_M, 50.0, BEAM_SPACING_RAD)
    guard, training, _ = cfar_window_for_mainlobe(mainlobe)

    assert guard >= mainlobe / 2
    assert training > guard


def test_peak_distance_is_one_mainlobe() -> None:
    """Two candidates closer than a mainlobe are not resolvable as separate sources."""
    mainlobe = beams_per_mainlobe(APERTURE_M, 50.0, BEAM_SPACING_RAD)
    _, _, peak_distance = cfar_window_for_mainlobe(mainlobe)

    assert peak_distance == math.ceil(mainlobe)


def test_window_stays_usable_for_a_sub_beam_mainlobe() -> None:
    """A sub-beam mainlobe must still yield a usable window, not zero cells."""
    guard, training, peak_distance = cfar_window_for_mainlobe(0.2)

    assert guard >= 1
    assert training >= 2
    assert peak_distance >= 1


@pytest.mark.parametrize(
    "kwargs", [{"mainlobe_beams": 0.0}, {"guard_scale": 0.0}, {"train_scale": -1.0}]
)
def test_window_rejects_non_positive_scales(kwargs) -> None:
    """Zero or negative scaling has no physical reading."""
    args = {"mainlobe_beams": 8.0}
    args.update(kwargs)
    with pytest.raises(ValueError):
        cfar_window_for_mainlobe(**args)
