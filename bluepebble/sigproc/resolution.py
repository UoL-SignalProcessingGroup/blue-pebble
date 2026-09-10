"""Array resolution helpers for sizing bearing-domain processing windows.

Several detector parameters are not free choices but properties of the array: how far
apart two sources must be before the beamformer can separate them, and therefore how
wide a CFAR window has to be before its training cells sample noise rather than the
target sitting in the middle of them. This module derives those from the geometry
instead of leaving them to be guessed.
"""

from __future__ import annotations

import math

__all__ = ["beams_per_mainlobe", "cfar_window_for_mainlobe"]

#: -3 dB mainlobe constant for a uniformly weighted line array.
UNIFORM_SHADING_FACTOR = 0.886


def beams_per_mainlobe(
    aperture_m: float,
    frequency_hz: float,
    beam_spacing_rad: float,
    sound_speed_ms: float = 1500.0,
    steering_rad: float = 0.0,
    shading_factor: float = UNIFORM_SHADING_FACTOR,
) -> float:
    """Width of the -3 dB mainlobe, in beams.

    For a line array the mainlobe subtends roughly ``shading_factor * lambda / L`` radians
    at broadside, broadening as ``1 / cos(steering)`` away from it. Dividing by the beam
    spacing converts that to the only unit the detector works in: bearing bins.

    Use the *lowest* frequency of interest. Mainlobe width scales with wavelength, so the
    low end of the band gives the widest lobe and therefore the conservative window.

    Parameters
    ----------
    aperture_m : float
        End-to-end array length, ``(num_sensors - 1) * sensor_spacing_m``.
    frequency_hz : float
        Frequency at which to evaluate the mainlobe. Must be positive.
    beam_spacing_rad : float
        Angular spacing between adjacent steering azimuths.
    sound_speed_ms : float, optional
        Sound speed, by default 1500.0.
    steering_rad : float, optional
        Steering angle from broadside. The mainlobe broadens as ``1 / cos``, so an
        off-broadside value gives the wider (safer) answer, by default 0.0 (broadside).
    shading_factor : float, optional
        Mainlobe constant for the shading in use, by default 0.886 (uniform weights).
        Tapers widen the mainlobe: a Hann window is roughly 1.5x this.

    Returns
    -------
    float
        Mainlobe width in beams. Not rounded -- callers that need an integer window
        should take the ceiling, which :func:`cfar_window_for_mainlobe` does.

    Raises
    ------
    ValueError
        If any of the geometry arguments is non-positive, or ``steering_rad`` is at or
        beyond endfire, where a line array has no resolution to speak of.

    """
    if aperture_m <= 0:
        raise ValueError(f"aperture_m ({aperture_m}) must be positive")
    if frequency_hz <= 0:
        raise ValueError(f"frequency_hz ({frequency_hz}) must be positive")
    if beam_spacing_rad <= 0:
        raise ValueError(f"beam_spacing_rad ({beam_spacing_rad}) must be positive")
    if sound_speed_ms <= 0:
        raise ValueError(f"sound_speed_ms ({sound_speed_ms}) must be positive")

    # Tested on the angle rather than its cosine: cos(pi/2) evaluates to ~6e-17, a small
    # positive float, so a sign test would let exact endfire through and return a mainlobe
    # of ~1e16 beams.
    if abs(steering_rad) >= math.pi / 2:
        raise ValueError(
            f"steering_rad ({steering_rad}) is at or beyond endfire, where the mainlobe "
            "of a line array is unbounded. Evaluate within +/- 90 degrees of broadside."
        )
    cos_steer = math.cos(steering_rad)

    wavelength_m = sound_speed_ms / frequency_hz
    beamwidth_rad = shading_factor * wavelength_m / (aperture_m * cos_steer)
    return beamwidth_rad / beam_spacing_rad


def cfar_window_for_mainlobe(
    mainlobe_beams: float,
    guard_scale: float = 2.0,
    train_scale: float = 4.0,
) -> tuple[int, int, int]:
    """Size a CFAR window and its peak consolidation from the mainlobe width.

    Guard cells exist to keep the target out of its own noise estimate, so they have to
    reach past the mainlobe: a window whose training cells fall inside the lobe measures
    the target and reports a badly compressed SNR. Peak consolidation has the same
    scale for the same reason -- two candidates closer than a mainlobe cannot be
    resolved as separate sources, whatever the prominence between them says.

    Parameters
    ----------
    mainlobe_beams : float
        Mainlobe width in beams, from :func:`beams_per_mainlobe`.
    guard_scale : float, optional
        Mainlobe widths the guard band spans, counting both sides, by default 2.0 --
        one full mainlobe of guard on each side of the cell under test. Reducing this
        below 1.0 puts training cells inside the lobe.
    train_scale : float, optional
        Training cells per guard cell, by default 4.0. Larger windows give a steadier
        noise estimate at the cost of averaging over more bearing structure.

    Returns
    -------
    tuple[int, int, int]
        ``(num_guard_cells, num_training_cells, peak_distance)``, ready to pass to a
        CFAR detector.

    Raises
    ------
    ValueError
        If ``mainlobe_beams`` is non-positive or either scale is non-positive.

    """
    if mainlobe_beams <= 0:
        raise ValueError(f"mainlobe_beams ({mainlobe_beams}) must be positive")
    if guard_scale <= 0:
        raise ValueError(f"guard_scale ({guard_scale}) must be positive")
    if train_scale <= 0:
        raise ValueError(f"train_scale ({train_scale}) must be positive")

    num_guard_cells = max(1, math.ceil(guard_scale * mainlobe_beams / 2))
    num_training_cells = max(2, math.ceil(train_scale * num_guard_cells))
    peak_distance = max(1, math.ceil(mainlobe_beams))
    return num_guard_cells, num_training_cells, peak_distance
