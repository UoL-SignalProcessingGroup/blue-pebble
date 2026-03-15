"""Tests for biological acoustic signal models."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from .support import (
    FakeBase,
    install_fake_stonesoup,
    install_repo_package,
    load_package_module_from_repo,
)

# Sampling rates that satisfy each model's bandpass filter requirements.
FS_SHRIMP = 48_000  # needs > 30 kHz for 2–15 kHz bandpass
FS_WHALE = 8_000  # sufficient for 50–1500 Hz bandpass


def _install_ssp_stub(monkeypatch, speed_mps: float = 1500.0) -> type:
    """Install a fixed-speed SoundSpeedProfile stub; return the class."""
    ssp_class = type(
        "SoundSpeedProfile",
        (FakeBase,),
        {"calculate": lambda self, depth: speed_mps},
    )
    ssp_module = types.ModuleType("bluepebble.models.environment.sound_speed_profile")
    ssp_module.SoundSpeedProfile = ssp_class
    env_module = types.ModuleType("bluepebble.models.environment")
    env_module.sound_speed_profile = ssp_module
    models_module = types.ModuleType("bluepebble.models")
    models_module.environment = env_module

    monkeypatch.setitem(sys.modules, "bluepebble.models", models_module)
    monkeypatch.setitem(sys.modules, "bluepebble.models.environment", env_module)
    monkeypatch.setitem(
        sys.modules, "bluepebble.models.environment.sound_speed_profile", ssp_module
    )
    return ssp_class


def _load_biological(monkeypatch):
    """Load biological module with lightweight stubs; return (module, ssp_class)."""
    install_fake_stonesoup(monkeypatch)
    install_repo_package(monkeypatch, "bluepebble", "bluepebble")
    install_repo_package(monkeypatch, "bluepebble.signal", "bluepebble/signal")
    ssp_class = _install_ssp_stub(monkeypatch)
    load_package_module_from_repo("bluepebble/signal/base.py", "bluepebble.signal.base")
    load_package_module_from_repo("bluepebble/signal/effects.py", "bluepebble.signal.effects")
    bio = load_package_module_from_repo(
        "bluepebble/signal/biological.py", "bluepebble.signal.biological"
    )
    return bio, ssp_class


def _source(amplitude_upa: float = 1.0, position: list | None = None) -> SimpleNamespace:
    """Build a minimal source state for biological signal tests."""
    pos = position if position is not None else [0.0, 0.0, 10.0]
    return SimpleNamespace(
        metadata={"amplitude_upa": amplitude_upa, "position_mapping": list(range(len(pos)))},
        state_vector=np.array([[p] for p in pos]),
    )


def _point_shrimp(bio, temperature_celsius=20.0, duration_s=0.1, **kw):
    return bio.PointSourceSnappingShrimpSignal(
        temperature_celsius=temperature_celsius,
        duration_s=duration_s,
        sampling_rate_hz=FS_SHRIMP,
        **kw,
    )


def _diffuse_shrimp(bio, ssp_class, temperature_celsius=20.0, duration_s=0.1, **kw):
    return bio.DiffuseSnappingShrimpSignal(
        temperature_celsius=temperature_celsius,
        duration_s=duration_s,
        sampling_rate_hz=FS_SHRIMP,
        ssp=ssp_class(),
        num_diffuse_sources=5,  # small for test speed
        **kw,
    )


def _whale(bio, duration_s=10.0, **kw):
    defaults = dict(
        duration_s=duration_s,
        sampling_rate_hz=FS_WHALE,
        start_freq_hz=200.0,
        end_freq_hz=800.0,
        call_duration_s=1.0,
        min_harmonics=1,
        max_harmonics=2,
        low_cutoff_hz=50.0,
        high_cutoff_hz=1500.0,
    )
    defaults.update(kw)
    return bio.WhaleCallSignal(**defaults)


def _rms(x: np.ndarray) -> float:
    """Return root-mean-square amplitude of array x."""
    return float(np.sqrt(np.mean(np.abs(x) ** 2)))


# ── Module-level helpers ──────────────────────────────────────────────────────


def test_snap_rate_from_temp_positive_at_warm(monkeypatch) -> None:
    """Snap rate should be positive at a temperature above the threshold."""
    bio, _ = _load_biological(monkeypatch)
    assert bio._get_snap_rate_from_temp(20.0, 137.0, -685.0) > 0.0


def test_snap_rate_from_temp_zero_at_threshold(monkeypatch) -> None:
    """Snap rate should be zero exactly at the threshold temperature (137*5 - 685 = 0)."""
    # 137*5 - 685 = 0
    bio, _ = _load_biological(monkeypatch)
    assert bio._get_snap_rate_from_temp(5.0, 137.0, -685.0) == pytest.approx(0.0)


def test_snap_rate_from_temp_clamped_below_zero(monkeypatch) -> None:
    """Snap rate must be clamped to 0 below the threshold temperature."""
    bio, _ = _load_biological(monkeypatch)
    assert bio._get_snap_rate_from_temp(0.0, 137.0, -685.0) == 0.0


def test_snap_rate_from_temp_scales_linearly(monkeypatch) -> None:
    """A higher temperature should produce a higher snap rate."""
    bio, _ = _load_biological(monkeypatch)
    r1 = bio._get_snap_rate_from_temp(20.0, 137.0, -685.0)
    r2 = bio._get_snap_rate_from_temp(25.0, 137.0, -685.0)
    assert r2 > r1


def test_get_source_amplitude_reads_metadata(monkeypatch) -> None:
    """_get_source_amplitude_upa should return the amplitude from source metadata."""
    bio, _ = _load_biological(monkeypatch)
    assert bio._get_source_amplitude_upa(_source(amplitude_upa=7.5)) == pytest.approx(7.5)


def test_get_source_amplitude_missing_key_raises(monkeypatch) -> None:
    """Missing amplitude_upa metadata key should raise ValueError."""
    bio, _ = _load_biological(monkeypatch)
    with pytest.raises(ValueError, match="amplitude_upa"):
        bio._get_source_amplitude_upa(SimpleNamespace(metadata={}))


def test_get_source_position_reads_state_vector(monkeypatch) -> None:
    """_get_source_position should extract coordinates from state_vector."""
    bio, _ = _load_biological(monkeypatch)
    pos = bio._get_source_position(_source(position=[1.0, 2.0, 50.0]))
    np.testing.assert_array_almost_equal(pos.flatten(), [1.0, 2.0, 50.0])


def test_get_source_position_missing_mapping_raises(monkeypatch) -> None:
    """Missing position_mapping metadata key should raise ValueError."""
    bio, _ = _load_biological(monkeypatch)
    with pytest.raises(ValueError, match="position_mapping"):
        bio._get_source_position(SimpleNamespace(metadata={"amplitude_upa": 1.0}))


def test_get_source_position_empty_mapping_raises(monkeypatch) -> None:
    """An empty position_mapping should raise ValueError about non-empty requirement."""
    bio, _ = _load_biological(monkeypatch)
    src = SimpleNamespace(
        metadata={"amplitude_upa": 1.0, "position_mapping": []},
        state_vector=np.zeros((3, 1)),
    )
    with pytest.raises(ValueError, match="non-empty"):
        bio._get_source_position(src)


# ── PointSourceSnappingShrimpSignal ──────────────────────────────────────────


def test_snap_template_length(monkeypatch) -> None:
    """Template length = delay + onset + snap sample counts."""
    bio, _ = _load_biological(monkeypatch)
    m = _point_shrimp(bio)
    template = m._create_snap_template()
    expected = (
        int(m.delay_duration * FS_SHRIMP)
        + int(m.onset_duration * FS_SHRIMP)
        + int(m.snap_duration * FS_SHRIMP)
    )
    assert len(template) == expected


def test_snap_template_finite(monkeypatch) -> None:
    """All samples in the snap template should be finite (no NaN/Inf)."""
    bio, _ = _load_biological(monkeypatch)
    assert np.all(np.isfinite(_point_shrimp(bio)._create_snap_template()))


def test_rate_function_constant_without_modulation(monkeypatch) -> None:
    """With diurnal_amplitude=tidal_amplitude=0 the rate equals the base rate exactly."""
    bio, _ = _load_biological(monkeypatch)
    m = _point_shrimp(bio, diurnal_amplitude=0.0, tidal_amplitude=0.0)
    t = np.linspace(0.0, 3600.0, 200)
    np.testing.assert_allclose(m._rate_function(t, 5.0), 5.0)


def test_rate_function_never_negative(monkeypatch) -> None:
    """Rate function must be ≥ 0 even with modulation that would invert it."""
    bio, _ = _load_biological(monkeypatch)
    m = _point_shrimp(bio, diurnal_amplitude=2.0, tidal_amplitude=2.0)
    t = np.linspace(0.0, 24 * 3600.0, 1000)
    assert np.all(m._rate_function(t, 1.0) >= 0.0)


def test_rate_function_varies_with_diurnal_modulation(monkeypatch) -> None:
    """Non-zero diurnal_amplitude should produce time-varying rate values."""
    bio, _ = _load_biological(monkeypatch)
    m = _point_shrimp(bio, diurnal_amplitude=0.5, tidal_amplitude=0.0)
    t = np.linspace(0.0, 24 * 3600.0, 500)
    rates = m._rate_function(t, 10.0)
    assert rates.max() > rates.min()


def test_generate_base_signal_zero_rate_returns_zeros(monkeypatch) -> None:
    """At 5 °C (snap rate = 0) the signal buffer stays all zeros."""
    bio, _ = _load_biological(monkeypatch)
    m = _point_shrimp(bio, temperature_celsius=5.0, duration_s=0.1)
    np.testing.assert_array_equal(m._generate_base_signal(_source()), 0.0)


def test_generate_base_signal_correct_length(monkeypatch) -> None:
    """_generate_base_signal must return exactly num_samples samples."""
    bio, _ = _load_biological(monkeypatch)
    m = _point_shrimp(bio, duration_s=0.1)
    assert len(m._generate_base_signal(_source())) == m.num_samples


def test_generate_base_signal_real_1d(monkeypatch) -> None:
    """_generate_base_signal must return a real-valued 1-D array."""
    bio, _ = _load_biological(monkeypatch)
    result = _point_shrimp(bio, duration_s=0.1)._generate_base_signal(_source())
    assert result.ndim == 1
    assert np.isrealobj(result)


def test_generate_base_signal_nonzero_at_high_rate(monkeypatch) -> None:
    """At 30 °C (~57 snaps/s) a 1 s signal should virtually always be non-zero."""
    bio, _ = _load_biological(monkeypatch)
    m = _point_shrimp(bio, temperature_celsius=30.0, duration_s=1.0)
    assert np.any(m._generate_base_signal(_source()) != 0.0)


def test_point_source_generate_output_shape_and_dtype(monkeypatch) -> None:
    """generate() must return a complex128 array of shape (num_sensors, num_samples)."""
    bio, _ = _load_biological(monkeypatch)
    m = _point_shrimp(bio, duration_s=0.05)
    out = m.generate(_source(), np.zeros(3), 0.0, 0.0)
    assert out.shape == (3, m.num_samples)
    assert out.dtype == np.complex128


def test_point_source_generate_applies_tloss(monkeypatch) -> None:
    """20 dB TL must reduce output amplitude by factor 10 relative to 0 dB."""
    bio, _ = _load_biological(monkeypatch)
    m = _point_shrimp(bio, duration_s=0.1)
    # Pin _generate_base_signal to a fixed signal so both generate() calls are comparable.
    fixed = np.ones(m.num_samples, dtype=np.float64)
    monkeypatch.setattr(m, "_generate_base_signal", lambda src: fixed)

    delays = np.zeros(1)
    out_0db = m.generate(_source(), delays, 0.0, 0.0)
    out_20db = m.generate(_source(), delays, 20.0, 0.0)

    assert _rms(out_0db) / _rms(out_20db) == pytest.approx(10.0, rel=0.01)


def test_point_source_generate_applies_effects(monkeypatch) -> None:
    """Effects in the effects list must be applied to the propagated signal."""
    bio, _ = _load_biological(monkeypatch)

    class ScaleEffect(bio.Effect):
        def apply(self, signals, sampling_rate_hz):
            return signals * 2.0

    m = _point_shrimp(bio, temperature_celsius=30.0, duration_s=0.5, effects=[ScaleEffect()])
    fixed = np.ones(m.num_samples, dtype=np.float64)
    monkeypatch.setattr(m, "_generate_base_signal", lambda src: fixed)

    m_plain = _point_shrimp(bio, temperature_celsius=30.0, duration_s=0.5)
    monkeypatch.setattr(m_plain, "_generate_base_signal", lambda src: fixed)

    delays = np.zeros(1)
    out_eff = m.generate(_source(), delays, 0.0, 0.0)
    out_plain = m_plain.generate(_source(), delays, 0.0, 0.0)
    np.testing.assert_allclose(np.abs(out_eff), np.abs(out_plain) * 2.0, rtol=1e-6)


# ── DiffuseSnappingShrimpSignal ───────────────────────────────────────────────


def test_diffuse_generate_base_signal_correct_length(monkeypatch) -> None:
    """_generate_base_signal must return exactly num_samples samples."""
    bio, ssp_class = _load_biological(monkeypatch)
    m = _diffuse_shrimp(bio, ssp_class)
    assert len(m._generate_base_signal(_source(), 1.0)) == m.num_samples


def test_diffuse_generate_base_signal_zero_at_cold_temp(monkeypatch) -> None:
    """At 5 °C the snap rate is 0 so all samples must be zero."""
    bio, ssp_class = _load_biological(monkeypatch)
    m = _diffuse_shrimp(bio, ssp_class, temperature_celsius=5.0)
    np.testing.assert_array_equal(m._generate_base_signal(_source(), 1.0), 0.0)


def test_diffuse_generate_base_signal_zero_lambda_fraction(monkeypatch) -> None:
    """A lambda_fraction of 0 must produce an all-zero sub-source signal."""
    bio, ssp_class = _load_biological(monkeypatch)
    m = _diffuse_shrimp(bio, ssp_class, temperature_celsius=30.0)
    np.testing.assert_array_equal(m._generate_base_signal(_source(), 0.0), 0.0)


def test_diffuse_generate_output_shape_and_dtype(monkeypatch) -> None:
    """generate() must return a complex128 array of shape (num_sensors, num_samples)."""
    bio, ssp_class = _load_biological(monkeypatch)
    m = _diffuse_shrimp(bio, ssp_class, duration_s=0.05)
    out = m.generate(_source(), np.zeros(4), 0.0, 0.0)
    assert out.shape == (4, m.num_samples)
    assert out.dtype == np.complex128


def test_diffuse_generate_nonzero_at_high_rate(monkeypatch) -> None:
    """At 30 °C the diffuse model should produce at least one non-zero sample."""
    bio, ssp_class = _load_biological(monkeypatch)
    m = _diffuse_shrimp(bio, ssp_class, temperature_celsius=30.0, duration_s=1.0)
    assert np.any(m.generate(_source(), np.zeros(2), 0.0, 0.0) != 0.0)


def test_diffuse_lambda_fraction_reduces_snap_density(monkeypatch) -> None:
    """Halving lambda_fraction should on average produce fewer snaps."""
    bio, ssp_class = _load_biological(monkeypatch)
    m = _diffuse_shrimp(bio, ssp_class, temperature_celsius=30.0, duration_s=2.0)
    src = _source()
    wins = sum(
        np.count_nonzero(m._generate_base_signal(src, 1.0))
        >= np.count_nonzero(m._generate_base_signal(src, 0.5))
        for _ in range(20)
    )
    assert wins >= 12  # strongly expected: full rate ≥ half rate on average


# ── WhaleCallSignal._create_call_template ─────────────────────────────────────


def test_create_call_template_length(monkeypatch) -> None:
    """Template length must equal duration × sampling_rate."""
    bio, _ = _load_biological(monkeypatch)
    m = _whale(bio)
    duration = 0.5
    assert len(m._create_call_template(duration, [200.0, 800.0])) == int(duration * FS_WHALE)


def test_create_call_template_empty_for_zero_duration(monkeypatch) -> None:
    """A zero-duration call should yield an empty template."""
    bio, _ = _load_biological(monkeypatch)
    assert len(_whale(bio)._create_call_template(0.0, [200.0, 800.0])) == 0


def test_create_call_template_finite_values(monkeypatch) -> None:
    """All template samples must be finite (no NaN/Inf)."""
    bio, _ = _load_biological(monkeypatch)
    template = _whale(bio)._create_call_template(0.5, [200.0, 800.0])
    assert np.all(np.isfinite(template))


def test_create_call_template_four_point_contour_uses_cubic(monkeypatch) -> None:
    """Four contour points trigger cubic interpolation; output must still be finite."""
    bio, _ = _load_biological(monkeypatch)
    contour = [200.0, 400.0, 600.0, 800.0]
    template = _whale(bio)._create_call_template(0.5, contour)
    assert len(template) == int(0.5 * FS_WHALE)
    assert np.all(np.isfinite(template))


def test_create_call_template_vibrato_changes_output(monkeypatch) -> None:
    """Enabling vibrato modifies the output relative to no vibrato."""
    bio, _ = _load_biological(monkeypatch)
    m_plain = _whale(bio, vibrato_rate_hz=0.0, vibrato_depth_hz=0.0)
    m_vib = _whale(bio, vibrato_rate_hz=3.0, vibrato_depth_hz=20.0)
    np.random.seed(0)
    t1 = m_plain._create_call_template(0.5, [200.0, 500.0])
    np.random.seed(0)
    t2 = m_vib._create_call_template(0.5, [200.0, 500.0])
    assert not np.allclose(t1, t2)


def test_create_call_template_biphonation_changes_output(monkeypatch) -> None:
    """Biphonation adds a second voice; output must differ from the pure tonal."""
    bio, _ = _load_biological(monkeypatch)
    m_plain = _whale(bio, add_biphonation=False)
    m_bi = _whale(bio, add_biphonation=True, biphonic_amplitude_ratio=0.5)
    np.random.seed(42)
    t1 = m_plain._create_call_template(0.5, [200.0, 500.0])
    np.random.seed(42)
    t2 = m_bi._create_call_template(0.5, [200.0, 500.0])
    assert not np.allclose(t1, t2)


def test_create_call_template_breathy_noise_changes_output(monkeypatch) -> None:
    """Adding breathy noise must change the template relative to the clean tonal."""
    bio, _ = _load_biological(monkeypatch)
    m_plain = _whale(bio, add_breathy_noise=False)
    m_breathy = _whale(bio, add_breathy_noise=True, breathy_noise_amount=0.3)
    np.random.seed(7)
    t1 = m_plain._create_call_template(0.5, [200.0, 500.0])
    np.random.seed(7)
    t2 = m_breathy._create_call_template(0.5, [200.0, 500.0])
    assert not np.allclose(t1, t2)


def test_create_call_template_energy_within_bandpass(monkeypatch) -> None:
    """≥ 80 % of signal energy should lie within the configured bandpass band."""
    bio, _ = _load_biological(monkeypatch)
    low, high = 200.0, 800.0
    m = _whale(bio, low_cutoff_hz=low, high_cutoff_hz=high)
    template = m._create_call_template(1.0, [300.0, 600.0])
    freqs = np.fft.rfftfreq(len(template), d=1.0 / FS_WHALE)
    power = np.abs(np.fft.rfft(template)) ** 2
    in_band = np.sum(power[(freqs >= low) & (freqs <= high)])
    assert in_band / np.sum(power) > 0.80


def test_create_call_template_sub_harmonics_change_output(monkeypatch) -> None:
    """Sub-harmonics add extra energy and should alter the waveform."""
    bio, _ = _load_biological(monkeypatch)
    m_plain = _whale(bio, sub_harmonic_ratios=None)
    m_sub = _whale(bio, sub_harmonic_ratios=[0.5], sub_harmonic_amplitude_ratio=0.5)
    np.random.seed(1)
    t1 = m_plain._create_call_template(0.5, [400.0, 600.0])
    np.random.seed(1)
    t2 = m_sub._create_call_template(0.5, [400.0, 600.0])
    assert not np.allclose(t1, t2)


def test_create_call_template_envelope_taper_reduces_edges(monkeypatch) -> None:
    """Tukey window taper must reduce RMS amplitude at the signal edges."""
    bio, _ = _load_biological(monkeypatch)
    m = _whale(bio, envelope_taper_ratio=0.25)
    template = m._create_call_template(0.5, [200.0, 500.0])
    n = len(template)
    edge_rms = np.sqrt(np.mean(template[: n // 8] ** 2))
    mid_rms = np.sqrt(np.mean(template[n // 4 : 3 * n // 4] ** 2))
    assert edge_rms < mid_rms


# ── WhaleCallSignal call-sequence helpers ─────────────────────────────────────


def test_random_call_sequence_events_have_required_keys(monkeypatch) -> None:
    """Each event dict from _generate_random_call_sequence must have the required keys."""
    bio, _ = _load_biological(monkeypatch)
    m = _whale(bio, duration_s=30.0, mean_call_interval_s=5.0, interval_jitter_s=1.0)
    events = m._generate_random_call_sequence(_source())
    assert len(events) > 0
    for ev in events:
        assert {"start_time", "duration", "contour_freqs", "amplitude"} <= ev.keys()


def test_random_call_sequence_positive_durations(monkeypatch) -> None:
    """All events in a random call sequence must have positive duration."""
    bio, _ = _load_biological(monkeypatch)
    m = _whale(bio, duration_s=30.0, mean_call_interval_s=5.0)
    for ev in m._generate_random_call_sequence(_source()):
        assert ev["duration"] > 0.0


def test_call_sequence_no_overlaps(monkeypatch) -> None:
    """_generate_call_sequence must guarantee non-overlapping events."""
    bio, _ = _load_biological(monkeypatch)
    m = _whale(bio, duration_s=60.0, mean_call_interval_s=3.0, interval_jitter_s=2.0)
    events = m._generate_call_sequence(_source())
    for i in range(1, len(events)):
        prev_end = events[i - 1]["start_time"] + events[i - 1]["duration"]
        assert events[i]["start_time"] >= prev_end - 1e-9


def test_call_sequence_events_within_duration(monkeypatch) -> None:
    """All call events must have start times within the signal duration."""
    bio, _ = _load_biological(monkeypatch)
    duration = 30.0
    m = _whale(bio, duration_s=duration, mean_call_interval_s=5.0)
    for ev in m._generate_call_sequence(_source()):
        assert 0.0 <= ev["start_time"] < duration


def test_structured_song_empty_without_themes(monkeypatch) -> None:
    """Structured song with no themes or phrases should return an empty sequence."""
    bio, _ = _load_biological(monkeypatch)
    m = _whale(bio, song_structure_enabled=True, song_themes=None, song_phrases=None)
    assert m._generate_structured_song_sequence(_source()) == []


def test_structured_song_generates_events(monkeypatch) -> None:
    """A valid structured song configuration should produce at least one event."""
    bio, _ = _load_biological(monkeypatch)
    m = _whale(
        bio,
        duration_s=60.0,
        song_structure_enabled=True,
        song_themes=[[50.0, 100.0], [-50.0, -100.0]],
        song_phrases=[[0, 1, 0]],
        theme_base_freq_hz=200.0,
        theme_duration_s=1.0,
        mean_call_interval_s=2.0,
    )
    events = m._generate_structured_song_sequence(_source())
    assert len(events) > 0


def test_structured_song_invalid_theme_index_skipped(monkeypatch) -> None:
    """Out-of-range theme indices in a phrase should be silently skipped."""
    bio, _ = _load_biological(monkeypatch)
    m = _whale(
        bio,
        duration_s=60.0,
        song_structure_enabled=True,
        song_themes=[[50.0]],
        song_phrases=[[0, 99]],  # 99 is out of range
        theme_base_freq_hz=200.0,
        theme_duration_s=1.0,
        mean_call_interval_s=2.0,
    )
    # Should not raise; invalid index is silently skipped
    events = m._generate_structured_song_sequence(_source())
    assert isinstance(events, list)


# ── WhaleCallSignal end-to-end ────────────────────────────────────────────────


def test_whale_generate_base_signal_length(monkeypatch) -> None:
    """_generate_base_signal must return exactly num_samples samples."""
    bio, _ = _load_biological(monkeypatch)
    m = _whale(bio, duration_s=5.0, mean_call_interval_s=2.0)
    assert len(m._generate_base_signal(_source())) == m.num_samples


def test_whale_generate_output_shape_and_dtype(monkeypatch) -> None:
    """generate() must return a complex128 array of shape (num_sensors, num_samples)."""
    bio, _ = _load_biological(monkeypatch)
    m = _whale(bio, duration_s=5.0, mean_call_interval_s=2.0)
    out = m.generate(_source(), np.zeros(3), 0.0, 0.0)
    assert out.shape == (3, m.num_samples)
    assert out.dtype == np.complex128


def test_whale_generate_nonzero(monkeypatch) -> None:
    """30 s window with 5 s mean interval should produce at least one call."""
    bio, _ = _load_biological(monkeypatch)
    m = _whale(bio, duration_s=30.0, mean_call_interval_s=5.0, interval_jitter_s=1.0)
    assert np.any(m._generate_base_signal(_source(amplitude_upa=1.0)) != 0.0)


def test_whale_generate_applies_tloss(monkeypatch) -> None:
    """20 dB TL must reduce RMS by a factor of 10 relative to 0 dB."""
    bio, _ = _load_biological(monkeypatch)
    m = _whale(bio, duration_s=5.0)
    fixed = np.ones(m.num_samples, dtype=np.float64)
    monkeypatch.setattr(m, "_generate_base_signal", lambda src: fixed)

    delays = np.zeros(1)
    ratio = _rms(m.generate(_source(), delays, 0.0, 0.0)) / _rms(
        m.generate(_source(), delays, 20.0, 0.0)
    )
    assert ratio == pytest.approx(10.0, rel=0.01)
