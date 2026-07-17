"""Tests for bluepebble global seed management."""

import numpy as np
import pytest

import bluepebble
import bluepebble._seed as _seed_module
from bluepebble._seed import _spawn_rng, get_rng, set_seed


@pytest.fixture(autouse=True)
def reset_global_seed():
    """Reset global seed state before and after each test."""
    _seed_module._seed_sequence = None
    _seed_module._global_rng = None
    yield
    _seed_module._seed_sequence = None
    _seed_module._global_rng = None


# ---------------------------------------------------------------------------
# set_seed
# ---------------------------------------------------------------------------


def test_set_seed_populates_seed_sequence():
    """set_seed stores a SeedSequence at module level."""
    set_seed(42)
    assert _seed_module._seed_sequence is not None


def test_set_seed_populates_global_rng():
    """set_seed creates a global Generator."""
    set_seed(42)
    assert _seed_module._global_rng is not None


def test_set_seed_accessible_via_bluepebble():
    """bluepebble.set_seed is the same function as _seed.set_seed."""
    bluepebble.set_seed(42)
    assert _seed_module._seed_sequence is not None


def test_set_seed_resets_state_on_repeated_call():
    """Calling set_seed a second time replaces the previous SeedSequence."""
    set_seed(1)
    seq_first = _seed_module._seed_sequence
    set_seed(1)
    assert _seed_module._seed_sequence is not seq_first


# ---------------------------------------------------------------------------
# get_rng
# ---------------------------------------------------------------------------


def test_get_rng_returns_generator():
    """get_rng returns a numpy Generator after set_seed."""
    set_seed(42)
    assert isinstance(get_rng(), np.random.Generator)


def test_get_rng_returns_same_object_on_repeated_calls():
    """get_rng returns the same Generator object every time."""
    set_seed(42)
    assert get_rng() is get_rng()


def test_get_rng_without_set_seed_returns_generator():
    """get_rng returns a Generator even when set_seed has not been called."""
    assert isinstance(get_rng(), np.random.Generator)


def test_get_rng_accessible_via_bluepebble():
    """bluepebble.get_rng is the same function as _seed.get_rng."""
    set_seed(42)
    assert isinstance(bluepebble.get_rng(), np.random.Generator)


def test_get_rng_reproducible_across_set_seed_calls():
    """The same seed produces the same first draw from get_rng."""
    set_seed(99)
    val_a = get_rng().uniform()

    _seed_module._seed_sequence = None
    _seed_module._global_rng = None

    set_seed(99)
    val_b = get_rng().uniform()

    assert val_a == val_b


# ---------------------------------------------------------------------------
# _spawn_rng
# ---------------------------------------------------------------------------


def test_spawn_rng_explicit_seed_is_deterministic():
    """Two _spawn_rng calls with the same integer seed produce identical first draws."""
    assert _spawn_rng(7).uniform() == _spawn_rng(7).uniform()


def test_spawn_rng_explicit_seed_ignores_global_seed():
    """An explicit seed overrides the global SeedSequence."""
    set_seed(42)
    assert _spawn_rng(7).uniform() == _spawn_rng(7).uniform()


def test_spawn_rng_none_without_global_seed_returns_generator():
    """_spawn_rng(None) returns a Generator when no global seed is set."""
    assert isinstance(_spawn_rng(None), np.random.Generator)


def test_spawn_rng_none_with_global_seed_returns_generator():
    """_spawn_rng(None) returns a Generator when a global seed is set."""
    set_seed(42)
    assert isinstance(_spawn_rng(None), np.random.Generator)


def test_spawn_rng_children_are_independent():
    """Two spawned children produce different sequences."""
    set_seed(42)
    assert _spawn_rng(None).uniform() != _spawn_rng(None).uniform()


def test_spawn_rng_children_are_reproducible():
    """The same global seed produces the same child sequence across runs."""
    set_seed(42)
    val_a = _spawn_rng(None).uniform()

    _seed_module._seed_sequence = None
    _seed_module._global_rng = None

    set_seed(42)
    val_b = _spawn_rng(None).uniform()

    assert val_a == val_b


# ---------------------------------------------------------------------------
# Signal integration
# ---------------------------------------------------------------------------


def test_white_noise_reproducible_with_global_seed():
    """Two simulations with the same global seed produce identical noise."""
    from bluepebble.signal.random import WhiteNoiseSignal  # noqa: PLC0415

    set_seed(42)
    sig_a = WhiteNoiseSignal(amplitude_upa=1.0, duration_s=0.01, sampling_rate_hz=1000)
    out_a = sig_a.generate(num_sensors=1)

    _seed_module._seed_sequence = None
    _seed_module._global_rng = None

    set_seed(42)
    sig_b = WhiteNoiseSignal(amplitude_upa=1.0, duration_s=0.01, sampling_rate_hz=1000)
    out_b = sig_b.generate(num_sensors=1)

    np.testing.assert_array_equal(out_a, out_b)


def test_two_instances_with_global_seed_produce_independent_noise():
    """Two WhiteNoiseSignal instances spawned from the same global seed differ."""
    from bluepebble.signal.random import WhiteNoiseSignal  # noqa: PLC0415

    set_seed(42)
    sig_a = WhiteNoiseSignal(amplitude_upa=1.0, duration_s=0.01, sampling_rate_hz=1000)
    sig_b = WhiteNoiseSignal(amplitude_upa=1.0, duration_s=0.01, sampling_rate_hz=1000)
    assert not np.array_equal(sig_a.generate(num_sensors=1), sig_b.generate(num_sensors=1))


def test_explicit_per_instance_seed_overrides_global():
    """An explicit seed on a signal instance is unaffected by set_seed."""
    from bluepebble.signal.random import WhiteNoiseSignal  # noqa: PLC0415

    set_seed(42)
    sig_a = WhiteNoiseSignal(amplitude_upa=1.0, duration_s=0.01, sampling_rate_hz=1000, seed=7)
    sig_b = WhiteNoiseSignal(amplitude_upa=1.0, duration_s=0.01, sampling_rate_hz=1000, seed=7)
    np.testing.assert_array_equal(sig_a.generate(num_sensors=1), sig_b.generate(num_sensors=1))


def test_no_global_seed_is_nondeterministic():
    """Without set_seed, seed=None gives non-deterministic noise."""
    from bluepebble.signal.random import WhiteNoiseSignal  # noqa: PLC0415

    sig_a = WhiteNoiseSignal(amplitude_upa=1.0, duration_s=0.01, sampling_rate_hz=1000)
    sig_b = WhiteNoiseSignal(amplitude_upa=1.0, duration_s=0.01, sampling_rate_hz=1000)
    assert not np.array_equal(sig_a.generate(num_sensors=1), sig_b.generate(num_sensors=1))


# ---------------------------------------------------------------------------
# ColouredNoiseSignal integration
# ---------------------------------------------------------------------------


def test_coloured_noise_reproducible_with_global_seed():
    """ColouredNoiseSignal output is identical across runs with the same global seed."""
    from bluepebble.signal.random import ColouredNoiseSignal  # noqa: PLC0415

    set_seed(42)
    sig_a = ColouredNoiseSignal(
        spectral_exponent=-1.0, amplitude_upa=1.0, duration_s=0.01, sampling_rate_hz=1000
    )
    out_a = sig_a.generate(num_sensors=1)

    _seed_module._seed_sequence = None
    _seed_module._global_rng = None

    set_seed(42)
    sig_b = ColouredNoiseSignal(
        spectral_exponent=-1.0, amplitude_upa=1.0, duration_s=0.01, sampling_rate_hz=1000
    )
    out_b = sig_b.generate(num_sensors=1)

    np.testing.assert_array_equal(out_a, out_b)


def test_coloured_noise_independent_from_white_noise_with_global_seed():
    """ColouredNoiseSignal and WhiteNoiseSignal spawned from the same global seed differ."""
    from bluepebble.signal.random import ColouredNoiseSignal, WhiteNoiseSignal  # noqa: PLC0415

    set_seed(42)
    white = WhiteNoiseSignal(amplitude_upa=1.0, duration_s=0.01, sampling_rate_hz=1000)
    coloured = ColouredNoiseSignal(
        spectral_exponent=-1.0, amplitude_upa=1.0, duration_s=0.01, sampling_rate_hz=1000
    )
    # The two instances receive independent RNG streams so their raw white-noise
    # bases must differ.
    assert not np.array_equal(white.generate(num_sensors=1), coloured.generate(num_sensors=1))


# ---------------------------------------------------------------------------
# SyntheticAnthropogenicSignal integration
# ---------------------------------------------------------------------------


def test_synthetic_anthropogenic_rng_reproducible_with_global_seed():
    """SyntheticAnthropogenicSignal._rng produces the same first draw across runs."""
    from bluepebble.signal.anthropogenic import SyntheticAnthropogenicSignal  # noqa: PLC0415

    set_seed(42)
    sig_a = SyntheticAnthropogenicSignal(
        duration_s=0.1,
        sampling_rate_hz=500,
        frame_len=50,
        hop_factor=4,
        noise_amplitude_upa=1.0,
    )
    draw_a = sig_a._rng.uniform()

    _seed_module._seed_sequence = None
    _seed_module._global_rng = None

    set_seed(42)
    sig_b = SyntheticAnthropogenicSignal(
        duration_s=0.1,
        sampling_rate_hz=500,
        frame_len=50,
        hop_factor=4,
        noise_amplitude_upa=1.0,
    )
    draw_b = sig_b._rng.uniform()

    assert draw_a == draw_b


def test_synthetic_anthropogenic_independent_from_white_noise():
    """SyntheticAnthropogenicSignal and WhiteNoiseSignal receive independent RNG streams."""
    from bluepebble.signal.anthropogenic import SyntheticAnthropogenicSignal  # noqa: PLC0415
    from bluepebble.signal.random import WhiteNoiseSignal  # noqa: PLC0415

    set_seed(42)
    white = WhiteNoiseSignal(amplitude_upa=1.0, duration_s=0.01, sampling_rate_hz=1000)
    synth = SyntheticAnthropogenicSignal(
        duration_s=0.1,
        sampling_rate_hz=500,
        frame_len=50,
        hop_factor=4,
        noise_amplitude_upa=1.0,
    )
    assert white._rng.uniform() != synth._rng.uniform()


# ---------------------------------------------------------------------------
# Reverb integration
# ---------------------------------------------------------------------------


def test_reverb_first_apply_reproducible_with_global_seed():
    """The first Reverb.apply() call produces the same output across runs with the same seed."""
    from bluepebble.signal.effects import Reverb  # noqa: PLC0415

    dry = (np.ones((2, 100)) + 1j * np.zeros((2, 100))).astype(complex)

    set_seed(42)
    reverb = Reverb(duration_s=0.05, wet_dry_mix=0.5)
    out_a = reverb.apply(dry, sampling_rate_hz=1000)

    _seed_module._seed_sequence = None
    _seed_module._global_rng = None

    set_seed(42)
    reverb = Reverb(duration_s=0.05, wet_dry_mix=0.5)
    out_b = reverb.apply(dry, sampling_rate_hz=1000)

    np.testing.assert_array_equal(out_a, out_b)


def test_reverb_successive_calls_differ_with_global_seed():
    """Successive Reverb.apply() calls spawn different children, giving different IRs."""
    from bluepebble.signal.effects import Reverb  # noqa: PLC0415

    dry = (np.ones((1, 100)) + 1j * np.zeros((1, 100))).astype(complex)

    set_seed(42)
    reverb = Reverb(duration_s=0.05, wet_dry_mix=0.5)
    out_first = reverb.apply(dry, sampling_rate_hz=1000)
    out_second = reverb.apply(dry, sampling_rate_hz=1000)

    assert not np.array_equal(out_first, out_second)


def test_reverb_explicit_seed_unaffected_by_global_seed():
    """An explicit seed on Reverb overrides the global seed and gives identical calls."""
    from bluepebble.signal.effects import Reverb  # noqa: PLC0415

    dry = (np.ones((1, 100)) + 1j * np.zeros((1, 100))).astype(complex)

    set_seed(42)
    reverb = Reverb(duration_s=0.05, wet_dry_mix=0.5, seed=7)
    out_a = reverb.apply(dry, sampling_rate_hz=1000)
    out_b = reverb.apply(dry, sampling_rate_hz=1000)

    # With an explicit seed the IR is recreated identically every call.
    np.testing.assert_array_equal(out_a, out_b)


# ---------------------------------------------------------------------------
# get_rng independence from signal streams
# ---------------------------------------------------------------------------


def test_get_rng_draws_do_not_affect_signal_rng_stream():
    """Draws from get_rng() do not alter the RNG stream seen by signal instances."""
    from bluepebble.signal.random import WhiteNoiseSignal  # noqa: PLC0415

    set_seed(42)
    sig_a = WhiteNoiseSignal(amplitude_upa=1.0, duration_s=0.01, sampling_rate_hz=1000)
    out_a = sig_a.generate(num_sensors=1)

    _seed_module._seed_sequence = None
    _seed_module._global_rng = None

    set_seed(42)
    # Consume several draws from the global rng before constructing the signal.
    get_rng().uniform(size=1000)
    sig_b = WhiteNoiseSignal(amplitude_upa=1.0, duration_s=0.01, sampling_rate_hz=1000)
    out_b = sig_b.generate(num_sensors=1)

    # Signal output must be identical regardless of get_rng() usage, because
    # signal instances spawn independent children from the SeedSequence.
    np.testing.assert_array_equal(out_a, out_b)


# ---------------------------------------------------------------------------
# Mixed scenario
# ---------------------------------------------------------------------------


def test_mixed_signal_types_all_reproducible_from_single_set_seed():
    """A full set_seed() call makes all seeded signal types reproduce identically."""
    from bluepebble.signal.anthropogenic import SyntheticAnthropogenicSignal  # noqa: PLC0415
    from bluepebble.signal.effects import Reverb  # noqa: PLC0415
    from bluepebble.signal.random import ColouredNoiseSignal, WhiteNoiseSignal  # noqa: PLC0415

    dry = (np.ones((1, 50)) + 1j * np.zeros((1, 50))).astype(complex)

    def _run():
        set_seed(99)
        white = WhiteNoiseSignal(amplitude_upa=1.0, duration_s=0.01, sampling_rate_hz=1000)
        coloured = ColouredNoiseSignal(
            spectral_exponent=-1.0, amplitude_upa=1.0, duration_s=0.01, sampling_rate_hz=1000
        )
        synth = SyntheticAnthropogenicSignal(
            duration_s=0.1,
            sampling_rate_hz=500,
            frame_len=50,
            hop_factor=4,
            noise_amplitude_upa=1.0,
        )
        reverb = Reverb(duration_s=0.01, wet_dry_mix=0.3)
        return (
            white.generate(num_sensors=1),
            coloured.generate(num_sensors=1),
            synth._rng.uniform(),
            reverb.apply(dry, sampling_rate_hz=1000),
        )

    _seed_module._seed_sequence = None
    _seed_module._global_rng = None
    run_a = _run()

    _seed_module._seed_sequence = None
    _seed_module._global_rng = None
    run_b = _run()

    np.testing.assert_array_equal(run_a[0], run_b[0])
    np.testing.assert_array_equal(run_a[1], run_b[1])
    assert run_a[2] == run_b[2]
    np.testing.assert_array_equal(run_a[3], run_b[3])
