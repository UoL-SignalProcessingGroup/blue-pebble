"""Tests for the normal-mode channel-filter simulator.

The modes are those of an ideal waveguide (isovelocity, pressure-release surface, rigid
bottom), which are known exactly, so these tests need no Acoustics Toolbox programs.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import signal as scipy_signal
from stonesoup.types.array import StateVector
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState

FS = 4000
FILTER_LENGTH = 4096
BAND = (800.0, 1200.0)
DEPTH = 100.0
SPEED = 1500.0
START = datetime(2020, 6, 1, 12, 0, tzinfo=UTC)
TONE_HZ = 1000.0

BP: SimpleNamespace | None = None


@pytest.fixture(autouse=True, scope="module")
def _isolate_bluepebble_import_state():
    """Import the real package for this module, then restore ``sys.modules``."""
    global BP
    saved_modules = dict(sys.modules)
    bluepebble_module = sys.modules.get("bluepebble")
    saved_package_attrs = set(vars(bluepebble_module)) if bluepebble_module is not None else None
    for name in list(sys.modules):
        if name == "bluepebble.simulator" or name.startswith("bluepebble.simulator."):
            del sys.modules[name]

    from bluepebble.models.environment import Constant, FlatBathymetry, LayeredSeabed, SeabedLayer
    from bluepebble.models.propagation import (
        KrakenModalPropagationModel,
        ModeSet,
        SphericalAcousticPropagationModel,
    )
    from bluepebble.signal import ContinuousWaveSignal, SpectrumLevelSignal
    from bluepebble.simulator import (
        ContinuousChannelFilterPassiveSonarArraySimulator,
        ContinuousModalPassiveSonarArraySimulator,
    )

    BP = SimpleNamespace(
        Constant=Constant,
        FlatBathymetry=FlatBathymetry,
        LayeredSeabed=LayeredSeabed,
        SeabedLayer=SeabedLayer,
        KrakenModalPropagationModel=KrakenModalPropagationModel,
        ModeSet=ModeSet,
        SphericalAcousticPropagationModel=SphericalAcousticPropagationModel,
        ContinuousWaveSignal=ContinuousWaveSignal,
        SpectrumLevelSignal=SpectrumLevelSignal,
        ContinuousChannelFilterPassiveSonarArraySimulator=(
            ContinuousChannelFilterPassiveSonarArraySimulator
        ),
        ContinuousModalPassiveSonarArraySimulator=ContinuousModalPassiveSonarArraySimulator,
    )
    yield
    BP = None
    compiled = {
        name: module
        for name, module in sys.modules.items()
        if name == "rtrs" or name.startswith("rtrs.")
    }
    sys.modules.clear()
    sys.modules.update(saved_modules)
    sys.modules.update(compiled)
    if bluepebble_module is not None:
        package = sys.modules["bluepebble"]
        for attribute in set(vars(package)) - saved_package_attrs:
            delattr(package, attribute)


# --------------------------------------------------------------------------------------
# The ideal waveguide
# --------------------------------------------------------------------------------------
def _channel_frequencies() -> np.ndarray:
    frequencies = np.fft.rfftfreq(FILTER_LENGTH, 1.0 / FS)
    return frequencies[(frequencies >= BAND[0]) & (frequencies <= BAND[1])]


def _gamma(order: int) -> float:
    return (order - 0.5) * np.pi / DEPTH


def _wavenumber(order: int, frequency_hz) -> np.ndarray:
    omega = 2.0 * np.pi * np.asarray(frequency_hz, dtype=float)
    return np.sqrt((omega / SPEED) ** 2 - _gamma(order) ** 2)


def _shape(order: int, depth_m) -> np.ndarray:
    return np.sqrt(2.0 / DEPTH) * np.sin(_gamma(order) * np.asarray(depth_m, dtype=float))


def _mode_set(orders) -> object:
    frequencies = _channel_frequencies()
    depths = np.linspace(0.0, DEPTH, 801)
    k = np.stack([_wavenumber(n, frequencies) for n in orders], axis=1)
    phi = np.stack([_shape(n, depths) for n in orders], axis=1)[:, None, :]
    phi = np.broadcast_to(phi, (depths.size, frequencies.size, len(orders)))
    return BP.ModeSet(
        frequencies_hz=frequencies,
        k=k,
        num_modes=np.full(frequencies.size, len(orders)),
        phi=phi,
        depths_m=depths,
        bottom_depth_m=DEPTH,
    )


def _pressure(orders, frequency_hz, source_depth, receiver_depth, range_m) -> complex:
    """FIELD's mode sum for the ideal guide, evaluated analytically."""
    from bluepebble.models.propagation.kraken import PREFACTOR

    terms = [
        _shape(n, source_depth)
        * _shape(n, receiver_depth)
        * np.exp(-1j * _wavenumber(n, frequency_hz) * range_m)
        / np.sqrt(_wavenumber(n, frequency_hz))
        for n in orders
    ]
    return complex(PREFACTOR * np.sum(terms) / np.sqrt(range_m))


def _model(orders):
    return BP.KrakenModalPropagationModel(
        ssp=BP.Constant(speed=SPEED),
        bathymetry=BP.FlatBathymetry(depth=-DEPTH),
        seabed=BP.LayeredSeabed(halfspace=BP.SeabedLayer(None, 1600.0, 1600.0, 1.5, 0.1)),
        mode_set=_mode_set(orders),
    )


class _StaticArray:
    """Sensors along y at one depth: broadside to a source on the x axis."""

    def __init__(self, num_sensors: int = 1, spacing: float = 1.0, depth: float = 50.0) -> None:
        """Lay the elements out."""
        self.num_sensors = num_sensors
        positions = np.vstack(
            [np.zeros(num_sensors), spacing * np.arange(num_sensors), np.full(num_sensors, depth)]
        )
        self.state = SimpleNamespace(
            array=SimpleNamespace(state_vector=positions, ref_state_vector=positions[:, [0]])
        )

    def get_platform_state_at(self, timestamp: datetime) -> SimpleNamespace:
        """Return the unchanging geometry."""
        return self.state


def _path(position_at, first_s: float, last_s: float, tone=True) -> GroundTruthPath:
    metadata = {"position_mapping": [0, 2, 4], "velocity_mapping": [1, 3, 5]}
    if tone:
        metadata |= {"frequencies_hz": [TONE_HZ], "amplitudes_upa": [1.0], "phases_rad": [0.0]}
    states = []
    for time_s in np.arange(first_s, last_s + 0.5, 0.5):
        x, y, z = position_at(float(time_s))
        states.append(
            GroundTruthState(
                StateVector([x, 0.0, y, 0.0, z, 0.0]),
                timestamp=START + timedelta(seconds=float(time_s)),
                metadata=metadata,
            )
        )
    return GroundTruthPath(states)


def _simulator(orders, position_at, duration_s=20.0, signal="tone", array=None, **kwargs):
    preroll_s = kwargs.pop("source_preroll_s", 6.0)
    if signal == "tone":
        model = BP.ContinuousWaveSignal(duration_s=duration_s + preroll_s, sampling_rate_hz=FS)
    else:
        model = BP.SpectrumLevelSignal(
            duration_s=duration_s + preroll_s, sampling_rate_hz=FS, level_db=0.0, seed=4
        )
    return BP.ContinuousModalPassiveSonarArraySimulator(
        array=array or _StaticArray(),
        propagation_model=kwargs.pop("propagation_model", None) or _model(orders),
        signal_models=[model],
        ground_truth_paths=[_path(position_at, -preroll_s, duration_s + 1.0)],
        start_time=START,
        duration_s=duration_s,
        source_preroll_s=preroll_s,
        frequency_band_hz=BAND,
        band_taper_hz=50.0,
        filter_length=FILTER_LENGTH,
        **kwargs,
    )


def _peak_frequency(samples: np.ndarray) -> float:
    """Frequency of the strongest line, from a heavily zero-padded windowed FFT."""
    window = scipy_signal.windows.blackmanharris(samples.size)
    size = 1 << 23
    # The output is one-sided, so the line is at a positive frequency of the full FFT.
    spectrum = np.abs(np.fft.fft(samples * window, size))[: size // 2]
    peak = int(np.argmax(spectrum))
    a, b, c = np.log(spectrum[peak - 1 : peak + 2])
    offset = 0.5 * (a - c) / (a - 2 * b + c)
    return (peak + offset) * FS / size


# --------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------
class TestLevel:
    """A stationary source is received at the mode sum's level."""

    @pytest.mark.parametrize("range_m", [700.0, 1234.5, 2500.0])
    def test_cw_amplitude_matches_mode_sum(self, range_m):
        """Two interfering modes: the tone's amplitude is |H| at its range, not spreading."""
        orders = [18, 22]
        output = _simulator(orders, lambda t: (range_m, 0.0, 30.0), duration_s=8.0).render()
        received = np.sqrt(2.0) * np.std(output[0][FS:-FS])
        expected = abs(_pressure(orders, TONE_HZ, 30.0, 50.0, range_m))
        assert 20 * np.log10(received / expected) == pytest.approx(0.0, abs=0.05)


class TestDoppler:
    """Each mode is shifted by its own phase speed, not by one sound speed."""

    def test_single_mode_doppler_uses_phase_speed(self):
        """Receding at 10 m/s: the line moves by k_m v / 2 pi with k_m at the received tone.

        For a source moving at ``v`` the received frequency of mode ``m`` solves
        ``omega = omega_0 - k_m(omega) v``. The alternatives that use the group speed or
        the water sound speed differ by more than 0.06 Hz, ten times the tolerance.
        """
        order, speed = 20, 10.0
        simulator = _simulator([order], lambda t: (1500.0 + speed * t, 0.0, 50.0), 40.0)
        output = simulator.render()[0]
        measured = _peak_frequency(output[2 * FS : -2 * FS])

        omega = 2 * np.pi * TONE_HZ
        for _ in range(50):
            omega = 2 * np.pi * TONE_HZ - _wavenumber(order, omega / (2 * np.pi)) * speed
        exact = omega / (2 * np.pi)
        group_speed = SPEED**2 * _wavenumber(order, TONE_HZ) / (2 * np.pi * TONE_HZ)
        assert measured == pytest.approx(exact, abs=0.006)
        assert abs(TONE_HZ * (1 - speed / group_speed) - exact) > 0.06
        assert abs(TONE_HZ * (1 - speed / SPEED) - exact) > 0.06

    def test_modal_fading_matches_the_moving_source_solution(self):
        """The envelope of a receding tone follows the exact moving-source mode sum.

        For a source receding at ``v`` the time integral over emission times turns each
        mode into a line at ``omega_m = omega_0 - k_m(omega_m) v``, with phase
        ``omega_0 t - k_m(omega_m) r(t)`` at the source's instantaneous range ``r(t)``.
        The interference therefore uses each mode's wavenumber at its own Doppler-shifted
        frequency; using ``k_m(omega_0)`` instead would move the pattern by metres.
        """
        from bluepebble.models.propagation.kraken import PREFACTOR

        orders, speed, start_m = [18, 22], 7.0, 1000.0
        simulator = _simulator(orders, lambda t: (start_m + speed * t, 0.0, 30.0), 60.0)
        output = simulator.render()[0]
        # A one-sided tone (a / sqrt(2)) e^{i phi} has the real envelope a.
        envelope = np.sqrt(2.0) * np.abs(output)[::100]
        times = np.arange(output.size)[::100] / FS
        ranges = start_m + speed * times

        expected = np.zeros(times.size, dtype=complex)
        for order in orders:
            omega = 2 * np.pi * TONE_HZ
            for _ in range(50):
                omega = 2 * np.pi * TONE_HZ - _wavenumber(order, omega / (2 * np.pi)) * speed
            k = _wavenumber(order, omega / (2 * np.pi))
            expected += (
                _shape(order, 30.0)
                * _shape(order, 50.0)
                * np.exp(-1j * k * ranges)
                / (np.sqrt(k * ranges))
            )
        expected = np.abs(PREFACTOR * expected)
        keep = (times > 2.0) & (times < 58.0)
        error_db = 20 * np.log10(envelope[keep] / expected[keep])
        assert np.max(np.abs(error_db)) < 0.1
        # The run spans more than one interference length, 2 pi / dk, and a deep fade.
        delta_k = _wavenumber(18, TONE_HZ) - _wavenumber(22, TONE_HZ)
        assert speed * 60.0 > 2 * np.pi / delta_k
        assert 20 * np.log10(expected[keep].max() / expected[keep].min()) > 10.0


class TestGeometry:
    """Knots are planned from the modes and evaluated at retarded positions."""

    def test_retarded_range(self):
        """Receding at v: r(t) = (r0 + v t) / (1 + v / c_ref) at the reception time t."""
        speed = 8.0
        simulator = _simulator([20], lambda t: (1500.0 + speed * t, 0.0, 50.0), 20.0)
        knots = simulator.plan_knots(0)
        c_ref = knots.bulk_speed_m_s
        expected = (1500.0 + speed * knots.knot_times_s) / (1.0 + speed / c_ref)
        np.testing.assert_allclose(knots.ranges_m[:, 0], expected, atol=1e-3)
        instant = _simulator(
            [20], lambda t: (1500.0 + speed * t, 0.0, 50.0), 20.0, retarded_time=False
        ).plan_knots(0)
        np.testing.assert_allclose(
            instant.ranges_m[:, 0], 1500.0 + speed * instant.knot_times_s, atol=1e-9
        )

    def test_knot_interval_bounds_the_phase_step(self):
        """No mode's residual phase moves more than max_phase_step_rad between knots."""
        simulator = _simulator([18, 22], lambda t: (1000.0 + 6.0 * t, 0.0, 30.0), 20.0)
        knots = simulator.plan_knots(0)
        steps = np.abs(np.diff(knots.ranges_m[:, 0])) * knots.max_residual_wavenumber
        assert np.max(steps) <= simulator.max_phase_step_rad * 1.001
        assert np.max(steps) > 0.8 * simulator.max_phase_step_rad

        frequencies = _channel_frequencies()
        omega = 2 * np.pi * frequencies[:, None]
        k = np.stack([_wavenumber(n, frequencies) for n in (18, 22)], axis=1)
        residual = np.abs(k - omega / knots.bulk_speed_m_s).max()
        assert knots.max_residual_wavenumber == pytest.approx(residual, rel=1e-9)

    def test_bulk_speed_needs_fewer_knots_and_still_fits(self):
        """Moving the bulk speed towards the phase speeds cuts the knots, within the filter."""
        motion = lambda t: (1000.0 + 6.0 * t, 0.0, 30.0)  # noqa: E731
        optimised = _simulator([18, 22], motion, 20.0).plan_knots(0)
        centred = _simulator([18, 22], motion, 20.0, optimise_bulk_speed=False).plan_knots(0)
        assert centred.bulk_speed_m_s == centred.dispersion.reference_speed_m_s
        assert optimised.bulk_speed_m_s > centred.bulk_speed_m_s
        assert optimised.max_residual_wavenumber < 0.5 * centred.max_residual_wavenumber
        assert optimised.knot_times_s.size < 0.5 * centred.knot_times_s.size
        # The retarded geometry follows the bulk speed in use.
        times = optimised.knot_times_s
        expected = (1000.0 + 6.0 * times) / (1.0 + 6.0 / optimised.bulk_speed_m_s)
        np.testing.assert_allclose(optimised.ranges_m[:, 0], expected, atol=1e-3)
        # Every arrival still fits in half the filter about the new bulk delay.
        ranges = optimised.ranges_m[:, 0]
        delays = ranges / optimised.bulk_speed_m_s
        late = ranges / optimised.dispersion.min_group_speed_m_s - delays
        early = ranges / optimised.dispersion.max_group_speed_m_s - delays
        assert max(late.max(), -early.min()) + 2.0 / 50.0 <= 0.5 * FILTER_LENGTH / FS

    @pytest.mark.parametrize("optimise", [False, True])
    def test_moving_source_matches_the_exact_solution(self, optimise):
        """A receding broadband source matches the exact moving-source field sample for sample.

        For ``x(t_e) = sum_j cos(w_j (t_e + P) + phi_j)``, ``P`` the preroll, emitted from
        range ``r0 + v t_e``, the field is ``sum_j sum_m A_m e^{i (w_jm t - k_m(w_jm) r0 +
        w_j P + phi_j)} / (1 + v / u_m)``, with ``w_jm = w_j - k_m(w_jm) v`` and ``u_m`` the
        group speed. Nothing is fitted: level, timing, Doppler and phase all have to agree.
        Whichever bulk speed the planner uses, the render matches to better than -45 dB at
        4 m/s.
        """
        from bluepebble.models.propagation.kraken import PREFACTOR

        orders, speed, start_m, preroll_s = [18, 22], 4.0, 900.0, 6.0
        rng = np.random.default_rng(5)
        tones = np.sort(rng.uniform(850.0, 1150.0, 40))
        phases = rng.uniform(0.0, 2 * np.pi, tones.size)
        simulator = _simulator(
            orders,
            lambda t: (start_m + speed * t, 0.0, 30.0),
            20.0,
            source_preroll_s=preroll_s,
            optimise_bulk_speed=optimise,
        )
        for state in simulator.ground_truth_paths[0]:
            state.metadata.update(
                {
                    "frequencies_hz": list(tones),
                    "amplitudes_upa": [1.0] * tones.size,
                    "phases_rad": list(phases),
                }
            )
        output = simulator.render(include_noise=False)[0]
        times = np.arange(output.size) / FS
        expected = np.zeros(times.size, dtype=complex)
        for tone, phase in zip(tones, phases, strict=True):
            for order in orders:
                omega = 2 * np.pi * tone
                for _ in range(60):
                    omega = 2 * np.pi * tone - _wavenumber(order, omega / (2 * np.pi)) * speed
                frequency = omega / (2 * np.pi)
                k = _wavenumber(order, frequency)
                slowness = (
                    _wavenumber(order, frequency + 1e-3) - _wavenumber(order, frequency - 1e-3)
                ) / (2 * np.pi * 2e-3)
                amplitude = PREFACTOR * _shape(order, 30.0) * _shape(order, 50.0)
                amplitude /= np.sqrt(k * (start_m + speed * times)) * (1 + slowness * speed)
                expected += amplitude * np.exp(
                    1j * (omega * times - k * start_m + 2 * np.pi * tone * preroll_s + phase)
                )
        keep = (times > 3.0) & (times < 17.0)
        # The one-sided field of each cos is half of e^{i ...} times sqrt(2).
        exact = expected[keep] / np.sqrt(2.0)
        error = np.linalg.norm(output[keep] - exact) / np.linalg.norm(exact)
        assert 20 * np.log10(error) < -45.0
        knots = simulator.plan_knots(0)
        if optimise:
            assert knots.bulk_speed_m_s > knots.dispersion.reference_speed_m_s

    def test_bulk_speed_changes_nothing_but_the_knots(self):
        """Broadband from a moving source: both bulk speeds render the same signal.

        Each is compared with a dense, centred reference; both agree with it to well
        within the -46 dB cross-fade bound.
        """
        motion = lambda t: (900.0 + 4.0 * t, 30.0 * np.sin(0.1 * t), 30.0)  # noqa: E731
        common = dict(signal="noise", array=_StaticArray(num_sensors=3, spacing=5.0))
        reference = _simulator(
            [18, 22],
            motion,
            10.0,
            optimise_bulk_speed=False,
            max_phase_step_rad=np.pi / 160,
            min_knot_interval_s=1e-3,
            **common,
        ).render(include_noise=False)
        scale = np.sqrt(np.mean(np.abs(reference[:, FS:-FS]) ** 2))
        for optimise in (True, False):
            output = _simulator(
                [18, 22], motion, 10.0, optimise_bulk_speed=optimise, **common
            ).render(include_noise=False)
            error = np.sqrt(np.mean(np.abs(output - reference)[:, FS:-FS] ** 2)) / scale
            assert 20 * np.log10(error) < -40.0

    def test_fixed_knot_interval(self):
        """A fixed interval overrides the plan."""
        simulator = _simulator([20], lambda t: (1500.0, 0.0, 50.0), 10.0, knot_interval_s=0.5)
        np.testing.assert_allclose(np.diff(simulator.plan_knots(0).knot_times_s), 0.5)

    def test_towed_array_history_is_interpolated(self):
        """An array known only at its move times is interpolated between them."""
        positions = [np.array([[2.0 * t], [0.0], [50.0]]) for t in range(3)]
        history = [
            SimpleNamespace(
                timestamp=START + timedelta(seconds=t), array=SimpleNamespace(state_vector=p)
            )
            for t, p in enumerate(positions)
        ]
        array = SimpleNamespace(
            num_sensors=1, platform_history=history, get_platform_state_at=lambda ts: None
        )
        simulator = _simulator([20], lambda t: (1500.0, 0.0, 50.0), 2.0, array=array)
        np.testing.assert_allclose(
            simulator._sensor_positions(np.array([0.5, 1.25]))[:, 0, 0], [1.0, 2.5]
        )


class TestChecks:
    """The simulator refuses runs it cannot render faithfully."""

    def test_dispersion_must_fit_the_filter(self):
        """Mode 1 and mode 40 separate by over a second at 30 km; 4096 taps cannot hold it."""
        simulator = _simulator([1, 40], lambda t: (30000.0, 0.0, 50.0), 4.0, source_preroll_s=25.0)
        with pytest.raises(ValueError, match="filter_length="):
            simulator.render()

    def test_needs_a_modal_model(self):
        """Any other model is rejected, as it has no modes to plan from."""
        simulator = _simulator(
            [20],
            lambda t: (1500.0, 0.0, 50.0),
            4.0,
            propagation_model=BP.SphericalAcousticPropagationModel(ssp=BP.Constant(speed=SPEED)),
        )
        with pytest.raises(TypeError, match="KrakenModalPropagationModel"):
            simulator.render()


class TestStreaming:
    """Streaming the filters is the channel-filter simulator, without the memory."""

    def test_matches_channel_filter_simulator_on_the_same_history(self):
        """Broadband noise, three sensors, a moving source: identical output."""
        simulator = _simulator(
            [18, 22],
            lambda t: (900.0 + 4.0 * t, 30.0 * np.sin(0.1 * t), 30.0),
            10.0,
            signal="noise",
            array=_StaticArray(num_sensors=3, spacing=5.0),
        )
        streamed = simulator.render(include_noise=False)
        history = simulator.compute_channel_history(0)
        reference = BP.ContinuousChannelFilterPassiveSonarArraySimulator(
            array=simulator.array,
            propagation_model=simulator.propagation_model,
            signal_models=simulator.signal_models,
            ground_truth_paths=simulator.ground_truth_paths,
            start_time=START,
            duration_s=simulator.duration_s,
            source_preroll_s=simulator.source_preroll_s,
            frequency_band_hz=BAND,
            band_taper_hz=50.0,
            filter_length=FILTER_LENGTH,
            channel_histories=[history],
            delay_interpolation="cubic",
        ).render(include_noise=False)
        assert np.max(np.abs(streamed - reference)) < 1e-5 * np.max(np.abs(reference))

    def test_source_is_silent_outside_its_path(self):
        """A path that ends mid-run silences the source after it, as in the parent class."""
        simulator = _simulator([20], lambda t: (1500.0, 0.0, 50.0), 10.0)
        states = [s for s in simulator.ground_truth_paths[0] if (s.timestamp - START).seconds < 4]
        simulator.ground_truth_paths[0] = GroundTruthPath(states)
        output = simulator.render()[0]
        assert np.std(output[6 * FS :]) == 0.0
        assert np.std(output[FS : 2 * FS]) > 0.0


class TestWithKraken:
    """End to end with real KRAKENC modes; skipped without the Acoustics Toolbox."""

    def test_stationary_level_matches_the_modal_transfer_function(self, tmp_path):
        """A CW source over a layered seabed is received at |H| from the model itself."""
        from bluepebble.models.propagation import AcousticsToolboxBinaries

        if AcousticsToolboxBinaries.find().krakenc is None:
            pytest.skip("KRAKENC not found")
        model = BP.KrakenModalPropagationModel(
            ssp=BP.Constant(speed=SPEED),
            bathymetry=BP.FlatBathymetry(depth=-DEPTH),
            seabed=BP.LayeredSeabed(
                halfspace=BP.SeabedLayer(None, 1750.0, 1750.0, 1.9, 0.1),
                layers=(BP.SeabedLayer(15.0, 1560.0, 1600.0, 1.6, 0.3),),
            ),
            cache_dir=str(tmp_path),
        )
        array = _StaticArray(num_sensors=4, spacing=3.0, depth=50.0)
        simulator = _simulator(
            None,
            lambda t: (2500.0, 0.0, 30.0),
            6.0,
            array=array,
            propagation_model=model,
        )
        output = simulator.render()
        transfer, _ = model.propagate_spectrum(
            array.state,
            next(iter(simulator.ground_truth_paths[0])),
            np.array([TONE_HZ]),
        )
        received = np.sqrt(2.0) * np.std(output[:, FS:-FS], axis=1)
        level_db = 20 * np.log10(received / np.abs(transfer[:, 0]))
        np.testing.assert_allclose(level_db, 0.0, atol=0.05)
        # The mode set was cached for the run's frequency grid.
        assert list(tmp_path.glob("modes_*.npz"))
