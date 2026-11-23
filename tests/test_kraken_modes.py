"""Tests for the KRAKEN mode files, the Python mode sum and its agreement with FIELD.

The tests that run the Acoustics Toolbox find its programs through ``BLUEPEBBLE_AT_BIN`` or
``PATH``, and are skipped if none of ``krakenc``, ``kraken`` and ``field`` is found.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

BP: SimpleNamespace | None = None


@pytest.fixture(autouse=True, scope="module")
def _isolate_bluepebble_import_state():
    """Import the real package for this module, then restore ``sys.modules``."""
    global BP
    saved_modules = dict(sys.modules)
    bluepebble_module = sys.modules.get("bluepebble")
    saved_package_attrs = set(vars(bluepebble_module)) if bluepebble_module is not None else None

    from bluepebble.models.environment import (
        Constant,
        FlatBathymetry,
        LayeredSeabed,
        SeabedLayer,
        TabulatedSoundSpeedProfile,
    )
    from bluepebble.models.propagation import KrakenModalPropagationModel, kraken

    BP = SimpleNamespace(
        Constant=Constant,
        FlatBathymetry=FlatBathymetry,
        LayeredSeabed=LayeredSeabed,
        SeabedLayer=SeabedLayer,
        TabulatedSoundSpeedProfile=TabulatedSoundSpeedProfile,
        KrakenModalPropagationModel=KrakenModalPropagationModel,
        kraken=kraken,
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


@pytest.fixture(scope="module")
def binaries(_isolate_bluepebble_import_state):
    """Find the toolbox programs, or skip."""
    found = BP.kraken.AcousticsToolboxBinaries.find()
    if found.krakenc is None or found.kraken is None or found.field is None:
        pytest.skip("Acoustics Toolbox programs not found")
    return found


def _pekeris():
    return BP.LayeredSeabed(halfspace=BP.SeabedLayer(None, 1600.0, 1600.0, 1.5, 0.15))


def _layered():
    """Build a SWellEx-like seabed: a graded sediment over a fast mudstone halfspace."""
    return BP.LayeredSeabed(
        halfspace=BP.SeabedLayer(None, 1881.0, 1881.0, 2.06, 0.06),
        layers=(BP.SeabedLayer(23.5, 1572.4, 1593.0, 1.76, 0.2, name="sediment"),),
    )


def _environment(seabed, thorp=True):
    return BP.kraken.KrakenEnvironment(
        ssp_depths_m=(0.0, 30.0, 70.0, 100.0),
        ssp_speeds_m_s=(1520.0, 1505.0, 1490.0, 1492.0),
        water_depth_m=100.0,
        seabed=seabed,
        thorp_attenuation=thorp,
    )


def _ideal_modes(frequencies, depths, orders, depth=100.0, speed=1500.0, loss=0.0):
    """Analytic modes of an isovelocity guide, pressure-release top and rigid bottom.

    ``phi = sqrt(2 / D) sin(gamma z)`` with ``gamma = (n - 1/2) pi / D`` satisfies the
    KRAKEN normalisation ``int phi^2 / rho dz = 1`` with ``rho = 1``.
    """
    frequencies = np.asarray(frequencies, dtype=float)
    depths = np.asarray(depths, dtype=float)
    gammas = (np.asarray(orders, dtype=float) - 0.5) * np.pi / depth
    omega = 2.0 * np.pi * frequencies[:, None]
    k = np.sqrt((omega / speed) ** 2 - gammas[None, :] ** 2) - 1j * loss
    phi = np.sqrt(2.0 / depth) * np.sin(gammas[None, None, :] * depths[:, None, None])
    phi = np.broadcast_to(phi, (depths.size, frequencies.size, gammas.size))
    return BP.kraken.ModeSet(
        frequencies_hz=frequencies,
        k=k,
        num_modes=np.full(frequencies.size, gammas.size),
        phi=phi,
        depths_m=depths,
        bottom_depth_m=depth,
    )


def _reference_sum(modes, source_depth, receiver_depths, ranges):
    """Evaluate the mode sum directly from FIELD's formula, to check the kernel."""
    phis = modes.mode_values(np.array([source_depth]))[0]
    phir = modes.mode_values(receiver_depths)
    terms = (
        phis[None]
        * phir
        * np.exp(-1j * modes.k[None] * ranges[:, None, None])
        / np.sqrt(modes.k[None])
    )
    terms = np.where(modes.mode_mask[None], terms, 0.0)
    return BP.kraken.PREFACTOR * terms.sum(axis=2) / np.sqrt(ranges)[:, None]


class TestModeSum:
    """The numba kernel computes FIELD's sum, with FIELD's interpolation."""

    def test_kernel_matches_formula_between_tabulated_depths(self):
        """Arbitrary depths and ranges, padded mode counts, complex wavenumbers."""
        rng = np.random.default_rng(3)
        frequencies = np.linspace(100.0, 300.0, 9)
        modes = _ideal_modes(frequencies, np.linspace(0.0, 100.0, 51), range(1, 8), loss=2e-5)
        num_modes = modes.num_modes.copy()
        num_modes[::2] = 4
        modes = BP.kraken.ModeSet(
            frequencies_hz=frequencies,
            k=modes.k,
            num_modes=num_modes,
            phi=modes.phi,
            depths_m=modes.depths_m,
        )
        depths = rng.uniform(0.0, 100.0, 13)
        ranges = rng.uniform(100.0, 20000.0, 13)
        got = modes.transfer(37.3, depths, ranges)
        expected = _reference_sum(modes, 37.3, depths, ranges)
        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-12)

    def test_halfspace_tail_follows_field(self):
        """Below the seabed a mode decays as exp(-gamma (z - D)) from its seabed value."""
        frequencies = np.array([200.0])
        base = _ideal_modes(frequencies, np.linspace(0.0, 100.0, 101), [1, 2])
        modes = BP.kraken.ModeSet(
            frequencies_hz=frequencies,
            k=base.k,
            num_modes=base.num_modes,
            phi=base.phi,
            depths_m=base.depths_m,
            bottom_depth_m=100.0,
            bottom_boundary="A",
            bottom_cp=np.array([1700.0 + 5.0j]),
            is_krakenc=True,
        )
        values = modes.mode_values(np.array([100.0, 112.0]))
        k_bottom2 = (2 * np.pi * 200.0 / (1700.0 + 5.0j)) ** 2
        gamma = np.sqrt(base.k[0] ** 2 - k_bottom2)
        np.testing.assert_allclose(values[1, 0], values[0, 0] * np.exp(-gamma * 12.0), rtol=1e-6)

    def test_depth_outside_table_is_rejected(self):
        """A depth the modes were not tabulated over must not be extrapolated silently."""
        modes = _ideal_modes([100.0], np.linspace(20.0, 60.0, 41), [1])
        with pytest.raises(ValueError, match="outside the mode table"):
            modes.transfer(40.0, np.array([80.0]), np.array([1000.0]))

    def test_group_delay_is_positive_and_modal(self):
        """The e^{+i omega t} convention: phase falls with frequency, at about r / v_g."""
        # 0.01 Hz steps keep the phase step below pi, so it unwraps.
        frequencies = np.linspace(304.5, 305.5, 101)
        modes = _ideal_modes(frequencies, np.linspace(0.0, 100.0, 101), [3])
        range_m = 5000.0
        transfer = modes.transfer(50.0, np.array([50.0]), np.array([range_m]))[0]
        delay = -np.gradient(np.unwrap(np.angle(transfer)), 2 * np.pi * frequencies)
        omega = 2 * np.pi * 305.0
        k = np.sqrt((omega / 1500.0) ** 2 - (2.5 * np.pi / 100.0) ** 2)
        group_speed = 1500.0**2 * k / omega
        assert np.median(delay) == pytest.approx(range_m / group_speed, rel=1e-3)

    def test_group_speeds(self):
        """Finite differences recover the analytic group speed."""
        frequencies = np.linspace(500.0, 520.0, 21)
        modes = _ideal_modes(frequencies, np.linspace(0.0, 100.0, 11), [1, 5])
        omega = 2 * np.pi * frequencies[:, None]
        expected = 1500.0**2 * modes.k.real / omega
        np.testing.assert_allclose(modes.group_speeds(), expected, rtol=1e-4)

    def test_cached_transfer_is_bit_identical(self):
        """Caching source excitations and depth weights changes no bit of the result."""
        rng = np.random.default_rng(11)
        frequencies = np.linspace(100.0, 300.0, 21)
        modes = _ideal_modes(frequencies, np.linspace(0.0, 100.0, 101), range(1, 9), loss=1e-5)
        for _ in range(3):
            source = float(rng.uniform(5.0, 95.0))
            depths = rng.uniform(0.0, 100.0, 7)
            ranges = rng.uniform(200.0, 9000.0, 7)
            for _ in range(2):  # the second call is served from the caches
                got = modes.transfer(source, depths, ranges)
                fresh = _ideal_modes(
                    frequencies, np.linspace(0.0, 100.0, 101), range(1, 9), loss=1e-5
                )
                index, weight, tail = fresh.depth_weights(depths)
                expected = np.empty_like(got)
                excitation = fresh.mode_values(np.array([source]))[0] / np.sqrt(fresh.k)
                BP.kraken.modes._mode_sum(
                    fresh.phi,
                    fresh.k,
                    fresh.num_modes,
                    np.ascontiguousarray(excitation),
                    index,
                    weight,
                    tail,
                    np.zeros((1, 1), complex),
                    ranges,
                    expected,
                )
                np.testing.assert_array_equal(got, expected)
            band = modes.transfer(source, depths, ranges, frequency_slice=slice(4, 9))
            np.testing.assert_array_equal(band, got[:, 4:9])

    def test_save_and_load(self, tmp_path):
        """A mode set survives the npz cache unchanged."""
        modes = _ideal_modes([100.0, 101.0], np.linspace(0.0, 100.0, 11), [1, 2, 3], loss=1e-5)
        loaded = BP.kraken.ModeSet.load(modes.save(tmp_path / "modes.npz"))
        np.testing.assert_array_equal(loaded.k, modes.k)
        np.testing.assert_array_equal(loaded.phi, modes.phi)
        assert loaded.is_krakenc == modes.is_krakenc


class TestEnvironmentFile:
    """The env writer produces a broadband, multi-media KRAKEN file."""

    def test_broadband_layered_file(self):
        """Options, media count, layers and the frequency list are where KRAKEN reads them."""
        text = BP.kraken.environment_text(
            _environment(_layered()), [100.0, 150.0], [25.0], np.linspace(0, 100, 5)
        )
        lines = text.splitlines()
        assert lines[2] == "2"
        assert lines[3] == "'CVFT.B'"
        assert "0 0.0 123.5" in lines
        assert lines[-2] == "2"
        assert lines[-1] == "100 150 /"
        assert "0 1881" in lines  # trapped modes only by default

    def test_profile_must_span_water(self):
        """An SSP that stops short of the seabed is an error, not an extrapolation."""
        with pytest.raises(ValueError, match="water depth"):
            BP.kraken.KrakenEnvironment((0.0, 50.0), (1500.0, 1500.0), 100.0, _pekeris())


def _fake_program(path):
    """Write an executable placeholder at ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n")
    path.chmod(0o755)
    return path


class TestFindingPrograms:
    """The programs are found from BLUEPEBBLE_AT_BIN or PATH, and nothing is bundled."""

    def test_found_in_a_listed_directory(self, tmp_path, monkeypatch):
        """A directory in BLUEPEBBLE_AT_BIN that holds the programs is used."""
        for name in ("kraken", "krakenc", "field"):
            _fake_program(tmp_path / f"{name}.exe")
        monkeypatch.setenv(BP.kraken.BINARY_PATH_VARIABLE, str(tmp_path))
        found = BP.kraken.AcousticsToolboxBinaries.find()
        for name in ("kraken", "krakenc", "field"):
            assert getattr(found, name) == (tmp_path / f"{name}.exe").resolve()

    def test_found_under_a_toolbox_root(self, tmp_path, monkeypatch):
        """A toolbox root keeps KRAKEN in Kraken/ and FIELD in KrakenField/."""
        krakenc = _fake_program(tmp_path / "Kraken" / "krakenc.exe")
        field = _fake_program(tmp_path / "KrakenField" / "field.exe")
        monkeypatch.setenv(BP.kraken.BINARY_PATH_VARIABLE, str(tmp_path))
        found = BP.kraken.AcousticsToolboxBinaries.find()
        assert found.krakenc == krakenc.resolve()
        assert found.field == field.resolve()

    def test_missing_program_says_how_to_install(self, monkeypatch):
        """With nothing configured, asking for a program explains where to put it."""
        monkeypatch.delenv(BP.kraken.BINARY_PATH_VARIABLE, raising=False)
        monkeypatch.setenv("PATH", "")
        found = BP.kraken.AcousticsToolboxBinaries.find()
        assert found.krakenc is None
        with pytest.raises(FileNotFoundError, match=BP.kraken.BINARY_PATH_VARIABLE):
            found.require("krakenc")


class TestAgainstField:
    """The mode sum reproduces FIELD's complex pressure from the same mode file."""

    @pytest.mark.parametrize("program", ["krakenc", "kraken"])
    @pytest.mark.parametrize("seabed", ["pekeris", "layered"])
    def test_complex_pressure_matches_field(self, binaries, tmp_path, program, seabed):
        """Three frequencies in one broadband run; receivers in water, sediment and below."""
        environment = _environment(_pekeris() if seabed == "pekeris" else _layered())
        frequencies = [100.0, 175.0, 250.0]
        bottom = environment.bottom_depth_m
        table = np.linspace(0.0, bottom, int(bottom) + 1)
        text = BP.kraken.environment_text(environment, frequencies, [25.0], table, rmax_km=10.0)
        BP.kraken.env.write_text(tmp_path / "case.env", text)
        mode_file = BP.kraken.read_mode_file(
            BP.kraken.run_program(binaries.require(program), tmp_path, "case", ".mod")
        )
        assert mode_file.is_krakenc == (program == "krakenc")
        modes = BP.kraken.ModeSet.from_mode_files([mode_file])

        receivers = np.array([3.0, 25.0, 47.3, 99.0, bottom - 5.0, bottom + 7.5, bottom + 30.0])
        ranges = np.linspace(250.0, 10000.0, 40)
        flp = BP.kraken.field_parameters_text([25.0], receivers, ranges / 1000.0)
        BP.kraken.env.write_text(tmp_path / "case.flp", flp)
        shade = BP.kraken.read_shade_file(
            BP.kraken.run_program(binaries.require("field"), tmp_path, "case", ".shd")
        )
        np.testing.assert_allclose(shade.frequencies_hz, frequencies)
        np.testing.assert_allclose(shade.ranges_m, ranges)

        # FIELD sorts the receiver depths, so evaluate on the shade file's own axes.
        np.testing.assert_allclose(shade.receiver_depths_m, np.sort(receivers), rtol=1e-6)
        field = shade.pressure[:, 0].astype(np.complex128)
        ours = modes.pressure_grid(25.0, shade.receiver_depths_m, shade.ranges_m)
        scale = np.max(np.abs(field), axis=(1, 2), keepdims=True)
        # The mode file and FIELD are single precision, so agreement stops near 1e-6.
        assert np.max(np.abs(ours - field) / scale) < 1e-4
        strong = np.abs(field) > 0.05 * scale
        assert np.max(np.abs(np.angle(ours[strong] / field[strong]))) < 1e-3

    def test_chunked_broadband_equals_one_run(self, binaries, tmp_path):
        """Splitting the grid over parallel broadband runs changes nothing.

        Every chunk shares KRAKEN's reference frequency, so every chunk meshes each
        frequency exactly as a single run over the whole grid would.
        """
        environment = _environment(_layered())
        frequencies = np.array([120.0, 121.5, 123.0, 124.5, 126.0])
        depths = np.linspace(10.0, 90.0, 33)
        common = {"binaries": binaries, "rmax_km": 5.0}
        chunked = BP.kraken.compute_mode_set(
            environment, frequencies, depths, frequencies_per_run=2, **common
        )
        whole = BP.kraken.compute_mode_set(
            environment, frequencies, depths, frequencies_per_run=len(frequencies), **common
        )
        # Inverse iteration starts from the previous frequency's result, so a chunk
        # boundary can move the last bit of a single-precision value.
        np.testing.assert_array_equal(chunked.num_modes, whole.num_modes)
        np.testing.assert_allclose(chunked.k, whole.k, rtol=1e-6)
        np.testing.assert_allclose(chunked.phi, whole.phi, rtol=1e-6, atol=1e-8)

    def test_broadband_run_agrees_with_single_frequency_runs(self, binaries):
        """A broadband run is the same physics as separate runs, to KRAKEN's mesh accuracy.

        Separate runs mesh each frequency differently, so mode shapes differ at the
        finite-difference error, about 0.1 %, while extrapolated wavenumbers agree closely.
        """
        environment = _environment(_layered())
        frequencies = np.array([120.0, 126.0])
        depths = np.linspace(10.0, 90.0, 33)
        broadband = BP.kraken.compute_mode_set(
            environment, frequencies, depths, binaries=binaries, rmax_km=5.0
        )
        for index, frequency in enumerate(frequencies):
            single = BP.kraken.compute_mode_set(
                environment, [frequency], depths, binaries=binaries, rmax_km=5.0
            )
            count = single.num_modes[0]
            assert broadband.num_modes[index] == count
            np.testing.assert_allclose(broadband.k[index, :count], single.k[0, :count], rtol=1e-6)
            scale = np.abs(single.phi[:, 0, :count]).max(axis=0)
            error = np.abs(broadband.phi[:, index, :count] - single.phi[:, 0, :count]).max(axis=0)
            assert np.all(error < 1e-2 * scale)

    @pytest.mark.parametrize("program", ["krakenc", "kraken"])
    def test_transfer_function_is_smooth_across_the_grid(self, binaries, program):
        """No mode is lost or replaced at isolated frequencies, so H has no one-bin glitches.

        Over a lossy layered seabed at 1 kHz there are about 85 modes. KRAKENC with its
        default Richardson extrapolation replaced part of the mode set at about one
        frequency in seven here; on its single fine mesh it is as smooth as KRAKEN.
        """
        environment = BP.kraken.KrakenEnvironment(
            (0.0, 100.0),
            (1500.0, 1500.0),
            100.0,
            BP.LayeredSeabed(
                halfspace=BP.SeabedLayer(None, 1750.0, 1750.0, 1.9, 0.1),
                layers=(BP.SeabedLayer(15.0, 1560.0, 1600.0, 1.6, 0.3),),
            ),
        )
        frequencies = np.arange(1100.0, 1160.0, 0.9765625)
        modes = BP.kraken.compute_mode_set(
            environment,
            frequencies,
            np.linspace(20.0, 60.0, 41),
            program=program,
            binaries=binaries,
            rmax_km=2.5,
        )
        assert modes.mode_count_drops().size == 0
        range_m = 1000.0
        transfer = modes.transfer(30.0, np.array([50.0]), np.array([range_m]))[0]
        transfer = transfer * np.exp(2j * np.pi * frequencies * range_m / 1450.0)
        departure = np.abs(transfer[1:-1] - 0.5 * (transfer[:-2] + transfer[2:]))
        assert np.max(departure) < 0.05 * np.sqrt(np.mean(np.abs(transfer) ** 2))

    def test_cache_reuses_mode_set(self, binaries, tmp_path):
        """A second request for the same modes is read from the cache, not recomputed."""
        environment = _environment(_pekeris())
        args = (environment, [150.0, 151.0], np.linspace(0.0, 100.0, 11))
        first = BP.kraken.compute_mode_set(*args, binaries=binaries, cache_dir=tmp_path)
        cached = list(tmp_path.glob("modes_*.npz"))
        assert len(cached) == 1
        second = BP.kraken.compute_mode_set(
            *args, binaries=BP.kraken.AcousticsToolboxBinaries(), cache_dir=tmp_path
        )
        np.testing.assert_array_equal(first.k, second.k)


class TestPropagationModel:
    """The propagation model wraps the modes as a SpectrumPropagationModel."""

    def _model(self, **kwargs):
        return BP.KrakenModalPropagationModel(
            ssp=BP.TabulatedSoundSpeedProfile(
                depths_m=[0.0, 30.0, 70.0, 100.0], speeds_m_s=[1520.0, 1505.0, 1490.0, 1492.0]
            ),
            bathymetry=BP.FlatBathymetry(depth=-100.0),
            seabed=_pekeris(),
            **kwargs,
        )

    def test_environment_samples_tabulated_profile(self):
        """A tabulated profile is passed to KRAKEN at its own points."""
        environment = self._model().kraken_environment()
        assert environment.ssp_depths_m == (0.0, 30.0, 70.0, 100.0)
        assert environment.ssp_speeds_m_s == (1520.0, 1505.0, 1490.0, 1492.0)

    def test_sloping_bathymetry_is_rejected(self):
        """Range-dependent environments are outside what the model can do."""
        from bluepebble.models.environment import WedgeBathymetry

        with pytest.raises(ValueError, match="flat"):
            BP.KrakenModalPropagationModel(
                ssp=BP.Constant(speed=1500.0),
                bathymetry=WedgeBathymetry(),
                seabed=_pekeris(),
            )

    def test_propagate_spectrum_is_the_mode_sum(self, binaries):
        """Transfer functions equal the mode sum at each sensor's range and depth."""
        model = self._model()
        sensors = np.array([[0.0, 10.0, 20.0], [0.0, 0.0, 0.0], [-40.0, -45.0, -50.0]])
        platform = SimpleNamespace(array=SimpleNamespace(state_vector=sensors))
        source = SimpleNamespace(
            state_vector=np.array([[3000.0], [0.0], [400.0], [0.0], [-30.0], [0.0]]),
            metadata={"position_mapping": [0, 2, 4]},
        )
        frequencies = np.linspace(200.0, 220.0, 5)
        transfer, delays = model.propagate_spectrum(platform, source, frequencies)
        modes = model.modes_for(frequencies, [30.0, 40.0, 50.0])
        ranges = np.hypot(3000.0 - sensors[0], 400.0)
        expected = modes.transfer(30.0, np.array([40.0, 45.0, 50.0]), ranges)
        np.testing.assert_allclose(transfer, expected)
        assert np.all(delays > ranges / 1600.0) and np.all(delays < ranges / 1300.0)
