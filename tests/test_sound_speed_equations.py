"""Tests for the empirical sound speed equations and the Copernicus equation dispatch."""

from __future__ import annotations

import inspect
import warnings

import numpy as np
import pytest

from .support import install_fake_stonesoup, load_module_from_repo

# Names of the equation classes that implement the EquationProfile contract.
EQUATION_CLASS_NAMES = ("Mackenzie", "Coppens", "UNESCO", "DelGrosso", "NPL")

# Analytic profiles with no temperature/salinity dependence, which must not satisfy it.
ANALYTIC_CLASS_NAMES = ("Constant", "Linear", "Arctan", "Munk")


def _load(monkeypatch, suffix: str):
    """Load a fresh copy of the sound speed profile module against a fake Stone Soup."""
    install_fake_stonesoup(monkeypatch)
    return load_module_from_repo(
        "bluepebble/models/environment/sound_speed_profile.py",
        f"bluepebble_ssp_equations_{suffix}",
    )


def _write_copernicus_pair(tmp_path, *, mask_column: bool = True):
    """Write a small synthetic Copernicus temperature/salinity NetCDF pair.

    The fields are shaped like a real Copernicus extract: a ``(time, depth, lat, lon)`` variable
    with a thermocline in depth and, optionally, a masked column standing in for land.
    """
    netCDF4 = pytest.importorskip("netCDF4")

    depth = np.array([0.0, 10.0, 50.0, 150.0, 400.0, 1000.0, 2000.0])
    latitude = np.linspace(60.0, 62.0, 4)
    longitude = np.linspace(-10.0, -6.0, 5)
    n_z, n_y, n_x = len(depth), len(latitude), len(longitude)

    temperature = (
        (12.0 - 9.0 * (1.0 - np.exp(-depth / 400.0)))[:, None, None]
        + 0.3 * np.cos(np.deg2rad(latitude))[None, :, None]
        + 0.1 * np.sin(np.deg2rad(longitude))[None, None, :]
    )
    salinity = (34.8 + 0.4 * (1.0 - np.exp(-depth / 800.0)))[:, None, None] + np.zeros(
        (n_z, n_y, n_x)
    )

    # Real Copernicus extracts encode land as the variable's _FillValue, which netCDF4 masks on
    # read; mirroring that keeps the fixture faithful to the files the loader actually sees.
    fill_value = -32767.0
    if mask_column:
        temperature[-2:, 0, 0] = fill_value
        salinity[-2:, 0, 0] = fill_value

    paths = {}
    for kind, variable, values in (
        ("temperature", "thetao", temperature),
        ("salinity", "so", salinity),
    ):
        path = tmp_path / f"{kind}.nc"
        with netCDF4.Dataset(path, "w") as dataset:
            dataset.createDimension("time", 1)
            dataset.createDimension("depth", n_z)
            dataset.createDimension("latitude", n_y)
            dataset.createDimension("longitude", n_x)
            dataset.createVariable("depth", "f8", ("depth",))[:] = depth
            dataset.createVariable("latitude", "f8", ("latitude",))[:] = latitude
            dataset.createVariable("longitude", "f8", ("longitude",))[:] = longitude
            handle = dataset.createVariable(
                variable, "f4", ("time", "depth", "latitude", "longitude"), fill_value=fill_value
            )
            handle[0, :, :, :] = values
        paths[kind] = str(path)

    return paths["temperature"], paths["salinity"]


# ---------------------------------------------------------------------------
# Published check values
# ---------------------------------------------------------------------------


def test_mackenzie_matches_published_check_value(monkeypatch) -> None:
    """Mackenzie's own check value: c(T=25, S=35, D=1000 m) = 1550.744 m/s."""
    ssp = _load(monkeypatch, "mackenzie_check")

    assert ssp.Mackenzie._equation(25.0, 35.0, 1000.0) == pytest.approx(1550.744, abs=1e-3)


def test_coppens_matches_published_form(monkeypatch) -> None:
    """Coppens at T=25, S=35, D=1 km, recomputed from the published polynomial."""
    ssp = _load(monkeypatch, "coppens_check")

    scaled_temp, depth_km = 2.5, 1.0
    surface = 1449.05 + 45.7 * scaled_temp - 5.21 * scaled_temp**2 + 0.23 * scaled_temp**3
    expected = (
        surface
        + (16.23 + 0.253 * scaled_temp) * depth_km
        + (0.213 - 0.1 * scaled_temp) * depth_km**2
    )

    assert ssp.Coppens._equation(25.0, 35.0, 1000.0) == pytest.approx(expected, abs=1e-9)
    assert ssp.Coppens._equation(25.0, 35.0, 1000.0) == pytest.approx(1551.157, abs=1e-3)


def test_unesco_matches_canonical_seawater_value(monkeypatch) -> None:
    """UNESCO (Chen-Millero) reproduces c(T=0, S=35, P=0) = 1449.14 m/s."""
    ssp = _load(monkeypatch, "unesco_check")

    assert ssp.UNESCO._equation(0.0, 35.0, 0.0) == pytest.approx(1449.14, abs=1e-2)


@pytest.mark.parametrize(
    ("class_name", "expected"),
    [("UNESCO", 1402.388), ("DelGrosso", 1402.392)],
)
def test_pure_water_constant_is_exact(monkeypatch, class_name: str, expected: float) -> None:
    """At T=0, S=0, P=0 each pressure-based equation collapses to its leading constant."""
    ssp = _load(monkeypatch, f"constant_{class_name.lower()}")

    equation_cls = getattr(ssp, class_name)

    assert equation_cls._equation(0.0, 0.0, 0.0) == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("class_name", ["Coppens", "NPL"])
def test_depth_based_equations_agree_at_reference_surface(monkeypatch, class_name: str) -> None:
    """Coppens and NPL both reduce to 1449.05 m/s at T=0, S=35 and zero depth."""
    ssp = _load(monkeypatch, f"surface_{class_name.lower()}")

    equation_cls = getattr(ssp, class_name)
    args = (0.0, 35.0, 0.0, 45.0) if class_name == "NPL" else (0.0, 35.0, 0.0)

    assert equation_cls._equation(*args) == pytest.approx(1449.05, abs=1e-9)


# ---------------------------------------------------------------------------
# Physical consistency across equations
# ---------------------------------------------------------------------------


def test_equations_agree_at_realistic_ocean_conditions(monkeypatch) -> None:
    """All five equations should agree closely where they are all valid."""
    ssp = _load(monkeypatch, "agreement")

    temp, salt, depth = 10.0, 35.0, 1000.0
    pressure = depth / 10.0
    speeds = np.array(
        [
            ssp.Mackenzie._equation(temp, salt, depth),
            ssp.Coppens._equation(temp, salt, depth),
            ssp.UNESCO._equation(temp, salt, pressure),
            ssp.DelGrosso._equation(temp, salt, pressure),
            ssp.NPL._equation(temp, salt, depth, 45.0),
        ],
        dtype=float,
    )

    assert np.all(np.isfinite(speeds))
    assert speeds.max() - speeds.min() < 1.0
    np.testing.assert_allclose(speeds, 1506.3, atol=1.0)


@pytest.mark.parametrize("class_name", EQUATION_CLASS_NAMES)
def test_sound_speed_gradient_is_physical(monkeypatch, class_name: str) -> None:
    """Sound speed should rise with depth at roughly 0.016-0.017 m/s per metre."""
    ssp = _load(monkeypatch, f"gradient_{class_name.lower()}")

    profile = getattr(ssp, class_name)(
        temperature_profile=lambda depth: np.full_like(np.asarray(depth, dtype=float), 10.0),
        salinity_profile=lambda depth: np.full_like(np.asarray(depth, dtype=float), 35.0),
    )

    shallow, deep = profile.calculate(900.0), profile.calculate(1100.0)
    gradient = (deep - shallow) / 200.0

    assert 0.015 < gradient < 0.018


# ---------------------------------------------------------------------------
# calculate() plumbing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("class_name", EQUATION_CLASS_NAMES)
def test_calculate_handles_scalars_arrays_and_negative_depth(monkeypatch, class_name: str) -> None:
    """Scalars return floats, arrays return arrays, and -z is treated as depth below surface."""
    ssp = _load(monkeypatch, f"shapes_{class_name.lower()}")

    profile = getattr(ssp, class_name)()

    scalar = profile.calculate(1000.0)
    assert isinstance(scalar, float)
    assert scalar == pytest.approx(profile.calculate(-1000.0))

    depths = np.array([0.0, 500.0, 1000.0])
    speeds = profile.calculate(depths)
    assert isinstance(speeds, np.ndarray)
    assert speeds.shape == depths.shape
    assert speeds[-1] == pytest.approx(scalar)
    np.testing.assert_allclose(speeds, profile.calculate(-depths))


@pytest.mark.parametrize("class_name", EQUATION_CLASS_NAMES)
def test_supplied_profiles_override_internal_approximations(monkeypatch, class_name: str) -> None:
    """temperature_profile and salinity_profile should displace the internal fallbacks."""
    ssp = _load(monkeypatch, f"override_{class_name.lower()}")
    equation_cls = getattr(ssp, class_name)

    default_profile = equation_cls()
    overridden = equation_cls(
        temperature_profile=lambda depth: np.full_like(np.asarray(depth, dtype=float), 25.0),
        salinity_profile=lambda depth: np.full_like(np.asarray(depth, dtype=float), 39.0),
    )

    assert overridden.calculate(500.0) != pytest.approx(default_profile.calculate(500.0))
    assert overridden._get_temperature(500.0) == pytest.approx(25.0)
    assert overridden._get_salinity(500.0) == pytest.approx(39.0)


def test_mackenzie_reproduces_check_value_through_supplied_profiles(monkeypatch) -> None:
    """Driving Mackenzie with constant T/S profiles reproduces its published check value."""
    ssp = _load(monkeypatch, "override_check")

    profile = ssp.Mackenzie(
        temperature_profile=lambda depth: np.full_like(np.asarray(depth, dtype=float), 25.0),
        salinity_profile=lambda depth: np.full_like(np.asarray(depth, dtype=float), 35.0),
    )

    assert profile.calculate(1000.0) == pytest.approx(1550.744, abs=1e-3)


def test_pressure_falls_back_to_leroy_parthiot_conversion(monkeypatch) -> None:
    """Without a pressure_profile, depth converts via Leroy and Parthiot (1998)."""
    ssp = _load(monkeypatch, "pressure_fallback")

    profile = ssp.UNESCO(pressure_region="standard")

    assert profile._get_pressure(0.0) == pytest.approx(0.0)
    # Published relation at 45 degrees; notably above the depth/10 rule of thumb it replaces.
    assert profile._get_pressure(1000.0) == pytest.approx(101.064, abs=1e-3)
    assert profile._get_pressure(4000.0) == pytest.approx(407.141, abs=1e-3)
    np.testing.assert_allclose(
        profile._get_pressure(np.array([0.0, 1000.0])),
        np.array([0.0, 101.064]),
        atol=1e-3,
    )


def test_pressure_conversion_increases_with_latitude(monkeypatch) -> None:
    """Gravity rises toward the poles, so the same depth sits under more pressure."""
    ssp = _load(monkeypatch, "pressure_latitude")

    at_equator = ssp.UNESCO(latitude=0.0, pressure_region="standard")._get_pressure(1000.0)
    at_45 = ssp.UNESCO(latitude=45.0, pressure_region="standard")._get_pressure(1000.0)
    at_pole = ssp.UNESCO(latitude=90.0, pressure_region="standard")._get_pressure(1000.0)

    assert at_equator < at_45 < at_pole
    # The relation is defined at 45 degrees, so that is where the gravity ratio is exactly 1.
    assert at_45 == pytest.approx(101.064, abs=1e-3)
    # Equator to pole is a small correction, well under 1 per cent.
    assert (at_pole - at_equator) / at_45 < 0.01


def test_pressure_conversion_exceeds_the_depth_over_ten_rule(monkeypatch) -> None:
    """The replaced rule of thumb read low; the gap should grow with depth."""
    ssp = _load(monkeypatch, "pressure_vs_rule_of_thumb")

    profile = ssp.UNESCO(pressure_region="standard")
    gaps = [profile._get_pressure(float(d)) - d / 10.0 for d in (1000, 2000, 4000)]

    assert all(gap > 0 for gap in gaps)
    assert gaps[0] < gaps[1] < gaps[2]
    assert gaps[0] == pytest.approx(1.06, abs=0.05)


def test_supplied_pressure_profile_is_used(monkeypatch) -> None:
    """A supplied pressure_profile should replace the depth-based fallback entirely."""
    ssp = _load(monkeypatch, "pressure_override")

    profile = ssp.UNESCO(
        temperature_profile=lambda depth: np.full_like(np.asarray(depth, dtype=float), 25.0),
        salinity_profile=lambda depth: np.full_like(np.asarray(depth, dtype=float), 35.0),
        pressure_profile=lambda depth: np.full_like(np.asarray(depth, dtype=float), 300.0),
    )

    # Depth is ignored entirely, so any depth yields the speed at 300 bar.
    expected = float(ssp.UNESCO._equation(25.0, 35.0, 300.0))
    assert profile.calculate(10.0) == pytest.approx(expected)
    assert profile.calculate(4000.0) == pytest.approx(expected)


def test_npl_latitude_term_matches_published_coefficient(monkeypatch) -> None:
    """The NPL depth-latitude correction contributes exactly 1.2e-6 * depth * (lat - 45)."""
    ssp = _load(monkeypatch, "npl_latitude")

    depth = 5000.0
    at_equator = ssp.NPL(latitude=0.0).calculate(depth)
    at_pole = ssp.NPL(latitude=90.0).calculate(depth)

    assert ssp.NPL().latitude == pytest.approx(45.0)
    assert at_pole - at_equator == pytest.approx(1.2e-6 * depth * 90.0, abs=1e-9)


# ---------------------------------------------------------------------------
# The EquationProfile contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("class_name", EQUATION_CLASS_NAMES)
def test_required_inputs_match_equation_signature(monkeypatch, class_name: str) -> None:
    """_required_inputs must name exactly the parameters _equation accepts."""
    ssp = _load(monkeypatch, f"contract_{class_name.lower()}")
    equation_cls = getattr(ssp, class_name)

    required = set(equation_cls._required_inputs)
    parameters = set(inspect.signature(equation_cls._equation).parameters)

    assert required == parameters
    assert required <= ssp.EQUATION_INPUTS


@pytest.mark.parametrize("class_name", EQUATION_CLASS_NAMES)
def test_equation_classes_satisfy_protocol(monkeypatch, class_name: str) -> None:
    """Every equation model should satisfy the EquationProfile structural type."""
    ssp = _load(monkeypatch, f"protocol_ok_{class_name.lower()}")

    assert isinstance(getattr(ssp, class_name)(), ssp.EquationProfile)


@pytest.mark.parametrize("class_name", ANALYTIC_CLASS_NAMES)
def test_analytic_profiles_do_not_satisfy_protocol(monkeypatch, class_name: str) -> None:
    """Profiles with no T/S equation must not pass as an equation_cls."""
    ssp = _load(monkeypatch, f"protocol_no_{class_name.lower()}")

    assert not isinstance(getattr(ssp, class_name)(), ssp.EquationProfile)


# ---------------------------------------------------------------------------
# Copernicus equation dispatch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("class_name", EQUATION_CLASS_NAMES)
def test_copernicus_evaluates_each_equation(monkeypatch, tmp_path, class_name: str) -> None:
    """Every equation class should evaluate over the loaded Copernicus grid."""
    ssp = _load(monkeypatch, f"copernicus_{class_name.lower()}")
    temperature_path, salinity_path = _write_copernicus_pair(tmp_path)

    profile = ssp.CopernicusSoundSpeedProfile(
        temperature_file_path=temperature_path,
        salinity_file_path=salinity_path,
        equation_cls=getattr(ssp, class_name),
    )
    profile._ensure_loaded()

    speeds = profile._c_zyx
    assert speeds.shape == (7, 4, 5)
    finite = speeds[np.isfinite(speeds)]
    assert finite.min() > 1400.0
    assert finite.max() < 1600.0


def test_copernicus_preserves_land_mask(monkeypatch, tmp_path) -> None:
    """Masked Copernicus cells should stay NaN rather than being silently filled."""
    ssp = _load(monkeypatch, "copernicus_mask")
    temperature_path, salinity_path = _write_copernicus_pair(tmp_path, mask_column=True)

    profile = ssp.CopernicusSoundSpeedProfile(
        temperature_file_path=temperature_path, salinity_file_path=salinity_path
    )
    profile._ensure_loaded()

    assert np.all(np.isnan(profile._c_zyx[-2:, 0, 0]))
    assert np.isfinite(profile._c_zyx[0, 0, 0])
    # The representative 1D profile must still be gap-free for pointwise queries.
    assert np.all(np.isfinite(profile._c_z_mean))


def test_copernicus_defaults_to_npl_and_matches_legacy_alias(monkeypatch, tmp_path) -> None:
    """The deprecated alias must stay equivalent to an explicit NPL configuration."""
    ssp = _load(monkeypatch, "copernicus_alias")
    temperature_path, salinity_path = _write_copernicus_pair(tmp_path)
    paths = {
        "temperature_file_path": temperature_path,
        "salinity_file_path": salinity_path,
    }

    default_profile = ssp.CopernicusSoundSpeedProfile(**paths)
    with pytest.deprecated_call():
        alias_profile = ssp.LeroyCopernicusSoundSpeedProfile(**paths)
    for profile in (default_profile, alias_profile):
        profile._ensure_loaded()

    assert default_profile.equation_cls is ssp.NPL
    assert alias_profile.equation_cls is ssp.NPL
    np.testing.assert_array_equal(default_profile._c_zyx, alias_profile._c_zyx)


def test_legacy_alias_emits_deprecation_warning(monkeypatch, tmp_path) -> None:
    """Instantiating the alias should point callers at the replacement."""
    ssp = _load(monkeypatch, "copernicus_deprecation")
    temperature_path, salinity_path = _write_copernicus_pair(tmp_path)

    with pytest.warns(DeprecationWarning, match="CopernicusSoundSpeedProfile"):
        ssp.LeroyCopernicusSoundSpeedProfile(
            temperature_file_path=temperature_path, salinity_file_path=salinity_path
        )


def test_replacement_class_does_not_warn(monkeypatch, tmp_path) -> None:
    """The supported class must not emit the deprecation warning of its alias."""
    ssp = _load(monkeypatch, "copernicus_no_deprecation")
    # Built outside the recording window: writing the fixture emits unrelated third-party
    # warnings from netCDF4 that would otherwise be caught here.
    temperature_path, salinity_path = _write_copernicus_pair(tmp_path)

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        ssp.CopernicusSoundSpeedProfile(
            temperature_file_path=temperature_path,
            salinity_file_path=salinity_path,
            equation_cls=ssp.NPL,
        )

    assert not [entry for entry in recorded if "deprecated" in str(entry.message).lower()]


def test_copernicus_honours_supplied_pressure_profile(monkeypatch, tmp_path) -> None:
    """A pressure_profile should feed the gridded evaluation of pressure-based equations."""
    ssp = _load(monkeypatch, "copernicus_pressure")
    temperature_path, salinity_path = _write_copernicus_pair(tmp_path, mask_column=False)
    paths = {
        "temperature_file_path": temperature_path,
        "salinity_file_path": salinity_path,
        "equation_cls": ssp.UNESCO,
    }

    fallback = ssp.CopernicusSoundSpeedProfile(**paths)
    supplied = ssp.CopernicusSoundSpeedProfile(
        **paths, pressure_profile=lambda depth: np.asarray(depth, dtype=float) / 10.0 * 1.05
    )
    for profile in (fallback, supplied):
        profile._ensure_loaded()

    assert not np.allclose(fallback._c_zyx, supplied._c_zyx)


@pytest.mark.parametrize("class_name", ANALYTIC_CLASS_NAMES)
def test_copernicus_rejects_equation_class_without_contract(
    monkeypatch, tmp_path, class_name: str
) -> None:
    """An equation_cls missing the contract should fail with a clear TypeError."""
    ssp = _load(monkeypatch, f"copernicus_reject_{class_name.lower()}")
    temperature_path, salinity_path = _write_copernicus_pair(tmp_path)

    # Loading is eager or lazy depending on whether the base class runs __post_init__, so the
    # error may surface from either construction or the explicit load.
    with pytest.raises(TypeError, match="does not satisfy the EquationProfile contract"):
        ssp.CopernicusSoundSpeedProfile(
            temperature_file_path=temperature_path,
            salinity_file_path=salinity_path,
            equation_cls=getattr(ssp, class_name),
        )._ensure_loaded()


def test_copernicus_rejects_unknown_required_input(monkeypatch, tmp_path) -> None:
    """An equation declaring an unsupported input name should be reported by name."""
    ssp = _load(monkeypatch, "copernicus_unknown_input")
    temperature_path, salinity_path = _write_copernicus_pair(tmp_path)

    class BadEquation:
        _required_inputs = ("temp", "turbidity")

        @staticmethod
        def _equation(temp, turbidity):
            return np.asarray(temp, dtype=float)

    with pytest.raises(TypeError, match="turbidity"):
        ssp.CopernicusSoundSpeedProfile(
            temperature_file_path=temperature_path,
            salinity_file_path=salinity_path,
            equation_cls=BadEquation,
        )._ensure_loaded()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def test_new_equations_are_publicly_exported() -> None:
    """The equation models should be reachable from the environment package API."""
    from bluepebble.models import environment

    for name in (*EQUATION_CLASS_NAMES, "CopernicusSoundSpeedProfile"):
        assert name in environment.__all__
        assert hasattr(environment, name)


# ---------------------------------------------------------------------------
# Range-of-validity warnings
# ---------------------------------------------------------------------------

# Published ranges, per the NPL technical guide. Depth/pressure limits are per equation.
STATED_RANGES = {
    "Mackenzie": {"temp": (2.0, 30.0), "salt": (25.0, 40.0), "depth": (0.0, 8000.0)},
    "Coppens": {"temp": (0.0, 35.0), "salt": (0.0, 45.0), "depth": (0.0, 4000.0)},
    "UNESCO": {"temp": (0.0, 40.0), "salt": (0.0, 40.0), "pressure": (0.0, 1000.0)},
    "DelGrosso": {"temp": (0.0, 30.0), "salt": (30.0, 40.0)},
    "NPL": {"salt": (-np.inf, 42.0)},
}

# Deepest depth each equation is stated to cover, for the no-false-positive check.
STATED_MAX_DEPTH = {
    "Mackenzie": 8000,
    "Coppens": 4000,
    "UNESCO": 1000,
    "DelGrosso": 1000,
    "NPL": 1000,
}


@pytest.mark.parametrize("class_name", EQUATION_CLASS_NAMES)
def test_valid_ranges_match_published_values(monkeypatch, class_name: str) -> None:
    """The encoded ranges must match the published ones they claim to represent."""
    ssp = _load(monkeypatch, f"ranges_{class_name.lower()}")

    encoded = getattr(ssp, class_name)._valid_ranges

    for name, expected in STATED_RANGES[class_name].items():
        assert encoded[name] == pytest.approx(expected), name
    assert set(encoded) <= ssp.EQUATION_INPUTS


def test_del_grosso_pressure_range_converts_from_kgf_per_cm2(monkeypatch) -> None:
    """Del Grosso states 0 to 1000 kg/cm^2; the stored range is that value in bar."""
    ssp = _load(monkeypatch, "ranges_delgrosso_pressure")

    low, high = ssp.DelGrosso._valid_ranges["pressure"]

    assert low == pytest.approx(0.0)
    assert high == pytest.approx(1000.0 / ssp.KGF_PER_CM2_PER_BAR)
    assert high * ssp.KGF_PER_CM2_PER_BAR == pytest.approx(1000.0)


@pytest.mark.parametrize("class_name", EQUATION_CLASS_NAMES)
def test_default_profiles_never_warn_within_stated_depth(monkeypatch, class_name: str) -> None:
    """Out-of-the-box use must be silent, or the warning is pure noise."""
    ssp = _load(monkeypatch, f"nowarn_{class_name.lower()}")
    depths = np.arange(0.0, STATED_MAX_DEPTH[class_name] + 1.0, 1.0)

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        getattr(ssp, class_name)().calculate(depths)

    assert not [
        entry for entry in recorded if issubclass(entry.category, ssp.SoundSpeedRangeWarning)
    ]


def test_range_warning_is_emitted_once_per_call(monkeypatch) -> None:
    """An array evaluation must aggregate into a single warning, not one per element."""
    ssp = _load(monkeypatch, "warn_once")
    depths = np.arange(0.0, 8000.0, 1.0)

    with pytest.warns(ssp.SoundSpeedRangeWarning, match="depth") as recorded:
        ssp.Coppens().calculate(depths)

    assert len(recorded) == 1
    assert "valid 0 to 4000" in str(recorded[0].message)


def test_range_warning_aggregates_multiple_breaches(monkeypatch) -> None:
    """Every breached input should appear in the one warning, with observed extremes."""
    ssp = _load(monkeypatch, "warn_aggregate")

    profile = ssp.Mackenzie(
        temperature_profile=lambda depth: np.full_like(np.asarray(depth, dtype=float), -1.5),
        salinity_profile=lambda depth: np.full_like(np.asarray(depth, dtype=float), 20.0),
    )

    with pytest.warns(ssp.SoundSpeedRangeWarning) as recorded:
        profile.calculate(9000.0)

    assert len(recorded) == 1
    message = str(recorded[0].message)
    for expected in ("temp -1.5", "salt 20", "depth 9000", "valid 2 to 30"):
        assert expected in message


def test_one_sided_range_reads_naturally(monkeypatch) -> None:
    """NPL bounds salinity only from above, so the message must not mention -inf."""
    ssp = _load(monkeypatch, "warn_one_sided")

    profile = ssp.NPL(
        salinity_profile=lambda depth: np.full_like(np.asarray(depth, dtype=float), 45.0)
    )

    with pytest.warns(ssp.SoundSpeedRangeWarning) as recorded:
        profile.calculate(100.0)

    message = str(recorded[0].message)
    assert "valid up to 42" in message
    assert "inf" not in message


def test_npl_has_no_depth_bound(monkeypatch) -> None:
    """NPL states only a salinity limit, so great depth alone must not warn."""
    ssp = _load(monkeypatch, "warn_npl_depth")

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        ssp.NPL().calculate(10000.0)

    assert not [
        entry for entry in recorded if issubclass(entry.category, ssp.SoundSpeedRangeWarning)
    ]


def test_validate_ranges_false_silences_the_check(monkeypatch) -> None:
    """Deliberate extrapolation should be possible without warning noise."""
    ssp = _load(monkeypatch, "warn_optout")

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        speed = ssp.Coppens(validate_ranges=False).calculate(9000.0)

    assert not [
        entry for entry in recorded if issubclass(entry.category, ssp.SoundSpeedRangeWarning)
    ]
    # Silencing the warning must not change the value.
    assert speed == pytest.approx(ssp.Coppens(validate_ranges=False).calculate(9000.0))


def test_range_warning_is_independently_filterable(monkeypatch) -> None:
    """The dedicated category must be suppressible without hiding other warnings."""
    ssp = _load(monkeypatch, "warn_filterable")

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", category=ssp.SoundSpeedRangeWarning)
        ssp.Coppens().calculate(9000.0)
        warnings.warn("unrelated", UserWarning, stacklevel=1)

    assert [str(entry.message) for entry in recorded] == ["unrelated"]


def test_copernicus_grid_warns_once_and_ignores_masked_cells(monkeypatch, tmp_path) -> None:
    """Land cells are NaN and must not be read as out-of-range values."""
    ssp = _load(monkeypatch, "warn_copernicus")
    temperature_path, salinity_path = _write_copernicus_pair(tmp_path, mask_column=True)

    # The fixture's salinity sits near 34.8, inside Del Grosso's range, so force a breach with a
    # supplied pressure profile while leaving the masked column NaN. Construction is inside the
    # block because loading is eager or lazy depending on whether the base runs __post_init__.
    with pytest.warns(ssp.SoundSpeedRangeWarning, match="pressure") as recorded:
        profile = ssp.CopernicusSoundSpeedProfile(
            temperature_file_path=temperature_path,
            salinity_file_path=salinity_path,
            equation_cls=ssp.DelGrosso,
            pressure_profile=lambda depth: np.asarray(depth, dtype=float) / 10.0 + 2000.0,
        )
        profile._ensure_loaded()

    assert len(recorded) == 1
    # The masked column must survive as NaN rather than being counted or filled.
    assert np.all(np.isnan(profile._c_zyx[-2:, 0, 0]))


# ---------------------------------------------------------------------------
# Leroy and Parthiot (1998) regional pressure corrections
# ---------------------------------------------------------------------------


def _range_warnings(ssp, recorded):
    return [entry for entry in recorded if issubclass(entry.category, ssp.SoundSpeedRangeWarning)]


def test_pressure_region_defaults_to_common_oceans(monkeypatch) -> None:
    """The common-ocean correction is the default, as the case covering most of the world."""
    ssp = _load(monkeypatch, "region_default")

    profile = ssp.UNESCO()
    standard = ssp.UNESCO(pressure_region="standard")._get_pressure(1000.0)

    assert profile.pressure_region == "common"
    # Table II row 0 at 1000 m: 1.0e-2 * 1000 / 1100 + 6.2e-6 * 1000 MPa, or 0.15291 bar.
    assert standard - profile._get_pressure(1000.0) == pytest.approx(0.15291, abs=1e-4)


@pytest.mark.parametrize(
    ("region", "depth", "expected_bar"),
    [
        ("common", 1000.0, (1.0e-2 * 1000 / 1100 + 6.2e-6 * 1000) * 10),
        ("north_eastern_atlantic", 3000.0, (8.0e-3 * 3000 / 3200 + 4.0e-6 * 3000) * 10),
        ("mediterranean", 1000.0, (-8.5e-6 * 1000 + 1.4e-9 * 1000**2) * 10),
        ("sea_of_japan", 2000.0, 7.8e-6 * 2000 * 10),
        ("black_sea", 200.0, 1.13e-4 * 200 * 10),
        ("baltic", 266.0, 1.8e-4 * 266 * 10),
    ],
)
def test_pressure_region_subtracts_its_table_ii_term(
    monkeypatch, region: str, depth: float, expected_bar: float
) -> None:
    """Each region subtracts its Table II corrective term from the standard-ocean pressure."""
    ssp = _load(monkeypatch, f"region_{region}")

    standard = ssp.UNESCO(pressure_region="standard")._get_pressure(depth)
    corrected = ssp.UNESCO(pressure_region=region)._get_pressure(depth)

    assert standard - corrected == pytest.approx(expected_bar, abs=1e-8)


def test_mediterranean_correction_raises_pressure(monkeypatch) -> None:
    """Unlike the other areas, the Mediterranean term increases pressure at shallow depth."""
    ssp = _load(monkeypatch, "region_mediterranean_sign")

    standard = ssp.UNESCO(pressure_region="standard")._get_pressure(1000.0)

    assert ssp.UNESCO(pressure_region="mediterranean")._get_pressure(1000.0) > standard
    assert ssp.UNESCO(pressure_region="common")._get_pressure(1000.0) < standard


@pytest.mark.parametrize("region", ["red_sea", "arctic"])
def test_uncorrected_regions_match_the_standard_ocean(monkeypatch, region: str) -> None:
    """Table II lists no corrective term for the Red Sea or the Arctic Ocean."""
    ssp = _load(monkeypatch, f"region_uncorrected_{region}")

    standard = ssp.UNESCO(pressure_region="standard")._get_pressure(3000.0)

    assert ssp.UNESCO(pressure_region=region)._get_pressure(3000.0) == pytest.approx(standard)


def test_every_region_correction_vanishes_at_the_surface(monkeypatch) -> None:
    """Pressure is relative to atmospheric, so every area must give zero at zero depth."""
    ssp = _load(monkeypatch, "region_surface")

    for region in ssp.PRESSURE_REGION_CORRECTIONS:
        pressure = ssp.UNESCO(pressure_region=region)._get_pressure(0.0)
        assert pressure == pytest.approx(0.0, abs=1e-12), region


def test_unknown_pressure_region_is_rejected(monkeypatch) -> None:
    """A misspelt region should fail loudly rather than silently applying no correction."""
    ssp = _load(monkeypatch, "region_unknown")

    with pytest.raises(ValueError, match="Unknown pressure_region 'atlantis'"):
        ssp.UNESCO(pressure_region="atlantis")._get_pressure(100.0)


def test_common_region_warns_outside_its_latitude_band(monkeypatch) -> None:
    """The common oceans are defined only between 60N and 40S."""
    ssp = _load(monkeypatch, "region_common_band_warns")

    with pytest.warns(ssp.SoundSpeedRangeWarning, match="pressure_region='common'"):
        ssp.UNESCO(latitude=70.0)._get_pressure(100.0)


@pytest.mark.parametrize("latitude", [-40.0, 0.0, 45.0, 60.0])
def test_common_region_is_silent_inside_its_latitude_band(monkeypatch, latitude: float) -> None:
    """The band edges are inclusive, and the default latitude sits well inside."""
    ssp = _load(monkeypatch, f"region_common_band_ok_{int(latitude)}")

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        ssp.UNESCO(latitude=latitude)._get_pressure(100.0)

    assert not _range_warnings(ssp, recorded)


def test_named_region_does_not_warn_at_high_latitude(monkeypatch) -> None:
    """Only the common-ocean term is tied to a latitude band."""
    ssp = _load(monkeypatch, "region_arctic_high_latitude")

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        ssp.UNESCO(latitude=80.0, pressure_region="arctic")._get_pressure(100.0)

    assert not _range_warnings(ssp, recorded)


def test_latitude_band_warning_respects_validate_ranges(monkeypatch) -> None:
    """validate_ranges=False silences the latitude check along with the equation ranges."""
    ssp = _load(monkeypatch, "region_band_optout")

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        ssp.UNESCO(latitude=70.0, validate_ranges=False)._get_pressure(100.0)

    assert not _range_warnings(ssp, recorded)
