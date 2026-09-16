"""Defines a collection of sound speed profile (SSP) models.

Choosing between the empirical equations
---------------------------------------
UNESCO (Chen-Millero) and Del Grosso are independent fits to different sets of experiments, and
each carries its own uncertainty. UNESCO is the International Standard, but its accuracy and
range of applicability relative to Del Grosso's equation remain debated, and some researchers
prefer Del Grosso, particularly for calculations inside its own domain of validity. The two
differ by a few tenths of a m/s in the open ocean, growing with pressure.

There is therefore no universally correct choice here: it depends on the accuracy and precision
acceptable for the application. Mackenzie, Coppens and NPL are simplified fits rather than
independent measurement campaigns, so agreement between them and the equations they were fitted
against is partly by construction.

For the debate, see Dushaw et al. (1993), Meinen and Watts (1997), Millero and Xu Li (1994),
Spiesberger and Metzger (1991a, 1991b) and Spiesberger (1993). Pike and Beiboer (1993), for the
Hydrographic Society, summarise the main algorithms with fuller advice on domains of validity
and on depth-to-pressure conversion than is reproduced here.
"""

import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar, Protocol, TypeAlias, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from stonesoup.base import Base, Property

FloatArray: TypeAlias = NDArray[np.float64]
Range1D: TypeAlias = tuple[float, float]
Grid3DResult: TypeAlias = tuple[FloatArray, FloatArray, FloatArray, FloatArray]
DepthInput: TypeAlias = float | ArrayLike
SpeedOutput: TypeAlias = float | FloatArray
ProfileCallable: TypeAlias = Callable[[DepthInput], SpeedOutput]

# Every quantity CopernicusSoundSpeedProfile can supply to an equation, by the keyword name an
# implementation must use in both _required_inputs and the _equation signature.
EQUATION_INPUTS: frozenset[str] = frozenset({"temp", "salt", "depth", "pressure", "latitude"})

# Units of each equation input, used only to make range warnings readable.
EQUATION_INPUT_UNITS: dict[str, str] = {
    "temp": "degrees C",
    "salt": "PSU",
    "depth": "m",
    "pressure": "bar",
    "latitude": "degrees",
}

# 1 bar = 100 kPa = 1.019716 kg/cm^2. Del Grosso's coefficients and its stated pressure range are
# both defined in kg/cm^2, while this module works in bar throughout.
KGF_PER_CM2_PER_BAR: float = 1.019716

# 1 MPa = 10 bar. Leroy and Parthiot's depth-pressure relation is expressed in MPa.
MPA_TO_BAR: float = 10.0

# Corrective terms delta-h_i(Z) from Leroy and Parthiot (1998), Table II: pressure in MPa for depth
# Z in m, subtracted from the standard-ocean pressure h(Z, phi). Comments give the Table II row and
# its stated accuracy. "standard" applies no correction: the oceanographers' standard ocean
# (0 degC, 35 ppt), which Table II also prescribes for the Arctic Ocean and the Red Sea.
PRESSURE_REGION_CORRECTIONS: dict[str, Callable[[FloatArray], FloatArray]] = {
    "standard": lambda z: np.zeros_like(z),
    # Row 0: common oceans, the open oceans between 60N and 40S (+/-8000 Pa). The paper's Eq. (12)
    # misprints the first coefficient as 0.8; the Leroy (2007) erratum confirms Table II's 1.0e-2.
    "common": lambda z: 1.0e-2 * z / (z + 100.0) + 6.2e-6 * z,
    # Row 1: North Eastern Atlantic, 30 to 35N (+/-3000 Pa).
    "north_eastern_atlantic": lambda z: 8.0e-3 * z / (z + 200.0) + 4.0e-6 * z,
    # Row 2: circumpolar Antarctic waters (+/-1000 Pa).
    "circumpolar_antarctic": lambda z: 8.0e-3 * z / (z + 1000.0) + 1.6e-6 * z,
    # Row 3: Mediterranean Sea (+/-2000 Pa). Opposite in sign to the others at shallow depth.
    "mediterranean": lambda z: -8.5e-6 * z + 1.4e-9 * z**2,
    # Rows 4 and 5: Red Sea (+/-2000 Pa) and Arctic Ocean (+/-1000 Pa) need no correction.
    "red_sea": lambda z: np.zeros_like(z),
    "arctic": lambda z: np.zeros_like(z),
    # Row 6(a): Sea of Japan, the more precise of its two options (+/-1000 Pa).
    "sea_of_japan": lambda z: 7.8e-6 * z,
    # Row 7: Sulu Sea (better than +/-1000 Pa).
    "sulu_sea": lambda z: 1.0e-2 * z / (z + 100.0) + 1.6e-5 * z + 1.0e-9 * z**2,
    # Row 8: Halmahera basin (better than +/-1000 Pa).
    "halmahera": lambda z: 8.0e-3 * z / (z + 50.0) + 1.3e-5 * z,
    # Row 9: Celebes basin and Weber deep (+/-2000 Pa).
    "celebes": lambda z: 1.2e-2 * z / (z + 100.0) + 7.0e-6 * z + 2.5e-10 * z**2,
    # Row 10: Black Sea (+/-1000 Pa).
    "black_sea": lambda z: 1.13e-4 * z,
    # Row 11: Baltic Sea (+/-1000 Pa).
    "baltic": lambda z: 1.8e-4 * z,
}

# Latitude band, in degrees with south negative, over which Leroy and Parthiot define the common
# oceans.
COMMON_OCEAN_LATITUDES: tuple[float, float] = (-40.0, 60.0)

ValidRanges: TypeAlias = dict[str, tuple[float, float]]


class SoundSpeedRangeWarning(UserWarning):
    """Warns that an input fell outside an equation's stated range of validity.

    Raised as its own class so it can be filtered independently of unrelated warnings, for
    example with ``warnings.filterwarnings("ignore", category=SoundSpeedRangeWarning)``.
    """


@runtime_checkable
class EquationProfile(Protocol):
    """Structural type for SSP models exposing a pure T/S/P equation over a gridded field.

    ``CopernicusSoundSpeedProfile.equation_cls`` requires this contract: the keyword names listed
    in ``_required_inputs`` (a subset of ``EQUATION_INPUTS``) are passed to ``_equation`` as
    keyword arguments, each already broadcast to the grid shape. Analytic profiles with no
    closed-form dependence on temperature and salinity (``Constant``, ``Linear``, ``Arctan``,
    ``Munk``) deliberately do not satisfy it and cannot be used as an ``equation_cls``.

    ``_valid_ranges`` records the published range of validity as ``{input: (low, high)}`` in the
    units ``_equation`` accepts, and may cover only some inputs (NPL, for instance, constrains
    salinity alone). It documents where the fit was established, not where it is accurate; see
    :meth:`SoundSpeedProfile._check_ranges`.
    """

    _required_inputs: ClassVar[tuple[str, ...]]
    _valid_ranges: ClassVar[ValidRanges]
    _equation: Callable[..., FloatArray]


class SoundSpeedProfile(ABC, Base):
    """Abstract base class for sound speed profile models."""

    temperature_profile: ProfileCallable | None = Property(
        default=None,
        doc="Optional callable mapping positive depth (m) to measure or forecast temperature "
        "(degrees °C). If not provided, falls back to an idealised internal approximation.",
    )
    salinity_profile: ProfileCallable | None = Property(
        default=None,
        doc="Optional callable mapping positive depth (m) to measure or forecast salinity "
        "(practical salinity units). If not provided, falls back to an idealised internal "
        "approximation.",
    )
    latitude: float = Property(
        default=45.0,
        doc="Latitude in degrees. Used by the depth-to-pressure conversion, and by the NPL "
        "equation's depth-latitude term. Defaults to 45.0, at which the gravity scaling and the "
        "NPL term both vanish.",
    )
    pressure_region: str = Property(
        default="common",
        doc="Which Leroy and Parthiot (1998) Table II correction to apply when converting depth "
        "to pressure; a key of PRESSURE_REGION_CORRECTIONS. Defaults to 'common', the open "
        "oceans between 60N and 40S. Use a named sea such as 'baltic' or 'mediterranean' for a "
        "closed basin, or 'standard' for the uncorrected standard ocean.",
    )
    pressure_profile: ProfileCallable | None = Property(
        default=None,
        doc="Optional callable mapping positive depth (m) to measured pressure (bar). If not "
        "provided, falls back to converting depth with Leroy and Parthiot (1998) at `latitude`, "
        "corrected for `pressure_region`.",
    )
    validate_ranges: bool = Property(
        default=True,
        doc="Whether to emit a SoundSpeedRangeWarning when an input falls outside the "
        "equation's published range of validity. Only affects the empirical equation models; "
        "set False to silence the check for deliberate extrapolation.",
    )

    def _check_ranges(self, equation_cls: type[EquationProfile], **values: ArrayLike) -> None:
        """Warn once per call if any input falls outside the equation's published range.

        The check is deliberately coarse: it reports the observed extremes of each input against
        the range over which the equation was fitted, aggregated into a single warning per call
        so that array and grid evaluations cannot produce a warning per element. Non-finite
        values are ignored, so masked Copernicus cells do not count as violations.

        Staying inside these ranges is necessary but not sufficient for accuracy. The equations
        drift apart continuously with depth, by of order 1 m/s at 4000 m, well inside every
        stated range, so absence of a warning is not a guarantee of agreement.

        Parameters
        ----------
        equation_cls : type[EquationProfile]
            Class whose ``_valid_ranges`` are applied. Equations that declare no ranges are
            skipped.
        **values : ArrayLike
            Input values by keyword name, in the units ``_equation`` accepts.

        """
        if not self.validate_ranges:
            return

        ranges = getattr(equation_cls, "_valid_ranges", None)
        if not ranges:
            return

        breaches = []
        for name, value in values.items():
            bounds = ranges.get(name)
            if bounds is None:
                continue
            low, high = bounds
            array = np.asarray(value, dtype=float)
            finite = array[np.isfinite(array)]
            if finite.size == 0:
                continue
            observed_min, observed_max = float(finite.min()), float(finite.max())
            if observed_min < low or observed_max > high:
                unit = EQUATION_INPUT_UNITS.get(name, "")
                # Ranges may be one-sided (NPL bounds salinity only from above).
                if np.isneginf(low):
                    limit = f"valid up to {high:g}"
                elif np.isposinf(high):
                    limit = f"valid from {low:g}"
                else:
                    limit = f"valid {low:g} to {high:g}"
                breaches.append(f"{name} {observed_min:g} to {observed_max:g} {unit} ({limit})")

        if breaches:
            name = getattr(equation_cls, "__name__", repr(equation_cls))
            warnings.warn(
                f"{name} evaluated outside its published range of validity: "
                f"{'; '.join(breaches)}. Results are extrapolated and may be inaccurate; "
                f"set validate_ranges=False to silence this.",
                SoundSpeedRangeWarning,
                stacklevel=3,
            )

    @abstractmethod
    def calculate(self, depth: DepthInput) -> SpeedOutput:
        """Calculate the sound speed at a given depth.

        Parameters
        ----------
        depth : DepthInput
            Depth in meters. Can be positive (oceanographic convention, measured downward from
            surface) or negative (3D coordinate system where surface == 0 and underwater is
            negative z).

        Returns
        -------
        SpeedOutput
            Sound speed in m/s.

        """
        ...

    def get_3d_grid(
        self,
        x_range: Range1D,
        y_range: Range1D,
        z_range: Range1D,
        x_res: float = 5000.0,
        y_res: float = 5000.0,
        z_res: float = 100.0,
    ) -> Grid3DResult:
        """Get a 3D grid representation of the sound speed profile.

        Parameters
        ----------
        x_range : Range1D
            Tuple of (x_min, x_max) in meters.
        y_range : Range1D
            Tuple of (y_min, y_max) in meters.
        z_range : Range1D
            Tuple of (z_min, z_max) in meters (negative depths).
        x_res : float, optional
            Grid resolution in x direction in meters (default 5000.0).
        y_res : float, optional
            Grid resolution in y direction in meters (default 5000.0).
        z_res : float, optional
            Grid resolution in z direction in meters (default 100.0).

        Returns
        -------
        Grid3DResult
            ``(x_grid, y_grid, z_grid, c_grid)`` where
            - ``x_grid`` : 1D array of x coordinates
            - ``y_grid`` : 1D array of y coordinates
            - ``z_grid`` : 1D array of z coordinates (negative depths)
            - ``c_grid`` : 3D array of sound speeds, flattened in C order

        """
        x_min, x_max = x_range
        y_min, y_max = y_range
        z_min, z_max = z_range

        # Create grid points
        x_points = int((x_max - x_min) / x_res) + 1
        y_points = int((y_max - y_min) / y_res) + 1
        z_points = int(abs(z_max - z_min) / z_res) + 1

        x_grid = np.linspace(x_min, x_max, max(2, x_points))
        y_grid = np.linspace(y_min, y_max, max(2, y_points))
        z_grid = np.linspace(z_min, z_max, max(2, z_points))

        # Calculate sound speed at each depth (z values are negative)
        c_at_depths = np.array([self.calculate(z) for z in z_grid])

        # Create 3D grid by tiling the 1D profile across x and y
        # Shape: (nx, ny, nz)
        c_grid_3d = np.tile(c_at_depths, (len(x_grid), len(y_grid), 1))

        # Flatten in C order (row-major) as expected by rtrs
        c_grid_flat = c_grid_3d.flatten(order="C")

        return x_grid, y_grid, z_grid, c_grid_flat

    def _get_temperature(self, depth: DepthInput) -> SpeedOutput:
        """Return temperature at depth, using the supplied profile if given.

        Parameters
        ----------
        depth : DepthInput
            Depth in meters (positive, below surface).

        Returns
        -------
        SpeedOutput
            Temperature in degrees Celsius.

        """
        if self.temperature_profile is not None:
            return self.temperature_profile(depth)
        return self._calc_temperature(depth)

    def _get_salinity(self, depth: DepthInput) -> SpeedOutput:
        """Return salinity at depth, using the supplied profile if given.

        Parameters
        ----------
        depth : DepthInput
            Depth in meters (positive, below surface).

        Returns
        -------
        SpeedOutput
            Salinity in practical salinity units (PSU).

        """
        if self.salinity_profile is not None:
            return self.salinity_profile(depth)
        return self._calc_salinity(depth)

    def _get_pressure(self, depth: DepthInput) -> SpeedOutput:
        """Return pressure at depth, using the supplied profile if given.

        Parameters
        ----------
        depth : DepthInput
            Depth in meters (positive, below surface).

        Returns
        -------
        SpeedOutput
            Pressure in bar.

        """
        if self.pressure_profile is not None:
            return self.pressure_profile(depth)
        return self._calc_pressure(depth)

    def _calc_temperature(self, depth: DepthInput) -> SpeedOutput:
        """Calculate ocean temperature based on vertical variation.

        Parameters
        ----------
        depth : DepthInput
            Depth in meters (positive, below surface).

        Returns
        -------
        SpeedOutput
            Temperature in degrees Celsius.

        """
        depth_array = np.asarray(depth, dtype=float)
        temperature = 10 * (1 - np.tanh((depth_array - 100.0) / 50.0)) + 2.0
        if depth_array.ndim == 0:
            return float(temperature)
        return np.asarray(temperature, dtype=float)

    def _calc_salinity(self, depth: DepthInput) -> SpeedOutput:
        """Calculate ocean salinity model based on vertical variation.

        Parameters
        ----------
        depth : DepthInput
            Depth in meters (positive, below surface).

        Returns
        -------
        SpeedOutput
            Salinity in practical salinity units (PSU).

        """
        depth_array = np.asarray(depth, dtype=float)
        salinity = 0.5 * (1 - np.tanh((depth_array - 200.0) / 100.0)) + 35.0
        if depth_array.ndim == 0:
            return float(salinity)
        return np.asarray(salinity, dtype=float)

    def _calc_pressure(self, depth: DepthInput) -> SpeedOutput:
        """Convert depth to pressure using Leroy and Parthiot (1998).

        Used only when no ``pressure_profile`` is supplied. Evaluates the paper's Eqs. (8) to
        (11), which give pressure in the oceanographers' standard ocean (an ideal medium at 0 degC
        and 35 ppt) at ``latitude``, then subtracts the Table II corrective term for
        ``pressure_region`` to account for the real temperature and salinity structure of that
        area. The default, ``"common"``, covers the open oceans between 60N and 40S; closed
        basins such as the Baltic or Mediterranean need their own term.

        The corrections are small in open ocean, about 0.15 bar at 1000 m for the common oceans,
        but proportionally larger in low-salinity seas: 1.8 per cent of pressure in the Baltic.
        With the appropriate correction the paper states an accuracy of 500 to 8000 Pa, or under
        0.02 m/s in sound speed.

        Eqs. (11) and (12) as printed in the paper contain misprints, both corrected by the
        erratum (Leroy, 2007) and applied here: the gravity constant is 9.7803, not 0.7803, and
        the common-ocean term's first coefficient is Table II's 1.0e-2, not 0.8. The erratum's
        only other correction concerns the pressure-to-depth equations, which are not used.

        Parameters
        ----------
        depth : DepthInput
            Depth in meters (positive, below surface).

        Returns
        -------
        SpeedOutput
            Pressure in bar, relative to atmospheric.

        Raises
        ------
        ValueError
            If ``pressure_region`` is not a key of ``PRESSURE_REGION_CORRECTIONS``.

        References
        ----------
        Leroy, C. C. and Parthiot, F. (1998). Depth-pressure relationships in the oceans and
        seas. Journal of the Acoustical Society of America, 103(3), 1346-1352.

        Leroy, C. C. (2007). Erratum: "Depth-pressure relationships in the oceans and seas".
        Journal of the Acoustical Society of America, 121(4), 2447.

        """
        correction = PRESSURE_REGION_CORRECTIONS.get(self.pressure_region)
        if correction is None:
            raise ValueError(
                f"Unknown pressure_region {self.pressure_region!r}; expected one of "
                f"{sorted(PRESSURE_REGION_CORRECTIONS)}."
            )

        latitude = float(self.latitude)
        low, high = COMMON_OCEAN_LATITUDES
        if self.pressure_region == "common" and self.validate_ranges:
            if not low <= latitude <= high:
                warnings.warn(
                    f"pressure_region='common' is defined for latitudes {low:g} to {high:g}, "
                    f"but latitude is {latitude:g}. Choose the correction for the area in "
                    "question, or 'standard' for none.",
                    SoundSpeedRangeWarning,
                    stacklevel=4,
                )

        depth_array = np.asarray(depth, dtype=float)

        # Eq. (9): standard-ocean pressure in MPa at 45 degrees latitude.
        pressure_45 = (
            1.00818e-2 * depth_array
            + 2.465e-8 * depth_array**2
            - 1.25e-13 * depth_array**3
            + 2.8e-19 * depth_array**4
        )
        # Eqs. (10) and (11): scale by local gravity; the ratio is 1 at 45 degrees.
        gravity = 9.7803 * (1.0 + 5.3e-3 * np.sin(np.deg2rad(latitude)) ** 2)
        gravity_ratio = (gravity - 2e-5 * depth_array) / (9.80612 - 2e-5 * depth_array)
        # Table II: subtract the area's corrective term from the standard-ocean pressure.
        pressure = (pressure_45 * gravity_ratio - correction(depth_array)) * MPA_TO_BAR
        if depth_array.ndim == 0:
            return float(pressure)
        return np.asarray(pressure, dtype=float)


class Constant(SoundSpeedProfile):
    """Constant sound speed profile model.

    This model assumes a uniform sound speed throughout the water column.

    Attributes
    ----------
    speed : float
        Constant sound speed in m/s.

    """

    speed: float = Property(default=1500.0, doc="Constant sound speed in m/s")

    def calculate(self, depth: DepthInput) -> SpeedOutput:
        """Return the constant sound speed.

        Parameters
        ----------
        depth : DepthInput
            Depth in meters (not used in this model).

        Returns
        -------
        SpeedOutput
            Sound speed in m/s.

        """
        depth_array = np.asarray(depth)
        if depth_array.ndim == 0:
            return float(self.speed)
        return np.full(depth_array.shape, self.speed, dtype=float)


class Linear(SoundSpeedProfile):
    """Linear sound speed profile model.

    This model assumes that the sound speed varies linearly with depth.

    Attributes
    ----------
    surface_speed : float
        Sound speed at the surface in m/s.
    gradient : float
        Sound speed gradient in s^-1 (change per meter).

    """

    surface_speed: float = Property(default=1500.0, doc="Sound speed at the surface in m/s")
    gradient: float = Property(
        default=0.017, doc="Sound speed gradient in s^-1 (change per meter)"
    )

    def calculate(self, depth: DepthInput) -> SpeedOutput:
        """Calculate sound speed using a linear profile.

        Parameters
        ----------
        depth : DepthInput
            Depth in meters. If negative (z-coordinate), converts to positive depth below surface
            for calculation.

        Returns
        -------
        SpeedOutput
            Sound speed in m/s.

        """
        # Convert negative z-coordinate to positive depth below surface
        depth_positive = abs(depth)
        c = self.surface_speed + self.gradient * depth_positive
        return c


class Arctan(SoundSpeedProfile):
    """Arctan sound speed profile model.

    This model describes the sound speed profile using an arctangent function,
    which can represent a smooth transition in sound speed with depth.

    Attributes
    ----------
    surface_speed : float
        The speed of sound at the surface in m/s. Defaults to 1500.0 m/s.
    mid_depth : float
        The depth at which the sound speed transition occurs in meters. Defaults to 1000.0 m.
    steepness : float
        The steepness of the transition. Higher values result in a sharper transition.
        Defaults to 0.005.

    """

    surface_speed: float = Property(default=1500.0, doc="Speed of sound at the surface in m/s")
    mid_depth: float = Property(
        default=1000.0, doc="Depth at which sound speed transition occurs in meters"
    )
    steepness: float = Property(default=0.005, doc="Steepness of the transition")

    def calculate(self, depth: DepthInput) -> SpeedOutput:
        """Calculate sound speed using the arctan profile.

        Parameters
        ----------
        depth : DepthInput
            Depth in meters. If negative (z-coordinate), converts to positive depth below surface
            for calculation.

        Returns
        -------
        SpeedOutput
            Sound speed in m/s.

        """
        # Convert negative z-coordinate to positive depth below surface
        depth_positive = abs(depth)
        c = self.surface_speed + 50.0 * np.arctan(
            self.steepness * (depth_positive - self.mid_depth)
        )
        return c


class Munk(SoundSpeedProfile):
    """Munk sound speed profile model.

    This model describes the sound speed profile using an analytical equation proposed by Walter
    Munk. It is characterised by a deep sound channel axis and is widely used in ocean acoustics.

    Attributes
    ----------
    surface_speed : float
        The speed of sound at the surface in m/s. Defaults to 1500.0 m/s.

    """

    surface_speed: float = Property(default=1500.0, doc="Speed of sound at the surface in m/s")

    def calculate(self, depth: DepthInput) -> SpeedOutput:
        """Calculate sound speed using the Munk equation.

        Parameters
        ----------
        depth : DepthInput
            Depth in meters. If negative (z-coordinate), converts to positive depth below surface
            for calculation.

        Returns
        -------
        SpeedOutput
            Sound speed in m/s.

        """
        # Convert negative z-coordinate to positive depth below surface
        depth_positive = abs(depth)

        zt = 2.0 * (depth_positive - 1300.0) / 1300.0
        c = self.surface_speed * (1.0 + 0.00737 * (zt - 1.0 + np.exp(-zt)))
        return c


class Mackenzie(SoundSpeedProfile):
    """Mackenzie sound speed profile model.

    This model calculates the sound speed using the nine-term Mackenzie equation, which is an
    empirical formula based on temperature, salinity, and depth. This implementation uses
    internal models for temperature and salinity as a function of depth unless
    ``temperature_profile`` and ``salinity_profile`` are supplied.

    The equation is valid for temperatures of 2 to 30 degrees Celsius, salinities of 25 to 40
    PSU, and depths of 0 to 8000 m.

    References
    ----------
    Mackenzie, K. V. (1981). Nine-term equation for sound speed in the oceans. Journal of the
    Acoustical Society of America, 70(3), 807-812.

    """

    _required_inputs = ("temp", "salt", "depth")
    _valid_ranges: ValidRanges = {
        "temp": (2.0, 30.0),
        "salt": (25.0, 40.0),
        "depth": (0.0, 8000.0),
    }

    @staticmethod
    def _equation(temp: ArrayLike, salt: ArrayLike, depth: ArrayLike) -> FloatArray:
        """Evaluate the Mackenzie nine-term polynomial.

        Pure function of temperature, salinity, and positive depth, with no data lookup.

        Parameters
        ----------
        temp : ArrayLike
            Temperature in degrees Celsius.
        salt : ArrayLike
            Salinity in PSU.
        depth : ArrayLike
            Positive depth in meters. Any shape broadcastable against temp/salt.

        Returns
        -------
        FloatArray
            Sound speed in m/s, with the broadcast shape of the inputs.

        """
        temp = np.asarray(temp, dtype=float)
        salt = np.asarray(salt, dtype=float)
        depth = np.asarray(depth, dtype=float)
        c = (
            1448.96
            + 4.591 * temp
            - 5.304e-2 * temp**2
            + 2.374e-4 * temp**3
            + 1.340 * (salt - 35)
            + 1.630e-2 * depth
            + 1.675e-7 * depth**2
            - 1.025e-2 * temp * (salt - 35)
            - 7.139e-13 * temp * depth**3
        )
        return c

    def calculate(self, depth: DepthInput) -> SpeedOutput:
        """Calculate sound speed using the Mackenzie nine-term equation.

        Parameters
        ----------
        depth : DepthInput
            Depth in meters. If negative (z-coordinate), converts to positive depth below surface
            for calculation.

        Returns
        -------
        SpeedOutput
            Sound speed in m/s.

        """
        depth_positive = abs(depth)
        temp = self._get_temperature(depth_positive)
        salt = self._get_salinity(depth_positive)
        self._check_ranges(type(self), temp=temp, salt=salt, depth=depth_positive)
        c = self._equation(temp, salt, depth_positive)
        if np.isscalar(depth):
            return float(c)
        return c


class Coppens(SoundSpeedProfile):
    """Coppens sound speed profile model.

    This model calculates the sound speed using the empirical equation of Coppens (1981), a
    compact formula based on temperature, salinity, and depth. The original equation is written
    in terms of ``T/10`` and depth in kilometres; ``_equation`` accepts degrees Celsius and
    metres and rescales internally, so its signature matches the other equations here. This
    implementation uses internal models for temperature and salinity as a function of depth
    unless ``temperature_profile`` and ``salinity_profile`` are supplied.

    The equation is valid for temperatures of 0 to 35 degrees Celsius, salinities of 0 to 45
    PSU, and depths of 0 to 4000 m.

    References
    ----------
    Coppens, A. B. (1981). Simple equations for the speed of sound in Neptunian waters. Journal
    of the Acoustical Society of America, 69(3), 862-863.

    """

    _required_inputs = ("temp", "salt", "depth")
    _valid_ranges: ValidRanges = {"temp": (0.0, 35.0), "salt": (0.0, 45.0), "depth": (0.0, 4000.0)}

    @staticmethod
    def _equation(temp: ArrayLike, salt: ArrayLike, depth: ArrayLike) -> FloatArray:
        """Evaluate the Coppens polynomial.

        Pure function of temperature (degrees C), salinity (PSU), and positive depth (m). The
        rescaling to t = T/10 and depth in km, required by the original equation, is handled
        internally so the public signature stays consistent with the other equations here.

        Parameters
        ----------
        temp : ArrayLike
            Temperature in degrees Celsius.
        salt : ArrayLike
            Salinity in PSU.
        depth : ArrayLike
            Positive depth in meters. Any shape broadcastable against temp/salt.

        Returns
        -------
        FloatArray
            Sound speed in m/s, with the broadcast shape of the inputs.

        """
        temp = np.asarray(temp, dtype=float)
        salt = np.asarray(salt, dtype=float)
        depth = np.asarray(depth, dtype=float)

        depth_km = depth / 1000.0
        t = temp / 10.0

        c0 = (
            1449.05
            + 45.7 * t
            - 5.21 * t**2
            + 0.23 * t**3
            + (1.333 - 0.126 * t + 0.009 * t**2) * (salt - 35)
        )
        c = (
            c0
            + (16.23 + 0.253 * t) * depth_km
            + (0.213 - 0.1 * t) * depth_km**2
            + (0.016 + 0.0002 * (salt - 35)) * (salt - 35) * t * depth_km
        )
        return c

    def calculate(self, depth: DepthInput) -> SpeedOutput:
        """Calculate sound speed using the Coppens equation.

        Parameters
        ----------
        depth : DepthInput
            Depth in meters. If negative (z-coordinate), converts to positive depth below surface
            for calculation.

        Returns
        -------
        SpeedOutput
            Sound speed in m/s.

        """
        depth_positive = abs(depth)
        temp = self._get_temperature(depth_positive)
        salt = self._get_salinity(depth_positive)
        self._check_ranges(type(self), temp=temp, salt=salt, depth=depth_positive)
        c = self._equation(temp, salt, depth_positive)
        if np.isscalar(depth):
            return float(c)
        return c


class UNESCO(SoundSpeedProfile):
    """UNESCO (Chen-Millero) sound speed profile model.

    This model calculates the sound speed using the UNESCO algorithm of Chen and Millero (1977),
    the standard international equation, which is expressed in terms of temperature, salinity
    and pressure rather than depth. Pressure is taken from ``pressure_profile`` when supplied,
    and otherwise converted from depth by the base class for ``latitude`` and
    ``pressure_region``. This implementation uses internal models for temperature and salinity
    as a function of depth unless ``temperature_profile`` and ``salinity_profile`` are supplied.

    The equation is valid for temperatures of 0 to 40 degrees Celsius, salinities of 0 to 40
    PSU, and pressures of 0 to 1000 bar.

    References
    ----------
    Chen, C.-T. and Millero, F. J. (1977). Speed of sound in seawater at high pressures. Journal
    of the Acoustical Society of America, 62(5), 1129-1135.

    """

    _required_inputs = ("temp", "salt", "pressure")
    _valid_ranges: ValidRanges = {
        "temp": (0.0, 40.0),
        "salt": (0.0, 40.0),
        "pressure": (0.0, 1000.0),
    }

    @staticmethod
    def _equation(temp: ArrayLike, salt: ArrayLike, pressure: ArrayLike) -> FloatArray:
        """Evaluate the UNESCO (Chen-Millero) polynomial.

        Pure function of temperature, salinity, and pressure, with no data lookup or unit
        conversion.

        Parameters
        ----------
        temp : ArrayLike
            Temperature in degrees Celsius.
        salt : ArrayLike
            Salinity in PSU.
        pressure : ArrayLike
            Pressure in bar. Any shape broadcastable against temp/salt.

        Returns
        -------
        FloatArray
            Sound speed in m/s, with the broadcast shape of the inputs.

        """
        temp = np.asarray(temp, dtype=float)
        salt = np.asarray(salt, dtype=float)
        pressure = np.asarray(pressure, dtype=float)

        c_w = (
            (
                1402.388
                + 5.03830 * temp
                - 5.81090e-2 * temp**2
                + 3.3432e-4 * temp**3
                - 1.47797e-6 * temp**4
                + 3.1419e-9 * temp**5
            )
            + (
                0.153563
                + 6.8999e-4 * temp
                - 8.1829e-6 * temp**2
                + 1.3632e-7 * temp**3
                - 6.1260e-10 * temp**4
            )
            * pressure
            + (
                3.1260e-5
                - 1.7111e-6 * temp
                + 2.5986e-8 * temp**2
                - 2.5353e-10 * temp**3
                + 1.0415e-12 * temp**4
            )
            * pressure**2
            + (-9.7729e-9 + 3.8513e-10 * temp - 2.3654e-12 * temp**2) * pressure**3
        )
        a = (
            (1.389 - 1.262e-2 * temp + 7.166e-5 * temp**2 + 2.008e-6 * temp**3 - 3.21e-8 * temp**4)
            + (
                9.4742e-5
                - 1.2583e-5 * temp
                - 6.4928e-8 * temp**2
                + 1.0515e-8 * temp**3
                - 2.0142e-10 * temp**4
            )
            * pressure
            + (-3.9064e-7 + 9.1061e-9 * temp - 1.6009e-10 * temp**2 + 7.994e-12 * temp**3)
            * pressure**2
            + (1.100e-10 + 6.651e-12 * temp - 3.391e-13 * temp**2) * pressure**3
        )
        b = -1.922e-2 - 4.42e-5 * temp + (7.3637e-5 + 1.7950e-7 * temp) * pressure
        d = 1.727e-3 - 7.9836e-6 * pressure

        c = c_w + a * salt + b * salt**1.5 + d * salt**2
        return c

    def calculate(self, depth: DepthInput) -> SpeedOutput:
        """Calculate sound speed using the UNESCO (Chen-Millero) equation.

        Parameters
        ----------
        depth : DepthInput
            Depth in meters. If negative (z-coordinate), converts to positive depth below surface
            for calculation.

        Returns
        -------
        SpeedOutput
            Sound speed in m/s.

        """
        depth_positive = abs(depth)
        temp = self._get_temperature(depth_positive)
        salt = self._get_salinity(depth_positive)
        pressure = self._get_pressure(depth_positive)
        self._check_ranges(type(self), temp=temp, salt=salt, pressure=pressure)
        c = self._equation(temp, salt, pressure)
        if np.isscalar(depth):
            return float(c)
        return c


class DelGrosso(SoundSpeedProfile):
    """Del Grosso sound speed profile model.

    This model calculates the sound speed using the equation of Del Grosso (1974), fitted to a
    separate set of experiments from the UNESCO algorithm and over a narrower range of
    conditions. Some researchers prefer it to UNESCO, particularly for calculations inside its
    own domain of validity, but the relative accuracy of the two is a live question in the
    literature rather than a settled one; see the module docstring. Del Grosso's coefficients
    are defined for pressure in kg/cm^2; ``_equation`` accepts bar and converts internally, so
    its signature matches UNESCO. Pressure is taken from ``pressure_profile`` when supplied, and
    otherwise converted from depth by the base class. This implementation uses internal models
    for temperature and salinity as a function of depth unless ``temperature_profile`` and
    ``salinity_profile`` are supplied.

    The equation is valid for temperatures of 0 to 30 degrees Celsius, salinities of 30 to 40
    PSU, and pressures of 0 to 1000 kg/cm^2 (approximately 981 bar). It should be expected to
    drift from the other models outside that range. Since the case for preferring it is tied to
    its domain, ``SoundSpeedRangeWarning`` is worth heeding for this model in particular.

    References
    ----------
    Del Grosso, V. A. (1974). New equation for the speed of sound in natural waters (with
    comparisons to other equations). Journal of the Acoustical Society of America, 56(4),
    1084-1091.

    """

    _required_inputs = ("temp", "salt", "pressure")
    _valid_ranges: ValidRanges = {
        "temp": (0.0, 30.0),
        "salt": (30.0, 40.0),
        # Del Grosso states 0 to 1000 kg/cm^2; expressed here in bar to match _equation.
        "pressure": (0.0, 1000.0 / KGF_PER_CM2_PER_BAR),
    }

    @staticmethod
    def _equation(temp: ArrayLike, salt: ArrayLike, pressure: ArrayLike) -> FloatArray:
        """Evaluate the Del Grosso polynomial.

        Pure function of temperature, salinity, and pressure in bar; the bar -> kg/cm^2
        conversion required by Del Grosso's own coefficients is handled internally, so the
        public signature stays consistent (pressure in bar) with UNESCO.

        Parameters
        ----------
        temp : ArrayLike
            Temperature in degrees Celsius.
        salt : ArrayLike
            Salinity in PSU.
        pressure : ArrayLike
            Pressure in bar. Any shape broadcastable against temp/salt.

        Returns
        -------
        FloatArray
            Sound speed in m/s, with the broadcast shape of the inputs.

        """
        temp = np.asarray(temp, dtype=float)
        salt = np.asarray(salt, dtype=float)
        pressure_bar = np.asarray(pressure, dtype=float)

        pressure_kgcm2 = pressure_bar * KGF_PER_CM2_PER_BAR

        c000 = 1402.392
        delta_ct = 0.5012285e1 * temp - 0.551184e-1 * temp**2 + 0.221649e-3 * temp**3
        delta_cs = 0.1329530e1 * salt + 0.1288598e-3 * salt**2
        delta_cp = (
            0.1560592 * pressure_kgcm2
            + 0.2449993e-4 * pressure_kgcm2**2
            - 0.8833959e-8 * pressure_kgcm2**3
        )
        delta_cstp = (
            -0.1275936e-1 * salt * temp
            + 0.6353509e-2 * temp * pressure_kgcm2
            + 0.2656174e-7 * temp**2 * pressure_kgcm2**2
            - 0.1593895e-5 * temp * pressure_kgcm2**2
            + 0.5222483e-9 * temp * pressure_kgcm2**3
            - 0.4383615e-6 * temp**3 * pressure_kgcm2
            - 0.1616745e-8 * salt**2 * pressure_kgcm2**2
            + 0.9688441e-4 * salt * temp**2
            + 0.4857614e-5 * salt**2 * temp * pressure_kgcm2
            - 0.3406824e-3 * salt * temp * pressure_kgcm2
        )

        c = c000 + delta_ct + delta_cs + delta_cp + delta_cstp
        return c

    def calculate(self, depth: DepthInput) -> SpeedOutput:
        """Calculate sound speed using the Del Grosso equation.

        Parameters
        ----------
        depth : DepthInput
            Depth in meters. If negative (z-coordinate), converts to positive depth below surface
            for calculation.

        Returns
        -------
        SpeedOutput
            Sound speed in m/s.

        """
        depth_positive = abs(depth)
        temp = self._get_temperature(depth_positive)
        salt = self._get_salinity(depth_positive)
        pressure = self._get_pressure(depth_positive)
        self._check_ranges(type(self), temp=temp, salt=salt, pressure=pressure)
        c = self._equation(temp, salt, pressure)
        if np.isscalar(depth):
            return float(c)
        return c


class NPL(SoundSpeedProfile):
    """NPL (Leroy-Robinson-Goldsmith) sound speed profile model.

    This model calculates the sound speed using the equation of Leroy, Robinson and Goldsmith
    (2008), designed for a single global fit across ordinary ocean conditions (salinity up to 42
    ppt). Unlike the other empirical models here, it uses depth and latitude directly rather than
    pressure. This implementation uses internal models for temperature and salinity as a function
    of depth.

    Attributes
    ----------
    latitude : float
        Latitude in degrees, used in the depth-latitude correction term. Defaults to 45.0, at
        which this term vanishes.

    References
    ----------
    Leroy, C. C., Robinson, S. P. and Goldsmith, M. J. (2008). A new equation for the accurate
    calculation of sound speed in all oceans. Journal of the Acoustical Society of America,
    124(5), 2774-2782.

    """

    _required_inputs = ("temp", "salt", "depth", "latitude")
    _valid_ranges: ValidRanges = {"salt": (-np.inf, 42.0)}

    @staticmethod
    def _equation(
        temp: ArrayLike, salt: ArrayLike, depth: ArrayLike, latitude: ArrayLike
    ) -> FloatArray:
        """Evaluate the Leroy-Robinson-Goldsmith polynomial.

        Pure function of already-positive depth, temperature, salinity, and latitude, with no
        depth-sign handling or data lookup. Shared with LeroyCopernicusSoundSpeedProfile so the
        two implementations cannot drift apart; do not duplicate this expression elsewhere.

        Parameters
        ----------
        temp : ArrayLike
            Temperature in degrees Celsius.
        salt : ArrayLike
            Salinity in PSU.
        depth : ArrayLike
            Positive depth in meters. Any shape broadcastable against temp/salt.
        latitude : ArrayLike
            Latitude in degrees. Any shape broadcastable against depth.

        Returns
        -------
        FloatArray
            Sound speed in m/s, with the broadcast shape of the inputs.

        """
        temp = np.asarray(temp, dtype=float)
        salt = np.asarray(salt, dtype=float)
        depth = np.asarray(depth, dtype=float)
        latitude = np.asarray(latitude, dtype=float)
        c = (
            1402.5
            + 5 * temp
            - 5.44e-2 * temp**2
            + 2.1e-4 * temp**3
            + 1.33 * salt
            - 1.23e-2 * salt * temp
            + 8.7e-5 * salt * temp**2
            + 1.56e-2 * depth
            + 2.55e-7 * depth**2
            - 7.3e-12 * depth**3
            + 1.2e-6 * depth * (latitude - 45)
            - 9.5e-13 * temp * depth**3
            + 3e-7 * temp**2 * depth
            + 1.43e-5 * salt * depth
        )
        return c

    def calculate(self, depth: DepthInput) -> SpeedOutput:
        """Calculate sound speed using the NPL equation.

        Parameters
        ----------
        depth : DepthInput
            Depth in meters. If negative (z-coordinate), converts to positive depth below surface
            for calculation.

        Returns
        -------
        SpeedOutput
            Sound speed in m/s.

        """
        # Convert negative z-coordinate to positive depth below surface
        depth_positive = abs(depth)

        temp = self._get_temperature(depth_positive)
        salt = self._get_salinity(depth_positive)

        self._check_ranges(
            type(self), temp=temp, salt=salt, depth=depth_positive, latitude=self.latitude
        )
        c = self._equation(temp, salt, depth_positive, self.latitude)
        if np.isscalar(depth):
            return float(c)
        return c


class CopernicusSoundSpeedProfile(SoundSpeedProfile):
    """SSP model built from Copernicus temperature/salinity, evaluated with a chosen equation.

    Notes
    -----
    - Copernicus depth is expected in oceanographic convention (``+z`` downward).
    - ``calculate`` accepts ``-z`` depth inputs and internally uses ``abs(depth)``.
    - ``get_3d_grid`` returns ``z_grid`` in RTRS convention (``+z`` downward).
    - Copernicus provides no pressure field; where the chosen equation requires pressure, it is
      obtained from ``pressure_profile`` if supplied, otherwise converted from depth for
      ``latitude`` and ``pressure_region`` (see SoundSpeedProfile._calc_pressure). Both are
      single values rather than taken from the grid, so set them to suit the extract.

    """

    temperature_file_path: str = Property(doc="Path to Copernicus temperature NetCDF file")
    salinity_file_path: str = Property(doc="Path to Copernicus salinity NetCDF file")
    equation_cls: type[EquationProfile] = Property(
        default=NPL,
        doc="Which empirical equation class to evaluate on the loaded Copernicus grid "
        "(e.g. NPL, Mackenzie, Coppens, UNESCO, DelGrosso). Must satisfy the EquationProfile "
        "contract, i.e. define both _equation and _required_inputs.",
    )
    reference_lat_deg: float | None = Property(
        default=None,
        doc="Reference latitude for local x/y conversion. Defaults to dataset midpoint.",
    )
    reference_lon_deg: float | None = Property(
        default=None,
        doc="Reference longitude for local x/y conversion. Defaults to dataset midpoint.",
    )
    fill_speed_m_s: float = Property(
        default=1480.0,
        doc="Fallback fill value for columns with no finite Copernicus values.",
    )

    def _evaluate_equation(
        self,
        z_m: ArrayLike,
        temp_zyx: ArrayLike,
        sal_zyx: ArrayLike,
        lat_deg: ArrayLike,
    ) -> FloatArray:
        """Evaluate self.equation_cls._equation over the full (z, y, x) Copernicus grid.

        Builds every quantity a supported equation might need (temp, salt, depth, pressure,
        latitude) at the correct broadcastable shape, then passes only the subset named in
        equation_cls._required_inputs.

        Parameters
        ----------
        z_m : ArrayLike
            1D array of positive depths in meters.
        temp_zyx : ArrayLike
            3D array of temperature, shape (depth, lat, lon).
        sal_zyx : ArrayLike
            3D array of salinity, matching temp_zyx.
        lat_deg : ArrayLike
            1D array of latitudes in degrees, matching the lat axis of temp_zyx/sal_zyx.

        Returns
        -------
        FloatArray
            Sound speed, shape (depth, lat, lon).

        """
        equation_cls = self.equation_cls
        name = getattr(equation_cls, "__name__", repr(equation_cls))
        required = getattr(equation_cls, "_required_inputs", None)
        if required is None or not hasattr(equation_cls, "_equation"):
            raise TypeError(
                f"equation_cls={name} does not satisfy the EquationProfile contract: it must "
                "define both _required_inputs and _equation. Analytic profiles such as Constant, "
                "Linear, Arctan and Munk have no temperature/salinity equation and cannot be "
                "used here; use NPL, Mackenzie, Coppens, UNESCO or DelGrosso."
            )

        z_array = np.asarray(z_m, dtype=float)
        depth4 = z_array[:, None, None]
        lat4 = np.asarray(lat_deg, dtype=float)[None, :, None]
        pressure4 = self._get_pressure(depth4)

        available = {
            "temp": np.asarray(temp_zyx, dtype=float),
            "salt": np.asarray(sal_zyx, dtype=float),
            "depth": depth4,
            "pressure": pressure4,
            "latitude": lat4,
        }
        unknown = sorted(set(required) - EQUATION_INPUTS)
        if unknown:
            raise TypeError(
                f"equation_cls={name} declares unsupported _required_inputs {unknown}; "
                f"supported names are {sorted(EQUATION_INPUTS)}."
            )
        kwargs = {input_name: available[input_name] for input_name in required}
        self._check_ranges(equation_cls, **kwargs)
        return np.asarray(equation_cls._equation(**kwargs), dtype=float)

    def _load_data(self) -> None:
        """Load Copernicus T/S fields and precompute sound speed on native grids."""
        temp_path = Path(self.temperature_file_path)
        sal_path = Path(self.salinity_file_path)
        if not temp_path.exists():
            raise FileNotFoundError(f"Copernicus temperature file not found: {temp_path}")
        if not sal_path.exists():
            raise FileNotFoundError(f"Copernicus salinity file not found: {sal_path}")

        try:
            import netCDF4 as nc
        except ImportError as exc:
            raise ImportError(
                "netCDF4 is required for CopernicusSoundSpeedProfile. "
                "Install with `pip install netCDF4`."
            ) from exc

        with nc.Dataset(temp_path, "r") as ds_t:
            t = self._to_float_with_nan(ds_t.variables["thetao"][0, :, :, :])
            self._z_m = np.asarray(ds_t.variables["depth"][:], dtype=float)
            self._lat_deg = np.asarray(ds_t.variables["latitude"][:], dtype=float)
            self._lon_deg = np.asarray(ds_t.variables["longitude"][:], dtype=float)

        with nc.Dataset(sal_path, "r") as ds_s:
            s = self._to_float_with_nan(ds_s.variables["so"][0, :, :, :])

        if t.shape != s.shape:
            raise ValueError(
                "Copernicus temperature and salinity arrays must have matching shapes."
            )
        if t.ndim != 3:
            raise ValueError("Copernicus arrays must have shape (depth, lat, lon).")
        if self._z_m.ndim != 1 or self._lat_deg.ndim != 1 or self._lon_deg.ndim != 1:
            raise ValueError("Copernicus depth/latitude/longitude coordinates must be 1D.")
        if np.any(np.diff(self._z_m) <= 0.0) or self._z_m[0] < 0.0:
            raise ValueError("Copernicus depth axis must be non-negative and strictly increasing.")

        lat0 = self.reference_lat_deg
        lon0 = self.reference_lon_deg
        if lat0 is None:
            lat0 = float(0.5 * (self._lat_deg.min() + self._lat_deg.max()))
        if lon0 is None:
            lon0 = float(0.5 * (self._lon_deg.min() + self._lon_deg.max()))

        self.reference_lat_deg = float(lat0)
        self.reference_lon_deg = float(lon0)

        self._x_m, _ = self._latlon_to_xy_m(
            np.full_like(self._lon_deg, self.reference_lat_deg),
            self._lon_deg,
            self.reference_lat_deg,
            self.reference_lon_deg,
        )
        _, self._y_m = self._latlon_to_xy_m(
            self._lat_deg,
            np.full_like(self._lat_deg, self.reference_lon_deg),
            self.reference_lat_deg,
            self.reference_lon_deg,
        )

        # Ensure monotonic increasing x/y for interpolation.
        if np.any(np.diff(self._x_m) < 0.0):
            x_order = np.argsort(self._x_m)
            self._x_m = self._x_m[x_order]
            self._lon_deg = self._lon_deg[x_order]
            t = t[:, :, x_order]
            s = s[:, :, x_order]
        if np.any(np.diff(self._y_m) < 0.0):
            y_order = np.argsort(self._y_m)
            self._y_m = self._y_m[y_order]
            self._lat_deg = self._lat_deg[y_order]
            t = t[:, y_order, :]
            s = s[:, y_order, :]

        self._c_zyx = self._evaluate_equation(self._z_m, t, s, self._lat_deg)

        # Representative 1D profile for pointwise speed requests in existing interfaces.
        valid_counts = np.sum(np.isfinite(self._c_zyx), axis=(1, 2))
        summed = np.nansum(self._c_zyx, axis=(1, 2))
        self._c_z_mean = np.divide(
            summed,
            valid_counts,
            out=np.full(len(self._z_m), np.nan, dtype=float),
            where=valid_counts > 0,
        )
        nan_mask = ~np.isfinite(self._c_z_mean)
        if np.any(nan_mask):
            finite_mask = np.isfinite(self._c_z_mean)
            if not np.any(finite_mask):
                self._c_z_mean = np.full_like(self._z_m, self.fill_speed_m_s, dtype=float)
            else:
                self._c_z_mean[nan_mask] = np.interp(
                    self._z_m[nan_mask],
                    self._z_m[finite_mask],
                    self._c_z_mean[finite_mask],
                )
        self._is_loaded = True

    def __post_init__(self) -> None:
        """Attempt eager load; public methods also support lazy loading."""
        self._is_loaded = False
        self._load_data()

    def _ensure_loaded(self) -> None:
        """Ensure cached Copernicus arrays are loaded."""
        if getattr(self, "_is_loaded", False):
            return
        self._load_data()

    @staticmethod
    def _to_float_with_nan(var_data: ArrayLike) -> FloatArray:
        arr = np.ma.array(var_data)
        arr = np.ma.filled(arr, np.nan)
        arr = np.asarray(arr, dtype=float)
        arr[~np.isfinite(arr)] = np.nan
        arr[np.abs(arr) > 1.0e4] = np.nan
        return arr

    @staticmethod
    def _latlon_to_xy_m(
        lat_deg: ArrayLike,
        lon_deg: ArrayLike,
        lat0_deg: float,
        lon0_deg: float,
    ) -> tuple[FloatArray, FloatArray]:
        lat_array = np.asarray(lat_deg, dtype=float)
        lon_array = np.asarray(lon_deg, dtype=float)
        lat0_rad = np.deg2rad(lat0_deg)
        dlon_rad = np.deg2rad(lon_array - lon0_deg)
        dlat_rad = np.deg2rad(lat_array - lat0_deg)

        a = 6_378_137.0
        f = 1.0 / 298.257223563
        e2 = f * (2.0 - f)

        sin_lat0 = np.sin(lat0_rad)
        w = np.sqrt(1.0 - e2 * sin_lat0**2)
        n = a / w
        m = a * (1.0 - e2) / (w**3)

        x = dlon_rad * n * np.cos(lat0_rad)
        y = dlat_rad * m
        return x, y

    @staticmethod
    def _fill_nans_2d(arr_2d: ArrayLike) -> FloatArray:
        out = np.array(arr_2d, dtype=float, copy=True)
        ny, nx = out.shape

        for row_idx in range(ny):
            row = out[row_idx, :]
            valid = np.isfinite(row)
            if np.any(valid):
                out[row_idx, :] = np.interp(np.arange(nx), np.where(valid)[0], row[valid])

        for col_idx in range(nx):
            col = out[:, col_idx]
            valid = np.isfinite(col)
            if np.any(valid):
                out[:, col_idx] = np.interp(np.arange(ny), np.where(valid)[0], col[valid])

        return out

    @staticmethod
    def _interp_2d_regular(
        z_old_yx: ArrayLike,
        x_old: ArrayLike,
        y_old: ArrayLike,
        x_new: ArrayLike,
        y_new: ArrayLike,
    ) -> FloatArray:
        z_old_yx = np.asarray(z_old_yx, dtype=float)
        x_old_array = np.asarray(x_old, dtype=float)
        y_old_array = np.asarray(y_old, dtype=float)
        x_new_array = np.asarray(x_new, dtype=float)
        y_new_array = np.asarray(y_new, dtype=float)
        z_x = np.vstack([np.interp(x_new_array, x_old_array, row) for row in z_old_yx])
        z_xy = np.vstack(
            [np.interp(y_new_array, y_old_array, z_x[:, i]) for i in range(z_x.shape[1])]
        ).T
        return z_xy

    @staticmethod
    def _interp_3d_horizontal(
        c_zyx: ArrayLike,
        x_old: ArrayLike,
        y_old: ArrayLike,
        x_new: ArrayLike,
        y_new: ArrayLike,
    ) -> FloatArray:
        c_array = np.asarray(c_zyx, dtype=float)
        x_old_array = np.asarray(x_old, dtype=float)
        y_old_array = np.asarray(y_old, dtype=float)
        x_new_array = np.asarray(x_new, dtype=float)
        y_new_array = np.asarray(y_new, dtype=float)
        nz = c_array.shape[0]
        out = np.empty((nz, len(y_new_array), len(x_new_array)), dtype=float)
        for k in range(nz):
            layer = CopernicusSoundSpeedProfile._fill_nans_2d(c_array[k, :, :])
            out[k, :, :] = CopernicusSoundSpeedProfile._interp_2d_regular(
                layer,
                x_old_array,
                y_old_array,
                x_new_array,
                y_new_array,
            )
        return out

    @staticmethod
    def _extrapolate_columns_to_depth(
        c_zyx: ArrayLike,
        z_in: ArrayLike,
        z_out: ArrayLike,
        c_fill: float,
    ) -> FloatArray:
        c_array = np.asarray(c_zyx, dtype=float)
        z_in_array = np.asarray(z_in, dtype=float)
        z_out_array = np.asarray(z_out, dtype=float)
        ny, nx = c_array.shape[1], c_array.shape[2]
        out = np.full((len(z_out_array), ny, nx), c_fill, dtype=float)
        for j in range(ny):
            for i in range(nx):
                col = c_array[:, j, i]
                valid = np.isfinite(col)
                if np.sum(valid) == 0:
                    continue
                if np.sum(valid) == 1:
                    out[:, j, i] = col[valid][0]
                    continue

                zv = z_in_array[valid]
                cv = col[valid]
                out[:, j, i] = np.interp(z_out_array, zv, cv)

                deep = z_out_array > zv[-1]
                slope = (cv[-1] - cv[-2]) / (zv[-1] - zv[-2])
                out[deep, j, i] = cv[-1] + slope * (z_out_array[deep] - zv[-1])

        return out

    @staticmethod
    def _normalise_z_range_to_positive(z_range: tuple[float, float]) -> tuple[float, float]:
        z0, z1 = float(z_range[0]), float(z_range[1])
        if z0 <= 0.0 and z1 <= 0.0:
            # -z for underwater.
            return min(abs(z0), abs(z1)), max(abs(z0), abs(z1))

        if z0 >= 0.0 and z1 >= 0.0:
            # Already oceanographic/RTRS (+z downward).
            return min(z0, z1), max(z0, z1)

        return min(abs(z0), abs(z1)), max(abs(z0), abs(z1))

    def calculate(self, depth: DepthInput) -> SpeedOutput:
        """Calculate representative sound speed at ``depth`` using a domain-mean profile."""
        self._ensure_loaded()
        depth_arr = np.asarray(depth, dtype=float)
        depth_pos = np.abs(depth_arr)
        depth_pos = np.clip(depth_pos, self._z_m[0], self._z_m[-1])
        c = np.interp(depth_pos, self._z_m, self._c_z_mean)
        if np.isscalar(depth):
            return float(c)
        return c

    def get_3d_grid(
        self,
        x_range: Range1D,
        y_range: Range1D,
        z_range: Range1D,
        x_res: float = 5000.0,
        y_res: float = 5000.0,
        z_res: float = 100.0,
    ) -> Grid3DResult:
        """Get a regular SSP cube resampled from Copernicus data for RTRS."""
        self._ensure_loaded()
        x_min, x_max = x_range
        y_min, y_max = y_range

        x_points = int((x_max - x_min) / x_res) + 1
        y_points = int((y_max - y_min) / y_res) + 1

        x_grid = np.linspace(x_min, x_max, max(2, x_points))
        y_grid = np.linspace(y_min, y_max, max(2, y_points))

        z_min_pos, z_max_pos = self._normalise_z_range_to_positive(z_range)
        z_points = int((z_max_pos - z_min_pos) / z_res) + 1
        z_grid = np.linspace(z_min_pos, z_max_pos, max(2, z_points))

        c_reg_h = self._interp_3d_horizontal(self._c_zyx, self._x_m, self._y_m, x_grid, y_grid)
        c_reg = self._extrapolate_columns_to_depth(
            c_reg_h,
            self._z_m,
            z_grid,
            c_fill=float(self.fill_speed_m_s),
        )

        # Convert (nz, ny, nx) -> (nx, ny, nz), then flatten C-order for RTRS.
        c_grid_flat = np.transpose(c_reg, (2, 1, 0)).flatten(order="C")

        return x_grid, y_grid, z_grid, c_grid_flat


class LeroyCopernicusSoundSpeedProfile(CopernicusSoundSpeedProfile):
    """Deprecated alias for ``CopernicusSoundSpeedProfile(equation_cls=NPL)``.

    Kept for backward compatibility, and equivalent in every respect to the class it aliases.
    Prefer ``CopernicusSoundSpeedProfile`` directly, naming ``equation_cls`` explicitly. The
    "Leroy" in the name refers to the NPL equation of Leroy, Robinson and Goldsmith (2008); see
    :class:`NPL`.

    .. deprecated::
        Use ``CopernicusSoundSpeedProfile(equation_cls=NPL)`` instead. Instantiating this class
        emits a ``DeprecationWarning``.

    """

    equation_cls: type[EquationProfile] = Property(default=NPL, doc="Fixed to NPL.")

    def __init__(self, *args, **kwargs) -> None:
        """Warn that this alias is deprecated, then initialise as usual."""
        warnings.warn(
            "LeroyCopernicusSoundSpeedProfile is deprecated; use "
            "CopernicusSoundSpeedProfile(equation_cls=NPL) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
