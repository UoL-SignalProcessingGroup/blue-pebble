r"""Sensor self-noise models.

This module collects spectral models for noise that originates at the sensor itself, as distinct
from ambient environmental noise fields. Sensor self-noise scales with sensor and platform state
(e.g. the flow past the hydrophone face, the preamp temperature) rather than with the ocean
environment.

Noise sources are tagged with the signal domain in which they act. Pressure-domain noise (e.g.
turbulent boundary layer flow noise) enters at the sensor face and is subsequently coloured by the
hydrophone transfer function. Voltage-domain noise (e.g. preamp Johnson-Nyquist noise) enters after
the transducer and bypasses the transfer function. The :attr:`SensorNoiseSpectrum.domain` attribute
lets the simulator route each source to the correct injection point.

Noise models consume two kinds of information:

- **Time-varying motion state** is passed in as a Stone Soup :class:`~stonesoup.types.state.State`,
  consistent with how the rest of the simulation pipeline represents platform state. Each subclass
  extracts what it needs (e.g. speed from the velocity components).
- **Sensor-specific parameters** (resistance, temperature, etc.) are declared as
  :class:`~stonesoup.base.Property` fields on the subclass itself; geometric parameters that vary
  per array element, such as streamwise offset, are passed as keyword arguments at call time.

Classes
-------
SensorNoiseSpectrum
    Abstract base class for sensor self-noise power spectral density
    models.
GoodyFlowNoiseSpectrum
    Turbulent boundary layer flow-noise model of Goody (2004).
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, ClassVar, Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from stonesoup.base import Base, Property
from stonesoup.types.state import State

FloatArray: TypeAlias = NDArray[np.floating[Any]]
NoiseDomain: TypeAlias = Literal["pressure", "voltage"]

REFERENCE_PRESSURE_PA: float = 1.0e-6
"""Reference pressure for underwater acoustics (1 uPa) in Pa."""


class SensorNoiseSpectrum(ABC, Base):
    r"""Abstract base class for sensor self-noise power spectral density models.

    Subclasses implement :meth:`psd` to return the one-sided PSD at a given set of frequencies for
    a given platform state, and declare their :attr:`domain` so the simulator knows where to inject
    the noise in the signal chain.

    Units depend on ``domain``:

    - ``"pressure"``: PSD in :math:`\text{Pa}^2/\text{Hz}`, added to the incident acoustic pressure
      before the hydrophone transfer function.
    - ``"voltage"``: PSD in :math:`\text{V}^2/\text{Hz}`, added to the sensor output after the
      hydrophone transfer function.

    The ``streamwise_position_m`` keyword argument is accepted by every subclass so the simulator
    can pass it uniformly, but is only meaningful for flow-type noise sources; other subclasses are
    free to ignore it.

    """

    domain: ClassVar[NoiseDomain]
    """Signal-chain domain in which this noise source acts."""

    @abstractmethod
    def psd(
        self,
        frequencies_hz: ArrayLike,
        platform_state: State,
        *,
        streamwise_position_m: float | None = None,
    ) -> FloatArray:
        """Evaluate the one-sided PSD at the given frequencies.

        Parameters
        ----------
        frequencies_hz : ArrayLike
            Frequencies in Hz at which to evaluate the PSD. Values are treated as non-negative;
            negative entries are interpreted via their absolute value.
        platform_state : stonesoup.types.state.State
            Platform state at the evaluation time. Subclasses extract whatever they need (e.g.
            speed from the velocity components).
        streamwise_position_m : float, optional
            Distance from the effective boundary layer origin to this sensor along the flow
            direction, in m. Required by flow-type noise sources; ignored by sources that do not
            depend on array geometry (e.g. preamp electronic noise).

        Returns
        -------
        FloatArray
            One-sided PSD in the units implied by :attr:`domain`, shape ``(num_frequencies,)``.

        """
        ...

    def level_db(
        self,
        frequencies_hz: ArrayLike,
        platform_state: State,
        *,
        streamwise_position_m: float | None = None,
    ) -> FloatArray:
        r"""Return the PSD expressed in dB re :math:`1\,\mu\text{Pa}^2/\text{Hz}`.

        Only meaningful for pressure-domain noise sources. For voltage-domain sources, use
        :meth:`psd` directly and convert to the reference of your choice.

        Parameters
        ----------
        frequencies_hz : ArrayLike
            Frequencies in Hz at which to evaluate the level.
        platform_state : stonesoup.types.state.State
            Platform state at the evaluation time.
        streamwise_position_m : float, optional
            Passed through to :meth:`psd`; see that method for details.

        Returns
        -------
        FloatArray
            Noise spectral level in dB re :math:`1\,\mu\text{Pa}^2/\text{Hz}`.

        Raises
        ------
        ValueError
            If this model is not a pressure-domain source.

        """
        if self.domain != "pressure":
            msg = (
                f"level_db (dB re 1 uPa^2/Hz) is only defined for "
                f"pressure-domain noise sources; this model has domain "
                f"{self.domain!r}."
            )
            raise ValueError(msg)
        psd = self.psd(
            frequencies_hz,
            platform_state,
            streamwise_position_m=streamwise_position_m,
        )
        return 10.0 * np.log10(psd / REFERENCE_PRESSURE_PA**2)


class GoodyFlowNoiseSpectrum(SensorNoiseSpectrum):
    r"""Turbulent boundary layer wall-pressure spectrum of Goody (2004).

    Models the hydrophone self-noise produced by turbulent boundary layer (TBL) pressure
    fluctuations acting on the sensor face as the sensor moves through water. The spectral shape is
    the semi-empirical dimensionless form of Goody (2004), validated extensively for zero pressure
    gradient TBLs:

    .. math::

        \Phi_{pp}(\omega) \frac{U_\infty}{\tau_w^2 \delta^*}
            = \frac{C_2 \, \tilde{\omega}^2}
                   {\left[\tilde{\omega}^{0.75} + C_1\right]^{3.7}
                    + \left[C_3 \, R_T^{-0.57} \, \tilde{\omega}\right]^7}

    where :math:`\tilde{\omega} = \omega \delta^* / U_\infty` is the outer-scaled angular frequency
    and :math:`R_T = (\delta^*/U_\infty)(\tau_w/\rho\nu)` is the ratio of outer to inner time
    scales. The model captures the low-frequency plateau, the overlap region, and the
    :math:`\omega^{-5}` viscous roll-off at high frequency.

    The free-stream speed :math:`U_\infty` is obtained from the platform state by taking the
    Euclidean norm of the velocity components indexed by :attr:`velocity_mapping`. The wall shear
    stress :math:`\tau_w` and displacement thickness :math:`\delta^*` are derived from that speed
    and the sensor streamwise position via a turbulent flat-plate TBL correlation. The default
    correlation is Schlichting's 1/7-power law:

    .. math::

        C_f = 0.0576 \, Re_x^{-1/5}, \qquad
        \delta = 0.37 \, x \, Re_x^{-1/5}, \qquad
        \delta^* = \delta / 8, \qquad
        \tau_w = \tfrac{1}{2} \rho U_\infty^2 C_f

    where :math:`Re_x = U_\infty x / \nu`.

    Outputs are returned as a one-sided PSD in :math:`\text{Pa}^2/\text{Hz}`. The dimensionless
    Goody formula is evaluated in the angular-frequency domain and converted via
    :math:`\Phi_{pp}(f) = 2\pi \, \Phi_{pp}(\omega)`.

    Parameters
    ----------
    fluid_density_kg_m3 : float, optional
        Fluid mass density in :math:`\text{kg/m}^3`. Defaults to ``1025.0`` (typical seawater).
    kinematic_viscosity_m2_s : float, optional
        Fluid kinematic viscosity in :math:`\text{m}^2/\text{s}`. Defaults to ``1.35e-6`` (seawater
        near 10 \u00B0C).
    velocity_mapping : sequence of int, optional
        Indices of the velocity components within ``platform_state.state_vector``. Defaults to
        ``(1, 3, 5)``, the Stone Soup convention for a 3-D position-velocity state ordered
        ``[x, vx, y, vy, z, vz]``.
    c1 : float, optional
        Goody constant :math:`C_1`.  Defaults to ``0.5``.
    c2 : float, optional
        Goody constant :math:`C_2`.  Defaults to ``3.0``.
    c3 : float, optional
        Goody constant :math:`C_3`.  Defaults to ``1.1``.

    Notes
    -----
    The Schlichting correlation assumes a turbulent zero-pressure-gradient flat-plate boundary
    layer. A ``UserWarning`` is emitted when :math:`Re_x < 5 \times 10^5`, below which the flow is
    likely laminar or transitional and the correlation is no longer valid.

    References
    ----------
    [1] M. Goody, "Empirical spectral model of surface pressure fluctuations," AIAA J., vol. 42,
        no. 9, pp. 1788-1794, Sep. 2004, doi: 10.2514/1.2486.

    [2] H. Schlichting, Boundary-Layer Theory, 7th ed. New York, NY, USA: McGraw-Hill, 1979,
        ch. 21.

    """

    domain: ClassVar[NoiseDomain] = "pressure"

    fluid_density_kg_m3: float = Property(
        default=1025.0,
        doc="Fluid mass density in kg/m^3.  Defaults to 1025.0 (seawater).",
    )
    kinematic_viscosity_m2_s: float = Property(
        default=1.35e-6,
        doc="Fluid kinematic viscosity in m^2/s. Defaults to 1.35e-6 (seawater ~10 C).",
    )
    velocity_mapping: Sequence[int] = Property(
        default=(1, 3, 5),
        doc=(
            "Indices of the velocity components within platform_state.state_vector. "
            "Defaults to (1, 3, 5), the Stone Soup convention for a 3-D position-velocity state."
        ),
    )
    c1: float = Property(
        default=0.5,
        doc="Goody constant C_1.  Defaults to 0.5.",
    )
    c2: float = Property(
        default=3.0,
        doc="Goody constant C_2.  Defaults to 3.0.",
    )
    c3: float = Property(
        default=1.1,
        doc="Goody constant C_3.  Defaults to 1.1.",
    )

    def psd(
        self,
        frequencies_hz: ArrayLike,
        platform_state: State,
        *,
        streamwise_position_m: float | None = None,
    ) -> FloatArray:
        """Evaluate the Goody one-sided wall-pressure PSD.

        Parameters
        ----------
        frequencies_hz : ArrayLike
            Frequencies in Hz at which to evaluate the PSD. Negative entries are folded to their
            absolute value to accommodate two-sided FFT frequency grids.
        platform_state : stonesoup.types.state.State
            Platform state. The free-stream speed is taken as the Euclidean norm of the velocity
            components indexed by :attr:`velocity_mapping`.
        streamwise_position_m : float
            Distance from the boundary layer origin to the sensor along the flow direction in m.
            Must be supplied and strictly positive.

        Returns
        -------
        FloatArray
            One-sided pressure PSD in Pa^2/Hz, shape ``(num_frequencies,)``.

        Raises
        ------
        ValueError
            If ``streamwise_position_m`` is ``None`` or non-positive, or if the speed derived from
            ``platform_state`` is non-positive.

        Warns
        -----
        UserWarning
            If the local Reynolds number ``Re_x`` is below 5e5, indicating a likely laminar or
            transitional boundary layer outside the validity of the turbulent flat-plate
            correlation.

        """
        if streamwise_position_m is None:
            msg = "streamwise_position_m is required by GoodyFlowNoiseSpectrum"
            raise ValueError(msg)
        if streamwise_position_m <= 0.0:
            msg = f"streamwise_position_m must be strictly positive, got {streamwise_position_m}"
            raise ValueError(msg)

        speed_mps = self._speed_from_state(platform_state)
        if speed_mps <= 0.0:
            msg = f"Speed derived from platform_state must be strictly positive, got {speed_mps}"
            raise ValueError(msg)

        frequencies = np.abs(np.asarray(frequencies_hz, dtype=float))
        omega = 2.0 * np.pi * frequencies

        delta_star, tau_w = self._boundary_layer_state(speed_mps, streamwise_position_m)
        r_t = (delta_star / speed_mps) * (
            tau_w / (self.fluid_density_kg_m3 * self.kinematic_viscosity_m2_s)
        )

        omega_tilde = omega * delta_star / speed_mps
        numerator = self.c2 * omega_tilde**2
        low_term = (omega_tilde**0.75 + self.c1) ** 3.7
        high_term = (self.c3 * r_t**-0.57 * omega_tilde) ** 7
        denominator = low_term + high_term

        phi_omega = (tau_w**2 * delta_star / speed_mps) * numerator / denominator
        return (2.0 * np.pi * phi_omega).astype(float)

    def _speed_from_state(self, platform_state: State) -> float:
        """Return the free-stream speed as ``|velocity|`` from the platform state."""
        state_vector = np.asarray(platform_state.state_vector, dtype=float).flatten()
        velocity = state_vector[list(self.velocity_mapping)]
        return float(np.linalg.norm(velocity))

    def _boundary_layer_state(
        self,
        speed_mps: float,
        streamwise_position_m: float,
    ) -> tuple[float, float]:
        r"""Return ``(delta_star, tau_w)`` from a turbulent flat-plate correlation.

        Uses Schlichting's 1/7-power-law results:
        :math:`C_f = 0.0576 Re_x^{-1/5}`,
        :math:`\delta = 0.37 x Re_x^{-1/5}`, and
        :math:`\delta^* = \delta / 8`.
        """
        import warnings

        re_x = speed_mps * streamwise_position_m / self.kinematic_viscosity_m2_s
        if re_x < 5.0e5:
            warnings.warn(
                (
                    f"Local Reynolds number Re_x={re_x:.3e} is below 5e5; the "
                    "turbulent flat-plate correlation used by "
                    "GoodyFlowNoiseSpectrum is not valid in this regime."
                ),
                UserWarning,
                stacklevel=3,
            )

        cf = 0.0576 * re_x**-0.2
        delta = 0.37 * streamwise_position_m * re_x**-0.2
        delta_star = delta / 8.0
        tau_w = 0.5 * self.fluid_density_kg_m3 * speed_mps**2 * cf
        return delta_star, tau_w


class JohnsonNyquistNoiseSpectrum(SensorNoiseSpectrum):
    r"""Johnson-Nyquist (thermal) noise of a resistive element.

    Models the white voltage noise produced by thermal agitation of charge carriers in a resistive
    element — typically the hydrophone's equivalent radiation resistance :math:`R_a` or a lumped
    input resistance of the preamplifier stage. The one-sided voltage PSD is frequency-independent
    (white):

    .. math::

        S_V(f) = 4 k_\mathrm{B} \, T \, R

    where :math:`k_\mathrm{B} = 1.380\,649 \times 10^{-23}` J K\ :sup:`-1` is Boltzmann's constant,
    :math:`T` is the absolute temperature of the resistive element, and :math:`R` is the effective
    resistance.

    This is a **voltage-domain** source: the PSD is returned in :math:`\text{V}^2/\text{Hz}` and is
    injected into the signal chain *after* the hydrophone transfer function. The ``platform_state``
    argument is accepted for interface compatibility but is not used; Johnson-Nyquist noise does
    not depend on platform motion.

    Notes
    -----
    When modelling a complete hydrophone/preamplifier front-end, instantiate one
    :class:`JohnsonNyquistNoiseSpectrum` per distinct resistive contributor (e.g. radiation
    resistance, bias resistor) and sum the resulting PSDs, or use
    :class:`PreamplifierNoiseSpectrum` which folds the amplifier's input-referred noise into
    a single object.

    References
    ----------
    [1] J. B. Johnson, "Thermal agitation of electricity in conductors," Phys. Rev., vol. 32,
        no. 1, pp. 97-109, Jul. 1928, doi: 10.1103/PhysRev.32.97.

    [2] H. Nyquist, "Thermal agitation of electric charge in conductors," Phys. Rev., vol. 32,
        no. 1, pp. 110-113, Jul. 1928, doi: 10.1103/PhysRev.32.110.

    """

    ...


class PreamplifierNoiseSpectrum(SensorNoiseSpectrum):
    r"""Input-referred noise of a preamplifier stage.

    Models the total input-referred voltage noise of a preamplifier (typically a low-noise
    JFET op-amp) connected to a piezoelectric hydrophone source. Two noise mechanisms are combined:

    **Voltage noise** (dominant at high impedance and high frequency):

    .. math::

        S_{V,\text{amp}}(f) = e_n^2 \left(1 + \frac{f_c}{f}\right)

    **Current noise** flowing through the source impedance (dominant at low frequency and low
    source impedance):

    .. math::

        S_{V,\text{current}}(f) = i_n^2 \, |Z_s(f)|^2

    The hydrophone source impedance is modelled as the parallel combination of the radiation
    resistance :math:`R_a` and the clamped capacitance :math:`C_0`:

    .. math::

        Z_s(f) = \frac{R_a}{1 + j \, 2\pi f \, R_a C_0}
        \implies |Z_s(f)| = \frac{R_a}{\sqrt{1 + (2\pi f R_a C_0)^2}}

    The total input-referred one-sided voltage PSD is then:

    .. math::

        S_V(f) = e_n^2 \left(1 + \frac{f_c}{f}\right)
               + i_n^2 \, |Z_s(f)|^2

    Both :math:`e_n` and :math:`i_n` are the white (flat) noise densities quoted on the amplifier
    datasheet; the :math:`1/f` flicker region is captured by the corner frequency :math:`f_c`.

    This is a **voltage-domain** source: the PSD is returned in :math:`\text{V}^2/\text{Hz}` and is
    injected into the signal chain *after* the hydrophone transfer function. The
    ``platform_state`` argument is accepted for interface compatibility but is not used.

    Parameters
    ----------
    voltage_noise_density_v_per_rthz : float
        Flat (white) input voltage noise density :math:`e_n` in V/√Hz. Must be strictly positive.
        Typical values for low-noise JFETs: 1-10 nV/√Hz.
    current_noise_density_a_per_rthz : float
        Flat (white) input current noise density :math:`i_n` in A/√Hz. Must be non-negative.
        Typical values: 1-100 fA/√Hz.
    radiation_resistance_ohm : float
        Hydrophone radiation resistance :math:`R_a` in :math:`\Omega`.  Used to compute the source
        impedance magnitude. Must be strictly positive.
    clamped_capacitance_f : float
        Hydrophone clamped (blocked) capacitance :math:`C_0` in F. Must be strictly positive.
    flicker_corner_freq_hz : float, optional
        Voltage-noise :math:`1/f` corner frequency :math:`f_c` in Hz. Below this frequency the
        voltage noise rises as :math:`1/f`. Defaults to ``100.0`` Hz, a representative value for
        JFET-input op-amps. Set to ``0.0`` to disable the flicker term entirely.

    Notes
    -----
    The model evaluates :math:`S_V(f)` at each supplied frequency independently. At
    :math:`f = 0` the :math:`1/f` flicker term diverges; any zero-frequency bin is silently
    replaced by the value at the smallest non-zero frequency in the supplied array.  If the caller
    supplies only :math:`f = 0`, a ``ValueError`` is raised.

    The source-impedance model (parallel :math:`R_a \| C_0`) is a first-order approximation valid
    well below the hydrophone resonance frequency. Near resonance the motional branch of the full
    Butterworth-Van Dyke equivalent circuit should be included.

    References
    ----------
    [1] C. D. Motchenbacher and J. A. Connelly, Low-Noise Electronic System Design. New York, NY,
        USA: Wiley-Interscience, 1993.

    [2] IEEE, "Sound System Equipment — Part 1: General," IEC 60268-1:2014, International
        Electrotechnical Commission, Geneva, Switzerland, 2014.

    [3] A. van der Ziel, Noise in Solid State Devices and Circuits. New York, NY, USA: Wiley, 1986,
        ch. 5.

    """

    ...
