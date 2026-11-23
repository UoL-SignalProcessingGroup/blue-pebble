"""Layered seabed descriptions for normal-mode propagation models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SeabedLayer:
    """One geoacoustic layer of the seabed, or the halfspace beneath them.

    Compressional speed varies linearly through a layer, from ``top_speed_m_s`` to
    ``bottom_speed_m_s``. Attenuation is in dB per metre per kHz, the units published
    for most geoacoustic models, so it scales linearly with frequency. Use
    :meth:`from_db_per_wavelength` for values given per wavelength.

    Attributes
    ----------
    thickness_m : float or None
        Layer thickness in metres. ``None`` marks the halfspace.
    top_speed_m_s, bottom_speed_m_s : float
        Compressional speed at the top and bottom of the layer, in m/s.
    density_g_cm3 : float
        Density in g/cm^3.
    attenuation_db_per_m_khz : float
        Compressional attenuation in dB/(m kHz).
    shear_speed_m_s : float
        Shear speed in m/s; 0 for a fluid layer. Shear needs KRAKENC.
    shear_attenuation_db_per_m_khz : float
        Shear attenuation in dB/(m kHz).
    name : str
        A label for printing.

    """

    thickness_m: float | None
    top_speed_m_s: float
    bottom_speed_m_s: float
    density_g_cm3: float
    attenuation_db_per_m_khz: float = 0.0
    shear_speed_m_s: float = 0.0
    shear_attenuation_db_per_m_khz: float = 0.0
    name: str = ""

    def __post_init__(self) -> None:
        """Check the layer is physical."""
        if self.thickness_m is not None and not self.thickness_m > 0:
            raise ValueError(f"Layer thickness must be positive, got {self.thickness_m}")
        if min(self.top_speed_m_s, self.bottom_speed_m_s) <= 0 or self.density_g_cm3 <= 0:
            raise ValueError("Layer speeds and density must be positive")
        if self.attenuation_db_per_m_khz < 0 or self.shear_attenuation_db_per_m_khz < 0:
            raise ValueError("Attenuation must be non-negative")
        if self.shear_speed_m_s < 0:
            raise ValueError("Shear speed must be non-negative")

    @classmethod
    def from_db_per_wavelength(
        cls,
        thickness_m: float | None,
        top_speed_m_s: float,
        bottom_speed_m_s: float,
        density_g_cm3: float,
        attenuation_db_per_wavelength: float,
        **kwargs: Any,
    ) -> SeabedLayer:
        """Build a layer from an attenuation in dB per wavelength.

        The conversion uses the speed at the top of the layer: ``dB/(m kHz) =
        dB/wavelength * 1000 / c``. It does not depend on frequency.

        Parameters
        ----------
        thickness_m, top_speed_m_s, bottom_speed_m_s, density_g_cm3 : float
            As for the class.
        attenuation_db_per_wavelength : float
            Compressional attenuation in dB per wavelength.
        **kwargs
            Other fields of the class.

        Returns
        -------
        SeabedLayer
            The layer.

        """
        return cls(
            thickness_m=thickness_m,
            top_speed_m_s=top_speed_m_s,
            bottom_speed_m_s=bottom_speed_m_s,
            density_g_cm3=density_g_cm3,
            attenuation_db_per_m_khz=attenuation_db_per_wavelength * 1000.0 / top_speed_m_s,
            **kwargs,
        )

    @property
    def attenuation_db_per_wavelength(self) -> float:
        """Compressional attenuation at the top of the layer, in dB per wavelength."""
        return self.attenuation_db_per_m_khz * self.top_speed_m_s / 1000.0


@dataclass(frozen=True)
class LayeredSeabed:
    """Sediment layers over a halfspace, for a range-independent waveguide.

    Attributes
    ----------
    halfspace : SeabedLayer
        The semi-infinite bottom. Only its top speed, density and attenuations are used.
    layers : tuple of SeabedLayer
        Layers from the seabed downward, each with a thickness. Empty for a single
        halfspace (a Pekeris bottom).

    """

    halfspace: SeabedLayer
    layers: tuple[SeabedLayer, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Check every layer but the halfspace has a thickness."""
        object.__setattr__(self, "layers", tuple(self.layers))
        if any(layer.thickness_m is None for layer in self.layers):
            raise ValueError("Every seabed layer above the halfspace needs a thickness")

    @property
    def total_thickness_m(self) -> float:
        """Combined thickness of the layers above the halfspace, in metres."""
        return float(sum(float(layer.thickness_m or 0.0) for layer in self.layers))

    @property
    def is_elastic(self) -> bool:
        """Whether any layer or the halfspace supports shear."""
        return any(layer.shear_speed_m_s > 0 for layer in (*self.layers, self.halfspace))

    @property
    def max_speed_m_s(self) -> float:
        """Fastest compressional speed in the seabed, in m/s."""
        return max(
            max(layer.top_speed_m_s, layer.bottom_speed_m_s)
            for layer in (*self.layers, self.halfspace)
        )
