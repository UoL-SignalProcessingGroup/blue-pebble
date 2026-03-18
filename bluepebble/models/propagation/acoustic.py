"""Defines acoustic propagation models for simulating sound propagation."""

import subprocess
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from shutil import which
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from stonesoup.base import Base, Property

from ...utils import read_shade_file
from ..environment import Bathymetry, SoundSpeedProfile

if TYPE_CHECKING:
    from stonesoup.platform.base import Platform
    from stonesoup.types.state import State

FloatArray: TypeAlias = NDArray[np.float64]
ComplexArray: TypeAlias = NDArray[np.complexfloating[Any, Any]]
PropagationLoss: TypeAlias = float | FloatArray
PropagationResult: TypeAlias = tuple[PropagationLoss, float]
SpectrumResult: TypeAlias = tuple[ComplexArray, float]


def _as_scalar_float(value: object, name: str) -> float:
    """Coerce a scalar-like numeric value to ``float``.

    Parameters
    ----------
    value : object
        Numeric scalar or size-1 array-like value.
    name : str
        Human-readable parameter name for validation errors.

    Returns
    -------
    float
        Scalar floating-point value.

    Raises
    ------
    ValueError
        If ``value`` cannot be interpreted as a scalar.

    """
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        return float(array)
    if array.size == 1:
        return float(array.reshape(-1)[0])
    raise ValueError(f"{name} must be scalar-like")


class _RtrsRunSimulation(Protocol):
    """Callable protocol for ``rtrs.run_simulation``."""

    def __call__(self, env_config: Mapping[str, object]) -> dict[str, Any]:
        """Run a single rtrs simulation."""
        ...


def _get_source_metadata(source: "State") -> Mapping[str, object]:
    """Return validated source metadata mapping."""
    metadata = getattr(source, "metadata", None)
    if not isinstance(metadata, Mapping):
        msg = "Source state metadata must be mapping-like"
        raise ValueError(msg)
    return metadata


def _get_source_position(source: "State") -> FloatArray:
    """Extract source position vector using ``position_mapping`` metadata."""
    metadata = _get_source_metadata(source)
    position_mapping = metadata.get("position_mapping")
    if position_mapping is None:
        msg = "Source metadata missing required key: position_mapping"
        raise ValueError(msg)

    mapping_array = np.asarray(position_mapping, dtype=int)
    if mapping_array.ndim != 1 or mapping_array.size == 0:
        msg = "Source position_mapping must be one-dimensional and non-empty"
        raise ValueError(msg)

    return np.asarray(source.state_vector[mapping_array.tolist()], dtype=float)


def _get_source_tonal_arrays(source: "State") -> tuple[FloatArray, FloatArray]:
    """Extract validated source tonal frequencies and amplitudes."""
    metadata = _get_source_metadata(source)
    required_keys = ("frequencies_hz", "amplitudes_upa")
    missing_keys = [key for key in required_keys if key not in metadata]
    if missing_keys:
        missing = ", ".join(missing_keys)
        msg = f"Source metadata missing required keys: {missing}"
        raise ValueError(msg)

    frequencies_hz = np.asarray(metadata["frequencies_hz"], dtype=float).reshape(-1)
    amplitudes_upa = np.asarray(metadata["amplitudes_upa"], dtype=float).reshape(-1)

    if frequencies_hz.size == 0:
        msg = "Source frequencies_hz metadata must be non-empty"
        raise ValueError(msg)
    if amplitudes_upa.size != frequencies_hz.size:
        msg = "Source frequencies_hz and amplitudes_upa metadata must have matching lengths"
        raise ValueError(msg)

    return cast(FloatArray, frequencies_hz), cast(FloatArray, amplitudes_upa)


def _get_rtrs_run_simulation() -> _RtrsRunSimulation:
    """Import and validate ``rtrs.run_simulation`` callable."""
    try:
        import rtrs  # type: ignore
    except ImportError:
        raise ImportError(
            "rtrs package is not installed. Please install it to use rtrsAcousticPropagationModel."
        ) from None

    run_simulation = getattr(rtrs, "run_simulation", None)
    if not callable(run_simulation):
        raise AttributeError("rtrs module does not define callable run_simulation")
    return cast(_RtrsRunSimulation, run_simulation)


class AcousticPropagationModel(ABC, Base):
    """An abstract base class for all acoustic propagation models.

    It defines a common interface and implements shared functionality.
    """

    ssp: SoundSpeedProfile = Property(doc="Sound speed profile")

    @abstractmethod
    def propagate(self, platform: "Platform", source: "State") -> PropagationResult:
        """Propagate a signal from a source to a platform.

        Notes
        -----
        Subclasses must implement this method.

        """
        ...

    def compute_sensor_delays(self, platform: "Platform", source: "State") -> FloatArray:
        """Compute time delays for each sensor in an array.

        The method calculates time-differences-of-arrival (TDOA) relative to the array reference
        sensor and accounts for sound speed at each sensor's depth.

        Parameters
        ----------
        platform : Platform
            Platform object representing the sensor array.
        source : State
            Source (State) object representing the acoustic point source.

        Returns
        -------
        FloatArray
            One-dimensional array of per-sensor delays in seconds.

        """
        source_position = _get_source_position(source)
        array_position = platform.array.state_vector
        array_ref_position = platform.array.ref_state_vector

        # Calculate distance from each sensor to the source
        distances = np.linalg.norm(
            source_position - array_position,
            axis=0,
        )

        # Calculate distance from the reference sensor to the source
        reference_distance = np.linalg.norm(source_position - array_ref_position)

        # Calculate speed of sound at the depth of each sensor
        speeds = np.asarray(self.ssp.calculate(array_position[2, :]), dtype=float)

        # Calculate time delays
        delays = (distances - reference_distance) / speeds
        return np.asarray(delays, dtype=float)


class SpectrumPropagationModel(ABC):
    """Mixin ABC for propagation models that support frequency-domain transfer functions.

    Inherit from this alongside :class:`AcousticPropagationModel` to declare that a model
    implements :meth:`propagate_spectrum`.  Simulators use ``isinstance`` against this class
    to discover the capability at construction time rather than relying on ``hasattr``.
    """

    @abstractmethod
    def propagate_spectrum(
        self,
        platform: "Platform",
        source: "State",
        frequencies_hz: ArrayLike,
    ) -> SpectrumResult:
        """Return per-sensor complex transfer functions and propagation time.

        Parameters
        ----------
        platform : Platform
            Platform state providing sensor array geometry.
        source : State
            Source state providing position and metadata.
        frequencies_hz : ArrayLike
            Frequency axis in Hz matching the desired FFT bins.

        Returns
        -------
        SpectrumResult
            Tuple ``(H_sensors, propagation_time_s)`` where ``H_sensors`` has
            shape ``(num_sensors, num_frequencies)``.

        """
        ...


class CylindricalAcousticPropagationModel(AcousticPropagationModel, SpectrumPropagationModel):
    """A simple acoustic model based on cylindrical spreading and absorption loss.

    This model provides a basic estimate of transmission loss without the computational overhead of
    more complex models like Bellhop.

    Attributes
    ----------
    attenuation_factor : float
        The absorption loss factor in dB/km.
    ssp : SoundSpeedProfile
        An instance of a sound speed profile.

    """

    attenuation_factor: float = Property(default=0.5, doc="The absorption loss factor in dB/km")

    def __post_init__(self) -> None:
        """Validate the attenuation factor after initialisation."""
        if self.attenuation_factor < 0:
            raise ValueError("Attenuation factor must be non-negative.")

    def propagate(self, platform: "Platform", source: "State") -> tuple[float, float]:
        """Propagate a signal using a cylindrical spreading loss model.

        The model combines cylindrical spreading (10*log10(r)) with a frequency-independent
        absorption term.

        Parameters
        ----------
        platform : Platform
            Platform object representing the sensor array.
        source : State
            Source (State) object representing the acoustic point source.

        Returns
        -------
        PropagationResult
            - ``tloss`` (float): Transmission loss in decibels (dB).
            - ``time`` (float): Direct-path travel time in seconds.

        """
        source_position = _get_source_position(source)
        array_ref_position = platform.array.ref_state_vector

        distance = float(np.linalg.norm(source_position - array_ref_position))
        speed = _as_scalar_float(self.ssp.calculate(array_ref_position[2]), "sound speed")
        time = distance / speed
        tloss = 10.0 * np.log10(distance) + self.attenuation_factor * (distance / 1000.0)
        return float(tloss), float(time)

    def propagate_spectrum(
        self, platform: "Platform", source: "State", frequencies_hz: ArrayLike
    ) -> SpectrumResult:
        """Propagate spectrum using cylindrical spreading.

        Calculates complex transfer functions H(f) for each sensor and frequency, accounting for
        cylindrical spreading and frequency-dependent absorption.

        Parameters
        ----------
        platform : Platform
            Platform object representing the sensor array.
        source : State
            Source (State) object representing the acoustic point source.
        frequencies_hz : ArrayLike
            Array of frequencies in Hz for which to compute transfer functions.

        Returns
        -------
        SpectrumResult
            ``(H_sensors, propagation_time_s)``, where ``H_sensors`` is a complex
            transfer-function array with shape ``(num_sensors, num_frequencies)``
            and ``propagation_time_s`` is the propagation time from the source to
            the reference sensor in seconds.

        """
        source_position = _get_source_position(source)
        array_position = platform.array.state_vector
        array_ref_position = platform.array.ref_state_vector

        # Calculate distances for each sensor. Shape: (num_sensors,)
        distances = np.linalg.norm(source_position - array_position, axis=0)

        # Calculate reference distance and propagation time
        reference_distance = np.linalg.norm(source_position - array_ref_position)
        speed = _as_scalar_float(self.ssp.calculate(array_ref_position[2]), "sound speed")
        propagation_time_s = float(reference_distance / speed)

        # Calculate cylindrical spreading loss: TL = 10*log10(r) + alpha*r
        # where alpha is attenuation in dB/km
        spreading_loss_db = 10 * np.log10(distances)  # Shape: (num_sensors,)
        absorption_loss_db = self.attenuation_factor * (distances / 1000.0)
        total_loss_db = spreading_loss_db + absorption_loss_db  # Shape: (num_sensors,)

        # Convert to linear amplitude scaling
        amplitude_scaling = 10 ** (-total_loss_db / 20.0)  # Shape: (num_sensors,)

        # Calculate phase shift for each sensor and frequency
        # Phase = 2pi * f * (d / c)
        speeds_per_sensor = self.ssp.calculate(array_position[2, :])  # Shape: (num_sensors,)
        time_delays = distances / speeds_per_sensor  # Shape: (num_sensors,)

        # Broadcast to (num_sensors, num_frequencies)
        # Delay convention follows FFT forward sign: delay tau => exp(-j2pi f tau)
        frequencies = np.asarray(frequencies_hz, dtype=float)
        phase_shifts = np.exp(
            -2j * np.pi * frequencies[np.newaxis, :] * time_delays[:, np.newaxis]
        )

        # Combine amplitude and phase
        # H(f) = amplitude * exp(-j*2pi*f*t)
        H_sensors = np.asarray(
            amplitude_scaling[:, np.newaxis] * phase_shifts, dtype=np.complex128
        )

        return H_sensors, float(propagation_time_s)


class SphericalAcousticPropagationModel(AcousticPropagationModel, SpectrumPropagationModel):
    """A simple acoustic model based on spherical spreading and absorption loss.

    This model provides a basic estimate of transmission loss without the computational overhead of
    more complex models like Bellhop.

    Attributes
    ----------
    attenuation_factor : float
        The absorption loss factor in dB/km.
    ssp : SoundSpeedProfile
        An instance of a sound speed profile.

    """

    attenuation_factor: float = Property(default=0.001, doc="The absorption loss factor in dB/km")

    def __post_init__(self) -> None:
        """Validate the attenuation factor after initialization."""
        if self.attenuation_factor < 0:
            raise ValueError("Attenuation factor must be non-negative.")

    def propagate(self, platform: "Platform", source: "State") -> tuple[float, float]:
        """Propagate a signal using a spherical spreading loss model.

        The model combines spherical spreading (20*log10(r)) with a frequency-independent
        absorption term.

        Parameters
        ----------
        platform : Platform
            Platform object representing the sensor array.
        source : State
            Source (State) object representing the acoustic point source.

        Returns
        -------
        PropagationResult
            - ``tloss`` (float): Transmission loss in decibels (dB).
            - ``time`` (float): Direct-path travel time in seconds.

        """
        source_position = _get_source_position(source)
        array_ref_position = platform.array.ref_state_vector

        distance = float(np.linalg.norm(source_position - array_ref_position))
        speed = _as_scalar_float(self.ssp.calculate(array_ref_position[2]), "sound speed")
        time = distance / speed
        tloss = 20.0 * np.log10(distance) + self.attenuation_factor * (distance / 1000.0)
        return float(tloss), float(time)

    def propagate_spectrum(
        self, platform: "Platform", source: "State", frequencies_hz: ArrayLike
    ) -> SpectrumResult:
        """Propagate spectrum using spherical spreading.

        This method calculates transfer functions H(f) for each sensor and frequency, accounting
        for spherical spreading and frequency-dependent absorption.

        Parameters
        ----------
        platform : Platform
            Platform object representing the sensor array.
        source : State
            Source (State) object representing the acoustic point source.
        frequencies_hz : ArrayLike
            Array of frequencies in Hz for which to compute transfer functions.

        Returns
        -------
        SpectrumResult
            ``(H_sensors, propagation_time_s)``, where ``H_sensors`` is a complex
            transfer-function array with shape ``(num_sensors, num_frequencies)``
            and ``propagation_time_s`` is the propagation time from the source to
            the reference sensor in seconds.

        """
        source_position = _get_source_position(source)
        array_position = platform.array.state_vector
        array_ref_position = platform.array.ref_state_vector

        # Calculate distances for each sensor
        distances = np.linalg.norm(
            source_position - array_position,
            axis=0,
        )  # Shape: (num_sensors,)

        # Calculate reference distance and propagation time
        reference_distance = np.linalg.norm(source_position - array_ref_position)
        speed = _as_scalar_float(self.ssp.calculate(array_ref_position[2]), "sound speed")
        propagation_time_s = float(reference_distance / speed)

        # Calculate spherical spreading loss: TL = 20*log10(r) + alpha*r
        # where alpha is attenuation in dB/km
        spreading_loss_db = 20 * np.log10(distances)  # Shape: (num_sensors,)
        absorption_loss_db = self.attenuation_factor * (distances / 1000.0)
        total_loss_db = spreading_loss_db + absorption_loss_db  # Shape: (num_sensors,)

        # Convert to linear amplitude scaling
        amplitude_scaling = 10 ** (-total_loss_db / 20.0)  # Shape: (num_sensors,)

        # Calculate phase shift for each sensor and frequency
        # Phase = 2pi * f * (d / c)
        speeds_per_sensor = self.ssp.calculate(array_position[2, :])  # Shape: (num_sensors,)
        time_delays = distances / speeds_per_sensor  # Shape: (num_sensors,)

        # Broadcast to (num_sensors, num_frequencies)
        # Delay convention follows FFT forward sign: delay tau => exp(-j2pi f tau)
        frequencies = np.asarray(frequencies_hz, dtype=float)
        phase_shifts = np.exp(
            -2j * np.pi * frequencies[np.newaxis, :] * time_delays[:, np.newaxis]
        )

        # Combine amplitude and phase
        # H(f) = amplitude * exp(-j*2pi*f*t)
        H_sensors = np.asarray(
            amplitude_scaling[:, np.newaxis] * phase_shifts, dtype=np.complex128
        )

        return H_sensors, float(propagation_time_s)


class BellhopAcousticPropagationModel(AcousticPropagationModel):
    """Representation of a Bellhop acoustic propagation model.

    This model calls an external Bellhop executable to perform propagation simulations. The Bellhop
    binary must be installed and available on the system or provided via ``exe_path``.

    Attributes
    ----------
    env_depth : float
        The depth of the environment in meters.
    ssp : SoundSpeedProfile
        An instance of a sound speed profile.
    exe_path : str | Path
        The path to the Bellhop executable (defaults to ``'bellhopcxx'``).

    """

    env_depth: float = Property(doc="The depth of the environment in meters")
    exe_path: str = Property(
        default="bellhopcxx",
        doc="The path to Bellhop executable, defaults to 'bellhopcxx'",
    )

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialise and validate Bellhop executable availability."""
        super().__init__(*args, **kwargs)
        self.exe_path = self._resolve_bellhop_executable(self.exe_path)

    @staticmethod
    def _resolve_bellhop_executable(exe_path: str | Path) -> str:
        """Resolve Bellhop executable path and raise if unavailable."""
        exe_name = str(exe_path)
        bellhop_path = which(exe_name)
        if bellhop_path is None:
            raise FileNotFoundError(
                f"Bellhop executable '{exe_name}' not found. Ensure it is installed and in "
                "your system's PATH, or provide the full path via the 'exe_path' property."
            )
        return bellhop_path

    def propagate(self, platform: "Platform", source: "State") -> tuple[float, float]:
        """Run a Bellhop simulation for a single source and receiver.

        The method writes a Bellhop environment file, executes the Bellhop binary, reads the
        resulting shade file and computes transmission loss and travel time.

        Parameters
        ----------
        platform : Platform
            Platform object representing the sensor array.
        source : State
            Source (State) object representing the acoustic point source.

        Returns
        -------
        tuple
            - ``tloss`` (float): Transmission loss in decibels (dB).
            - ``time`` (float): Direct-path travel time in seconds.

        Raises
        ------
        subprocess.CalledProcessError
            If the external Bellhop executable returns a non-zero exit code.

        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            self._create_env_file(platform, source, output_dir=temp_path)

            env_file_path = temp_path / "env"

            try:
                subprocess.run(
                    [self.exe_path, str(env_file_path)],
                    capture_output=True,
                    check=True,
                    text=True,  # Decode stdout/stderr as text
                )
            except subprocess.CalledProcessError as e:
                print("--- Bellhop Execution Failed ---")
                print(f"Bellhop returned with exit code: {e.returncode}")
                print("\n--- Bellhop's Standard Output ---")
                print(e.stdout)
                print("\n--- Bellhop's Error Messages (STDERR) ---")
                print(e.stderr)
                print("\nCheck the error messages above for clues from Bellhop.")
                print(f"The input file that caused the error was: '{env_file_path}'")
                # Re-raise the exception so the program still stops
                raise

            shd_path = temp_path / "env.shd"
            pressure, _ = read_shade_file(shd_path)

        np.seterr(divide="ignore")
        pressure_array = np.asarray(pressure)
        pressure = np.squeeze(pressure_array)
        pressure_magnitude = np.abs(pressure)
        tloss = -20 * np.log10(np.maximum(pressure_magnitude, 1e-12))

        distance = float(
            np.linalg.norm(source.state_vector[[0, 2, 4]] - platform.array.ref_state_vector)
        )
        speed = float(np.asarray(self.ssp.calculate(platform.array.ref_state_vector[2])).item())
        time = float(distance / speed)

        # Handle cases where pressure is effectively zero by returning a large finite loss.
        if np.atleast_1d(pressure_magnitude)[-1] < 1e-12:
            return 999.0, float(time)  # Return a large, finite loss value

        return float(np.atleast_1d(tloss)[-1]), float(time)

    def _create_env_file(
        self,
        platform: "Platform",
        source: "State",
        output_dir: Path = Path("."),
        options: str = "SVW",
        bottom_bc: str = "A",
        runtype: str = "Cb",
        nbeams: int = 0,
        beam_angles: list[float] | None = None,
    ) -> None:
        """Create the Bellhop environment file (.env) from a template.

        The method collects simulation parameters, formats them according to Bellhop's input
        specification and writes the environment file.

        Parameters
        ----------
        platform
            Platform object containing array/state information.
        source
            Source object containing its state and metadata.
        output_dir : Path, optional
            Directory to save the generated file (default current directory).
        options : str, optional
            Bellhop option string (default ``"SVW"``).
        bottom_bc : str, optional
            Bottom boundary condition code (default ``"A"``).
        runtype : str, optional
            Bellhop run type code (default ``"Cb"``).
        nbeams : int, optional
            Number of beams for the simulation (0 lets Bellhop choose).
        beam_angles : list[float], optional
            Minimum and maximum beam launch angles in degrees
            (default ``[-89.0, 89.0]``).

        """
        if beam_angles is None:
            beam_angles = [-89.0, 89.0]

        # 1. Calculate All Required Values
        # ==================================
        title = "'env'"

        depth = np.arange(0, self.env_depth + 1, 100)
        sound_speed = self.ssp.calculate(depth)
        ssp = np.column_stack((depth, sound_speed))

        source_position = _get_source_position(source)
        array_ref_position = platform.array.ref_state_vector

        frequencies_hz, amplitudes_upa = _get_source_tonal_arrays(source)
        frequency = frequencies_hz[int(np.argmax(amplitudes_upa))]
        max_range_m = np.linalg.norm(source_position - array_ref_position)

        # Source and receiver depths
        source_depth = float(np.abs(source_position[2]).item())
        receiver_depth = float(np.abs(array_ref_position[2]).item())

        # Define bathymetry and check if a .bty file is needed
        bathy = [[0, self.env_depth]]  # Simple flat bottom
        bottom_bc_final = "A~" if len(bathy) > 2 else bottom_bc
        if len(bathy) > 2:
            self._write_bathy_file(bathy, "env", output_dir)

        # Prepare Sound Speed Profile (SSP) string
        ssp_header = f"{ssp[0, 0]:.1f} {ssp[0, 1]:.1f}  /\n"
        ssp_body = "\n".join([f"{z:.1f} {c:.1f}  /" for z, c in ssp[1:]])
        ssp_string = ssp_header + ssp_body

        # Bottom half-space properties
        bottom_props_str = "1700.0 0.0 1.5 0.5"

        # 2. Define The File Template
        # ============================
        env_template = f"""
            {title}
            {frequency:.1f}
            1
            '{options}'
            0 0 {self.env_depth:.1f}
            {ssp_string}
            '{bottom_bc_final}' 0.0
            {self.env_depth:.1f} {bottom_props_str} /
            1
            {source_depth:.1f} /
            1
            {receiver_depth:.1f} /
            {int(max_range_m) + 1}
            0.0 {max_range_m / 1000.0} /
            '{runtype}'
            {nbeams}
            {beam_angles[0]} {beam_angles[1]} /
            0.0 {self.env_depth * 1.2:.1f} {(max_range_m / 1000.0) * 1.1:.2f}
        """.strip()

        # 3. Write The File
        # =================
        filename_env = output_dir / "env.env"
        with open(filename_env, "w") as file:
            file.write(env_template)

    def _write_bathy_file(
        self,
        bathy: list[list[float]],
        filename: str,
        output_dir: Path,
    ) -> None:
        """Write bathymetry file for complex bottom topography."""
        # This method would write a .bty file for complex bathymetry
        # For now, it's a placeholder since we use simple flat bottom
        pass


class rtrsAcousticPropagationModel(AcousticPropagationModel, SpectrumPropagationModel):
    """Representation of an rtrs acoustic propagation model.

    Uses the rtrs Python bindings for 3D ray-tracing with support for 3D SSP and 2D bathymetry.

    Attributes
    ----------
    ssp : SoundSpeedProfile
        An instance of a sound speed profile.
    bathymetry : Bathymetry
        An instance of a bathymetry model.
    step_m : float
        Ray tracing step size in meters (default 15.0).
    ssp_resolution : tuple
        Resolution for SSP grid (x, y, z) in meters.
    azimuth_search_width : float
        Angular width in degrees to search for azimuth angles that will hit
        the receiver.
    azimuth_resolution : float
        Angular resolution for azimuth search in degrees.
    elevation_range : tuple
        Minimum and maximum elevation angles in degrees.
    elevation_resolution : float
        Angular resolution for elevation in degrees.
    use_all_frequencies : bool
        If True, run rtrs for all tonal frequencies and return per-frequency
        TL values. If False, use only the loudest frequency.
    water_density_g_cm3 : float | None
        Optional water density value passed to rtrs bathymetry config.
    bottom_model : dict
        Bottom boundary model configuration for rtrs (e.g., rigid/acoustic/elastic).
    store_ray_paths : bool
        Whether rtrs should store full ray paths.
    integration_method : str
        Beam integration method for rtrs ("euler" or "rk2").

    """

    bathymetry: Bathymetry = Property(doc="Bathymetry model")
    step_m: float = Property(default=15.0, doc="Ray tracing step size in meters")
    ssp_resolution: tuple[float, float, float] = Property(
        default=(5000.0, 5000.0, 100.0),
        doc="Resolution for SSP grid (x, y, z) in meters",
    )
    azimuth_search_width: float = Property(
        default=1.0,
        doc="Angular width in degrees to search for azimuth angles",
    )
    azimuth_resolution: float = Property(
        default=0.5, doc="Angular resolution for azimuth search in degrees"
    )
    elevation_range: tuple[float, float] = Property(
        default=(-70.0, 70.0), doc="Min and max elevation angles in degrees"
    )
    elevation_resolution: float = Property(
        default=1.0, doc="Angular resolution for elevation in degrees"
    )
    use_all_frequencies: bool = Property(
        default=False,
        doc="If True, run rtrs for all tonal frequencies. If False, use only the "
        "loudest frequency. Not used for propagated spectrum method.",
    )
    water_density_g_cm3: float | None = Property(
        default=None,
        doc="Optional water density passed to rtrs bathymetry config (g/cm^3)",
    )
    bottom_model: dict[str, object] | None = Property(
        default=None,
        doc="Bottom boundary model dictionary for rtrs",
    )
    store_ray_paths: bool = Property(
        default=False,
        doc="If True, store full ray paths in rtrs output, not needed for Blue Pebble",
    )
    integration_method: str = Property(
        default="euler",
        doc='Beam integration method for rtrs ("euler" or "rk2")',
    )

    def __post_init__(self) -> None:
        """Validate rtrs-specific configuration options."""
        if self.integration_method not in {"euler", "rk2"}:
            raise ValueError("integration_method must be 'euler' or 'rk2'.")
        if self.bottom_model is not None and (
            not isinstance(self.bottom_model, dict) or "model" not in self.bottom_model
        ):
            raise ValueError("bottom_model must be a dict containing at least a 'model' key.")

    def _resolved_bottom_model(self) -> dict[str, object]:
        """Return bottom model config with a safe default."""
        if self.bottom_model is None:
            return {"model": "rigid"}
        return dict(self.bottom_model)

    def _calculate_launch_azimuths(
        self,
        source_position: ArrayLike,
        receiver_position: ArrayLike,
    ) -> list[float]:
        """Calculate launch azimuth angles to ensure rays cross the receiver.

        Parameters
        ----------
        source_position : array-like
            3D position vector of the source.
        receiver_position : array-like
            3D position vector of the receiver.

        Returns
        -------
        list
            Azimuth angles in degrees.

        """
        # Calculate the direct azimuth to the receiver
        source_pos = np.asarray(source_position, dtype=float)
        receiver_pos = np.asarray(receiver_position, dtype=float)
        dx = receiver_pos[0] - source_pos[0]
        dy = receiver_pos[1] - source_pos[1]

        # Calculate azimuth in degrees (0° = +x axis, 90° = +y axis)
        direct_azimuth = np.degrees(np.arctan2(dy, dx))

        # Generate azimuth angles around the direct path
        half_width = self.azimuth_search_width / 2.0
        num_angles = int(self.azimuth_search_width / self.azimuth_resolution) + 1

        azimuths = np.linspace(
            direct_azimuth - half_width, direct_azimuth + half_width, num_angles
        )

        azimuths = -azimuths + 90

        return azimuths.tolist()

    def _calculate_max_steps_and_range(self, distance: float) -> tuple[int, float]:
        """Calculate max_steps and max_range based on source-receiver distance.

        Parameters
        ----------
        distance : float
            Distance from source to receiver in meters.

        Returns
        -------
        tuple
            (max_steps, max_range_m).

        """
        # Add 20% margin to the distance
        max_range_m = distance * 1.2

        # Calculate max_steps based on step size
        max_steps = int(max_range_m / self.step_m) + 1000  # Add buffer

        return max_steps, max_range_m

    def propagate(self, platform: "Platform", source: "State") -> PropagationResult:
        """Run an rtrs simulation for a single source and receiver.

        The method prepares the rtrs environment, runs the ray-tracing simulation and returns
        transmission loss and travel time.

        Parameters
        ----------
        platform
            Object representing the sensor platform.
        source
            Object representing the acoustic source.

        Returns
        -------
        PropagationResult
            - ``tloss`` (float or FloatArray): Transmission loss in dB. If
              ``use_all_frequencies`` is ``True``, returns one value per frequency.
            - ``time`` (float): Direct-path travel time in seconds.

        """
        run_simulation = _get_rtrs_run_simulation()
        source_position = _get_source_position(source)
        array_ref_position = platform.array.ref_state_vector

        # Flatten to 1D arrays for easier indexing
        source_pos = source_position.flatten()
        array_pos = array_ref_position.flatten()

        # Calculate distance and dynamic parameters
        distance = float(np.linalg.norm(source_pos - array_pos))
        max_steps, max_range_m = self._calculate_max_steps_and_range(distance)

        # Calculate launch azimuths
        launch_azimuths = self._calculate_launch_azimuths(source_pos, array_pos)

        # Calculate launch elevations
        num_elev = (
            int((self.elevation_range[1] - self.elevation_range[0]) / self.elevation_resolution)
            + 1
        )
        launch_elevations = np.linspace(
            self.elevation_range[0], self.elevation_range[1], num_elev
        ).tolist()

        # print(f"Launch azimuths: {launch_azimuths}")
        # print(f"Number of elevations: {len(launch_elevations)}")
        # print(f"Number of azimuths: {len(launch_azimuths)}")
        # print(f"Number of rays: {len(launch_elevations) * len(launch_azimuths)}")
        # print(f"Source position: {source_pos}")
        # print(f"Array position: {array_pos}")
        # print(f"Distance (m): {distance}")
        # print(f"Max steps: {max_steps}, Max range (m): {max_range_m}")

        # Determine spatial extent for grids
        x_coords = [source_pos[0], array_pos[0]]
        y_coords = [source_pos[1], array_pos[1]]
        z_coords = [source_pos[2], array_pos[2]]

        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        z_min, z_max = min(z_coords), max(z_coords)

        # Add margins to spatial extent
        margin = 0.1  # 10% margin
        x_range_width = max(x_max - x_min, 1000.0)  # Minimum 1km width
        y_range_width = max(y_max - y_min, 1000.0)
        z_range_depth = abs(z_max - z_min)

        x_margin = x_range_width * margin
        y_margin = y_range_width * margin
        z_margin = max(z_range_depth * margin, 500.0)  # Minimum 500m margin

        x_range = (x_min - x_margin, x_max + x_margin)
        y_range = (y_min - y_margin, y_max + y_margin)
        z_range = (
            z_min - z_margin,
            min(z_max + z_margin, 0.0),
        )  # Don't go above surface

        # Get bathymetry depth at receiver for lower bound
        bathy_depth_at_receiver = self.bathymetry.get_depth(array_pos[0], array_pos[1])
        z_range = (max(z_range[0], -bathy_depth_at_receiver), z_range[1])

        # Generate 3D SSP grid
        x_ssp, y_ssp, z_ssp, c_ssp = self.ssp.get_3d_grid(
            x_range,
            y_range,
            z_range,
            self.ssp_resolution[0],
            self.ssp_resolution[1],
            self.ssp_resolution[2],
        )

        # Generate 2D bathymetry grid
        x_bty, y_bty, z_bty = self.bathymetry.get_grid(x_range, y_range)
        z_bty_flat = z_bty.flatten(order="C")

        # Get frequency/frequencies from source
        source_frequencies_hz, source_amplitudes_upa = _get_source_tonal_arrays(source)
        if self.use_all_frequencies:
            frequencies = source_frequencies_hz.tolist()
        else:
            # Use only the peak frequency
            frequency = source_frequencies_hz[int(np.argmax(source_amplitudes_upa))]
            frequencies = [float(frequency)]

        # Build rtrs environment configuration
        env_config = {
            "ssp": {
                "x_ssp_m": x_ssp.tolist(),
                "y_ssp_m": y_ssp.tolist(),
                "z_ssp_m": z_ssp.tolist(),
                "c_m_s": c_ssp.tolist(),
            },
            "bathymetry": {
                "x_bty_m": x_bty.tolist(),
                "y_bty_m": y_bty.tolist(),
                "z_bty_m": z_bty_flat.tolist(),
                "bottom_model": self._resolved_bottom_model(),
            },
            "source": {
                "position": [
                    float(source_pos[0]),
                    float(source_pos[1]),
                    float(-source_pos[2]),
                ],
                "freq_hz": frequencies,
                "launch_elev_deg": launch_elevations,
                "launch_azim_deg": launch_azimuths,
            },
            "receivers": {
                "config_type": "array",
                "x_rcvr_m": [float(array_pos[0])],
                "y_rcvr_m": [float(array_pos[1])],
                "z_rcvr_m": [-float(array_pos[2])],
            },
            "beam": {
                "step_m": float(self.step_m),
                "max_steps": int(max_steps),
                "max_range_m": float(max_range_m),
                "store_ray_paths": bool(self.store_ray_paths),
                "integration_method": self.integration_method,
            },
        }

        if self.water_density_g_cm3 is not None:
            env_config["bathymetry"]["water_density_g_cm3"] = float(self.water_density_g_cm3)

        # Run rtrs simulation
        result = run_simulation(env_config)

        # Extract pressure field
        pf = result["pressure_field"]
        shape = tuple(pf["shape"])  # (nfreq, nreceivers, 1, 1)

        # Reconstruct complex pressure
        re = np.array(pf["pressure_re"], dtype=np.float32).reshape(shape)
        im = np.array(pf["pressure_im"], dtype=np.float32).reshape(shape)
        pressure = re + 1j * im

        # Calculate transmission loss per frequency
        if self.use_all_frequencies:
            # Return TL for each frequency
            tloss_per_freq = []
            for freq_idx in range(len(frequencies)):
                pressure_magnitude = np.abs(pressure[freq_idx, 0, 0, 0])

                # Handle zero pressure case
                if pressure_magnitude < 1e-12:
                    tloss_per_freq.append(999.0)
                else:
                    tloss_per_freq.append(-20 * np.log10(pressure_magnitude))

            tloss_out: PropagationLoss = np.asarray(tloss_per_freq, dtype=np.float64)
        else:
            # Return single TL value
            pressure_magnitude = np.abs(pressure[0, 0, 0, 0])

            # Handle zero pressure case
            if pressure_magnitude < 1e-12:
                tloss_out = 999.0
            else:
                tloss_out = float(-20 * np.log10(pressure_magnitude))

        # Calculate travel time
        speed = _as_scalar_float(self.ssp.calculate(array_pos[2]), "sound speed")
        time = float(distance / speed)

        return tloss_out, float(time)

    def propagate_spectrum(
        self,
        platform: "Platform",
        source: "State",
        frequencies_hz: ArrayLike,
    ) -> SpectrumResult:
        """Run rtrs simulation for broadband spectrum propagation.

        Computes complex transfer functions H(f) for each frequency bin and sensor. Suitable for
        STFT-based broadband processing where H(f) is applied to each STFT frame.

        Parameters
        ----------
        platform
            Sensor platform with array geometry.
        source
            Acoustic source position.
        frequencies_hz : ArrayLike
            Array of frequencies in Hz (from STFT bins).

        Returns
        -------
        SpectrumResult
            - ``transfer_functions`` : Complex array of shape (num_sensors, num_frequencies)
                containing H(f).
            - ``propagation_time_s`` : Mean travel time in seconds.

        """
        run_simulation = _get_rtrs_run_simulation()
        source_position = _get_source_position(source)
        array_position = platform.array.state_vector
        array_ref_position = platform.array.ref_state_vector

        # Flatten to 1D
        source_pos = source_position.flatten()
        array_ref_pos = array_ref_position.flatten()

        # Calculate distance and dynamic parameters
        distance = float(np.linalg.norm(source_pos - array_ref_pos))
        max_steps, max_range_m = self._calculate_max_steps_and_range(distance)

        # Calculate launch angles
        launch_azimuths = self._calculate_launch_azimuths(source_pos, array_ref_pos)

        num_elev = (
            int((self.elevation_range[1] - self.elevation_range[0]) / self.elevation_resolution)
            + 1
        )
        launch_elevations = np.linspace(
            self.elevation_range[0], self.elevation_range[1], num_elev
        ).tolist()

        # Determine spatial extent for grids
        # Include all sensor positions in the array
        x_coords = [source_pos[0]] + array_position[0, :].tolist()
        y_coords = [source_pos[1]] + array_position[1, :].tolist()

        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        # Add margins
        margin = 0.1
        x_range_width = max(x_max - x_min, 1000.0)
        y_range_width = max(y_max - y_min, 1000.0)

        x_margin = x_range_width * margin
        y_margin = y_range_width * margin

        x_range = (x_min - x_margin, x_max + x_margin)
        y_range = (y_min - y_margin, y_max + y_margin)

        # Generate 2D bathymetry grid
        x_bty, y_bty, z_bty = self.bathymetry.get_grid(x_range, y_range)
        z_bty_flat = -z_bty.flatten(order="C")

        # Generate 3D SSP grid
        z_range = [0.0, np.max(z_bty_flat)]
        x_ssp, y_ssp, z_ssp, c_ssp = self.ssp.get_3d_grid(
            x_range,
            y_range,
            z_range,
            self.ssp_resolution[0],
            self.ssp_resolution[1],
            self.ssp_resolution[2],
        )

        # Build rtrs environment configuration with all sensors as receivers
        env_config = {
            "ssp": {
                "x_ssp_m": x_ssp.tolist(),
                "y_ssp_m": y_ssp.tolist(),
                "z_ssp_m": z_ssp.tolist(),
                "c_m_s": c_ssp.tolist(),
            },
            "bathymetry": {
                "x_bty_m": x_bty.tolist(),
                "y_bty_m": y_bty.tolist(),
                "z_bty_m": z_bty_flat.tolist(),
                "bottom_model": self._resolved_bottom_model(),
            },
            "source": {
                "position": [
                    float(source_pos[0]),
                    float(source_pos[1]),
                    float(-source_pos[2]),
                ],
                "freq_hz": np.asarray(frequencies_hz, dtype=float).tolist(),
                "launch_elev_deg": launch_elevations,
                "launch_azim_deg": launch_azimuths,
            },
            "receivers": {
                "config_type": "array",
                "x_rcvr_m": array_position[0, :].tolist(),
                "y_rcvr_m": array_position[1, :].tolist(),
                "z_rcvr_m": (-array_position[2, :]).tolist(),  # Negate for rtrs convention
            },
            "beam": {
                "step_m": float(self.step_m),
                "max_steps": int(max_steps),
                "max_range_m": float(max_range_m),
                "store_ray_paths": bool(self.store_ray_paths),
                "integration_method": self.integration_method,
            },
        }

        if self.water_density_g_cm3 is not None:
            env_config["bathymetry"]["water_density_g_cm3"] = float(self.water_density_g_cm3)

        # Run rtrs simulation
        result = run_simulation(env_config)

        # Extract pressure field
        pf = result["pressure_field"]
        shape = tuple(pf["shape"])  # (nfreq, nreceivers, 1, 1)

        # Reconstruct complex pressure
        re = np.array(pf["pressure_re"], dtype=np.float32).reshape(shape)
        im = np.array(pf["pressure_im"], dtype=np.float32).reshape(shape)
        pressure = re + 1j * im

        # Extract transfer functions: shape (num_frequencies, num_sensors)
        # rtrs returns shape (nfreq, nreceivers, 1, 1), squeeze to (nfreq, nreceivers)
        transfer_functions = pressure[:, :, 0, 0]

        # Transpose to (num_sensors, num_frequencies) for consistency with processing
        transfer_functions = transfer_functions.T

        # Complex conjugate to match phase convention of rtrs (e^{+i\omega t}) to the
        # simulation convention (e^{-i\omega t})
        transfer_functions = np.conj(transfer_functions)

        # Calculate mean travel time
        speed = _as_scalar_float(self.ssp.calculate(array_ref_pos[2]), "sound speed")
        propagation_time_s = float(distance / speed)

        return transfer_functions, propagation_time_s
