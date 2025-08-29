"""Defines acoustic propagation models for simulating sound propagation.

© Copyright 2025 Joshua J. Wakefield.
Licensed under the MIT License.
"""

import subprocess
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from stonesoup.base import Base, Property

from nereus.stonesoup.models.environment import SoundSpeedProfile
from nereus.stonesoup.utils import read_shade_file


class AcousticPropagationModel(ABC, Base):
    """An abstract base class for all acoustic propagation models.

    It defines a common interface and implements shared functionality.
    """

    ssp = Property(SoundSpeedProfile, doc="Sound speed profile")

    @abstractmethod
    def propagate(self, platform, source) -> tuple:
        """Propagate a signal from a source to a platform.

        This method must be implemented by all subclasses.
        """
        pass

    def compute_sensor_delays(self, platform, source) -> np.ndarray:
        """Compute time delays for each sensor in an array.

        Args:
            platform: The platform object representing the sensor array.
            source: The source object representing the acoustic source.

        This method is shared by all propagation models.

        """
        source_position = source.state_vector[source.metadata["position_mapping"]]
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
        speeds = self.ssp.calculate(array_position[2, :])

        # Calculate time delays
        delays = (distances - reference_distance) / speeds
        return delays


class CylindricalAcousticPropagationModel(AcousticPropagationModel):
    """A simple acoustic model based on cylindrical spreading and absorption loss.

    This model provides a basic estimate of transmission loss without the
    computational overhead of more complex models like Bellhop.

    Attributes:
        attenuation_factor (float): The absorption loss factor in dB/km.
        ssp (SoundSpeedProfile): An instance of a sound speed profile.

    """

    attenuation_factor = Property(
        float, default=0.5, doc="The absorption loss factor in dB/km"
    )

    def __post_init__(self):
        """Validate the attenuation factor after initialization."""
        if self.attenuation_factor < 0:
            raise ValueError("Attenuation factor must be non-negative.")

    def propagate(self, platform, source):
        """Propagates a signal using a cylindrical spreading loss model.

        This method calculates the transmission loss based on a simple model that
        combines cylindrical spreading (10*log10(r)) with a frequency-
        independent absorption term.

        Args:
            platform: An object representing the sensor platform.
            source: An object representing the acoustic source.

        Returns:
            A tuple containing:
            - tloss (float): The transmission loss in decibels (dB).
            - time (float): The direct path signal travel time in seconds.

        """
        source_position = source.state_vector[source.metadata["position_mapping"]]
        array_ref_position = platform.array.ref_state_vector

        distance = np.linalg.norm(source_position - array_ref_position)
        speed = self.ssp.calculate(array_ref_position[2])
        time = distance / speed
        tloss = 10 * np.log10(distance) + self.attenuation_factor * (distance / 1000)
        return tloss, time


class SphericalAcousticPropagationModel(AcousticPropagationModel):
    """A simple acoustic model based on spherical spreading and absorption loss.

    This model provides a basic estimate of transmission loss without the
    computational overhead of more complex models like Bellhop.

    Attributes:
        attenuation_factor (float): The absorption loss factor in dB/km.
        ssp (SoundSpeedProfile): An instance of a sound speed profile.

    """

    attenuation_factor = Property(
        float, default=0.001, doc="The absorption loss factor in dB/km"
    )

    def __post_init__(self):
        """Validate the attenuation factor after initialization."""
        if self.attenuation_factor < 0:
            raise ValueError("Attenuation factor must be non-negative.")

    def propagate(self, platform, source):
        """Propagates a signal using a spherical spreading loss model.

        This method calculates the transmission loss based on a simple model that
        combines spherical spreading (20*log10(r)) with a frequency-
        independent absorption term.

        Args:
            platform: An object representing the sensor platform.
            source: An object representing the acoustic source.

        Returns:
            A tuple containing:
            - tloss (float): The transmission loss in decibels (dB).
            - time (float): The direct path signal travel time in seconds.

        """
        source_position = source.state_vector[source.metadata["position_mapping"]]
        array_ref_position = platform.array.ref_state_vector

        distance = np.linalg.norm(source_position - array_ref_position)
        speed = self.ssp.calculate(array_ref_position[2])
        time = distance / speed
        tloss = 20 * np.log10(distance) + self.attenuation_factor * (distance / 1000)
        return tloss, time


class BellhopAcousticPropagationModel(AcousticPropagationModel):
    """A class to represent a Bellhop acoustic propagation model.

    This model calls an external Bellhop executable to perform the propagation
    simulation. It's the user's responsibility to ensure that the Bellhop
    executable is installed and its path is provided correctly.

    Attributes:
        env_depth (float): The depth of the environment in meters.
        ssp (SoundSpeedProfile): An instance of a sound speed profile.
        exe_path (str | Path): The path to the Bellhop executable.
        sound_speed_profile (np.ndarray): A two-column array of [depth, sound_speed].

    """

    env_depth = Property(float, doc="The depth of the environment in meters")
    exe_path = Property(str, doc="The path to the Bellhop executable")

    def __post_init__(self):
        """Initialize the Bellhop acoustic propagation model."""
        exe_path_obj = Path(self.exe_path)
        if not exe_path_obj.is_file():
            raise FileNotFoundError(f"Bellhop executable not found at: {self.exe_path}")
        self.exe_path = exe_path_obj

        depth = np.arange(0, self.env_depth + 1, 100)
        sound_speed = self.ssp.calculate(depth)
        self.sound_speed_profile = np.column_stack((depth, sound_speed))

    def propagate(self, platform, source):
        """Run a Bellhop simulation for a single source and receiver.

        This method generates the necessary environment file, calls the external
        Bellhop executable, reads the resulting shade file, and computes the
        transmission loss and travel time.

        Args:
            platform: An object representing the sensor platform.
            source: An object representing the acoustic source.

        Returns:
            A tuple containing:
            - tloss (float): The transmission loss in decibels (dB).
            - time (float): The direct path signal travel time in seconds.

        Raises:
            subprocess.CalledProcessError: If the external Bellhop executable
                returns a non-zero exit code, indicating a failure.

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
        pressure = np.squeeze(pressure)
        tloss = np.abs(pressure)
        tloss = -20 * np.log10(tloss + 1e-12)  # Avoid log(0) by adding a small constant

        distance = np.linalg.norm(
            source.state_vector[[0, 2, 4]] - platform.array.ref_state_vector
        )
        speed = self.ssp.calculate(platform.array.ref_state_vector[2])
        time = distance / speed

        # Handle cases where pressure is zero, resulting in infinite tloss
        if np.isinf(tloss[-1]):
            return 999.0, time  # Return a large, finite loss value
        return tloss[-1], time

    def _create_env_file(
        self,
        platform,
        source,
        output_dir=Path("."),
        options="SVW",
        bottom_bc="A",
        runtype="Ib",
        nbeams=0,
        beam_angles=None,
    ):
        if beam_angles is None:
            beam_angles = [-89.0, 89.0]
        """Create the Bellhop environment file (.env) from a template.

        This private method gathers all necessary simulation parameters,
        calculates derived values, and writes them to a text file with the
        precise formatting required by the Bellhop executable.

        Args:
            receiver: An object representing the sensor platform, containing its state.
            source: An object representing the acoustic source, containing its state.
            output_dir (Path, optional): The directory to save the file in.
                Defaults to the current directory.
            options (str, optional): A string of option characters for Bellhop
                that controls the model's behavior. Defaults to "SVW".
            bottom_bc (str, optional): The bottom boundary condition code,
                e.g., 'A' for an acousto-elastic half-space. Defaults to "A".
            runtype (str, optional): The Bellhop run type code, e.g., 'Ib' for
                eigenrays and arrivals. Defaults to "Ib".
            nbeams (int, optional): The number of beams for the simulation;
                0 lets Bellhop choose. Defaults to 0.
            beam_angles (List[float], optional): The minimum and maximum beam
                launch angles in degrees. Defaults to [-89.0, 89.0].
            **kwargs: Absorbs any other unused keyword arguments for backward
                compatibility.

        """
        # 1. Calculate All Required Values
        # ==================================
        title = "'env'"
        frequency = source.frequency[np.argmax(source.amplitude)]
        max_range_m = np.linalg.norm(
            source.state_vector[[0, 2, 4]] - platform.array.ref_state_vector
        )

        # Source and receiver depths
        source_depth = np.abs(source.state_vector[4])
        receiver_depth = np.abs(platform.array.ref_state_vector[2])

        # Define bathymetry and check if a .bty file is needed
        bathy = [[0, self.env_depth]]  # Simple flat bottom
        bottom_bc_final = "A~" if len(bathy) > 2 else bottom_bc
        if len(bathy) > 2:
            self._write_bathy_file(bathy, "env", output_dir)

        # Prepare Sound Speed Profile (SSP) string
        ssp_header = (
            f"{self.sound_speed_profile[0, 0]:.1f} "
            f"{self.sound_speed_profile[0, 1]:.1f}  /\n"
        )
        ssp_body = "\n".join(
            [f"{z:.1f} {c:.1f}  /" for z, c in self.sound_speed_profile[1:]]
        )
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

    def _write_bathy_file(self, bathy, filename, output_dir):
        """Write bathymetry file for complex bottom topography."""
        # This method would write a .bty file for complex bathymetry
        # For now, it's a placeholder since we use simple flat bottom
        pass
