"""Defines sensor models and array generators for the Nereus plugin.

© Copyright 2025 Joshua J. Wakefield.
Licensed under the MIT License.
"""


# class Hydrophone(Sensor):
#     """A simple sensor to represent a single hydrophone."""


# class HorizontalArrayGenerator(Base):
#     """A factory component to generate a list of ideal hydrophone sensors.

#     This component is designed to be used in a YAML configuration to create
#     the `sensors` list for a Stone Soup Platform. It calculates the 3D
#     position of each sensor in a linear array based on a starting point,
#     the number of sensors, and their spacing.
#     """

#     # --- Configurable Properties ---
#     # These attributes are set by the Stone Soup loader from the YAML file.
#     start_position: list
#     num_sensors: int
#     spacing_m: float

#     def __init__(self, *args, **kwargs):
#         """Initialise the HorizontalArrayGenerator and validate parameters.

#         Args:
#             *args: Positional arguments passed to the parent Base class.
#             **kwargs: Keyword arguments passed to the parent Base class.

#         Raises:
#             ValueError: If `num_sensors` is less than 1, if `spacing_m` is not
#                 a positive value, or if `start_position` is not a 3D coordinate.

#         """
#         # The Base class constructor handles setting the properties from kwargs.
#         super().__init__(*args, **kwargs)

#         # --- Parameter Validation ---
#         if self.num_sensors < 1:
#             raise ValueError("num_sensors must be at least 1.")
#         if self.spacing_m <= 0:
#             raise ValueError("spacing_m must be a positive value.")
#         if len(self.start_position) != 3:
#             raise ValueError("start_position must be a 3D coordinate [x, y, z].")

#         # --- Sensor Generation ---
#         # Generate the list of hydrophone objects based on the configuration.
#         self.sensors = []
#         for i in range(self.num_sensors):
#             # Calculate the position of the current sensor along the x-axis
#             # relative to the start_position.
#             pos = np.array(self.start_position) + np.array([i * self.spacing_m, 0, 0])
#             self.sensors.append(Hydrophone(position=pos.tolist()))

#     def __iter__(self):
#         """Iterate through the list of sensors."""
#         return iter(self.sensors)
