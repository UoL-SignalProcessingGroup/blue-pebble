# `bluepebble.arrays`

Sensor arrays mounted on or towed behind a platform. A platform hosts arrays, and each array
hosts its sensors. Arrays use Stone Soup's own mounting: mount one with the platform's
`add_sensor`, and its `mounting_offset` places the array's reference point relative to the
platform's reference point (here, the bow).

## Mounting arrays

```python
from stonesoup.types.array import StateVector
from stonesoup.types.state import State

from bluepebble.arrays import BowArray, FlankArray, TowedArray
from bluepebble.platform import SonarPlatform, Sonobuoy

ship = SonarPlatform(states=..., position_mapping=[0, 2, 4], transition_models=...,
                     transition_times=...)

# A towed array has its own dynamics; its mounting_offset is the tow point.
ship.add_sensor(TowedArray(num_sensors=64, cable_length_m=200.0, sensor_spacing_m=1.0,
                           array_depth_m=-60.0, mounting_offset=StateVector([-40.0, 0.0, 0.0])))

# Rigid arrays: geometry follows the platform's state at each time.
ship.add_sensor(BowArray(dome_radius_m=1.5, azimuth_extent_rad=2.1, elevation_extent_rad=0.5,
                         sensor_spacing_m=0.2, sensor_size_m=0.1, freq_max_hz=3000.0,
                         mounting_offset=StateVector([0.0, 0.0, -3.0])))
for side, y in (("port", 5.0), ("starboard", -5.0)):
    ship.add_sensor(FlankArray(side=side, aperture_length_m=20.0, sensor_spacing_m=0.2,
                               sensor_size_m=0.1, freq_max_hz=3000.0,
                               mounting_offset=StateVector([-15.0, y, -4.0])))

# A sonobuoy is a fixed platform with an omnidirectional hydrophone below it.
buoy = Sonobuoy(states=State(StateVector([2000.0, 0.0, 0.0])), position_mapping=[0, 1, 2],
                hydrophone_depth_m=50.0)

ship.move(t)
ship.arrays  # (TowedArray, BowArray, FlankArray, FlankArray)
```

Mount a towed array before the flank panels so their footprint is checked against its tow point.

```{eval-rst}
.. automodule:: bluepebble.arrays
   :members:
```
