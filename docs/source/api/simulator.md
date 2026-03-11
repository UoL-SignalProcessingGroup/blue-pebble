# `bluepebble.simulator`

Stone Soup-compatible simulator entrypoints for passive sonar array data generation.

Current primary simulator entrypoints are:

- `ContinuousSTFTPassiveSonarArraySimulator` for broadband STFT-domain propagation and reconstruction.
- `DiscretePassiveSonarArraySimulator` for discrete timestep broadband workflows.
- `ContinuousFractionalDelayPassiveSonarArraySimulator` for continuous fractional-delay synthesis.

For compatibility with older notebooks, `DepreciatedDiscretePassiveSonarArraySimulator` remains available but is deprecated for new work.

`PassiveSonarSensorData` now lives in `bluepebble.types` and is documented with the types API surface.

```{eval-rst}
.. automodule:: bluepebble.simulator
   :members:
   :no-index:
```
