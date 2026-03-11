# `bluepebble.simulator`

Stone Soup-compatible simulator entrypoints for passive sonar array data generation.

Current primary simulator entrypoints are:

- `ContinuousSTFTPassiveSonarArraySimulator` for broadband STFT-domain propagation and reconstruction.
- `DiscretePassiveSonarArraySimulator` for discrete timestep broadband workflows.
- `ContinuousFractionalDelayPassiveSonarArraySimulator` for continuous fractional-delay synthesis.

For compatibility with older notebooks, `DepreciatedDiscretePassiveSonarArraySimulator` remains available but is deprecated for new work.

```{eval-rst}
.. automodule:: bluepebble.simulator
   :members:
   :no-index:
```
