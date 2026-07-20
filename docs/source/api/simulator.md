# `bluepebble.simulator`

Stone Soup-compatible simulator entrypoints for passive sonar array data generation.

Current primary simulator entrypoints are:

- `ContinuousSTFTPassiveSonarArraySimulator` for broadband STFT-domain propagation and reconstruction.
- `DiscretePassiveSonarArraySimulator` for discrete timestep broadband workflows.
- `ContinuousFractionalDelayPassiveSonarArraySimulator` for continuous fractional-delay synthesis.

For compatibility with older notebooks, `DeprecatedDiscretePassiveSonarArraySimulator` remains available but is deprecated for new work. It emits a `DeprecationWarning` on instantiation.

`PassiveSonarSensorData` now lives in `bluepebble.types` and is documented with the types API surface.

## Doppler

No propagation model computes a Doppler term. `H(f)` is instantaneous and the source spectrum is fixed. A received frequency shift emerges only from how the propagation delay `tau(t) = range(t) / c` evolves during reconstruction, which it depends on the simulator.

- `ContinuousFractionalDelayPassiveSonarArraySimulator` — best for doppler: time-domain resampling `x(n - tau[n]·fs)`, no frame-rate limit. Channel `H(f)` is collapsed to one broadband gain + delay (frequency-flat).
- `ContinuousSTFTPassiveSonarArraySimulator` (`stft_interp`, `wola_interp`) — Doppler correct only below `fs / (2·hop)`, where `hop = frame_len // hop_factor`; aliases above it. Channel `H(f)` is full per-frequency-bin.
- `ContinuousSTFTPassiveSonarArraySimulator` (`cola`) — Doppler is staircased between knots. Channel `H(f)` is full per-frequency-bin (nearest knot).
- `DiscretePassiveSonarArraySimulator` — No Doppler: independent frozen-geometry snapshots. Channel `H(f)` is full per-frequency-bin, frozen per step.

No single simulator gives both correct wideband Doppler and a full frequency-dependent channel: raise `hop_factor` or switch to the fractional-delay simulator to lift the STFT alias limit.

```{eval-rst}
.. automodule:: bluepebble.simulator
   :members:
```
