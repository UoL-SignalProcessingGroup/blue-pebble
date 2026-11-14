# `bluepebble.simulator`

Stone Soup-compatible simulator entrypoints for passive and active sonar array data generation.

Current primary simulator entrypoints are:

- `ContinuousSTFTPassiveSonarArraySimulator` for broadband STFT-domain propagation and reconstruction.
- `DiscretePassiveSonarArraySimulator` for discrete timestep broadband workflows.
- `ContinuousFractionalDelayPassiveSonarArraySimulator` for continuous fractional-delay synthesis.
- `ActiveSonarSimulator` for active sonar pings, monostatic or bistatic, with any spectrum propagation model. A signal with `baseband=True` is simulated as its complex envelope at a sampling rate that only has to cover the pulse's bandwidth.

`PassiveSonarSensorData` and `ActiveSonarSensorData` live in `bluepebble.types` and are documented with the types API surface.

## Doppler

No propagation model computes a Doppler term. `H(f)` is instantaneous and the source spectrum is fixed. A received frequency shift emerges only from how the propagation delay `tau(t) = range(t) / c` evolves during reconstruction, which depends on the simulator.

- `ContinuousFractionalDelayPassiveSonarArraySimulator` is best for Doppler. It resamples in the time domain, `x(n - tau[n]·fs)`, with no frame-rate limit. The channel `H(f)` is collapsed to one broadband gain + delay (frequency-flat).
- `ContinuousSTFTPassiveSonarArraySimulator` (`stft_interp`, `wola_interp`) gives correct Doppler only below `fs / (2·hop)` (where `hop = frame_len // hop_factor`) and aliases above it. The channel `H(f)` is full per-frequency-bin.
- `ContinuousSTFTPassiveSonarArraySimulator` (`cola`) staircases Doppler between knots. The channel `H(f)` is full per-frequency-bin (nearest knot).
- `DiscretePassiveSonarArraySimulator` has no Doppler, since each step is an independent frozen-geometry snapshot. The channel `H(f)` is full per-frequency-bin, frozen per step.

`ActiveSonarSimulator` has no Doppler either: each ping's geometry is frozen at the transmit time.

No single passive simulator gives both correct wideband Doppler and a full frequency-dependent channel: raise `hop_factor` or switch to the fractional-delay simulator to lift the STFT alias limit.

```{eval-rst}
.. automodule:: bluepebble.simulator
   :members:
```
