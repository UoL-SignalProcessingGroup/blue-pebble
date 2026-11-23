# `bluepebble.simulator`

Stone Soup-compatible simulator entrypoints for passive and active sonar array data generation.

Current primary simulator entrypoints are:

- `ContinuousSTFTPassiveSonarArraySimulator` for broadband STFT-domain propagation and reconstruction.
- `DiscretePassiveSonarArraySimulator` for discrete timestep broadband workflows.
- `ContinuousFractionalDelayPassiveSonarArraySimulator` for continuous fractional-delay synthesis.
- `ContinuousChannelFilterPassiveSonarArraySimulator` for continuous synthesis with a time-domain bulk delay and short per-sensor channel filters, keeping both Doppler and the full `H(f)`.
- `ContinuousModalPassiveSonarArraySimulator` for the same synthesis driven by a `KrakenModalPropagationModel`, with the channel evaluated densely from modes computed once.
- `ActiveSonarSimulator` for active sonar pings, monostatic or bistatic, with any spectrum propagation model. A signal with `baseband=True` is simulated as its complex envelope at a sampling rate that only has to cover the pulse's bandwidth.

`PassiveSonarSensorData` and `ActiveSonarSensorData` live in `bluepebble.types` and are documented with the types API surface.

## Doppler

No propagation model computes a Doppler term. `H(f)` is instantaneous and the source spectrum is fixed. A received frequency shift emerges only from how the propagation delay `tau(t) = range(t) / c` evolves during reconstruction, which depends on the simulator.

- `ContinuousFractionalDelayPassiveSonarArraySimulator` is best for Doppler. It resamples in the time domain, `x(n - tau[n]·fs)`, with no frame-rate limit. The channel `H(f)` is collapsed to one broadband gain + delay (frequency-flat).
- `ContinuousSTFTPassiveSonarArraySimulator` (`stft_interp`, `wola_interp`) gives correct Doppler only below `fs / (2·hop)` (where `hop = frame_len // hop_factor`) and aliases above it. The channel `H(f)` is full per-frequency-bin.
- `ContinuousSTFTPassiveSonarArraySimulator` (`cola`) staircases Doppler between knots. The channel `H(f)` is full per-frequency-bin (nearest knot).
- `ContinuousChannelFilterPassiveSonarArraySimulator` delays each source in the time domain by one bulk delay, as the fractional-delay simulator does, so Doppler has no frame-rate limit. A short filter per sensor and knot carries the rest of the channel, so `H(f)` is full per-frequency (linearly crossfaded between knots).
- `ContinuousModalPassiveSonarArraySimulator` is the channel-filter simulator with knots planned from the modes, close enough that each mode keeps its own Doppler shift, `-k_m v_r`.
- `DiscretePassiveSonarArraySimulator` has no Doppler, since each step is an independent frozen-geometry snapshot. The channel `H(f)` is full per-frequency-bin, frozen per step.

`ActiveSonarSimulator` has no Doppler either: each ping's geometry is frozen at the transmit time.

The channel-filter simulators give both correct wideband Doppler and a full frequency-dependent channel. With the STFT simulator, raise `hop_factor` to lift its alias limit.

```{eval-rst}
.. automodule:: bluepebble.simulator
   :members:
```
