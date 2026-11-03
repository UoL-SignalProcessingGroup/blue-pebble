# Testing

The automated test suite is deterministic and lightweight. It focuses on numerical behaviour, validation logic and public interfaces that can be exercised without a full Stone Soup runtime. Most external tooling is mocked, but the suite includes a small number of `rtrs` smoke tests that run when the package is available.

To see the current test inventory:

```bash
pytest --collect-only -q
```

## How the suite is structured

- The tests are primarily unit tests, with a few light integration-style checks.
- `tests/support.py` provides small Stone Soup and Blue Pebble stubs so modules can be imported in isolation.
- Optional external dependencies are usually mocked rather than executed.
- The tests prefer explicit expected values and deterministic assertions over broad random coverage. Statistical checks, such as achieved false-alarm rates, use fixed seeds and tolerances sized to their sampling error.

## Coverage by area

### Package API

`tests/test_package_api.py`

- public names exposed by `bluepebble`
- lazy submodule import and caching through `__getattr__`
- rejection of unknown attributes, and removed APIs raising an error that names their replacement

### Signal models and utilities

`tests/test_signal_utils.py`, `tests/test_signal_base.py`, `tests/test_signal_models.py`, `tests/test_signal_anthropogenic.py`, `tests/test_signal_biological.py`, `tests/test_signal_active.py`

- STFT utilities, inverse-STFT round-trip behaviour and fade helpers
- `Signal.generate` validation and attenuation/delay application branches
- ambient-noise and effects helpers
- anthropogenic source synthesis from each source's metadata, and the deprecated constructor arguments
- biological signal generation, call template construction and snapping shrimp models
- active pulses: CW tone, level and phase; LFM sweep, edge taper and configuration checks

### Seed management

`tests/test_seed.py`

- `set_seed` and `get_rng` behaviour and accessibility via `bluepebble`
- reproducibility of independent RNG streams spawned from a global seed
- per-instance seed override and non-deterministic default behaviour
- cross-signal-type reproducibility from a single `set_seed` call

### Beamforming and array resolution

`tests/test_beamformer.py`, `tests/test_sigproc_resolution.py`

- delay-and-sum domain validation, input-shape checks and zero-delay behaviour
- shading normalisation and rejection of zero-sum shading
- STFT short-input and overlap edge cases
- MVDR validation, finite output on deterministic inputs, and multiband output matching separate single-band runs
- steering delays and array-axis wrapping at ±π
- mainlobe width against the textbook beamwidth, and the CFAR windows derived from it
- steering at an elevation: bit-identical horizontal delays at zero elevation, sin(e) and cos(e) scaling on vertical and horizontal lines, mirror pairing below the horizontal, and a broadband plane wave located in bearing and elevation by a crossed array

### Matched filtering

`tests/test_sigproc_matchedfilter.py`

- the correlation peak at the echo delay with the echo's energy, and agreement with scipy's direct correlation
- multi-channel rows filtered independently, and the replica taper
- shape validation, and parabolic peak interpolation recovering an exact vertex

### Environment models

`tests/test_environment_models.py`

- sound-speed profile handling for negative depths
- gridded sound-speed profile generation
- flat, wedge and seamount bathymetry

### Detection

`tests/test_detector_algorithms.py`, `tests/test_detector_algorithms_validation.py`, `tests/test_detector_cfar_correctness.py`, `tests/test_detector_noise_calibration.py`, `tests/test_detector_passive.py`, `tests/test_detector_multiband.py`, `tests/test_detector_metrics.py`

- CA-CFAR and OS-CFAR construction, validation and the two threshold modes (fixed `threshold_factor`, or `target_pfa` with a noise calibration)
- closed-form thresholds and detection probabilities pinned to published values (Rohling; Gandhi and Kassam; Chalabi)
- noise-floor estimators against brute-force references, and achieved false-alarm rates by quadrature and direct simulation
- noise calibration: exact recovery on i.i.d. Gamma data, generalised Pareto tail extrapolation, and held-out Pfa on correlated noise
- peak consolidation, including clusters that straddle the ±180° wrap
- passive and multiband detector wiring from beamformed data to Stone Soup detections with `Bearing` state vectors
- timestep metrics, parameter sweeps and theoretical ROC helpers

### Plotting helpers

`tests/test_plotter.py`

- axis-scaling and layout helper functions
- spectrogram validation and figure construction
- ROC/PR plotting and combined subplot behaviour
- bearing-time records: wrapped bearings, overlays, legend grouping and the mathematical, true and relative bearing conventions
- world plotting validation and stationary-platform rendering

### Propagation models

`tests/test_propagation_models.py`

- analytic cylindrical and spherical propagation formulas
- sensor delay calculation and rejection of invalid attenuation factors
- `rtrs` option validation, missing-package behaviour, transmission loss and spectrum transfer functions
- real `rtrs` smoke tests for scalar and per-frequency transmission loss and broadband transfer functions

`tests/test_propagation_bellhop.py`, `tests/test_scattering_models.py`

- Bellhop environment and bathymetry files, arrivals parsing (including phase), world-to-depth
  conversion, invalid geometry and executable errors, using a stub executable
- Bellhop as a spectrum propagation model: transfer functions summed from arrivals (conjugate at
  negative frequencies), receivers mapped to their grid points, no-arrival sensors, run counts
  per receiver geometry, and `propagate` loss and travel time
- with `bellhopcxx` on `PATH` (skipped otherwise, as in CI): direct and surface-reflected paths
  against spherical spreading and the image source; transfer functions against the analytic
  Lloyd's mirror field; bottom-reflection phase against the Rayleigh coefficient; shared and
  per-sensor runs agreeing over a flat bottom; and, with rtrs installed, Bellhop against a
  converged rtrs in the same sediment half-space
- constant target strength gain and its frequency independence

### Simulators

`tests/test_simulator_acoustic.py`, `tests/test_simulator_modules.py`

- behavioural checks for passive and broadband simulator outputs, including noise-only runs and absent targets
- simulator base-class helpers for model resolution, target lookup, noise shaping, beamforming and sensor payloads
- discrete simulator source-signal resolution and timestep validation
- continuous STFT and fractional-delay simulator interpolation, fading and reconstruction branches

`tests/test_simulator_active.py`

- active echoes at the monostatic round-trip delay and the bistatic range sum, including a
  sonobuoy receiving a hull array's ping, with the out leg traced from transmitter to target
- echo level against the sonar equation with spherical spreading, and baffle gains on
  transmit and receive
- an LFM matched-filter peak, per-sensor delays across a line array, no wrap-around of late
  echoes, and silence from targets that cannot be heard
- default ping times, a missing transmitter state, seeded noise and settings validation
- with `bellhopcxx` on `PATH`: the direct echo against spherical spreading, and a bistatic
  sonobuoy echo
- with rtrs installed: the direct echo against spherical spreading

### Towed-array models

`tests/test_towedarray_models.py`, `tests/test_arrays_base.py`,
`tests/test_towedarray_platform_deprecated.py`

These run against the real Stone Soup platform and mounting classes.

- follower geometry for overlapping and separated sensor states
- horizontal offset handling and depth consistency
- sonar platform depth, mounting arrays and the move cascade to them
- array mounting, sensor layout, `mounting_offset` as the tow point, state capture and heading
  wrapping at ±π
- platform and sensor path retrieval
- the towed array's forward axis and baffle gains
- the deprecated `TowedArrayPlatform` reproducing `SonarPlatform` + `TowedArray` exactly, and the
  simulators' deprecated `platform=` argument

`tests/test_bowarray.py`, `tests/test_flankarray.py`, `tests/test_sonobuoy.py`

- bow dome layout on its sphere, mounting depth, forward-facing normals, axis and history through
  a turn
- flank panels per side: rotation offset, outward normals, grid layout and reference, the
  bow-to-tow-point fit check, and configuration errors
- sonobuoy hydrophone position and depth sign, and `SingleSensorArray` on a plain Stone Soup platform
- sensor spacing checks (touching sensors, grating-lobe limit)
- `RigidArray`'s reference point matching Stone Soup's own mounted position, and the heading of a
  fixed platform

### Types

`tests/test_types_sensordata.py`

- `PassiveSonarSensorData` construction, defaults and multiband band-label validation
- `ActiveSonarSensorData` construction and waveform shape validation
- types package public API exports

## Current gaps

The main areas not yet covered well are:

- real end-to-end runs with a full Stone Soup installation, from simulation through to tracking (the tutorials exercise this path, but they run only in the docs build)
- stronger numerical checks on MVDR, such as beam patterns against known steering responses
- execution of the examples, which the docs build renders without running

## Running the tests

From a configured development environment:

```bash
pytest
```

With coverage, as CI runs it on Python 3.12:

```bash
pytest --cov=bluepebble --cov-report=term-missing --cov-fail-under=55
```

Useful targeted runs:

```bash
pytest tests/test_detector_algorithms.py
pytest tests/test_detector_noise_calibration.py
pytest tests/test_propagation_models.py
pytest tests/test_simulator_acoustic.py
```
