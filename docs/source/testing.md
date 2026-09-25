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

`tests/test_signal_utils.py`, `tests/test_signal_base.py`, `tests/test_signal_models.py`, `tests/test_signal_anthropogenic.py`, `tests/test_signal_biological.py`

- STFT utilities, inverse-STFT round-trip behaviour and fade helpers
- `Signal.generate` validation and attenuation/delay application branches
- ambient-noise and effects helpers
- anthropogenic source synthesis from each source's metadata, and the deprecated constructor arguments
- biological signal generation, call template construction and snapping shrimp models

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

### Simulators

`tests/test_simulator_acoustic.py`, `tests/test_simulator_modules.py`

- behavioural checks for passive and broadband simulator outputs, including noise-only runs and absent targets
- simulator base-class helpers for model resolution, target lookup, noise shaping, beamforming and sensor payloads
- discrete simulator source-signal resolution and timestep validation
- continuous STFT and fractional-delay simulator interpolation, fading and reconstruction branches

### Towed-array models

`tests/test_towedarray_models.py`

- follower geometry for overlapping and separated platform states
- horizontal offset handling and depth consistency
- platform construction, sensor initialisation, state capture and heading wrapping at ±π
- host and sensor path retrieval

### Types

`tests/test_types_sensordata.py`

- `PassiveSonarSensorData` construction, defaults and multiband band-label validation
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
