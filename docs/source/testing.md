# Testing

The automated test suite is intentionally deterministic and lightweight. It focuses on numerical behaviour, validation logic, and public interfaces that can usually be exercised without a full Stone Soup runtime.

Most external tooling is mocked, but the suite also includes a small number of `rtrs` smoke tests when the tool is available locally.

There are currently **293 tests** across **18 test modules** under `tests/`.

## How the suite is structured

- The tests are primarily unit tests with a few light integration-style checks.
- `tests/support.py` provides small Stone Soup and Blue Pebble stubs so modules can be imported in isolation.
- Optional external dependencies are usually mocked rather than executed.
- The tests prefer explicit expected values and deterministic assertions over broad random coverage.

## Coverage by area

Current module counts:

- `tests/test_package_api.py`: 4 tests
- `tests/test_signal_utils.py`: 7 tests
- `tests/test_signal_base.py`: 11 tests
- `tests/test_signal_models.py`: 8 tests
- `tests/test_signal_anthropogenic.py`: 15 tests
- `tests/test_signal_biological.py`: 48 tests
- `tests/test_seed.py`: 28 tests
- `tests/test_beamformer.py`: 16 tests
- `tests/test_environment_models.py`: 5 tests
- `tests/test_detector_algorithms.py`: 15 tests
- `tests/test_detector_metrics.py`: 7 tests
- `tests/test_detector_passive.py`: 18 tests
- `tests/test_plotter.py`: 27 tests
- `tests/test_propagation_models.py`: 15 tests
- `tests/test_simulator_acoustic.py`: 14 tests
- `tests/test_simulator_modules.py`: 13 tests
- `tests/test_towedarray_models.py`: 39 tests
- `tests/test_types_sensordata.py`: 3 tests

### Package API

`tests/test_package_api.py`

- public names exposed by `bluepebble`
- lazy submodule import and caching through `__getattr__`
- rejection of unknown public attributes

### Signal models and utilities

`tests/test_signal_utils.py`, `tests/test_signal_base.py`, `tests/test_signal_models.py`, `tests/test_signal_anthropogenic.py`, `tests/test_signal_biological.py`

- STFT utilities, inverse-STFT round-trip behaviour, and fade helpers
- `Signal.generate` validation and attenuation/delay application branches
- ambient-noise and effects helpers
- tonal and broadband anthropogenic model validation, caching, and reset behaviour
- biological signal generation, call template construction, and snapping shrimp models

### Seed management

`tests/test_seed.py`

- `set_seed` and `get_rng` behaviour and accessibility via `bluepebble`
- reproducibility of independent RNG streams spawned from a global seed
- per-instance seed override and non-deterministic default behaviour
- cross-signal-type reproducibility from a single `set_seed` call

### Beamforming

`tests/test_beamformer.py`

- delay-and-sum domain validation and input-shape checks
- time-domain and frequency-domain zero-delay beamforming behaviour
- shading normalisation and rejection of zero-sum shading
- broadband power behaviour when no frequency bins are active
- STFT short-input and overlap edge cases
- time-domain cropping behaviour under large integer delays
- MVDR validation and finite-output checks on deterministic inputs
- MVDR zero-power behaviour when the selected band excludes all bins
- steering-delay calculation for simple horizontal array geometry

### Environment models

`tests/test_environment_models.py`

- sound-speed profile handling for negative depths
- gridded sound-speed profile generation
- flat bathymetry validation
- wedge bathymetry surface clamping
- seamount interpolation behaviour

### Detection

`tests/test_detector_algorithms.py`, `tests/test_detector_metrics.py`, `tests/test_detector_passive.py`

- threshold, peak, CA-CFAR, and OS-CFAR behaviour
- wrapped-bearing timestep metrics and parameter-sweep helpers
- passive detector chain wiring from beamformer outputs to detections

### Plotting helpers

`tests/test_plotter.py`

- axis-scaling and layout helper functions
- spectrogram validation and figure construction
- ROC/PR plotting and combined subplot behaviour
- BTR validation, wrapped bearings, detection/track/truth overlays, and legend grouping
- world plotting validation and stationary-platform rendering

### Propagation models

`tests/test_propagation_models.py`

- analytic cylindrical and spherical propagation formulas
- sensor delay calculation
- rejection of invalid attenuation factors
- `rtrs` option validation, missing-package behaviour, single-frequency and per-frequency transmission loss, and spectrum transfer-function shaping
- real `rtrs` smoke tests for scalar TL, per-frequency TL, and broadband transfer functions

### Simulators

`tests/test_simulator_acoustic.py`, `tests/test_simulator_modules.py`

- compatibility-focused behavioural checks for passive and broadband simulator outputs
- simulator base-class helpers for model resolution, target lookup, noise shaping, beamforming, and sensor payload construction
- discrete simulator source-signal resolution and timestep validation
- continuous simulator interpolation, fading, and reconstruction helper branches

### Towed-array models

`tests/test_towedarray_models.py`

- follower geometry behaviour for overlapping and separated platform states
- horizontal offset handling and depth consistency
- platform construction, sensor initialisation, and state capture
- host and sensor path retrieval

### Types

`tests/test_types_sensordata.py`

- `PassiveSonarSensorData` construction and default values
- types package public API exports

## Current gaps

The main areas not yet covered well are:

- real end-to-end integration with a full Stone Soup installation
- deeper beamformer coverage, especially non-trivial steering cases and stronger MVDR numerical assertions
- higher-level passive detector pipelines beyond the algorithm primitives and metrics helpers
- packaging/build checks and documentation notebook coherence as part of the default test suite

## Running the tests

From a configured development environment:

```bash
pytest
```

To run with coverage reporting:

```bash
pytest --cov=bluepebble --cov-report=term-missing --cov-report=xml
```

To inspect the currently collected test inventory:

```bash
pytest --collect-only -q
```

Useful targeted runs:

```bash
pytest tests/test_signal_base.py
pytest tests/test_signal_anthropogenic.py
pytest tests/test_detector_algorithms.py
pytest tests/test_propagation_models.py
pytest tests/test_simulator_acoustic.py
pytest tests/test_simulator_modules.py
```
