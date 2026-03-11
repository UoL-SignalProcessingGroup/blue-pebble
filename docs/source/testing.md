# Testing

The automated test suite is intentionally deterministic and lightweight. It focuses on numerical behaviour, validation logic, and public interfaces that can usually be exercised without a full Stone Soup runtime.

Most external tooling is mocked, but the suite also includes a small number of real Bellhop and `rtrs` smoke tests when those tools are available locally.

At the time of writing (11 March 2026), there are currently **167 tests** across **15 test modules** under `tests/`.

## How the suite is structured

- The tests are primarily unit tests with a few light integration-style checks.
- `tests/support.py` provides small Stone Soup and Blue Pebble stubs so modules can be imported in isolation.
- Optional external dependencies are usually mocked rather than executed.
- The tests prefer explicit expected values and deterministic assertions over broad random coverage.

## Coverage by area

Current module counts:

- `tests/test_package_api.py`: 3 tests
- `tests/test_signal_utils.py`: 4 tests
- `tests/test_signal_base.py`: 8 tests
- `tests/test_signal_models_and_bellhop.py`: 7 tests
- `tests/test_signal_anthropogenic.py`: 20 tests
- `tests/test_beamformer.py`: 16 tests
- `tests/test_environment_models.py`: 5 tests
- `tests/test_detector_algorithms.py`: 15 tests
- `tests/test_detector_metrics.py`: 7 tests
- `tests/test_detector_passive.py`: 3 tests
- `tests/test_plotter.py`: 20 tests
- `tests/test_propagation_models.py`: 28 tests
- `tests/test_simulator_acoustic.py`: 12 tests
- `tests/test_simulator_modules.py`: 14 tests
- `tests/test_towedarray_models.py`: 5 tests

### Package API

`tests/test_package_api.py`

- public names exposed by `bluepebble`
- lazy submodule import and caching through `__getattr__`
- rejection of unknown public attributes

### Signal models and utilities

`tests/test_signal_utils.py`, `tests/test_signal_base.py`, `tests/test_signal_models_and_bellhop.py`, `tests/test_signal_anthropogenic.py`

- STFT utilities, inverse-STFT round-trip behaviour, and fade helpers
- `Signal.generate` validation and attenuation/delay application branches
- ambient-noise and effects helpers
- Bellhop shade-file parsing helpers
- tonal and broadband anthropogenic model validation, caching, and reset behaviour

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
- Bellhop executable resolution, environment-file generation, subprocess success and failure paths, zero-pressure handling, scalar-vs-array coercion, and higher-dimensional shade output
- real Bellhop smoke tests against the project-local `bellhopcxx` executable
- `rtrs` option validation, missing-package behaviour, single-frequency and per-frequency transmission loss, and spectrum transfer-function shaping
- real `rtrs` smoke tests for scalar TL, per-frequency TL, and broadband transfer functions

### Simulators

`tests/test_simulator_acoustic.py`, `tests/test_simulator_modules.py`

- compatibility-focused behavioural checks for passive and broadband simulator outputs
- simulator base-class helpers for model resolution, target lookup, noise shaping, beamforming, and sensor payload construction
- discrete simulator source-signal resolution and timestep validation
- continuous simulator interpolation, fading, and reconstruction helper branches
- deprecated discrete simulator compatibility and warning behaviour

### Towed-array models

`tests/test_towedarray_models.py`

- follower geometry behaviour for overlapping and separated platform states
- horizontal offset handling and depth consistency

## Current gaps

The main areas not yet covered well are:

- real end-to-end integration with a full Stone Soup installation
- broader Bellhop executable coverage beyond the current smoke tests
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

Some propagation tests are optional:

- real `rtrs` tests run only when the `rtrs` package is installed
- real Bellhop smoke tests run only when the project-local `bellhopcxx` executable is present

Useful targeted runs:

```bash
pytest tests/test_signal_base.py
pytest tests/test_signal_anthropogenic.py
pytest tests/test_detector_algorithms.py
pytest tests/test_propagation_models.py
pytest tests/test_simulator_acoustic.py
pytest tests/test_simulator_modules.py
```
