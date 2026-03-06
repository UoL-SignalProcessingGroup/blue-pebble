# Testing

The current automated test suite is intentionally deterministic and lightweight. It focuses on
numerical behaviour, validation logic, and public interfaces that can usually be exercised without
a full Stone Soup runtime. Most external tooling is mocked, but the suite now also includes a
small number of real Bellhop and `rtrs` smoke tests when those tools are available locally.

At the time of writing, the suite contains **108 tests** across **9 test modules** under `tests/`.

## How the suite is structured

- The tests are primarily unit tests with a few light integration-style checks.
- `tests/support.py` provides small Stone Soup and Blue Pebble stubs so modules can be imported in
  isolation.
- Optional external dependencies are usually mocked rather than executed, with a small number of
  real `rtrs` and Bellhop smoke tests when those tools are installed locally.
- The tests prefer explicit expected values and tight, deterministic assertions over broad random
  coverage.

## Coverage by area

Current module counts:

- `tests/test_package_api.py`: 3 tests
- `tests/test_signal_utils.py`: 4 tests
- `tests/test_beamformer.py`: 16 tests
- `tests/test_environment_models.py`: 5 tests
- `tests/test_detector_algorithms.py`: 15 tests
- `tests/test_detector_metrics.py`: 7 tests
- `tests/test_plotter.py`: 18 tests
- `tests/test_simulator_acoustic.py`: 12 tests
- `tests/test_propagation_models.py`: 28 tests

### Package API

`tests/test_package_api.py`

- public names exposed by `bluepebble`
- lazy submodule import and caching through `__getattr__`
- rejection of unknown public attributes

### Signal utilities

`tests/test_signal_utils.py`

- STFT window validation
- STFT/inverse-STFT round-trip behaviour
- fade-in application and no-op cases

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

### Detector algorithms

`tests/test_detector_algorithms.py`

- threshold detector strictness and empty outputs
- peak detection spacing, suppression of nearby weaker peaks, and no-peak cases
- CA-CFAR isolated targets, flat backgrounds, edge handling, default `wrap` mode, `same` mode, and
  oversized training windows
- OS-CFAR rank validation, isolated targets, edge detections, and float-rank handling from
  parameter sweeps

### Detector metrics

`tests/test_detector_metrics.py`

- timestep metrics for empty, wrapped-bearing, and false-positive-heavy cases
- chained detector behaviour across sparse intermediate outputs
- `SweepResult` derived metrics and parameter-selection helpers
- synthetic parameter sweeps with deterministic expected counts

### Plotting helpers

`tests/test_plotter.py`

- axis-scaling and layout helper functions
- spectrogram validation and figure construction
- ROC/PR plotting and combined subplot behaviour
- BTR validation, wrapped bearings, detection/track/truth overlays, and legend grouping
- world plotting validation and stationary-platform rendering

### Acoustic simulators

`tests/test_simulator_acoustic.py`

- passive simulator signal summation, noise addition, beamforming, timestamp ordering, and empty
  snapshot handling
- broadband simulator validation for timestep and target requirements
- broadband reconstruction with noise truncation and padding
- multiple-target summation, missing-target timesteps, and beamformer sensor ordering

### Propagation models

`tests/test_propagation_models.py`

- analytic cylindrical and spherical propagation formulas
- sensor delay calculation
- rejection of invalid attenuation factors
- Bellhop executable resolution, environment-file generation, subprocess success and failure paths,
  zero-pressure handling, scalar-vs-array coercion, and higher-dimensional shade output
- real Bellhop smoke tests against the project-local `bellhopcxx` executable
- `rtrs` option validation, missing-package behaviour, single-frequency and per-frequency
  transmission loss, and spectrum transfer-function shaping
- real `rtrs` smoke tests for scalar TL, per-frequency TL, and broadband transfer functions

## Current gaps

The main areas not yet covered well are:

- real end-to-end integration with a full Stone Soup installation
- broader Bellhop executable coverage beyond the current smoke tests
- deeper beamformer coverage, especially non-trivial steering cases and stronger MVDR numerical
  assertions
- higher-level passive detector pipelines beyond the algorithm primitives and metrics helpers
- packaging/build checks and documentation notebook coherence as part of the test suite

## Running the tests

From a configured development environment:

```bash
pytest
```

To run with coverage reporting:

```bash
pytest --cov=bluepebble --cov-report=term-missing --cov-report=xml
```

In CI, coverage is collected on the Python 3.12 matrix job to keep overall runtime manageable.

Some propagation tests are optional:

- real `rtrs` tests run only when the `rtrs` package is installed
- real Bellhop smoke tests run only when the project-local `bellhopcxx` executable is present

Useful targeted runs:

```bash
pytest tests/test_detector_algorithms.py
pytest tests/test_beamformer.py
pytest tests/test_propagation_models.py
pytest tests/test_simulator_acoustic.py
```
