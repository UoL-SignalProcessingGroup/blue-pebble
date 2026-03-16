# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

- Package name: `bluepebble`
- Language: Python 3.11+
- Domain: underwater acoustics, passive sonar simulation, signal processing, and Stone Soup integration
- Architecture: Blue Pebble is a Stone Soup plugin and should align with Stone Soup extension patterns
- Primary source tree: `bluepebble/`
- Supporting trees: `docs/`, `.github/`

## Environment Setup

Preferred environment: the existing dev container in `.devcontainer/`

Local setup:
```bash
python3 -m pip install -U pip
python3 -m pip install -e ".[dev]"

# For docs or notebooks/examples
python3 -m pip install -e ".[docs,examples]"
```

## Commands

```bash
# Lint
ruff check .

# Lint and auto-fix
ruff check . --fix

# Format
ruff format .

# Type-check
pyright bluepebble/

# Tests
pytest

# Run a single test file or test by name
pytest tests/test_beamformer.py
pytest tests/ -k "test_cfar"

# Run tests with coverage report
pytest tests/ --cov=bluepebble --cov-report=term-missing

# Build package
python -m build --sdist --wheel

# Build docs
cd docs && make html

# Install for development (from repo root)
pip install -e ".[dev]"
```

## Validation Expectations

- Python changes: run `ruff check .` and relevant `pytest` coverage for the affected area
- Packaging/dependency changes: run `python -m build --sdist --wheel`
- Docs changes: run `cd docs && make html`
- Examples/tutorials: ensure the touched notebook or documentation path remains coherent and up to date

## Working Rules

- Prefer small, targeted changes over broad refactors.
- Preserve public API compatibility unless the task explicitly requires a breaking change.
- Prefer a deprecation period of at least one release cycle before removing or changing public interfaces.
- Treat Stone Soup compatibility as a core design constraint for public-facing simulation and tracking components.
- When adding or reshaping components that conceptually extend Stone Soup behavior, prefer inheriting from the relevant Stone Soup modules or base classes instead of introducing parallel abstractions.
- Use clear, domain-meaningful names over shorthand.
- Validate inputs explicitly and use standard Python exceptions/warnings.
- Use NumPy-style docstrings for public APIs.
- Use British English in documentation and docstrings.

## Scientific/Numerical Guidance

- Treat behavioral changes in propagation, beamforming, detection, and tracking logic as high-risk.
- Prefer deterministic tests for numerical methods.
- Use explicit tolerances such as `rtol` and `atol` where exact equality is not appropriate.
- Call out performance implications if a change affects simulation-heavy paths.

## External Tooling Constraints

- Bellhop/BellhopCUDA support is optional and depends on an external executable.
- Do not assume external acoustic propagation executables are available unless the task or environment confirms it.
- Avoid making Bellhop-dependent validation a default requirement for unrelated changes.

## Dependency Policy

- Prefer the standard library or existing project dependencies.
- Do not add new dependencies without explicit approval.
- If a dependency is added, update `pyproject.toml` with an appropriate version constraint and justify the need.

## Commit Hygiene

- Use Conventional Commits for commit messages.
- Prefer scopes when they clarify the affected area, e.g. `feat(docs): ...` or `fix(plotter): ...`.

## PR/Review Priorities

- Numerical correctness
- Backwards compatibility
- Focused, reviewable diffs
- Adequate validation for changed behavior
- Clear explanation of compatibility impact, if any

## Release/Versioning Notes

- Versioning is dynamic via `setuptools_scm`.
- Do not manually set `project.version` in `pyproject.toml`.
- Git tags are the source of release versions.

## Stone Soup Integration Pattern

Every public class in this library inherits from Stone Soup's `Base` and declares its constructor parameters as class-level `Property` annotations — not in `__init__`. Stone Soup's metaclass reads these to generate `__init__`, provide declarative configuration, and support serialisation.

```python
class MyModel(Base):
    param_one: float = Property(doc="Description of param")
    param_two: int = Property(default=10, doc="Optional param")
```

Rules:
- **Never** use `from __future__ import annotations` in a file that defines a Stone Soup `Base` subclass. PEP 563 stringifies annotations, which breaks the metaclass's ability to read them on Python ≥3.14, causing `ValueError: Type was not specified`. Files with only `TYPE_CHECKING` guards are fine — but any file with a `Property`-bearing class must not have this import.
- Always supply the type via the class-level annotation (`param: float = Property(...)`). Do not also pass the type as the first positional argument to `Property` — Stone Soup raises `ValueError: Type was specified both` when both are present.
- When overriding `__init__`, always call `super().__init__(*args, **kwargs)` first.

## Architecture: Simulation Pipeline

A complete passive sonar simulation connects components in this order:

```
TowedArrayPlatform  ──►  PassiveSonarArraySimulator  ──►  PassiveSonarDetector
         ▲                        ▲
         │              ┌─────────┴──────────────┐
 GroundTruthPath    Signal model(s)    AcousticPropagationModel
                    AmbientNoise       Beamformer + SteeringCalculator
```

1. **`TowedArrayPlatform`** (`platform/towedarray.py`) — wraps Stone Soup's `MultiTransitionMovingPlatform`. It simulates a towed linear array by maintaining a chain of `MovingMovable` followers behind the host. After each call to `move()`, it captures a `PlatformState` (containing `HostState` and `ArrayState`) that downstream components query.

2. **Simulators** (`simulator/`) — two concrete implementations:
   - `DiscretePassiveSonarArraySimulator`: generates one independent broadband snapshot per platform timestamp. Requires the propagation model to implement `propagate_spectrum`. The full source waveform is computed once, then partitioned into per-timestamp chunks.
   - `ContinuousPassiveSonarArraySimulator` (`simulator/continuous.py`): WOLA/COLA-based overlap-add processing for waveform continuity across frames. Used when phase coherence between timesteps matters.

3. **Signal models** (`signal/`) — produce per-sensor complex waveforms. Two sibling branches inherit from the private `_SignalBase` (shared parameter contract):
   - **`Signal`** (per-timestep path) — subclasses implement `_generate_base_signal(source)`. `Signal.generate()` validates inputs, normalises the waveform via `_prepare_waveform()`, then applies attenuation and per-sensor phase delays in the frequency domain via `_apply_propagation()`. Used by `DiscretePassiveSonarArraySimulator`.
   - **`AnthropogenicSignal`** (STFT-first path) — subclasses implement `_generate_base_signal(source)` once; `compute_stft(source)` caches the result as an STFT. Used by `ContinuousSTFTPassiveSonarArraySimulator` and `ContinuousFractionalDelayPassiveSonarArraySimulator`.

4. **Propagation models** (`models/propagation/acoustic.py`) — implement `propagate(platform_state, source_state)` returning `(tloss_db, propagation_time_s)`, and optionally `propagate_spectrum(platform_state, source_state, frequencies_hz)` returning per-sensor transfer functions `H(f)`. Backends range from analytical spreading laws to external Bellhop/BellhopCUDA calls.

5. **Beamformer + SteeringCalculator** (`sigproc/beamformer.py`) — `SteeringCalculator.calculate(platform_state)` computes per-direction per-sensor delays from array geometry and a `SoundSpeedProfile`. The result feeds `Beamformer.beamform(sensor_signals, steering_delays_s)`. The two main implementations are `DelayAndSumBeamformer` (time/frequency/broadband-power domains, Numba-accelerated) and `MinimumVarianceDistortionlessResponseBeamformer` (STFT-based Capon).

6. **Detector** (`detector/`) — `PassiveSonarDetector` consumes the simulator's generator, computes SNR from beamformed power, and runs a `detection_chain` (list of `DetectionAlgorithm` instances applied sequentially). Detections are emitted as Stone Soup `Detection` objects with bearing state vectors.

## Key Conventions

**Type aliases** — each module defines local `TypeAlias` names (e.g., `FloatArray`, `ComplexArray`, `DetectionArray`) rather than importing a shared set. This is intentional for readability within each file.

**`TYPE_CHECKING` guards** — files that would create circular imports at runtime guard the import under `if TYPE_CHECKING:`. Do NOT add `from __future__ import annotations` to any file that defines a Stone Soup `Base` subclass (see above); only add it to pure-utility or helper files with no `Property` declarations.

**Pyright configuration** — `pyright.toml` sets `typeCheckingMode = "basic"` and silences `reportAssignmentType` / `reportArgumentType` across all `bluepebble/` subdirectories. This is because Stone Soup's metaclass-driven `Property` system causes false positives that cannot be resolved without stub files. Do not tighten these suppression rules without verifying against the full Stone Soup type surface.

**Numba JIT** — `_time_das` and `_frequency_das` in `beamformer.py` are compiled with explicit Numba type signatures and `cache=True`. Changes to their signatures require updating the `@njit(...)` decorator tuple, not just the function signature.

**Bellhop integration** — `external_tools/bellhopcuda/` contains the BellhopCUDA binary. The `BellhopModel` and `BellhopCUDAModel` propagation backends invoke this via `subprocess`. Tests that require Bellhop are skipped when the executable is absent. `read_shade_file` in `utils/bellhop.py` parses the binary `.shd` output format.

**Git** - Do not add "Authored by Claude" to commit messages.
