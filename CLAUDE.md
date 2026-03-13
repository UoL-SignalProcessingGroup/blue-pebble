# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

See `AGENTS.md` for the canonical list of commands (format, lint, test, build). Key additions:

```bash
# Run a single test file or test by name
pytest tests/test_beamformer.py
pytest tests/ -k "test_cfar"

# Run tests with coverage report
pytest tests/ --cov=bluepebble --cov-report=term-missing

# Lint
ruff check .

# Lint and auto-fix
ruff check . --fix

# Format
ruff format .

# Type-check
pyright bluepebble/

# Install for development (from repo root)
pip install -e ".[dev]"
```

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

3. **Signal models** (`signal/`) — produce per-sensor complex waveforms:
   - `DiscreteTimestepSignal` subclasses implement `generate(source, sensor_delays_s, tloss_db, propagation_time_s)` and are consumed per-timestep.
   - `ContinuousTimestepSignal` / `BroadbandStftSignalBase` subclasses implement `compute_stft(source)` / `get_source_signal()` and are consumed once up-front.
   - `Signal.generate()` in `signal/base.py` applies attenuation and per-sensor phase delays in the frequency domain via FFT — subclasses typically implement `_generate_base_signal(source)` rather than overriding `generate()`.

4. **Propagation models** (`models/propagation/acoustic.py`) — implement `propagate(platform_state, source_state)` returning `(tloss_db, propagation_time_s)`, and optionally `propagate_spectrum(platform_state, source_state, frequencies_hz)` returning per-sensor transfer functions `H(f)`. Backends range from analytical spreading laws to external Bellhop/BellhopCUDA calls.

5. **Beamformer + SteeringCalculator** (`sigproc/beamformer.py`) — `SteeringCalculator.calculate(platform_state)` computes per-direction per-sensor delays from array geometry and a `SoundSpeedProfile`. The result feeds `Beamformer.beamform(sensor_signals, steering_delays_s)`. The two main implementations are `DelayAndSumBeamformer` (time/frequency/broadband-power domains, Numba-accelerated) and `MinimumVarianceDistortionlessResponseBeamformer` (STFT-based Capon).

6. **Detector** (`detector/`) — `PassiveSonarDetector` consumes the simulator's generator, computes SNR from beamformed power, and runs a `detection_chain` (list of `DetectionAlgorithm` instances applied sequentially). Detections are emitted as Stone Soup `Detection` objects with bearing state vectors.

## Key Conventions

**Type aliases** — each module defines local `TypeAlias` names (e.g., `FloatArray`, `ComplexArray`, `DetectionArray`) rather than importing a shared set. This is intentional for readability within each file.

**`TYPE_CHECKING` guards** — files that would create circular imports at runtime guard the import under `if TYPE_CHECKING:`. Do NOT add `from __future__ import annotations` to any file that defines a Stone Soup `Base` subclass (see above); only add it to pure-utility or helper files with no `Property` declarations.

**Pyright configuration** — `pyright.toml` sets `typeCheckingMode = "basic"` and silences `reportAssignmentType` / `reportArgumentType` across all `bluepebble/` subdirectories. This is because Stone Soup's metaclass-driven `Property` system causes false positives that cannot be resolved without stub files. Do not tighten these suppression rules without verifying against the full Stone Soup type surface.

**Numba JIT** — `_time_das` and `_frequency_das` in `beamformer.py` are compiled with explicit Numba type signatures and `cache=True`. Changes to their signatures require updating the `@njit(...)` decorator tuple, not just the function signature.

**Bellhop integration** — `external_tools/bellhopcuda/` contains the BellhopCUDA binary. The `BellhopModel` and `BellhopCUDAModel` propagation backends invoke this via `subprocess`. Tests that require Bellhop are skipped when the executable is absent. `read_shade_file` in `utils/bellhop.py` parses the binary `.shd` output format.

**`signal/biological.py`** — currently has 0% test coverage. It contains whale call and snapping shrimp signal models. Treat changes here with extra care and add tests.
