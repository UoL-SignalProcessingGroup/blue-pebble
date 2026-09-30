# Profiling and Optimisation

This page is a step-by-step guide to making code faster without changing what it computes:
measure a representative workload, save its outputs, find where the time goes, change one thing,
check the outputs still match, and measure again. Optimise only once the code is correct and
tested, since the existing tests are what the optimised version is checked against.

## 1. Decide whether it is worth it

Speed matters for code that runs once per scan, per trial or per frequency bin, where small costs
multiply, and for anything a tutorial runs, since the tutorials execute in every documentation
build and Read the Docs stops a build after fifteen minutes. Code that runs once per script rarely
repays the effort.

## 2. Time a representative workload

Measure at the sizes the code will actually meet (sensor counts, beam counts, scan lengths,
number of steps), since costs scale differently and a toy case can point at the wrong place. Take
baselines from plain runs, never from a profiler, which adds its own overhead, and compare before
and after on the same machine:

```bash
/usr/bin/time -p python docs/tutorials/getting_started.py
```

For a single function, make one untimed call first (Numba compiles on first use and NumPy warms
up), then report the best and median of several:

```python
import timeit
from functools import partial

call = partial(beamformer.beamform, signals, delays)
call()
times = timeit.repeat(call, number=1, repeat=7)
```

`benchmarks/das_beamformer.py` is a worked example and a template for new benchmarks.

## 3. Save reference outputs before changing anything

Without saved outputs you can only compare the new code with itself. The benchmark script saves
and checks them:

```bash
python benchmarks/das_beamformer.py --save before.npz
python benchmarks/das_beamformer.py --check before.npz   # after the change
```

To run the committed version side by side with the working tree, load it from git as a separate
module, giving it a dotted name inside the package so its relative imports resolve:

```bash
git show HEAD:bluepebble/sigproc/conventional.py > /tmp/conventional_head.py
```

```python
import importlib.util

spec = importlib.util.spec_from_file_location(
    "bluepebble.sigproc._conventional_head", "/tmp/conventional_head.py"
)
head = importlib.util.module_from_spec(spec)
spec.loader.exec_module(head)
```

If it contains cached Numba functions, delete the `__pycache__` beside the copy before each run.

To test on the inputs a function really meets, wrap it for one run of a script and pickle what it
receives and returns:

```python
import pickle
import runpy

import bluepebble.simulator.continuous as continuous

cls = continuous.ContinuousSTFTPassiveSonarArraySimulator
original = cls._synthesise_stft_interp


def capture(self, ctx, targets_data):
    result = original(self, ctx, targets_data)
    if targets_data:
        with open("inputs.pkl", "wb") as f:
            pickle.dump({"ctx": ctx, "targets": targets_data, "result": result}, f)
    return result


cls._synthesise_stft_interp = capture
runpy.run_path("docs/tutorials/getting_started.py", run_name="__main__")
```

## 4. Find the expensive stage with `cProfile`

```bash
python -m cProfile -o profile.prof docs/tutorials/getting_started.py
```

```python
import pstats

stats = pstats.Stats("profile.prof")
stats.sort_stats("cumulative").print_stats("bluepebble", 25)
stats.sort_stats("tottime").print_stats(25)
```

Cumulative time finds the expensive stage, self time (`tottime`) the function doing the work, and
call counts (`ncalls`) expose loops. Hundreds of thousands of calls to a small function usually
mean a Python loop doing work that one call could. `cProfile` exaggerates code that makes many
small calls, such as Stone Soup attribute access, so confirm those findings with plain timing.

Profile the workload at four times its length too. A function whose share grows with the run is
scanning a history on every step.

## 5. Find the expensive lines with Scalene

[Scalene](https://github.com/plasma-umass/scalene) samples the program and splits each line's
time into Python and native (NumPy, BLAS, Numba). High Python time calls for fewer Python-level
operations; all-native time means the work itself has to shrink. Install it in your own
environment:

```bash
pip install scalene
scalene run --cpu-only --program-path . -o profile.json docs/tutorials/getting_started.py
scalene view profile.json            # browser; add --cli -r for the terminal
```

`--program-path .` is needed because Scalene otherwise profiles only the script's own folder, and
`--profile-only bluepebble` drops the script too (the folder is `blue-pebble-dev`). Script
arguments go after `---`. Scalene's overhead falls unevenly, so read its figures as relative
shares.

## 6. Watch a long run with `py-spy` (optional)

[`py-spy`](https://github.com/benfred/py-spy) attaches to a running process, which suits the
minutes-long examples. On macOS it needs root: `sudo py-spy top --pid <PID>`.

## 7. Recognise the pattern

| Symptom | Likely cause | Fix |
|---|---|---|
| A line with large native time building an array much bigger than its result | Broadcasting to a large intermediate, then summing | A matrix product (`@`) or `np.einsum` |
| Huge call counts to a small NumPy function in a loop | One NumPy call per frame, bin or element | One call on the whole array, e.g. `np.fft.ifft(stft, axis=1)` |
| A Python loop over sensors running several array operations each time | A temporary array and a pass over memory per operation | A fused `@njit(parallel=True)` kernel with `prange` over sensors; `rocket-fft` provides `np.fft` inside it |
| `np.exp` or `np.sin` recomputed over evenly spaced values | Recomputing what changes by a fixed factor each step | A recurrence, recomputed exactly every few dozen steps so rounding cannot build up |
| Time per step grows with run length | Scanning a history from the start each step | Keep the latest state, or index the history in a dictionary |
| Millions of `StateVector.__getitem__` or `Base.__get__` calls | Reading Stone Soup states element by element | Convert to a NumPy array once, outside the loop |
| The same result recomputed from identical inputs | Unchanged geometry or steering recomputed | A cache with an explicit invalidation rule, after checking its memory cost |

When writing a Numba kernel:

- Compile on first use (`@njit(cache=True)`, no type signature), so users who never call it do not
  pay at import.
- Give each output element one writer, so results do not depend on the thread count.
- Coverage cannot trace kernels; use `NUMBA_DISABLE_JIT=1` for the true figure.
- For parallel simulations, use processes with `NUMBA_NUM_THREADS=1`. Numba's default `workqueue`
  layer aborts if several threads of one process run kernels at once.
- Profilers show a kernel as one block, so time it directly.

## 8. Prototype the fix in isolation

Before editing the package, time the current and proposed versions side by side at realistic
sizes and check they agree. It is cheap to discard an idea here, and ratios are more trustworthy
than absolute times.

## 9. Apply the change and verify it

- **Against the reference**, cell by cell and including the weakest cells, since a figure relative
  to the peak can hide large relative errors in deep nulls. Rounding-level differences (about
  `1e-15` relative in double precision) are expected when a sum's order changes.
- **Downstream**, by running what consumes the output, e.g. confirming identical detections.
- **Determinism**, by repeating calls and running with BLAS limited to one thread
  (`VECLIB_MAXIMUM_THREADS=1` on macOS, `OPENBLAS_NUM_THREADS=1` with OpenBLAS).
- **The tests**, by breaking the new code in the way each new test targets and confirming it fails.
- **The usual checks**, `ruff check .`, `pyright` and the full test suite.

## 10. Measure again and record the result

Re-run the plain timing. Stop when the target is met, or when the largest remaining cost is
external (such as the `rtrs` ray tracer) or can only be cut by trading away accuracy. Put the
before and after figures in the commit message under a `perf` type.
