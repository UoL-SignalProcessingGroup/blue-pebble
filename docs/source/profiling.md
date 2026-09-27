# Profiling and Optimisation

This page is a step-by-step guide to making new code faster without changing what it computes.
The loop is always the same: measure a representative workload, save its outputs, find where the
time goes, change one thing, check the outputs still match, and measure again.

Optimise only once the code is correct and tested. A faster wrong answer is still wrong, and the
existing tests are what the optimised version is checked against.

## 1. Decide whether it is worth it

Speed matters most for code that runs once per scan, per trial or per frequency bin, since that is
where small costs multiply. It also matters for anything a tutorial runs, because the tutorials
execute in every documentation build and Read the Docs stops a build after fifteen minutes. Code
that runs once per script rarely repays the effort.

## 2. Time a representative workload

Measure with the sizes the code will actually meet (sensor counts, beam counts, scan lengths,
number of steps), since the costs scale differently and a toy-sized case can point at the wrong
place. Time a plain run first, with no profiler attached, so you know the real total:

```bash
/usr/bin/time -p python docs/tutorials/getting_started.py
```

For a single function, time repeated calls and report the best and the median, since one run is
noisy. The first call is often slower than the rest (Numba compiles kernels on first use unless
they are cached, and NumPy and BLAS warm up), so make an untimed call before timing.

```python
import timeit
from functools import partial

call = partial(beamformer.beamform, signals, delays)
call()  # warm-up, untimed
times = timeit.repeat(call, number=1, repeat=7)
print(f"best {min(times) * 1e3:.1f} ms")
```

`benchmarks/das_beamformer.py` is a worked example of a standalone benchmark, and a template for
new ones.

## 3. Save reference outputs before changing anything

Save what the current code produces, for the same inputs you will use afterwards. Without this you
can only compare the new code with itself. The benchmark script does it with `--save` and
`--check`:

```bash
python benchmarks/das_beamformer.py --save before.npz
# ...change the code...
python benchmarks/das_beamformer.py --check before.npz
```

For a wider check, the committed version can be run side by side with the working tree by
loading it from git as a separate module. Give it a dotted name inside the package so its relative
imports resolve:

```python
import importlib.util
import subprocess

source = subprocess.run(
    ["git", "show", "HEAD:bluepebble/sigproc/conventional.py"],
    capture_output=True, text=True, check=True,
).stdout
path = "/tmp/conventional_head.py"
with open(path, "w") as f:
    f.write(source)
spec = importlib.util.spec_from_file_location("bluepebble.sigproc._conventional_head", path)
head = importlib.util.module_from_spec(spec)
spec.loader.exec_module(head)
OldBeamformer = head.DelayAndSumBeamformer
```

If the copied module contains Numba functions with `cache=True`, delete the `__pycache__` Numba
writes next to the copy before each run, or Numba fails to reload its own cache.

## 4. Find the expensive stage with `cProfile`

`cProfile` is built into Python and records every function call, so it gives exact call counts
and the time spent in each function:

```bash
python -m cProfile -o profile.prof docs/tutorials/getting_started.py
```

```python
import pstats

stats = pstats.Stats("profile.prof")
stats.sort_stats("cumulative").print_stats("bluepebble", 25)  # stages, including callees
stats.sort_stats("tottime").print_stats(25)  # where the work itself happens
```

Read three things from it:

- **Cumulative time** finds the expensive stage, e.g. the simulator's synthesis or the detector.
- **Self time** (`tottime`) finds the function doing the work within that stage.
- **Call counts** (`ncalls`) expose loops. Hundreds of thousands of calls to a small function, such
  as one `np.fft.ifft` per STFT frame or one `StateVector.__getitem__` per element, usually means
  a Python loop doing work that could be done in one call.

`cProfile` adds a fixed cost to every call it records, so code making many small calls looks
slower than it is. On the towed-array platform, whose Stone Soup properties make every attribute
access a Python call, it roughly quadrupled the apparent cost. Confirm such findings with plain
timing before acting on them. It also stops at function boundaries, so the NumPy work inside a
function appears as a single number.

## 5. Find the expensive lines with Scalene

[Scalene](https://github.com/plasma-umass/scalene) samples the running program instead of hooking
every call, so its overhead is small. It reports time per line and splits it into Python time and
native time (NumPy, BLAS, Numba and other compiled code), and it also profiles memory per line.
That split shows whether a slow line is paying for the interpreter or for the numerical work
itself, which decides the fix. It is not a project dependency, so install it in your own
environment:

```bash
pip install scalene
scalene --cli --reduced-profile --profile-only bluepebble docs/tutorials/getting_started.py
```

`--profile-only bluepebble` restricts the report to package code, and `--reduced-profile` hides
lines with little time. Run `scalene --help` for the options in your installed version.

## 6. Watch a long run with `py-spy` (optional)

[`py-spy`](https://github.com/benfred/py-spy) is also a sampling profiler, and it can attach to a
process that is already running. That suits the examples, which take minutes each. It also draws
flame graphs of the whole run. On macOS it must run as root.

```bash
pip install py-spy
sudo py-spy record -o profile.svg -- python docs/examples/comparing_bathymetry.py
sudo py-spy top --pid 12345  # live view of a running process
```

## 7. Recognise the pattern

Most slow code in this package has turned out to be one of a few patterns.

| Symptom in the profile | Likely cause | Fix |
|---|---|---|
| One line with large native time, building an array much bigger than its result | Broadcasting to a large intermediate, then summing it | A matrix product (`@`) or `np.einsum`. The broadband delay-and-sum beamformer went from `np.sum((A[:, :, None] * S[None]) * w, axis=1)` to `(A * w) @ S`, about 3x faster. |
| Huge call counts to a small NumPy function inside a loop | Calling NumPy once per frame, bin or element | Pass the whole array in one call, e.g. `np.fft.ifft(stft, axis=1)` for every frame at once |
| A transcendental function (`np.exp`, `np.sin`) recomputed in a loop over evenly spaced values | Recomputing something that changes by a fixed factor each step | Hoist it out of the loop, or use a recurrence (for steering phases, `A_next = A * exp(j*2*pi*df*delays)`), recomputing exactly every few dozen steps so rounding cannot build up |
| Time per step grows as the run gets longer | Scanning a history from the start on every step, so the total grows with the square of the run length | Keep the latest state, or index the history in a dictionary |
| Millions of `StateVector.__getitem__` or `Base.__get__` calls | Reading Stone Soup states element by element in a hot loop | Convert to a NumPy array once, outside the loop |
| The same expensive result computed again with identical inputs | Recomputing geometry or steering that has not changed | Cache it, with an explicit rule for when the cache is invalid, and check its memory cost first |

Numba kernels, such as the time and frequency-domain delay-and-sum, appear as a single native
block in every profiler. Time them directly.

## 8. Prototype the fix in isolation

Before editing the package, time the current and proposed versions side by side on synthetic data
at realistic sizes, and check they agree. A few lines of script is enough, and it is cheap to
discard an idea that does not pay off. Ratios are more trustworthy than absolute times here,
since single runs are noisy.

## 9. Apply the change and verify it

Check that the change is numerically faithful, not just faster:

- **Against the reference.** Compare cell by cell, not only against the peak, and look at the
  weakest cells too. A figure relative to the peak can hide large relative errors in deep nulls.
  Rounding-level differences (around `1e-15` relative in double precision) are expected when the
  order of a sum changes. Anything larger needs explaining.
- **Downstream.** Run what consumes the output. For signal processing that means calibrating a
  detector on the old and new outputs and confirming identical thresholds and detections.
- **Determinism.** Repeated calls, and a run with BLAS limited to one thread
  (`VECLIB_MAXIMUM_THREADS=1` on macOS, `OPENBLAS_NUM_THREADS=1` with OpenBLAS), should give
  identical output. Otherwise seeded runs stop being reproducible.
- **The usual checks.** `ruff check .`, `pyright` and the full test suite.

## 10. Measure again and record the result

Re-run the plain timing from step 2. Stop when the target is met, or when the largest remaining
cost is external (such as the `rtrs` ray tracer) or can only be reduced by trading away accuracy.
Put the before and after figures in the commit message, under a `perf` type, for example
`perf(sigproc): steer broadband DAS bins with a matrix product`.

## Pitfalls

- **Profiler overhead.** `cProfile` exaggerates code that makes many small calls. Sampling
  profilers and plain timing do not.
- **Warm-up.** First calls include Numba compilation and cache loading, so leave them out of
  timings.
- **Threads.** `rtrs` and BLAS use several threads, so CPU time can exceed wall time, and
  profilers report threaded code differently. Judge by wall time.
- **Machine.** Timings, and rounding at the level of `1e-16`, differ between machines and BLAS
  libraries. Compare before and after on the same machine.
- **Unrepresentative sizes.** A cost that is negligible at 32 sensors can dominate at 200, and
  anything that grows with the square of the run length only shows up in long runs.
