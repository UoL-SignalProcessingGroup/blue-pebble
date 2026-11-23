# Installation

## Core package

Install the core package with pip:

```bash
pip install blue-pebble
```

This installs the core framework including all propagation backends (analytical models and ray tracing via rtrs).

## Bellhop (optional)

{class}`~bluepebble.models.propagation.BellhopAcousticPropagationModel` runs Bellhop's C++ port,
`bellhopcxx`, as an external program. It is not installed by pip. Build it from
[bellhopcuda](https://github.com/A-New-BellHope/bellhopcuda) (requires CMake and a C++17
compiler; CUDA is not needed) and put the `bellhopcxx` executable on your `PATH`:

```bash
git clone --recursive https://github.com/A-New-BellHope/bellhopcuda.git
cd bellhopcuda
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBHC_ENABLE_CUDA=OFF -DBHC_BUILD_EXAMPLES=OFF
cmake --build build --target bellhopcxx -j 8
ln -s "$PWD/bin/bellhopcxx" ~/.local/bin/bellhopcxx  # or anywhere on your PATH
```

Alternatively, pass the executable's full path to the model as `exe_path`. The tests that run
Bellhop are skipped when `bellhopcxx` is not on `PATH`.

## Acoustics Toolbox (optional)

{class}`~bluepebble.models.propagation.KrakenModalPropagationModel` runs the Acoustics Toolbox
programs `krakenc` (or `kraken`) and `field` as external programs. They are not installed by pip.
Build the toolbox from [its source](http://oalib.hlsresearch.com/AcousticsToolbox/) with
gfortran (`make` in the toolbox root), then either put the programs on your `PATH` or point
`BLUEPEBBLE_AT_BIN` at the toolbox root:

```bash
export BLUEPEBBLE_AT_BIN=/path/to/at  # finds Kraken/krakenc.exe and KrakenField/field.exe
```

`BLUEPEBBLE_AT_BIN` can also list directories that hold the programs, separated by `:`. The
tests that run the toolbox are skipped when it is not found.

## Development install

For local development, install the project in editable mode with the development dependencies:

```bash
python3 -m pip install -U pip
python3 -m pip install -e ".[dev]"
```

## Documentation and examples

If you are running the tutorials and examples or building the documentation locally, install the docs and examples extras:

```bash
python3 -m pip install -e ".[docs,examples]"
```

