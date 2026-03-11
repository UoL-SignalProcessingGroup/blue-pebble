# Installation

## Core package

Install the core package with pip:

```bash
pip install blue-pebble
```

This installs the core framework with the built-in propagation models.

## Development install

For local development, install the project in editable mode with the development dependencies:

```bash
python3 -m pip install -U pip
python3 -m pip install -e ".[dev]"
```

## Documentation and examples

If you are working with the published notebooks or building the documentation locally, install the docs and examples extras:

```bash
python3 -m pip install -e ".[docs,examples]"
```

## Optional Bellhop support

Bellhop and bellhopcuda support is optional and depends on an external executable that is not distributed with Blue Pebble. Documentation and most package functionality should remain usable without it.

We recommend [bellhopcuda](https://github.com/A-New-BellHope/bellhopcuda), a modern C++/CUDA port.

> **Important**
> Blue Pebble does not distribute Bellhop or bellhopcuda. These must be installed separately.

### Installing bellhopcuda

**Windows (precompiled)**
Download precompiled binaries from the [bellhopcuda releases page](https://github.com/A-New-BellHope/bellhopcuda/releases).

Place `bellhopcxx.exe` somewhere on your system `PATH`, or provide its path explicitly in Blue Pebble.

**Linux/macOS (build from source, outside Docker)**

Install the required native build tools first.

On Debian/Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y git cmake build-essential
```

On macOS with Homebrew:

```bash
brew install cmake
xcode-select --install
```

If you want to keep Bellhop local to the Blue Pebble repository rather than installing it
system-wide, build it into a project subdirectory such as `external_tools/bellhopcuda`:

```bash
git clone --recurse-submodules https://github.com/A-New-BellHope/bellhopcuda.git external_tools/bellhopcuda
cd external_tools/bellhopcuda
mkdir -p build
cd build
cmake -DBHC_ENABLE_CUDA=OFF -DBHC_BUILD_EXAMPLES=OFF ..
cmake --build . -j
```

This produces a local executable at:

```text
external_tools/bellhopcuda/bin/bellhopcxx
```

You can either:

- add that directory to `PATH`, or
- pass the executable path explicitly to Blue Pebble

For a repo-local shell session:

```bash
export PATH="$PWD/external_tools/bellhopcuda/bin:$PATH"
```

### Using Bellhop in Blue Pebble

```python
from bluepebble.models.propagation import BellhopAcousticPropagationModel

model = BellhopAcousticPropagationModel(
    env_depth=3000,
    ssp=my_ssp,
)
```

If needed, provide the full path explicitly:

```python
from bluepebble.models.propagation import BellhopAcousticPropagationModel

model = BellhopAcousticPropagationModel(
    env_depth=3000,
    ssp=my_ssp,
    exe_path="external_tools/bellhopcuda/bin/bellhopcxx",
)
```
