# Installation

## Core package

Install the core package with pip:

```bash
pip install nereus
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

Bellhop and bellhopcuda support is optional and depends on an external executable that is not distributed with Nereus. Documentation and most package functionality should remain usable without it.

We recommend [bellhopcuda](https://github.com/A-New-BellHope/bellhopcuda), a modern C++/CUDA port.

> **Important**
> Nereus does not distribute Bellhop or bellhopcuda. These must be installed separately.

### Installing bellhopcuda

**Windows (precompiled)**
Download precompiled binaries from the [bellhopcuda releases page](https://github.com/A-New-BellHope/bellhopcuda/releases).

Place `bellhopcxx.exe` somewhere on your system `PATH`, or provide its path explicitly in Nereus.

**Linux/macOS (build from source)**

```bash
git clone https://github.com/A-New-BellHope/bellhopcuda.git
cd bellhopcuda
# Follow the build instructions in the bellhopcuda README.
```

Ensure the resulting `bellhopcxx` executable is available on your `PATH`.

### Using Bellhop in Nereus

```python
from nereus.models.propagation import BellhopAcousticPropagationModel

model = BellhopAcousticPropagationModel(
    env_depth=3000,
    ssp=my_ssp,
)
```

If needed, provide the full path explicitly:

```python
from nereus.models.propagation import BellhopAcousticPropagationModel

model = BellhopAcousticPropagationModel(
    env_depth=3000,
    ssp=my_ssp,
    exe_path="/full/path/to/bellhopcxx",
)
```
