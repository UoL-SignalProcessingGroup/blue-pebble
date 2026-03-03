# Blue Pebble

[![PyPI version](https://img.shields.io/pypi/v/blue-pebble.svg)](https://pypi.org/project/blue-pebble/)
[![Python versions](https://img.shields.io/pypi/pyversions/blue-pebble.svg)](https://pypi.org/project/blue-pebble/)
[![Ruff](https://img.shields.io/badge/lint-ruff-46a2f1)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/jjwakefield/blue-pebble/blob/main/LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

**Blue Pebble** is a research-oriented simulation framework for underwater acoustic sensing, currently focused on passive sonar signal processing, acoustic propagation modelling, beamforming, detection, and multi-target tracking.

Designed as a plugin for [Stone Soup](https://stonesoup.rtfd.io/), Blue Pebble supports research in:

- Underwater acoustics
- Passive sonar signal processing
- Towed array modelling
- Acoustic propagation modelling
- Beamforming and detection theory
- Target tracking and data association

Blue Pebble provides modular acoustic propagation backends, ranging from analytical spreading laws to external ray-tracing solvers (e.g., Bellhop), enabling trade-offs between physical fidelity and computational efficiency.

## Research Applications

Blue Pebble is intended for controlled, simulation-based studies, including:

- Evaluation of tracking and data association algorithms  
- End-to-end sonar performance analysis  
- Synthetic dataset generation for validation  
- Sensitivity analysis of propagation effects on detection and estimation  

Although current functionality centres on passive sonar, the architecture supports extension to additional modalities (e.g., active or multistatic configurations).

## Architecture

Blue Pebble follows a modular design that separates physical modelling from signal processing and tracking logic. Core components include:

- **Platform dynamics** - Kinematic modelling of ownship, targets, and arrays  
- **Acoustic propagation** - Pluggable propagation backends (analytical or external solvers)  
- **Signal generation** - Source modelling and noise synthesis  
- **Beamforming** - Array processing algorithms  
- **Detection** - Measurement formation and statistical thresholding  
- **Tracking** - Integration with Stone Soup estimators and data association  

This separation enables systematic experimentation across modelling assumptions and algorithmic choices without tightly coupling components.

## Features

Implemented capabilities include:

- Multi-body kinematic modelling for flexible towed arrays  
- Analytical spreading models and external ray-tracing integration (e.g., Bellhop)  
- Configurable source signature synthesis  
- Ambient, biological, and ownship noise modelling  
- Multiple beamforming algorithms  
- Detection algorithms with performance metrics  
- Passive sonar simulation pipelines  
- Native integration with Stone Soup tracking workflows  
- Plotting utilities for bearings and Cartesian tracks  
- Notebook-based tutorials and worked examples

## Installation

### Basic Installation (Core Models Only)

```bash
pip install blue-pebble
```

This installs the core framework with built-in propagation models.

### Optional: Ray Tracing with Bellhop

Blue Pebble supports Bellhop via an external executable.

We recommend [bellhopcuda](https://github.com/A-New-BellHope/bellhopcuda), a modern C++/CUDA port.

> **Important**: Blue Pebble does **not** distribute Bellhop or bellhopcuda. These must be installed separately.

#### Installing bellhopcuda

**Windows (Precompiled)**
Download precompiled binaries from the [bellhopcuda Releases page](https://github.com/A-New-BellHope/bellhopcuda/releases)

Place `bellhopcxx.exe` somewhere on your system `PATH`, or provide its path explicitly in Blue Pebble.

**Linux/macOS (Build from Source, Outside Docker)**

Install the native build tools first.

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

To keep Bellhop local to this repository rather than installing it system-wide:

```bash
git clone --recurse-submodules https://github.com/A-New-BellHope/bellhopcuda.git external_tools/bellhopcuda
cd external_tools/bellhopcuda
mkdir -p build
cd build
cmake -DBHC_ENABLE_CUDA=OFF -DBHC_BUILD_EXAMPLES=OFF ..
cmake --build . -j
```

This produces a local executable at `external_tools/bellhopcuda/bin/bellhopcxx`.

You can then either add it to `PATH`:

```bash
export PATH="$PWD/external_tools/bellhopcuda/bin:$PATH"
```

or pass the path explicitly in Blue Pebble.

#### Using Bellhop in Blue Pebble

Blue Pebble automatically detects the executable:

```python
from bluepebble.models.propagation import BellhopAcousticPropagationModel

model = BellhopAcousticPropagationModel(
    env_depth=3000,
    ssp=my_ssp
)
```

If needed, provide the full path:

```python
from bluepebble.models.propagation import BellhopAcousticPropagationModel

model = BellhopAcousticPropagationModel(
    env_depth=3000,
    ssp=my_ssp,
    exe_path="external_tools/bellhopcuda/bin/bellhopcxx"
)
```

## Development

### Recommended: Dev Container (Easiest Setup)

For a fully configured development environment (including bellhopcuda and other build dependencies), use the included Dev Container.

#### Requirements

- Docker Engine (Docker Desktop on Windows/macOS, or Docker on Linux)

Optional:
- Visual Studio Code
- VS Code Dev Containers extension

Clone the repository:
```bash
git clone https://github.com/jjwakefield/blue-pebble.git
```

#### Using the Dev Container (VS Code Workflow)

If using Visual Studio Code with the Dev Containers extension:
```bash
cd blue-pebble
code .
```

When prompted, select **"Reopen in Container."**

VS Code will:
- Build the Docker image
- Start the container
- Mount the repository
- Configure the Python interpreter automatically

This provides a fully configured development environment including:
- Python
- Required build dependencies
- Optional propagation model backends (e.g., bellhopcuda)

#### Using the Container Without VS Code (CLI Workflow)

You can build and run the container manually:
```bash
docker build -t blue-pebble-dev .
docker run -it --rm -v $(pwd):/workspace blue-pebble-dev
```

On Windows PowerShell:
```bash
docker run -it --rm -v ${PWD}:/workspace blue-pebble-dev
```

This starts an interactive shell inside the container.

## Citation

If you use Blue Pebble in academic work, please cite the associated conference paper and the software release (via DOI when available).

```bibtex
@inproceedings{wakefield2026sonar,
  title={A Sonar Signal Processing Plugin for Stone Soup},
  author={Wakefield, Joshua J and Boulton, Finley and Colquitt, Daniel J. and Ralph, Jason F. and Williams, Duncan P.},
  booktitle={2026 29th International Conference on Information Fusion (FUSION)},
  pages={1--8},
  year={2026},
  organization={IEEE}
}
```

## License

Blue Pebble is licensed under the MIT license.

See `LICENSE` and `NOTICE.md` for details.

## Third-Party Software

Optional backends (e.g., bellhopcuda) are licensed separately.

Blue Pebble does not distribute these components in its PyPI package. Users are responsible for complying with the licenses of any external tools they install.


## Future Enhancements

Planned and potential extensions include:

### Environmental Modelling
- Integration of real environmental datasets (bathymetry, range-dependent sound speed profiles)
- Coherent ambient noise modelling (wind, rain, wave-induced noise)
- Improved acoustic volume attenuation and boundary loss modelling
- Systematic environmental uncertainty modelling (sound speed and sensor position errors)

### Signal and Source Modelling
- Incorporation of measured source signatures
- Expanded source directivity modelling
- Additional sensing geometries (hull-mounted arrays, sonobuoys, distributed arrays)

### Detection and Performance Analysis
- Alternative SNR and beam power outputs (e.g., angle-dependent CFAR variants)

### Extended Sensing Modalities
- Active sonar modelling
- Multistatic and bistatic configurations
