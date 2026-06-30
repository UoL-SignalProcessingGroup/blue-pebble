# Blue Pebble

[![PyPI version](https://img.shields.io/pypi/v/bluepebble.svg)](https://pypi.org/project/bluepebble/)
[![Python versions](https://img.shields.io/pypi/pyversions/bluepebble.svg)](https://pypi.org/project/bluepebble/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/UoL-SignalProcessingGroup/blue-pebble/blob/main/LICENSE)

**Blue Pebble** is a research-oriented simulation framework for underwater acoustic sensing, currently focused on passive sonar signal processing, acoustic propagation modelling, beamforming, detection, and multi-target tracking.

Designed as a plugin for [Stone Soup](https://stonesoup.rtfd.io/), Blue Pebble supports research in:

- Underwater acoustics
- Passive sonar signal processing
- Towed array modelling
- Acoustic propagation modelling
- Beamforming and detection theory
- Target tracking and data association

Blue Pebble provides modular acoustic propagation backends, ranging from analytical spreading laws to external ray-tracing solvers (e.g., rtrs), enabling trade-offs between physical fidelity and computational efficiency.

> **Naming conventions:** The project is referred to as **Blue Pebble** throughout documentation. The PyPI package and import name are both **bluepebble** (e.g., `pip install bluepebble`, `import bluepebble`).

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
- Analytical spreading models and external ray-tracing integration (e.g., rtrs)  
- Configurable source signature synthesis  
- Ambient, biological, and ownship noise modelling  
- Multiple beamforming algorithms  
- Detection algorithms with performance metrics  
- Passive sonar simulation pipelines  
- Integration of real environmental datasets (bathymetry, range-dependent sound speed profiles)
- Incorporation of measured source signatures
- Native integration with Stone Soup tracking workflows  
- Plotting utilities for bearings and Cartesian tracks  
- Notebook-based tutorials and worked examples

## Installation

### Basic Installation (Core Models Only)

```bash
pip install bluepebble
```

This installs the core framework with built-in propagation models.

### Optional: Ray Tracing with rtrs

Blue Pebble supports ray traced propagation via the rtrs package.

rtrs is currently private, but will be made public in the near future. In the meantime, provided you have access it can be installed as follows.

#### Installing rtrs

Clone the repository:
```bash
git clone https://github.com/fincb/rtrs.git
```

With a Blue Pebble virtual environment activated, install with pip:
```bash
pip install -e /path/to/rtrs
```

## Development

### Recommended: Dev Container (Easiest Setup)

For a fully configured development environment, use the included Dev Container.

#### Requirements

- Docker Engine (Docker Desktop on Windows/macOS, or Docker on Linux)

Optional:
- Visual Studio Code
- VS Code Dev Containers extension

Clone the repository:
```bash
git clone https://github.com/UoL-SignalProcessingGroup/blue-pebble.git
```

#### Using the Dev Container (VS Code Workflow)

If using Visual Studio Code with the Dev Containers extension:
```bash
cd blue-pebble
code .
```

To include the optional rtrs backend, set `RTRS_URL` in your shell before opening VS Code:
```bash
export RTRS_URL="https://<token>@github.com/fincb/rtrs.git"
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
- Optional propagation model backends (e.g., rtrs)

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

To include the optional rtrs backend, pass its URL as a build argument:
```bash
docker build --build-arg RTRS_URL="https://<token>@github.com/fincb/rtrs.git" -t blue-pebble-dev .
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

## Third-Party Components

**Software:** The optional rtrs backend is licensed under the MIT licence and is not distributed with Blue Pebble. Users are responsible for installing it separately.

**Data:** Blue Pebble does not distribute external data in its PyPI package. Users are responsible for complying with the licences of any external data they utilise.

## Future Enhancements

Planned and potential extensions include:

### Environmental Modelling
- Coherent ambient noise modelling (wind, rain, wave-induced noise)
- Systematic environmental uncertainty modelling (sound speed and sensor position errors)

### Signal and Source Modelling
- Expanded source directivity modelling

### Detection and Performance Analysis
- Alternative SNR and beam power outputs (e.g., angle-dependent CFAR variants)
- Bearing × time × frequency output volume to support multi-band downstream processing
- Multi-band detector operating across frequency bands simultaneously
- 2D CFAR with training cells spanning both bearing and time, giving the detector access to a limited time history

### Sensing Modalities
- Active sonar modelling
- Multistatic and bistatic configurations
- Additional sensing geometries (hull-mounted arrays, sonobuoys, distributed arrays)
- Explicit hydrophone modelling
