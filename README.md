# Nereus

Nereus is a high-fidelity simulation framework for underwater acoustic scenarios, built as a plugin for the [Stone Soup](https://stonesoup.rtfd.io/) tracking library. It provides tools for generating realistic sensor data, simulating flexible towed hydrophone arrays, and evaluating state estimation algorithms.

## Features

- Multi-body kinematic model for flexible towed arrays ("follow-the-leader" dynamics)
- Acoustic propagation models (Bellhop, cylindrical, spherical)
- Source signature generation (multi-tone, configurable)
- Biological source simulation (e.g., whale calls, snapping shrimp)
- Ambient noise field simulation (white/pink noise)
- Beamforming (delay-and-sum, frequency domain)
- Detection algorithms (CFAR, peak, threshold)
- Passive sonar simulation and detection chain
- Integration with Stone Soup for tracking and data association
- Plotting utilities for bearings and Cartesian tracks
- Example Jupyter notebooks for scenario setup, simulation, and tracking

## Getting Started (Recommended Method: Dev Container)

The easiest way to get started is by using the included Dev Container, which sets up a complete, pre-configured development environment with all dependencies, including the Bellhop acoustic model.

### Prerequisites

1.  **Docker Desktop**: [Download here](https://www.docker.com/products/docker-desktop)
2.  **Visual Studio Code**: [Download here](https://code.visualstudio.com/)
3.  **VS Code Dev Containers Extension**: [Install from Marketplace](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

### Launching the Environment

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/jjwakefield/nereus.git
    cd nereus
    ```
2.  **Open in VS Code**:
    ```bash
    code .
    ```
3.  **Reopen in Container**:
    When prompted, click "Reopen in Container" in the bottom-right corner.

VS Code will build the container and connect to it. Your environment is now ready with all dependencies, including the compiled Bellhop executable.

---

## Manual Installation (Alternative Method)

If you prefer not to use Docker, you can set up the project manually:

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/jjwakefield/nereus.git
    cd nereus
    ```
2.  **Create and Activate a Virtual Environment** (Recommended):
    ```bash
    conda create --name nereus-env python=3.12
    conda activate nereus-env
    ```
3.  **Install Nereus**:
    ```bash
    pip install -e .
    ```

### Bellhop Acoustic Model Dependency

This project uses the **Bellhop** acoustic ray tracing model. If installing manually, you must install a compatible Bellhop executable yourself.

We recommend [bellhopcuda](https://github.com/A-New-BellHope/bellhopcuda), a modern C++/CUDA port.

#### Installation Steps

**Windows Users:**  
Download pre-compiled binaries from the [bellhopcuda Releases Page](https://github.com/A-New-BellHope/bellhopcuda/releases).

**Linux/macOS/Windows (Build from Source):**
```bash
git clone https://github.com/A-New-BellHope/bellhopcuda.git
cd bellhopcuda
# Follow build instructions in their README.md
```

#### Make the Executable Accessible

Once you have the Bellhop executable (`bellhopcxx.exe` or `bellhopcxx`):

- **Option A:** Add its directory to your system PATH.
- **Option B:** Copy it into `nereus/models/propagation/` (not tracked by Git).

## Usage

- See the `examples/` directory for Jupyter notebooks demonstrating scenario setup, simulation, detection, and tracking.
- The main modules are:
    - [`nereus.platform.TowedArrayPlatform`](nereus/platform/towedarray.py)
    - [`nereus.simulator.PassiveSonarArraySimulator`](nereus/simulator/acoustic.py)
    - [`nereus.detector.PassiveSonarDetector`](nereus/detector/passive.py)
    - [`nereus.plotter.BearingsPlotter`](nereus/plotter.py), [`nereus.plotter.CartesianPlotter`](nereus/plotter.py)

## Documentation

- Build API docs with Sphinx:  
    ```sh
    cd docs
    make html
    ```
- See [docs/source/index.rst](docs/source/index.rst) for structure.

## License

MIT License. See [LICENSE](LICENSE) and [NOTICE.MD](NOTICE.MD) for details and attributions.

---