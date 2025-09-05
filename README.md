Nereus is intended as a high-fidelity simulation framework for underwater acoustic scenarios, built using the components of Stone Soup. It provides a toolkit for generating realistic sensor data and testing state estimation algorithms.

At its core, the framework features a multi-body kinematic model of a flexible towed hydrophone array, simulating the 'follow-the-leader' dynamics of a vessel towing a chain of sensors in 3D space. Targets are modelled as acoustic objects, each with a dynamic ground truth path and an acoustic signature defined by a spectrum of discrete tonal frequencies.

## Features

- Acoustic propagation
- Source signature generation
- Ambient noise fields
- Beamforming
- Detection algorithms

## Getting Started (Recommended Method: Dev Container)

The easiest way to get started is by using the included Dev Container, which sets up a complete, pre-configured development environment with all dependencies, including the Bellhop acoustic model.

### Prerequisites

1.  **Docker Desktop**: Install it from the [official Docker website](https://www.docker.com/products/docker-desktop/).
2.  **Visual Studio Code**: Install it from the [official VS Code website](https://code.visualstudio.com/).
3.  **VS Code Dev Containers Extension**: Install this from the [VS Code Marketplace](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers).

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
    A pop-up will appear in the bottom-right corner asking to "Reopen in Container". Click it.

That's it! VS Code will build the container and connect to it. Your environment is now ready with all dependencies, including the compiled Bellhop executable.

---

## Manual Installation (Alternative Method)

If you prefer not to use Docker, you can set up the project manually by following these steps.

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
    Install the project in editable mode, which will also install all its dependencies.
    ```bash
    pip install -e .
    ```

### Bellhop Acoustic Model Dependency

This project uses the **Bellhop** acoustic ray tracing model. If you are installing manually, you must install a compatible version of Bellhop yourself, as the executable is not distributed with this project.

We recommend using the **bellhopcxx / bellhopcuda** project, which is a modern, multithreaded C++/CUDA port of the original Bellhop.

#### Installation Steps:

**For Windows Users (Recommended Method):**

You can download pre-compiled binaries directly, which is the easiest way to get started.

1.  **Download a Release:** Go to the [**bellhopcuda Releases Page**](https://github.com/A-New-BellHope/bellhopcuda/releases) on GitHub.
2.  **Get the Executable:** Download the latest release `.zip` file and extract it. Inside, you will find the `bellhopcxx.exe` file.
3.  **Make it Accessible:** Proceed to the "Make the Executable Accessible" step below.

**For Linux, macOS, or Windows Users (Build from Source):**

If you are not on Windows or want to build the latest version from source, follow these steps.

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/A-New-BellHope/bellhopcuda.git
    ```
2.  **Follow Build Instructions:**
    Navigate into the cloned directory and follow the build instructions provided in their `README.md`.
3.  **Make it Accessible:**
    Proceed to the next step.

#### Make the Executable Accessible

Once you have the Bellhop executable (`bellhopcxx.exe` or `bellhopcxx`), you must make it accessible to Nereus. You have two options:

* **Option A: Add to System PATH**
    The best approach is to add the directory containing the executable to your system's PATH environment variable. This allows Nereus and other programs to find it automatically.

* **Option B: Place in Project Directory**
    Alternatively, you can copy the executable directly into the `nereus/models/propagation/` directory of this project. Note that you will need to do this manually, and the file will not be tracked by Git.

The Bellhop model will now be accessible.