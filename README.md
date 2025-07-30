# Nereus

Nereus is a Python framework for underwater acoustic tracking, derived from the open-source Stone Soup project. It provides a WIP toolkit for state estimation, designed specifically for sonar and underwater acoustic environments. The framework supports various tracking algorithms, such as Kalman filters and particle filters, and integrates with external models like the Bellhop acoustic model for realistic propagation simulations. It also includes built-in signal processing and plotting utilities to aid in the development and analysis of tracking scenarios.

## Features

- Data association
- Bayesian filters
- Track management
- Source and platform models
- Acoustic propagation
- Signal simulation
- Beamforming
- Detection
- Visualisation

## Installation

1.  **Clone the Repository**:
    ```bash
    git clone [https://github.com/jjwakefield/nereus.git](https://github.com/jjwakefield/nereus.git)
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

## Important Notice: Origin of Code

This project is a derivative work of the open-source framework **[Stone Soup](https://stonesoup.rtfd.io/)**. The architecture and significant portions of the source code were adapted from that project. For full attribution and the original license, please see the **[NOTICE.md](NOTICE.md)** file.

## Basic Usage

Here is a minimal example of the Nereus API. For a complete, runnable example that generates a plot, click the "details" section below.

<details>
<summary><b>Click to view the full example script</b></summary>

```python
from datetime import datetime, timedelta

import numpy as np

from nereus.association.hypothesisers import BasicHypothesiser
from nereus.filters.predictors import KalmanPredictor
from nereus.filters.updaters import KalmanUpdater
from nereus.models.measurement_models import LinearGaussianMeasurementModel
from nereus.models.transition_models import (
    CombinedLinearGaussianTransitionModel,
    NearlyConstantVelocity,
)
from nereus.types.detections import TrueDetection
from nereus.types.states import (
    GaussianState,
    GroundTruthPath,
    GroundTruthState,
    StateVector,
    Track,
)
from nereus.utils.plotting import CartesianPlotter

np.random.seed(42)  # For reproducibility

start_time = datetime.now().replace(
    hour=0,
    minute=0,
    second=0,
    microsecond=0,
)
time_interval = timedelta(seconds=1)
num_steps: int = 100
timesteps = [start_time + i * time_interval for i in range(num_steps)]

# 1. Define the initial state of the target (the prior)
# Format is [x, vx, y, vy]
initial_state_vector = StateVector([0, 1, 0, 1])
ndim_state = initial_state_vector.shape[0]

# 2. Define the physical models
# How the target moves (Constant Velocity with some noise)
process_noise_std = 0.1
transition_model = CombinedLinearGaussianTransitionModel(
    [
        NearlyConstantVelocity(process_noise_std, time_interval),
        NearlyConstantVelocity(process_noise_std, time_interval),
    ]
)
ground_truth = GroundTruthPath([GroundTruthState(initial_state_vector, start_time)])
for t in timesteps[1:]:
    # Generate the next state using the transition model
    next_state = transition_model.function(ground_truth[-1], noise=True)
    ground_truth.append(GroundTruthState(next_state, t))

# 3. Define the measurement model
# How we observe the target (we can only measure x and y)
mapping = (0, 2)  # Indices of the state vector we can measure
noise_covariance = np.diag([5, 5]) ** 2  # Measurement noise covariance
measurement_model = LinearGaussianMeasurementModel(
    ndim_state=ndim_state, mapping=mapping, noise_covar=noise_covariance
)
all_detections = []
for t in timesteps:
    detection_set = set()
    for state in ground_truth:
        if state.timestamp == t:
            # Simulate a measurement with noise
            measurement = measurement_model.function(state, noise=True)
            detection_set.add(
                TrueDetection(
                    state_vector=measurement,
                    measurement_model=measurement_model,
                    timestamp=t,
                )
            )

    all_detections.append(detection_set)

# 4. Define the Kalman filter
predictor = KalmanPredictor(transition_model=transition_model)
updater = KalmanUpdater(measurement_model=measurement_model)
hypothesiser = BasicHypothesiser(measurement_model=measurement_model)
prior = GaussianState(
    mean=initial_state_vector + np.random.normal(0, 10, ndim_state),
    covar=np.diag([5] * ndim_state) ** 2,
    timestamp=start_time,
)
track = Track([prior])
for measurement_set in all_detections:
    # 4a. Always predict the next state based on the previous one
    prior = predictor.predict(track[-1], time_interval)

    # 4b. Generate hypotheses for the new set of detections
    hypotheses = hypothesiser.hypothesise(prior, measurement_set)

    # 4c. Check if any hypothesis is based on an actual detection then update the track
    hypothesis = next(hyp for hyp in hypotheses if hyp)
    if hypothesis is not None:
        # If a hypothesis exists, update the track with the new state
        posterior = updater.update(hypothesis)
    else:
        # If no hypothesis, just append the predicted state
        posterior = prior

    # 4d. Add the new state (either updated or not) to the track
    track.append(posterior)

# 5. Plot the results
plotter = CartesianPlotter()
plotter.plot(
    [ground_truth],
    [track],
    all_detections,
    timesteps=timesteps,
    mapping=mapping,
)
plotter.show()

```
</details>

## Dependencies

### Bellhop Acoustic Model

This project uses the **Bellhop** acoustic ray tracing model for underwater propagation modeling. To use this functionality, you must install a compatible version of Bellhop yourself, as the executable is not distributed with this project due to licensing requirements.

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
    git clone [https://github.com/A-New-BellHope/bellhopcuda.git](https://github.com/A-New-BellHope/bellhopcuda.git)
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
    Alternatively, you can copy the executable into the `nereus/models/bellhop/` directory of this project. Note that you will need to do this manually, and the file will not be tracked by Git.

The Bellhop model will now function correctly.