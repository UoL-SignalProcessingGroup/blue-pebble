# Blue Pebble

**Blue Pebble** is a research-oriented simulation framework for underwater acoustic sensing, currently focused on passive sonar signal processing, acoustic propagation modelling, beamforming, detection, and multi-target tracking.

Designed as a plugin for [Stone Soup](https://stonesoup.rtfd.io/), Blue Pebble supports research in:

- Underwater acoustics
- Passive sonar signal processing
- Towed array modelling
- Acoustic propagation modelling
- Beamforming and detection theory
- Target tracking and data association

---

## Research Applications

Blue Pebble is intended for controlled, simulation-based studies, including:

- Evaluation of tracking and data association algorithms  
- End-to-end sonar performance analysis  
- Synthetic dataset generation for validation  
- Sensitivity analysis of propagation effects on detection and estimation  

Although current functionality centres on passive sonar, the architecture supports extension to additional modalities (e.g., active or multistatic configurations).

---

## Architecture

Blue Pebble follows a modular design that separates physical modelling from signal processing and tracking logic. Core components include:

- **Platform dynamics** - Kinematic modelling of ownship, targets, and arrays  
- **Acoustic propagation** - Pluggable propagation backends (analytical or external solvers)  
- **Signal generation** - Source modelling and noise synthesis  
- **Beamforming** - Array processing algorithms  
- **Detection** - Measurement formation and statistical thresholding  
- **Tracking** - Integration with Stone Soup estimators and data association  

This separation enables systematic experimentation across modelling assumptions and algorithmic choices without tightly coupling components.

---

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
- Gallery-based tutorials and worked examples

```{toctree}
:maxdepth: 2
:caption: Contents

installation
getting_started
auto_tutorials/index
auto_examples/index
api/index
development
testing
citation
roadmap
```
