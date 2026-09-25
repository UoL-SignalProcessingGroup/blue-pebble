# Blue Pebble

[![PyPI version](https://img.shields.io/pypi/v/blue-pebble.svg)](https://pypi.org/project/blue-pebble/)
[![Python versions](https://img.shields.io/pypi/pyversions/blue-pebble.svg)](https://pypi.org/project/blue-pebble/)
[![Documentation](https://readthedocs.org/projects/blue-pebble/badge/?version=latest)](https://blue-pebble.readthedocs.io/en/latest/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/UoL-SignalProcessingGroup/blue-pebble/blob/main/LICENSE)

**Blue Pebble** is a research-oriented simulation framework for underwater acoustic sensing, built as a plugin for [Stone Soup](https://stonesoup.rtfd.io/). It currently focuses on passive sonar: acoustic propagation modelling, beamforming, detection and multi-target tracking.

## Installation

```bash
pip install blue-pebble
```

The package installs as `blue-pebble` and imports as `bluepebble`.

## Getting Started

Start with the [Getting Started tutorial](https://blue-pebble.readthedocs.io/en/latest/auto_tutorials/getting_started.html), which simulates a towed array and beamforms, detects and tracks a single target. The [documentation](https://blue-pebble.readthedocs.io/en/latest/) also has further tutorials, worked examples and the API reference.

## Features

- Towed-array kinematics for the ownship, targets and array elements
- Acoustic propagation, from analytical spreading laws to ray tracing with [rtrs](https://pypi.org/project/rtrs/)
- Environments built from analytical profiles or measured bathymetry and sound-speed data
- Source and noise synthesis, including biological, anthropogenic and ownship noise
- Delay-and-sum and MVDR (Minimum Variance Distortionless Response) beamforming
- CFAR (Constant False Alarm Rate) detection, calibrated from noise or with a fixed threshold
- Detections that feed straight into Stone Soup trackers

## Contributing

See [CONTRIBUTING.md](https://github.com/UoL-SignalProcessingGroup/blue-pebble/blob/main/CONTRIBUTING.md) for the development setup and workflow, and the [roadmap](https://blue-pebble.readthedocs.io/en/latest/roadmap.html) for planned extensions.

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

## Licence

Blue Pebble is released under the MIT licence; see [LICENSE](https://github.com/UoL-SignalProcessingGroup/blue-pebble/blob/main/LICENSE) and [NOTICE.md](https://github.com/UoL-SignalProcessingGroup/blue-pebble/blob/main/NOTICE.md). Its ray-tracing dependency, [rtrs](https://pypi.org/project/rtrs/), is also MIT-licensed and installs automatically. The package includes no external data, so users are responsible for complying with the licences of any data they use with it.
