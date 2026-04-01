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

## Optional: Ray Tracing with rtrs

Blue Pebble supports ray traced propagation via the rtrs package.

rtrs is currently private, but will be made public in the near future. In the meantime, provided you have access it can be installed as follows.

Clone the repository:
```bash
git clone https://github.com/fincb/rtrs.git
```

With a Blue Pebble virtual environment activated, install with pip:
```bash
pip install -e /path/to/rtrs
```
