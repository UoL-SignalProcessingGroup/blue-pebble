# Installation

## Core package

Install the core package with pip:

```bash
pip install blue-pebble
```

This installs the core framework including all propagation backends (analytical models and ray tracing via rtrs).

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

