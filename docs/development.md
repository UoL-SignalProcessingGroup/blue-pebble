# Development

Blue Pebble is developed as a Stone Soup-compatible plugin, so documentation and examples should remain aligned with the public package surface and the surrounding scientific workflow.

## Common checks

```bash
ruff check .
pytest
python -m build --sdist --wheel
make -C docs html
```

## Documentation workflow

- Public APIs should use NumPy-style docstrings.
- Notebook files under `docs/tutorials/` and `docs/examples/` are the canonical published sources.
- Sphinx renders the committed notebook outputs directly; the docs build does not execute notebooks.
- When notebook content changes, refresh the outputs locally before committing so the published documentation remains coherent.

## Development environment

The preferred fully configured environment is the existing dev container in `.devcontainer/devcontainer.json`.

### Requirements

- Docker Engine

Optional:

- Visual Studio Code
- the VS Code Dev Containers extension

### Using the dev container in VS Code

Clone the repository and open it in Visual Studio Code:

```bash
git clone https://github.com/jjwakefield/blue-pebble.git
cd blue-pebble
code .
```

When prompted, select **Reopen in Container**.

This provides a configured environment including Python, the required build dependencies, and the optional tooling used by the project.

### Using the container from the command line

If you prefer a CLI workflow, build and run the container manually:

```bash
docker build -t blue-pebble-dev .
docker run -it --rm -v $(pwd):/workspace blue-pebble-dev
```

On Windows PowerShell:

```bash
docker run -it --rm -v ${PWD}:/workspace blue-pebble-dev
```

This starts an interactive shell inside the container.
