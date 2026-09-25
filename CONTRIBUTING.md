# Contributing to Blue Pebble

Blue Pebble follows Stone Soup-style contribution practices, adapted for this plugin and toolchain.

## Development Setup

1. Clone the repository and enter it.
2. Create and activate a Python 3.11+ virtual environment.
3. Install development dependencies:

```bash
python3 -m pip install -U pip
python3 -m pip install -e ".[dev]"
```

4. If working on docs/examples, also install:

```bash
python3 -m pip install -e ".[docs,examples]"
```

### Dev Container

Alternatively, the dev container in `.devcontainer/` gives a fully configured environment, including `rtrs`. It needs Docker Engine (Docker Desktop on Windows and macOS). Visual Studio Code with the Dev Containers extension is optional.

In VS Code, open the repository and select **Reopen in Container** when prompted:

```bash
git clone https://github.com/UoL-SignalProcessingGroup/blue-pebble.git
cd blue-pebble
code .
```

Without VS Code, build and run the container from the command line, which starts an interactive shell inside it:

```bash
docker build -t blue-pebble-dev .
docker run -it --rm -v $(pwd):/workspace blue-pebble-dev
```

On Windows PowerShell, use `${PWD}` in place of `$(pwd)`.

## Code Style

- British English
- Use clear, descriptive names (prefer domain clarity over shorthand).
- Use standard Python exceptions/warnings and validate inputs explicitly.
- Deprecate public interfaces for at least one release cycle before removing or changing them. Once removed, the old name should raise an error pointing to its replacement (see `_REMOVED` in `bluepebble/detector/__init__.py`).
- Treat each subpackage's `__init__.py` as its public API. `__all__` lists everything public, and code outside a subpackage imports from the subpackage rather than its internal modules.
- Keep components modular with one primary responsibility each.
- Write comments that explain why, not what the code already says.
- Follow PEP 8 and project linting rules.
- Lint with `ruff` (this project uses `ruff`, not `flake8`):

```bash
ruff check .
```

## Documentation

- British English
- Use NumPy-style docstrings for public APIs.
- Add or update docs for behaviour/API changes.
- For significant new capability, include a minimal reproducible example in `docs/examples/` or `docs/tutorials/`.
- Keep docstrings concise. Say each thing once, and don't restate a default that's already visible in the signature.
- Docstrings for a model or algorithm should give its key equations and list its assumptions under their own `Assumptions` heading.
- Where relevant, cite papers or references in the docstring of the class or function they support, not only in the module header.
- Spell out an abbreviation once, on first use, e.g. CFAR (Constant False Alarm Rate).
- `.py` files under `docs/tutorials/` and `docs/examples/` are the canonical published sources for documentation.

Build the docs locally with:

```bash
make -C docs html
```

The build executes the tutorials but renders the examples as code only, since running every example takes longer than Read the Docs allows. Sphinx-Gallery caches executed scripts in `docs/source/auto_tutorials` and `docs/source/auto_examples` and only re-runs a script when it changes, so avoid deleting those directories. `make -C docs clean` leaves them in place deliberately.

## Tests

- Use `pytest`.
- Add tests for new behaviour and regressions.
- Add tests for expected failures/exceptions where relevant.
- For numerical methods, use deterministic checks and appropriate tolerances (`rtol`, `atol`).

Run local checks before opening a PR:

```bash
ruff check .
pyright
pytest --cov=bluepebble --cov-report=term-missing --cov-fail-under=55
python -m build --sdist --wheel
make -C docs html
```

CI runs `ruff`, the tests and the package build on Python 3.11 to 3.14, with the coverage gate on 3.12 only. It doesn't run `pyright` or the docs build, so run those locally.

## Licence

By contributing, you agree your contribution is under the repository licence (MIT), unless explicitly stated otherwise in the pull request.

## External Dependencies

- Prefer Python standard library or existing well-maintained libraries.
- New dependencies should have permissive or weak-copyleft licensing and a clear maintenance story.
- Add dependencies to `pyproject.toml` with appropriate version constraints.

## Pull Requests

- Use feature branches (for example `feat/<topic>` or `fix/<topic>`).
- Keep PRs focused and reviewable.
- Use a Conventional Commit-formatted PR title, for example `ci: add pytest step to workflow`. The `pr-title` CI check enforces this.
- Prefer the form `type(scope): summary`; scope is encouraged when it adds clarity, but it is not required.
- Mark a breaking change with `!` in the title, e.g. `feat(detector)!: ...`, and the `breaking` label.
- Label each PR `breaking`, `enhancement`, `bug` or `documentation`. The label decides which section of the generated release notes the PR appears under.
- Prefer squash merging so the PR title becomes the commit message on `main`.
- `main` is protected, so merging needs an approving review and passing checks.
- In the PR description, include:
  - what changed
  - why it changed
  - how you validated it (tests, docs build, example output)
  - any compatibility impact

## Versioning and Releases

This project uses dynamic versioning via `setuptools_scm`.

- Do not manually set `project.version` in `pyproject.toml`.
- Versions are resolved from Git tags.
- Use semantic tags for releases, for example `v3.0.0`, `v3.1.0`, `v3.1.1`, or pre-release tags such as `v3.0.0rc1`.
- Publishing a GitHub Release builds the package and uploads it to PyPI through trusted publishing (`.github/workflows/publish.yml`). There's no manual upload step, and PyPI never lets a version number be reused, so check everything before publishing.
- Release notes are generated from the merged PRs' titles and labels (`.github/release.yml`). Add migration notes for breaking changes by hand.
