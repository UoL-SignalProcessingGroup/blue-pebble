# Contributing to Nereus

Nereus follows Stone Soup-style contribution practices, adapted for this plugin and toolchain.

## Development Setup

1. Clone the repository and enter it.
2. Create and activate a Python 3.10+ virtual environment.
3. Install development dependencies:

```bash
python3 -m pip install -U pip
python3 -m pip install -e ".[dev]"
```

4. If working on docs/examples, also install:

```bash
python3 -m pip install -e ".[docs,examples]"
```

## Code Style

- Use clear, descriptive names (prefer domain clarity over shorthand).
- Use standard Python exceptions/warnings and validate inputs explicitly.
- Prefer a deprecation period of at least one release cycle before removing/changing public interfaces.
- Keep components modular with one primary responsibility each.
- Follow PEP 8 and project linting rules.
- Lint with `ruff` (this project uses `ruff`, not `flake8`):

```bash
ruff check .
```

## Documentation

- Use NumPy-style docstrings for public APIs.
- Add or update docs for behavior/API changes.
- For significant new capability, include a minimal reproducible example in `docs/examples/` or `docs/tutorials/`.
- Where relevant, include references to equations, assumptions, or papers.

## Tests

- Use `pytest`.
- Add tests for new behavior and regressions.
- Add tests for expected failures/exceptions where relevant.
- For numerical methods, use deterministic checks and appropriate tolerances (`rtol`, `atol`).

Run local checks before opening a PR:

```bash
ruff check .
pytest
```

## License

By contributing, you agree your contribution is under the repository license (MIT), unless explicitly stated otherwise in the pull request.

## External Dependencies

- Prefer Python standard library or existing well-maintained libraries.
- New dependencies should have permissive or weak-copyleft licensing and a clear maintenance story.
- Add dependencies to `pyproject.toml` with appropriate version constraints.

## Pull Requests

- Use feature branches (for example `feat/<topic>` or `fix/<topic>`).
- Keep PRs focused and reviewable.
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
