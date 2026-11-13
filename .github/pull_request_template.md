# Pull Request

Commit messages and the PR title follow
[Conventional Commits](https://www.conventionalcommits.org/): `type(scope): summary`. The
`pr-title` check enforces this on the PR title.

- Types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`
- Scope is encouraged but not required.
- Add `!` after the type or scope for a breaking change, e.g. `feat(detector)!: ...`

Examples:
- `ci: add pytest step to workflow`
- `fix(plotter): handle empty detections`
- `docs(api): clarify rtrs installation`

## Summary

- What changed?
- Why did it change?

## Scientific validation

How do we know the results are right, not just that the code runs? Delete this section if the
change cannot affect numerical output (e.g. CI, docs wording).

- Method: what model or algorithm does this implement or change, and which reference is it from?
- Evidence: what was it compared against (analytical result, reference implementation,
  published figure, previous output), and with what tolerance?
- Refactors and optimisations: do outputs reproduce the previous version to rounding?
- Assumptions and limitations: are any new ones stated in the docstrings?

## Checks

- `ruff check .`
- `pyrefly check`
- `pytest`
- `make -C docs html`

## Compatibility

- Any public API, behavioural, or documentation impact?
