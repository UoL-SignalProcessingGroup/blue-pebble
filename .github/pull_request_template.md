# Pull Request

PR title format: `type(scope): summary`

Examples:
- `ci: add pytest step to workflow`
- `fix(plotter): handle empty detections`
- `docs(api): clarify rtrs installation`

Scope is encouraged but not required.

## Summary

- What changed?
- Why did it change?

## Validation

- `ruff check .`
- `pytest`
- `make -C docs html`

## Compatibility

- Any public API, behavioural, or documentation impact?
