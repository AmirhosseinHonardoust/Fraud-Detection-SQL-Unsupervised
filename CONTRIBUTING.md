# Contributing

## Setup

```bash
pip install -r requirements-dev.txt
pre-commit install   # optional but recommended: runs the gate on every commit
```

## Before opening a PR

Run the same quality gate CI runs (lint, format check, type check, tests +
coverage):

```bash
make gate
```

or individually: `make lint`, `make format-check`, `make typecheck`, `make test`,
`make coverage`. All must pass — CI enforces the same checks on Python 3.11 and
3.12.

## Guidelines

- Keep changes minimal and focused; avoid unrelated renames or file moves.
- Add or update tests for any behavior change. `make coverage` fails if
  coverage on `src/` drops below the threshold in `pyproject.toml`.
- For any refactor, show that behavior is unchanged (e.g. recompute outputs
  before/after and diff them) in the PR description.
- Match existing patterns in the codebase (type hints, docstring style, the
  try/except lazy-import pattern in `src/detect_fraud_unsupervised.py`).
