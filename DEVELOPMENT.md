# Development Guide

This guide covers local setup, common development commands, testing, documentation, notebooks, and validation before opening a pull request. For contribution policy and PR expectations, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Prerequisites

- Python 3.11+
- Git
- [uv](https://docs.astral.sh/uv/) for dependency management
- [gh](https://cli.github.com/) for optional GitHub CLI workflows

Development tools such as Ruff, ty, pre-commit, pytest, and pytest-cov are installed by the development dependency group.

## Local Setup

Clone the repository and install development dependencies:

```bash
git clone https://github.com/<your-username>/Anonymizer.git
cd Anonymizer
make bootstrap
```

Install docs or notebook dependencies when needed:

```bash
make install-dev-docs
make install-dev-notebooks
```

Install pre-commit hooks once after cloning:

```bash
make install-pre-commit
```

If you work from a fork, add the upstream remote:

```bash
git remote add upstream https://github.com/NVIDIA-NeMo/Anonymizer.git
```

## Day-to-Day Workflow

Start from the latest `main`:

```bash
git checkout main
git pull --ff-only origin main  # use upstream main when origin is your fork
git checkout -b <username>/<type>/<issue-number>-<short-description>
```

Common Makefile targets:

```bash
make bootstrap              # install dev dependencies
make install-dev-docs       # install dev + docs dependencies
make install-dev-notebooks  # install dev + notebook dependencies
make install-pre-commit     # install pre-commit hooks
make check                  # read-only format, lint, typecheck, lock, SPDX checks
make test                   # unit tests
make coverage               # unit tests with coverage report
make docs-build             # strict docs build
make docs-serve             # local docs server
make convert-notebooks      # regenerate tutorial notebooks
```

## Validation Before Opening a PR

Run the smallest useful check while iterating, then run the full relevant set before requesting review.

For most code changes:

```bash
make check
make test
```

For changes that affect coverage-sensitive code:

```bash
make coverage
```

For end-to-end behavior:

```bash
make test-e2e
```

For docs changes:

```bash
make install-dev-docs
make docs-build
```

For tutorial source changes:

```bash
make install-dev-notebooks
make convert-notebooks
make docs-build
```

`make convert-notebooks` executes `docs/notebook_source/*.py` and writes generated notebooks to `docs/notebooks/`. Review the generated notebook diffs before committing them.

## Testing

Run all unit tests:

```bash
make test
```

Run a specific test file:

```bash
uv run --group dev pytest tests/engine/test_detection_workflow.py
```

Run a specific test:

```bash
uv run --group dev pytest tests/engine/test_detection_workflow.py::test_name
```

Run coverage:

```bash
make coverage
```

Testing expectations:

- New features should include tests for the new behavior.
- Bug fixes should include regression tests.
- Tests should use fabricated data and must not introduce real PII.
- Prefer behavior-focused tests over assertions that depend on private implementation details.
- Mock external model, network, and file-system boundaries rather than internal helpers.

## Code Quality

Format and lint:

```bash
make format
make format-check
```

Run all read-only checks:

```bash
make check
```

Run the advisory type checker:

```bash
make typecheck
```

`ty` is currently advisory. Treat new type-check findings in touched code as review-worthy even when CI does not block on them.

Check lockfile freshness:

```bash
make lock-check
```

Check or repair SPDX headers:

```bash
make copyright-check
make copyright
```

## Pre-Commit Hooks

Install hooks once:

```bash
make install-pre-commit
```

Hooks run Ruff format and lint, ty type checking, uv lock verification, DCO signoff checks, and basic file hygiene.

If `pyproject.toml` changes and `uv.lock` is stale, the uv-lock hook may regenerate `uv.lock` and fail the commit. Add the updated `uv.lock` and retry.

## Documentation

Serve docs locally:

```bash
make docs-serve
```

Build docs in strict mode:

```bash
make docs-build
```

Update docs when a change affects public API behavior, CLI behavior, examples, notebooks, configuration, contributor workflow, or release process.

## Notebooks

Tutorial notebooks are generated from Python sources:

- Source files: `docs/notebook_source/*.py`
- Generated notebooks: `docs/notebooks/*.ipynb`

When editing tutorial sources, regenerate notebooks with:

```bash
make convert-notebooks
```

Notebook execution can require configured model provider credentials. If notebooks cannot be regenerated locally, state that clearly in the PR and include the exact failure mode.

## Releases

Build a wheel locally:

```bash
make build-wheel
```

Release tags use `vMAJOR.MINOR.PATCH`, while the Python package version is the unprefixed version.

Release publishing is handled by `.github/workflows/release.yml`.
