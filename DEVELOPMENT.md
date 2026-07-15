# Development Guide

This guide covers local setup, common development commands, testing, documentation, notebooks, and validation before opening a pull request. For contribution policy and PR expectations, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Prerequisites

- Python 3.11+
- Git
- [mise](https://mise.jdx.dev/) for pinned development tools and task execution
- [uv](https://docs.astral.sh/uv/) for dependency management
- [gh](https://cli.github.com/) for optional GitHub CLI workflows

Development tools such as Ruff, ty, pre-commit, pytest, and pytest-cov are installed by the development dependency group.

## Local Setup

Clone the repository and install development dependencies:

```bash
git clone https://github.com/<your-username>/Anonymizer.git
cd Anonymizer
make setup
mise run bootstrap
```

Install docs or notebook dependencies when needed:

```bash
mise run install-dev-docs
mise run install-dev-notebooks
```

Install pre-commit hooks once after cloning:

```bash
mise run install-pre-commit
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

Common mise tasks:

```bash
mise run bootstrap              # install dev dependencies
mise run install-dev-docs       # install dev + docs dependencies
mise run install-dev-notebooks  # install dev + notebook dependencies
mise run install-pre-commit     # install pre-commit hooks
mise run check                  # read-only format, lint, typecheck, lock, SPDX checks
mise run test                   # unit tests
mise run coverage               # unit tests with coverage report
mise run docs:build             # strict docs build
mise run docs:serve             # local docs server
mise run convert-notebooks      # regenerate tutorial notebooks
```

The Makefile retains compatibility targets for existing scripts. New development commands belong in `.mise/tasks/`.

## Validation Before Opening a PR

Run the smallest useful check while iterating, then run the full relevant set before requesting review.

For most code changes:

```bash
mise run check
mise run test
```

For changes that affect coverage-sensitive code:

```bash
mise run coverage
```

For end-to-end behavior:

```bash
mise run test:e2e
```

For docs changes:

```bash
mise run install-dev-docs
mise run docs:build
```

For tutorial source changes:

```bash
mise run install-dev-notebooks
mise run convert-notebooks
mise run docs:build
```

`mise run convert-notebooks` executes `docs/notebook_source/*.py` and writes generated notebooks to `docs/notebooks/`. Review the generated notebook diffs before committing them.

## Testing

Run all unit tests:

```bash
mise run test
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
mise run coverage
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
mise run format
mise run format-check
```

Run all read-only checks:

```bash
mise run check
```

Run the advisory type checker:

```bash
mise run typecheck
```

`ty` is currently advisory. Treat new type-check findings in touched code as review-worthy even when CI does not block on them.

Check lockfile freshness:

```bash
mise run lock-check
```

Check or repair SPDX headers:

```bash
mise run copyright-check
mise run copyright
```

## Pre-Commit Hooks

Install hooks once:

```bash
mise run install-pre-commit
```

Hooks run Ruff format and lint, uv lock verification, DCO signoff checks, and basic file hygiene. `ty` is installed
for `mise run typecheck` and `mise run check`, but it is not currently run as a pre-commit hook.

If `pyproject.toml` changes and `uv.lock` is stale, the uv-lock hook may regenerate `uv.lock` and fail the commit. Add the updated `uv.lock` and retry.

## Secrets and Credentials

Do not commit API keys, service tokens, private keys, passwords, real endpoint secrets, or credential-bearing logs.

Use environment variables, local `.env` files, or GitHub Actions secrets for credentials. `.env` and `.env.*` are ignored
by Git in this repository, but still review diffs before committing.

If a secret is committed or pushed by mistake, treat it as compromised: rotate or revoke it, then remove it from the
repository history before sharing the branch further.

## Documentation

Serve docs locally:

```bash
mise run docs:serve
```

Build docs in strict mode:

```bash
mise run docs:build
```

Update docs when a change affects public API behavior, CLI behavior, examples, notebooks, configuration, contributor workflow, or release process.

## Notebooks

Tutorial notebooks are generated from Python sources:

- Source files: `docs/notebook_source/*.py`
- Generated notebooks: `docs/notebooks/*.ipynb`

When editing tutorial sources, regenerate notebooks with:

```bash
mise run convert-notebooks
```

Notebook execution can require configured model provider credentials. If notebooks cannot be regenerated locally, state
that clearly in the PR and include the exact failure mode. Do not save credentials or credential-bearing outputs in
generated notebooks.

## Releases

Build a wheel locally:

```bash
mise run build-wheel
```

Release tags use `vMAJOR.MINOR.PATCH` for stable releases and `vMAJOR.MINOR.PATCHrcN` for release candidates, while the Python package version is the unprefixed version.

Release publishing is handled by `.github/workflows/release.yml`.
