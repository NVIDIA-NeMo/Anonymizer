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
```

`make setup` installs Mise when needed, then runs the default `dev` setup profile. If Mise is already installed, run the
same onboarding flow directly:

```bash
mise run setup
```

The default profile installs pinned tools, development dependencies, and repository hooks. Select another profile during
onboarding, or synchronize one profile later:

```bash
mise run setup docs
mise run setup notebooks
mise run deps:sync docs
mise run deps:sync notebooks
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
mise run setup                  # tools, dev dependencies, and repository hooks
mise run setup all              # tools, every dependency group, and repository hooks
mise run deps:sync docs         # synchronize dev + docs dependencies
mise run deps:sync notebooks    # synchronize dev + notebook dependencies
mise run hooks:install          # reinstall repository hooks
mise run check                  # read-only format, lint, type, lock, and SPDX checks
mise run check ::: test         # read-only checks plus unit tests
mise run test                   # unit tests
mise run test:all               # unit and opt-in end-to-end tests (credentials may be required)
mise run test:coverage          # unit tests with coverage report
mise run docs:build             # strict docs build
mise run docs:serve             # local docs server
mise run notebooks:execute      # execute sources and replace generated notebooks
```

`setup` and `deps:sync` use `uv sync --locked` and fail when `uv.lock` does not match the project metadata. After changing
dependencies in `pyproject.toml`, run `mise run lock:update`, review the lockfile diff, then rerun the required setup or
dependency profile.

Task names follow `<domain>[:<action>[:<qualifier>...]]`. Colons separate concepts; public task names do not use hyphens
or underscores. Run `mise tasks` for the complete tree. Commands containing `check` leave tracked files unchanged.

The Makefile only exposes `help`, `install-mise`, and `setup`. Developer commands belong in `.mise/tasks/`.

## Validation Before Opening a PR

Run the smallest useful check while iterating, then run the full relevant set before requesting review.

For most code changes, run the local pre-PR gate:

```bash
mise run check ::: test
```

For changes that affect coverage-sensitive code:

```bash
mise run test:coverage
```

For end-to-end behavior:

```bash
mise run test:e2e
```

To run both the unit and end-to-end suites:

```bash
mise run test:all
```

For docs changes:

```bash
mise run docs:build
```

For tutorial source changes:

```bash
mise run notebooks:execute
mise run docs:build
```

`mise run notebooks:execute` executes `docs/notebook_source/*.py` and replaces generated notebooks in `docs/notebooks/`.
It may require model-provider credentials. Review the generated notebook diffs before committing them.
Tasks that use `uv run --locked --group <profile>` synchronize that locked profile before running. Use `deps:sync` when
you want to prepare a profile without running another task.

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
mise run test:coverage
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
mise run check:format
mise run check:lint
```

Ruff formats and lints tracked Python files and rendered notebooks. The ty configuration includes `docs`, so the
blocking type check also checks code cells in `docs/notebooks/*.ipynb`.

Run all read-only checks:

```bash
mise run check
```

`mise run check` runs `mise run check:format`, `mise run check:lint`, `mise run check:type`, `mise run check:lock`, and
`mise run check:license:headers`. CI runs the same tasks as separate steps so failures identify the affected stage.

Run the blocking type checker:

```bash
mise run check:type
```

`ty` checks `src`, `tests`, `tests_e2e`, `docs`, `scripts`, and `tools`. Errors and warnings fail locally and in CI.

Check lockfile freshness:

```bash
mise run check:lock
```

Regenerate the lockfile after an intentional dependency change:

```bash
mise run lock:update
```

Check or repair SPDX headers:

```bash
mise run check:license:headers
mise run license:headers:fix
```

## Pre-Commit Hooks

Install hooks once:

```bash
mise run hooks:install
```

Before Git records a commit, the hooks check file hygiene, format and lint staged Python files, repair SPDX headers, and
run the repository-wide `mise run check`. That aggregate includes the blocking ty check and read-only lock verification.
The commit-message hook rejects commits without a DCO `Signed-off-by` line.

If a hook changes a file, review the change, stage it again, and retry the commit. If lock verification fails after a
dependency change, run `mise run lock:update` and review the result. Do not bypass repository hooks with
`git commit --no-verify`.

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
mise run notebooks:execute
```

Notebook execution can require configured model provider credentials. If notebooks cannot be regenerated locally, state
that clearly in the PR and include the exact failure mode. Do not save credentials or credential-bearing outputs in
generated notebooks.

## Releases

Build a wheel locally:

```bash
mise run build:wheel
```

Release tags use `vMAJOR.MINOR.PATCH` for stable releases and `vMAJOR.MINOR.PATCHrcN` for release candidates, while the Python package version is the unprefixed version.

Release publishing is handled by `.github/workflows/release.yml`.
