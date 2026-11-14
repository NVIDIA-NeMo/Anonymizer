# Development Guide

This guide covers local setup, common development commands, testing, documentation, notebooks, and validation before opening a pull request. For contribution policy and PR expectations, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Prerequisites

- Python 3.11+
- Git
- [uv](https://docs.astral.sh/uv/) 0.13.4 or newer for dependency management
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
mise run docs:fern:check
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

`make check` runs `make format-check`, `make typecheck`, `make lock-check`, and `make copyright-check`. CI runs the
same four Make targets as separate steps in its `Check` job so failures identify the affected stage.

Run the blocking type checker:

```bash
make typecheck
```

`ty` checks `src`, `tests`, `tests_e2e`, `docs`, `scripts`, and `tools`. Errors and warnings fail locally and in CI.

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

Before Git records a commit, the hooks check file hygiene, format and lint staged Python files, repair SPDX headers,
verify `uv.lock` when `pyproject.toml` changes, and run the repository-wide `make check`. The commit-message hook
rejects commits without a DCO `Signed-off-by` line.

If a hook changes a file, review the change, stage it again, and retry the commit. In particular, the uv-lock hook may
regenerate `uv.lock` when `pyproject.toml` changes. Do not bypass repository hooks with `git commit --no-verify`.

## Secrets and Credentials

Do not commit API keys, service tokens, private keys, passwords, real endpoint secrets, or credential-bearing logs.

Use environment variables, local `.env` files, or GitHub Actions secrets for credentials. `.env` and `.env.*` are ignored
by Git in this repository, but still review diffs before committing.

If a secret is committed or pushed by mistake, treat it as compromised: rotate or revoke it, then remove it from the
repository history before sharing the branch further.

## Documentation

Update docs when a change affects public API behavior, CLI behavior, examples,
notebooks, configuration, contributor workflow, or release process. Fern is being
introduced alongside the production MkDocs site; see the
[temporary migration status](docs/fern/FERN_MIGRATION.md) for parallel authoring
requirements and cutover work.

### MkDocs

```bash
make install-dev-docs
make docs-serve       # local server at http://127.0.0.1:8000
make docs-build       # strict build
```

### Fern Validation and Previews

Fern prose lives in `docs/**/*.mdx`. The site configuration is
`docs/fern/docs.yml`, and nightly navigation is in
`docs/fern/versions/nightly.yml`. Use explicit slugs to keep page URLs stable
when reorganizing the sidebar. Images use the existing source-relative assets;
use native Fern components for cards, tabs, and admonitions.

For Fern, install [mise](https://mise.jdx.dev/) 2026.7.5 or newer, then install
the Fern tools from the repository root:

```bash
mise trust
mise install
```

This installs Node.js 22, Fern CLI 5.201.0, and the check script's jq/yq
dependencies from `.mise.toml` and `mise.lock`. No separate npm installation is needed. Mise is used only for Fern tools
and `docs:fern:*` tasks in this PR. Keep using Make and uv for MkDocs and Python
development. Run these tasks from the repository root:

| Task | Purpose |
| --- | --- |
| `mise run docs:fern:check` | Validate version alignment, configuration, MDX, and internal links. |
| `mise run docs:fern:serve` | Start the local Fern development server. |
| `mise run docs:fern:preview` | Generate a hosted preview; run the check task first. |
| `mise run docs:fern:notebooks` | Regenerate notebook MDX from saved cells and outputs without executing cells. |

Validation runs `fern check --local --warnings --strict-broken-links` without
`FERN_TOKEN`. It also fails on MDX parse warnings, which Fern can report with a
successful exit code. Keep the Fern version in `.mise.toml`, `mise.lock`,
`docs/fern/fern.config.json`, and any workflow CLI pins aligned; validation
rejects a configuration mismatch. The CLI may need
network access to resolve its configured version. Local validation does not
prove that a hosted build will succeed or that every page renders correctly.

For a hosted preview, authenticate with `fern login` or set `FERN_TOKEN`, then run:

```bash
mise run docs:fern:check && mise run docs:fern:preview
```

Command output, including the hosted preview URL, appears directly in the
terminal or GitHub Actions log. Hosted previews send the selected documentation
to Fern. For the NVIDIA theme in the local server, authentication is also needed;
the unauthenticated local server starts without the central theme.
An unauthenticated missing-redirects warning concerns comparison with the hosted
instance; it does not establish that redirects work.

`fern-docs-ci.yml` runs the same `docs:fern:check` task on every PR, pushes to
`main`, merge queue checks, and manual dispatch. Its stable job name is
**Fern Check**, and it requires no publishing token. `fern-docs-preview.yml` is
manually dispatched from any repository branch. It runs `docs:fern:check` before
`docs:fern:preview`, supplies the repository or organization `FERN_TOKEN` secret
only to the preview step. Both workflows use the existing shared Python
environment action, then install Fern and its Node/jq/yq dependencies with the
mise action. The notebook prerequisite installs the locked
Python docs dependencies through uv. Neither publishes production
documentation. See the [migration status](docs/fern/FERN_MIGRATION.md#current-status)
for production publishing and instance setup.

#### Notebooks and visual review

Tutorial Python sources and their generated `.ipynb` files keep the existing
workflow described under [Notebooks](#notebooks). Fern's current exporter reads
saved notebooks and outputs without executing cells. Generated MDX lives beside
the notebooks; extracted images go under `docs/fern/assets/notebooks/`. These
files are ignored by Git and regenerated by the check, serve, and preview tasks.
Do not edit them directly. Notebook handling will change in a follow-up PR.

Check notebook reading pages, saved outputs, code formatting, and dataset and
notebook download links in the preview.

Preview affected pages even when validation passes. Check cards, sidebar
entries, tabs, admonitions, formulas, charts, Dev Notes bylines, and API symbol
anchors as applicable. For navigation or theme changes, also check desktop and
mobile layouts, light/dark themes, and search.

#### Versioned documentation

`nightly` follows `main`. Release documentation must reflect the exact release
source, including API output, notebook reading pages, assets, and download links
pinned to the release tag. Treat frozen snapshots as immutable except for
deliberate backports. A `latest` alias should point to the current frozen GA
release, rather than nightly content.

When adding release publication, publish the complete version manifest so that
publishing an older release cannot remove newer versions. Serialize main and
release publications. Existing historical documentation at
`https://nvidia-nemo.github.io/Anonymizer/` remains available; preserve
version-specific URLs and assets rather than redirecting all releases to nightly.

## Notebooks

Tutorial notebooks are generated from Python sources:

- Source files: `docs/notebook_source/*.py`
- Generated notebooks: `docs/notebooks/*.ipynb`

When editing tutorial sources, regenerate notebooks with:

```bash
make convert-notebooks
```

Notebook execution can require configured model provider credentials. If notebooks cannot be regenerated locally, state
that clearly in the PR and include the exact failure mode. Do not save credentials or credential-bearing outputs in
generated notebooks.

## Releases

Build a wheel locally:

```bash
make build-wheel
```

Release tags use `vMAJOR.MINOR.PATCH` for stable releases and `vMAJOR.MINOR.PATCHrcN` for release candidates, while the Python package version is the unprefixed version.

Release publishing is handled by `.github/workflows/release.yml`.
