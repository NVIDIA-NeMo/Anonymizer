# Development Guide

This guide covers local setup, common development commands, testing, documentation, notebooks, and validation before opening a pull request. For contribution policy and PR expectations, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Prerequisites

- Python 3.11+
- Git
- [mise](https://mise.jdx.dev/) 2026.10.15 or newer, which installs the pinned uv, Ruff, and ty and runs every task
- [gh](https://cli.github.com/) for optional GitHub CLI workflows

Development tools such as Ruff, ty, pre-commit, pytest, and pytest-cov are installed by the development dependency group.

## Local Setup

Clone the repository and install development dependencies:

```bash
git clone https://github.com/<your-username>/Anonymizer.git
cd Anonymizer
make setup
```

`make setup` installs Mise when needed, trusts the repository's `.mise.toml`, then runs the default `dev` setup profile.

After setup, verify that `mise --version` works in your terminal. If the command isn't found, follow mise's [shell setup instructions](https://mise.jdx.dev/getting-started.html#activate-mise) for your environment. This is a one-time setup per machine and shell.

If Mise is already installed, trust the configuration once and run the same onboarding flow directly:

```bash
mise trust
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
mise run docs:fern:check
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

`mise run check` runs `mise run check:format`, `mise run check:lint`, `mise run check:type`, `mise run check:lock`,
`mise run check:license:headers`, and `mise run check:docs:pairs`. CI runs the same tasks as separate steps so failures identify the affected stage.

Run the blocking type checker:

```bash
mise run check:type
```

`ty` checks `src`, `tests`, `tests_e2e`, `docs`, `scripts`, and `tools`. Errors and warnings fail locally and in CI.

Check that `uv.lock` and the serving runtime's `tools/inference-service-runtime/uv.lock` are current:

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

Update docs when a change affects public API behavior, CLI behavior, examples,
notebooks, configuration, contributor workflow, or release process. Fern is being
introduced alongside the production MkDocs site; see the
[temporary migration status](docs/fern/FERN_MIGRATION.md) for parallel authoring
requirements and cutover work.

### MkDocs

```bash
mise run docs:serve    # local server at http://127.0.0.1:8000
mise run docs:build    # strict build
```

### Fern Validation and Previews

Fern prose lives in `docs/**/*.mdx`. The site configuration is
`docs/fern/docs.yml`, and nightly navigation is in
`docs/fern/versions/nightly.yml`. Use explicit slugs to keep page URLs stable
when reorganizing the sidebar. Images use the existing source-relative assets;
use native Fern components for cards, tabs, and admonitions.

`mise run setup` installs the Fern tools with the rest of the pinned tools. To
install them together with the docs dependency profile:

```bash
mise run setup docs
```

This installs Node.js 22, Fern CLI 5.201.0, and the check script's jq/yq
dependencies from `.mise.toml` and `mise.lock`. No separate npm installation is
needed. Run these tasks from the repository root:

| Task | Purpose |
| --- | --- |
| `mise run docs:fern:check` | Validate version alignment, configuration, MDX, and internal links. |
| `mise run docs:fern:serve` | Start the local Fern development server. |
| `mise run docs:fern:preview` | Generate a hosted preview; run the check task first. |
| `mise run docs:fern:notebooks` | Regenerate notebook MDX from saved cells and outputs without executing cells. |

Validation runs `fern check --local --warnings --strict-broken-links` without
`FERN_TOKEN`. It also fails on MDX parse warnings, which Fern can report with a
successful exit code. Validation also fails when the `npm:fern-api` version in
`.mise.toml` differs from the version in `docs/fern/fern.config.json`; update
both, then run `mise lock`. The CLI may need
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

`fern-docs-ci.yml` runs the same `docs:fern:check` task on PRs and pushes to
`main` that change `docs/` or the Fern tooling (see its `paths` filter), on
merge queue checks, and on manual dispatch. Its stable job name is
**Fern Check**, and it requires no publishing token. `fern-docs-preview.yml` is
manually dispatched from any repository branch. It runs `docs:fern:check` before
`docs:fern:preview`, and passes the organization `DOCS_FERN_TOKEN` secret to the
preview step only, as `FERN_TOKEN`. Both workflows use the shared Python environment
action with the `docs` profile, which installs mise and every pinned tool,
including Fern and its Node/jq/yq dependencies, and syncs the locked docs
dependencies. Neither publishes production
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

Pushing a `v*` tag starts a production release immediately. The run builds the wheel, checks it against the tag (`release:verify`: the wheel version must equal the tag without its `v`, `twine check --strict` must pass, and a clean install must import), publishes to Test PyPI, then publishes to PyPI, creates the GitHub release for the tag, and deploys the versioned docs. Test PyPI runs first on every release, so a Test PyPI failure stops the run before PyPI.

Create the tag on the exact commit you intend to release. The package version comes from Git tags (`uv-dynamic-versioning`), so a commit without a tag builds a version with a `+<hash>` suffix, which PyPI and Test PyPI do not accept.

To rehearse before tagging, run `mise run release:check <version>` with the tag minus its `v`, in normalized form (`0.5.0rc1`, not `0.5.0.rc1` or `0.5.0-rc1`). It builds the wheel at that version (`UV_DYNAMIC_VERSIONING_BYPASS`), then runs `release:verify`. It replaces `dist/`, needs no credentials, and publishes nothing. Local versions (`+...`) are rejected because PyPI would reject them.

The manual workflow (`workflow_dispatch`) takes three inputs. What it does depends on the ref you run it on:

| Run on | `version` | Result |
| --- | --- | --- |
| Branch | Required, with `dry-run` on | Runs `release:check` in CI. Uploads nothing and creates no release. |
| Tag | Must be empty | Builds and verifies the tag's version, then uploads it to Test PyPI. With `dry-run` off it also publishes to PyPI. |

`create-gh-release` applies only when `dry-run` is off. A dry run never creates the release or deploys docs. The Test PyPI upload uses `--skip-existing`, so re-running a tag whose version is already on Test PyPI passes without uploading.

`mise run publish:pypi <version>` builds and publishes an explicit version straight to PyPI, with no Test PyPI step and no tag check. It runs `release:verify` before uploading. It is a manual fallback, not the release path; `--dry-run` builds and validates without uploading.

## Local inference service environment

The vLLM serving runtime has an independent Python 3.12 uv project at
`tools/inference-service-runtime`. Its lockfile resolves separately from
Anonymizer and DataDesigner. From the repository root:

```bash
uv sync --project tools/inference-service-runtime --locked --group test
uv run --project tools/inference-service-runtime --locked --group test pytest --confcutdir tests/tools tests/tools/test_inference_service.py tests/tools/test_vllm_*.py
```

`--confcutdir` keeps the serving tests independent of Anonymizer's test fixtures.
`mise run check:lock` checks both lockfiles without installing the serving stack.
See the [deployment guide](docs/concepts/inference-services.md) for launch commands
and the container workflow.
