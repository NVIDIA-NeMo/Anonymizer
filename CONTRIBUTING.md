# Contributing to NeMo Anonymizer

Thank you for your interest in contributing to NeMo Anonymizer. Please read our [Code of Conduct](CODE_OF_CONDUCT.md) before contributing.

This document covers contribution policy and pull request expectations. For local setup, test commands, docs commands, and day-to-day development tasks, see [DEVELOPMENT.md](DEVELOPMENT.md).

## Table of Contents

- [Contribution Flow](#contribution-flow)
- [Repository Rules](#repository-rules)
  - [Branch Naming](#branch-naming)
  - [Conventional Commits](#conventional-commits)
  - [Release Tags](#release-tags)
  - [Branch Protection](#branch-protection)
- [Pull Request Expectations](#pull-request-expectations)
- [Agent-Assisted Development](#agent-assisted-development)
- [Developer Certificate of Origin](#developer-certificate-of-origin)
- [Issues and Discussions](#issues-and-discussions)

## Contribution Flow

1. Search existing issues and pull requests before starting work.
2. Create or link an issue for the change. Bug fixes, features, and non-trivial development tasks should start from an issue so maintainers can confirm scope.
3. Create a branch from the latest `main` using the [branch naming](#branch-naming) convention.
4. For non-trivial changes, write a plan document before implementation. See [Agent-Assisted Development](#agent-assisted-development).
5. Implement the change following [AGENTS.md](AGENTS.md), [STYLEGUIDE.md](STYLEGUIDE.md), and [DEVELOPMENT.md](DEVELOPMENT.md).
6. Run the relevant local checks from [DEVELOPMENT.md](DEVELOPMENT.md#validation-before-opening-a-pr).
7. Open a pull request with a conventional title, a linked issue, and a completed checklist.
8. Address review feedback. CODEOWNERS are assigned automatically.

## Repository Rules

This repository uses GitHub rulesets, branch protection, CODEOWNERS, semantic PR title checks, and DCO checks. These are enforced automatically, but contributors should understand the expected shape before opening a PR.

### Branch Naming

All branches except `main` must follow one of these patterns:

```text
<author>/<description>
<author>/<issue-id>-<description>
<author>/<type>/<description>
<author>/<type>/<issue-id>-<description>
```

Rules:

- `<author>`: GitHub username or team name, lowercase, with alphanumeric characters and hyphens.
- `<issue-id>`: Optional GitHub issue number prefix, such as `123-`.
- `<description>`: Short lowercase description with alphanumeric characters and hyphens.
- `<type>`: Optional category. Valid types are `feature`, `bugfix`, `hotfix`, `release`, `docs`, `chore`, and `test`.

Examples:

| Branch Name                       | Valid                |
| --------------------------------- | -------------------- |
| `jsmith/add-login-feature`        | Yes                  |
| `jsmith/123-add-login-feature`    | Yes                  |
| `jsmith/feature/123-add-login`    | Yes                  |
| `aagonzales/bugfix/456-fix-crash` | Yes                  |
| `feature/add-login`               | No, missing author   |
| `JSmith/123-Add-Login`            | No, use lowercase    |

### Conventional Commits

Pull request titles must follow [Conventional Commits](https://www.conventionalcommits.org/) because squash merging uses the PR title as the commit message:

```text
<type>(<scope>): <description>
```

or without a scope:

```text
<type>: <description>
```

Rules:

- `<type>` is required and must be lowercase.
- `<scope>` is optional and names the affected area.
- `<description>` is required and should be concise.
- Add `!` after type or scope for breaking changes.

Valid types:

| Type       | Description                                      |
| ---------- | ------------------------------------------------ |
| `feat`     | New feature                                      |
| `fix`      | Bug fix                                          |
| `docs`     | Documentation changes                            |
| `style`    | Code style changes without logic changes         |
| `refactor` | Code refactoring without feature or fix changes  |
| `perf`     | Performance improvements                         |
| `test`     | Adding or updating tests                         |
| `build`    | Build system or dependency changes               |
| `ci`       | CI/CD configuration                              |
| `chore`    | Maintenance tasks                                |
| `revert`   | Reverting previous commits                       |

Examples:

| PR Title                                  | Valid             |
| ----------------------------------------- | ----------------- |
| `feat: add user authentication`           | Yes               |
| `fix(auth): resolve token expiration bug` | Yes               |
| `docs: update API documentation`          | Yes               |
| `chore(deps)!: bump major dependencies`   | Yes, breaking     |
| `Added new feature`                       | No, missing type  |
| `FIX: resolve bug`                        | No, type must be lowercase |

### Release Tags

GitHub release tags use a `v` prefix followed by the PEP 440 package version. The supported public release forms are stable releases and release candidates:

```text
vMAJOR.MINOR.PATCH
vMAJOR.MINOR.PATCHrcN
```

The Python package version remains the PEP 440 version without the `v` prefix. Release automation builds the wheel, extracts the unprefixed package `VERSION`, creates GitHub releases as `v${VERSION}`, marks `rcN` versions as prereleases, and deploys docs using the unprefixed package version.

Examples:

| Tag                         | Valid              |
| --------------------------- | ------------------ |
| `v1.0.0`                    | Yes                |
| `v2.1.3`                    | Yes                |
| `v1.0.0rc1`                 | Yes                |
| `v1.0.0rc2`                 | Yes                |
| `1.0.0`                     | No, missing `v`    |
| `v1.0.0-rc.1`               | No, use `rc1`      |
| `v1.0.0-alpha`              | No, unsupported    |
| `release-1.0`               | No, wrong format   |

### Branch Protection

The `main` branch has these protections:

| Rule                            | Setting     |
| ------------------------------- | ----------- |
| Required approvals              | 1           |
| Code owner review               | Required    |
| Require conversation resolution | Yes         |
| Linear history                  | Required    |
| Force pushes                    | Blocked     |
| Deletions                       | Blocked     |
| Merge strategy                  | Squash only |

## Pull Request Expectations

Every PR should include:

- A linked issue using `Fixes #NNN`, `Closes #NNN`, or `Resolves #NNN`, or, for maintainer-owned
  changes only, a clear explanation for why no issue is needed. External contributors must link a real
  issue with the maintainer-applied `triaged` label before the PR can merge. The linked-issue workflow
  checks the visible PR body, ignores HTML comments, verifies that the issue exists, and reruns the
  linked-issue check for open PRs that reference an issue when `triaged` is added to that issue.
- A conventional PR title.
- A summary of user-visible behavior, developer-facing behavior, or policy changed by the PR.
- Relevant tests or a brief explanation for why tests do not apply.
- Documentation updates when public behavior, CLI behavior, examples, notebooks, or contributor workflow changes.
- A public API impact check. If a public symbol or default changes, check whether [`skills/anonymizer/SKILL.md`](skills/anonymizer/SKILL.md) also needs an update.
- A PII fixture check. Do not add real PII to tests, docs, notebooks, or artifacts. Use synthetic examples.
- A secrets check. Do not commit API keys, service tokens, private keys, credentials, or real endpoint secrets. Use
  environment variables, local `.env` files, or GitHub Actions secrets instead.

PRs with checks in GitHub CLI's `fail` bucket receive stale reminders after a period of inactivity. Cancelled checks do
not trigger stale reminders or auto-close. External PRs with failing checks may be closed automatically after a reminder
if there is still no author activity. Collaborator PRs receive reminders only. Maintainers can add the `keep-open` label
to suppress stale reminders and auto-close behavior.

CODEOWNERS:

- `src`, `tests`, and `docs`: `@NVIDIA-NeMo/anonymizer-reviewers`
- All remaining files, including `pyproject.toml`, `uv.lock`, `SECURITY.md`, `LICENSE`, and `.github/`: `@NVIDIA-NeMo/anonymizer-maintainers`

## Agent-Assisted Development

Coding agents can help with implementation, review, tests, and documentation, but contributors remain responsible for the final change.

For non-trivial changes, draft a plan first. Non-trivial includes:

- Changes spanning more than one of the `config`, `engine`, or `interface` subsystems.
- New or changed public APIs.
- Changes to rewrite pipeline behavior or data-flow invariants.
- Changes to repository policy, CI, release automation, or documentation publishing.
- Changes to invariants called out in [AGENTS.md](AGENTS.md) or [STYLEGUIDE.md](STYLEGUIDE.md).

Plan document expectations:

- Save the plan at `plans/<issue-number>/<short-name>.md`.
- Explain the goal, affected subsystems, trade-offs considered, validation strategy, and rollout plan.
- Get maintainer review before or during implementation, depending on the change scope.
- Link the plan document from the implementation PR. If maintainers decide a plan is not needed, include `No plan required: <reason>` in the implementation PR.

Implementation expectations:

- Agents and humans should read [AGENTS.md](AGENTS.md) and [STYLEGUIDE.md](STYLEGUIDE.md) before non-trivial work.
- Agent-authored changes should be self-reviewed before requesting human review.
- Keep generated or exploratory artifacts out of the PR unless they are intentionally part of the deliverable.

## Developer Certificate of Origin

All contributions must be signed off to certify that you have the right to submit the code. Add a `Signed-off-by` line to your commit messages:

```bash
git commit -s -m "feat: add new feature"
```

This adds:

```text
Signed-off-by: Your Name <your.email@example.com>
```

By signing off, you certify the [Developer Certificate of Origin](DCO). See the full [DCO](DCO) file for details.

## Issues and Discussions

We provide structured issue templates for bug reports, feature requests, and development tasks.

For general questions, use [GitHub Discussions](https://github.com/NVIDIA-NeMo/Anonymizer/discussions) instead of opening an issue.
