<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Fern migration

This is a temporary file for the MkDocs to Fern migration and will be removed
when the migration is complete. See [DEVELOPMENT.md](../../DEVELOPMENT.md#fern-validation-and-previews)
for tool installation, authoring, validation, previews, and versioning conventions.

1. Add MDX content, Fern configuration, mise tasks, validation, and manual preview
   workflows in this PR.
2. Ask in the `cdd-fern` Slack channel for the official docs URL and `FERN_TOKEN`.
3. Review the hosted preview and iterate on content alongside the remaining work.
4. Provide a way to reach historical GitHub Pages documentation from Fern.
5. Generate API reference pages and restore API navigation and symbol links.
6. Improve notebook handling and rendering in a follow-up PR.
7. Configure production publishing and frozen release versioning.
8. Request publication in `cdd-fern` once the site is ready.
9. Update public documentation links in repository files, package metadata,
   the bundled anonymizer skill, notebook sources, and GitHub settings.
10. Delete verified legacy source counterparts, MkDocs configuration and
    dependencies, temporary dual-authoring instructions, and this file. Preserve
    historical GitHub Pages versions and their assets at their existing URLs.

## Current status

Fern prose lives in `docs/**/*.mdx`, with site configuration and nightly
navigation in `docs/fern/`. MkDocs and mike still serve production and release
documentation; existing build, notebook, and release workflows remain active.

Fern configures only `nightly` and the preview instance
`nemo-anonymizer.docs.buildwithfern.com`. The NVIDIA Fern team still needs to
confirm instance provisioning, the official production URL and mount, and token
access. No production publishing or frozen Fern release versions are configured.
The NVIDIA global theme supplies shared branding.

The API Reference entry is omitted pending generated API pages. Five tutorial
reading pages are generated from saved notebook cells and outputs without
execution. Notebook export behavior is intentionally retained for the follow-up.
The five developer notes keep their post dates and bylines; the MkDocs blog
archive remains on the legacy site.

## Required during migration: update both formats

Every change to a paired documentation page must update both its `.md` and
`.mdx` files in the same change. These are independent sources with no automatic
prose synchronization: `.md` serves production MkDocs and `.mdx` serves Fern.
Keep their content equivalent using each renderer's syntax.

- New prose pages need both formats and matching navigation entries in
  `mkdocs.yml` and `docs/fern/versions/nightly.yml`.
- For notebooks, edit `docs/notebook_source/*.py` and regenerate `.ipynb` with
  `make convert-notebooks`. Fern tasks regenerate the ignored MDX companions;
  do not edit them directly.
- Standalone repository Markdown does not need an MDX counterpart. Do not
  update frozen release snapshots for routine changes.
- Verify both renderers with `make docs-build` and
  `mise run docs:fern:check`, and preview affected pages in both sites.
- Retain legacy pages, configuration, and shared assets until cutover. Do not
  rerun the migration copy step over reviewed prose.

## Before publication and cleanup

Review the hosted preview for navigation, search, visual parity, bylines,
notebook cells and outputs, downloads, API coverage, and anchors. Restore
Pydantic field documentation and public API links as part of the API follow-up.
Validation cannot prove hosted rendering or redirect correctness.

Configure a repository or organization `FERN_TOKEN` secret for the manual
`fern-docs-preview.yml` workflow. PR validation, including fork PRs, uses no
publishing token. Add the **Fern Check** job to required checks after its first
successful run. Production publishing remains a separate change.

Release automation must build from the release tag, retain notebook artifact
ordering from `build-notebooks.yml`, `docs.yml`, and `release.yml`, freeze pages,
API output, downloads, and assets, and publish a complete version manifest.
`latest` must identify a frozen GA release, never nightly. Serialize main and
release publications and keep credentials out of untrusted PR execution.

Before changing public links, refresh the deployed version inventory:

```bash
git fetch origin gh-pages
git show origin/gh-pages:versions.json
```

Inventory version sitemaps, important page URLs and fragments, notebook downloads,
and blog post slugs. Test destination routes and anchors individually. Keep
historical versions and their assets serving at
`https://nvidia-nemo.github.io/Anonymizer/`; Fern redirects cannot redirect
requests received by GitHub Pages. Any old dev/latest redirects require explicit
old-host rules and route/fragment checks. Retain a known-good deployment for
rollback and preserve the `gh-pages` branch after source cleanup.

Once publication, public links, and releases are verified, retire routine
MkDocs/mike deployment. Remove only legacy prose with verified MDX replacements
and obsolete MkDocs configuration, scripts, tasks, and dependencies. Keep shared
assets, notebook sources, `docs/data/README.md`, license files, and repository
Markdown. Remove temporary migration guidance from CONTRIBUTING.md,
DEVELOPMENT.md, AGENTS.md, and STYLEGUIDE.md, then delete this file. Repeat site
and historical-link checks after cleanup.
