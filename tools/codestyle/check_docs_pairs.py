#!/usr/bin/env -S uv run --script
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///

"""Check that paired MkDocs (.md) and Fern (.mdx) pages change together.

During the Fern migration each prose page under docs/ exists as an independent
.md file (MkDocs) and .mdx file (Fern). This check compares the working tree,
including uncommitted changes, with the merge base of the target branch. Every
changed page must change in both formats, which covers edits, deletions, and new
pages. The exception is a page that had exactly one format at the merge base, so
adding its missing format, or editing a page that is intentionally single-format,
passes.

docs/fern/ is Fern infrastructure, not prose, and is not checked. Generated
notebook MDX is ignored by Git and never appears in the comparison.
"""

from __future__ import annotations

import os
import subprocess
import sys

DOCS = "docs/"
EXCLUDED = "docs/fern/"


def git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout


def git_paths(*args: str) -> set[str]:
    """Return the paths a git listing command prints, one per line."""
    return set(git(*args, "--", DOCS).splitlines())


def resolve_base() -> str:
    """Return the first existing ref among origin/$GITHUB_BASE_REF, origin/main, and main."""
    candidates = ["origin/main", "main"]
    if base_ref := os.environ.get("GITHUB_BASE_REF"):
        candidates.insert(0, f"origin/{base_ref}")
    for candidate in candidates:
        probe = ["git", "rev-parse", "--verify", "--quiet", f"{candidate}^{{commit}}"]
        if subprocess.run(probe, stdout=subprocess.DEVNULL).returncode == 0:
            return candidate
    raise SystemExit(
        "check:docs:pairs: no base ref found. Fetch the target branch, for example `git fetch origin main`."
    )


def page_stem(path: str) -> str | None:
    """Return the extension-free page path for a checked .md or .mdx file."""
    if not path.startswith(DOCS) or path.startswith(EXCLUDED):
        return None
    for suffix in (".mdx", ".md"):
        if path.endswith(suffix):
            return path.removesuffix(suffix)
    return None


def find_violations(*, base_files: set[str], changed: set[str]) -> list[str]:
    violations = []
    for stem in sorted({stem for path in changed if (stem := page_stem(path))}):
        md, mdx = f"{stem}.md", f"{stem}.mdx"
        single_format_at_base = (md in base_files) != (mdx in base_files)
        if not single_format_at_base and (md in changed) != (mdx in changed):
            edited, other = (md, mdx) if md in changed else (mdx, md)
            violations.append(f"{edited} changed but {other} did not")
    return violations


def main() -> int:
    base = resolve_base()
    merge_base_run = subprocess.run(["git", "merge-base", base, "HEAD"], capture_output=True, text=True)
    if merge_base_run.returncode != 0:
        raise SystemExit(
            f"check:docs:pairs: no merge base between {base} and HEAD. Fetch full history, "
            "for example `git fetch --unshallow origin`."
        )
    merge_base = merge_base_run.stdout.strip()
    violations = find_violations(
        base_files=git_paths("ls-tree", "-r", "--name-only", merge_base),
        changed=git_paths("diff", "--name-only", "--no-renames", merge_base)
        | git_paths("ls-files", "--others", "--exclude-standard"),
    )
    if not violations:
        print(f"check:docs:pairs: paired .md/.mdx pages are in sync relative to {base}.")
        return 0
    print(f"check:docs:pairs: update both formats in the same change (relative to {base}):", file=sys.stderr)
    for violation in violations:
        print(f"  {violation}", file=sys.stderr)
    print("See docs/fern/FERN_MIGRATION.md.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
