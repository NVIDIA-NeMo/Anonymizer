#!/usr/bin/env -S uv run
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""diff-lockfile: compare uv.lock between git refs and report dependency changes.

Parses two versions of a uv.lock file (base ref vs head ref) and categorizes
every package version change as added, removed, upgraded, or downgraded.

Usage::

    uv run tools/diff-lockfile.py                          # auto merge-base vs HEAD
    uv run tools/diff-lockfile.py origin/main              # explicit base ref
    uv run tools/diff-lockfile.py --json | jq '.[]'        # JSON output for tooling
    git bisect run uv run tools/diff-lockfile.py \\
        --run tests/test_foo.py                             # bisect driver

Models:
    Package        -- snapshot of a single package (name, version, source)
    PackageChange  -- one version delta (old Package -> new Package)
    LockfileDiff   -- container of all PackageChange records (JSON array)
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "typer>=0.15",
#     "pydantic>=2",
#     "gitpython>=3.1",
#     "packaging>=24",
#     "rich>=13",
# ]
# ///

from __future__ import annotations

import subprocess
import tomllib
from enum import Enum
from typing import Annotated, Optional

import git
import typer
from packaging.version import Version
from pydantic import BaseModel, PlainSerializer, PlainValidator, RootModel
from rich.console import Console
from rich.text import Text

# ---------------------------------------------------------------------------
# Custom Pydantic type: packaging.version.Version that serialises as a string
# ---------------------------------------------------------------------------

VersionStr = Annotated[
    Version,
    PlainValidator(lambda v: v if isinstance(v, Version) else Version(str(v))),
    PlainSerializer(str, return_type=str),
]


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ChangeType(str, Enum):
    """Classification of a package version change."""

    added = "added"
    removed = "removed"
    upgraded = "upgraded"
    downgraded = "downgraded"


class Package(BaseModel):
    """Snapshot of a single package at a specific git ref."""

    name: str
    version: VersionStr
    source: str = ""


class PackageChange(BaseModel):
    """A single package version change between two lockfile refs."""

    name: str
    change: ChangeType
    old: Package | None = None
    new: Package | None = None
    ref: str = ""


class LockfileDiff(RootModel[list[PackageChange]]):
    """Container of all package changes.  Serialises as a JSON array."""


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def get_lockfile_content(repo: git.Repo, ref: str, path: str) -> str:
    """Read the contents of a file at a given git ref.

    Args:
        repo: An initialised ``git.Repo`` instance.
        ref: A git ref (branch, tag, SHA, ``HEAD``, etc.).
        path: Path to the file inside the repository tree.

    Returns:
        The UTF-8 decoded file contents.

    Raises:
        KeyError: If *path* does not exist at *ref*.
    """
    commit = repo.commit(ref)
    blob = commit.tree / path
    return blob.data_stream.read().decode()


def _extract_source(raw: dict) -> str:  # type: ignore[type-arg]
    """Return a human-readable source string from a ``[[package]]`` entry."""
    src = raw.get("source", {})
    if isinstance(src, dict):
        return src.get("registry", src.get("git", src.get("path", "")))
    return str(src) if src else ""


def parse_packages(content: str) -> dict[str, Package]:
    """Parse a ``uv.lock`` file and return a mapping of packages.

    Args:
        content: Raw UTF-8 text of the lockfile.

    Returns:
        A dict keyed by package name (or ``name@source`` when a name
        appears more than once with different sources).  Values are
        :class:`Package` instances.
    """
    data = tomllib.loads(content)

    name_counts: dict[str, int] = {}
    for pkg in data.get("package", []):
        n = pkg["name"]
        name_counts[n] = name_counts.get(n, 0) + 1

    packages: dict[str, Package] = {}
    for pkg in data.get("package", []):
        name: str = pkg["name"]
        version: str | None = pkg.get("version")
        if version is None:
            # Workspace members (editable installs) have no pinned version
            continue
        source = _extract_source(pkg)

        # Disambiguate packages that appear under multiple sources
        key = f"{name}@{source}" if name_counts.get(name, 1) > 1 else name
        packages[key] = Package(name=name, version=Version(version), source=source)

    return packages


def diff_packages(
    base: dict[str, Package],
    head: dict[str, Package],
    ref: str,
) -> LockfileDiff:
    """Compare two parsed lockfile snapshots and emit a list of changes.

    Every version-string difference is forced into *upgraded* or *downgraded*
    using :class:`packaging.version.Version` ordering (including ``+local``
    segments).  Source-only changes (same version, different registry) are
    treated as no-ops.

    Args:
        base: Packages parsed from the base ref.
        head: Packages parsed from the head ref.
        ref: A string describing the comparison, e.g. ``"abc123..def456"``.

    Returns:
        A :class:`LockfileDiff` containing one :class:`PackageChange` per
        differing package.
    """
    changes: list[PackageChange] = []
    all_keys = sorted(set(base) | set(head))

    for key in all_keys:
        base_pkg = base.get(key)
        head_pkg = head.get(key)

        if base_pkg is None and head_pkg is not None:
            changes.append(
                PackageChange(name=head_pkg.name, change=ChangeType.added, new=head_pkg, ref=ref),
            )
        elif head_pkg is None and base_pkg is not None:
            changes.append(
                PackageChange(name=base_pkg.name, change=ChangeType.removed, old=base_pkg, ref=ref),
            )
        elif base_pkg is not None and head_pkg is not None and base_pkg.version != head_pkg.version:
            direction = ChangeType.upgraded if head_pkg.version > base_pkg.version else ChangeType.downgraded
            changes.append(
                PackageChange(name=base_pkg.name, change=direction, old=base_pkg, new=head_pkg, ref=ref),
            )

    return LockfileDiff(changes)


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

_CHANGE_ORDER = [ChangeType.upgraded, ChangeType.downgraded, ChangeType.added, ChangeType.removed]

_CHANGE_COLOUR = {
    ChangeType.upgraded: "green",
    ChangeType.downgraded: "yellow",
    ChangeType.added: "cyan",
    ChangeType.removed: "red",
}


def format_rich(diff: LockfileDiff, console: Console | None = None) -> None:
    """Render a :class:`LockfileDiff` to the terminal using Rich.

    Output is grouped by change type in the order: upgraded, downgraded,
    added, removed.  A summary line is printed at the end.

    Args:
        diff: The diff to render.
        console: Optional Rich console; a new one is created if *None*.
    """
    console = console or Console()

    if not diff.root:
        console.print("[dim]No dependency changes.[/dim]")
        return

    grouped: dict[ChangeType, list[PackageChange]] = {ct: [] for ct in _CHANGE_ORDER}
    for change in diff.root:
        grouped[change.change].append(change)

    max_name = max(len(c.name) for c in diff.root)

    for change_type in _CHANGE_ORDER:
        items = grouped[change_type]
        if not items:
            continue

        colour = _CHANGE_COLOUR[change_type]
        console.print(Text(f"\n{change_type.value.title()} ({len(items)}):", style=f"bold {colour}"))

        for item in sorted(items, key=lambda c: c.name):
            padded = item.name.ljust(max_name)
            if item.old and item.new:
                console.print(f"  {padded}  {item.old.version}  ->  {item.new.version}")
            elif item.new:
                console.print(f"  {padded}  {item.new.version}")
            elif item.old:
                console.print(f"  {padded}  {item.old.version}")

    counts = {ct: len(grouped[ct]) for ct in _CHANGE_ORDER if grouped[ct]}
    summary = ", ".join(f"{v} {k.value}" for k, v in counts.items())
    console.print(f"\n[bold]Summary:[/bold] {summary}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

app = typer.Typer(
    help="Compare uv.lock between git refs and report dependency version changes.",
    no_args_is_help=False,
)


def _resolve_base_ref(repo: git.Repo, base_ref: str | None) -> str:
    """Return the base ref, computing merge-base with origin/main when needed.

    Args:
        repo: An initialised ``git.Repo`` instance.
        base_ref: An explicit ref, or *None* to auto-compute.

    Returns:
        A resolved git ref string (SHA or symbolic name).
    """
    if base_ref is not None:
        return base_ref
    merge_bases = repo.merge_base("origin/main", "HEAD")
    if not merge_bases:
        typer.echo("Error: could not determine merge-base with origin/main", err=True)
        raise typer.Exit(code=1)
    return merge_bases[0].hexsha


@app.command()
def main(
    base_ref: Annotated[
        Optional[str],
        typer.Argument(help="Base git ref to compare against. Defaults to merge-base with origin/main."),
    ] = None,
    head: Annotated[str, typer.Option(help="Head git ref (default: HEAD).")] = "HEAD",
    lockfile: Annotated[str, typer.Option(help="Lockfile path within the repo.")] = "uv.lock",
    json_output: Annotated[bool, typer.Option("--json", help="Output as JSON instead of Rich text.")] = False,
    run: Annotated[
        Optional[str],
        typer.Option("--run", help="Bisect driver mode: pytest node ID to run after printing the diff."),
    ] = None,
) -> None:
    """Compare uv.lock between two git refs and report dependency changes."""
    repo = git.Repo(search_parent_directories=True)
    resolved_base = _resolve_base_ref(repo, base_ref)

    # In bisect mode, always diff against the parent commit
    effective_base = "HEAD~1" if run else resolved_base

    try:
        base_content = get_lockfile_content(repo, effective_base, lockfile)
    except KeyError:
        typer.echo(f"Lockfile '{lockfile}' not found at ref '{effective_base}'.", err=True)
        raise typer.Exit(code=0)

    try:
        head_content = get_lockfile_content(repo, head, lockfile)
    except KeyError:
        typer.echo(f"Lockfile '{lockfile}' not found at ref '{head}'.", err=True)
        raise typer.Exit(code=0)

    base_pkgs = parse_packages(base_content)
    head_pkgs = parse_packages(head_content)

    base_sha = repo.commit(effective_base).hexsha[:12]
    head_sha = repo.commit(head).hexsha[:12]
    ref_label = f"{base_sha}..{head_sha}"

    diff = diff_packages(base_pkgs, head_pkgs, ref=ref_label)

    if json_output:
        typer.echo(diff.model_dump_json(indent=2))
    else:
        format_rich(diff)

    if run is not None:
        _bisect_run(run)


# ---------------------------------------------------------------------------
# Bisect driver
# ---------------------------------------------------------------------------


def _bisect_run(test_target: str) -> None:
    """Execute the bisect driver workflow.

    1. Run ``uv sync --frozen`` (exit 125 on failure to signal bisect skip).
    2. Run ``uv run --frozen pytest {test_target} -x``.
    3. Pass through the pytest exit code.

    Args:
        test_target: A pytest node ID (file, file::class, file::test, etc.).
    """
    sync = subprocess.run(["uv", "sync", "--frozen"], capture_output=True)
    if sync.returncode != 0:
        typer.echo("uv sync --frozen failed; signalling bisect skip (exit 125).", err=True)
        raise typer.Exit(code=125)

    result = subprocess.run(["uv", "run", "--frozen", "pytest", test_target, "-x"])
    raise typer.Exit(code=result.returncode)


if __name__ == "__main__":
    app()
