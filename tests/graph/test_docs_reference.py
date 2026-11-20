# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The graph SDK reference page names every public name and every rejection and reason code."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import anonymizer.graph as graph

ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "src" / "anonymizer" / "engine" / "graph_core"
REFERENCE = ROOT / "docs" / "concepts" / "graph-sdk" / "reference.md"

# Where the kernel writes a code as a string literal: a rejection, a mention rule, a failed or closed step, a reason.
CODE_SITES = (
    r'GraphRejection\(\s*"(\w+)"',
    r'return "(\w+)"',
    r'Failed\([^()]*?"(\w+)"',
    r'Closed\(\w+, "(\w+)"',
    r'Reason\("(\w+)"',
    r'Reason\("(\w+)" if [^()]* else "(\w+)"\)',
)


def _kernel_codes() -> set[str]:
    source = "\n".join(p.read_text() for p in CORE.glob("*.py"))
    found: set[str] = set()
    for pattern in CODE_SITES:
        for match in re.finditer(pattern, source):
            found.update(g for g in match.groups() if g)
    # mention_problem's codes and the reason codes come back as plain returns too; keep only snake_case codes.
    return {c for c in found if re.fullmatch(r"[a-z]+(_[a-z]+)+|[a-z]+", c)}


def _documented(page: Path) -> set[str]:
    return set(re.findall(r"`([A-Za-z_]\w*)`", page.read_text()))


def test_the_scan_finds_codes_from_every_site() -> None:
    codes = _kernel_codes()
    for code in ("datum_not_text", "mention_out_of_range", "raised", "binding_failed", "withheld", "annotation"):
        assert code in codes


@pytest.mark.parametrize("page", [REFERENCE, REFERENCE.with_suffix(".mdx")], ids=["md", "mdx"])
def test_reference_names_every_public_name(page: Path) -> None:
    assert set(graph.__all__) - _documented(page) == set()


@pytest.mark.parametrize("page", [REFERENCE, REFERENCE.with_suffix(".mdx")], ids=["md", "mdx"])
def test_reference_names_every_rejection_and_reason_code(page: Path) -> None:
    assert _kernel_codes() - _documented(page) == set()


def test_repository_links_on_the_graph_pages_name_files_that_exist() -> None:
    pages = [ROOT / "docs" / "concepts" / "graph-sdk.md", *(ROOT / "docs" / "concepts" / "graph-sdk").glob("*.md")]
    link = re.compile(r"https://github\.com/NVIDIA-NeMo/Anonymizer/blob/main/([^)#\s]+)")
    paths = {m.group(1) for page in pages for m in link.finditer(page.read_text())}
    assert paths, "expected at least the example-file link"
    assert {p for p in paths if not (ROOT / p).is_file()} == set()
