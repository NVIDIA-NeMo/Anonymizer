# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run every Python example on the graph SDK pages and compare it with the output the page shows."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parents[2] / "docs" / "concepts"
PAGES = sorted([DOCS / "graph-sdk.md", *(DOCS / "graph-sdk").glob("*.md")])
FENCE = re.compile(r"^```(\w*)\n(.*?)^```\n", re.S | re.M)


def _blocks(page: Path) -> list[tuple[str, str]]:
    return [(m.group(1), m.group(2)) for m in FENCE.finditer(page.read_text())]


def _examples(page: Path) -> list[tuple[str, str | None]]:
    """Each Python block with the ``text`` block right after it, if any: the output the page shows for it."""
    text = page.read_text()
    found = []
    for m in FENCE.finditer(text):
        if m.group(1) != "python":
            continue
        gap = len(text[m.end() :]) - len(text[m.end() :].lstrip("\n"))
        after = FENCE.match(text, m.end() + gap)
        found.append((m.group(2), after.group(2) if after and after.group(1) == "text" else None))
    return found


CASES = [
    pytest.param(page, i, code, shown, id=f"{page.stem}-{i}")
    for page in PAGES
    for i, (code, shown) in enumerate(_examples(page))
]


def _normalize(out: str) -> str:
    return "\n".join(line.rstrip() for line in out.strip("\n").splitlines())


@pytest.mark.parametrize(("page", "index", "code", "shown"), CASES)
def test_example_runs_and_prints_what_the_page_shows(
    page: Path, index: int, code: str, shown: str | None, tmp_path: Path
) -> None:
    done = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True, timeout=120, check=False
    )
    assert done.returncode == 0, f"{page.name} example {index} failed:\n{done.stderr}"
    if shown is not None:
        assert _normalize(done.stdout) == _normalize(shown), f"{page.name} example {index} output differs"


def test_every_page_has_examples_found() -> None:
    assert sum(len(_examples(p)) for p in PAGES) >= 30


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.stem)
def test_mdx_twin_has_the_same_code_blocks(page: Path) -> None:
    assert _blocks(page) == _blocks(page.with_suffix(".mdx"))
