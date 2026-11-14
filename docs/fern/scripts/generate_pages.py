# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate Fern notebook pages from saved outputs without executing cells."""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

import nbformat
from bs4 import BeautifulSoup
from markdownify import markdownify
from nbconvert import MarkdownExporter

ROOT = Path(__file__).resolve().parents[3]
DOCS = ROOT / "docs"
FERN = DOCS / "fern"
LICENSE = (
    "{/* SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. */}\n"
    "{/* SPDX-License-Identifier: Apache-2.0 */}\n"
)


def make_mdx_safe(text: str) -> str:
    """Escape prose expressions while leaving fenced examples intact."""
    text = re.sub(r"<!--([\s\S]*?)-->", "", text)
    lines = []
    fence = None
    for line in text.splitlines():
        marker = re.match(r"^\s*(`{3,}|~{3,})", line)
        if marker:
            if fence is None:
                fence = marker[1][0]
            elif marker[1][0] == fence:
                fence = None
            lines.append(line)
            continue
        if fence is None:
            line = re.sub(r":(?:meth|class|func|attr|mod):(?=`)", "", line)
            parts = re.split(r"(`+[^`]*`+)", line)
            for index, part in enumerate(parts):
                if part.startswith("`"):
                    parts[index] = html.unescape(part)
                    if line.lstrip().startswith("|"):
                        parts[index] = parts[index].replace("|", r"\|")
                else:
                    part = part.replace(r"\_", "&#95;")
                    part = part.replace("{", "&#123;").replace("}", "&#125;")
                    part = re.sub(r"<(https?://[^>]+)>", r"[\1](\1)", part)
                    part = part.replace("<", "&lt;")
                    parts[index] = part
            line = "".join(parts)
        lines.append(line)
    return "\n".join(lines) + "\n"


def generate_notebooks() -> None:
    """Export the saved notebook cells and outputs without requiring LLM keys."""
    for source in sorted((DOCS / "notebooks").glob("*.ipynb")):
        notebook = nbformat.read(source, as_version=4)
        for cell in notebook.cells:
            for output in cell.get("outputs", []):
                data = output.get("data", {})
                if "text/html" in data:
                    soup = BeautifulSoup(data.pop("text/html"), "html.parser")
                    for tag in soup.select("style, script"):
                        tag.decompose()
                    data["text/markdown"] = markdownify(str(soup), heading_style="ATX")
        text, resources = MarkdownExporter().from_notebook_node(notebook)
        for name, content in resources.get("outputs", {}).items():
            target = FERN / "assets" / "notebooks" / source.stem / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            text = text.replace(f"]({name})", f"](/assets/notebooks/{source.stem}/{name})")
        text = text.replace("https://nvidia-nemo.github.io/Anonymizer/latest/tutorials/", "/tutorials/overview")
        text = re.sub(r"\]\(\.\./(\d\d_[^/)]+)/\)", r"](/tutorials/\1)", text)
        text = re.sub(r"\]\(\.\./\.\./concepts/([^/)]+)/?(#[^)]*)?\)", r"](/concepts/\1\2)", text)
        title = next((line.lstrip("# ") for line in text.splitlines() if line.startswith("# ")), source.stem)
        target = source.with_suffix(".mdx")
        target.write_text(f"---\ntitle: {json.dumps(title)}\n---\n{LICENSE}\n{make_mdx_safe(text)}")


if __name__ == "__main__":
    generate_notebooks()
