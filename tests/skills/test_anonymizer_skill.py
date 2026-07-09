# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_DIR = REPO_ROOT / "skills" / "anonymizer"
PUBLISHED_DOCS_BASE = "https://nvidia-nemo.github.io/Anonymizer/"

EXPECTED_EVAL_IDS = {
    "anonymizer-positive-mode-choice",
    "anonymizer-positive-hash-cross-record-consistency",
    "anonymizer-positive-failed-records-first",
    "anonymizer-positive-self-hosted-gliner",
    "anonymizer-negative-general-privacy-explainer",
    "anonymizer-negative-repository-source-development",
}

LOCAL_DOC_RE = re.compile(r"(?P<path>(?:\.\./)*docs/[A-Za-z0-9_./-]+\.md)")


def test_publication_artifacts_are_present() -> None:
    required_paths = (
        SKILL_DIR / "SKILL.md",
        SKILL_DIR / "references" / "interactive.md",
        SKILL_DIR / "evals" / "evals.json",
        SKILL_DIR / "skill-card.md",
        SKILL_DIR / "BENCHMARK.md",
    )

    for path in required_paths:
        assert path.is_file(), f"Missing required skill artifact: {path.relative_to(REPO_ROOT)}"

    assert not (SKILL_DIR / "README.md").exists()
    assert not (SKILL_DIR / "workflows" / "interactive.md").exists()


def test_skill_routes_to_renamed_interactive_reference() -> None:
    skill_text = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")

    assert "references/interactive.md" in skill_text
    assert "workflows/interactive.md" not in skill_text


def test_eval_cases_have_activation_shape() -> None:
    evals = _load_evals()

    assert len(evals) == 6
    assert {case["id"] for case in evals} == EXPECTED_EVAL_IDS
    assert sum(1 for case in evals if case["should_trigger"]) == 4
    assert sum(1 for case in evals if not case["should_trigger"]) == 2

    for case in evals:
        assert set(case) == {
            "id",
            "question",
            "expected_skill",
            "should_trigger",
            "expected_script",
            "ground_truth",
            "expected_behavior",
        }
        assert isinstance(case["id"], str) and case["id"]
        assert isinstance(case["question"], str) and case["question"]
        assert isinstance(case["should_trigger"], bool)
        assert case["expected_script"] is None
        assert isinstance(case["ground_truth"], str) and case["ground_truth"]
        assert isinstance(case["expected_behavior"], list) and case["expected_behavior"]
        assert all(isinstance(item, str) and item for item in case["expected_behavior"])

        if case["should_trigger"]:
            assert case["expected_skill"] == "anonymizer"
        else:
            assert case["expected_skill"] is None


def test_local_docs_links_have_published_fallbacks() -> None:
    for relative_path in ("SKILL.md", "references/interactive.md"):
        path = SKILL_DIR / relative_path
        text = path.read_text(encoding="utf-8")
        paragraphs = re.split(r"\n\s*\n", text)
        matched_docs_link = False

        for paragraph in paragraphs:
            for match in LOCAL_DOC_RE.finditer(paragraph):
                matched_docs_link = True
                expected_url = _published_url_for(match.group("path"))
                assert expected_url in paragraph, (
                    f"{path.relative_to(REPO_ROOT)} references {match.group('path')} "
                    f"without nearby fallback {expected_url}"
                )

        assert matched_docs_link, f"{path.relative_to(REPO_ROOT)} should reference local docs"


def _load_evals() -> list[dict[str, Any]]:
    with (SKILL_DIR / "evals" / "evals.json").open(encoding="utf-8") as fh:
        evals = json.load(fh)

    assert isinstance(evals, list)
    return evals


def _published_url_for(local_doc_path: str) -> str:
    docs_path = local_doc_path[local_doc_path.index("docs/") + len("docs/") :]
    route = docs_path.removesuffix(".md")
    if route == "index":
        return PUBLISHED_DOCS_BASE
    if route.endswith("/index"):
        route = route[: -len("/index")]
    return f"{PUBLISHED_DOCS_BASE}{route}/"
