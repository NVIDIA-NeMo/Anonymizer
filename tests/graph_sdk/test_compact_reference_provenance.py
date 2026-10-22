# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pin relocated reference sources and unchanged independently frozen corpora."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tests.graph_sdk.reference.evidence_v1.p7_reference.fixtures import write_bundle
from tests.graph_sdk.reference.foundation_v2.generate import encode_corpus

ROOT = Path(__file__).parent / "reference"


@pytest.mark.parametrize("name", ["foundation_v2", "evidence_v1"])
def test_adopted_reference_matches_recorded_provenance(name: str) -> None:
    directory = ROOT / name
    adoption = json.loads((directory / "adoption.json").read_bytes())
    for filename, expected in adoption["files"].items():
        digest = hashlib.sha256((directory / filename).read_bytes()).hexdigest()
        assert digest == expected["adopted_sha256"], filename
        if not expected["relocated"]:
            assert digest == expected["frozen_sha256"]
    assert not adoption["files"]["predicates.py" if name == "foundation_v2" else "p7_reference/model.py"]["relocated"]


def test_foundation_regeneration_matches_frozen_bytes() -> None:
    frozen = (ROOT / "foundation_v2/cases.json").read_bytes()
    assert encode_corpus() == encode_corpus() == frozen


def test_evidence_regeneration_preserves_corpus_and_semantic_manifest(tmp_path: Path) -> None:
    first = write_bundle(tmp_path / "first")
    second = write_bundle(tmp_path / "second")
    assert first == second
    assert (tmp_path / "first/cases.jsonl").read_bytes() == (tmp_path / "second/cases.jsonl").read_bytes()
    assert (tmp_path / "first/cases.jsonl").read_bytes() == (ROOT / "evidence_v1/cases.jsonl").read_bytes()
    frozen = json.loads((ROOT / "evidence_v1/manifest.json").read_bytes())
    assert {key: value for key, value in first.items() if key != "sources"} == {
        key: value for key, value in frozen.items() if key != "sources"
    }
    adoption = json.loads((ROOT / "evidence_v1/adoption.json").read_bytes())
    assert first["sources"] == {
        Path(filename).name: row["adopted_sha256"]
        for filename, row in adoption["files"].items()
        if filename.startswith("p7_reference/")
    }


def test_publication_regeneration_and_reduced_source_provenance() -> None:
    from tests.graph_sdk.reference.publication_v3.reference import encode_json, generate_cases, generate_text_cases

    directory = ROOT / "publication_v3"
    retirement = json.loads((directory / "retirement.json").read_bytes())
    assert (
        hashlib.sha256((directory / "reference.py").read_bytes()).hexdigest() == retirement["maintained_source_sha256"]
    )
    assert hashlib.sha256((directory / "retirement.patch").read_bytes()).hexdigest() == retirement["patch_sha256"]
    manifest = json.loads((directory / "manifest.json").read_bytes())
    for name, generate in (("full-six", generate_cases), ("simple-text", generate_text_cases)):
        first = b"".join(encode_json(row) + b"\n" for row in generate())
        second = b"".join(encode_json(row) + b"\n" for row in generate())
        assert first == second
        assert hashlib.sha256(first).hexdigest() == manifest["corpora"][name]["cases_sha256"]
        assert retirement["frozen_source_sha256"] == manifest["corpora"][name]["generator_sha256"]
        if name == "simple-text":
            assert first == (directory / "cases.jsonl").read_bytes()
