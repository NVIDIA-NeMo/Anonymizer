# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent qualification_v1 reference: model."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import TypeAlias, cast

Json: TypeAlias = str | int | bool | None | list["Json"] | dict[str, "Json"]


Obj: TypeAlias = dict[str, Json]


CONTRACT_SHA256 = "239bdaf97eda6b90caeb13d29826abead08e2beff6297460c26409b3e1f5d87c"


STRUCTURAL_CONTRACT_SHA256 = "88c0ef075b225847f1b2668d2a749307c220eceb50215718db722119d828dc6b"


MATERIALIZED_VERSION_CONTRACT_SHA256 = "165c7c95bce31a7c5808860f28d012bbe1986bf0712cebdc86fb08ad07afcb21"


MAP_ITEM_EVIDENCE_CONTRACT_SHA256 = "d5e270fe413f4f5632b522e3ce57e0c913a143060997d63b7673268ae9acfbd8"


GENERATOR_VERSION = "qualification-v1-generator-32-dispatched-input-shape-v13"


SELF_TEST_VERSION = "qualification-v1-self-test-32-dispatched-input-shape-v13"


CORPUS_PATH = "future-contracts/r3-map-item-v13/qualification_v1_cases.json"


V10_IDS_SHA256 = "043c433056b1ecb21d17ef48efa8bcca6a678fd6b722e7a91ea2e1b3a05c9544"


TERMINAL_CATEGORIES = {"blocked", "cancelled", "failure", "inconsistent", "lost", "success"}


REASON_CODES = {
    "cancel_requested",
    "contradictory",
    "duplicate",
    "execution_failed",
    "foreign",
    "missing",
    "prerequisite",
    "stale",
    "transport_lost",
}


def arr(x: Json) -> list[Json]:
    return cast(list[Json], x) if isinstance(x, list) else []


def obj(x: Json) -> Obj:
    return cast(Obj, x) if isinstance(x, dict) else {}


def reject(code: str) -> Obj:
    return {"code": code, "status": "rejected"}


def canonical_bytes(xs: Iterable[Obj]) -> bytes:
    return (json.dumps(tuple(xs), indent=2, sort_keys=True) + "\n").encode()


LIMIT_KEYS = {
    "max_absence_revisions",
    "max_consumed_per_assessment",
    "max_coverage_atoms",
    "max_fixed_point_steps",
    "max_port_facts",
    "max_productions",
    "max_provenance_edges",
    "max_required_decisions",
    "max_revision_entries",
    "max_submissions",
    "max_verified_evidence",
    "max_artifacts",
    "max_artifact_bytes",
}


def limits(**kw: int) -> Obj:
    x: Obj = {
        "max_artifact_bytes": 64,
        "max_artifacts": 32,
        "max_absence_revisions": 2,
        "max_consumed_per_assessment": 3,
        "max_coverage_atoms": 2,
        "max_fixed_point_steps": 3,
        "max_port_facts": 16,
        "max_productions": 2,
        "max_provenance_edges": 16,
        "max_required_decisions": 2,
        "max_revision_entries": 16,
        "max_submissions": 3,
        "max_verified_evidence": 3,
    }
    x.update(kw)
    return x
