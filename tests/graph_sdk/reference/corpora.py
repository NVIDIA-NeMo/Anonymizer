# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate reference fixtures in memory and check their reviewed digests.

Export a corpus for inspection with:
    python -m tests.graph_sdk.reference.corpora effects /tmp/effects.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from tests.graph_sdk.reference import activation_v1, data_v1, effects_v1, qualification_v1, workflow_static_v1

CorpusName = Literal["effects", "qualification", "data", "activation", "workflow_static"]


@lru_cache(maxsize=5)
def corpus_bytes(name: CorpusName) -> bytes:
    """Return deterministic JSON authenticated against the committed manifest."""
    if name == "effects":
        data = effects_v1.canonical_bytes(effects_v1.generate_cases())
    elif name == "qualification":
        data = qualification_v1.canonical_bytes(qualification_v1.generate_cases())
    elif name == "data":
        data = data_v1.canonical_bytes(data_v1.generate_cases())
    elif name == "activation":
        data = activation_v1.canonical_bytes(activation_v1.generate_cases())
    elif name == "workflow_static":
        data = workflow_static_v1.canonical_bytes(workflow_static_v1.generate_cases())
    else:
        raise ValueError(f"Unknown reference corpus: {name!r}")
    manifest = json.loads((Path(__file__).parent / f"{name}_v1_manifest.json").read_bytes())
    if hashlib.sha256(data).hexdigest() != manifest["corpus_sha256"]:
        raise AssertionError(f"Generated {name} corpus differs from its reviewed manifest")
    return data


def load_cases(name: Literal["effects", "qualification"]) -> list[dict[str, Any]]:
    """Return an independent copy so mutation tests cannot contaminate callers."""
    return json.loads(corpus_bytes(name))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", choices=("effects", "qualification", "data", "activation", "workflow_static"))
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_bytes(corpus_bytes(args.corpus))


if __name__ == "__main__":
    main()
