# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Load unchanged historical fixtures with their reviewed content digests.

These are frozen regression answers, not runtime independent interpreters.
Active independent finite models live in the compact reference packages.
Export an authenticated corpus with:
    python -m tests.graph_sdk.reference.corpora effects /tmp/effects.json
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

CorpusName = Literal["effects", "qualification", "data", "activation", "workflow_static"]
_NAMES = ("effects", "qualification", "data", "activation", "workflow_static")


@lru_cache(maxsize=5)
def corpus_bytes(name: CorpusName) -> bytes:
    """Authenticate the archive and original uncompressed historical JSON."""
    if name not in _NAMES:
        raise ValueError(f"Unknown reference corpus: {name!r}")
    directory = Path(__file__).parent
    archive = json.loads((directory / "frozen" / "manifest.json").read_bytes())[name]
    compressed = (directory / "frozen" / f"{name}.json.gz").read_bytes()
    if (
        len(compressed) != archive["compressed_bytes"]
        or hashlib.sha256(compressed).hexdigest() != archive["compressed_sha256"]
    ):
        raise AssertionError(f"Archived {name} corpus differs from its reviewed manifest")
    data = gzip.decompress(compressed)
    manifest = json.loads((directory / f"{name}_v1_manifest.json").read_bytes())
    if len(data) != archive["original_bytes"] or hashlib.sha256(data).hexdigest() != manifest["corpus_sha256"]:
        raise AssertionError(f"Archived {name} corpus differs from its reviewed manifest")
    return data


def load_cases(name: Literal["effects", "qualification"]) -> list[dict[str, Any]]:
    """Return an independent copy so mutation tests cannot contaminate callers."""
    return json.loads(corpus_bytes(name))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", choices=_NAMES)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_bytes(corpus_bytes(args.corpus))


if __name__ == "__main__":
    main()
