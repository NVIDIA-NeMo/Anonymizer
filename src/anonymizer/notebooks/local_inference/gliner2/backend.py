# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native GLiNER2 loading and inference for the notebook server."""

from __future__ import annotations

import bisect
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

SUPPORTED_DEVICES = ("auto", "cuda", "mps", "cpu")
MODEL_ID = "fastino/gliner2-privacy-filter-PII-multi"
MODEL_REVISION = "59894c087cb2923b01f337d4ee72f6ff84d5bdd6"


def resolve_device(requested: str) -> str:
    """Resolve and validate a notebook inference device."""
    import torch  # ty: ignore[unresolved-import] -- installed in the isolated server environment

    normalized = requested.strip().lower()
    if normalized not in SUPPORTED_DEVICES:
        raise ValueError(f"Unsupported GLiNER2 device {requested!r}; expected one of {SUPPORTED_DEVICES!r}.")
    if normalized == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested for local GLiNER2, but it is unavailable. "
            "In Colab, select Runtime > Change runtime type > GPU and retry."
        )
    if normalized == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested for local GLiNER2, but it is unavailable on this machine.")
    return normalized


def load_model(device: str) -> Any:
    """Download the pinned checkpoint and load it on ``device``."""
    from gliner2 import GLiNER2  # ty: ignore[unresolved-import] -- isolated server dependency
    from huggingface_hub import snapshot_download

    snapshot_path = snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION)
    compatible_snapshot = _prepare_transformers_v4_snapshot(Path(snapshot_path))
    return GLiNER2.from_pretrained(compatible_snapshot, map_location=device)


def _prepare_transformers_v4_snapshot(snapshot: Path, *, cache_root: Path | None = None) -> Path:
    """Adapt the pinned checkpoint's v5 tokenizer field for GLiNER2's v4 dependency."""
    root = cache_root or Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache")) / "nemo-anonymizer"
    target = root / "model-adapters" / MODEL_REVISION
    tokenizer_config = target / "tokenizer_config.json"
    if tokenizer_config.is_file():
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix="model-adapter-", dir=target.parent))
    temporary = temporary_root / "snapshot"
    try:
        shutil.copytree(snapshot, temporary, copy_function=_link_or_copy)
        config_path = temporary / "tokenizer_config.json"
        config = json.loads(config_path.read_text())
        extra_tokens = config.pop("extra_special_tokens", None)
        if extra_tokens is not None:
            config["additional_special_tokens"] = extra_tokens
        config_path.unlink()
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        try:
            os.replace(temporary, target)
        except OSError:
            if not tokenizer_config.is_file():
                raise
        return target
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)


def _link_or_copy(source: str, destination: str) -> str:
    """Hard-link cached model artifacts when possible, copying as a fallback."""
    resolved_source = str(Path(source).resolve(strict=True))
    try:
        os.link(resolved_source, destination)
        return destination
    except OSError:
        return shutil.copy2(resolved_source, destination)


def overlap_policy(flat_ner: bool) -> str:
    """Translate the existing server flag to GLiNER2 overlap semantics."""
    return "disallow" if flat_ner else "longest"


def flatten_result(result: object) -> list[dict[str, Any]]:
    """Convert GLiNER2's label-keyed result into Anonymizer entity dictionaries."""
    if not isinstance(result, dict):
        raise ValueError(f"Unexpected GLiNER2 result type: {type(result)!r}")
    entities_by_label = result.get("entities", {})
    if not isinstance(entities_by_label, dict):
        raise ValueError("Unexpected GLiNER2 result: 'entities' must be a mapping.")

    flattened: list[dict[str, Any]] = []
    for label, values in entities_by_label.items():
        if not isinstance(values, list):
            raise ValueError(f"Unexpected GLiNER2 values for label {label!r}: expected a list.")
        for value in values:
            if not isinstance(value, dict):
                raise ValueError(f"Unexpected GLiNER2 entity for label {label!r}: {value!r}")
            entity = {
                "text": str(value["text"]),
                "label": str(label),
                "start": int(value["start"]),
                "end": int(value["end"]),
                "score": float(value["confidence"]),
            }
            flattened.append(entity)
    return flattened


def detect_entities_for_texts(
    model: Any,
    texts: list[str],
    labels: list[str],
    *,
    threshold: float,
    chunk_length: int,
    overlap: int,
    flat_ner: bool,
    inference_batch_size: int,
) -> list[list[dict[str, Any]]]:
    """Detect entities while retaining the existing character-chunk contract."""
    if not labels:
        return [[] for _ in texts]
    _validate_chunk_params(chunk_length, overlap)
    if inference_batch_size < 1:
        raise ValueError("inference_batch_size must be >= 1")

    chunk_records: list[tuple[int, int, str]] = []
    for text_index, source_text in enumerate(texts):
        for chunk, offset in _create_text_chunks(source_text, chunk_length, overlap):
            chunk_records.append((text_index, offset, chunk))
    results: list[list[dict[str, Any]]] = [[] for _ in texts]
    if not chunk_records:
        return results

    raw_results = model.batch_extract_entities(
        [chunk for _, _, chunk in chunk_records],
        labels,
        batch_size=inference_batch_size,
        threshold=threshold,
        include_confidence=True,
        include_spans=True,
        overlap_policy=overlap_policy(flat_ner),
    )
    if not isinstance(raw_results, list) or len(raw_results) != len(chunk_records):
        raise ValueError("Unexpected GLiNER2 batch result shape.")
    for (text_index, offset, _), raw_result in zip(chunk_records, raw_results, strict=True):
        for entity in flatten_result(raw_result):
            entity["start"] += offset
            entity["end"] += offset
            results[text_index].append(entity)
    policy = overlap_policy(flat_ner)
    return [_resolve_document_overlaps(entities, policy) for entities in results]


def _validate_chunk_params(chunk_length: int, overlap: int) -> None:
    if chunk_length < 1:
        raise ValueError("chunk_length must be >= 1")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_length:
        raise ValueError("overlap must be less than chunk_length")


def _create_text_chunks(text: str, chunk_length: int, overlap: int) -> list[tuple[str, int]]:
    chunks: list[tuple[str, int]] = []
    start = 0
    while start < len(text):
        chunks.append((text[start : start + chunk_length], start))
        if start + chunk_length >= len(text):
            break
        start += chunk_length - overlap
    return chunks


def _resolve_document_overlaps(
    entities: list[dict[str, Any]],
    policy: str,
) -> list[dict[str, Any]]:
    """Apply GLiNER2 overlap semantics again after document-offset remapping."""
    ranked = sorted(
        enumerate(entities),
        key=lambda row: (
            -float(row[1]["score"]),
            int(row[1]["start"]),
            int(row[1]["end"]),
            row[0],
        ),
    )
    distinct: list[tuple[int, dict[str, Any]]] = []
    seen_entities: set[tuple[int, int, str]] = set()
    for row in ranked:
        entity_key = (int(row[1]["start"]), int(row[1]["end"]), str(row[1]["label"]))
        if entity_key not in seen_entities:
            seen_entities.add(entity_key)
            distinct.append(row)

    if policy == "longest":
        selected = [
            row
            for row in distinct
            if not any(
                int(other[1]["start"]) <= int(row[1]["start"])
                and int(row[1]["end"]) <= int(other[1]["end"])
                and (int(other[1]["start"]) < int(row[1]["start"]) or int(row[1]["end"]) < int(other[1]["end"]))
                for other in distinct
            )
        ]
    elif policy == "disallow":
        representatives: dict[tuple[int, int], tuple[int, dict[str, Any]]] = {}
        for row in distinct:
            boundaries = (int(row[1]["start"]), int(row[1]["end"]))
            representatives.setdefault(boundaries, row)
        selected_boundaries = {
            (int(entity["start"]), int(entity["end"]))
            for _, entity in _maximum_score_non_overlapping(list(representatives.values()))
        }
        selected = [row for row in distinct if (int(row[1]["start"]), int(row[1]["end"])) in selected_boundaries]
    else:  # pragma: no cover - overlap_policy() has a closed result set
        raise ValueError(f"Unsupported overlap policy: {policy!r}")

    return sorted(
        (entity for _, entity in selected),
        key=lambda entity: (int(entity["start"]), int(entity["end"]), str(entity["label"])),
    )


def _maximum_score_non_overlapping(
    ranked: list[tuple[int, dict[str, Any]]],
) -> list[tuple[int, dict[str, Any]]]:
    """Return the maximum-total-score disjoint span set."""
    by_end = sorted(
        ranked,
        key=lambda row: (
            int(row[1]["end"]),
            int(row[1]["start"]),
            -float(row[1]["score"]),
            row[0],
        ),
    )
    ends = [int(entity["end"]) for _, entity in by_end]
    predecessors = [
        bisect.bisect_right(ends, int(entity["start"]), 0, index) - 1 for index, (_, entity) in enumerate(by_end)
    ]
    best: list[tuple[float, tuple[int, ...]]] = [(0.0, ())]

    def rank_selection(selection: tuple[int, ...]) -> tuple[tuple[float, int, int, int], ...]:
        return tuple(
            sorted(
                (
                    -float(by_end[index][1]["score"]),
                    int(by_end[index][1]["start"]),
                    int(by_end[index][1]["end"]),
                    by_end[index][0],
                )
                for index in selection
            )
        )

    for index, (_, entity) in enumerate(by_end):
        previous_score, previous_selection = best[predecessors[index] + 1]
        with_entity = (previous_score + float(entity["score"]), previous_selection + (index,))
        without_entity = best[index]
        if (
            with_entity[0] > without_entity[0]
            or (with_entity[0] == without_entity[0] and len(with_entity[1]) > len(without_entity[1]))
            or (
                with_entity[0] == without_entity[0]
                and len(with_entity[1]) == len(without_entity[1])
                and rank_selection(with_entity[1]) < rank_selection(without_entity[1])
            )
        ):
            best.append(with_entity)
        else:
            best.append(without_entity)
    return [by_end[index] for index in best[-1][1]]
