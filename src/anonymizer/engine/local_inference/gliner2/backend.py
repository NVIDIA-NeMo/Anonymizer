# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native GLiNER2 loading and inference for the native server."""

from __future__ import annotations

import bisect
import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from operator import attrgetter
from pathlib import Path
from typing import Any, NamedTuple, Protocol, cast, get_args

try:
    from .contract import (  # noqa: TID252 - direct-script fallback keeps server dependencies isolated
        Attention,
        AttentionChoice,
        AttentionRequest,
        DetectedEntity,
        Device,
        DeviceRequest,
        Dtype,
        DtypeRequest,
        Label,
        OverlapPolicy,
        Span,
    )
except ImportError:  # Direct execution inside the isolated server environment.
    from contract import (  # ty: ignore[unresolved-import] -- direct-script import path
        Attention,
        AttentionChoice,
        AttentionRequest,
        DetectedEntity,
        Device,
        DeviceRequest,
        Dtype,
        DtypeRequest,
        Label,
        OverlapPolicy,
        Span,
    )

SUPPORTED_DEVICES = get_args(DeviceRequest)
SUPPORTED_DTYPES = get_args(DtypeRequest)
SUPPORTED_ATTENTION = get_args(AttentionRequest)
DTYPE_ENV = "ANONYMIZER_LOCAL_GLINER2_DTYPE"
ATTENTION_ENV = "ANONYMIZER_LOCAL_GLINER2_ATTENTION"
DEFAULT_INFERENCE_BATCH_SIZE = 32
DEFAULT_CHUNK_LENGTH = 128
DEFAULT_OVERLAP = 32
MODEL_ID = "fastino/gliner2-privacy-filter-PII-multi"
MODEL_REVISION = "59894c087cb2923b01f337d4ee72f6ff84d5bdd6"

_DTYPE_ALIASES: Mapping[str, str] = {"fp32": "float32", "bf16": "bfloat16"}
_WARMUP_TEXT = "Alice Smith lives in Paris."
_WARMUP_LABELS = ["first_name", "city"]

logger = logging.getLogger(__name__)


class _WordSplitter(Protocol):
    def __call__(self, text: str, lower: bool, /) -> Iterable[tuple[str, int, int]]: ...


class _Processor(Protocol):
    @property
    def word_splitter(self) -> _WordSplitter: ...


class GLiNER2Like(Protocol):
    """The part of the GLiNER2 model this module uses."""

    @property
    def processor(self) -> _Processor: ...

    def batch_extract_entities(self, texts: list[str], labels: list[str], **options: object) -> Sequence[object]: ...


@dataclass(frozen=True)
class LoadedModel:
    """A loaded GLiNER2 model and the device/precision/attention it actually runs with."""

    model: GLiNER2Like
    device: Device
    dtype: Dtype
    attention: Attention


def resolve_device(requested: str) -> Device:
    """Resolve and validate a native inference device."""
    import torch

    normalized = requested.strip().lower()
    if normalized not in SUPPORTED_DEVICES:
        raise ValueError(f"Unsupported GLiNER2 device {requested!r}; expected one of {SUPPORTED_DEVICES!r}.")
    device = cast(DeviceRequest, normalized)
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested for local GLiNER2, but it is unavailable. "
            "In Colab, select Runtime > Change runtime type > GPU and retry."
        )
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested for local GLiNER2, but it is unavailable on this machine.")
    return device


def resolve_dtype(requested: str, device: Device) -> Dtype:
    """Resolve the inference precision: bfloat16 on CUDA, float32 elsewhere."""
    normalized = requested.strip().lower()
    normalized = _DTYPE_ALIASES.get(normalized, normalized)
    if normalized not in SUPPORTED_DTYPES:
        raise ValueError(f"Unsupported GLiNER2 dtype {requested!r}; expected one of {SUPPORTED_DTYPES!r}.")
    dtype = cast(DtypeRequest, normalized)
    if dtype == "auto":
        return "bfloat16" if device == "cuda" else "float32"
    return dtype


def resolve_attention(requested: str, device: Device) -> AttentionChoice:
    """Resolve the attention implementation and whether it was explicitly requested.

    The automatic choice is DisentangledFlash on CUDA and standard attention elsewhere.
    """
    normalized = requested.strip().lower()
    if normalized not in SUPPORTED_ATTENTION:
        raise ValueError(f"Unsupported GLiNER2 attention {requested!r}; expected one of {SUPPORTED_ATTENTION!r}.")
    attention = cast(AttentionRequest, normalized)
    if attention == "auto":
        return AttentionChoice("disentangled_flash" if device == "cuda" else "standard", explicit=False)
    return AttentionChoice(attention, explicit=True)


def load_model(device: Device, *, dtype: str = "auto", attention: str = "auto") -> LoadedModel:
    """Download the pinned checkpoint and load it on ``device`` with the resolved precision and attention.

    An explicitly requested DisentangledFlash that cannot be enabled raises. The automatic CUDA
    default falls back to standard attention with a warning instead.
    """
    from huggingface_hub import snapshot_download

    resolved_dtype = resolve_dtype(dtype, device)
    resolved_attention, explicit_attention = resolve_attention(attention, device)
    snapshot = Path(snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION))
    model = _load_weights(snapshot, device, resolved_dtype)
    if resolved_attention == "disentangled_flash":
        try:
            model.enable_disentangled_flash()
            _warm_up(model)
        except Exception as exc:
            if explicit_attention:
                raise RuntimeError(
                    f"DisentangledFlash attention was requested via {ATTENTION_ENV} but could not be enabled "
                    f"on {device}: {exc}"
                ) from exc
            logger.warning(
                "DisentangledFlash could not be enabled on %s (%s); falling back to standard attention.",
                device,
                exc,
            )
            model = _load_weights(snapshot, device, resolved_dtype)
            resolved_attention = "standard"
    return LoadedModel(model=model, device=device, dtype=resolved_dtype, attention=resolved_attention)


def _load_weights(snapshot: Path, device: Device, dtype: Dtype) -> Any:
    import torch
    from gliner2 import GLiNER2

    model = GLiNER2.from_pretrained(snapshot, map_location=device)
    if dtype == "bfloat16":
        model = model.to(dtype=torch.bfloat16)
    model.eval()
    return model


def _warm_up(model: GLiNER2Like) -> None:
    """Run one tiny extraction so kernel compilation failures surface at load time."""
    model.batch_extract_entities([_WARMUP_TEXT], _WARMUP_LABELS, batch_size=1, threshold=0.5)


def overlap_policy(flat_ner: bool) -> OverlapPolicy:
    """Translate the existing server flag to GLiNER2 overlap semantics."""
    return "disallow" if flat_ner else "longest"


def flatten_result(result: object) -> list[DetectedEntity]:
    """Convert GLiNER2's label-keyed result into entities in chunk coordinates."""
    if not isinstance(result, dict):
        raise ValueError(f"Unexpected GLiNER2 result type: {type(result)!r}")
    entities_by_label = result.get("entities", {})
    if not isinstance(entities_by_label, dict):
        raise ValueError("Unexpected GLiNER2 result: 'entities' must be a mapping.")

    flattened: list[DetectedEntity] = []
    for label, values in entities_by_label.items():
        if not isinstance(values, list):
            raise ValueError(f"Unexpected GLiNER2 values for label {label!r}: expected a list.")
        for value in values:
            if not isinstance(value, dict):
                raise ValueError(f"Unexpected GLiNER2 entity for label {label!r}: {value!r}")
            flattened.append(
                DetectedEntity(
                    span=Span(int(value["start"]), int(value["end"])),
                    label=Label(str(label)),
                    text=str(value["text"]),
                    score=float(value["confidence"]),
                )
            )
    return flattened


class Chunk(NamedTuple):
    """A word window of one input text, with its character offset in that text."""

    text_index: int
    offset: int
    text: str


def detect_entities_for_texts(
    model: GLiNER2Like,
    texts: list[str],
    labels: list[str],
    *,
    threshold: float,
    chunk_length: int,
    overlap: int,
    flat_ner: bool,
    inference_batch_size: int,
) -> list[list[DetectedEntity]]:
    """Detect entities in overlapping native word windows with source-safe spans."""
    if not labels:
        return [[] for _ in texts]
    _validate_chunk_params(chunk_length, overlap)
    if inference_batch_size < 1:
        raise ValueError("inference_batch_size must be >= 1")

    chunks = [
        chunk
        for text_index, source_text in enumerate(texts)
        for chunk in _create_text_chunks(text_index, source_text, chunk_length, overlap, model.processor.word_splitter)
    ]
    results: list[list[DetectedEntity]] = [[] for _ in texts]
    if not chunks:
        return results

    policy = overlap_policy(flat_ner)
    raw_results = model.batch_extract_entities(
        [chunk.text for chunk in chunks],
        labels,
        batch_size=inference_batch_size,
        threshold=threshold,
        include_confidence=True,
        include_spans=True,
        overlap_policy=policy,
    )
    if not isinstance(raw_results, list) or len(raw_results) != len(chunks):
        raise ValueError("Unexpected GLiNER2 batch result shape.")
    for chunk, raw_result in zip(chunks, raw_results, strict=True):
        for raw_entity in flatten_result(raw_result):
            entity = _ground_chunk_entity(raw_entity, chunk.text)
            if entity is not None:
                results[chunk.text_index].append(entity.shifted(chunk.offset))
    return [_resolve_document_overlaps(entities, policy) for entities in results]


def _validate_chunk_params(chunk_length: int, overlap: int) -> None:
    if chunk_length < 1:
        raise ValueError("chunk_length must be >= 1")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_length:
        raise ValueError("overlap must be less than chunk_length")


def _create_text_chunks(
    text_index: int,
    text: str,
    chunk_length: int,
    overlap: int,
    word_splitter: _WordSplitter,
) -> list[Chunk]:
    """Window the model's native word boundaries without changing source coordinates."""
    words = [(start, end) for _, start, end in word_splitter(text, False)]
    chunks: list[Chunk] = []
    for first in range(0, len(words), chunk_length - overlap):
        last = min(first + chunk_length, len(words))
        start, end = words[first][0], words[last - 1][1]
        chunks.append(Chunk(text_index, start, text[start:end]))
        if last == len(words):
            break
    return chunks


def _ground_chunk_entity(entity: DetectedEntity, chunk: str) -> DetectedEntity | None:
    """Remove the pinned GLiNER2 implementation's single synthetic terminal period."""
    start, end = entity.span
    invalid_span = f"GLiNER2 returned inconsistent offsets [{start}, {end}) for a chunk of length {len(chunk)}"
    if not entity.text.strip():
        raise ValueError(invalid_span)
    if 0 <= start < end <= len(chunk) and entity.text == chunk[start:end]:
        return entity
    appended = bool(chunk) and not chunk.endswith((".", "!", "?"))
    if appended and 0 <= start <= len(chunk) and end == len(chunk) + 1 and entity.text == chunk[start:] + ".":
        if start == len(chunk):
            return None
        return replace(entity, span=Span(start, len(chunk)), text=chunk[start:])
    raise ValueError(invalid_span)


def _keep_outermost(candidates: Sequence[DetectedEntity]) -> list[DetectedEntity]:
    """Drop every entity whose span lies strictly inside another candidate's span."""
    return [e for e in candidates if not any(other.span.strictly_contains(e.span) for other in candidates)]


def _best_disjoint(candidates: Sequence[DetectedEntity]) -> list[DetectedEntity]:
    """Keep the entities on the span set of maximum total score, with every label on each chosen span."""
    representatives: dict[Span, DetectedEntity] = {}
    for entity in candidates:
        representatives.setdefault(entity.span, entity)
    chosen = {entity.span for entity in _maximum_score_non_overlapping(list(representatives.values()))}
    return [e for e in candidates if e.span in chosen]


Resolver = Callable[[Sequence[DetectedEntity]], list[DetectedEntity]]
_RESOLVERS: Mapping[OverlapPolicy, Resolver] = {"longest": _keep_outermost, "disallow": _best_disjoint}


def _resolve_document_overlaps(entities: Sequence[DetectedEntity], policy: OverlapPolicy) -> list[DetectedEntity]:
    """Apply GLiNER2 overlap semantics again after document-offset remapping."""
    distinct: list[DetectedEntity] = []
    seen: set[tuple[Span, Label]] = set()
    for entity in sorted(entities, key=attrgetter("rank")):
        if (entity.span, entity.label) not in seen:
            seen.add((entity.span, entity.label))
            distinct.append(entity)
    return sorted(_RESOLVERS[policy](distinct), key=lambda e: (*e.span, e.label))


class Selection(NamedTuple):
    """A set of disjoint entities and its total score."""

    score: float
    picks: tuple[DetectedEntity, ...]

    def with_pick(self, entity: DetectedEntity) -> Selection:
        return Selection(self.score + entity.score, (*self.picks, entity))

    def order(self) -> tuple[float, int, tuple[tuple[float, int, int], ...]]:
        """Preference key, smallest first: higher score, then more picks, then better-ranked picks."""
        return (-self.score, -len(self.picks), tuple(sorted(e.rank for e in self.picks)))


def _maximum_score_non_overlapping(candidates: Sequence[DetectedEntity]) -> list[DetectedEntity]:
    """Return the maximum-total-score disjoint span set."""
    by_end = sorted(candidates, key=lambda e: (e.span.end, e.span.start, -e.score))
    ends = [e.span.end for e in by_end]
    best = [Selection(0.0, ())]
    for index, entity in enumerate(by_end):
        predecessor = bisect.bisect_right(ends, entity.span.start, 0, index) - 1
        # min() keeps the earlier argument on ties, so skipping the entity wins unless taking it is strictly better.
        best.append(min(best[index], best[predecessor + 1].with_pick(entity), key=Selection.order))
    return list(best[-1].picks)
