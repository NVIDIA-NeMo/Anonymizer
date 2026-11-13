# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Types shared by the native GLiNER2 backend and server.

The server runs as a script from an isolated package directory, so this module imports only the standard
library (and pydantic, which arrives with FastAPI) and is imported by sibling name there. Do not import
``anonymizer`` modules from here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal, NamedTuple, NewType

from pydantic import BaseModel, Field, JsonValue, field_validator

DEFAULT_THRESHOLD = 0.3
DEFAULT_FLAT_NER = False
DEFAULT_INFERENCE_BATCH_SIZE = 32
DEFAULT_CHUNK_LENGTH = 128
DEFAULT_OVERLAP = 32

Label = NewType("Label", str)

Device = Literal["cuda", "mps", "cpu"]
Dtype = Literal["float32", "bfloat16"]
Attention = Literal["standard", "disentangled_flash"]
OverlapPolicy = Literal["longest", "disallow"]

DeviceRequest = Literal["auto", "cuda", "mps", "cpu"]
DtypeRequest = Literal["auto", "float32", "bfloat16"]
AttentionRequest = Literal["auto", "standard", "disentangled_flash"]


class Span(NamedTuple):
    """A half-open character range ``[start, end)``."""

    start: int
    end: int

    def shifted(self, offset: int) -> Span:
        return Span(self.start + offset, self.end + offset)

    def strictly_contains(self, other: Span) -> bool:
        """Whether ``other`` lies inside this span and is not the same span."""
        return self.start <= other.start and other.end <= self.end and self != other


@dataclass(frozen=True, slots=True)
class DetectedEntity:
    """One detected entity with source-text coordinates."""

    span: Span
    label: Label
    text: str
    score: float

    def shifted(self, offset: int) -> DetectedEntity:
        return replace(self, span=self.span.shifted(offset))

    @property
    def rank(self) -> tuple[float, int, int]:
        """Sort key: highest score first, then earliest span."""
        return (-self.score, *self.span)


class DetectionRequest(BaseModel):
    """The detector chat-completion request: the text to scan plus the detection settings.

    Unknown keys (``max_tokens`` and the like) are ignored, as OpenAI-compatible clients send them.
    """

    model: str | None = None
    messages: JsonValue = Field(default_factory=list)
    labels: list[str] = Field(default_factory=list)
    threshold: float = DEFAULT_THRESHOLD
    chunk_length: int = DEFAULT_CHUNK_LENGTH
    overlap: int = DEFAULT_OVERLAP
    flat_ner: bool = DEFAULT_FLAT_NER
    batch_size: int = DEFAULT_INFERENCE_BATCH_SIZE

    @field_validator("labels", mode="before")
    @classmethod
    def _labels_as_strings(cls, value: object) -> object:
        if not value:
            return []
        if isinstance(value, list):
            return [str(label) for label in value]
        return value

    def text(self) -> str:
        """The text to scan: the content of the last message, with content parts joined."""
        if not isinstance(self.messages, list) or not self.messages:
            return ""
        last = self.messages[-1]
        if not isinstance(last, dict):
            return ""
        content = last.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
        return str(content)


class WireEntity(BaseModel):
    """One entity as the detector reports it, in source-text coordinates."""

    text: str
    label: str
    start: int
    end: int
    score: float

    @classmethod
    def from_entity(cls, entity: DetectedEntity) -> WireEntity:
        return cls(
            text=entity.text, label=entity.label, start=entity.span.start, end=entity.span.end, score=entity.score
        )


class DetectionContent(BaseModel):
    """The JSON document carried in the assistant message content."""

    entities: list[WireEntity]

    @classmethod
    def from_entities(cls, entities: Sequence[DetectedEntity]) -> DetectionContent:
        return cls(entities=[WireEntity.from_entity(entity) for entity in entities])


class ChatMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class ChatChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: Literal["stop"] = "stop"


class ChatUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class DetectionResponse(BaseModel):
    """The detector chat-completion response."""

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    usage: ChatUsage = Field(default_factory=ChatUsage)


class ModelCard(BaseModel):
    """What the server reports about the checkpoint it loaded and how it runs it."""

    id: str
    object: Literal["model"] = "model"
    revision: str
    device: Device
    dtype: Dtype
    attention: Attention


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard]
