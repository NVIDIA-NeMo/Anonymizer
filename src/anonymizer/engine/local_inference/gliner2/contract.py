# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Types shared by the native GLiNER2 backend and server.

The server runs as a script from an isolated package directory, so this module imports only the standard
library (and pydantic, which arrives with FastAPI) and is imported by sibling name there. Do not import
``anonymizer`` modules from here.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, NamedTuple, NewType

Label = NewType("Label", str)

Device = Literal["cuda", "mps", "cpu"]
Dtype = Literal["float32", "bfloat16"]
Attention = Literal["standard", "disentangled_flash"]
OverlapPolicy = Literal["longest", "disallow"]

DeviceRequest = Literal["auto", "cuda", "mps", "cpu"]
DtypeRequest = Literal["auto", "float32", "bfloat16"]
AttentionRequest = Literal["auto", "standard", "disentangled_flash"]


class AttentionChoice(NamedTuple):
    """The resolved attention implementation and whether the caller asked for it explicitly."""

    attention: Attention
    explicit: bool


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
