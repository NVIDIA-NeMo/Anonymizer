#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark-only DataDesigner structured-output parser compatibility modes."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from enum import StrEnum
from typing import Any

from data_designer.engine.models.parsers.errors import ParserException
from data_designer.engine.models.recipes import response_recipes as recipes
from data_designer.engine.processing.gsonschema.validators import JSONSchemaValidationError, validate


class DDParserCompatMode(StrEnum):
    none = "none"
    raw_json = "raw_json"


@contextmanager
def dd_parser_compat_context(mode: DDParserCompatMode) -> Iterator[None]:
    """Temporarily patch DataDesigner parsers for benchmark-only compatibility."""
    if mode == DDParserCompatMode.none:
        yield
        return
    if mode != DDParserCompatMode.raw_json:
        raise ValueError(f"unsupported DataDesigner parser compatibility mode: {mode}")

    original_pydantic = recipes.PydanticResponseRecipe._build_parser_fn
    original_structured = recipes.StructuredResponseRecipe._build_parser_fn
    recipes.PydanticResponseRecipe._build_parser_fn = _tolerant_pydantic_builder(original_pydantic)  # type: ignore[method-assign]
    recipes.StructuredResponseRecipe._build_parser_fn = _tolerant_structured_builder(original_structured)  # type: ignore[method-assign]
    try:
        yield
    finally:
        recipes.PydanticResponseRecipe._build_parser_fn = original_pydantic  # type: ignore[method-assign]
        recipes.StructuredResponseRecipe._build_parser_fn = original_structured  # type: ignore[method-assign]


def _tolerant_pydantic_builder(original: Callable[..., Callable[[str], Any]]) -> Callable[..., Callable[[str], Any]]:
    def build_parser(self: Any) -> Callable[[str], Any]:
        base_parse = original(self)

        def parse(response: str) -> Any:
            try:
                return base_parse(response)
            except ParserException as exc:
                try:
                    return self.data_type.model_validate(_load_embedded_json(response))
                except Exception:
                    raise exc

        return parse

    return build_parser


def _tolerant_structured_builder(
    original: Callable[..., Callable[[str], dict]],
) -> Callable[..., Callable[[str], dict]]:
    def build_parser(self: Any) -> Callable[[str], dict]:
        base_parse = original(self)

        def parse(response: str) -> dict:
            try:
                return base_parse(response)
            except ParserException as exc:
                try:
                    return validate(_load_embedded_json(response), **self._validate_args)
                except (json.JSONDecodeError, JSONSchemaValidationError, TypeError, ValueError):
                    raise exc

        return parse

    return build_parser


def _load_embedded_json(response: str) -> Any:
    """Return the largest JSON object/array embedded in a model response."""
    decoder = json.JSONDecoder()
    stripped = response.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    best: tuple[int, int, Any] | None = None
    for start, char in enumerate(response):
        if char not in "{[":
            continue
        try:
            parsed, end = decoder.raw_decode(response, start)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict | list):
            continue
        # Prefer the candidate that consumes the most response text. That
        # selects the outer response object instead of nested item objects.
        if best is None or end > best[1] or (end == best[1] and start < best[0]):
            best = (start, end, parsed)

    if best is None:
        raise json.JSONDecodeError("No embedded JSON object or array found", response, 0)
    return best[2]
