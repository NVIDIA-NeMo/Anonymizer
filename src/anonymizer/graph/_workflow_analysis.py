# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One visit budget for a complete compiler invocation, including nested analyses."""

from __future__ import annotations

from collections.abc import Callable, Container, Iterable, Iterator, Set
from contextvars import ContextVar
from dataclasses import dataclass, fields, is_dataclass
from functools import wraps
from typing import ParamSpec, TypeVar

from anonymizer.graph._values import ContractViolation, ValidationCode

_T = TypeVar("_T")
_P = ParamSpec("_P")


@dataclass(slots=True)
class _AnalysisMeter:
    limit: int
    work: int = 0

    def tick(self) -> None:
        """Reject before the first visit beyond the admitted work envelope."""
        if self.work == self.limit:
            raise ContractViolation(ValidationCode.ANALYSIS_LIMIT_EXCEEDED)
        self.work += 1


_ANALYSIS: ContextVar[_AnalysisMeter | None] = ContextVar("workflow_analysis", default=None)


def _visits(values: Iterable[_T]) -> Iterator[_T]:
    """Charge each inspected declaration/relation entry, including repeated visits."""
    meter = _ANALYSIS.get()
    for value in values:
        if meter is not None:
            meter.tick()
        yield value


def _bounded_analysis(
    limit_of: Callable[..., int | None],
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    """Share the outer invocation's meter; always restore context on exit."""

    def decorate(function: Callable[_P, _T]) -> Callable[_P, _T]:
        @wraps(function)
        def bounded(*args: _P.args, **kwargs: _P.kwargs) -> _T:
            if _ANALYSIS.get() is not None:
                return function(*args, **kwargs)
            limit = limit_of(*args, **kwargs)
            if limit is None:
                return function(*args, **kwargs)
            token = _ANALYSIS.set(_AnalysisMeter(limit))
            try:
                return function(*args, **kwargs)
            finally:
                _ANALYSIS.reset(token)

        return bounded

    return decorate


def _subset(left: Iterable[_T], right: object) -> bool:
    """Inspect all entries so hash iteration order cannot change visit cost."""
    if not isinstance(right, Container):
        return False
    return all([item in right for item in _visits(left)])


def _equal(left: object, right: object) -> bool:
    """Expose collection comparisons instead of hiding scans in Python operators."""
    if isinstance(left, Set) and isinstance(right, Set):
        return len(left) == len(right) and _subset(left, right)
    if isinstance(left, tuple) and isinstance(right, tuple):
        return len(left) == len(right) and all(
            [_equal(a, b) for a, b in zip(_visits(left), _visits(right), strict=True)]
        )
    if is_dataclass(left) and not isinstance(left, type) and type(left) is type(right):
        # The schema has fixed arity. Recursion exposes only variable collections.
        return all([_equal(getattr(left, field.name), getattr(right, field.name)) for field in fields(left)])
    return left == right


def _union(left: Iterable[_T], right: Iterable[_T]) -> set[_T]:
    result = set(_visits(left))
    result.update(_visits(right))
    return result


def _difference(left: Iterable[_T], right: Container[_T]) -> set[_T]:
    return {item for item in _visits(left) if item not in right}


def _intersection(left: Iterable[_T], right: Container[_T]) -> set[_T]:
    return {item for item in _visits(left) if item in right}


def _visited_pop(values: list[_T]) -> _T:
    meter = _ANALYSIS.get()
    if meter is not None:
        meter.tick()
    return values.pop()
