# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opaque invocation identities and keyed terminal evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeAlias, TypeVar, final

from anonymizer.engine.execution.accounting_plan import _TaskKey

T = TypeVar("T")


class _PrivateEvidenceValue:
    def __repr__(self) -> str:
        return f"<private {type(self).__name__.strip('_').replace('_', ' ').lower()}>"

    def __reduce__(self) -> str | tuple[object, ...]:
        raise TypeError("private accounting evidence is not serializable")


@final
@dataclass(frozen=True, slots=True, repr=False)
class _InvocationId(_PrivateEvidenceValue):
    value: str


@final
@dataclass(frozen=True, slots=True, repr=False)
class _AttemptId(_PrivateEvidenceValue):
    value: str


@final
@dataclass(frozen=True, slots=True, repr=False)
class _RowToken(_PrivateEvidenceValue):
    value: str


@final
@dataclass(frozen=True, slots=True, repr=False)
class _Dispatch(_PrivateEvidenceValue):
    invocation_id: _InvocationId
    task: _TaskKey
    attempt_id: _AttemptId
    row_token: _RowToken


@final
@dataclass(frozen=True, slots=True, repr=False)
class _SuccessRecord(_PrivateEvidenceValue, Generic[T]):
    dispatch: _Dispatch
    candidate: T


@final
@dataclass(frozen=True, slots=True, repr=False)
class _FailureRecord(_PrivateEvidenceValue):
    dispatch: _Dispatch


_TerminalRecord: TypeAlias = _SuccessRecord[T] | _FailureRecord
