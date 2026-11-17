# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public graph authoring and local execution.

Declare datums and the relations between them, compose operations into a workflow, call :func:`prepare` to
admit the workflow before any operation runs, then call :func:`protect` to run it synchronously on the calling
thread. The kernel registers the kinds ``text``, ``mentions``, ``context``, ``datum``, ``assessment``,
``session-view`` and ``decisions``; callers register any other kind their operations produce with
:func:`register_kind`. A :class:`Session` passed as ``session=`` keeps assignments across calls.
"""

from __future__ import annotations

from anonymizer.engine.graph_core.authoring import (
    Assessment,
    ContextScope,
    ContextSource,
    ContextView,
    Datum,
    Dependency,
    GraphScope,
    Limits,
    Mention,
    Operation,
    OutputGroup,
    ProtectionGraph,
    Requests,
    Workflow,
)
from anonymizer.engine.graph_core.execution import (
    Advisory,
    InspectionSummary,
    PreparedProtection,
    ProtectionResult,
    Reason,
    TargetOutcome,
    prepare,
    protect,
)
from anonymizer.engine.graph_core.kernel import CancelToken, GraphRejection, register_kind
from anonymizer.engine.graph_core.session import (
    Decision,
    Session,
    SessionMode,
    SessionSnapshot,
    SessionView,
    SubjectKey,
)

__all__ = [
    "Advisory",
    "Assessment",
    "CancelToken",
    "ContextScope",
    "ContextSource",
    "ContextView",
    "Datum",
    "Decision",
    "Dependency",
    "GraphRejection",
    "GraphScope",
    "InspectionSummary",
    "Limits",
    "Mention",
    "Operation",
    "OutputGroup",
    "PreparedProtection",
    "ProtectionGraph",
    "ProtectionResult",
    "Reason",
    "Requests",
    "Session",
    "SessionMode",
    "SessionSnapshot",
    "SessionView",
    "SubjectKey",
    "TargetOutcome",
    "Workflow",
    "prepare",
    "protect",
    "register_kind",
]
