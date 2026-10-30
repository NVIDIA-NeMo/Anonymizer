# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public graph authoring and local execution.

Declare datums and the relations between them, compose operations into a workflow, call :func:`prepare` to
admit the workflow before any operation runs, then call :func:`protect` to run it synchronously on the calling
thread. Callers register the artifact kinds their operations produce with :func:`register_kind`; the kernel
defines none.
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
    InspectionSummary,
    PreparedProtection,
    ProtectionResult,
    TargetOutcome,
    prepare,
    protect,
)
from anonymizer.engine.graph_core.kernel import CancelToken, GraphRejection, register_kind

__all__ = [
    "Assessment",
    "CancelToken",
    "ContextScope",
    "ContextSource",
    "ContextView",
    "Datum",
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
    "Requests",
    "TargetOutcome",
    "Workflow",
    "prepare",
    "protect",
    "register_kind",
]
