# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private graph kernel and protection semantics (tier T4); not part of the public ``anonymizer.graph`` surface.

Declare datums and relations, compose operations into a workflow, :func:`prepare` (admission before effects), then
:func:`protect` (synchronous, on the calling thread). Routes register the artifact kinds they need with
:func:`register_kind`; the kernel itself knows none.
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
