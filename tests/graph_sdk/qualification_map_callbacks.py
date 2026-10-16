# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Operation callbacks for reference map qualification scenarios."""

from __future__ import annotations

from dataclasses import dataclass

from anonymizer.engine.graph_sdk.executor import (
    AssessmentFinding,
    LocalAssessmentResult,
    LocalCompleted,
    LocalFailure,
)
from anonymizer.engine.graph_sdk.requests import (
    AssociationInput,
    AssociationResult,
    PortArtifact,
    SemanticAssociation,
    TextArtifactValue,
    TextCollectionItem,
    TextCollectionValue,
)
from anonymizer.graph.workflow import (
    ArtifactType,
)


@dataclass
class _ReferenceMapCallback:
    role: str
    count: int
    text_type: ArtifactType
    collection_type: ArtifactType
    calls: int = 0
    member_failure: bool = False
    expander_failure: bool = False
    versioned_items: bool = False
    membership_port: str = "members"
    expansion_outcome: str = "ok"
    item_promise: str = "P_ITEM"

    async def run(self, request: tuple[AssociationInput, ...]) -> LocalCompleted | LocalFailure:
        self.calls += 1
        (item,) = request
        assert isinstance(item.association, SemanticAssociation)
        if (
            self.role == "FAILED_SOURCE"
            or (self.role == "MN" and self.member_failure)
            or (self.role == "EXP" and self.expander_failure)
        ):
            return LocalFailure(failure="permanent")
        outputs: tuple[PortArtifact, ...] = ()
        assessments: tuple[LocalAssessmentResult, ...] = ()
        if self.role == "EXP":
            outputs = (
                PortArtifact(
                    port=self.membership_port,
                    artifact_type=self.collection_type,
                    artifact=None,
                    value=TextCollectionValue(
                        items=tuple(
                            TextCollectionItem(
                                key=0 if self.versioned_items else index,
                                version=index + 1 if self.versioned_items else 1,
                                value=TextArtifactValue(text=str(index)),
                            )
                            for index in range(self.count)
                        )
                    ),
                ),
            )
        elif self.role in {"MN", "N"}:
            outputs = (
                PortArtifact(
                    port="evidence", artifact_type=self.text_type, artifact=None, value=TextArtifactValue(text="E")
                ),
            )
            if self.role == "N":
                subject = next(port for port in item.inputs if port.port == "subject")
                outputs += (
                    PortArtifact(port="result", artifact_type=self.text_type, artifact=None, value=subject.value),
                )
            assessments = (
                LocalAssessmentResult(
                    association=item.association,
                    promise="P" if self.role == "N" else self.item_promise,
                    evidence_port="evidence",
                    finding=AssessmentFinding(status="satisfied", code="observed"),
                ),
            )
        return LocalCompleted(
            results=(
                AssociationResult(
                    association=item.association,
                    outcome=self.expansion_outcome if self.role == "EXP" else "ok",
                    outputs=outputs,
                    consumed_context_ports=frozenset(),
                ),
            ),
            assessments=assessments,
        )
