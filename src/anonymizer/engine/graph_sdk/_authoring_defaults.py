# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Engine-private authoring shorthands that pass the empty declarations engine graphs never use."""

from __future__ import annotations

from functools import partial

from anonymizer.engine.graph_sdk._execution_values import ExecutionLimits
from anonymizer.engine.graph_sdk._graph_values import DatumId
from anonymizer.engine.graph_sdk._ingress_values import IngressLimits
from anonymizer.engine.graph_sdk._workflow_admission import compile_graph
from anonymizer.engine.graph_sdk._workflow_values import OperationSpec, OutcomeSpec
from anonymizer.engine.graph_sdk.data import DataGraph, DataLimits, ValidatedDataGraph

_outcome_spec = partial(
    OutcomeSpec, produced_ports=frozenset(), context=frozenset(), evidence=frozenset(), model_requirements=frozenset()
)
_operation_spec = partial(OperationSpec, inputs=(), outputs=(), output_dependencies=())
_compile_graph = partial(compile_graph, input_bindings=(), output_bindings=(), sequence=(), protection=())
_execution_limits = partial(
    ExecutionLimits, max_remote_outstanding=None, max_runtime_artifact_bytes=None, max_collection_items=None
)
_ingress_limits = partial(
    IngressLimits, max_collection_items=None, max_metadata_items=None, max_text_bytes=None, max_metadata_bytes=None
)


def _validate_targets(graph: DataGraph, *, targets: tuple[DatumId, ...], limits: DataLimits) -> ValidatedDataGraph:
    return graph.validate(
        targets=targets,
        source_relations=(),
        contexts=(),
        dependencies=(),
        coherence=(),
        atomic=(),
        output_regions=(),
        limits=limits,
    )
