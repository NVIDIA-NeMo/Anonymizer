# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import Mock

import pandas as pd

from anonymizer.config.anonymizer_config import AnonymizerConfig, Rewrite
from anonymizer.config.models import ModelSelection
from anonymizer.config.replace_strategies import Redact
from anonymizer.engine.constants import (
    COL_CONTEXT_BINDING_ID,
    COL_CONTEXT_ORDINAL,
    COL_CONTEXT_OWNER_WORK_ID,
    COL_CONTEXT_TEXT,
    COL_FINAL_ENTITIES,
    COL_REPLACED_TEXT,
    COL_REWRITTEN_TEXT,
    COL_TEXT,
)
from anonymizer.engine.detection.detection_workflow import EntityDetectionResult, EntityDetectionWorkflow
from anonymizer.engine.execution.context_workframes import (
    _BackendArtifactId,
    _ContextPayload,
    _ContextPayloadToken,
)
from anonymizer.engine.execution.invocation import _CompiledInvocation
from anonymizer.engine.execution.pandas_runtime import _PandasRuntime
from anonymizer.engine.private_row_verification import _InvocationRowVerifier
from anonymizer.engine.replace.replace_runner import ReplacementResult, ReplacementWorkflow
from anonymizer.engine.rewrite.rewrite_workflow import RewriteResult, RewriteWorkflow


def _detected(frame: pd.DataFrame) -> EntityDetectionResult:
    return EntityDetectionResult(dataframe=frame.assign(**{COL_FINAL_ENTITIES: [{"entities": []}]}), failed_records=[])


def test_runtime_delegates_replace_with_existing_workflow_arguments(stub_slim_model_selection: ModelSelection) -> None:
    frame = pd.DataFrame({COL_TEXT: ["Alice"]})
    verifier = _InvocationRowVerifier(frame)
    bound = verifier.bind(frame)
    detection = Mock(spec=EntityDetectionWorkflow)
    detection.run.side_effect = lambda dataframe, **_: _detected(dataframe)
    replace = Mock(spec=ReplacementWorkflow)
    replace.run.side_effect = lambda dataframe, **_: ReplacementResult(
        dataframe=dataframe.assign(**{COL_REPLACED_TEXT: ["[REDACTED]"]}), failed_records=[]
    )
    runtime = _PandasRuntime(
        detection_workflow=detection,
        replace_runner=replace,
        rewrite_runner=Mock(spec=RewriteWorkflow),
        combined_rewrite_runner=Mock(spec=RewriteWorkflow),
    )

    result = runtime.run(
        bound,
        invocation=_CompiledInvocation.compile(AnonymizerConfig(replace=Redact()), stub_slim_model_selection),
        data_summary="support tickets",
        preview_num_records=3,
        verifier=verifier,
    )

    assert "__anonymizer_private_row_correlation__" not in result.dataframe
    assert detection.run.call_args.kwargs["tag_latent_entities"] is False
    assert replace.run.call_args.kwargs["preview_num_records"] == 1


def test_runtime_selects_combined_rewrite_graph(stub_slim_model_selection: ModelSelection) -> None:
    frame = pd.DataFrame({COL_TEXT: ["Alice"]})
    verifier = _InvocationRowVerifier(frame)
    bound = verifier.bind(frame)
    detection = Mock(spec=EntityDetectionWorkflow)
    detection.run.side_effect = lambda dataframe, **_: _detected(dataframe)
    combined = Mock(spec=RewriteWorkflow)
    combined.run.side_effect = lambda dataframe, **_: RewriteResult(
        dataframe=dataframe.assign(**{COL_REWRITTEN_TEXT: ["Someone"]}), failed_records=[]
    )
    rewrite = Mock(spec=RewriteWorkflow)
    runtime = _PandasRuntime(
        detection_workflow=detection,
        replace_runner=Mock(spec=ReplacementWorkflow),
        rewrite_runner=rewrite,
        combined_rewrite_runner=combined,
    )

    runtime.run(
        bound,
        invocation=_CompiledInvocation.compile(
            AnonymizerConfig(rewrite=Rewrite(use_combined_graph=True)), stub_slim_model_selection
        ),
        data_summary=None,
        preview_num_records=None,
        verifier=verifier,
    )

    rewrite.run.assert_not_called()
    combined.run.assert_called_once()


def test_context_framing_does_not_change_phase5_workflow_or_prompt_inputs(
    stub_slim_model_selection: ModelSelection,
) -> None:
    context_canary = "CONTEXT-CANARY-bob@example.test"
    frame = pd.DataFrame({COL_TEXT: ["target text"]})
    verifier = _InvocationRowVerifier(frame, correlations=("target-work",))
    bound = verifier.bind(frame)
    context_frame = pd.DataFrame(
        {
            COL_CONTEXT_BINDING_ID: ["binding-work"],
            COL_CONTEXT_OWNER_WORK_ID: ["target-work"],
            COL_CONTEXT_ORDINAL: [0],
            COL_CONTEXT_TEXT: [_ContextPayload(context_canary, _ContextPayloadToken("binding-work"))],
        }
    )
    detection = Mock(spec=EntityDetectionWorkflow)

    def detect(dataframe: pd.DataFrame, **_kwargs: object) -> EntityDetectionResult:
        assert context_canary not in dataframe.to_string()
        assert COL_CONTEXT_TEXT not in dataframe.columns
        return _detected(dataframe)

    detection.run.side_effect = detect
    replace = Mock(spec=ReplacementWorkflow)
    replace.run.side_effect = lambda dataframe, **_: ReplacementResult(
        dataframe=dataframe.assign(**{COL_REPLACED_TEXT: ["target text"]}), failed_records=[]
    )
    runtime = _PandasRuntime(
        detection_workflow=detection,
        replace_runner=replace,
        rewrite_runner=Mock(spec=RewriteWorkflow),
        combined_rewrite_runner=Mock(spec=RewriteWorkflow),
    )

    result = runtime.run_context(
        bound,
        context_dataframe=context_frame,
        artifact_id=_BackendArtifactId("artifact-work"),
        invocation=_CompiledInvocation.compile(AnonymizerConfig(replace=Redact()), stub_slim_model_selection),
        data_summary=None,
        preview_num_records=None,
        verifier=verifier,
    )

    assert len(result.context_binding_evidence) == 1
    assert result.context_binding_evidence[0].ordinal == 0
    assert result.closure_attestations[0].closed is True
    detection.run.assert_called_once()
    replace.run.assert_called_once()
    assert context_canary not in repr(detection.run.call_args)
    assert context_canary not in repr(replace.run.call_args)
