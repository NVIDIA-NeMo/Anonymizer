# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real DataDesigner detection and local Redact with scripted HTTP transport."""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import pytest
import yaml
from data_designer.config import ModelConfig, ModelProvider
from data_designer.engine.dataset_builders.utils.async_concurrency import ensure_async_engine_loop
from data_designer.interface import DataDesigner

from anonymizer import Anonymizer, AnonymizerConfig, AnonymizerInput
from anonymizer.config.models import DetectionModelSelection, ReplaceModelSelection
from anonymizer.config.replace_strategies import Redact
from anonymizer.engine.constants import COL_FINAL_ENTITIES, COL_REPLACED_TEXT, COL_TEXT
from anonymizer.engine.detection.detection_workflow import EntityDetectionWorkflow
from anonymizer.engine.ndd.adapter import RECORD_ID_COLUMN, NddAdapter
from anonymizer.engine.replace.replace_runner import ReplacementWorkflow
from tests.retained import retained


@pytest.fixture
def scripted_detection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[DataDesigner, list[ModelConfig], DetectionModelSelection, Counter[str]]]:
    calls: Counter[str] = Counter()
    clients: list[Any] = []

    def respond(request: httpx.Request) -> httpx.Response:
        assert "_anonymizer_source_" not in request.content.decode()
        body = json.loads(request.content)
        model = body["model"]
        messages = json.dumps(body["messages"])
        role = (
            "detector"
            if model == "detector"
            else "validator"
            if "Validate entity tags" in messages
            else "augmenter"
            if "Find untagged sensitive entities" in messages
            else "readiness"
        )
        calls[role] += 1
        if role == "detector" and "Alice fails." in messages:
            return httpx.Response(
                400,
                json={
                    "error": {
                        "message": "scripted row failure",
                        "type": "invalid_request_error",
                        "code": "invalid_request_error",
                    },
                    "usage": {"prompt_tokens": 1, "completion_tokens": 0, "total_tokens": 1},
                },
            )
        if role == "detector":
            entities = [{"text": "Alice", "label": "first_name", "start": 0, "end": 5, "score": 0.9}]
            if "Alice Bob" in messages:
                entities.append({"text": "Bob", "label": "first_name", "start": 6, "end": 9, "score": 0.9})
            content = {"entities": [] if "No entities" in messages else entities}
        elif role == "validator":
            content = {
                "decisions": [
                    {"id": entity_id, "decision": "keep"}
                    for entity_id in ("first_name_0_5", "first_name_6_9")
                    if entity_id in messages
                ]
            }
        else:
            assert model == "singleton"
            content = {"entities": []}
        return httpx.Response(
            200,
            json={
                "id": "scripted",
                "object": "chat.completion",
                "created": 0,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(content)
                            if role == "detector"
                            else "```json\n" + json.dumps(content) + "\n```",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    designer = DataDesigner(
        artifact_path=tmp_path / "artifacts",
        model_providers=[
            ModelProvider(
                name="scripted", endpoint="http://scripted.invalid/v1", provider_type="openai", api_key="EMPTY"
            )
        ],
        auto_configure_logging=False,
    )
    original = designer._create_resource_provider

    def resources(*args: Any, **kwargs: Any) -> Any:
        provider: Any = original(*args, **kwargs)
        registry = provider.model_registry
        factory = registry._model_facade_factory

        def facade(*args: Any, **kwargs: Any) -> Any:
            result = factory(*args, **kwargs)
            client = result._client
            assert client._inner._retry_config.max_retries == 0
            assert client._inner._aclient is None and client._inner._client is None
            client._inner._transport = httpx.MockTransport(respond)
            clients.append(client)
            return result

        registry._model_facade_factory = facade
        return provider

    monkeypatch.setattr(designer, "_create_resource_provider", resources)
    models = [
        ModelConfig(alias=name, model=name, provider="scripted", skip_health_check=name == "detector")
        for name in ("detector", "singleton")
    ]
    selected = DetectionModelSelection(
        entity_detector="detector",
        entity_validator=["singleton"],
        entity_augmenter="singleton",
        latent_detector="singleton",
    )
    try:
        yield designer, models, selected, calls
    finally:

        async def close() -> None:
            for client in clients:
                await client.aclose()

        asyncio.run_coroutine_threadsafe(close(), ensure_async_engine_loop()).result(timeout=30)


@pytest.mark.parametrize("preview", [False, True])
def test_real_detection_source_correspondence_and_redact(
    scripted_detection: tuple[DataDesigner, list[ModelConfig], DetectionModelSelection, Counter[str]], preview: bool
) -> None:
    designer, models, selected, calls = scripted_detection
    source = pd.DataFrame(
        {COL_TEXT: ["Alice travels.", "Alice stays."], "arbitrary_extra": [17, 29]}, index=pd.Index([4, 4])
    )
    detected = EntityDetectionWorkflow(NddAdapter(designer)).run(
        source,
        model_configs=models,
        selected_models=selected,
        gliner_detection_threshold=0.3,
        tag_latent_entities=False,
        preview_num_records=2 if preview else None,
    )
    replaced = ReplacementWorkflow().run(
        detected.dataframe,
        replace_method=Redact(),
        model_configs=models,
        selected_models=ReplaceModelSelection(replacement_generator="unused"),
    )
    assert detected.failed_records == replaced.failed_records == []
    assert replaced.dataframe[COL_REPLACED_TEXT].tolist() == [
        "[REDACTED_FIRST_NAME] travels.",
        "[REDACTED_FIRST_NAME] stays.",
    ]
    assert replaced.dataframe["arbitrary_extra"].tolist() == [17, 29]
    assert not any(str(name).startswith("_anonymizer_source_") for name in replaced.dataframe.columns)
    assert calls == {"detector": 2, "validator": 2, "augmenter": 2, "readiness": 1}


def public_anonymizer(scripted_detection, **kwargs):
    designer, models, selected, _ = scripted_detection
    return Anonymizer(
        data_designer=designer,
        model_configs=yaml.safe_dump(
            {
                "model_configs": [model.model_dump(mode="json", exclude_none=True) for model in models],
                "selected_models": {"detection": selected.model_dump()},
            }
        ),
        **kwargs,
    )


@pytest.mark.parametrize("preview", [False, True])
def test_public_source_frame(
    scripted_detection: tuple[DataDesigner, list[ModelConfig], DetectionModelSelection, Counter[str]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    preview: bool,
) -> None:
    designer, models, selected, calls = scripted_detection
    detections = []
    detect = EntityDetectionWorkflow.run

    def detected(self, frame, **kwargs):
        detections.append((frame, kwargs))
        return detect(self, frame, **kwargs)

    def displaced(*args, **kwargs):
        raise AssertionError("production Redact must not call the legacy replacement dispatcher")

    monkeypatch.setattr(EntityDetectionWorkflow, "run", detected)
    monkeypatch.setattr(ReplacementWorkflow, "run", displaced)
    path = tmp_path / "source.csv"
    pd.DataFrame({"text": ["Alice travels.", "Alice stays.", "Alice leaves."], "extra": [17, 29, 43]}).to_csv(
        path, index=False
    )
    anonymizer = public_anonymizer(scripted_detection, artifact_path=tmp_path / "public")
    config = AnonymizerConfig(replace=Redact(), emit_telemetry=False)
    data = AnonymizerInput(source=str(path))
    result = (
        anonymizer.preview(config=config, data=data, num_records=2)
        if preview
        else anonymizer.run(config=config, data=data)
    )
    expected = ["[REDACTED_FIRST_NAME] travels.", "[REDACTED_FIRST_NAME] stays.", "[REDACTED_FIRST_NAME] leaves."][
        : 2 if preview else 3
    ]
    assert result.dataframe["text_replaced"].tolist() == expected
    assert result.trace_dataframe["extra"].tolist() == [17, 29, 43][: len(expected)]
    assert result.failed_records == []
    assert result._execution is not None
    assert len(detections) == 1
    assert len(detections[0][0]) == len(expected)
    assert detections[0][1]["source_frame"] is not None
    assert detections[0][1]["request_bridge"] is None
    assert result._execution._execution.context.prepared.declared_request_upper_bound is None
    assert not result._execution.requests.dispatches
    assert not any(str(name).startswith("_anonymizer_source_") for name in result.trace_dataframe.columns)
    assert calls == {**dict.fromkeys(("detector", "validator", "augmenter"), len(expected)), "readiness": 1}


@pytest.mark.parametrize("rows,preview", [(6, None), (7, None), (20, None), (20, 7), (1025, None)])
def test_public_batch_span_items_are_aggregated(scripted_detection, tmp_path, rows, preview):
    path = tmp_path / "batch.csv"
    pd.DataFrame({"text": ["Alice."] * rows, "extra": list(range(rows))}).to_csv(path, index=False)
    anonymizer = public_anonymizer(scripted_detection, artifact_path=tmp_path / "public")
    config = AnonymizerConfig(replace=Redact(), emit_telemetry=False)
    data = AnonymizerInput(source=str(path))
    result = (
        anonymizer.run(config=config, data=data)
        if preview is None
        else anonymizer.preview(config=config, data=data, num_records=preview)
    )
    selected = rows if preview is None else preview
    assert result.dataframe["text_replaced"].tolist() == ["[REDACTED_FIRST_NAME]."] * selected
    assert result.trace_dataframe["extra"].tolist() == list(range(selected))
    assert result.failed_records == []
    assert len(result._execution.frames) == 1
    assert len(result._execution.final_outputs) == selected
    assert scripted_detection[3] == {
        **dict.fromkeys(("detector", "validator", "augmenter"), selected),
        "readiness": 1,
    }


def test_real_detection_dropped_row_preserves_peer_and_failure(
    scripted_detection: tuple[DataDesigner, list[ModelConfig], DetectionModelSelection, Counter[str]],
) -> None:
    designer, models, selected, calls = scripted_detection
    source = pd.DataFrame(
        {COL_TEXT: ["Alice travels.", "Alice fails."], RECORD_ID_COLUMN: ["keep", "drop"], "extra": [17, 29]}
    )
    result = EntityDetectionWorkflow(NddAdapter(designer)).run(
        source,
        model_configs=models,
        selected_models=selected,
        gliner_detection_threshold=0.3,
        tag_latent_entities=False,
        preview_num_records=2,
    )
    assert result.dataframe[COL_TEXT].tolist() == ["Alice travels."]
    assert result.dataframe["extra"].tolist() == [17]
    assert [(failure.record_id, failure.step, failure.reason) for failure in result.failed_records] == [
        ("drop", "entity-detection", "Record missing from workflow output")
    ]
    assert calls == {"detector": 2, "validator": 1, "augmenter": 1, "readiness": 1}


@pytest.mark.parametrize(
    "preview,unattributed_loss,chunks", [(True, False, 1), (False, False, 1), (True, True, 1), (True, False, 2)]
)
def test_real_dd_whole_frame_uses_invocation_readiness_and_source_requests(
    scripted_detection, monkeypatch, preview, unattributed_loss, chunks
):
    designer, models, selected, calls = scripted_detection
    prefix = "Alice" if chunks == 1 else "Alice Bob"
    source = pd.DataFrame(
        {
            COL_TEXT: [f"{prefix} travels.", f"{prefix} stays."],
            RECORD_ID_COLUMN: ["duplicate", "duplicate"],
            "extra": [{"numbers": [17]}, {"numbers": [29]}],
        },
        index=pd.Index([4, 4]),
    )
    options: dict[str, Any] = {
        "gliner_detection_threshold": 0.3,
        "tag_latent_entities": False,
        "validation_max_entities_per_call": 1,
        "preview_num_records": 2 if preview else None,
    }
    legacy = EntityDetectionWorkflow(NddAdapter(designer)).run(
        source,
        model_configs=models,
        selected_models=selected,
        **options,
    )
    expected = (
        ReplacementWorkflow()
        .run(
            legacy.dataframe,
            replace_method=Redact(),
            model_configs=models,
            selected_models=ReplaceModelSelection(replacement_generator="singleton"),
        )
        .dataframe
    )
    calls.clear()
    if unattributed_loss:
        original = designer.preview

        def lose_output(*args, **kwargs):
            result = original(*args, **kwargs)
            result.dataset = result.dataset.iloc[:1]
            return result

        monkeypatch.setattr(designer, "preview", lose_output)

    result = asyncio.run(run_graph_profile(scripted_detection, source, options, chunks))
    assert result._execution.context.prepared.configuration.purpose == "protection"
    assert result.requests.dispatched_count == 5 + 2 * chunks
    assert calls == {"detector": 2, "validator": 2 * chunks, "augmenter": 2, "readiness": 1}
    assert len(result.frames) == (0 if unattributed_loss else 1)
    assert len(result.final_outputs) == (0 if unattributed_loss else 2)
    if not unattributed_loss:
        projected = result.frames[0].project(result)
        pd.testing.assert_frame_equal(projected, expected)
        assert result.frames[0].failed_records == ()
        assert COL_REPLACED_TEXT not in result.frames[0]._content
        assert COL_FINAL_ENTITIES not in result.frames[0]._content
        checks = [port for port in result.ports if port.port == "assessment"]
        assert len(checks) == 2
        assert [dict(result.artifacts)[port.artifact].text for port in checks] == ["span_redaction_matches"] * 2
        artifacts = dict(result.artifacts)
        assert sorted(artifacts[output.candidate.artifact].text for output in result.final_outputs) == sorted(
            expected[COL_REPLACED_TEXT]
        )
    assert any(fact.disposition == "closed" for fact in result.cleanup)


async def run_graph_profile(
    scripted_detection, source, options, chunks=1, artifact_limit=16, remote_limit=1, replace_method=None
):
    from anonymizer.engine.detection._graph_profile import execute_detection_redact
    from anonymizer.engine.graph_sdk.blocking import BlockingPool
    from anonymizer.engine.graph_sdk.executor import ExecutionLimits

    designer, models, selected, _ = scripted_detection
    pool = BlockingPool(max_workers=1)
    try:
        return await execute_detection_redact(
            source,
            detection=EntityDetectionWorkflow(NddAdapter(designer)),
            model_configs=models,
            selected_models=selected,
            detection_options=options,
            replace_method=replace_method,
            validation_chunks=chunks,
            pool=pool,
            limits=ExecutionLimits(
                max_local_in_flight=1,
                max_remote_outstanding=remote_limit,
                max_runtime_artifacts=artifact_limit,
                max_runtime_artifact_bytes=10000,
                max_collection_items=16,
            ),
        )
    finally:
        await pool.close()


@pytest.mark.parametrize(
    "command,method,template,token",
    [
        ("run", "redact", "<{label}>", "<first_name>"),
        ("preview", "redact", "<{label}>", "<first_name>"),
        ("run", "hash", "<{label}_{digest}>", "<FIRST_NAME_3bc51062973c>"),
        ("preview", "annotate", "<{text}:{label}>", "<Alice:first_name>"),
    ],
)
def test_cli_local_replace_uses_actual_graph(
    scripted_detection, monkeypatch, tmp_path, capsys, command, method, template, token
):
    import anonymizer.interface.anonymizer as facade
    import anonymizer.interface.cli.main as cli

    executions = []
    run = facade.run_detection_redact

    def capture(*args, **kwargs):
        result = run(*args, **kwargs)
        executions.append(result._execution)
        return result

    monkeypatch.setattr(facade, "run_detection_redact", capture)
    monkeypatch.setattr(cli, "Anonymizer", lambda **kwargs: public_anonymizer(scripted_detection))
    monkeypatch.setattr(cli, "_configure_logging", lambda opts: None)
    path = tmp_path / "cli.csv"
    pd.DataFrame({"text": ["Alice travels.", "Alice stays."]}).to_csv(path, index=False)
    arguments = [
        command,
        "--source",
        str(path),
        "--replace",
        method,
        "--format-template",
        template,
        "--no-normalize-label",
    ]
    if command == "preview":
        arguments += ["--num-records", "1"]
    with pytest.raises(SystemExit) as stopped:
        cli.app(arguments)
    assert stopped.value.code == 0
    assert len(executions) == 1 and executions[0] is not None
    if command == "run":
        assert pd.read_csv(tmp_path / "cli_anonymized.csv")["text_replaced"].tolist() == [
            f"{token} travels.",
            f"{token} stays.",
        ]
    else:
        assert f"{token} travels." in capsys.readouterr().out
    assert scripted_detection[3] == {
        "detector": 2 if command == "run" else 1,
        "validator": 2 if command == "run" else 1,
        "augmenter": 2 if command == "run" else 1,
        "readiness": 1,
    }


@pytest.mark.parametrize(
    "texts,preview,error",
    [
        ([], False, "Failed to load generated dataset"),
        (["Alice travels."], True, "Dataset is empty"),
        (["Alice fails."], False, "scripted row failure"),
    ],
)
def test_public_graph_preserves_empty_and_all_failed_errors(scripted_detection, tmp_path, texts, preview, error):
    from anonymizer.interface.errors import AnonymizerWorkflowError

    path = tmp_path / "empty.csv"
    pd.DataFrame({"text": pd.Series(texts, dtype=str)}).to_csv(path, index=False)
    facade = public_anonymizer(scripted_detection)
    config, data = AnonymizerConfig(replace=Redact(), emit_telemetry=False), AnonymizerInput(source=str(path))
    with pytest.raises(AnonymizerWorkflowError, match=error) as caught:
        if preview:
            facade.preview(config=config, data=data, num_records=0)
        else:
            facade.run(config=config, data=data)
    execution = retained(caught.value)
    assert execution is not None and not execution.final_outputs and not execution.frames
    assert execution.requests.dispatched_count == 0
    assert execution._execution.context.prepared.declared_request_upper_bound is None


def test_public_graph_preserves_failed_source_order_and_injected_result(scripted_detection, tmp_path):
    designer, _, _, calls = scripted_detection
    path = tmp_path / "failures.csv"
    pd.DataFrame(
        {
            "text": ["Alice fails.", "Alice travels.", "Alice fails."],
            RECORD_ID_COLUMN: ["first", "kept", "last"],
            "extra": [17, 29, 43],
        }
    ).to_csv(path, index=False)
    config = AnonymizerConfig(replace=Redact(format_template="<{label}>", normalize_label=False), emit_telemetry=False)
    data = AnonymizerInput(source=str(path))
    legacy = public_anonymizer(
        scripted_detection, detection_workflow=EntityDetectionWorkflow(NddAdapter(designer))
    ).run(config=config, data=data)
    assert legacy._execution is not None
    assert legacy._execution._execution.context.prepared.declared_request_upper_bound is None
    assert not legacy._execution.final_outputs
    calls.clear()
    result = public_anonymizer(scripted_detection).run(config=config, data=data)
    assert result._execution is not None
    pd.testing.assert_frame_equal(result.dataframe, legacy.dataframe)
    pd.testing.assert_frame_equal(result.trace_dataframe, legacy.trace_dataframe)
    assert result.failed_records == legacy.failed_records
    assert [failure.record_id for failure in result.failed_records] == ["first", "last"]
    assert result.dataframe["text_replaced"].tolist() == ["<first_name> travels."]
    assert result.trace_dataframe["extra"].tolist() == [29]
    assert calls == {"detector": 3, "validator": 1, "augmenter": 1, "readiness": 1}


@pytest.mark.parametrize("case", ["rate_limit", "second_chunk", "readiness_unknown"])
def test_public_opaque_dd_preserves_backend_retry_and_usage_behavior(scripted_detection, monkeypatch, tmp_path, case):
    from data_designer.engine.models.clients.model_request_executor import ModelRequestExecutor

    from anonymizer import Detect
    from anonymizer.engine.harness.frames import PublishedFrame
    from anonymizer.engine.harness.kernel import qualify_detection_redact
    from tests.engine.test_detection_graph_publication import _modify_dd_responses

    monkeypatch.setattr(ModelRequestExecutor, "_retry_delay_seconds", lambda self, attempt: 0)
    remaining = [1 if case == "rate_limit" else 4 if case == "second_chunk" else 0]

    def response(request, answer):
        body = json.loads(request.content)
        messages = json.dumps(body["messages"])
        retry = (case == "rate_limit" and body["model"] == "detector") or (
            case == "second_chunk" and "Validate entity tags" in messages and "first_name_6_9" in messages
        )
        if remaining[0] and retry:
            remaining[0] -= 1
            return httpx.Response(429 if case == "rate_limit" else 503, json={"error": {"message": "transient"}})
        if case == "readiness_unknown":
            payload = answer.json()
            payload.pop("usage", None)
            return httpx.Response(answer.status_code, json=payload)
        return answer

    _modify_dd_responses(scripted_detection, monkeypatch, response)
    path = tmp_path / "source.csv"
    pd.DataFrame({"text": ["Alice Bob travels." if case == "second_chunk" else "Alice travels."]}).to_csv(
        path, index=False
    )
    result = public_anonymizer(scripted_detection).run(
        config=AnonymizerConfig(
            replace=Redact(), detect=Detect(validation_max_entities_per_call=1), emit_telemetry=False
        ),
        data=AnonymizerInput(source=str(path)),
    )
    expected = (
        "[REDACTED_FIRST_NAME] [REDACTED_FIRST_NAME] travels."
        if case == "second_chunk"
        else "[REDACTED_FIRST_NAME] travels."
    )
    assert result.dataframe.text_replaced.tolist() == [expected]
    calls = scripted_detection[3]
    assert dict(calls) == {
        "detector": 2 if case == "rate_limit" else 1,
        "validator": 7 if case == "second_chunk" else 1,
        "augmenter": 1,
        "readiness": 1,
    }
    execution = result._execution
    assert execution is not None and isinstance(execution.frames[0], PublishedFrame)
    assert execution._execution.context.prepared.declared_request_upper_bound is None
    assert execution._execution.capabilities[0].attribution == "per_task"
    assert not execution.requests.dispatches
    assert not qualify_detection_redact(execution).qualified
