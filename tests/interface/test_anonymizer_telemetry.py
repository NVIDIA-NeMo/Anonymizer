# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for telemetry emission from Anonymizer.run() / preview()."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

from anonymizer.config.anonymizer_config import AnonymizerConfig, AnonymizerInput, Rewrite
from anonymizer.config.replace_strategies import Annotate, Hash, Redact, Substitute
from anonymizer.engine.constants import COL_FINAL_ENTITIES, COL_REPLACED_TEXT, COL_REWRITTEN_TEXT, COL_TEXT
from anonymizer.engine.detection.detection_workflow import EntityDetectionResult, EntityDetectionWorkflow
from anonymizer.engine.ndd.adapter import FailedRecord
from anonymizer.engine.replace.replace_runner import ReplacementResult, ReplacementWorkflow
from anonymizer.engine.rewrite.rewrite_workflow import RewriteResult, RewriteWorkflow
from anonymizer.interface.anonymizer import Anonymizer
from anonymizer.telemetry import (
    NOT_APPLICABLE,
    AnonymizerEvent,
    TaskEnum,
    TaskStatusEnum,
)


@pytest.fixture
def stub_input(tmp_path: Path) -> AnonymizerInput:
    csv_path = tmp_path / "input.csv"
    pd.DataFrame({"text": ["Alice works at Acme"]}).to_csv(csv_path, index=False)
    return AnonymizerInput(source=str(csv_path))


def _make_anonymizer(
    detection_return: EntityDetectionResult | None = None,
    replace_return: ReplacementResult | None = None,
    rewrite_return: RewriteResult | None = None,
) -> tuple[Anonymizer, Mock, Mock, Mock]:
    detection_workflow = Mock(spec=EntityDetectionWorkflow)
    detection_workflow.run.return_value = detection_return or EntityDetectionResult(
        dataframe=pd.DataFrame({COL_TEXT: ["Alice works at Acme"], COL_FINAL_ENTITIES: [{"entities": []}]}),
        failed_records=[],
    )
    _replace_df = pd.DataFrame(
        {COL_TEXT: ["Alice works at Acme"], COL_REPLACED_TEXT: ["[REDACTED] works at [REDACTED]"]}
    )
    _replace_df.attrs["original_text_column"] = "text"
    replace_runner = Mock(spec=ReplacementWorkflow)
    replace_runner.run.return_value = replace_return or ReplacementResult(
        dataframe=_replace_df,
        failed_records=[],
    )
    _rewrite_df = pd.DataFrame(
        {
            COL_TEXT: ["Alice works at Acme"],
            COL_REWRITTEN_TEXT: ["Beth works at Globex"],
            "utility_score": [0.85],
            "leakage_mass": [0.3],
            "weighted_leakage_rate": [0.23],
            "any_high_leaked": [False],
            "needs_human_review": [False],
        }
    )
    _rewrite_df.attrs["original_text_column"] = "text"
    rewrite_runner = Mock(spec=RewriteWorkflow)
    rewrite_runner.run.return_value = rewrite_return or RewriteResult(
        dataframe=_rewrite_df,
        failed_records=[],
    )
    anonymizer = Anonymizer(
        detection_workflow=detection_workflow,
        replace_runner=replace_runner,
        rewrite_runner=rewrite_runner,
    )
    return anonymizer, detection_workflow, replace_runner, rewrite_runner


class _FakeHandler:
    """Drop-in replacement for TelemetryHandler that records what was enqueued."""

    enqueued: list[AnonymizerEvent] = []

    def __init__(self, *_, **__):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def enqueue(self, event: AnonymizerEvent) -> None:
        _FakeHandler.enqueued.append(event)

    def flush(self) -> None:  # pragma: no cover - context manager handles it
        pass


@pytest.fixture
def captured_events(monkeypatch: pytest.MonkeyPatch) -> list[AnonymizerEvent]:
    """Patch the TelemetryHandler used inside Anonymizer to capture events.

    Also re-enables NEMO_TELEMETRY_ENABLED for this test (the autouse fixture
    in conftest sets it to "false").
    """
    monkeypatch.setenv("NEMO_TELEMETRY_ENABLED", "true")
    _FakeHandler.enqueued = []
    monkeypatch.setattr("anonymizer.interface.anonymizer.TelemetryHandler", _FakeHandler)
    return _FakeHandler.enqueued


# =============================================================================
# Init-time side effects
# =============================================================================


class TestInitSideEffects:
    def test_session_prefix_is_set(self) -> None:
        # The autouse fixture clears NEMO_SESSION_PREFIX before each test, so
        # we observe Anonymizer.__init__ setting it.
        _make_anonymizer()
        assert os.environ.get("NEMO_SESSION_PREFIX") == "anonymizer-"

    def test_session_prefix_respects_existing_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """setdefault should not override a user-provided prefix."""
        monkeypatch.setenv("NEMO_SESSION_PREFIX", "custom-")
        _make_anonymizer()
        assert os.environ["NEMO_SESSION_PREFIX"] == "custom-"

    def test_deployment_type_defaults_to_sdk(self) -> None:
        _make_anonymizer()
        assert os.environ.get("NEMO_DEPLOYMENT_TYPE") == "sdk"


# =============================================================================
# Emission on the success / error / cancel paths
# =============================================================================


class TestRunEmitsTelemetry:
    def test_run_emits_completed_event(
        self,
        captured_events: list[AnonymizerEvent],
        stub_input: AnonymizerInput,
    ) -> None:
        anonymizer, *_ = _make_anonymizer()
        anonymizer.run(config=AnonymizerConfig(replace=Redact()), data=stub_input)

        assert len(captured_events) == 1
        event = captured_events[0]
        assert event.task == TaskEnum.BATCH
        assert event.task_status == TaskStatusEnum.COMPLETED
        assert event.transformation_type == "redact"
        assert event.job_duration_sec >= 0

    def test_run_emits_error_event_and_reraises(
        self,
        captured_events: list[AnonymizerEvent],
        stub_input: AnonymizerInput,
    ) -> None:
        anonymizer, detection_wf, _, _ = _make_anonymizer()
        detection_wf.run.side_effect = RuntimeError("kaboom")

        with pytest.raises(RuntimeError, match="kaboom"):
            anonymizer.run(config=AnonymizerConfig(replace=Redact()), data=stub_input)

        assert len(captured_events) == 1
        assert captured_events[0].task_status == TaskStatusEnum.ERROR

    def test_run_emits_canceled_event_on_keyboard_interrupt(
        self,
        captured_events: list[AnonymizerEvent],
        stub_input: AnonymizerInput,
    ) -> None:
        anonymizer, detection_wf, _, _ = _make_anonymizer()
        detection_wf.run.side_effect = KeyboardInterrupt()

        with pytest.raises(KeyboardInterrupt):
            anonymizer.run(config=AnonymizerConfig(replace=Redact()), data=stub_input)

        assert len(captured_events) == 1
        assert captured_events[0].task_status == TaskStatusEnum.CANCELED


class TestPreviewEmitsTelemetry:
    def test_preview_emits_task_preview(
        self,
        captured_events: list[AnonymizerEvent],
        stub_input: AnonymizerInput,
    ) -> None:
        anonymizer, *_ = _make_anonymizer()
        anonymizer.preview(config=AnonymizerConfig(replace=Redact()), data=stub_input, num_records=5)

        assert len(captured_events) == 1
        assert captured_events[0].task == TaskEnum.PREVIEW
        assert captured_events[0].task_status == TaskStatusEnum.COMPLETED


# =============================================================================
# Opt-out
# =============================================================================


class TestOptOut:
    def test_config_emit_telemetry_false_skips_emission(
        self,
        captured_events: list[AnonymizerEvent],
        stub_input: AnonymizerInput,
    ) -> None:
        anonymizer, *_ = _make_anonymizer()
        anonymizer.run(
            config=AnonymizerConfig(replace=Redact(), emit_telemetry=False),
            data=stub_input,
        )
        assert captured_events == []

    def test_env_var_disables_emission(
        self,
        monkeypatch: pytest.MonkeyPatch,
        captured_events: list[AnonymizerEvent],
        stub_input: AnonymizerInput,
    ) -> None:
        """``NEMO_TELEMETRY_ENABLED=false`` short-circuits BEFORE event construction.

        The path must not pay the tiktoken cost or build the event when the env-var
        opt-out is active — so the FakeHandler should never see anything enqueued.
        """
        # captured_events fixture set this to "true"; flip back.
        monkeypatch.setenv("NEMO_TELEMETRY_ENABLED", "false")
        anonymizer, *_ = _make_anonymizer()
        anonymizer.run(config=AnonymizerConfig(replace=Redact()), data=stub_input)
        assert captured_events == []


# =============================================================================
# Field population
# =============================================================================


class TestFieldPopulation:
    def test_substitute_populates_replacement_generator(
        self,
        captured_events: list[AnonymizerEvent],
        stub_input: AnonymizerInput,
    ) -> None:
        anonymizer, *_ = _make_anonymizer()
        anonymizer.run(config=AnonymizerConfig(replace=Substitute()), data=stub_input)

        event = captured_events[0]
        assert event.transformation_type == "substitute"
        assert event.replacement_generator_model != NOT_APPLICABLE
        # Rewrite-only fields stay not_applicable
        assert event.rewriter_model == NOT_APPLICABLE
        assert event.judge_model == NOT_APPLICABLE
        assert event.max_repair_iterations == -1

    def test_rewrite_populates_rewrite_models(
        self,
        captured_events: list[AnonymizerEvent],
        stub_input: AnonymizerInput,
    ) -> None:
        anonymizer, *_ = _make_anonymizer()
        anonymizer.run(
            config=AnonymizerConfig(rewrite=Rewrite(max_repair_iterations=2, strict_entity_protection=True)),
            data=stub_input,
        )

        event = captured_events[0]
        assert event.transformation_type == "rewrite"
        assert event.rewriter_model != NOT_APPLICABLE
        assert event.repairer_model != NOT_APPLICABLE
        assert event.max_repair_iterations == 2
        assert event.strict_entity_protection is True
        # judge runs in evaluate(), not run() — stays not_applicable here
        assert event.judge_model == NOT_APPLICABLE
        # Substitute-only field stays not_applicable
        assert event.replacement_generator_model == NOT_APPLICABLE

    def test_custom_data_summary_detected(
        self,
        captured_events: list[AnonymizerEvent],
        tmp_path: Path,
    ) -> None:
        csv_path = tmp_path / "input.csv"
        pd.DataFrame({"text": ["Alice"]}).to_csv(csv_path, index=False)
        data = AnonymizerInput(source=str(csv_path), data_summary="medical records about clinical trials")

        anonymizer, *_ = _make_anonymizer()
        anonymizer.run(config=AnonymizerConfig(replace=Redact()), data=data)

        assert captured_events[0].custom_data_summary_provided is True

    @pytest.mark.parametrize(
        "strategy_factory,expected_value",
        [
            (Redact, "redact"),
            (Hash, "hash"),
            (Substitute, "substitute"),
            (Annotate, "annotate"),
        ],
    )
    def test_transformation_type_matches_schema_enum(
        self,
        captured_events: list[AnonymizerEvent],
        stub_input: AnonymizerInput,
        strategy_factory,
        expected_value: str,
    ) -> None:
        """Every replace strategy must map to one of the schema's enum values."""
        anonymizer, *_ = _make_anonymizer()
        anonymizer.run(config=AnonymizerConfig(replace=strategy_factory()), data=stub_input)

        assert captured_events[0].transformation_type == expected_value

    def test_default_model_hosts_reflect_bundled_providers(
        self,
        captured_events: list[AnonymizerEvent],
        stub_input: AnonymizerInput,
    ) -> None:
        """Plain Anonymizer() classifies hosts from bundled providers.yaml."""
        anonymizer, *_ = _make_anonymizer()
        anonymizer.run(config=AnonymizerConfig(replace=Redact()), data=stub_input)

        assert captured_events[0].model_hosts == ["nvidia-build"]


# =============================================================================
# Failure-count aggregation
# =============================================================================


class TestFailureAggregation:
    def test_failure_counts_grouped_by_workflow_name(
        self,
        captured_events: list[AnonymizerEvent],
        stub_input: AnonymizerInput,
    ) -> None:
        detection_failures = [
            FailedRecord(record_id="r1", step="entity-detection", reason="x"),
            FailedRecord(record_id="r2", step="entity-detection", reason="y"),
        ]
        detection_return = EntityDetectionResult(
            dataframe=pd.DataFrame({COL_TEXT: ["a"], COL_FINAL_ENTITIES: [{"entities": []}]}),
            failed_records=detection_failures,
        )
        anonymizer, *_ = _make_anonymizer(detection_return=detection_return)
        anonymizer.run(config=AnonymizerConfig(replace=Redact()), data=stub_input)

        event = captured_events[0]
        assert event.entity_detection_failure_count == 2
        assert event.num_failure_records == 2

    def test_repair_iteration_suffixes_aggregate(
        self,
        captured_events: list[AnonymizerEvent],
        stub_input: AnonymizerInput,
    ) -> None:
        """rewrite-repair-1 and rewrite-repair-2 both aggregate into rewrite_repair_failure_count."""
        rewrite_failures = [
            FailedRecord(record_id="r1", step="rewrite-repair-1", reason="x"),
            FailedRecord(record_id="r2", step="rewrite-repair-2", reason="y"),
            FailedRecord(record_id="r3", step="rewrite-repair-2", reason="z"),
        ]
        _df = pd.DataFrame({COL_TEXT: ["a"], COL_REWRITTEN_TEXT: ["b"]})
        _df.attrs["original_text_column"] = "text"
        rewrite_return = RewriteResult(dataframe=_df, failed_records=rewrite_failures)

        anonymizer, *_ = _make_anonymizer(rewrite_return=rewrite_return)
        anonymizer.run(config=AnonymizerConfig(rewrite=Rewrite()), data=stub_input)

        event = captured_events[0]
        assert event.rewrite_repair_failure_count == 3
        # Distinct iteration numbers seen
        assert event.repair_iterations_triggered == 2

    def test_unknown_step_lands_in_catch_all(
        self,
        captured_events: list[AnonymizerEvent],
        stub_input: AnonymizerInput,
    ) -> None:
        unknown_failures = [FailedRecord(record_id="r1", step="some-future-step", reason="x")]
        detection_return = EntityDetectionResult(
            dataframe=pd.DataFrame({COL_TEXT: ["a"], COL_FINAL_ENTITIES: [{"entities": []}]}),
            failed_records=unknown_failures,
        )
        anonymizer, *_ = _make_anonymizer(detection_return=detection_return)
        anonymizer.run(config=AnonymizerConfig(replace=Redact()), data=stub_input)

        event = captured_events[0]
        assert event.unknown_step_failure_count == 1
        assert event.entity_detection_failure_count == 0

    def test_no_failures_counts_zero(
        self,
        captured_events: list[AnonymizerEvent],
        stub_input: AnonymizerInput,
    ) -> None:
        anonymizer, *_ = _make_anonymizer()
        anonymizer.run(config=AnonymizerConfig(replace=Redact()), data=stub_input)

        event = captured_events[0]
        assert event.num_failure_records == 0
        assert event.entity_detection_failure_count == 0
        assert event.unknown_step_failure_count == 0
