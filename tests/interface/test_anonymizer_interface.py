# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextvars
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock, PropertyMock, patch

import pandas as pd
import pytest
import yaml
from data_designer.config.models import ModelConfig

from anonymizer import RunConfig
from anonymizer.config.anonymizer_config import (
    AnonymizerConfig,
    AnonymizerInput,
    Detect,
    DetectionScope,
    EvaluateConfig,
    Rewrite,
    TextRecord,
    TextRecordsInput,
)
from anonymizer.config.models import ModelSelection, ReplaceModelSelection
from anonymizer.config.options import DetectionOptions, RunContext
from anonymizer.config.regex import RegexLimits, RegexRule
from anonymizer.config.replace_strategies import Annotate, Hash, Redact, ReplaceMethod, Substitute
from anonymizer.config.rewrite import PrivacyGoal
from anonymizer.engine.constants import (
    COL_DETECTED_ENTITIES,
    COL_DETECTION_VALID,
    COL_ENTITIES_BY_VALUE,
    COL_ENTITY_COVERAGE,
    COL_FINAL_ENTITIES,
    COL_JUDGE_EVALUATION,
    COL_MISSED_ENTITIES,
    COL_NEEDS_HUMAN_REVIEW,
    COL_REPLACED_TEXT,
    COL_REPLACEMENT_MAP,
    COL_REWRITTEN_TEXT,
    COL_TAGGED_TEXT,
    COL_TEXT,
    RECORD_ID_COLUMN,
)
from anonymizer.engine.detection.detection_workflow import (
    EntityDetectionResult,
    EntityDetectionWorkflow,
    _build_entities_by_value,
)
from anonymizer.engine.evaluation.entity_coverage_judge import EntityCoverageWorkflow
from anonymizer.engine.graph_core.kernel import CancelToken, run_in_frame
from anonymizer.engine.ndd.adapter import FailedRecord
from anonymizer.engine.ndd.model_loader import (
    load_default_model_providers,
    load_models_config,
    validate_model_alias_references,
)
from anonymizer.engine.replace.llm_replace_workflow import LlmReplaceWorkflow
from anonymizer.engine.replace.replace_runner import ReplacementResult, ReplacementWorkflow
from anonymizer.engine.rewrite.rewrite_workflow import RewriteResult, RewriteWorkflow
from anonymizer.interface.anonymizer import Anonymizer, _resolve_model_providers
from anonymizer.interface.errors import AnonymizerWorkflowError, InvalidConfigError, InvalidInputError
from anonymizer.interface.workflows import DetectionWorkflow
from tests.conftest import stub_run_with_detection


@pytest.fixture
def stub_input(tmp_path: Path) -> AnonymizerInput:
    csv_path = tmp_path / "input.csv"
    pd.DataFrame({"text": ["Alice"]}).to_csv(csv_path, index=False)
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
    replace_runner = Mock(spec=ReplacementWorkflow)
    stub_run_with_detection(replace_runner)
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
    rewrite_runner = Mock(spec=RewriteWorkflow)
    stub_run_with_detection(rewrite_runner)
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


def test_run_merges_failed_records_from_both_stages(
    stub_anonymizer_config: AnonymizerConfig,
    stub_input: AnonymizerInput,
) -> None:
    detection_failures = [FailedRecord(record_id="r1", step="detection", reason="timeout")]
    replace_failures = [FailedRecord(record_id="r2", step="replace", reason="parse error")]

    detection_result = EntityDetectionResult(
        dataframe=pd.DataFrame({COL_TEXT: ["Alice"], COL_FINAL_ENTITIES: [{"entities": []}]}),
        failed_records=detection_failures,
    )
    _replace_df = pd.DataFrame({COL_TEXT: ["Alice"], COL_REPLACED_TEXT: ["Alice"]})
    replace_return = ReplacementResult(dataframe=_replace_df, failed_records=replace_failures)

    anonymizer, _, _, _ = _make_anonymizer(detection_return=detection_result, replace_return=replace_return)
    result = anonymizer.run(config=stub_anonymizer_config, data=stub_input)

    assert len(result.failed_records) == 2
    assert result.failed_records[0].step == "detection"
    assert result.failed_records[1].step == "replace"


def test_run_in_memory_records_exposes_ids_and_entities_in_public_dataframe() -> None:
    ids = ["turn-2", "turn-1"]
    texts = ["Bob works at Acme", "Alice works at Globex"]
    entities = [
        {"entities": [{"value": "Bob", "label": "first_name", "start_position": 0, "end_position": 3}]},
        {"entities": [{"value": "Alice", "label": "first_name", "start_position": 0, "end_position": 5}]},
    ]
    detected = pd.DataFrame({"id": ids, COL_TEXT: texts, COL_FINAL_ENTITIES: entities})
    replaced = detected.assign(
        **{
            COL_REPLACED_TEXT: ["[REDACTED_FIRST_NAME] works at Acme", "[REDACTED_FIRST_NAME] works at Globex"],
            COL_TAGGED_TEXT: ["<Bob, first_name> works at Acme", "<Alice, first_name> works at Globex"],
        }
    )
    anonymizer, detection_workflow, _, _ = _make_anonymizer(
        detection_return=EntityDetectionResult(dataframe=detected, failed_records=[]),
        replace_return=ReplacementResult(dataframe=replaced, failed_records=[]),
    )

    result = anonymizer.run(
        config=AnonymizerConfig(replace=Redact(), emit_telemetry=False),
        data=TextRecordsInput(
            records=[TextRecord(id=record_id, text=text) for record_id, text in zip(ids, texts, strict=True)]
        ),
    )

    assert detection_workflow.run.call_args.args[0]["id"].tolist() == ids
    assert detection_workflow.run.call_args.args[0][RECORD_ID_COLUMN].tolist() == ids
    assert result.dataframe["id"].tolist() == ids
    assert result.dataframe["final_entities"].tolist() == entities


def test_run_in_memory_records_tracks_dropped_caller_id_and_survivor_order() -> None:
    from data_designer.interface.data_designer import DataDesigner

    ids = ["turn-2", "dropped", "turn-1"]
    texts = ["Bob works at Acme", "Drop this record", "Alice works at Globex"]
    surviving_ids = ["turn-2", "turn-1"]
    surviving_texts = [texts[0], texts[2]]
    detected = pd.DataFrame(
        {
            "id": surviving_ids,
            COL_TEXT: surviving_texts,
            RECORD_ID_COLUMN: surviving_ids,
            COL_DETECTED_ENTITIES: [{"entities": []}, {"entities": []}],
            COL_TAGGED_TEXT: surviving_texts,
        }
    )

    def create(config_builder, **_kwargs):
        seed = pd.read_parquet(config_builder.get_seed_config().source.path)
        key_column = next(c for c in seed.columns if c.startswith("_anonymizer_source_"))
        survivors = seed[seed[RECORD_ID_COLUMN].isin(surviving_ids)]
        echoed = detected.assign(**{key_column: survivors[key_column].tolist()})
        return SimpleNamespace(load_dataset=Mock(return_value=echoed))

    data_designer = Mock(spec=DataDesigner)
    data_designer.create.side_effect = create

    result = Anonymizer(data_designer=data_designer).run(
        config=AnonymizerConfig(replace=Redact(), emit_telemetry=False),
        data=TextRecordsInput(
            records=[TextRecord(id=record_id, text=text) for record_id, text in zip(ids, texts, strict=True)]
        ),
    )

    assert result.dataframe["id"].tolist() == surviving_ids
    assert result.failed_records == [
        FailedRecord(
            record_id="dropped",
            step="entity-detection",
            reason="Record missing from workflow output",
        )
    ]


def test_run_enables_latent_detection_when_rewrite_configured(stub_input: AnonymizerInput) -> None:
    rewrite = Rewrite()
    config = AnonymizerConfig(rewrite=rewrite)
    anonymizer, detection_wf, _, _ = _make_anonymizer()
    anonymizer.run(config=config, data=stub_input)

    assert detection_wf.run.call_args.kwargs["latent_goal"] == rewrite.privacy_goal


def test_run_disables_latent_detection_without_rewrite(
    stub_anonymizer_config: AnonymizerConfig,
    stub_input: AnonymizerInput,
) -> None:
    anonymizer, detection_wf, _, _ = _make_anonymizer()
    anonymizer.run(config=stub_anonymizer_config, data=stub_input)

    assert detection_wf.run.call_args.kwargs["latent_goal"] is None


def test_run_passes_detect_entity_labels_to_detection_workflow(stub_input: AnonymizerInput) -> None:
    config = AnonymizerConfig(detect={"entity_labels": ["server_name"]}, replace=Redact())
    anonymizer, detection_wf, _, _ = _make_anonymizer()

    anonymizer.run(config=config, data=stub_input)

    assert detection_wf.run.call_args.kwargs["options"].entity_labels == ["server_name"]


def test_run_passes_entity_label_examples_only_to_detection_workflow(stub_input: AnonymizerInput) -> None:
    examples = {"vendor_api_key": ["acme_live_abc123"]}
    config = AnonymizerConfig(
        detect={"entity_labels": ["vendor_api_key"], "entity_label_examples": examples},
        replace=Redact(),
    )
    anonymizer, detection_wf, replace_runner, _ = _make_anonymizer()

    result = anonymizer.run(config=config, data=stub_input)

    assert detection_wf.run.call_args.kwargs["options"].entity_label_examples == examples
    assert result.entity_labels == ["vendor_api_key"]
    assert not hasattr(result, "entity_label_examples")
    assert "entity_label_examples" not in replace_runner.run_with_detection.call_args.kwargs
    assert "acme_live_abc123" not in (result.dataframe.to_json() or "")


def test_rewrite_run_passes_entity_label_examples_to_detection_workflow(stub_input: AnonymizerInput) -> None:
    examples = {"vendor_api_key": ["acme_live_abc123"]}
    config = AnonymizerConfig(
        detect={"entity_labels": ["vendor_api_key"], "entity_label_examples": examples},
        rewrite=Rewrite(),
    )
    anonymizer, detection_wf, _, rewrite_runner = _make_anonymizer()

    result = anonymizer.run(config=config, data=stub_input)

    assert detection_wf.run.call_args.kwargs["options"].entity_label_examples == examples
    assert detection_wf.run.call_args.kwargs["latent_goal"] is not None
    assert result.entity_labels == ["vendor_api_key"]
    assert "entity_label_examples" not in rewrite_runner.run_with_detection.call_args.kwargs
    assert "acme_live_abc123" not in (result.dataframe.to_json() or "")


def test_preview_passes_entity_label_examples_to_detection_workflow(stub_input: AnonymizerInput) -> None:
    examples = {"api_key": ["sk-ant-api03-abc123"]}
    config = AnonymizerConfig(detect={"entity_label_examples": examples}, replace=Redact())
    anonymizer, detection_wf, _, _ = _make_anonymizer()

    anonymizer.preview(config=config, data=stub_input, num_records=1)

    assert detection_wf.run.call_args.kwargs["options"].entity_label_examples == examples


class _Replay040DetectionWorkflow:
    """A detection workflow written against the pre-v1 keyword list: no `options`, `context` or `latent_goal`."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def run(
        self,
        dataframe: pd.DataFrame,
        *,
        model_configs: list[ModelConfig],
        selected_models: object,
        gliner_detection_threshold: float,
        entity_labels: list[str] | None = None,
        privacy_goal: object | None = None,
        tag_latent_entities: bool = True,
    ) -> EntityDetectionResult:
        self.calls.append("run")
        raise AssertionError("unreachable")

    def build_detection_config(
        self,
        dataframe: pd.DataFrame,
        *,
        seed_path: str | Path,
        model_configs: list[ModelConfig],
        selected_models: object,
    ) -> Mock:
        self.calls.append("config")
        return Mock()

    def build_detection_builder_for_seed(
        self, *, seed_path: str | Path, model_configs: list[ModelConfig], selected_models: object
    ) -> Mock:
        self.calls.append("builder")
        return Mock()


class _RunOnlyDetectionWorkflow:
    def run(
        self,
        dataframe: pd.DataFrame,
        *,
        options: DetectionOptions,
        context: RunContext,
        latent_goal: PrivacyGoal | None = None,
    ) -> EntityDetectionResult:
        raise AssertionError("unreachable")


@pytest.mark.parametrize("mode", ["run", "preview"])
@pytest.mark.parametrize("rewrite", [False, True], ids=["replace", "rewrite"])
def test_keyword_list_detection_workflow_fails_at_first_call(
    stub_input: AnonymizerInput, mode: str, rewrite: bool
) -> None:
    config = AnonymizerConfig(rewrite=Rewrite()) if rewrite else AnonymizerConfig(replace=Redact())
    from data_designer.interface.data_designer import DataDesigner

    detector = _Replay040DetectionWorkflow()
    data_designer = Mock(spec=DataDesigner)
    anonymizer = Anonymizer(data_designer=data_designer, detection_workflow=cast(DetectionWorkflow, detector))

    with pytest.raises(TypeError, match="options"):
        getattr(anonymizer, mode)(config=config, data=stub_input)

    assert detector.calls == []
    data_designer.create.assert_not_called()
    data_designer.preview.assert_not_called()


def test_keyword_list_detection_workflow_fails_on_export(stub_input: AnonymizerInput, tmp_path: Path) -> None:
    config = AnonymizerConfig(replace=Redact())
    detector = _Replay040DetectionWorkflow()
    anonymizer = Anonymizer(detection_workflow=cast(DetectionWorkflow, detector))

    with pytest.raises(TypeError, match="options"):
        anonymizer.export_detection_config(config=config, data=stub_input, seed_path=tmp_path / "seed.parquet")
    with pytest.raises(TypeError, match="options"):
        anonymizer.export_detection_builder_for_seed(config=config, seed_path=tmp_path / "seed.parquet")
    assert detector.calls == []


@pytest.mark.parametrize("method", ["export_detection_config", "export_detection_builder_for_seed"])
def test_export_needs_a_workflow_that_supports_export(stub_input: AnonymizerInput, tmp_path: Path, method: str) -> None:
    anonymizer = Anonymizer(detection_workflow=_RunOnlyDetectionWorkflow())
    seed_path = tmp_path / "seed.parquet"
    kwargs: dict[str, Any] = {"config": AnonymizerConfig(replace=Redact()), "seed_path": seed_path}
    if method == "export_detection_config":
        kwargs["data"] = stub_input

    with pytest.raises(InvalidConfigError, match="does not support export"):
        getattr(anonymizer, method)(**kwargs)
    assert not seed_path.exists()


@pytest.mark.parametrize("mode", ["run", "preview"])
@pytest.mark.parametrize("rewrite", [False, True], ids=["replace", "rewrite"])
def test_detection_workflow_receives_options_and_context(tmp_path: Path, mode: str, rewrite: bool) -> None:
    csv_path = tmp_path / "input.csv"
    pd.DataFrame({"text": ["Alice", "Bob", "Carol"]}).to_csv(csv_path, index=False)
    data = AnonymizerInput(source=str(csv_path), data_summary="Support tickets")
    config = AnonymizerConfig(
        detect=Detect(excluded_entity_labels=["email"]),
        replace=None if rewrite else Redact(),
        rewrite=Rewrite() if rewrite else None,
    )
    anonymizer, detection_wf, _, _ = _make_anonymizer()

    if mode == "run":
        anonymizer.run(config=config, data=data)
    else:
        anonymizer.preview(config=config, data=data, num_records=5)

    kwargs = detection_wf.run.call_args.kwargs
    assert set(kwargs) == {"options", "context", "latent_goal"}
    assert kwargs["options"] == config.detect
    assert kwargs["options"] is not config.detect
    context = kwargs["context"]
    assert isinstance(context, RunContext)
    assert context.data_summary == "Support tickets"
    assert context.preview_num_records == (None if mode == "run" else 3)
    assert context.model_configs == tuple(anonymizer._model_configs)
    assert all(a is not b for a, b in zip(context.model_configs, anonymizer._model_configs, strict=True))
    assert context.models == anonymizer._selected_models
    assert context.models is not anonymizer._selected_models


def test_export_paths_pass_options_and_context(stub_input: AnonymizerInput, tmp_path: Path) -> None:
    config = AnonymizerConfig(detect=Detect(gliner_threshold=0.4), replace=Redact())
    anonymizer, detection_wf, _, _ = _make_anonymizer()
    seed_path = tmp_path / "seed.parquet"

    anonymizer.export_detection_config(config=config, data=stub_input, seed_path=seed_path)
    anonymizer.export_detection_builder_for_seed(
        config=config, seed_path=seed_path, job_index=2, num_jobs=4, data_summary="Notes"
    )

    config_kwargs = detection_wf.build_detection_config.call_args.kwargs
    assert set(config_kwargs) == {"seed_path", "options", "context"}
    assert config_kwargs["seed_path"] == seed_path
    assert config_kwargs["options"] == config.detect
    assert config_kwargs["context"].preview_num_records is None
    seed_kwargs = detection_wf.build_detection_builder_for_seed.call_args.kwargs
    assert set(seed_kwargs) == {"seed_path", "options", "context", "job_index", "num_jobs"}
    assert (seed_kwargs["job_index"], seed_kwargs["num_jobs"]) == (2, 4)
    assert seed_kwargs["options"] == config.detect
    assert seed_kwargs["context"].data_summary == "Notes"
    assert seed_kwargs["context"].preview_num_records is None


@pytest.mark.parametrize("mode", ["run", "preview", "export_detection_config", "export_detection_builder_for_seed"])
@pytest.mark.parametrize(
    "detect",
    [
        Detect(),
        Detect(
            builtin_regexes=False,
            regex_rules=[RegexRule(label="ticket", pattern=r"TKT-\d+")],
            regex_limits=RegexLimits(
                max_matches_per_rule=1234, min_timeout_seconds=0.2, timeout_seconds_per_million_characters=2
            ),
        ),
    ],
)
def test_detection_entrypoints_pass_regex_options(
    stub_input: AnonymizerInput, tmp_path: Path, mode: str, detect: Detect
) -> None:
    anonymizer, detector, _, _ = _make_anonymizer()
    config = AnonymizerConfig(detect=detect, replace=Redact())

    if mode == "export_detection_config":
        anonymizer.export_detection_config(config=config, data=stub_input, seed_path=tmp_path / "seed.parquet")
        call = detector.build_detection_config.call_args
    elif mode == "export_detection_builder_for_seed":
        anonymizer.export_detection_builder_for_seed(config=config, seed_path=tmp_path / "seed.parquet")
        call = detector.build_detection_builder_for_seed.call_args
    else:
        getattr(anonymizer, mode)(config=config, data=stub_input)
        call = detector.run.call_args

    options = call.kwargs["options"]
    assert options.builtin_regexes is detect.builtin_regexes
    assert options.regex_rules == detect.regex_rules
    assert options.regex_limits == detect.regex_limits


@pytest.mark.parametrize("strategy", [Redact(), Substitute(), Annotate(), Hash()], ids=lambda s: type(s).__name__)
def test_latent_goal_is_none_without_latent_stage(stub_input: AnonymizerInput, strategy: ReplaceMethod) -> None:
    anonymizer, detection_wf, _, _ = _make_anonymizer()

    anonymizer.run(config=AnonymizerConfig(replace=strategy), data=stub_input)

    assert detection_wf.run.call_args.kwargs["latent_goal"] is None


def test_latent_goal_is_the_rewrite_goal(stub_input: AnonymizerInput) -> None:
    goal = PrivacyGoal(protect="Names and employers", preserve="The meaning of each ticket")
    anonymizer, detection_wf, _, _ = _make_anonymizer()

    anonymizer.run(config=AnonymizerConfig(rewrite=Rewrite(privacy_goal=goal)), data=stub_input)

    assert detection_wf.run.call_args.kwargs["latent_goal"] == goal


def test_export_detection_paths_pass_entity_label_examples(
    stub_input: AnonymizerInput,
    tmp_path: Path,
) -> None:
    examples = {"vendor_api_key": ["acme_live_abc123"]}
    config = AnonymizerConfig(
        detect={"entity_labels": ["vendor_api_key"], "entity_label_examples": examples},
        replace=Redact(),
    )
    anonymizer, detection_wf, _, _ = _make_anonymizer()
    detection_wf.build_detection_config.return_value = Mock()
    detection_wf.build_detection_builder_for_seed.return_value = Mock()
    seed_path = tmp_path / "seed.parquet"

    anonymizer.export_detection_config(config=config, data=stub_input, seed_path=seed_path)
    anonymizer.export_detection_builder_for_seed(config=config, seed_path=seed_path)

    assert detection_wf.build_detection_config.call_args.kwargs["options"].entity_label_examples == examples
    assert detection_wf.build_detection_builder_for_seed.call_args.kwargs["options"].entity_label_examples == examples


def test_run_propagates_excluded_entity_labels(stub_input: AnonymizerInput) -> None:
    config = AnonymizerConfig(detect={"excluded_entity_labels": ["email"]}, replace=Redact())
    anonymizer, detection_wf, _, _ = _make_anonymizer()

    result = anonymizer.run(config=config, data=stub_input)

    assert detection_wf.run.call_args.kwargs["options"].excluded_entity_labels == ["email"]
    assert result.excluded_entity_labels == ["email"]


def test_preview_propagates_excluded_entity_labels(stub_input: AnonymizerInput) -> None:
    config = AnonymizerConfig(detect={"excluded_entity_labels": ["email"]}, replace=Redact())
    anonymizer, detection_wf, _, _ = _make_anonymizer()

    result = anonymizer.preview(config=config, data=stub_input, num_records=1)

    assert detection_wf.run.call_args.kwargs["options"].excluded_entity_labels == ["email"]
    assert result.excluded_entity_labels == ["email"]


def test_resolve_model_providers_raises_on_invalid_yaml(tmp_path: Path) -> None:
    yaml_path = tmp_path / "bad.yaml"
    yaml_path.write_text("not_providers: []")

    with pytest.raises(ValueError, match="providers"):
        _resolve_model_providers(yaml_path)


def test_anonymizer_default_passes_bundled_providers_to_data_designer() -> None:
    bundled = load_default_model_providers()
    with patch("anonymizer.interface.anonymizer.DataDesigner") as mock_data_designer:
        Anonymizer(
            detection_workflow=Mock(),
            replace_runner=Mock(),
            rewrite_runner=Mock(),
        )
    mock_data_designer.assert_called_once()
    passed_providers = mock_data_designer.call_args.kwargs["model_providers"]
    assert {provider.name for provider in passed_providers} == {provider.name for provider in bundled}


def test_anonymizer_custom_model_providers_override_bundled_defaults() -> None:
    from anonymizer import ModelProvider

    # Custom providers replace the full list, so retain the local detector while overriding OpenRouter.
    custom_providers = [
        ModelProvider(name="local-gliner2", endpoint="http://127.0.0.1:8001/v1"),
        ModelProvider(name="openrouter", endpoint="https://example.com/v1"),
    ]
    with patch("anonymizer.interface.anonymizer.DataDesigner") as mock_data_designer:
        Anonymizer(
            model_providers=custom_providers,
            detection_workflow=Mock(),
            replace_runner=Mock(),
            rewrite_runner=Mock(),
        )
    passed_providers = mock_data_designer.call_args.kwargs["model_providers"]
    assert passed_providers is custom_providers


def test_anonymizer_applies_data_designer_run_config_to_managed_instance() -> None:
    run_config = RunConfig(buffer_size=20, max_in_flight_tasks=64)

    with patch("anonymizer.interface.anonymizer.DataDesigner") as mock_data_designer:
        Anonymizer(
            data_designer_run_config=run_config,
            detection_workflow=Mock(),
            replace_runner=Mock(),
            rewrite_runner=Mock(),
        )

    mock_data_designer.return_value.set_run_config.assert_called_once_with(run_config)


def test_anonymizer_applies_data_designer_run_config_to_supplied_instance() -> None:
    from data_designer.interface.data_designer import DataDesigner

    data_designer = Mock(spec=DataDesigner)
    run_config = RunConfig(buffer_size=20, max_in_flight_tasks=64)

    Anonymizer(
        data_designer=data_designer,
        data_designer_run_config=run_config,
        detection_workflow=Mock(),
        replace_runner=Mock(),
        rewrite_runner=Mock(),
    )

    data_designer.set_run_config.assert_called_once_with(run_config)


def test_anonymizer_rejects_missing_provider_as_invalid_config_error() -> None:
    yaml_str = """
model_configs:
  - alias: custom-detector
    model: test/model
"""
    with pytest.raises(InvalidConfigError, match="missing required field 'provider'"):
        Anonymizer(
            model_configs=yaml_str,
            detection_workflow=Mock(),
            replace_runner=Mock(),
            rewrite_runner=Mock(),
        )


def test_anonymizer_rejects_unknown_model_provider_as_invalid_config_error() -> None:
    from anonymizer import ModelProvider

    yaml_str = """
model_configs:
  - alias: custom-detector
    model: test/model
    provider: unknown-provider
"""
    providers = [ModelProvider(name="nvidia", endpoint="https://example.com/v1")]
    with pytest.raises(InvalidConfigError, match="unknown-provider"):
        Anonymizer(
            model_configs=yaml_str,
            model_providers=providers,
            detection_workflow=Mock(),
            replace_runner=Mock(),
            rewrite_runner=Mock(),
        )


def test_anonymizer_rejects_empty_model_providers_list() -> None:
    with pytest.raises(InvalidConfigError, match="at least one provider"):
        Anonymizer(
            model_providers=[],
            detection_workflow=Mock(),
            replace_runner=Mock(),
            rewrite_runner=Mock(),
        )


def test_anonymizer_skips_provider_validation_with_supplied_data_designer() -> None:
    from data_designer.interface.data_designer import DataDesigner

    # Provider not in bundled defaults; would raise if validated against them.
    yaml_str = """
model_configs:
  - alias: custom-detector
    model: test/model
    provider: my-own-provider
"""
    anonymizer = Anonymizer(
        model_configs=yaml_str,
        data_designer=Mock(spec=DataDesigner),
        detection_workflow=Mock(),
        replace_runner=Mock(),
        rewrite_runner=Mock(),
    )
    assert {config.provider for config in anonymizer._model_configs} == {"my-own-provider"}


def test_run_exposes_trace_dataframe_and_filters_internal_columns(
    stub_anonymizer_config: AnonymizerConfig,
    stub_input: AnonymizerInput,
) -> None:
    _replace_df = pd.DataFrame(
        {
            COL_TEXT: ["Alice"],
            COL_REPLACED_TEXT: ["[REDACTED]"],
            COL_TAGGED_TEXT: ["<first_name>Alice</first_name>"],
            COL_DETECTED_ENTITIES: [{"entities": [{"value": "Alice", "label": "first_name"}]}],
            COL_REPLACEMENT_MAP: [{"replacements": []}],
        }
    )
    replace_return = ReplacementResult(dataframe=_replace_df, failed_records=[])
    anonymizer, _, _, _ = _make_anonymizer(replace_return=replace_return)

    result = anonymizer.run(config=stub_anonymizer_config, data=stub_input)

    assert COL_DETECTED_ENTITIES in result.trace_dataframe.columns
    assert COL_REPLACEMENT_MAP in result.trace_dataframe.columns
    assert COL_DETECTED_ENTITIES not in result.dataframe.columns
    assert COL_REPLACEMENT_MAP not in result.dataframe.columns
    assert set(result.dataframe.columns) == {"text", "text_replaced", "text_with_spans"}


def test_preview_exposes_trace_dataframe_for_display(
    stub_anonymizer_config: AnonymizerConfig,
    stub_input: AnonymizerInput,
) -> None:
    _replace_df = pd.DataFrame(
        {
            COL_TEXT: ["Alice"],
            COL_REPLACED_TEXT: ["[REDACTED]"],
            COL_DETECTED_ENTITIES: [{"entities": [{"value": "Alice", "label": "first_name"}]}],
            COL_REPLACEMENT_MAP: [{"replacements": []}],
        }
    )
    replace_return = ReplacementResult(dataframe=_replace_df, failed_records=[])
    anonymizer, _, _, _ = _make_anonymizer(replace_return=replace_return)

    preview = anonymizer.preview(
        config=stub_anonymizer_config,
        data=stub_input,
        num_records=1,
    )

    assert COL_DETECTED_ENTITIES in preview.trace_dataframe.columns
    assert COL_DETECTED_ENTITIES not in preview.dataframe.columns


def test_run_restores_original_text_column_names_for_user_dataframe(
    stub_anonymizer_config: AnonymizerConfig,
    tmp_path: Path,
) -> None:
    replace_df = pd.DataFrame(
        {
            COL_TEXT: ["Alice bio text"],
            COL_REPLACED_TEXT: ["[REDACTED] bio text"],
            COL_TAGGED_TEXT: ["<first_name>Alice</first_name> bio text"],
            COL_DETECTED_ENTITIES: [{"entities": [{"value": "Alice", "label": "first_name"}]}],
        }
    )
    replace_return = ReplacementResult(dataframe=replace_df, failed_records=[])
    anonymizer, _, _, _ = _make_anonymizer(replace_return=replace_return)

    input_csv = tmp_path / "input.csv"
    pd.DataFrame({"bio": ["Alice bio text"]}).to_csv(input_csv, index=False)

    result = anonymizer.run(
        config=stub_anonymizer_config,
        data=AnonymizerInput(source=str(input_csv), text_column="bio"),
    )

    assert "bio" in result.dataframe.columns
    assert "bio_replaced" in result.dataframe.columns
    assert "bio_with_spans" in result.dataframe.columns
    assert "text" not in result.dataframe.columns
    assert "replaced_text" not in result.dataframe.columns
    assert result.dataframe["bio"].iloc[0] == "Alice bio text"
    assert result.dataframe["bio_replaced"].iloc[0] == "[REDACTED] bio text"
    assert result.resolved_text_column == "bio"


def test_run_ignores_workflow_output_attrs_for_text_column_resolution(
    stub_anonymizer_config: AnonymizerConfig,
    tmp_path: Path,
) -> None:
    """Behavioral guarantee: the orchestrator's text-column resolution comes
    from :class:`ResolvedInput`, never from workflow output ``DataFrame.attrs``.

    Construct workflow mocks whose returned dataframes carry deliberately
    misleading ``attrs`` (a key that historical implementations might have
    relied on). If the orchestrator silently consulted ``attrs`` it would
    produce a column named ``WRONG_COL`` instead of ``bio``, and
    ``display_record`` would fail to find the text. We assert the user-facing
    dataframe and the display HTML both reflect the user's actual
    ``text_column`` request.
    """
    detection_df = pd.DataFrame(
        {
            COL_TEXT: ["Alice bio text"],
            COL_FINAL_ENTITIES: [{"entities": [{"value": "Alice", "label": "first_name"}]}],
        }
    )
    detection_df.attrs.update({"resolved_text_column": "WRONG_COL", "requested_text_column": "WRONG_COL"})
    replace_df = pd.DataFrame(
        {
            COL_TEXT: ["Alice bio text"],
            COL_REPLACED_TEXT: ["[REDACTED] bio text"],
            COL_TAGGED_TEXT: ["<first_name>Alice</first_name> bio text"],
            COL_DETECTED_ENTITIES: [
                {"entities": [{"value": "Alice", "label": "first_name", "start_position": 0, "end_position": 5}]}
            ],
            COL_REPLACEMENT_MAP: [
                {"replacements": [{"original": "Alice", "label": "first_name", "synthetic": "[REDACTED]"}]}
            ],
        }
    )
    replace_df.attrs.update({"resolved_text_column": "WRONG_COL", "extra": "noise"})

    anonymizer, _, _, _ = _make_anonymizer(
        detection_return=EntityDetectionResult(dataframe=detection_df, failed_records=[]),
        replace_return=ReplacementResult(dataframe=replace_df, failed_records=[]),
    )

    input_csv = tmp_path / "input.csv"
    pd.DataFrame({"bio": ["Alice bio text"]}).to_csv(input_csv, index=False)
    result = anonymizer.run(
        config=stub_anonymizer_config,
        data=AnonymizerInput(source=str(input_csv), text_column="bio"),
    )

    assert result.resolved_text_column == "bio"
    assert "bio" in result.dataframe.columns
    assert "bio_replaced" in result.dataframe.columns
    assert "WRONG_COL" not in result.dataframe.columns
    assert "WRONG_COL" not in result.trace_dataframe.columns

    from anonymizer.interface.display import render_record_html

    rendered = render_record_html(
        result.trace_dataframe.iloc[0],
        record_index=0,
        resolved_text_column=result.resolved_text_column,
    )
    assert "Alice" in rendered
    assert "bio text" in rendered
    assert "[REDACTED]" in rendered
    assert "WRONG_COL" not in rendered


def test_run_with_colliding_internal_text_column_raises(
    stub_anonymizer_config: AnonymizerConfig,
    tmp_path: Path,
) -> None:
    input_csv = tmp_path / "input.csv"
    pd.DataFrame({COL_TEXT: ["internal"], "bio": ["Alice bio text"]}).to_csv(input_csv, index=False)
    anonymizer, _, _, _ = _make_anonymizer()

    with pytest.raises(InvalidInputError, match="reserved internal column"):
        anonymizer.run(
            config=stub_anonymizer_config,
            data=AnonymizerInput(source=str(input_csv), text_column="bio"),
        )


def test_run_with_colliding_output_column_renames_input_and_warns(
    stub_anonymizer_config: AnonymizerConfig,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Colliding input columns are renamed (not blocked) and a warning is logged.

    We use a custom ``replace_return`` that propagates the renamed input column
    so we can verify that it survives into the trace dataframe alongside the
    fresh pipeline output. This documents the lossless behaviour: the user's
    original ``bio_replaced`` data ends up under ``bio_replaced__input`` while
    the pipeline writes its own ``bio_replaced``.
    """
    input_csv = tmp_path / "input.csv"
    pd.DataFrame({"bio": ["Alice bio text"], "bio_replaced": ["pre-existing"]}).to_csv(input_csv, index=False)

    replace_df = pd.DataFrame(
        {
            COL_TEXT: ["Alice bio text"],
            COL_REPLACED_TEXT: ["[REDACTED] bio text"],
            "bio_replaced__input": ["pre-existing"],
        }
    )
    anonymizer, _, _, _ = _make_anonymizer(
        replace_return=ReplacementResult(dataframe=replace_df, failed_records=[]),
    )

    with caplog.at_level("WARNING", logger="anonymizer"):
        result = anonymizer.run(
            config=stub_anonymizer_config,
            data=AnonymizerInput(source=str(input_csv), text_column="bio"),
        )

    assert any("collide with Anonymizer output column names" in rec.message for rec in caplog.records)
    assert result.dataframe["bio_replaced"].iloc[0] == "[REDACTED] bio text"
    assert result.trace_dataframe["bio_replaced__input"].iloc[0] == "pre-existing"


def test_run_with_text_column_matching_static_output_preserves_both_columns(
    stub_anonymizer_config: AnonymizerConfig,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Setting ``text_column`` to a fixed output name (e.g. ``final_entities``)
    must not produce duplicate columns in the trace/user dataframes.

    The reader renames the input column to ``final_entities__input`` so the
    pipeline's own ``final_entities`` output (the detected entities dict) and
    the user's original text live side-by-side as distinct columns.
    """
    input_csv = tmp_path / "input.csv"
    pd.DataFrame({"final_entities": ["Alice bio text"]}).to_csv(input_csv, index=False)

    entities_payload = {"entities": [{"value": "Alice", "label": "first_name"}]}
    replace_df = pd.DataFrame(
        {
            COL_TEXT: ["Alice bio text"],
            COL_REPLACED_TEXT: ["[REDACTED] bio text"],
            COL_TAGGED_TEXT: ["<first_name>Alice</first_name> bio text"],
            COL_FINAL_ENTITIES: [entities_payload],
        }
    )
    anonymizer, _, _, _ = _make_anonymizer(
        replace_return=ReplacementResult(dataframe=replace_df, failed_records=[]),
    )

    with caplog.at_level("WARNING", logger="anonymizer"):
        result = anonymizer.run(
            config=stub_anonymizer_config,
            data=AnonymizerInput(source=str(input_csv), text_column="final_entities"),
        )

    assert any("collide with Anonymizer output column names" in rec.message for rec in caplog.records)

    for frame_name, frame in (("dataframe", result.dataframe), ("trace_dataframe", result.trace_dataframe)):
        assert list(frame.columns).count("final_entities") == 1, (
            f"{frame_name} has duplicate 'final_entities' columns: {list(frame.columns)}"
        )
        assert "final_entities__input" in frame.columns, f"{frame_name} missing renamed user text column"
        assert frame["final_entities__input"].iloc[0] == "Alice bio text"
        assert frame["final_entities"].iloc[0] == entities_payload


def test_validate_config_passes_for_valid_replace_config(stub_anonymizer_config: AnonymizerConfig) -> None:
    anonymizer, _, _, _ = _make_anonymizer()
    anonymizer.validate_config(stub_anonymizer_config)


def test_validate_config_passes_for_valid_substitute_config() -> None:
    anonymizer, _, _, _ = _make_anonymizer()
    anonymizer.validate_config(AnonymizerConfig(replace=Substitute()))


def test_validate_config_passes_for_valid_rewrite_config() -> None:
    anonymizer, _, _, _ = _make_anonymizer()
    anonymizer.validate_config(AnonymizerConfig(rewrite=Rewrite()))


def test_validate_config_raises_on_unknown_detection_alias(
    stub_anonymizer_config: AnonymizerConfig,
    stub_known_model_configs: list[ModelConfig],
    stub_slim_model_selection: ModelSelection,
) -> None:
    anonymizer, _, _, _ = _make_anonymizer()
    anonymizer._model_configs = stub_known_model_configs
    anonymizer._selected_models = stub_slim_model_selection
    anonymizer._selected_models = anonymizer._selected_models.model_copy(
        update={
            "detection": anonymizer._selected_models.detection.model_copy(
                update={"entity_detector": "bad-detection-alias"}
            )
        }
    )

    with pytest.raises(InvalidConfigError, match="bad-detection-alias"):
        anonymizer.validate_config(stub_anonymizer_config)


def test_validate_config_raises_on_unknown_replace_alias_for_substitute(
    stub_known_model_configs: list[ModelConfig],
    stub_slim_model_selection: ModelSelection,
) -> None:
    anonymizer, _, _, _ = _make_anonymizer()
    anonymizer._model_configs = stub_known_model_configs
    anonymizer._selected_models = stub_slim_model_selection
    anonymizer._selected_models = anonymizer._selected_models.model_copy(
        update={
            "replace": ReplaceModelSelection(
                replacement_generator="bad-replace-alias",
            )
        }
    )

    with pytest.raises(InvalidConfigError, match="bad-replace-alias"):
        anonymizer.validate_config(AnonymizerConfig(replace=Substitute()))


def test_validate_config_skips_replace_alias_for_non_substitute(
    stub_anonymizer_config: AnonymizerConfig,
    stub_known_model_configs: list[ModelConfig],
    stub_slim_model_selection: ModelSelection,
) -> None:
    anonymizer, _, _, _ = _make_anonymizer()
    anonymizer._model_configs = stub_known_model_configs
    anonymizer._selected_models = stub_slim_model_selection
    anonymizer._selected_models = anonymizer._selected_models.model_copy(
        update={
            "replace": ReplaceModelSelection(
                replacement_generator="bad-replace-alias",
            )
        }
    )

    anonymizer.validate_config(stub_anonymizer_config)


def test_validate_config_raises_on_unknown_rewrite_alias(
    stub_known_model_configs: list[ModelConfig],
    stub_slim_model_selection: ModelSelection,
) -> None:
    anonymizer, _, _, _ = _make_anonymizer()
    anonymizer._model_configs = stub_known_model_configs
    anonymizer._selected_models = stub_slim_model_selection
    anonymizer._selected_models = anonymizer._selected_models.model_copy(
        update={"rewrite": anonymizer._selected_models.rewrite.model_copy(update={"rewriter": "bad-rewrite-alias"})}
    )

    with pytest.raises(InvalidConfigError, match="bad-rewrite-alias"):
        anonymizer.validate_config(AnonymizerConfig(rewrite=Rewrite()))


def test_validate_config_skips_latent_detector_without_rewrite(
    stub_anonymizer_config: AnonymizerConfig,
    stub_known_model_configs: list[ModelConfig],
    stub_slim_model_selection: ModelSelection,
) -> None:
    anonymizer, _, _, _ = _make_anonymizer()
    anonymizer._model_configs = stub_known_model_configs
    anonymizer._selected_models = stub_slim_model_selection.model_copy(
        update={"rewrite": stub_slim_model_selection.rewrite.model_copy(update={"rewriter": "bad-rewrite-alias"})}
    )

    anonymizer.validate_config(stub_anonymizer_config)


def test_validate_config_raises_on_unknown_latent_detector_in_rewrite(
    stub_known_model_configs: list[ModelConfig],
    stub_slim_model_selection: ModelSelection,
) -> None:
    anonymizer, _, _, _ = _make_anonymizer()
    anonymizer._model_configs = stub_known_model_configs
    anonymizer._selected_models = stub_slim_model_selection.model_copy(
        update={
            "detection": stub_slim_model_selection.detection.model_copy(update={"latent_detector": "bad-latent-alias"})
        }
    )

    with pytest.raises(InvalidConfigError, match="bad-latent-alias"):
        anonymizer.validate_config(AnonymizerConfig(rewrite=Rewrite()))


def test_validate_config_skips_rewrite_alias_without_rewrite(
    stub_anonymizer_config: AnonymizerConfig,
    stub_known_model_configs: list[ModelConfig],
    stub_slim_model_selection: ModelSelection,
) -> None:
    anonymizer, _, _, _ = _make_anonymizer()
    anonymizer._model_configs = stub_known_model_configs
    anonymizer._selected_models = stub_slim_model_selection
    anonymizer._selected_models = anonymizer._selected_models.model_copy(
        update={"rewrite": anonymizer._selected_models.rewrite.model_copy(update={"rewriter": "bad-rewrite-alias"})}
    )

    anonymizer.validate_config(stub_anonymizer_config)


def test_run_raises_invalid_config_before_workflows(
    stub_input: AnonymizerInput,
    stub_known_model_configs: list[ModelConfig],
    stub_slim_model_selection: ModelSelection,
) -> None:
    anonymizer, detection_wf, replace_runner, _ = _make_anonymizer()
    anonymizer._model_configs = stub_known_model_configs
    anonymizer._selected_models = stub_slim_model_selection
    anonymizer._selected_models = anonymizer._selected_models.model_copy(
        update={
            "replace": ReplaceModelSelection(
                replacement_generator="bad-replace-alias",
            )
        }
    )

    with pytest.raises(InvalidConfigError, match="bad-replace-alias"):
        anonymizer.run(config=AnonymizerConfig(replace=Substitute()), data=stub_input)

    detection_wf.run.assert_not_called()
    replace_runner.run_with_detection.assert_not_called()


def test_preview_raises_invalid_config_before_workflows(
    stub_input: AnonymizerInput,
    stub_known_model_configs: list[ModelConfig],
    stub_slim_model_selection: ModelSelection,
) -> None:
    anonymizer, detection_wf, replace_runner, _ = _make_anonymizer()
    anonymizer._model_configs = stub_known_model_configs
    anonymizer._selected_models = stub_slim_model_selection
    anonymizer._selected_models = anonymizer._selected_models.model_copy(
        update={"rewrite": anonymizer._selected_models.rewrite.model_copy(update={"rewriter": "bad-rewrite-alias"})}
    )

    with pytest.raises(InvalidConfigError, match="bad-rewrite-alias"):
        anonymizer.preview(config=AnonymizerConfig(rewrite=Rewrite()), data=stub_input, num_records=1)

    detection_wf.run.assert_not_called()
    replace_runner.run_with_detection.assert_not_called()


# ---------------------------------------------------------------------------
# Rewrite wiring tests
# ---------------------------------------------------------------------------


def test_run_rewrite_calls_rewrite_runner(stub_input: AnonymizerInput) -> None:
    config = AnonymizerConfig(rewrite=Rewrite())
    anonymizer, _, _, rewrite_runner = _make_anonymizer()

    anonymizer.run(config=config, data=stub_input)

    rewrite_runner.run_with_detection.assert_called_once()
    call_kwargs = rewrite_runner.run_with_detection.call_args.kwargs
    assert config.rewrite is not None
    assert call_kwargs["privacy_goal"] == config.rewrite.privacy_goal
    assert call_kwargs["evaluation"] == config.rewrite.evaluation


def test_run_rewrite_uses_combined_runner_when_enabled(stub_input: AnonymizerInput) -> None:
    config = AnonymizerConfig(rewrite=Rewrite(use_combined_graph=True))
    anonymizer, _, _, rewrite_runner = _make_anonymizer()
    combined_runner = Mock(spec=RewriteWorkflow)
    stub_run_with_detection(combined_runner)
    combined_runner.run.return_value = rewrite_runner.run.return_value
    anonymizer._combined_rewrite_runner = combined_runner

    anonymizer.run(config=config, data=stub_input)

    rewrite_runner.run_with_detection.assert_not_called()
    combined_runner.run_with_detection.assert_called_once()


def test_run_rewrite_output_columns(stub_input: AnonymizerInput) -> None:
    config = AnonymizerConfig(rewrite=Rewrite())
    anonymizer, _, _, _ = _make_anonymizer()

    result = anonymizer.run(config=config, data=stub_input)

    assert "text_rewritten" in result.dataframe.columns
    assert "utility_score" in result.dataframe.columns
    assert "leakage_mass" in result.dataframe.columns
    assert "weighted_leakage_rate" in result.dataframe.columns
    assert "any_high_leaked" in result.dataframe.columns
    assert "needs_human_review" in result.dataframe.columns


def test_run_rewrite_internal_columns_only_in_trace(stub_input: AnonymizerInput) -> None:
    rewrite_df = pd.DataFrame(
        {
            COL_TEXT: ["Alice works at Acme"],
            COL_REWRITTEN_TEXT: ["Beth works at Globex"],
            "_domain": ["general"],
            "_sensitivity_disposition": [None],
            "utility_score": [0.85],
            "leakage_mass": [0.3],
            "weighted_leakage_rate": [0.23],
            "any_high_leaked": [False],
            "needs_human_review": [False],
        }
    )
    rewrite_return = RewriteResult(dataframe=rewrite_df, failed_records=[])
    config = AnonymizerConfig(rewrite=Rewrite())
    anonymizer, _, _, _ = _make_anonymizer(rewrite_return=rewrite_return)

    result = anonymizer.run(config=config, data=stub_input)

    assert "_domain" in result.trace_dataframe.columns
    assert "_domain" not in result.dataframe.columns
    assert "_sensitivity_disposition" in result.trace_dataframe.columns
    assert "_sensitivity_disposition" not in result.dataframe.columns


def test_run_rewrite_merges_failed_records(stub_input: AnonymizerInput) -> None:
    detection_failures = [FailedRecord(record_id="r1", step="detection", reason="timeout")]
    rewrite_failures = [FailedRecord(record_id="r2", step="rewrite", reason="parse error")]

    detection_result = EntityDetectionResult(
        dataframe=pd.DataFrame({COL_TEXT: ["Alice"], COL_FINAL_ENTITIES: [{"entities": []}]}),
        failed_records=detection_failures,
    )
    _rewrite_df = pd.DataFrame(
        {
            COL_TEXT: ["Alice"],
            COL_REWRITTEN_TEXT: ["Beth"],
            "utility_score": [0.85],
            "leakage_mass": [0.3],
            "weighted_leakage_rate": [0.23],
            "any_high_leaked": [False],
            "needs_human_review": [False],
        }
    )
    rewrite_return = RewriteResult(dataframe=_rewrite_df, failed_records=rewrite_failures)

    config = AnonymizerConfig(rewrite=Rewrite())
    anonymizer, _, _, _ = _make_anonymizer(detection_return=detection_result, rewrite_return=rewrite_return)
    result = anonymizer.run(config=config, data=stub_input)

    assert len(result.failed_records) == 2
    assert result.failed_records[0].step == "detection"
    assert result.failed_records[1].step == "rewrite"


def test_validate_config_raises_on_unknown_replace_alias_in_rewrite_mode(
    stub_known_model_configs: list[ModelConfig],
    stub_slim_model_selection: ModelSelection,
) -> None:
    """Rewrite mode uses replace aliases for the replacement map; validate them up front."""
    anonymizer, _, _, _ = _make_anonymizer()
    anonymizer._model_configs = stub_known_model_configs
    anonymizer._selected_models = stub_slim_model_selection
    anonymizer._selected_models = anonymizer._selected_models.model_copy(
        update={
            "replace": ReplaceModelSelection(
                replacement_generator="bad-replace-alias",
            )
        }
    )

    with pytest.raises(InvalidConfigError, match="bad-replace-alias"):
        anonymizer.validate_config(AnonymizerConfig(rewrite=Rewrite()))


def test_evaluate_raises_value_error_on_legacy_result_without_replace_method() -> None:
    """A pickled result from before `replace_method` existed should surface the
    actionable ValueError, not an AttributeError from the missing attribute."""
    anonymizer, _, _, _ = _make_anonymizer()
    legacy_result = SimpleNamespace(
        dataframe=pd.DataFrame(),
        trace_dataframe=pd.DataFrame(),
        resolved_text_column="text",
    )

    with pytest.raises(ValueError, match="replace_method"):
        anonymizer.evaluate(legacy_result)  # ty: ignore[invalid-argument-type]


# ---------------------------------------------------------------------------
# Tests: Anonymizer.evaluate() for rewrite results
# ---------------------------------------------------------------------------


def test_run_rewrite_does_not_include_judge_in_user_dataframe(stub_input: AnonymizerInput) -> None:
    """run() output must not include COL_JUDGE_EVALUATION — it only appears after evaluate()."""
    config = AnonymizerConfig(rewrite=Rewrite())
    anonymizer, _, _, _ = _make_anonymizer()

    result = anonymizer.run(config=config, data=stub_input)

    assert COL_JUDGE_EVALUATION not in result.dataframe.columns


def test_evaluate_rewrite_result_adds_judge_columns(stub_input: AnonymizerInput) -> None:
    """anonymizer.evaluate() on a rewrite result must add COL_JUDGE_EVALUATION."""
    config = AnonymizerConfig(
        detect={
            "entity_labels": ["vendor_api_key"],
            "entity_label_examples": {"vendor_api_key": ["acme_live_abc123"]},
        },
        rewrite=Rewrite(),
    )
    anonymizer, _, _, rewrite_runner = _make_anonymizer()

    run_result = anonymizer.run(config=config, data=stub_input)

    eval_df = pd.DataFrame(
        {
            COL_TEXT: ["Alice works at Acme"],
            COL_REWRITTEN_TEXT: ["Beth works at Globex"],
            "utility_score": [0.85],
            "leakage_mass": [0.3],
            "weighted_leakage_rate": [0.23],
            "any_high_leaked": [False],
            "needs_human_review": [False],
            COL_JUDGE_EVALUATION: [{"privacy": {"score": "high"}}],
            COL_DETECTION_VALID: [1.0],
        }
    )
    rewrite_runner.evaluate.return_value = RewriteResult(dataframe=eval_df, failed_records=[])

    evaluated = anonymizer.evaluate(run_result)

    assert COL_JUDGE_EVALUATION in evaluated.dataframe.columns
    assert "entity_label_examples" not in rewrite_runner.evaluate.call_args.kwargs
    assert "acme_live_abc123" not in (evaluated.dataframe.to_json() or "")


def test_evaluate_rewrite_result_adds_detection_valid(stub_input: AnonymizerInput) -> None:
    """anonymizer.evaluate() on a rewrite result must add COL_DETECTION_VALID."""
    config = AnonymizerConfig(rewrite=Rewrite())
    anonymizer, _, _, rewrite_runner = _make_anonymizer()

    run_result = anonymizer.run(config=config, data=stub_input)

    eval_df = pd.DataFrame(
        {
            COL_TEXT: ["Alice works at Acme"],
            COL_REWRITTEN_TEXT: ["Beth works at Globex"],
            "utility_score": [0.85],
            "leakage_mass": [0.3],
            "weighted_leakage_rate": [0.23],
            "any_high_leaked": [False],
            "needs_human_review": [False],
            COL_JUDGE_EVALUATION: [None],
            COL_DETECTION_VALID: [0.9],
            COL_ENTITIES_BY_VALUE: [{}],
        }
    )
    rewrite_runner.evaluate.return_value = RewriteResult(dataframe=eval_df, failed_records=[])

    evaluated = anonymizer.evaluate(run_result, config=EvaluateConfig(compute_detection_validity=True))

    assert COL_DETECTION_VALID in evaluated.dataframe.columns


def test_evaluate_config_detection_validity_defaults_off() -> None:
    """Goal 2: detection-validity scoring is opt-in — the config default must be False
    so a plain ``EvaluateConfig()`` never triggers the extra detection judge."""
    assert EvaluateConfig().compute_detection_validity is False


def test_evaluate_defaults_skip_detection_validity_in_runner_call(stub_input: AnonymizerInput) -> None:
    """With the default ``EvaluateConfig()``, evaluate() must forward
    ``compute_detection_validity=False`` to the rewrite runner (the judge is not run)."""
    config = AnonymizerConfig(rewrite=Rewrite())
    anonymizer, _, _, rewrite_runner = _make_anonymizer()

    run_result = anonymizer.run(config=config, data=stub_input)

    eval_df = pd.DataFrame(
        {
            COL_TEXT: ["Alice works at Acme"],
            COL_REWRITTEN_TEXT: ["Beth works at Globex"],
            "utility_score": [0.85],
            "leakage_mass": [0.3],
            "weighted_leakage_rate": [0.23],
            "any_high_leaked": [False],
            "needs_human_review": [False],
            COL_JUDGE_EVALUATION: [None],
            COL_ENTITIES_BY_VALUE: [{}],
        }
    )
    rewrite_runner.evaluate.return_value = RewriteResult(dataframe=eval_df, failed_records=[])

    anonymizer.evaluate(run_result, config=EvaluateConfig())

    assert rewrite_runner.evaluate.call_args.kwargs["compute_detection_validity"] is False


def test_evaluate_rewrite_raises_without_rewrite_config() -> None:
    """evaluate() must raise ValueError when result has no rewrite_config and no replace_method."""
    anonymizer, _, _, _ = _make_anonymizer()
    bare_result = SimpleNamespace(
        dataframe=pd.DataFrame(),
        trace_dataframe=pd.DataFrame(),
        resolved_text_column="text",
        rewrite_config=None,
    )

    with pytest.raises(ValueError):
        anonymizer.evaluate(bare_result)  # ty: ignore[invalid-argument-type]


def test_evaluate_rewrite_raises_on_bad_rewrite_judge_alias(
    stub_input: AnonymizerInput,
    stub_known_model_configs: list[ModelConfig],
    stub_slim_model_selection: ModelSelection,
) -> None:
    """evaluate() on a rewrite result must raise InvalidConfigError for an unknown rewrite_judge alias.

    This is the integration-level counterpart to the unit tests on
    validate_model_alias_references — it guards the path from Anonymizer.evaluate()
    through to the validator so that a misconfigured evaluate.rewrite_judge is caught
    before any LLM call is made.
    """
    config = AnonymizerConfig(rewrite=Rewrite())
    anonymizer, _, _, _ = _make_anonymizer()
    anonymizer._model_configs = stub_known_model_configs
    anonymizer._selected_models = stub_slim_model_selection.model_copy(
        update={
            "evaluate": stub_slim_model_selection.evaluate.model_copy(
                update={"rewrite_judge": "bad-rewrite-judge-alias"}
            )
        }
    )

    run_result = anonymizer.run(config=config, data=stub_input)

    with pytest.raises(InvalidConfigError, match="bad-rewrite-judge-alias"):
        anonymizer.evaluate(run_result)


def test_evaluate_rewrite_calls_validate_with_check_rewrite_false(stub_input: AnonymizerInput) -> None:
    """evaluate() on a rewrite result must NOT validate rewrite pipeline model aliases.

    Passing check_rewrite=True would require domain-classifier / rewrite-generator
    aliases that are irrelevant for post-hoc evaluation. This test asserts the call
    uses check_rewrite=False so users with evaluate-only configs are not blocked.
    """
    from unittest.mock import patch as _patch

    config = AnonymizerConfig(rewrite=Rewrite())
    anonymizer, _, _, rewrite_runner = _make_anonymizer()

    run_result = anonymizer.run(config=config, data=stub_input)

    eval_df = pd.DataFrame(
        {
            COL_TEXT: ["Alice works at Acme"],
            COL_REWRITTEN_TEXT: ["Beth works at Globex"],
            "utility_score": [0.85],
            "leakage_mass": [0.3],
            "weighted_leakage_rate": [0.23],
            "any_high_leaked": [False],
            "needs_human_review": [False],
            COL_JUDGE_EVALUATION: [None],
            COL_DETECTION_VALID: [1.0],
        }
    )
    rewrite_runner.evaluate.return_value = RewriteResult(dataframe=eval_df, failed_records=[])

    with _patch(
        "anonymizer.interface.anonymizer.validate_model_alias_references",
        wraps=validate_model_alias_references,
    ) as mock_validate:
        anonymizer.evaluate(run_result)

    rewrite_eval_calls = [call for call in mock_validate.call_args_list if call.kwargs.get("check_evaluate") is True]
    assert rewrite_eval_calls, "validate_model_alias_references was not called with check_evaluate=True"
    for call in rewrite_eval_calls:
        assert call.kwargs.get("check_rewrite") is False, (
            "evaluate() on a rewrite result must pass check_rewrite=False to avoid "
            "requiring rewrite pipeline model aliases that are unused during evaluation"
        )


def test_run_and_preview_persist_data_summary(stub_input: AnonymizerInput) -> None:
    """run()/preview() must preserve input context for later evaluation."""
    data = stub_input.model_copy(update={"data_summary": "Customer support transcripts."})
    config = AnonymizerConfig(
        detect={"regex_rules": [RegexRule(label="ticket", pattern=r"TICKET-\d+")]},
        rewrite=Rewrite(),
    )
    anonymizer, _, _, _ = _make_anonymizer()

    result = anonymizer.run(config=config, data=data)
    preview = anonymizer.preview(config=config, data=data, num_records=1)

    assert result.data_summary == "Customer support transcripts."
    assert preview.data_summary == "Customer support transcripts."
    assert result.entity_labels is preview.entity_labels is None
    assert result.effective_entity_labels == preview.effective_entity_labels
    assert result.effective_entity_labels is not None
    assert "ticket" in result.effective_entity_labels


def test_effective_entity_labels_match_detection_scope_on_run_and_preview(stub_input: AnonymizerInput) -> None:
    config = AnonymizerConfig(
        detect={
            "excluded_entity_labels": ["email"],
            "regex_rules": [RegexRule(label="ticket", pattern=r"TICKET-\d+")],
        },
        replace=Redact(),
    )
    anonymizer, _, _, _ = _make_anonymizer()

    result = anonymizer.run(config=config, data=stub_input)
    preview = anonymizer.preview(config=config, data=stub_input, num_records=1)

    for output in (result, preview):
        assert output.detection_scope is not None
        assert output.effective_entity_labels is not None
        assert set(output.effective_entity_labels) == set(output.detection_scope.effective_labels)
        assert "ticket" in output.effective_entity_labels
        assert "email" not in output.effective_entity_labels


def test_run_detects_custom_regex_rule_entity_through_the_facade(tmp_path: Path) -> None:
    """A custom rule reaches the detection run: it needs no model, so every model request is refused."""
    rule = RegexRule(
        label="ticket", pattern=r"TKT-[0-9]+", validate_matches_with_llm=False, detect_additional_matches=False
    )
    config = AnonymizerConfig(
        detect=Detect(entity_labels=["ticket"], builtin_regexes=False, regex_rules=[rule]), replace=Redact()
    )
    data = TextRecordsInput(records=[TextRecord(id="a", text="Escalate TKT-4821 today.")])
    models = load_models_config()["model_configs"]
    models_yaml = tmp_path / "models.yaml"
    models_yaml.write_text(yaml.safe_dump({"model_configs": [{**m, "skip_health_check": True} for m in models]}))
    anonymizer = Anonymizer(model_configs=models_yaml, artifact_path=tmp_path / "artifacts")

    with (
        patch("httpx.Client.send", side_effect=AssertionError("unexpected synchronous model request")),
        patch("httpx.AsyncClient.send", side_effect=AssertionError("unexpected asynchronous model request")),
    ):
        result = anonymizer.run(config=config, data=data)

    assert result.failed_records == []
    assert [(e["value"], e["label"]) for e in result.dataframe["final_entities"].iloc[0]["entities"]] == [
        ("TKT-4821", "ticket")
    ]
    assert result.dataframe["text_replaced"].tolist() == ["Escalate [REDACTED_TICKET] today."]


def test_evaluate_passes_detection_context_to_coverage_judge(stub_input: AnonymizerInput) -> None:
    """evaluate() must forward persisted detection context to EntityCoverageWorkflow."""
    data = stub_input.model_copy(update={"data_summary": "Customer support transcripts."})
    config = AnonymizerConfig(detect={"excluded_entity_labels": ["email"]}, rewrite=Rewrite())
    anonymizer, _, _, rewrite_runner = _make_anonymizer()
    run_result = anonymizer.run(config=config, data=data)

    eval_df = pd.DataFrame(
        {
            COL_TEXT: ["Alice works at Acme"],
            COL_REWRITTEN_TEXT: ["Beth works at Globex"],
            "utility_score": [0.85],
            "leakage_mass": [0.3],
            "weighted_leakage_rate": [0.23],
            "any_high_leaked": [False],
            "needs_human_review": [False],
            COL_JUDGE_EVALUATION: [None],
            COL_DETECTION_VALID: [1.0],
        }
    )
    rewrite_runner.evaluate.return_value = RewriteResult(dataframe=eval_df, failed_records=[])

    with patch("anonymizer.interface.anonymizer.EntityCoverageWorkflow") as mock_coverage_wf:
        mock_coverage_wf.return_value.run_non_critical.return_value = (eval_df, [])
        evaluated = anonymizer.evaluate(run_result)

    assert mock_coverage_wf.call_args.kwargs["data_summary"] == "Customer support transcripts."
    assert mock_coverage_wf.call_args.kwargs["excluded_entity_labels"] == ["email"]
    assert evaluated.data_summary == "Customer support transcripts."
    assert evaluated.excluded_entity_labels == ["email"]
    assert mock_coverage_wf.call_args.kwargs["effective_entity_labels"] == run_result.effective_entity_labels
    assert evaluated.effective_entity_labels == run_result.effective_entity_labels


_MISSED = [{"value": "Smith", "label": "last_name", "reasoning": "kept verbatim"}]


@pytest.mark.parametrize(
    ("initial_flag", "coverage", "missed", "expected_flag"),
    [
        (False, 0.5, _MISSED, True),
        (True, 0.5, _MISSED, True),
        (False, 1.0, [], False),
        (False, None, None, False),
    ],
    ids=["missed-flags", "already-flagged", "no-missed-unchanged", "coverage-unavailable-unchanged"],
)
def test_evaluate_rewrite_flags_rows_with_missed_entities(
    stub_input: AnonymizerInput,
    initial_flag: bool,
    coverage: float | None,
    missed: list[dict[str, str]] | None,
    expected_flag: bool,
) -> None:
    """evaluate() sets needs_human_review when the coverage judge reports missed entities."""
    anonymizer, _, _, rewrite_runner = _make_anonymizer()
    run_result = anonymizer.run(config=AnonymizerConfig(rewrite=Rewrite()), data=stub_input)
    eval_df = pd.DataFrame(
        {
            COL_TEXT: ["Alice Smith works at Acme"],
            COL_REWRITTEN_TEXT: ["Beth Smith works at Globex"],
            COL_NEEDS_HUMAN_REVIEW: pd.array([initial_flag], dtype="boolean"),
        }
    )
    rewrite_runner.evaluate.return_value = RewriteResult(dataframe=eval_df, failed_records=[])
    coverage_df = eval_df.assign(**{COL_ENTITY_COVERAGE: [coverage], COL_MISSED_ENTITIES: [missed]})

    with patch("anonymizer.interface.anonymizer.EntityCoverageWorkflow") as mock_coverage_wf:
        mock_coverage_wf.return_value.run_non_critical.return_value = (coverage_df, [])
        evaluated = anonymizer.evaluate(run_result)

    assert evaluated.dataframe[COL_NEEDS_HUMAN_REVIEW].tolist() == [expected_flag]
    assert str(evaluated.dataframe[COL_NEEDS_HUMAN_REVIEW].dtype) == "boolean"
    assert run_result.dataframe[COL_NEEDS_HUMAN_REVIEW].tolist() == [False] * len(run_result.dataframe)


@pytest.mark.parametrize(
    "detect",
    [
        {"entity_labels": ["first_name", "email"], "detector": {"labels": ["first_name"], "labels_fixed": True}},
        {"detector": {"labels_fixed": True}},
        {"detector": {"offset_unit": "utf16_code_unit"}},
    ],
)
def test_conflicting_detector_declarations_rejected_before_detection(stub_input: AnonymizerInput, detect: dict) -> None:
    anonymizer, detection_wf, _, _ = _make_anonymizer()
    with pytest.raises(InvalidConfigError):
        anonymizer.run(config=AnonymizerConfig(detect=detect, replace=Redact()), data=stub_input)
    detection_wf.run.assert_not_called()


def test_fixed_detector_labels_missing_a_reserved_label_raise_invalid_config_at_preflight(
    stub_input: AnonymizerInput,
) -> None:
    detect = {
        "entity_labels": ["first_name", "nagroup_g1"],
        "detector": {"labels": ["first_name"], "labels_fixed": True},
    }
    anonymizer, detection_wf, _, _ = _make_anonymizer()
    with pytest.raises(InvalidConfigError, match="nagroup_g1"):
        anonymizer.validate_config(AnonymizerConfig(detect=detect, replace=Redact()))
    with pytest.raises(InvalidConfigError, match="nagroup_g1"):
        anonymizer.run(config=AnonymizerConfig(detect=detect, replace=Redact()), data=stub_input)
    detection_wf.run.assert_not_called()


def test_detection_scope_is_immutable_and_hashable(stub_input: AnonymizerInput) -> None:
    anonymizer, _, _, _ = _make_anonymizer()

    scope = anonymizer.preview(config=AnonymizerConfig(replace=Redact()), data=stub_input).detection_scope

    assert scope is not None
    hash(scope)
    assert not hasattr(scope.effective_labels, "append")
    assert not hasattr(scope.effective_labels, "clear")


def test_preview_reports_scope_with_exclusions(stub_input: AnonymizerInput) -> None:
    detect = {
        "entity_labels": ["first_name", "email"],
        "excluded_entity_labels": ["email"],
        "detector": {"labels": ["first_name", "email", "city"], "labels_fixed": True, "offset_unit": "code_point"},
    }
    anonymizer, _, _, _ = _make_anonymizer()

    result = anonymizer.preview(config=AnonymizerConfig(detect=detect, replace=Redact()), data=stub_input)

    assert result.detection_scope is not None
    assert result.detection_scope.effective_labels == ("first_name",)
    assert result.detection_scope.allowed_labels == ("first_name",)
    assert result.detection_scope.excluded_labels == ("email",)
    assert not hasattr(result.detection_scope, "augmentation_may_add_labels")


def test_result_scope_is_the_config_scope(stub_input: AnonymizerInput) -> None:
    sentinel = DetectionScope(
        effective_labels=("zz_extra",), allowed_labels=None, excluded_labels=(), gliner_threshold=0.3
    )
    anonymizer, _, _, _ = _make_anonymizer()

    with patch.object(Detect, "scope", new_callable=PropertyMock, return_value=sentinel):
        scope = anonymizer.preview(config=AnonymizerConfig(replace=Redact()), data=stub_input).detection_scope

    assert scope == sentinel


@pytest.mark.parametrize("method", ["run", "preview"])
def test_in_place_edit_after_call_start_does_not_change_the_call(stub_input: AnonymizerInput, method: str) -> None:
    config = AnonymizerConfig(
        detect=Detect(entity_labels=["first_name", "email"], entity_label_examples={"email": ["a@b.example"]}),
        replace=Redact(),
    )
    scope_before = config.detect.scope
    anonymizer, detection_workflow, _, _ = _make_anonymizer()
    original_side_effect = detection_workflow.run.side_effect
    original_return = detection_workflow.run.return_value
    received: dict[str, Any] = {}

    def edit_mid_call(*args: Any, **kwargs: Any) -> Any:
        received["entity_labels"] = kwargs["options"].entity_labels
        assert config.detect.entity_labels is not None
        config.detect.entity_labels.append("zz_late")
        config.detect.entity_label_examples["first_name"] = ["Zed"]
        if original_side_effect is not None:
            return original_side_effect(*args, **kwargs)
        return original_return

    detection_workflow.run.side_effect = edit_mid_call
    result = getattr(anonymizer, method)(config=config, data=stub_input)

    assert "zz_late" in (config.detect.entity_labels or [])
    assert "zz_late" not in received["entity_labels"]
    assert result.entity_labels is not None and "zz_late" not in result.entity_labels
    assert result.detection_scope == scope_before


@pytest.mark.parametrize("method", ["run", "preview", "export_detection_config", "export_detection_builder_for_seed"])
def test_workflow_cannot_change_the_caller_config(stub_input: AnonymizerInput, method: str, tmp_path: Path) -> None:
    config = AnonymizerConfig(
        detect=Detect(entity_labels=["first_name", "email"], entity_label_examples={"email": ["a@b.example"]}),
        replace=Redact(),
    )
    snapshot = config.model_dump()
    anonymizer, detection_workflow, _, _ = _make_anonymizer()
    target = {
        "run": detection_workflow.run,
        "preview": detection_workflow.run,
        "export_detection_config": detection_workflow.build_detection_config,
        "export_detection_builder_for_seed": detection_workflow.build_detection_builder_for_seed,
    }[method]
    original_side_effect = target.side_effect
    original_return = target.return_value

    def mutate(*args: Any, **kwargs: Any) -> Any:
        kwargs["options"].entity_labels.append("zz_workflow")
        for examples in kwargs["options"].entity_label_examples.values():
            examples.append("zz_workflow")
        if original_side_effect is not None:
            return original_side_effect(*args, **kwargs)
        return original_return

    target.side_effect = mutate
    if method in ("run", "preview"):
        getattr(anonymizer, method)(config=config, data=stub_input)
    elif method == "export_detection_config":
        anonymizer.export_detection_config(config=config, data=stub_input, seed_path=tmp_path / "seed.parquet")
    else:
        anonymizer.export_detection_builder_for_seed(config=config, seed_path=tmp_path / "seed.parquet")

    assert target.call_count == 1
    assert config.model_dump() == snapshot


def test_rewrite_goal_reaching_the_detector_matches_the_caller_goal(stub_input: AnonymizerInput) -> None:
    goal = PrivacyGoal(protect="Names and street addresses", preserve="Clinical meaning of the note")
    config = AnonymizerConfig(rewrite=Rewrite(privacy_goal=goal))
    anonymizer, detection_workflow, _, rewrite_runner = _make_anonymizer()

    anonymizer.run(config=config, data=stub_input)

    received = detection_workflow.run.call_args.kwargs["latent_goal"]
    assert received == goal
    assert rewrite_runner.run_with_detection.call_args.kwargs["privacy_goal"] == goal


class _FirstWordDetector:
    """Detects each row's first word as a ``first_name``; records the row count of every call."""

    def __init__(self) -> None:
        self.calls: list[int] = []

    def run(self, dataframe: pd.DataFrame, **_: object) -> EntityDetectionResult:
        self.calls.append(len(dataframe))
        frame = dataframe.copy()
        names = [str(text).split()[0] for text in frame[COL_TEXT]]
        entities = [
            {"entities": [{"id": "a", "value": n, "label": "first_name", "start_position": 0, "end_position": len(n)}]}
            for n in names
        ]
        frame[COL_FINAL_ENTITIES] = entities
        frame[COL_ENTITIES_BY_VALUE] = [_build_entities_by_value(e) for e in entities]
        return EntityDetectionResult(frame, [], tuple(range(len(frame))))


def _dropped_second(dataframe: pd.DataFrame) -> list[FailedRecord]:
    ids = dataframe[RECORD_ID_COLUMN] if RECORD_ID_COLUMN in dataframe.columns else dataframe.index
    return [FailedRecord(str(ids[1]), "judge", "dropped")]


class _PartialReplaceRunner(ReplacementWorkflow):
    def evaluate(self, dataframe: pd.DataFrame, **_: object) -> ReplacementResult:
        kept = dataframe.iloc[[0]].copy()
        kept[COL_ENTITY_COVERAGE] = 1.0
        return ReplacementResult(dataframe=kept, failed_records=_dropped_second(dataframe))


class _StubRewriteRunner(RewriteWorkflow):
    """Rewrite with the model stages stubbed; ``evaluate`` keeps the first row and fails the second."""

    def _produce(self, rows, **options):
        def produce() -> None:
            rows.frame[COL_REWRITTEN_TEXT] = [f"R<{t}>" for t in rows.frame[COL_TEXT]]
            for column in ("utility_score", "leakage_mass", "weighted_leakage_rate"):
                rows.frame[column] = 1.0 if column == "utility_score" else 0.0
            rows.frame["any_high_leaked"] = False

        return produce

    def evaluate(self, dataframe: pd.DataFrame, **_: object) -> RewriteResult:  # ty: ignore[invalid-method-override]
        kept = dataframe.iloc[[0]].copy()
        kept[COL_JUDGE_EVALUATION] = [{"privacy": 1}]
        return RewriteResult(kept, _dropped_second(dataframe))


def _two_row_anonymizer(tmp_path: Path) -> tuple[Anonymizer, _FirstWordDetector, AnonymizerInput]:
    csv_path = tmp_path / "input.csv"
    pd.DataFrame({"text": ["Alice Smith lives here", "Bob Jones is here"]}).to_csv(csv_path, index=False)
    detector = _FirstWordDetector()
    anonymizer = Anonymizer(
        artifact_path=tmp_path / "artifacts",
        detection_workflow=cast(EntityDetectionWorkflow, detector),
        replace_runner=_PartialReplaceRunner(),
        rewrite_runner=_StubRewriteRunner(Mock()),
    )
    return anonymizer, detector, AnonymizerInput(source=str(csv_path))


def _stub_coverage(_self: object, dataframe: pd.DataFrame, **_: object) -> tuple[pd.DataFrame, list[FailedRecord]]:
    return dataframe.assign(**{COL_ENTITY_COVERAGE: 1.0}), []


_MODES = {"replace": AnonymizerConfig(replace=Redact()), "rewrite": AnonymizerConfig(rewrite=Rewrite())}


@pytest.mark.parametrize("mode", list(_MODES))
def test_evaluate_keeps_partial_result_from_custom_runner(tmp_path: Path, mode: str) -> None:
    anonymizer, _, data = _two_row_anonymizer(tmp_path)
    run_result = anonymizer.run(config=_MODES[mode], data=data)

    with patch.object(EntityCoverageWorkflow, "run_non_critical", _stub_coverage):
        evaluated = anonymizer.evaluate(run_result)

    assert len(evaluated.dataframe) == 1
    assert [(r.step, r.reason) for r in evaluated.failed_records] == [("judge", "dropped")]


@pytest.mark.parametrize("mode", list(_MODES))
def test_run_calls_the_detector_on_empty_input(tmp_path: Path, mode: str) -> None:
    anonymizer, detector, data = _two_row_anonymizer(tmp_path)
    pd.DataFrame({"text": []}).to_csv(tmp_path / "input.csv", index=False)

    result = anonymizer.run(config=_MODES[mode], data=data)

    assert result.dataframe.empty
    assert detector.calls == [0]


def _cancelled_token() -> CancelToken:
    token = CancelToken()
    token.cancel()
    return token


_NESTED_MODES = {**_MODES, "rewrite-combined": AnonymizerConfig(rewrite=Rewrite(use_combined_graph=True))}


@pytest.mark.parametrize("mode", list(_NESTED_MODES))
def test_nested_run_in_cancelled_frame_raises_without_detecting(tmp_path: Path, mode: str) -> None:
    anonymizer, detector, data = _two_row_anonymizer(tmp_path)

    with pytest.raises(AnonymizerWorkflowError, match="cancelled"):
        run_in_frame(anonymizer.run, config=_NESTED_MODES[mode], data=data, token=_cancelled_token())

    assert detector.calls == []


def test_nested_evaluate_in_cancelled_frame_raises_without_judging(tmp_path: Path) -> None:
    anonymizer, _, data = _two_row_anonymizer(tmp_path)
    run_result = anonymizer.run(config=_MODES["replace"], data=data)

    with patch.object(_PartialReplaceRunner, "evaluate") as judge:
        with pytest.raises(AnonymizerWorkflowError, match="cancelled"):
            run_in_frame(anonymizer.evaluate, run_result, token=_cancelled_token())

    judge.assert_not_called()


@pytest.mark.parametrize("method", [Redact(), Substitute()], ids=["local", "substitute"])
def test_replace_route_in_cancelled_frame_raises(method: Redact | Substitute) -> None:
    entities = [
        {"entities": [{"id": "a", "value": n, "label": "first_name", "start_position": 0, "end_position": len(n)}]}
        for n in ("Alice", "Bob")
    ]
    frame = pd.DataFrame(
        {
            COL_TEXT: ["Alice Smith", "Bob Jones"],
            COL_FINAL_ENTITIES: entities,
            COL_ENTITIES_BY_VALUE: [_build_entities_by_value(e) for e in entities],
        }
    )
    llm_workflow = Mock(spec=LlmReplaceWorkflow)

    with pytest.raises(AnonymizerWorkflowError, match="cancelled"):
        run_in_frame(
            ReplacementWorkflow(llm_workflow=llm_workflow).run,
            frame,
            replace_method=method,
            model_configs=[],
            selected_models=ReplaceModelSelection(replacement_generator="x"),
            token=_cancelled_token(),
        )

    llm_workflow.generate_map_only.assert_not_called()


_BINDING: contextvars.ContextVar[str] = contextvars.ContextVar("test_facade_binding", default="caller")


class _BindingReplaceRunner(_PartialReplaceRunner):
    """An overridden runner whose run and evaluate set a context value."""

    def run(self, dataframe: pd.DataFrame, **_: object) -> ReplacementResult:
        _BINDING.set("run")
        return ReplacementResult(dataframe=dataframe.assign(**{COL_REPLACED_TEXT: "[REDACTED]"}), failed_records=[])

    def evaluate(self, dataframe: pd.DataFrame, **options: object) -> ReplacementResult:
        _BINDING.set("evaluate")
        return super().evaluate(dataframe, **options)


@pytest.mark.parametrize("entry", ["run", "preview", "evaluate"])
def test_entries_do_not_leak_context_values_set_by_a_runner(tmp_path: Path, entry: str) -> None:
    anonymizer, _, data = _two_row_anonymizer(tmp_path)
    anonymizer._replace_runner = _BindingReplaceRunner()
    config = _MODES["replace"]
    run_result = anonymizer.run(config=config, data=data)
    token = _BINDING.set("caller")
    try:
        if entry == "evaluate":
            with patch.object(EntityCoverageWorkflow, "run_non_critical", _stub_coverage):
                anonymizer.evaluate(run_result)
        elif entry == "preview":
            anonymizer.preview(config=config, data=data, num_records=2)
        else:
            anonymizer.run(config=config, data=data)
        assert _BINDING.get() == "caller"
    finally:
        _BINDING.reset(token)
