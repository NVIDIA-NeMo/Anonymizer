# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from data_designer.config.models import ModelConfig

from anonymizer.config.models import (
    DetectionModelSelection,
    ModelSelection,
    ReplaceModelSelection,
)
from anonymizer.engine.ndd.model_loader import (
    DEFAULT_CONFIG_DIR,
    WorkflowName,
    get_model_alias,
    load_default_model_selection,
    load_models_config,
    load_workflow_config,
    load_workflow_selections,
    parse_model_configs,
    validate_model_alias_references,
)


def test_load_workflow_config_contains_selected_models() -> None:
    config = load_workflow_config(WorkflowName.detection)
    assert "model_configs" in config
    assert "selected_models" in config
    entity_detector = config["selected_models"]["entity_detector"]
    assert isinstance(entity_detector, str) and entity_detector


def test_get_model_alias_reads_workflow_mapping() -> None:
    alias = get_model_alias(workflow_name=WorkflowName.detection, role="entity_validator")
    assert isinstance(alias, str) and alias


WORKFLOW_YAMLS = [p.stem for p in DEFAULT_CONFIG_DIR.glob("*.yaml") if p.stem != "models"]


@pytest.mark.parametrize("workflow_name", WORKFLOW_YAMLS)
def test_default_workflow_aliases_exist_in_models(workflow_name: str) -> None:
    """Every alias referenced in a workflow YAML must be defined in models.yaml."""
    models_config = load_models_config()
    known_aliases = {m["alias"] for m in models_config.get("model_configs", [])}
    selections = load_workflow_selections(WorkflowName(workflow_name))
    unknown = set(selections.values()) - known_aliases
    assert not unknown, f"Workflow '{workflow_name}' references unknown aliases: {unknown}. Known: {known_aliases}"


def test_load_default_model_selection_populates_all_workflows() -> None:
    selection = load_default_model_selection()
    # Detection
    assert selection.detection.entity_detector
    assert selection.detection.entity_validator  # list[str]
    assert isinstance(selection.detection.entity_validator, list)
    assert all(isinstance(alias, str) and alias for alias in selection.detection.entity_validator)
    assert selection.detection.entity_augmenter
    assert selection.detection.latent_detector
    # Replace
    assert selection.replace.replacement_generator
    # Rewrite — all 8 roles must be populated
    assert selection.rewrite.domain_classifier
    assert selection.rewrite.disposition_analyzer
    assert selection.rewrite.meaning_extractor
    assert selection.rewrite.qa_generator
    assert selection.rewrite.rewriter
    assert selection.rewrite.evaluator
    assert selection.rewrite.repairer
    assert selection.rewrite.judge


def test_parse_model_configs_none_uses_defaults() -> None:
    result = parse_model_configs(None)
    assert len(result.model_configs) > 0
    assert result.selected_models.detection.entity_detector == "gliner-pii-detector"


def test_parse_model_configs_yaml_string_extracts_selections() -> None:
    yaml_str = """
selected_models:
  detection:
    entity_detector: custom-detector
model_configs:
  - alias: custom-detector
    model: test/model
  - alias: gpt-oss-120b
    model: test/gpt
"""
    result = parse_model_configs(yaml_str)
    assert result.selected_models.detection.entity_detector == "custom-detector"
    assert result.selected_models.replace.replacement_generator == "gpt-oss-120b"
    assert len(result.model_configs) == 2


def test_parse_model_configs_yaml_without_selections_uses_defaults() -> None:
    yaml_str = """
model_configs:
  - alias: stub-model
    model: test/stub
"""
    result = parse_model_configs(yaml_str)
    assert len(result.model_configs) == 1
    assert result.selected_models.detection.entity_detector == "gliner-pii-detector"


def test_validate_model_alias_references_accepts_valid_detection_aliases(
    stub_known_model_configs: list[ModelConfig],
    stub_slim_model_selection: ModelSelection,
) -> None:
    validate_model_alias_references(
        stub_known_model_configs,
        stub_slim_model_selection,
    )


def test_validate_model_alias_references_raises_on_unknown_detection_alias(
    stub_known_model_configs: list[ModelConfig],
    stub_slim_model_selection: ModelSelection,
) -> None:
    selected_models = stub_slim_model_selection.model_copy(
        update={
            "detection": DetectionModelSelection(
                entity_detector="bad-detection-alias",
                entity_validator="known",
                entity_augmenter="known",
                latent_detector="known",
            )
        }
    )

    with pytest.raises(ValueError, match="bad-detection-alias"):
        validate_model_alias_references(
            stub_known_model_configs,
            selected_models,
        )


def test_validate_model_alias_references_skips_latent_detector_when_not_rewrite(
    stub_known_model_configs: list[ModelConfig],
    stub_slim_model_selection: ModelSelection,
) -> None:
    selected_models = stub_slim_model_selection.model_copy(
        update={
            "detection": stub_slim_model_selection.detection.model_copy(update={"latent_detector": "bad-latent-alias"})
        }
    )

    validate_model_alias_references(
        stub_known_model_configs,
        selected_models,
    )


def test_validate_model_alias_references_raises_on_unknown_replace_alias_when_enabled(
    stub_known_model_configs: list[ModelConfig],
    stub_slim_model_selection: ModelSelection,
) -> None:
    selected_models = stub_slim_model_selection.model_copy(
        update={"replace": ReplaceModelSelection(replacement_generator="bad-replace-alias")}
    )

    with pytest.raises(ValueError, match="bad-replace-alias"):
        validate_model_alias_references(
            stub_known_model_configs,
            selected_models,
            check_substitute=True,
        )


def test_validate_model_alias_references_skips_replace_alias_when_not_enabled(
    stub_known_model_configs: list[ModelConfig],
    stub_slim_model_selection: ModelSelection,
) -> None:
    selected_models = stub_slim_model_selection.model_copy(
        update={"replace": ReplaceModelSelection(replacement_generator="bad-replace-alias")}
    )

    validate_model_alias_references(
        stub_known_model_configs,
        selected_models,
    )


def test_validate_model_alias_references_raises_on_unknown_rewrite_alias_when_enabled(
    stub_known_model_configs: list[ModelConfig],
    stub_slim_model_selection: ModelSelection,
) -> None:
    selected_models = stub_slim_model_selection.model_copy(
        update={"rewrite": stub_slim_model_selection.rewrite.model_copy(update={"rewriter": "bad-rewrite-alias"})}
    )

    with pytest.raises(ValueError, match="bad-rewrite-alias"):
        validate_model_alias_references(
            stub_known_model_configs,
            selected_models,
            check_rewrite=True,
        )


def test_validate_model_alias_references_raises_on_unknown_latent_detector_when_rewrite_enabled(
    stub_known_model_configs: list[ModelConfig],
    stub_slim_model_selection: ModelSelection,
) -> None:
    selected_models = stub_slim_model_selection.model_copy(
        update={
            "detection": stub_slim_model_selection.detection.model_copy(update={"latent_detector": "bad-latent-alias"})
        }
    )

    with pytest.raises(ValueError, match="bad-latent-alias"):
        validate_model_alias_references(
            stub_known_model_configs,
            selected_models,
            check_rewrite=True,
        )


def test_validate_model_alias_references_skips_rewrite_alias_when_not_enabled(
    stub_known_model_configs: list[ModelConfig],
    stub_slim_model_selection: ModelSelection,
) -> None:
    selected_models = stub_slim_model_selection.model_copy(
        update={"rewrite": stub_slim_model_selection.rewrite.model_copy(update={"rewriter": "bad-rewrite-alias"})}
    )

    validate_model_alias_references(
        stub_known_model_configs,
        selected_models,
    )


class TestEntityValidatorNormalization:
    """``DetectionModelSelection.entity_validator`` accepts scalar or list input.

    Scalars normalize to single-item lists so every downstream consumer
    sees ``list[str]``.
    """

    def test_scalar_normalizes_to_single_item_list(self) -> None:
        selection = DetectionModelSelection(
            entity_detector="d",
            entity_validator="v",
            entity_augmenter="a",
            latent_detector="l",
        )
        assert selection.entity_validator == ["v"]

    def test_list_preserved(self) -> None:
        selection = DetectionModelSelection(
            entity_detector="d",
            entity_validator=["v1", "v2", "v3"],
            entity_augmenter="a",
            latent_detector="l",
        )
        assert selection.entity_validator == ["v1", "v2", "v3"]

    def test_empty_list_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one model alias"):
            DetectionModelSelection(
                entity_detector="d",
                entity_validator=[],
                entity_augmenter="a",
                latent_detector="l",
            )

    def test_whitespace_only_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one model alias"):
            DetectionModelSelection(
                entity_detector="d",
                entity_validator=["  ", ""],
                entity_augmenter="a",
                latent_detector="l",
            )

    def test_non_string_non_list_rejected(self) -> None:
        with pytest.raises((ValueError, TypeError)):
            DetectionModelSelection(
                entity_detector="d",
                entity_validator=42,  # type: ignore[arg-type]
                entity_augmenter="a",
                latent_detector="l",
            )


class TestValidateAliasReferencesHandlesValidatorPool:
    """``validate_model_alias_references`` must expand list-valued roles to one check per alias."""

    def test_accepts_all_pool_aliases_present(
        self,
        stub_slim_model_selection: ModelSelection,
    ) -> None:
        """Pool of aliases all present in the model pool — passes."""
        configs = [
            ModelConfig(alias="v1", model="test/v1"),
            ModelConfig(alias="v2", model="test/v2"),
            ModelConfig(alias="known", model="some/model"),
        ]
        selected_models = stub_slim_model_selection.model_copy(
            update={
                "detection": stub_slim_model_selection.detection.model_copy(update={"entity_validator": ["v1", "v2"]})
            }
        )
        validate_model_alias_references(configs, selected_models)

    def test_raises_on_any_pool_alias_missing(
        self,
        stub_known_model_configs: list[ModelConfig],
        stub_slim_model_selection: ModelSelection,
    ) -> None:
        """If any alias in the validator pool is unknown, error names that alias by index."""
        selected_models = stub_slim_model_selection.model_copy(
            update={
                "detection": stub_slim_model_selection.detection.model_copy(
                    update={"entity_validator": ["known", "missing-one"]}
                )
            }
        )
        with pytest.raises(ValueError, match=r"entity_validator\[1\].*missing-one"):
            validate_model_alias_references(stub_known_model_configs, selected_models)
