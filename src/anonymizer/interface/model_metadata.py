# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Save safe task/model assignments for evaluation reports, without credentials."""

from __future__ import annotations

from data_designer.config.models import ModelConfig

from anonymizer.config.anonymizer_config import AnonymizerConfig, EvaluateConfig
from anonymizer.config.models import ModelSelection
from anonymizer.config.replace_strategies import Substitute


def _snapshot(
    tasks: list[tuple[str, str, str | list[str]]], models: list[ModelConfig], *, phase: str
) -> list[dict[str, str]]:
    lookup = {model.alias: model for model in models}
    rows = []
    for group, task, aliases in tasks:
        for alias in dict.fromkeys([aliases] if isinstance(aliases, str) else aliases):
            model = lookup.get(alias)
            rows.append(
                {
                    "phase": phase,
                    "group": group,
                    "task": task,
                    "alias": alias,
                    "model": model.model if model else "Unknown",
                    "provider": model.provider if model else "Unknown",
                }
            )
    return rows


def run_model_metadata(
    config: AnonymizerConfig, selection: ModelSelection, models: list[ModelConfig]
) -> list[dict[str, str]]:
    """Snapshot applicable run-time task assignments, including conditional roles."""
    detection = selection.detection
    tasks: list[tuple[str, str, str | list[str]]] = [
        ("Anonymization", "Entity detection", detection.entity_detector),
        ("Anonymization", "Entity validation", detection.entity_validator),
        ("Anonymization", "Entity augmentation", detection.entity_augmenter),
    ]
    if config.rewrite is not None or isinstance(config.replace, Substitute):
        tasks.append(("Anonymization", "Replacement generation", selection.replace.replacement_generator))
    if config.rewrite is not None:
        rewrite = selection.rewrite
        tasks.extend(
            [
                ("Anonymization", "Latent detection", detection.latent_detector),
                ("Anonymization", "Domain classification", rewrite.domain_classifier),
                ("Anonymization", "Entity classification (custom labels)", rewrite.entity_classifier),
                ("Evaluation", "Sensitivity / disposition", rewrite.disposition_analyzer),
                ("Anonymization", "Meaning extraction", rewrite.meaning_extractor),
                ("Anonymization", "QA generation", rewrite.qa_generator),
                ("Anonymization", "Rewriting", rewrite.rewriter),
                ("Evaluation", "Privacy / quality QA evaluation", rewrite.evaluator),
            ]
        )
        if config.rewrite.max_repair_iterations > 0:
            tasks.append(("Anonymization", "Repair (when needed)", rewrite.repairer))
    return _snapshot(tasks, models, phase="run")


def evaluation_model_metadata(
    config: EvaluateConfig, selection: ModelSelection, models: list[ModelConfig], *, rewrite: bool, substitute: bool
) -> list[dict[str, str]]:
    """Snapshot judge roles requested for this evaluation; preserve run roles separately."""
    evaluate = selection.evaluate
    tasks: list[tuple[str, str, str | list[str]]] = [("Evaluation", "Entity coverage", evaluate.entity_coverage_judge)]
    if config.compute_detection_validity:
        tasks.append(("Evaluation", "Detection validity", evaluate.detection_validity_judge))
    if rewrite:
        tasks.append(("Evaluation", "Privacy / quality / style judge", evaluate.rewrite_judge))
    elif substitute:
        tasks.extend(
            [
                ("Evaluation", "Type fidelity", evaluate.replace_type_fidelity_judge),
                ("Evaluation", "Attribute fidelity", evaluate.replace_attribute_fidelity_judge),
                ("Evaluation", "Relational consistency", evaluate.replace_relational_consistency_judge),
            ]
        )
    if not rewrite and config.compute_privacy_leakage:
        tasks.extend(
            [
                (
                    "Evaluation",
                    "Entity classification (custom labels)",
                    evaluate.privacy_entity_classifier or evaluate.privacy_sensitivity_analyzer,
                ),
                ("Evaluation", "Sensitivity assessment", evaluate.privacy_sensitivity_analyzer),
                ("Evaluation", "Privacy QA answering", evaluate.privacy_qa_judge),
            ]
        )
    return _snapshot(tasks, models, phase="evaluate")
