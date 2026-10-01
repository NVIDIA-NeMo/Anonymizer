# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate every rewrite-pipeline LLM prompt against DataDesigner's Jinja sandbox rules.

Plain Jinja accepts templates that NDD rejects at run time (nested ``{% for %}`` loops,
``{% set %}``, over-deep or over-large ASTs, unknown references). A rejected template is a
non-retryable failure for every row, so catch it here rather than in a live run.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest
from data_designer.config.column_configs import LLMStructuredColumnConfig, LLMTextColumnConfig
from data_designer.engine.processing.ginja.environment import UserTemplateSandboxEnvironment
from jinja2 import Environment, meta

from anonymizer.config.models import ReplaceModelSelection, RewriteModelSelection
from anonymizer.config.rewrite import EvaluationCriteria, PrivacyGoal
from anonymizer.engine.rewrite.combined_rewrite_workflow import CombinedRewriteWorkflow

_PRIVACY_GOAL = PrivacyGoal(
    protect="Protect direct identifiers and quasi-identifier combinations from re-identification.",
    preserve="General utility and semantic meaning of the original text.",
)


@pytest.mark.parametrize("strict", [False, True])
def test_all_rewrite_pipeline_prompts_pass_ndd_template_validation(
    stub_rewrite_model_selection: RewriteModelSelection,
    stub_replace_model_selection: ReplaceModelSelection,
    strict: bool,
) -> None:
    graph = CombinedRewriteWorkflow(adapter=Mock()).build_graph(
        selected_models=stub_rewrite_model_selection,
        replace_model_selection=stub_replace_model_selection,
        privacy_goal=_PRIVACY_GOAL,
        evaluation=EvaluationCriteria(max_repair_iterations=1),
        data_summary="Biography profiles",
        strict_entity_protection=strict,
    )
    prompts = {
        column.name: column.prompt
        for column in graph.columns
        if isinstance(column, (LLMStructuredColumnConfig, LLMTextColumnConfig))
    }
    assert prompts, "expected LLM columns in the rewrite graph"

    for name, prompt in prompts.items():
        # Allow every variable the template references; this test targets structural rules.
        references = sorted(meta.find_undeclared_variables(Environment().parse(prompt)))
        env = UserTemplateSandboxEnvironment(allowed_references=references)
        try:
            env.validate_template(prompt)
        except Exception as exc:  # noqa: BLE001 - re-raise with the offending column named
            raise AssertionError(f"Prompt for column {name!r} fails NDD template validation: {exc}") from exc
