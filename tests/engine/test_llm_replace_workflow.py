# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import random
from contextlib import nullcontext
from typing import Literal
from unittest.mock import Mock

import pandas as pd
import pytest
from data_designer.config.models import ModelConfig

from anonymizer.config.models import ReplaceModelSelection
from anonymizer.engine.constants import COL_ENTITIES_BY_VALUE, COL_REPLACEMENT_MAP, COL_TEXT
from anonymizer.engine.ndd.adapter import WorkflowRunResult
from anonymizer.engine.replace.llm_replace_workflow import (
    _INTERNAL_COLUMNS,
    LlmReplaceWorkflow,
    _filter_replacement_map_to_input_entities,
    _fold,
    _fold_replacement_maps,
)
from anonymizer.engine.schemas.detection import EntitiesByValueSchema
from anonymizer.measurement import measurement_session


def _positional(dataframe: pd.DataFrame, failed_records: list) -> WorkflowRunResult:
    """An adapter result that reports one source position per returned row, in input order."""
    return WorkflowRunResult(
        dataframe=dataframe, failed_records=failed_records, source_positions=tuple(range(len(dataframe)))
    )


def test_generate_map_only_returns_replacement_map(
    stub_model_configs: list[ModelConfig],
    stub_replace_model_selection: ReplaceModelSelection,
) -> None:
    """Workflow operates on plain DataFrames; pipeline metadata is the orchestrator's job."""
    adapter = Mock()
    adapter.run_workflow.return_value = WorkflowRunResult(
        dataframe=pd.DataFrame(
            {
                "text": ["Alice works at Acme"],
                COL_REPLACEMENT_MAP: [{"replacements": []}],
                "_anonymizer_row_order": [0],
            }
        ),
        failed_records=[],
    )
    workflow = LlmReplaceWorkflow(adapter=adapter)

    input_df = pd.DataFrame(
        {
            "text": ["Alice works at Acme"],
            "_entities_by_value": [[{"value": "Alice", "labels": ["first_name"]}]],
            "tagged_text": ["<<SENSITIVE:first_name>>Alice<</SENSITIVE:first_name>> works at Acme"],
        }
    )

    result = workflow.generate_map_only(
        input_df,
        model_configs=stub_model_configs,
        selected_models=stub_replace_model_selection,
    )

    assert COL_REPLACEMENT_MAP in result.dataframe.columns


def test_generate_map_only_skips_llm_for_rows_without_entities(
    stub_model_configs: list[ModelConfig],
    stub_replace_model_selection: ReplaceModelSelection,
) -> None:
    adapter = Mock()
    workflow = LlmReplaceWorkflow(adapter=adapter)

    input_df = pd.DataFrame(
        {
            COL_TEXT: ["CT guided biopsy revealed a fatty mass which was diagnosed as lipoma."],
            COL_ENTITIES_BY_VALUE: [{"entities_by_value": []}],
            "tagged_text": ["CT guided biopsy revealed a fatty mass which was diagnosed as lipoma."],
        }
    )

    result = workflow.generate_map_only(
        input_df,
        model_configs=stub_model_configs,
        selected_models=stub_replace_model_selection,
    )

    adapter.run_workflow.assert_not_called()
    assert result.dataframe[COL_REPLACEMENT_MAP].iloc[0] == {"replacements": []}


def test_generate_map_only_filters_hallucinated_replacements(
    stub_model_configs: list[ModelConfig],
    stub_replace_model_selection: ReplaceModelSelection,
) -> None:
    adapter = Mock()
    adapter.run_workflow.return_value = _positional(
        dataframe=pd.DataFrame(
            {
                COL_TEXT: ["Alice works at Acme"],
                COL_ENTITIES_BY_VALUE: [{"entities_by_value": [{"value": "Alice", "labels": ["first_name"]}]}],
                "tagged_text": ["<<PII:first_name>>Alice<</PII:first_name>> works at Acme"],
                "_anonymizer_row_order": [0],
                COL_REPLACEMENT_MAP: [
                    {
                        "replacements": [
                            {"original": "Alice", "label": "first_name", "synthetic": "Maya"},
                            {"original": "Acme", "label": "organization", "synthetic": "NovaCorp"},
                        ]
                    }
                ],
            }
        ),
        failed_records=[],
    )
    workflow = LlmReplaceWorkflow(adapter=adapter)

    input_df = pd.DataFrame(
        {
            COL_TEXT: ["Alice works at Acme"],
            COL_ENTITIES_BY_VALUE: [{"entities_by_value": [{"value": "Alice", "labels": ["first_name"]}]}],
            "tagged_text": ["<<PII:first_name>>Alice<</PII:first_name>> works at Acme"],
        }
    )

    result = workflow.generate_map_only(
        input_df,
        model_configs=stub_model_configs,
        selected_models=stub_replace_model_selection,
    )

    assert result.dataframe[COL_REPLACEMENT_MAP].iloc[0] == {
        "replacements": [{"original": "Alice", "label": "first_name", "synthetic": "Maya"}]
    }


def test_generate_map_only_preserves_original_anonymizer_row_order_with_mixed_rows(
    stub_model_configs: list[ModelConfig],
    stub_replace_model_selection: ReplaceModelSelection,
) -> None:
    adapter = Mock()
    adapter.run_workflow.return_value = _positional(
        dataframe=pd.DataFrame(
            {
                COL_TEXT: ["Alice works at Acme"],
                COL_ENTITIES_BY_VALUE: [{"entities_by_value": [{"value": "Alice", "labels": ["first_name"]}]}],
                "tagged_text": ["<<PII:first_name>>Alice<</PII:first_name>> works at Acme"],
                "_anonymizer_row_order": [1],
                COL_REPLACEMENT_MAP: [
                    {"replacements": [{"original": "Alice", "label": "first_name", "synthetic": "Maya"}]}
                ],
            }
        ),
        failed_records=[],
    )
    workflow = LlmReplaceWorkflow(adapter=adapter)

    input_df = pd.DataFrame(
        {
            COL_TEXT: ["No entities here", "Alice works at Acme", "Still no entities"],
            COL_ENTITIES_BY_VALUE: [
                {"entities_by_value": []},
                {"entities_by_value": [{"value": "Alice", "labels": ["first_name"]}]},
                {"entities_by_value": []},
            ],
            "tagged_text": [
                "No entities here",
                "<<PII:first_name>>Alice<</PII:first_name>> works at Acme",
                "Still no entities",
            ],
        }
    )

    result = workflow.generate_map_only(
        input_df,
        model_configs=stub_model_configs,
        selected_models=stub_replace_model_selection,
    )

    assert result.dataframe[COL_TEXT].tolist() == ["No entities here", "Alice works at Acme", "Still no entities"]
    assert result.dataframe[COL_REPLACEMENT_MAP].tolist() == [
        {"replacements": []},
        {"replacements": [{"original": "Alice", "label": "first_name", "synthetic": "Maya"}]},
        {"replacements": []},
    ]


def test_generate_map_only_strips_internal_prompt_columns(
    stub_model_configs: list[ModelConfig],
    stub_replace_model_selection: ReplaceModelSelection,
) -> None:
    """Workflow-internal prompt-construction columns must be stripped from the
    result. They carry pyarrow-backed pandas extension dtypes that would break
    a downstream `trace_dataframe.to_parquet` round-trip, and nothing
    downstream of this workflow consumes them.
    """
    adapter = Mock()
    adapter.run_workflow.return_value = _positional(
        dataframe=pd.DataFrame(
            {
                COL_TEXT: ["Alice works at Acme"],
                COL_ENTITIES_BY_VALUE: [{"entities_by_value": [{"value": "Alice", "labels": ["first_name"]}]}],
                "tagged_text": ["<<PII:first_name>>Alice<</PII:first_name>> works at Acme"],
                "_anonymizer_row_order": [0],
                COL_REPLACEMENT_MAP: [
                    {"replacements": [{"original": "Alice", "label": "first_name", "synthetic": "Maya"}]}
                ],
                # NDD passes input columns through to its output; simulate that so the
                # drop in the entity-rows return path actually has work to do.
                **{col: [""] for col in _INTERNAL_COLUMNS},
            }
        ),
        failed_records=[],
    )
    workflow = LlmReplaceWorkflow(adapter=adapter)

    input_df = pd.DataFrame(
        {
            COL_TEXT: ["Alice works at Acme"],
            COL_ENTITIES_BY_VALUE: [{"entities_by_value": [{"value": "Alice", "labels": ["first_name"]}]}],
            "tagged_text": ["<<PII:first_name>>Alice<</PII:first_name>> works at Acme"],
        }
    )

    result = workflow.generate_map_only(
        input_df,
        model_configs=stub_model_configs,
        selected_models=stub_replace_model_selection,
    )

    for col in _INTERNAL_COLUMNS:
        assert col not in result.dataframe.columns, f"workflow-internal column {col!r} leaked into result"


def test_generate_map_only_strips_internal_prompt_columns_when_no_entities(
    stub_model_configs: list[ModelConfig],
    stub_replace_model_selection: ReplaceModelSelection,
) -> None:
    """Same guarantee as above for the early-return path (no entities → no NDD call)."""
    adapter = Mock()
    workflow = LlmReplaceWorkflow(adapter=adapter)

    input_df = pd.DataFrame(
        {
            COL_TEXT: ["No entities here"],
            COL_ENTITIES_BY_VALUE: [{"entities_by_value": []}],
            "tagged_text": ["No entities here"],
        }
    )

    result = workflow.generate_map_only(
        input_df,
        model_configs=stub_model_configs,
        selected_models=stub_replace_model_selection,
    )

    adapter.run_workflow.assert_not_called()
    for col in _INTERNAL_COLUMNS:
        assert col not in result.dataframe.columns, f"workflow-internal column {col!r} leaked into result"


# ---------------------------------------------------------------------------
# PII-free logging regression tests for _filter_replacement_map_to_input_entities
#
# The replacement map filter runs after the LLM proposes substitutions and
# emits a DEBUG summary plus a WARNING when the filter empties out. Both
# log paths must report counts and labels only, never raw entity values.
# ---------------------------------------------------------------------------

_ORIGINAL_PII = ("Jane Doe", "jane.doe@example.com", "+1-555-867-5309")
_SYNTHETIC_PII = ("Maya Chen", "maya.chen@example.com", "+1-555-000-1234")


def _assert_no_pii_in_logs(caplog: pytest.LogCaptureFixture, extra_secrets: tuple[str, ...] = ()) -> None:
    for secret in (*_ORIGINAL_PII, *_SYNTHETIC_PII, *extra_secrets):
        assert secret not in caplog.text, f"PII leak in logs: {secret!r} appeared in:\n{caplog.text}"


def test_filter_replacement_map_debug_log_does_not_leak_pii(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The DEBUG summary on the happy path must not emit raw entity values."""
    parsed_entities = EntitiesByValueSchema.model_validate(
        {
            "entities_by_value": [
                {"value": "Jane Doe", "labels": ["first_name"]},
                {"value": "jane.doe@example.com", "labels": ["email"]},
                {"value": "+1-555-867-5309", "labels": ["phone_number"]},
            ]
        }
    )
    raw_map = {
        "replacements": [
            {"original": "Jane Doe", "label": "first_name", "synthetic": "Maya Chen"},
            {"original": "jane.doe@example.com", "label": "email", "synthetic": "maya.chen@example.com"},
            {"original": "+1-555-867-5309", "label": "phone_number", "synthetic": "+1-555-000-1234"},
        ]
    }

    with caplog.at_level(logging.DEBUG, logger="anonymizer"):
        result = _filter_replacement_map_to_input_entities(
            raw_map=raw_map, parsed_entities=parsed_entities, record_id="row-abc123"
        )

    assert "Replacement map: requested=" in caplog.text
    assert "requested=3" in caplog.text
    _assert_no_pii_in_logs(caplog)
    assert len(result["replacements"]) == 3


def test_filter_replacement_map_anomaly_summaries_do_not_leak_pii(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Anomaly logs expose counts while retaining labels and values in returned maps."""
    parsed_entities = EntitiesByValueSchema.model_validate(
        {
            "entities_by_value": [
                {"value": "Jane Doe", "labels": ["first_name"]},
                {"value": "+1-555-867-5309", "labels": ["phone_number"]},
            ]
        }
    )
    raw_map = {
        "replacements": [
            {"original": "Jane Doe", "label": "first_name", "synthetic": "Maya Chen"},
            {"original": "Acme Corp", "label": "organization_name", "synthetic": "NovaCorp"},
        ]
    }

    with caplog.at_level(logging.DEBUG, logger="anonymizer"):
        _filter_replacement_map_to_input_entities(raw_map=raw_map, parsed_entities=parsed_entities, record_id="row-xyz")

    assert "unrequested=1" in caplog.text
    assert "unfilled=1" in caplog.text
    assert "organization_name" not in caplog.text
    assert "phone_number" not in caplog.text

    _assert_no_pii_in_logs(caplog, extra_secrets=("Acme Corp", "NovaCorp"))


def test_fold_repairs_synthetic_original_collisions_without_pii(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Synthetic values must not reuse another protected original from the same row."""
    maps = _maps([("date", "1979-01-01", "1980-02-02"), ("date", "1980-02-02", "1991-03-04")])

    with caplog.at_level(logging.WARNING, logger="anonymizer"):
        result = _fold_replacement_maps(maps, "record")

    assert result == _maps([("date", "1979-01-01", "[SUBSTITUTE_DATE_1]"), ("date", "1980-02-02", "1991-03-04")])
    assert "placeholder substitutes to avoid collisions" in caplog.text
    assert "date" not in caplog.text
    _assert_no_pii_in_logs(caplog, extra_secrets=("1979-01-01", "1980-02-02", "1991-03-04"))


def test_filter_replacement_map_empty_warning_does_not_leak_pii(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The empty-after-filtering WARNING must report counts only."""
    parsed_entities = EntitiesByValueSchema.model_validate(
        {
            "entities_by_value": [
                {"value": "Jane Doe", "labels": ["first_name"]},
            ]
        }
    )
    raw_map = {
        "replacements": [
            {"original": "Acme Corp", "label": "organization_name", "synthetic": "NovaCorp"},
        ]
    }

    with caplog.at_level(logging.WARNING, logger="anonymizer"):
        result = _filter_replacement_map_to_input_entities(
            raw_map=raw_map, parsed_entities=parsed_entities, record_id="row-empty"
        )

    assert "Replacement map empty after filtering" in caplog.text
    assert "first_name" not in caplog.text
    _assert_no_pii_in_logs(caplog, extra_secrets=("Acme Corp", "NovaCorp"))
    assert result == {"replacements": []}


def _maps(*records: list[tuple[str, str, str]]) -> list[dict]:
    return [{"replacements": [{"label": lb, "original": o, "synthetic": sy} for lb, o, sy in r]} for r in records]


_ACME_TWICE = _maps([("company", "Acme", "Nova")], [("company", "Acme", "Zeta")])
_ANN_TWICE = _maps([("first_name", "Ann", "Bea")], [("first_name", "Ann", "Cy")])
_CASE = _maps([("person", "John Smith", "Mark Lee"), ("person", "JOHN SMITH", "Paul Hu")])
_NNBSP = _maps([("person", "John Smith", "Mark Lee"), ("person", "John\u202fsmith", "Paul Hu")])
_TWO_LABELS = _maps([("city", "Paris", "Lyon")], [("first_name", "Paris", "Dana")])
_COLLISION = _maps([("company", "Acme", "Nova")], [("company", "Acme", "Zeta"), ("company", "Initech", "Nova")])
_COLLISION_RUN = _maps(
    [("company", "Acme", "Nova")], [("company", "Acme", "Nova"), ("company", "Initech", "[SUBSTITUTE_COMPANY_1]")]
)
_NEWCOMER_FIRST = _maps([("company", "Acme", "Nova")], [("company", "Initech", "Nova"), ("company", "Acme", "Zeta")])
_NEWCOMER_FIRST_RUN = _maps(
    [("company", "Acme", "Nova")], [("company", "Initech", "[SUBSTITUTE_COMPANY_1]"), ("company", "Acme", "Nova")]
)
_HOLDER_AFTER = _maps(
    [("first_name", "Alice", "Maya")],
    [("first_name", "Bob", "Nora"), ("first_name", "Alice", "Nora"), ("first_name", "Maya", "Lena")],
)
_HOLDER_AFTER_FOLDED = _maps(
    [("first_name", "Alice", "Maya")],
    [
        ("first_name", "Bob", "Nora"),
        ("first_name", "Alice", "[SUBSTITUTE_FIRST_NAME_1]"),
        ("first_name", "Maya", "Lena"),
    ],
)
_CROSS_KEY = _maps([("first_name", "Alice", "Maya")], [("first_name", "Carl", "Maya")])
_CROSS_KEY_RUN = _maps([("first_name", "Alice", "Maya")], [("first_name", "Carl", "[SUBSTITUTE_FIRST_NAME_1]")])
_SINGLE = _maps([("company", "Acme", "Nova"), ("person", "Bob", "Tim")])
_VARIANT = _maps([("first_name", "Alice", "Maya"), ("first_name", "ALICE", "NORA"), ("first_name", "MAYA", "LENA")])
_VARIANT_FOLDED = _maps(
    [
        ("first_name", "Alice", "[SUBSTITUTE_FIRST_NAME_1]"),
        ("first_name", "ALICE", "[SUBSTITUTE_FIRST_NAME_1]"),
        ("first_name", "MAYA", "LENA"),
    ]
)
_SAME_OWN = _maps([("first_name", "Alice", "Maya"), ("first_name", "Bob", "Maya")])
_SAME_OWN_FOLDED = _maps([("first_name", "Alice", "Maya"), ("first_name", "Bob", "[SUBSTITUTE_FIRST_NAME_1]")])
_UNMAPPED = _maps([("first_name", "Alice", "Maya")])
_UNMAPPED_FOLDED = _maps([("first_name", "Alice", "[SUBSTITUTE_FIRST_NAME_1]")])
_DOTLESS = _maps([("first_name", "Ayşe", "Yıldız"), ("first_name", "AYŞE", "Can"), ("first_name", "YILDIZ", "Deniz")])
_RUN_VARIANT = _maps(
    [("first_name", "Alice", "Maya")],
    [("first_name", "Ann", "Mia"), ("first_name", "ALICE", "NORA"), ("first_name", "MAYA", "LENA")],
)


@pytest.mark.parametrize(
    ("maps", "record", "run", "requested"),
    [
        (*row, None)
        for row in [
            (_ACME_TWICE, _ACME_TWICE, _maps([("company", "Acme", "Nova")], [("company", "Acme", "Nova")])),
            (_CASE, *[_maps([("person", "John Smith", "Mark Lee"), ("person", "JOHN SMITH", "MARK LEE")])] * 2),
            (_NNBSP, *[_maps([("person", "John Smith", "Mark Lee"), ("person", "John\u202fsmith", "Mark Lee")])] * 2),
            (_TWO_LABELS, _TWO_LABELS, _TWO_LABELS),
            (_ANN_TWICE, _ANN_TWICE, _maps([("first_name", "Ann", "Bea")], [("first_name", "Ann", "Bea")])),
            (_COLLISION, _COLLISION, _COLLISION_RUN),
            (_CROSS_KEY, _CROSS_KEY, _CROSS_KEY_RUN),
            (_SINGLE, _SINGLE, _SINGLE),
            (_VARIANT, *[_VARIANT_FOLDED] * 2),
            (_RUN_VARIANT, _RUN_VARIANT, _RUN_VARIANT),
            (_DOTLESS, _DOTLESS, _DOTLESS),
            (_SAME_OWN, *[_SAME_OWN_FOLDED] * 2),
            (_NEWCOMER_FIRST, _NEWCOMER_FIRST, _NEWCOMER_FIRST_RUN),
            (_HOLDER_AFTER, *[_HOLDER_AFTER_FOLDED] * 2),
        ]
    ]
    + [(_UNMAPPED, *[_UNMAPPED_FOLDED] * 2, [{"Alice", "Maya"}])],
    ids=[
        "cross-record",
        "case",
        "nnbsp",
        "two-labels",
        "first-name",
        "collision",
        "cross-key",
        "single",
        "variant",
        "run-variant",
        "dotless-i",
        "same-own",
        "newcomer-first",
        "holder-fallback-after-proposer",
        "unmapped-entity",
    ],
)
def test_fold_replacement_maps(
    maps: list[dict], record: list[dict], run: list[dict], requested: list[set[str]] | None
) -> None:
    assert _fold_replacement_maps(maps, "record", requested) == record
    assert _fold_replacement_maps(maps, "run", requested) == run


@pytest.mark.parametrize("consistency", ["record", "run"])
def test_fold_never_emits_a_colliding_substitute(consistency: Literal["record", "run"]) -> None:
    rng = random.Random(20261106)
    names = ["Ayşe", "Yıldız", "İsmail", "Irmak", "ılgın", "Maya", "Alice", "Lena"]

    def variant(name: str) -> str:
        cased = rng.choice([name, name.upper(), name.lower(), name.title()])
        return rng.choice(["", " "]) + cased.replace("a", rng.choice(["a", "A"])) + rng.choice(["", "  "])

    maps = [
        {
            "replacements": [
                {
                    "label": rng.choice(["first_name", "city"]),
                    "original": variant(o),
                    "synthetic": variant(rng.choice(names)),
                }
                for o in rng.choices(names, k=rng.randint(1, 6))
            ]
        }
        for _ in range(300)
    ]
    unmapped = [{variant(o) for o in rng.choices(names, k=rng.randint(0, 2))} for _ in maps]
    run_wide: dict[str, set[tuple[str, str]]] = {}
    for raw, extra, out in zip(maps, unmapped, _fold_replacement_maps(maps, consistency, unmapped), strict=True):
        originals = {_fold(e["original"]) for e in raw["replacements"]} | {_fold(v) for v in extra}
        by_value: dict[str, set[tuple[str, str]]] = {}
        for e in out["replacements"]:
            value = _fold(e["synthetic"])
            assert value not in originals, (raw, out)
            by_value.setdefault(value, set()).add((e["label"], _fold(e["original"])))
        assert all(len(keys) == 1 for keys in by_value.values()), (raw, out)
        for value, keys in by_value.items():
            run_wide.setdefault(value, set()).update(keys)
    if consistency == "run":
        assert all(len(keys) == 1 for keys in run_wide.values()), run_wide


_FALLBACKS = _maps(
    [("first_name", "Alice", "Maya"), ("first_name", "ALICE", "NORA"), ("first_name", "Bob", "Maya")],
    [("city", "Paris", "Lyon"), ("city", "LYON", "Nice")],
)
_FALLBACK_COUNTS = {"city": {"placeholder_protected": 1}, "first_name": {"own": 1, "placeholder_taken": 1}}
# Records 1 and 2 propose one substitute for different names; record 3 repeats record 2's name. In run mode Bob
# falls back to a run-wide placeholder once and record 3 reuses it without a new count.
_REUSED = _maps([("first_name", "Alice", "Maya")], [("first_name", "Bob", "Maya")], [("first_name", "Bob", "Zed")])


@pytest.mark.parametrize(
    ("maps", "expected"),
    [
        (_FALLBACKS, {"record": (5, 2, _FALLBACK_COUNTS), "run": (5, 2, _FALLBACK_COUNTS)}),
        (_REUSED, {"record": (3, 0, {}), "run": (3, 1, {"first_name": {"placeholder_taken": 1}})}),
    ],
    ids=["within_record", "across_records"],
)
@pytest.mark.parametrize("measured", [True, False])
@pytest.mark.parametrize("consistency", ["record", "run"])
def test_generate_map_only_records_fold_fallbacks(
    stub_model_configs: list[ModelConfig],
    stub_replace_model_selection: ReplaceModelSelection,
    consistency: Literal["record", "run"],
    measured: bool,
    maps: list[dict],
    expected: dict[str, tuple[int, int, dict]],
) -> None:
    entities = [
        {"entities_by_value": [{"value": e["original"], "labels": [e["label"]]} for e in m["replacements"]]}
        for m in maps
    ]
    texts = [str(i) for i in range(len(maps))]
    adapter = Mock()
    adapter.run_workflow.return_value = _positional(
        dataframe=pd.DataFrame(
            {
                COL_TEXT: texts,
                COL_ENTITIES_BY_VALUE: entities,
                "tagged_text": texts,
                "_anonymizer_row_order": list(range(len(maps))),
                COL_REPLACEMENT_MAP: maps,
            }
        ),
        failed_records=[],
    )
    input_df = pd.DataFrame({COL_TEXT: texts, COL_ENTITIES_BY_VALUE: entities, "tagged_text": texts})
    with measurement_session() if measured else nullcontext() as collector:
        result = LlmReplaceWorkflow(adapter=adapter).generate_map_only(
            input_df,
            model_configs=stub_model_configs,
            selected_models=stub_replace_model_selection,
            consistency=consistency,
        )

    assert result.dataframe[COL_REPLACEMENT_MAP].tolist() == _fold_replacement_maps(maps, consistency)
    if collector is None:
        return
    [stage] = [r for r in collector.records if r.get("stage") == "LlmReplaceWorkflow.fold"]
    entry_count, with_placeholder, fallback_counts = expected[consistency]
    assert {k: stage[k] for k in ("consistency", "input_row_count", "entry_count", "records_with_placeholder")} == {
        "consistency": consistency,
        "input_row_count": len(maps),
        "entry_count": entry_count,
        "records_with_placeholder": with_placeholder,
    }
    assert stage["fallback_counts"] == fallback_counts


@pytest.mark.parametrize("consistency", ["record", "run"])
def test_generate_map_only_records_fold_when_no_row_has_entities(
    stub_model_configs: list[ModelConfig],
    stub_replace_model_selection: ReplaceModelSelection,
    consistency: Literal["record", "run"],
) -> None:
    adapter = Mock()
    input_df = pd.DataFrame({COL_TEXT: ["a", "b"], COL_ENTITIES_BY_VALUE: [{"entities_by_value": []}] * 2})
    with measurement_session() as collector:
        result = LlmReplaceWorkflow(adapter=adapter).generate_map_only(
            input_df,
            model_configs=stub_model_configs,
            selected_models=stub_replace_model_selection,
            consistency=consistency,
        )

    adapter.run_workflow.assert_not_called()
    assert result.failed_records == []
    assert result.source_positions == (0, 1)
    assert result.dataframe[COL_REPLACEMENT_MAP].tolist() == [{"replacements": []}] * 2
    [stage] = [r for r in collector.records if r.get("stage") == "LlmReplaceWorkflow.fold"]
    assert {k: stage[k] for k in ("consistency", "input_row_count", "entry_count", "records_with_placeholder")} == {
        "consistency": consistency,
        "input_row_count": 2,
        "entry_count": 0,
        "records_with_placeholder": 0,
    }
    assert stage["fallback_counts"] == {}
