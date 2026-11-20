# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pickle

import pandas as pd

from anonymizer.config.replace_strategies import Hash
from anonymizer.engine.constants import COL_DETECTED_ENTITIES
from anonymizer.interface.results import AnonymizerResult, PreviewResult


def test_anonymizer_result_repr_is_compact() -> None:
    result = AnonymizerResult(
        dataframe=pd.DataFrame({"bio": ["a"], "bio_replaced": ["b"]}),
        trace_dataframe=pd.DataFrame(
            {
                "__nemo_anonymizer_text_input__": ["a"],
                "__nemo_anonymizer_text_output__": ["b"],
                COL_DETECTED_ENTITIES: [[]],
            }
        ),
        resolved_text_column="bio",
        failed_records=[],
    )
    rendered = repr(result)
    assert rendered.startswith("AnonymizerResult(")
    assert "rows=1" in rendered
    assert "columns=2" in rendered
    assert "trace_columns=3" in rendered
    assert "failed_records=0" in rendered
    assert "__nemo_anonymizer_text_input__" not in rendered
    assert "bio_replaced" not in rendered


def test_preview_result_repr_is_compact() -> None:
    preview = PreviewResult(
        dataframe=pd.DataFrame({"bio": ["a"], "bio_replaced": ["b"]}),
        trace_dataframe=pd.DataFrame(
            {
                "__nemo_anonymizer_text_input__": ["a"],
                "__nemo_anonymizer_text_output__": ["b"],
                COL_DETECTED_ENTITIES: [[]],
            }
        ),
        resolved_text_column="bio",
        failed_records=[],
        preview_num_records=10,
    )
    rendered = repr(preview)
    assert rendered.startswith("PreviewResult(")
    assert "rows=1" in rendered
    assert "columns=2" in rendered
    assert "trace_columns=3" in rendered
    assert "failed_records=0" in rendered
    assert "preview_num_records=10" in rendered
    assert "__nemo_anonymizer_text_input__" not in rendered
    assert "bio_replaced" not in rendered


def test_anonymizer_result_preserves_positional_data_summary_contract() -> None:
    result = AnonymizerResult(
        pd.DataFrame(),
        pd.DataFrame(),
        "bio",
        [],
        None,
        None,
        ["first_name"],
        "Customer support transcripts.",
    )

    assert result.data_summary == "Customer support transcripts."
    assert result.excluded_entity_labels is None


def test_preview_result_preserves_positional_data_summary_contract() -> None:
    result = PreviewResult(
        pd.DataFrame(),
        pd.DataFrame(),
        "bio",
        [],
        3,
        None,
        None,
        ["first_name"],
        "Customer support transcripts.",
    )

    assert result.data_summary == "Customer support transcripts."
    assert result.excluded_entity_labels is None


def test_anonymizer_result_positional_label_list_lands_in_effective_entity_labels() -> None:
    result = AnonymizerResult(
        pd.DataFrame(),
        pd.DataFrame(),
        "bio",
        [],
        None,
        None,
        None,
        None,
        None,
        None,
        ["custom_id"],
    )

    assert result.effective_entity_labels == ["custom_id"]
    assert result.detection_scope is None


def test_preview_result_positional_label_list_lands_in_effective_entity_labels() -> None:
    result = PreviewResult(
        pd.DataFrame(),
        pd.DataFrame(),
        "bio",
        [],
        3,
        None,
        None,
        None,
        None,
        None,
        None,
        ["custom_id"],
    )

    assert result.effective_entity_labels == ["custom_id"]
    assert result.detection_scope is None


def _results_with(method: Hash) -> list[AnonymizerResult | PreviewResult]:
    frame = pd.DataFrame({"bio": ["a"]})
    return [
        AnonymizerResult(
            dataframe=frame, trace_dataframe=frame, resolved_text_column="bio", failed_records=[], replace_method=method
        ),
        PreviewResult(
            dataframe=frame,
            trace_dataframe=frame,
            resolved_text_column="bio",
            failed_records=[],
            preview_num_records=1,
            replace_method=method,
        ),
    ]


def test_a_saved_result_keeps_no_hash_key() -> None:
    for result in _results_with(Hash(key="s3cr3t-key-material")):
        assert isinstance(result.replace_method, Hash)
        assert result.replace_method.key is None
        saved = pickle.dumps(result)
        assert b"s3cr3t-key-material" not in saved
        restored = pickle.loads(saved)
        assert isinstance(restored.replace_method, Hash) and restored.replace_method.key is None
