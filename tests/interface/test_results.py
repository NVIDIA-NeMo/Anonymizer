# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pandas as pd

from anonymizer.interface.results import AnonymizerResult, PreviewResult


def test_anonymizer_result_repr_is_compact() -> None:
    result = AnonymizerResult(
        dataframe=pd.DataFrame({"bio": ["a"], "bio_replaced": ["b"]}),
        trace_dataframe=pd.DataFrame(
            {
                "__nemo_anonymizer_text_input__": ["a"],
                "__nemo_anonymizer_text_output__": ["b"],
                "_detected_entities": [[]],
            }
        ),
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
                "_detected_entities": [[]],
            }
        ),
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
