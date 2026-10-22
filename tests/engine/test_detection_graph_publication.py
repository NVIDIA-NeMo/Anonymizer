# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Source-partition and private-storage faults of the opaque detection/Redact invocation."""

from __future__ import annotations

import pandas as pd
import pytest

from anonymizer.engine.constants import COL_TEXT
from tests.engine import test_detection_source_frame as detection_tests

scripted_detection = detection_tests.scripted_detection

# Hang-guard bound for waits on normally-fast in-memory thread/future signals
# in this module. Not a measurement of expected runtime -- only a generous
# ceiling so a hung test fails instead of blocking CI, chosen to stay robust
# under a loaded CI host.
_HANG_GUARD_TIMEOUT = 30


def _source():
    return pd.DataFrame({COL_TEXT: ["Alice travels.", "Alice stays."], "extra": [17, 29]})


def _options():
    return {"gliner_detection_threshold": 0.3, "tag_latent_entities": False, "preview_num_records": 2}


@pytest.mark.parametrize("fault", [None, "foreign", "duplicate", "unregistered_batch"])
def test_private_storage_batch_identity_is_ordered_and_closed(tmp_path, fault):
    from data_designer.engine.storage.artifact_storage import ArtifactStorage, BatchStage

    from anonymizer.engine.ndd._source_frame import WorkflowSourceFrame
    from anonymizer.interface.errors import AnonymizerWorkflowError

    source = WorkflowSourceFrame.capture(pd.DataFrame({COL_TEXT: ["first", "second"]}))
    storage = ArtifactStorage(artifact_path=tmp_path, dataset_name="identity")
    original_write = storage.write_parquet_file
    with source._bind_storage(storage):
        if fault in {"foreign", "duplicate"}:
            invalid = source.seed()
            invalid[source.key_column] = [source.keys[0], "foreign" if fault == "foreign" else source.keys[0]]
            with pytest.raises(AnonymizerWorkflowError, match="invalid source correspondence"):
                storage.write_batch_to_parquet_file(0, invalid, BatchStage.PARTIAL_RESULT)
            assert not list(tmp_path.rglob("*.parquet"))
            return
        for index in (1, 0):
            storage.write_batch_to_parquet_file(index, source.seed().iloc[[index]], BatchStage.PARTIAL_RESULT)
            storage.move_partial_result_to_final_file_path(index)
        if fault == "unregistered_batch":
            original_write("batch_99999.parquet", pd.DataFrame({COL_TEXT: ["unregistered"]}), BatchStage.FINAL_RESULT)
            with pytest.raises(AnonymizerWorkflowError, match="unknown source batches"):
                storage.load_dataset()
        else:
            returned, positions = source.project(storage.load_dataset())
            assert returned[COL_TEXT].tolist() == ["first", "second"]
            assert positions == (0, 1)
        for path in tmp_path.rglob("*.parquet"):
            assert source.key_column not in pd.read_parquet(path).columns
    assert storage.write_parquet_file == original_write
