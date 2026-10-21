# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Source-partition and private-storage faults of the opaque detection/Redact invocation."""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pandas as pd
import pytest

from anonymizer.engine.constants import COL_TEXT
from tests.engine import test_detection_source_frame as detection_tests
from tests.engine.test_detection_source_frame import run_graph_profile

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


@pytest.mark.parametrize("fault", ["overlap", "foreign", "unattributed", "duplicate"])
def test_delegated_source_partition_rejects_uncorrelated_completion(scripted_detection, monkeypatch, fault):
    from anonymizer.engine.detection._graph_invocation import DetectionRedactInvocation
    from anonymizer.engine.graph_sdk.requests import SemanticAssociation
    from anonymizer.graph import TaskAttemptId

    original = DetectionRedactInvocation._run

    def corrupted(self, inputs, requests):
        result = original(self, inputs, requests)
        association = result.results[0].association
        assert isinstance(association, SemanticAssociation)
        if fault == "unattributed":
            return replace(result, results=result.results[1:])
        if fault == "duplicate":
            raw = replace(result)
            object.__setattr__(raw, "failed_sources", (association, association))
            return raw
        if fault == "foreign":
            association = SemanticAssociation(task=TaskAttemptId.new(activation=association.task.activation))
        return replace(result, failed_sources=frozenset({association}))

    monkeypatch.setattr(DetectionRedactInvocation, "_run", corrupted)
    result = asyncio.run(run_graph_profile(scripted_detection, _source(), _options()))
    assert not result.final_outputs and not result.frames
    assert all(state.complete for state in result.states)


@pytest.mark.parametrize("failed", [False, True])
def test_create_projection_keeps_source_identity_out_of_persisted_parquet(scripted_detection, tmp_path, failed):
    from anonymizer.engine.constants import COL_REPLACED_TEXT
    from anonymizer.engine.harness.kernel import project_detection_redact

    source = pd.DataFrame(
        {COL_TEXT: ["Alice travels.", "Alice fails." if failed else "Alice stays."], "extra": [17, 29]}
    )
    result = asyncio.run(run_graph_profile(scripted_detection, source, {**_options(), "preview_num_records": None}))
    projected = project_detection_redact(result)
    assert projected.dataframe["extra"].tolist() == ([17] if failed else [17, 29])
    assert projected.dataframe[COL_REPLACED_TEXT].tolist() == (
        ["[REDACTED_FIRST_NAME] travels."]
        if failed
        else ["[REDACTED_FIRST_NAME] travels.", "[REDACTED_FIRST_NAME] stays."]
    )
    paths = list((tmp_path / "artifacts").rglob("*.parquet"))
    assert paths
    for path in paths:
        assert not any(name.startswith("_anonymizer_source_") for name in pd.read_parquet(path).columns)
    for path in (tmp_path / "artifacts").rglob("*.json"):
        assert "_anonymizer_source_" not in path.read_text()


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
