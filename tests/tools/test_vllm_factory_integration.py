# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the pinned external vLLM Factory preparation boundary."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from inference_service_compiler import vllm_factory_integration as integration


def test_factory_plugin_spec_is_immutable_and_complete() -> None:
    """Each supported plugin resolves all of its metadata through one product."""
    gliner = integration.factory_plugin_spec("deberta_gliner")
    gliner2 = integration.factory_plugin_spec("deberta_gliner2")

    assert gliner.io_processor == "deberta_gliner_io"
    assert gliner.characterized_models == frozenset({"nvidia/gliner-pii"})
    assert gliner2.io_processor == "deberta_gliner2_io"
    assert gliner2.characterized_models == frozenset({"fastino/gliner2-privacy-filter-PII-multi"})


def test_prepare_model_injects_pinned_revision_into_upstream_python_api(tmp_path: Path) -> None:
    """Anonymizer closes vLLM Factory's missing Hugging Face revision argument."""
    download = mock.Mock(return_value="/cache/file")
    list_files = mock.Mock(return_value=[])
    tokenizer = mock.Mock(return_value=mock.Mock())
    model_prep = SimpleNamespace(
        hf_hub_download=download,
        list_repo_files=list_files,
    )

    def prepare_model_for_vllm_if_needed(**kwargs):
        model_prep.list_repo_files(kwargs["model_ref"])
        model_prep.hf_hub_download(repo_id=kwargs["model_ref"], filename="config.json")
        transformers.AutoTokenizer.from_pretrained(kwargs["model_ref"])
        Path(kwargs["output_dir"]).mkdir(parents=True, exist_ok=True)
        return kwargs["output_dir"]

    model_prep.prepare_model_for_vllm_if_needed = prepare_model_for_vllm_if_needed
    transformers = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer))

    def import_module(name: str):
        return model_prep if name == "forge.model_prep" else transformers

    with mock.patch.object(integration.importlib, "import_module", side_effect=import_module):
        prepared = integration.prepare_model(
            model_id="nvidia/gliner-pii",
            revision="bd23e8ef",
            plugin="deberta_gliner",
            prepared_model_root=str(tmp_path),
        )

    assert Path(prepared).is_dir()
    list_files.assert_called_once_with("nvidia/gliner-pii", revision="bd23e8ef")
    download.assert_called_once_with(
        repo_id="nvidia/gliner-pii",
        filename="config.json",
        revision="bd23e8ef",
    )
    tokenizer.assert_called_once_with("nvidia/gliner-pii", revision="bd23e8ef")
