# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pinned integration with the external vLLM Factory project."""

from __future__ import annotations

import importlib
import json
import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal, assert_never

from inference_service_compiler.models import FactoryPlugin

VLLM_FACTORY_SOURCE_REVISION = "7d6ff68ce68f9f7c0a9d72f9645bcf6d335d02f0"
VLLM_FACTORY_SOURCE_URL = "https://github.com/latenceainew/vllm-factory.git"
VLLM_FACTORY_DEPENDENCY = f"vllm-factory[gliner] @ git+{VLLM_FACTORY_SOURCE_URL}@{VLLM_FACTORY_SOURCE_REVISION}"

FactoryIoProcessor = Literal["deberta_gliner_io", "deberta_gliner2_io"]


@dataclass(frozen=True, slots=True)
class FactoryPluginSpec:
    """Immutable, characterized contract for one supported Factory plugin."""

    io_processor: FactoryIoProcessor
    characterized_models: frozenset[str]


def factory_plugin_spec(plugin: FactoryPlugin) -> FactoryPluginSpec:
    """Return the exhaustive immutable specification for a Factory plugin."""
    match plugin:
        case "deberta_gliner":
            return FactoryPluginSpec("deberta_gliner_io", frozenset({"nvidia/gliner-pii"}))
        case "deberta_gliner2":
            return FactoryPluginSpec(
                "deberta_gliner2_io",
                frozenset({"fastino/gliner2-privacy-filter-PII-multi"}),
            )
        case _:
            assert_never(plugin)


def prepare_model(
    *,
    model_id: str,
    revision: str | None,
    plugin: FactoryPlugin,
    prepared_model_root: str,
) -> str:
    """Prepare one pinned model through vLLM Factory's Python API."""
    if revision is None:
        raise ValueError("vLLM Factory models require a pinned Hugging Face revision")

    output = _prepared_model_path(
        root=Path(prepared_model_root),
        model_id=model_id,
        revision=revision,
        plugin=plugin,
    )
    provenance_path = output / ".anonymizer-vllm-factory.json"
    provenance = {
        "model_id": model_id,
        "model_revision": revision,
        "plugin": plugin,
        "vllm_factory_source": VLLM_FACTORY_SOURCE_URL,
        "vllm_factory_revision": VLLM_FACTORY_SOURCE_REVISION,
    }
    force = not _matches_provenance(provenance_path, provenance)

    model_prep = importlib.import_module("forge.model_prep")
    transformers = importlib.import_module("transformers")
    with _pin_hugging_face_revision(model_prep, transformers, model_id, revision):
        prepared = model_prep.prepare_model_for_vllm_if_needed(
            model_ref=model_id,
            plugin=plugin,
            output_dir=str(output),
            force=force,
        )
    if prepared == model_id:
        raise RuntimeError(f"vLLM Factory did not prepare GLiNER model {model_id!r}")
    output.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return str(output)


def io_processor_for(plugin: FactoryPlugin) -> FactoryIoProcessor:
    """Resolve the vLLM IOProcessor entry point for a supported factory plugin."""
    return factory_plugin_spec(plugin).io_processor


def supports_model(plugin: FactoryPlugin, model_id: str) -> bool:
    """Return whether this source revision was characterized for the pair."""
    return model_id in factory_plugin_spec(plugin).characterized_models


def _prepared_model_path(*, root: Path, model_id: str, revision: str, plugin: FactoryPlugin) -> Path:
    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "--", model_id).strip("-")
    return root / safe_model / revision / plugin


def _matches_provenance(path: Path, expected: dict[str, str]) -> bool:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False
    return value == expected


@contextmanager
def _pin_hugging_face_revision(
    model_prep: Any,
    transformers: Any,
    model_id: str,
    revision: str,
) -> Iterator[None]:
    """Supply the missing revision argument at vLLM Factory's hub boundary."""
    original_download = model_prep.hf_hub_download
    original_list = model_prep.list_repo_files
    original_tokenizer = transformers.AutoTokenizer.from_pretrained

    def pinned_download(*args: Any, **kwargs: Any) -> Any:
        kwargs["revision"] = revision
        return original_download(*args, **kwargs)

    def pinned_list(*args: Any, **kwargs: Any) -> Any:
        kwargs["revision"] = revision
        return original_list(*args, **kwargs)

    def pinned_tokenizer(source: str, *args: Any, **kwargs: Any) -> Any:
        if source == model_id:
            kwargs["revision"] = revision
        return original_tokenizer(source, *args, **kwargs)

    model_prep.hf_hub_download = pinned_download
    model_prep.list_repo_files = pinned_list
    transformers.AutoTokenizer.from_pretrained = pinned_tokenizer
    try:
        yield
    finally:
        model_prep.hf_hub_download = original_download
        model_prep.list_repo_files = original_list
        transformers.AutoTokenizer.from_pretrained = original_tokenizer
