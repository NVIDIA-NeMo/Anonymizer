# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Distributed-executor entrypoint for running the detection workflow on a SLURM cluster.

For running detection at scale, an external DataDesigner runtime (e.g. a SLURM
orchestrator) provisions the model servers, partitions the dataset across workers, and
runs the workflow. Such runtimes usually ship a *serialized* config to each worker and
rebuild it with ``from_config`` — but the detection workflow can't go through that path:
it uses ``CustomColumnConfig`` columns whose ``generator_function`` is a live Python
callable (DataDesigner custom columns are "library only"), which do not survive JSON
serialization.

This module is the alternative: a factory the runtime imports and calls **in-process on
each worker** to get the live ``DataDesignerConfigBuilder`` (callables intact). The custom
columns reference their LLM by *alias* and receive model facades injected by the
DataDesigner runtime, so the runtime's provider wiring (alias → provisioned server) still
routes their calls correctly. The seed parquet is read from the path the runtime provides
(not rewritten — workers may share it), and ``num_jobs > 1`` selects this worker's ordered
partition.

The runtime calls:
    build_detection_builder(seed_path=..., job_index=..., num_jobs=..., spec={...})
where ``spec`` is the JSON-serializable detection spec produced by the submitting side.
Requires ``nemo-anonymizer`` installed in the worker environment.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from data_designer.config.config_builder import DataDesignerConfigBuilder

# Placeholder provider endpoint; the distributed runtime overrides providers at run time
# (the workflow is only *built* here, never run against this endpoint).
_PLACEHOLDER_ENDPOINT = "http://overridden-by-runtime:8000/v1"


def build_detection_builder(
    *,
    seed_path: str,
    spec: dict[str, Any],
    job_index: int = 0,
    num_jobs: int = 1,
) -> DataDesignerConfigBuilder:
    """Return the live detection ``DataDesignerConfigBuilder`` for one distributed worker.

    Args:
        seed_path: Path to the seed parquet the runtime placed on this worker (read, not
            written). Record ids are assumed already attached by the submitting side.
        spec: JSON-serializable detection spec with keys:
            ``model_configs_yaml`` (str): the Anonymizer model_configs YAML (selected_models
                + model_configs aliases) — the alias ``model`` ids must match the served
                model names the runtime provisions, so its provider wiring can map them.
            ``provider_names`` (list[str]): provider names referenced by the YAML; placeholder
                ``ModelProvider``s are created for them (the runtime supplies the real ones).
            ``detect`` (dict): ``gliner_threshold`` (float) and optional ``entity_labels``
                (list[str] | None).
            ``data_summary`` (str | None): optional dataset description for prompts.
        job_index: index of this worker's ordered partition of the seed.
        num_jobs: total number of partitions the seed is split across.
    """
    from anonymizer import Anonymizer, AnonymizerConfig, ModelProvider, Redact  # noqa: PLC0415
    from anonymizer.config.anonymizer_config import Detect  # noqa: PLC0415

    providers = [
        ModelProvider(name=name, endpoint=_PLACEHOLDER_ENDPOINT, provider_type="openai", api_key="EMPTY")
        for name in spec["provider_names"]
    ]
    anonymizer = Anonymizer(model_configs=spec["model_configs_yaml"], model_providers=providers)

    if "detect" not in spec:
        raise KeyError("spec must include required 'detect' section")
    detect = spec["detect"]
    detect_kwargs: dict[str, Any] = {"gliner_threshold": detect["gliner_threshold"]}
    if detect.get("entity_labels") is not None:
        detect_kwargs["entity_labels"] = detect["entity_labels"]
    config = AnonymizerConfig(detect=Detect(**detect_kwargs), replace=Redact())

    return anonymizer.export_detection_builder_for_seed(
        config=config,
        seed_path=seed_path,
        job_index=job_index,
        num_jobs=num_jobs,
        data_summary=spec.get("data_summary"),
    )
