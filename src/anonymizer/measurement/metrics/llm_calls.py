# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math


def estimate_llm_calls_by_stage(
    *,
    mode: str,
    strategy: str,
    has_grouped_entities: bool,
    validation_chunk_count: int | None,
    repair_iterations: int = 0,
    replace_map_generation_uses_llm: bool = True,
) -> dict[str, int | None]:
    """Estimate nominal model calls for one record, split by workflow stage."""
    detection_calls = None if validation_chunk_count is None else 2 + validation_chunk_count
    replace_map_generation = 0
    if replace_map_generation_uses_llm and has_grouped_entities and (mode == "rewrite" or strategy == "Substitute"):
        replace_map_generation = 1

    if mode != "rewrite":
        return {
            "entity_detection": detection_calls,
            "replace_map_generation": replace_map_generation,
        }

    rewrite_body_calls = has_grouped_entities
    return {
        "entity_detection": detection_calls,
        "latent_entity_detection": 1 if rewrite_body_calls else 0,
        "replace_map_generation": replace_map_generation,
        "rewrite_pipeline": 5 if rewrite_body_calls else 0,
        "rewrite_evaluate": 3 * (1 + repair_iterations) if rewrite_body_calls else 0,
        "rewrite_repair": repair_iterations if rewrite_body_calls else 0,
        "rewrite_final_judge": 1 if rewrite_body_calls else 0,
    }


def _validation_chunk_count(
    detected_candidate_count: int | None,
    *,
    validation_max_entities_per_call: int,
) -> int | None:
    if detected_candidate_count is None:
        return None
    if detected_candidate_count <= 0:
        return 0
    return int(math.ceil(detected_candidate_count / validation_max_entities_per_call))
