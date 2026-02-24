# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

# ---------------------------------------------------------------------------
# Column names for the anonymizer pipeline
#
# Shared vocabulary across detection, replacement, and display layers.
# ---------------------------------------------------------------------------

# Input
COL_TEXT = "__nemo_anonymizer_text_input__"

# Step 1: GLiNER detection
COL_RAW_DETECTED = "_raw_detected_entities"

# Step 2: parse_detected_entities
COL_SEED_ENTITIES = "_seed_entities"
COL_INITIAL_TAGGED_TEXT = "_initial_tagged_text"
COL_SEED_ENTITIES_JSON = "_seed_entities_json"
COL_TAG_NOTATION = "_tag_notation"

# Step 3: LLM augmentation
COL_AUGMENTED_ENTITIES = "_augmented_entities"

# Step 4: merge_and_build_candidates
COL_MERGED_ENTITIES = "_merged_entities"
COL_MERGED_TAGGED_TEXT = "_merged_tagged_text"
COL_VALIDATION_CANDIDATES = "_validation_candidates"

# Step 5: LLM validation
COL_VALIDATED_ENTITIES = "_validated_entities"

# Step 6: apply_validation_and_finalize
COL_DETECTED_ENTITIES = "_detected_entities"
COL_TAGGED_TEXT = "tagged_text"
COL_ENTITIES_BY_VALUE = "_entities_by_value"
COL_REPLACED_TEXT = "__nemo_anonymizer_text_output__"
COL_REPLACEMENT_MAP = "_replacement_map"

# Latent detection (optional second workflow)
COL_LATENT_ENTITIES = "_latent_entities"

# Final output
COL_FINAL_ENTITIES = "final_entities"
