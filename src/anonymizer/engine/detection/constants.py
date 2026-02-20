# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

# ---------------------------------------------------------------------------
# Column names for the entity detection pipeline
#
# Every column produced or consumed by detection steps is defined here.
# These are referenced in:
#   - custom_columns.py (row dict keys)
#   - detection_workflow.py  (NDD column configs + Jinja2 prompt templates)
#   - tests
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

# Latent detection (optional second workflow)
COL_LATENT_ENTITIES = "_latent_entities"

# Final output
COL_FINAL_ENTITIES = "final_entities"

# ---------------------------------------------------------------------------
# Jinja2 helper
# ---------------------------------------------------------------------------


def _jinja(col: str) -> str:
    """Wrap a column constant in Jinja2 template syntax: ``{{ col }}``."""
    return "{{ " + col + " }}"


# ---------------------------------------------------------------------------
# Default entity labels for GLiNER detection
# ---------------------------------------------------------------------------

DEFAULT_ENTITY_LABELS: list[str] = [
    "occupation",
    "certificate_license_number",
    "first_name",
    "date_of_birth",
    "ssn",
    "medical_record_number",
    "password",
    "unique_id",
    "phone_number",
    "national_id",
    "swift_bic",
    "company_name",
    "country",
    "license_plate",
    "tax_id",
    "employee_id",
    "pin",
    "state",
    "email",
    "date_time",
    "api_key",
    "biometric_identifier",
    "credit_debit_card",
    "coordinate",
    "device_identifier",
    "city",
    "postcode",
    "bank_routing_number",
    "vehicle_identifier",
    "health_plan_beneficiary_number",
    "url",
    "ipv4",
    "last_name",
    "cvv",
    "customer_id",
    "date",
    "user_name",
    "street_address",
    "ipv6",
    "account_number",
    "time",
    "age",
    "fax_number",
    "county",
    "gender",
    "sexuality",
    "political_view",
    "race_ethnicity",
    "religious_belief",
    "language",
    "blood_type",
    "mac_address",
    "http_cookie",
    "employment_status",
    "education_level",
]
