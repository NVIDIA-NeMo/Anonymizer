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
COL_VALIDATION_SKELETON = "_validation_skeleton"
COL_VALIDATION_DECISIONS = "_validation_decisions"
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

# ---------------------------------------------------------------------------
# Rewrite pipeline
# ---------------------------------------------------------------------------

# Internal columns (prefixed with _)

COL_DOMAIN = "_domain"
COL_DOMAIN_SUPPLEMENT = "_domain_supplement"
COL_SENSITIVITY_DISPOSITION = "_sensitivity_disposition"
COL_SENSITIVITY_DISPOSITION_BLOCK = "_sensitivity_disposition_block"
COL_REWRITE_DISPOSITION_BLOCK = "_rewrite_disposition_block"
COL_REPLACEMENT_MAP_FOR_PROMPT = "_replacement_map_for_prompt"
COL_FULL_REWRITE = "_full_rewrite"
COL_MEANING_UNITS = "_meaning_units"
COL_MEANING_UNITS_SERIALIZED = "_meaning_units_serialized"
COL_QUALITY_QA = "_quality_qa"
COL_PRIVACY_QA = "_privacy_qa"
COL_REWRITTEN_TEXT = "_rewritten_text"  # pre-repair intermediate; renamed to {text_col}_rewritten in user output
COL_QUALITY_QA_REANSWER = "_quality_qa_reanswer"
COL_QUALITY_QA_COMPARE = "_quality_qa_compare"
COL_PRIVACY_QA_REANSWER = "_privacy_qa_reanswer"
COL_NEEDS_REPAIR = "_needs_repair"
COL_LEAKED_PRIVACY_ITEMS = "_leaked_privacy_items"
COL_REWRITTEN_TEXT_NEXT = COL_REWRITTEN_TEXT + "__next"
COL_REPAIR_ITERATIONS = "_repair_iterations"
COL_JUDGE_EVALUATION = "_judge_evaluation"

# User-facing output columns
COL_UTILITY_SCORE = "utility_score"
COL_LEAKAGE_MASS = "leakage_mass"
COL_ANY_HIGH_LEAKED = "any_high_leaked"
COL_NEEDS_HUMAN_REVIEW = "needs_human_review"

# ---------------------------------------------------------------------------
# Entity labels and examples
#
# Source of truth for default entity labels and their examples.
# Used by both detection (validation/augment prompts) and replacement prompts.
# ---------------------------------------------------------------------------

ENTITY_LABEL_EXAMPLES: dict[str, list[str]] = {
    "occupation": ["registered nurse", "truck driver", "software engineer", "retail salesperson"],
    "certificate_license_number": ["ENG-TX-20240513", "A9825473", "RN-123456", "CPA-78901"],
    "first_name": ["Michael", "Isabella", "Carlos", "Wei"],
    "date_of_birth": ["1986-12-29", "03/15/1990", "1991-12-05", "1988-03-02"],
    "ssn": ["007-52-4910", "252-96-0016", "523-25-1554", "228-94-9430"],
    "medical_record_number": ["MRN-345672", "BOS-00025836", "MRN-567234", "00058362"],
    "password": ["Rainbow@2025", "P@ssw0rd!", "River@2025", "River2025!"],
    "unique_id": ["2e008d4415b57d036b51", "d2b796a8-161f-4d0c-b3e5-2c9f8a1b3c92", "987654321"],
    "phone_number": ["949-307-5488", "(212) 555-7890", "+254 712 345 678", "+1-800-555-0199"],
    "national_id": ["128456189092325", "45789-0123-456", "AB123456C", "JQ 12 54 26 7"],
    "swift_bic": ["QWERTUS45ZYX", "BOFAUS3N", "DEUTDEFF", "XPLAAU6RZ"],
    "company_name": ["VerdantBio", "Hartford Construction Group", "MediaPulse", "Lumina Entertainment"],
    "country": ["United States", "Japan", "Russia", "United Kingdom"],
    "license_plate": ["JXK-732", "KTP-9837", "IRB 5721", "D SZ 5814"],
    "tax_id": ["489-32-1765", "16-3189372", "781534867390", "EIN 98-7654321"],
    "employee_id": ["MK4567", "21MKT347Z", "EMP-001234", "SM345"],
    "pin": ["358495", "1634", "248593", "9404"],
    "state": ["Texas", "California", "Gyeonggi", "Krasnodar Krai"],
    "email": ["derez_lester94@icloud.com", "clayton.burke@hotmail.com", "amina.e@sudanlinklogistics.com"],
    "date_time": ["2023-07-31T16:34:56", "2024-08-08T10:21:02", "March 15, 2024 2:30 PM"],
    "api_key": ["a1b2c3d4-e5f6-78g9-h0i1-j2k3l4m5n6o7", "sk-abc123def456"],
    "biometric_identifier": ["BIO-5739126845", "M49283715672", "BIO-782654913"],
    "credit_debit_card": ["4920 1254 5278 9812", "5412 3656 9820 1634", "5123 3587 8301 2745"],
    "coordinate": ["47.6062, -122.3321", "36.7783, -119.4179", "51.207812, 4.429671"],
    "device_identifier": ["a1b2c3d4e5f6g7h8", "IMEI:123456789012345", "490154203237518"],
    "city": ["Houston", "San Diego", "Doha", "Lahore"],
    "postcode": ["77450", "92101", "SW1A 1AA", "50630-100"],
    "bank_routing_number": ["061102356", "801232597", "012745278"],
    "vehicle_identifier": ["WBA4J52K9MJ129456", "VSK5G71F34R000153", "SCF4K3L5J9M212645"],
    "health_plan_beneficiary_number": ["AET-7820-1264-15", "FL123496719", "WA-0003284668"],
    "url": ["https://shopify.com", "https://internal.corp.net/dashboard", "https://bestbuy.com?auth=6589"],
    "ipv4": ["192.168.1.1", "195.150.21.234", "157.200.130.19", "194.126.23.77"],
    "last_name": ["Smith", "Rodriguez", "McKenzie", "Nakamura"],
    "cvv": ["161", "884", "447", "760"],
    "customer_id": ["CUS439012", "SM-19382", "CUST-00012345", "SM-78321"],
    "date": ["07/15/2024", "2023-09-15", "March 15, 2024", "15/07/2024"],
    "user_name": ["jeffreymoon87", "stacy.flynn", "e.sullivan", "Henry1985"],
    "street_address": ["661 NE Regents Dr", "739 Main St", "4/87 Collins Street", "22 Boulevard Haussmann"],
    "ipv6": ["2001:0db8:85a3:0000:0000:8a2e:0370:7334", "2a02:4d60:1031::85e1:7341:9203:4c56"],
    "account_number": ["FR72-1534-5678-9082-3156-28", "230915-872513", "125456289"],
    "time": ["7:23 AM", "18:30", "18:23:45", "10h30"],
    "age": ["76", "51", "35", "41"],
    "fax_number": ["326-316-9410", "(512) 876-4321", "(212) 555-6789", "+1-800-555-0123"],
    "county": ["Los Angeles County", "Harris County", "Cook County", "Clark County"],
    "gender": ["female", "male", "transgender", "non-binary"],
    "sexuality": ["homosexual", "heterosexual", "gay", "lesbian"],
    "political_view": ["Republican", "Democrat", "Blue Dog Democrat", "Tea Party"],
    "race_ethnicity": ["white", "African-American", "Korean", "Hispanic"],
    "religious_belief": ["Christian", "Protestant", "Church of England", "Buddhist"],
    "language": ["English", "Spanish", "Mandarin", "Tagalog"],
    "blood_type": ["A+", "B+", "O positive", "AB-"],
    "mac_address": ["49:FD:EE:1A:3B:7C", "23:14:B5:67:89:AB", "A9:A5:CC:12:54:56"],
    "http_cookie": ["session_id=abc123xyz", "jwt_token=...; Path=/auth"],
    "employment_status": ["full-time", "part-time", "self-employed", "contractor"],
    "education_level": ["high school", "bachelor's degree", "some college", "graduate level"],
    "university": ["Stanford University", "University of Oxford", "National University of Singapore", "UCLA"],
    "court_name": [
        "U.S. District Court for the Northern District of California",
        "Supreme Court of the United Kingdom",
        "European Court of Human Rights",
        "Los Angeles County Superior Court",
    ],
    "prison_detention_facility": [
        "San Quentin State Prison",
        "Rikers Island Correctional Facility",
        "ADX Florence Supermax Prison",
        "Cook County Jail",
    ],
    "nationality": ["American", "Japanese", "Brazilian", "Nigerian"],
    "degree": ["Bachelor of Science", "Master of Arts", "Doctor of Philosophy", "Juris Doctor", "B.S.", "Ph.D"],
    "field_of_study": ["Computer Science", "Mechanical Engineering", "International Relations", "Public Health"],
    "place_name": ["Brooklyn", "Lake Tahoe", "Silicon Valley", "Downtown Seattle"],
    "landmark": ["Eiffel Tower", "Golden Gate Bridge", "Statue of Liberty", "Big Ben"],
    "organization_name": [
        "World Health Organization",
        "Amnesty International",
        "OpenAI",
        "International Monetary Fund",
    ],
    "monetary_amount": ["$125", "$18,400", "$1.2 million", "€245,000", "200 PLN", "50 EUR", "CAD 36,000", "¥850,000"],
}

DEFAULT_ENTITY_LABELS: list[str] = list(ENTITY_LABEL_EXAMPLES.keys())


# ---------------------------------------------------------------------------
# Prompt utilities
# ---------------------------------------------------------------------------


def _jinja(col: str, *, key: str | None = None) -> str:
    """Wrap a column name in Jinja2 template syntax for use in NDD prompts.

    When *key* is given the expression becomes ``{{ col['key'] }}``,
    providing a single grep-able call site for nested schema access.
    """
    expr = col if key is None else f"{col}['{key}']"
    return "{{ " + expr + " }}"
