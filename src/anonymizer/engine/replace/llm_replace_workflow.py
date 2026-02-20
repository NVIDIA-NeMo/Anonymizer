# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from data_designer.config.column_configs import LLMStructuredColumnConfig
from data_designer.config.models import ModelConfig
from pydantic import BaseModel, Field

from anonymizer.config.models import ReplaceModelSelection
from anonymizer.engine.ndd.adapter import FailedRecord, NddAdapter
from anonymizer.engine.ndd.model_loader import resolve_model_alias


class EntityReplacement(BaseModel):
    original: str = Field(min_length=1, description="The original entity value")
    label: str = Field(min_length=1, description="The entity label/type")
    synthetic: str = Field(min_length=1, description="The synthetic replacement value")


class ReplacementMap(BaseModel):
    replacements: list[EntityReplacement] = Field(default_factory=list, description="List of entity replacements")


@dataclass(frozen=True)
class LlmReplaceResult:
    dataframe: pd.DataFrame
    failed_records: list[FailedRecord]


class LlmReplaceWorkflow:
    """Generate replacement maps via LLM workflow."""

    def __init__(self, adapter: NddAdapter, config_dir: Path | None = None) -> None:
        self._adapter = adapter
        self._config_dir = config_dir

    def generate_map_only(
        self,
        dataframe: pd.DataFrame,
        *,
        model_configs: list[ModelConfig] | str | Path,
        model_providers: list[Any] | str | Path | None,
        selected_models: ReplaceModelSelection,
        instructions: str | None = None,
        entities_column: str = "_entities_by_value",
        preview_num_records: int | None = None,
    ) -> LlmReplaceResult:
        replace_alias = resolve_model_alias(
            "replace_workflow",
            "replacement_generator",
            selected_models,
            self._config_dir,
        )

        working_df = dataframe.copy()
        working_df["_entity_examples"] = working_df.apply(
            lambda row: _create_entity_examples(row.get(entities_column, [])),
            axis=1,
        )
        working_df["_entities_for_replace"] = working_df.apply(
            lambda row: _normalize_entities_by_value(row.get(entities_column, [])),
            axis=1,
        )
        working_df["_entities_for_replace_json"] = working_df["_entities_for_replace"].apply(json.dumps)

        run_result = self._adapter.run_workflow(
            working_df,
            model_configs=model_configs,
            model_providers=model_providers,
            columns=[
                LLMStructuredColumnConfig(
                    name="_replacement_map",
                    prompt=_get_replacement_mapping_prompt(
                        entities_column="_entities_for_replace",
                        instructions=instructions,
                    ),
                    model_alias=replace_alias,
                    output_format=ReplacementMap,
                )
            ],
            workflow_name="replace-map-generation",
            preview_num_records=preview_num_records,
        )
        output_df = run_result.dataframe.copy()
        output_df.attrs = {**run_result.dataframe.attrs, **dataframe.attrs}
        return LlmReplaceResult(dataframe=output_df, failed_records=run_result.failed_records)


def _normalize_entities_by_value(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    normalized: list[dict[str, Any]] = []
    for entity in raw:
        if not isinstance(entity, dict):
            continue
        enriched = {**entity}
        labels = enriched.get("labels", [])
        enriched["labels_str"] = ", ".join(str(label) for label in labels) if isinstance(labels, list) else ""
        normalized.append(enriched)
    return normalized


def _create_entity_examples(entities_by_value: Any) -> str:
    normalized = _normalize_entities_by_value(entities_by_value)
    labels: set[str] = set()
    for entity in normalized:
        labels.update(str(label) for label in entity.get("labels", []) if str(label))
    if not labels:
        return ""
    examples = {
        label: _EXAMPLE_LOOKUP.get(label, "(generate realistic synthetic replacement)") for label in sorted(labels)
    }
    return json.dumps(examples, ensure_ascii=True)


def _get_replacement_mapping_prompt(*, entities_column: str, instructions: str | None = None) -> str:
    instruction_block = "\nAdditional instructions: %s\n" % instructions if instructions else ""
    return """Generate synthetic replacements for sensitive entities. ONE value per entity, used consistently.
%s
Context: {{ tagged_text }}

Entities to replace:
{%% for entity in %s %%}
- "{{ entity.value }}" ({{ entity.labels_str }})
{%% endfor %%}

Examples: {{ _entity_examples }}

Rules:
1. Related entities must stay consistent:
   - Geographic: city/state/zip must match (Portland→Austin, Oregon→Texas, 97201→78701)
   - Personal: name/email must match (Sarah Chen→Michael Torres, sarah.chen@x.com→michael.torres@x.com)
   - Organizational: company/domain must match (Acme Corp→TechStart, acme.com→techstart.com)
   - Temporal: age/birthdate must match (DOB 1989-05-15→1985-03-20, age 35→39)
   - Contact: phone country code/country must match (+1→+44, USA→UK)

2. Preserve wildcards and patterns:
   - CHANGE concrete parts, KEEP wildcards (* %% ?) in same positions
   - "10.0.*.*" → "192.168.*.*" (changed 10.0, kept *.*)
   - "file_*.log" → "output_*.log" (changed file, kept *)
   - "user_%%@%%.com" → "person_%%@%%.net" (changed user/com, kept %%@%%)
   - DON'T return original unchanged! Change the non-wildcard parts

3. Maintain format and type:
   - Same structure, length patterns, character types
   - Same demographic characteristics (Indian name → Indian name)

CRITICAL: Every entity MUST have a different synthetic value. Never return original=synthetic.
""" % (
        instruction_block,
        entities_column,
    )


_EXAMPLE_LOOKUP: dict[str, str] = {
    "certificate_license_number": "(e.g. ENG-TX-20240513, A9825473, ENG-NY-20230514, TX-97324856)",
    "first_name": "(e.g. Michael, John, Ethan, Isabella)",
    "date_of_birth": "(e.g. 1986-12-29, 1991-12-05, 1990-01-01, 1988-03-02)",
    "ssn": "(e.g. 007-52-4910, 252-96-0016, 523-25-1554, 228-94-9430)",
    "medical_record_number": "(e.g. MRN-345672, BOS-00025836, MRN-567234, 00058362)",
    "password": "(e.g. Rainbow@2025, Michael1995, River@2025, River2025!)",
    "unique_id": "(e.g. 2e008d4415b57d036b51, d2b796a8-161f-4d0c-b3e5-2c9f8a1b3c92, 987654321)",
    "phone_number": "(e.g. 949-307-5488, 251-542-6421, (212) 555-7890, +254 712 345 678)",
    "national_id": "(e.g. 128456189092325, 45789-0123-456, 12545638935, JQ 12 54 26 7)",
    "swift_bic": "(e.g. QWERTUS45ZYX, VXTRUS7K, ZWYTUYMC82B, XPLAAU6RZ)",
    "company_name": "(e.g. VerdantBio, Hartford Construction Group, MediaPulse, Lumina Entertainment)",
    "country": "(e.g. USA, United States, Russia, United Kingdom)",
    "license_plate": "(e.g. JXK-732, KTP-9837, IRB 5721, D SZ 5814)",
    "tax_id": "(e.g. 489-32-1765, 16-3189372, 781534867390, 153-25-6709)",
    "employee_id": "(e.g. MK4567, 21MKT347Z, SM345)",
    "pin": "(e.g. 358495, 1634, 248593, 9404)",
    "state": "(e.g. Texas, CA, Gyeonggi, Krasnodar Krai)",
    "email": "(e.g. derez_lester94@icloud.com, clayton.burke@hotmail.com, amina.e@sudanlinklogistics.com)",
    "date_time": "(e.g. 2023-07-31T16:34:56, 2024-08-08T10:21:02, 2023-09-23T07:34:40)",
    "api_key": "(e.g. a1b2c3d4-e5f6-78g9-h0i1-j2k3l4m5n6o7, zT6mK2pLw8nF4rGjQ9dC1bV7aX0eY5kP)",
    "biometric_identifier": "(e.g. BIO-5739126845, M49283715672, BIO-782654913)",
    "credit_debit_card": "(e.g. 4920 1254 5278 9812, 5412 3656 9820 1634, 5123 3587 8301 2745)",
    "coordinate": "(e.g. 47.6062, -122.3321; 36.7783, -119.4179; 51.207812, 4.429671)",
    "device_identifier": "(e.g. a1b2c3d4e5f6g7h8, 302450982654821, 490154203237518)",
    "city": "(e.g. Houston, Brooklyn, Doha, Lahore)",
    "postcode": "(e.g. 77450, 11201, 452001, 50630-100)",
    "bank_routing_number": "(e.g. 061102356, 801232597, 012745278)",
    "vehicle_identifier": "(e.g. WBA4J52K9MJ129456, VSK5G71F34R000153, SCF4K3L5J9M212645)",
    "health_plan_beneficiary_number": "(e.g. AET-7820-1264-15, FL123496719, WA-0003284668)",
    "url": "(e.g. https://shopify.com, https://marriott.com, https://bestbuy.com?auth=6589)",
    "ipv4": "(e.g. 192.168.1.1, 195.150.21.234, 157.200.130.19, 194.126.23.77)",
    "last_name": "(e.g. Smith, Williams, McKenzie, Hargreaves)",
    "time": "(e.g. 7:23 AM, 18:30, 18:23:45, 10h30)",
    "cvv": "(e.g. 161, 884, 447, 760)",
    "customer_id": "(e.g. CUS439012, SM-19382, 192837425, SM-78321)",
    "date": "(e.g. 07/15/2024, 2023-09-15, 2023-11-15, 15/07/2024)",
    "user_name": "(e.g. jeffreymoon87, stacy.flynn, e.sullivan, Henry1985)",
    "street_address": "(e.g. 661 NE Regents Dr, 739 Main St, 4/87 Collins Street, 22 Boulevard Haussmann)",
    "ipv6": "(e.g. 2001:0db8:85a3:0000:0000:8a2e:0370:7334, 2a02:4d60:1031::85e1:7341:9203:4c56)",
    "account_number": "(e.g. FR72-1534-5678-9082-3156-28, 230915-872513, 125456289)",
    "age": "(e.g. 76, 51, 75, 41)",
    "fax_number": "(e.g. 326-316-9410, 201-565-6897, (512) 876-4321, (212) 555-6789)",
    "county": "(e.g. Los Angeles County, Harris County, Maricopa County, Clark County)",
    "gender": "(e.g. female, male, transgender, non-binary)",
    "sexuality": "(e.g. straight, gay, lesbian, bisexual)",
    "political_view": "(e.g. Republican, Democrat, Blue Dog Democrat, Tea Party)",
    "race_ethnicity": "(e.g. white, black, Korean, Italian)",
    "religious_belief": "(e.g. Christian, Protestant, Church of England, Russian Orthodoxy)",
    "language": "(e.g. English, Spanish, French, German)",
    "blood_type": "(e.g. A+, B+, O positive, A positive)",
    "mac_address": "(e.g. 49:FD:EE:1A:3B:7C, 23:14:B5:67:89:AB, A9:A5:CC:12:54:56)",
    "http_cookie": "(e.g. jwt_token=...; Path=/auth)",
    "employment_status": "(e.g. full-time, part-time, self-employed, contractor)",
    "education_level": "(e.g. high school, bachelor's degree, some college, graduate level)",
    "occupation": "(e.g. registered nurse, truck driver, customer service representative, retail salesperson)",
}
