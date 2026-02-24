# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations


def _jinja(col: str) -> str:
    """Wrap a column constant in Jinja2 template syntax: ``{{ col }}``."""
    return "{{ " + col + " }}"


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
