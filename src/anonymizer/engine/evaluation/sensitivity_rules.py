# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared sensitivity guidance for Rewrite and Replace privacy evaluation."""

from __future__ import annotations

SENSITIVITY_RISK_SCOPE = """Assess whether retaining the information could identify or enable linkage to a person, organization, record, system, or other protected asset, or expose confidential information such as credentials, access details, or proprietary information."""

DIRECT_IDENTIFIER_SENSITIVITY_RULES = """Direct identifiers:
Assign high. They must always be protected."""

QUASI_IDENTIFIER_SENSITIVITY_RULES = """Generic quasi-identifiers:
Assign low unless the document supplies distinguishing context that makes
this information materially useful for linkage or recognition.

Generic information describes an unnamed role, institution type, activity,
setting, or unspecified fact without distinguishing a particular instance.
Examples include "bank", "court", "lawyer", "hospital", "local library",
"community food pantry", and "unspecified date".

An entity label, capitalization, or the presence of identifying information
elsewhere is not sufficient reason to assign medium. Formal references such
as "Court" or "Government" may still be generic.
"The only surgeon in the village", however, has distinguishing context.

Specific quasi-identifiers:
Assign medium unless retaining the information adds no meaningful linkage
or recognition risk in this document.

Specific information expresses a named affiliation or location, a concrete
attribute, or a precise fact—for example, a named employer, city, nationality,
language, exact age, date, or amount. Proper nouns often indicate specificity,
but capitalization alone does not.

Specific does not automatically mean identifying. For example, "English"
may qualify for low when it adds no meaningful re-identification risk."""

SENSITIVITY_CONTEXT_RULES = """Do not assume another planned edit makes information safe.
Do not equate commonness with safety or require unique identification to
establish risk. Do not invent unsupported identification paths."""
