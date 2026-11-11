# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

DEFAULT_IGNORED_ENTITY_VALUES: frozenset[str] = frozenset(
    {
        "PayPal",
        "paypal",
        "Visa",
        "visa",
        "Visa card",
        "Mastercard",
        "mastercard",
        "credit_card",
        "credit card",
        "Credit Card",
        "gift_card",
        "gift card",
        "Gift Card",
        "email",
        "email address",
    }
)
