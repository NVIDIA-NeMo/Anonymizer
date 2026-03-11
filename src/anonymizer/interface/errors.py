# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations


class AnonymizerError(Exception):
    """Base error for Anonymizer interface operations."""


class InvalidInputError(AnonymizerError):
    """Raised when input data or configuration is invalid."""


class InvalidConfigError(AnonymizerError):
    """Raised when model aliases or semantic configuration are invalid."""


class AnonymizerIOError(AnonymizerError):
    """Raised when file IO operations fail."""
