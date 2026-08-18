# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations


class AnonymizerError(Exception):
    """Base error for Anonymizer interface operations."""


class InvalidInputError(AnonymizerError):
    """Raised when input data or configuration is invalid."""


class InvalidConfigError(AnonymizerError):
    """Raised when model, provider, alias, or semantic configuration is invalid."""


class AnonymizerIOError(AnonymizerError):
    """Raised when file IO operations fail."""


class AnonymizerWorkflowError(AnonymizerError):
    """Raised when an underlying workflow step (preview, execution, or dataset load) fails.

    Internal boundaries may preserve a backend exception as ``__cause__`` for
    diagnostics. Public privacy-sensitive operations such as ``run()`` and
    ``preview()`` deliberately suppress causes and use a generic message so
    backend details, row correlations, and input values cannot escape.
    """
