# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Failures the command line reports as bad input or a failed effect."""

from __future__ import annotations

from inference_service_compiler.models import RuntimeDiagnostic, RuntimeErrorCode


class InferenceServiceError(Exception):
    """Base for every failure the command line reports as bad input or a failed effect."""


class PlanIntegrityError(InferenceServiceError, ValueError):
    """A serialized plan does not match its digest or its spec's compilation."""


class RuntimeEffectError(InferenceServiceError, RuntimeError):
    """A launch, probe, or process observation failed with a serializable partial-effects record."""

    def __init__(self, diagnostic: RuntimeDiagnostic) -> None:
        super().__init__(diagnostic.message)
        self.diagnostic = diagnostic

    @classmethod
    def of(
        cls,
        code: RuntimeErrorCode,
        message: str,
        *,
        known_effects: tuple[str, ...] = (),
        cleanup_complete: bool | None = None,
    ) -> RuntimeEffectError:
        """Build the error from its diagnostic fields."""
        return cls(
            RuntimeDiagnostic(
                code=code, message=message, known_effects=known_effects, cleanup_complete=cleanup_complete
            )
        )
