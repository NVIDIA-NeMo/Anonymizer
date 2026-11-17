# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure compilation from inference spec to an immutable run plan."""

from __future__ import annotations

import hashlib
import hmac
import json

from inference_service_compiler.backends import traits
from inference_service_compiler.errors import PlanIntegrityError
from inference_service_compiler.models import RunPlan, ServiceSpec


def compile_spec(spec: ServiceSpec) -> RunPlan:
    """Compile semantic spec without starting, probing, or allocating anything."""
    plan = RunPlan(spec=spec, argv=traits(spec).argv, plan_digest="")
    return plan.model_copy(update={"plan_digest": digest_plan(plan)})


def digest_plan(plan: RunPlan) -> str:
    """Return the stable digest of a plan excluding its digest field."""
    payload = plan.model_dump(mode="json", exclude={"plan_digest"})
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_plan(serialized: str | bytes) -> RunPlan:
    """Parse a v3 plan and reject transport mutation or a stale compilation."""
    plan = RunPlan.model_validate_json(serialized)
    verify_plan(plan)
    return plan


def verify_plan(plan: RunPlan) -> None:
    """Check the digest and that compiling ``plan.spec`` reproduces the plan; does not authenticate it."""
    expected = digest_plan(plan)
    if not hmac.compare_digest(plan.plan_digest, expected):
        raise PlanIntegrityError(f"plan digest mismatch: declared {plan.plan_digest!r}, computed {expected!r}")
    if compile_spec(plan.spec) != plan:
        raise PlanIntegrityError("plan does not match the compilation of its spec; recompile the profile")
