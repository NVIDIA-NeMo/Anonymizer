# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
from data_designer.config.models import ModelConfig

import anonymizer.engine.execution.phase8_ndd_backend as phase8_backend
from anonymizer.config.anonymizer_config import AnonymizerConfig, Rewrite
from anonymizer.config.models import ModelSelection
from anonymizer.engine.execution.invocation import _CompiledInvocation
from anonymizer.engine.execution.phase8_cleanup import (
    _issue_phase8_cleanup_receipt,
    _Phase8CleanupComponent,
    _Phase8CleanupPhase,
    _Phase8CleanupStatus,
)
from anonymizer.engine.execution.phase8_ndd_backend import (
    _compile_phase8_capability,
    _operation_column,
    _Phase8DispatchResult,
    _Phase8NddBackend,
    _Phase8Operation,
    _snapshot_phase8_capability,
)
from anonymizer.engine.execution.phase8_service import _Phase8CapabilityGuard
from anonymizer.engine.ndd.adapter import NddAdapter


def _invocation(selection: ModelSelection, configs: list[ModelConfig]) -> _CompiledInvocation:
    return _CompiledInvocation.compile(
        AnonymizerConfig(rewrite=Rewrite(strict_entity_protection=True), emit_telemetry=False),
        selection,
        configs,
    )


def test_phase8_capability_freezes_retention_roles_models_limits_and_prompts(
    stub_slim_model_selection: ModelSelection,
    stub_known_model_configs: list[ModelConfig],
) -> None:
    capability = _compile_phase8_capability(_invocation(stub_slim_model_selection, stub_known_model_configs))

    assert capability is not None
    assert capability.version == "phase8-grouped-rewrite-capability/v1"
    assert capability.profile == "anonymizer-phase8-grouped-rewrite/v1"
    assert capability.workframe_schema == "phase8-group-workframe/v1"
    assert capability.retention == "retention_disabled"
    assert tuple(role for role, _alias, _digest in capability.model_roles) == (
        "disposition_analyzer",
        "rewriter",
        "evaluator",
        "repairer",
    )
    assert len({digest for _role, _alias, digest in capability.model_roles}) == 1
    assert dict(capability.limits)["max_repair_iterations"] == 3
    assert len(capability.prompt_contract_digest) == 64


def test_phase8_capability_rejects_missing_model_alias_or_duplicate_model_config(
    stub_slim_model_selection: ModelSelection,
    stub_known_model_configs: list[ModelConfig],
) -> None:
    invocation = _invocation(stub_slim_model_selection, stub_known_model_configs)
    missing = invocation.selected_models.model_copy(deep=True)
    missing.rewrite.repairer = "missing"

    assert _compile_phase8_capability(replace(invocation, selected_models=missing)) is None
    assert (
        _compile_phase8_capability(
            replace(invocation, model_configs=(*invocation.model_configs, invocation.model_configs[0]))
        )
        is None
    )


def test_phase8_operation_routes_use_exact_roles_without_alias_fallback(
    stub_slim_model_selection: ModelSelection,
) -> None:
    selection = stub_slim_model_selection.model_copy(deep=True)
    routes = {
        _Phase8Operation.ANALYZE: "disposition_analyzer",
        _Phase8Operation.REWRITE: "rewriter",
        _Phase8Operation.EVALUATE: "evaluator",
        _Phase8Operation.REPAIR: "repairer",
    }
    configs = []
    for role in routes.values():
        alias = f"alias-{role}"
        setattr(selection.rewrite, role, alias)
        configs.append(ModelConfig(alias=alias, model=f"model-{role}", provider="stub"))
    invocation = _invocation(selection, configs)

    assert {operation: _operation_column(operation, invocation)[0].model_alias for operation in _Phase8Operation} == {
        operation: f"alias-{role}" for operation, role in routes.items()
    }


def test_phase8_capability_checks_compile_open_every_dispatch_and_close(
    stub_slim_model_selection: ModelSelection,
    stub_known_model_configs: list[ModelConfig],
) -> None:
    invocation = _invocation(stub_slim_model_selection, stub_known_model_configs)
    compiled = _compile_phase8_capability(invocation)
    assert compiled is not None
    backend = _CountingCapabilityBackend(invocation)

    assert _snapshot_phase8_capability(backend, invocation) == compiled
    guard = _Phase8CapabilityGuard(backend, invocation, compiled)
    assert not cast(_Phase8DispatchResult, guard.run_operation(_Phase8Operation.ANALYZE, {})).failed
    assert not cast(_Phase8DispatchResult, guard.run_operation(_Phase8Operation.REWRITE, {})).failed
    receipt = guard.retire_phase8(object())

    assert backend.snapshots == 4
    assert getattr(receipt, "status", None) is _Phase8CleanupStatus.VERIFIED
    assert guard.backend is None and guard.invocation is None and guard.expected is None


@pytest.mark.parametrize("drift", ["retention_unknown", "retention_enabled", "prompt", "model", "role"])
def test_phase8_capability_drift_fails_before_dispatch_and_makes_close_unconfirmed(
    drift: str,
    stub_slim_model_selection: ModelSelection,
    stub_known_model_configs: list[ModelConfig],
) -> None:
    invocation = _invocation(stub_slim_model_selection, stub_known_model_configs)
    compiled = _compile_phase8_capability(invocation)
    assert compiled is not None
    backend = _CountingCapabilityBackend(invocation, drift=drift)
    guard = _Phase8CapabilityGuard(backend, invocation, compiled)

    result = cast(_Phase8DispatchResult, guard.run_operation(_Phase8Operation.ANALYZE, {}))
    receipt = guard.retire_phase8(object())

    assert result.failed and result.failure_kind == "invocation_inconsistent"
    assert backend.dispatches == 0
    assert getattr(receipt, "status", None) is _Phase8CleanupStatus.UNCONFIRMED


def test_phase8_ndd_backend_detects_prompt_or_model_drift_before_adapter_call(
    monkeypatch: pytest.MonkeyPatch,
    stub_slim_model_selection: ModelSelection,
    stub_known_model_configs: list[ModelConfig],
) -> None:
    invocation = _invocation(stub_slim_model_selection, stub_known_model_configs)
    adapter = _AdapterSpy()
    backend = _Phase8NddBackend(cast(NddAdapter, adapter), invocation)
    monkeypatch.setitem(phase8_backend._PROMPTS, "analyze", "changed")

    result = backend.run_operation(_Phase8Operation.ANALYZE, {})

    assert result.failed and result.failure_kind == "invocation_inconsistent"
    assert adapter.calls == 0


def test_oversized_phase8_workframe_fails_locally_without_adapter_call(
    stub_slim_model_selection: ModelSelection,
    stub_known_model_configs: list[ModelConfig],
) -> None:
    invocation = _invocation(stub_slim_model_selection, stub_known_model_configs)
    capability = _compile_phase8_capability(invocation)
    assert capability is not None
    adapter = _AdapterSpy()
    backend = _Phase8NddBackend(cast(NddAdapter, adapter), invocation)
    oversized = "x" * (dict(capability.limits)["max_workframe_utf8_bytes_per_operation"] + 1)

    result = backend.run_operation(_Phase8Operation.ANALYZE, {"payload": oversized})

    assert result.failed and result.failure_kind == "local_failure"
    assert adapter.calls == 0


def test_phase8_ndd_dispatch_calls_only_adapter_run_workflow(
    stub_slim_model_selection: ModelSelection,
    stub_known_model_configs: list[ModelConfig],
) -> None:
    invocation = _invocation(stub_slim_model_selection, stub_known_model_configs)
    adapter = _AdapterSpy()
    backend = _Phase8NddBackend(cast(NddAdapter, adapter), invocation)

    result = backend.run_operation(_Phase8Operation.ANALYZE, {})

    assert result.failed and result.failure_kind == "invocation_inconsistent"
    assert adapter.calls == 1


def test_phase8_provider_execution_has_one_adapter_boundary_and_no_direct_datadesigner_call() -> None:
    paths = tuple(Path("src/anonymizer/engine/execution").glob("phase8_*.py"))
    calls: list[str] = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                function = node.func
                if isinstance(function, ast.Attribute):
                    calls.append(function.attr)

    assert calls.count("run_workflow") == 1
    assert "create" not in calls
    assert "preview" not in calls


class _AdapterSpy:
    def __init__(self) -> None:
        self.calls = 0

    def private_execution(self):
        return nullcontext()

    def run_workflow(self, *_args: object, **_kwargs: object) -> object:
        self.calls += 1
        return object()


class _CountingCapabilityBackend:
    def __init__(self, invocation: _CompiledInvocation, *, drift: str | None = None) -> None:
        self.invocation = invocation
        self.drift = drift
        self.snapshots = 0
        self.dispatches = 0

    def phase8_capability(self, _invocation: object) -> object:
        self.snapshots += 1
        capability = _compile_phase8_capability(self.invocation)
        assert capability is not None
        if self.drift == "retention_unknown":
            return replace(capability, retention="retention_unknown")
        if self.drift == "retention_enabled":
            return replace(capability, retention="retention_enabled")
        if self.drift == "prompt":
            return replace(capability, prompt_contract_digest="0" * 64)
        if self.drift == "model":
            role, alias, _digest = capability.model_roles[0]
            return replace(capability, model_roles=((role, alias, "0" * 64), *capability.model_roles[1:]))
        if self.drift == "role":
            _role, alias, digest = capability.model_roles[0]
            return replace(capability, model_roles=(("repairer", alias, digest), *capability.model_roles[1:]))
        return capability

    def run_operation(self, operation: _Phase8Operation, _request: dict[str, object]) -> object:
        self.dispatches += 1
        return _Phase8DispatchResult(operation, None)

    def retire_phase8(self, identity: object) -> object:
        return _issue_phase8_cleanup_receipt(
            _Phase8CleanupPhase.PRE_REDUCTION,
            _Phase8CleanupComponent.BACKEND,
            _Phase8CleanupStatus.VERIFIED,
            identity,
        )
