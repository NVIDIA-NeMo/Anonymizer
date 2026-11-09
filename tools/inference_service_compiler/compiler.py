# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure compilation from inference spec to an immutable run plan."""

from __future__ import annotations

import hashlib
import hmac
import json
from typing import Never, assert_never

from inference_service_compiler.models import (
    CommandSpec,
    EndpointAddress,
    EntityDetection,
    FrozenModel,
    Generation,
    LiteralArgument,
    LocalInferenceServiceSpec,
    ReadinessCheck,
    RunPlan,
)
from inference_service_compiler.vllm_factory_integration import (
    io_processor_for,
    supports_model,
)


class CompilerDiagnostic(FrozenModel):
    """Serializable reason that semantic spec cannot be compiled."""

    code: str
    message: str
    details: dict[str, str]


class CompilationError(ValueError):
    """A service spec failed a closed compiler compatibility rule."""

    def __init__(self, diagnostic: CompilerDiagnostic) -> None:
        super().__init__(diagnostic.message)
        self.diagnostic = diagnostic


class PlanIntegrityError(ValueError):
    """A serialized plan does not match its declared consistency checksum."""


def compile_profile(spec: LocalInferenceServiceSpec) -> RunPlan:
    """Compile semantic spec without starting, probing, or allocating anything."""
    required = spec.task.required_capabilities()
    command = _compile_vllm(spec)
    placement = spec.local
    endpoint = EndpointAddress(host=placement.host, port=placement.port)
    plan = RunPlan(
        plan_digest="",
        spec=spec,
        command=command,
        endpoint=endpoint,
        readiness=ReadinessCheck(
            path="/models",
            timeout_seconds=spec.local.startup_timeout_seconds,
            bearer_token_environment_variable=spec.vllm.api_key_env,
        ),
        served_model_name=spec.served_model_name,
        required_capabilities=required,
    )
    return plan.model_copy(update={"plan_digest": digest_plan(plan)})


def digest_plan(plan: RunPlan) -> str:
    """Return the stable digest of a plan excluding its digest field."""
    payload = plan.model_dump(mode="json", exclude={"plan_digest"})
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def load_plan(serialized: str | bytes) -> RunPlan:
    """Parse a closed v2 plan and reject accidental transport mutation."""
    plan = RunPlan.model_validate_json(serialized)
    verify_plan(plan)
    return plan


def verify_plan(plan: RunPlan) -> None:
    """Check transport consistency, without authenticating or recompiling a plan."""
    expected = digest_plan(plan)
    if not hmac.compare_digest(plan.plan_digest, expected):
        raise PlanIntegrityError(f"plan digest mismatch: declared {plan.plan_digest!r}, computed {expected!r}")


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _compile_vllm(spec: LocalInferenceServiceSpec) -> CommandSpec:
    match spec.task:
        case EntityDetection() as task:
            _validate_factory_detection(spec, task)
        case Generation() as task:
            if spec.vllm.factory is not None:
                _raise_unsupported_task_engine(task.kind, "vllm")
        case _:
            assert_never(spec.task)
    config = {key.replace("_", "-"): value for key, value in spec.vllm.config.items()}
    if len(config) != len(spec.vllm.config):
        raise CompilationError(
            CompilerDiagnostic(
                code="duplicate-vllm-option", message="vLLM options contain duplicate spellings", details={}
            )
        )
    if spec.vllm.factory is not None:
        required = {"runner": "pooling", "io-processor-plugin": io_processor_for(spec.vllm.factory.plugin)}
        for key, value in required.items():
            if config.get(key) != value:
                raise CompilationError(
                    CompilerDiagnostic(
                        code="invalid-factory-option",
                        message=f"Factory requires {key} = {value!r} in vllm.config",
                        details={},
                    )
                )
    reserved = {
        "model",
        "host",
        "port",
        "revision",
        "tokenizer-revision",
        "served-model-name",
        "api-key",
        "hf-token",
        "config",
    }
    conflict = reserved.intersection(key.replace("_", "-") for key in config)
    if conflict:
        raise CompilationError(
            CompilerDiagnostic(
                code="reserved-vllm-option",
                message=f"vLLM options are owned by the profile: {sorted(conflict)}",
                details={},
            )
        )
    config.update({"model": spec.model.model_id, "host": spec.local.host, "port": spec.local.port})
    if spec.model.revision is not None:
        config.update({"revision": spec.model.revision, "tokenizer-revision": spec.model.revision})
    if spec.vllm.served_model_name is not None:
        config["served-model-name"] = [spec.vllm.served_model_name]
    if spec.model.adapter is not None:
        if {"enable-lora", "lora-modules"}.intersection(config):
            raise CompilationError(
                CompilerDiagnostic(
                    code="reserved-vllm-option", message="LoRA settings are owned by model.adapter", details={}
                )
            )
        config.update({"enable-lora": True, "lora-modules": [f"{spec.model.adapter.name}={spec.model.adapter.path}"]})
    argv = [
        spec.vllm.python_executable,
        "tools/inference_service_compiler/vllm_server.py",
        "--config-json",
        json.dumps(config, sort_keys=True),
    ]
    if spec.vllm.factory is not None:
        argv.extend(
            [
                "--vllm-factory-plugin",
                spec.vllm.factory.plugin,
                "--prepared-model-root",
                spec.vllm.factory.prepared_model_root,
            ]
        )
    return CommandSpec(argv=tuple(LiteralArgument(value=value) for value in argv))


def _validate_factory_detection(spec: LocalInferenceServiceSpec, task: EntityDetection) -> None:
    factory = spec.vllm.factory
    if factory is None:
        _raise_unsupported_task_engine(task.kind, "vllm")
    if spec.model.revision is None:
        raise CompilationError(
            CompilerDiagnostic(
                code="unpinned-model-revision",
                message="vLLM Factory entity detection requires a pinned model revision",
                details={"model": spec.model.model_id},
            )
        )
    if not supports_model(factory.plugin, spec.model.model_id):
        raise CompilationError(
            CompilerDiagnostic(
                code="unsupported-model-engine",
                message=(
                    f"model {spec.model.model_id!r} is not characterized for vLLM Factory plugin {factory.plugin!r}"
                ),
                details={"model": spec.model.model_id, "plugin": factory.plugin},
            )
        )
    if spec.model.adapter is not None:
        raise CompilationError(
            CompilerDiagnostic(
                code="unsupported-model-adapter",
                message="vLLM Factory entity detection does not support a model adapter",
                details={"engine": "vllm", "task": task.kind},
            )
        )


def _raise_unsupported_task_engine(task: str, engine: str) -> Never:
    raise CompilationError(
        CompilerDiagnostic(
            code="unsupported-task-engine",
            message=f"engine {engine!r} does not support task {task!r}",
            details={"engine": engine, "task": task},
        )
    )
