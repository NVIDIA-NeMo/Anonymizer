# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure compilation from inference intent to an immutable run plan."""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass
from typing import Never, assert_never

from pydantic import BaseModel

from inference_service_compiler.models import (
    Capability,
    CommandArgument,
    CommandSpec,
    CompatibilityEvidence,
    EndpointContract,
    EntityDetection,
    FrozenModel,
    Generation,
    HttpProbe,
    InferenceIntent,
    LiteralArgument,
    LocalProcessRuntime,
    RunPlan,
    SecretEnvironmentVariable,
    Vllm,
)
from inference_service_compiler.vllm_factory_integration import (
    VLLM_FACTORY_DEPENDENCY,
    supports_model,
)

VLLM_API_KEY_ENV = "VLLM_API_KEY"
VLLM_DEPENDENCY = "vllm==0.27.1"
FLASHINFER_CUDA_TOOLCHAIN_DEPENDENCIES = (
    "nvidia-cuda-nvcc==13.0.88",
    "nvidia-cuda-crt==13.0.88",
    "nvidia-nvvm==13.0.88",
)


class CompilerDiagnostic(FrozenModel):
    """Serializable reason that semantic intent cannot be compiled."""

    code: str
    message: str
    details: dict[str, str]


class CompilationError(ValueError):
    """Intent failed a closed compiler compatibility rule."""

    def __init__(self, diagnostic: CompilerDiagnostic) -> None:
        super().__init__(diagnostic.message)
        self.diagnostic = diagnostic


class PlanIntegrityError(ValueError):
    """A serialized plan does not match its declared digest."""


@dataclass(frozen=True, slots=True)
class ServiceCompilation:
    """Complete immutable product of selecting one service implementation."""

    command: CommandSpec
    runtime: LocalProcessRuntime
    declared_capabilities: tuple[Capability, ...]
    compatibility_evidence: tuple[CompatibilityEvidence, ...]


def compile_intent(intent: InferenceIntent, *, source_revision: str) -> RunPlan:
    """Compile semantic intent without starting, probing, or allocating anything."""
    if not source_revision:
        raise ValueError("source_revision must not be empty")
    required = intent.task.required_capabilities()
    compilation = _compile_vllm(intent)
    placement = intent.local
    endpoint = EndpointContract(host=placement.host, port=placement.port)
    plan = RunPlan(
        plan_digest="",
        intent_digest=digest_model(intent),
        intent=intent,
        command=compilation.command,
        runtime=compilation.runtime,
        endpoint=endpoint,
        readiness=HttpProbe(
            host=placement.host,
            port=placement.port,
            path="/v1/models",
            timeout_seconds=intent.local.startup_timeout_seconds,
            bearer_token_environment_variable=intent.vllm.api_key_env,
        ),
        expected_model=intent.expected_model,
        required_capabilities=required,
        declared_capabilities=compilation.declared_capabilities,
        compatibility_evidence=compilation.compatibility_evidence,
        dependencies=_plan_dependencies(intent),
        source_revision=source_revision,
    )
    return plan.model_copy(update={"plan_digest": digest_plan(plan)})


def digest_model(model: BaseModel) -> str:
    """Return a stable SHA-256 digest of one typed transport value."""
    payload = model.model_dump(mode="json")
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def digest_plan(plan: RunPlan) -> str:
    """Return the stable digest of a plan excluding its digest field."""
    payload = plan.model_dump(mode="json", exclude={"plan_digest"})
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def load_plan(serialized: str | bytes) -> RunPlan:
    """Parse a closed v2 plan and reject transport mutation."""
    plan = RunPlan.model_validate_json(serialized)
    verify_plan(plan)
    return plan


def verify_plan(plan: RunPlan) -> None:
    """Reject a plan whose declared digest does not match its contents."""
    expected = digest_plan(plan)
    if not hmac.compare_digest(plan.plan_digest, expected):
        raise PlanIntegrityError(f"plan digest mismatch: declared {plan.plan_digest!r}, computed {expected!r}")


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _compile_vllm(intent: InferenceIntent) -> ServiceCompilation:
    engine = intent.vllm
    match intent.task:
        case EntityDetection() as task:
            factory = engine.factory
            if factory is None:
                _raise_unsupported_task_engine(task.kind, "vllm")
            if intent.model.revision is None:
                raise CompilationError(
                    CompilerDiagnostic(
                        code="unpinned-model-revision",
                        message="vLLM Factory entity detection requires a pinned model revision",
                        details={"model": intent.model.model_id},
                    )
                )
            if not supports_model(factory.plugin, intent.model.model_id):
                raise CompilationError(
                    CompilerDiagnostic(
                        code="unsupported-model-engine",
                        message=(
                            f"model {intent.model.model_id!r} is not characterized for "
                            f"vLLM Factory plugin {factory.plugin!r}"
                        ),
                        details={"model": intent.model.model_id, "plugin": factory.plugin},
                    )
                )
            if intent.model.adapter is not None:
                raise CompilationError(
                    CompilerDiagnostic(
                        code="unsupported-model-adapter",
                        message="vLLM Factory entity detection does not support a model adapter",
                        details={"engine": "vllm", "task": task.kind},
                    )
                )
            command, runtime = _vllm_command(intent, engine)
            return ServiceCompilation(
                command=command,
                runtime=runtime,
                declared_capabilities=task.required_capabilities(),
                compatibility_evidence=(
                    CompatibilityEvidence(
                        rule="vllm-factory-entity-detection-v1",
                        outcome="runtime-probe-required",
                        detail=(
                            "vLLM Factory supplies model preparation, pooling inference, and IO processing; "
                            "the Anonymizer adapter preserves dynamic labels, offsets, and scores"
                        ),
                    ),
                ),
            )
        case Generation() as task:
            if engine.factory is not None:
                _raise_unsupported_task_engine(task.kind, "vllm")
            command, runtime = _vllm_command(intent, engine)
            return ServiceCompilation(
                command=command,
                runtime=runtime,
                declared_capabilities=("chat-completions",),
                compatibility_evidence=(
                    CompatibilityEvidence(
                        rule="vllm-openai-compatible-v1",
                        outcome="characterized",
                        detail="vLLM exposes chat completions for generation",
                    ),
                ),
            )
        case _:
            assert_never(intent.task)


def _vllm_command(
    intent: InferenceIntent,
    engine: Vllm,
) -> tuple[CommandSpec, LocalProcessRuntime]:
    engine_arguments = _vllm_engine_arguments(intent, engine)
    placement = intent.local
    argv = _literal_arguments(
        engine.python_executable,
        "tools/inference_service_compiler/vllm_server.py",
        intent.model.model_id,
        "--host",
        placement.host,
        "--port",
        str(placement.port),
    )
    return CommandSpec(argv=argv + engine_arguments, environment=_vllm_environment(engine)), LocalProcessRuntime()


def _vllm_engine_arguments(intent: InferenceIntent, engine: Vllm) -> tuple[CommandArgument, ...]:
    arguments: list[CommandArgument] = []
    if intent.model.revision is not None:
        arguments.extend(
            _literal_arguments(
                "--revision",
                intent.model.revision,
                "--tokenizer-revision",
                intent.model.revision,
            )
        )
    if engine.served_model_name is not None:
        arguments.extend(_literal_arguments("--served-model-name", engine.served_model_name))
    if engine.tensor_parallel_size is not None:
        arguments.extend(_literal_arguments("--tensor-parallel-size", str(engine.tensor_parallel_size)))
    if engine.gpu_memory_utilization is not None:
        arguments.extend(_literal_arguments("--gpu-memory-utilization", str(engine.gpu_memory_utilization)))
    if engine.max_model_len is not None:
        arguments.extend(_literal_arguments("--max-model-len", str(engine.max_model_len)))
    if engine.max_num_seqs is not None:
        arguments.extend(_literal_arguments("--max-num-seqs", str(engine.max_num_seqs)))
    if engine.eager:
        arguments.extend(_literal_arguments("--enforce-eager"))
    if engine.enable_prefix_caching:
        arguments.extend(_literal_arguments("--enable-prefix-caching"))
    if engine.async_scheduling:
        arguments.extend(_literal_arguments("--async-scheduling"))
    if engine.mamba_backend is not None:
        arguments.extend(_literal_arguments("--mamba-backend", engine.mamba_backend))
    if engine.mamba_ssm_cache_dtype != "auto":
        arguments.extend(_literal_arguments("--mamba-ssm-cache-dtype", engine.mamba_ssm_cache_dtype))
    if engine.enable_mamba_cache_stochastic_rounding:
        arguments.extend(_literal_arguments("--enable-mamba-cache-stochastic-rounding"))
    if engine.mamba_cache_philox_rounds:
        arguments.extend(_literal_arguments("--mamba-cache-philox-rounds", str(engine.mamba_cache_philox_rounds)))
    if engine.factory is not None:
        arguments.extend(
            _literal_arguments(
                "--vllm-factory-plugin",
                engine.factory.plugin,
                "--prepared-model-root",
                engine.factory.prepared_model_root,
            )
        )
    if intent.model.adapter is not None:
        arguments.extend(
            _literal_arguments(
                "--enable-lora",
                "--lora-modules",
                f"{intent.model.adapter.name}={intent.model.adapter.path}",
            )
        )
    return tuple(arguments)


def _vllm_environment(engine: Vllm) -> tuple[SecretEnvironmentVariable, ...]:
    if engine.api_key_env is None:
        return ()
    return (
        SecretEnvironmentVariable(
            name=VLLM_API_KEY_ENV,
            source_environment_variable=engine.api_key_env,
        ),
    )


def _literal_arguments(*values: str) -> tuple[CommandArgument, ...]:
    return tuple(LiteralArgument(value=value) for value in values)


def _plan_dependencies(intent: InferenceIntent) -> tuple[str, ...]:
    dependencies = [VLLM_DEPENDENCY]
    if intent.vllm.factory is not None:
        dependencies.append(VLLM_FACTORY_DEPENDENCY)
    if intent.vllm.mamba_backend == "flashinfer":
        dependencies.extend(FLASHINFER_CUDA_TOOLCHAIN_DEPENDENCIES)
    return tuple(dependencies)


def _raise_unsupported_task_engine(task: str, engine: str) -> Never:
    raise CompilationError(
        CompilerDiagnostic(
            code="unsupported-task-engine",
            message=f"engine {engine!r} does not support task {task!r}",
            details={"engine": engine, "task": task},
        )
    )
