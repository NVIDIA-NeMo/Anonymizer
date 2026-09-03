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
    CompatibilityAssessment,
    EndpointAddress,
    EntityDetection,
    FrozenModel,
    Generation,
    LiteralArgument,
    LocalInferenceServiceSpec,
    ReadinessCheck,
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


def compile_profile(spec: LocalInferenceServiceSpec, *, source_revision: str) -> RunPlan:
    """Compile semantic spec without starting, probing, or allocating anything."""
    if not source_revision:
        raise CompilationError(
            CompilerDiagnostic(
                code="invalid-source-revision",
                message="source_revision must not be empty",
                details={},
            )
        )
    required = spec.task.required_capabilities()
    command, assessments = _compile_vllm(spec)
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
        compatibility_assessments=assessments,
        dependencies=_plan_dependencies(spec),
        source_revision=source_revision,
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


def _compile_vllm(spec: LocalInferenceServiceSpec) -> tuple[CommandSpec, tuple[CompatibilityAssessment, ...]]:
    match spec.task:
        case EntityDetection() as task:
            return _compile_entity_detection(spec, task)
        case Generation() as task:
            return _compile_generation(spec, task)
        case _:
            assert_never(spec.task)


def _compile_entity_detection(
    spec: LocalInferenceServiceSpec, task: EntityDetection
) -> tuple[CommandSpec, tuple[CompatibilityAssessment, ...]]:
    _validate_factory_detection(spec, task)
    command = _vllm_command(spec, spec.vllm)
    return command, (
        CompatibilityAssessment(
            rule="vllm-factory-entity-detection-v1",
            outcome="runtime-probe-required",
            detail=(
                "vLLM Factory supplies model preparation, pooling inference, and IO processing; "
                "the Anonymizer adapter preserves dynamic labels, offsets, and scores"
            ),
        ),
    )


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


def _compile_generation(
    spec: LocalInferenceServiceSpec, task: Generation
) -> tuple[CommandSpec, tuple[CompatibilityAssessment, ...]]:
    if spec.vllm.factory is not None:
        _raise_unsupported_task_engine(task.kind, "vllm")
    command = _vllm_command(spec, spec.vllm)
    return command, (
        CompatibilityAssessment(
            rule="vllm-openai-compatible-v1",
            outcome="characterized",
            detail="vLLM exposes chat completions for generation",
        ),
    )


def _vllm_command(
    spec: LocalInferenceServiceSpec,
    engine: Vllm,
) -> CommandSpec:
    engine_arguments = _vllm_engine_arguments(spec, engine)
    placement = spec.local
    argv = _literal_arguments(
        engine.python_executable,
        "tools/inference_service_compiler/vllm_server.py",
        spec.model.model_id,
        "--host",
        placement.host,
        "--port",
        str(placement.port),
    )
    return CommandSpec(argv=argv + engine_arguments, environment=_vllm_environment(engine))


def _vllm_engine_arguments(spec: LocalInferenceServiceSpec, engine: Vllm) -> tuple[LiteralArgument, ...]:
    arguments: list[LiteralArgument] = []
    if spec.model.revision is not None:
        arguments.extend(
            _literal_arguments(
                "--revision",
                spec.model.revision,
                "--tokenizer-revision",
                spec.model.revision,
            )
        )
    arguments.extend(_optional_vllm_arguments(engine))
    arguments.extend(
        LiteralArgument(value=flag)
        for flag, enabled in (
            ("--enforce-eager", engine.eager),
            ("--enable-prefix-caching", engine.enable_prefix_caching),
            ("--async-scheduling", engine.async_scheduling),
        )
        if enabled
    )
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
    if spec.model.adapter is not None:
        arguments.extend(
            _literal_arguments(
                "--enable-lora",
                "--lora-modules",
                f"{spec.model.adapter.name}={spec.model.adapter.path}",
            )
        )
    return tuple(arguments)


def _optional_vllm_arguments(engine: Vllm) -> tuple[LiteralArgument, ...]:
    values = (
        ("--served-model-name", engine.served_model_name),
        ("--tensor-parallel-size", engine.tensor_parallel_size),
        ("--gpu-memory-utilization", engine.gpu_memory_utilization),
        ("--max-model-len", engine.max_model_len),
        ("--max-num-seqs", engine.max_num_seqs),
    )
    return tuple(
        argument for flag, value in values if value is not None for argument in _literal_arguments(flag, str(value))
    )


def _vllm_environment(engine: Vllm) -> tuple[SecretEnvironmentVariable, ...]:
    if engine.api_key_env is None:
        return ()
    return (
        SecretEnvironmentVariable(
            name=VLLM_API_KEY_ENV,
            source_environment_variable=engine.api_key_env,
        ),
    )


def _literal_arguments(*values: str) -> tuple[LiteralArgument, ...]:
    return tuple(LiteralArgument(value=value) for value in values)


def _plan_dependencies(spec: LocalInferenceServiceSpec) -> tuple[str, ...]:
    dependencies = [VLLM_DEPENDENCY]
    if spec.vllm.factory is not None:
        dependencies.append(VLLM_FACTORY_DEPENDENCY)
    if spec.vllm.mamba_backend == "flashinfer":
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
