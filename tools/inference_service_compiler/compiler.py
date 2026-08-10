# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure compilation from inference intent to an immutable run plan."""

from __future__ import annotations

import hashlib
import hmac
import json

from pydantic import BaseModel

from inference_service_compiler.models import (
    Capability,
    CommandArgument,
    CommandSpec,
    CompatibilityEvidence,
    DockerPlacement,
    DockerRuntime,
    EndpointContract,
    EntityDetection,
    EnvironmentVariable,
    FrozenModel,
    Generation,
    HttpProbe,
    InferenceIntent,
    LiteralArgument,
    LocalProcessPlacement,
    LocalProcessRuntime,
    NativeGlinerEngine,
    RunPlan,
    SecretEnvironmentVariable,
    VllmEngine,
)

VLLM_API_KEY_ENV = "VLLM_API_KEY"


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


def compile_intent(intent: InferenceIntent, *, source_revision: str) -> RunPlan:
    """Compile semantic intent without starting, probing, or allocating anything."""
    if not source_revision:
        raise ValueError("source_revision must not be empty")
    required = intent.task.required_capabilities()
    command, runtime, declared, evidence = _compile_service(intent)
    placement = intent.placement
    endpoint = EndpointContract(host=placement.host, port=placement.port)
    plan = RunPlan(
        plan_digest="",
        intent_digest=digest_model(intent),
        intent=intent,
        command=command,
        runtime=runtime,
        endpoint=endpoint,
        readiness=HttpProbe(
            host=placement.host,
            port=placement.port,
            path="/v1/models",
            timeout_seconds=intent.lifecycle.startup_timeout_seconds,
            bearer_token_environment_variable=(
                intent.engine.api_key_env if isinstance(intent.engine, VllmEngine) else None
            ),
        ),
        expected_model=intent.expected_model,
        required_capabilities=required,
        declared_capabilities=declared,
        compatibility_evidence=evidence,
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
    """Parse a closed v1 plan and reject transport mutation."""
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


def _compile_service(
    intent: InferenceIntent,
) -> tuple[
    CommandSpec,
    LocalProcessRuntime | DockerRuntime,
    tuple[Capability, ...],
    tuple[CompatibilityEvidence, ...],
]:
    match intent.engine:
        case NativeGlinerEngine() as engine:
            if not isinstance(intent.task, EntityDetection):
                _raise_unsupported_task_engine(intent.task.kind, engine.kind)
            if not isinstance(intent.placement, LocalProcessPlacement):
                raise CompilationError(
                    CompilerDiagnostic(
                        code="unsupported-engine-placement",
                        message="native GLiNER is characterized only as a local process",
                        details={"engine": engine.kind, "placement": intent.placement.kind},
                    )
                )
            return _compile_native_gliner(intent, engine)
        case VllmEngine() as engine:
            return _compile_vllm(intent, engine)


def _compile_native_gliner(
    intent: InferenceIntent,
    engine: NativeGlinerEngine,
) -> tuple[CommandSpec, LocalProcessRuntime, tuple[Capability, ...], tuple[CompatibilityEvidence, ...]]:
    placement = intent.placement
    if not isinstance(placement, LocalProcessPlacement):
        raise TypeError(f"expected LocalProcessPlacement, got {type(placement)!r}")
    argv = _literal_arguments(
        "uv",
        "run",
        "--script",
        "tools/inference_service_compiler/native_gliner.py",
        "--host",
        placement.host,
        "--port",
        str(placement.port),
        "--model",
        engine.family,
        "--checkpoint",
        intent.model.model_id,
    )
    if engine.log_format != "plain":
        argv += _literal_arguments("--log-format", engine.log_format)
    if intent.model.revision is not None:
        argv += _literal_arguments("--revision", intent.model.revision)
    return (
        CommandSpec(argv=argv, environment=_native_environment(engine)),
        LocalProcessRuntime(),
        intent.task.required_capabilities(),
        (
            CompatibilityEvidence(
                rule="native-gliner-entity-detection-v1",
                outcome="characterized",
                detail="Anonymizer chat-completion adapter preserves dynamic labels, offsets, and scores",
            ),
        ),
    )


def _native_environment(engine: NativeGlinerEngine) -> tuple[EnvironmentVariable, ...]:
    return (
        EnvironmentVariable(name="DEVICE", value=engine.device),
        EnvironmentVariable(name="GLINER_BATCH_MODE", value=str(engine.batch_mode).lower()),
        EnvironmentVariable(name="GLINER_MAX_BATCH_REQUESTS", value=str(engine.max_batch_requests)),
        EnvironmentVariable(name="GLINER_BATCH_WAIT_MS", value=str(float(engine.batch_wait_ms))),
    )


def _compile_vllm(
    intent: InferenceIntent,
    engine: VllmEngine,
) -> tuple[
    CommandSpec,
    LocalProcessRuntime | DockerRuntime,
    tuple[Capability, ...],
    tuple[CompatibilityEvidence, ...],
]:
    command, runtime = _vllm_command(intent, engine)
    declared = ("chat-completions",)
    outcome = "characterized" if isinstance(intent.task, Generation) else "runtime-probe-required"
    evidence = (
        CompatibilityEvidence(
            rule="vllm-openai-compatible-v1",
            outcome=outcome,
            detail=(
                "vLLM exposes chat completions for generation"
                if outcome == "characterized"
                else "entity-detection capabilities must be established by the run probe"
            ),
        ),
    )
    return command, runtime, declared, evidence


def _vllm_command(
    intent: InferenceIntent,
    engine: VllmEngine,
) -> tuple[CommandSpec, LocalProcessRuntime | DockerRuntime]:
    engine_arguments = _vllm_engine_arguments(intent, engine)
    match intent.placement:
        case LocalProcessPlacement() as placement:
            argv = _literal_arguments(
                engine.executable,
                "serve",
                intent.model.model_id,
                "--host",
                placement.host,
                "--port",
                str(placement.port),
            )
            return CommandSpec(
                argv=argv + engine_arguments, environment=_vllm_environment(engine)
            ), LocalProcessRuntime()
        case DockerPlacement() as placement:
            return _docker_vllm_command(intent, engine, placement, engine_arguments)


def _docker_vllm_command(
    intent: InferenceIntent,
    engine: VllmEngine,
    placement: DockerPlacement,
    engine_arguments: tuple[CommandArgument, ...],
) -> tuple[CommandSpec, DockerRuntime]:
    values = [
        placement.runtime,
        "run",
        "--detach",
        "--rm",
        "--gpus",
        placement.gpus,
        "--ipc",
        "host",
        "--publish",
        f"{placement.host}:{placement.port}:8000",
    ]
    if placement.hugging_face_cache is not None:
        values.extend(["--volume", f"{placement.hugging_face_cache}:/root/.cache/huggingface"])
    if engine.api_key_env is not None:
        values.extend(["--env", VLLM_API_KEY_ENV])
    values.extend([placement.image, "--model", intent.model.model_id])
    return (
        CommandSpec(
            argv=_literal_arguments(*values) + engine_arguments,
            environment=_vllm_environment(engine),
        ),
        DockerRuntime(image=placement.image),
    )


def _vllm_engine_arguments(intent: InferenceIntent, engine: VllmEngine) -> tuple[CommandArgument, ...]:
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
    if engine.eager:
        arguments.extend(_literal_arguments("--enforce-eager"))
    if intent.model.adapter is not None:
        arguments.extend(
            _literal_arguments(
                "--enable-lora",
                "--lora-modules",
                f"{intent.model.adapter.name}={intent.model.adapter.path}",
            )
        )
    return tuple(arguments)


def _vllm_environment(engine: VllmEngine) -> tuple[SecretEnvironmentVariable, ...]:
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


def _raise_unsupported_task_engine(task: str, engine: str) -> None:
    raise CompilationError(
        CompilerDiagnostic(
            code="unsupported-task-engine",
            message=f"engine {engine!r} does not support task {task!r}",
            details={"engine": engine, "task": task},
        )
    )
