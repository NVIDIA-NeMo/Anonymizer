# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Versioned transport and immutable intermediate-representation models."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

INTENT_SCHEMA_VERSION = "inference-service.intent/v1"
PLAN_SCHEMA_VERSION = "inference-service.run-plan/v1"
CAPABILITY_PROBE_RECEIPT_SCHEMA_VERSION = "inference-service.capability-probe-receipt/v1"
LAUNCH_RECEIPT_SCHEMA_VERSION = "inference-service.launch-receipt/v1"
STATUS_RECEIPT_SCHEMA_VERSION = "inference-service.status-receipt/v1"
CANCELLATION_RECEIPT_SCHEMA_VERSION = "inference-service.cancellation-receipt/v1"

Capability = Literal["chat-completions", "dynamic-labels", "offsets", "scores"]


class FrozenModel(BaseModel):
    """Closed immutable base for compiler values and transport records."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class EntityDetection(FrozenModel):
    """Entity detection requirements independent of a serving engine."""

    kind: Literal["entity-detection"] = "entity-detection"
    dynamic_labels: bool
    offsets: bool
    scores: bool

    def required_capabilities(self) -> tuple[Capability, ...]:
        """Return the endpoint capabilities required by this task."""
        capabilities: list[Capability] = []
        if self.dynamic_labels:
            capabilities.append("dynamic-labels")
        if self.offsets:
            capabilities.append("offsets")
        if self.scores:
            capabilities.append("scores")
        return tuple(capabilities)


class Generation(FrozenModel):
    """Text-generation requirements independent of a serving engine."""

    kind: Literal["generation"] = "generation"
    chat: Literal[True] = True

    def required_capabilities(self) -> tuple[Capability, ...]:
        """Return the endpoint capabilities required by this task."""
        return ("chat-completions",)


TaskSpec = Annotated[EntityDetection | Generation, Field(discriminator="kind")]


class LoraAdapter(FrozenModel):
    """A LoRA artifact and the stable name exposed by the serving engine."""

    path: str = Field(min_length=1)
    name: str = Field(min_length=1)


class HuggingFaceModel(FrozenModel):
    """A Hugging Face model identifier and optional immutable revision."""

    kind: Literal["hugging-face"] = "hugging-face"
    model_id: str = Field(min_length=1)
    revision: str | None = Field(default=None, min_length=1)
    adapter: LoraAdapter | None = None


ModelSpec = Annotated[HuggingFaceModel, Field(discriminator="kind")]


class NativeGlinerEngine(FrozenModel):
    """The characterized local GLiNER or GLiNER2 Python runtime."""

    kind: Literal["native-gliner"] = "native-gliner"
    family: Literal["nvidia-gliner", "gliner2"] = "nvidia-gliner"
    device: str = Field(default="auto", min_length=1)
    batch_mode: bool = True
    max_batch_requests: int = Field(default=32, ge=1)
    batch_wait_ms: float = Field(default=10, ge=0)
    log_format: Literal["plain", "json"] = "plain"


class VllmEngine(FrozenModel):
    """vLLM's OpenAI-compatible server with bounded common options."""

    kind: Literal["vllm"] = "vllm"
    executable: str = Field(default="vllm", min_length=1)
    served_model_name: str | None = Field(default=None, min_length=1)
    api_key_env: str | None = Field(default=None, min_length=1)
    tensor_parallel_size: int | None = Field(default=None, ge=1)
    gpu_memory_utilization: float | None = Field(default=None, gt=0, le=1)
    max_model_len: int | None = Field(default=None, ge=1)
    eager: bool = False


EngineSpec = Annotated[NativeGlinerEngine | VllmEngine, Field(discriminator="kind")]


class LocalProcessPlacement(FrozenModel):
    """A process on the caller's host."""

    kind: Literal["local-process"] = "local-process"
    host: str = Field(default="127.0.0.1", min_length=1)
    port: int = Field(default=8000, ge=1, le=65535)


class DockerPlacement(FrozenModel):
    """A managed local Docker container with direct host access."""

    kind: Literal["docker"] = "docker"
    host: str = Field(default="127.0.0.1", min_length=1)
    port: int = Field(default=8000, ge=1, le=65535)
    image: str = Field(min_length=1)
    runtime: Literal["docker"] = "docker"
    gpus: str = Field(default="all", min_length=1)
    hugging_face_cache: str | None = Field(default=None, min_length=1)


PlacementSpec = Annotated[LocalProcessPlacement | DockerPlacement, Field(discriminator="kind")]


class DirectAccess(FrozenModel):
    """A direct HTTP endpoint exposed by the managed runtime."""

    kind: Literal["direct"] = "direct"


AccessSpec = Annotated[DirectAccess, Field(discriminator="kind")]


class ManagedLifecycle(FrozenModel):
    """The compiler owns launch, inspection, cancellation, and cleanup."""

    kind: Literal["managed"] = "managed"
    startup_timeout_seconds: float = Field(default=120, gt=0)
    shutdown_timeout_seconds: float = Field(default=30, gt=0)


LifecycleSpec = Annotated[ManagedLifecycle, Field(discriminator="kind")]


class InferenceIntent(FrozenModel):
    """Complete semantic input to pure inference-service compilation."""

    schema_version: Literal["inference-service.intent/v1"] = INTENT_SCHEMA_VERSION
    task: TaskSpec
    model: ModelSpec
    engine: EngineSpec
    placement: PlacementSpec
    access: AccessSpec
    lifecycle: LifecycleSpec

    @property
    def expected_model(self) -> str:
        """Return the model ID that the compiled endpoint must serve."""
        if self.model.adapter is not None:
            return self.model.adapter.name
        if isinstance(self.engine, VllmEngine) and self.engine.served_model_name is not None:
            return self.engine.served_model_name
        return self.model.model_id


class LiteralArgument(FrozenModel):
    """One non-secret argv value."""

    kind: Literal["literal"] = "literal"
    value: str


CommandArgument = LiteralArgument


class EnvironmentVariable(FrozenModel):
    """One ordinary non-secret process environment value."""

    kind: Literal["literal"] = "literal"
    name: str = Field(min_length=1)
    value: str


class SecretEnvironmentVariable(FrozenModel):
    """One process environment value resolved from a named secret source."""

    kind: Literal["secret-reference"] = "secret-reference"
    name: str = Field(min_length=1)
    source_environment_variable: str = Field(min_length=1)


EnvironmentSpec = Annotated[EnvironmentVariable | SecretEnvironmentVariable, Field(discriminator="kind")]


class CommandSpec(FrozenModel):
    """Complete argv and ordinary environment for one managed service."""

    argv: tuple[CommandArgument, ...] = Field(min_length=1)
    environment: tuple[EnvironmentSpec, ...] = ()
    working_directory: str = "."

    def render_argv(self) -> tuple[str, ...]:
        """Render the complete non-secret process argument vector."""
        return tuple(argument.value for argument in self.argv)

    def render_environment(self, *, resolve_secrets: Mapping[str, str] | None = None) -> dict[str, str]:
        """Render environment values with redacted or explicitly supplied secrets."""
        values: dict[str, str] = {}
        for variable in self.environment:
            match variable:
                case EnvironmentVariable(name=name, value=value):
                    values[name] = value
                case SecretEnvironmentVariable(name=name, source_environment_variable=source):
                    if resolve_secrets is None:
                        values[name] = f"<secret:{source}>"
                    elif source in resolve_secrets:
                        values[name] = resolve_secrets[source]
                    else:
                        raise ValueError(f"secret environment variable {source!r} is not resolved")
        return values


class EndpointContract(FrozenModel):
    """Direct OpenAI-compatible endpoint produced by a run."""

    scheme: Literal["http"] = "http"
    host: str
    port: int
    base_path: Literal["/v1"] = "/v1"

    @property
    def url(self) -> str:
        """Return the normalized endpoint URL."""
        return f"{self.scheme}://{self.host}:{self.port}{self.base_path}"


class HttpProbe(FrozenModel):
    """One bounded readiness or capability-probe request."""

    scheme: Literal["http"] = "http"
    host: str
    port: int
    path: str
    expected_status: int = 200
    timeout_seconds: float = Field(gt=0)
    bearer_token_environment_variable: str | None = Field(default=None, min_length=1)

    @property
    def url(self) -> str:
        """Return the complete probe URL."""
        return f"{self.scheme}://{self.host}:{self.port}{self.path}"


class LocalProcessRuntime(FrozenModel):
    """Runtime facts needed to launch and stop a local process."""

    kind: Literal["local-process"] = "local-process"
    cleanup: Literal["terminate-process-group"] = "terminate-process-group"


class DockerRuntime(FrozenModel):
    """Runtime facts needed to launch and remove a local container."""

    kind: Literal["docker"] = "docker"
    image: str
    cleanup: Literal["remove-container"] = "remove-container"


RuntimeSpec = Annotated[LocalProcessRuntime | DockerRuntime, Field(discriminator="kind")]


class CompatibilityEvidence(FrozenModel):
    """One compiler rule supporting or qualifying the selected combination."""

    rule: str
    outcome: Literal["characterized", "runtime-probe-required"]
    detail: str


class RunPlan(FrozenModel):
    """Portable, immutable, effect-free instructions for one service run."""

    schema_version: Literal["inference-service.run-plan/v1"] = PLAN_SCHEMA_VERSION
    plan_digest: str
    intent_digest: str = Field(min_length=1)
    intent: InferenceIntent
    command: CommandSpec
    runtime: RuntimeSpec
    endpoint: EndpointContract
    readiness: HttpProbe
    expected_model: str = Field(min_length=1)
    required_capabilities: tuple[Capability, ...]
    declared_capabilities: tuple[Capability, ...]
    compatibility_evidence: tuple[CompatibilityEvidence, ...]
    dependencies: tuple[str, ...] = ()
    source_revision: str = Field(min_length=1)


class CapabilityProbeReceipt(FrozenModel):
    """Runtime evidence for the endpoint capabilities required by a plan."""

    schema_version: Literal["inference-service.capability-probe-receipt/v1"] = CAPABILITY_PROBE_RECEIPT_SCHEMA_VERSION
    plan_digest: str
    endpoint: EndpointContract
    observed_at: str
    models: tuple[str, ...]
    observed_capabilities: tuple[Capability, ...]
    passed: bool


class LocalProcessHandle(FrozenModel):
    """Reconnectable identity for a managed local process."""

    kind: Literal["local-process"] = "local-process"
    external_id: str
    pid: int = Field(ge=1)
    process_group_id: int = Field(ge=1)
    start_marker: str | None
    stdout_path: str
    stderr_path: str


class DockerHandle(FrozenModel):
    """Reconnectable identity for a managed Docker container."""

    kind: Literal["docker"] = "docker"
    external_id: str
    container_id: str


HandleRecord = Annotated[LocalProcessHandle | DockerHandle, Field(discriminator="kind")]


class LaunchReceipt(FrozenModel):
    """Known launch effects, reconnectable identity, and readiness evidence."""

    schema_version: Literal["inference-service.launch-receipt/v1"] = LAUNCH_RECEIPT_SCHEMA_VERSION
    plan_digest: str
    launched_at: str
    shutdown_timeout_seconds: float = Field(gt=0)
    handle: HandleRecord
    probe: CapabilityProbeReceipt


class StatusReceipt(FrozenModel):
    """Observed state for a reconnectable managed handle."""

    schema_version: Literal["inference-service.status-receipt/v1"] = STATUS_RECEIPT_SCHEMA_VERSION
    plan_digest: str
    observed_at: str
    handle: HandleRecord
    state: Literal["running", "stopped"]


class CancellationReceipt(FrozenModel):
    """Cancellation outcome and cleanup state for a managed handle."""

    schema_version: Literal["inference-service.cancellation-receipt/v1"] = CANCELLATION_RECEIPT_SCHEMA_VERSION
    plan_digest: str
    canceled_at: str
    handle: HandleRecord
    outcome: Literal["terminated", "already-stopped", "forced"]
    cleanup_complete: bool


class RuntimeDiagnostic(FrozenModel):
    """Serializable runtime failure including every known external effect."""

    code: str
    message: str
    known_effects: tuple[str, ...] = ()
    cleanup_complete: bool | None = None


class CachedModel(FrozenModel):
    """One immutable Hugging Face cache snapshot available to local runtimes."""

    repository: str
    revision: str
    snapshot_path: str


class CachedModels(FrozenModel):
    """Versioned discovery result that performs no model downloads."""

    schema_version: Literal["inference-service.cached-models/v1"] = "inference-service.cached-models/v1"
    cache_root: str
    models: tuple[CachedModel, ...]
