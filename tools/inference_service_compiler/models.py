# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Versioned transport and immutable intermediate-representation models."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Literal, assert_never

from pydantic import BaseModel, ConfigDict, Field

INTENT_SCHEMA_VERSION = "inference-service.intent/v2"
PLAN_SCHEMA_VERSION = "inference-service.run-plan/v2"
CAPABILITY_PROBE_RECEIPT_SCHEMA_VERSION = "inference-service.capability-probe-receipt/v1"
LAUNCH_RECEIPT_SCHEMA_VERSION = "inference-service.launch-receipt/v1"
STATUS_RECEIPT_SCHEMA_VERSION = "inference-service.status-receipt/v1"
CANCELLATION_RECEIPT_SCHEMA_VERSION = "inference-service.cancellation-receipt/v1"

Capability = Literal["chat-completions", "dynamic-labels", "offsets", "scores"]
FactoryPlugin = Literal["deberta_gliner", "deberta_gliner2"]


def parse_factory_plugin(value: str) -> FactoryPlugin:
    """Parse the Factory plugin name at an untyped process boundary."""
    match value:
        case "deberta_gliner" | "deberta_gliner2":
            return value
        case _:
            raise ValueError(f"unsupported vLLM Factory plugin {value!r}")


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

    model_id: str = Field(min_length=1)
    revision: str | None = Field(default=None, min_length=1)
    adapter: LoraAdapter | None = None


class VllmFactoryIntegration(FrozenModel):
    """A supported vLLM Factory structured-prediction plugin."""

    plugin: FactoryPlugin
    prepared_model_root: str = Field(default="/tmp/anonymizer-vllm-factory", min_length=1)


class Vllm(FrozenModel):
    """vLLM's OpenAI-compatible server with bounded common options."""

    python_executable: str = Field(default=".venv/bin/python", min_length=1)
    served_model_name: str | None = Field(default=None, min_length=1)
    api_key_env: str | None = Field(default=None, min_length=1)
    tensor_parallel_size: int | None = Field(default=None, ge=1)
    gpu_memory_utilization: float | None = Field(default=None, gt=0, le=1)
    max_model_len: int | None = Field(default=None, ge=1)
    max_num_seqs: int | None = Field(default=None, ge=1)
    eager: bool = False
    enable_prefix_caching: bool = False
    async_scheduling: bool = False
    mamba_backend: Literal["triton", "flashinfer"] | None = None
    mamba_ssm_cache_dtype: Literal["auto", "float32", "float16", "bfloat16"] = "auto"
    enable_mamba_cache_stochastic_rounding: bool = False
    mamba_cache_philox_rounds: int = Field(default=0, ge=0)
    factory: VllmFactoryIntegration | None = None


class LocalProcess(FrozenModel):
    """The only supported inference-host deployment domain."""

    host: str = Field(default="127.0.0.1", min_length=1)
    port: int = Field(default=8000, ge=1, le=65535)

    startup_timeout_seconds: float = Field(default=120, gt=0)
    shutdown_timeout_seconds: float = Field(default=30, gt=0)


class InferenceIntent(FrozenModel):
    """Complete semantic input to pure inference-service compilation."""

    schema_version: Literal["inference-service.intent/v2"] = INTENT_SCHEMA_VERSION
    task: TaskSpec
    model: HuggingFaceModel
    vllm: Vllm
    local: LocalProcess

    @property
    def expected_model(self) -> str:
        """Return the model ID that the compiled endpoint must serve."""
        if self.model.adapter is not None:
            return self.model.adapter.name
        if self.vllm.served_model_name is not None:
            return self.vllm.served_model_name
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
                case _:
                    assert_never(variable)
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


class CompatibilityEvidence(FrozenModel):
    """One compiler rule supporting or qualifying the selected combination."""

    rule: str
    outcome: Literal["characterized", "runtime-probe-required"]
    detail: str


class RunPlan(FrozenModel):
    """Portable, immutable, effect-free instructions for one service run."""

    schema_version: Literal["inference-service.run-plan/v2"] = PLAN_SCHEMA_VERSION
    plan_digest: str
    intent_digest: str = Field(min_length=1)
    intent: InferenceIntent
    command: CommandSpec
    runtime: LocalProcessRuntime
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


class LaunchReceipt(FrozenModel):
    """Known launch effects, reconnectable identity, and readiness evidence."""

    schema_version: Literal["inference-service.launch-receipt/v1"] = LAUNCH_RECEIPT_SCHEMA_VERSION
    plan_digest: str
    launched_at: str
    shutdown_timeout_seconds: float = Field(gt=0)
    handle: LocalProcessHandle
    probe: CapabilityProbeReceipt


class StatusReceipt(FrozenModel):
    """Observed state for a reconnectable managed handle."""

    schema_version: Literal["inference-service.status-receipt/v1"] = STATUS_RECEIPT_SCHEMA_VERSION
    plan_digest: str
    observed_at: str
    handle: LocalProcessHandle
    state: Literal["running", "stopped"]


class CancellationReceipt(FrozenModel):
    """Cancellation outcome and cleanup state for a managed handle."""

    schema_version: Literal["inference-service.cancellation-receipt/v1"] = CANCELLATION_RECEIPT_SCHEMA_VERSION
    plan_digest: str
    canceled_at: str
    handle: LocalProcessHandle
    outcome: Literal["terminated", "already-stopped", "forced"]
    cleanup_complete: bool


class RuntimeDiagnostic(FrozenModel):
    """Serializable runtime failure including every known external effect."""

    code: str
    message: str
    known_effects: tuple[str, ...] = ()
    cleanup_complete: bool | None = None
