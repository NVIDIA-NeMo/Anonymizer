# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Versioned transport and immutable intermediate-representation models."""

from __future__ import annotations

import re
from typing import Annotated, Literal, NewType

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, JsonValue, field_validator

SPEC_SCHEMA_VERSION = "inference-service.spec/v3"
PLAN_SCHEMA_VERSION = "inference-service.run-plan/v3"
LAUNCH_RECEIPT_SCHEMA_VERSION = "inference-service.launch-receipt/v3"
STATUS_RECEIPT_SCHEMA_VERSION = "inference-service.status-receipt/v3"
STOP_RECEIPT_SCHEMA_VERSION = "inference-service.stop-receipt/v3"

GLINER2_MODEL_ID = "fastino/gliner2-privacy-filter-PII-multi"

# Options the tool derives from the spec; a profile cannot set them.
VLLM_OWNED_OPTIONS = frozenset(
    {
        "model",
        "host",
        "port",
        "uds",
        "revision",
        "tokenizer-revision",
        "served-model-name",
        "api-key",
        "hf-token",
        "config",
        "middleware",
        "enable-lora",
        "lora-modules",
    }
)
_OPTION_KEY = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")

Pid = NewType("Pid", int)
ProcessGroupId = NewType("ProcessGroupId", int)
Sha256Hex = NewType("Sha256Hex", str)
"""Lowercase hexadecimal SHA-256 digest."""

ServiceState = Literal["running", "stopped", "ownership-unverified"]
StopOutcome = Literal["terminated", "already-stopped", "forced"]

OptionValue = bool | int | float | str | list[str | int | float] | dict[str, JsonValue]


class FrozenModel(BaseModel):
    """Closed immutable base for compiler values and transport records."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class LoraAdapter(FrozenModel):
    """A LoRA artifact and the stable name exposed by the serving engine."""

    path: str = Field(min_length=1)
    name: str = Field(min_length=1)


class HuggingFaceModel(FrozenModel):
    """A Hugging Face model identifier and optional immutable revision."""

    model_id: str = Field(min_length=1)
    revision: str | None = Field(default=None, min_length=1)
    adapter: LoraAdapter | None = None


class VllmBackend(FrozenModel):
    """vLLM's OpenAI-compatible server for a generation model."""

    kind: Literal["vllm"] = "vllm"
    model: HuggingFaceModel
    served_model_name: str | None = Field(default=None, min_length=1)
    options: dict[str, OptionValue] = Field(default_factory=dict)
    """vLLM serve options by their long flag name without dashes, e.g. ``max-model-len``."""
    probe_extra_body: dict[str, JsonValue] = Field(default_factory=dict)
    """Extra chat-completion fields for the readiness probe, e.g. to disable reasoning."""

    @field_validator("options")
    @classmethod
    def _validate_options(cls, options: dict[str, OptionValue]) -> dict[str, OptionValue]:
        for key, value in options.items():
            if not _OPTION_KEY.fullmatch(key):
                raise ValueError(f"vLLM option {key!r} must be a lowercase kebab-case flag name")
            if key.startswith("no-"):
                raise ValueError(f"vLLM option {key!r}: set {key.removeprefix('no-')!r} to false instead")
            if key in VLLM_OWNED_OPTIONS:
                raise ValueError(f"vLLM option {key!r} is derived from the profile's model and endpoint fields")
            if value == []:
                raise ValueError(f"vLLM option {key!r} cannot be an empty list")
        return options

    @property
    def model_name(self) -> str:
        """Return the model ID that the endpoint serves."""
        if self.model.adapter is not None:
            return self.model.adapter.name
        return self.served_model_name or self.model.model_id


class Gliner2Backend(FrozenModel):
    """Anonymizer's native GLiNER2 detector server, pinned to one checkpoint."""

    kind: Literal["gliner2"] = "gliner2"
    device: Literal["auto", "cuda", "mps", "cpu"] = "auto"

    @property
    def model_name(self) -> str:
        """Return the model ID that the endpoint serves."""
        return GLINER2_MODEL_ID


Backend = Annotated[VllmBackend | Gliner2Backend, Field(discriminator="kind")]


class ServiceSpec(FrozenModel):
    """Complete semantic input to pure inference-service compilation."""

    schema_version: Literal["inference-service.spec/v3"] = SPEC_SCHEMA_VERSION
    backend: Backend
    python_executable: str | None = Field(default=None, min_length=1)
    """Serving interpreter; when omitted the backend's default environment is used."""
    host: str = Field(default="127.0.0.1", min_length=1)
    port: int = Field(ge=1024, le=65535)
    api_key_env: str | None = Field(default=None, min_length=1)
    """Controller environment variable holding the endpoint's bearer token."""
    startup_timeout_seconds: float = Field(default=120, gt=0)
    shutdown_timeout_seconds: float = Field(default=30, gt=0)


class RunPlan(FrozenModel):
    """Portable, frozen, effect-free instructions for one service run."""

    schema_version: Literal["inference-service.run-plan/v3"] = PLAN_SCHEMA_VERSION
    spec: ServiceSpec
    argv: tuple[str, ...] = Field(min_length=1)
    plan_digest: str

    @property
    def endpoint(self) -> tuple[str, int]:
        """Return the host and port the service listens on."""
        return self.spec.host, self.spec.port

    @property
    def base_url(self) -> str:
        """Return the OpenAI-compatible base URL of the endpoint."""
        return f"http://{self.spec.host}:{self.spec.port}/v1"

    @property
    def served_model_name(self) -> str:
        """Return the model ID that the endpoint must serve."""
        return self.spec.backend.model_name


class ConnectionInfo(FrozenModel):
    """Non-secret client inputs derived from a compiled run plan."""

    url: str
    model: str
    api_key_env: str | None = None


class LocalProcessHandle(FrozenModel):
    """Observed local-process coordinates checked against PID reuse.

    The service is launched as a session leader, so its PID is also its process group ID.
    """

    pid: Pid = Field(ge=1)
    start_marker: str = Field(min_length=1)
    launch_token_sha256: Sha256Hex = Field(pattern=r"^[0-9a-f]{64}$")
    """Non-secret digest of the launch token every process in the group inherits."""
    stdout_path: str
    stderr_path: str

    @property
    def process_group_id(self) -> ProcessGroupId:
        """Return the process group created by the launch."""
        return ProcessGroupId(self.pid)

    @property
    def label(self) -> str:
        """Return the process identity used in diagnostics."""
        return f"{self.pid}:{self.start_marker}"


class LaunchReceipt(FrozenModel):
    """Durable record of a launched plan and its observed process."""

    schema_version: Literal["inference-service.launch-receipt/v3"] = LAUNCH_RECEIPT_SCHEMA_VERSION
    plan: RunPlan
    launched_at: AwareDatetime
    handle: LocalProcessHandle


class StatusReceipt(FrozenModel):
    """Recorded state observation for a managed handle."""

    schema_version: Literal["inference-service.status-receipt/v3"] = STATUS_RECEIPT_SCHEMA_VERSION
    plan_digest: str
    observed_at: AwareDatetime
    handle: LocalProcessHandle
    state: ServiceState
    """``ownership-unverified``: the leader is gone and the surviving group cannot be shown to be this launch's."""


class StopReceipt(FrozenModel):
    """Recorded cleanup outcome for a managed handle."""

    schema_version: Literal["inference-service.stop-receipt/v3"] = STOP_RECEIPT_SCHEMA_VERSION
    plan_digest: str
    stopped_at: AwareDatetime
    handle: LocalProcessHandle
    outcome: StopOutcome
    cleanup_complete: bool


RuntimeErrorCode = Literal[
    "validation-failed",
    "probe-failed",
    "served-model-mismatch",
    "missing-secret",
    "launch-exited",
    "launch-ownership-not-observed",
    "readiness-timeout",
    "unexpected-readiness-failure",
    "process-identity-mismatch",
    "process-identity-unavailable",
    "process-group-ownership-unverified",
    "missing-process-start-marker",
]


class RuntimeDiagnostic(FrozenModel):
    """Serializable runtime failure including every known external effect."""

    code: RuntimeErrorCode
    message: str
    known_effects: tuple[str, ...] = ()
    cleanup_complete: bool | None = None
