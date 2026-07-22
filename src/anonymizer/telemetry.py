# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Anonymous telemetry handler for NeMo Anonymizer.

Emits one ``anonymizer_event`` per ``Anonymizer.run()`` / ``Anonymizer.preview()``
invocation. Telemetry is opt-out via:

- ``NEMO_TELEMETRY_ENABLED=false`` environment variable
- ``AnonymizerConfig(emit_telemetry=False)``
- ``--no-emit-telemetry`` CLI flag

Related environment variables (read at runtime, not import time):

- ``NEMO_TELEMETRY_ENABLED``: set to ``false`` / ``0`` / ``no`` to disable.
- ``NEMO_DEPLOYMENT_TYPE``: ``cli``, ``sdk``, ``nmp``. Defaults to ``sdk``.
- ``NEMO_TELEMETRY_ENDPOINT``: override the destination URL.
- ``NEMO_SESSION_PREFIX``: prepended to session IDs. Set to ``"anonymizer-"``
  automatically by ``Anonymizer.__init__`` for dashboard filtering.
"""

from __future__ import annotations

import asyncio
import logging
import os
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    import httpx
    from data_designer.config.models import ModelProvider

logger = logging.getLogger("anonymizer")

CLIENT_ID = "184482118588404"
NEMO_TELEMETRY_VERSION = "nemo-telemetry/1.0"
DEFAULT_ENDPOINT = "https://events.telemetry.data.nvidia.com/v1.1/events/json"
MAX_RETRIES = 3
CPU_ARCHITECTURE = platform.uname().machine


class NemoSourceEnum(str, Enum):
    ANONYMIZER = "anonymizer"
    UNDEFINED = "undefined"


class TaskEnum(str, Enum):
    BATCH = "batch"
    PREVIEW = "preview"


class TaskStatusEnum(str, Enum):
    COMPLETED = "completed"
    ERROR = "error"
    CANCELED = "canceled"


class DeploymentTypeEnum(str, Enum):
    CLI = "cli"
    SDK = "sdk"
    NMP = "nmp"
    NVIDIA_INTERNAL = "nvidia-internal"
    UNDEFINED = "undefined"


class ModelHostEnum(str, Enum):
    NVIDIA_BUILD = "nvidia-build"
    NVIDIA_INTERNAL = "nvidia-internal"
    OPENROUTER = "openrouter"
    LOCAL = "local"
    OTHER = "other"


NOT_APPLICABLE = "not_applicable"


def _telemetry_enabled() -> bool:
    return os.getenv("NEMO_TELEMETRY_ENABLED", "true").lower() in ("1", "true", "yes")


def _telemetry_endpoint() -> str:
    return os.getenv("NEMO_TELEMETRY_ENDPOINT", DEFAULT_ENDPOINT)


def _deployment_type() -> DeploymentTypeEnum:
    raw = os.getenv("NEMO_DEPLOYMENT_TYPE", "sdk").lower()
    try:
        return DeploymentTypeEnum(raw)
    except ValueError:
        return DeploymentTypeEnum.UNDEFINED


def _session_prefix() -> str | None:
    return os.getenv("NEMO_SESSION_PREFIX")


_tokenizer = None  # lazily cached cl100k_base encoder


def _get_tokenizer():
    """Lazily initialize and cache the tiktoken encoder.

    Mirrors DataDesigner's approach (cl100k_base — the GPT-3.5/4 family encoder).
    The exact LLM anonymizer hits may tokenize differently; this is a consistent
    cross-run estimate, not an exact count.
    """
    global _tokenizer
    if _tokenizer is None:
        import tiktoken

        _tokenizer = tiktoken.get_encoding("cl100k_base")
    return _tokenizer


def avg_tokens_per_record(texts) -> int:
    """Mean tiktoken count across the input texts.

    Returns -1 on empty input or tokenizer failure. Telemetry is best-effort.
    Accepts any iterable of strings (list, pd.Series).
    """
    try:
        tokenizer = _get_tokenizer()
        counts = [len(tokenizer.encode(str(t), disallowed_special=())) for t in texts]
        if not counts:
            return -1
        return int(sum(counts) / len(counts))
    except Exception:
        return -1


def classify_model_host(provider: ModelProvider | None) -> ModelHostEnum:
    """Classify a ModelProvider's endpoint URL into one of the ModelHostEnum values.

    Substring-matches known host fragments against the (lower-cased) endpoint URL.
    """
    if provider is None:
        return ModelHostEnum.OTHER
    # ``endpoint`` is a single URL string; the ``in`` checks below are substring
    # searches against that string (not iteration over characters).
    endpoint = (getattr(provider, "endpoint", "") or "").lower()
    if "build.nvidia.com" in endpoint or "integrate.api.nvidia.com" in endpoint:
        return ModelHostEnum.NVIDIA_BUILD
    if "inference-api.nvidia.com" in endpoint:
        return ModelHostEnum.NVIDIA_INTERNAL
    if "openrouter.ai" in endpoint:
        return ModelHostEnum.OPENROUTER
    local_hosts = ("localhost", "127.0.0.1", "0.0.0.0", "[::1]")
    if any(host in endpoint for host in local_hosts):
        return ModelHostEnum.LOCAL
    return ModelHostEnum.OTHER


def collect_model_hosts(hosts: list[ModelHostEnum]) -> list[str]:
    """Sort + dedupe per-provider host classifications into the wire-format list.

    Returns ``["other"]`` when no hosts were observed — never an empty list, so the
    telemetry payload always carries at least one host string.
    """
    unique = sorted({h.value for h in hosts if h is not None})
    return unique or [ModelHostEnum.OTHER.value]


def sort_join_aliases(aliases: list[str]) -> str:
    """Canonical pool serialization: sorted ascending, comma-joined, no spaces."""
    cleaned = [a.strip() for a in aliases if a and a.strip()]
    if not cleaned:
        return NOT_APPLICABLE
    return ",".join(sorted(cleaned))


class AnonymizerEvent(BaseModel):
    """Pydantic model for the anonymizer_event payload.

    Field aliases match the camelCase schema in
    ``aire/microservices/nemo-telemetry`` ``schemas/anonymous_events.json``.
    """

    _event_name: ClassVar[str] = "anonymizer_event"
    # Matches the schemaMeta.schemaVersion of nemo-telemetry's anonymous_events.json.
    _schema_version: ClassVar[str] = "1.9"

    # Identity
    nemo_source: NemoSourceEnum = Field(default=NemoSourceEnum.ANONYMIZER, alias="nemoSource")
    task: TaskEnum
    task_status: TaskStatusEnum = Field(alias="taskStatus")
    deployment_type: DeploymentTypeEnum = Field(default_factory=_deployment_type, alias="deploymentType")

    # Timing
    job_duration_sec: float = Field(default=-1.0, alias="jobDurationSec")

    # Record counts (raw integers, legal-cleared)
    num_input_records: int = Field(default=-1, alias="numInputRecords")
    num_success_records: int = Field(default=-1, alias="numSuccessRecords")
    num_failure_records: int = Field(default=-1, alias="numFailureRecords")
    avg_tokens_per_record: int = Field(default=-1, alias="avgTokensPerRecord")
    input_tokens: int = Field(default=0, alias="inputTokens")

    # Configuration
    transformation_type: str = Field(alias="transformationType")
    custom_data_summary_provided: bool = Field(default=False, alias="customDataSummaryProvided")
    custom_privacy_goal_provided: bool = Field(default=False, alias="customPrivacyGoalProvided")
    custom_substitute_instructions_provided: bool = Field(default=False, alias="customSubstituteInstructionsProvided")
    max_repair_iterations: int = Field(default=-1, alias="maxRepairIterations")
    strict_entity_protection: bool = Field(default=False, alias="strictEntityProtection")
    repair_iterations_triggered: int = Field(default=-1, alias="repairIterationsTriggered")

    # Models per step. The first three always run regardless of strategy, so they have
    # no default; the rest fall back to ``NOT_APPLICABLE`` when their step doesn't run.
    entity_detector_model: str = Field(alias="entityDetectorModel")
    entity_validator_model: str = Field(alias="entityValidatorModel")
    entity_augmenter_model: str = Field(alias="entityAugmenterModel")
    latent_detector_model: str = Field(default=NOT_APPLICABLE, alias="latentDetectorModel")
    replacement_generator_model: str = Field(default=NOT_APPLICABLE, alias="replacementGeneratorModel")
    domain_classifier_model: str = Field(default=NOT_APPLICABLE, alias="domainClassifierModel")
    disposition_analyzer_model: str = Field(default=NOT_APPLICABLE, alias="dispositionAnalyzerModel")
    meaning_extractor_model: str = Field(default=NOT_APPLICABLE, alias="meaningExtractorModel")
    qa_generator_model: str = Field(default=NOT_APPLICABLE, alias="qaGeneratorModel")
    rewriter_model: str = Field(default=NOT_APPLICABLE, alias="rewriterModel")
    evaluator_model: str = Field(default=NOT_APPLICABLE, alias="evaluatorModel")
    repairer_model: str = Field(default=NOT_APPLICABLE, alias="repairerModel")
    judge_model: str = Field(default=NOT_APPLICABLE, alias="judgeModel")
    model_hosts: list[str] = Field(default_factory=list, alias="modelHosts")

    # Failure attribution (workflow_name granularity)
    entity_detection_failure_count: int = Field(default=0, alias="entityDetectionFailureCount")
    latent_detection_failure_count: int = Field(default=0, alias="latentDetectionFailureCount")
    replace_map_generation_failure_count: int = Field(default=0, alias="replaceMapGenerationFailureCount")
    rewrite_pipeline_failure_count: int = Field(default=0, alias="rewritePipelineFailureCount")
    rewrite_evaluate_failure_count: int = Field(default=0, alias="rewriteEvaluateFailureCount")
    rewrite_repair_failure_count: int = Field(default=0, alias="rewriteRepairFailureCount")
    rewrite_final_judge_failure_count: int = Field(default=0, alias="rewriteFinalJudgeFailureCount")
    unknown_step_failure_count: int = Field(default=0, alias="unknownStepFailureCount")

    model_config = {"populate_by_name": True}


@dataclass
class QueuedEvent:
    event: AnonymizerEvent
    timestamp: datetime
    retry_count: int = 0


def _get_iso_timestamp(dt: datetime | None = None) -> str:
    if dt is None:
        dt = datetime.now(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def build_payload(
    events: list[QueuedEvent], *, source_client_version: str, session_id: str = "undefined"
) -> dict[str, Any]:
    if not events:
        raise ValueError("build_payload requires at least one event")
    return {
        "browserType": "undefined",  # do not change
        "clientId": CLIENT_ID,
        "clientType": "Native",  # do not change
        "clientVariant": "Release",  # do not change
        "clientVer": source_client_version,
        "cpuArchitecture": CPU_ARCHITECTURE,
        "deviceGdprBehOptIn": "None",  # do not change
        "deviceGdprFuncOptIn": "None",  # do not change
        "deviceGdprTechOptIn": "None",  # do not change
        "deviceId": "undefined",  # do not change
        "deviceMake": "undefined",  # do not change
        "deviceModel": "undefined",  # do not change
        "deviceOS": "undefined",  # do not change
        "deviceOSVersion": "undefined",  # do not change
        "deviceType": "undefined",  # do not change
        "eventProtocol": "1.6",  # do not change
        "eventSchemaVer": events[0].event._schema_version,
        "eventSysVer": NEMO_TELEMETRY_VERSION,
        "externalUserId": "undefined",  # do not change
        "gdprBehOptIn": "None",  # do not change
        "gdprFuncOptIn": "None",  # do not change
        "gdprTechOptIn": "None",  # do not change
        "idpId": "undefined",  # do not change
        "integrationId": "undefined",  # do not change
        "productName": "undefined",  # do not change
        "productVersion": "undefined",  # do not change
        "sentTs": _get_iso_timestamp(),
        "sessionId": session_id,
        "userId": "undefined",  # do not change
        "events": [
            {
                "ts": _get_iso_timestamp(q.timestamp),
                "parameters": q.event.model_dump(by_alias=True, mode="json"),
                "name": q.event._event_name,
            }
            for q in events
        ],
    }


class TelemetryHandler:
    """Fire-and-flush telemetry handler for Anonymizer.

    Anonymizer runs are short, so we skip DD's background-daemon-thread mode
    entirely. Usage:

        with TelemetryHandler(source_client_version=__version__, session_id=...) as h:
            h.enqueue(event)
        # on __exit__, queued events are flushed synchronously

    All errors are swallowed; telemetry must never disrupt the pipeline.
    """

    def __init__(
        self,
        *,
        source_client_version: str = "undefined",
        session_id: str = "undefined",
        max_queue_size: int = 50,
        max_retries: int = MAX_RETRIES,
    ):
        self._max_queue_size = max_queue_size
        self._max_retries = max_retries
        self._events: list[QueuedEvent] = []
        self._dlq: list[QueuedEvent] = []
        self._source_client_version = source_client_version
        prefix = _session_prefix()
        self._session_id = f"{prefix}{session_id}" if prefix else session_id

    def enqueue(self, event: AnonymizerEvent) -> None:
        if not _telemetry_enabled():
            return
        if not isinstance(event, AnonymizerEvent):
            return
        self._events.append(QueuedEvent(event=event, timestamp=datetime.now(timezone.utc)))

    def flush(self) -> None:
        if not (self._events or self._dlq):
            return
        try:
            self._run_sync(self._flush_events())
        except Exception:
            # Telemetry is best-effort; never raise into the caller's pipeline.
            # Logged at DEBUG so it's surfaceable via `logging.getLogger("anonymizer")`
            # without spamming default-level output.
            logger.debug("Telemetry flush failed", exc_info=True)

    @staticmethod
    def _run_sync(coro: Any) -> Any:
        """Run an awaitable synchronously from any caller context.

        Mirrors DataDesigner's ``TelemetryHandler._run_sync`` (see
        ``data_designer/engine/models/telemetry.py``). Plain ``asyncio.run`` raises
        ``RuntimeError`` when called from a thread that already has a running event
        loop — e.g. a Jupyter kernel, an async test runner, or any caller invoking
        ``Anonymizer.run()`` from within ``asyncio.run(...)``. In that case the
        previous implementation swallowed the error and silently dropped every
        queued event. We instead off-load to a worker thread, which gets its own
        fresh loop, so flush works identically in sync and async contexts.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop:
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        return asyncio.run(coro)

    def __enter__(self) -> TelemetryHandler:
        return self

    def __exit__(self, *_: object) -> None:
        self.flush()

    async def _flush_events(self) -> None:
        dlq, self._dlq = self._dlq, []
        new, self._events = self._events, []
        events = dlq + new
        if not events:
            return
        try:
            import httpx

            async with httpx.AsyncClient() as client:
                await self._send_events_with_client(client, events)
        except Exception:
            self._add_to_dlq(events)

    async def _send_events_with_client(self, client: httpx.AsyncClient, events: list[QueuedEvent]) -> None:
        if not events:
            return
        payload = build_payload(
            events,
            source_client_version=self._source_client_version,
            session_id=self._session_id,
        )
        try:
            response = await client.post(_telemetry_endpoint(), json=payload)
            if response.status_code in (400, 422) or response.is_success:
                return
            if response.status_code == 413:
                if len(events) == 1:
                    return
                mid = len(events) // 2
                await self._send_events_with_client(client, events[:mid])
                await self._send_events_with_client(client, events[mid:])
                return
            if response.status_code == 408 or response.status_code >= 500:
                self._add_to_dlq(events)
        except Exception:
            self._add_to_dlq(events)

    def _add_to_dlq(self, events: list[QueuedEvent]) -> None:
        """Bookkeeping for events that hit a retryable failure.

        Note: in anonymizer's fire-and-flush usage (``with TelemetryHandler(...) as h``),
        ``flush()`` runs exactly once per handler lifetime, so DLQ entries are not
        actually retried — they're effectively dropped. The structure is preserved to
        match DataDesigner's handler shape and to support a future long-lived /
        timer-driven usage pattern without restructuring. Telemetry is best-effort.
        """
        for q in events:
            q.retry_count += 1
            if q.retry_count > self._max_retries:
                continue
            self._dlq.append(q)
