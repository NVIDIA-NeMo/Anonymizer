# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A step whose model keeps giving no usable response stops early, through ``Anonymizer.run`` and real DataDesigner.

Only the HTTP client is scripted, per model alias; every Rewrite step, detection included, runs its real columns.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import threading
import time
from collections import Counter
from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from data_designer.config.models import ModelProvider
from data_designer.config.run_config import RunConfig
from data_designer.engine.models.clients import factory as client_factory
from data_designer.engine.models.clients.types import AssistantMessage, ChatCompletionRequest, ChatCompletionResponse

from anonymizer.config.anonymizer_config import AnonymizerConfig, AnonymizerInput, Rewrite
from anonymizer.config.rewrite import PrivacyGoal
from anonymizer.engine.rewrite import evaluate
from anonymizer.interface.anonymizer import Anonymizer
from anonymizer.interface.errors import AnonymizerWorkflowError

_ROLES = {
    "detector": ("detection", "entity_detector"),
    "validator": ("detection", "entity_validator"),
    "augmenter": ("detection", "entity_augmenter"),
    "latent": ("detection", "latent_detector"),
    "replacer": ("replace", "replacement_generator"),
    "domain": ("rewrite", "domain_classifier"),
    "disposer": ("rewrite", "disposition_analyzer"),
    "meaning": ("rewrite", "meaning_extractor"),
    "qa": ("rewrite", "qa_generator"),
    "rewriter": ("rewrite", "rewriter"),
    "evaluator": ("rewrite", "evaluator"),
    "repairer": ("rewrite", "repairer"),
}


def _disposition(count: int) -> str:
    entry = {"sensitivity": "high", "protection_method_suggestion": "replace", "low_sensitivity_reason": None}
    return json.dumps({"sensitivity_disposition": [{"id": i, **entry} for i in range(1, count + 1)]})


_GOOD = {
    "detector": '{"entities": []}',
    "validator": '{"decisions": []}',
    "augmenter": '{"entities": [{"value": "7312", "context_before": "ending in ", '
    '"context_after": " will be charged", "label": "credit_debit_card"}]}',
    "latent": '{"latent_entities": []}',
    "replacer": '{"replacements": [{"original": "7312", "label": "credit_debit_card", "synthetic": "4521"}]}',
    "domain": '{"domain": "FINANCIAL", "domain_confidence": 0.9}',
    "disposer": _disposition(1),
    "meaning": '{"units": [{"id": 1, "aspect": "Charge", "unit": "A card is charged.", "importance": "important"}]}',
    "qa": '{"items": [{"id": 1, "aspect": "Charge", "importance": "important", '
    '"question": "What is charged?", "reference_answer": "A card."}]}',
    "rewriter": '{"rewritten_text": "A card will be charged."}',
    "repairer": '{"rewritten_text": "A card will be charged."}',
}
_EVALUATOR = {
    "privacy": '{"answers": [{"id": 1, "answer": "no", "confidence": 0.9, "reason": "Absent.", "evidence": []}]}',
    "compare": '{"per_item": [{"id": 1, "score": 0.9, "reason": "Close."}]}',
    "quality": '{"answers": [{"id": 1, "answer": "A card."}]}',
}
_PROSE = "I think this is about finance."
_REQUESTS_PER_ROW = 6  # a first request and the corrections and restarts the default run config allows

# Whether request ``n`` (counted per alias) for the record numbered ``record`` gets prose instead of a response.
Failing = Callable[[int, int], bool]


_STOP_WARNING = "Stopped model requests for step _domain"
_REWRITE_ROLES = [role for role, (group, _) in _ROLES.items() if group == "rewrite" and role != "domain"]


@dataclass
class _Traffic:
    """Request timing and who reached the domain step, shared by every scripted client.

    ``latency`` adds seconds before a role's answer. A role in ``gated`` answers only once the domain step has
    stopped (``released``), so the healthy rows are still part-way through their later steps when the stop fires,
    the order a slow production model gives. ``timeout`` only stops a mistake from hanging the suite. ``asked``
    holds the record numbers whose domain request reached the model, ``answered`` those it gave a usable answer.
    """

    latency: dict[str, float] = field(default_factory=dict)
    gated: set[str] = field(default_factory=set)
    released: threading.Event = field(default_factory=threading.Event)
    timeout: float = 30.0
    asked: set[int] = field(default_factory=set)
    answered: set[int] = field(default_factory=set)

    def delay_blocking(self, role: str) -> None:
        if role in self.gated:
            self.released.wait(self.timeout)
        time.sleep(self.latency.get(role, 0.0))

    async def delay(self, role: str) -> None:
        if role in self.gated:
            deadline = time.monotonic() + self.timeout
            while not self.released.is_set() and time.monotonic() < deadline:
                await asyncio.sleep(0.005)
        await asyncio.sleep(self.latency.get(role, 0.0))


class _Release(logging.Handler):
    """Sets ``event`` on the domain step's stop WARNING."""

    def __init__(self, event: threading.Event) -> None:
        super().__init__(logging.WARNING)
        self._event = event

    def emit(self, record: logging.LogRecord) -> None:
        if record.getMessage().startswith(_STOP_WARNING):
            self._event.set()


class _Client:
    """Responds per alias; the failing alias answers prose whenever ``failing`` says so."""

    def __init__(
        self, role: str, failing: dict[str, Failing], calls: Counter[str], lock: threading.Lock, traffic: _Traffic
    ) -> None:
        self._role, self._failing, self._calls, self._lock, self._traffic = role, failing, calls, lock, traffic

    def _plan(self, request: ChatCompletionRequest) -> tuple[str, bool]:
        """The request's prompt, and whether it gets prose."""
        prompt = json.dumps(request.messages, default=str)
        match = re.search(r"Record (\d+):", prompt)
        record = int(match.group(1)) if match else -1
        failing = self._failing.get(self._role)
        with self._lock:
            n = self._calls[self._role]
            self._calls[self._role] += 1
            prose = failing is not None and failing(n, record)
            if self._role == "domain":
                self._traffic.asked.add(record)
                if not prose:
                    self._traffic.answered.add(record)
        return prompt, prose

    def _response(self, prompt: str, prose: bool) -> ChatCompletionResponse:
        if prose:
            return ChatCompletionResponse(message=AssistantMessage(content=_PROSE))
        if self._role == "evaluator":
            kind = "privacy" if "privacy auditor" in prompt else "compare" if "A student took" in prompt else "quality"
            content = _EVALUATOR[kind]
        else:
            content = _GOOD[self._role]
        return ChatCompletionResponse(message=AssistantMessage(content=content))

    def completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        prompt, prose = self._plan(request)
        self._traffic.delay_blocking(self._role)
        return self._response(prompt, prose)

    async def acompletion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        prompt, prose = self._plan(request)
        await self._traffic.delay(self._role)
        return self._response(prompt, prose)

    def supports_chat_completion(self) -> bool:
        return True

    def close(self) -> None:
        return None

    async def aclose(self) -> None:
        return None


@pytest.fixture
def traffic() -> Generator[_Traffic]:
    traffic = _Traffic()
    handler = _Release(traffic.released)
    response_logger = logging.getLogger("anonymizer.ndd.response")
    response_logger.addHandler(handler)
    try:
        yield traffic
    finally:
        response_logger.removeHandler(handler)


@pytest.fixture
def scripted(monkeypatch: pytest.MonkeyPatch, traffic: _Traffic) -> tuple[dict[str, Failing], Counter[str]]:
    failing: dict[str, Failing] = {}
    calls: Counter[str] = Counter()
    lock = threading.Lock()

    def create(model_config: Any, *args: Any, **kwargs: Any) -> _Client:
        return _Client(model_config.alias.removeprefix("scripted-"), failing, calls, lock, traffic)

    monkeypatch.setattr(client_factory, "create_model_client", create)
    return failing, calls


def _model_configs() -> str:
    selected: dict[str, dict[str, str]] = {}
    for role, (group, key) in _ROLES.items():
        selected.setdefault(group, {})[key] = f"scripted-{role}"
    configs = [
        {"alias": f"scripted-{role}", "model": f"scripted-{role}", "provider": "scripted", "skip_health_check": True}
        for role in _ROLES
    ]
    return json.dumps({"selected_models": selected, "model_configs": configs})


def _run(tmp_path: Path, rows: int, *, combined: bool, disable_early_shutdown: bool = False) -> Any:
    csv = tmp_path / "input.csv"
    texts = [f"Record {i}: card ending in 7312 will be charged." for i in range(rows)]
    pd.DataFrame({"text": texts}).to_csv(csv, index=False)
    anonymizer = Anonymizer(
        model_configs=_model_configs(),
        model_providers=[ModelProvider(name="scripted", endpoint="http://scripted.invalid/v1", api_key="unused")],
        artifact_path=tmp_path / "artifacts",
        data_designer_run_config=RunConfig(disable_early_shutdown=disable_early_shutdown),
    )
    goal = PrivacyGoal(protect="Card numbers and other direct identifiers", preserve="That a charge happens")
    rewrite = Rewrite(privacy_goal=goal, use_combined_graph=combined)
    return anonymizer.run(
        config=AnonymizerConfig(rewrite=rewrite, emit_telemetry=False),
        data=AnonymizerInput(source=str(csv), text_column="text"),
    )


@contextlib.contextmanager
def _captured(caplog: pytest.LogCaptureFixture) -> Generator[None]:
    """Capture warnings during ``Anonymizer.run()``, which stops the package's records from propagating."""
    package = logging.getLogger("anonymizer")
    package.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING):
            yield
    finally:
        package.removeHandler(caplog.handler)


def _stop_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    messages = [r.getMessage() for r in dict.fromkeys(caplog.records)]
    return [m for m in messages if m.startswith("Stopped model requests")]


def _assert_partitioned(result: Any, rows: int, reason: str) -> None:
    """Every input row is either returned or listed once in failed_records with ``reason``."""
    failed = [record.record_id for record in result.failed_records]
    assert len(result.dataframe) + len(failed) == rows
    assert len(set(failed)) == len(failed)
    assert {record.reason for record in result.failed_records} <= {reason}


def _returned_records(result: Any) -> set[int]:
    return {int(text.removeprefix("Record ").split(":")[0]) for text in result.dataframe["text"]}


def _assert_returns_the_rows_asked(result: Any, traffic: _Traffic, rows: int) -> None:
    """The rows whose domain step was asked come back, those it answered among them; every other row is listed once
    as refused by the stopped domain step."""
    assert traffic.answered
    assert traffic.answered <= traffic.asked
    assert _returned_records(result) == traffic.asked
    assert len(result.failed_records) == rows - len(traffic.asked)
    _assert_partitioned(result, rows, "stopped_early (_domain)")


_MODES = pytest.mark.parametrize("combined", [False, True], ids=["staged", "combined"])


@_MODES
def test_a_step_that_never_responds_stops_and_lists_the_rows_not_run(
    tmp_path: Path,
    scripted: tuple[dict[str, Failing], Counter[str]],
    traffic: _Traffic,
    caplog: pytest.LogCaptureFixture,
    combined: bool,
) -> None:
    failing, calls = scripted
    failing["domain"] = lambda n, record: True
    rows = 30

    with _captured(caplog):
        result = _run(tmp_path, rows, combined=combined)  # does not raise, though no row may finish

    # Without the stop every row spends its requests: 180. The stop allows about one window of rows.
    assert calls["domain"] < rows * _REQUESTS_PER_ROW
    assert result.failed_records  # the rows after the stop are listed, not dropped
    # The rows asked before the stop are kept with the domain's fallback, like any row whose response never parsed.
    assert _returned_records(result) == traffic.asked
    _assert_partitioned(result, rows, "stopped_early (_domain)")
    assert _stop_warnings(caplog) == [
        "Stopped model requests for step _domain: 10 of its last 10 rows had no usable response, a rate of 1.00 "
        "over shutdown_error_window=10, at or above shutdown_error_rate=0.50. Rows that finished are returned; rows "
        "not run are listed in failed_records as stopped_early (_domain). Check that step's model, or set "
        "disable_early_shutdown=True to run every row."
    ]


@_MODES
def test_a_step_that_stops_responding_returns_the_rows_that_finished(
    tmp_path: Path, scripted: tuple[dict[str, Failing], Counter[str]], traffic: _Traffic, combined: bool
) -> None:
    """The later steps answer only after the domain step stops, so every healthy row is part-way when it does."""
    failing, calls = scripted
    healthy, rows = 20, 60
    failing["domain"] = lambda n, record: record >= healthy  # records 0-19 answer; the rest never do
    traffic.gated.update(_REWRITE_ROLES)
    traffic.latency.update(dict.fromkeys(_REWRITE_ROLES, 0.05))

    result = _run(tmp_path, rows, combined=combined)

    assert traffic.released.is_set()
    _assert_returns_the_rows_asked(result, traffic, rows)
    assert calls["domain"] < healthy + (rows - healthy) * _REQUESTS_PER_ROW


@_MODES
def test_a_stop_keeps_the_rows_past_it_at_scale(
    tmp_path: Path, scripted: tuple[dict[str, Failing], Counter[str]], traffic: _Traffic, combined: bool
) -> None:
    """Half the rows never get a domain answer while every other model is slow."""
    failing, _ = scripted
    healthy, rows = 50, 100
    failing["domain"] = lambda n, record: record >= healthy
    traffic.latency.update({role: 0.05 for role in _ROLES if role != "domain"})

    result = _run(tmp_path, rows, combined=combined)

    assert traffic.released.is_set()
    _assert_returns_the_rows_asked(result, traffic, rows)


@_MODES
@pytest.mark.parametrize(
    "failing_step",
    [
        pytest.param(lambda n, record: n < 2 * _REQUESTS_PER_ROW, id="two-bad-rows-then-recovers"),
        pytest.param(lambda n, record: n % 2 == 0, id="half-of-the-requests"),
    ],
)
def test_a_step_that_recovers_within_the_window_never_stops(
    tmp_path: Path,
    scripted: tuple[dict[str, Failing], Counter[str]],
    caplog: pytest.LogCaptureFixture,
    combined: bool,
    failing_step: Failing,
) -> None:
    failing, _ = scripted
    failing["domain"] = failing_step
    rows = 20

    with _captured(caplog):
        result = _run(tmp_path, rows, combined=combined)

    assert len(result.dataframe) == rows
    assert result.failed_records == []
    assert _stop_warnings(caplog) == []


@_MODES
def test_half_of_the_rows_failing_stops_the_step(
    tmp_path: Path, scripted: tuple[dict[str, Failing], Counter[str]], combined: bool
) -> None:
    failing, calls = scripted
    failing["domain"] = lambda n, record: record % 2 == 1  # odd records never get a usable response
    rows = 30

    result = _run(tmp_path, rows, combined=combined)

    assert result.failed_records
    _assert_partitioned(result, rows, "stopped_early (_domain)")
    assert calls["domain"] < rows // 2 * (1 + _REQUESTS_PER_ROW)


@_MODES
def test_a_dataset_smaller_than_the_window_never_stops(
    tmp_path: Path, scripted: tuple[dict[str, Failing], Counter[str]], combined: bool
) -> None:
    failing, calls = scripted
    failing["domain"] = lambda n, record: True
    rows = 8  # below shutdown_error_window=10

    result = _run(tmp_path, rows, combined=combined)

    assert len(result.dataframe) == rows
    assert result.failed_records == []
    assert calls["domain"] == rows * _REQUESTS_PER_ROW


@_MODES
def test_disable_early_shutdown_never_stops(
    tmp_path: Path, scripted: tuple[dict[str, Failing], Counter[str]], combined: bool
) -> None:
    failing, calls = scripted
    failing["domain"] = lambda n, record: True
    rows = 20

    result = _run(tmp_path, rows, combined=combined, disable_early_shutdown=True)

    assert len(result.dataframe) == rows
    assert result.failed_records == []
    assert calls["domain"] == rows * _REQUESTS_PER_ROW


@_MODES
def test_evaluation_stops_by_step_not_by_model(
    tmp_path: Path,
    scripted: tuple[dict[str, Failing], Counter[str]],
    caplog: pytest.LogCaptureFixture,
    combined: bool,
) -> None:
    """The evaluator model serves three steps; each stops under its own name."""
    failing, _ = scripted
    failing["evaluator"] = lambda n, record: True
    rows = 30

    with _captured(caplog):
        result = _run(tmp_path, rows, combined=combined)

    stopped = [m.removeprefix("Stopped model requests for step ").split(":")[0] for m in _stop_warnings(caplog)]
    assert stopped
    steps = {"_privacy_qa_reanswer", "_quality_qa_reanswer", "_quality_qa_compare"}
    # Combined mode runs every round in one graph and names each round's step by its column.
    assert {step.split("__iteration_")[0] for step in stopped} <= steps
    assert len(stopped) == len(set(stopped))  # one WARNING per stopped step
    assert result.failed_records
    # Each listed row names the one stopped step that refused it.
    assert {record.reason for record in result.failed_records} <= {f"stopped_early ({step})" for step in stopped}


@_MODES
def test_a_dataset_smaller_than_the_window_never_stops_across_repair_rounds(
    tmp_path: Path,
    scripted: tuple[dict[str, Failing], Counter[str]],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    combined: bool,
) -> None:
    """Every row leaks on its first evaluation, so it is repaired, and every re-evaluation is prose. Combined mode
    runs both rounds in one graph; each round still counts on its own, as each staged round is its own run."""
    failing, _ = scripted
    leaked = '{"answers": [{"id": 1, "answer": "yes", "confidence": 0.9, "reason": "Present.", "evidence": []}]}'
    monkeypatch.setitem(_EVALUATOR, "privacy", leaked)
    failing["evaluator"] = lambda n, record: n >= 24  # the first evaluation of 8 rows (3 requests each) is usable
    rows = 8

    with _captured(caplog):
        result = _run(tmp_path, rows, combined=combined)

    assert len(result.dataframe) == rows
    assert result.failed_records == []
    assert _stop_warnings(caplog) == []


@_MODES
def test_a_row_that_fails_for_another_reason_after_a_stop_is_not_stopped_early(
    tmp_path: Path,
    scripted: tuple[dict[str, Failing], Counter[str]],
    monkeypatch: pytest.MonkeyPatch,
    combined: bool,
) -> None:
    """The domain step stops partway, and a later column fails for every row that reaches it. Without the stop the run
    raises because no row finishes; with it, it still raises rather than listing those rows as stopped early."""
    failing, _ = scripted
    failing["domain"] = lambda n, record: n >= 30

    def broken_prompt(row: Any) -> str:
        raise TypeError("a bug in a later column")

    monkeypatch.setattr(evaluate, "_render_privacy_reanswer_prompt", broken_prompt)

    with pytest.raises(AnonymizerWorkflowError):
        _run(tmp_path, 60, combined=combined)


def test_a_disposition_with_the_wrong_entry_count_logs_both_counts(
    tmp_path: Path,
    scripted: tuple[dict[str, Failing], Counter[str]],
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The per-row warning names the count check the last response failed, with counts only."""
    monkeypatch.setitem(_GOOD, "disposer", _disposition(3))  # one entity is supplied

    with _captured(caplog):
        result = _run(tmp_path, 2, combined=False)

    assert len(result.dataframe) == 2
    messages = [r.getMessage() for r in dict.fromkeys(caplog.records)]
    assert [m for m in messages if m.startswith("_raw_sensitivity_disposition unavailable")] == [
        "_raw_sensitivity_disposition unavailable after 6 responses (entry_count_mismatch: 3 entries, 1 expected); "
        "keeping the row."
    ] * 2
