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


@dataclass
class _Hold:
    """Keeps a failing alias's prose answers back until ``until`` says the rows that should finish have finished.

    DataDesigner shuts the whole run down once enough rows fail, and drops every row still part-way through its
    columns. Which rows were part-way then depends on how the event loop interleaved the requests, so a test that
    needs particular rows to come back holds the failures until those rows have their last answer, plus ``settle``
    seconds for DataDesigner to record them as complete. ``timeout`` only stops a mistake from hanging the suite.
    """

    until: Callable[[Counter[str]], bool]
    settle: float = 0.2
    timeout: float = 30.0
    _deadline: float = field(default=0.0, repr=False)
    _released_at: float | None = field(default=None, repr=False)

    def _remaining(self, answered: Counter[str]) -> float:
        """Seconds still to wait; 0 once released, or once the timeout has passed."""
        now = time.monotonic()
        if not self._deadline:
            self._deadline = now + self.timeout
        if self._released_at is None and self.until(answered):
            self._released_at = now
        if self._released_at is None:
            return 0.0 if now > self._deadline else 0.005
        return max(0.0, self._released_at + self.settle - now)

    async def wait(self, answered: Counter[str]) -> None:
        while delay := self._remaining(answered):
            await asyncio.sleep(delay)

    def wait_blocking(self, answered: Counter[str]) -> None:
        while delay := self._remaining(answered):
            time.sleep(delay)


class _Client:
    """Responds per alias; the failing alias answers prose whenever ``failing`` says so."""

    def __init__(
        self,
        role: str,
        failing: dict[str, Failing],
        calls: Counter[str],
        lock: threading.Lock,
        holds: dict[str, _Hold],
        answered: Counter[str],
    ) -> None:
        self._role, self._failing, self._calls, self._lock = role, failing, calls, lock
        self._hold, self._answered = holds.get(role), answered

    def _plan(self, request: ChatCompletionRequest) -> tuple[str, bool]:
        """The request's prompt, and whether it gets prose."""
        prompt = json.dumps(request.messages, default=str)
        with self._lock:
            n = self._calls[self._role]
            self._calls[self._role] += 1
        match = re.search(r"Record (\d+):", prompt)
        failing = self._failing.get(self._role)
        return prompt, failing is not None and failing(n, int(match.group(1)) if match else -1)

    def _response(self, prompt: str, prose: bool) -> ChatCompletionResponse:
        if prose:
            return ChatCompletionResponse(message=AssistantMessage(content=_PROSE))
        if self._role == "evaluator":
            kind = "privacy" if "privacy auditor" in prompt else "compare" if "A student took" in prompt else "quality"
            content = _EVALUATOR[kind]
        else:
            content = _GOOD[self._role]
        with self._lock:
            self._answered[self._role] += 1
        return ChatCompletionResponse(message=AssistantMessage(content=content))

    def completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        prompt, prose = self._plan(request)
        if prose and self._hold is not None:
            self._hold.wait_blocking(self._answered)
        return self._response(prompt, prose)

    async def acompletion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        prompt, prose = self._plan(request)
        if prose and self._hold is not None:
            await self._hold.wait(self._answered)
        return self._response(prompt, prose)

    def supports_chat_completion(self) -> bool:
        return True

    def close(self) -> None:
        return None

    async def aclose(self) -> None:
        return None


@pytest.fixture
def holds() -> dict[str, _Hold]:
    return {}


@pytest.fixture
def scripted(monkeypatch: pytest.MonkeyPatch, holds: dict[str, _Hold]) -> tuple[dict[str, Failing], Counter[str]]:
    failing: dict[str, Failing] = {}
    calls: Counter[str] = Counter()
    answered: Counter[str] = Counter()  # usable answers returned, per alias
    lock = threading.Lock()

    def create(model_config: Any, *args: Any, **kwargs: Any) -> _Client:
        return _Client(model_config.alias.removeprefix("scripted-"), failing, calls, lock, holds, answered)

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


_MODES = pytest.mark.parametrize("combined", [False, True], ids=["staged", "combined"])


@_MODES
def test_a_step_that_never_responds_stops_and_lists_the_rows_not_run(
    tmp_path: Path,
    scripted: tuple[dict[str, Failing], Counter[str]],
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
    _assert_partitioned(result, rows, "stopped_early (_domain)")
    assert _stop_warnings(caplog) == [
        "Stopped model requests for step _domain: 10 of its last 10 rows had no usable response, a rate of 1.00 "
        "over shutdown_error_window=10, at or above shutdown_error_rate=0.50. Rows that finished are returned; rows "
        "not run are listed in failed_records as stopped_early (_domain). Check that step's model, or set "
        "disable_early_shutdown=True to run every row."
    ]


@_MODES
def test_a_step_that_stops_responding_returns_the_rows_that_finished(
    tmp_path: Path, scripted: tuple[dict[str, Failing], Counter[str]], holds: dict[str, _Hold], combined: bool
) -> None:
    failing, calls = scripted
    healthy, rows = 15, 30
    failing["domain"] = lambda n, record: record >= healthy  # records 0-14 answer; the rest never do
    # The stop shuts the run down and drops rows still part-way through their columns, so the failures wait until the
    # healthy rows have their last answer: the rewrite, or in combined mode the last of the 3 evaluations per row.
    last, per_row = ("evaluator", 3) if combined else ("rewriter", 1)
    holds["domain"] = _Hold(until=lambda answered: answered[last] >= healthy * per_row)

    result = _run(tmp_path, rows, combined=combined)

    assert healthy <= len(result.dataframe) < rows  # the rows that finished come back
    _assert_partitioned(result, rows, "stopped_early (_domain)")
    assert calls["domain"] < healthy + (rows - healthy) * _REQUESTS_PER_ROW


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
    # A refused row names the step its own request was refused by; a row with no trace names every stopped step.
    for record in result.failed_records:
        assert record.reason.startswith("stopped_early (")
        assert set(record.reason.removeprefix("stopped_early (").removesuffix(")").split(", ")) <= set(stopped)
    assert any(record.reason in {f"stopped_early ({step})" for step in stopped} for record in result.failed_records)


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
