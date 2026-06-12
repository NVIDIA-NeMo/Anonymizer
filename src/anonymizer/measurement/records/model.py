# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from anonymizer.measurement._coerce import _coerce_int, _safe_rate, _safe_ratio


def _model_workflow_fields(fields: dict[str, Any], observed_usage: dict[str, int | None]) -> dict[str, Any]:
    return {
        **fields,
        **observed_usage,
        "observed_failed_request_rate": _safe_ratio(
            observed_usage["observed_failed_requests"],
            observed_usage["observed_total_requests"],
        ),
        **_throughput_fields(
            elapsed_sec=cast(float, fields["elapsed_sec"]),
            input_row_count=cast(int, fields["input_row_count"]),
            output_row_count=cast(int | None, fields["output_row_count"]),
            total_tokens=observed_usage["observed_total_tokens"],
            total_requests=observed_usage["observed_total_requests"],
            successful_requests=observed_usage["observed_successful_requests"],
        ),
    }


def _throughput_fields(
    *,
    elapsed_sec: float,
    input_row_count: int | None,
    output_row_count: int | None,
    total_tokens: int | None,
    total_requests: int | None,
    successful_requests: int | None,
) -> dict[str, float | None]:
    return {
        "input_rows_per_sec": _safe_rate(input_row_count, elapsed_sec),
        "output_rows_per_sec": _safe_rate(output_row_count, elapsed_sec),
        "observed_tokens_per_sec": _safe_rate(total_tokens, elapsed_sec),
        "observed_requests_per_sec": _safe_rate(total_requests, elapsed_sec),
        "observed_tokens_per_successful_request": _safe_ratio(total_tokens, successful_requests),
    }


def _row_throughput_fields(
    *,
    elapsed_sec: float,
    input_row_count: int | None,
    output_row_count: int | None,
) -> dict[str, float | None]:
    if input_row_count is not None and input_row_count < 0:
        input_row_count = None
    if output_row_count is not None and output_row_count < 0:
        output_row_count = None
    return {
        "input_rows_per_sec": _safe_rate(input_row_count, elapsed_sec),
        "output_rows_per_sec": _safe_rate(output_row_count, elapsed_sec),
    }


def _summarize_model_usage(model_usage: Mapping[str, Any] | None) -> dict[str, int | None]:
    totals = _empty_model_usage_totals()
    for usage in (model_usage or {}).values():
        if not isinstance(usage, Mapping):
            continue
        _add_model_usage_totals(totals, usage)

    if totals["total_tokens"] == 0:
        totals["total_tokens"] = totals["input_tokens"] + totals["output_tokens"]
    if totals["total_requests"] == 0:
        totals["total_requests"] = totals["successful_requests"] + totals["failed_requests"]

    return {
        "observed_input_tokens": totals["input_tokens"],
        "observed_output_tokens": totals["output_tokens"],
        "observed_total_tokens": totals["total_tokens"],
        "observed_reasoning_tokens": totals["reasoning_tokens"] if totals["has_reasoning_tokens"] else None,
        "observed_successful_requests": totals["successful_requests"],
        "observed_failed_requests": totals["failed_requests"],
        "observed_total_requests": totals["total_requests"],
    }


def _empty_model_usage_totals() -> dict[str, int | bool]:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "reasoning_tokens": 0,
        "has_reasoning_tokens": False,
        "successful_requests": 0,
        "failed_requests": 0,
        "total_requests": 0,
    }


def _add_model_usage_totals(totals: dict[str, int | bool], usage: Mapping[str, Any]) -> None:
    token_usage = usage.get("token_usage")
    if isinstance(token_usage, Mapping):
        totals["input_tokens"] += _coerce_int(token_usage.get("input_tokens"), default=0)
        totals["output_tokens"] += _coerce_int(token_usage.get("output_tokens"), default=0)
        totals["total_tokens"] += _coerce_int(token_usage.get("total_tokens"), default=0)
        if token_usage.get("reasoning_tokens") is not None:
            totals["has_reasoning_tokens"] = True
            totals["reasoning_tokens"] += _coerce_int(token_usage.get("reasoning_tokens"), default=0)

    request_usage = usage.get("request_usage")
    if isinstance(request_usage, Mapping):
        totals["successful_requests"] += _coerce_int(request_usage.get("successful_requests"), default=0)
        totals["failed_requests"] += _coerce_int(request_usage.get("failed_requests"), default=0)
        totals["total_requests"] += _coerce_int(request_usage.get("total_requests"), default=0)
