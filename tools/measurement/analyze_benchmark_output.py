#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Analyze joined benchmark measurements and detection artifact sidecars.

Usage:
    uv run python tools/measurement/analyze_benchmark_output.py benchmark-runs/suite-id
    uv run python tools/measurement/analyze_benchmark_output.py benchmark-runs/suite-id --output analysis
    uv run python tools/measurement/analyze_benchmark_output.py benchmark-runs/suite-id --detection-artifacts current.jsonl
    uv run python tools/measurement/analyze_benchmark_output.py benchmark-runs/suite-id --json
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Annotated

import cyclopts
from measurement_tools.benchmark_analysis_io import (
    analyze_benchmark_output as analyze_benchmark_output,
)
from measurement_tools.benchmark_analysis_io import (
    read_jsonl_table as read_jsonl_table,
)
from measurement_tools.benchmark_analysis_io import (
    read_trace_summary_table as read_trace_summary_table,
)
from measurement_tools.benchmark_analysis_io import (
    write_analysis_tables as write_analysis_tables,
)
from measurement_tools.benchmark_analysis_models import (
    BenchmarkOutputAnalysis as BenchmarkOutputAnalysis,
)
from measurement_tools.benchmark_analysis_models import (
    CaseAnalysisRow as CaseAnalysisRow,
)
from measurement_tools.benchmark_analysis_models import (
    GroupAnalysisRow as GroupAnalysisRow,
)
from measurement_tools.benchmark_analysis_models import (
    ModelUsageAnalysisRow as ModelUsageAnalysisRow,
)
from measurement_tools.benchmark_analysis_models import (
    ModelUsageGroupAnalysisRow as ModelUsageGroupAnalysisRow,
)
from measurement_tools.benchmark_analysis_models import (
    _EvaluationRollup as _EvaluationRollup,
)
from measurement_tools.benchmark_group_analysis import (
    build_group_rows as build_group_rows,
)
from measurement_tools.benchmark_model_usage import (
    build_model_usage_group_rows as build_model_usage_group_rows,
)
from measurement_tools.benchmark_model_usage import (
    build_model_usage_rows as build_model_usage_rows,
)
from measurement_tools.cli import LogFormat, configure_logging, log_bad_input
from measurement_tools.tables import AnalysisExportResult as AnalysisExportResult
from measurement_tools.tables import ExportFormat

app = cyclopts.App(help=__doc__)
logger = logging.getLogger("measurement.benchmark_output")


def render_result(result: BenchmarkOutputAnalysis, *, json_output: bool) -> str:
    if json_output:
        return result.model_dump_json(indent=2)
    lines = [
        f"Analyzed {result.case_count} case(s) across {result.group_count} group(s); "
        f"model rows={result.model_usage_count}"
    ]
    for group in result.groups:
        label = (
            f"{group.workload_id}/{group.config_id}/"
            f"{group.experimental_detection_strategy}/{group.experimental_replacement_strategy}"
        )
        lines.append(
            f"- {label}: cases={group.case_count}, median_entities={group.median_final_entity_count}, "
            f"failed_cases={group.failed_case_count}/{group.case_count}, "
            f"median_requests={group.median_observed_total_requests}, median_tokens={group.median_observed_total_tokens}, "
            f"median_input_tok_s={group.median_input_text_tokens_per_pipeline_sec}, "
            f"micro_relaxed_f1={group.micro_entity_relaxed_f1}, "
            f"empty_with_gt={group.total_empty_detection_with_ground_truth_count}, "
            f"median_failed_request_rate={group.median_observed_failed_request_rate}, "
            f"median_aug_new_final={group.median_augmented_new_final_value_count}"
        )
    return "\n".join(lines)


@app.default
def main(
    benchmark_dir: Path,
    *,
    output: Annotated[Path | None, cyclopts.Parameter(("--output", "-o"))] = None,
    detection_artifacts: Annotated[Path | None, cyclopts.Parameter("--detection-artifacts")] = None,
    format: Annotated[ExportFormat, cyclopts.Parameter("--format")] = ExportFormat.parquet,
    json_output: Annotated[bool, cyclopts.Parameter("--json")] = False,
    log_format: Annotated[LogFormat, cyclopts.Parameter("--log-format")] = LogFormat.plain,
) -> None:
    configure_logging(log_format)
    try:
        result = analyze_benchmark_output(benchmark_dir, detection_artifacts=detection_artifacts)
        if output is not None:
            write_analysis_tables(result, output, format)
    except ValueError as exc:
        log_bad_input(logger, str(exc))
        raise SystemExit(125) from exc
    sys.stdout.write(render_result(result, json_output=json_output) + "\n")


if __name__ == "__main__":
    app()
