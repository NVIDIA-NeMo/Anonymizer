# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recording harness for the built-in routes.

``NddAdapter.run_workflow`` is replaced by a scripted function that returns canned frames per DataDesigner
workflow name, so the real facade, routes, graph kernel and measurement code run with no model, detector or
network. Each case drives ``run()``, ``preview()`` and ``evaluate()`` and records what the caller can observe
(frames, failed records, outcomes, measurement stage records) and what ran underneath (the workflow runs the
adapter was asked for, whether each happened inside a graph activation, the number of graph invocations).

The committed recordings in ``recordings/`` are what ``test_parity.py`` compares a fresh recording against. After
a deliberate change to a route's output, regenerate them and review the diff::

    uv run --frozen --group dev python -m tests.parity.harness

Not covered: DataDesigner itself never executes. Output dtypes and column sets are those of the scripted frames
(the scripted adapter returns the seed plus one column per declared column), and the adapter's own work (record
id attachment excepted), JSON storage, failure reasons and ``ndd_workflow`` measurement records are not run.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import pickle
import re
import tempfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd

from anonymizer import Anonymizer
from anonymizer.config.anonymizer_config import (
    AnonymizerConfig,
    EvaluateConfig,
    Rewrite,
    TextRecord,
    TextRecordsInput,
)
from anonymizer.config.replace_strategies import Annotate, Hash, Redact, Substitute
from anonymizer.engine.constants import (
    COL_ANY_HIGH_LEAKED,
    COL_DETECTED_ENTITIES,
    COL_DOMAIN,
    COL_DOMAIN_SUPPLEMENT,
    COL_DOMAIN_SUPPLEMENT_PRIVACY,
    COL_ENTITY_CLASSIFICATION,
    COL_JUDGE_EVALUATION,
    COL_LATENT_ENTITIES,
    COL_LEAKAGE_MASS,
    COL_NEEDS_REPAIR,
    COL_PRIVACY_QA,
    COL_PRIVACY_QA_REANSWER,
    COL_QUALITY_QA,
    COL_QUALITY_QA_COMPARE,
    COL_QUALITY_QA_REANSWER,
    COL_REPLACEMENT_APPLICATION,
    COL_REPLACEMENT_MAP,
    COL_REWRITE_REPLACEMENT_READY,
    COL_REWRITTEN_TEXT,
    COL_REWRITTEN_TEXT_NEXT,
    COL_SENSITIVITY_DISPOSITION,
    COL_TAGGED_TEXT,
    COL_TEXT,
    COL_UTILITY_SCORE,
    COL_WEIGHTED_LEAKAGE_RATE,
    RECORD_ID_COLUMN,
)
from anonymizer.engine.detection.postprocess import EntitySpan, build_tagged_text
from anonymizer.engine.evaluation.detection_judge import DetectionJudgeWorkflow
from anonymizer.engine.evaluation.entity_coverage_judge import EntityCoverageWorkflow
from anonymizer.engine.evaluation.replace.attribute_fidelity_judge import AttributeFidelityJudgeWorkflow
from anonymizer.engine.evaluation.replace.relational_consistency_judge import RelationalConsistencyJudgeWorkflow
from anonymizer.engine.evaluation.replace.type_fidelity_judge import TypeFidelityJudgeWorkflow
from anonymizer.engine.graph_core import execution
from anonymizer.engine.graph_core.observe import _ACTIVATION
from anonymizer.engine.ndd.adapter import FailedRecord, NddAdapter, WorkflowRunResult
from anonymizer.measurement import MeasurementCollector, measurement_session

RECORDINGS = Path(__file__).parent / "recordings"

RECORDS = [
    ("a", "Alice works at Acme."),
    ("b", "The sky is blue."),
    ("c", "Bob lives in Paris."),
    ("d", "Dora paints in Lyon."),
    ("e", "Eve visits Rome."),
]
LEXICON = {
    "Alice": "first_name",
    "Acme": "organization_name",
    "Bob": "first_name",
    "Paris": "city",
    "Dora": "first_name",
    "Lyon": "city",
    "Eve": "first_name",
    "Rome": "city",
}
SYNTHETIC = {
    "Alice": "Maria",
    "Acme": "Globex",
    "Bob": "Carl",
    "Paris": "Oslo",
    "Dora": "Elena",
    "Lyon": "Bergen",
    "Eve": "Nora",
    "Rome": "Turin",
}
PREVIEW_ROWS = 4

_JUDGES = (
    DetectionJudgeWorkflow,
    EntityCoverageWorkflow,
    TypeFidelityJudgeWorkflow,
    RelationalConsistencyJudgeWorkflow,
    AttributeFidelityJudgeWorkflow,
)
_JUDGE_PAYLOADS = {judge.RAW_COL: judge.DEFAULT_PAYLOAD for judge in _JUDGES}


@dataclass(frozen=True)
class Case:
    """One route: its config, how the scripted detector labels text, and what the scripts drop or send to repair."""

    name: str
    config: Callable[[], AnonymizerConfig]
    evaluate: EvaluateConfig = field(default_factory=EvaluateConfig)
    lexicon: dict[str, str] = field(default_factory=lambda: dict(LEXICON))
    drops: dict[str, str] = field(default_factory=dict)  # workflow name -> text marker of the row it drops
    repairs: str | None = None  # text marker of the row whose first evaluation asks for repair
    heals: bool = True  # whether that row passes after one repair; if not, repair runs out
    missed: str | None = None  # text marker of the row the coverage judge reports an entity for


CASES = [
    Case("redact", lambda: AnonymizerConfig(replace=Redact()), drops={"entity-detection": "Bob"}),
    Case("hash", lambda: AnonymizerConfig(replace=Hash()), drops={"replace-judges": "Dora"}),
    Case("annotate", lambda: AnonymizerConfig(replace=Annotate())),
    Case(
        "substitute",
        lambda: AnonymizerConfig(replace=Substitute()),
        EvaluateConfig(compute_detection_validity=True),
        drops={"replace-map-generation": "Bob"},
        missed="Alice",
    ),
    Case(
        "rewrite_staged",
        lambda: AnonymizerConfig(rewrite=Rewrite(max_repair_iterations=2)),
        EvaluateConfig(compute_detection_validity=True),
        drops={"entity-detection": "sky", "latent-entity-detection": "Dora", "rewrite-pipeline": "Alice"},
        repairs="Bob",
        missed="Bob",
    ),
    Case(
        "rewrite_combined",
        lambda: AnonymizerConfig(rewrite=Rewrite(use_combined_graph=True)),
        drops={"rewrite-combined": "Bob", "entity-coverage-judge": "Dora"},
        missed="Alice",
    ),
    Case(
        "rewrite_custom_label",
        lambda: AnonymizerConfig(rewrite=Rewrite(max_repair_iterations=1)),
        drops={"rewrite-evaluate-0": "Bob"},
        lexicon={**LEXICON, "Acme": "workplace_unit"},
        repairs="Alice",
        heals=False,
    ),
]


# ---------------------------------------------------------------------------
# The scripted adapter
# ---------------------------------------------------------------------------


def _spans(text: str, lexicon: dict[str, str]) -> list[EntitySpan]:
    found = sorted((text.find(value), value, label) for value, label in lexicon.items() if value in text)
    return [
        EntitySpan(f"{label}_{start}_{start + len(value)}", value, label, start, start + len(value), 0.9, "detector")
        for start, value, label in found
    ]


def _entities(text: str, lexicon: dict[str, str]) -> dict[str, Any]:
    return {"entities": [span.as_dict() for span in _spans(text, lexicon)]}


def _rewrite(text: str) -> str:
    for value, synthetic in SYNTHETIC.items():
        text = text.replace(value, synthetic)
    return text


class Script:
    """Canned frames per workflow name; every call is appended to ``log``."""

    def __init__(self, case: Case) -> None:
        self.case = case
        self.log: list[dict[str, Any]] = []
        self.texts: dict[str, str] = {}  # record id -> original text, learned from the first seed that carries both

    def _text(self, row: dict[str, Any]) -> str:
        return self.texts.get(str(row[RECORD_ID_COLUMN]), "")

    def run_workflow(
        self,
        adapter: NddAdapter,
        df: pd.DataFrame,
        *,
        columns: list[Any],
        workflow_name: str,
        preview_num_records: int | None = None,
        **_: Any,
    ) -> WorkflowRunResult:
        seed = adapter._attach_record_ids(df).reset_index(drop=True)
        if COL_TEXT in seed.columns:
            self.texts.update(zip(seed[RECORD_ID_COLUMN].astype(str), seed[COL_TEXT].astype(str)))
        if preview_num_records is not None:
            seed = seed.head(preview_num_records)
        marker = self.case.drops.get(workflow_name)
        kept = [i for i, row in enumerate(seed.to_dict("records")) if not (marker and marker in self._text(row))]
        out = seed.iloc[kept].reset_index(drop=True)
        rows = out.to_dict("records")
        values = self._canned(workflow_name, rows)
        for column in columns:
            if getattr(column, "drop", False):
                continue
            for name in (column.name, *getattr(column, "side_effect_columns", ())):
                out[name] = pd.Series(values.get(name, [None] * len(rows)), index=out.index)
        failed = [
            FailedRecord(record_id=str(seed.iloc[i][RECORD_ID_COLUMN]), step=workflow_name, reason="Record missing")
            for i in range(len(seed))
            if i not in kept
        ]
        self.log.append(
            {
                "workflow": workflow_name,
                "seed_columns": list(seed.columns),
                "seed_rows": len(seed),
                "output_rows": len(out),
                "preview_num_records": preview_num_records,
                "inside_activation": _ACTIVATION.get() is not None,
            }
        )
        return WorkflowRunResult(dataframe=out, failed_records=failed, source_positions=tuple(kept))

    def _canned(self, workflow: str, rows: list[dict[str, Any]]) -> dict[str, list[Any]]:
        case, texts = self.case, [self._text(row) for row in rows]
        iteration = int(workflow.rsplit("-", 1)[1]) if workflow[-1].isdigit() else 0
        repairing = [bool(case.repairs and case.repairs in text) for text in texts]
        if workflow == "entity-detection":
            return {
                COL_DETECTED_ENTITIES: [_entities(text, case.lexicon) for text in texts],
                COL_TAGGED_TEXT: [build_tagged_text(text=text, entities=_spans(text, case.lexicon)) for text in texts],
            }
        if workflow == "latent-entity-detection":
            latent = {
                "category": "latent_identifier",
                "label": "employer",
                "value": "a company",
                "confidence": "medium",
                "evidence": ["works at"],
                "rationale": "The text states where the person works, which implies an employer.",
            }
            return {COL_LATENT_ENTITIES: [{"latent_entities": [latent] if "works" in text else []} for text in texts]}
        if workflow == "rewrite-label-classification":
            labels = rows[0]["_entity_classification_unmapped_labels"]
            local = rows[0]["_entity_classification_local"]
            return {COL_ENTITY_CLASSIFICATION: [{**local, **dict.fromkeys(labels, "quasi_identifier")}]}
        if workflow == "replace-map-generation":
            return {COL_REPLACEMENT_MAP: [_replacement_map(text, case.lexicon) for text in texts]}
        if workflow in ("rewrite-pipeline", "rewrite-combined"):
            return {
                COL_DOMAIN: [{"domain": "OTHER", "domain_confidence": 0.9}] * len(rows),
                COL_DOMAIN_SUPPLEMENT: ["supplement"] * len(rows),
                COL_DOMAIN_SUPPLEMENT_PRIVACY: ["privacy supplement"] * len(rows),
                COL_SENSITIVITY_DISPOSITION: [{"sensitivity_disposition": []}] * len(rows),
                COL_QUALITY_QA: [{"items": []}] * len(rows),
                COL_PRIVACY_QA: [{"items": []}] * len(rows),
                COL_REWRITTEN_TEXT: [_rewrite(text) for text in texts],
                COL_REWRITE_REPLACEMENT_READY: [True] * len(rows),
                COL_REPLACEMENT_APPLICATION: [_application(text, case.lexicon) for text in texts],
                **_evaluation(repairing, iteration=1 if workflow == "rewrite-combined" else None, heals=case.heals),
            }
        if workflow.startswith("rewrite-evaluate-"):
            return _evaluation(repairing, iteration=iteration, heals=case.heals)
        if workflow.startswith("rewrite-repair-"):
            return {
                "_leaked_privacy_items": [f"leaked items {iteration}"] * len(rows),
                COL_REWRITTEN_TEXT_NEXT: [_rewrite(text) + " (repaired)" for text in texts],
            }
        if workflow == "rewrite-final-judge":
            judgment = {name: {"score": 4, "reasoning": "ok"} for name in ("privacy", "quality", "style")}
            return {COL_JUDGE_EVALUATION: [judgment] * len(rows)}
        if workflow in ("replace-judges", "replace-detection-judge", "entity-coverage-judge"):
            payloads = {raw: [dict(payload)] * len(rows) for raw, payload in _JUDGE_PAYLOADS.items()}
            payloads["_entity_coverage_judge"] = [
                {"candidate_entities": [{"value": "works", "label": "occupation", "reasoning": "a job description"}]}
                if case.missed and case.missed in text
                else {"candidate_entities": []}
                for text in texts
            ]
            return payloads
        raise AssertionError(f"no canned frame for workflow {workflow!r}")


def _application(text: str, lexicon: dict[str, str]) -> dict[str, Any]:
    spans = len(_spans(text, lexicon))
    # The generator stores the empty label mapping as a JSON string, which the workflow restores.
    return {
        "targeted_span_count": spans,
        "applied_span_count": spans,
        "skipped_span_count": 0,
        "skipped_span_label_counts": "{}",
    }


def _replacement_map(text: str, lexicon: dict[str, str]) -> dict[str, Any]:
    return {
        "replacements": [
            {"original": e.value, "label": e.label, "synthetic": SYNTHETIC[e.value]} for e in _spans(text, lexicon)
        ]
    }


def _evaluation(repairing: list[bool], *, iteration: int | None, heals: bool) -> dict[str, list[Any]]:
    """One evaluation's metric columns; a row marked for repair fails every evaluation before the last allowed."""
    failing = [flag and (not heals or (iteration or 0) < 1) for flag in repairing]
    n = len(repairing)
    return {
        COL_QUALITY_QA_REANSWER: [{"answers": []}] * n,
        COL_PRIVACY_QA_REANSWER: [{"answers": []}] * n,
        COL_QUALITY_QA_COMPARE: [{"results": []}] * n,
        COL_UTILITY_SCORE: [0.9] * n,
        COL_LEAKAGE_MASS: [2.0 if flag else 0.0 for flag in failing],
        COL_WEIGHTED_LEAKAGE_RATE: [0.0] * n,
        COL_ANY_HIGH_LEAKED: failing,
        COL_NEEDS_REPAIR: failing,
    }


@contextlib.contextmanager
def scripted(case: Case) -> Iterator[tuple[Script, list[int]]]:
    """Patch the adapter and count graph invocations; yields the script and the invocation counter."""
    script, invocations = Script(case), [0]
    original = execution._protect

    def counted(prepared: Any) -> Any:
        invocations[0] += 1
        return original(prepared)

    def run_workflow(adapter: NddAdapter, df: pd.DataFrame, **kwargs: Any) -> WorkflowRunResult:
        return script.run_workflow(adapter, df, **kwargs)

    with patch.object(NddAdapter, "run_workflow", run_workflow), patch.object(execution, "_protect", counted):
        yield script, invocations


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------

_TIMING_FIELDS = {"elapsed_sec", "timestamp_unix_sec", "run_id"}


def _cell(value: Any) -> Any:
    """A JSON-safe, lossless-enough form of one cell: containers recurse, NaN and NA are tagged, numpy unwraps."""
    if isinstance(value, dict):
        return {str(k): _cell(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)) or hasattr(value, "tolist") and not hasattr(value, "item"):
        return [_cell(v) for v in (value.tolist() if hasattr(value, "tolist") else value)]
    if value is None:
        return None
    if value is pd.NA:
        return {"$": "NA"}
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and value != value:
        return {"$": "NaN"}
    if isinstance(value, (bool, int, float, str)):
        return value
    return {"$": f"{type(value).__name__}:{value}"}


def frame_record(frame: pd.DataFrame) -> dict[str, Any]:
    """Column order, dtypes (by column name), and one dict per row in row order."""
    return {
        "columns": [str(c) for c in frame.columns],
        "dtypes": {str(c): str(t) for c, t in frame.dtypes.items()},
        "rows": [{str(c): _cell(v) for c, v in row.items()} for row in frame.to_dict("records")],
    }


def _stage_record(record: dict[str, Any]) -> dict[str, Any]:
    """A stage record with the fields that vary per run replaced by a tag, so only their presence is compared."""
    return {k: ("<varies>" if k in _TIMING_FIELDS or k.endswith("_per_sec") else _cell(v)) for k, v in record.items()}


def _outcomes(result: Any) -> list[dict[str, Any]] | None:
    try:
        outcomes = [result.outcome(i) for i in range(len(result.trace_dataframe))]
    except IndexError:
        return None
    return [
        {
            "terminal": o.terminal,
            "qualification": o.qualification,
            "available": o.available,
            "protected": o.protected,
            "reasons": [_cell(vars(r)) if hasattr(r, "__dict__") else str(r) for r in o.reasons],
        }
        for o in outcomes
    ]


def _consumers(result: Any) -> dict[str, Any]:
    """What the trace's consumers make of a result: display, Parquet, pickle and the user/trace column relation."""
    printed: list[str] = []
    with patch("builtins.print", lambda *args, **_: printed.append(" ".join(map(str, args)))):
        with patch.dict("sys.modules", {"IPython": None, "IPython.display": None}):
            if len(result.trace_dataframe):  # an empty result still compares, as a diff
                result.display_record(0)
    # Label colours come from hash(label), which varies per process: normalize them.
    html = re.sub(r"#[0-9a-fA-F]{6}", "#colour", "".join(printed))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "user.parquet"
        result.dataframe.to_parquet(path)
        saved = pd.read_parquet(path)
    restored = pickle.loads(pickle.dumps(result))  # noqa: S301
    return {
        "display_record": {"chars": len(html), "sha256": hashlib.sha256(html.encode()).hexdigest()[:16]},
        "parquet": {"columns": list(saved.columns), "dtypes": {c: str(t) for c, t in saved.dtypes.items()}},
        "pickle": {
            "trace_equal": restored.trace_dataframe.equals(result.trace_dataframe),
            "user_equal": restored.dataframe.equals(result.dataframe),
            "outcomes_kept": _outcomes(restored) == _outcomes(result),
        },
        "user_columns_in_trace": sorted(set(result.dataframe.columns) - set(result.trace_dataframe.columns)) == [],
    }


def _observe(call: Callable[[], Any], case: Case) -> tuple[dict[str, Any], Any]:
    collector = MeasurementCollector(record_hash_key="parity")
    with scripted(case) as (script, invocations), measurement_session(collector):
        result = call()
    records = collector.records
    return {
        "user_frame": frame_record(result.dataframe),
        "trace_frame": frame_record(result.trace_dataframe),
        "failed_records": [
            {"record_id": f.record_id, "step": f.step, "reason": f.reason} for f in result.failed_records
        ],
        "outcomes": _outcomes(result),
        "stages": [_stage_record(r) for r in records if r["record_type"] == "stage"],
        "record_metrics": [_stage_record(r) for r in records if r["record_type"] == "record"],
        "run_record_fields": sorted({k for r in records if r["record_type"] == "run" for k in r} - _TIMING_FIELDS),
        "workflow_runs": script.log,
        "graph_invocations": invocations[0],
        "consumers": _consumers(result),
    }, result


def record_case(case: Case) -> dict[str, Any]:
    """Drive run(), preview() and evaluate() of one case and return everything observed."""
    data = TextRecordsInput(records=[TextRecord(id=i, text=t) for i, t in RECORDS])
    config = case.config()
    with tempfile.TemporaryDirectory() as artifacts:
        anonymizer = Anonymizer(artifact_path=artifacts)
        run_record, run_result = _observe(lambda: anonymizer.run(config=config, data=data), case)
        preview_record, _ = _observe(
            lambda: anonymizer.preview(config=config, data=data, num_records=PREVIEW_ROWS), case
        )
        evaluate_record, _ = _observe(lambda: anonymizer.evaluate(run_result, config=case.evaluate), case)
    return {"run": run_record, "preview": preview_record, "evaluate": evaluate_record}


def recording_path(case: Case) -> Path:
    return RECORDINGS / f"{case.name}.json"


def _dump(value: Any, indent: int = 0) -> str:
    """JSON with one line per mapping entry and per list element, so a changed row is a one-line diff."""
    pad, close = " " * (indent + 1), " " * indent
    if isinstance(value, dict) and value:
        return (
            "{\n"
            + ",\n".join(f"{pad}{json.dumps(k)}: {_dump(v, indent + 1)}" for k, v in value.items())
            + f"\n{close}}}"
        )
    if isinstance(value, list) and any(isinstance(v, (dict, list)) for v in value):
        return "[\n" + ",\n".join(pad + json.dumps(v) for v in value) + f"\n{close}]"
    return json.dumps(value)


def regenerate() -> None:
    RECORDINGS.mkdir(exist_ok=True)
    for case in CASES:
        recording_path(case).write_text(_dump(record_case(case)) + "\n")
        print(f"wrote {recording_path(case)}")


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def _leaves(value: Any, path: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        return {k: v for key, item in value.items() for k, v in _leaves(item, f"{path}.{key}" if path else key).items()}
    if isinstance(value, list):
        return {k: v for i, item in enumerate(value) for k, v in _leaves(item, f"{path}[{i}]").items()}
    return {path: value}


def differences(recorded: Any, fresh: Any) -> list[str]:
    """One line per leaf that differs, named by its path (``run.user_frame.rows[1].text_replaced`` and so on)."""
    old, new = _leaves(recorded), _leaves(fresh)
    absent = "<absent>"
    return [
        f"{path}: recorded {old.get(path, absent)!r}, now {new.get(path, absent)!r}"
        for path in dict.fromkeys([*old, *new])
        if path not in old or path not in new or old[path] != new[path]
    ]


if __name__ == "__main__":
    regenerate()
