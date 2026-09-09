# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import json
import subprocess
import sys
import zipfile
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import cast
from unittest.mock import patch

import pandas as pd
import pytest

import anonymizer
import anonymizer.interface.anonymizer as facade
from anonymizer.config.anonymizer_config import AnonymizerConfig, AnonymizerInput, Rewrite
from anonymizer.config.replace_strategies import Redact
from anonymizer.engine.constants import COL_ENTITY_COVERAGE, COL_REWRITTEN_TEXT, COL_TEXT
from anonymizer.engine.ndd.adapter import FailedRecord
from anonymizer.engine.rewrite.rewrite_workflow import RewriteResult
from anonymizer.interface import _result_compatibility as compatibility
from anonymizer.interface.results import AnonymizerResult
from anonymizer.telemetry import TaskEnum, TaskStatusEnum
from tests.interface.test_anonymizer_interface import _make_anonymizer

_REFERENCE_DIRECTORY = Path(__file__).parent / "reference_models"
_MUTATION_MANIFEST_PATH = _REFERENCE_DIRECTORY / "phase9_result_compatibility_v1_mutations.json"
_MUTATION_DIGEST = "951d5dc36c7619507bea6c1ec90305df749048af9872099fb42ef4fb1208b2d4"
_CONTRACT_PATH = Path(compatibility.__file__).with_name("result_compatibility_contract.json")
_MATERIALIZER_PATH = Path(compatibility.__file__)


def _load_mutant(
    tmp_path: Path,
    mutation_id: str,
    old: str,
    new: str,
    *,
    source_path: Path = _MATERIALIZER_PATH,
) -> ModuleType:
    source = source_path.read_text(encoding="utf-8")
    assert source.count(old) == 1, mutation_id
    mutant_path = tmp_path / f"phase9_mutant_{mutation_id.replace('-', '_')}.py"
    mutant_path.write_text(source.replace(old, new), encoding="utf-8")
    module_name = f"phase9_mutant_{mutation_id.replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, mutant_path)
    if spec is None or spec.loader is None:  # pragma: no cover - importlib defensive guard
        raise RuntimeError("could not load Phase 9 mutant")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _projection_observation(module: ModuleType) -> tuple[object, ...]:
    nested = {"entities": [{"value": "Alice"}]}
    index = pd.Index([3, 1, 3], dtype="Int64", name="source-row")
    source = pd.DataFrame(
        {
            "entity_coverage": pd.Series([0.5, None, 1.0], index=index, dtype="Float64"),
            "__nemo_anonymizer_text_output__": pd.Series(["A", None, "B"], index=index, dtype="string"),
            "__nemo_anonymizer_text_input__": pd.Series(["a", None, "b"], index=index, dtype="string"),
            "tagged_text": pd.Series(["a", None, "b"], index=index, dtype="object"),
            "final_entities": pd.Series([nested, {"entities": []}, nested], index=index, dtype="object"),
            "_rewritten_text": pd.Series(["person a", None, "person b"], index=index, dtype="string"),
            "utility_score": pd.Series([0.9, None, 0.8], index=index, dtype="Float64"),
            "ignored": pd.Series([1, None, 3], index=index, dtype="Int64"),
        },
        index=index,
    )
    source.attrs.update({"dataset": {"kind": "mutation-witness"}})
    trace = module._rename_output_columns(source, resolved_text_column="bio")
    public = module._build_user_dataframe(trace, resolved_text_column="bio")
    return (
        tuple(trace.columns),
        tuple(public.columns),
        tuple(str(dtype) for dtype in public.dtypes),
        tuple(public.index.tolist()),
        type(public.index),
        tuple(public.index.names),
        public.attrs,
        public is trace,
        public.iloc[0].to_dict(),
    )


_PROJECTION_MUTANTS = [
    pytest.param(
        "column-order",
        "return trace[[column for column in trace.columns if column in allowed]].copy()",
        "return trace[sorted(column for column in trace.columns if column in allowed)].copy()",
        id="column-order",
    ),
    pytest.param(
        "index-reset",
        "return trace[[column for column in trace.columns if column in allowed]].copy()",
        "return trace[[column for column in trace.columns if column in allowed]].copy().reset_index(drop=True)",
        id="index-reset",
    ),
    pytest.param(
        "dtype-coercion",
        "return trace[[column for column in trace.columns if column in allowed]].copy()",
        'return trace[[column for column in trace.columns if column in allowed]].copy().astype("object")',
        id="dtype-coercion",
    ),
    pytest.param(
        "attrs-drop",
        "return trace[[column for column in trace.columns if column in allowed]].copy()",
        (
            "result = trace[[column for column in trace.columns if column in allowed]].copy()\n"
            "    result.attrs.clear()\n"
            "    return result"
        ),
        id="attrs-drop",
    ),
    pytest.param(
        "copy-aliasing",
        "return trace[[column for column in trace.columns if column in allowed]].copy()",
        "return trace",
        id="copy-aliasing",
    ),
    pytest.param(
        "rename-source",
        'rename_map[COL_REPLACED_TEXT] = f"{resolved_text_column}_replaced"',
        'rename_map[COL_REPLACED_TEXT] = f"{resolved_text_column}_replacement"',
        id="rename-source",
    ),
    pytest.param(
        "mode-projection",
        'if f"{text_column}_rewritten" in trace.columns:',
        'if False and f"{text_column}_rewritten" in trace.columns:',
        id="mode-projection",
    ),
]


@pytest.mark.parametrize(("mutation_id", "old", "new"), _PROJECTION_MUTANTS)
def test_executable_projection_mutants_are_killed(
    tmp_path: Path,
    mutation_id: str,
    old: str,
    new: str,
) -> None:
    mutant = _load_mutant(tmp_path, mutation_id, old, new)
    expected = (
        (
            "entity_coverage",
            "bio_replaced",
            "bio",
            "bio_with_spans",
            "final_entities",
            "bio_rewritten",
            "utility_score",
            "ignored",
        ),
        ("entity_coverage", "bio", "bio_rewritten", "utility_score"),
        ("Float64", "string", "string", "Float64"),
        (3, 1, 3),
        pd.Index,
        ("source-row",),
        {"dataset": {"kind": "mutation-witness"}},
        False,
        {
            "entity_coverage": 0.5,
            "bio": "a",
            "bio_rewritten": "person a",
            "utility_score": 0.9,
        },
    )

    assert _projection_observation(compatibility) == expected
    with pytest.raises(AssertionError):
        assert _projection_observation(mutant) == expected


def _run_rewrite_factory(module: ModuleType) -> object:
    config = AnonymizerConfig(rewrite=Rewrite())
    source = pd.DataFrame(
        {
            "__nemo_anonymizer_text_input__": ["Alice"],
            "_rewritten_text": ["A person"],
        }
    )
    return module._materialize_run_result(
        source,
        config=config,
        resolved_text_column="text",
        failed_records=[],
        data_summary=None,
    )


def _rewrite_metadata_is_exact(module: ModuleType) -> bool:
    config = AnonymizerConfig(rewrite=Rewrite())
    source = pd.DataFrame(
        {
            "__nemo_anonymizer_text_input__": ["Alice"],
            "_rewritten_text": ["A person"],
        }
    )
    result = module._materialize_run_result(
        source,
        config=config,
        resolved_text_column="text",
        failed_records=[],
        data_summary=None,
    )
    return config.rewrite is not None and result.rewrite_config is config.rewrite.privacy_goal


def _preview_count(module: ModuleType) -> int:
    config = AnonymizerConfig(rewrite=Rewrite())
    run_result = _run_rewrite_factory(compatibility)
    return cast(
        int,
        module._materialize_preview_result(run_result, config=config, preview_num_records=10).preview_num_records,
    )


_FACTORY_MUTANTS: list[tuple[str, str, str, Callable[[ModuleType], object], object]] = [
    (
        "rewrite-metadata",
        (
            "failed_records=failed_records,\n"
            "        replace_method=config.replace,\n"
            "        rewrite_config=config.rewrite.privacy_goal if config.rewrite is not None else None,"
        ),
        (
            "failed_records=failed_records,\n"
            "        replace_method=config.replace,\n"
            "        rewrite_config=config.rewrite.evaluation if config.rewrite is not None else None,"
        ),
        _rewrite_metadata_is_exact,
        True,
    ),
    (
        "preview-count",
        "preview_num_records=preview_num_records,",
        "preview_num_records=len(result.dataframe),",
        _preview_count,
        10,
    ),
]


@pytest.mark.parametrize(("mutation_id", "old", "new", "observe", "expected"), _FACTORY_MUTANTS)
def test_executable_factory_mutants_are_killed(
    tmp_path: Path,
    mutation_id: str,
    old: str,
    new: str,
    observe: Callable[[ModuleType], object],
    expected: object,
) -> None:
    mutant = _load_mutant(tmp_path, mutation_id, old, new)

    assert observe(compatibility) == expected
    with pytest.raises(AssertionError):
        assert observe(mutant) == expected


_PACKAGE_ROOT = _MATERIALIZER_PATH.parents[1]
_RESULTS_PATH = _MATERIALIZER_PATH.with_name("results.py")
_FACADE_PATH = _MATERIALIZER_PATH.with_name("anonymizer.py")
_PUBLIC_PACKAGE_PATH = _PACKAGE_ROOT / "__init__.py"


def _mutate_source(path: Path, mutation_id: str, old: str, new: str) -> tuple[str, str]:
    source = path.read_text(encoding="utf-8")
    assert source.count(old) == 1, mutation_id
    return source, source.replace(old, new)


def _class_fields(source: str, class_name: str) -> list[str]:
    tree = ast.parse(source)
    class_node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    return [
        node.target.id
        for node in class_node.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    ]


def _assert_public_field_contract(source: str) -> None:
    assert _class_fields(source, "AnonymizerResult") == [
        "dataframe",
        "trace_dataframe",
        "resolved_text_column",
        "failed_records",
        "replace_method",
        "rewrite_config",
        "entity_labels",
        "data_summary",
        "_display_cycle_index",
    ]


def _assert_evaluation_failure_contract(source: str) -> None:
    assert "all_failed: list[FailedRecord] = list(rewrite_result.failed_records)" in source
    assert "all_failed.extend(coverage_failed)" in source
    assert "failed_records=replace_result.failed_records," in source


def _assert_telemetry_count_contract(source: str) -> None:
    assert "failure_count = len(failed)" in source
    assert "success_count = max(total_records - failure_count, 0)" in source


def _assert_exception_contract(source: str) -> None:
    assert '_PUBLIC_PIPELINE_FAILURE_MESSAGE = "Anonymization pipeline failed."' in source
    assert source.count("raise public_error from None") == 2


def _assert_materialization_order(source: str) -> None:
    method = source[source.index("    def _run_internal_impl(") : source.index("    def _validate_preflight_config(")]
    assert method.index(".run(") < method.index("record_record_metrics(") < method.index("_materialize_run_result(")


def _assert_no_direct_data_designer_execution(sources: dict[Path, str]) -> None:
    execution_sites: set[Path] = set()
    for path, source in sources.items():
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr in {"create", "preview"} and "data_designer" in ast.unparse(node.func.value):
                execution_sites.add(path)
    assert execution_sites == {Path("engine/ndd/adapter.py")}


def _public_exports(source: str) -> set[str]:
    tree = ast.parse(source)
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets)
    )
    return {
        item.value
        for item in cast(ast.List, assignment.value).elts
        if isinstance(item, ast.Constant) and isinstance(item.value, str)
    }


def _assert_no_private_public_exports(source: str) -> None:
    assert {
        "_materialize_run_result",
        "_materialize_preview_result",
        "_materialize_evaluation_result",
    }.isdisjoint(_public_exports(source))


def _assert_wheel_artifacts(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
    assert {
        "anonymizer/interface/_result_compatibility.py",
        "anonymizer/interface/result_compatibility_contract.json",
    }.issubset(names)


def test_remaining_contract_mutants_are_killed_public_field_drift() -> None:
    source, mutant = _mutate_source(
        _RESULTS_PATH,
        "public-field-drift",
        "class AnonymizerResult(_DisplayMixin):\n",
        "class AnonymizerResult(_DisplayMixin):\n    contract_version: str = 'v2'\n",
    )

    _assert_public_field_contract(source)
    with pytest.raises(AssertionError):
        _assert_public_field_contract(mutant)


def test_remaining_contract_mutants_are_killed_evaluation_failures(tmp_path: Path) -> None:
    mutant = _load_mutant(
        tmp_path,
        "evaluation-failures",
        "all_failed: list[FailedRecord] = list(rewrite_result.failed_records)",
        "all_failed: list[FailedRecord] = [*output.failed_records, *rewrite_result.failed_records]",
        source_path=_FACADE_PATH,
    )
    rewrite = Rewrite()
    prior = FailedRecord(record_id="prior", step="run", reason="prior")
    current = FailedRecord(record_id="current", step="judge", reason="current")
    judged = pd.DataFrame(
        {
            COL_TEXT: ["Alice"],
            COL_REWRITTEN_TEXT: ["A person"],
            COL_ENTITY_COVERAGE: [None],
        }
    )

    def observe(module: ModuleType) -> list[str]:
        anonymizer_instance, _, _, rewrite_runner = _make_anonymizer()
        rewrite_runner.evaluate.return_value = RewriteResult(dataframe=judged, failed_records=[current])
        output = AnonymizerResult(
            dataframe=pd.DataFrame(),
            trace_dataframe=judged,
            resolved_text_column="text",
            failed_records=[prior],
            rewrite_config=rewrite.privacy_goal,
        )
        with patch.object(module, "EntityCoverageWorkflow") as coverage_workflow:
            coverage_workflow.return_value.run_non_critical.return_value = (judged, [])
            result = module.Anonymizer.evaluate(anonymizer_instance, output)
        return [record.record_id for record in result.failed_records]

    assert observe(facade) == ["current"]
    with pytest.raises(AssertionError):
        assert observe(mutant) == ["current"]


def test_remaining_contract_mutants_are_killed_telemetry_counts(tmp_path: Path) -> None:
    mutant = _load_mutant(
        tmp_path,
        "telemetry-counts",
        "failure_count = len(failed)",
        "failure_count = len({record.record_id for record in failed})",
        source_path=_FACADE_PATH,
    )
    source = tmp_path / "telemetry.csv"
    pd.DataFrame({"text": ["a", "b"]}).to_csv(source, index=False)
    failures = [
        FailedRecord(record_id="opaque-a", step="unknown", reason="unavailable"),
        FailedRecord(record_id="opaque-a", step="unknown", reason="unavailable"),
        FailedRecord(record_id="opaque-b", step="unknown", reason="unavailable"),
    ]

    def observe(module: ModuleType) -> tuple[int, int]:
        anonymizer_instance, *_ = _make_anonymizer()
        result = AnonymizerResult(pd.DataFrame(), pd.DataFrame(), "text", failures)
        event = module.Anonymizer._build_telemetry_event(
            anonymizer_instance,
            task=TaskEnum.BATCH,
            status=TaskStatusEnum.COMPLETED,
            config=AnonymizerConfig(replace=Redact()),
            data=AnonymizerInput(source=str(source)),
            input_df=pd.DataFrame({COL_TEXT: ["a", "b"]}),
            result=result,
            duration_sec=0.0,
        )
        return event.num_failure_records, event.num_success_records

    assert observe(facade) == (3, 0)
    with pytest.raises(AssertionError):
        assert observe(mutant) == (3, 0)


def test_remaining_contract_mutants_are_killed_exception_drift(tmp_path: Path) -> None:
    mutant = _load_mutant(
        tmp_path,
        "exception-drift",
        '_PUBLIC_PIPELINE_FAILURE_MESSAGE = "Anonymization pipeline failed."',
        '_PUBLIC_PIPELINE_FAILURE_MESSAGE = "private provider failure"',
        source_path=_FACADE_PATH,
    )
    source = tmp_path / "exception.csv"
    pd.DataFrame({"text": ["Alice"]}).to_csv(source, index=False)

    def observe(module: ModuleType) -> tuple[str, str, object]:
        anonymizer_instance, detection_workflow, _, _ = _make_anonymizer()
        detection_workflow.run.side_effect = RuntimeError("private provider failure")
        caught: BaseException | None = None
        try:
            module.Anonymizer.run(
                anonymizer_instance,
                config=AnonymizerConfig(replace=Redact()),
                data=AnonymizerInput(source=str(source)),
            )
        except BaseException as exc:  # noqa: BLE001 - witness records exact public error
            caught = exc
        assert caught is not None
        return type(caught).__name__, str(caught), caught.__cause__

    assert observe(facade) == ("AnonymizerWorkflowError", "Anonymization pipeline failed.", None)
    with pytest.raises(AssertionError):
        assert observe(mutant) == ("AnonymizerWorkflowError", "Anonymization pipeline failed.", None)


def test_remaining_contract_mutants_are_killed_graph_admission(tmp_path: Path) -> None:
    mutant = _load_mutant(
        tmp_path,
        "graph-admission",
        "def _require_dataframe(value: object) -> pd.DataFrame:\n",
        (
            "def _require_dataframe(value: object) -> pd.DataFrame:\n"
            "    if type(value).__name__ == '_GraphProtectionResult':\n"
            "        return pd.DataFrame({'private-id': [repr(value)]})\n"
        ),
    )
    from anonymizer.engine.execution.graph import _DatumId
    from anonymizer.engine.execution.protection_service import (
        _GraphProtectionResult,
        _GraphProtectionSucceeded,
    )

    private_outcome = _GraphProtectionResult((_GraphProtectionSucceeded(_DatumId("private-id"), "private text", True),))
    config = AnonymizerConfig(replace=Redact())

    with pytest.raises(TypeError, match="pandas DataFrame"):
        compatibility._materialize_run_result(
            cast(pd.DataFrame, private_outcome),
            config=config,
            resolved_text_column="text",
            failed_records=[],
            data_summary=None,
        )
    admitted = mutant._materialize_run_result(
        cast(pd.DataFrame, private_outcome),
        config=config,
        resolved_text_column="text",
        failed_records=[],
        data_summary=None,
    )
    with pytest.raises(AssertionError):
        assert "private-id" not in admitted.trace_dataframe.to_json()


def test_remaining_contract_mutants_are_killed_early_materialization() -> None:
    source, mutant = _mutate_source(
        _FACADE_PATH,
        "early-materialization",
        "        execution = _PandasRuntime(\n",
        (
            "        _materialize_run_result(\n"
            "            context.dataframe,\n"
            "            config=config,\n"
            "            resolved_text_column=context.resolved_text_column,\n"
            "            failed_records=[],\n"
            "            data_summary=data.data_summary,\n"
            "        )\n"
            "        execution = _PandasRuntime(\n"
        ),
    )

    _assert_materialization_order(source)
    with pytest.raises(AssertionError):
        _assert_materialization_order(mutant)


def test_remaining_contract_mutants_are_killed_private_leakage(tmp_path: Path) -> None:
    mutant = _load_mutant(
        tmp_path,
        "private-leakage",
        "return trace[[column for column in trace.columns if column in allowed]].copy()",
        (
            "return trace[[column for column in trace.columns if column in allowed] "
            '+ (["private-id"] if "private-id" in trace.columns else [])].copy()'
        ),
    )
    source = pd.DataFrame(
        {
            "__nemo_anonymizer_text_input__": ["Alice"],
            "__nemo_anonymizer_text_output__": ["[REDACTED]"],
            "private-id": ["synthetic-secret@example.test"],
        }
    )
    config = AnonymizerConfig(replace=Redact())

    baseline = compatibility._materialize_run_result(
        source,
        config=config,
        resolved_text_column="text",
        failed_records=[],
        data_summary=None,
    )
    mutated = mutant._materialize_run_result(
        source,
        config=config,
        resolved_text_column="text",
        failed_records=[],
        data_summary=None,
    )

    assert "private-id" not in baseline.dataframe.columns
    with pytest.raises(AssertionError):
        assert "private-id" not in mutated.dataframe.columns


def test_remaining_contract_mutants_are_killed_ndd_bypass() -> None:
    sources = {
        path.relative_to(_PACKAGE_ROOT): path.read_text(encoding="utf-8") for path in _PACKAGE_ROOT.rglob("*.py")
    }
    mutant_sources = dict(sources)
    compatibility_path = Path("interface/_result_compatibility.py")
    mutant_sources[compatibility_path] += "\n_data_designer.create()\n"

    _assert_no_direct_data_designer_execution(sources)
    with pytest.raises(AssertionError):
        _assert_no_direct_data_designer_execution(mutant_sources)


def test_remaining_contract_mutants_are_killed_public_export() -> None:
    source, mutant = _mutate_source(
        _PUBLIC_PACKAGE_PATH,
        "public-export",
        '    "Anonymizer",\n',
        '    "Anonymizer",\n    "_materialize_run_result",\n',
    )

    _assert_no_private_public_exports(source)
    with pytest.raises(AssertionError):
        _assert_no_private_public_exports(mutant)


def test_remaining_contract_mutants_are_killed_wheel_omission(tmp_path: Path) -> None:
    repository_root = Path(__file__).parents[2]
    build_directory = tmp_path / "built"
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(build_directory)],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    built_wheels = list(build_directory.glob("*.whl"))
    assert len(built_wheels) == 1
    baseline = built_wheels[0]
    mutant = tmp_path / "mutant.whl"
    with zipfile.ZipFile(baseline) as source_archive, zipfile.ZipFile(mutant, "w") as mutant_archive:
        for member in source_archive.infolist():
            if member.filename != "anonymizer/interface/result_compatibility_contract.json":
                mutant_archive.writestr(member, source_archive.read(member.filename))

    _assert_wheel_artifacts(baseline)
    with pytest.raises(AssertionError):
        _assert_wheel_artifacts(mutant)


def test_mutation_manifest_covers_the_exact_frozen_contract_set() -> None:
    manifest = json.loads(_MUTATION_MANIFEST_PATH.read_text(encoding="utf-8"))
    contract = json.loads(_CONTRACT_PATH.read_text(encoding="utf-8"))
    mutations = manifest["mutations"]
    encoded = json.dumps(mutations, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()

    assert manifest["mutation_count"] == len(mutations) == 19
    assert manifest["digest"] == hashlib.sha256(encoded).hexdigest() == _MUTATION_DIGEST
    assert [mutation["rule"] for mutation in mutations] == contract["contract"]["mutation_contract"]
    assert all(mutation["witness"] for mutation in mutations)


def test_materializer_remains_after_runtime_and_measurement_boundaries() -> None:
    from anonymizer.interface.anonymizer import Anonymizer

    source = inspect.getsource(Anonymizer._run_internal_impl)

    assert source.index(".run(") < source.index("record_record_metrics(") < source.index("_materialize_run_result(")


def test_result_materializer_has_no_private_graph_or_correlation_dependency() -> None:
    source = _MATERIALIZER_PATH.read_text(encoding="utf-8")

    assert "PRIVATE_CORRELATION_COLUMN" not in source
    assert "anonymizer.engine.execution" not in source
    assert "DataDesigner" not in source
    assert "record_id" not in source


def test_data_designer_execution_stays_in_the_ndd_adapter() -> None:
    package_root = _MATERIALIZER_PATH.parents[1]
    execution_sites: set[Path] = set()
    for path in package_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr not in {"create", "preview"}:
                continue
            if "data_designer" in ast.unparse(node.func.value):
                execution_sites.add(path.relative_to(package_root))

    assert execution_sites == {Path("engine/ndd/adapter.py")}


def test_adapter_is_not_exported_from_the_public_package() -> None:
    assert not hasattr(anonymizer, "_materialize_run_result")
    assert not hasattr(anonymizer, "_materialize_preview_result")
    assert not hasattr(anonymizer, "_materialize_evaluation_result")
    assert compatibility.__name__ == "anonymizer.interface._result_compatibility"
