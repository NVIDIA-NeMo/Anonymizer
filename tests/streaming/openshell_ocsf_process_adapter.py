# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test-only buffered adapter for OpenShell OCSF Process Activity JSONL records."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import cast

from anonymizer.interface.anonymizer import Anonymizer
from tests.streaming.structured_trace_prototype import (
    CodecBounds,
    FailureCode,
    FieldRole,
    ProjectedItem,
    SourceFormat,
    StructuredItemError,
    TraceMapping,
    project_complete_item,
    protect_and_emit,
)

MAPPING_VERSION = "openshell-ocsf-process-activity-1007/v1"
FIDELITY_CLASS = "semantic-jsonl-item-v1"

_ACTIVITY_LABELS = {1: "Launch", 2: "Terminate"}
_SEVERITY_LABELS = {
    0: "Unknown",
    1: "Informational",
    2: "Low",
    3: "Medium",
    4: "High",
    5: "Critical",
    6: "Fatal",
    99: "Other",
}
_STATUS_LABELS = {0: "Unknown", 1: "Success", 2: "Failure", 99: "Other"}
_LAUNCH_LABELS = {0: "Unknown", 1: "Spawn", 2: "Fork", 3: "Exec", 99: "Other"}
_ACTION_LABELS = {0: "Unknown", 1: "Allowed", 2: "Denied", 3: "Observed", 4: "Modified", 99: "Other"}
_DISPOSITION_LABELS = {
    0: "Unknown",
    1: "Allowed",
    2: "Blocked",
    3: "Quarantined",
    4: "Isolated",
    5: "Deleted",
    6: "Dropped",
    7: "Custom Action",
    8: "Approved",
    9: "Restored",
    10: "Exonerated",
    11: "Corrected",
    12: "Partially Corrected",
    13: "Uncorrected",
    14: "Delayed",
    15: "Detected",
    16: "No Action",
    17: "Logged",
    18: "Tagged",
    19: "Alert",
    20: "Count",
    21: "Reset",
    22: "Captcha",
    23: "Challenge",
    24: "Access Revoked",
    25: "Rejected",
    26: "Unauthorized",
    27: "Error",
    99: "Other",
}
_REQUIRED_EVENT_FIELDS = frozenset(
    {
        "class_uid",
        "class_name",
        "category_uid",
        "category_name",
        "activity_id",
        "activity_name",
        "type_uid",
        "type_name",
        "time",
        "severity_id",
        "severity",
        "metadata",
        "device",
        "container",
        "process",
    }
)
_OPTIONAL_EVENT_FIELDS = frozenset(
    {
        "status_id",
        "status",
        "message",
        "actor",
        "launch_type_id",
        "launch_type",
        "exit_code",
        "action_id",
        "action",
        "disposition_id",
        "disposition",
    }
)


def project_process_activity_item(source: bytes, *, bounds: CodecBounds) -> ProjectedItem:
    """Validate and project one complete OpenShell Process Activity JSONL record."""
    document = _decode_and_validate(source, bounds=bounds)
    mapping = _mapping_for(document)
    projected = project_complete_item(
        source,
        source_format=SourceFormat.JSONL,
        mapping=mapping,
        bounds=bounds,
    )
    if projected.manifest.fidelity != FIDELITY_CLASS:
        raise StructuredItemError(FailureCode.MAPPING_MISMATCH)
    return projected


def protect_process_activity_item(
    source: bytes,
    *,
    bounds: CodecBounds,
    anonymizer: Anonymizer,
    source_ref: Path,
    emit: Callable[[bytes], None],
) -> bytes:
    """Protect and schema-check one complete item before external emission."""
    document = _decode_and_validate(source, bounds=bounds)
    staged: list[bytes] = []
    protected = protect_and_emit(
        source,
        source_format=SourceFormat.JSONL,
        mapping=_mapping_for(document),
        bounds=bounds,
        anonymizer=anonymizer,
        source_ref=source_ref,
        emit=staged.append,
    )
    if staged != [protected]:
        raise StructuredItemError(FailureCode.SEGMENT_PROCESSING_FAILED)
    _decode_and_validate(protected, bounds=bounds)
    emit(protected)
    return protected


def replay_process_activity_corpus(
    corpus: bytes,
    *,
    max_corpus_bytes: int,
    max_records: int,
    bounds: CodecBounds,
    anonymizer: Anonymizer,
    source_ref: Path,
    emit: Callable[[bytes], None],
) -> tuple[bytes, ...]:
    """Replay bounded JSONL records without making OpenShell a dispatcher."""
    if max_corpus_bytes <= 0 or max_records <= 0:
        raise StructuredItemError(FailureCode.MAPPING_MISMATCH)
    if not corpus or len(corpus) > max_corpus_bytes:
        raise StructuredItemError(FailureCode.ITEM_TOO_LARGE)
    lines = corpus.splitlines(keepends=True)
    if len(lines) > max_records:
        raise StructuredItemError(FailureCode.ITEM_TOO_LARGE)
    if any(not line.endswith(b"\n") or not line[:-1].strip() for line in lines):
        raise StructuredItemError(FailureCode.INVALID_SOURCE)

    protected_items: list[bytes] = []
    for line in lines:
        protected_items.append(
            protect_process_activity_item(
                line,
                bounds=bounds,
                anonymizer=anonymizer,
                source_ref=source_ref,
                emit=emit,
            )
        )
    return tuple(protected_items)


def _decode_and_validate(source: bytes, *, bounds: CodecBounds) -> dict[str, object]:
    if len(source) > bounds.max_bytes:
        raise StructuredItemError(FailureCode.ITEM_TOO_LARGE)
    try:
        lines = source.splitlines()
        if len(lines) != 1 or not source.endswith(b"\n") or not lines[0].strip():
            raise StructuredItemError(FailureCode.INVALID_SOURCE)
        document = json.loads(
            lines[0].decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_number,
            parse_float=_parse_finite_float,
        )
        if not isinstance(document, dict):
            raise StructuredItemError(FailureCode.MAPPING_MISMATCH)
        typed_document = cast(dict[str, object], document)
        _validate_process_activity(typed_document)
        return typed_document
    except StructuredItemError:
        raise
    except RecursionError:
        raise StructuredItemError(FailureCode.STRUCTURE_TOO_DEEP) from None
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        raise StructuredItemError(FailureCode.INVALID_SOURCE) from None
    except Exception:
        raise StructuredItemError(FailureCode.MAPPING_MISMATCH) from None


def _validate_process_activity(event: dict[str, object]) -> None:
    _require_keys(event, required=_REQUIRED_EVENT_FIELDS, optional=_OPTIONAL_EVENT_FIELDS)
    _validate_base_event(event)
    _validate_optional_event_fields(event)
    source_identity = _validate_metadata(_require_object(event["metadata"]))
    _validate_device(_require_object(event["device"]))
    _validate_container(_require_object(event["container"]), source_identity=source_identity)
    _validate_process(_require_object(event["process"]))
    if "actor" in event:
        actor = _require_object(event["actor"])
        _require_keys(actor, required={"process"})
        _validate_process(_require_object(actor["process"]))


def _validate_base_event(event: Mapping[str, object]) -> None:
    if _require_int(event["class_uid"]) != 1007 or _require_string(event["class_name"]) != "Process Activity":
        raise StructuredItemError(FailureCode.MAPPING_MISMATCH)
    if _require_int(event["category_uid"]) != 1 or _require_string(event["category_name"]) != "System Activity":
        raise StructuredItemError(FailureCode.MAPPING_MISMATCH)
    activity_id = _require_int(event["activity_id"])
    activity_name = _require_string(event["activity_name"])
    if _ACTIVITY_LABELS.get(activity_id) != activity_name:
        raise StructuredItemError(FailureCode.MAPPING_MISMATCH)
    if _require_int(event["type_uid"]) != 100_700 + activity_id:
        raise StructuredItemError(FailureCode.MAPPING_MISMATCH)
    if _require_string(event["type_name"]) != f"Process Activity: {activity_name}":
        raise StructuredItemError(FailureCode.MAPPING_MISMATCH)
    _require_int(event["time"])
    _validate_pair(event, "severity_id", "severity", _SEVERITY_LABELS, required=True)


def _validate_optional_event_fields(event: Mapping[str, object]) -> None:
    _validate_pair(event, "status_id", "status", _STATUS_LABELS)
    _validate_pair(event, "launch_type_id", "launch_type", _LAUNCH_LABELS)
    _validate_pair(event, "action_id", "action", _ACTION_LABELS)
    _validate_pair(event, "disposition_id", "disposition", _DISPOSITION_LABELS)
    if "message" in event:
        _require_string(event["message"])
    if "exit_code" in event:
        _require_int(event["exit_code"])


def _validate_metadata(metadata: dict[str, object]) -> str:
    _require_keys(metadata, required={"version", "product", "profiles", "uid"})
    source_identity = _require_string(metadata["uid"])
    if _require_string(metadata["version"]) != "1.7.0" or not source_identity:
        raise StructuredItemError(FailureCode.MAPPING_MISMATCH)
    if metadata["profiles"] != ["security_control", "container", "host"]:
        raise StructuredItemError(FailureCode.MAPPING_MISMATCH)
    product = _require_object(metadata["product"])
    _require_keys(product, required={"name", "vendor_name", "version"})
    if (
        _require_string(product["name"]) != "OpenShell Sandbox Supervisor"
        or _require_string(product["vendor_name"]) != "OpenShell"
        or not _require_string(product["version"])
    ):
        raise StructuredItemError(FailureCode.MAPPING_MISMATCH)
    return source_identity


def _validate_device(device: dict[str, object]) -> None:
    _require_keys(device, required={"hostname", "os"})
    _require_string(device["hostname"])
    os_info = _require_object(device["os"])
    _require_keys(os_info, required={"name"})
    if _require_string(os_info["name"]) != "Linux":
        raise StructuredItemError(FailureCode.MAPPING_MISMATCH)


def _validate_container(container: dict[str, object], *, source_identity: str) -> None:
    _require_keys(container, required={"name", "uid", "image"})
    _require_string(container["name"])
    if _require_string(container["uid"]) != source_identity:
        raise StructuredItemError(FailureCode.MAPPING_MISMATCH)
    image = _require_object(container["image"])
    _require_keys(image, required={"name"})
    _require_string(image["name"])


def _validate_process(process: dict[str, object]) -> None:
    _require_keys(process, required={"name", "pid"}, optional={"cmd_line", "parent_process"})
    _require_string(process["name"])
    _require_int(process["pid"])
    if "cmd_line" in process:
        _require_string(process["cmd_line"])
    if "parent_process" in process:
        _validate_process(_require_object(process["parent_process"]))


def _mapping_for(document: Mapping[str, object]) -> TraceMapping:
    fields: dict[str, FieldRole] = {}
    for pointer, _ in _iter_scalars(document):
        fields[pointer] = FieldRole.TARGET if _is_target(pointer) else FieldRole.STRUCTURAL
    return TraceMapping(
        version=MAPPING_VERSION,
        fields=fields,
        source_identity_pointer="/metadata/uid",
        ordered_identity_pointers=("/type_name",),
    )


def _is_target(pointer: str) -> bool:
    tokens = pointer.removeprefix("/").split("/")
    if pointer == "/message" or pointer == "/device/hostname" or pointer == "/container/name":
        return True
    if pointer == "/container/image/name":
        return True
    return tokens[-1] in {"name", "cmd_line"} and tokens[0] in {"process", "actor"}


def _iter_scalars(value: object, *, pointer: str = "") -> list[tuple[str, object]]:
    if isinstance(value, dict):
        result: list[tuple[str, object]] = []
        for key, item in cast(dict[str, object], value).items():
            escaped = key.replace("~", "~0").replace("/", "~1")
            result.extend(_iter_scalars(item, pointer=f"{pointer}/{escaped}"))
        return result
    if isinstance(value, list):
        result = []
        for index, item in enumerate(value):
            result.extend(_iter_scalars(item, pointer=f"{pointer}/{index}"))
        return result
    return [(pointer, value)]


def _validate_pair(
    value: Mapping[str, object],
    id_key: str,
    label_key: str,
    labels: Mapping[int, str],
    *,
    required: bool = False,
) -> None:
    present = id_key in value or label_key in value
    if required and not present:
        raise StructuredItemError(FailureCode.MAPPING_MISMATCH)
    if not present:
        return
    if id_key not in value or label_key not in value:
        raise StructuredItemError(FailureCode.MAPPING_MISMATCH)
    if labels.get(_require_int(value[id_key])) != _require_string(value[label_key]):
        raise StructuredItemError(FailureCode.MAPPING_MISMATCH)


def _require_keys(
    value: Mapping[str, object],
    *,
    required: set[str] | frozenset[str],
    optional: set[str] | frozenset[str] | None = None,
) -> None:
    keys = set(value)
    allowed = required | (optional or set())
    if not required.issubset(keys) or not keys.issubset(allowed):
        raise StructuredItemError(FailureCode.MAPPING_MISMATCH)


def _require_object(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise StructuredItemError(FailureCode.MAPPING_MISMATCH)
    return cast(dict[str, object], value)


def _require_string(value: object) -> str:
    if not isinstance(value, str):
        raise StructuredItemError(FailureCode.MAPPING_MISMATCH)
    return value


def _require_int(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise StructuredItemError(FailureCode.MAPPING_MISMATCH)
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _reject_nonfinite_number(_: str) -> float:
    raise ValueError("non-finite JSON number")


def _parse_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("non-finite JSON number")
    return parsed
