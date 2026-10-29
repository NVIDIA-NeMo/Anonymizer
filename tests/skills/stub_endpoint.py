# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provider-free OpenAI-compatible endpoint for running the skill's script template.

Answers ``/chat/completions`` deterministically from the request alone. Requests for a model whose name contains
``gliner`` get a detector payload; every other request carries a ``<response_schema>`` block and gets a schema-valid
JSON instance. The pipeline's detector, validator, augmenter, replacement, rewrite, QA and judge roles all go through
this one endpoint, so a missing provider or alias fails the way it would against a real service.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

# Synthetic PII shared with the test's input rows.
VOCAB = [
    ("Alice Smith", "first_name"),
    ("Bob Jones", "first_name"),
    ("Carol Lee", "first_name"),
    ("Paris", "city"),
    ("Acme Corp", "company_name"),
]
_LABELS = dict(VOCAB)


def _synthetic(value: str) -> str:
    return f"Syn{hashlib.sha256(value.encode()).hexdigest()[:6]}"


def _messages_text(body: dict[str, Any]) -> str:
    parts: list[str] = []
    for message in body.get("messages", []):
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts.extend(str(item.get("text", "")) for item in content if isinstance(item, dict))
    return "\n".join(parts)


def _response_schema(text: str) -> dict[str, Any] | None:
    found = re.findall(r"<response_schema>\s*(\{.*?\})\s*</response_schema>", text, re.S)
    if not found:
        return None
    try:
        return json.loads(found[-1])
    except json.JSONDecodeError:
        return None


def _source_text(text: str) -> str:
    """Longest prompt line that mentions a vocabulary item."""
    lines = [line for line in text.splitlines() if any(value in line for value, _ in VOCAB)]
    return max(lines, key=len) if lines else "stub text"


def _redact(text: str) -> str:
    for value, _ in sorted(VOCAB, key=lambda item: -len(item[0])):
        text = text.replace(value, _synthetic(value))
    return text


def _detector_payload(text: str) -> dict[str, Any]:
    entities = []
    for value, label in VOCAB:
        start = text.find(value)
        while start >= 0:
            entities.append({"text": value, "label": label, "start": start, "end": start + len(value), "score": 0.9})
            start = text.find(value, start + 1)
    return {"entities": sorted(entities, key=lambda entity: entity["start"])}


def _fill_template(template: Any) -> Any:
    if isinstance(template, dict):
        out: dict[str, Any] = {}
        for key, value in template.items():
            if key == "answer":
                out[key] = value or ("no" if "question" in template else "stub answer")
            elif key in {"confidence", "score"}:
                out[key] = 1.0
            elif key == "reason":
                out[key] = value or "stub reason"
            elif key == "evidence":
                out[key] = []
            else:
                out[key] = _fill_template(value)
        return out
    if isinstance(template, list):
        return [_fill_template(item) for item in template]
    return template


class _Instance:
    """Minimal JSON-Schema instance generator."""

    def __init__(self, schema: dict[str, Any], text: str) -> None:
        self.root, self.text = schema, text

    def _resolve(self, node: dict[str, Any]) -> dict[str, Any]:
        while "$ref" in node:
            target: Any = self.root
            for part in node["$ref"].lstrip("#/").split("/"):
                target = target[part]
            node = {**target, **{key: value for key, value in node.items() if key != "$ref"}}
        return node

    def make(self, node: dict[str, Any], name: str = "") -> Any:
        node = self._resolve(node)
        if "const" in node:
            return node["const"]
        if "enum" in node:
            return node["enum"][0]
        for key in ("anyOf", "oneOf"):
            if key in node:
                options = [self._resolve(option) for option in node[key]]
                return self.make(([o for o in options if o.get("type") != "null"] or options)[0], name)
        kind = node.get("type")
        if isinstance(kind, list):
            kind = next((k for k in kind if k != "null"), "null")
        if kind == "object" or "properties" in node:
            return {key: self.make(value, key) for key, value in (node.get("properties") or {}).items()}
        if kind == "array":
            return [self.make(node.get("items") or {}, name) for _ in range(max(int(node.get("minItems", 0)), 0))]
        if kind == "string":
            if "rewrit" in name.lower() or name.lower() in {"text", "output", "repaired_text"}:
                return _redact(_source_text(self.text))
            return f"stub {name}".strip()
        if kind == "integer":
            return int(node.get("minimum", 0))
        if kind == "number":
            low, high = node.get("minimum"), node.get("maximum")
            if low is not None and high is not None:
                return (low + high) / 2
            return float(low) if low is not None else 0.5
        if kind == "boolean":
            return name.startswith("all_")
        return None


def _disposition(schema: dict[str, Any], text: str) -> list[dict[str, Any]]:
    defs = schema.get("$defs", {})

    def pick(name: str, preferred: str) -> str:
        values = defs.get(name, {}).get("enum", [preferred])
        return preferred if preferred in values else values[0]

    found = re.search(r"(\{'entities_by_value': \[.*?\]\})", text, re.S)
    entities: list[tuple[str, str]] = []
    if found:
        try:
            parsed = ast.literal_eval(found.group(1))
            entities = [(e["value"], (e.get("labels") or ["unknown"])[0]) for e in parsed["entities_by_value"]]
        except (ValueError, SyntaxError, KeyError, TypeError):
            entities = []
    entities = entities or [(value, label) for value, label in VOCAB if value in text] or [("stub", "unknown")]
    return [
        {
            "id": index,
            "source": pick("EntitySource", "tagged"),
            "category": pick("EntityCategory", "direct_identifier"),
            "sensitivity": pick("SensitivityLevel", "high"),
            "entity_label": label,
            "entity_value": value,
            "protection_reason": "stub: identifies the subject of the document",
            "protection_method_suggestion": pick("ProtectionMethod", "replace"),
            "combined_risk_level": pick("CombinedRiskLevel", "high"),
        }
        for index, (value, label) in enumerate(entities, start=1)
    ]


def _structured_payload(schema: dict[str, Any], text: str) -> Any:
    props = set((schema.get("properties") or {}).keys())
    names = [value for value, _ in VOCAB if value in text]
    if props == {"replacements"}:
        return {
            "replacements": [
                {"original": value, "label": _LABELS[value], "synthetic": _synthetic(value)} for value in names
            ]
        }
    if props == {"entities"}:
        return {"entities": []}  # augmenter: nothing beyond the detector
    if props == {"decisions"}:
        match = re.search(r"Template: (\{.*?\})\s*$", text, re.M)
        if match:
            try:
                template = ast.literal_eval(match.group(1))
                return {"decisions": [{**d, "decision": "keep", "reason": "stub keep"} for d in template["decisions"]]}
            except (ValueError, SyntaxError, KeyError, TypeError):
                pass
        return {"decisions": []}
    if props == {"items"}:
        generator = _Instance(schema, text)
        return {"items": [generator.make(schema["properties"]["items"].get("items") or {}, "items")]}
    if props == {"sensitivity_disposition"}:
        return {"sensitivity_disposition": _disposition(schema, text)}
    if props == {"rewritten_text"}:
        match = re.search(r"Tagged text:\s*\n(.*?)\n", text)
        tagged = match.group(1) if match else _source_text(text)
        return {"rewritten_text": _redact(re.sub(r"</?[a-z_]+>", "", tagged))}
    template = re.search(r"<answer_template>\s*(\{.*?\})\s*</answer_template>", text, re.S)
    if template:
        try:
            return _fill_template(json.loads(template.group(1)))
        except json.JSONDecodeError:
            pass
    return _Instance(schema, text).make(schema)


def _completion(content: str, model: str) -> dict[str, Any]:
    return {
        "id": "stub-" + hashlib.sha256(content.encode()).hexdigest()[:16],
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


class _Handler(BaseHTTPRequestHandler):
    server: StubEndpoint

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    def _send(self, payload: dict[str, Any]) -> None:
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        self._send({"object": "list", "data": []})

    def do_POST(self) -> None:  # noqa: N802
        raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        body = json.loads(raw) if raw else {}
        model = str(body.get("model", ""))
        text = _messages_text(body)
        with self.server.lock:
            self.server.models_seen.add(model)
        schema = _response_schema(text)
        if "gliner" in model:
            users = [m.get("content") for m in body.get("messages", []) if m.get("role") == "user"]
            content = json.dumps(_detector_payload(users[-1] if users and isinstance(users[-1], str) else text))
        elif schema is None:
            content = _redact(_source_text(text))
        else:
            content = "```json\n" + json.dumps(_structured_payload(schema, text)) + "\n```"
        self._send(_completion(content, model))


class StubEndpoint(ThreadingHTTPServer):
    """Stub served from a background thread on an ephemeral localhost port."""

    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.lock = threading.Lock()
        self.models_seen: set[str] = set()
        self._thread = threading.Thread(target=self.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}/v1"

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self.shutdown()
        self.server_close()
        self._thread.join(timeout=5)
