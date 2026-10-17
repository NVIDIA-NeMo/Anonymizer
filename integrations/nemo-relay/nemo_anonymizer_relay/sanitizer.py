# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded sanitization over copied observability values."""

from __future__ import annotations

import asyncio
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Protocol

from nemo_anonymizer_relay.backend import RedactionSpan
from nemo_anonymizer_relay.projection import (
    TextLeaf,
    collect_text_leaves,
    omit_unsupported_media,
    replace_text_leaves,
)

Json = Any

_SECRET_PATTERNS = (
    re.compile(r"\b(?:sk-(?:proj-)?|nvapi-)[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bglpat-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bnpm_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),
    re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]{12,}=*"),
)
_SECRET_VALUE_KEYS = frozenset(
    {
        "access_token",
        "api_key",
        "api_secret",
        "authentication_token",
        "authorization",
        "auth_token",
        "bearer_token",
        "client_secret",
        "cookie",
        "password",
        "passwd",
        "private_key",
        "proxy_authorization",
        "secret",
        "set_cookie",
        "x_api_key",
    }
)
_COLLAPSED_SECRET_VALUE_KEYS = frozenset(key.replace("_", "") for key in _SECRET_VALUE_KEYS)
_SECRET_KEY_SUFFIXES = (
    "api_key",
    "authorization",
    "cookie",
    "credential",
    "password",
    "passwd",
    "private_key",
    "secret",
    "secret_access_key",
    "token",
)
_COLLAPSED_SECRET_KEY_SUFFIXES = tuple(suffix.replace("_", "") for suffix in _SECRET_KEY_SUFFIXES)


class Detector(Protocol):
    def detect(self, texts: list[str]) -> list[list[RedactionSpan]]: ...


class ObservationSanitizer:
    """Sanitize copied Relay events without retaining source strings.

    Source strings and mostly-unredacted outputs are never retained after a
    callback completes.
    """

    def __init__(
        self,
        detector: Detector,
        *,
        max_leaves: int = 1024,
        max_bytes: int = 256 * 1024,
        max_leaf_bytes: int = 64 * 1024,
    ) -> None:
        self._detector = detector
        self._max_leaves = max_leaves
        self._max_bytes = max_bytes
        self._max_leaf_bytes = max_leaf_bytes
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="nemo-anonymizer")
        self._detector_state_lock = threading.Lock()
        self._detector_active = False
        self._closed = False
        # The stable Anonymizer/Data Designer API does not document concurrent
        # calls on one pipeline. Serialize calls until that contract exists.
        self._lock = asyncio.Lock()

    async def sanitize(
        self,
        value: Json,
        *,
        preserve_protocol_values: bool = False,
    ) -> Json:
        """Sanitize one copied observability value without mutating its source."""

        if self._closed:
            raise RuntimeError("observation sanitizer is closed")

        # Credentials are not useful model input. Remove recognizable tokens
        # and values under credential-bearing keys before the detector or
        # evaluator can receive them.
        protected = _redact_secrets(omit_unsupported_media(value))
        leaves = collect_text_leaves(
            protected,
            max_leaves=self._max_leaves,
            max_bytes=self._max_bytes,
            max_leaf_bytes=self._max_leaf_bytes,
            preserve_protocol_values=preserve_protocol_values,
        )
        if not leaves:
            return protected
        decisions = await self._resolve_decisions(leaves)
        if any(leaf.key is not None and decisions[leaf.text] for leaf in leaves):
            raise RuntimeError("Anonymizer detected a sensitive mapping key")
        replacements = {leaf.path: self._apply(leaf.text, decisions[leaf.text]) for leaf in leaves if leaf.key is None}
        return replace_text_leaves(protected, replacements)

    async def _resolve_decisions(self, leaves: list[TextLeaf]) -> dict[str, tuple[RedactionSpan, ...]]:
        unique_texts = list(dict.fromkeys(leaf.text for leaf in leaves))
        async with self._lock:
            if not self._reserve_detector():
                raise RuntimeError("Anonymizer detector is still completing prior work")
            try:
                future = asyncio.get_running_loop().run_in_executor(
                    self._executor,
                    self._detect_reserved,
                    unique_texts,
                )
                future.add_done_callback(_consume_future_exception)
            except BaseException:
                self._release_detector()
                raise
            # A Relay timeout cancels this coroutine but cannot stop a
            # synchronous model call. Shield the executor future so the
            # reservation remains held until that call actually exits; later
            # events fail closed instead of filling an unbounded executor queue.
            spans = await asyncio.shield(future)
        if len(spans) != len(unique_texts):
            raise RuntimeError("Anonymizer returned the wrong number of decisions")

        decisions: dict[str, tuple[RedactionSpan, ...]] = {}
        for text, text_spans in zip(unique_texts, spans, strict=True):
            decision = tuple(text_spans)
            self._validate_decision(text, decision)
            decisions[text] = decision
        return decisions

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # A synchronous detector keeps running after its Relay callback is
        # cancelled. Drain that one bounded executor task before releasing the
        # reusable backend and its private artifact directory.
        self._executor.shutdown(wait=True, cancel_futures=True)
        close_detector = getattr(self._detector, "close", None)
        if callable(close_detector):
            close_detector()

    def _reserve_detector(self) -> bool:
        with self._detector_state_lock:
            if self._detector_active:
                return False
            self._detector_active = True
            return True

    def _release_detector(self) -> None:
        with self._detector_state_lock:
            self._detector_active = False

    def _detect_reserved(self, texts: list[str]) -> list[list[RedactionSpan]]:
        try:
            try:
                return self._detector.detect(texts)
            except Exception:
                raise RuntimeError("Anonymizer detector failed") from None
        finally:
            self._release_detector()

    @staticmethod
    def _apply(text: str, spans: tuple[RedactionSpan, ...]) -> str:
        redacted = text
        for span in reversed(spans):
            redacted = f"{redacted[: span.start]}[REDACTED]{redacted[span.end :]}"
        return redacted

    @staticmethod
    def _validate_decision(text: str, spans: tuple[RedactionSpan, ...]) -> None:
        previous_end = 0
        for span in spans:
            if span.start < previous_end or span.end > len(text) or span.end <= span.start:
                raise RuntimeError("Anonymizer returned an unsafe entity span")
            previous_end = span.end


def _consume_future_exception(future: asyncio.Future[Any]) -> None:
    """Prevent a post-timeout executor failure from becoming an unhandled log."""

    if not future.cancelled():
        future.exception()


def _redact_secret_string(value: str) -> str:
    for pattern in _SECRET_PATTERNS:
        value = pattern.sub("[REDACTED_SECRET]", value)
    return value


def _redact_secrets(value: Json) -> Json:
    if isinstance(value, str):
        return _redact_secret_string(value)
    if isinstance(value, list):
        return [_redact_secrets(item) for item in value]
    if not isinstance(value, dict):
        return value
    rewritten: dict[str, Json] = {}
    for key, item in value.items():
        if isinstance(key, str) and _redact_secret_string(key) != key:
            raise RuntimeError("copied observability contains a secret-shaped mapping key")
        rewritten[key] = (
            "[REDACTED_SECRET]"
            if isinstance(key, str) and item is not None and _is_secret_value_key(key)
            else _redact_secrets(item)
        )
    return rewritten


def _is_secret_value_key(key: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", key.strip().lower()).strip("_")
    collapsed = normalized.replace("_", "")
    return (
        normalized in _SECRET_VALUE_KEYS
        or collapsed in _COLLAPSED_SECRET_VALUE_KEYS
        or any(normalized == suffix or normalized.endswith(f"_{suffix}") for suffix in _SECRET_KEY_SUFFIXES)
        or any(collapsed.endswith(suffix) for suffix in _COLLAPSED_SECRET_KEY_SUFFIXES)
    )


__all__ = ["ObservationSanitizer"]
