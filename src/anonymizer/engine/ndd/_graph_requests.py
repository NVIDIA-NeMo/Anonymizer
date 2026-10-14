# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Version-qualified DD HTTP entry borrowed from the invocation request owner.

This development binding observes each actual parser attempt in its generation chain.
The caller registers NddPhysicalTransport with ForeignLoopTransport before
execution. Its cleanup closes actual DD clients after the worker drains. DD owns
conversation restarts and correction turns; provider retries remain unsupported.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from importlib.metadata import version
from threading import Lock
from typing import Any

import httpx
from data_designer.engine.models.clients.retry import create_retry_transport
from data_designer.engine.models.parsers.errors import ParserException

from anonymizer.engine.graph_sdk._execution_values import DelegatedRequests
from anonymizer.engine.graph_sdk.requests import (
    AssociationResult,
    DispatchEnvelope,
    ExactUsage,
    ExternalSettlement,
    PhysicalRequestId,
    StopUnknown,
    TransportFailure,
    TransportSuccess,
    UnknownUsage,
)
from anonymizer.engine.ndd._source_frame import current_source_work
from anonymizer.interface.errors import AnonymizerWorkflowError


@dataclass
class _Exchange:
    owner: NddPhysicalTransport
    transport: httpx.AsyncBaseTransport
    request: httpx.Request
    response: httpx.Response | None = None
    error: Exception | None = None


_EXCHANGE: ContextVar[_Exchange] = ContextVar("anonymizer_dd_physical_exchange")


@dataclass
class _Generation:
    parser: Callable[[str], Any]
    correction_steps: int = 0
    content: str | None = None
    value: Any = None
    error: Exception | None = None
    previous: PhysicalRequestId | None = None
    attempts: int = 0

    def parse(self, content: str) -> Any:
        if content != self.content:
            raise AnonymizerWorkflowError("DD parser content differs from the physical receipt")
        if self.error is not None:
            raise self.error
        return self.value


_GENERATION: ContextVar[_Generation | None] = ContextVar("anonymizer_dd_generation", default=None)


class NddPhysicalTransport:
    """Dispatch the actual HTTP exchange after the graph's foreign-loop gate."""

    def __init__(self) -> None:
        self.loop = asyncio.get_running_loop()
        self.clients: list[Any] = []
        self.closed = False
        self._lock = Lock()

    def register(self, client: Any) -> None:
        with self._lock:
            if self.closed:
                raise AnonymizerWorkflowError("DD physical resource is closed")
            if not any(client is existing for existing in self.clients):
                self.clients.append(client)

    async def close(self) -> None:
        if asyncio.get_running_loop() is not self.loop:
            raise AnonymizerWorkflowError("DD physical resource closed on a foreign loop")
        with self._lock:
            clients = tuple(self.clients)
            self.closed = True
        results = await asyncio.gather(*(client.aclose() for client in clients), return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException):
                raise result

    async def dispatch(self, request: DispatchEnvelope) -> TransportSuccess | TransportFailure:
        if self.closed or asyncio.get_running_loop() is not self.loop:
            raise AnonymizerWorkflowError("DD physical resource is not live on its owning loop")
        exchange = _EXCHANGE.get()
        work = current_source_work()
        if (
            exchange.owner is not self
            or exchange.response is not None
            or len(request.associations) != 1
            or (work is not None and request.associations[0].association != work.association)
        ):
            raise AnonymizerWorkflowError("DD physical exchange has invalid source ownership")
        try:
            response = await exchange.transport.handle_async_request(exchange.request)
            await response.aread()
        except Exception as error:
            exchange.error = error
            raise
        exchange.response = response
        try:
            payload = response.json()
        except ValueError:
            payload = None
        usage = payload.get("usage", {}) if isinstance(payload, dict) else {}
        incoming = usage.get("prompt_tokens") if isinstance(usage, dict) else None
        outgoing = usage.get("completion_tokens") if isinstance(usage, dict) else None
        observed_usage = (
            ExactUsage(input_units=incoming, output_units=outgoing)
            if type(incoming) is int and type(outgoing) is int and min(incoming, outgoing) >= 0
            else UnknownUsage()
        )
        settlement = ExternalSettlement(
            request=request.request,
            disposition="completed",
            remote_stopped=True,
            usage=observed_usage,
        )
        if response.is_error:
            return TransportFailure(
                failure="retryable" if response.status_code >= 500 else "permanent", settlement=settlement
            )
        if not isinstance(payload, dict):
            return TransportFailure(failure="malformed_response", settlement=settlement)
        generation = _GENERATION.get()
        if work is not None and generation is not None:
            try:
                content = (payload["choices"][0]["message"]["content"] or "").strip()
            except (KeyError, IndexError, TypeError, AttributeError):
                return TransportFailure(failure="malformed_response", settlement=settlement)
            generation.content, generation.error = content, None
            try:
                generation.value = generation.parser(content)
            except Exception as error:
                generation.error = error
                return TransportFailure(
                    failure="malformed_response" if isinstance(error, ParserException) else "permanent",
                    settlement=settlement,
                )
        return TransportSuccess(
            results=(
                AssociationResult(
                    association=request.associations[0].association,
                    outcome="ready" if current_source_work() is None else "ok",
                    outputs=(),
                    consumed_context_ports=frozenset(),
                ),
            ),
            settlement=settlement,
        )

    async def cancel(self, request: object) -> StopUnknown:
        return StopUnknown()


class NddRequestBridge:
    """Bind actual DD generation identity to pre-admitted independent chains."""

    def __init__(
        self,
        requests: DelegatedRequests,
        slots: Mapping[tuple[str, int], int | tuple[int, ...]],
        physical: NddPhysicalTransport,
    ) -> None:
        self.physical = physical
        self.requests = requests
        self.slots = dict(slots)
        self.rejected = False

    @contextmanager
    def bind(self, designer: Any) -> Iterator[None]:
        self.requests.require_transport(self.physical)
        if version("data-designer") != "0.9.4":
            raise AnonymizerWorkflowError("DD graph request binding requires DataDesigner 0.9.4")
        original = designer._create_resource_provider
        bridge = self

        class Gate(httpx.AsyncBaseTransport):
            def __init__(self, model: Any) -> None:
                self.inner = model._transport
                self.model = model

            async def aclose(self) -> None:
                if self.inner is not None:
                    await self.inner.aclose()

            async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
                if self.inner is None:
                    self.inner = create_retry_transport(
                        self.model._retry_config,
                        strip_rate_limit_codes=True,
                        transport=httpx.AsyncHTTPTransport(limits=self.model._limits),
                    )
                exchange = _Exchange(bridge.physical, self.inner, request)
                token = _EXCHANGE.set(exchange)
                try:
                    work = current_source_work()
                    if work is None:
                        await bridge.requests.readiness()
                        await bridge.requests.probe()
                    else:
                        slot = bridge.slots.get((work.column, work.generation))
                        if slot is None:
                            raise AnonymizerWorkflowError("DD generation has no admitted physical slot")
                        slots = (slot,) if isinstance(slot, int) else slot
                        generation = _GENERATION.get()
                        attempt = generation.attempts if generation is not None else 0
                        if attempt >= len(slots):
                            raise AnonymizerWorkflowError("DD generation exceeded its admitted parser capacity")
                        previous = generation.previous if generation is not None else None
                        evidence = await bridge.requests.aparser_rejected(previous) if previous is not None else None
                        physical = await bridge.requests.request(
                            slots[attempt],
                            association=work.association,
                            source_kind=(
                                "initial"
                                if previous is None
                                else "conversation_restart"
                                if generation is not None and attempt % (generation.correction_steps + 1) == 0
                                else "correction"
                            ),
                            predecessor=previous,
                            evidence=evidence,
                        )
                        if generation is not None:
                            generation.previous = physical
                            generation.attempts += 1
                    if exchange.error is not None:
                        raise exchange.error
                    if exchange.response is None:
                        raise AnonymizerWorkflowError("DD physical request did not return a response")
                    return exchange.response
                except BaseException as error:
                    if error is not exchange.error:
                        bridge.rejected = True
                    raise
                finally:
                    _EXCHANGE.reset(token)

        def resources(*args: Any, **kwargs: Any) -> Any:
            provider = original(*args, **kwargs)
            factory = provider.model_registry._model_facade_factory

            def facade(*args: Any, **kwargs: Any) -> Any:
                result = factory(*args, **kwargs)
                bridge.physical.register(result._client)
                inner = result._client._inner
                if inner._retry_config.max_retries != 0 or inner._aclient is not None or inner._client is not None:
                    bridge.rejected = True
                    raise AnonymizerWorkflowError("DD client is not eligible for graph request binding")
                if inner._transport is not None and not isinstance(inner._transport, httpx.AsyncBaseTransport):
                    bridge.rejected = True
                    raise AnonymizerWorkflowError("DD physical transport must be explicitly registered")
                inner._transport = Gate(inner)
                generate = result.agenerate

                async def agenerate(*args: Any, **kwargs: Any) -> Any:
                    work = current_source_work()
                    if work is None:
                        return await generate(*args, **kwargs)
                    slots = bridge.slots.get((work.column, work.generation))
                    capacity = len(slots) if isinstance(slots, tuple) else 1
                    corrections = kwargs.get("max_correction_steps", 0)
                    restarts = kwargs.get("max_conversation_restarts", 0)
                    if (
                        type(corrections) is not int
                        or corrections < 0
                        or type(restarts) is not int
                        or restarts < 0
                        or (corrections + 1) * (restarts + 1) > capacity
                        or kwargs.get("tool_alias") is not None
                    ):
                        bridge.rejected = True
                        raise AnonymizerWorkflowError("DD parser policy exceeds admitted generation capacity")
                    generation = _Generation(kwargs.get("parser", lambda value: value), corrections)
                    kwargs["parser"] = generation.parse
                    token = _GENERATION.set(generation)
                    try:
                        return await generate(*args, **kwargs)
                    finally:
                        _GENERATION.reset(token)

                result.agenerate = agenerate
                return result

            provider.model_registry._model_facade_factory = facade
            return provider

        designer._create_resource_provider = resources
        try:
            yield
            if self.rejected:
                raise AnonymizerWorkflowError("DD request binding rejected physical work")
        finally:
            designer._create_resource_provider = original
