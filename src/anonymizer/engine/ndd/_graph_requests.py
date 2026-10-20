# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capability-checked DD HTTP entry borrowed from the invocation request owner.

This development binding observes each actual parser attempt in its generation chain.
The caller registers NddPhysicalTransport with ForeignLoopTransport before
execution. Its cleanup closes actual DD clients after the worker drains. DD owns
conversation restarts, correction turns and transient-failure retries; a bridge that
declares ``retries`` accounts each retry as its own retry child.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import logging
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
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
    FailureClass,
    PhysicalRequestId,
    StopUnknown,
    TransportFailure,
    TransportSuccess,
    UnknownUsage,
)
from anonymizer.engine.ndd._source_frame import current_source_work
from anonymizer.interface.errors import AnonymizerWorkflowError

SUPPORTED_DATA_DESIGNER_RANGE = ">=0.9.4,<0.10"
"""DataDesigner releases this graph request binding supports.

The binding relies on private DataDesigner hooks, so it checks for them instead of an exact version (see
``require_binding_capabilities``). Must stay in sync with the ``data-designer`` specifier in
``pyproject.toml``; ``tests/engine/test_graph_requests_version_pin.py`` checks this.
"""

_REGISTRY_MODULE = "data_designer.engine.models.registry"
_FACADE_MODULE = "data_designer.engine.models.facade"
_HTTP_CLIENT_FIELDS = ("_retry_config", "_aclient", "_client", "_transport", "_limits")


def _installed_data_designer() -> str:
    try:
        return version("data-designer")
    except PackageNotFoundError:
        return "unknown"


def _missing_capability(attribute: str) -> AnonymizerWorkflowError:
    return AnonymizerWorkflowError(
        f"DD graph request binding requires DataDesigner's private `{attribute}`, which the installed "
        f"data-designer {_installed_data_designer()} does not provide (supported: data-designer "
        f"{SUPPORTED_DATA_DESIGNER_RANGE})"
    )


def require_binding_capabilities(designer: Any) -> None:
    """Check, before any DD work runs, every private hook resolvable without a live DD resource.

    Instance attributes that DataDesigner assigns in constructors (the registry's facade factory, the
    facade's client and the HTTP client's fields) are checked when each object is first created, still
    before any request is sent.
    """
    if not callable(getattr(type(designer), "_create_resource_provider", None)):
        raise _missing_capability("DataDesigner._create_resource_provider")
    try:
        registry = importlib.import_module(_REGISTRY_MODULE).ModelRegistry
        facade = importlib.import_module(_FACADE_MODULE).ModelFacade
    except (ImportError, AttributeError) as error:
        raise _missing_capability(f"{_REGISTRY_MODULE}.ModelRegistry / {_FACADE_MODULE}.ModelFacade") from error
    try:
        parameters = inspect.signature(registry.__init__).parameters
    except (TypeError, ValueError):
        parameters = {}
    if "model_facade_factory" not in parameters:
        raise _missing_capability("ModelRegistry(model_facade_factory=...)")
    if not inspect.iscoroutinefunction(getattr(facade, "agenerate", None)):
        raise _missing_capability("ModelFacade.agenerate")


def _require_attribute(owner: Any, name: str, label: str) -> Any:
    try:
        return getattr(owner, name)
    except AttributeError:
        raise _missing_capability(label) from None


@dataclass
class _Exchange:
    owner: NddPhysicalTransport
    transport: httpx.AsyncBaseTransport
    request: httpx.Request
    response: httpx.Response | None = None
    error: Exception | None = None
    failure: str | None = None
    retries: bool = False


_RETRY_FAILURES = frozenset({"retryable", "transport_unknown", "rejected_before_acceptance"})


@dataclass
class _Chain:
    """One generation's physical attempts for one record, in order; DD decides whether and when to retry."""

    used: int = 0
    previous: PhysicalRequestId | None = None
    retry: bool = False
    beyond_admitted: int = 0


def _observe_generation(generation: _Generation, payload: Any) -> FailureClass | None:
    """Record DD's parse of a completed response for the facade; return its failure class, if any."""
    try:
        content = (payload["choices"][0]["message"]["content"] or "").strip()
    except (KeyError, IndexError, TypeError, AttributeError):
        return "malformed_response"
    generation.content, generation.error = content, None
    try:
        generation.value = generation.parser(content)
    except Exception as error:
        generation.error = error
        return "malformed_response" if isinstance(error, ParserException) else "permanent"
    return None


logger = logging.getLogger(__name__)

_EXCHANGE: ContextVar[_Exchange] = ContextVar("anonymizer_dd_physical_exchange")


@dataclass
class _Generation:
    parser: Callable[[str], Any]
    correction_steps: int = 0
    content: str | None = None
    value: Any = None
    error: Exception | None = None
    turns: int = 0

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
        self._exchanges: dict[PhysicalRequestId, asyncio.Task[Any]] = {}

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
        task = asyncio.current_task()
        if task is not None:
            self._exchanges[request.request] = task
        try:
            response = await exchange.transport.handle_async_request(exchange.request)
            await response.aread()
        except httpx.TransportError as error:
            if not exchange.retries:
                exchange.error = error
                raise
            # DD sees the same error and may retry it. A refused connection never sent the request;
            # any other transport error leaves the provider's side unknown.
            exchange.error = error
            refused = isinstance(error, (httpx.ConnectError, httpx.ConnectTimeout))
            exchange.failure = "rejected_before_acceptance" if refused else "transport_unknown"
            return TransportFailure(
                failure=exchange.failure,
                settlement=ExternalSettlement(
                    request=request.request,
                    disposition="rejected" if refused else "unknown",
                    remote_stopped=True if refused else None,
                    usage=ExactUsage(input_units=0, output_units=0) if refused else UnknownUsage(),
                ),
            )
        except Exception as error:
            exchange.error = error
            raise
        finally:
            self._exchanges.pop(request.request, None)
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
            # DD retries 5xx and rate limits (429); other client errors are final.
            retryable = response.status_code >= 500 or (exchange.retries and response.status_code == 429)
            exchange.failure = "retryable" if retryable else "permanent"
            return TransportFailure(failure=exchange.failure, settlement=settlement)
        if not isinstance(payload, dict):
            exchange.failure = "malformed_response"
            return TransportFailure(failure="malformed_response", settlement=settlement)
        generation = _GENERATION.get()
        if work is not None and generation is not None:
            failure = _observe_generation(generation, payload)
            if failure is not None:
                return TransportFailure(failure=failure, settlement=settlement)
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
        """Abort the local HTTP exchange; the provider gives no stop evidence, so the request stays unknown."""
        exchange = self._exchanges.get(request) if isinstance(request, PhysicalRequestId) else None
        if exchange is not None and not exchange.done():
            exchange.cancel()
        return StopUnknown()


class NddRequestBridge:
    """Bind actual DD generation identity to pre-admitted independent chains."""

    def __init__(
        self,
        requests: DelegatedRequests,
        slots: Mapping[tuple[str, int], int | tuple[int, ...]],
        physical: NddPhysicalTransport,
        *,
        retries: bool = False,
    ) -> None:
        """``retries``: the admitted chains declare retry follow-ups, so DD's transient-failure retries are
        accounted as retry children instead of rejecting the binding."""
        self.physical = physical
        self.retries = retries
        self.requests = requests
        self.slots = dict(slots)
        self.chains: dict[Any, _Chain] = {}
        self.rejected = False
        self.capability_error: AnonymizerWorkflowError | None = None

    def _capability(self, error: AnonymizerWorkflowError) -> AnonymizerWorkflowError:
        """Record a capability failure so it aborts the run even if DD absorbs the raise per row."""
        self.rejected = True
        if self.capability_error is None:
            self.capability_error = error
        return error

    @contextmanager
    def bind(self, designer: Any) -> Iterator[None]:
        self.requests.require_transport(self.physical)
        require_binding_capabilities(designer)
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
                exchange = _Exchange(bridge.physical, self.inner, request, retries=bridge.retries)
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
                        chain = bridge.chains.setdefault((work.association, work.column, work.generation), _Chain())
                        if chain.used >= len(slots) or chain.beyond_admitted:
                            if not (bridge.retries and chain.retry):
                                raise AnonymizerWorkflowError("DD generation exceeded its admitted request capacity")
                            # DD keeps retrying past the admitted leaves (429 deferrals are unbounded). The
                            # invocation already declares an unknown request bound and no hard budget, so these
                            # attempts run as DD-owned, unaccounted requests instead of failing every row.
                            chain.beyond_admitted += 1
                            response = await self.inner.handle_async_request(request)
                            await response.aread()
                            if generation is not None and not response.is_error:
                                try:
                                    payload = response.json()
                                except ValueError:
                                    payload = None
                                _observe_generation(generation, payload)
                            return response
                        previous = chain.previous
                        # A retry follows a transient failure; a parser turn follows a rejected response.
                        retry = previous is not None and chain.retry
                        evidence = (
                            await bridge.requests.aparser_rejected(previous)
                            if previous is not None and not retry
                            else None
                        )
                        physical = await bridge.requests.request(
                            slots[chain.used],
                            association=work.association,
                            source_kind=(
                                "initial"
                                if previous is None
                                else "retry"
                                if retry
                                else "conversation_restart"
                                if generation is not None and generation.turns % (generation.correction_steps + 1) == 0
                                else "correction"
                            ),
                            predecessor=previous,
                            evidence=evidence,
                        )
                        chain.used += 1
                        chain.previous = physical
                        chain.retry = bridge.retries and exchange.failure in _RETRY_FAILURES
                        if generation is not None and not chain.retry:
                            generation.turns += 1
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
            try:
                registry = _require_attribute(provider, "model_registry", "ResourceProvider.model_registry")
                factory = _require_attribute(registry, "_model_facade_factory", "ModelRegistry._model_facade_factory")
                if not callable(factory):
                    raise _missing_capability("ModelRegistry._model_facade_factory")
            except AnonymizerWorkflowError as error:
                raise bridge._capability(error) from None

            def facade(*args: Any, **kwargs: Any) -> Any:
                result = factory(*args, **kwargs)
                try:
                    client = _require_attribute(result, "_client", "ModelFacade._client")
                    inner = _require_attribute(client, "_inner", "ModelRequestExecutor._inner")
                    for field in _HTTP_CLIENT_FIELDS:
                        _require_attribute(inner, field, f"HttpModelClient.{field}")
                    _require_attribute(inner._retry_config, "max_retries", "RetryConfig.max_retries")
                except AnonymizerWorkflowError as error:
                    raise bridge._capability(error) from None
                bridge.physical.register(client)
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
            beyond = sum(chain.beyond_admitted for chain in self.chains.values())
            if beyond:
                logger.warning(
                    "%d DataDesigner retry request(s) beyond the admitted per-record allowance ran outside graph "
                    "request accounting; this invocation declares no request upper bound.",
                    beyond,
                )
            if self.capability_error is not None:
                raise self.capability_error
