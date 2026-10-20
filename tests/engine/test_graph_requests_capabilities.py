# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The graph request bridge checks DataDesigner's private hooks, not its exact version.

Every case runs against fake DataDesigner modules and classes, so it pins the bridge's behavior on a
release shaped differently from the installed one.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, cast

import pytest

from anonymizer.engine.ndd import _graph_requests
from anonymizer.engine.ndd._graph_requests import SUPPORTED_DATA_DESIGNER_RANGE, NddRequestBridge
from anonymizer.interface.errors import AnonymizerWorkflowError

_SECRET = "Alice Example lives at 1 Main St"


class _Registry:
    def __init__(self, *, model_facade_factory: Callable[..., Any] | None = None) -> None:
        self._model_facade_factory = model_facade_factory


class _Facade:
    async def agenerate(self, *args: Any, **kwargs: Any) -> Any:
        return _SECRET


class _Designer:
    def __init__(self, provider: Any) -> None:
        self.provider = provider
        self.created = 0

    def _create_resource_provider(self, *args: Any, **kwargs: Any) -> Any:
        self.created += 1
        return self.provider


class _Physical:
    def __init__(self) -> None:
        self.clients: list[Any] = []

    def register(self, client: Any) -> None:
        self.clients.append(client)


def _http_client(**overrides: Any) -> SimpleNamespace:
    fields = {
        "_retry_config": SimpleNamespace(max_retries=0),
        "_aclient": None,
        "_client": None,
        "_transport": None,
        "_limits": None,
    }
    fields.update(overrides)
    return SimpleNamespace(**{key: value for key, value in fields.items() if value is not _MISSING})


_MISSING = object()


def _facade(client: Any) -> Any:
    facade = _Facade()
    setattr(facade, "_client", client)
    return facade


@pytest.fixture
def fake_dd(monkeypatch: pytest.MonkeyPatch) -> types.SimpleNamespace:
    """Install fake DataDesigner registry/facade modules and report installed version 0.9.9."""
    registry = types.ModuleType("fake_registry")
    setattr(registry, "ModelRegistry", _Registry)
    facade = types.ModuleType("fake_facade")
    setattr(facade, "ModelFacade", _Facade)
    monkeypatch.setitem(sys.modules, _graph_requests._REGISTRY_MODULE, registry)
    monkeypatch.setitem(sys.modules, _graph_requests._FACADE_MODULE, facade)
    monkeypatch.setattr(_graph_requests, "version", lambda _name: "0.9.9")
    return types.SimpleNamespace(registry=registry, facade=facade)


def _bridge() -> tuple[NddRequestBridge, _Physical]:
    physical = _Physical()
    requests: Any = SimpleNamespace(require_transport=lambda _physical: None)
    return NddRequestBridge(requests, {}, cast(Any, physical)), physical


def _provider(factory: Callable[..., Any]) -> SimpleNamespace:
    return SimpleNamespace(model_registry=_Registry(model_facade_factory=factory))


def _assert_capability_message(error: BaseException, attribute: str) -> None:
    message = str(error)
    assert message == (
        f"DD graph request binding requires DataDesigner's private `{attribute}`, which the installed "
        f"data-designer 0.9.9 does not provide (supported: data-designer {SUPPORTED_DATA_DESIGNER_RANGE})"
    )
    assert _SECRET not in message


def test_pyproject_range_matches_supported_range() -> None:
    import tomllib
    from pathlib import Path

    pyproject = tomllib.loads((Path(__file__).resolve().parents[2] / "pyproject.toml").read_text())
    declared = [dep for dep in pyproject["project"]["dependencies"] if dep.startswith("data-designer")]
    assert declared == [f"data-designer{SUPPORTED_DATA_DESIGNER_RANGE}"]


def test_any_release_in_range_with_the_hooks_binds(fake_dd: types.SimpleNamespace) -> None:
    client = SimpleNamespace(_inner=_http_client())
    designer = _Designer(_provider(lambda: _facade(client)))
    bridge, physical = _bridge()

    with bridge.bind(designer):
        provider = designer._create_resource_provider()
        facade = provider.model_registry._model_facade_factory()

    assert designer.created == 1
    assert physical.clients == [client]
    assert isinstance(client._inner._transport, _graph_requests.httpx.AsyncBaseTransport)
    assert facade.agenerate is not _Facade.agenerate
    assert designer._create_resource_provider.__func__ is _Designer._create_resource_provider


def test_missing_resource_provider_hook_fails_at_bind(fake_dd: types.SimpleNamespace) -> None:
    class Designer:
        pass

    bridge, _ = _bridge()
    with pytest.raises(AnonymizerWorkflowError) as raised, bridge.bind(Designer()):
        pytest.fail("bind must not yield")
    _assert_capability_message(raised.value, "DataDesigner._create_resource_provider")


def test_registry_without_facade_factory_parameter_fails_at_bind(
    fake_dd: types.SimpleNamespace,
) -> None:
    class Registry:
        def __init__(self, *, model_configs: Any = None) -> None:
            pass

    fake_dd.registry.ModelRegistry = Registry
    designer = _Designer(_provider(lambda: None))
    bridge, _ = _bridge()
    with pytest.raises(AnonymizerWorkflowError) as raised, bridge.bind(designer):
        pytest.fail("bind must not yield")
    _assert_capability_message(raised.value, "ModelRegistry(model_facade_factory=...)")
    assert designer.created == 0


def test_facade_without_async_generate_fails_at_bind(fake_dd: types.SimpleNamespace) -> None:
    class Facade:
        def generate(self) -> None:
            pass

    fake_dd.facade.ModelFacade = Facade
    designer = _Designer(_provider(lambda: None))
    bridge, _ = _bridge()
    with pytest.raises(AnonymizerWorkflowError) as raised, bridge.bind(designer):
        pytest.fail("bind must not yield")
    _assert_capability_message(raised.value, "ModelFacade.agenerate")
    assert designer.created == 0


def test_provider_without_facade_factory_fails_at_creation_even_if_swallowed(
    fake_dd: types.SimpleNamespace,
) -> None:
    registry = _Registry()
    del registry._model_facade_factory
    designer = _Designer(SimpleNamespace(model_registry=registry))
    bridge, _ = _bridge()

    with pytest.raises(AnonymizerWorkflowError) as raised, bridge.bind(designer):
        with pytest.raises(AnonymizerWorkflowError) as inner:
            designer._create_resource_provider()
        _assert_capability_message(inner.value, "ModelRegistry._model_facade_factory")
        # DataDesigner may absorb the raise as a dropped row; leaving the binding still aborts the run.
    assert raised.value is inner.value


@pytest.mark.parametrize(
    ("client", "attribute"),
    [
        (SimpleNamespace(), "ModelRequestExecutor._inner"),
        (SimpleNamespace(_inner=_http_client(_limits=_MISSING)), "HttpModelClient._limits"),
        (SimpleNamespace(_inner=_http_client(_transport=_MISSING)), "HttpModelClient._transport"),
        (SimpleNamespace(_inner=_http_client(_retry_config=SimpleNamespace())), "RetryConfig.max_retries"),
    ],
    ids=["inner", "limits", "transport", "max_retries"],
)
def test_facade_without_client_fields_fails_before_any_request_even_if_swallowed(
    fake_dd: types.SimpleNamespace, client: Any, attribute: str
) -> None:
    designer = _Designer(_provider(lambda: _facade(client)))
    bridge, physical = _bridge()

    with pytest.raises(AnonymizerWorkflowError) as raised, bridge.bind(designer):
        factory = designer._create_resource_provider().model_registry._model_facade_factory
        with pytest.raises(AnonymizerWorkflowError) as inner:
            factory()
        _assert_capability_message(inner.value, attribute)
    assert raised.value is inner.value
    assert physical.clients == []


def test_facade_without_client_fails_at_creation(fake_dd: types.SimpleNamespace) -> None:
    designer = _Designer(_provider(lambda: _Facade()))
    bridge, _ = _bridge()
    with pytest.raises(AnonymizerWorkflowError) as raised, bridge.bind(designer):
        designer._create_resource_provider().model_registry._model_facade_factory()
    _assert_capability_message(raised.value, "ModelFacade._client")
