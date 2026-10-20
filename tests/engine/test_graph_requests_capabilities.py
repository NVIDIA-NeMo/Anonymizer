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

import httpx
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
    retry = types.ModuleType("fake_retry")
    setattr(retry, "create_retry_transport", lambda *args, **kwargs: None)
    parsers = types.ModuleType("fake_parser_errors")
    setattr(parsers, "ParserException", type("ParserException", (Exception,), {}))
    readiness = types.ModuleType("fake_readiness")
    setattr(readiness, "_SKIP_MODEL_HEALTH_CHECKS_ENV", "FAKE_SKIP_HEALTH_CHECKS")
    scheduler = types.ModuleType("fake_scheduler")

    class AsyncTaskScheduler:
        def __init__(self, *, salvage_max_rounds: int = 7) -> None:
            pass

    setattr(scheduler, "AsyncTaskScheduler", AsyncTaskScheduler)
    storage = types.ModuleType("fake_artifact_storage")
    setattr(storage, "BatchStage", SimpleNamespace(PARTIAL_RESULT="partial", FINAL_RESULT="final"))
    setattr(storage, "SDG_CONFIG_FILENAME", "sdg.json")
    for name, module in (
        (_graph_requests._REGISTRY_MODULE, registry),
        (_graph_requests._FACADE_MODULE, facade),
        (_graph_requests._RETRY_MODULE, retry),
        (_graph_requests._PARSER_ERRORS_MODULE, parsers),
        (_graph_requests._READINESS_MODULE, readiness),
        (_graph_requests._SCHEDULER_MODULE, scheduler),
        (_graph_requests._ARTIFACT_STORAGE_MODULE, storage),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(_graph_requests, "version", lambda _name: "0.9.9")
    return types.SimpleNamespace(
        registry=registry,
        facade=facade,
        retry=retry,
        parsers=parsers,
        readiness=readiness,
        scheduler=scheduler,
        storage=storage,
    )


def _bridge() -> tuple[NddRequestBridge, _Physical]:
    physical = _Physical()
    requests: Any = SimpleNamespace(require_transport=lambda _physical: None)
    return NddRequestBridge(requests, {}, cast(Any, physical)), physical


def _provider(factory: Callable[..., Any]) -> SimpleNamespace:
    return SimpleNamespace(model_registry=_Registry(model_facade_factory=factory))


def _assert_capability_message(error: BaseException, attribute: str) -> None:
    message = str(error)
    assert message == (
        f"Anonymizer's DataDesigner binding requires DataDesigner's private `{attribute}`, which the installed "
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


@pytest.mark.parametrize(
    ("module", "name"),
    [("retry", "create_retry_transport"), ("parsers", "ParserException")],
)
def test_missing_module_hook_fails_at_bind(fake_dd: types.SimpleNamespace, module: str, name: str) -> None:
    delattr(getattr(fake_dd, module), name)
    designer = _Designer(_provider(lambda: None))
    bridge, _ = _bridge()
    with pytest.raises(AnonymizerWorkflowError) as raised, bridge.bind(designer):
        pytest.fail("bind must not yield")
    path = {"retry": _graph_requests._RETRY_MODULE, "parsers": _graph_requests._PARSER_ERRORS_MODULE}[module]
    _assert_capability_message(raised.value, f"{path}.{name}")
    assert designer.created == 0


def test_native_map_lookups_resolve_from_the_installed_release(fake_dd: types.SimpleNamespace) -> None:
    assert _graph_requests.skip_health_checks_env() == "FAKE_SKIP_HEALTH_CHECKS"
    assert _graph_requests.scheduler_salvage_rounds() == 7


def test_missing_health_check_switch_fails_with_capability_error(fake_dd: types.SimpleNamespace) -> None:
    delattr(fake_dd.readiness, "_SKIP_MODEL_HEALTH_CHECKS_ENV")
    with pytest.raises(AnonymizerWorkflowError) as raised:
        _graph_requests.skip_health_checks_env()
    _assert_capability_message(raised.value, f"{_graph_requests._READINESS_MODULE}._SKIP_MODEL_HEALTH_CHECKS_ENV")


def test_scheduler_without_salvage_rounds_fails_with_capability_error(fake_dd: types.SimpleNamespace) -> None:
    class AsyncTaskScheduler:
        def __init__(self) -> None:
            pass

    setattr(fake_dd.scheduler, "AsyncTaskScheduler", AsyncTaskScheduler)
    with pytest.raises(AnonymizerWorkflowError) as raised:
        _graph_requests.scheduler_salvage_rounds()
    _assert_capability_message(raised.value, "AsyncTaskScheduler(salvage_max_rounds=...)")


def test_bridge_modules_import_no_data_designer_internals_at_module_scope() -> None:
    """Importing ``anonymizer`` must not fail on a release that moves a hook the bridge checks at bind time."""
    import ast
    from pathlib import Path

    src = Path(__file__).resolve().parents[2] / "src" / "anonymizer"
    lazy = {
        "engine/ndd/_graph_requests.py",
        "engine/replace/_native_map_requests.py",
        "engine/harness/kernel.py",
        "engine/detection/_graph_profile.py",
    }
    offenders = []
    for path in sorted(src.rglob("*.py")):
        relative = path.relative_to(src).as_posix()
        for node in ast.parse(path.read_text()).body:
            if (
                not isinstance(node, ast.ImportFrom)
                or node.module is None
                or not node.module.startswith("data_designer")
            ):
                continue
            names = [alias.name for alias in node.names]
            if any(name.startswith("_") for name in names) or (
                relative in lazy and node.module.startswith("data_designer.engine")
            ):
                offenders.append(f"{relative}: from {node.module} import {', '.join(names)}")
    assert offenders == []


@pytest.mark.parametrize(
    ("error", "wrapped"),
    [
        (httpx.ReadError("read failed"), True),
        (httpx.WriteError("write failed"), True),
        (httpx.RemoteProtocolError("peer closed"), True),
        (httpx.ConnectError("refused"), False),
        (httpx.ReadTimeout("slow"), False),
        (httpx.LocalProtocolError("bad request"), False),
    ],
    ids=lambda value: type(value).__name__ if isinstance(value, Exception) else str(value),
)
def test_single_attempt_errors_keep_the_released_retry_classification(error: Exception, wrapped: bool) -> None:
    """DD sees the same retry decision on every supported release: only errors 0.9.4 retried stay retryable."""
    seen = _graph_requests._single_attempt_error(cast(Any, error))
    if not wrapped:
        assert seen is error
        return
    assert isinstance(seen, _graph_requests.SingleAttemptTransportError)
    assert seen.__cause__ is error
    assert str(seen) == str(error)
    from data_designer.engine.models.clients.errors import ProviderErrorKind, infer_error_kind_from_exception

    assert infer_error_kind_from_exception(seen) is ProviderErrorKind.API_ERROR


class _BuilderDesigner:
    def _create_dataset_builder(self) -> None:
        pass


def test_source_frame_hooks_resolve(fake_dd: types.SimpleNamespace) -> None:
    _graph_requests.require_source_frame_capabilities(_BuilderDesigner())


def test_designer_without_dataset_builder_hook_fails_before_any_work(fake_dd: types.SimpleNamespace) -> None:
    with pytest.raises(AnonymizerWorkflowError) as raised:
        _graph_requests.require_source_frame_capabilities(SimpleNamespace())
    _assert_capability_message(raised.value, "DataDesigner._create_dataset_builder")


@pytest.mark.parametrize(
    ("change", "attribute"),
    [
        ("stage", "BatchStage.FINAL_RESULT"),
        ("filename", f"{_graph_requests._ARTIFACT_STORAGE_MODULE}.SDG_CONFIG_FILENAME"),
    ],
)
def test_artifact_storage_module_hooks_fail_before_any_work(
    fake_dd: types.SimpleNamespace, change: str, attribute: str
) -> None:
    if change == "stage":
        setattr(fake_dd.storage, "BatchStage", SimpleNamespace(PARTIAL_RESULT="partial"))
    else:
        delattr(fake_dd.storage, "SDG_CONFIG_FILENAME")
    with pytest.raises(AnonymizerWorkflowError) as raised:
        _graph_requests.require_source_frame_capabilities(_BuilderDesigner())
    _assert_capability_message(raised.value, attribute)


def _builder(**overrides: Any) -> SimpleNamespace:
    storage = {
        "write_parquet_file": lambda *args, **kwargs: None,
        "load_dataset": lambda *args, **kwargs: None,
        "write_metadata": lambda *args, **kwargs: None,
        "mkdir_if_needed": lambda *args, **kwargs: None,
        "base_dataset_path": "base",
        "final_dataset_path": "final",
    }
    builder = {
        "_initialize_generators_and_graph": lambda: ([], None),
        "_write_builder_config": lambda: None,
        "_data_designer_config": object(),
    }
    for key, value in overrides.items():
        target = storage if key in storage else builder
        if value is _MISSING:
            del target[key]
        else:
            target[key] = value
    return SimpleNamespace(**builder, artifact_storage=SimpleNamespace(**storage))


def test_complete_dataset_builder_passes(fake_dd: types.SimpleNamespace) -> None:
    builder = _builder()
    assert _graph_requests.require_dataset_builder_capabilities(builder) is builder.artifact_storage


@pytest.mark.parametrize(
    ("missing", "attribute"),
    [
        ("_initialize_generators_and_graph", "DatasetBuilder._initialize_generators_and_graph"),
        ("_write_builder_config", "DatasetBuilder._write_builder_config"),
        ("_data_designer_config", "DatasetBuilder._data_designer_config"),
        ("write_parquet_file", "ArtifactStorage.write_parquet_file"),
        ("load_dataset", "ArtifactStorage.load_dataset"),
        ("write_metadata", "ArtifactStorage.write_metadata"),
        ("mkdir_if_needed", "ArtifactStorage.mkdir_if_needed"),
        ("final_dataset_path", "ArtifactStorage.final_dataset_path"),
    ],
)
def test_dataset_builder_without_hook_fails_at_creation(
    fake_dd: types.SimpleNamespace, missing: str, attribute: str
) -> None:
    with pytest.raises(AnonymizerWorkflowError) as raised:
        _graph_requests.require_dataset_builder_capabilities(_builder(**{missing: _MISSING}))
    _assert_capability_message(raised.value, attribute)
