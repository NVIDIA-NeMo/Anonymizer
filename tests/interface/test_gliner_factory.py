# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the public GLiNER runtime factory."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from data_designer.config.models import ModelProvider
from pydantic import ValidationError

import anonymizer


def test_native_factory_uses_inputs_validated_before_runtime_acquisition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from anonymizer.engine.local_inference import runtime as controller

    model_path = tmp_path / "models.yaml"
    provider_path = tmp_path / "providers.yaml"
    model_path.write_text(
        json.dumps(
            {
                "model_configs": [{"alias": "custom-llm", "model": "test/llm", "provider": "custom"}],
                "selected_models": {"replace": {"replacement_generator": "custom-llm"}},
            }
        )
    )
    provider_path.write_text(
        json.dumps({"providers": [{"name": "custom", "endpoint": "https://example.test/v1", "api_key": "KEY"}]})
    )
    runtime = controller._LocalRuntime(Mock(), "http://127.0.0.1:1234/v1", "cpu", "cpu", "token")

    @contextmanager
    def acquire(device: str, dtype: str) -> Iterator[controller._LocalRuntime]:
        model_path.unlink()
        provider_path.unlink()
        yield runtime

    monkeypatch.setattr(controller, "singleton_runtime", acquire)
    client = anonymizer.create_anonymizer(
        gliner=anonymizer.NativeGliner(device="cpu"),
        model_configs=model_path,
        model_providers=provider_path,
        artifact_path=tmp_path / "artifacts",
    )
    assert client._selected_models.replace.replacement_generator == "custom-llm"
    assert next(model for model in client._model_configs if model.alias == "custom-llm").model == "test/llm"
    assert client._resolved_providers[0].name == "custom"


def test_external_endpoint_preserves_bundled_generation_providers() -> None:
    client = anonymizer.create_anonymizer(
        gliner=anonymizer.GlinerEndpoint(url="http://127.0.0.1:43210/v1", model="org/gliner-served", api_key_env=None)
    )
    assert client._selected_models.detection.entity_detector == "anonymizer-gliner-detector"
    assert any(provider.name == "openrouter" for provider in client._resolved_providers)
    injected = next(
        provider for provider in client._resolved_providers if provider.name == "anonymizer-gliner-endpoint"
    )
    assert injected.endpoint == "http://127.0.0.1:43210/v1"


@pytest.mark.parametrize("device", ["", "gpu", "CUDA"])
def test_native_request_rejects_unsupported_devices(device: str) -> None:
    with pytest.raises(ValidationError):
        anonymizer.NativeGliner(device=device)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [("url", ""), ("model", ""), ("api_key_env", "")],
)
def test_external_endpoint_rejects_empty_connection_values(field: str, value: str) -> None:
    values = {
        "url": "https://gliner.example.test/v1",
        "model": "org/model",
        "api_key_env": "GLINER_TOKEN",
    }
    values[field] = value

    with pytest.raises(ValidationError):
        anonymizer.GlinerEndpoint(**values)


def test_external_endpoint_injects_detector_and_preserves_other_configuration() -> None:
    assert hasattr(anonymizer, "GlinerEndpoint")
    assert hasattr(anonymizer, "create_anonymizer")
    GlinerEndpoint = anonymizer.GlinerEndpoint
    create_anonymizer = anonymizer.create_anonymizer
    source = json.dumps(
        {
            "model_configs": [
                {"alias": "old-detector", "model": "test/old", "provider": "custom"},
                {"alias": "custom-llm", "model": "test/llm", "provider": "custom"},
            ],
            "selected_models": {
                "detection": {
                    "entity_detector": "old-detector",
                    "entity_validator": "custom-llm",
                    "entity_augmenter": "custom-llm",
                },
                "replace": {"replacement_generator": "custom-llm"},
            },
        }
    )
    provider = ModelProvider(name="custom", endpoint="https://example.test/v1", api_key="CUSTOM_KEY")
    result = create_anonymizer(
        gliner=GlinerEndpoint(
            url="https://gliner.example.test/v1", model="org/gliner-served", api_key_env="GLINER_TOKEN"
        ),
        model_configs=source,
        model_providers=[provider],
    )
    injected = next(config for config in result._model_configs if config.alias == "anonymizer-gliner-detector")
    assert injected.model == "org/gliner-served"
    assert injected.provider == "anonymizer-gliner-endpoint"
    assert injected.skip_health_check is True
    assert next(config for config in result._model_configs if config.alias == "custom-llm").model == "test/llm"
    assert result._selected_models.detection.entity_detector == "anonymizer-gliner-detector"
    assert result._selected_models.detection.entity_validator == ["custom-llm"]
    assert result._selected_models.detection.entity_augmenter == "custom-llm"
    assert result._selected_models.replace.replacement_generator == "custom-llm"
    assert result._resolved_providers[0] is provider
    endpoint_provider = result._resolved_providers[-1]
    assert endpoint_provider.name == "anonymizer-gliner-endpoint"
    assert endpoint_provider.endpoint == "https://gliner.example.test/v1"
    assert endpoint_provider.api_key == "GLINER_TOKEN"


@pytest.mark.parametrize("collision", ["anonymizer-gliner-detector", "anonymizer-gliner-endpoint"])
def test_external_endpoint_rejects_reserved_name_collisions(collision: str) -> None:
    GlinerEndpoint = anonymizer.GlinerEndpoint
    create_anonymizer = anonymizer.create_anonymizer
    model_configs: str | None = None
    providers: list[ModelProvider] | None = None
    if collision == "anonymizer-gliner-detector":
        model_configs = json.dumps(
            {"model_configs": [{"alias": collision, "model": "test/model", "provider": "nvidia"}]}
        )
    else:
        providers = [ModelProvider(name=collision, endpoint="https://example.test/v1")]

    with pytest.raises(ValueError, match="reserved"):
        create_anonymizer(
            gliner=GlinerEndpoint(
                url="https://gliner.example.test/v1",
                model="org/model",
                api_key_env="GLINER_TOKEN",
            ),
            model_configs=model_configs,
            model_providers=providers,
        )


def test_external_endpoint_does_not_import_or_control_native_runtime() -> None:
    GlinerEndpoint = anonymizer.GlinerEndpoint
    create_anonymizer = anonymizer.create_anonymizer
    with patch("subprocess.Popen") as popen:
        create_anonymizer(
            gliner=GlinerEndpoint(
                url="https://gliner.example.test/v1",
                model="org/model",
                api_key_env="GLINER_TOKEN",
            )
        )

    popen.assert_not_called()


def test_built_wheel_external_factory_excludes_heavy_runtime_dependencies(tmp_path: Path) -> None:
    build = subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(tmp_path)],
        cwd=Path(__file__).parents[2],
        text=True,
        capture_output=True,
        check=False,
    )
    assert build.returncode == 0, build.stderr
    wheel = next(tmp_path.glob("*.whl"))
    installed = tmp_path / "installed"
    install = subprocess.run(
        ["uv", "pip", "install", "--target", str(installed), "--no-deps", str(wheel)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert install.returncode == 0, install.stderr
    outside_checkout = tmp_path / "outside-checkout"
    outside_checkout.mkdir()
    script = f"""
import sys
sys.path.insert(0, {str(installed)!r})
from anonymizer import GlinerEndpoint, create_anonymizer
instance = create_anonymizer(
    gliner=GlinerEndpoint(url='https://example.test/v1', model='org/model', api_key_env='TOKEN')
)
assert type(instance).__name__ == 'Anonymizer'
forbidden = ('torch', 'gliner2', 'fastapi', 'vllm', 'tools.inference_service_compiler')
def is_forbidden(name):
    return any(name == module or name.startswith(module + '.') for module in forbidden)
assert all(is_forbidden(module) for module in forbidden)
loaded = sorted(name for name in sys.modules if is_forbidden(name))
assert not loaded, loaded
assert 'anonymizer.engine.local_inference.runtime' not in sys.modules
assert __import__('anonymizer').__file__.startswith({str(installed)!r})
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=outside_checkout,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_native_factory_preserves_word_chunk_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from anonymizer.engine.local_inference import runtime as _runtime

    runtime = _runtime._LocalRuntime(
        process=Mock(),
        endpoint="http://127.0.0.1:1234/v1",
        requested_device="cpu",
        selected_device="cpu",
        token="token",
    )
    from contextlib import nullcontext

    monkeypatch.setattr(_runtime, "singleton_runtime", lambda _device, _dtype: nullcontext(runtime))
    provider = ModelProvider(name="custom", endpoint="https://example.test/v1", api_key="CUSTOM_KEY")
    source = json.dumps({"model_configs": [{"alias": "custom-llm", "model": "test/llm", "provider": "custom"}]})
    run_config = anonymizer.RunConfig(buffer_size=123)
    client = anonymizer.create_anonymizer(
        gliner=anonymizer.NativeGliner(device="cpu"),
        model_configs=source,
        model_providers=[provider],
        artifact_path=tmp_path,
        data_designer_run_config=run_config,
    )
    assert any(item.alias == "custom-llm" for item in client._model_configs)
    assert client._resolved_providers[0] is provider
    assert client._data_designer._artifact_path == tmp_path
    assert client._data_designer._run_config == run_config
    detector = next(
        model for model in client._model_configs if model.alias == client._selected_models.detection.entity_detector
    )
    assert detector.inference_parameters.extra_body == {"chunk_length": 128, "overlap": 32}


def test_session_builds_real_client_and_preserves_configuration(tmp_path: Path) -> None:
    from anonymizer.engine.local_inference import runtime as controller

    process = Mock()
    process.poll.return_value = None
    candidate = controller._LocalRuntime(process, "http://127.0.0.1:1234/v1", "cpu", "cpu", "token")
    provider = ModelProvider(name="custom", endpoint="https://example.test/v1", api_key="CUSTOM_KEY")
    source = json.dumps({"model_configs": [{"alias": "custom-llm", "model": "test/llm", "provider": "custom"}]})
    run_config = anonymizer.RunConfig(buffer_size=123)
    with (
        patch.object(controller, "_start_runtime", return_value=candidate),
        patch.object(controller, "_wait_until_ready"),
        anonymizer.native_runtime(device="cpu") as session,
    ):
        client = session.create_anonymizer(
            model_configs=source,
            model_providers=[provider],
            artifact_path=tmp_path,
            data_designer_run_config=run_config,
        )
        assert client._data_designer._artifact_path == tmp_path
        assert client._data_designer._run_config == run_config
        assert session.endpoint.url == candidate.endpoint
        assert client._selected_models.detection.entity_detector == "anonymizer-gliner-detector"
        assert any(item.alias == "custom-llm" for item in client._model_configs)
        assert client._resolved_providers[0] is provider
        assert client._resolved_providers[-1].endpoint == candidate.endpoint
        detector = next(item for item in client._model_configs if item.alias == "anonymizer-gliner-detector")
        assert detector.inference_parameters.extra_body == {"chunk_length": 128, "overlap": 32}
        process.terminate.assert_not_called()
    process.terminate.assert_called_once()


def test_session_cleans_up_when_client_construction_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    from anonymizer.engine.local_inference import runtime as controller

    monkeypatch.setenv(controller.LOCAL_TOKEN_ENV, "original")
    process = Mock()
    process.poll.return_value = None
    candidate = controller._LocalRuntime(process, "http://127.0.0.1:1234/v1", "cpu", "cpu", "token")

    def start(_: str, __: str = "auto") -> controller._LocalRuntime:
        controller._set_token_environment("token")
        return candidate

    with (
        patch.object(controller, "_start_runtime", side_effect=start),
        patch.object(controller, "_wait_until_ready"),
        pytest.raises(ValueError, match="reserved"),
        anonymizer.native_runtime(device="cpu") as session,
    ):
        session.create_anonymizer(
            model_providers=[ModelProvider(name="anonymizer-gliner-endpoint", endpoint="https://example.test")]
        )
    process.terminate.assert_called_once()
    assert controller.os.environ[controller.LOCAL_TOKEN_ENV] == "original"


def test_notebook_factory_remains_a_native_singleton_wrapper() -> None:
    from anonymizer.notebooks import _runtime

    with patch.object(_runtime, "create_public_anonymizer", return_value=Mock()) as factory:
        _runtime.create_anonymizer(gliner_device="cpu")
    assert factory.call_args.kwargs["gliner"] == anonymizer.NativeGliner(device="cpu")


def test_native_dtype_reaches_the_runtime_for_both_construction_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    from contextlib import nullcontext

    from anonymizer.engine.local_inference import runtime as _runtime

    runtime = _runtime._LocalRuntime(Mock(), "http://127.0.0.1:1234/v1", "cpu", "cpu", "token")
    requests: list[tuple[str, str]] = []

    def acquire(device: str, dtype: str) -> object:
        requests.append((device, dtype))
        return nullcontext(runtime)

    monkeypatch.setattr(_runtime, "singleton_runtime", acquire)
    monkeypatch.setattr(_runtime, "runtime_context", acquire)
    assert anonymizer.NativeGliner().dtype == "auto"
    anonymizer.create_anonymizer(gliner=anonymizer.NativeGliner(device="cpu", dtype="bfloat16"))
    with anonymizer.native_runtime(device="cpu", dtype="bfloat16"):
        pass
    with anonymizer.native_runtime():
        pass
    assert requests == [("cpu", "bfloat16"), ("cpu", "bfloat16"), ("auto", "auto")]
