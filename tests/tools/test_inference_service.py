# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Behavior tests for the source-tree inference service compiler."""

from __future__ import annotations

import importlib
import json
import stat
import sys
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest import mock

import httpx
import pytest
from pydantic import ValidationError

REPO_ROOT = Path(__file__).resolve().parents[2]
CLI_PATH = REPO_ROOT / "tools" / "inference_service.py"
TOOLS_ROOT = REPO_ROOT / "tools"
NATIVE_GLINER_PATH = TOOLS_ROOT / "inference_service_compiler" / "native_gliner.py"
VLLM_SERVER_PATH = TOOLS_ROOT / "inference_service_compiler" / "vllm_server.py"
PROFILE_ROOT = TOOLS_ROOT / "inference_service_profiles"
REMOVED_GLINER_PATH = TOOLS_ROOT / "serve_gliner.py"


def load_compiler_modules() -> tuple[ModuleType, ModuleType]:
    """Load source-tree modules through the same import root as the CLI."""
    sys.path.insert(0, str(TOOLS_ROOT))
    try:
        models = importlib.import_module("inference_service_compiler.models")
        compiler = importlib.import_module("inference_service_compiler.compiler")
    finally:
        sys.path.pop(0)
    return models, compiler


def load_cli_module() -> ModuleType:
    """Load the CLI through the source-tree import root."""
    sys.path.insert(0, str(TOOLS_ROOT))
    try:
        return importlib.import_module("inference_service_compiler.cli")
    finally:
        sys.path.pop(0)


def load_runtime_module() -> ModuleType:
    """Load the runtime through the source-tree import root."""
    sys.path.insert(0, str(TOOLS_ROOT))
    try:
        return importlib.import_module("inference_service_compiler.runtime")
    finally:
        sys.path.pop(0)


def build_generation_plan(
    models: ModuleType,
    compiler: ModuleType,
    *,
    api_key_env: str | None = None,
    docker: bool = False,
) -> Any:
    """Build one local vLLM plan without runtime effects."""
    intent = models.InferenceIntent(
        task=models.Generation(chat=True),
        model=models.HuggingFaceModel(model_id="openai/gpt-oss-20b", revision="abcdef0123456789"),
        engine=models.VllmEngine(api_key_env=api_key_env),
        placement=(
            models.DockerPlacement(
                host="127.0.0.1",
                port=8000,
                image="vllm/vllm-openai:v0.27.1",
                gpus="all",
            )
            if docker
            else models.LocalProcessPlacement(host="127.0.0.1", port=8000)
        ),
        access=models.DirectAccess(),
        lifecycle=models.ManagedLifecycle(startup_timeout_seconds=30),
    )
    return compiler.compile_intent(intent, source_revision="3f68c145")


def test_cli_is_a_directly_executable_source_tree_entrypoint() -> None:
    """The compiler has one stable CLI without creating an installable package."""
    assert CLI_PATH.is_file()
    assert CLI_PATH.stat().st_mode & stat.S_IXUSR


def test_compile_native_gliner_local_process_plan() -> None:
    """Native detection compiles into a deterministic effect-free local plan."""
    models, compiler = load_compiler_modules()
    assert callable(getattr(compiler, "compile_intent", None))

    intent = models.InferenceIntent(
        task=models.EntityDetection(dynamic_labels=True, offsets=True, scores=True),
        model=models.HuggingFaceModel(model_id="nvidia/gliner-pii", revision="0123456789abcdef"),
        engine=models.NativeGlinerEngine(family="nvidia-gliner", device="cpu"),
        placement=models.LocalProcessPlacement(host="127.0.0.1", port=8001),
        access=models.DirectAccess(),
        lifecycle=models.ManagedLifecycle(startup_timeout_seconds=120),
    )

    first = compiler.compile_intent(intent, source_revision="3f68c145")
    second = compiler.compile_intent(intent, source_revision="3f68c145")

    assert first == second
    assert first.schema_version == "inference-service.run-plan/v1"
    assert first.intent_digest == compiler.digest_model(intent)
    assert first.plan_digest == compiler.digest_plan(first)
    assert first.endpoint.url == "http://127.0.0.1:8001/v1"
    assert first.readiness.url == "http://127.0.0.1:8001/v1/models"
    assert first.expected_model == "nvidia/gliner-pii"
    assert first.declared_capabilities == ("dynamic-labels", "offsets", "scores")
    assert first.required_capabilities == ("dynamic-labels", "offsets", "scores")
    assert first.runtime.kind == "local-process"
    assert first.command.render_argv() == (
        "uv",
        "run",
        "--script",
        "tools/inference_service_compiler/native_gliner.py",
        "--host",
        "127.0.0.1",
        "--port",
        "8001",
        "--model",
        "nvidia-gliner",
        "--checkpoint",
        "nvidia/gliner-pii",
        "--revision",
        "0123456789abcdef",
    )

    with pytest.raises(Exception, match="frozen"):
        first.endpoint.port = 9000


def test_compile_vllm_docker_plan_keeps_secrets_symbolic() -> None:
    """Docker plans pin the image and never serialize an API-key value."""
    models, compiler = load_compiler_modules()
    intent = models.InferenceIntent(
        task=models.Generation(chat=True),
        model=models.HuggingFaceModel(model_id="openai/gpt-oss-20b", revision="abcdef0123456789"),
        engine=models.VllmEngine(
            served_model_name="anonymizer-local",
            api_key_env="LOCAL_VLLM_API_KEY",
            tensor_parallel_size=2,
            gpu_memory_utilization=0.8,
            max_model_len=4096,
            eager=True,
        ),
        placement=models.DockerPlacement(
            host="127.0.0.1",
            port=8000,
            image="vllm/vllm-openai:v0.27.1",
            gpus="all",
        ),
        access=models.DirectAccess(),
        lifecycle=models.ManagedLifecycle(startup_timeout_seconds=300),
    )

    plan = compiler.compile_intent(intent, source_revision="3f68c145")
    rendered = plan.model_dump_json()

    assert plan.runtime.kind == "docker"
    assert plan.runtime.image == "vllm/vllm-openai:v0.27.1"
    assert plan.endpoint.url == "http://127.0.0.1:8000/v1"
    assert plan.expected_model == "anonymizer-local"
    assert plan.required_capabilities == ("chat-completions",)
    assert plan.declared_capabilities == ("chat-completions",)
    assert "LOCAL_VLLM_API_KEY" in rendered
    assert "test-secret" not in rendered
    assert "test-secret" not in plan.command.render_argv()
    assert plan.command.render_environment() == {"VLLM_API_KEY": "<secret:LOCAL_VLLM_API_KEY>"}
    assert plan.command.render_environment(resolve_secrets={"LOCAL_VLLM_API_KEY": "test-secret"}) == {
        "VLLM_API_KEY": "test-secret"
    }
    assert plan.command.render_argv()[:6] == (
        "docker",
        "run",
        "--detach",
        "--rm",
        "--gpus",
        "all",
    )
    environment_index = plan.command.render_argv().index("--env")
    assert plan.command.render_argv()[environment_index : environment_index + 2] == ("--env", "VLLM_API_KEY")
    revision_index = plan.command.render_argv().index("--revision")
    assert plan.command.render_argv()[revision_index : revision_index + 4] == (
        "--revision",
        "abcdef0123456789",
        "--tokenizer-revision",
        "abcdef0123456789",
    )


def test_compile_local_vllm_plan_uses_the_python_server_factory() -> None:
    """Local vLLM runs through the source-owned Python runtime, not its CLI binary."""
    models, compiler = load_compiler_modules()
    intent = models.InferenceIntent(
        task=models.Generation(chat=True),
        model=models.HuggingFaceModel(model_id="openai/gpt-oss-20b", revision="abcdef0123456789"),
        engine=models.VllmEngine(python_executable=".venv/bin/python"),
        placement=models.LocalProcessPlacement(host="127.0.0.1", port=8000),
        access=models.DirectAccess(),
        lifecycle=models.ManagedLifecycle(),
    )

    plan = compiler.compile_intent(intent, source_revision="3f68c145")

    assert plan.command.render_argv()[:3] == (
        ".venv/bin/python",
        "tools/inference_service_compiler/vllm_server.py",
        "openai/gpt-oss-20b",
    )
    assert "serve" not in plan.command.render_argv()
    assert VLLM_SERVER_PATH.is_file()


def test_compiler_accepts_gliner_through_external_vllm_factory() -> None:
    """GLiNER compiles through the pinned factory plugin and Python vLLM runtime."""
    models, compiler = load_compiler_modules()
    intent = models.InferenceIntent(
        task=models.EntityDetection(dynamic_labels=True, offsets=True, scores=True),
        model=models.HuggingFaceModel(model_id="nvidia/gliner-pii", revision="bd23e8ef4425fd04"),
        engine=models.VllmEngine(factory=models.VllmFactoryIntegration(plugin="deberta_gliner")),
        placement=models.LocalProcessPlacement(),
        access=models.DirectAccess(),
        lifecycle=models.ManagedLifecycle(),
    )

    plan = compiler.compile_intent(intent, source_revision="3f68c145")

    assert plan.declared_capabilities == ("dynamic-labels", "offsets", "scores")
    assert "vllm==0.27.1" in plan.dependencies
    assert (
        "vllm-factory[gliner] @ git+https://github.com/latenceainew/vllm-factory.git@"
        "7d6ff68ce68f9f7c0a9d72f9645bcf6d335d02f0"
    ) in plan.dependencies
    assert "--vllm-factory-plugin" in plan.command.render_argv()
    assert "deberta_gliner" in plan.command.render_argv()


def test_compiler_rejects_gliner_through_stock_vllm() -> None:
    """Entity detection requires an explicit vLLM Factory plugin."""
    models, compiler = load_compiler_modules()
    intent = models.InferenceIntent(
        task=models.EntityDetection(dynamic_labels=True, offsets=True, scores=True),
        model=models.HuggingFaceModel(model_id="nvidia/gliner-pii"),
        engine=models.VllmEngine(),
        placement=models.LocalProcessPlacement(),
        access=models.DirectAccess(),
        lifecycle=models.ManagedLifecycle(),
    )

    with pytest.raises(compiler.CompilationError) as exc_info:
        compiler.compile_intent(intent, source_revision="3f68c145")

    assert exc_info.value.diagnostic.code == "unsupported-task-engine"


def test_compiler_rejects_unpinned_or_uncharacterized_factory_models() -> None:
    """Factory plans close both checkpoint provenance and plugin compatibility."""
    models, compiler = load_compiler_modules()

    def compile_model(model_id: str, revision: str | None):
        return compiler.compile_intent(
            models.InferenceIntent(
                task=models.EntityDetection(dynamic_labels=True, offsets=True, scores=True),
                model=models.HuggingFaceModel(model_id=model_id, revision=revision),
                engine=models.VllmEngine(factory=models.VllmFactoryIntegration(plugin="deberta_gliner")),
                placement=models.LocalProcessPlacement(),
                access=models.DirectAccess(),
                lifecycle=models.ManagedLifecycle(),
            ),
            source_revision="3f68c145",
        )

    with pytest.raises(compiler.CompilationError) as unpinned:
        compile_model("nvidia/gliner-pii", None)
    assert unpinned.value.diagnostic.code == "unpinned-model-revision"

    with pytest.raises(compiler.CompilationError) as unsupported:
        compile_model("urchade/gliner_small-v2.1", "abcdef0123456789")
    assert unsupported.value.diagnostic.code == "unsupported-model-engine"


def test_compiler_rejects_unsupported_native_generation() -> None:
    """Compatibility failures are typed compiler diagnostics, not runtime surprises."""
    models, compiler = load_compiler_modules()
    intent = models.InferenceIntent(
        task=models.Generation(chat=True),
        model=models.HuggingFaceModel(model_id="openai/gpt-oss-20b"),
        engine=models.NativeGlinerEngine(),
        placement=models.LocalProcessPlacement(),
        access=models.DirectAccess(),
        lifecycle=models.ManagedLifecycle(),
    )

    with pytest.raises(compiler.CompilationError) as exc_info:
        compiler.compile_intent(intent, source_revision="3f68c145")

    assert exc_info.value.diagnostic.code == "unsupported-task-engine"
    assert exc_info.value.diagnostic.details == {
        "engine": "native-gliner",
        "task": "generation",
    }


def test_transport_rejects_unknown_fields_and_attach_variants() -> None:
    """The initial union is closed and has no attach or external-endpoint path."""
    models, _compiler = load_compiler_modules()
    payload = {
        "schema_version": "inference-service.intent/v1",
        "task": {"kind": "generation", "chat": True},
        "model": {"kind": "hugging-face", "model_id": "openai/gpt-oss-20b"},
        "engine": {"kind": "vllm"},
        "placement": {"kind": "attach", "url": "http://example.invalid/v1"},
        "access": {"kind": "direct"},
        "lifecycle": {"kind": "managed"},
        "unexpected": True,
    }

    with pytest.raises(Exception):
        models.InferenceIntent.model_validate(payload)

    with pytest.raises(Exception):
        models.Generation(chat=False)


def test_plan_digest_detects_transport_mutation() -> None:
    """Loading a modified plan fails before any runtime effect can occur."""
    models, compiler = load_compiler_modules()
    intent = models.InferenceIntent(
        task=models.Generation(chat=True),
        model=models.HuggingFaceModel(model_id="openai/gpt-oss-20b"),
        engine=models.VllmEngine(),
        placement=models.LocalProcessPlacement(),
        access=models.DirectAccess(),
        lifecycle=models.ManagedLifecycle(),
    )
    plan = compiler.compile_intent(intent, source_revision="3f68c145")
    payload = json.loads(plan.model_dump_json())
    payload["endpoint"]["port"] = 9000

    with pytest.raises(compiler.PlanIntegrityError, match="plan digest mismatch"):
        compiler.load_plan(json.dumps(payload))


def test_compile_command_accepts_toml_profile_and_writes_json_plan(tmp_path: Path) -> None:
    """Operators author TOML while generated plans retain the JSON transport."""
    cli = load_cli_module()
    profile_path = tmp_path / "generation.toml"
    plan_path = tmp_path / "plan.json"
    profile_path.write_text(
        """\
schema_version = "inference-service.intent/v1"

[task]
kind = "generation"
chat = true

[model]
kind = "hugging-face"
model_id = "openai/gpt-oss-20b"

[engine]
kind = "vllm"

[placement]
kind = "local-process"
host = "127.0.0.1"
port = 8000

[access]
kind = "direct"

[lifecycle]
kind = "managed"
""",
        encoding="utf-8",
    )

    with mock.patch.object(sys, "argv", ["inference-service"]):
        with pytest.raises(SystemExit) as exc_info:
            cli.app(
                [
                    "compile",
                    "--profile",
                    str(profile_path),
                    "--source-revision",
                    "3f68c145",
                    "--output",
                    str(plan_path),
                ]
            )

    assert exc_info.value.code == 0
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "inference-service.run-plan/v1"
    assert payload["source_revision"] == "3f68c145"
    assert payload["plan_digest"]


def test_equivalent_toml_profiles_compile_to_the_same_plan(tmp_path: Path) -> None:
    """TOML comments and table order do not change semantic plan identity."""
    cli = load_cli_module()
    _models, compiler = load_compiler_modules()
    first_path = tmp_path / "first.toml"
    second_path = tmp_path / "second.toml"
    first_path.write_text(
        """\
schema_version = "inference-service.intent/v1"
[task]
kind = "generation"
chat = true
[model]
kind = "hugging-face"
model_id = "openai/gpt-oss-20b"
[engine]
kind = "vllm"
[placement]
kind = "local-process"
host = "127.0.0.1"
port = 8000
[access]
kind = "direct"
[lifecycle]
kind = "managed"
""",
        encoding="utf-8",
    )
    second_path.write_text(
        """\
# The order and formatting are for humans; the compiler sees one typed value.
schema_version = "inference-service.intent/v1"
[lifecycle]
kind = "managed"
[access]
kind = "direct"
[placement]
port = 8000
host = "127.0.0.1"
kind = "local-process"
[engine]
kind = "vllm"
[model]
model_id = "openai/gpt-oss-20b"
kind = "hugging-face"
[task]
chat = true
kind = "generation"
""",
        encoding="utf-8",
    )

    first = compiler.compile_intent(cli.load_profile(first_path), source_revision="3f68c145")
    second = compiler.compile_intent(cli.load_profile(second_path), source_revision="3f68c145")

    assert first == second


def test_toml_profile_rejects_unknown_engine_fields(tmp_path: Path) -> None:
    """Closed profile tables reject unknown settings before compilation."""
    cli = load_cli_module()
    profile_path = tmp_path / "invalid.toml"
    profile_path.write_text(
        """\
schema_version = "inference-service.intent/v1"
[task]
kind = "generation"
chat = true
[model]
kind = "hugging-face"
model_id = "openai/gpt-oss-20b"
[engine]
kind = "vllm"
unknown = true
[placement]
kind = "local-process"
[access]
kind = "direct"
[lifecycle]
kind = "managed"
""",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        cli.load_profile(profile_path)


def test_reference_toml_profiles_are_pinned_and_compile() -> None:
    """Bundled operator profiles stay parseable, pinned, and compatible."""
    cli = load_cli_module()
    _models, compiler = load_compiler_modules()

    profile_paths = tuple(sorted(PROFILE_ROOT.glob("*.toml")))

    assert {path.name for path in profile_paths} == {
        "gliner2.toml",
        "gpt-oss-120b.toml",
        "gpt-oss-20b.toml",
        "nemotron-3.5-lightning.toml",
        "nvidia-gliner.toml",
        "qwen3-30b-a3b-instruct.toml",
        "vllm-local.toml",
    }
    for path in profile_paths:
        intent = cli.load_profile(path)
        assert intent.model.revision is not None
        plan = compiler.compile_intent(intent, source_revision="3f68c145")
        assert plan
        if path.name == "nemotron-3.5-lightning.toml":
            assert plan.dependencies == (
                "vllm==0.27.1",
                "nvidia-cuda-nvcc==13.0.88",
                "nvidia-cuda-crt==13.0.88",
                "nvidia-nvvm==13.0.88",
            )


def test_probe_records_generation_capabilities() -> None:
    """A live probe records models and observed task capabilities in a v1 receipt."""
    models, compiler = load_compiler_modules()
    runtime = load_runtime_module()
    assert callable(getattr(runtime, "probe_endpoint", None))
    plan = build_generation_plan(models, compiler)

    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "openai/gpt-oss-20b"}]})
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(200, json={"choices": [{"message": {"content": "ready"}}]})
        return httpx.Response(404)

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        receipt = runtime.probe_endpoint(plan, client=client)

    assert receipt.schema_version == "inference-service.capability-probe-receipt/v1"
    assert receipt.plan_digest == plan.plan_digest
    assert receipt.models == ("openai/gpt-oss-20b",)
    assert receipt.observed_capabilities == ("chat-completions",)
    assert receipt.passed is True


def test_probe_supports_reasoning_models() -> None:
    """The capability probe obtains content from models that reason before answering."""
    models, compiler = load_compiler_modules()
    runtime = load_runtime_module()
    plan = build_generation_plan(models, compiler)

    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "openai/gpt-oss-20b"}]})
        if request.url.path == "/v1/chat/completions":
            payload = json.loads(request.content)
            enough_output = payload.get("max_tokens", 0) > 8
            low_reasoning = payload.get("chat_template_kwargs", {}).get("reasoning_effort") == "low"
            content = "ready" if enough_output and low_reasoning else None
            return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})
        return httpx.Response(404)

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        receipt = runtime.probe_endpoint(plan, client=client)

    assert receipt.passed is True


def test_probe_uses_the_resolved_bearer_secret() -> None:
    """Secured readiness and task probes authenticate without serializing the value."""
    models, compiler = load_compiler_modules()
    runtime = load_runtime_module()
    plan = build_generation_plan(models, compiler, api_key_env="LOCAL_VLLM_API_KEY")

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer test-secret"
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "openai/gpt-oss-20b"}]})
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(200, json={"choices": [{"message": {"content": "ready"}}]})
        return httpx.Response(404)

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        receipt = runtime.probe_endpoint(
            plan,
            client=client,
            secret_values={"LOCAL_VLLM_API_KEY": "test-secret"},
        )

    assert receipt.passed is True
    assert "test-secret" not in receipt.model_dump_json()


def test_probe_rejects_the_wrong_served_model() -> None:
    """Capabilities from another model do not satisfy the compiled contract."""
    models, compiler = load_compiler_modules()
    runtime = load_runtime_module()
    plan = build_generation_plan(models, compiler)

    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "other/model"}]})
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(200, json={"choices": [{"message": {"content": "ready"}}]})
        return httpx.Response(404)

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        receipt = runtime.probe_endpoint(plan, client=client)

    assert receipt.models == ("other/model",)
    assert receipt.observed_capabilities == ("chat-completions",)
    assert receipt.passed is False


def test_probe_enforces_the_declared_readiness_status() -> None:
    """A parseable response with the wrong status does not satisfy readiness."""
    models, compiler = load_compiler_modules()
    runtime = load_runtime_module()
    plan = build_generation_plan(models, compiler)

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(201, json={"data": [{"id": "openai/gpt-oss-20b"}]})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        with pytest.raises(runtime.RuntimeEffectError, match="readiness probe returned status 201, expected 200"):
            runtime.probe_endpoint(plan, client=client)


def test_launch_local_process_returns_reconnectable_handle(tmp_path: Path) -> None:
    """Launching a plan records external process identity and readiness evidence."""
    models, compiler = load_compiler_modules()
    runtime = load_runtime_module()
    assert callable(getattr(runtime, "launch_plan", None))
    plan = build_generation_plan(models, compiler, api_key_env="LOCAL_VLLM_API_KEY")
    probe = models.CapabilityProbeReceipt(
        plan_digest=plan.plan_digest,
        endpoint=plan.endpoint,
        observed_at="2026-08-07T00:00:00+00:00",
        models=("openai/gpt-oss-20b",),
        observed_capabilities=("chat-completions",),
        passed=True,
    )
    process = mock.Mock(pid=4242)

    with (
        mock.patch.object(runtime.subprocess, "Popen", return_value=process) as popen,
        mock.patch.object(runtime, "probe_endpoint", return_value=probe),
        mock.patch.object(runtime, "read_process_start_marker", return_value="100"),
        mock.patch.object(runtime, "is_handle_running", return_value=True),
    ):
        receipt = runtime.launch_plan(
            plan,
            secret_values={"LOCAL_VLLM_API_KEY": "test-secret"},
            log_directory=tmp_path,
        )

    launched_argv = tuple(popen.call_args.args[0])
    assert launched_argv == plan.command.render_argv()
    assert "test-secret" not in launched_argv
    assert popen.call_args.kwargs["env"]["VLLM_API_KEY"] == "test-secret"
    assert receipt.schema_version == "inference-service.launch-receipt/v1"
    assert receipt.plan_digest == plan.plan_digest
    assert receipt.handle.kind == "local-process"
    assert receipt.handle.external_id == "4242:100"
    assert receipt.handle.pid == 4242
    assert receipt.probe == probe
    assert Path(receipt.handle.stdout_path).parent == tmp_path


def test_launch_rejects_an_unresolved_secret_before_effects(tmp_path: Path) -> None:
    """Missing secret references fail before a process or container is started."""
    models, compiler = load_compiler_modules()
    runtime = load_runtime_module()
    plan = build_generation_plan(models, compiler, api_key_env="LOCAL_VLLM_API_KEY")

    with mock.patch.object(runtime.subprocess, "Popen") as popen:
        with pytest.raises(runtime.RuntimeEffectError) as exc_info:
            runtime.launch_plan(plan, secret_values={}, log_directory=tmp_path)

    popen.assert_not_called()
    assert exc_info.value.diagnostic.code == "missing-secret"
    assert exc_info.value.diagnostic.known_effects == ()


def test_launch_rejects_an_in_memory_mutated_plan_before_effects(tmp_path: Path) -> None:
    """The Python runtime boundary verifies plans as strictly as the JSON CLI."""
    models, compiler = load_compiler_modules()
    runtime = load_runtime_module()
    plan = build_generation_plan(models, compiler)
    changed = plan.model_copy(update={"expected_model": "other/model"})

    with mock.patch.object(runtime.subprocess, "Popen") as popen:
        with pytest.raises(compiler.PlanIntegrityError, match="plan digest mismatch"):
            runtime.launch_plan(changed, secret_values={}, log_directory=tmp_path)

    popen.assert_not_called()


def test_launch_docker_returns_container_identity(tmp_path: Path) -> None:
    """Docker launch captures the stable container ID instead of the client PID."""
    models, compiler = load_compiler_modules()
    runtime = load_runtime_module()
    plan = build_generation_plan(models, compiler, docker=True)
    probe = models.CapabilityProbeReceipt(
        plan_digest=plan.plan_digest,
        endpoint=plan.endpoint,
        observed_at="2026-08-07T00:00:00+00:00",
        models=("openai/gpt-oss-20b",),
        observed_capabilities=("chat-completions",),
        passed=True,
    )
    completed = mock.Mock(returncode=0, stdout="abc123\n", stderr="")

    with (
        mock.patch.object(runtime.subprocess, "run", return_value=completed) as run,
        mock.patch.object(runtime, "probe_endpoint", return_value=probe),
        mock.patch.object(runtime, "is_handle_running", return_value=True),
    ):
        receipt = runtime.launch_plan(plan, secret_values={}, log_directory=tmp_path)

    assert tuple(run.call_args.args[0]) == plan.command.render_argv()
    assert receipt.handle.kind == "docker"
    assert receipt.handle.external_id == "abc123"
    assert receipt.handle.container_id == "abc123"


def test_inspect_and_cancel_local_process_emit_versioned_receipts(tmp_path: Path) -> None:
    """A later CLI invocation can inspect and cancel the recorded process identity."""
    models, compiler = load_compiler_modules()
    runtime = load_runtime_module()
    plan = build_generation_plan(models, compiler)
    probe = models.CapabilityProbeReceipt(
        plan_digest=plan.plan_digest,
        endpoint=plan.endpoint,
        observed_at="2026-08-07T00:00:00+00:00",
        models=("openai/gpt-oss-20b",),
        observed_capabilities=("chat-completions",),
        passed=True,
    )
    handle = models.LocalProcessHandle(
        external_id="4242:100",
        pid=4242,
        process_group_id=4242,
        start_marker="100",
        stdout_path=str(tmp_path / "stdout.log"),
        stderr_path=str(tmp_path / "stderr.log"),
    )
    launch = models.LaunchReceipt(
        plan_digest=plan.plan_digest,
        launched_at="2026-08-07T00:00:00+00:00",
        shutdown_timeout_seconds=10,
        handle=handle,
        probe=probe,
    )

    with mock.patch.object(runtime, "is_handle_running", return_value=True):
        status = runtime.inspect_run(launch)
    with (
        mock.patch.object(runtime, "is_handle_running", side_effect=[True, False]),
        mock.patch.object(runtime.os, "killpg") as killpg,
    ):
        cancellation = runtime.cancel_run(launch)

    assert status.schema_version == "inference-service.status-receipt/v1"
    assert status.state == "running"
    assert cancellation.schema_version == "inference-service.cancellation-receipt/v1"
    assert cancellation.outcome == "terminated"
    assert cancellation.cleanup_complete is True
    killpg.assert_called_once_with(4242, runtime.signal.SIGTERM)


def test_process_identity_handles_spaces_and_zombies(tmp_path: Path) -> None:
    """Linux process identity parsing handles spaced names and treats zombies as stopped."""
    models, _compiler = load_compiler_modules()
    runtime = load_runtime_module()
    fields = ["S", *(str(index) for index in range(4, 22)), "98765"]
    payload = f"4242 (worker with spaces) {' '.join(fields)}"
    assert runtime._parse_process_stat(payload) == ("S", "98765")
    handle = models.LocalProcessHandle(
        external_id="4242:98765",
        pid=4242,
        process_group_id=4242,
        start_marker="98765",
        stdout_path=str(tmp_path / "stdout.log"),
        stderr_path=str(tmp_path / "stderr.log"),
    )
    with (
        mock.patch.object(runtime, "read_process_start_marker", return_value="98765"),
        mock.patch.object(runtime, "read_process_state", return_value="Z"),
        mock.patch.object(runtime.os, "kill") as kill,
    ):
        assert runtime.is_handle_running(handle) is False
    kill.assert_not_called()


def test_vllm_plan_preserves_lora_model_artifact() -> None:
    """LoRA remains a model artifact while vLLM owns its launch spelling."""
    models, compiler = load_compiler_modules()
    intent = models.InferenceIntent(
        task=models.Generation(chat=True),
        model=models.HuggingFaceModel(
            model_id="openai/gpt-oss-20b",
            adapter=models.LoraAdapter(path="/models/privacy-adapter", name="privacy"),
        ),
        engine=models.VllmEngine(),
        placement=models.LocalProcessPlacement(),
        access=models.DirectAccess(),
        lifecycle=models.ManagedLifecycle(),
    )

    plan = compiler.compile_intent(intent, source_revision="3f68c145")

    assert plan.command.render_argv()[-3:] == (
        "--enable-lora",
        "--lora-modules",
        "privacy=/models/privacy-adapter",
    )


def test_discover_cached_models_returns_versioned_source_paths(tmp_path: Path) -> None:
    """Cached-model discovery survives as typed source-tree output without downloads."""
    models, _compiler = load_compiler_modules()
    runtime = load_runtime_module()
    assert callable(getattr(runtime, "discover_cached_models", None))
    snapshot = tmp_path / "models--openai--gpt-oss-20b" / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)

    result = runtime.discover_cached_models(tmp_path)

    assert result.schema_version == "inference-service.cached-models/v1"
    assert result.cache_root == str(tmp_path)
    assert result.models == (
        models.CachedModel(repository="openai/gpt-oss-20b", revision="abc123", snapshot_path=str(snapshot)),
    )


def test_models_command_writes_versioned_cache_discovery(tmp_path: Path) -> None:
    """The CLI retains PR 212's cached-model discovery as typed JSON."""
    cli = load_cli_module()
    cache_root = tmp_path / "hub"
    snapshot = cache_root / "models--openai--gpt-oss-20b" / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    output = tmp_path / "models.json"

    with pytest.raises(SystemExit) as exc_info:
        cli.app(["models", "--cache-root", str(cache_root), "--output", str(output)])

    assert exc_info.value.code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "inference-service.cached-models/v1"
    assert payload["models"][0]["repository"] == "openai/gpt-oss-20b"


def test_native_gliner_cutover_has_no_legacy_entrypoint() -> None:
    """The characterized server moves into the compiler tree without a wrapper."""
    assert NATIVE_GLINER_PATH.is_file()
    assert NATIVE_GLINER_PATH.stat().st_mode & stat.S_IXUSR
    assert not REMOVED_GLINER_PATH.exists()


def test_native_gliner_plan_preserves_batch_environment() -> None:
    """Compiler plans retain the characterized request-coalescing controls."""
    models, compiler = load_compiler_modules()
    intent = models.InferenceIntent(
        task=models.EntityDetection(dynamic_labels=True, offsets=True, scores=True),
        model=models.HuggingFaceModel(model_id="nvidia/gliner-pii"),
        engine=models.NativeGlinerEngine(
            device="cuda",
            batch_mode=True,
            max_batch_requests=64,
            batch_wait_ms=10,
        ),
        placement=models.LocalProcessPlacement(port=9000),
        access=models.DirectAccess(),
        lifecycle=models.ManagedLifecycle(),
    )

    plan = compiler.compile_intent(intent, source_revision="3f68c145")

    assert {item.name: item.value for item in plan.command.environment} == {
        "DEVICE": "cuda",
        "GLINER_BATCH_MODE": "true",
        "GLINER_MAX_BATCH_REQUESTS": "64",
        "GLINER_BATCH_WAIT_MS": "10.0",
    }


def test_failed_readiness_cleans_up_the_launched_process(tmp_path: Path) -> None:
    """A failed readiness probe reports and cleans every known launch effect."""
    models, compiler = load_compiler_modules()
    runtime = load_runtime_module()
    plan = build_generation_plan(models, compiler)
    process = mock.Mock(pid=4242)
    failure = runtime.RuntimeEffectError(models.RuntimeDiagnostic(code="probe-failed", message="not ready"))

    with (
        mock.patch.object(runtime.subprocess, "Popen", return_value=process),
        mock.patch.object(runtime, "read_process_start_marker", return_value="100"),
        mock.patch.object(runtime, "wait_for_readiness", side_effect=failure),
        mock.patch.object(runtime, "is_handle_running", side_effect=[True, False]),
        mock.patch.object(runtime.os, "killpg") as killpg,
    ):
        with pytest.raises(runtime.RuntimeEffectError) as exc_info:
            runtime.launch_plan(plan, secret_values={}, log_directory=tmp_path)

    assert exc_info.value.diagnostic.known_effects == ("4242:100",)
    assert exc_info.value.diagnostic.cleanup_complete is True
    killpg.assert_called_once_with(4242, runtime.signal.SIGTERM)


def test_readiness_stops_polling_when_the_managed_process_exits(tmp_path: Path) -> None:
    """A crashed server fails immediately and points the operator to its stderr log."""
    models, compiler = load_compiler_modules()
    runtime = load_runtime_module()
    plan = build_generation_plan(models, compiler)
    handle = models.LocalProcessHandle(
        external_id="4242:100",
        pid=4242,
        process_group_id=4242,
        start_marker="100",
        stdout_path=str(tmp_path / "stdout.log"),
        stderr_path=str(tmp_path / "stderr.log"),
    )

    with (
        mock.patch.object(runtime, "is_handle_running", return_value=False),
        mock.patch.object(runtime, "probe_endpoint") as probe,
        pytest.raises(runtime.RuntimeEffectError) as exc_info,
    ):
        runtime.wait_for_readiness(plan, handle=handle)

    assert exc_info.value.diagnostic.code == "launch-exited"
    assert str(tmp_path / "stderr.log") in exc_info.value.diagnostic.message
    probe.assert_not_called()
