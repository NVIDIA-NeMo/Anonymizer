# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Observable contracts for the managed local inference service."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import socket
import stat
import subprocess
import sys
import time
import venv
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import httpx
import pytest
from pydantic import ValidationError

from inference_service_compiler import backends, cli, compiler, launch_ownership, models, runtime
from inference_service_compiler.profiles import load_profile

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
PROFILES = TOOLS / "inference_service_profiles"
CLI = TOOLS / "inference_service.py"
FIXTURES = Path(__file__).parent / "fixtures"
TOKEN_SHA256 = "0" * 64


def generation(*, backend: dict[str, object] | None = None, **fields: object) -> models.ServiceSpec:
    return models.ServiceSpec.model_validate(
        {
            "backend": {"kind": "vllm", "model": {"model_id": "openai/gpt-oss-20b", "revision": "abc"}}
            | (backend or {}),
            "port": 8000,
            "startup_timeout_seconds": 3,
            "shutdown_timeout_seconds": 0.01,
        }
        | fields
    )


def detection(**fields: object) -> models.ServiceSpec:
    return models.ServiceSpec.model_validate({"backend": {"kind": "gliner2"}, "port": 8001} | fields)


def launch_receipt(plan: models.RunPlan, handle: models.LocalProcessHandle) -> models.LaunchReceipt:
    return models.LaunchReceipt(plan=plan, launched_at=datetime(2026, 8, 7, tzinfo=UTC), handle=handle)


def chat_server(model: str, seen: list[httpx.Request]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": model}]})
        return httpx.Response(200, json={"choices": [{"message": {"content": "ready"}}]})

    return httpx.MockTransport(handler)


def detector_server(entities: list[dict[str, object]]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": models.GLINER2_MODEL_ID}]})
        content = json.dumps({"entities": entities})
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    return httpx.MockTransport(handler)


def test_cli_remains_a_directly_executable_source_entrypoint() -> None:
    assert CLI.is_file()
    assert CLI.stat().st_mode & stat.S_IXUSR


@pytest.mark.parametrize("path", sorted(PROFILES.glob("*.toml")), ids=lambda path: path.name)
def test_shipped_profiles_compile_and_reload(path: Path) -> None:
    plan = compiler.compile_spec(load_profile(path))
    assert compiler.load_plan(plan.model_dump_json()) == plan
    assert plan.served_model_name


def test_vllm_plan_hands_the_child_its_typed_launch_spec() -> None:
    spec = generation(backend={"served_model_name": "local-generator", "options": {"max-model-len": 8192}})
    plan = compiler.compile_spec(spec)
    python, entry, flag, payload = plan.argv
    assert (python, entry, flag) == (backends.VLLM_PYTHON, backends.VLLM_ENTRY, "--launch-spec-json")
    assert models.ServiceSpec.model_validate_json(payload) == spec
    assert plan.base_url == "http://127.0.0.1:8000/v1"
    assert plan.served_model_name == "local-generator"


def test_gliner2_plan_runs_the_native_server_through_its_isolated_environment() -> None:
    plan = compiler.compile_spec(detection(host="127.0.0.2", port=9001))
    assert plan.argv == (
        ".venv/bin/python",
        "-m",
        "anonymizer.engine.local_inference.serve",
        "--host",
        "127.0.0.2",
        "--port",
        "9001",
    )
    assert plan.served_model_name == "fastino/gliner2-privacy-filter-PII-multi"
    assert backends.traits(detection(backend={"kind": "gliner2", "device": "cpu"})).child_environment == {
        "ANONYMIZER_LOCAL_GLINER2_DEVICE": "cpu"
    }


def test_served_model_name_prefers_the_lora_adapter() -> None:
    spec = generation(backend={"model": {"model_id": "base", "adapter": {"path": "/a", "name": "privacy"}}})
    assert compiler.compile_spec(spec).served_model_name == "privacy"


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"port": 9000}, "derived from the profile"),
        ({"middleware": "user.middleware"}, "derived from the profile"),
        ({"served-model-name": "alias"}, "derived from the profile"),
        ({"no-enable-lora": True}, "to false instead"),
        ({"max_model_len": 1}, "kebab-case"),
        ({"max-model-len=1": 1}, "kebab-case"),
        ({"hf-overrides.x": 1}, "kebab-case"),
        ({"allowed-origins": []}, "empty list"),
    ],
)
def test_vllm_options_reject_tool_owned_and_ambiguous_keys(options: dict[str, object], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        generation(backend={"options": options})


def test_spec_rejects_privileged_ports_and_unknown_backends() -> None:
    with pytest.raises(ValidationError):
        generation(port=80)
    with pytest.raises(ValidationError):
        models.ServiceSpec.model_validate({"backend": {"kind": "factory"}, "port": 8000})
    with pytest.raises(ValidationError):
        detection(backend={"kind": "gliner2", "device": "tpu"})


def test_compile_command_writes_a_verified_plan(tmp_path: Path) -> None:
    output = tmp_path / "plan.json"
    with pytest.raises(SystemExit) as exc_info:
        cli.app(["compile", "--profile", str(PROFILES / "vllm-local.toml"), "--output", str(output)])
    assert exc_info.value.code == 0
    assert compiler.load_plan(output.read_text(encoding="utf-8")).served_model_name == "anonymizer-local"


def test_connection_command_emits_only_non_secret_client_inputs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    plan = compiler.compile_spec(generation(api_key_env="LOCAL_KEY"))
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(plan.model_dump_json(), encoding="utf-8")

    cli.connection(plan=plan_path)

    assert json.loads(capsys.readouterr().out) == {
        "api_key_env": "LOCAL_KEY",
        "model": plan.served_model_name,
        "url": plan.base_url,
    }


def test_compile_command_translates_non_directory_profile_paths(tmp_path: Path) -> None:
    """Filesystem path-shape errors retain the documented bad-input exit."""
    not_a_directory = tmp_path / "profile-file"
    not_a_directory.write_text("not a directory", encoding="utf-8")
    with pytest.raises(SystemExit) as exc_info:
        cli.app(["compile", "--profile", str(not_a_directory / "profile.toml")])
    assert exc_info.value.code == 125


def test_plan_digest_detects_transport_mutation() -> None:
    changed = json.loads(compiler.compile_spec(generation()).model_dump_json())
    changed["spec"]["port"] = 9000
    with pytest.raises(compiler.PlanIntegrityError, match="plan digest mismatch"):
        compiler.load_plan(json.dumps(changed))


def test_plan_must_match_the_compilation_of_its_spec() -> None:
    """A re-digested plan whose argv no longer follows from its spec is still rejected."""
    plan = compiler.compile_spec(generation())
    forged = plan.model_copy(update={"argv": ("/usr/bin/python3", "-c", "pass")})
    forged = forged.model_copy(update={"plan_digest": compiler.digest_plan(forged)})
    with pytest.raises(compiler.PlanIntegrityError, match="recompile"):
        compiler.load_plan(forged.model_dump_json())


def test_vllm_probe_sends_only_the_profile_requested_extras() -> None:
    seen: list[httpx.Request] = []
    plain = compiler.compile_spec(generation())
    runtime.probe_endpoint(plain, transport=chat_server(plain.served_model_name, seen))
    payload = json.loads(seen[-1].content)
    assert payload["model"] == plain.served_model_name
    assert "reasoning_effort" not in payload
    assert "chat_template_kwargs" not in payload

    reasoning = compiler.compile_spec(generation(backend={"probe_extra_body": {"reasoning_effort": "low"}}))
    runtime.probe_endpoint(reasoning, transport=chat_server(reasoning.served_model_name, seen))
    assert json.loads(seen[-1].content)["reasoning_effort"] == "low"


def test_probe_uses_bearer_secret_without_serializing_it() -> None:
    plan = compiler.compile_spec(generation(api_key_env="LOCAL_KEY"))
    seen: list[httpx.Request] = []
    runtime.probe_endpoint(
        plan, secret_values={"LOCAL_KEY": "test-secret"}, transport=chat_server(plan.served_model_name, seen)
    )
    assert {request.headers["authorization"] for request in seen} == {"Bearer test-secret"}


def test_probe_rejects_wrong_model_and_status() -> None:
    plan = compiler.compile_spec(generation())
    with pytest.raises(runtime.RuntimeEffectError, match="not 'openai/gpt-oss-20b'"):
        runtime.probe_endpoint(plan, transport=chat_server("other/model", []))
    with pytest.raises(runtime.RuntimeEffectError, match="status 201"):
        runtime.probe_endpoint(plan, transport=httpx.MockTransport(lambda _request: httpx.Response(201, json={})))


def test_gliner2_probe_accepts_labelled_offsets_and_scores() -> None:
    plan = compiler.compile_spec(detection())
    entity: dict[str, object] = {"label": "person", "start": 0, "end": 12, "score": 0.98}
    runtime.probe_endpoint(plan, transport=detector_server([entity]))


@pytest.mark.parametrize(
    ("entities", "reason"),
    [
        ([], "no entities"),
        ([{"label": "location", "start": 0, "end": 12, "score": 0.9}], "ignored the requested labels"),
        ([{"label": "person", "score": 0.9}], "invalid offsets"),
        ([{"label": "person", "start": 0, "end": 500, "score": 0.9}], "invalid offsets"),
        ([{"label": "person", "start": 0, "end": 12}], "numeric score"),
    ],
)
def test_gliner2_probe_reports_why_detection_is_not_ready(entities: list[dict[str, object]], reason: str) -> None:
    plan = compiler.compile_spec(detection())
    with pytest.raises(runtime.RuntimeEffectError, match=reason):
        runtime.probe_endpoint(plan, transport=detector_server(entities))


def test_gliner2_launch_delivers_token_and_device_to_the_native_server(tmp_path: Path) -> None:
    plan = compiler.compile_spec(detection(api_key_env="DETECTOR_KEY", backend={"kind": "gliner2", "device": "cpu"}))
    process = mock.Mock(pid=4242)
    with (
        mock.patch.object(runtime.subprocess, "Popen", return_value=process) as popen,
        mock.patch.object(runtime, "wait_for_readiness"),
        mock.patch.object(runtime, "read_process_start_marker", return_value="100"),
    ):
        runtime.launch_plan(plan, secret_values={"DETECTOR_KEY": "detector-secret"}, log_directory=tmp_path)
    environment = popen.call_args.kwargs["env"]
    assert environment["ANONYMIZER_LOCAL_GLINER2_TOKEN"] == "detector-secret"
    assert environment["ANONYMIZER_LOCAL_GLINER2_DEVICE"] == "cpu"
    assert "VLLM_API_KEY" not in environment


def test_gliner2_validation_prepares_the_isolated_environment_in_its_interpreter(tmp_path: Path) -> None:
    assert backends.traits(detection()).validate_argv == (
        ".venv/bin/python",
        "-m",
        "anonymizer.engine.local_inference.serve",
        "--prepare",
    )
    bare = tmp_path / "bare"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(bare)
    with pytest.raises(runtime.RuntimeEffectError, match="validation failed"):
        runtime.validate_plan(compiler.compile_spec(detection(python_executable=str(bare / "bin/python"))))


def test_plan_integrity_is_enforced_before_launch(tmp_path: Path) -> None:
    plan = compiler.compile_spec(generation())
    with pytest.raises(compiler.PlanIntegrityError):
        runtime.launch_plan(plan.model_copy(update={"argv": ("changed",)}), secret_values={}, log_directory=tmp_path)


def test_status_and_stop_reject_a_reused_pid_without_signaling(monkeypatch: pytest.MonkeyPatch) -> None:
    plan = compiler.compile_spec(generation())
    handle = models.LocalProcessHandle(
        launch_token_sha256=TOKEN_SHA256, pid=1, start_marker="old", stdout_path="out", stderr_path="err"
    )
    launch = launch_receipt(plan, handle)
    monkeypatch.setattr(runtime, "read_process_start_marker", lambda _pid: "new")

    with pytest.raises(runtime.RuntimeEffectError) as status_error:
        runtime.status_run(launch)
    with (
        mock.patch.object(runtime.os, "killpg") as killpg,
        pytest.raises(runtime.RuntimeEffectError) as stop_error,
    ):
        runtime.stop_run(launch)

    assert status_error.value.diagnostic.code == "process-identity-mismatch"
    assert stop_error.value.diagnostic.code == "process-identity-mismatch"
    killpg.assert_not_called()


def test_stop_rejects_a_mismatched_process_group_without_signaling() -> None:
    plan = compiler.compile_spec(generation())
    handle = models.LocalProcessHandle(
        launch_token_sha256=TOKEN_SHA256,
        pid=4242,
        start_marker="100",
        stdout_path="out",
        stderr_path="err",
    )
    launch = launch_receipt(plan, handle)

    with (
        mock.patch.object(runtime, "read_process_start_marker", return_value="100"),
        mock.patch.object(runtime, "read_process_state", return_value="S"),
        mock.patch.object(runtime.os, "getpgid", return_value=9000),
        mock.patch.object(runtime.os, "kill"),
        mock.patch.object(runtime.os, "killpg") as killpg,
        pytest.raises(runtime.RuntimeEffectError) as exc_info,
    ):
        runtime.stop_run(launch)

    assert exc_info.value.diagnostic.code == "process-identity-mismatch"
    killpg.assert_not_called()


def test_launch_records_process_identity_and_resolves_secrets(tmp_path: Path) -> None:
    plan = compiler.compile_spec(generation(api_key_env="LOCAL_KEY"))
    process = mock.Mock(pid=4242)
    with (
        mock.patch.object(runtime.subprocess, "Popen", return_value=process) as popen,
        mock.patch.object(runtime, "wait_for_readiness"),
        mock.patch.object(runtime, "read_process_start_marker", return_value="100"),
    ):
        receipt = runtime.launch_plan(plan, secret_values={"LOCAL_KEY": "test-secret"}, log_directory=tmp_path)
    assert tuple(popen.call_args.args[0]) == plan.argv
    assert popen.call_args.kwargs["env"]["VLLM_API_KEY"] == "test-secret"
    assert "test-secret" not in popen.call_args.args[0]
    assert receipt.handle.label == "4242:100"


def test_launch_requires_scoped_ownership_evidence(tmp_path: Path) -> None:
    plan = compiler.compile_spec(generation())
    handle = models.LocalProcessHandle(
        launch_token_sha256=TOKEN_SHA256,
        pid=4242,
        start_marker="100",
        stdout_path="out",
        stderr_path="err",
    )
    observed_environment: dict[str, str] = {}
    observed_token: str | None = None

    def launch_process(
        _plan: models.RunPlan,
        environment: dict[str, str],
        _log_directory: Path,
        _launch_token_sha256: str,
    ) -> models.LocalProcessHandle:
        observed_environment.update(environment)
        return handle

    def wait_for_readiness(
        _plan: models.RunPlan,
        *,
        secret_values: object,
        handle: models.LocalProcessHandle,
        launch_token: str | None = None,
    ) -> None:
        del secret_values, handle
        nonlocal observed_token
        observed_token = launch_token

    with (
        mock.patch.object(runtime, "_launch_process", side_effect=launch_process),
        mock.patch.object(runtime, "wait_for_readiness", side_effect=wait_for_readiness),
    ):
        receipt = runtime.launch_plan(plan, secret_values={}, log_directory=tmp_path)

    environment_token = observed_environment.get("ANONYMIZER_INFERENCE_LAUNCH_TOKEN")
    assert environment_token
    assert observed_token == environment_token
    assert environment_token not in receipt.model_dump_json()


def test_launch_ownership_endpoint_requires_and_proves_the_scoped_token() -> None:
    async def next_handler(_request: object) -> object:
        return mock.sentinel.next_response

    def request(path: str, token: str | None) -> SimpleNamespace:
        headers = {} if token is None else {launch_ownership.LAUNCH_OWNERSHIP_HEADER: token}
        return SimpleNamespace(url=SimpleNamespace(path=path), headers=headers)

    with mock.patch.dict(
        runtime.os.environ,
        {launch_ownership.LAUNCH_TOKEN_ENVIRONMENT_VARIABLE: "launch-secret"},
        clear=True,
    ):
        wrong = asyncio.run(
            launch_ownership.launch_ownership(request(launch_ownership.LAUNCH_OWNERSHIP_PATH, "wrong"), next_handler)
        )
        owned = asyncio.run(
            launch_ownership.launch_ownership(
                request(launch_ownership.LAUNCH_OWNERSHIP_PATH, "launch-secret"), next_handler
            )
        )
        passed_through = asyncio.run(launch_ownership.launch_ownership(request("/v1/models", None), next_handler))

    assert wrong.status_code == 404
    assert json.loads(owned.body) == {
        launch_ownership.LAUNCH_OWNERSHIP_PROOF_FIELD: launch_ownership.launch_token_proof("launch-secret")
    }
    assert passed_through is mock.sentinel.next_response


def test_compatible_preexisting_endpoint_cannot_prove_launch_ownership() -> None:
    plan = compiler.compile_spec(generation())
    compatible_response = httpx.Response(200, json={"data": [{"id": plan.served_model_name}]})

    with (
        mock.patch.object(runtime.httpx, "get", return_value=compatible_response),
        pytest.raises(runtime.RuntimeEffectError) as exc_info,
    ):
        runtime._probe_launch_ownership(plan, "launch-secret")

    assert exc_info.value.diagnostic.code == "launch-ownership-not-observed"


def test_launch_ownership_probe_receives_endpoint_authentication() -> None:
    plan = compiler.compile_spec(generation(api_key_env="LOCAL_KEY"))
    observed_secrets: object = None

    def ownership_probe(
        _plan: models.RunPlan,
        _launch_token: str,
        secret_values: object = None,
    ) -> None:
        nonlocal observed_secrets
        observed_secrets = secret_values

    with (
        mock.patch.object(runtime, "_probe_launch_ownership", side_effect=ownership_probe),
        mock.patch.object(runtime, "probe_endpoint"),
    ):
        runtime.wait_for_readiness(
            plan,
            secret_values={"LOCAL_KEY": "endpoint-secret"},
            launch_token="launch-secret",
        )

    assert observed_secrets == {"LOCAL_KEY": "endpoint-secret"}


@pytest.mark.parametrize("failure", [RuntimeError("unexpected readiness failure"), KeyboardInterrupt()])
def test_post_spawn_readiness_failures_always_clean_up(tmp_path: Path, failure: BaseException) -> None:
    plan = compiler.compile_spec(generation())
    handle = models.LocalProcessHandle(
        launch_token_sha256=TOKEN_SHA256,
        pid=4242,
        start_marker="100",
        stdout_path="out",
        stderr_path="err",
    )

    with (
        mock.patch.object(runtime, "_launch_process", return_value=handle),
        mock.patch.object(runtime, "wait_for_readiness", side_effect=failure),
        mock.patch.object(runtime, "_cleanup_handle", return_value=True) as cleanup,
        pytest.raises(type(failure)),
    ):
        runtime.launch_plan(plan, secret_values={}, log_directory=tmp_path)

    cleanup.assert_called_once_with(handle, plan.spec.shutdown_timeout_seconds)


def test_launch_command_stops_service_when_receipt_write_fails(tmp_path: Path) -> None:
    plan = compiler.compile_spec(generation())
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(plan.model_dump_json(), encoding="utf-8")
    handle = models.LocalProcessHandle(
        launch_token_sha256=TOKEN_SHA256,
        pid=4242,
        start_marker="100",
        stdout_path="out",
        stderr_path="err",
    )
    receipt = launch_receipt(plan, handle)

    with (
        mock.patch.object(cli, "launch_plan", return_value=receipt),
        mock.patch.object(cli, "write_json", side_effect=PermissionError("read-only destination")),
        mock.patch.object(cli, "stop_run") as stop,
        pytest.raises(SystemExit) as exc_info,
    ):
        cli.launch(plan=plan_path, output=tmp_path / "receipt.json", log_directory=tmp_path)

    assert exc_info.value.code == 125
    stop.assert_called_once_with(receipt)


def test_launch_command_rejects_a_directory_receipt_target_before_launch(tmp_path: Path) -> None:
    plan = compiler.compile_spec(generation())
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(plan.model_dump_json(), encoding="utf-8")
    output = tmp_path / "receipt"
    output.mkdir()
    handle = models.LocalProcessHandle(
        launch_token_sha256=TOKEN_SHA256,
        pid=4242,
        start_marker="100",
        stdout_path="out",
        stderr_path="err",
    )
    receipt = launch_receipt(plan, handle)

    with (
        mock.patch.object(cli, "launch_plan", return_value=receipt) as launch,
        mock.patch.object(cli, "stop_run"),
        pytest.raises(SystemExit) as exc_info,
    ):
        cli.launch(plan=plan_path, output=output, log_directory=tmp_path)

    assert exc_info.value.code == 125
    launch.assert_not_called()


def test_launch_refuses_an_unmarked_process_and_cleans_up(tmp_path: Path) -> None:
    plan = compiler.compile_spec(generation())
    process = mock.Mock(pid=4242)
    process.poll.return_value = 0
    with (
        mock.patch.object(runtime.subprocess, "Popen", return_value=process),
        mock.patch.object(runtime, "read_process_start_marker", return_value=None),
        mock.patch.object(runtime.os, "killpg") as killpg,
        pytest.raises(runtime.RuntimeEffectError) as exc_info,
    ):
        runtime.launch_plan(plan, secret_values={}, log_directory=tmp_path)
    killpg.assert_called_once_with(4242, runtime.signal.SIGTERM)
    process.wait.assert_called_once_with(timeout=1)
    assert exc_info.value.diagnostic.code == "missing-process-start-marker"
    assert exc_info.value.diagnostic.cleanup_complete is True


def test_launch_refuses_an_unmarked_process_when_cleanup_cannot_reap(tmp_path: Path) -> None:
    plan = compiler.compile_spec(generation())
    process = mock.Mock(pid=4242)
    process.poll.return_value = None
    process.wait.side_effect = (
        runtime.subprocess.TimeoutExpired(cmd=plan.argv, timeout=1),
        runtime.subprocess.TimeoutExpired(cmd=plan.argv, timeout=1),
    )
    with (
        mock.patch.object(runtime.subprocess, "Popen", return_value=process),
        mock.patch.object(runtime, "read_process_start_marker", return_value=None),
        mock.patch.object(runtime.os, "killpg") as killpg,
        pytest.raises(runtime.RuntimeEffectError) as exc_info,
    ):
        runtime.launch_plan(plan, secret_values={}, log_directory=tmp_path)
    assert killpg.call_args_list == [mock.call(4242, runtime.signal.SIGTERM), mock.call(4242, runtime.signal.SIGKILL)]
    assert all(call.kwargs["timeout"] > 0 for call in process.wait.call_args_list)
    assert exc_info.value.diagnostic.code == "missing-process-start-marker"
    assert exc_info.value.diagnostic.cleanup_complete is False


def test_missing_secret_fails_before_process_start(tmp_path: Path) -> None:
    plan = compiler.compile_spec(generation(api_key_env="LOCAL_KEY"))
    with mock.patch.object(runtime.subprocess, "Popen") as popen:
        with pytest.raises(runtime.RuntimeEffectError) as exc_info:
            runtime.launch_plan(plan, secret_values={}, log_directory=tmp_path)
    popen.assert_not_called()
    assert exc_info.value.diagnostic.code == "missing-secret"
    assert exc_info.value.diagnostic.known_effects == ()


def test_process_stat_handles_spaces_and_zombies(tmp_path: Path) -> None:
    fields = ["S", *(str(index) for index in range(4, 22)), "98765"]
    assert runtime._parse_process_stat(f"4242 (worker with spaces) {' '.join(fields)}") == ("S", "98765")
    handle = models.LocalProcessHandle(
        launch_token_sha256=TOKEN_SHA256,
        pid=4242,
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


def test_readiness_stops_when_the_managed_process_exits(tmp_path: Path) -> None:
    plan = compiler.compile_spec(generation())
    handle = models.LocalProcessHandle(
        launch_token_sha256=TOKEN_SHA256,
        pid=4242,
        start_marker="100",
        stdout_path=str(tmp_path / "stdout.log"),
        stderr_path=str(tmp_path / "stderr.log"),
    )
    with (
        mock.patch.object(runtime, "is_handle_running", return_value=False),
        mock.patch.object(runtime, "probe_endpoint") as probe,
        pytest.raises(runtime.RuntimeEffectError, match="exited before readiness"),
    ):
        runtime.wait_for_readiness(plan, handle=handle)
    probe.assert_not_called()


@pytest.mark.skipif(sys.platform != "linux", reason="managed process identity uses Linux /proc")
def test_launch_uses_independent_interpreter_and_cleans_python_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second interpreter can prove ownership and be stopped by the controller."""
    serving = tmp_path / "serving"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(serving)
    server = tmp_path / "tools/inference_service_compiler/vllm_server.py"
    server.parent.mkdir(parents=True)
    shutil.copyfile(Path(__file__).parent / "fixtures/inference_process.py", server)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    spec = generation(
        python_executable=str(serving / "bin/python"),
        port=port,
        startup_timeout_seconds=10,
        shutdown_timeout_seconds=0.1,
    )
    plan = compiler.compile_spec(spec)
    monkeypatch.chdir(tmp_path)
    for variable in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV"):
        monkeypatch.setenv(variable, str(tmp_path / "foreign-python"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "models"))
    receipt = runtime.launch_plan(plan, secret_values={}, log_directory=tmp_path / "logs")
    try:
        observed = json.loads((tmp_path / "observed.json").read_text())
        assert observed["prefix"] == str(serving) != sys.prefix
        assert observed["pid"] == receipt.handle.pid == receipt.handle.process_group_id
        assert str(tmp_path / "foreign-python") not in observed["path"]
        assert observed["environment"] == {
            "PYTHONHOME": None,
            "PYTHONPATH": None,
            "VIRTUAL_ENV": None,
            "CUDA_VISIBLE_DEVICES": "0",
            "HF_HOME": str(tmp_path / "models"),
            "VLLM_API_KEY": None,
        }
        assert runtime.is_handle_running(receipt.handle)
    finally:
        runtime.stop_run(receipt)
        os.waitpid(receipt.handle.pid, 0)
    assert not runtime.is_handle_running(receipt.handle)


def test_real_cli_compile_validate_connection_and_tampered_plan(tmp_path: Path) -> None:
    """The source CLI validates through the configured interpreter and rejects modified plans."""
    import subprocess

    server = tmp_path / "tools/inference_service_compiler/vllm_server.py"
    server.parent.mkdir(parents=True)
    shutil.copyfile(Path(__file__).parent / "fixtures/inference_process.py", server)
    profile = tmp_path / "profile.toml"
    profile.write_text(
        f'python_executable = "{sys.executable}"\nport = 8123\n'
        '[backend]\nkind = "vllm"\n[backend.model]\nmodel_id = "model"\n'
        "[backend.options]\nenable-prefix-caching = false\n"
    )
    plan_path = tmp_path / "plan.json"

    def call(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(CLI), *args], cwd=tmp_path, text=True, capture_output=True, check=False
        )

    result = call("compile", "--profile", str(profile), "--output", str(plan_path))
    assert result.returncode == 0, result.stderr
    assert call("validate", "--plan", str(plan_path)).returncode == 0
    result = call("connection", "--plan", str(plan_path))
    assert result.returncode == 0
    assert json.loads(result.stdout)["model"] == "model"
    plan = json.loads(plan_path.read_text())
    plan["spec"]["backend"]["options"]["enable-prefix-caching"] = True
    plan_path.write_text(json.dumps(plan))
    assert call("validate", "--plan", str(plan_path)).returncode == 125


@pytest.mark.skipif(sys.platform != "linux", reason="managed process identity uses Linux /proc")
def test_real_cli_authenticated_lifecycle(tmp_path: Path) -> None:
    """The CLI delivers credentials only to the child and cleans up its owned process."""
    import subprocess

    serving = tmp_path / "serving"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(serving)
    server = tmp_path / "tools/inference_service_compiler/vllm_server.py"
    server.parent.mkdir(parents=True)
    shutil.copyfile(Path(__file__).parent / "fixtures/inference_process.py", server)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    profile = tmp_path / "profile.toml"
    profile.write_text(
        f'python_executable = "{serving / "bin/python"}"\napi_key_env = "LOCAL_KEY"\n'
        f"port = {port}\nstartup_timeout_seconds = 10\nshutdown_timeout_seconds = 0.1\n"
        '[backend]\nkind = "vllm"\n[backend.model]\nmodel_id = "openai/gpt-oss-20b"\n'
    )
    plan, receipt = tmp_path / "plan.json", tmp_path / "receipt.json"
    env = os.environ | {"LOCAL_KEY": "private-secret", "PYTHONPATH": "/foreign", "VIRTUAL_ENV": "/foreign"}

    def call(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(CLI), *args], cwd=tmp_path, text=True, capture_output=True, env=env, check=False
        )

    result = call("compile", "--profile", str(profile), "--output", str(plan))
    assert result.returncode == 0, result.stderr
    assert call("validate", "--plan", str(plan)).returncode == 0
    result = call("launch", "--plan", str(plan), "--output", str(receipt), "--log-directory", str(tmp_path / "logs"))
    assert result.returncode == 0, result.stderr
    try:
        observed = json.loads((tmp_path / "observed.json").read_text())
        assert observed["prefix"] == str(serving)
        assert observed["environment"]["VLLM_API_KEY"] == "private-secret"
        assert observed["environment"]["PYTHONPATH"] is None
        assert "private-secret" not in plan.read_text() + receipt.read_text()
        result = call("status", "--receipt", str(receipt))
        assert json.loads(result.stdout)["state"] == "running"
    finally:
        result = call("stop", "--receipt", str(receipt))
        assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["outcome"] in ("terminated", "forced")


@dataclass
class ProcessGroup:
    """A real process group started from the fixture's ``group`` mode."""

    leader: subprocess.Popen[bytes]
    workers: list[int]
    handle: models.LocalProcessHandle

    @property
    def receipt(self) -> models.LaunchReceipt:
        return launch_receipt(compiler.compile_spec(generation(shutdown_timeout_seconds=0.5)), self.handle)

    def reap_leader(self) -> None:
        self.leader.wait(timeout=10)

    def live_workers(self) -> list[int]:
        return [pid for pid in self.workers if runtime.read_process_state(pid) not in {None, "Z", "X"}]


StartGroup = Callable[..., ProcessGroup]


@pytest.fixture
def start_group(tmp_path: Path) -> Iterator[StartGroup]:
    """Start real fixture process groups and kill whatever is left at teardown."""
    started: list[ProcessGroup] = []

    def start(
        *,
        token: str | None,
        recorded_token: str = "launch-token",
        workers: int = 2,
        ignore_term: bool = False,
        leader_exits: bool = False,
        hold_mib: int = 0,
    ) -> ProcessGroup:
        ready = tmp_path / f"group-{len(started)}.json"
        environment = {
            name: value
            for name, value in os.environ.items()
            if name != launch_ownership.LAUNCH_TOKEN_ENVIRONMENT_VARIABLE
        }
        if token is not None:
            environment[launch_ownership.LAUNCH_TOKEN_ENVIRONMENT_VARIABLE] = token
        flags = [
            "--hold-mib",
            str(hold_mib),
            *(["--ignore-term"] if ignore_term else []),
            *(["--leader-exits"] if leader_exits else []),
        ]
        leader = subprocess.Popen(
            [
                sys.executable,
                str(FIXTURES / "inference_process.py"),
                "group",
                "--workers",
                str(workers),
                "--ready",
                str(ready),
                *flags,
            ],
            env=environment,
            start_new_session=True,
        )
        deadline = time.monotonic() + 20
        while not ready.exists():
            assert time.monotonic() < deadline, "fixture process group did not start"
            time.sleep(0.05)
        marker = runtime.read_process_start_marker(leader.pid)
        assert marker is not None
        group = ProcessGroup(
            leader=leader,
            workers=json.loads(ready.read_text())["workers"],
            handle=models.LocalProcessHandle(
                pid=leader.pid,
                start_marker=marker,
                launch_token_sha256=launch_ownership.launch_token_proof(recorded_token),
                stdout_path=str(tmp_path / "stdout.log"),
                stderr_path=str(tmp_path / "stderr.log"),
            ),
        )
        started.append(group)
        return group

    yield start
    for group in started:
        for pid in (group.leader.pid, *group.workers):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        group.leader.wait(timeout=10)


@pytest.mark.skipif(sys.platform != "linux", reason="managed process identity uses Linux /proc")
def test_stop_terminates_a_group_whose_leader_is_alive(start_group: StartGroup) -> None:
    group = start_group(token="launch-token")

    assert runtime.status_run(group.receipt).state == "running"
    stopped = runtime.stop_run(group.receipt)

    assert (stopped.outcome, stopped.cleanup_complete) == ("terminated", True)
    assert group.live_workers() == []
    assert runtime.stop_run(group.receipt).outcome == "already-stopped"


@pytest.mark.skipif(sys.platform != "linux", reason="managed process identity uses Linux /proc")
def test_stop_forces_a_group_that_ignores_sigterm_and_waits_for_it_to_go(start_group: StartGroup) -> None:
    """Workers holding memory take a moment to die after SIGKILL; the stop must wait for them."""
    group = start_group(token="launch-token", workers=4, ignore_term=True, hold_mib=256)

    stopped = runtime.stop_run(group.receipt)

    assert (stopped.outcome, stopped.cleanup_complete) == ("forced", True)
    assert group.live_workers() == []


@pytest.mark.skipif(sys.platform != "linux", reason="managed process identity uses Linux /proc")
def test_stop_reaches_this_launchs_workers_after_the_leader_is_gone(start_group: StartGroup) -> None:
    """A crashed leader must not hide workers that still hold GPU memory."""
    group = start_group(token="launch-token", leader_exits=True)
    group.reap_leader()

    assert runtime.status_run(group.receipt).state == "running"
    stopped = runtime.stop_run(group.receipt)

    assert (stopped.outcome, stopped.cleanup_complete) == ("terminated", True)
    assert group.live_workers() == []


@pytest.mark.skipif(sys.platform != "linux", reason="managed process identity uses Linux /proc")
@pytest.mark.parametrize("foreign_token", [None, "another-launch"])
def test_stale_receipt_refuses_a_leaderless_group_it_cannot_prove_it_owns(
    start_group: StartGroup, foreign_token: str | None
) -> None:
    """A receipt whose leader is gone must not signal a group that merely has the recorded ID."""
    group = start_group(token=foreign_token, recorded_token="launch-token", leader_exits=True)
    group.reap_leader()

    assert runtime.status_run(group.receipt).state == "ownership-unverified"
    with (
        mock.patch.object(runtime.os, "killpg", wraps=runtime.os.killpg) as killpg,
        pytest.raises(runtime.RuntimeEffectError) as exc_info,
    ):
        runtime.stop_run(group.receipt)

    assert exc_info.value.diagnostic.code == "process-group-ownership-unverified"
    assert exc_info.value.diagnostic.cleanup_complete is False
    killpg.assert_not_called()
    assert sorted(group.live_workers()) == sorted(group.workers)


@pytest.mark.skipif(sys.platform != "linux", reason="managed process identity uses Linux /proc")
def test_group_holding_only_an_unreaped_leader_counts_as_stopped(start_group: StartGroup) -> None:
    group = start_group(token="launch-token", workers=0, leader_exits=True)
    deadline = time.monotonic() + 10
    while runtime.read_process_state(group.leader.pid) != "Z":
        assert time.monotonic() < deadline
        time.sleep(0.05)

    assert runtime.status_run(group.receipt).state == "stopped"
    assert runtime.stop_run(group.receipt).outcome == "already-stopped"


@pytest.mark.skipif(sys.platform != "linux", reason="managed process identity uses Linux /proc")
@pytest.mark.skipif(
    not hasattr(os, "geteuid") or os.geteuid() != 0 or shutil.which("unshare") is None,
    reason="needs root and unshare to force PID reuse in a private PID namespace",
)
def test_stale_receipt_does_not_signal_a_new_session_that_reused_the_group_id(tmp_path: Path) -> None:
    """Regression: once a group fully exits its ID can go to an unrelated session."""
    environment = os.environ | {"PYTHONPATH": str(TOOLS)}
    result = subprocess.run(
        [
            "unshare",
            "--pid",
            "--fork",
            "--mount-proc",
            sys.executable,
            str(FIXTURES / "pid_reuse_probe.py"),
            str(tmp_path),
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    observed = json.loads(result.stdout)

    assert observed["foreign_pgid"] == observed["recorded_pgid"]
    assert observed["status"] == "ownership-unverified"
    assert observed["stop"] == {"code": "process-group-ownership-unverified", "cleanup_complete": False}
    assert observed["foreign_workers_alive"] is True


def _fixture_server_plan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **backend: object) -> models.RunPlan:
    serving = tmp_path / "serving"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(serving)
    server = tmp_path / "tools/inference_service_compiler/vllm_server.py"
    server.parent.mkdir(parents=True)
    shutil.copyfile(FIXTURES / "inference_process.py", server)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    monkeypatch.chdir(tmp_path)
    return compiler.compile_spec(
        generation(
            backend=backend,
            python_executable=str(serving / "bin/python"),
            port=port,
            startup_timeout_seconds=60,
            shutdown_timeout_seconds=1,
        )
    )


@pytest.mark.skipif(sys.platform != "linux", reason="managed process identity uses Linux /proc")
def test_launch_fails_fast_and_cleans_up_when_the_endpoint_serves_another_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _fixture_server_plan(tmp_path, monkeypatch, served_model_name="some/other-model")

    started = time.monotonic()
    with pytest.raises(runtime.RuntimeEffectError) as exc_info:
        runtime.launch_plan(plan, secret_values={}, log_directory=tmp_path / "logs")

    assert time.monotonic() - started < 30
    diagnostic = exc_info.value.diagnostic
    assert diagnostic.code == "served-model-mismatch"
    assert "some/other-model" in diagnostic.message
    assert diagnostic.cleanup_complete is True
    pid = int(diagnostic.known_effects[0].split(":")[0])
    os.waitpid(pid, 0)


def test_launch_environment_passes_only_what_the_service_needs(monkeypatch: pytest.MonkeyPatch) -> None:
    plan = compiler.compile_spec(generation(api_key_env="LOCAL_KEY"))
    monkeypatch.setenv("LOCAL_KEY", "source-secret")
    monkeypatch.setenv("VLLM_API_KEY", "inherited-secret")
    monkeypatch.setenv("WANDB_API_KEY", "unrelated-secret")
    monkeypatch.setenv("GITHUB_ACCESS_TOKEN", "unrelated-secret")
    monkeypatch.setenv("HF_TOKEN", "hub-token")
    monkeypatch.setenv("PYTHONPATH", "/controller")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")

    environment = runtime._launch_environment(plan, "endpoint-secret", "launch-token")

    assert environment["VLLM_API_KEY"] == "endpoint-secret"
    assert environment[launch_ownership.LAUNCH_TOKEN_ENVIRONMENT_VARIABLE] == "launch-token"
    assert environment["SPT_NOENV"] == "1"
    assert environment["HF_TOKEN"] == "hub-token"
    assert environment["CUDA_VISIBLE_DEVICES"] == "1"
    for name in ("LOCAL_KEY", "WANDB_API_KEY", "GITHUB_ACCESS_TOKEN", "PYTHONPATH"):
        assert name not in environment
    assert "source-secret" not in environment.values()


def test_gliner2_launch_environment_drops_an_inherited_token_without_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(backends.GLINER2_TOKEN_ENV, "inherited")

    environment = runtime._launch_environment(compiler.compile_spec(detection()), None, "launch-token")

    assert backends.GLINER2_TOKEN_ENV not in environment
    assert environment[backends.GLINER2_DEVICE_ENV] == "auto"
