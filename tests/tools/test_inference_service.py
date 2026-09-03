# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Observable contracts for the local-process vLLM inference host."""

from __future__ import annotations

import asyncio
import json
import stat
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import httpx
import pytest
from pydantic import ValidationError

from inference_service_compiler import cli, compiler, lifecycle, models, runtime
from inference_service_compiler.profiles import load_profile

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
PROFILES = TOOLS / "inference_service_profiles"
CLI = TOOLS / "inference_service.py"


def generation(**vllm: object) -> models.LocalInferenceServiceSpec:
    return models.LocalInferenceServiceSpec(
        task=models.Generation(),
        model=models.HuggingFaceModel(model_id="openai/gpt-oss-20b", revision="abc"),
        vllm=models.Vllm.model_validate(vllm),
        local=models.LocalProcess(port=8000, startup_timeout_seconds=3, shutdown_timeout_seconds=0.01),
    )


def launch_receipt(
    plan: models.RunPlan,
    handle: models.LocalProcessHandle,
) -> models.LaunchReceipt:
    return models.LaunchReceipt(
        plan_digest=plan.plan_digest,
        launched_at="2026-08-07T00:00:00+00:00",
        shutdown_timeout_seconds=0.01,
        handle=handle,
        probe=models.CapabilityProbeReceipt(
            plan_digest=plan.plan_digest,
            endpoint=plan.endpoint,
            observed_at="2026-08-07T00:00:00+00:00",
            models=(plan.served_model_name,),
            observed_capabilities=plan.required_capabilities,
            passed=True,
        ),
    )


def test_cli_remains_a_directly_executable_source_entrypoint() -> None:
    assert CLI.is_file()
    assert CLI.stat().st_mode & stat.S_IXUSR


def test_all_shipped_profiles_compile() -> None:
    plans = [
        compiler.compile_profile(load_profile(path), source_revision="test") for path in sorted(PROFILES.glob("*.toml"))
    ]
    assert plans
    assert all(plan.schema_version == "inference-service.run-plan/v2" for plan in plans)
    assert all(plan.readiness.path == "/models" for plan in plans)
    assert all(plan.served_model_name for plan in plans)


def test_plan_keeps_one_endpoint_address_and_one_served_model_vocabulary() -> None:
    plan = compiler.compile_profile(generation(served_model_name="local-generator"), source_revision="test")

    serialized = plan.model_dump(mode="json")
    assert plan.endpoint.url == "http://127.0.0.1:8000/v1"
    assert plan.readiness.path == "/models"
    assert plan.served_model_name == "local-generator"
    assert "host" not in serialized["readiness"]
    assert "port" not in serialized["readiness"]
    assert "intent_digest" not in serialized
    assert "declared_capabilities" not in serialized


def test_compile_command_writes_a_digest_verified_plan(tmp_path: Path) -> None:
    output = tmp_path / "plan.json"
    with pytest.raises(SystemExit) as exc_info:
        cli.app(
            [
                "compile",
                "--profile",
                str(PROFILES / "vllm-local.toml"),
                "--source-revision",
                "test",
                "--output",
                str(output),
            ]
        )
    assert exc_info.value.code == 0
    plan = compiler.load_plan(output.read_text(encoding="utf-8"))
    assert plan.served_model_name == "anonymizer-local"


def test_compile_command_translates_non_directory_profile_paths(tmp_path: Path) -> None:
    """Filesystem path-shape errors retain the documented bad-input exit."""
    not_a_directory = tmp_path / "profile-file"
    not_a_directory.write_text("not a directory", encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        cli.app(
            [
                "compile",
                "--profile",
                str(not_a_directory / "profile.toml"),
                "--source-revision",
                "test",
            ]
        )

    assert exc_info.value.code == 125


def test_compile_command_translates_empty_source_revision() -> None:
    """Known compiler input errors retain the documented bad-input exit."""
    with pytest.raises(SystemExit) as exc_info:
        cli.app(
            [
                "compile",
                "--profile",
                str(PROFILES / "vllm-local.toml"),
                "--source-revision",
                "",
            ]
        )

    assert exc_info.value.code == 125


def test_generation_argv_keeps_local_vllm_controls_and_omits_defaults() -> None:
    plan = compiler.compile_profile(
        generation(api_key_env="LOCAL_KEY", tensor_parallel_size=2, max_model_len=4096, eager=True),
        source_revision="test",
    )
    argv = plan.command.render_argv()
    assert argv[:3] == (".venv/bin/python", "tools/inference_service_compiler/vllm_server.py", "openai/gpt-oss-20b")
    assert ("--tensor-parallel-size", "2") == argv[
        argv.index("--tensor-parallel-size") : argv.index("--tensor-parallel-size") + 2
    ]
    assert "--enforce-eager" in argv and "--enable-prefix-caching" not in argv
    assert plan.command.secret_sources == ("LOCAL_KEY",)
    assert plan.command.render_environment() == {"VLLM_API_KEY": "<secret:LOCAL_KEY>"}
    assert plan.command.resolve_environment({"LOCAL_KEY": "secret-value"}) == {"VLLM_API_KEY": "secret-value"}


def test_factory_detection_is_task_bounded() -> None:
    valid = models.LocalInferenceServiceSpec(
        task=models.EntityDetection(dynamic_labels=True, offsets=True, scores=True),
        model=models.HuggingFaceModel(model_id="nvidia/gliner-pii", revision="abc"),
        vllm=models.Vllm(factory=models.VllmFactoryIntegration(plugin="deberta_gliner")),
        local=models.LocalProcess(),
    )
    assert "--vllm-factory-plugin" in compiler.compile_profile(valid, source_revision="test").command.render_argv()
    with pytest.raises(compiler.CompilationError, match="does not support"):
        compiler.compile_profile(
            generation(factory=models.VllmFactoryIntegration(plugin="deberta_gliner")), source_revision="test"
        )


def test_factory_plugin_is_closed_at_the_spec_boundary() -> None:
    """Profiles cannot select an uncharacterized Factory plugin."""
    with pytest.raises(ValidationError):
        models.VllmFactoryIntegration.model_validate({"plugin": "unsupported"})


def test_factory_detection_requires_a_pin_and_characterized_model() -> None:
    for model_id, revision, message in (
        ("nvidia/gliner-pii", None, "pinned model revision"),
        ("unknown/model", "abc", "not characterized"),
    ):
        spec = models.LocalInferenceServiceSpec(
            task=models.EntityDetection(dynamic_labels=True, offsets=True, scores=True),
            model=models.HuggingFaceModel(model_id=model_id, revision=revision),
            vllm=models.Vllm(factory=models.VllmFactoryIntegration(plugin="deberta_gliner")),
            local=models.LocalProcess(),
        )
        with pytest.raises(compiler.CompilationError, match=message):
            compiler.compile_profile(spec, source_revision="test")


def test_removed_domains_are_invalid_profile_fields() -> None:
    with pytest.raises(ValidationError):
        models.LocalInferenceServiceSpec.model_validate(
            {
                "schema_version": "inference-service.local-spec/v2",
                "task": {"kind": "generation"},
                "model": {"model_id": "x"},
                "vllm": {},
                "local": {},
                "placement": {"kind": "docker"},
            }
        )


def test_plan_digest_detects_transport_mutation() -> None:
    plan = compiler.compile_profile(generation(), source_revision="test")
    changed = json.loads(plan.model_dump_json())
    changed["endpoint"]["port"] = 9000
    with pytest.raises(compiler.PlanIntegrityError, match="plan digest mismatch"):
        compiler.load_plan(json.dumps(changed))


def test_lora_is_rendered_as_a_model_artifact() -> None:
    spec = models.LocalInferenceServiceSpec(
        task=models.Generation(),
        model=models.HuggingFaceModel(
            model_id="openai/gpt-oss-20b",
            adapter=models.LoraAdapter(path="/models/privacy-adapter", name="privacy"),
        ),
        vllm=models.Vllm(),
        local=models.LocalProcess(),
    )
    argv = compiler.compile_profile(spec, source_revision="test").command.render_argv()
    assert argv[-3:] == ("--enable-lora", "--lora-modules", "privacy=/models/privacy-adapter")


def test_probe_payload_is_task_aware_and_reasoning_safe() -> None:
    plan = compiler.compile_profile(generation(), source_revision="test")
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "openai/gpt-oss-20b"}]})
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok", "reasoning_content": "r"}}]})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        receipt = runtime.probe_endpoint(plan, client=client)
        assert client.is_closed is False
    assert receipt.passed and receipt.observed_capabilities == ("chat-completions",)
    payload = json.loads(requests[-1].content)
    assert payload["max_tokens"] == 128
    assert payload["chat_template_kwargs"] == {"enable_thinking": False, "reasoning_effort": "low"}


def test_probe_uses_bearer_secret_without_serializing_it() -> None:
    plan = compiler.compile_profile(generation(api_key_env="LOCAL_KEY"), source_revision="test")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer test-secret"
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": plan.served_model_name}]})
        return httpx.Response(200, json={"choices": [{"message": {"content": "ready"}}]})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        receipt = runtime.probe_endpoint(plan, client=client, secret_values={"LOCAL_KEY": "test-secret"})
    assert receipt.passed
    assert "test-secret" not in receipt.model_dump_json()


def test_probe_rejects_wrong_model_and_status() -> None:
    plan = compiler.compile_profile(generation(), source_revision="test")

    def wrong_model(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "other/model"}]})
        return httpx.Response(200, json={"choices": [{"message": {"content": "ready"}}]})

    with httpx.Client(transport=httpx.MockTransport(wrong_model)) as client:
        assert runtime.probe_endpoint(plan, client=client).passed is False
    with httpx.Client(transport=httpx.MockTransport(lambda _request: httpx.Response(201, json={}))) as client:
        with pytest.raises(runtime.RuntimeEffectError, match="status 201"):
            runtime.probe_endpoint(plan, client=client)


def test_plan_integrity_is_enforced_before_launch(tmp_path: Path) -> None:
    plan = compiler.compile_profile(generation(), source_revision="test")
    with pytest.raises(compiler.PlanIntegrityError):
        runtime.launch_plan(
            plan.model_copy(update={"source_revision": "changed"}), secret_values={}, log_directory=tmp_path
        )


def test_status_and_stop_reject_a_reused_pid_without_signaling(monkeypatch: pytest.MonkeyPatch) -> None:
    plan = compiler.compile_profile(generation(), source_revision="test")
    handle = models.LocalProcessHandle(
        external_id="1:x", pid=1, process_group_id=1, start_marker="old", stdout_path="out", stderr_path="err"
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
    plan = compiler.compile_profile(generation(), source_revision="test")
    handle = models.LocalProcessHandle(
        external_id="4242:100",
        pid=4242,
        process_group_id=4242,
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
        mock.patch.object(runtime.time, "monotonic", side_effect=[0.0, 1.0]),
        pytest.raises(runtime.RuntimeEffectError) as exc_info,
    ):
        runtime.stop_run(launch)

    assert exc_info.value.diagnostic.code == "process-identity-mismatch"
    killpg.assert_not_called()


def test_launch_records_process_identity_and_resolves_secrets(tmp_path: Path) -> None:
    plan = compiler.compile_profile(generation(api_key_env="LOCAL_KEY"), source_revision="test")
    probe = models.CapabilityProbeReceipt(
        plan_digest=plan.plan_digest,
        endpoint=plan.endpoint,
        observed_at="2026-08-07T00:00:00+00:00",
        models=(plan.served_model_name,),
        observed_capabilities=("chat-completions",),
        passed=True,
    )
    process = mock.Mock(pid=4242)
    with (
        mock.patch.object(runtime.subprocess, "Popen", return_value=process) as popen,
        mock.patch.object(runtime, "wait_for_readiness", return_value=probe),
        mock.patch.object(runtime, "read_process_start_marker", return_value="100"),
    ):
        receipt = runtime.launch_plan(plan, secret_values={"LOCAL_KEY": "test-secret"}, log_directory=tmp_path)
    assert tuple(popen.call_args.args[0]) == plan.command.render_argv()
    assert popen.call_args.kwargs["env"]["VLLM_API_KEY"] == "test-secret"
    assert "test-secret" not in popen.call_args.args[0]
    assert receipt.handle.external_id == "4242:100"


def test_launch_requires_scoped_ownership_evidence(tmp_path: Path) -> None:
    plan = compiler.compile_profile(generation(), source_revision="test")
    handle = models.LocalProcessHandle(
        external_id="4242:100",
        pid=4242,
        process_group_id=4242,
        start_marker="100",
        stdout_path="out",
        stderr_path="err",
    )
    probe = launch_receipt(plan, handle).probe
    observed_environment: dict[str, str] = {}
    observed_token: str | None = None

    def launch_process(
        _plan: models.RunPlan,
        _argv: tuple[str, ...],
        environment: dict[str, str],
        _log_directory: Path,
    ) -> models.LocalProcessHandle:
        observed_environment.update(environment)
        return handle

    def wait_for_readiness(
        _plan: models.RunPlan,
        *,
        secret_values: object,
        handle: models.LocalProcessHandle,
        launch_token: str | None = None,
    ) -> models.CapabilityProbeReceipt:
        del secret_values, handle
        nonlocal observed_token
        observed_token = launch_token
        return probe

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
        headers = {} if token is None else {lifecycle.LAUNCH_OWNERSHIP_HEADER: token}
        return SimpleNamespace(url=SimpleNamespace(path=path), headers=headers)

    with mock.patch.dict(
        runtime.os.environ,
        {lifecycle.LAUNCH_TOKEN_ENVIRONMENT_VARIABLE: "launch-secret"},
        clear=True,
    ):
        wrong = asyncio.run(lifecycle.launch_ownership(request(lifecycle.LAUNCH_OWNERSHIP_PATH, "wrong"), next_handler))
        owned = asyncio.run(
            lifecycle.launch_ownership(request(lifecycle.LAUNCH_OWNERSHIP_PATH, "launch-secret"), next_handler)
        )
        passed_through = asyncio.run(lifecycle.launch_ownership(request("/v1/models", None), next_handler))

    assert wrong.status_code == 404
    assert json.loads(owned.body) == {
        lifecycle.LAUNCH_OWNERSHIP_PROOF_FIELD: lifecycle.launch_token_proof("launch-secret")
    }
    assert passed_through is mock.sentinel.next_response


def test_compatible_preexisting_endpoint_cannot_prove_launch_ownership() -> None:
    plan = compiler.compile_profile(generation(), source_revision="test")
    compatible_response = httpx.Response(200, json={"data": [{"id": plan.served_model_name}]})

    with (
        mock.patch.object(runtime.httpx, "get", return_value=compatible_response),
        pytest.raises(runtime.RuntimeEffectError) as exc_info,
    ):
        runtime._probe_launch_ownership(plan, "launch-secret")

    assert exc_info.value.diagnostic.code == "launch-ownership-not-observed"


def test_launch_ownership_probe_receives_endpoint_authentication() -> None:
    plan = compiler.compile_profile(generation(api_key_env="LOCAL_KEY"), source_revision="test")
    probe = launch_receipt(
        plan,
        models.LocalProcessHandle(
            external_id="4242:100",
            pid=4242,
            process_group_id=4242,
            start_marker="100",
            stdout_path="out",
            stderr_path="err",
        ),
    ).probe
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
        mock.patch.object(runtime, "probe_endpoint", return_value=probe),
    ):
        runtime.wait_for_readiness(
            plan,
            secret_values={"LOCAL_KEY": "endpoint-secret"},
            launch_token="launch-secret",
        )

    assert observed_secrets == {"LOCAL_KEY": "endpoint-secret"}


@pytest.mark.parametrize("failure", [RuntimeError("unexpected readiness failure"), KeyboardInterrupt()])
def test_post_spawn_readiness_failures_always_clean_up(tmp_path: Path, failure: BaseException) -> None:
    plan = compiler.compile_profile(generation(), source_revision="test")
    handle = models.LocalProcessHandle(
        external_id="4242:100",
        pid=4242,
        process_group_id=4242,
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

    cleanup.assert_called_once_with(handle, plan.spec.local.shutdown_timeout_seconds)


def test_launch_command_stops_service_when_receipt_write_fails(tmp_path: Path) -> None:
    plan = compiler.compile_profile(generation(), source_revision="test")
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(plan.model_dump_json(), encoding="utf-8")
    handle = models.LocalProcessHandle(
        external_id="4242:100",
        pid=4242,
        process_group_id=4242,
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
    plan = compiler.compile_profile(generation(), source_revision="test")
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(plan.model_dump_json(), encoding="utf-8")
    output = tmp_path / "receipt"
    output.mkdir()
    handle = models.LocalProcessHandle(
        external_id="4242:100",
        pid=4242,
        process_group_id=4242,
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
    plan = compiler.compile_profile(generation(), source_revision="test")
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
    plan = compiler.compile_profile(generation(), source_revision="test")
    process = mock.Mock(pid=4242)
    process.poll.return_value = None
    process.wait.side_effect = (
        runtime.subprocess.TimeoutExpired(cmd=plan.command.render_argv(), timeout=1),
        runtime.subprocess.TimeoutExpired(cmd=plan.command.render_argv(), timeout=1),
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
    plan = compiler.compile_profile(generation(api_key_env="LOCAL_KEY"), source_revision="test")
    with mock.patch.object(runtime.subprocess, "Popen") as popen:
        with pytest.raises(runtime.RuntimeEffectError) as exc_info:
            runtime.launch_plan(plan, secret_values={}, log_directory=tmp_path)
    popen.assert_not_called()
    assert exc_info.value.diagnostic.code == "missing-secret"
    assert exc_info.value.diagnostic.known_effects == ()


def test_status_stop_and_forced_cleanup_are_versioned(tmp_path: Path) -> None:
    plan = compiler.compile_profile(generation(), source_revision="test")
    handle = models.LocalProcessHandle(
        external_id="4242:100",
        pid=4242,
        process_group_id=4242,
        start_marker="100",
        stdout_path=str(tmp_path / "stdout.log"),
        stderr_path=str(tmp_path / "stderr.log"),
    )
    launch = launch_receipt(plan, handle)
    with mock.patch.object(runtime, "is_handle_running", return_value=True):
        assert runtime.status_run(launch).state == "running"
    with (
        mock.patch.object(runtime, "is_handle_running", side_effect=[True, True, False]),
        mock.patch.object(runtime.time, "monotonic", side_effect=[0.0, 0.0, 1.0]),
        mock.patch.object(runtime.os, "killpg") as killpg,
    ):
        stopped = runtime.stop_run(launch)
    assert stopped.outcome == "forced"
    assert stopped.cleanup_complete is True
    assert [call.args[1] for call in killpg.call_args_list] == [runtime.signal.SIGTERM, runtime.signal.SIGKILL]


def test_process_stat_handles_spaces_and_zombies(tmp_path: Path) -> None:
    fields = ["S", *(str(index) for index in range(4, 22)), "98765"]
    assert runtime._parse_process_stat(f"4242 (worker with spaces) {' '.join(fields)}") == ("S", "98765")
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


def test_failed_readiness_cleans_up_the_known_process(tmp_path: Path) -> None:
    plan = compiler.compile_profile(generation(), source_revision="test")
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


def test_readiness_stops_when_the_managed_process_exits(tmp_path: Path) -> None:
    plan = compiler.compile_profile(generation(), source_revision="test")
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
        pytest.raises(runtime.RuntimeEffectError, match="exited before readiness"),
    ):
        runtime.wait_for_readiness(plan, handle=handle)
    probe.assert_not_called()
