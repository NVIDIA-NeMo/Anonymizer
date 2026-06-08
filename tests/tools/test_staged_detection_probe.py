# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from anonymizer.engine.schemas import ValidationCandidateSchema

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_tool(module_name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(module)
    return module


class SequencedClient:
    def __init__(self, tool: ModuleType, outputs: list[str]) -> None:
        self._tool = tool
        self._outputs = list(outputs)
        self.prompts: list[str] = []

    def complete(self, request):  # type: ignore[no-untyped-def]
        self.prompts.append(request.prompt)
        content = self._outputs.pop(0)
        return self._tool.DirectCompletion(
            content=content,
            elapsed_sec=0.5,
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )


class StaticSeedClient:
    def __init__(self, tool: ModuleType, content: str) -> None:
        self._tool = tool
        self.content = content
        self.requests = []

    def detect(self, request):  # type: ignore[no-untyped-def]
        self.requests.append(request)
        return self._tool.DirectCompletion(
            content=self.content,
            elapsed_sec=0.25,
            usage={"prompt_tokens": 1, "completion_tokens": 3, "total_tokens": 4},
        )


def test_staged_detection_reuses_validation_and_augmentation_flow() -> None:
    tool = load_tool(
        "measurement_staged_detection_probe_case",
        REPO_ROOT / "tools/measurement/staged_detection_probe.py",
    )
    client = SequencedClient(
        tool,
        [
            '{"entities": [{"value": "Alice", "label": "first_name", "reason": "name"}]}',
            '{"decisions": [{"id": "first_name_0_5", "decision": "keep", "reason": "real name"}]}',
            '{"entities": [{"value": "NVIDIA", "label": "organization_name", "reason": "employer"}]}',
        ],
    )

    result = tool.run_staged_detection_case(
        tool.StagedDetectionRequest(
            case_id="case-1",
            text="Alice works at NVIDIA.",
            labels=["first_name", "organization_name"],
            row_index=0,
        ),
        client=client,
    )

    assert result.status == tool.CaseStatus.completed
    assert result.seed_suggestion_count == 1
    assert result.seed_entity_count == 1
    assert result.validation_decision_count == 1
    assert result.augmented_suggestion_count == 1
    assert result.final_entity_count == 2
    assert result.final_label_counts == {"first_name": 1, "organization_name": 1}
    assert result.artifact.final_source_counts == {"direct_seed": 1, "augmenter": 1}
    assert result.phase_model_work == tool.PhaseModelWork(seed=True, validation=True, augmentation=True)
    assert result.phase_skip_reasons == tool.PhaseSkipReasons()
    assert result.model_phase_count == 3
    assert result.phase_model_requests == tool.PhaseModelRequests(seed=1, validation=1, augmentation=1)
    assert result.model_request_count == 3
    assert result.total_usage == {"prompt_tokens": 30, "completion_tokens": 15, "total_tokens": 45}
    assert len(client.prompts) == 3


def test_staged_detection_can_disable_augmentation_phase() -> None:
    tool = load_tool(
        "measurement_staged_detection_probe_disable_augmentation",
        REPO_ROOT / "tools/measurement/staged_detection_probe.py",
    )
    client = SequencedClient(
        tool,
        [
            '{"entities": [{"value": "Alice", "label": "first_name", "reason": "name"}]}',
            '{"decisions": [{"id": "first_name_0_5", "decision": "keep", "reason": "real name"}]}',
        ],
    )

    result = tool.run_staged_detection_case(
        tool.StagedDetectionRequest(
            case_id="case-1",
            text="Alice works at NVIDIA.",
            labels=["first_name", "organization_name"],
            row_index=0,
        ),
        client=client,
        skip_augmentation=True,
    )

    assert result.status == tool.CaseStatus.completed
    assert result.final_label_counts == {"first_name": 1}
    assert result.phase_model_work == tool.PhaseModelWork(seed=True, validation=True, augmentation=False)
    assert result.phase_skip_reasons == tool.PhaseSkipReasons(augmentation="disabled")
    assert result.phase_model_requests == tool.PhaseModelRequests(seed=1, validation=1, augmentation=0)
    assert result.model_request_count == 2
    assert len(client.prompts) == 2


def test_staged_detection_execution_exposes_output_row_without_serializing_it() -> None:
    tool = load_tool(
        "measurement_staged_detection_probe_execution",
        REPO_ROOT / "tools/measurement/staged_detection_probe.py",
    )
    client = SequencedClient(
        tool,
        [
            '{"entities": [{"value": "Alice", "label": "first_name", "reason": "name"}]}',
            '{"decisions": [{"id": "first_name_0_5", "decision": "keep", "reason": "real name"}]}',
            '{"entities": []}',
        ],
    )

    execution = tool.execute_staged_detection_case(
        tool.StagedDetectionRequest(
            case_id="case-1",
            text="Alice works remotely.",
            labels=["first_name"],
            row_index=0,
        ),
        client=client,
    )

    assert execution.case.status == tool.CaseStatus.completed
    assert execution.row[tool.COL_DETECTED_ENTITIES]["entities"][0]["value"] == "Alice"
    assert "row" not in execution.case.model_dump()


def test_staged_detection_validation_drop_removes_seed_entity() -> None:
    tool = load_tool(
        "measurement_staged_detection_probe_drop",
        REPO_ROOT / "tools/measurement/staged_detection_probe.py",
    )
    client = SequencedClient(
        tool,
        [
            '{"entities": [{"value": "name", "label": "first_name", "reason": "placeholder"}]}',
            '{"decisions": [{"id": "first_name_3_7", "decision": "drop", "reason": "placeholder"}]}',
            '{"entities": []}',
        ],
    )

    result = tool.run_staged_detection_case(
        tool.StagedDetectionRequest(
            case_id="case-1",
            text="my name is hidden",
            labels=["first_name"],
            row_index=0,
        ),
        client=client,
    )

    assert result.status == tool.CaseStatus.completed
    assert result.seed_entity_count == 1
    assert result.validation_decision_count == 1
    assert result.final_entity_count == 0
    assert result.final_label_counts == {}


def test_staged_detection_invalid_reclass_label_keeps_seed_label() -> None:
    tool = load_tool(
        "measurement_staged_detection_probe_invalid_reclass_label",
        REPO_ROOT / "tools/measurement/staged_detection_probe.py",
    )
    client = SequencedClient(
        tool,
        [
            '{"entities": [{"value": "Alice", "label": "first_name", "reason": "name"}]}',
            (
                '{"decisions": ['
                '{"id": "first_name_0_5", "decision": "reclass", '
                '"proposed_label": "drop", "reason": "invalid label"}'
                "]}"
            ),
        ],
    )

    result = tool.run_staged_detection_case(
        tool.StagedDetectionRequest(
            case_id="case-1",
            text="Alice works remotely.",
            labels=["first_name"],
            row_index=0,
        ),
        client=client,
        skip_augmentation=True,
    )

    assert result.status == tool.CaseStatus.completed
    assert result.final_entity_count == 1
    assert result.final_label_counts == {"first_name": 1}


def test_staged_detection_discards_invalid_augmentation_labels() -> None:
    tool = load_tool(
        "measurement_staged_detection_probe_invalid_augmentation_label",
        REPO_ROOT / "tools/measurement/staged_detection_probe.py",
    )
    client = SequencedClient(
        tool,
        [
            '{"entities": [{"value": "Alice", "label": "first_name", "reason": "name"}]}',
            '{"decisions": [{"id": "first_name_0_5", "decision": "keep", "reason": "real name"}]}',
            '{"entities": [{"value": "NVIDIA", "label": "drop", "reason": "invalid label"}]}',
        ],
    )

    result = tool.run_staged_detection_case(
        tool.StagedDetectionRequest(
            case_id="case-1",
            text="Alice works at NVIDIA.",
            labels=["first_name", "organization_name"],
            row_index=0,
        ),
        client=client,
    )

    assert result.status == tool.CaseStatus.completed
    assert result.augmented_suggestion_count == 0
    assert result.final_entity_count == 1
    assert result.final_label_counts == {"first_name": 1}


def test_staged_detection_preserves_more_specific_seed_label_on_native_reclass() -> None:
    tool = load_tool(
        "measurement_staged_detection_probe_specific_label_reclass",
        REPO_ROOT / "tools/measurement/staged_detection_probe.py",
    )
    client = SequencedClient(
        tool,
        [
            '{"entities": [{"value": "23 October 1992", "label": "date_of_birth", "reason": "birth date"}]}',
            (
                '{"decisions": ['
                '{"id": "date_of_birth_26_41", "decision": "reclass", '
                '"proposed_label": "date", "reason": "date expression"}'
                "]}"
            ),
        ],
    )

    result = tool.run_staged_detection_case(
        tool.StagedDetectionRequest(
            case_id="case-1",
            text="The applicant was born on 23 October 1992.",
            labels=["date", "date_of_birth"],
            row_index=0,
        ),
        client=client,
        skip_augmentation=True,
    )

    assert result.status == tool.CaseStatus.completed
    assert result.final_entity_count == 1
    assert result.final_label_counts == {"date_of_birth": 1}


def test_staged_detection_allows_date_of_birth_reclass_without_birth_context() -> None:
    tool = load_tool(
        "measurement_staged_detection_probe_generic_date_reclass",
        REPO_ROOT / "tools/measurement/staged_detection_probe.py",
    )
    client = SequencedClient(
        tool,
        [
            '{"entities": [{"value": "23 October 1992", "label": "date_of_birth", "reason": "date"}]}',
            (
                '{"decisions": ['
                '{"id": "date_of_birth_3_18", "decision": "reclass", '
                '"proposed_label": "date", "reason": "filing date"}'
                "]}"
            ),
        ],
    )

    result = tool.run_staged_detection_case(
        tool.StagedDetectionRequest(
            case_id="case-1",
            text="On 23 October 1992 the applicant filed an action.",
            labels=["date", "date_of_birth"],
            row_index=0,
        ),
        client=client,
        skip_augmentation=True,
    )

    assert result.status == tool.CaseStatus.completed
    assert result.final_entity_count == 1
    assert result.final_label_counts == {"date": 1}


def test_staged_detection_demotes_native_birth_date_without_birth_context() -> None:
    tool = load_tool(
        "measurement_staged_detection_probe_generic_date_birth_label",
        REPO_ROOT / "tools/measurement/staged_detection_probe.py",
    )
    client = SequencedClient(
        tool,
        [
            '{"entities": [{"value": "23 October 1992", "label": "date", "reason": "date"}]}',
            (
                '{"decisions": ['
                '{"id": "date_3_18", "decision": "reclass", '
                '"proposed_label": "date_of_birth", "reason": "ambiguous date"}'
                "]}"
            ),
        ],
    )

    result = tool.run_staged_detection_case(
        tool.StagedDetectionRequest(
            case_id="case-1",
            text="On 23 October 1992 the applicant filed an action.",
            labels=["date", "date_of_birth"],
            row_index=0,
        ),
        client=client,
        skip_augmentation=True,
    )

    assert result.status == tool.CaseStatus.completed
    assert result.final_entity_count == 1
    assert result.final_label_counts == {"date": 1}


def test_staged_detection_can_seed_from_direct_gliner_payload_without_llm_seed_prompt() -> None:
    tool = load_tool(
        "measurement_staged_detection_probe_gliner_seed",
        REPO_ROOT / "tools/measurement/staged_detection_probe.py",
    )
    seed_client = StaticSeedClient(
        tool,
        '{"entities": [{"text": "Alice", "label": "first_name", "start": 0, "end": 5, "score": 0.99}]}',
    )
    llm_client = SequencedClient(
        tool,
        [
            '{"decisions": [{"id": "first_name_0_5", "decision": "keep", "reason": "real name"}]}',
            '{"entities": [{"value": "NVIDIA", "label": "organization_name", "reason": "employer"}]}',
        ],
    )

    result = tool.run_staged_detection_case(
        tool.StagedDetectionRequest(
            case_id="case-1",
            text="Alice works at NVIDIA.",
            labels=["first_name", "organization_name"],
            row_index=0,
        ),
        client=llm_client,
        seed_client=seed_client,
        seed_source=tool.SeedSource.gliner,
    )

    assert result.status == tool.CaseStatus.completed
    assert result.seed_source == tool.SeedSource.gliner
    assert result.seed_entity_count == 1
    assert result.final_label_counts == {"first_name": 1, "organization_name": 1}
    assert result.total_usage == {"prompt_tokens": 21, "completion_tokens": 13, "total_tokens": 34}
    assert len(seed_client.requests) == 1
    assert len(llm_client.prompts) == 2


def test_staged_detection_promotes_gliner_date_seed_in_birth_context() -> None:
    tool = load_tool(
        "measurement_staged_detection_probe_gliner_birth_date_seed",
        REPO_ROOT / "tools/measurement/staged_detection_probe.py",
    )
    seed_client = StaticSeedClient(
        tool,
        ('{"entities": [{"text": "23 October 1992", "label": "date", "start": 26, "end": 41, "score": 0.99}]}'),
    )
    llm_client = SequencedClient(
        tool,
        ['{"decisions": [{"id": "date_of_birth_26_41", "decision": "keep", "reason": "birth date"}]}'],
    )

    result = tool.run_staged_detection_case(
        tool.StagedDetectionRequest(
            case_id="case-1",
            text="The applicant was born on 23 October 1992.",
            labels=["date", "date_of_birth"],
            row_index=0,
        ),
        client=llm_client,
        seed_client=seed_client,
        seed_source=tool.SeedSource.gliner,
        skip_augmentation=True,
    )

    assert result.status == tool.CaseStatus.completed
    assert result.seed_source == tool.SeedSource.gliner
    assert result.seed_entity_count == 1
    assert result.final_label_counts == {"date_of_birth": 1}


def test_staged_detection_keeps_gliner_date_seed_without_birth_context() -> None:
    tool = load_tool(
        "measurement_staged_detection_probe_gliner_generic_date_seed",
        REPO_ROOT / "tools/measurement/staged_detection_probe.py",
    )
    seed_client = StaticSeedClient(
        tool,
        ('{"entities": [{"text": "23 October 1992", "label": "date", "start": 3, "end": 18, "score": 0.99}]}'),
    )
    llm_client = SequencedClient(
        tool,
        ['{"decisions": [{"id": "date_3_18", "decision": "keep", "reason": "filing date"}]}'],
    )

    result = tool.run_staged_detection_case(
        tool.StagedDetectionRequest(
            case_id="case-1",
            text="On 23 October 1992 the applicant filed an action.",
            labels=["date", "date_of_birth"],
            row_index=0,
        ),
        client=llm_client,
        seed_client=seed_client,
        seed_source=tool.SeedSource.gliner,
        skip_augmentation=True,
    )

    assert result.status == tool.CaseStatus.completed
    assert result.seed_source == tool.SeedSource.gliner
    assert result.seed_entity_count == 1
    assert result.final_label_counts == {"date": 1}


def test_staged_detection_validation_prompt_preserves_degree_label_guidance() -> None:
    tool = load_tool(
        "measurement_staged_detection_probe_degree_guidance",
        REPO_ROOT / "tools/measurement/staged_detection_probe.py",
    )
    request = tool.StagedDetectionRequest(
        case_id="case-1",
        text="He earned his Bachelor of Science in physics.",
        labels=["degree", "education_level", "field_of_study"],
    )
    candidates = tool.ValidationCandidatesSchema(
        candidates=[
            ValidationCandidateSchema(
                id="education_level_14_33",
                value="Bachelor of Science",
                label="education_level",
                context_before="He earned his ",
                context_after=" in physics.",
            )
        ]
    )

    prompt = tool._validation_prompt(request, candidates)

    assert "degree" in prompt
    assert "education_level" in prompt
    assert "Bachelor of Science" in prompt
    assert "Prefer degree for credential names" in prompt


def test_staged_detection_augmentation_prompt_discourages_grouped_person_and_surname_spans() -> None:
    tool = load_tool(
        "measurement_staged_detection_probe_augmentation_guidance",
        REPO_ROOT / "tools/measurement/staged_detection_probe.py",
    )
    request = tool.StagedDetectionRequest(
        case_id="case-1",
        text="Her parents, Mark and Linda, live near West Baker Drive and Baker's grocery.",
        labels=["first_name", "last_name", "organization_name", "place_name", "company_name"],
    )

    prompt = tool._augmentation_prompt(request, {tool.COL_SEED_ENTITIES_JSON: "[]"})

    assert "split personal names connected by 'and'" in prompt
    assert "Do not label a list of people as organization_name" in prompt
    assert "also return the surname substring as last_name" in prompt


def test_staged_detection_can_seed_from_rules_without_llm_seed_prompt() -> None:
    tool = load_tool(
        "measurement_staged_detection_probe_rules_seed",
        REPO_ROOT / "tools/measurement/staged_detection_probe.py",
    )
    llm_client = SequencedClient(
        tool,
        [
            '{"decisions": [{"id": "email_6_23", "decision": "keep", "reason": "email address"}]}',
            '{"entities": [{"value": "NVIDIA", "label": "organization_name", "reason": "employer"}]}',
        ],
    )

    result = tool.run_staged_detection_case(
        tool.StagedDetectionRequest(
            case_id="case-1",
            text="Email alice@example.com at NVIDIA.",
            labels=["email", "organization_name"],
            row_index=0,
        ),
        client=llm_client,
        seed_source=tool.SeedSource.rules,
    )

    assert result.status == tool.CaseStatus.completed
    assert result.seed_source == tool.SeedSource.rules
    assert result.phase_usage.seed == {}
    assert result.phase_model_work == tool.PhaseModelWork(seed=False, validation=True, augmentation=True)
    assert result.phase_skip_reasons.seed == "deterministic_rules"
    assert result.phase_skip_reasons.validation is None
    assert result.model_phase_count == 2
    assert result.phase_model_requests == tool.PhaseModelRequests(seed=0, validation=1, augmentation=1)
    assert result.model_request_count == 2
    assert result.seed_suggestion_count == 1
    assert result.seed_entity_count == 1
    assert result.final_label_counts == {"email": 1, "organization_name": 1}
    assert result.artifact.final_source_counts == {"augmenter": 1, "rule": 1}
    assert result.total_usage == {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}
    assert len(llm_client.prompts) == 2


def test_staged_detection_can_add_rules_to_direct_llm_seed_without_validating_rules() -> None:
    tool = load_tool(
        "measurement_staged_detection_probe_rules_plus_direct_seed",
        REPO_ROOT / "tools/measurement/staged_detection_probe.py",
    )
    llm_client = SequencedClient(
        tool,
        [
            '{"entities": [{"value": "NVIDIA", "label": "organization_name", "reason": "employer"}]}',
            '{"decisions": [{"id": "organization_name_27_33", "decision": "keep", "reason": "employer"}]}',
            '{"entities": []}',
        ],
    )

    result = tool.run_staged_detection_case(
        tool.StagedDetectionRequest(
            case_id="case-1",
            text="Email alice@example.com at NVIDIA.",
            labels=["email", "organization_name"],
            row_index=0,
        ),
        client=llm_client,
        seed_source=tool.SeedSource.rules_plus_direct_llm,
    )

    assert result.status == tool.CaseStatus.completed
    assert result.seed_source == tool.SeedSource.rules_plus_direct_llm
    assert result.seed_suggestion_count == 2
    assert result.seed_entity_count == 2
    assert result.validation_candidate_count == 1
    assert result.validation_decision_count == 1
    assert result.final_label_counts == {"email": 1, "organization_name": 1}
    assert result.artifact.final_source_counts == {"direct_seed": 1, "rule": 1}
    assert result.phase_model_requests == tool.PhaseModelRequests(seed=1, validation=1, augmentation=1)
    assert result.model_request_count == 3
    assert '"label":"email"' not in llm_client.prompts[1]
    assert '"label":"organization_name"' in llm_client.prompts[1]


def test_staged_detection_baseline_comparison_skips_rows_without_signature_hashes() -> None:
    tool = load_tool(
        "measurement_staged_detection_probe_missing_baseline_signatures",
        REPO_ROOT / "tools/measurement/staged_detection_probe.py",
    )
    llm_client = SequencedClient(
        tool,
        [
            '{"entities": [{"value": "Alice", "label": "first_name", "reason": "person name"}]}',
            '{"decisions": [{"id": "first_name_0_5", "decision": "keep", "reason": "person name"}]}',
            '{"entities": []}',
        ],
    )
    case = tool.run_staged_detection_case(
        tool.StagedDetectionRequest(
            case_id="case-1",
            text="Alice works remotely.",
            labels=["first_name"],
            row_index=0,
        ),
        client=llm_client,
    )

    compared = tool._case_with_comparison(case, {"row_index": 0, "final_entity_count": 1})

    assert compared.comparison is None


def test_staged_detection_can_trust_rules_without_validation_prompt() -> None:
    tool = load_tool(
        "measurement_staged_detection_probe_rules_trusted_seed",
        REPO_ROOT / "tools/measurement/staged_detection_probe.py",
    )
    llm_client = SequencedClient(tool, ['{"entities": []}'])

    result = tool.run_staged_detection_case(
        tool.StagedDetectionRequest(
            case_id="case-1",
            text=(
                "$ docker run -e DATABASE_URL='postgres://app_user:fakeDbPass123!@db.example.test:5432/app' "
                "-e API_KEY=ghp_FAKEtoken1234567890abcdef myapp:latest\nPassword: fakeLoginPass!"
            ),
            labels=["api_key", "password", "email", "url"],
            row_index=0,
        ),
        client=llm_client,
        seed_source=tool.SeedSource.rules_trusted,
    )

    assert result.status == tool.CaseStatus.completed
    assert result.seed_source == tool.SeedSource.rules_trusted
    assert result.phase_usage.seed == {}
    assert result.phase_usage.validation == {}
    assert result.phase_model_work == tool.PhaseModelWork(seed=False, validation=False, augmentation=True)
    assert result.phase_skip_reasons.seed == "deterministic_rules"
    assert result.phase_skip_reasons.validation == "trusted_rules"
    assert result.phase_skip_reasons.augmentation is None
    assert result.model_phase_count == 1
    assert result.phase_model_requests == tool.PhaseModelRequests(seed=0, validation=0, augmentation=1)
    assert result.model_request_count == 1
    assert result.rule_covered_label_set is True
    assert result.validation_decision_count == 3
    assert result.final_label_counts == {"api_key": 1, "password": 1, "url": 1}
    assert result.artifact.final_source_counts == {"rule": 3}
    assert result.total_usage == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    assert len(llm_client.prompts) == 1


def test_staged_detection_can_skip_augmentation_when_all_labels_are_rule_covered() -> None:
    tool = load_tool(
        "measurement_staged_detection_probe_rules_trusted_no_augment",
        REPO_ROOT / "tools/measurement/staged_detection_probe.py",
    )
    llm_client = SequencedClient(tool, [])

    result = tool.run_staged_detection_case(
        tool.StagedDetectionRequest(
            case_id="case-1",
            text="Email alice@example.com",
            labels=["email"],
            row_index=0,
        ),
        client=llm_client,
        seed_source=tool.SeedSource.rules_trusted,
        skip_augmentation_when_rule_covered=True,
    )

    assert result.status == tool.CaseStatus.completed
    assert result.phase_usage.augmentation == {}
    assert result.phase_model_work == tool.PhaseModelWork(seed=False, validation=False, augmentation=False)
    assert result.phase_skip_reasons == tool.PhaseSkipReasons(
        seed="deterministic_rules",
        validation="trusted_rules",
        augmentation="rule_covered_labels",
    )
    assert result.model_phase_count == 0
    assert result.phase_model_requests == tool.PhaseModelRequests(seed=0, validation=0, augmentation=0)
    assert result.model_request_count == 0
    assert result.rule_covered_label_set is True
    assert result.augmented_suggestion_count == 0
    assert result.final_label_counts == {"email": 1}
    assert result.total_usage == {}
    assert len(llm_client.prompts) == 0


def test_staged_detection_rules_router_short_circuits_rule_covered_labels() -> None:
    tool = load_tool(
        "measurement_staged_detection_probe_rules_router_short_circuit",
        REPO_ROOT / "tools/measurement/staged_detection_probe.py",
    )
    llm_client = SequencedClient(tool, [])

    result = tool.run_staged_detection_case(
        tool.StagedDetectionRequest(
            case_id="case-1",
            text="Email alice@example.com and token ghp_FAKEtoken1234567890abcdef",
            labels=["email", "api_key"],
            row_index=0,
        ),
        client=llm_client,
        seed_source=tool.SeedSource.rules_router,
    )

    assert result.status == tool.CaseStatus.completed
    assert result.seed_source == tool.SeedSource.rules_router
    assert result.phase_model_work == tool.PhaseModelWork(seed=False, validation=False, augmentation=False)
    assert result.phase_skip_reasons == tool.PhaseSkipReasons(
        seed="deterministic_rules",
        validation="trusted_rules",
        augmentation="rule_covered_labels",
    )
    assert result.model_phase_count == 0
    assert result.phase_model_requests == tool.PhaseModelRequests(seed=0, validation=0, augmentation=0)
    assert result.model_request_count == 0
    assert result.elapsed_sec is not None and result.elapsed_sec > 0.0
    assert result.model_elapsed_sec == 0.0
    assert result.rule_covered_label_set is True
    assert result.final_label_counts == {"api_key": 1, "email": 1}
    assert result.artifact.final_source_counts == {"rule": 2}
    assert result.total_usage == {}
    assert len(llm_client.prompts) == 0


def test_staged_detection_rules_router_uses_direct_seed_for_contextual_labels() -> None:
    tool = load_tool(
        "measurement_staged_detection_probe_rules_router_mixed_labels",
        REPO_ROOT / "tools/measurement/staged_detection_probe.py",
    )
    llm_client = SequencedClient(
        tool,
        [
            '{"entities": [{"value": "Alice", "label": "first_name", "reason": "person name"}]}',
            '{"decisions": [{"id": "first_name_0_5", "decision": "keep", "reason": "person name"}]}',
            '{"entities": []}',
        ],
    )

    result = tool.run_staged_detection_case(
        tool.StagedDetectionRequest(
            case_id="case-1",
            text="Alice emails alice@example.com.",
            labels=["email", "first_name"],
            row_index=0,
        ),
        client=llm_client,
        seed_source=tool.SeedSource.rules_router,
    )

    assert result.status == tool.CaseStatus.completed
    assert result.seed_source == tool.SeedSource.rules_router
    assert result.rule_covered_label_set is False
    assert result.phase_model_work == tool.PhaseModelWork(seed=True, validation=True, augmentation=True)
    assert result.phase_skip_reasons == tool.PhaseSkipReasons()
    assert result.phase_model_requests == tool.PhaseModelRequests(seed=1, validation=1, augmentation=1)
    assert result.model_request_count == 3
    assert result.final_label_counts == {"email": 1, "first_name": 1}
    assert result.artifact.final_source_counts == {"direct_seed": 1, "rule": 1}
    assert '"label":"email"' not in llm_client.prompts[1]
    assert '"label":"first_name"' in llm_client.prompts[1]


def test_staged_detection_can_chunk_validation_into_local_excerpts() -> None:
    tool = load_tool(
        "measurement_staged_detection_probe_chunked_validation",
        REPO_ROOT / "tools/measurement/staged_detection_probe.py",
    )
    client = SequencedClient(
        tool,
        [
            (
                '{"entities": ['
                '{"value": "Alice", "label": "first_name", "reason": "name"},'
                '{"value": "Paris", "label": "city", "reason": "city"}'
                "]}"
            ),
            '{"decisions": [{"id": "first_name_0_5", "decision": "keep", "reason": "real name"}]}',
            '{"decisions": [{"id": "city_61_66", "decision": "keep", "reason": "city"}]}',
            '{"entities": []}',
        ],
    )

    result = tool.run_staged_detection_case(
        tool.StagedDetectionRequest(
            case_id="case-1",
            text="Alice works in a very long remote biography before moving to Paris.",
            labels=["first_name", "city"],
            row_index=0,
        ),
        client=client,
        validation_prompt_mode=tool.ValidationPromptMode.chunked_excerpt,
        validation_max_entities_per_call=1,
        validation_excerpt_window_chars=8,
    )

    assert result.status == tool.CaseStatus.completed
    assert result.final_label_counts == {"city": 1, "first_name": 1}
    assert result.phase_model_requests == tool.PhaseModelRequests(seed=1, validation=2, augmentation=1)
    assert result.model_phase_count == 3
    assert result.model_request_count == 4
    assert len(client.prompts) == 4
    assert "Alice" in client.prompts[1]
    assert "Paris" not in client.prompts[1]
    assert "Paris" in client.prompts[2]
