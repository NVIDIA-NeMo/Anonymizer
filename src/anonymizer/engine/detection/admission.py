# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable, Collection, Sequence
from typing import TypeVar

from anonymizer.engine.constants import SOURCE_KIND_AUGMENTER
from anonymizer.engine.detection.postprocess import is_reserved_entity_label
from anonymizer.engine.schemas import (
    AdmissionReason,
    AdmissionResult,
    AdmissionRoute,
    AdmissionStatus,
    DetectionResult,
    EffectiveLabelPolicy,
    Observation,
    ObservationRef,
    RawValidationDecisionSchema,
    ValidationChoice,
    ValidationInventoryEntry,
    ValidationRequestDecisionSets,
    ValidationRequestInventory,
)

_T = TypeVar("_T")
logger = logging.getLogger("anonymizer.detection.admission")


def apply_validation_admission(
    *,
    detection_result: DetectionResult,
    inventories: tuple[ValidationRequestInventory, ...],
    decision_sets: ValidationRequestDecisionSets,
    effective_label_policy: EffectiveLabelPolicy,
) -> tuple[AdmissionResult, ...]:
    """Consume request-scoped validation decisions into terminal outcomes once."""
    if any(inventory.effective_label_policy != effective_label_policy for inventory in inventories):
        raise ValueError("validation request inventories do not match the effective label policy")
    observations_by_ref = {observation.ref: observation for observation in detection_result.observations}
    inventories_by_id = _index_unique(
        inventories,
        key_name="validation inventory request ID",
        key=lambda item: item.request_id,
    )
    responses_by_id = _index_unique(
        decision_sets.responses,
        key_name="validation response request ID",
        key=lambda item: item.request_id,
    )
    if set(responses_by_id) != set(inventories_by_id):
        raise ValueError("validation responses must match dispatched request inventories exactly")

    admissions_by_observation: dict[ObservationRef, AdmissionResult] = {}
    for request_id, inventory in inventories_by_id.items():
        response = responses_by_id[request_id]
        assigned_aliases = {entry.alias for entry in inventory.entries}
        duplicate_aliases = find_duplicate_assigned_aliases(
            decisions=response.decisions,
            assigned_aliases=assigned_aliases,
        )
        if duplicate_aliases:
            raise ValueError(
                f"validation response {request_id!r} contains duplicate assigned aliases {duplicate_aliases}"
            )
        decisions_by_alias = {
            decision.id: decision for decision in response.decisions if decision.id in assigned_aliases
        }
        unknown_alias_count = sum(decision.id not in assigned_aliases for decision in response.decisions)
        if unknown_alias_count:
            logger.warning(
                "validation response %s: ignored %d entries with unknown aliases",
                request_id,
                unknown_alias_count,
            )
        for entry in inventory.entries:
            observation = observations_by_ref.get(entry.observation_ref)
            if observation is None:
                raise ValueError("validation inventory references an unknown observation")
            if entry.observation_ref in admissions_by_observation:
                raise ValueError("an observation was assigned to more than one validation request")
            _validate_inventory_entry(observation, entry)
            decision = decisions_by_alias.get(entry.alias)
            admissions_by_observation[entry.observation_ref] = _admit_llm_observation(
                observation=observation,
                decision=decision,
            )

    results: list[AdmissionResult] = []
    for observation in detection_result.observations:
        if observation.admission_route is AdmissionRoute.direct_acceptance:
            reason = (
                AdmissionReason.augmenter_output
                if observation.source_kind == SOURCE_KIND_AUGMENTER
                else AdmissionReason.direct_acceptance
            )
            results.append(
                AdmissionResult(
                    observation_ref=observation.ref,
                    route=observation.admission_route,
                    status=AdmissionStatus.accepted,
                    effective_label=observation.label,
                    reason=reason,
                )
            )
            continue
        admission = admissions_by_observation.get(observation.ref)
        if admission is None:
            raise ValueError("an LLM-validated observation was not assigned to a validation request")
        results.append(admission)
    return tuple(_apply_label_eligibility(result, effective_label_policy) for result in results)


def _apply_label_eligibility(
    result: AdmissionResult,
    policy: EffectiveLabelPolicy,
) -> AdmissionResult:
    if result.status is AdmissionStatus.rejected:
        return result
    normalized_label = _normalize_label(result.effective_label or "")
    allowed = None if policy.allowed_labels is None else {_normalize_label(label) for label in policy.allowed_labels}
    excluded = {_normalize_label(label) for label in policy.excluded_labels}
    if (
        not normalized_label
        or is_reserved_entity_label(normalized_label)
        or normalized_label in excluded
        or (allowed is not None and normalized_label not in allowed)
    ):
        return AdmissionResult(
            observation_ref=result.observation_ref,
            route=result.route,
            status=AdmissionStatus.rejected,
            reason=AdmissionReason.ineligible_label,
        )
    return result.model_copy(update={"effective_label": normalized_label})


def _normalize_label(label: str) -> str:
    return label.strip().casefold()


def find_duplicate_assigned_aliases(
    *,
    decisions: Sequence[RawValidationDecisionSchema],
    assigned_aliases: Collection[str],
) -> list[str]:
    """Return repeated aliases authorized by a validation request."""
    assigned = set(assigned_aliases)
    counts = Counter(decision.id for decision in decisions if decision.id in assigned)
    return sorted(alias for alias, count in counts.items() if count > 1)


def _admit_llm_observation(
    *,
    observation: Observation,
    decision: RawValidationDecisionSchema | None,
) -> AdmissionResult:
    observation_ref = observation.ref
    if decision is None or decision.decision is None:
        return AdmissionResult(
            observation_ref=observation_ref,
            route=observation.admission_route,
            status=AdmissionStatus.fallback_protected,
            effective_label=observation.label,
            reason=AdmissionReason.missing_decision_fallback,
        )
    choice = decision.decision
    if choice is ValidationChoice.drop:
        return AdmissionResult(
            observation_ref=observation_ref,
            route=observation.admission_route,
            status=AdmissionStatus.rejected,
            reason=AdmissionReason.validator_drop,
        )
    if choice is ValidationChoice.reclass:
        proposed_label = decision.proposed_label.strip()
        if not proposed_label:
            return AdmissionResult(
                observation_ref=observation_ref,
                route=observation.admission_route,
                status=AdmissionStatus.fallback_protected,
                effective_label=observation.label,
                reason=AdmissionReason.unusable_reclass_fallback,
            )
        return AdmissionResult(
            observation_ref=observation_ref,
            route=observation.admission_route,
            status=AdmissionStatus.accepted,
            effective_label=proposed_label,
            reason=AdmissionReason.validator_reclass,
        )
    return AdmissionResult(
        observation_ref=observation_ref,
        route=observation.admission_route,
        status=AdmissionStatus.accepted,
        effective_label=observation.label,
        reason=AdmissionReason.validator_keep,
    )


def _validate_inventory_entry(observation: Observation, entry: ValidationInventoryEntry) -> None:
    if (
        observation.value != entry.value
        or observation.label != entry.label
        or observation.start_position != entry.start_position
        or observation.end_position != entry.end_position
    ):
        raise ValueError("validation inventory entry does not match its authoritative observation")


def _index_unique(
    items: tuple[_T, ...],
    *,
    key_name: str,
    key: Callable[[_T], str],
) -> dict[str, _T]:
    indexed: dict[str, _T] = {}
    for item in items:
        item_key = key(item)
        if item_key in indexed:
            raise ValueError(f"duplicate {key_name}: {item_key!r}")
        indexed[item_key] = item
    return indexed
