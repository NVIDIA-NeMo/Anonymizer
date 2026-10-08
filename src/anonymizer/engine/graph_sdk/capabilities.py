# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Immutable implementation capability declarations for graph preparation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Literal, Never, SupportsIndex, TypeAlias

from anonymizer.graph.workflow import NodeId, OperationSpec

ImplementationEffect: TypeAlias = Literal["local", "external"]
Attribution: TypeAlias = Literal["per_task", "keyed_shared_request", "aggregate_only"]
RequestVisibility: TypeAlias = Literal["none", "before_dispatch", "dispatch_and_settlement"]
PreDispatchControl: TypeAlias = Literal["none", "executor"]
RetryOwner: TypeAlias = Literal["none", "executor", "implementation"]
ErrorReporting: TypeAlias = Literal["typed_terminal", "raises_after_dispatch", "opaque"]
CancellationGuarantee: TypeAlias = Literal["before_dispatch_only", "cooperative_ack", "unobservable"]
SettlementGuarantee: TypeAlias = Literal["synchronous", "explicit_ack", "uncertain_possible"]
UsageCertainty: TypeAlias = Literal["exact", "upper_bound", "unknown"]
ResourceLifetime: TypeAlias = Literal["stateless", "caller_owned", "executor_owned"]


class PreparationCode(str, Enum):
    """Closed rejection codes for pure graph preparation."""

    INVALID_TYPE = "invalid_type"
    INVALID_VALUE = "invalid_value"
    LIMIT_EXCEEDED = "limit_exceeded"
    FOREIGN_OWNER = "foreign_owner"
    DUPLICATE = "duplicate"
    MISSING_INPUT = "missing_input"
    MISSING_STATE = "missing_state"
    UNSUPPORTED_CAPABILITY = "unsupported_capability"
    HARD_BUDGET_INCOMPATIBLE = "hard_budget_incompatible"
    PROTECTION_INELIGIBLE = "protection_ineligible"
    CONTRADICTORY = "contradictory"
    CAPABILITY_CHANGED = "capability_changed"


class PreparationRejected(ValueError):
    """A content-free preparation rejection."""

    __slots__ = ("code",)

    def __init__(self, code: PreparationCode) -> None:
        self.code = code
        super().__init__(code.value)

    def __repr__(self) -> str:
        return f"<{type(self).__name__}>"


def _reject(code: PreparationCode) -> Never:
    raise PreparationRejected(code) from None


class _PrivateValue:
    __slots__ = ()

    def __repr__(self) -> str:
        return f"<{type(self).__name__}>"

    def __copy__(self) -> _PrivateValue:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> _PrivateValue:
        del memo
        return self

    def __reduce_ex__(self, protocol: SupportsIndex, /) -> Never:
        del protocol
        raise TypeError("preparation value serialization is not supported")


def _integer(value: object, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _reject(PreparationCode.INVALID_TYPE)
    if value < (1 if positive else 0):
        _reject(PreparationCode.INVALID_VALUE)
    return value


def _text(value: object) -> str:
    if not isinstance(value, str):
        _reject(PreparationCode.INVALID_TYPE)
    if not value:
        _reject(PreparationCode.INVALID_VALUE)
    return value


@dataclass(frozen=True, slots=True, repr=False)
class ConfigNull(_PrivateValue):
    """A tagged null configuration value."""


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ConfigBoolean(_PrivateValue):
    value: bool

    def __post_init__(self) -> None:
        if not isinstance(self.value, bool):
            _reject(PreparationCode.INVALID_TYPE)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ConfigInteger(_PrivateValue):
    value: int

    def __post_init__(self) -> None:
        if isinstance(self.value, bool) or not isinstance(self.value, int):
            _reject(PreparationCode.INVALID_TYPE)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ConfigText(_PrivateValue):
    value: str

    def __post_init__(self) -> None:
        _text(self.value)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ConfigNumber(_PrivateValue):
    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        if isinstance(self.numerator, bool) or not isinstance(self.numerator, int):
            _reject(PreparationCode.INVALID_TYPE)
        _integer(self.denominator, positive=True)
        divisor = math.gcd(self.numerator, self.denominator)
        numerator = self.numerator // divisor
        denominator = self.denominator // divisor
        if numerator == 0:
            denominator = 1
        object.__setattr__(self, "numerator", numerator)
        object.__setattr__(self, "denominator", denominator)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ConfigSequence(_PrivateValue):
    items: tuple[ConfigValue, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.items, tuple) or any(not _is_config_value(item) for item in self.items):
            _reject(PreparationCode.INVALID_TYPE)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ConfigField(_PrivateValue):
    name: str
    value: ConfigValue

    def __post_init__(self) -> None:
        _text(self.name)
        if not _is_config_value(self.value):
            _reject(PreparationCode.INVALID_TYPE)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class FrozenConfig(_PrivateValue):
    fields: tuple[ConfigField, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.fields, tuple) or any(not isinstance(field, ConfigField) for field in self.fields):
            _reject(PreparationCode.INVALID_TYPE)
        names = [field.name for field in self.fields]
        if len(names) != len(set(names)):
            _reject(PreparationCode.DUPLICATE)
        if names != sorted(names):
            _reject(PreparationCode.INVALID_VALUE)


ConfigValue: TypeAlias = (
    ConfigNull | ConfigBoolean | ConfigInteger | ConfigText | ConfigNumber | ConfigSequence | FrozenConfig
)


def _is_config_value(value: object) -> bool:
    return isinstance(
        value,
        (ConfigNull, ConfigBoolean, ConfigInteger, ConfigText, ConfigNumber, ConfigSequence, FrozenConfig),
    )


def config_size(configuration: FrozenConfig) -> tuple[int, int]:
    """Return maximum depth and atom count for one tagged configuration."""
    if not isinstance(configuration, FrozenConfig):
        _reject(PreparationCode.INVALID_TYPE)
    maximum = 0
    atoms = 0
    pending: list[tuple[ConfigValue, int]] = [(configuration, 1)]
    while pending:
        value, depth = pending.pop()
        maximum = max(maximum, depth)
        atoms += 1
        if isinstance(value, FrozenConfig):
            for field in value.fields:
                atoms += 1
                pending.append((field.value, depth + 1))
        elif isinstance(value, ConfigSequence):
            pending.extend((item, depth + 1) for item in value.items)
    return maximum, atoms


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ImplementationRef(_PrivateValue):
    name: str
    revision: int

    def __post_init__(self) -> None:
        _text(self.name)
        _integer(self.revision, positive=True)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ImplementationCapability(_PrivateValue):
    implementation: ImplementationRef
    operation: OperationSpec
    configuration: FrozenConfig
    effect: ImplementationEffect
    attribution: Attribution
    request_visibility: RequestVisibility
    pre_dispatch_control: PreDispatchControl
    retry_owner: RetryOwner
    error_reporting: ErrorReporting
    cancellation: CancellationGuarantee
    settlement: SettlementGuarantee
    usage: UsageCertainty
    resource_lifetime: ResourceLifetime
    max_physical_requests_per_activation: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.implementation, ImplementationRef)
            or not isinstance(self.operation, OperationSpec)
            or not isinstance(self.configuration, FrozenConfig)
        ):
            _reject(PreparationCode.INVALID_TYPE)
        values = (
            self.effect,
            self.attribution,
            self.request_visibility,
            self.pre_dispatch_control,
            self.retry_owner,
            self.error_reporting,
            self.cancellation,
            self.settlement,
            self.usage,
            self.resource_lifetime,
        )
        if any(not isinstance(value, str) for value in values):
            _reject(PreparationCode.INVALID_TYPE)
        allowed = (
            {"local", "external"},
            {"per_task", "keyed_shared_request", "aggregate_only"},
            {"none", "before_dispatch", "dispatch_and_settlement"},
            {"none", "executor"},
            {"none", "executor", "implementation"},
            {"typed_terminal", "raises_after_dispatch", "opaque"},
            {"before_dispatch_only", "cooperative_ack", "unobservable"},
            {"synchronous", "explicit_ack", "uncertain_possible"},
            {"exact", "upper_bound", "unknown"},
            {"stateless", "caller_owned", "executor_owned"},
        )
        if any(value not in options for value, options in zip(values, allowed, strict=True)):
            _reject(PreparationCode.INVALID_VALUE)
        _integer(self.max_physical_requests_per_activation)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ImplementationSelection(_PrivateValue):
    node: NodeId
    implementation: ImplementationRef
    configuration: FrozenConfig

    def __post_init__(self) -> None:
        if (
            not isinstance(self.node, NodeId)
            or not isinstance(self.implementation, ImplementationRef)
            or not isinstance(self.configuration, FrozenConfig)
        ):
            _reject(PreparationCode.INVALID_TYPE)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class SelectedImplementation(_PrivateValue):
    node: NodeId
    capability: ImplementationCapability

    def __post_init__(self) -> None:
        if not isinstance(self.node, NodeId) or not isinstance(self.capability, ImplementationCapability):
            _reject(PreparationCode.INVALID_TYPE)


def validate_capability(capability: ImplementationCapability) -> None:
    """Validate one complete declaration before selection or recheck."""
    maximum = max((outcome.ceiling.max_model_requests for outcome in capability.operation.outcomes), default=0)
    if capability.max_physical_requests_per_activation > maximum:
        _reject(PreparationCode.CONTRADICTORY)
    if capability.effect == "local":
        valid = (
            capability.max_physical_requests_per_activation == 0
            and capability.request_visibility == "none"
            and capability.pre_dispatch_control == "none"
            and capability.retry_owner == "none"
            and capability.settlement == "synchronous"
            and capability.usage == "exact"
            and capability.resource_lifetime in {"stateless", "caller_owned"}
        )
    else:
        valid = capability.max_physical_requests_per_activation > 0 and capability.request_visibility != "none"
    if not valid:
        _reject(PreparationCode.CONTRADICTORY)


def validate_catalog(capabilities: object, *, maximum: int) -> tuple[ImplementationCapability, ...]:
    """Validate a bounded capability catalog with the D05 precheck ordering."""
    if not isinstance(capabilities, tuple):
        _reject(PreparationCode.INVALID_TYPE)
    if len(capabilities) > maximum:
        _reject(PreparationCode.LIMIT_EXCEEDED)
    if any(not isinstance(capability, ImplementationCapability) for capability in capabilities):
        _reject(PreparationCode.INVALID_TYPE)
    typed = capabilities
    if len(typed) != len(set(typed)):
        _reject(PreparationCode.DUPLICATE)
    for capability in typed:
        validate_capability(capability)
    return typed
