"""Versioned Plane Agent runtime contract.

The classes in this module are the only wire-facing vocabulary in the
runtime adapter.  They deliberately contain references and bounded values,
not Plane database handles, credentials, transports, or Hermes objects.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Union


PROTOCOL = "plane.agent-runtime/v1"
MAX_REFERENCE_LENGTH = 256
MAX_TEXT_LENGTH = 32_000
MAX_PROGRESS_LENGTH = 4_000
MAX_EVENT_BYTES = 16_384
MAX_ACCEPTANCE_CRITERIA = 64
MAX_CONTEXT_REFS = 128
MAX_EAGER_OPERATIONS = 128
MAX_NEW_CONTEXT_EVENT_REFS = 128
MAX_RUN_SNAPSHOT_BYTES = 128 * 1024
MAX_INVOCATION_BYTES = 16 * 1024


class ContractError(ValueError):
    """Raised when an untrusted runtime contract value is invalid."""


class BindingError(ContractError):
    """Raised when a value is bound to a different run or invocation."""


class SequenceError(ContractError):
    """Raised when an event is duplicated inconsistently or out of order."""


class BoundsError(ContractError):
    """Raised when a contract payload exceeds its bounded surface."""


class LeaseError(ContractError):
    """Raised when an invocation lease is not valid for execution."""


JSONScalar = Union[None, bool, int, float, str]
JSONValue = Union[JSONScalar, list, dict]


def _require_text(value: Any, name: str, *, max_length: int = MAX_REFERENCE_LENGTH) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{name} must be a non-empty string")
    if len(value) > max_length:
        raise BoundsError(f"{name} exceeds {max_length} characters")
    return value


def _require_optional_text(
    value: Any, name: str, *, max_length: int = MAX_REFERENCE_LENGTH
) -> str | None:
    if value is None:
        return None
    return _require_text(value, name, max_length=max_length)


def _require_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ContractError(f"{name} must be an integer >= {minimum}")
    return value


def _freeze_json(value: Any, path: str = "payload") -> Any:
    """Deep-freeze JSON data so frozen contracts are actually immutable."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractError(f"{path} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        frozen = {
            _require_text(key, f"{path} key"): _freeze_json(item, f"{path}.{key}")
            for key, item in value.items()
        }
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item, f"{path}[]") for item in value)
    raise ContractError(f"{path} must contain JSON-compatible values")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{name} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise ContractError(f"{name} keys must be strings")
    return value


def _reject_unknown(data: Mapping[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(set(data).difference(allowed))
    if unknown:
        raise ContractError(f"{name} has unknown field(s): {', '.join(unknown)}")


def _sequence(value: Any, name: str, *, maximum: int | None = None) -> tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        raise ContractError(f"{name} must be an array")
    result = tuple(value)
    if maximum is not None and len(result) > maximum:
        raise BoundsError(f"{name} exceeds {maximum} items")
    return result


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def parse_utc_timestamp(value: Any, name: str = "timestamp") -> datetime:
    """Parse the one canonical, timezone-aware UTC timestamp representation."""

    if not isinstance(value, str) or not value:
        raise ContractError(f"{name} must be a non-empty UTC timestamp")
    if not value.endswith("Z"):
        raise ContractError(f"{name} must use canonical UTC 'Z' notation")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ContractError(f"{name} must be a valid ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ContractError(f"{name} must be timezone-aware UTC")
    parsed = parsed.astimezone(timezone.utc)
    canonical = parsed.isoformat(timespec="auto").replace("+00:00", "Z")
    if value != canonical:
        raise ContractError(f"{name} is not canonical: expected {canonical!r}")
    return parsed


def _canonical_utc_timestamp(value: Any, name: str) -> str:
    return parse_utc_timestamp(value, name).isoformat(timespec="auto").replace("+00:00", "Z")


def _check_wire_size(value: Mapping[str, Any], name: str, maximum: int) -> None:
    size = len(_canonical_json(value))
    if size > maximum:
        raise BoundsError(f"{name} exceeds {maximum} canonical JSON bytes (got {size})")


@dataclass(frozen=True)
class AssignmentSnapshot:
    """The immutable assignment slice needed by one run."""

    version: str
    target_ref: str
    objective: str
    acceptance_criteria: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "version", _require_text(self.version, "assignment.version"))
        object.__setattr__(self, "target_ref", _require_text(self.target_ref, "assignment.targetRef"))
        object.__setattr__(
            self,
            "objective",
            _require_text(self.objective, "assignment.objective", max_length=MAX_TEXT_LENGTH),
        )
        criteria = tuple(
            _require_text(item, "assignment.acceptanceCriteria[]", max_length=MAX_TEXT_LENGTH)
            for item in self.acceptance_criteria
        )
        if not criteria:
            raise ContractError("assignment.acceptanceCriteria must not be empty")
        if len(criteria) > MAX_ACCEPTANCE_CRITERIA:
            raise BoundsError(f"assignment.acceptanceCriteria exceeds {MAX_ACCEPTANCE_CRITERIA} items")
        object.__setattr__(self, "acceptance_criteria", criteria)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "targetRef": self.target_ref,
            "objective": self.objective,
            "acceptanceCriteria": list(self.acceptance_criteria),
        }

    @classmethod
    def from_dict(cls, raw: Any) -> "AssignmentSnapshot":
        data = _mapping(raw, "assignment")
        _reject_unknown(data, {"version", "targetRef", "objective", "acceptanceCriteria"}, "assignment")
        return cls(
            version=_require_text(data.get("version"), "assignment.version"),
            target_ref=_require_text(data.get("targetRef"), "assignment.targetRef"),
            objective=_require_text(data.get("objective"), "assignment.objective", max_length=MAX_TEXT_LENGTH),
            acceptance_criteria=tuple(
                _require_text(item, "assignment.acceptanceCriteria[]", max_length=MAX_TEXT_LENGTH)
                for item in _sequence(data.get("acceptanceCriteria"), "assignment.acceptanceCriteria")
            ),
        )


@dataclass(frozen=True)
class VersionedContextRef:
    ref: str
    digest: str
    kind: str = "context"

    def __post_init__(self) -> None:
        object.__setattr__(self, "ref", _require_text(self.ref, "context.ref"))
        object.__setattr__(self, "digest", _require_text(self.digest, "context.digest"))
        object.__setattr__(self, "kind", _require_text(self.kind, "context.kind"))

    def to_dict(self) -> dict[str, str]:
        return {"ref": self.ref, "digest": self.digest, "kind": self.kind}

    @classmethod
    def from_dict(cls, raw: Any) -> "VersionedContextRef":
        data = _mapping(raw, "context")
        _reject_unknown(data, {"ref", "digest", "kind"}, "context")
        return cls(
            ref=_require_text(data.get("ref"), "context.ref"),
            digest=_require_text(data.get("digest"), "context.digest"),
            kind=_require_text(data.get("kind", "context"), "context.kind"),
        )


@dataclass(frozen=True)
class OperationDescriptor:
    operation_ref: str
    descriptor_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "operation_ref", _require_text(self.operation_ref, "operation.ref"))
        object.__setattr__(
            self, "descriptor_digest", _require_text(self.descriptor_digest, "operation.digest")
        )

    def to_dict(self) -> dict[str, str]:
        return {"operationRef": self.operation_ref, "descriptorDigest": self.descriptor_digest}

    @classmethod
    def from_dict(cls, raw: Any) -> "OperationDescriptor":
        data = _mapping(raw, "operation")
        _reject_unknown(data, {"operationRef", "descriptorDigest"}, "operation")
        return cls(
            operation_ref=_require_text(data.get("operationRef"), "operation.ref"),
            descriptor_digest=_require_text(data.get("descriptorDigest"), "operation.digest"),
        )


@dataclass(frozen=True)
class ToolPresentation:
    eager_operations: tuple[OperationDescriptor, ...]
    catalog_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "eager_operations", tuple(self.eager_operations))
        if len(self.eager_operations) > MAX_EAGER_OPERATIONS:
            raise BoundsError(f"toolPresentation.eagerOperations exceeds {MAX_EAGER_OPERATIONS} items")
        object.__setattr__(self, "catalog_digest", _require_text(self.catalog_digest, "toolPresentation.catalogDigest"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "eagerOperations": [operation.to_dict() for operation in self.eager_operations],
            "catalogDigest": self.catalog_digest,
        }

    @classmethod
    def from_dict(cls, raw: Any) -> "ToolPresentation":
        data = _mapping(raw, "toolPresentation")
        _reject_unknown(data, {"eagerOperations", "catalogDigest"}, "toolPresentation")
        return cls(
            eager_operations=tuple(
                OperationDescriptor.from_dict(item)
                for item in _sequence(
                    data.get("eagerOperations"),
                    "toolPresentation.eagerOperations",
                    maximum=MAX_EAGER_OPERATIONS,
                )
            ),
            catalog_digest=_require_text(data.get("catalogDigest"), "toolPresentation.catalogDigest"),
        )


@dataclass(frozen=True)
class RuntimeModelRoute:
    model: str
    route_ref: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "model", _require_text(self.model, "model.model"))
        object.__setattr__(self, "route_ref", _require_text(self.route_ref, "model.routeRef"))

    def to_dict(self) -> dict[str, str]:
        return {"model": self.model, "routeRef": self.route_ref}

    @classmethod
    def from_dict(cls, raw: Any) -> "RuntimeModelRoute":
        data = _mapping(raw, "model")
        _reject_unknown(data, {"model", "routeRef"}, "model")
        return cls(
            model=_require_text(data.get("model"), "model.model"),
            route_ref=_require_text(data.get("routeRef"), "model.routeRef"),
        )


@dataclass(frozen=True)
class RuntimeBudget:
    iterations: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "iterations", _require_int(self.iterations, "budget.iterations"))
        object.__setattr__(self, "input_tokens", _require_int(self.input_tokens, "budget.inputTokens"))
        object.__setattr__(self, "output_tokens", _require_int(self.output_tokens, "budget.outputTokens"))

    def within(self, limit: "RuntimeBudget") -> bool:
        return (
            self.iterations <= limit.iterations
            and self.input_tokens <= limit.input_tokens
            and self.output_tokens <= limit.output_tokens
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "iterations": self.iterations,
            "inputTokens": self.input_tokens,
            "outputTokens": self.output_tokens,
        }

    @classmethod
    def from_dict(cls, raw: Any, name: str = "budget") -> "RuntimeBudget":
        data = _mapping(raw, name)
        _reject_unknown(data, {"iterations", "inputTokens", "outputTokens"}, name)
        return cls(
            iterations=_require_int(data.get("iterations"), f"{name}.iterations"),
            input_tokens=_require_int(data.get("inputTokens"), f"{name}.inputTokens"),
            output_tokens=_require_int(data.get("outputTokens"), f"{name}.outputTokens"),
        )


@dataclass(frozen=True)
class RuntimeBudgetPolicy:
    total: RuntimeBudget

    def to_dict(self) -> dict[str, Any]:
        return {"total": self.total.to_dict()}

    @classmethod
    def from_dict(cls, raw: Any) -> "RuntimeBudgetPolicy":
        data = _mapping(raw, "totalBudgetPolicy")
        _reject_unknown(data, {"total"}, "totalBudgetPolicy")
        return cls(total=RuntimeBudget.from_dict(data.get("total"), "totalBudgetPolicy.total"))


@dataclass(frozen=True)
class ContractDigests:
    snapshot: str
    invocation: str
    event: str
    exit: str

    def __post_init__(self) -> None:
        for name in ("snapshot", "invocation", "event", "exit"):
            object.__setattr__(self, name, _require_text(getattr(self, name), f"contractDigests.{name}"))

    def to_dict(self) -> dict[str, str]:
        return {
            "snapshot": self.snapshot,
            "invocation": self.invocation,
            "event": self.event,
            "exit": self.exit,
        }

    @classmethod
    def from_dict(cls, raw: Any) -> "ContractDigests":
        data = _mapping(raw, "contractDigests")
        _reject_unknown(data, {"snapshot", "invocation", "event", "exit"}, "contractDigests")
        return cls(
            snapshot=_require_text(data.get("snapshot"), "contractDigests.snapshot"),
            invocation=_require_text(data.get("invocation"), "contractDigests.invocation"),
            event=_require_text(data.get("event"), "contractDigests.event"),
            exit=_require_text(data.get("exit"), "contractDigests.exit"),
        )


@dataclass(frozen=True)
class RunSnapshot:
    protocol: str
    run_id: str
    assignment: AssignmentSnapshot
    actor_ref: str
    profile_version: str
    behavioral_prompt: str
    context: tuple[VersionedContextRef, ...]
    tool_presentation: ToolPresentation
    model: RuntimeModelRoute
    total_budget_policy: RuntimeBudgetPolicy
    contract_digests: ContractDigests

    def __post_init__(self) -> None:
        if self.protocol != PROTOCOL:
            raise ContractError(f"unsupported protocol: {self.protocol!r}")
        object.__setattr__(self, "run_id", _require_text(self.run_id, "runId"))
        object.__setattr__(self, "actor_ref", _require_text(self.actor_ref, "actorRef"))
        object.__setattr__(self, "profile_version", _require_text(self.profile_version, "profileVersion"))
        object.__setattr__(
            self,
            "behavioral_prompt",
            _require_text(self.behavioral_prompt, "behavioralPrompt", max_length=MAX_TEXT_LENGTH),
        )
        object.__setattr__(self, "context", tuple(self.context))
        if len(self.context) > MAX_CONTEXT_REFS:
            raise BoundsError(f"context exceeds {MAX_CONTEXT_REFS} items")
        _check_wire_size(self.to_dict(), "runSnapshot", MAX_RUN_SNAPSHOT_BYTES)

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol,
            "runId": self.run_id,
            "assignment": self.assignment.to_dict(),
            "actorRef": self.actor_ref,
            "profileVersion": self.profile_version,
            "behavioralPrompt": self.behavioral_prompt,
            "context": [item.to_dict() for item in self.context],
            "toolPresentation": self.tool_presentation.to_dict(),
            "model": self.model.to_dict(),
            "totalBudgetPolicy": self.total_budget_policy.to_dict(),
            "contractDigests": self.contract_digests.to_dict(),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    def digest(self) -> str:
        return _digest(self.to_dict())

    @classmethod
    def from_dict(cls, raw: Any) -> "RunSnapshot":
        data = _mapping(raw, "runSnapshot")
        _reject_unknown(
            data,
            {
                "protocol",
                "runId",
                "assignment",
                "actorRef",
                "profileVersion",
                "behavioralPrompt",
                "context",
                "toolPresentation",
                "model",
                "totalBudgetPolicy",
                "contractDigests",
            },
            "runSnapshot",
        )
        protocol = data.get("protocol")
        if protocol != PROTOCOL:
            raise ContractError(f"unsupported protocol: {protocol!r}")
        return cls(
            protocol=protocol,
            run_id=_require_text(data.get("runId"), "runId"),
            assignment=AssignmentSnapshot.from_dict(data.get("assignment")),
            actor_ref=_require_text(data.get("actorRef"), "actorRef"),
            profile_version=_require_text(data.get("profileVersion"), "profileVersion"),
            behavioral_prompt=_require_text(
                data.get("behavioralPrompt"), "behavioralPrompt", max_length=MAX_TEXT_LENGTH
            ),
            context=tuple(
                VersionedContextRef.from_dict(item)
                for item in _sequence(data.get("context"), "context", maximum=MAX_CONTEXT_REFS)
            ),
            tool_presentation=ToolPresentation.from_dict(data.get("toolPresentation")),
            model=RuntimeModelRoute.from_dict(data.get("model")),
            total_budget_policy=RuntimeBudgetPolicy.from_dict(data.get("totalBudgetPolicy")),
            contract_digests=ContractDigests.from_dict(data.get("contractDigests")),
        )

    @classmethod
    def from_json(cls, raw: str) -> "RunSnapshot":
        try:
            return cls.from_dict(json.loads(raw))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            if isinstance(exc, ContractError):
                raise
            raise ContractError(f"invalid run snapshot JSON: {exc}") from exc


@dataclass(frozen=True)
class InvocationTrigger:
    kind: str
    event_ref: str | None = None

    def __post_init__(self) -> None:
        allowed = {"initial", "human_input", "recoverable_restart", "continuation"}
        if self.kind not in allowed:
            raise ContractError(f"unsupported invocation trigger: {self.kind!r}")
        if self.kind == "initial" and self.event_ref is not None:
            raise ContractError("initial trigger cannot carry an event reference")
        if self.kind != "initial" and self.event_ref is None:
            raise ContractError(f"{self.kind} trigger requires an event reference")
        object.__setattr__(self, "event_ref", _require_optional_text(self.event_ref, "trigger.eventRef"))

    def to_dict(self) -> dict[str, str]:
        data: dict[str, str] = {"kind": self.kind}
        if self.event_ref is not None:
            data["eventRef"] = self.event_ref
        return data

    @classmethod
    def from_dict(cls, raw: Any) -> "InvocationTrigger":
        data = _mapping(raw, "trigger")
        _reject_unknown(data, {"kind", "eventRef"}, "trigger")
        return cls(
            kind=_require_text(data.get("kind"), "trigger.kind"),
            event_ref=_require_optional_text(data.get("eventRef"), "trigger.eventRef"),
        )


@dataclass(frozen=True)
class RuntimeLease:
    lease_id: str
    holder_ref: str
    expires_at: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "lease_id", _require_text(self.lease_id, "lease.leaseId"))
        object.__setattr__(self, "holder_ref", _require_text(self.holder_ref, "lease.holderRef"))
        object.__setattr__(self, "expires_at", _canonical_utc_timestamp(self.expires_at, "lease.expiresAt"))

    def to_dict(self) -> dict[str, str]:
        return {"leaseId": self.lease_id, "holderRef": self.holder_ref, "expiresAt": self.expires_at}

    @classmethod
    def from_dict(cls, raw: Any) -> "RuntimeLease":
        data = _mapping(raw, "lease")
        _reject_unknown(data, {"leaseId", "holderRef", "expiresAt"}, "lease")
        return cls(
            lease_id=_require_text(data.get("leaseId"), "lease.leaseId"),
            holder_ref=_require_text(data.get("holderRef"), "lease.holderRef"),
            expires_at=_require_text(data.get("expiresAt"), "lease.expiresAt"),
        )


@dataclass(frozen=True)
class InvocationEnvelope:
    protocol: str
    invocation_id: str
    run_id: str
    run_snapshot_digest: str
    trigger: InvocationTrigger
    new_context_event_refs: tuple[str, ...]
    checkpoint_ref: str | None
    remaining_budget: RuntimeBudget
    lease: RuntimeLease
    causation_ref: str
    cancellation_ref: str

    def __post_init__(self) -> None:
        if self.protocol != PROTOCOL:
            raise ContractError(f"unsupported protocol: {self.protocol!r}")
        object.__setattr__(self, "invocation_id", _require_text(self.invocation_id, "invocationId"))
        object.__setattr__(self, "run_id", _require_text(self.run_id, "runId"))
        object.__setattr__(self, "run_snapshot_digest", _require_text(self.run_snapshot_digest, "runSnapshotDigest"))
        object.__setattr__(
            self,
            "new_context_event_refs",
            tuple(_require_text(ref, "newContextEventRefs[]") for ref in self.new_context_event_refs),
        )
        if len(self.new_context_event_refs) > MAX_NEW_CONTEXT_EVENT_REFS:
            raise BoundsError(
                f"newContextEventRefs exceeds {MAX_NEW_CONTEXT_EVENT_REFS} items"
            )
        if len(set(self.new_context_event_refs)) != len(self.new_context_event_refs):
            raise ContractError("newContextEventRefs must be unique")
        if self.trigger.kind == "initial" and self.checkpoint_ref is not None:
            raise ContractError("initial trigger cannot carry a checkpoint reference")
        object.__setattr__(self, "checkpoint_ref", _require_optional_text(self.checkpoint_ref, "checkpointRef"))
        object.__setattr__(self, "causation_ref", _require_text(self.causation_ref, "causationRef"))
        object.__setattr__(self, "cancellation_ref", _require_text(self.cancellation_ref, "cancellationRef"))
        _check_wire_size(self.to_dict(), "invocation", MAX_INVOCATION_BYTES)

    def to_dict(self) -> dict[str, Any]:
        data = {
            "protocol": self.protocol,
            "invocationId": self.invocation_id,
            "runId": self.run_id,
            "runSnapshotDigest": self.run_snapshot_digest,
            "trigger": self.trigger.to_dict(),
            "newContextEventRefs": list(self.new_context_event_refs),
            "checkpointRef": self.checkpoint_ref,
            "remainingBudget": self.remaining_budget.to_dict(),
            "lease": self.lease.to_dict(),
            "causationRef": self.causation_ref,
            "cancellationRef": self.cancellation_ref,
        }
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    @classmethod
    def from_dict(cls, raw: Any) -> "InvocationEnvelope":
        data = _mapping(raw, "invocation")
        _reject_unknown(
            data,
            {
                "protocol",
                "invocationId",
                "runId",
                "runSnapshotDigest",
                "trigger",
                "newContextEventRefs",
                "checkpointRef",
                "remainingBudget",
                "lease",
                "causationRef",
                "cancellationRef",
            },
            "invocation",
        )
        protocol = data.get("protocol")
        if protocol != PROTOCOL:
            raise ContractError(f"unsupported protocol: {protocol!r}")
        return cls(
            protocol=protocol,
            invocation_id=_require_text(data.get("invocationId"), "invocationId"),
            run_id=_require_text(data.get("runId"), "runId"),
            run_snapshot_digest=_require_text(data.get("runSnapshotDigest"), "runSnapshotDigest"),
            trigger=InvocationTrigger.from_dict(data.get("trigger")),
            new_context_event_refs=tuple(
                _require_text(ref, "newContextEventRefs[]")
                for ref in _sequence(
                    data.get("newContextEventRefs"),
                    "newContextEventRefs",
                    maximum=MAX_NEW_CONTEXT_EVENT_REFS,
                )
            ),
            checkpoint_ref=_require_optional_text(data.get("checkpointRef"), "checkpointRef"),
            remaining_budget=RuntimeBudget.from_dict(data.get("remainingBudget"), "remainingBudget"),
            lease=RuntimeLease.from_dict(data.get("lease")),
            causation_ref=_require_text(data.get("causationRef"), "causationRef"),
            cancellation_ref=_require_text(data.get("cancellationRef"), "cancellationRef"),
        )

    @classmethod
    def from_json(cls, raw: str) -> "InvocationEnvelope":
        try:
            return cls.from_dict(json.loads(raw))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            if isinstance(exc, ContractError):
                raise
            raise ContractError(f"invalid invocation JSON: {exc}") from exc


@dataclass(frozen=True)
class RuntimeFailure:
    code: str
    message: str
    retryable: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", _require_text(self.code, "failure.code"))
        object.__setattr__(self, "message", _require_text(self.message, "failure.message", max_length=MAX_TEXT_LENGTH))
        if not isinstance(self.retryable, bool):
            raise ContractError("failure.retryable must be a boolean")

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "retryable": self.retryable}

    @classmethod
    def from_dict(cls, raw: Any) -> "RuntimeFailure":
        data = _mapping(raw, "failure")
        _reject_unknown(data, {"code", "message", "retryable"}, "failure")
        return cls(
            code=_require_text(data.get("code"), "failure.code"),
            message=_require_text(data.get("message"), "failure.message", max_length=MAX_TEXT_LENGTH),
            retryable=data.get("retryable"),
        )


@dataclass(frozen=True)
class ProgressObserved:
    message: str
    payload: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "message",
            _require_text(self.message, "progress.message", max_length=MAX_PROGRESS_LENGTH),
        )
        frozen = _freeze_json(self.payload, "progress.payload")
        if not isinstance(frozen, Mapping):
            raise ContractError("progress.payload must be an object")
        object.__setattr__(self, "payload", frozen)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "progress", "message": self.message, "payload": _thaw_json(self.payload)}


@dataclass(frozen=True)
class TranscriptObserved:
    transcript_ref: str
    text: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "transcript_ref", _require_text(self.transcript_ref, "transcript.ref"))
        object.__setattr__(self, "text", _require_text(self.text, "transcript.text", max_length=MAX_TEXT_LENGTH))

    def to_dict(self) -> dict[str, str]:
        return {"kind": "transcript", "transcriptRef": self.transcript_ref, "text": self.text}


@dataclass(frozen=True)
class ConversationPublicationObserved:
    transcript_ref: str
    publication_ref: str
    receipt_ref: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "transcript_ref", _require_text(self.transcript_ref, "publication.transcriptRef"))
        object.__setattr__(self, "publication_ref", _require_text(self.publication_ref, "publication.publicationRef"))
        object.__setattr__(self, "receipt_ref", _require_text(self.receipt_ref, "publication.receiptRef"))

    def to_dict(self) -> dict[str, str]:
        return {
            "kind": "conversation_publication",
            "transcriptRef": self.transcript_ref,
            "publicationRef": self.publication_ref,
            "receiptRef": self.receipt_ref,
        }


@dataclass(frozen=True)
class InputRequestObserved:
    request_ref: str
    prompt: str
    receipt_ref: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_ref", _require_text(self.request_ref, "inputRequest.requestRef"))
        object.__setattr__(
            self,
            "prompt",
            _require_text(self.prompt, "inputRequest.prompt", max_length=MAX_TEXT_LENGTH),
        )
        object.__setattr__(self, "receipt_ref", _require_text(self.receipt_ref, "inputRequest.receiptRef"))

    def to_dict(self) -> dict[str, str]:
        return {
            "kind": "input_request",
            "requestRef": self.request_ref,
            "prompt": self.prompt,
            "receiptRef": self.receipt_ref,
        }


@dataclass(frozen=True)
class ArtifactObserved:
    artifact_ref: str
    digest: str
    receipt_ref: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifact_ref", _require_text(self.artifact_ref, "artifact.artifactRef"))
        object.__setattr__(self, "digest", _require_text(self.digest, "artifact.digest"))
        object.__setattr__(self, "receipt_ref", _require_text(self.receipt_ref, "artifact.receiptRef"))

    def to_dict(self) -> dict[str, str]:
        return {
            "kind": "artifact",
            "artifactRef": self.artifact_ref,
            "digest": self.digest,
            "receiptRef": self.receipt_ref,
        }


@dataclass(frozen=True)
class UsageObserved:
    usage: RuntimeBudget

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "usage", "usage": self.usage.to_dict()}


@dataclass(frozen=True)
class OutcomeSubmissionObserved:
    submission_ref: str
    receipt_ref: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "submission_ref", _require_text(self.submission_ref, "submission.submissionRef"))
        object.__setattr__(self, "receipt_ref", _require_text(self.receipt_ref, "submission.receiptRef"))

    def to_dict(self) -> dict[str, str]:
        return {
            "kind": "outcome_submission",
            "submissionRef": self.submission_ref,
            "receiptRef": self.receipt_ref,
        }


@dataclass(frozen=True)
class FailureObserved:
    code: str
    message: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", _require_text(self.code, "failure.code"))
        object.__setattr__(self, "message", _require_text(self.message, "failure.message", max_length=MAX_TEXT_LENGTH))

    def to_dict(self) -> dict[str, str]:
        return {"kind": "failure", "code": self.code, "message": self.message}


@dataclass(frozen=True)
class BlockerObserved:
    code: str
    message: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", _require_text(self.code, "blocker.code"))
        object.__setattr__(self, "message", _require_text(self.message, "blocker.message", max_length=MAX_TEXT_LENGTH))

    def to_dict(self) -> dict[str, str]:
        return {"kind": "blocker", "code": self.code, "message": self.message}


EventBody = Union[
    ProgressObserved,
    TranscriptObserved,
    ConversationPublicationObserved,
    InputRequestObserved,
    ArtifactObserved,
    UsageObserved,
    OutcomeSubmissionObserved,
    FailureObserved,
    BlockerObserved,
]


_EVENT_BODY_TYPES: dict[str, type[EventBody]] = {
    "progress": ProgressObserved,
    "transcript": TranscriptObserved,
    "conversation_publication": ConversationPublicationObserved,
    "input_request": InputRequestObserved,
    "artifact": ArtifactObserved,
    "usage": UsageObserved,
    "outcome_submission": OutcomeSubmissionObserved,
    "failure": FailureObserved,
    "blocker": BlockerObserved,
}


def _event_body_from_dict(raw: Any) -> EventBody:
    data = _mapping(raw, "event.body")
    kind = _require_text(data.get("kind"), "event.body.kind")
    try:
        body_type = _EVENT_BODY_TYPES[kind]
    except KeyError as exc:
        raise ContractError(f"unsupported event body kind: {kind!r}") from exc
    if kind == "progress":
        _reject_unknown(data, {"kind", "message", "payload"}, "event.body.progress")
        return ProgressObserved(data.get("message"), data.get("payload", {}))
    if kind == "transcript":
        _reject_unknown(data, {"kind", "transcriptRef", "text"}, "event.body.transcript")
        return TranscriptObserved(data.get("transcriptRef"), data.get("text"))
    if kind == "conversation_publication":
        _reject_unknown(
            data,
            {"kind", "transcriptRef", "publicationRef", "receiptRef"},
            "event.body.conversation_publication",
        )
        return ConversationPublicationObserved(
            data.get("transcriptRef"), data.get("publicationRef"), data.get("receiptRef")
        )
    if kind == "input_request":
        _reject_unknown(data, {"kind", "requestRef", "prompt", "receiptRef"}, "event.body.input_request")
        return InputRequestObserved(data.get("requestRef"), data.get("prompt"), data.get("receiptRef"))
    if kind == "artifact":
        _reject_unknown(
            data,
            {"kind", "artifactRef", "digest", "receiptRef"},
            "event.body.artifact",
        )
        return ArtifactObserved(data.get("artifactRef"), data.get("digest"), data.get("receiptRef"))
    if kind == "usage":
        _reject_unknown(data, {"kind", "usage"}, "event.body.usage")
        return UsageObserved(RuntimeBudget.from_dict(data.get("usage"), "event.body.usage"))
    if kind == "outcome_submission":
        _reject_unknown(
            data,
            {"kind", "submissionRef", "receiptRef"},
            "event.body.outcome_submission",
        )
        return OutcomeSubmissionObserved(data.get("submissionRef"), data.get("receiptRef"))
    if kind == "failure":
        _reject_unknown(data, {"kind", "code", "message"}, "event.body.failure")
        return FailureObserved(data.get("code"), data.get("message"))
    if kind == "blocker":
        _reject_unknown(data, {"kind", "code", "message"}, "event.body.blocker")
        return BlockerObserved(data.get("code"), data.get("message"))
    raise AssertionError(f"unhandled event body type: {body_type}")


@dataclass(frozen=True)
class RuntimeEvent:
    protocol: str
    run_id: str
    invocation_id: str
    sequence: int
    event_id: str
    correlation_ref: str
    idempotency_key: str
    body: EventBody

    def __post_init__(self) -> None:
        if self.protocol != PROTOCOL:
            raise ContractError(f"unsupported protocol: {self.protocol!r}")
        object.__setattr__(self, "run_id", _require_text(self.run_id, "runId"))
        object.__setattr__(self, "invocation_id", _require_text(self.invocation_id, "invocationId"))
        object.__setattr__(self, "sequence", _require_int(self.sequence, "sequence", minimum=1))
        object.__setattr__(self, "event_id", _require_text(self.event_id, "eventId"))
        object.__setattr__(self, "correlation_ref", _require_text(self.correlation_ref, "correlationRef"))
        object.__setattr__(self, "idempotency_key", _require_text(self.idempotency_key, "idempotencyKey"))
        if not isinstance(self.body, tuple(_EVENT_BODY_TYPES.values())):
            raise ContractError("event.body has an unsupported type")
        payload_size = len(_canonical_json(self.to_dict()))
        if payload_size > MAX_EVENT_BYTES:
            raise BoundsError(f"event exceeds {MAX_EVENT_BYTES} bytes")

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol,
            "runId": self.run_id,
            "invocationId": self.invocation_id,
            "sequence": self.sequence,
            "eventId": self.event_id,
            "correlationRef": self.correlation_ref,
            "idempotencyKey": self.idempotency_key,
            "body": self.body.to_dict(),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    @classmethod
    def from_dict(cls, raw: Any) -> "RuntimeEvent":
        data = _mapping(raw, "event")
        _reject_unknown(
            data,
            {
                "protocol",
                "runId",
                "invocationId",
                "sequence",
                "eventId",
                "correlationRef",
                "idempotencyKey",
                "body",
            },
            "event",
        )
        protocol = data.get("protocol")
        if protocol != PROTOCOL:
            raise ContractError(f"unsupported protocol: {protocol!r}")
        return cls(
            protocol=protocol,
            run_id=_require_text(data.get("runId"), "runId"),
            invocation_id=_require_text(data.get("invocationId"), "invocationId"),
            sequence=_require_int(data.get("sequence"), "sequence", minimum=1),
            event_id=_require_text(data.get("eventId"), "eventId"),
            correlation_ref=_require_text(data.get("correlationRef"), "correlationRef"),
            idempotency_key=_require_text(data.get("idempotencyKey"), "idempotencyKey"),
            body=_event_body_from_dict(data.get("body")),
        )

    @classmethod
    def from_json(cls, raw: str) -> "RuntimeEvent":
        try:
            return cls.from_dict(json.loads(raw))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            if isinstance(exc, ContractError):
                raise
            raise ContractError(f"invalid runtime event JSON: {exc}") from exc


@dataclass(frozen=True)
class RuntimeExit:
    kind: str
    final_sequence: int
    failure: RuntimeFailure | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"completed", "waiting_for_input", "failed", "blocked", "cancelled"}:
            raise ContractError(f"unsupported exit kind: {self.kind!r}")
        object.__setattr__(self, "final_sequence", _require_int(self.final_sequence, "finalSequence", minimum=0))
        if self.kind in {"failed", "blocked"} and self.failure is None:
            raise ContractError(f"{self.kind} exit requires failure details")
        if self.kind not in {"failed", "blocked"} and self.failure is not None:
            raise ContractError(f"{self.kind} exit cannot carry failure details")

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": PROTOCOL,
            "kind": self.kind,
            "finalSequence": self.final_sequence,
            "failure": self.failure.to_dict() if self.failure is not None else None,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    @classmethod
    def from_dict(cls, raw: Any) -> "RuntimeExit":
        data = _mapping(raw, "exit")
        _reject_unknown(data, {"protocol", "kind", "finalSequence", "failure"}, "exit")
        if data.get("protocol") != PROTOCOL:
            raise ContractError(f"unsupported protocol: {data.get('protocol')!r}")
        failure = data.get("failure")
        return cls(
            kind=_require_text(data.get("kind"), "exit.kind"),
            final_sequence=_require_int(data.get("finalSequence"), "exit.finalSequence", minimum=0),
            failure=RuntimeFailure.from_dict(failure) if failure is not None else None,
        )

    @classmethod
    def from_json(cls, raw: str) -> "RuntimeExit":
        try:
            return cls.from_dict(json.loads(raw))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            if isinstance(exc, ContractError):
                raise
            raise ContractError(f"invalid runtime exit JSON: {exc}") from exc


@dataclass(frozen=True)
class PublicationReceipt:
    publication_ref: str
    receipt_ref: str
    transcript_ref: str
    run_id: str | None = None
    invocation_id: str | None = None
    idempotency_key: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "publication_ref", _require_text(self.publication_ref, "receipt.publicationRef"))
        object.__setattr__(self, "receipt_ref", _require_text(self.receipt_ref, "receipt.receiptRef"))
        object.__setattr__(self, "transcript_ref", _require_text(self.transcript_ref, "receipt.transcriptRef"))
        object.__setattr__(self, "run_id", _require_optional_text(self.run_id, "receipt.runId"))
        object.__setattr__(self, "invocation_id", _require_optional_text(self.invocation_id, "receipt.invocationId"))
        object.__setattr__(self, "idempotency_key", _require_optional_text(self.idempotency_key, "receipt.idempotencyKey"))


@dataclass(frozen=True)
class ProductReceipt:
    """Receipt returned by one narrow, host-authorized product operation."""

    resource_ref: str
    receipt_ref: str
    run_id: str
    invocation_id: str
    idempotency_key: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "resource_ref", _require_text(self.resource_ref, "receipt.resourceRef"))
        object.__setattr__(self, "receipt_ref", _require_text(self.receipt_ref, "receipt.receiptRef"))
        object.__setattr__(self, "run_id", _require_text(self.run_id, "receipt.runId"))
        object.__setattr__(self, "invocation_id", _require_text(self.invocation_id, "receipt.invocationId"))
        object.__setattr__(self, "idempotency_key", _require_text(self.idempotency_key, "receipt.idempotencyKey"))


TERMINAL_KINDS = frozenset(
    {"completed", "waiting_for_input", "failed", "blocked", "cancelled"}
)


@dataclass(frozen=True)
class TerminalProposal:
    """A terminal observation submitted to Plane for lifecycle reconciliation.

    This is deliberately not a Plane state record.  It is the small value sent
    across the injected ``TerminalReconciliationPort`` seam.  The port owns
    legal-transition decisions and durable idempotency; Hermes only supplies
    bounded evidence and preserves the deterministic idempotency key.
    """

    run_id: str
    invocation_id: str
    kind: str
    final_sequence: int
    evidence_event_ids: tuple[str, ...] = ()
    evidence_receipt_refs: tuple[str, ...] = ()
    failure: RuntimeFailure | None = None
    source: str = "runtime"

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _require_text(self.run_id, "terminal.runId"))
        object.__setattr__(
            self, "invocation_id", _require_text(self.invocation_id, "terminal.invocationId")
        )
        if self.kind not in TERMINAL_KINDS:
            raise ContractError(f"unsupported terminal proposal kind: {self.kind!r}")
        object.__setattr__(
            self,
            "final_sequence",
            _require_int(self.final_sequence, "terminal.finalSequence", minimum=0),
        )
        event_ids = tuple(
            _require_text(value, "terminal.evidenceEventIds[]") for value in self.evidence_event_ids
        )
        receipt_refs = tuple(
            _require_text(value, "terminal.evidenceReceiptRefs[]")
            for value in self.evidence_receipt_refs
        )
        if len(event_ids) > MAX_NEW_CONTEXT_EVENT_REFS:
            raise BoundsError(
                f"terminal.evidenceEventIds exceeds {MAX_NEW_CONTEXT_EVENT_REFS} items"
            )
        if len(receipt_refs) > MAX_NEW_CONTEXT_EVENT_REFS:
            raise BoundsError(
                f"terminal.evidenceReceiptRefs exceeds {MAX_NEW_CONTEXT_EVENT_REFS} items"
            )
        if len(set(event_ids)) != len(event_ids):
            raise ContractError("terminal.evidenceEventIds must be unique")
        if len(set(receipt_refs)) != len(receipt_refs):
            raise ContractError("terminal.evidenceReceiptRefs must be unique")
        object.__setattr__(self, "evidence_event_ids", event_ids)
        object.__setattr__(self, "evidence_receipt_refs", receipt_refs)
        if self.kind in {"failed", "blocked"} and self.failure is None:
            raise ContractError(f"{self.kind} terminal proposal requires failure details")
        if self.kind not in {"failed", "blocked"} and self.failure is not None:
            raise ContractError(f"{self.kind} terminal proposal cannot carry failure details")
        if self.source not in {"runtime", "supervisor"}:
            raise ContractError(f"unsupported terminal proposal source: {self.source!r}")
        if self.source == "supervisor" and (
            self.kind != "failed"
            or self.failure is None
            or self.failure.code != "process_died"
        ):
            raise ContractError("supervisor terminal proposals must synthesize process death")
        object.__setattr__(self, "source", self.source)

    @property
    def idempotency_key(self) -> str:
        return (
            f"terminal:{self.source}:{self.run_id}:{self.invocation_id}:"
            f"{self.kind}:{self.final_sequence}"
        )


@dataclass(frozen=True)
class TerminalReconciliationReceipt:
    """The host/Plane result for one terminal proposal."""

    receipt_ref: str
    audit_ref: str
    run_id: str
    invocation_id: str
    kind: str
    idempotency_key: str
    accepted: bool
    legal_transition: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "receipt_ref", _require_text(self.receipt_ref, "terminalReceipt.receiptRef"))
        object.__setattr__(self, "audit_ref", _require_text(self.audit_ref, "terminalReceipt.auditRef"))
        object.__setattr__(self, "run_id", _require_text(self.run_id, "terminalReceipt.runId"))
        object.__setattr__(
            self,
            "invocation_id",
            _require_text(self.invocation_id, "terminalReceipt.invocationId"),
        )
        if self.kind not in TERMINAL_KINDS:
            raise ContractError(f"unsupported terminal receipt kind: {self.kind!r}")
        object.__setattr__(
            self,
            "idempotency_key",
            _require_text(self.idempotency_key, "terminalReceipt.idempotencyKey"),
        )
        if not isinstance(self.accepted, bool) or not isinstance(self.legal_transition, bool):
            raise ContractError("terminal receipt decisions must be booleans")
