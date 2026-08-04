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
MAX_TERMINAL_EVIDENCE = 128
MAX_TERMINAL_PRODUCT_RECEIPTS = 256
MAX_EVENTS_PER_INVOCATION = 512
MAX_EVENT_STREAM_BYTES = 256 * 1024
MAX_OPTIONAL_EVENT_TAIL = 256
MAX_TRANSCRIPT_OBSERVATIONS = 64
MAX_TRANSCRIPT_BYTES = 64 * 1024
MAX_ARTIFACT_PROPOSALS = 128
MAX_ARTIFACT_PROPOSAL_BYTES = 64 * 1024
MAX_INPUT_PROPOSALS = 16
MAX_INPUT_PROPOSAL_BYTES = 64 * 1024
MAX_OUTCOME_PROPOSALS = 1
MAX_OUTCOME_PROPOSAL_BYTES = 64 * 1024
MAX_MESSAGE_PROPOSALS = 64
MAX_MESSAGE_PROPOSAL_BYTES = 64 * 1024
MAX_TERMINAL_PROPOSAL_BYTES = 128 * 1024
MAX_TERMINAL_RECEIPT_BYTES = 128 * 1024
MAX_TERMINAL_PROOFS = 5


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


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    """Return the canonical UTF-8 representation used by every wire bound."""

    return _canonical_json(value)


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
    size = len(canonical_json_bytes(value))
    if size > maximum:
        raise BoundsError(f"{name} exceeds {maximum} canonical JSON bytes (got {size})")


def _check_raw_wire_size(raw: Any, name: str, maximum: int) -> str:
    if not isinstance(raw, str):
        raise ContractError(f"{name} JSON must be a string")
    try:
        size = len(raw.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise ContractError(f"{name} JSON must be valid UTF-8 text") from exc
    if size > maximum:
        raise BoundsError(f"{name} JSON exceeds {maximum} UTF-8 bytes (got {size})")
    return raw


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
    workspace_ref: str
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
        object.__setattr__(self, "workspace_ref", _require_text(self.workspace_ref, "workspaceRef"))
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
            "workspaceRef": self.workspace_ref,
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
                "workspaceRef",
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
            workspace_ref=_require_text(data.get("workspaceRef"), "workspaceRef"),
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
            raw = _check_raw_wire_size(raw, "runSnapshot", MAX_RUN_SNAPSHOT_BYTES)
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
            raw = _check_raw_wire_size(raw, "invocation", MAX_INVOCATION_BYTES)
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
class MessageProposalObserved:
    """A bounded message proposal; it is never a product mutation."""

    message_ref: str
    transcript_ref: str
    content: str
    proposal_receipt_ref: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "message_ref", _require_text(self.message_ref, "message.messageRef"))
        object.__setattr__(self, "transcript_ref", _require_text(self.transcript_ref, "message.transcriptRef"))
        object.__setattr__(
            self,
            "content",
            _require_text(self.content, "message.content", max_length=MAX_TEXT_LENGTH),
        )
        object.__setattr__(
            self,
            "proposal_receipt_ref",
            _require_text(self.proposal_receipt_ref, "message.proposalReceiptRef"),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "kind": "message_proposal",
            "messageRef": self.message_ref,
            "transcriptRef": self.transcript_ref,
            "content": self.content,
            "proposalReceiptRef": self.proposal_receipt_ref,
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
    content: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "submission_ref", _require_text(self.submission_ref, "submission.submissionRef"))
        object.__setattr__(self, "receipt_ref", _require_text(self.receipt_ref, "submission.receiptRef"))
        object.__setattr__(
            self,
            "content",
            _require_text(self.content, "submission.content", max_length=MAX_TEXT_LENGTH),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "kind": "outcome_submission",
            "submissionRef": self.submission_ref,
            "receiptRef": self.receipt_ref,
            "content": self.content,
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
    MessageProposalObserved,
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
    "message_proposal": MessageProposalObserved,
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
    if kind == "message_proposal":
        _reject_unknown(
            data,
            {"kind", "messageRef", "transcriptRef", "content", "proposalReceiptRef"},
            "event.body.message_proposal",
        )
        return MessageProposalObserved(
            data.get("messageRef"),
            data.get("transcriptRef"),
            data.get("content"),
            data.get("proposalReceiptRef"),
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
            {"kind", "submissionRef", "receiptRef", "content"},
            "event.body.outcome_submission",
        )
        return OutcomeSubmissionObserved(
            data.get("submissionRef"), data.get("receiptRef"), data.get("content")
        )
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
            raw = _check_raw_wire_size(raw, "runtimeEvent", MAX_EVENT_BYTES)
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


TERMINAL_KINDS = frozenset(
    {"completed", "waiting_for_input", "failed", "blocked", "cancelled"}
)

TERMINAL_PROOF_KINDS = frozenset(
    {"operation_attempt", "application", "gateway", "audit", "product_event"}
)


@dataclass(frozen=True)
class TerminalProof:
    """One fully bound proof for a terminal reconciliation boundary."""

    proof_kind: str
    proof_ref: str
    resource_ref: str
    run_id: str
    invocation_id: str
    actor_ref: str
    workspace_ref: str
    snapshot_digest: str
    terminal_slot: str
    terminal_kind: str
    proposal_digest: str

    def __post_init__(self) -> None:
        if self.proof_kind not in TERMINAL_PROOF_KINDS:
            raise ContractError(f"unsupported terminal proof kind: {self.proof_kind!r}")
        for name in (
            "proof_ref",
            "resource_ref",
            "run_id",
            "invocation_id",
            "actor_ref",
            "workspace_ref",
            "snapshot_digest",
            "terminal_slot",
            "proposal_digest",
        ):
            object.__setattr__(
                self,
                name,
                _require_text(getattr(self, name), f"terminalProof.{name}"),
            )
        if self.terminal_kind not in TERMINAL_KINDS:
            raise ContractError(f"unsupported terminal proof terminal kind: {self.terminal_kind!r}")
        prefixes = {
            "operation_attempt": "operation",
            "application": "application",
            "gateway": "gateway",
            "audit": "audit",
            "product_event": "product-event",
        }
        if self.resource_ref != f"{prefixes[self.proof_kind]}:{self.terminal_slot}":
            raise BindingError("terminal proof resource is not bound to its proof kind and slot")
        if self.proof_ref != f"terminal-proof:{self.proof_kind}:{self.terminal_slot}":
            raise BindingError("terminal proof identity is not bound to its proof kind and slot")

    def to_dict(self) -> dict[str, str]:
        return {
            "proofKind": self.proof_kind,
            "proofRef": self.proof_ref,
            "resourceRef": self.resource_ref,
            "runId": self.run_id,
            "invocationId": self.invocation_id,
            "actorRef": self.actor_ref,
            "workspaceRef": self.workspace_ref,
            "snapshotDigest": self.snapshot_digest,
            "terminalSlot": self.terminal_slot,
            "terminalKind": self.terminal_kind,
            "proposalDigest": self.proposal_digest,
        }

    @classmethod
    def from_dict(cls, raw: Any) -> "TerminalProof":
        data = _mapping(raw, "terminalProof")
        _reject_unknown(
            data,
            {
                "proofKind",
                "proofRef",
                "resourceRef",
                "runId",
                "invocationId",
                "actorRef",
                "workspaceRef",
                "snapshotDigest",
                "terminalSlot",
                "terminalKind",
                "proposalDigest",
            },
            "terminalProof",
        )
        return cls(
            proof_kind=_require_text(data.get("proofKind"), "terminalProof.proofKind"),
            proof_ref=_require_text(data.get("proofRef"), "terminalProof.proofRef"),
            resource_ref=_require_text(data.get("resourceRef"), "terminalProof.resourceRef"),
            run_id=_require_text(data.get("runId"), "terminalProof.runId"),
            invocation_id=_require_text(data.get("invocationId"), "terminalProof.invocationId"),
            actor_ref=_require_text(data.get("actorRef"), "terminalProof.actorRef"),
            workspace_ref=_require_text(data.get("workspaceRef"), "terminalProof.workspaceRef"),
            snapshot_digest=_require_text(
                data.get("snapshotDigest"), "terminalProof.snapshotDigest"
            ),
            terminal_slot=_require_text(data.get("terminalSlot"), "terminalProof.terminalSlot"),
            terminal_kind=_require_text(data.get("terminalKind"), "terminalProof.terminalKind"),
            proposal_digest=_require_text(
                data.get("proposalDigest"), "terminalProof.proposalDigest"
            ),
        )


@dataclass(frozen=True)
class CancellationAuthorityReceipt:
    """Host-authoritative, invocation-bound cancellation receipt."""

    resource_ref: str
    receipt_ref: str
    run_id: str
    invocation_id: str
    actor_ref: str
    workspace_ref: str
    snapshot_digest: str
    idempotency_key: str
    gateway_receipt_ref: str
    audit_ref: str
    kind: str = "cancellation"

    def __post_init__(self) -> None:
        for name in (
            "resource_ref",
            "receipt_ref",
            "run_id",
            "invocation_id",
            "actor_ref",
            "workspace_ref",
            "snapshot_digest",
            "idempotency_key",
            "gateway_receipt_ref",
            "audit_ref",
        ):
            object.__setattr__(
                self,
                name,
                _require_text(getattr(self, name), f"cancellationReceipt.{name}"),
            )
        if self.kind != "cancellation":
            raise ContractError("cancellation authority receipt must have cancellation kind")

    def to_dict(self) -> dict[str, str]:
        return {
            "resourceRef": self.resource_ref,
            "receiptRef": self.receipt_ref,
            "runId": self.run_id,
            "invocationId": self.invocation_id,
            "actorRef": self.actor_ref,
            "workspaceRef": self.workspace_ref,
            "snapshotDigest": self.snapshot_digest,
            "idempotencyKey": self.idempotency_key,
            "gatewayReceiptRef": self.gateway_receipt_ref,
            "auditRef": self.audit_ref,
            "kind": self.kind,
        }

    @classmethod
    def from_dict(cls, raw: Any) -> "CancellationAuthorityReceipt":
        data = _mapping(raw, "cancellationReceipt")
        _reject_unknown(
            data,
            {
                "resourceRef",
                "receiptRef",
                "runId",
                "invocationId",
                "actorRef",
                "workspaceRef",
                "snapshotDigest",
                "idempotencyKey",
                "gatewayReceiptRef",
                "auditRef",
                "kind",
            },
            "cancellationReceipt",
        )
        return cls(
            resource_ref=_require_text(data.get("resourceRef"), "cancellationReceipt.resourceRef"),
            receipt_ref=_require_text(data.get("receiptRef"), "cancellationReceipt.receiptRef"),
            run_id=_require_text(data.get("runId"), "cancellationReceipt.runId"),
            invocation_id=_require_text(
                data.get("invocationId"), "cancellationReceipt.invocationId"
            ),
            actor_ref=_require_text(data.get("actorRef"), "cancellationReceipt.actorRef"),
            workspace_ref=_require_text(
                data.get("workspaceRef"), "cancellationReceipt.workspaceRef"
            ),
            snapshot_digest=_require_text(
                data.get("snapshotDigest"), "cancellationReceipt.snapshotDigest"
            ),
            idempotency_key=_require_text(
                data.get("idempotencyKey"), "cancellationReceipt.idempotencyKey"
            ),
            gateway_receipt_ref=_require_text(
                data.get("gatewayReceiptRef"), "cancellationReceipt.gatewayReceiptRef"
            ),
            audit_ref=_require_text(data.get("auditRef"), "cancellationReceipt.auditRef"),
            kind=_require_text(data.get("kind"), "cancellationReceipt.kind"),
        )

    def to_json(self) -> str:
        payload = self.to_dict()
        _check_wire_size(payload, "cancellationReceipt", MAX_TERMINAL_RECEIPT_BYTES)
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> "CancellationAuthorityReceipt":
        try:
            raw = _check_raw_wire_size(raw, "cancellationReceipt", MAX_TERMINAL_RECEIPT_BYTES)
            return cls.from_dict(json.loads(raw))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            if isinstance(exc, ContractError):
                raise
            raise ContractError(f"invalid cancellation receipt JSON: {exc}") from exc


@dataclass(frozen=True)
class ProductReceipt:
    """Receipt returned by one narrow, host-authorized product operation."""

    resource_ref: str
    receipt_ref: str
    run_id: str
    invocation_id: str
    idempotency_key: str
    kind: str = "generic"
    actor_ref: str | None = None
    workspace_ref: str | None = None
    snapshot_digest: str | None = None
    terminal_slot: str | None = None
    proposal_digest: str | None = None
    terminal_kind: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "resource_ref", _require_text(self.resource_ref, "receipt.resourceRef"))
        object.__setattr__(self, "receipt_ref", _require_text(self.receipt_ref, "receipt.receiptRef"))
        object.__setattr__(self, "run_id", _require_text(self.run_id, "receipt.runId"))
        object.__setattr__(self, "invocation_id", _require_text(self.invocation_id, "receipt.invocationId"))
        object.__setattr__(self, "idempotency_key", _require_text(self.idempotency_key, "receipt.idempotencyKey"))
        object.__setattr__(self, "kind", _require_text(self.kind, "receipt.kind"))
        for name in ("actor_ref", "workspace_ref", "snapshot_digest", "terminal_slot", "proposal_digest"):
            object.__setattr__(
                self,
                name,
                _require_optional_text(getattr(self, name), f"receipt.{name}"),
            )
        object.__setattr__(self, "terminal_kind", _require_optional_text(self.terminal_kind, "receipt.terminalKind"))

    def to_dict(self) -> dict[str, str]:
        return {
            "resourceRef": self.resource_ref,
            "receiptRef": self.receipt_ref,
            "runId": self.run_id,
            "invocationId": self.invocation_id,
            "idempotencyKey": self.idempotency_key,
            "kind": self.kind,
            "actorRef": self.actor_ref,
            "workspaceRef": self.workspace_ref,
            "snapshotDigest": self.snapshot_digest,
            "terminalSlot": self.terminal_slot,
            "proposalDigest": self.proposal_digest,
            "terminalKind": self.terminal_kind,
        }

    @classmethod
    def from_dict(cls, raw: Any) -> "ProductReceipt":
        data = _mapping(raw, "productReceipt")
        _reject_unknown(
            data,
            {
                "resourceRef",
                "receiptRef",
                "runId",
                "invocationId",
                "idempotencyKey",
                "kind",
                "actorRef",
                "workspaceRef",
                "snapshotDigest",
                "terminalSlot",
                "proposalDigest",
                "terminalKind",
            },
            "productReceipt",
        )
        return cls(
            resource_ref=_require_text(data.get("resourceRef"), "receipt.resourceRef"),
            receipt_ref=_require_text(data.get("receiptRef"), "receipt.receiptRef"),
            run_id=_require_text(data.get("runId"), "receipt.runId"),
            invocation_id=_require_text(data.get("invocationId"), "receipt.invocationId"),
            idempotency_key=_require_text(data.get("idempotencyKey"), "receipt.idempotencyKey"),
            kind=_require_text(data.get("kind", "generic"), "receipt.kind"),
            actor_ref=_require_optional_text(data.get("actorRef"), "receipt.actorRef"),
            workspace_ref=_require_optional_text(data.get("workspaceRef"), "receipt.workspaceRef"),
            snapshot_digest=_require_optional_text(data.get("snapshotDigest"), "receipt.snapshotDigest"),
            terminal_slot=_require_optional_text(data.get("terminalSlot"), "receipt.terminalSlot"),
            proposal_digest=_require_optional_text(data.get("proposalDigest"), "receipt.proposalDigest"),
            terminal_kind=_require_optional_text(data.get("terminalKind"), "receipt.terminalKind"),
        )


@dataclass(frozen=True)
class ArtifactProposal:
    """Untrusted artifact input carried to the atomic terminal application."""

    artifact_ref: str
    digest: str
    event_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifact_ref", _require_text(self.artifact_ref, "artifact.artifactRef"))
        object.__setattr__(self, "digest", _require_text(self.digest, "artifact.digest"))
        object.__setattr__(self, "event_id", _require_text(self.event_id, "artifact.eventId"))

    def to_dict(self) -> dict[str, str]:
        return {
            "artifactRef": self.artifact_ref,
            "digest": self.digest,
            "eventId": self.event_id,
        }

    @classmethod
    def from_dict(cls, raw: Any) -> "ArtifactProposal":
        data = _mapping(raw, "artifactProposal")
        _reject_unknown(data, {"artifactRef", "digest", "eventId"}, "artifactProposal")
        return cls(data.get("artifactRef"), data.get("digest"), data.get("eventId"))


@dataclass(frozen=True)
class OutcomeProposal:
    """Required completed-terminal evidence; never stored only in an evidence tail."""

    submission_ref: str
    content: str
    event_id: str
    proposal_receipt_ref: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "submission_ref", _require_text(self.submission_ref, "outcome.submissionRef"))
        object.__setattr__(
            self,
            "content",
            _require_text(self.content, "outcome.content", max_length=MAX_TEXT_LENGTH),
        )
        object.__setattr__(self, "event_id", _require_text(self.event_id, "outcome.eventId"))
        object.__setattr__(
            self,
            "proposal_receipt_ref",
            _require_text(self.proposal_receipt_ref, "outcome.proposalReceiptRef"),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "submissionRef": self.submission_ref,
            "content": self.content,
            "eventId": self.event_id,
            "proposalReceiptRef": self.proposal_receipt_ref,
        }

    @classmethod
    def from_dict(cls, raw: Any) -> "OutcomeProposal":
        data = _mapping(raw, "outcomeProposal")
        _reject_unknown(
            data,
            {"submissionRef", "content", "eventId", "proposalReceiptRef"},
            "outcomeProposal",
        )
        return cls(
            data.get("submissionRef"),
            data.get("content"),
            data.get("eventId"),
            data.get("proposalReceiptRef"),
        )


@dataclass(frozen=True)
class InputRequestProposal:
    """Required waiting-terminal evidence; product state is applied by the port."""

    request_ref: str
    prompt: str
    event_id: str
    proposal_receipt_ref: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_ref", _require_text(self.request_ref, "input.requestRef"))
        object.__setattr__(self, "prompt", _require_text(self.prompt, "input.prompt", max_length=MAX_TEXT_LENGTH))
        object.__setattr__(self, "event_id", _require_text(self.event_id, "input.eventId"))
        object.__setattr__(
            self,
            "proposal_receipt_ref",
            _require_text(self.proposal_receipt_ref, "input.proposalReceiptRef"),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "requestRef": self.request_ref,
            "prompt": self.prompt,
            "eventId": self.event_id,
            "proposalReceiptRef": self.proposal_receipt_ref,
        }

    @classmethod
    def from_dict(cls, raw: Any) -> "InputRequestProposal":
        data = _mapping(raw, "inputRequestProposal")
        _reject_unknown(
            data,
            {"requestRef", "prompt", "eventId", "proposalReceiptRef"},
            "inputRequestProposal",
        )
        return cls(
            data.get("requestRef"),
            data.get("prompt"),
            data.get("eventId"),
            data.get("proposalReceiptRef"),
        )


@dataclass(frozen=True)
class MessageProposal:
    """Optional visible message evidence applied only by terminal reconciliation."""

    message_ref: str
    content: str
    event_id: str
    proposal_receipt_ref: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "message_ref", _require_text(self.message_ref, "message.messageRef"))
        object.__setattr__(self, "content", _require_text(self.content, "message.content", max_length=MAX_TEXT_LENGTH))
        object.__setattr__(self, "event_id", _require_text(self.event_id, "message.eventId"))
        object.__setattr__(
            self,
            "proposal_receipt_ref",
            _require_text(self.proposal_receipt_ref, "message.proposalReceiptRef"),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "messageRef": self.message_ref,
            "content": self.content,
            "eventId": self.event_id,
            "proposalReceiptRef": self.proposal_receipt_ref,
        }

    @classmethod
    def from_dict(cls, raw: Any) -> "MessageProposal":
        data = _mapping(raw, "messageProposal")
        _reject_unknown(
            data,
            {"messageRef", "content", "eventId", "proposalReceiptRef"},
            "messageProposal",
        )
        return cls(
            data.get("messageRef"),
            data.get("content"),
            data.get("eventId"),
            data.get("proposalReceiptRef"),
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
    actor_ref: str
    workspace_ref: str
    snapshot_digest: str
    kind: str
    final_sequence: int
    evidence_event_ids: tuple[str, ...] = ()
    evidence_receipt_refs: tuple[str, ...] = ()
    failure: RuntimeFailure | None = None
    source: str = "runtime"
    outcome_proposal: OutcomeProposal | None = None
    input_request_proposal: InputRequestProposal | None = None
    cancellation_receipt: CancellationAuthorityReceipt | None = None
    artifact_proposals: tuple[ArtifactProposal, ...] = ()
    message_proposals: tuple[MessageProposal, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _require_text(self.run_id, "terminal.runId"))
        object.__setattr__(
            self, "invocation_id", _require_text(self.invocation_id, "terminal.invocationId")
        )
        object.__setattr__(self, "actor_ref", _require_text(self.actor_ref, "terminal.actorRef"))
        object.__setattr__(self, "workspace_ref", _require_text(self.workspace_ref, "terminal.workspaceRef"))
        object.__setattr__(
            self,
            "snapshot_digest",
            _require_text(self.snapshot_digest, "terminal.snapshotDigest"),
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
        if len(event_ids) > MAX_TERMINAL_EVIDENCE:
            raise BoundsError(
                f"terminal.evidenceEventIds exceeds {MAX_TERMINAL_EVIDENCE} items"
            )
        if len(receipt_refs) > MAX_TERMINAL_EVIDENCE:
            raise BoundsError(
                f"terminal.evidenceReceiptRefs exceeds {MAX_TERMINAL_EVIDENCE} items"
            )
        if len(set(event_ids)) != len(event_ids):
            raise ContractError("terminal.evidenceEventIds must be unique")
        if len(set(receipt_refs)) != len(receipt_refs):
            raise ContractError("terminal.evidenceReceiptRefs must be unique")
        object.__setattr__(self, "evidence_event_ids", event_ids)
        object.__setattr__(self, "evidence_receipt_refs", receipt_refs)
        artifacts = tuple(self.artifact_proposals)
        if len(artifacts) > MAX_ARTIFACT_PROPOSALS:
            raise BoundsError(
                f"terminal.artifactProposals exceeds {MAX_ARTIFACT_PROPOSALS} items"
            )
        if not all(isinstance(item, ArtifactProposal) for item in artifacts):
            raise ContractError("terminal.artifactProposals contains an unsupported value")
        if len({item.artifact_ref for item in artifacts}) != len(artifacts):
            raise ContractError("terminal.artifactProposals must contain unique artifact references")
        object.__setattr__(self, "artifact_proposals", artifacts)
        messages = tuple(self.message_proposals)
        if len(messages) > MAX_MESSAGE_PROPOSALS:
            raise BoundsError(
                f"terminal.messageProposals exceeds {MAX_MESSAGE_PROPOSALS} items"
            )
        if not all(isinstance(item, MessageProposal) for item in messages):
            raise ContractError("terminal.messageProposals contains an unsupported value")
        if len({item.message_ref for item in messages}) != len(messages):
            raise ContractError("terminal.messageProposals must contain unique message references")
        object.__setattr__(self, "message_proposals", messages)
        if 1 + len(artifacts) + len(messages) > MAX_TERMINAL_PRODUCT_RECEIPTS:
            raise BoundsError(
                "terminal product proposal count exceeds the bounded receipt surface"
            )
        if self.kind in {"failed", "blocked"} and self.failure is None:
            raise ContractError(f"{self.kind} terminal proposal requires failure details")
        if self.kind not in {"failed", "blocked"} and self.failure is not None:
            raise ContractError(f"{self.kind} terminal proposal cannot carry failure details")
        if self.kind == "completed":
            if self.outcome_proposal is None:
                raise ContractError("completed terminal proposal requires outcome evidence")
            if self.input_request_proposal is not None or self.cancellation_receipt is not None:
                raise ContractError("completed terminal proposal has wrong-kind evidence")
            if (
                self.outcome_proposal.event_id not in event_ids
                or self.outcome_proposal.proposal_receipt_ref not in receipt_refs
            ):
                raise ContractError("completed terminal proposal evicted mandatory outcome evidence")
        elif self.kind == "waiting_for_input":
            if self.input_request_proposal is None:
                raise ContractError("waiting_for_input terminal proposal requires input evidence")
            if self.outcome_proposal is not None or self.cancellation_receipt is not None:
                raise ContractError("waiting_for_input terminal proposal has wrong-kind evidence")
            if (
                self.input_request_proposal.event_id not in event_ids
                or self.input_request_proposal.proposal_receipt_ref not in receipt_refs
            ):
                raise ContractError("waiting terminal proposal evicted mandatory input evidence")
        elif self.kind == "cancelled":
            if self.cancellation_receipt is None:
                raise ContractError("cancelled terminal proposal requires cancellation authority evidence")
            if self.outcome_proposal is not None or self.input_request_proposal is not None:
                raise ContractError("cancelled terminal proposal has wrong-kind evidence")
            if (
                self.cancellation_receipt.run_id != self.run_id
                or self.cancellation_receipt.invocation_id != self.invocation_id
            ):
                raise BindingError("cancellation evidence is bound to a different invocation")
            if self.cancellation_receipt.receipt_ref not in receipt_refs:
                raise ContractError("cancelled terminal proposal evicted mandatory cancellation evidence")
        elif self.outcome_proposal is not None or self.input_request_proposal is not None or self.cancellation_receipt is not None:
            raise ContractError(f"{self.kind} terminal proposal has wrong-kind evidence")
        if self.source not in {"runtime", "supervisor"}:
            raise ContractError(f"unsupported terminal proposal source: {self.source!r}")
        if self.source == "supervisor" and (
            self.kind != "failed"
            or self.failure is None
            or self.failure.code != "process_died"
        ):
            raise ContractError("supervisor terminal proposals must synthesize process death")
        object.__setattr__(self, "source", self.source)
        _check_wire_size(self.to_dict(), "terminalProposal", MAX_TERMINAL_PROPOSAL_BYTES)

    def to_dict(self) -> dict[str, Any]:
        return {
            "runId": self.run_id,
            "invocationId": self.invocation_id,
            "actorRef": self.actor_ref,
            "workspaceRef": self.workspace_ref,
            "snapshotDigest": self.snapshot_digest,
            "kind": self.kind,
            "finalSequence": self.final_sequence,
            "evidenceEventIds": list(self.evidence_event_ids),
            "evidenceReceiptRefs": list(self.evidence_receipt_refs),
            "failure": self.failure.to_dict() if self.failure is not None else None,
            "source": self.source,
            "outcomeProposal": self.outcome_proposal.to_dict() if self.outcome_proposal else None,
            "inputRequestProposal": (
                self.input_request_proposal.to_dict() if self.input_request_proposal else None
            ),
            "cancellationReceipt": (
                self.cancellation_receipt.to_dict() if self.cancellation_receipt else None
            ),
            "artifactProposals": [item.to_dict() for item in self.artifact_proposals],
            "messageProposals": [item.to_dict() for item in self.message_proposals],
        }

    @classmethod
    def from_dict(cls, raw: Any) -> "TerminalProposal":
        data = _mapping(raw, "terminalProposal")
        _reject_unknown(
            data,
            {
                "runId",
                "invocationId",
                "actorRef",
                "workspaceRef",
                "snapshotDigest",
                "kind",
                "finalSequence",
                "evidenceEventIds",
                "evidenceReceiptRefs",
                "failure",
                "source",
                "outcomeProposal",
                "inputRequestProposal",
                "cancellationReceipt",
                "artifactProposals",
                "messageProposals",
            },
            "terminalProposal",
        )
        failure = data.get("failure")
        cancellation = data.get("cancellationReceipt")
        return cls(
            run_id=_require_text(data.get("runId"), "terminal.runId"),
            invocation_id=_require_text(data.get("invocationId"), "terminal.invocationId"),
            actor_ref=_require_text(data.get("actorRef"), "terminal.actorRef"),
            workspace_ref=_require_text(data.get("workspaceRef"), "terminal.workspaceRef"),
            snapshot_digest=_require_text(data.get("snapshotDigest"), "terminal.snapshotDigest"),
            kind=_require_text(data.get("kind"), "terminal.kind"),
            final_sequence=_require_int(data.get("finalSequence"), "terminal.finalSequence"),
            evidence_event_ids=tuple(
                _require_text(item, "terminal.evidenceEventIds[]")
                for item in _sequence(
                    data.get("evidenceEventIds"),
                    "terminal.evidenceEventIds",
                    maximum=MAX_TERMINAL_EVIDENCE,
                )
            ),
            evidence_receipt_refs=tuple(
                _require_text(item, "terminal.evidenceReceiptRefs[]")
                for item in _sequence(
                    data.get("evidenceReceiptRefs"),
                    "terminal.evidenceReceiptRefs",
                    maximum=MAX_TERMINAL_EVIDENCE,
                )
            ),
            failure=RuntimeFailure.from_dict(failure) if failure is not None else None,
            source=_require_text(data.get("source", "runtime"), "terminal.source"),
            outcome_proposal=(
                OutcomeProposal.from_dict(data.get("outcomeProposal"))
                if data.get("outcomeProposal") is not None
                else None
            ),
            input_request_proposal=(
                InputRequestProposal.from_dict(data.get("inputRequestProposal"))
                if data.get("inputRequestProposal") is not None
                else None
            ),
            cancellation_receipt=(
                CancellationAuthorityReceipt.from_dict(cancellation)
                if cancellation is not None
                else None
            ),
            artifact_proposals=tuple(
                ArtifactProposal.from_dict(item)
                for item in _sequence(
                    data.get("artifactProposals"),
                    "terminal.artifactProposals",
                    maximum=MAX_TERMINAL_EVIDENCE,
                )
            ),
            message_proposals=tuple(
                MessageProposal.from_dict(item)
                for item in _sequence(
                    data.get("messageProposals"),
                    "terminal.messageProposals",
                    maximum=MAX_MESSAGE_PROPOSALS,
                )
            ),
        )

    def to_json(self) -> str:
        payload = self.to_dict()
        _check_wire_size(payload, "terminalProposal", MAX_TERMINAL_PROPOSAL_BYTES)
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    def digest(self) -> str:
        return _digest(self.to_dict())

    @classmethod
    def from_json(cls, raw: str) -> "TerminalProposal":
        try:
            raw = _check_raw_wire_size(raw, "terminalProposal", MAX_TERMINAL_PROPOSAL_BYTES)
            return cls.from_dict(json.loads(raw))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            if isinstance(exc, ContractError):
                raise
            raise ContractError(f"invalid terminal proposal JSON: {exc}") from exc

    @property
    def idempotency_key(self) -> str:
        """Return the one terminal slot shared by runtime and supervisor.

        Proposal details remain part of the value compared by the durable
        reconciliation port.  They deliberately do not participate in the
        key: a later competing proposal must collide with the same slot and
        be rejected or reconciled, never create a second accepted mutation.
        """

        return f"terminal:{self.run_id}:{self.invocation_id}"


@dataclass(frozen=True)
class TerminalReconciliationReceipt:
    """The host/Plane result for one terminal proposal."""

    receipt_ref: str
    run_id: str
    invocation_id: str
    kind: str
    idempotency_key: str
    accepted: bool
    legal_transition: bool
    proofs: tuple[TerminalProof, ...] = ()
    product_receipts: tuple[ProductReceipt, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "receipt_ref", _require_text(self.receipt_ref, "terminalReceipt.receiptRef"))
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
        proofs = tuple(self.proofs)
        if len(proofs) > MAX_TERMINAL_PROOFS:
            raise BoundsError(f"terminalReceipt.proofs exceeds {MAX_TERMINAL_PROOFS} items")
        if not all(isinstance(item, TerminalProof) for item in proofs):
            raise ContractError("terminalReceipt.proofs contains an unsupported value")
        if len({item.proof_kind for item in proofs}) != len(proofs):
            raise ContractError("terminalReceipt.proofs must contain unique proof kinds")
        if len({item.proof_ref for item in proofs}) != len(proofs):
            raise ContractError("terminalReceipt.proofs must contain unique proof identities")
        if len({item.resource_ref for item in proofs}) != len(proofs):
            raise ContractError("terminalReceipt.proofs must contain unique proof resources")
        if proofs:
            first = proofs[0]
            if any(
                (
                    item.run_id != first.run_id
                    or item.invocation_id != first.invocation_id
                    or item.actor_ref != first.actor_ref
                    or item.workspace_ref != first.workspace_ref
                    or item.snapshot_digest != first.snapshot_digest
                    or item.terminal_slot != first.terminal_slot
                    or item.terminal_kind != first.terminal_kind
                    or item.proposal_digest != first.proposal_digest
                )
                for item in proofs[1:]
            ):
                raise BindingError("terminal receipt proofs disagree on their common binding")
            if any(
                item.terminal_slot != self.idempotency_key
                or item.terminal_kind != self.kind
                for item in proofs
            ):
                raise BindingError("terminal receipt proofs are not bound to the receipt")
        object.__setattr__(self, "proofs", proofs)
        product_receipts = tuple(self.product_receipts)
        if len(product_receipts) > MAX_TERMINAL_PRODUCT_RECEIPTS:
            raise BoundsError(
                f"terminalReceipt.productReceipts exceeds {MAX_TERMINAL_PRODUCT_RECEIPTS} items"
            )
        if not all(isinstance(item, ProductReceipt) for item in product_receipts):
            raise ContractError("terminalReceipt.productReceipts contains an unsupported value")
        if any(
            item.run_id != self.run_id or item.invocation_id != self.invocation_id
            for item in product_receipts
        ):
            raise BindingError("terminal product receipt is not bound to the terminal slot")
        object.__setattr__(self, "product_receipts", product_receipts)
        if self.accepted and self.legal_transition:
            if len(proofs) != MAX_TERMINAL_PROOFS or {
                item.proof_kind for item in proofs
            } != TERMINAL_PROOF_KINDS:
                raise ContractError("accepted terminal receipt requires exactly one proof of every kind")
            if not product_receipts:
                raise ContractError("accepted terminal receipt requires product receipts")
            if any(
                item.terminal_kind != self.kind
                or any(
                    getattr(item, name) is None
                    for name in (
                        "actor_ref",
                        "workspace_ref",
                        "snapshot_digest",
                        "terminal_slot",
                        "proposal_digest",
                    )
                )
                for item in product_receipts
            ):
                raise ContractError("accepted terminal receipt has an incompletely bound product proof")
        _check_wire_size(self.to_dict(), "terminalReceipt", MAX_TERMINAL_RECEIPT_BYTES)

    def to_dict(self) -> dict[str, Any]:
        return {
            "receiptRef": self.receipt_ref,
            "runId": self.run_id,
            "invocationId": self.invocation_id,
            "kind": self.kind,
            "idempotencyKey": self.idempotency_key,
            "accepted": self.accepted,
            "legalTransition": self.legal_transition,
            "proofs": [item.to_dict() for item in self.proofs],
            "productReceipts": [item.to_dict() for item in self.product_receipts],
        }

    @classmethod
    def from_dict(cls, raw: Any) -> "TerminalReconciliationReceipt":
        data = _mapping(raw, "terminalReceipt")
        _reject_unknown(
            data,
            {
                "receiptRef",
                "runId",
                "invocationId",
                "kind",
                "idempotencyKey",
                "accepted",
                "legalTransition",
                "proofs",
                "productReceipts",
            },
            "terminalReceipt",
        )
        return cls(
            receipt_ref=_require_text(data.get("receiptRef"), "terminalReceipt.receiptRef"),
            run_id=_require_text(data.get("runId"), "terminalReceipt.runId"),
            invocation_id=_require_text(data.get("invocationId"), "terminalReceipt.invocationId"),
            kind=_require_text(data.get("kind"), "terminalReceipt.kind"),
            idempotency_key=_require_text(data.get("idempotencyKey"), "terminalReceipt.idempotencyKey"),
            accepted=data.get("accepted"),
            legal_transition=data.get("legalTransition"),
            proofs=tuple(
                TerminalProof.from_dict(item)
                for item in _sequence(
                    data.get("proofs"),
                    "terminalReceipt.proofs",
                    maximum=MAX_TERMINAL_PROOFS,
                )
            ),
            product_receipts=tuple(
                ProductReceipt.from_dict(item)
                for item in _sequence(
                    data.get("productReceipts"),
                    "terminalReceipt.productReceipts",
                    maximum=MAX_TERMINAL_PRODUCT_RECEIPTS,
                )
            ),
        )

    def to_json(self) -> str:
        payload = self.to_dict()
        _check_wire_size(payload, "terminalReceipt", MAX_TERMINAL_RECEIPT_BYTES)
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> "TerminalReconciliationReceipt":
        try:
            raw = _check_raw_wire_size(raw, "terminalReceipt", MAX_TERMINAL_RECEIPT_BYTES)
            return cls.from_dict(json.loads(raw))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            if isinstance(exc, ContractError):
                raise
            raise ContractError(f"invalid terminal receipt JSON: {exc}") from exc
