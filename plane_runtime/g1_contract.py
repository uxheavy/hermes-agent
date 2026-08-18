"""Small, dependency-free implementation of the accepted G1 wire contract.

The Plane repository owns the generated schemas.  This service keeps only the
frozen manifest digests and the boundary invariants it must enforce before a
snapshot or frame reaches Hermes.  It never imports Plane or treats a runtime
observation as a product mutation.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence


PROTOCOL = "plane.agent-runtime/v1"
MAX_REFERENCE_BYTES = 128
MAX_TEXT_BYTES = 4_096
MAX_PROMPT_BYTES = 32_768
MAX_TOKEN_BYTES = 256
MAX_SNAPSHOT_BYTES = 1_048_576
MAX_INVOCATION_BYTES = 16 * 1024
MAX_EVENT_BYTES = 16 * 1024
MAX_INTEGER = 2_147_483_647
MAX_EAGER_OPERATIONS = 64
MAX_EAGER_INPUT_SCHEMA_BYTES = 16 * 1024
MAX_EAGER_PRESENTATION_BYTES = 512 * 1024
MAX_EAGER_SCHEMA_PROPERTIES = 4096

# Frozen bytes from the paired Plane G1 runtime contract manifest.
G1_CONTRACT_DIGESTS = {
    "runSnapshot": "1d04c2a36f07d0e8128c3616e7dcae29af104fe4aa44d71cb1b7f43e55c0869b",
    "invocationEnvelope": "b7a15d74406f1624cdb7cd95b42edfd1ffee596abe57e4f00ed60e2e23ded995",
    "runtimeEvent": "78da5ce9d112b6545ea471e5fcae25ff5dfeb2e5db74a8d5796d0ee026823a27",
    "runtimeExit": "ed127d0ebec8f5d432ce87a6be1a8eb41b31caf808badc27ed23cd0ba9115a24",
    "runtimeDurableState": "444c944ec8a5054f33c8662470529a1f4565d42ff06138438beceeef7967a0da",
}
G1_MANIFEST_DIGEST = "bc45b732e691ca9650e2f741f91344ddaec41c92da63bdfeafd98ea184e1d73e"

_ROLES = {"worker", "delegator", "gardener", "chief_of_staff", "hr", "evaluator", "custom"}
_TRIGGERS = {"initial", "human_input", "recoverable_restart", "continuation"}
_EVENT_KINDS = {
    "progress_observed",
    "conversation_publication_observed",
    "input_request_observed",
    "artifact_observed",
    "usage_observed",
    "outcome_submission_observed",
    "failure_observed",
    "blocker_observed",
    "cancellation_observed",
    "transcript_evidence_observed",
}
_PRODUCT_REF_PREFIXES = {
    "conversation": "conversation:",
    "input_request": "input-request:",
    "artifact": "artifact:",
    "outcome_submission": "outcome-submission:",
    "run_failure": "product-event:",
    "run_blocker": "product-event:",
    "run_cancellation": "product-event:",
}
_APPLIED_PRODUCT_KINDS = set(_PRODUCT_REF_PREFIXES)
_FAILURE_CODES = {
    "runtime_error",
    "lease_expired",
    "invalid_continuation",
    "budget_exhausted",
    "outcome_unknown",
    "cancelled",
}
RUNTIME_FAILURE_CAUSES = frozenset(
    {
        "host_operation_failure",
        "cancellation_monitor_failure",
        "invalid_usage_accounting",
        "static_configuration_failure",
        "dependency_failure",
        "permission_failure",
        "resource_failure",
        "timeout_failure",
        "provider_client_failure",
        "runtime_unknown_failure",
        "provider_auth_failure",
        "provider_entitlement_failure",
        "provider_rate_limit",
        "provider_request_failure",
        "provider_transport_failure",
        "provider_unknown_failure",
    }
)


class G1ContractError(ValueError):
    """Raised when an untrusted value is not valid G1 runtime evidence."""


def _canonical(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise G1ContractError("runtime value is not canonical JSON") from exc


def content_digest(value: Mapping[str, Any]) -> str:
    return f"content:{hashlib.sha256(_canonical(value)).hexdigest()}"


def snapshot_digest(value: Mapping[str, Any]) -> str:
    return f"snapshot:{hashlib.sha256(_canonical(value)).hexdigest()}"


def _freeze(value: Any, path: str = "value") -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise G1ContractError(f"{path} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): _freeze(v, f"{path}.{k}") for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v, f"{path}[]") for v in value)
    raise G1ContractError(f"{path} contains an unsupported value")


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(k, str) for k in value):
        raise G1ContractError(f"{name} must be an object")
    return dict(value)


def _reject_unknown(value: Mapping[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(set(value).difference(allowed))
    if unknown:
        raise G1ContractError(f"{name} has unknown field(s): {', '.join(unknown)}")


def _required(value: Mapping[str, Any], fields: set[str], name: str) -> None:
    missing = sorted(fields.difference(value))
    if missing:
        raise G1ContractError(f"{name} is missing field(s): {', '.join(missing)}")


def _text(value: Any, name: str, *, maximum: int = MAX_TEXT_BYTES) -> str:
    if not isinstance(value, str) or not value:
        raise G1ContractError(f"{name} must be a non-empty string")
    if len(value.encode("utf-8")) > maximum:
        raise G1ContractError(f"{name} exceeds {maximum} UTF-8 bytes")
    return value


def _ref(value: Any, name: str, namespace: str) -> str:
    value = _text(value, name, maximum=MAX_REFERENCE_BYTES)
    if not value.startswith(f"{namespace}:") or len(value.split(":", 1)[1]) == 0:
        raise G1ContractError(f"{name} must use the {namespace}: reference namespace")
    suffix = value.split(":", 1)[1]
    if len(suffix) > 120 or not suffix[0].isalnum() or any(char not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._~/-" for char in suffix):
        raise G1ContractError(f"{name} contains invalid reference characters")
    return value


def _token(value: Any, name: str) -> str:
    return _text(value, name, maximum=MAX_TOKEN_BYTES)


def _bounded_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= MAX_INTEGER:
        raise G1ContractError(f"{name} must be a bounded non-negative integer")
    return value


def _bounded_byte_count(value: Any, name: str) -> int:
    value = _bounded_integer(value, name)
    if value > 1_048_576:
        raise G1ContractError(f"{name} exceeds the bounded byte-count maximum")
    return value


def _bounded_budget(value: Any, name: str = "budget") -> dict[str, int]:
    data = _object(value, name)
    _reject_unknown(data, {"inputTokens", "outputTokens", "durationMs"}, name)
    _required(data, {"inputTokens", "outputTokens", "durationMs"}, name)
    return {
        "inputTokens": _bounded_integer(data["inputTokens"], f"{name}.inputTokens"),
        "outputTokens": _bounded_integer(data["outputTokens"], f"{name}.outputTokens"),
        "durationMs": _bounded_integer(data["durationMs"], f"{name}.durationMs"),
    }


def _bounded_payload(
    value: Any,
    name: str = "payload",
) -> dict[str, Any]:
    data = _object(value, name)
    kind = data.get("kind")
    if kind == "inline_text":
        _reject_unknown(data, {"kind", "contentType", "text"}, name)
        _required(data, {"kind", "contentType", "text"}, name)
        if data["contentType"] != "text/plain":
            raise G1ContractError(f"{name}.contentType must be text/plain")
        _text(data["text"], f"{name}.text")
        return data
    if kind == "payload_ref":
        _reject_unknown(data, {"kind", "payloadRef", "contentType", "contentDigest", "sizeBytes"}, name)
        _required(data, {"kind", "payloadRef", "contentType", "contentDigest", "sizeBytes"}, name)
        _ref(data["payloadRef"], f"{name}.payloadRef", "payload")
        _token(data["contentType"], f"{name}.contentType")
        _content_ref(data["contentDigest"], f"{name}.contentDigest")
        size_bytes = _bounded_byte_count(data["sizeBytes"], f"{name}.sizeBytes")
        return data
    raise G1ContractError(f"{name}.kind is not a supported bounded payload")


def _content_ref(value: Any, name: str) -> str:
    value = _text(value, name, maximum=73)
    if len(value) != 72 or not value.startswith("content:"):
        raise G1ContractError(f"{name} must be a content digest")
    if any(char not in "0123456789abcdef" for char in value[8:]):
        raise G1ContractError(f"{name} must contain a lowercase hexadecimal digest")
    return value


def validate_eager_input_schema(value: Any, name: str = "inputSchema") -> dict[str, Any]:
    """Validate and preserve one bounded canonical JSON Schema object."""

    data = _object(value, name)
    if len(data) > MAX_EAGER_SCHEMA_PROPERTIES:
        raise G1ContractError(
            f"{name} must be a bounded canonical JSON Schema object"
        )

    def validate_json_value(item: Any, path: str) -> Any:
        if item is None or isinstance(item, (bool, int, str)):
            return item
        if isinstance(item, float):
            if not math.isfinite(item):
                raise G1ContractError(
                    f"{name} must be a bounded canonical JSON Schema object"
                )
            return item
        if isinstance(item, Mapping):
            if any(not isinstance(key, str) for key in item):
                raise G1ContractError(
                    f"{name} must be a bounded canonical JSON Schema object"
                )
            return {key: validate_json_value(value, f"{path}.{key}") for key, value in item.items()}
        if isinstance(item, list):
            return [validate_json_value(child, f"{path}[]") for child in item]
        raise G1ContractError(
            f"{name} must be a bounded canonical JSON Schema object"
        )

    try:
        validated = validate_json_value(data, name)
        encoded = _canonical(validated)
    except RecursionError as exc:
        raise G1ContractError(
            f"{name} must be a bounded canonical JSON Schema object"
        ) from exc
    if len(encoded) > MAX_EAGER_INPUT_SCHEMA_BYTES:
        raise G1ContractError(
            f"{name} must be a bounded canonical JSON Schema object"
        )
    return validated


def _snapshot_ref(value: Any, name: str) -> str:
    value = _text(value, name, maximum=73)
    if len(value) != 73 or not value.startswith("snapshot:"):
        raise G1ContractError(f"{name} must be a snapshot digest")
    if any(char not in "0123456789abcdef" for char in value[9:]):
        raise G1ContractError(f"{name} must contain a lowercase hexadecimal digest")
    return value


def _validate_snapshot(raw: Any) -> dict[str, Any]:
    data = _object(raw, "RunSnapshot")
    _reject_unknown(
        data,
        {
            "protocol", "workspaceRef", "runId", "assignment", "actorRef", "profile",
            "context", "toolCatalog", "runtimePolicy", "totalBudget", "contractDigests",
            "contentDigest",
        },
        "RunSnapshot",
    )
    _required(data, {
        "protocol", "workspaceRef", "runId", "assignment", "actorRef", "profile", "context",
        "toolCatalog", "runtimePolicy", "totalBudget", "contractDigests", "contentDigest",
    }, "RunSnapshot")
    if data["protocol"] != PROTOCOL:
        raise G1ContractError("RunSnapshot.protocol is unsupported")
    _ref(data["workspaceRef"], "RunSnapshot.workspaceRef", "workspace")
    _ref(data["runId"], "RunSnapshot.runId", "run")
    _ref(data["actorRef"], "RunSnapshot.actorRef", "actor")

    assignment = _object(data["assignment"], "RunSnapshot.assignment")
    _reject_unknown(assignment, {"assignmentRef", "revision", "targetRef", "objective", "acceptanceCriteria"}, "assignment")
    _required(assignment, {"assignmentRef", "revision", "targetRef", "objective", "acceptanceCriteria"}, "assignment")
    _ref(assignment["assignmentRef"], "assignment.assignmentRef", "assignment")
    _token(assignment["revision"], "assignment.revision")
    _ref(assignment["targetRef"], "assignment.targetRef", "target")
    _text(assignment["objective"], "assignment.objective")
    criteria = assignment["acceptanceCriteria"]
    if not isinstance(criteria, list) or not 1 <= len(criteria) <= 32:
        raise G1ContractError("assignment.acceptanceCriteria must contain 1..32 items")
    for item in criteria:
        _text(item, "assignment.acceptanceCriteria[]")

    profile = _object(data["profile"], "RunSnapshot.profile")
    _reject_unknown(profile, {"profileRef", "revision", "role", "behavioralPrompt"}, "profile")
    _required(profile, {"profileRef", "revision", "role", "behavioralPrompt"}, "profile")
    _ref(profile["profileRef"], "profile.profileRef", "profile-version")
    _token(profile["revision"], "profile.revision")
    if profile["role"] not in _ROLES:
        raise G1ContractError("profile.role is unsupported")
    _text(profile["behavioralPrompt"], "profile.behavioralPrompt", maximum=MAX_PROMPT_BYTES)

    context = data["context"]
    if not isinstance(context, list) or len(context) > 64:
        raise G1ContractError("context must contain at most 64 items")
    for index, item in enumerate(context):
        item_data = _object(item, f"context[{index}]")
        _reject_unknown(item_data, {"contextRef", "revision", "contentDigest"}, f"context[{index}]")
        _required(item_data, {"contextRef", "revision", "contentDigest"}, f"context[{index}]")
        _ref(item_data["contextRef"], f"context[{index}].contextRef", "context")
        _token(item_data["revision"], f"context[{index}].revision")
        _content_ref(item_data["contentDigest"], f"context[{index}].contentDigest")

    catalog = _object(data["toolCatalog"], "RunSnapshot.toolCatalog")
    _reject_unknown(catalog, {"catalogDigest", "modelToolset", "eagerOperations"}, "toolCatalog")
    _required(catalog, {"catalogDigest", "eagerOperations"}, "toolCatalog")
    _content_ref(catalog["catalogDigest"], "toolCatalog.catalogDigest")
    if "modelToolset" in catalog and catalog["modelToolset"] not in {"standard", "code_mode_only"}:
        raise G1ContractError("toolCatalog.modelToolset is unsupported")
    operations = catalog["eagerOperations"]
    if not isinstance(operations, list) or len(operations) > MAX_EAGER_OPERATIONS:
        raise G1ContractError(
            f"toolCatalog.eagerOperations must contain at most {MAX_EAGER_OPERATIONS} items"
        )
    for index, item in enumerate(operations):
        operation = _object(item, f"toolCatalog.eagerOperations[{index}]")
        operation_name = f"toolCatalog.eagerOperations[{index}]"
        _reject_unknown(
            operation,
            {"operationRef", "schemaDigest", "inputSchema", "disclosure"},
            operation_name,
        )
        _required(
            operation,
            {"operationRef", "schemaDigest", "inputSchema", "disclosure"},
            operation_name,
        )
        _ref(operation["operationRef"], f"{operation_name}.operationRef", "operation")
        _content_ref(operation["schemaDigest"], f"{operation_name}.schemaDigest")
        validate_eager_input_schema(operation["inputSchema"], f"{operation_name}.inputSchema")
        if operation["disclosure"] != "eager":
            raise G1ContractError(f"{operation_name}.disclosure must be eager")
    if len(_canonical(catalog)) > MAX_EAGER_PRESENTATION_BYTES:
        raise G1ContractError(
            f"toolCatalog exceeds {MAX_EAGER_PRESENTATION_BYTES} canonical JSON bytes"
        )

    policy = _object(data["runtimePolicy"], "RunSnapshot.runtimePolicy")
    _reject_unknown(
        policy,
        {
            "model",
            "adapter",
            "isolation",
            "maxEventPayloadBytes",
            "maxArtifactBytes",
            "maxReceiptBytes",
            "maxCodeModeInputBytes",
            "maxCodeModeOutputBytes",
            "maxCodeModeCalls",
        },
        "runtimePolicy",
    )
    _required(policy, {"model", "adapter", "isolation", "maxEventPayloadBytes", "maxArtifactBytes", "maxReceiptBytes"}, "runtimePolicy")
    model = _object(policy["model"], "runtimePolicy.model")
    _reject_unknown(model, {"provider", "model"}, "runtimePolicy.model")
    _required(model, {"provider", "model"}, "runtimePolicy.model")
    _token(model["provider"], "runtimePolicy.model.provider")
    _token(model["model"], "runtimePolicy.model.model")
    _token(policy["adapter"], "runtimePolicy.adapter")
    if policy["isolation"] != "single-invocation":
        raise G1ContractError("runtimePolicy.isolation must be single-invocation")
    for key in ("maxEventPayloadBytes", "maxArtifactBytes", "maxReceiptBytes"):
        _bounded_byte_count(policy[key], f"runtimePolicy.{key}")
    for key in ("maxCodeModeInputBytes", "maxCodeModeOutputBytes"):
        if key in policy:
            _bounded_byte_count(policy[key], f"runtimePolicy.{key}")
    if "maxCodeModeCalls" in policy:
        _bounded_integer(policy["maxCodeModeCalls"], "runtimePolicy.maxCodeModeCalls")
    _bounded_budget(data["totalBudget"], "totalBudget")

    digests = _object(data["contractDigests"], "contractDigests")
    _reject_unknown(digests, set(G1_CONTRACT_DIGESTS), "contractDigests")
    _required(digests, set(G1_CONTRACT_DIGESTS), "contractDigests")
    if digests != G1_CONTRACT_DIGESTS:
        raise G1ContractError("contractDigests do not match the accepted G1 manifest")
    _snapshot_ref(data["contentDigest"], "RunSnapshot.contentDigest")
    expected = snapshot_digest({key: data[key] for key in data if key != "contentDigest"})
    if data["contentDigest"] != expected:
        raise G1ContractError("RunSnapshot.contentDigest does not match canonical immutable content")
    if len(_canonical(data)) > MAX_SNAPSHOT_BYTES:
        raise G1ContractError("RunSnapshot exceeds its wire bound")
    return data


def _validate_invocation(raw: Any) -> dict[str, Any]:
    data = _object(raw, "InvocationEnvelope")
    allowed = {
        "protocol", "workspaceRef", "actorRef", "runId", "invocationId", "runSnapshotDigest",
        "trigger", "newContextEventRefs", "checkpointRef", "remainingBudget", "lease",
        "cancellationRef", "causationRef", "correlationId", "idempotencyKey",
    }
    _reject_unknown(data, allowed, "InvocationEnvelope")
    _required(data, allowed - {"checkpointRef"}, "InvocationEnvelope")
    if data["protocol"] != PROTOCOL:
        raise G1ContractError("InvocationEnvelope.protocol is unsupported")
    for key, namespace in (("workspaceRef", "workspace"), ("actorRef", "actor"), ("runId", "run"), ("invocationId", "invocation"), ("cancellationRef", "cancellation"), ("causationRef", "causation"), ("correlationId", "correlation"), ("idempotencyKey", "idempotency")):
        _ref(data[key], f"InvocationEnvelope.{key}", namespace)
    _snapshot_ref(data["runSnapshotDigest"], "InvocationEnvelope.runSnapshotDigest")
    if "checkpointRef" in data:
        _ref(data["checkpointRef"], "InvocationEnvelope.checkpointRef", "checkpoint")
    trigger = _object(data["trigger"], "InvocationEnvelope.trigger")
    kind = trigger.get("kind")
    if kind not in _TRIGGERS:
        raise G1ContractError("InvocationEnvelope.trigger.kind is unsupported")
    if kind == "initial":
        _reject_unknown(trigger, {"kind"}, "trigger")
    elif kind == "human_input":
        _reject_unknown(trigger, {"kind", "eventRef", "pendingInputEventRef", "answerFactDigest"}, "trigger")
        _required(trigger, {"kind", "eventRef", "pendingInputEventRef", "answerFactDigest"}, "trigger")
        _ref(trigger["eventRef"], "trigger.eventRef", "event")
        _ref(trigger["pendingInputEventRef"], "trigger.pendingInputEventRef", "event")
        _content_ref(trigger["answerFactDigest"], "trigger.answerFactDigest")
    else:
        _reject_unknown(trigger, {"kind", "eventRef", "pendingInputEventRef"}, "trigger")
        _required(trigger, {"kind", "eventRef"}, "trigger")
        _ref(trigger["eventRef"], "trigger.eventRef", "event")
        if "pendingInputEventRef" in trigger:
            _ref(trigger["pendingInputEventRef"], "trigger.pendingInputEventRef", "event")
    refs = data["newContextEventRefs"]
    if not isinstance(refs, list) or len(refs) > 64:
        raise G1ContractError("newContextEventRefs must contain at most 64 items")
    for ref in refs:
        _ref(ref, "newContextEventRefs[]", "event")
    _bounded_budget(data["remainingBudget"], "remainingBudget")
    lease = _object(data["lease"], "InvocationEnvelope.lease")
    _reject_unknown(lease, {"leaseId", "expiresAt", "renewAfterMs"}, "lease")
    _required(lease, {"leaseId", "expiresAt", "renewAfterMs"}, "lease")
    _ref(lease["leaseId"], "lease.leaseId", "lease")
    _text(lease["expiresAt"], "lease.expiresAt", maximum=64)
    _bounded_integer(lease["renewAfterMs"], "lease.renewAfterMs")
    if len(_canonical(data)) > MAX_INVOCATION_BYTES:
        raise G1ContractError("InvocationEnvelope exceeds its wire bound")
    return data


@dataclass(frozen=True)
class G1RunSnapshot:
    """Deeply immutable, validated G1 snapshot."""

    raw: Mapping[str, Any]

    @classmethod
    def from_dict(cls, value: Any) -> "G1RunSnapshot":
        return cls(_freeze(_validate_snapshot(value), "RunSnapshot"))

    @property
    def digest(self) -> str:
        return str(self.raw["contentDigest"])

    @property
    def run_id(self) -> str:
        return str(self.raw["runId"])

    @property
    def workspace_ref(self) -> str:
        return str(self.raw["workspaceRef"])

    @property
    def actor_ref(self) -> str:
        return str(self.raw["actorRef"])

    @property
    def profile_version_ref(self) -> str:
        return str(self.raw["profile"]["profileRef"])

    @property
    def objective(self) -> str:
        return str(self.raw["assignment"]["objective"])

    @property
    def target_ref(self) -> str:
        return str(self.raw["assignment"]["targetRef"])

    @property
    def acceptance_criteria(self) -> tuple[str, ...]:
        return tuple(str(item) for item in self.raw["assignment"]["acceptanceCriteria"])

    @property
    def behavioral_prompt(self) -> str:
        return str(self.raw["profile"]["behavioralPrompt"])

    @property
    def context_refs(self) -> tuple[str, ...]:
        return tuple(str(item["contextRef"]) for item in self.raw["context"])

    @property
    def eager_operations(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self.raw["toolCatalog"]["eagerOperations"])

    @property
    def model_toolset(self) -> str:
        return str(self.raw["toolCatalog"].get("modelToolset", "standard"))

    @property
    def model_provider(self) -> str:
        return str(self.raw["runtimePolicy"]["model"]["provider"])

    @property
    def model_name(self) -> str:
        return str(self.raw["runtimePolicy"]["model"]["model"])

    @property
    def adapter_name(self) -> str:
        return str(self.raw["runtimePolicy"]["adapter"])

    @property
    def total_budget(self) -> Mapping[str, int]:
        return self.raw["totalBudget"]

    def to_dict(self) -> dict[str, Any]:
        return _thaw(self.raw)


@dataclass(frozen=True)
class G1InvocationEnvelope:
    """Deeply immutable, validated G1 invocation envelope."""

    raw: Mapping[str, Any]

    @classmethod
    def from_dict(cls, value: Any) -> "G1InvocationEnvelope":
        return cls(_freeze(_validate_invocation(value), "InvocationEnvelope"))

    @property
    def invocation_id(self) -> str:
        return str(self.raw["invocationId"])

    @property
    def run_snapshot_digest(self) -> str:
        return str(self.raw["runSnapshotDigest"])

    @property
    def idempotency_key(self) -> str:
        return str(self.raw["idempotencyKey"])

    @property
    def correlation_id(self) -> str:
        return str(self.raw["correlationId"])

    @property
    def causation_ref(self) -> str:
        return str(self.raw["causationRef"])

    @property
    def lease(self) -> Mapping[str, Any]:
        return self.raw["lease"]

    @property
    def remaining_budget(self) -> Mapping[str, int]:
        return self.raw["remainingBudget"]

    def to_dict(self) -> dict[str, Any]:
        return _thaw(self.raw)


def bind_snapshot_and_invocation(snapshot: G1RunSnapshot, invocation: G1InvocationEnvelope) -> None:
    envelope = invocation.raw
    expected = {
        "workspaceRef": snapshot.raw["workspaceRef"],
        "actorRef": snapshot.raw["actorRef"],
        "runId": snapshot.raw["runId"],
        "runSnapshotDigest": snapshot.digest,
    }
    if any(envelope[key] != value for key, value in expected.items()):
        raise G1ContractError("InvocationEnvelope is not bound to the immutable RunSnapshot")
    for key, limit in (("inputTokens", "inputTokens"), ("outputTokens", "outputTokens"), ("durationMs", "durationMs")):
        if envelope["remainingBudget"][key] > snapshot.total_budget[key]:
            raise G1ContractError(f"remainingBudget.{limit} exceeds the cumulative run budget")


def _publication(
    value: Any,
    *,
    product_kind: str | None = None,
    proposal_allowed: bool = True,
    observation_only: bool = False,
) -> dict[str, Any]:
    data = _object(value, "publication")
    action = data.get("action")
    if action == "observation_only":
        _reject_unknown(data, {"action"}, "publication")
        if product_kind is not None or not observation_only:
            raise G1ContractError("product publication cannot be observation_only")
        return data
    if observation_only:
        raise G1ContractError("observation publication must be observation_only")
    if action not in {"proposal", "applied"} or (action == "proposal" and not proposal_allowed):
        raise G1ContractError("publication action is unsupported")
    expected = {"action", "productKind", "productRef", "operationAttemptRef"}
    if action == "applied":
        expected |= {"operationRef", "applicationServiceRef", "gatewayReceiptRef", "receiptRef", "auditReceiptRef", "productEventRef"}
        if data.get("productKind") == "run_cancellation":
            expected |= {"cancellationRef"}
    _reject_unknown(data, expected, "publication")
    _required(data, expected, "publication")
    if product_kind is not None and data["productKind"] != product_kind:
        raise G1ContractError("publication productKind does not match event body")
    actual_kind = str(data["productKind"])
    if actual_kind not in _APPLIED_PRODUCT_KINDS:
        raise G1ContractError("publication productKind is unsupported")
    _ref(data["productRef"], "publication.productRef", _PRODUCT_REF_PREFIXES[actual_kind].rstrip(":"))
    _ref(data["operationAttemptRef"], "publication.operationAttemptRef", "operation-attempt")
    if action == "applied":
        for key, namespace in (("operationRef", "operation"), ("applicationServiceRef", "application-service"), ("gatewayReceiptRef", "gateway-receipt"), ("receiptRef", "receipt"), ("auditReceiptRef", "audit-receipt"), ("productEventRef", "product-event")):
            _ref(data[key], f"publication.{key}", namespace)
        if actual_kind in {"run_failure", "run_blocker", "run_cancellation"} and data["productRef"] != data["productEventRef"]:
            raise G1ContractError("terminal publication productRef must equal productEventRef")
        if actual_kind == "run_cancellation":
            _ref(data["cancellationRef"], "publication.cancellationRef", "cancellation")
    return data


def _validate_failure(value: Any, name: str = "failure") -> dict[str, Any]:
    data = _object(value, name)
    _reject_unknown(
        data,
        {"code", "message", "retryable", "cause", "callbackPhase", "operationRefDigest"},
        name,
    )
    _required(data, {"code", "message", "retryable"}, name)
    if data["code"] not in _FAILURE_CODES or not isinstance(data["retryable"], bool):
        raise G1ContractError(f"{name} is invalid")
    if "cause" in data and (
        data["code"] != "runtime_error" or data["cause"] not in RUNTIME_FAILURE_CAUSES
    ):
        raise G1ContractError(f"{name}.cause is invalid")
    diagnostic_fields = {"callbackPhase", "operationRefDigest"}
    present_diagnostic_fields = diagnostic_fields.intersection(data)
    if present_diagnostic_fields and present_diagnostic_fields != diagnostic_fields:
        raise G1ContractError(f"{name} host diagnostic fields must be provided together")
    if present_diagnostic_fields:
        if data["callbackPhase"] not in {
            "before_host_call",
            "host_return",
            "model_observation_emit",
            "adapter_event",
        }:
            raise G1ContractError(f"{name}.callbackPhase is invalid")
        operation_ref_digest = data["operationRefDigest"]
        if (
            not isinstance(operation_ref_digest, str)
            or len(operation_ref_digest) != 64
            or any(char not in "0123456789abcdef" for char in operation_ref_digest)
        ):
            raise G1ContractError(f"{name}.operationRefDigest is invalid")
    _text(data["message"], f"{name}.message")
    return data


def _validate_event(
    raw: Any,
    *,
    max_event_bytes: int = MAX_EVENT_BYTES,
    max_artifact_bytes: int = 1_048_576,
) -> dict[str, Any]:
    data = _object(raw, "RuntimeEvent")
    allowed = {"protocol", "trust", "workspaceRef", "actorRef", "runId", "invocationId", "sequence", "eventId", "idempotencyKey", "correlationId", "causationRef", "observedAt", "body"}
    _reject_unknown(data, allowed, "RuntimeEvent")
    _required(data, allowed, "RuntimeEvent")
    if data["protocol"] != PROTOCOL or data["trust"] != "untrusted":
        raise G1ContractError("RuntimeEvent protocol or trust is invalid")
    for key, namespace in (("workspaceRef", "workspace"), ("actorRef", "actor"), ("runId", "run"), ("invocationId", "invocation"), ("eventId", "event"), ("idempotencyKey", "idempotency"), ("correlationId", "correlation"), ("causationRef", "causation")):
        _ref(data[key], f"RuntimeEvent.{key}", namespace)
    _bounded_integer(data["sequence"], "RuntimeEvent.sequence")
    _text(data["observedAt"], "RuntimeEvent.observedAt", maximum=64)
    body = _object(data["body"], "RuntimeEvent.body")
    kind = body.get("kind")
    if kind not in _EVENT_KINDS:
        raise G1ContractError("RuntimeEvent.body.kind is unsupported")
    if kind in {"progress_observed", "conversation_publication_observed", "outcome_submission_observed", "transcript_evidence_observed"}:
        _reject_unknown(body, {"kind", "payload", "publication"}, "RuntimeEvent.body")
        _required(body, {"kind", "payload", "publication"}, "RuntimeEvent.body")
        _bounded_payload(
            body["payload"],
            "RuntimeEvent.body.payload",
        )
        product_kind = {"conversation_publication_observed": "conversation", "outcome_submission_observed": "outcome_submission"}.get(kind)
        _publication(
            body["publication"],
            product_kind=product_kind,
            proposal_allowed=kind != "outcome_submission_observed",
            observation_only=kind in {"progress_observed", "transcript_evidence_observed"},
        )
    elif kind == "input_request_observed":
        _reject_unknown(body, {"kind", "question", "publication"}, "RuntimeEvent.body")
        _required(body, {"kind", "question", "publication"}, "RuntimeEvent.body")
        _text(body["question"], "RuntimeEvent.body.question")
        _publication(body["publication"], product_kind="input_request")
    elif kind == "artifact_observed":
        _reject_unknown(body, {"kind", "artifact", "publication"}, "RuntimeEvent.body")
        _required(body, {"kind", "artifact", "publication"}, "RuntimeEvent.body")
        artifact = _object(body["artifact"], "RuntimeEvent.body.artifact")
        _reject_unknown(artifact, {"artifactRef", "contentDigest", "mediaType", "sizeBytes"}, "artifact")
        _required(artifact, {"artifactRef", "contentDigest", "mediaType", "sizeBytes"}, "artifact")
        _ref(artifact["artifactRef"], "artifact.artifactRef", "artifact")
        _content_ref(artifact["contentDigest"], "artifact.contentDigest")
        _token(artifact["mediaType"], "artifact.mediaType")
        artifact_size = _bounded_byte_count(artifact["sizeBytes"], "artifact.sizeBytes")
        if artifact_size > max_artifact_bytes:
            raise G1ContractError("artifact.sizeBytes exceeds the selected runtime policy")
        _publication(body["publication"], product_kind="artifact")
    elif kind == "usage_observed":
        _reject_unknown(body, {"kind", "usage", "publication"}, "RuntimeEvent.body")
        _required(body, {"kind", "usage", "publication"}, "RuntimeEvent.body")
        _bounded_budget(body["usage"], "RuntimeEvent.body.usage")
        _publication(body["publication"], observation_only=True)
    elif kind == "failure_observed":
        _reject_unknown(body, {"kind", "failure", "publication"}, "RuntimeEvent.body")
        _required(body, {"kind", "failure", "publication"}, "RuntimeEvent.body")
        _validate_failure(body["failure"], "RuntimeEvent.body.failure")
        _publication(body["publication"], product_kind="run_failure", proposal_allowed=False)
    elif kind == "blocker_observed":
        _reject_unknown(body, {"kind", "reason", "publication"}, "RuntimeEvent.body")
        _required(body, {"kind", "reason", "publication"}, "RuntimeEvent.body")
        _text(body["reason"], "RuntimeEvent.body.reason")
        _publication(body["publication"], product_kind="run_blocker", proposal_allowed=False)
    elif kind == "cancellation_observed":
        _reject_unknown(body, {"kind", "reason", "cancellationRef", "publication"}, "RuntimeEvent.body")
        _required(body, {"kind", "reason", "cancellationRef", "publication"}, "RuntimeEvent.body")
        _text(body["reason"], "RuntimeEvent.body.reason")
        _ref(body["cancellationRef"], "RuntimeEvent.body.cancellationRef", "cancellation")
        publication = _publication(body["publication"], product_kind="run_cancellation", proposal_allowed=False)
        if "cancellationRef" in publication and publication["cancellationRef"] != body["cancellationRef"]:
            raise G1ContractError("cancellation publication is not bound to the event")
    if len(_canonical(data)) > max_event_bytes:
        raise G1ContractError("RuntimeEvent exceeds the selected runtime policy")
    return data


def _validate_exit(raw: Any) -> dict[str, Any]:
    data = _object(raw, "RuntimeExit")
    allowed = {"protocol", "authority", "workspaceRef", "actorRef", "runId", "invocationId", "finalSequence", "idempotencyKey", "correlationId", "causationRef", "kind", "inputEventRef", "failure"}
    _reject_unknown(data, allowed, "RuntimeExit")
    _required(data, {"protocol", "authority", "workspaceRef", "actorRef", "runId", "invocationId", "finalSequence", "idempotencyKey", "correlationId", "causationRef", "kind"}, "RuntimeExit")
    if data["protocol"] != PROTOCOL or data["authority"] != "runtime_evidence_only":
        raise G1ContractError("RuntimeExit protocol or authority is invalid")
    for key, namespace in (("workspaceRef", "workspace"), ("actorRef", "actor"), ("runId", "run"), ("invocationId", "invocation"), ("idempotencyKey", "idempotency"), ("correlationId", "correlation"), ("causationRef", "causation")):
        _ref(data[key], f"RuntimeExit.{key}", namespace)
    _bounded_integer(data["finalSequence"], "RuntimeExit.finalSequence")
    kind = data["kind"]
    if kind not in {"completed", "waiting_for_input", "failed", "blocked", "cancelled"}:
        raise G1ContractError("RuntimeExit.kind is unsupported")
    if kind == "waiting_for_input":
        if "inputEventRef" not in data:
            raise G1ContractError("waiting_for_input RuntimeExit requires inputEventRef")
        _ref(data["inputEventRef"], "RuntimeExit.inputEventRef", "event")
    elif "inputEventRef" in data:
        raise G1ContractError("only waiting_for_input RuntimeExit may carry inputEventRef")
    if kind in {"failed", "blocked", "cancelled"}:
        if "failure" not in data:
            raise G1ContractError(f"{kind} RuntimeExit requires failure")
        _validate_failure(data["failure"], "RuntimeExit.failure")
    elif "failure" in data:
        raise G1ContractError(f"{kind} RuntimeExit cannot carry failure")
    return data


def _policy_bounds(snapshot: Mapping[str, Any]) -> tuple[int, int, int]:
    policy = snapshot["runtimePolicy"]
    return (
        _bounded_byte_count(policy["maxEventPayloadBytes"], "runtimePolicy.maxEventPayloadBytes"),
        _bounded_byte_count(policy["maxArtifactBytes"], "runtimePolicy.maxArtifactBytes"),
        _bounded_byte_count(policy["maxReceiptBytes"], "runtimePolicy.maxReceiptBytes"),
    )


def validate_g1_frame(frame: Mapping[str, Any], snapshot: Mapping[str, Any], invocation: Mapping[str, Any]) -> dict[str, Any]:
    event_bytes, artifact_bytes, _receipt_bytes = _policy_bounds(snapshot)
    parsed = (
        _validate_event(
            frame,
            max_event_bytes=event_bytes,
            max_artifact_bytes=artifact_bytes,
        )
        if "trust" in frame
        else _validate_exit(frame)
        if "authority" in frame
        else None
    )
    if parsed is None:
        raise G1ContractError("frame is neither a RuntimeEvent nor a RuntimeExit")
    expected = {
        "workspaceRef": snapshot["workspaceRef"],
        "actorRef": snapshot["actorRef"],
        "runId": snapshot["runId"],
        "invocationId": invocation["invocationId"],
        "correlationId": invocation["correlationId"],
        "causationRef": invocation["causationRef"],
    }
    for key, value in expected.items():
        if parsed[key] != value:
            raise G1ContractError(f"frame {key} is not bound to the invocation")
    if "authority" in parsed and parsed.get("idempotencyKey") != invocation["idempotencyKey"]:
        raise G1ContractError("RuntimeExit.idempotencyKey is not bound to the invocation")
    return parsed


def validate_g1_frames(
    frames: Sequence[Mapping[str, Any]],
    snapshot: Mapping[str, Any],
    invocation: Mapping[str, Any],
    *,
    max_stream_bytes: int = 512 * 1024,
) -> tuple[dict[str, Any], ...]:
    """Validate one complete direct JSON-lines response from the service."""

    snapshot_value = G1RunSnapshot.from_dict(snapshot)
    invocation_value = G1InvocationEnvelope.from_dict(invocation)
    bind_snapshot_and_invocation(snapshot_value, invocation_value)
    event_bytes, _artifact_bytes, _receipt_bytes = _policy_bounds(snapshot_value.raw)
    if isinstance(max_stream_bytes, bool) or not isinstance(max_stream_bytes, int) or max_stream_bytes < event_bytes:
        raise G1ContractError("max_stream_bytes is outside the runtime evidence bound")
    result: list[dict[str, Any]] = []
    seen_event_ids: set[str] = set()
    seen_idempotency_keys: set[str] = set()
    exit_seen = False
    expected_sequence = 0
    stream_bytes = 0
    for frame in frames:
        if exit_seen:
            raise G1ContractError("runtime emitted a frame after RuntimeExit")
        parsed = validate_g1_frame(frame, snapshot_value.raw, invocation_value.raw)
        stream_bytes += len(_canonical(parsed))
        if stream_bytes > max_stream_bytes:
            raise G1ContractError("runtime evidence stream exceeds the configured retained bound")
        if "trust" in parsed:
            if parsed["eventId"] in seen_event_ids:
                raise G1ContractError("runtime emitted a duplicate eventId")
            if parsed["idempotencyKey"] in seen_idempotency_keys:
                raise G1ContractError("runtime emitted a duplicate event idempotencyKey")
            seen_event_ids.add(parsed["eventId"])
            seen_idempotency_keys.add(parsed["idempotencyKey"])
            if parsed["sequence"] != expected_sequence:
                raise G1ContractError("runtime event sequence is out of order or gapped")
            expected_sequence += 1
        else:
            expected_final_sequence = expected_sequence - 1 if expected_sequence else 0
            if parsed["finalSequence"] != expected_final_sequence:
                raise G1ContractError("RuntimeExit.finalSequence does not match accepted event sequence")
            exit_seen = True
        result.append(parsed)
    if not exit_seen:
        raise G1ContractError("runtime response does not contain RuntimeExit")
    return tuple(result)


def _derived_ref(namespace: str, invocation_id: str, sequence: int, body: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(
        _canonical({"invocationId": invocation_id, "sequence": sequence, "body": body})
    ).hexdigest()[:24]
    return f"{namespace}:runtime-{digest}"


def build_event(
    *,
    snapshot: G1RunSnapshot,
    invocation: G1InvocationEnvelope,
    sequence: int,
    body: Mapping[str, Any],
) -> dict[str, Any]:
    """Build and validate one deterministic untrusted observation frame."""

    event_id = _derived_ref("event", invocation.invocation_id, sequence, body)
    frame = {
        "protocol": PROTOCOL,
        "trust": "untrusted",
        "workspaceRef": snapshot.workspace_ref,
        "actorRef": snapshot.actor_ref,
        "runId": snapshot.run_id,
        "invocationId": invocation.invocation_id,
        "sequence": sequence,
        "eventId": event_id,
        "idempotencyKey": _derived_ref("idempotency", invocation.invocation_id, sequence, body),
        "correlationId": invocation.correlation_id,
        "causationRef": invocation.causation_ref,
        # The immutable lease timestamp keeps deterministic replay stable while
        # still carrying the host-selected observation time through the wire.
        "observedAt": invocation.lease["expiresAt"],
        "body": dict(body),
    }
    event_bytes, artifact_bytes, _receipt_bytes = _policy_bounds(snapshot.to_dict())
    return _validate_event(
        frame,
        max_event_bytes=event_bytes,
        max_artifact_bytes=artifact_bytes,
    )


def build_exit(
    *,
    snapshot: G1RunSnapshot,
    invocation: G1InvocationEnvelope,
    final_sequence: int,
    kind: str,
    input_event_ref: str | None = None,
    failure: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build and validate one runtime-evidence-only terminal frame."""

    frame: dict[str, Any] = {
        "protocol": PROTOCOL,
        "authority": "runtime_evidence_only",
        "workspaceRef": snapshot.workspace_ref,
        "actorRef": snapshot.actor_ref,
        "runId": snapshot.run_id,
        "invocationId": invocation.invocation_id,
        "finalSequence": final_sequence,
        "idempotencyKey": invocation.idempotency_key,
        "correlationId": invocation.correlation_id,
        "causationRef": invocation.causation_ref,
        "kind": kind,
    }
    if input_event_ref is not None:
        frame["inputEventRef"] = input_event_ref
    if failure is not None:
        frame["failure"] = dict(failure)
    return _validate_exit(frame)


__all__ = [
    "G1_CONTRACT_DIGESTS",
    "G1_MANIFEST_DIGEST",
    "G1ContractError",
    "G1InvocationEnvelope",
    "G1RunSnapshot",
    "PROTOCOL",
    "RUNTIME_FAILURE_CAUSES",
    "bind_snapshot_and_invocation",
    "build_event",
    "build_exit",
    "content_digest",
    "snapshot_digest",
    "validate_eager_input_schema",
    "validate_g1_frame",
    "validate_g1_frames",
]
