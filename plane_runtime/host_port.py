"""Credential-free Plane host callbacks for the Hermes kernel.

This module is the narrow Hermes-side seam for ``plane.agent-runtime/v1``.
It contains no Plane application imports and no Plane credential handling.  A
trusted host owns the callable transport and binds identity, authorization,
and credentials on the other side of that transport.  Hermes exposes only
bounded, invocation-scoped tool callbacks to the existing agent loop.

The registered tools deliberately live in a dynamic ``plane_runtime``
toolset.  They are installed only by :class:`HermesKernelAdapter`, and their
handlers resolve a context-local port, so another Hermes conversation cannot
reuse this invocation's host binding.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Mapping, Protocol


HOST_PROTOCOL = "plane.agent-runtime/v1"
PLANE_RUNTIME_TOOLSET = "plane_runtime"
PLANE_OPERATION_TOOL = "plane_operation"
PLANE_PUBLISH_TOOL = "plane_publish"
PLANE_DISCOVERY_OPERATION = "plane.operations.discover@1"

MAX_HOST_REQUEST_BYTES = 16 * 1024
MAX_HOST_RESULT_BYTES = 16 * 1024
MAX_HOST_INPUT_BYTES = 8 * 1024
MAX_HOST_RESULT_TEXT_BYTES = 12 * 1024
MAX_HOST_CALLS = 32
MAX_HOST_OPERATION_REF_BYTES = 256
MAX_HOST_CONTENT_BYTES = 4 * 1024

_ACTIONS = {"discover", "read", "mutate", "code", "publish"}
_SOURCES = {"model", "code"}
_RESULT_STATUSES = {
    "ok",
    "replayed",
    "denied",
    "conflict",
    "unavailable",
    "invalid",
}


class PlaneHostError(ValueError):
    """Base class for a malformed, unavailable, or rejected host callback."""


class PlaneHostUnavailable(PlaneHostError):
    """The trusted host transport could not produce a usable response."""


class PlaneHostCancelled(PlaneHostError):
    """The invocation was cancelled before a host callback could run."""


class PlaneHostBoundsError(PlaneHostError):
    """A host callback exceeded the invocation-local bound."""


def _canonical(value: Any, name: str) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PlaneHostError(f"{name} is not JSON-compatible") from exc


def _bounded_json(value: Any, name: str, maximum: int) -> bytes:
    encoded = _canonical(value, name)
    if len(encoded) > maximum:
        raise PlaneHostBoundsError(
            f"{name} exceeds {maximum} canonical UTF-8 bytes"
        )
    return encoded


def _text(value: Any, name: str, maximum: int) -> str:
    if not isinstance(value, str) or not value:
        raise PlaneHostError(f"{name} must be a non-empty string")
    if len(value.encode("utf-8")) > maximum:
        raise PlaneHostBoundsError(f"{name} exceeds {maximum} UTF-8 bytes")
    return value


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(k, str) for k in value):
        raise PlaneHostError(f"{name} must be an object")
    return dict(value)


def _reject_unknown(value: Mapping[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(set(value).difference(allowed))
    if unknown:
        raise PlaneHostError(f"{name} has unknown field(s): {', '.join(unknown)}")


def _strict_wire_object(value: Any, name: str) -> dict[str, Any]:
    """Parse a host response and reject duplicate, unknown JSON structure later."""

    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise PlaneHostUnavailable(f"{name} is not UTF-8") from exc
    if isinstance(value, str):
        raw_bytes = value.encode("utf-8")
        if len(raw_bytes) > MAX_HOST_RESULT_BYTES or value != value.strip():
            raise PlaneHostUnavailable(f"{name} is not canonical")
        try:
            value = json.loads(value, object_pairs_hook=_duplicate_rejecting_pairs)
        except (TypeError, json.JSONDecodeError) as exc:
            raise PlaneHostUnavailable(f"{name} is malformed JSON") from exc
        if _canonical(value, name) != raw_bytes:
            raise PlaneHostUnavailable(f"{name} is not canonical")
    return _object(value, name)


def _duplicate_rejecting_pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise json.JSONDecodeError("duplicate object key", "", 0)
        result[key] = value
    return result


def _digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical(value, "host request")).hexdigest()


@dataclass(frozen=True)
class HostCallRequest:
    """The complete host request derived from model input and trusted context."""

    run_id: str
    invocation_id: str
    correlation_id: str
    action: str
    operation_ref: str
    input: Mapping[str, Any]
    source: str
    request_ref: str = field(init=False)
    idempotency_key: str = field(init=False)

    def __post_init__(self) -> None:
        run_id = _text(self.run_id, "host.runId", 256)
        invocation_id = _text(self.invocation_id, "host.invocationId", 256)
        correlation_id = _text(self.correlation_id, "host.correlationId", 256)
        operation_ref = _text(
            self.operation_ref, "host.operationRef", MAX_HOST_OPERATION_REF_BYTES
        )
        if self.action not in _ACTIONS:
            raise PlaneHostError(f"unsupported host action: {self.action!r}")
        if self.source not in _SOURCES:
            raise PlaneHostError(f"unsupported host source: {self.source!r}")
        payload = _object(self.input, "host.input")
        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(self, "invocation_id", invocation_id)
        object.__setattr__(self, "correlation_id", correlation_id)
        object.__setattr__(self, "operation_ref", operation_ref)
        object.__setattr__(self, "input", payload)
        identity = {
            "protocol": HOST_PROTOCOL,
            "runId": run_id,
            "invocationId": invocation_id,
            "action": self.action,
            "operationRef": operation_ref,
            "input": payload,
        }
        digest = _digest(identity)
        object.__setattr__(self, "request_ref", f"host-request:{digest}")
        object.__setattr__(self, "idempotency_key", f"host-idempotency:{digest}")
        _bounded_json(self.to_dict(), "host.request", MAX_HOST_REQUEST_BYTES)

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": HOST_PROTOCOL,
            "runId": self.run_id,
            "invocationId": self.invocation_id,
            "correlationId": self.correlation_id,
            "action": self.action,
            "operationRef": self.operation_ref,
            "input": dict(self.input),
            "source": self.source,
            "requestRef": self.request_ref,
            "idempotencyKey": self.idempotency_key,
        }


@dataclass(frozen=True)
class HostCallResult:
    """Strict, bounded result returned by the trusted Plane host."""

    request_ref: str
    correlation_id: str
    idempotency_key: str
    status: str
    replayed: bool
    output: Any = None
    error_code: str | None = None
    error_message: str | None = None
    publication: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        _text(self.request_ref, "host.result.requestRef", 256)
        _text(self.correlation_id, "host.result.correlationId", 256)
        _text(self.idempotency_key, "host.result.idempotencyKey", 256)
        if self.status not in _RESULT_STATUSES:
            raise PlaneHostUnavailable(f"unsupported host result status: {self.status!r}")
        if not isinstance(self.replayed, bool):
            raise PlaneHostUnavailable("host.result.replayed must be a boolean")
        if self.status == "replayed" and not self.replayed:
            raise PlaneHostUnavailable("replayed host result must set replayed=true")
        if self.status in {"denied", "conflict", "unavailable", "invalid"}:
            _text(self.error_code, "host.result.errorCode", 128)
            _text(self.error_message, "host.result.errorMessage", 2048)
        elif self.error_code is not None or self.error_message is not None:
            raise PlaneHostUnavailable("successful host result cannot carry an error")
        if self.publication is not None:
            _object(self.publication, "host.result.publication")
        _bounded_json(self.to_dict(), "host.result", MAX_HOST_RESULT_BYTES)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "protocol": HOST_PROTOCOL,
            "requestRef": self.request_ref,
            "correlationId": self.correlation_id,
            "idempotencyKey": self.idempotency_key,
            "status": self.status,
            "replayed": self.replayed,
            "output": self.output,
        }
        if self.error_code is not None:
            data["errorCode"] = self.error_code
        if self.error_message is not None:
            data["errorMessage"] = self.error_message
        if self.publication is not None:
            data["publication"] = dict(self.publication)
        return data

    @classmethod
    def from_wire(cls, raw: Any) -> "HostCallResult":
        data = _strict_wire_object(raw, "host.result")
        _reject_unknown(
            data,
            {
                "protocol",
                "requestRef",
                "correlationId",
                "idempotencyKey",
                "status",
                "replayed",
                "output",
                "errorCode",
                "errorMessage",
                "publication",
            },
            "host.result",
        )
        required = {
            "protocol",
            "requestRef",
            "correlationId",
            "idempotencyKey",
            "status",
            "replayed",
            "output",
        }
        if not required.issubset(data):
            missing = sorted(required.difference(data))
            raise PlaneHostUnavailable(
                f"host.result is missing field(s): {', '.join(missing)}"
            )
        if data["protocol"] != HOST_PROTOCOL:
            raise PlaneHostUnavailable("host.result protocol is unsupported")
        publication = data.get("publication")
        if publication is not None:
            publication = _object(publication, "host.result.publication")
        result = cls(
            request_ref=_text(data["requestRef"], "host.result.requestRef", 256),
            correlation_id=_text(data["correlationId"], "host.result.correlationId", 256),
            idempotency_key=_text(data["idempotencyKey"], "host.result.idempotencyKey", 256),
            status=data["status"],
            replayed=data["replayed"],
            output=data["output"],
            error_code=data.get("errorCode"),
            error_message=data.get("errorMessage"),
            publication=publication,
        )
        _bounded_json(result.to_dict(), "host.result", MAX_HOST_RESULT_BYTES)
        return result

    def model_payload(self) -> str:
        """Return only bounded host output to the model/tool transcript."""

        payload = self.to_dict()
        encoded = _bounded_json(payload, "host.modelResult", MAX_HOST_RESULT_TEXT_BYTES)
        return encoded.decode("utf-8")


class PlaneHostRpc(Protocol):
    """Transport-only host callable; implementations live outside Hermes."""

    def __call__(self, request: Mapping[str, Any]) -> Mapping[str, Any] | str | bytes:
        ...


class PlaneHostPort(Protocol):
    """Logical host port used by Hermes tools and restricted code execution."""

    def invoke(self, request: HostCallRequest) -> HostCallResult:
        ...


class CallablePlaneHostPort:
    """Validate a host RPC result without importing or knowing Plane code."""

    def __init__(self, rpc: PlaneHostRpc) -> None:
        if not callable(rpc):
            raise TypeError("Plane host RPC must be callable")
        self._rpc = rpc

    def invoke(self, request: HostCallRequest) -> HostCallResult:
        if not isinstance(request, HostCallRequest):
            raise PlaneHostUnavailable("host request is not a validated HostCallRequest")
        try:
            raw = self._rpc(request.to_dict())
            result = HostCallResult.from_wire(raw)
        except PlaneHostError:
            raise
        except Exception as exc:
            raise PlaneHostUnavailable("Plane host RPC was unavailable") from exc
        if (
            result.request_ref != request.request_ref
            or result.correlation_id != request.correlation_id
            or result.idempotency_key != request.idempotency_key
        ):
            raise PlaneHostUnavailable("host result is not bound to the request")
        return result


@dataclass(frozen=True)
class HostCallRecord:
    request: HostCallRequest
    result: HostCallResult


@dataclass
class PlaneHostBinding:
    """One invocation's port, cancellation, event sink, and call budget."""

    port: PlaneHostPort
    run_id: str
    invocation_id: str
    correlation_id: str
    cancellation: Callable[[], bool]
    emit_body: Callable[[Mapping[str, Any]], None] | None = None
    max_calls: int = MAX_HOST_CALLS
    records: list[HostCallRecord] = field(default_factory=list)
    _fatal_error: str | None = field(default=None, init=False, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)

    def __post_init__(self) -> None:
        if not callable(getattr(self.port, "invoke", None)):
            raise TypeError("Plane host port must expose invoke()")
        if not callable(self.cancellation):
            raise TypeError("Plane host cancellation must be callable")
        if (
            isinstance(self.max_calls, bool)
            or not isinstance(self.max_calls, int)
            or self.max_calls <= 0
        ):
            raise ValueError("Plane host max_calls must be a positive integer")

    @property
    def fatal_error(self) -> str | None:
        return self._fatal_error

    @property
    def publication_count(self) -> int:
        return sum(1 for item in self.records if item.request.action == "publish")

    def _fail(self, message: str) -> None:
        if self._fatal_error is None:
            self._fatal_error = message[:2048]

    def _is_cancelled(self) -> bool:
        try:
            cancelled = self.cancellation()
        except Exception as exc:
            self._fail("cancellation signal failed")
            raise PlaneHostUnavailable("cancellation signal failed") from exc
        if not isinstance(cancelled, bool):
            self._fail("cancellation signal was not boolean")
            raise PlaneHostUnavailable("cancellation signal was not boolean")
        return cancelled

    def call(
        self,
        *,
        action: str,
        operation_ref: str,
        input: Mapping[str, Any],
        source: str,
    ) -> HostCallResult:
        with self._lock:
            if self._is_cancelled():
                self._fail("Plane host callback cancelled")
                raise PlaneHostCancelled("Plane host callback cancelled")
            if len(self.records) >= self.max_calls:
                self._fail("Plane host call budget exhausted")
                raise PlaneHostBoundsError("Plane host call budget exhausted")
            try:
                request = HostCallRequest(
                    run_id=self.run_id,
                    invocation_id=self.invocation_id,
                    correlation_id=self.correlation_id,
                    action=action,
                    operation_ref=operation_ref,
                    input=input,
                    source=source,
                )
                _bounded_json(request.input, "host.input", MAX_HOST_INPUT_BYTES)
            except PlaneHostError as exc:
                self._fail(str(exc) or "Plane host request was invalid")
                raise
            try:
                result = self.port.invoke(request)
            except PlaneHostError as exc:
                self._fail(str(exc) or "Plane host callback failed")
                raise
            except Exception as exc:
                self._fail("Plane host callback failed")
                raise PlaneHostUnavailable("Plane host callback failed") from exc
            self.records.append(HostCallRecord(request, result))
            self._emit_call_observation(request, result)
            if result.status in {"unavailable", "invalid", "conflict"}:
                self._fail(result.error_message or "Plane host rejected the callback")
            if self._is_cancelled():
                self._fail("Plane host callback cancelled")
                raise PlaneHostCancelled("Plane host callback cancelled")
            return result

    def publish(
        self,
        *,
        kind: str,
        operation_ref: str,
        resource_ref: str,
        content: str,
    ) -> HostCallResult:
        try:
            if kind not in {"conversation", "outcome"}:
                raise PlaneHostError("publication kind must be conversation or outcome")
            operation_ref = _text(
                operation_ref,
                "publication.operationRef",
                MAX_HOST_OPERATION_REF_BYTES,
            )
            resource_ref = _text(resource_ref, "publication.resourceRef", 256)
            content = _text(content, "publication.content", MAX_HOST_CONTENT_BYTES)
        except PlaneHostError as exc:
            self._fail(str(exc) or "publication request was invalid")
            raise
        result = self.call(
            action="publish",
            operation_ref=operation_ref,
            input={"kind": kind, "resourceRef": resource_ref, "content": content},
            source="model",
        )
        if result.status not in {"ok", "replayed"}:
            self._fail("explicit publication was not authorized by the Plane host")
            return result
        if result.publication is None:
            self._fail("explicit publication has no gateway publication receipt")
            raise PlaneHostUnavailable("explicit publication has no gateway publication receipt")
        try:
            publication = _validated_publication(
                result.publication,
                kind=kind,
                resource_ref=resource_ref,
            )
        except PlaneHostError as exc:
            self._fail(str(exc) or "explicit publication receipt was invalid")
            raise PlaneHostUnavailable("explicit publication receipt was invalid") from exc
        if self.emit_body is not None:
            try:
                self.emit_body(
                    {
                        "kind": (
                            "conversation_publication_observed"
                            if kind == "conversation"
                            else "outcome_submission_observed"
                        ),
                        "payload": {
                            "kind": "inline_text",
                            "contentType": "text/plain",
                            "text": content,
                        },
                        "publication": publication,
                    }
                )
            except Exception as exc:
                self._fail("Plane publication observation could not be emitted")
                raise PlaneHostUnavailable(
                    "Plane publication observation could not be emitted"
                ) from exc
        return result

    def _emit_call_observation(
        self, request: HostCallRequest, result: HostCallResult
    ) -> None:
        if self.emit_body is None:
            return
        message = (
            f"Plane host {request.source} {request.action} "
            f"{request.operation_ref} -> {result.status}"
        )
        try:
            self.emit_body(
                {
                    "kind": "progress_observed",
                    "payload": {
                        "kind": "inline_text",
                        "contentType": "text/plain",
                        "text": message[:4096],
                    },
                    "publication": {"action": "observation_only"},
                }
            )
        except Exception as exc:
            self._fail("Plane host observation could not be emitted")
            raise PlaneHostUnavailable("Plane host observation could not be emitted") from exc


def _validated_publication(
    value: Mapping[str, Any], *, kind: str, resource_ref: str
) -> dict[str, Any]:
    data = _object(value, "host.result.publication")
    product_kind = "conversation" if kind == "conversation" else "outcome_submission"
    required = {"action", "productKind", "productRef", "operationAttemptRef"}
    if data.get("action") == "applied":
        required |= {
            "operationRef",
            "applicationServiceRef",
            "gatewayReceiptRef",
            "receiptRef",
            "auditReceiptRef",
            "productEventRef",
        }
    _reject_unknown(data, required, "host.result.publication")
    if not required.issubset(data):
        missing = sorted(required.difference(data))
        raise PlaneHostUnavailable(
            f"host.result.publication is missing field(s): {', '.join(missing)}"
        )
    if data["productKind"] != product_kind:
        raise PlaneHostUnavailable("host publication kind is not bound to the action")
    if data["productRef"] != resource_ref:
        raise PlaneHostUnavailable("host publication is not bound to the requested resource")
    if data["action"] not in {"proposal", "applied"}:
        raise PlaneHostUnavailable("host publication action is unsupported")
    if kind == "outcome" and data["action"] != "applied":
        raise PlaneHostUnavailable("outcome publication must be gateway-applied")
    namespaces = {
        "productRef": "conversation" if kind == "conversation" else "outcome-submission",
        "operationAttemptRef": "operation-attempt",
    }
    if data["action"] == "applied":
        namespaces.update(
            {
                "operationRef": "operation",
                "applicationServiceRef": "application-service",
                "gatewayReceiptRef": "gateway-receipt",
                "receiptRef": "receipt",
                "auditReceiptRef": "audit-receipt",
                "productEventRef": "product-event",
            }
        )
    for field_name, namespace in namespaces.items():
        ref = _text(data[field_name], f"publication.{field_name}", 256)
        if not ref.startswith(f"{namespace}:"):
            raise PlaneHostUnavailable(
                f"publication.{field_name} must use {namespace}: namespace"
            )
    return data


_CURRENT_BINDING: contextvars.ContextVar[PlaneHostBinding | None] = contextvars.ContextVar(
    "plane_runtime_host_binding", default=None
)
_PLANE_CODE_MODE: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "plane_runtime_code_mode", default=False
)


def current_plane_host() -> PlaneHostBinding | None:
    return _CURRENT_BINDING.get()


@contextmanager
def plane_code_mode() -> Iterator[None]:
    """Mark one existing execute_code parent-RPC dispatch as Code Mode."""

    token = _PLANE_CODE_MODE.set(True)
    try:
        yield
    finally:
        _PLANE_CODE_MODE.reset(token)


@contextmanager
def bind_plane_host(binding: PlaneHostBinding) -> Iterator[PlaneHostBinding]:
    if not isinstance(binding, PlaneHostBinding):
        raise TypeError("Plane host binding must be a PlaneHostBinding")
    token = _CURRENT_BINDING.set(binding)
    try:
        yield binding
    finally:
        _CURRENT_BINDING.reset(token)


def _error_payload(message: str, *, code: str = "plane_host_error") -> str:
    return json.dumps(
        {"status": "error", "error": {"code": code, "message": message[:2048]}},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _binding_or_error() -> PlaneHostBinding:
    binding = current_plane_host()
    if binding is None:
        raise PlaneHostUnavailable("Plane host is not bound to this invocation")
    return binding


def _handle_plane_operation(args: Mapping[str, Any], **_: Any) -> str:
    try:
        data = _object(args, "plane_operation")
        _reject_unknown(data, {"action", "operationRef", "input"}, "plane_operation")
        action = _text(data.get("action"), "plane_operation.action", 32)
        if action not in {"discover", "read", "mutate", "code"}:
            raise PlaneHostError("plane_operation.action is unsupported")
        operation_ref = data.get("operationRef") or (
            PLANE_DISCOVERY_OPERATION if action == "discover" else None
        )
        operation_ref = _text(
            operation_ref, "plane_operation.operationRef", MAX_HOST_OPERATION_REF_BYTES
        )
        input_value = _object(data.get("input", {}), "plane_operation.input")
        if action == "code" and not _PLANE_CODE_MODE.get():
            raise PlaneHostError("plane_operation code action is restricted to execute_code")
        result = _binding_or_error().call(
            action=action,
            operation_ref=operation_ref,
            input=input_value,
            source="model" if action != "code" else "code",
        )
        return result.model_payload()
    except PlaneHostCancelled as exc:
        return _error_payload(str(exc), code="cancelled")
    except PlaneHostError as exc:
        return _error_payload(str(exc))


def _handle_plane_publish(args: Mapping[str, Any], **_: Any) -> str:
    try:
        data = _object(args, "plane_publish")
        _reject_unknown(
            data,
            {"kind", "operationRef", "resourceRef", "content"},
            "plane_publish",
        )
        kind = _text(data.get("kind"), "plane_publish.kind", 32)
        operation_ref = _text(
            data.get("operationRef"),
            "plane_publish.operationRef",
            MAX_HOST_OPERATION_REF_BYTES,
        )
        resource_ref = _text(data.get("resourceRef"), "plane_publish.resourceRef", 256)
        content = _text(data.get("content"), "plane_publish.content", MAX_HOST_CONTENT_BYTES)
        result = _binding_or_error().publish(
            kind=kind,
            operation_ref=operation_ref,
            resource_ref=resource_ref,
            content=content,
        )
        return result.model_payload()
    except PlaneHostCancelled as exc:
        return _error_payload(str(exc), code="cancelled")
    except PlaneHostError as exc:
        return _error_payload(str(exc))


_INSTALL_LOCK = threading.Lock()
_INSTALLED = False


def install_plane_tools() -> None:
    """Install the narrow dynamic toolset into Hermes' existing registry."""

    global _INSTALLED
    with _INSTALL_LOCK:
        if _INSTALLED:
            return
        from tools.registry import registry

        registry.register(
            PLANE_OPERATION_TOOL,
            PLANE_RUNTIME_TOOLSET,
            {
                "name": PLANE_OPERATION_TOOL,
                "description": (
                    "Discover or invoke one Plane operation through the trusted "
                    "host gateway. Results are bounded and remain untrusted."
                ),
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["discover", "read", "mutate", "code"],
                        },
                        "operationRef": {"type": "string"},
                        "input": {"type": "object"},
                    },
                    "required": ["action"],
                },
            },
            _handle_plane_operation,
            description="Plane host operation discovery/read/mutation callback",
            max_result_size_chars=MAX_HOST_RESULT_TEXT_BYTES,
        )
        registry.register(
            PLANE_PUBLISH_TOOL,
            PLANE_RUNTIME_TOOLSET,
            {
                "name": PLANE_PUBLISH_TOOL,
                "description": (
                    "Explicitly ask the Plane host to publish a conversation or "
                    "outcome. Ordinary final text never calls this implicitly."
                ),
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "kind": {"type": "string", "enum": ["conversation", "outcome"]},
                        "operationRef": {"type": "string"},
                        "resourceRef": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["kind", "operationRef", "resourceRef", "content"],
                },
            },
            _handle_plane_publish,
            description="Explicit Plane product publication/outcome proposal",
            max_result_size_chars=MAX_HOST_RESULT_TEXT_BYTES,
        )
        _INSTALLED = True


__all__ = [
    "CallablePlaneHostPort",
    "HostCallRecord",
    "HostCallRequest",
    "HostCallResult",
    "PlaneHostBinding",
    "PlaneHostBoundsError",
    "PlaneHostCancelled",
    "PlaneHostError",
    "PlaneHostPort",
    "PlaneHostUnavailable",
    "bind_plane_host",
    "current_plane_host",
    "install_plane_tools",
    "plane_code_mode",
]
