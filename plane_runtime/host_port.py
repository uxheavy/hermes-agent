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
import errno
import hashlib
import json
import math
import select
import socket
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Callable, Iterator, Literal, Mapping, Protocol

from .g1_contract import CODE_MODE_PHASES, G1ContractError, validate_eager_input_schema


HOST_PROTOCOL = "plane.agent-runtime/v1"
PLANE_OPERATION_TOOLSET = "plane_runtime_operations"
PLANE_PUBLICATION_TOOLSET = "plane_runtime_publication"
PLANE_CODE_MODE_TOOLSET = "plane_runtime_code_mode"
PLANE_OPERATION_TOOL = "plane_operation"
PLANE_PUBLISH_TOOL = "plane_publish"
PLANE_CODE_MODE_TOOL = "plane_execute_typescript"
PLANE_CODE_MODE_SCHEMA_VERSION = "plane.code-mode/v1"
PLANE_CODE_MODE_EXECUTE_OPERATION = "plane.code-mode.execute@1"
PLANE_OUTCOME_PUBLISH_OPERATION = "operation:agent.outcome.publish"
PLANE_DISCOVERY_OPERATION = "plane.operations.discover@1"
PLANE_CATALOG_SEARCH_OPERATION = "operation:catalog.search"
PLANE_CATALOG_DESCRIBE_OPERATION = "operation:catalog.describe"

MAX_HOST_REQUEST_BYTES = 16 * 1024
MAX_HOST_RESULT_BYTES = 16 * 1024
MAX_HOST_INPUT_BYTES = 8 * 1024
MAX_HOST_RESULT_TEXT_BYTES = 12 * 1024
MAX_CODE_MODE_SOURCE_BYTES = 4 * 1024
MAX_HOST_CALLS = 32
MAX_HOST_OPERATION_REF_BYTES = 256
MAX_HOST_CONTENT_BYTES = 4 * 1024

HOST_CALLBACK_PHASES = frozenset(
    {
        "before_host_call",
        "host_return",
        "model_observation_emit",
        "adapter_event",
    }
)

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

_CODE_MODE_HOST_STATUSES = frozenset(_RESULT_STATUSES)
_CODE_MODE_CONTRACT_ERRORS = frozenset(
    {
        "VALIDATION_ERROR",
        "PREPARED_CALL_INVALID",
        "PROTOCOL_ERROR",
        "SOURCE_TOO_LARGE",
        "BUDGET_EXCEEDED",
    }
)

HostResultDisposition = Literal[
    "continue_with_tool_result",
    "poison_invocation",
]

_HOST_RESULT_DISPOSITIONS: Mapping[tuple[str, str | None], HostResultDisposition] = MappingProxyType(
    {
        ("ok", None): "continue_with_tool_result",
        ("replayed", None): "continue_with_tool_result",
        ("invalid", "VALIDATION_ERROR"): "continue_with_tool_result",
        # A generated Code Mode module may fail in the restricted isolate.
        # Return that finite, bounded result to the model so it can correct
        # the module in the same invocation; host/transport failures remain
        # poison unless explicitly classified above.
        ("invalid", "CODE_MODE_FAILED"): "continue_with_tool_result",
        ("denied", "NOT_AUTHORIZED"): "continue_with_tool_result",
        ("conflict", "PLANE_CONFLICT"): "continue_with_tool_result",
    }
)


class PlaneHostError(ValueError):
    """Base class for a malformed, unavailable, or rejected host callback."""


class PlaneHostUnavailable(PlaneHostError):
    """The trusted host transport could not produce a usable response."""


class PlaneHostCancelled(PlaneHostError):
    """The invocation was cancelled before a host callback could run."""


class PlaneHostBoundsError(PlaneHostError):
    """A host callback exceeded the invocation-local bound."""


class PlaneHostSchemaNotDisclosed(PlaneHostError):
    """An operation needs progressive schema disclosure before invocation."""


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


def _operation_ref_digest(operation_ref: str) -> str:
    return hashlib.sha256(operation_ref.encode("utf-8")).hexdigest()


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
            result = _bound_host_result(request, raw)
        except PlaneHostError:
            raise
        except Exception as exc:
            raise PlaneHostUnavailable("Plane host RPC was unavailable") from exc
        return result


def _bound_host_result(request: HostCallRequest, raw: Any) -> HostCallResult:
    result = HostCallResult.from_wire(raw)
    if (
        result.request_ref != request.request_ref
        or result.correlation_id != request.correlation_id
        or result.idempotency_key != request.idempotency_key
    ):
        raise PlaneHostUnavailable("host result is not bound to the request")
    return result


def _host_result_disposition(result: HostCallResult) -> HostResultDisposition:
    return _HOST_RESULT_DISPOSITIONS.get(
        (result.status, result.error_code), "poison_invocation"
    )


def _prepared_read_refs_from_search_result(output: Any) -> tuple[str, ...]:
    """Return the one opaque read reference prepared for a model search.

    Plane owns the target behind the reference. Hermes only recognizes the
    model-facing envelope and never reads or reconstructs target identifiers.
    A search with more than one prepared item is intentionally left to the
    model because there is no safe single handoff to choose.
    """

    if not isinstance(output, Mapping):
        return ()
    result = output.get("result")
    if isinstance(output.get("results"), list):
        result = output
    if not isinstance(result, Mapping):
        return ()
    items = result.get("results")
    if not isinstance(items, list):
        return ()
    prepared_refs: list[str] = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        call = item.get("workItemReadCall")
        if not isinstance(call, Mapping):
            prepared_ref = _opaque_prepared_ref(call)
            if prepared_ref is None:
                return ()
            prepared_refs.append(prepared_ref)
            continue
        if set(call) == {"preparedCallRef"}:
            prepared_ref = _opaque_prepared_ref(call.get("preparedCallRef"))
        elif (
            call.get("action") == "read"
            and call.get("operationRef") == "operation:work_item.read"
        ):
            input_value = call.get("input")
            prepared_ref = (
                _opaque_prepared_ref(input_value.get("preparedCallRef"))
                if isinstance(input_value, Mapping)
                and set(input_value) == {"preparedCallRef"}
                else None
            )
        else:
            prepared_ref = None
        if prepared_ref is None:
            return ()
        prepared_refs.append(prepared_ref)
    return tuple(prepared_refs)


def _prepared_read_ref_from_search_result(output: Any) -> str | None:
    """Return a single prepared read ref for compatibility with callers."""

    refs = _prepared_read_refs_from_search_result(output)
    return refs[0] if len(refs) == 1 else None


def _prepared_read_refs_from_code_mode_result(output: Any) -> tuple[str, ...]:
    """Return prepared reads left unconsumed by one trusted Code Mode result.

    Plane's Code Mode envelope carries the module result separately from its
    bounded operation observations.  Require both pieces before arming the
    Hermes continuation: a search observation proves which operation produced
    the result, while the absence of a successful read observation proves the
    opaque prepared call was not consumed by that same Code Mode turn.
    """

    if not isinstance(output, Mapping):
        return ()
    observations = output.get("observations")
    if not isinstance(observations, list):
        return ()
    search_observed = False
    read_consumed = False
    for observation in observations:
        if not isinstance(observation, Mapping):
            return ()
        if observation.get("source") != "code" or observation.get("action") != "code":
            return ()
        operation_ref = observation.get("operationRef")
        status = observation.get("status")
        if operation_ref == "operation:search_workspace" and status in {"ok", "replayed"}:
            search_observed = True
        elif operation_ref == "operation:work_item.read" and status in {"ok", "replayed"}:
            read_consumed = True
    if not search_observed or read_consumed:
        return ()
    return _prepared_read_refs_from_search_result(output.get("result"))


def _opaque_prepared_ref(value: Any) -> str | None:
    if (
        not isinstance(value, str)
        or not value.startswith("prepared-call:")
        or len(value.encode("utf-8")) > 256
    ):
        return None
    return value


def _ready_to_call_prepared_ref(value: Any) -> str | None:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"action", "operationRef", "input"}
        or value.get("action") != "read"
        or value.get("operationRef") != "operation:work_item.read"
    ):
        return None
    nested_input = value.get("input")
    if not isinstance(nested_input, Mapping) or set(nested_input) != {"preparedCallRef"}:
        return None
    return _opaque_prepared_ref(nested_input.get("preparedCallRef"))


def _wrapped_ready_to_call_prepared_ref(value: Any) -> str | None:
    if isinstance(value, Mapping):
        return _ready_to_call_prepared_ref(value)
    if not isinstance(value, str) or not value.startswith("{"):
        return None
    if len(value.encode("utf-8")) > MAX_HOST_INPUT_BYTES:
        return None
    try:
        decoded = json.loads(value, object_pairs_hook=_duplicate_rejecting_pairs)
    except (TypeError, ValueError):
        return None
    return _ready_to_call_prepared_ref(decoded)


def _normalize_prepared_read_input(
    action: str, operation_ref: str, input_value: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Accept only finite, lossless shapes of the prepared read handoff.

    The model-facing search result can reach the host callback as the opaque
    reference itself, one ``input`` wrapper, the complete ready-to-call
    envelope, one named ``workItemReadCall`` wrapper, or that exact envelope
    wrapped under ``preparedCallRef`` as an object or JSON string. These
    branches are deliberately explicit rather than recursive: only the exact
    read binding and one bounded opaque reference are collapsed to the
    canonical input; every other shape remains untouched for Plane to reject.
    """

    if action != "read" or operation_ref != "operation:work_item.read":
        return input_value
    def canonical_ref(candidate: Any) -> Mapping[str, Any] | None:
        if (
            not isinstance(candidate, Mapping)
            or set(candidate) != {"preparedCallRef"}
            or not isinstance(candidate.get("preparedCallRef"), str)
            or not candidate["preparedCallRef"].startswith("prepared-call:")
            or len(candidate["preparedCallRef"].encode("utf-8")) > 256
        ):
            return None
        return {"preparedCallRef": candidate["preparedCallRef"]}

    def ready_envelope(candidate: Any) -> Mapping[str, Any] | None:
        if (
            not isinstance(candidate, Mapping)
            or set(candidate) != {"action", "operationRef", "input"}
            or candidate.get("action") != "read"
            or candidate.get("operationRef") != operation_ref
        ):
            return None
        return canonical_ref(candidate.get("input"))

    def wrapped_ready_envelope(candidate: Any) -> Mapping[str, Any] | None:
        prepared_ref = _wrapped_ready_to_call_prepared_ref(candidate)
        return canonical_ref({"preparedCallRef": prepared_ref}) if prepared_ref else None

    # The accepted forms are intentionally enumerated. In particular, a named
    # wrapper may contain only the ready envelope, not another wrapper.
    if set(input_value) == {"preparedCallRef"}:
        normalized = canonical_ref(input_value)
        if normalized is None:
            normalized = wrapped_ready_envelope(input_value["preparedCallRef"])
    elif set(input_value) == {"input"}:
        normalized = canonical_ref(input_value.get("input"))
    elif set(input_value) == {"action", "operationRef", "input"}:
        normalized = ready_envelope(input_value)
    elif set(input_value) == {"workItemReadCall"}:
        normalized = ready_envelope(input_value.get("workItemReadCall"))
    else:
        normalized = None
    return normalized if normalized is not None else input_value


class UnixSocketPlaneHostPort:
    """Invocation-scoped canonical JSONL client for the trusted Plane host.

    The endpoint is trusted bootstrap configuration, not part of the Plane
    request contract.  A fresh connection is used for every callback so the
    client retains no run-lifetime transport state.  The peer must return one
    canonical JSON object followed by one newline and close the connection.
    """

    _MAX_SOCKET_PATH_BYTES = 103
    _POLL_SECONDS = 0.05

    def __init__(
        self,
        socket_path: str,
        *,
        timeout_seconds: float = 2.0,
        cancellation: Callable[[], bool] | None = None,
    ) -> None:
        if (
            not isinstance(socket_path, str)
            or not socket_path.startswith("/")
            or not socket_path
            or any(ord(char) < 0x20 or ord(char) == 0x7F for char in socket_path)
            or len(socket_path.encode("utf-8")) > self._MAX_SOCKET_PATH_BYTES
        ):
            raise PlaneHostError("Plane host socket configuration is invalid")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(float(timeout_seconds))
            or timeout_seconds <= 0
            or timeout_seconds > 60
        ):
            raise PlaneHostError("Plane host socket timeout is invalid")
        if cancellation is not None and not callable(cancellation):
            raise PlaneHostError("Plane host cancellation signal is invalid")
        self._socket_path = socket_path
        self._timeout_seconds = float(timeout_seconds)
        self._cancellation = cancellation or (lambda: False)
        self._closed = False
        self._lock = threading.Lock()

    def __enter__(self) -> "UnixSocketPlaneHostPort":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def close(self) -> None:
        """Make later calls fail closed; no socket survives a callback."""

        with self._lock:
            self._closed = True

    def _check_state(self) -> None:
        if self._closed:
            raise PlaneHostUnavailable("Plane host RPC is closed")
        try:
            cancelled = self._cancellation()
        except Exception as exc:
            raise PlaneHostUnavailable("Plane host cancellation signal failed") from exc
        if type(cancelled) is not bool:
            raise PlaneHostUnavailable("Plane host cancellation signal was invalid")
        if cancelled:
            raise PlaneHostCancelled("Plane host callback cancelled")

    def _wait(self, channel: socket.socket, *, writable: bool, deadline: float) -> None:
        while True:
            self._check_state()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise PlaneHostUnavailable("Plane host RPC deadline exceeded")
            try:
                readable, writable_ready, _ = select.select(
                    [] if writable else [channel],
                    [channel] if writable else [],
                    [],
                    min(remaining, self._POLL_SECONDS),
                )
            except (OSError, ValueError) as exc:
                raise PlaneHostUnavailable("Plane host RPC wait failed") from exc
            if (writable and writable_ready) or (not writable and readable):
                return

    def _connect(self, channel: socket.socket, deadline: float) -> None:
        try:
            error = channel.connect_ex(self._socket_path)
        except OSError as exc:
            raise PlaneHostUnavailable("Plane host RPC was unavailable") from exc
        if error == 0:
            return
        if error not in {
            errno.EINPROGRESS,
            errno.EALREADY,
            errno.EWOULDBLOCK,
            errno.EAGAIN,
        }:
            raise PlaneHostUnavailable("Plane host RPC was unavailable")
        self._wait(channel, writable=True, deadline=deadline)
        try:
            error = channel.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
        except OSError as exc:
            raise PlaneHostUnavailable("Plane host RPC was unavailable") from exc
        if error:
            raise PlaneHostUnavailable("Plane host RPC was unavailable")

    def _send(self, channel: socket.socket, payload: bytes, deadline: float) -> None:
        offset = 0
        while offset < len(payload):
            self._check_state()
            try:
                sent = channel.send(payload[offset:])
            except BlockingIOError:
                self._wait(channel, writable=True, deadline=deadline)
                continue
            except OSError as exc:
                raise PlaneHostUnavailable("Plane host RPC was unavailable") from exc
            if sent <= 0:
                raise PlaneHostUnavailable("Plane host RPC peer closed")
            offset += sent

    def _receive(self, channel: socket.socket, deadline: float) -> bytes:
        response = bytearray()
        while True:
            self._check_state()
            if len(response) > MAX_HOST_RESULT_BYTES + 1:
                raise PlaneHostBoundsError("Plane host RPC result exceeded its bound")
            try:
                chunk = channel.recv(
                    min(4096, MAX_HOST_RESULT_BYTES + 2 - len(response))
                )
            except BlockingIOError:
                self._wait(channel, writable=False, deadline=deadline)
                continue
            except OSError as exc:
                raise PlaneHostUnavailable("Plane host RPC was unavailable") from exc
            if not chunk:
                break
            response.extend(chunk)
        if len(response) > MAX_HOST_RESULT_BYTES + 1 or response.count(b"\n") != 1 or not response.endswith(b"\n"):
            raise PlaneHostUnavailable("Plane host RPC response framing was invalid")
        return bytes(response[:-1])

    def invoke(self, request: HostCallRequest) -> HostCallResult:
        if not isinstance(request, HostCallRequest):
            raise PlaneHostUnavailable("host request is not a validated HostCallRequest")
        self._check_state()
        payload = _bounded_json(request.to_dict(), "host.request", MAX_HOST_REQUEST_BYTES) + b"\n"
        deadline = time.monotonic() + self._timeout_seconds
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as channel:
                channel.setblocking(False)
                self._connect(channel, deadline)
                self._send(channel, payload, deadline)
                try:
                    channel.shutdown(socket.SHUT_WR)
                except OSError:
                    pass
                raw = self._receive(channel, deadline)
        except (PlaneHostCancelled, PlaneHostBoundsError, PlaneHostUnavailable):
            raise
        except Exception as exc:
            raise PlaneHostUnavailable("Plane host RPC was unavailable") from exc
        return _bound_host_result(request, raw)


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
    diagnostic_callback: Callable[[Mapping[str, Any]], None] | None = None
    eager_operation_refs: frozenset[str] = field(default_factory=frozenset)
    max_calls: int = MAX_HOST_CALLS
    records: list[HostCallRecord] = field(default_factory=list)
    code_mode_phase: str = "none"
    described_operation_refs: set[str] = field(default_factory=set, init=False, repr=False)
    _fatal_error: str | None = field(default=None, init=False, repr=False)
    _fatal_error_after_terminal: bool = field(default=False, init=False, repr=False)
    _host_operation_diagnostic: dict[str, Any] | None = field(
        default=None, init=False, repr=False
    )
    _terminal_action_reason: str | None = field(default=None, init=False, repr=False)
    _terminal_action_result: HostCallResult | None = field(default=None, init=False, repr=False)
    _terminal_action_request: HostCallRequest | None = field(default=None, init=False, repr=False)
    _prepared_read_handoff_pending: bool = field(default=False, init=False, repr=False)
    _code_mode_phase_hint: str | None = field(default=None, init=False, repr=False)
    _code_mode_continuation_used: bool = field(default=False, init=False, repr=False)
    _outcome_publication_metadata: dict[str, Any] | None = field(
        default=None, init=False, repr=False
    )
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
        if any(
            not isinstance(operation_ref, str)
            or not operation_ref.startswith("operation:")
            for operation_ref in self.eager_operation_refs
        ):
            raise ValueError("Plane host eager operation refs must be operation references")
        if self.code_mode_phase not in CODE_MODE_PHASES:
            raise ValueError("Plane host Code Mode phase is unsupported")

    @property
    def fatal_error(self) -> str | None:
        return self._fatal_error

    @property
    def fatal_error_after_terminal(self) -> bool:
        """Whether the first fatal host result arrived after terminal state."""

        with self._lock:
            return self._fatal_error_after_terminal

    @property
    def host_operation_diagnostic(self) -> dict[str, Any] | None:
        """Return only bounded, non-secret facts about the last host callback."""

        with self._lock:
            if self._host_operation_diagnostic is None:
                return None
            return dict(self._host_operation_diagnostic)

    @property
    def publication_count(self) -> int:
        return sum(1 for item in self.records if item.request.action == "publish")

    def terminal_action_reason(self) -> str | None:
        """Return the one-shot terminal reason after its observation is emitted."""

        with self._lock:
            return self._terminal_action_reason

    def prepared_read_handoff_pending(self) -> bool:
        """Return whether a search produced an unconsumed prepared read."""

        with self._lock:
            return self._prepared_read_handoff_pending

    def code_mode_phase_hint(self) -> str | None:
        """Return the one trusted Code Mode phase hint, if armed."""

        with self._lock:
            return self._code_mode_phase_hint

    def take_code_mode_phase_hint(self) -> str | None:
        """Consume the finite Code Mode continuation hint once."""

        with self._lock:
            phase = self._code_mode_phase_hint
            if phase is not None:
                self._code_mode_phase_hint = None
                self._code_mode_continuation_used = True
            return phase

    def outcome_publication_metadata(self) -> dict[str, Any] | None:
        """Return the last validated outcome publication's bounded facts."""

        with self._lock:
            if self._outcome_publication_metadata is None:
                return None
            return dict(self._outcome_publication_metadata)

    def _fail(self, message: str) -> None:
        if self._fatal_error is None:
            self._fatal_error = message[:2048]
            self._fatal_error_after_terminal = self._terminal_action_reason is not None

    def _set_callback_phase(
        self, phase: str, request: HostCallRequest | None = None
    ) -> None:
        if phase not in HOST_CALLBACK_PHASES:
            raise ValueError("unsupported Plane host callback phase")
        if request is not None:
            operation_ref_digest = _operation_ref_digest(request.operation_ref)
        elif self._host_operation_diagnostic is not None:
            operation_ref_digest = self._host_operation_diagnostic["operationRefDigest"]
        else:
            return
        diagnostic = {
            "callbackPhase": phase,
            "operationRefDigest": operation_ref_digest,
        }
        self._host_operation_diagnostic = diagnostic
        if self.diagnostic_callback is not None:
            try:
                self.diagnostic_callback(diagnostic)
            except Exception:
                # Diagnostics are observational only and must never alter the
                # host authorization or callback result.
                pass

    def _set_code_mode_diagnostic(
        self,
        request: HostCallRequest,
        *,
        status: str,
        error_code: str | None = None,
    ) -> None:
        """Attach only finite Code Mode result facts to the host diagnostic."""

        if request.action != "code":
            return
        bounded_status = status if status in _CODE_MODE_HOST_STATUSES else "unavailable"
        if error_code == "CODE_MODE_FAILED":
            failure_class = "code_mode"
        elif error_code == "CALLBACK_FAILED":
            failure_class = "callback"
        elif error_code in _CODE_MODE_CONTRACT_ERRORS or bounded_status == "invalid":
            failure_class = "contract"
        elif bounded_status == "unavailable":
            failure_class = "transport"
        else:
            failure_class = "unknown"
        diagnostic = dict(self._host_operation_diagnostic or {})
        diagnostic["codeModeHostStatus"] = bounded_status
        diagnostic["codeModeFailureClass"] = failure_class
        self._host_operation_diagnostic = diagnostic
        if self.diagnostic_callback is not None:
            try:
                self.diagnostic_callback(diagnostic)
            except Exception:
                pass

    def _schema_is_disclosed(self, operation_ref: str) -> bool:
        return operation_ref in self.eager_operation_refs or operation_ref in self.described_operation_refs

    def _require_schema_disclosure(self, *, action: str, operation_ref: str) -> None:
        if action == "discover":
            return
        if action == "code" and operation_ref == PLANE_CODE_MODE_EXECUTE_OPERATION:
            return
        if operation_ref in {
            PLANE_CATALOG_SEARCH_OPERATION,
            PLANE_CATALOG_DESCRIBE_OPERATION,
        }:
            return
        if not self._schema_is_disclosed(operation_ref):
            raise PlaneHostSchemaNotDisclosed(
                "SCHEMA_NOT_DISCLOSED: use discovery, operation:catalog.search, "
                "then operation:catalog.describe for this operation before invocation"
            )

    def _record_catalog_description(self, request: HostCallRequest, result: HostCallResult) -> None:
        if request.operation_ref != PLANE_CATALOG_DESCRIBE_OPERATION:
            return
        if result.status not in {"ok", "replayed"}:
            return
        try:
            output = _object(result.output, "catalog.describe.output")
            operation = _object(output.get("operation"), "catalog.describe.output.operation")
            operation_ref = operation.get("operationRef")
            operation_id = operation.get("operationId")
            if isinstance(operation_ref, str):
                described_ref = operation_ref
            elif isinstance(operation_id, str):
                described_ref = f"operation:{operation_id}"
            else:
                raise G1ContractError("catalog.describe result has no operation identity")
            if not described_ref.startswith("operation:"):
                raise G1ContractError("catalog.describe result has an invalid operation identity")
            if isinstance(operation_id, str) and described_ref != f"operation:{operation_id}":
                raise G1ContractError("catalog.describe result operation identity does not match")
            requested_id = request.input.get("operation_id")
            if not isinstance(requested_id, str) or not requested_id:
                raise G1ContractError("catalog.describe request has no operation identity")
            requested_ref = (
                requested_id
                if requested_id.startswith("operation:")
                else f"operation:{requested_id}"
            )
            if described_ref != requested_ref:
                raise G1ContractError("catalog.describe result does not match the requested operation")
            validate_eager_input_schema(operation.get("inputSchema"), "catalog.describe.inputSchema")
        except (G1ContractError, PlaneHostError, TypeError) as exc:
            self._fail("catalog.describe result was malformed or mismatched")
            raise PlaneHostUnavailable("catalog.describe result was malformed or mismatched") from exc
        self.described_operation_refs.add(described_ref)

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

    @staticmethod
    def _outcome_publication_identity(
        request: HostCallRequest,
    ) -> tuple[str, str, str] | None:
        if request.operation_ref != PLANE_OUTCOME_PUBLISH_OPERATION:
            return None
        if request.action == "publish":
            kind = request.input.get("kind")
            resource_ref = request.input.get("resourceRef")
        else:
            kind = "outcome"
            resource_ref = request.input.get("outcome_ref")
            if resource_ref is None:
                resource_ref = request.input.get("resourceRef")
        content = request.input.get("content")
        if (
            kind != "outcome"
            or not isinstance(resource_ref, str)
            or not isinstance(content, str)
        ):
            return None
        return kind, resource_ref, content

    def _terminal_result_for(self, request: HostCallRequest) -> HostCallResult | None:
        """Reuse equal outcome receipts or expose a terminal conflict."""

        if self._terminal_action_reason is None:
            return None
        if self._terminal_action_result is None or self._terminal_action_request is None:
            self._fail("terminal publication receipt is unavailable")
            raise PlaneHostUnavailable("terminal publication receipt is unavailable")
        if self._outcome_publication_identity(request) == self._outcome_publication_identity(
            self._terminal_action_request
        ):
            return self._terminal_action_result
        return HostCallResult(
            request_ref=request.request_ref,
            correlation_id=request.correlation_id,
            idempotency_key=request.idempotency_key,
            status="conflict",
            replayed=False,
            output=None,
            error_code="PLANE_CONFLICT",
            error_message="the invocation already has an applied outcome publication",
        )

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
            self._require_schema_disclosure(action=action, operation_ref=operation_ref)
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
            terminal_result = self._terminal_result_for(request)
            if terminal_result is not None:
                return terminal_result
            if len(self.records) >= self.max_calls:
                self._fail("Plane host call budget exhausted")
                raise PlaneHostBoundsError("Plane host call budget exhausted")
            self._set_callback_phase("before_host_call", request)
            try:
                result = self.port.invoke(request)
            except PlaneHostError as exc:
                self._set_code_mode_diagnostic(
                    request, status="unavailable", error_code="HOST_UNAVAILABLE"
                )
                self._fail(str(exc) or "Plane host callback failed")
                raise
            except Exception as exc:
                self._set_code_mode_diagnostic(
                    request, status="unavailable", error_code="HOST_UNAVAILABLE"
                )
                self._fail("Plane host callback failed")
                raise PlaneHostUnavailable("Plane host callback failed") from exc
            self._set_callback_phase("host_return", request)
            self._set_code_mode_diagnostic(
                request, status=result.status, error_code=result.error_code
            )
            self.records.append(HostCallRecord(request, result))
            try:
                self._record_catalog_description(request, result)
            except Exception:
                self._set_callback_phase("adapter_event")
                raise
            self._emit_call_observation(request, result)
            try:
                self._observe_publication(request, result)
            except Exception:
                self._set_callback_phase("adapter_event")
                raise
            prepared_ref: str | None = None
            if (
                operation_ref == "operation:search_workspace"
                and result.status in {"ok", "replayed"}
            ):
                prepared_refs = _prepared_read_refs_from_search_result(result.output)
                if prepared_refs:
                    self._prepared_read_handoff_pending = True
                if len(prepared_refs) == 1:
                    prepared_ref = prepared_refs[0]
                if prepared_ref is not None:
                    # The gateway has already prepared and authorized this
                    # opaque handoff. Consume it through the same binding
                    # and port before the model can terminate on ordinary
                    # text. The bounded search result remains visible, with
                    # the canonical read receipt attached for the next model
                    # turn; no target identifier crosses this seam.
                    prepared_read = self.call(
                        action="read",
                        operation_ref="operation:work_item.read",
                        input={"preparedCallRef": prepared_ref},
                        source="model",
                    )
                    combined_output = dict(result.output) if isinstance(result.output, Mapping) else {}
                    combined_output["preparedReadResult"] = prepared_read.to_dict()
                    result = replace(result, output=combined_output)
                    if prepared_read.status in {"ok", "replayed"} and self.code_mode_phase == "post_search":
                        with self._lock:
                            self._code_mode_phase_hint = "post_search"
            if (
                operation_ref == "operation:work_item.read"
                and isinstance(input.get("preparedCallRef"), str)
                and result.status in {"ok", "replayed"}
            ):
                self._prepared_read_handoff_pending = False
            if action == "code":
                with self._lock:
                    self._code_mode_phase_hint = None
                    should_arm = (
                        operation_ref == PLANE_CODE_MODE_EXECUTE_OPERATION
                        and result.status in {"ok", "replayed"}
                        and self.code_mode_phase == "post_search"
                        and not self._code_mode_continuation_used
                    )
                if should_arm and _prepared_read_refs_from_code_mode_result(result.output):
                    with self._lock:
                        if not self._code_mode_continuation_used:
                            self._code_mode_phase_hint = "post_search"
            if _host_result_disposition(result) == "poison_invocation":
                self._fail(result.error_message or "Plane host rejected the callback")
            if self._is_cancelled():
                self._set_callback_phase("adapter_event")
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
            if (
                _host_result_disposition(result) == "continue_with_tool_result"
                and result.status in {"invalid", "conflict"}
            ):
                return result
            self._fail("explicit publication was not authorized by the Plane host")
            return result
        return result

    def _observe_publication(
        self, request: HostCallRequest, result: HostCallResult
    ) -> None:
        """Observe the versioned publication receipt from every host route."""

        if request.action == "publish":
            kind = request.input.get("kind")
            resource_ref = request.input.get("resourceRef")
            content = request.input.get("content")
            if kind not in {"conversation", "outcome"}:
                self._fail("publication kind must be conversation or outcome")
                raise PlaneHostUnavailable("publication kind must be conversation or outcome")
        elif request.operation_ref == PLANE_OUTCOME_PUBLISH_OPERATION:
            kind = "outcome"
            resource_ref = request.input.get("outcome_ref")
            if resource_ref is None:
                resource_ref = request.input.get("resourceRef")
            content = request.input.get("content")
        else:
            return

        if result.status not in {"ok", "replayed"}:
            return
        publication = result.publication
        if (
            publication is None
            and request.action != "publish"
            and request.operation_ref == PLANE_OUTCOME_PUBLISH_OPERATION
            and result.status == "ok"
        ):
            try:
                publication = _outcome_publication_from_operation_result(
                    result.output,
                    resource_ref=resource_ref,
                )
            except PlaneHostError as exc:
                self._fail(str(exc) or "generic outcome publication receipt was invalid")
                raise PlaneHostUnavailable(
                    "generic outcome publication receipt was invalid"
                ) from exc
        if publication is None:
            if request.action == "publish" or result.status == "ok":
                self._fail("publication has no gateway publication receipt")
                raise PlaneHostUnavailable("publication has no gateway publication receipt")
            return

        try:
            resource_ref = _text(resource_ref, "publication.resourceRef", 256)
            content = _text(content, "publication.content", MAX_HOST_CONTENT_BYTES)
            publication = _validated_publication(
                publication,
                kind=kind,
                resource_ref=resource_ref,
            )
            if (
                request.operation_ref == PLANE_OUTCOME_PUBLISH_OPERATION
                and publication.get("operationRef") != PLANE_OUTCOME_PUBLISH_OPERATION
            ):
                raise PlaneHostUnavailable(
                    "host publication is not bound to agent.outcome.publish"
                )
        except PlaneHostError as exc:
            self._fail(str(exc) or "publication receipt was invalid")
            raise PlaneHostUnavailable("publication receipt was invalid") from exc

        with self._lock:
            if request.operation_ref == PLANE_OUTCOME_PUBLISH_OPERATION and kind == "outcome":
                self._outcome_publication_metadata = {
                    "status": result.status,
                    "replayed": result.replayed,
                    "publication_action": publication.get("action", "none"),
                    "operation_ref": request.operation_ref,
                    "terminal_armed": False,
                }
            if (
                request.operation_ref == PLANE_OUTCOME_PUBLISH_OPERATION
                and kind == "outcome"
                and result.status == "ok"
                and result.replayed is False
                and publication["action"] == "applied"
            ):
                # The Plane host has already applied the product mutation. Arm
                # the kernel stop before emitting the corresponding runtime
                # observation so a late event-handoff failure cannot turn an
                # applied publication into a pre-terminal host failure.
                self._terminal_action_reason = "product_outcome_published"
                self._terminal_action_result = result
                self._terminal_action_request = request
                if self._outcome_publication_metadata is not None:
                    self._outcome_publication_metadata["terminal_armed"] = True
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
            self._set_callback_phase("model_observation_emit", request)
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


def _outcome_publication_from_operation_result(
    value: Any, *, resource_ref: Any
) -> dict[str, str]:
    """Normalize the generic operation receipt into the publication binding."""

    output = _object(value, "host.result.output")
    if (
        output.get("ok") is not True
        or output.get("replayed") is not False
        or output.get("operationRef") != PLANE_OUTCOME_PUBLISH_OPERATION
    ):
        raise PlaneHostUnavailable("generic outcome publication receipt is not applied")
    result = _object(output.get("result"), "host.result.output.result")
    outcome = _object(result.get("outcome"), "host.result.output.result.outcome")
    request_id = _text(output.get("requestId"), "host.result.output.requestId", 256)
    gateway_receipt = _text(
        output.get("gatewayReceipt"),
        "host.result.output.gatewayReceipt",
        256,
    )
    audit_receipt = _text(
        output.get("auditReceipt"),
        "host.result.output.auditReceipt",
        256,
    )
    product_event_ref = _text(
        outcome.get("productEventRef"),
        "host.result.output.result.outcome.productEventRef",
        256,
    )
    product_ref = _text(resource_ref, "publication.resourceRef", 256)
    if not product_ref.startswith("outcome-submission:"):
        raise PlaneHostUnavailable("generic outcome publication resource is invalid")
    return {
        "action": "applied",
        "productKind": "outcome_submission",
        "productRef": product_ref,
        "operationAttemptRef": f"operation-attempt:{request_id}",
        "operationRef": PLANE_OUTCOME_PUBLISH_OPERATION,
        "applicationServiceRef": "application-service:agent-lifecycle",
        "gatewayReceiptRef": f"gateway-receipt:{gateway_receipt}",
        "receiptRef": f"receipt:{request_id}",
        "auditReceiptRef": f"audit-receipt:{audit_receipt}",
        "productEventRef": product_event_ref,
    }


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
    """Mark one existing Hermes execute_code parent-RPC dispatch as Code Mode."""

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


def _handle_plane_code_mode(args: Mapping[str, Any], **_: Any) -> str:
    """Submit TypeScript source to the versioned Plane host action.

    The source is opaque to Hermes.  It is never evaluated here and the
    capsule deliberately carries no credentials, endpoint, filesystem, or
    network data.  Plane parses the capsule and invokes its existing
    credential-free child isolate; callbacks from that isolate stay on the
    Plane-owned operation gateway.
    """

    try:
        data = _object(args, PLANE_CODE_MODE_TOOL)
        _reject_unknown(data, {"typescript_source"}, PLANE_CODE_MODE_TOOL)
        source = _text(
            data.get("typescript_source"),
            f"{PLANE_CODE_MODE_TOOL}.typescript_source",
            MAX_CODE_MODE_SOURCE_BYTES,
        )
        if not source.strip():
            raise PlaneHostError(
                f"{PLANE_CODE_MODE_TOOL}.typescript_source must be non-empty TypeScript"
            )
        capsule = {
            "schemaVersion": PLANE_CODE_MODE_SCHEMA_VERSION,
            "entrypoint": "default",
            "source": source,
            "input": {},
        }
        _bounded_json(capsule, "codeMode.capsule", MAX_HOST_INPUT_BYTES)
        result = _binding_or_error().call(
            action="code",
            operation_ref=PLANE_CODE_MODE_EXECUTE_OPERATION,
            input=capsule,
            source="code",
        )
        return result.model_payload()
    except PlaneHostCancelled as exc:
        return _error_payload(str(exc), code="cancelled")
    except PlaneHostError as exc:
        return _error_payload(str(exc))


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
        input_value = dict(
            _normalize_prepared_read_input(action, operation_ref, input_value)
        )
        if action == "code" and not _PLANE_CODE_MODE.get():
            raise PlaneHostError(
                "plane_operation code action is restricted to plane_execute_typescript"
            )
        result = _binding_or_error().call(
            action=action,
            operation_ref=operation_ref,
            input=input_value,
            source="model" if action != "code" else "code",
        )
        return result.model_payload()
    except PlaneHostCancelled as exc:
        return _error_payload(str(exc), code="cancelled")
    except PlaneHostSchemaNotDisclosed as exc:
        return _error_payload(str(exc), code="SCHEMA_NOT_DISCLOSED")
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
    except PlaneHostSchemaNotDisclosed as exc:
        return _error_payload(str(exc), code="SCHEMA_NOT_DISCLOSED")
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
            PLANE_CODE_MODE_TOOL,
            PLANE_CODE_MODE_TOOLSET,
            {
                "name": PLANE_CODE_MODE_TOOL,
                "description": (
                    "what it does: run bounded TypeScript in Plane's credential-free "
                    "Code Mode restricted isolate and return a bounded Plane host result. "
                    "when to use: use this when the commission supplies or requires "
                    "Plane Code Mode TypeScript composition. "
                    "input contract: typescript_source must be a complete TypeScript "
                    "module exporting default async function receiving {host,input} "
                    "(written as ({host,input})); do not "
                    "import modules or use network, filesystem, process, or credentials; "
                    "use only the typed host callbacks supplied by Plane. "
                    "returns: a bounded structured HostCallResult with status, replayed, "
                    "output, and optional error or publication. "
                    "errors/recovery: correct the source and call again only for "
                    "validation errors; do not retry unknown outcomes."
                ),
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "typescript_source": {
                            "type": "string",
                            "maxLength": 4096,
                            "description": (
                                "Complete bounded TypeScript module exporting default "
                                "async function ({host,input})."
                            ),
                        },
                    },
                    "required": ["typescript_source"],
                },
            },
            _handle_plane_code_mode,
            description="Bounded TypeScript Plane Code Mode execution",
            max_result_size_chars=MAX_HOST_RESULT_TEXT_BYTES,
        )
        registry.register(
            PLANE_OPERATION_TOOL,
            PLANE_OPERATION_TOOLSET,
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
                            "enum": ["discover", "read", "mutate"],
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
            PLANE_PUBLICATION_TOOLSET,
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
    "PlaneHostSchemaNotDisclosed",
    "PlaneHostPort",
    "PlaneHostUnavailable",
    "MAX_CODE_MODE_SOURCE_BYTES",
    "PLANE_CODE_MODE_EXECUTE_OPERATION",
    "PLANE_CODE_MODE_SCHEMA_VERSION",
    "PLANE_OPERATION_TOOLSET",
    "PLANE_PUBLICATION_TOOLSET",
    "PLANE_CODE_MODE_TOOLSET",
    "PLANE_CODE_MODE_TOOL",
    "PLANE_OUTCOME_PUBLISH_OPERATION",
    "bind_plane_host",
    "current_plane_host",
    "install_plane_tools",
    "plane_code_mode",
]
