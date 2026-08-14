"""Hermes-only execution adapter behind the G1 runtime seam."""

from __future__ import annotations

import contextlib
import io
import json
import re
import socket
import threading
import time
import uuid
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Mapping, MutableMapping, Protocol, Sequence

import httpx

from .g1_contract import (
    G1ContractError,
    G1InvocationEnvelope,
    G1RunSnapshot,
    MAX_PROMPT_BYTES,
    MAX_TEXT_BYTES,
    RUNTIME_FAILURE_CAUSES,
)
from .host_port import (
    PlaneHostBinding,
    PlaneHostPort,
    bind_plane_host,
    install_plane_tools,
)
from .presentation import PresentationBoundsError, build_model_guidance


_CREDENTIAL_PROTOCOL = "plane.agent-runtime/credentials/v1"
_MAX_CREDENTIALS = 16
_MAX_CREDENTIAL_KEY_BYTES = 128
_MAX_CREDENTIAL_VALUE_BYTES = 16 * 1024
_MAX_UNIX_SOCKET_PATH_BYTES = 103
_PROVIDER_RELAY_ORIGIN = "http://plane-provider-relay.invalid"
_PROVIDER_RELAY_FIELDS = frozenset(
    {"host", "path", "provider", "relayToken", "invocationSocket"}
)
_PROVIDER_RELAY_DUMMY_API_KEY = "plane-provider-relay"
_MAX_PROVIDER_ERROR_BODY_BYTES = 4096
_CODE_MODE_RUNTIME_POLICY_FIELDS = (
    "maxCodeModeInputBytes",
    "maxCodeModeOutputBytes",
    "maxCodeModeCalls",
)
_OUTCOME_UNKNOWN_RUNTIME_MESSAGE = (
    "Provider outcome is unknown; Plane reconciliation is required before retrying."
)


class ProviderOutcomeUnknownError(RuntimeError):
    """A provider request may have reached upstream and must not be replayed."""

    code = "outcome_unknown"
    retryable = False
    upstream_initiated = True
    terminal_failure = True

    def __init__(self, *, status_code: int) -> None:
        self.status_code = status_code
        super().__init__("provider outcome is unknown; reconcile before retrying")


class _ProviderRelayBodyStream(httpx.SyncByteStream):
    """Turn post-header relay read failures into the typed terminal signal."""

    def __init__(self, stream: Any, *, status_code: int) -> None:
        self._stream = stream
        self._status_code = status_code

    def __iter__(self):
        try:
            yield from self._stream
        except ProviderOutcomeUnknownError:
            raise
        except Exception as exc:
            raise ProviderOutcomeUnknownError(status_code=self._status_code) from exc

    def close(self) -> None:
        close = getattr(self._stream, "close", None)
        if not callable(close):
            return
        try:
            close()
        except ProviderOutcomeUnknownError:
            raise
        except Exception as exc:
            raise ProviderOutcomeUnknownError(status_code=self._status_code) from exc


def _raise_on_provider_outcome_unknown(response: Any) -> None:
    """Decode the relay's bounded ambiguity marker at the HTTP boundary."""

    response.stream = _ProviderRelayBodyStream(
        response.stream,
        status_code=int(response.status_code),
    )
    if response.status_code < 400:
        return
    content_length = response.headers.get("content-length")
    try:
        content_length = int(content_length) if content_length is not None else -1
        if content_length < 0 or content_length > _MAX_PROVIDER_ERROR_BODY_BYTES:
            return
    except (TypeError, ValueError):
        return
    try:
        body = response.read()
    except ProviderOutcomeUnknownError:
        raise
    except Exception as exc:
        raise ProviderOutcomeUnknownError(status_code=int(response.status_code)) from exc
    if len(body) > _MAX_PROVIDER_ERROR_BODY_BYTES:
        return
    try:
        payload = json.loads(body)
    except (TypeError, ValueError, json.JSONDecodeError):
        return
    if (
        isinstance(payload, dict)
        and payload.get("error") == "outcome_unknown"
        and payload.get("retryable") is False
        and payload.get("upstreamInitiated") is True
    ):
        raise ProviderOutcomeUnknownError(status_code=int(response.status_code))


def validate_absolute_unix_socket_path(value: object) -> str | None:
    """Validate the bounded absolute socket path shared by trusted bootstrap paths."""

    if value is None:
        return None
    if (
        not isinstance(value, str)
        or not value.startswith("/")
        or any(ord(char) < 0x20 or ord(char) == 0x7F for char in value)
        or len(value.encode("utf-8")) > _MAX_UNIX_SOCKET_PATH_BYTES
    ):
        raise ValueError("provider relay socket configuration is invalid")
    return value


@dataclass(frozen=True)
class _ProviderRelaySpec:
    """Bounded provider contract enforced before the child gets an SDK client."""

    host: str
    path: str
    api_mode: str

    @property
    def logical_base_path(self) -> str:
        base_path, separator, _leaf = self.path.rpartition("/")
        if not separator or not base_path.startswith("/"):
            raise RuntimeError("provider relay registry contains an invalid path")
        return base_path

    @property
    def logical_base_url(self) -> str:
        return _PROVIDER_RELAY_ORIGIN + self.logical_base_path


# The relay is the trusted egress authority.  These entries describe only the
# exact upstream target and the Hermes transport needed for that target; they
# are not user-controlled endpoint configuration.  The logical base URL is
# derived from ``path`` so the Responses/chat-completions suffix is appended by
# the existing Hermes client in the normal way.
_PROVIDER_RELAY_SPECS: Mapping[str, _ProviderRelaySpec] = MappingProxyType(
    {
        "openai-codex": _ProviderRelaySpec(
            host="chatgpt.com",
            path="/backend-api/codex/responses",
            api_mode="codex_responses",
        ),
        "xai": _ProviderRelaySpec(
            host="api.x.ai",
            path="/v1/chat/completions",
            api_mode="chat_completions",
        ),
        "xai-oauth": _ProviderRelaySpec(
            host="api.x.ai",
            path="/v1/chat/completions",
            api_mode="chat_completions",
        ),
    }
)


def _provider_relay_spec(provider: object) -> _ProviderRelaySpec:
    if not isinstance(provider, str) or provider not in _PROVIDER_RELAY_SPECS:
        raise ValueError("provider relay metadata is invalid")
    return _PROVIDER_RELAY_SPECS[provider]


def provider_relay_base_url(provider: str) -> str:
    """Return the logical relay base URL for one supported provider."""

    return _provider_relay_spec(provider).logical_base_url


# Compatibility for existing xAI adapter callers.  Routing is registry-owned;
# this alias is not consulted by relay validation or client construction.
PROVIDER_RELAY_BASE_URL = provider_relay_base_url("xai")


@dataclass(frozen=True)
class _ProviderRelayConfig:
    provider: str
    relay_token: str = field(repr=False)
    invocation_socket: str = field(repr=False)
    base_url: str

    def http_client_factory(self) -> Callable[[], Any]:
        """Build fresh SDK-owned HTTP clients bound to this invocation relay."""

        def create_client() -> Any:
            client_request_id = str(uuid.uuid4())

            def apply_relay_headers(request: Any) -> None:
                # OpenAI builds request-level headers from ``api_key``.  Apply
                # the relay credential at the HTTPX request boundary so the
                # agent can retain a fixed dummy API key without replacing the
                # relay authorization with a provider credential.
                request.headers["Authorization"] = f"Bearer {self.relay_token}"
                request.headers["X-Plane-Relay-Provider"] = self.provider
                request.headers["X-Request-ID"] = str(uuid.uuid4())

            transport = httpx.HTTPTransport(uds=self.invocation_socket, retries=0)
            return httpx.Client(
                transport=transport,
                base_url=self.base_url,
                headers={
                    "Authorization": f"Bearer {self.relay_token}",
                    "X-Plane-Relay-Provider": self.provider,
                    "X-Request-ID": client_request_id,
                },
                follow_redirects=False,
                timeout=None,
                event_hooks={
                    "request": [apply_relay_headers],
                    "response": [_raise_on_provider_outcome_unknown],
                },
            )

        setattr(create_client, "_plane_provider_relay", True)
        return create_client


def prepare_provider_relay_credentials(
    credentials: MutableMapping[str, str],
    *,
    expected_provider: str,
    provider_relay_socket: str | None,
) -> tuple[Mapping[str, str], Callable[[], Any] | None]:
    """Consume relay controls and return dummy Hermes credentials plus a factory.

    A normal credential map is returned unchanged for compatibility.  A relay
    map must contain exactly the private control fields, and its socket must
    match the separately forwarded trusted bootstrap argument.  Relay
    controls are removed before the resulting source can be observed by
    ``InlineCredentialSource`` or ``AIAgent``.
    """

    provider_relay_socket = validate_absolute_unix_socket_path(provider_relay_socket)
    relay_keys = set(credentials) & _PROVIDER_RELAY_FIELDS
    if not relay_keys:
        if provider_relay_socket is not None:
            raise ValueError("provider relay metadata is required")
        return credentials, None
    if set(credentials) != _PROVIDER_RELAY_FIELDS:
        raise ValueError("provider relay metadata is invalid")
    if provider_relay_socket is None:
        raise ValueError("provider relay socket is required for relay metadata")

    spec = _provider_relay_spec(expected_provider)
    host = credentials.get("host")
    path = credentials.get("path")
    provider = credentials.get("provider")
    relay_token = credentials.get("relayToken")
    invocation_socket = credentials.get("invocationSocket")
    if (
        host != spec.host
        or path != spec.path
        or provider != expected_provider
        or not isinstance(relay_token, str)
        or not relay_token
        or len(relay_token.encode("utf-8")) > _MAX_CREDENTIAL_VALUE_BYTES
        or any(ord(char) < 0x20 or ord(char) == 0x7F for char in relay_token)
        or not isinstance(invocation_socket, str)
        or validate_absolute_unix_socket_path(invocation_socket) != provider_relay_socket
    ):
        raise ValueError("provider relay metadata is invalid")

    credentials.clear()
    config = _ProviderRelayConfig(
        provider,
        relay_token,
        provider_relay_socket,
        spec.logical_base_url,
    )
    return (
        {
            "api_key": _PROVIDER_RELAY_DUMMY_API_KEY,
            "base_url": spec.logical_base_url,
            "api_mode": spec.api_mode,
        },
        config.http_client_factory(),
    )


def _code_mode_is_available(snapshot: G1RunSnapshot) -> bool:
    """Translate Plane's immutable Code Mode availability declaration.

    The values are validated by the G1 contract before this adapter sees the
    snapshot.  Presence of all three positive limits is the existing Plane
    contract's declaration that this invocation may use restricted Code Mode;
    it is not a permission check and does not replace live host authorization.
    """

    policy = snapshot.raw["runtimePolicy"]
    return all(policy.get(key, 0) > 0 for key in _CODE_MODE_RUNTIME_POLICY_FIELDS)


def _strict_json_object(raw: bytes, expected: set[str], name: str) -> dict[str, Any]:
    """Parse one canonical object and reject duplicate/unknown/trailing data."""
    if raw != raw.strip() or len(raw) > _MAX_CREDENTIAL_VALUE_BYTES:
        raise G1ContractError(f"{name} is not canonical")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise G1ContractError(f"{name} contains duplicate keys")
            result[key] = value
        return result

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise G1ContractError(f"{name} is malformed") from exc
    if not isinstance(value, dict) or set(value) != expected:
        raise G1ContractError(f"{name} has an invalid key set")
    canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if canonical != raw:
        raise G1ContractError(f"{name} is not canonical")
    return value


def _strict_credentials(value: object) -> dict[str, str]:
    if not isinstance(value, dict) or len(value) > _MAX_CREDENTIALS:
        raise G1ContractError("credential broker credentials are invalid")
    credentials: dict[str, str] = {}
    for key, item in value.items():
        if (
            not isinstance(key, str)
            or not isinstance(item, str)
            or not key
            or not item
            or len(key.encode("utf-8")) > _MAX_CREDENTIAL_KEY_BYTES
            or len(item.encode("utf-8")) > _MAX_CREDENTIAL_VALUE_BYTES
            or any(ord(char) < 0x20 or ord(char) == 0x7F for char in key + item)
        ):
            raise G1ContractError("credential broker credentials are invalid")
        credentials[key] = item
    return credentials


class HermesCredentialSource(Protocol):
    """Trusted host source; credentials never come from a runtime envelope."""

    def resolve(self, provider: str) -> Mapping[str, str]:
        ...


class HermesCheckpointSource(Protocol):
    """Trusted host source for a previously approved continuation checkpoint."""

    def load(self, checkpoint_ref: str) -> Sequence[Mapping[str, Any]]:
        ...


@dataclass(frozen=True)
class EnvironmentCredentialSource:
    """Fail closed; ambient environment is never a credential channel."""

    def resolve(self, provider: str) -> Mapping[str, str]:
        del provider
        return {}


@dataclass
class InlineCredentialSource:
    """Private bootstrap handoff; never an environment or model data source."""

    credentials: Mapping[str, str]
    expected_provider: str

    def resolve(self, provider: str) -> Mapping[str, str]:
        if provider != self.expected_provider:
            return {}
        return dict(self.credentials)


@dataclass(frozen=True)
class UnixSocketCredentialSource:
    """Read one provider credential from an explicitly supplied test socket.

    Production G1 does not construct this compatibility seam; its credential
    source is the private bootstrap pipe. A missing, malformed, or over-sized
    broker response fails closed.
    """

    # Explicit path is required because the production bootstrap no longer
    # has a fixed credential socket authority.  This remains a local test or
    # compatibility seam only.
    path: str
    timeout_seconds: float = 2.0

    def resolve(self, provider: str) -> Mapping[str, str]:
        request = json.dumps(
            {"protocol": _CREDENTIAL_PROTOCOL, "provider": provider},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8") + b"\n"
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as channel:
            channel.settimeout(self.timeout_seconds)
            channel.connect(self.path)
            channel.sendall(request)
            response = bytearray()
            while len(response) <= 16 * 1024:
                chunk = channel.recv(min(4096, 16 * 1024 + 1 - len(response)))
                if not chunk:
                    break
                response.extend(chunk)
                if response.endswith(b"\n"):
                    break
        if not response.endswith(b"\n"):
            raise G1ContractError("credential broker response is incomplete")
        value = _strict_json_object(response[:-1], {"credentials", "protocol"}, "credential broker response")
        if value["protocol"] != _CREDENTIAL_PROTOCOL:
            raise G1ContractError("credential broker response protocol is invalid")
        return _strict_credentials(value["credentials"])


class HermesAuthStoreCredentialSource:
    """Resolve the active Hermes credential in the trusted host process."""

    def resolve(self, provider: str) -> Mapping[str, str]:
        from agent.credential_pool import load_pool

        pool = load_pool(provider)
        entry = pool.current() or next(iter(pool.entries()), None)
        if entry is None or not entry.runtime_api_key:
            return {}
        credentials: dict[str, str] = {"api_key": entry.runtime_api_key}
        if entry.runtime_base_url:
            credentials["base_url"] = entry.runtime_base_url
        return credentials


@dataclass(frozen=True)
class HermesKernelResult:
    kind: str
    output_text: str = ""
    question: str | None = None
    failure_code: str | None = None
    failure_message: str | None = None
    retryable: bool = True
    usage: Mapping[str, int] | None = None
    model_calls: int | None = None
    failure_cause: str | None = None


class NeverCancelled:
    """Explicit no-cancellation seam for callers that own no control signal."""

    def __call__(self) -> bool:
        return False


class _CancellationMonitor:
    """Poll trusted cancellation and interrupt a running AIAgent promptly."""

    def __init__(self, probe: Callable[[], bool], agent: Any) -> None:
        self._probe = probe
        self._agent = agent
        self._stop = threading.Event()
        self._requested = threading.Event()
        self._failed = threading.Event()
        self._thread = threading.Thread(target=self._run, name="plane-runtime-cancellation", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                value = self._probe()
            except Exception:
                self._failed.set()
                return
            if not isinstance(value, bool):
                self._failed.set()
                return
            if value:
                self._requested.set()
                try:
                    self._agent.interrupt("runtime cancellation requested")
                except Exception:
                    self._failed.set()
                return
            self._stop.wait(0.05)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=0.5)

    @property
    def requested(self) -> bool:
        return self._requested.is_set()

    @property
    def failed(self) -> bool:
        return self._failed.is_set() or self._thread.is_alive()


def redact_runtime_text(value: str, secrets: Sequence[str]) -> str:
    """Remove host credential values from all runtime-visible text."""

    redacted = str(value)
    for secret in secrets:
        if secret:
            redacted = redacted.replace(secret, "[redacted]")
    # Provider error messages occasionally include bearer-style material even
    # when the credential source did not expose the exact original string.
    return re.sub(r"(?i)\b(?:sk|api|token|bearer)[_-][A-Za-z0-9._-]{12,}\b", "[redacted]", redacted)


def bound_runtime_text(value: str, maximum_bytes: int) -> str:
    """Keep model and error text within the inline payload bound."""

    maximum_bytes = max(1, min(MAX_TEXT_BYTES, int(maximum_bytes)))
    text = str(value)
    encoded = text.encode("utf-8")
    if len(encoded) <= maximum_bytes:
        return text
    marker = "\n[truncated]"
    available = max(0, maximum_bytes - len(marker.encode("utf-8")))
    return encoded[:available].decode("utf-8", errors="ignore") + marker


class HermesKernelAdapter:
    """Translate one G1 invocation to one existing ``AIAgent`` turn.

    This is deliberately the only Plane-runtime module that imports Hermes'
    agent class.  The factory and host sources are dependency seams for local
    verification; production uses the lazy default factory and trusted host
    credential source.  ``http_client_factory`` is an optional invocation-
    scoped callable that returns a fresh SDK-owned HTTP client for each
    OpenAI-compatible client construction; it remains private to the live
    agent and is never part of provider configuration or runtime evidence.
    """

    def __init__(
        self,
        *,
        agent_factory: Callable[..., Any] | None = None,
        credential_source: HermesCredentialSource | None = None,
        checkpoint_source: HermesCheckpointSource | None = None,
        enabled_toolsets: Sequence[str] = (),
        host_port: PlaneHostPort | None = None,
        http_client_factory: Callable[[], Any] | None = None,
    ) -> None:
        if http_client_factory is not None and not callable(http_client_factory):
            raise TypeError("http_client_factory must be callable or None")
        self._agent_factory = agent_factory or self._default_agent_factory
        self._credential_source = credential_source or EnvironmentCredentialSource()
        self._checkpoint_source = checkpoint_source
        self._enabled_toolsets = tuple(enabled_toolsets)
        self._host_port = host_port
        self._http_client_factory = http_client_factory
        if host_port is not None:
            # Dynamic and invocation-gated: never part of Hermes' default schema.
            install_plane_tools()

    @staticmethod
    def _default_agent_factory(**kwargs: Any) -> Any:
        # Keep the replaceable kernel import lazy and local to this adapter.
        from run_agent import AIAgent

        return AIAgent(**kwargs)

    def dispatch(
        self,
        snapshot: G1RunSnapshot,
        invocation: G1InvocationEnvelope,
        cancellation: Callable[[], bool],
        emit_body: Callable[[Mapping[str, Any]], None],
        *,
        model_call_allowance: int | None = None,
    ) -> HermesKernelResult:
        try:
            initially_cancelled = cancellation()
        except Exception:
            return HermesKernelResult(
                kind="failed",
                failure_code="runtime_error",
                failure_message="trusted cancellation probe failed",
                failure_cause="cancellation_monitor_failure",
                retryable=False,
            )
        if not isinstance(initially_cancelled, bool):
            return HermesKernelResult(
                kind="failed",
                failure_code="runtime_error",
                failure_message="trusted cancellation probe returned an invalid value",
                failure_cause="cancellation_monitor_failure",
                retryable=False,
            )
        if initially_cancelled:
            return HermesKernelResult(
                kind="cancelled",
                failure_code="cancelled",
                failure_message="runtime cancellation was requested",
                retryable=False,
            )
        checkpoint_ref = invocation.raw.get("checkpointRef")
        prefill_messages: Sequence[Mapping[str, Any]] | None = None
        if checkpoint_ref is not None:
            if self._checkpoint_source is None:
                return HermesKernelResult(
                    kind="failed",
                    failure_code="invalid_continuation",
                    failure_message="continuation checkpoint is not available to the trusted host",
                    retryable=False,
                )
            try:
                prefill_messages = self._checkpoint_source.load(str(checkpoint_ref))
            except Exception:
                return HermesKernelResult(
                    kind="failed",
                    failure_code="invalid_continuation",
                    failure_message="continuation checkpoint could not be reconstructed",
                    retryable=False,
                )
        if model_call_allowance is None or isinstance(model_call_allowance, bool) or not isinstance(model_call_allowance, int) or model_call_allowance < 0:
            return HermesKernelResult(
                kind="failed",
                failure_code="runtime_error",
                failure_message="trusted model-call allowance is required",
                failure_cause="static_configuration_failure",
                retryable=False,
            )
        if model_call_allowance == 0:
            return HermesKernelResult(
                kind="failed",
                failure_code="budget_exhausted",
                failure_message="model-call allowance is exhausted",
                retryable=False,
                model_calls=0,
            )

        event_limit = int(snapshot.raw["runtimePolicy"]["maxEventPayloadBytes"])
        try:
            prompt = build_model_guidance(snapshot)
        except PresentationBoundsError:
            return HermesKernelResult(
                kind="failed",
                failure_code="runtime_error",
                failure_message="Plane invocation guidance exceeds its prompt bound",
                failure_cause="static_configuration_failure",
                retryable=False,
            )
        try:
            credentials = dict(self._credential_source.resolve(snapshot.model_provider))
        except Exception:
            return HermesKernelResult(
                kind="failed",
                failure_code="runtime_error",
                failure_message="trusted host credential resolution failed",
                retryable=True,
            )
        if not credentials:
            return HermesKernelResult(
                kind="failed",
                failure_code="runtime_error",
                failure_message="trusted host credential source is required",
                failure_cause="static_configuration_failure",
                retryable=False,
            )
        credential_values = tuple(credentials.values())
        emit_body(
            {
                "kind": "progress_observed",
                "payload": {"kind": "inline_text", "contentType": "text/plain", "text": "Hermes invocation started."},
                "publication": {"action": "observation_only"},
            }
        )
        streamed: list[str] = []
        streamed_bytes = 0

        def on_delta(delta: Any) -> None:
            nonlocal streamed_bytes
            if delta is not None:
                text = redact_runtime_text(str(delta), credential_values)
                if streamed_bytes < MAX_PROMPT_BYTES:
                    bounded = bound_runtime_text(text, MAX_PROMPT_BYTES - streamed_bytes)
                    streamed.append(bounded)
                    streamed_bytes += len(bounded.encode("utf-8"))

        def on_step(*args: Any, **kwargs: Any) -> None:
            del args, kwargs
            try:
                cancelled = cancellation()
            except Exception:
                return
            if cancelled is False:
                emit_body(
                    {
                        "kind": "progress_observed",
                        "payload": {"kind": "inline_text", "contentType": "text/plain", "text": "Hermes tool loop advanced."},
                        "publication": {"action": "observation_only"},
                    }
                )

        enabled_toolsets = [
            toolset
            for toolset in self._enabled_toolsets
            if toolset != "code_execution"
        ]
        if _code_mode_is_available(snapshot):
            enabled_toolsets.append("code_execution")
        if self._host_port is not None:
            enabled_toolsets.append("plane_runtime")
        # Preserve caller ordering while keeping adapter-added toolsets
        # idempotent when a compatibility caller already supplied one.
        enabled_toolsets = list(dict.fromkeys(enabled_toolsets))

        agent_kwargs: dict[str, Any] = {
            "provider": snapshot.model_provider,
            "model": snapshot.model_name,
            # This is the model-dispatch output ceiling.  IterationBudget is
            # only a loop safeguard; it is not the token-budget authority.
            "max_tokens": max(1, int(invocation.remaining_budget["outputTokens"])),
            "max_iterations": model_call_allowance,
            "session_id": invocation.invocation_id,
            "enabled_toolsets": enabled_toolsets,
            "quiet_mode": True,
            "skip_context_files": True,
            "skip_memory": True,
            "save_trajectories": False,
            "checkpoints_enabled": False,
            "stream_delta_callback": on_delta,
            "step_callback": on_step,
        }
        try:
            from agent.iteration_budget import IterationBudget

            agent_kwargs["iteration_budget"] = IterationBudget(
                model_call_allowance
            )
        except Exception:
            return HermesKernelResult(
                kind="failed",
                failure_code="runtime_error",
                failure_message="Hermes iteration budget could not be constructed",
                failure_cause="static_configuration_failure",
                retryable=False,
            )
        if credentials.get("api_key"):
            agent_kwargs["api_key"] = credentials["api_key"]
        if credentials.get("base_url"):
            agent_kwargs["base_url"] = credentials["base_url"]
        if credentials.get("api_mode"):
            agent_kwargs["api_mode"] = credentials["api_mode"]
        if prefill_messages is not None:
            agent_kwargs["prefill_messages"] = [dict(message) for message in prefill_messages]
        if self._http_client_factory is not None:
            agent_kwargs["http_client_factory"] = self._http_client_factory

        started_at = time.monotonic()
        agent: Any | None = None
        cancellation_monitor: _CancellationMonitor | None = None
        host_binding = (
            PlaneHostBinding(
                port=self._host_port,
                run_id=snapshot.run_id,
                invocation_id=invocation.invocation_id,
                correlation_id=invocation.correlation_id,
                cancellation=cancellation,
                emit_body=emit_body,
                eager_operation_refs=frozenset(
                    str(operation["operationRef"])
                    for operation in snapshot.eager_operations
                ),
            )
            if self._host_port is not None
            else None
        )
        try:
            agent = self._agent_factory(**agent_kwargs)
            if host_binding is not None:
                setattr(agent, "_terminal_action_check", host_binding.terminal_action_reason)
            # Plane's provider allowance is a hard invocation boundary. The
            # interactive Hermes summary fallback would spend an additional
            # provider call after that boundary, so return a finite budget
            # failure instead.
            setattr(agent, "_plane_runtime_terminal_budget_failure", True)
            cancellation_monitor = _CancellationMonitor(cancellation, agent)
            cancellation_monitor.start()
            if host_binding is None:
                with contextlib.redirect_stdout(io.StringIO()):
                    result = agent.run_conversation(snapshot.objective, system_message=prompt)
            else:
                with bind_plane_host(host_binding):
                    with contextlib.redirect_stdout(io.StringIO()):
                        result = agent.run_conversation(snapshot.objective, system_message=prompt)
        except Exception as exc:
            if cancellation_monitor is not None:
                cancellation_monitor.close()
            if cancellation_monitor is not None and cancellation_monitor.requested:
                return HermesKernelResult(
                    kind="cancelled",
                    failure_code="cancelled",
                    failure_message="runtime cancellation was requested",
                    retryable=False,
                    model_calls=self._observed_model_calls(agent, None),
                )
            if cancellation_monitor is not None and cancellation_monitor.failed:
                return HermesKernelResult(
                    kind="failed",
                    failure_code="runtime_error",
                    failure_message="trusted cancellation could not interrupt Hermes",
                    failure_cause="cancellation_monitor_failure",
                    retryable=False,
                    model_calls=self._observed_model_calls(agent, None),
                )
            if (
                getattr(exc, "terminal_failure", False) is True
                and getattr(exc, "code", None) == "outcome_unknown"
                and getattr(exc, "retryable", None) is False
                and getattr(exc, "upstream_initiated", False) is True
            ):
                return HermesKernelResult(
                    kind="failed",
                    failure_code="outcome_unknown",
                    failure_message=_OUTCOME_UNKNOWN_RUNTIME_MESSAGE,
                    retryable=False,
                    model_calls=self._observed_model_calls(agent, None),
                )
            message = bound_runtime_text(redact_runtime_text(str(exc), credential_values), event_limit)
            return HermesKernelResult(
                kind="failed",
                failure_code="runtime_error",
                failure_message=message or "Hermes invocation failed",
                retryable=True,
                model_calls=self._observed_model_calls(agent, None),
            )

        if cancellation_monitor is not None:
            cancellation_monitor.close()
        if cancellation_monitor is not None and cancellation_monitor.failed:
            return HermesKernelResult(
                kind="failed",
                failure_code="runtime_error",
                failure_message="trusted cancellation probe failed",
                failure_cause="cancellation_monitor_failure",
                retryable=False,
                model_calls=self._observed_model_calls(agent, result),
            )
        if cancellation_monitor is not None and cancellation_monitor.requested:
            return HermesKernelResult(
                kind="cancelled",
                failure_code="cancelled",
                failure_message="runtime cancellation was requested",
                retryable=False,
            )
        if host_binding is not None and host_binding.fatal_error is not None:
            return HermesKernelResult(
                kind="failed",
                failure_code="runtime_error",
                failure_message="Plane host operation failed",
                failure_cause="host_operation_failure",
                retryable=False,
                model_calls=self._observed_model_calls(agent, result),
            )
        if not isinstance(result, Mapping):
            raise G1ContractError("Hermes adapter returned a non-object result")
        usage_values = (
            getattr(agent, "session_input_tokens", 0),
            getattr(agent, "session_output_tokens", 0),
        )
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in usage_values):
            return HermesKernelResult(
                kind="failed",
                failure_code="runtime_error",
                failure_message="Hermes usage accounting was invalid",
                failure_cause="invalid_usage_accounting",
                retryable=False,
                model_calls=self._observed_model_calls(agent, result),
            )
        usage = {
            "inputTokens": int(usage_values[0]),
            "outputTokens": int(usage_values[1]),
            "durationMs": max(0, int((time.monotonic() - started_at) * 1000)),
        }
        model_calls = self._observed_model_calls(agent, result)
        if any(usage[name] > invocation.remaining_budget[name] for name in usage):
            return HermesKernelResult(
                kind="failed",
                failure_code="budget_exhausted",
                failure_message="Hermes usage exceeded the cumulative invocation budget",
                retryable=False,
                usage=usage,
                model_calls=model_calls,
            )
        if result.get("interrupted") is True:
            return HermesKernelResult(
                kind="cancelled",
                failure_code="cancelled",
                failure_message="Hermes invocation was interrupted",
                retryable=False,
                model_calls=model_calls,
            )
        if result.get("failed") is True:
            if result.get("failure_reason") == "outcome_unknown":
                return HermesKernelResult(
                    kind="failed",
                    failure_code="outcome_unknown",
                    failure_message=_OUTCOME_UNKNOWN_RUNTIME_MESSAGE,
                    retryable=False,
                    usage=usage,
                    model_calls=model_calls,
                )
            if result.get("failure_reason") == "budget_exhausted":
                return HermesKernelResult(
                    kind="failed",
                    failure_code="budget_exhausted",
                    failure_message="model-call allowance is exhausted",
                    retryable=False,
                    usage=usage,
                    model_calls=model_calls,
                )
            return HermesKernelResult(
                kind="failed",
                failure_code="runtime_error",
                failure_message=bound_runtime_text(
                    redact_runtime_text(str(result.get("error") or "Hermes invocation failed"), credential_values),
                    event_limit,
                ),
                retryable=True,
                usage=usage,
                model_calls=model_calls,
            )
        question = result.get("input_request") or result.get("waiting_for_input")
        if isinstance(question, str) and question:
            return HermesKernelResult(
                kind="waiting_for_input",
                question=bound_runtime_text(redact_runtime_text(question, credential_values), event_limit),
                usage=usage,
                model_calls=model_calls,
            )
        output = result.get("final_response")
        if output is None:
            output = "".join(streamed)
        terminal_action = str(result.get("turn_exit_reason", "")).startswith(
            "terminal_action("
        )
        output_text = bound_runtime_text(
            redact_runtime_text(
                str(output or ("" if terminal_action else "Hermes invocation completed.")),
                credential_values,
            ),
            event_limit,
        )
        emit_body(
            {
                "kind": "usage_observed",
                "usage": usage,
                "publication": {"action": "observation_only"},
            }
        )
        if output_text:
            emit_body(
                {
                    "kind": "transcript_evidence_observed",
                    "payload": {"kind": "inline_text", "contentType": "text/plain", "text": output_text},
                    "publication": {"action": "observation_only"},
                }
            )
        return HermesKernelResult(kind="completed", output_text=output_text, usage=usage, model_calls=model_calls)

    @staticmethod
    def _observed_model_calls(agent: Any | None, result: Mapping[str, Any] | None) -> int | None:
        """Read Hermes' narrow call counter; never infer calls from tokens."""
        for source in (agent, result):
            value = getattr(source, "session_api_calls", None) if source is agent else (source or {}).get("api_calls")
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                return value
        return None


class DeterministicKernelAdapter:
    """No-network adapter used by G1 subprocess fixtures."""

    def dispatch(
        self,
        snapshot: G1RunSnapshot,
        invocation: G1InvocationEnvelope,
        cancellation: Callable[[], bool],
        emit_body: Callable[[Mapping[str, Any]], None],
    ) -> HermesKernelResult:
        if cancellation():
            return HermesKernelResult("cancelled", failure_code="cancelled", failure_message="runtime cancellation was requested", retryable=False)
        emit_body(
            {
                "kind": "progress_observed",
                "payload": {"kind": "inline_text", "contentType": "text/plain", "text": "Deterministic Hermes adapter started."},
                "publication": {"action": "observation_only"},
            }
        )
        emit_body(
            {
                "kind": "usage_observed",
                "usage": {"inputTokens": 0, "outputTokens": 0, "durationMs": 0},
                "publication": {"action": "observation_only"},
            }
        )
        output = bound_runtime_text(
            f"Deterministic outcome for {snapshot.objective}",
            int(snapshot.raw["runtimePolicy"]["maxEventPayloadBytes"]),
        )
        emit_body(
            {
                "kind": "transcript_evidence_observed",
                "payload": {"kind": "inline_text", "contentType": "text/plain", "text": output},
                "publication": {"action": "observation_only"},
            }
        )
        return HermesKernelResult("completed", output_text=output, model_calls=0)


__all__ = [
    "DeterministicKernelAdapter",
    "EnvironmentCredentialSource",
    "InlineCredentialSource",
    "HermesAuthStoreCredentialSource",
    "UnixSocketCredentialSource",
    "HermesCheckpointSource",
    "HermesCredentialSource",
    "HermesKernelAdapter",
    "HermesKernelResult",
    "NeverCancelled",
    "PROVIDER_RELAY_BASE_URL",
    "RUNTIME_FAILURE_CAUSES",
    "bound_runtime_text",
    "prepare_provider_relay_credentials",
    "provider_relay_base_url",
    "redact_runtime_text",
    "validate_absolute_unix_socket_path",
]
