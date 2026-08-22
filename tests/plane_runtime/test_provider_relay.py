"""Focused tests for the private provider-relay adapter boundary."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import threading
import types
from unittest import mock

import pytest

from plane_runtime.g1_bootstrap_contract import G1BootstrapFrames
from plane_runtime.g1_contract import G1InvocationEnvelope, G1RunSnapshot
from plane_runtime.g1_runtime_image import bootstrap
from plane_runtime.hermes_adapter import (
    HermesKernelAdapter,
    HermesKernelResult,
    InlineCredentialSource,
    PROVIDER_RELAY_BASE_URL,
    ProviderOutcomeUnknownError,
    prepare_provider_relay_credentials,
    provider_relay_base_url,
)
from plane_runtime.g1_service import serve_once_g1
from plane_runtime.service import main as service_main

from tests.plane_runtime.test_g1_runtime_process import make_invocation, make_snapshot, _digest


def _hermes_request(
    *, provider: str = "xai", model: str = "grok-test"
) -> tuple[dict[str, object], dict[str, object], bytes]:
    snapshot = make_snapshot()
    snapshot["runtimePolicy"] = dict(snapshot["runtimePolicy"])  # type: ignore[arg-type]
    snapshot["runtimePolicy"]["adapter"] = "hermes"  # type: ignore[index]
    snapshot["runtimePolicy"]["model"] = {"provider": provider, "model": model}  # type: ignore[index]
    snapshot["contentDigest"] = _digest(
        "snapshot", {key: value for key, value in snapshot.items() if key != "contentDigest"}
    )
    invocation = make_invocation(snapshot)
    request = json.dumps(
        {"invocation": invocation, "run": snapshot},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return snapshot, invocation, request


def _relay_credentials(
    socket_path: str,
    token: str = "relay-secret",
    *,
    provider: str = "xai",
) -> dict[str, str]:
    if provider == "openai-codex":
        host = "chatgpt.com"
        path = "/backend-api/codex/responses"
    else:
        host = "api.x.ai"
        path = "/v1/chat/completions"
    return {
        "host": host,
        "path": path,
        "provider": provider,
        "relayToken": token,
        "invocationSocket": socket_path,
    }


class _LocalCodexRelay:
    """One-request provider-free HTTP/SSE relay on an AF_UNIX socket."""

    def __init__(self, response: bytes | None = None) -> None:
        self._directory = tempfile.TemporaryDirectory(prefix="plane-codex-relay-")
        self.path = os.path.join(self._directory.name, "relay.sock")
        self._response = response
        self.requests: list[tuple[str, bytes]] = []
        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server.bind(self.path)
        self._server.listen(2)
        self._server.settimeout(0.05)
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)

    def __enter__(self) -> "_LocalCodexRelay":
        self._thread.start()
        self._ready.wait(timeout=1)
        return self

    def __exit__(self, *_args: object) -> None:
        self._stop.set()
        self._server.close()
        self._thread.join(timeout=1)
        self._directory.cleanup()

    def _serve(self) -> None:
        self._ready.set()
        while not self._stop.is_set():
            try:
                channel, _ = self._server.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            with channel:
                channel.settimeout(1)
                raw = bytearray()
                while b"\r\n\r\n" not in raw:
                    chunk = channel.recv(4096)
                    if not chunk:
                        return
                    raw.extend(chunk)
                header_bytes, body = bytes(raw).split(b"\r\n\r\n", 1)
                headers = header_bytes.split(b"\r\n")
                content_length = next(
                    int(line.split(b":", 1)[1].strip())
                    for line in headers[1:]
                    if line.lower().startswith(b"content-length:")
                )
                while len(body) < content_length:
                    body += channel.recv(4096)
                self.requests.append((headers[0].decode("ascii"), body[:content_length]))
                channel.sendall(self._response or _codex_sse_response())


def _codex_sse_response() -> bytes:
    events = [
        {"type": "response.created", "response": {"id": "resp-relay", "status": "in_progress"}},
        {
            "type": "response.output_item.added",
            "item": {"type": "message", "id": "msg-relay", "role": "assistant", "status": "in_progress", "content": []},
        },
        {"type": "response.output_text.delta", "delta": "relay-complete"},
        {
            "type": "response.output_item.done",
            "item": {
                "type": "message",
                "id": "msg-relay",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "relay-complete", "annotations": []}],
            },
        },
        {
            "type": "response.completed",
            "response": {
                "id": "resp-relay",
                "status": "completed",
                "output": [],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            },
        },
    ]
    body = b"".join(
        b"data: " + json.dumps(event, separators=(",", ":")).encode() + b"\n\n"
        for event in events
    ) + b"data: [DONE]\n\n"
    return (
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Type: text/event-stream\r\n"
        b"Cache-Control: no-cache\r\n"
        b"Connection: close\r\n"
        b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body
    )


def _broken_codex_sse_response() -> bytes:
    body = b'data: {"type":"response.created"}\n\n'
    return (
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Type: text/event-stream\r\n"
        b"Connection: close\r\n"
        b"Content-Length: "
        + str(len(body) + 1).encode()
        + b"\r\n\r\n"
        + body
    )


def _outcome_unknown_response(*, status_code: int = 502) -> bytes:
    body = json.dumps(
        {
            "error": "outcome_unknown",
            "retryable": False,
            "upstreamInitiated": True,
        },
        separators=(",", ":"),
    ).encode()
    return (
        b"HTTP/1.1 "
        + str(status_code).encode()
        + b" Bad Gateway\r\n"
        b"Content-Type: application/json\r\n"
        b"Connection: close\r\n"
        b"Content-Length: "
        + str(len(body)).encode()
        + b"\r\n\r\n"
        + body
    )


class _BinaryStdin:
    def __init__(self, value: bytes) -> None:
        self.buffer = io.BytesIO(value)


def test_provider_relay_metadata_is_consumed_and_client_factory_is_exact() -> None:
    socket_path = "/tmp/provider-relay.sock"
    credentials = _relay_credentials(socket_path)

    agent_credentials, factory = prepare_provider_relay_credentials(
        credentials,
        expected_provider="xai",
        provider_relay_socket=socket_path,
    )

    assert credentials == {}
    assert agent_credentials == {
        "api_key": "plane-provider-relay",
        "base_url": PROVIDER_RELAY_BASE_URL,
        "api_mode": "chat_completions",
    }
    assert factory is not None
    assert getattr(factory, "_plane_provider_relay", False) is True

    clients = [object(), object()]
    with mock.patch("httpx.HTTPTransport") as transport, mock.patch(
        "httpx.Client", side_effect=clients
    ) as client:
        first = factory()
        second = factory()

    assert first is clients[0]
    assert second is clients[1]
    assert transport.call_args_list == [
        mock.call(uds=socket_path, retries=0),
        mock.call(uds=socket_path, retries=0),
    ]
    assert client.call_args_list[0].kwargs == {
        "transport": transport.return_value,
        "base_url": PROVIDER_RELAY_BASE_URL,
        "headers": {
            "Authorization": "Bearer relay-secret",
            "X-Plane-Relay-Provider": "xai",
            "X-Request-ID": client.call_args_list[0].kwargs["headers"]["X-Request-ID"],
        },
        "follow_redirects": False,
        "timeout": None,
        "event_hooks": client.call_args_list[0].kwargs["event_hooks"],
    }
    assert client.call_args_list[0].kwargs["headers"]["X-Request-ID"]
    assert client.call_args_list[1].kwargs["headers"]["X-Request-ID"] != client.call_args_list[0].kwargs["headers"]["X-Request-ID"]
    assert len(client.call_args_list[0].kwargs["event_hooks"]["response"]) == 1
    import httpx

    request = httpx.Request("POST", PROVIDER_RELAY_BASE_URL + "/chat/completions")
    request.headers["Authorization"] = "Bearer plane-provider-relay"
    client.call_args_list[0].kwargs["event_hooks"]["request"][0](request)
    assert request.headers["Authorization"] == "Bearer relay-secret"
    assert request.headers["X-Plane-Relay-Provider"] == "xai"
    assert request.headers["X-Request-ID"]
    assert request.headers["X-Request-ID"] != "plane-provider-relay"


def test_openai_codex_relay_metadata_derives_native_endpoint_and_mode() -> None:
    socket_path = "/tmp/provider-relay.sock"
    credentials = _relay_credentials(socket_path, provider="openai-codex")

    agent_credentials, factory = prepare_provider_relay_credentials(
        credentials,
        expected_provider="openai-codex",
        provider_relay_socket=socket_path,
    )

    assert credentials == {}
    assert agent_credentials == {
        "api_key": "plane-provider-relay",
        "base_url": "http://plane-provider-relay.invalid/backend-api/codex",
        "api_mode": "codex_responses",
    }
    assert provider_relay_base_url("openai-codex") == agent_credentials["base_url"]
    assert factory is not None

    with mock.patch("httpx.HTTPTransport") as transport, mock.patch(
        "httpx.Client", return_value=object()
    ) as client:
        factory()

    transport.assert_called_once_with(uds=socket_path, retries=0)
    assert client.call_args.kwargs["base_url"] == agent_credentials["base_url"]
    assert client.call_args.kwargs["headers"]["X-Plane-Relay-Provider"] == "openai-codex"

    import httpx

    request = httpx.Request(
        "POST", agent_credentials["base_url"] + "/responses"
    )
    request.headers["Authorization"] = "Bearer plane-provider-relay"
    client.call_args.kwargs["event_hooks"]["request"][0](request)
    assert request.url.path == "/backend-api/codex/responses"
    assert request.headers["Authorization"] == "Bearer relay-secret"
    assert request.headers["X-Plane-Relay-Provider"] == "openai-codex"


def test_openai_codex_relay_rejects_mismatched_provider_and_path() -> None:
    socket_path = "/tmp/provider-relay.sock"
    credentials = _relay_credentials(socket_path, provider="openai-codex")

    mismatched_provider = credentials.copy()
    mismatched_provider["provider"] = "xai"
    with pytest.raises(ValueError, match="metadata"):
        prepare_provider_relay_credentials(
            mismatched_provider,
            expected_provider="openai-codex",
            provider_relay_socket=socket_path,
        )

    invalid_path = credentials.copy()
    invalid_path["path"] = "/backend-api/codex/chat/completions"
    with pytest.raises(ValueError, match="metadata"):
        prepare_provider_relay_credentials(
            invalid_path,
            expected_provider="openai-codex",
            provider_relay_socket=socket_path,
        )


def test_openai_codex_service_adapter_uses_native_sse_over_bounded_uds_relay() -> None:
    snapshot, invocation, _request = _hermes_request(
        provider="openai-codex", model="gpt-5.6-luna"
    )
    request_line = json.dumps(
        {"invocation": invocation, "run": snapshot},
        sort_keys=True,
        separators=(",", ":"),
    )

    with _LocalCodexRelay() as relay:
        source_credentials, factory = prepare_provider_relay_credentials(
            _relay_credentials(relay.path, provider="openai-codex"),
            expected_provider="openai-codex",
            provider_relay_socket=relay.path,
        )
        output = io.StringIO()
        diagnostics = io.StringIO()
        with tempfile.TemporaryDirectory(prefix="hermes-plane-relay-home-") as home:
            with mock.patch.dict(os.environ, {"HERMES_HOME": home}), mock.patch(
                "agent.agent_init.fetch_model_metadata", return_value=None
            ), mock.patch(
                "agent.context_compressor.get_model_context_length", return_value=272000
            ):
                import run_agent

                previous_home = run_agent._hermes_home
                run_agent._hermes_home = Path(home)
                try:
                    status = serve_once_g1(
                        request_line,
                        output,
                        production=True,
                        diagnostics=diagnostics,
                        model_call_allowance=1,
                        credential_source=InlineCredentialSource(
                            source_credentials, "openai-codex"
                        ),
                        http_client_factory=factory,
                    )
                finally:
                    run_agent._hermes_home = previous_home

    assert status == 0
    frames = [json.loads(line) for line in output.getvalue().splitlines()]
    assert frames[-1]["kind"] == "completed"
    assert any(
        frame.get("body", {}).get("kind") == "transcript_evidence_observed"
        for frame in frames
    )
    assert '"protocol":"plane.agent-runtime/internal-usage/v1"' in diagnostics.getvalue()
    assert len(relay.requests) == 1
    request_line_seen, request_body = relay.requests[0]
    assert request_line_seen == "POST /backend-api/codex/responses HTTP/1.1"
    payload = json.loads(request_body)
    assert payload["model"] == "gpt-5.6-luna"
    assert "relay-secret" not in request_body.decode()


def test_openai_codex_outcome_unknown_stops_without_replay_or_fallback() -> None:
    snapshot, invocation, _request = _hermes_request(
        provider="openai-codex", model="gpt-5.6-luna"
    )
    request_line = json.dumps(
        {"invocation": invocation, "run": snapshot},
        sort_keys=True,
        separators=(",", ":"),
    )

    with _LocalCodexRelay(response=_outcome_unknown_response()) as relay:
        source_credentials, factory = prepare_provider_relay_credentials(
            _relay_credentials(relay.path, provider="openai-codex"),
            expected_provider="openai-codex",
            provider_relay_socket=relay.path,
        )
        output = io.StringIO()
        diagnostics = io.StringIO()
        with tempfile.TemporaryDirectory(prefix="hermes-plane-relay-home-") as home:
            with mock.patch.dict(os.environ, {"HERMES_HOME": home}), mock.patch(
                "agent.agent_init.fetch_model_metadata", return_value=None
            ), mock.patch(
                "agent.context_compressor.get_model_context_length", return_value=272000
            ):
                import run_agent

                previous_home = run_agent._hermes_home
                run_agent._hermes_home = Path(home)
                try:
                    with mock.patch.object(
                        run_agent.AIAgent, "_try_activate_fallback", return_value=False
                    ) as activate_fallback:
                        status = serve_once_g1(
                            request_line,
                            output,
                            production=True,
                            diagnostics=diagnostics,
                            model_call_allowance=12,
                            credential_source=InlineCredentialSource(
                                source_credentials, "openai-codex"
                            ),
                            http_client_factory=factory,
                        )
                finally:
                    run_agent._hermes_home = previous_home

    assert status == 0
    frames = [json.loads(line) for line in output.getvalue().splitlines()]
    assert frames[-1]["kind"] == "failed"
    assert frames[-1]["failure"] == {
        "code": "outcome_unknown",
        "message": "Provider outcome is unknown; Plane reconciliation is required before retrying.",
        "retryable": False,
    }
    assert len(relay.requests) == 1
    activate_fallback.assert_not_called()
    assert '"modelCalls":' in diagnostics.getvalue()
    assert "relay-secret" not in output.getvalue()
    assert "Return a deterministic runtime outcome." not in output.getvalue()


def test_provider_relay_latches_outcome_unknown_before_next_request() -> None:
    with _LocalCodexRelay(response=_outcome_unknown_response()) as relay:
        credentials = _relay_credentials(relay.path, provider="openai-codex")
        _source_credentials, factory = prepare_provider_relay_credentials(
            credentials,
            expected_provider="openai-codex",
            provider_relay_socket=relay.path,
        )
        client = factory()
        try:
            with pytest.raises(ProviderOutcomeUnknownError):
                client.post("/responses", json={"input": "first"})
            with pytest.raises(ProviderOutcomeUnknownError):
                client.post("/responses", json={"input": "must-not-send"})
        finally:
            client.close()

    assert len(relay.requests) == 1
    latch = getattr(factory, "_plane_provider_relay_outcome_unknown_latch")
    assert latch.is_set() is True


def test_openai_codex_midstream_failure_stops_without_replay_fallback_or_dump() -> None:
    snapshot, invocation, _request = _hermes_request(
        provider="openai-codex", model="gpt-5.6-luna"
    )
    request_line = json.dumps(
        {"invocation": invocation, "run": snapshot},
        sort_keys=True,
        separators=(",", ":"),
    )

    with _LocalCodexRelay(response=_broken_codex_sse_response()) as relay:
        source_credentials, factory = prepare_provider_relay_credentials(
            _relay_credentials(relay.path, provider="openai-codex"),
            expected_provider="openai-codex",
            provider_relay_socket=relay.path,
        )
        output = io.StringIO()
        diagnostics = io.StringIO()
        with tempfile.TemporaryDirectory(prefix="hermes-plane-relay-home-") as home:
            with mock.patch.dict(
                os.environ,
                {"HERMES_HOME": home, "HERMES_DUMP_REQUESTS": "1"},
            ), mock.patch(
                "agent.agent_init.fetch_model_metadata", return_value=None
            ), mock.patch(
                "agent.context_compressor.get_model_context_length", return_value=272000
            ):
                import run_agent

                previous_home = run_agent._hermes_home
                run_agent._hermes_home = Path(home)
                try:
                    with mock.patch.object(
                        run_agent.AIAgent, "_try_activate_fallback", return_value=False
                    ) as activate_fallback:
                        status = serve_once_g1(
                            request_line,
                            output,
                            production=True,
                            diagnostics=diagnostics,
                            model_call_allowance=12,
                            credential_source=InlineCredentialSource(
                                source_credentials, "openai-codex"
                            ),
                            http_client_factory=factory,
                        )
                finally:
                    run_agent._hermes_home = previous_home
            request_dumps = list(Path(home).rglob("request_dump_*.json"))

    assert status == 0
    frames = [json.loads(line) for line in output.getvalue().splitlines()]
    assert frames[-1]["kind"] == "failed"
    assert frames[-1]["failure"] == {
        "code": "outcome_unknown",
        "message": "Provider outcome is unknown; Plane reconciliation is required before retrying.",
        "retryable": False,
    }
    assert len(relay.requests) == 1
    activate_fallback.assert_not_called()
    assert request_dumps == []
    assert "relay-secret" not in output.getvalue()
    assert "Return a deterministic runtime outcome." not in output.getvalue()


def test_production_service_passes_only_dummy_openai_codex_credentials() -> None:
    snapshot, _invocation, request = _hermes_request(provider="openai-codex", model="gpt-5.5")
    socket_path = "/tmp/provider-relay.sock"
    frames = G1BootstrapFrames(
        1,
        _relay_credentials(socket_path, provider="openai-codex"),
        request,
    )
    payload = bytes(frames.child_bytes())
    frames.clear()
    output = io.StringIO()

    with mock.patch("plane_runtime.g1_service.HermesKernelAdapter") as adapter:
        def dispatch_result(_snapshot, _invocation, _cancellation, emit_body, **_kwargs):
            emit_body(
                {
                    "kind": "progress_observed",
                    "payload": {
                        "kind": "inline_text",
                        "contentType": "text/plain",
                        "text": "started",
                    },
                    "publication": {"action": "observation_only"},
                }
            )
            return HermesKernelResult(kind="completed")

        adapter.return_value.dispatch.side_effect = dispatch_result
        with mock.patch("sys.stdin", _BinaryStdin(payload)), mock.patch("sys.stdout", output):
            assert service_main(
                [
                    "--once",
                    "--g1-production",
                    "--g1-bootstrap-child",
                    "--provider-relay-socket",
                    socket_path,
                ]
            ) == 0

    adapter_kwargs = adapter.call_args.kwargs
    source = adapter_kwargs["credential_source"]
    assert isinstance(source, InlineCredentialSource)
    assert source.expected_provider == "openai-codex"
    assert source.credentials == {
        "api_key": "plane-provider-relay",
        "base_url": "http://plane-provider-relay.invalid/backend-api/codex",
        "api_mode": "codex_responses",
    }
    assert "relay-secret" not in output.getvalue()
    assert "chatgpt.com" not in output.getvalue()
    assert snapshot["runtimePolicy"]["model"]["provider"] == "openai-codex"  # type: ignore[index]


def test_default_adapter_factory_passes_openai_codex_identity_and_transport() -> None:
    snapshot, _invocation, _request = _hermes_request(
        provider="openai-codex", model="gpt-5.5"
    )
    snapshot_model = G1RunSnapshot.from_dict(snapshot)
    invocation_model = G1InvocationEnvelope.from_dict(make_invocation(snapshot_model.to_dict()))
    captured: dict[str, object] = {}
    factory = lambda: object()

    class FakeAgent:
        session_api_calls = 1

        def run_conversation(self, message: str, *, system_message: str) -> dict[str, str]:
            del message, system_message
            return {"final_response": "done"}

    def constructor(**kwargs: object) -> FakeAgent:
        captured.update(kwargs)
        return FakeAgent()

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = constructor  # type: ignore[attr-defined]
    relay_credentials = _relay_credentials(
        "/tmp/provider-relay.sock", provider="openai-codex"
    )
    agent_credentials, _relay_factory = prepare_provider_relay_credentials(
        relay_credentials,
        expected_provider="openai-codex",
        provider_relay_socket="/tmp/provider-relay.sock",
    )
    with mock.patch.dict(sys.modules, {"run_agent": fake_run_agent}):
        result = HermesKernelAdapter(
            credential_source=InlineCredentialSource(
                agent_credentials, "openai-codex"
            ),
            http_client_factory=factory,
        ).dispatch(
            snapshot_model,
            invocation_model,
            lambda: False,
            lambda _body: None,
            model_call_allowance=1,
        )

    assert result.kind == "completed"
    assert captured["provider"] == "openai-codex"
    assert captured["model"] == "gpt-5.5"
    assert captured["base_url"] == "http://plane-provider-relay.invalid/backend-api/codex"
    assert captured["api_mode"] == "codex_responses"
    assert captured["api_key"] == "plane-provider-relay"
    assert captured["http_client_factory"] is factory


def test_provider_relay_rejects_missing_or_invalid_socket() -> None:
    credentials = _relay_credentials("/tmp/provider-relay.sock")

    with pytest.raises(ValueError, match="socket"):
        prepare_provider_relay_credentials(
            credentials.copy(), expected_provider="xai", provider_relay_socket=None
        )
    with pytest.raises(ValueError, match="socket"):
        prepare_provider_relay_credentials(
            credentials.copy(), expected_provider="xai", provider_relay_socket="relative.sock"
        )
    with pytest.raises(ValueError, match="socket"):
        prepare_provider_relay_credentials(
            credentials.copy(), expected_provider="xai", provider_relay_socket="/tmp/" + "x" * 104
        )
    bad_host = credentials.copy()
    bad_host["host"] = "api.example.invalid"
    with pytest.raises(ValueError, match="metadata"):
        prepare_provider_relay_credentials(
            bad_host, expected_provider="xai", provider_relay_socket="/tmp/provider-relay.sock"
        )


def test_provider_relay_default_credentials_remain_unchanged() -> None:
    credentials = {"api_key": "existing-secret", "base_url": "https://example.invalid/v1"}
    agent_credentials, factory = prepare_provider_relay_credentials(
        credentials, expected_provider="xai", provider_relay_socket=None
    )
    assert agent_credentials is credentials
    assert factory is None
    assert credentials == {"api_key": "existing-secret", "base_url": "https://example.invalid/v1"}


def test_production_service_passes_relay_factory_and_only_dummy_credentials() -> None:
    snapshot, invocation, request = _hermes_request()
    socket_path = "/tmp/provider-relay.sock"
    frames = G1BootstrapFrames(1, _relay_credentials(socket_path), request)
    payload = bytes(frames.child_bytes())
    frames.clear()
    output = io.StringIO()

    with mock.patch("plane_runtime.g1_service.HermesKernelAdapter") as adapter:
        def dispatch_result(_snapshot, _invocation, _cancellation, emit_body, **_kwargs):
            emit_body(
                {
                    "kind": "progress_observed",
                    "payload": {
                        "kind": "inline_text",
                        "contentType": "text/plain",
                        "text": "started",
                    },
                    "publication": {"action": "observation_only"},
                }
            )
            return HermesKernelResult(kind="completed")

        adapter.return_value.dispatch.side_effect = dispatch_result
        with mock.patch("sys.stdin", _BinaryStdin(payload)), mock.patch("sys.stdout", output):
            assert service_main(
                [
                    "--once",
                    "--g1-production",
                    "--g1-bootstrap-child",
                    "--provider-relay-socket",
                    socket_path,
                ]
            ) == 0

    adapter_kwargs = adapter.call_args.kwargs
    source = adapter_kwargs["credential_source"]
    assert isinstance(source, InlineCredentialSource)
    assert source.credentials == {
        "api_key": "plane-provider-relay",
        "base_url": PROVIDER_RELAY_BASE_URL,
        "api_mode": "chat_completions",
    }
    assert callable(adapter_kwargs["http_client_factory"])
    assert "relay-secret" not in output.getvalue()
    assert "relayToken" not in output.getvalue()
    assert snapshot["runtimePolicy"]["model"]["provider"] == "xai"  # type: ignore[index]


def test_production_service_rejects_relay_credentials_without_socket() -> None:
    _snapshot, _invocation, request = _hermes_request()
    frames = G1BootstrapFrames(1, _relay_credentials("/tmp/provider-relay.sock"), request)
    payload = bytes(frames.child_bytes())
    frames.clear()

    with mock.patch("plane_runtime.g1_service.HermesKernelAdapter") as adapter:
        with mock.patch("sys.stdin", _BinaryStdin(payload)), mock.patch("sys.stdout", io.StringIO()):
            assert service_main(["--once", "--g1-production", "--g1-bootstrap-child"]) == 2
    adapter.assert_not_called()


def test_default_adapter_factory_passes_factory_to_exact_aiagent_constructor() -> None:
    snapshot, _invocation, _request = _hermes_request()
    snapshot_model = G1RunSnapshot.from_dict(snapshot)
    invocation_model = G1InvocationEnvelope.from_dict(make_invocation(snapshot_model.to_dict()))
    captured: dict[str, object] = {}
    factory = lambda: object()

    class FakeAgent:
        session_api_calls = 1

        def run_conversation(self, message: str, *, system_message: str) -> dict[str, str]:
            del message, system_message
            return {"final_response": "done"}

    def constructor(**kwargs: object) -> FakeAgent:
        captured.update(kwargs)
        return FakeAgent()

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = constructor  # type: ignore[attr-defined]
    with mock.patch.dict(sys.modules, {"run_agent": fake_run_agent}):
        result = HermesKernelAdapter(
            credential_source=InlineCredentialSource(
                {"api_key": "plane-provider-relay", "api_mode": "chat_completions"}, "xai"
            ),
            http_client_factory=factory,
        ).dispatch(
            snapshot_model,
            invocation_model,
            lambda: False,
            lambda _body: None,
            model_call_allowance=1,
        )

    assert result.kind == "completed"
    assert captured["http_client_factory"] is factory
    assert captured["api_key"] == "plane-provider-relay"
    assert captured["api_mode"] == "chat_completions"


def test_bootstrap_forwards_and_validates_provider_relay_socket() -> None:
    dispatch = json.dumps(
        {"modelCallAllowance": 1, "protocol": "plane.agent-runtime/dispatch-control/v1"},
        sort_keys=True,
        separators=(",", ":"),
    ).encode() + b"\n"
    control = json.dumps(
        {"credentials": {}, "protocol": "plane.agent-runtime/credential-control/v1"},
        sort_keys=True,
        separators=(",", ":"),
    ).encode() + b"\n"
    request = b'{"invocation":{},"run":{}}\n'

    with mock.patch.object(bootstrap, "_run", return_value=0) as run:
        with mock.patch("sys.stdin", _BinaryStdin(dispatch + control + request)):
            assert bootstrap.main(
                ["--once", "--g1-production", "--provider-relay-socket", "/tmp/provider-relay.sock"]
            ) == 0
    assert run.call_args.args[2] == "/tmp/provider-relay.sock"

    for invalid in ("relative.sock", "/tmp/" + "x" * 104, "/tmp/bad\npath"):
        run.reset_mock()
        with mock.patch("sys.stdin", _BinaryStdin(dispatch + control + request)):
            assert bootstrap.main(
                ["--once", "--g1-production", "--provider-relay-socket", invalid]
            ) == 2
        run.assert_not_called()
