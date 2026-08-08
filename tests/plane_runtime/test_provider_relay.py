"""Focused tests for the private provider-relay adapter boundary."""

from __future__ import annotations

import io
import json
import sys
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
    prepare_provider_relay_credentials,
)
from plane_runtime.service import main as service_main

from tests.plane_runtime.test_g1_runtime_process import make_invocation, make_snapshot, _digest


def _hermes_request() -> tuple[dict[str, object], dict[str, object], bytes]:
    snapshot = make_snapshot()
    snapshot["runtimePolicy"] = dict(snapshot["runtimePolicy"])  # type: ignore[arg-type]
    snapshot["runtimePolicy"]["adapter"] = "hermes"  # type: ignore[index]
    snapshot["runtimePolicy"]["model"] = {"provider": "xai", "model": "grok-test"}  # type: ignore[index]
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


def _relay_credentials(socket_path: str, token: str = "relay-secret") -> dict[str, str]:
    return {
        "host": "api.x.ai",
        "path": "/v1/chat/completions",
        "provider": "xai",
        "relayToken": token,
        "invocationSocket": socket_path,
    }


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
    import httpx

    request = httpx.Request("POST", PROVIDER_RELAY_BASE_URL + "/chat/completions")
    request.headers["Authorization"] = "Bearer plane-provider-relay"
    client.call_args_list[0].kwargs["event_hooks"]["request"][0](request)
    assert request.headers["Authorization"] == "Bearer relay-secret"
    assert request.headers["X-Plane-Relay-Provider"] == "xai"
    assert request.headers["X-Request-ID"]
    assert request.headers["X-Request-ID"] != "plane-provider-relay"


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
