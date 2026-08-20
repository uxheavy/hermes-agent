"""Focused tests for the credential-free Hermes Plane host seam."""

from __future__ import annotations

import hashlib
import json
import io
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping
import unittest
from unittest import mock

from plane_runtime.host_port import (
    CallablePlaneHostPort,
    PlaneHostBinding,
    PlaneHostBoundsError,
    PlaneHostCancelled,
    PlaneHostError,
    PlaneHostUnavailable,
    HostCallRequest,
    PLANE_OUTCOME_PUBLISH_OPERATION,
    UnixSocketPlaneHostPort,
    bind_plane_host,
    install_plane_tools,
    plane_code_mode,
)
from tools.registry import registry

_RuntimePlaneHostBinding = PlaneHostBinding
_TEST_EAGER_OPERATION_REFS = frozenset(
    {
        "operation:search_workspace",
        "operation:work_item.read",
        "operation:work-item-get@1",
        "operation:read@1",
        "operation:read@2",
        "operation:mutate@1",
        "operation:work-item-update@1",
        "operation:conversation-publish@1",
        PLANE_OUTCOME_PUBLISH_OPERATION,
        "operation:compose@1",
    }
)


class PlaneHostBinding(_RuntimePlaneHostBinding):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("eager_operation_refs", _TEST_EAGER_OPERATION_REFS)
        super().__init__(*args, **kwargs)


# Hermes' logging monitor may still emit after an AIAgent returns. Keep this
# hermetic home alive for the process instead of deleting it mid-test.
_TEST_HERMES_HOME = tempfile.TemporaryDirectory(prefix="hermes-plane-host-")


class _LocalHostServer:
    """One-request JSONL server used to exercise the real Unix socket seam."""

    def __init__(self, responder):
        self._directory = tempfile.TemporaryDirectory(prefix="plane-host-rpc-")
        self.path = os.path.join(self._directory.name, "host.sock")
        self._responder = responder
        self.requests: list[dict] = []
        self.raw_requests: list[bytes] = []
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server.bind(self.path)
        self._server.listen(4)
        self._server.settimeout(0.05)
        self._thread = threading.Thread(target=self._serve, daemon=True)

    def __enter__(self):
        self._thread.start()
        self._ready.wait(timeout=1)
        return self

    def __exit__(self, *_args):
        self._stop.set()
        self._server.close()
        self._thread.join(timeout=1)
        self._directory.cleanup()

    def _serve(self):
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
                while b"\n" not in raw and len(raw) <= 16 * 1024:
                    chunk = channel.recv(4096)
                    if not chunk:
                        break
                    raw.extend(chunk)
                if not raw.endswith(b"\n"):
                    continue
                self.raw_requests.append(bytes(raw))
                request = json.loads(bytes(raw[:-1]).decode("utf-8"))
                self.requests.append(request)
                response = self._responder(request)
                if response is not None:
                    try:
                        channel.sendall(response)
                    except OSError:
                        pass


def _result(request: dict, *, status: str = "ok", output: object = None, **extra: object) -> dict:
    return {
        "protocol": "plane.agent-runtime/v1",
        "requestRef": request["requestRef"],
        "correlationId": request["correlationId"],
        "idempotencyKey": request["idempotencyKey"],
        "status": status,
        "replayed": status == "replayed",
        "output": output if output is not None else {"accepted": True},
        **extra,
    }


def _applied_outcome_publication(
    *, operation_ref: str = PLANE_OUTCOME_PUBLISH_OPERATION
) -> dict[str, str]:
    return {
        "action": "applied",
        "productKind": "outcome_submission",
        "productRef": "outcome-submission:test",
        "operationAttemptRef": "operation-attempt:attempt-1",
        "operationRef": operation_ref,
        "applicationServiceRef": "application-service:conversation",
        "gatewayReceiptRef": "gateway-receipt:receipt-1",
        "receiptRef": "receipt:receipt-1",
        "auditReceiptRef": "audit-receipt:audit-1",
        "productEventRef": "product-event:event-1",
    }


class HostPortTests(unittest.TestCase):
    def test_host_operation_diagnostic_is_bounded_and_phase_exact(self) -> None:
        operation_ref = "operation:work-item-get@1"
        operation_ref_digest = hashlib.sha256(operation_ref.encode("utf-8")).hexdigest()
        raw_input = "input-secret-must-not-leak"
        raw_result = "result-secret-must-not-leak"

        returned = PlaneHostBinding(
            port=CallablePlaneHostPort(
                lambda request: _result(
                    request,
                    output={"rawResult": raw_result},
                )
            ),
            run_id="run:diagnostic",
            invocation_id="invocation:diagnostic",
            correlation_id="correlation:diagnostic",
            cancellation=lambda: False,
        )
        returned.call(
            action="read",
            operation_ref=operation_ref,
            input={"rawInput": raw_input},
            source="model",
        )
        self.assertEqual(
            returned.host_operation_diagnostic,
            {
                "callbackPhase": "host_return",
                "operationRefDigest": operation_ref_digest,
            },
        )

        before_call = PlaneHostBinding(
            port=CallablePlaneHostPort(
                lambda _request: (_ for _ in ()).throw(
                    RuntimeError("provider-secret-before-return")
                )
            ),
            run_id="run:diagnostic",
            invocation_id="invocation:diagnostic",
            correlation_id="correlation:diagnostic",
            cancellation=lambda: False,
        )
        with self.assertRaises(PlaneHostUnavailable):
            before_call.call(
                action="read",
                operation_ref=operation_ref,
                input={"rawInput": raw_input},
                source="model",
            )
        self.assertEqual(
            before_call.host_operation_diagnostic,
            {
                "callbackPhase": "before_host_call",
                "operationRefDigest": operation_ref_digest,
            },
        )

        observation_emit = PlaneHostBinding(
            port=CallablePlaneHostPort(
                lambda request: _result(request, output={"rawResult": raw_result})
            ),
            run_id="run:diagnostic",
            invocation_id="invocation:diagnostic",
            correlation_id="correlation:diagnostic",
            cancellation=lambda: False,
            emit_body=mock.Mock(side_effect=RuntimeError("provider-secret-observation")),
        )
        with self.assertRaises(PlaneHostUnavailable):
            observation_emit.call(
                action="read",
                operation_ref=operation_ref,
                input={"rawInput": raw_input},
                source="model",
            )
        self.assertEqual(
            observation_emit.host_operation_diagnostic,
            {
                "callbackPhase": "model_observation_emit",
                "operationRefDigest": operation_ref_digest,
            },
        )

        event_count = 0

        def emit_event(_body: Mapping[str, object]) -> None:
            nonlocal event_count
            event_count += 1
            if event_count == 2:
                raise RuntimeError("provider-secret-adapter-event")

        adapter_event = PlaneHostBinding(
            port=CallablePlaneHostPort(
                lambda request: _result(
                    request,
                    publication=_applied_outcome_publication(),
                )
            ),
            run_id="run:diagnostic",
            invocation_id="invocation:diagnostic",
            correlation_id="correlation:diagnostic",
            cancellation=lambda: False,
            emit_body=emit_event,
        )
        with self.assertRaises(PlaneHostUnavailable):
            adapter_event.publish(
                kind="outcome",
                operation_ref=PLANE_OUTCOME_PUBLISH_OPERATION,
                resource_ref="outcome-submission:diagnostic",
                content=raw_input,
            )
        adapter_diagnostic = adapter_event.host_operation_diagnostic
        assert adapter_diagnostic is not None
        self.assertEqual(adapter_diagnostic["callbackPhase"], "adapter_event")
        self.assertEqual(
            adapter_diagnostic["operationRefDigest"],
            hashlib.sha256(
                PLANE_OUTCOME_PUBLISH_OPERATION.encode("utf-8")
            ).hexdigest(),
        )

        serialized = json.dumps(
            {
                "returned": returned.host_operation_diagnostic,
                "beforeCall": before_call.host_operation_diagnostic,
                "observationEmit": observation_emit.host_operation_diagnostic,
                "adapterEvent": adapter_diagnostic,
            },
            sort_keys=True,
        )
        self.assertNotIn(raw_input, serialized)
        self.assertNotIn(raw_result, serialized)
        self.assertNotIn("provider-secret", serialized)

    def test_ambiguous_prepared_search_handoff_stays_pending_until_read(self) -> None:
        """A multi-result search cannot silently become an ordinary text exit."""

        def respond(request: dict) -> bytes:
            if request["operationRef"] == "operation:search_workspace":
                output = {
                    "ok": True,
                    "result": {
                        "results": [
                            {
                                "objectType": "work_item",
                                "workItemReadCall": {
                                    "action": "read",
                                    "operationRef": "operation:work_item.read",
                                    "input": {"preparedCallRef": "prepared-call:first"},
                                },
                            },
                            {
                                "objectType": "work_item",
                                "workItemReadCall": {
                                    "action": "read",
                                    "operationRef": "operation:work_item.read",
                                    "input": {"preparedCallRef": "prepared-call:second"},
                                },
                            },
                        ]
                    },
                }
            else:
                self.assertEqual(request["operationRef"], "operation:work_item.read")
                output = {"ok": True, "result": {"work_item": {"title": "assigned"}}}
            return (
                json.dumps(
                    _result(request, output=output),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
                + b"\n"
            )

        binding = PlaneHostBinding(
            port=CallablePlaneHostPort(lambda request: json.loads(respond(request))),
            run_id="run:ambiguous",
            invocation_id="invocation:ambiguous",
            correlation_id="correlation:ambiguous",
            cancellation=lambda: False,
        )
        search = binding.call(
            action="read",
            operation_ref="operation:search_workspace",
            input={"query": "assigned", "limit": 2},
            source="model",
        )

        assert search.status == "ok"
        assert binding.prepared_read_handoff_pending() is True

        read = binding.call(
            action="read",
            operation_ref="operation:work_item.read",
            input={"preparedCallRef": "prepared-call:first"},
            source="model",
        )
        assert read.status == "ok"
        assert binding.prepared_read_handoff_pending() is False

    def test_cross_process_model_search_consumes_prepared_read_before_text_exit(self) -> None:
        """A child using the real model-facing tool dispatch cannot exit after search."""

        def respond(request: dict) -> bytes:
            if request["operationRef"] == "operation:search_workspace":
                output = {
                    "ok": True,
                    "result": {
                        "results": [
                            {
                                "objectType": "work_item",
                                "workItemReadCall": {
                                    "action": "read",
                                    "operationRef": "operation:work_item.read",
                                    "input": {"preparedCallRef": "prepared-call:cross-process"},
                                },
                            }
                        ]
                    },
                }
            else:
                self.assertEqual(request["operationRef"], "operation:work_item.read")
                self.assertEqual(
                    request["input"],
                    {"preparedCallRef": "prepared-call:cross-process"},
                )
                output = {"ok": True, "result": {"work_item": {"title": "assigned"}}}
            return (
                json.dumps(
                    _result(request, output=output),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
                + b"\n"
            )

        child = """
import json
import sys
from plane_runtime.host_port import (
    PlaneHostBinding,
    UnixSocketPlaneHostPort,
    bind_plane_host,
    install_plane_tools,
)
from tools.registry import registry

binding = PlaneHostBinding(
    port=UnixSocketPlaneHostPort(sys.argv[1], timeout_seconds=2),
    run_id="run:cross-process",
    invocation_id="invocation:cross-process",
    correlation_id="correlation:cross-process",
    cancellation=lambda: False,
    eager_operation_refs=frozenset({"operation:search_workspace", "operation:work_item.read"}),
)
install_plane_tools()
with bind_plane_host(binding):
    result = registry.dispatch(
        "plane_operation",
        {
            "action": "read",
            "operationRef": "operation:search_workspace",
            "input": {"query": "assigned", "limit": 1},
        },
    )
    json.loads(result)
print("text_response")
"""

        with _LocalHostServer(respond) as server:
            completed = subprocess.run(
                [sys.executable, "-c", child, server.path],
                check=True,
                capture_output=True,
                text=True,
                env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[2])},
            )
            self.assertEqual(completed.stdout.strip(), "text_response")
            self.assertEqual(
                [request["operationRef"] for request in server.requests],
                ["operation:search_workspace", "operation:work_item.read"],
            )

    def test_production_one_shot_service_binds_socket_host_before_real_hermes_turn(self) -> None:
        from plane_runtime.hermes_adapter import HermesKernelAdapter
        from plane_runtime.g1_bootstrap_contract import G1BootstrapFrames
        from plane_runtime.service import main as service_main
        from run_agent import AIAgent
        from tests.plane_runtime.test_g1_runtime_process import (
            G1InvocationEnvelope,
            G1RunSnapshot,
            _digest,
            make_invocation,
            make_snapshot,
        )

        snapshot_raw = make_snapshot()
        snapshot_raw["runtimePolicy"] = dict(snapshot_raw["runtimePolicy"])  # type: ignore[index]
        snapshot_raw["runtimePolicy"]["adapter"] = "hermes"  # type: ignore[index]
        snapshot_raw["runtimePolicy"]["model"] = {  # type: ignore[index]
            "provider": "test-provider",
            "model": "deterministic-local",
        }
        snapshot_raw["toolCatalog"] = {
            **snapshot_raw["toolCatalog"],  # type: ignore[index]
            "eagerOperations": [
                *snapshot_raw["toolCatalog"]["eagerOperations"],  # type: ignore[index]
                {
                    "operationRef": "operation:search_workspace",
                    "schemaDigest": "content:" + "f" * 64,
                    "inputSchema": {"type": "object"},
                    "disclosure": "eager",
                },
            ],
        }
        snapshot_raw["contentDigest"] = _digest(  # type: ignore[assignment]
            "snapshot",
            {key: value for key, value in snapshot_raw.items() if key != "contentDigest"},
        )
        snapshot = G1RunSnapshot.from_dict(snapshot_raw)
        invocation = G1InvocationEnvelope.from_dict(make_invocation(snapshot.to_dict()))
        request_line = json.dumps(
            {"run": snapshot.to_dict(), "invocation": invocation.to_dict()},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ) + "\n"

        class Credentials:
            def resolve(self, provider: str) -> dict[str, str]:
                self.provider = provider
                return {"api_key": "model-only-secret", "base_url": "http://127.0.0.1"}

        class Completions:
            def __init__(self) -> None:
                self.calls = 0

            def create(self, **_kwargs):
                self.calls += 1
                if self.calls == 1:
                    call = SimpleNamespace(
                        id="call-search",
                        function=SimpleNamespace(
                            name="tool_call",
                            arguments=json.dumps(
                                {
                                    "name": "plane_operation",
                                    "arguments": {
                                        "action": "read",
                                        "operationRef": "operation:search_workspace",
                                        "input": {"query": "assigned", "limit": 1},
                                    },
                                }
                            ),
                        ),
                        extra_content=None,
                    )
                    message = SimpleNamespace(
                        content=None,
                        tool_calls=[call],
                        reasoning=None,
                        reasoning_content=None,
                        refusal=None,
                    )
                    finish_reason = "tool_calls"
                else:
                    message = SimpleNamespace(
                        content="ordinary final evidence",
                        tool_calls=None,
                        reasoning=None,
                        reasoning_content=None,
                        refusal=None,
                    )
                    finish_reason = "stop"
                return SimpleNamespace(
                    choices=[SimpleNamespace(finish_reason=finish_reason, message=message)],
                    usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
                )

        class Client:
            def __init__(self) -> None:
                self.chat = SimpleNamespace(completions=Completions())

        def agent_factory(**kwargs):
            agent = AIAgent(**kwargs)
            client = Client()
            agent._create_request_openai_client = lambda **_kwargs: client  # type: ignore[method-assign]
            return agent

        class BinaryStdin:
            def __init__(self, value: bytes) -> None:
                self.buffer = io.BytesIO(value)

        bootstrap_frames = G1BootstrapFrames(
            4,
            {"api_key": "model-only-secret", "base_url": "http://127.0.0.1"},
            request_line[:-1].encode(),
        )
        bootstrap_input = bytes(bootstrap_frames.child_bytes())
        bootstrap_frames.clear()

        bodies: list[dict] = []
        def respond(request: dict) -> bytes:
            if request["operationRef"] == "operation:search_workspace":
                output = {
                    "ok": True,
                    "result": {
                        "results": [
                            {
                                "objectType": "work_item",
                                "workItemReadCall": {
                                    "action": "read",
                                    "operationRef": "operation:work_item.read",
                                    "input": {"preparedCallRef": "prepared-call:opaque"},
                                },
                            }
                        ]
                    },
                }
            else:
                self.assertEqual(request["operationRef"], "operation:work_item.read")
                self.assertEqual(request["input"], {"preparedCallRef": "prepared-call:opaque"})
                output = {"ok": True, "result": {"work_item": {"title": "assigned"}}}
            return (
                json.dumps(
                    _result(request, output=output),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
                + b"\n"
            )

        with _LocalHostServer(respond) as server:
            output = io.StringIO()
            diagnostics = io.StringIO()
            import run_agent

            run_agent._hermes_home = Path(_TEST_HERMES_HOME.name)
            with mock.patch.object(
                HermesKernelAdapter,
                "_default_agent_factory",
                staticmethod(agent_factory),
            ), mock.patch("sys.stdin", BinaryStdin(bootstrap_input)), mock.patch(
                "sys.stdout", output
            ), mock.patch("sys.stderr", diagnostics), mock.patch.dict(
                os.environ, {"HERMES_HOME": _TEST_HERMES_HOME.name}
            ):
                status = service_main(
                    [
                        "--once",
                        "--g1-production",
                        "--g1-bootstrap-child",
                        "--model-call-allowance",
                        "4",
                        "--plane-host-socket",
                        server.path,
                    ]
                )

            self.assertEqual(status, 0)
            self.assertEqual(
                [request["action"] for request in server.requests], ["read", "read"]
            )
            self.assertEqual(
                [request["operationRef"] for request in server.requests],
                ["operation:search_workspace", "operation:work_item.read"],
            )
            self.assertEqual(
                server.requests[1]["input"], {"preparedCallRef": "prepared-call:opaque"}
            )
            self.assertNotIn(server.path, output.getvalue())
            self.assertIn('"modelCalls":2', diagnostics.getvalue())
            self.assertTrue(
                any(
                    frame.get("body", {}).get("kind")
                    == "transcript_evidence_observed"
                    for frame in map(json.loads, output.getvalue().splitlines())
                )
            )

    def test_unix_socket_client_round_trips_exact_canonical_contract(self) -> None:
        with _LocalHostServer(
            lambda request: (
                json.dumps(
                    _result(request, output={"accepted": True}),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
                + b"\n"
            )
        ) as server:
            binding = PlaneHostBinding(
                port=UnixSocketPlaneHostPort(server.path, timeout_seconds=1),
                run_id="run:test",
                invocation_id="invocation:test",
                correlation_id="correlation:test",
                cancellation=lambda: False,
            )
            result = binding.call(
                action="read",
                operation_ref="operation:work-item-get@1",
                input={"workItemRef": "work-item:test"},
                source="model",
            )

            self.assertEqual(result.status, "ok")
            self.assertEqual(server.requests[0]["protocol"], "plane.agent-runtime/v1")
            self.assertEqual(
                server.raw_requests[0],
                json.dumps(
                    server.requests[0],
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
                + b"\n",
            )
            self.assertEqual(
                set(server.requests[0]),
                {
                    "protocol",
                    "runId",
                    "invocationId",
                    "correlationId",
                    "action",
                    "operationRef",
                    "input",
                    "source",
                    "requestRef",
                    "idempotencyKey",
                },
            )
            self.assertNotIn(server.path, json.dumps(server.requests))

    def test_unix_socket_client_fails_closed_on_missing_malformed_and_mismatched_results(self) -> None:
        request_ref = "host-request:changed"

        def malformed(_request):
            return b'{"output":null,"output":null}\n'

        with _LocalHostServer(malformed) as server:
            port = UnixSocketPlaneHostPort(server.path, timeout_seconds=1)
            request = PlaneHostBinding(
                port=port,
                run_id="run:test",
                invocation_id="invocation:test",
                correlation_id="correlation:test",
                cancellation=lambda: False,
            )
            with self.assertRaises(PlaneHostUnavailable) as raised:
                request.call(
                    action="read",
                    operation_ref="operation:read@1",
                    input={},
                    source="model",
                )
            self.assertNotIn(server.path, str(raised.exception))

        def mismatched(request):
            response = _result(request)
            response["requestRef"] = request_ref
            return (
                json.dumps(response, sort_keys=True, separators=(",", ":")).encode()
                + b"\n"
            )

        with _LocalHostServer(mismatched) as server:
            binding = PlaneHostBinding(
                port=UnixSocketPlaneHostPort(server.path),
                run_id="run:test",
                invocation_id="invocation:test",
                correlation_id="correlation:test",
                cancellation=lambda: False,
            )
            with self.assertRaises(PlaneHostUnavailable):
                binding.call(
                    action="read",
                    operation_ref="operation:read@1",
                    input={},
                    source="model",
                )

        def rejected(raw_response):
            responder = raw_response if callable(raw_response) else lambda _request: raw_response
            with _LocalHostServer(responder) as server:
                binding = PlaneHostBinding(
                    port=UnixSocketPlaneHostPort(server.path),
                    run_id="run:test",
                    invocation_id="invocation:test",
                    correlation_id="correlation:test",
                    cancellation=lambda: False,
                )
                with self.assertRaises(PlaneHostError):
                    binding.call(
                        action="read",
                        operation_ref="operation:read@1",
                        input={},
                        source="model",
                    )

        def unknown(request):
            response = _result(request)
            response["unknown"] = True
            return json.dumps(response, sort_keys=True, separators=(",", ":")).encode() + b"\n"

        def truncated(request):
            return json.dumps(_result(request), sort_keys=True, separators=(",", ":")).encode()

        rejected(unknown)
        rejected(truncated)
        rejected(lambda _request: b"{" + (b"x" * (17 * 1024)) + b"}\n")
        rejected(lambda _request: None)

        missing = os.path.join(tempfile.gettempdir(), "plane-host-rpc-missing.sock")
        with self.assertRaises(PlaneHostUnavailable) as raised:
            UnixSocketPlaneHostPort(missing).invoke(
                HostCallRequest(
                    run_id="run:test",
                    invocation_id="invocation:test",
                    correlation_id="correlation:test",
                    action="read",
                    operation_ref="operation:read@1",
                    input={},
                    source="model",
                )
            )
        self.assertNotIn(missing, str(raised.exception))

    def test_unix_socket_client_honors_deadline_and_active_cancellation(self) -> None:
        release = threading.Event()

        def delayed(request):
            release.wait(timeout=1)
            return (
                json.dumps(_result(request), sort_keys=True, separators=(",", ":")).encode()
                + b"\n"
            )

        cancelled = threading.Event()
        with _LocalHostServer(delayed) as server:
            binding = PlaneHostBinding(
                port=UnixSocketPlaneHostPort(
                    server.path,
                    timeout_seconds=1,
                    cancellation=lambda: cancelled.is_set(),
                ),
                run_id="run:test",
                invocation_id="invocation:test",
                correlation_id="correlation:test",
                cancellation=lambda: cancelled.is_set(),
            )
            threading.Timer(0.05, cancelled.set).start()
            with self.assertRaises(PlaneHostCancelled):
                binding.call(
                    action="read",
                    operation_ref="operation:read@1",
                    input={},
                    source="model",
                )
            release.set()

        timeout_release = threading.Event()

        def timeout_delayed(request):
            timeout_release.wait(timeout=1)
            return (
                json.dumps(_result(request), sort_keys=True, separators=(",", ":")).encode()
                + b"\n"
            )

        with _LocalHostServer(timeout_delayed) as server:
            started = time.monotonic()
            with self.assertRaises(PlaneHostUnavailable):
                UnixSocketPlaneHostPort(server.path, timeout_seconds=0.05).invoke(
                    HostCallRequest(
                        run_id="run:test",
                        invocation_id="invocation:test",
                        correlation_id="correlation:test",
                        action="read",
                        operation_ref="operation:read@1",
                        input={},
                        source="model",
                    )
                )
            self.assertLess(time.monotonic() - started, 0.5)
            timeout_release.set()

    def test_request_identity_is_stable_and_credential_free(self) -> None:
        requests: list[dict] = []

        def rpc(request: dict) -> dict:
            requests.append(request)
            return _result(request, output={"value": "read"})

        port = CallablePlaneHostPort(rpc)
        first = PlaneHostBinding(
            port=port,
            run_id="run:test",
            invocation_id="invocation:test",
            correlation_id="correlation:test",
            cancellation=lambda: False,
        ).call(
            action="read",
            operation_ref="operation:work-item-get@1",
            input={"workItemRef": "work-item:test"},
            source="model",
        )
        second = PlaneHostBinding(
            port=port,
            run_id="run:test",
            invocation_id="invocation:test",
            correlation_id="correlation:test",
            cancellation=lambda: False,
        ).call(
            action="read",
            operation_ref="operation:work-item-get@1",
            input={"workItemRef": "work-item:test"},
            source="model",
        )

        self.assertEqual(requests[0]["requestRef"], requests[1]["requestRef"])
        self.assertEqual(requests[0]["idempotencyKey"], requests[1]["idempotencyKey"])
        self.assertEqual(first.model_payload(), second.model_payload())
        self.assertNotIn("credential", json.dumps(requests))
        self.assertNotIn("api_key", json.dumps(requests))

    def test_denied_authorization_is_recoverable_but_unknown_outcome_stays_fatal(self) -> None:
        calls = 0

        def denied_then_ok(request: dict) -> dict:
            nonlocal calls
            calls += 1
            if calls == 1:
                return _result(
                    request,
                    status="denied",
                    errorCode="NOT_AUTHORIZED",
                    errorMessage="operation is not authorized",
                )
            return _result(request, output={"accepted": True})

        binding = PlaneHostBinding(
            port=CallablePlaneHostPort(denied_then_ok),
            run_id="run:test",
            invocation_id="invocation:test",
            correlation_id="correlation:test",
            cancellation=lambda: False,
        )
        denied = binding.call(
            action="read",
            operation_ref="operation:read@1",
            input={},
            source="model",
        )
        recovered = binding.call(
            action="read",
            operation_ref="operation:read@2",
            input={},
            source="model",
        )
        self.assertEqual(denied.status, "denied")
        self.assertEqual(recovered.status, "ok")
        self.assertIsNone(binding.fatal_error)

        unknown = PlaneHostBinding(
            port=CallablePlaneHostPort(
                lambda request: _result(
                    request,
                    status="unavailable",
                    errorCode="OUTCOME_UNKNOWN",
                    errorMessage="host outcome is unknown",
                )
            ),
            run_id="run:test",
            invocation_id="invocation:test",
            correlation_id="correlation:test",
            cancellation=lambda: False,
        )
        unknown_result = unknown.call(
            action="mutate",
            operation_ref="operation:mutate@1",
            input={},
            source="model",
        )
        self.assertEqual(unknown_result.status, "unavailable")
        self.assertIsNotNone(unknown.fatal_error)

    def test_code_mode_failure_is_bounded_and_corrected_call_can_continue(self) -> None:
        calls = 0

        def failed_then_ok(request: dict) -> dict:
            nonlocal calls
            calls += 1
            if calls == 1:
                return _result(
                    request,
                    status="invalid",
                    errorCode="CODE_MODE_FAILED",
                    errorMessage="Code Mode module failed in the restricted isolate",
                )
            return _result(request, output={"operationId": "work_item.rename"})

        binding = PlaneHostBinding(
            port=CallablePlaneHostPort(failed_then_ok),
            run_id="run:test",
            invocation_id="invocation:test",
            correlation_id="correlation:test",
            cancellation=lambda: False,
        )
        failed = binding.call(
            action="code",
            operation_ref="plane.code-mode.execute@1",
            input={"source": "malformed-generated-module"},
            source="code",
        )
        corrected = binding.call(
            action="code",
            operation_ref="plane.code-mode.execute@1",
            input={"source": "corrected-generated-module"},
            source="code",
        )

        self.assertEqual(failed.status, "invalid")
        self.assertEqual(failed.error_code, "CODE_MODE_FAILED")
        self.assertEqual(corrected.status, "ok")
        self.assertEqual(corrected.output, {"operationId": "work_item.rename"})
        self.assertIsNone(binding.fatal_error)

    def test_plane_conflict_is_model_observable_but_idempotency_conflict_stays_fatal(self) -> None:
        conflict = PlaneHostBinding(
            port=CallablePlaneHostPort(
                lambda request: _result(
                    request,
                    status="conflict",
                    errorCode="PLANE_CONFLICT",
                    errorMessage="the invocation already has an applied outcome publication",
                )
            ),
            run_id="run:test",
            invocation_id="invocation:test",
            correlation_id="correlation:test",
            cancellation=lambda: False,
        )
        result = conflict.call(
            action="mutate",
            operation_ref="operation:work-item-update@1",
            input={},
            source="model",
        )
        self.assertEqual(result.status, "conflict")
        self.assertEqual(result.error_code, "PLANE_CONFLICT")
        self.assertEqual(json.loads(result.model_payload())["errorCode"], "PLANE_CONFLICT")
        self.assertIsNone(conflict.fatal_error)

        idempotency_conflict = PlaneHostBinding(
            port=CallablePlaneHostPort(
                lambda request: _result(
                    request,
                    status="conflict",
                    errorCode="IDEMPOTENCY_CONFLICT",
                    errorMessage="the request is not the original operation",
                )
            ),
            run_id="run:test",
            invocation_id="invocation:test",
            correlation_id="correlation:test",
            cancellation=lambda: False,
        )
        idempotency_result = idempotency_conflict.call(
            action="mutate",
            operation_ref="operation:work-item-update@1",
            input={},
            source="model",
        )
        self.assertEqual(idempotency_result.status, "conflict")
        self.assertIsNotNone(idempotency_conflict.fatal_error)

    def test_terminal_action_returns_conflict_for_differing_duplicate_and_late_mutation(self) -> None:
        calls: list[dict] = []

        def rpc(request: dict) -> dict:
            calls.append(request)
            if request["action"] == "publish":
                return _result(
                    request,
                    output={"published": True},
                    publication=_applied_outcome_publication(),
                )
            return _result(request, output={"accepted": True})

        binding = PlaneHostBinding(
            port=CallablePlaneHostPort(rpc),
            run_id="run:test",
            invocation_id="invocation:test",
            correlation_id="correlation:test",
            cancellation=lambda: False,
        )
        binding.publish(
            kind="outcome",
            operation_ref=PLANE_OUTCOME_PUBLISH_OPERATION,
            resource_ref="outcome-submission:test",
            content="first outcome",
        )

        differing_duplicate = binding.publish(
            kind="outcome",
            operation_ref=PLANE_OUTCOME_PUBLISH_OPERATION,
            resource_ref="outcome-submission:test",
            content="different outcome",
        )
        late_mutation = binding.call(
            action="mutate",
            operation_ref="operation:work-item-update@1",
            input={"workItemRef": "work-item:test", "title": "late"},
            source="model",
        )

        for result in (differing_duplicate, late_mutation):
            self.assertEqual(result.status, "conflict")
            self.assertEqual(result.error_code, "PLANE_CONFLICT")
            self.assertEqual(json.loads(result.model_payload())["status"], "conflict")
        self.assertEqual(len(calls), 1)
        self.assertIsNone(binding.fatal_error)

    def test_successful_host_result_with_error_fields_remains_rejected(self) -> None:
        binding = PlaneHostBinding(
            port=CallablePlaneHostPort(
                lambda request: _result(
                    request,
                    errorCode="SHOULD_NOT_BE_ACCEPTED",
                    errorMessage="successful result with an error",
                )
            ),
            run_id="run:test",
            invocation_id="invocation:test",
            correlation_id="correlation:test",
            cancellation=lambda: False,
        )

        with self.assertRaises(PlaneHostUnavailable):
            binding.call(
                action="read",
                operation_ref="operation:read@1",
                input={},
                source="model",
            )
        self.assertIsNotNone(binding.fatal_error)

    def test_nonrecoverable_host_results_poison_the_invocation(self) -> None:
        cases = (
            ("denied", "CALLBACK_BINDING_INVALID"),
            ("invalid", "UNKNOWN_VALIDATION_FAILURE"),
            ("conflict", "IDEMPOTENCY_CONFLICT"),
            ("unavailable", "OUTCOME_UNKNOWN"),
        )

        for status, error_code in cases:
            with self.subTest(status=status, error_code=error_code):
                binding = PlaneHostBinding(
                    port=CallablePlaneHostPort(
                        lambda request, status=status, error_code=error_code: _result(
                            request,
                            status=status,
                            errorCode=error_code,
                            errorMessage="bounded host failure",
                        )
                    ),
                    run_id="run:test",
                    invocation_id="invocation:test",
                    correlation_id="correlation:test",
                    cancellation=lambda: False,
                )

                result = binding.call(
                    action="read",
                    operation_ref="operation:read@1",
                    input={},
                    source="model",
                )

                self.assertEqual(result.status, status)
                self.assertIsNotNone(binding.fatal_error)

    def test_changed_replay_binding_and_noncanonical_response_fail_closed(self) -> None:
        def changed_response(request: dict) -> dict:
            response = _result(request)
            response["requestRef"] = "host-request:changed"
            return response

        binding = PlaneHostBinding(
            port=CallablePlaneHostPort(changed_response),
            run_id="run:test",
            invocation_id="invocation:test",
            correlation_id="correlation:test",
            cancellation=lambda: False,
        )
        with self.assertRaises(PlaneHostUnavailable):
            binding.call(
                action="mutate",
                operation_ref="operation:work-item-update@1",
                input={"workItemRef": "work-item:test", "title": "new"},
                source="model",
            )
        self.assertIsNotNone(binding.fatal_error)

        def noncanonical_response(request: dict) -> str:
            return json.dumps(_result(request), indent=2)

        binding = PlaneHostBinding(
            port=CallablePlaneHostPort(noncanonical_response),
            run_id="run:test",
            invocation_id="invocation:test",
            correlation_id="correlation:test",
            cancellation=lambda: False,
        )
        with self.assertRaises(PlaneHostUnavailable):
            binding.call(
                action="read",
                operation_ref="operation:work-item-get@1",
                input={},
                source="model",
            )

    def test_cancellation_and_call_bounds_are_invocation_scoped(self) -> None:
        cancelled = True
        binding = PlaneHostBinding(
            port=CallablePlaneHostPort(lambda request: _result(request)),
            run_id="run:test",
            invocation_id="invocation:test",
            correlation_id="correlation:test",
            cancellation=lambda: cancelled,
        )
        with self.assertRaises(PlaneHostCancelled):
            binding.call(
                action="read",
                operation_ref="operation:read@1",
                input={},
                source="model",
            )

        calls = []
        bounded = PlaneHostBinding(
            port=CallablePlaneHostPort(lambda request: calls.append(request) or _result(request)),
            run_id="run:test",
            invocation_id="invocation:test",
            correlation_id="correlation:test",
            cancellation=lambda: False,
            max_calls=1,
        )
        bounded.call(
            action="read",
            operation_ref="operation:read@1",
            input={},
            source="model",
        )
        with self.assertRaises(ValueError):
            bounded.call(
                action="read",
                operation_ref="operation:read@1",
                input={},
                source="model",
            )
        self.assertEqual(len(calls), 1)

        oversized = PlaneHostBinding(
            port=CallablePlaneHostPort(lambda request: _result(request)),
            run_id="run:test",
            invocation_id="invocation:test",
            correlation_id="correlation:test",
            cancellation=lambda: False,
        )
        with self.assertRaises(PlaneHostBoundsError):
            oversized.call(
                action="read",
                operation_ref="operation:read@1",
                input={"value": "x" * 9000},
                source="model",
            )
        self.assertIsNotNone(oversized.fatal_error)

    def test_explicit_publication_requires_a_gateway_publication_shape(self) -> None:
        bodies: list[dict] = []

        def rpc(request: dict) -> dict:
            return _result(
                request,
                output={"published": True},
                publication={
                    "action": "applied",
                    "productKind": "conversation",
                    "productRef": "conversation:conversation-1",
                    "operationAttemptRef": "operation-attempt:attempt-1",
                    "operationRef": "operation:conversation-publish@1",
                    "applicationServiceRef": "application-service:conversation",
                    "gatewayReceiptRef": "gateway-receipt:receipt-1",
                    "receiptRef": "receipt:receipt-1",
                    "auditReceiptRef": "audit-receipt:audit-1",
                    "productEventRef": "product-event:event-1",
                },
            )

        binding = PlaneHostBinding(
            port=CallablePlaneHostPort(rpc),
            run_id="run:test",
            invocation_id="invocation:test",
            correlation_id="correlation:test",
            cancellation=lambda: False,
            emit_body=bodies.append,
        )
        binding.publish(
            kind="conversation",
            operation_ref="operation:conversation-publish@1",
            resource_ref="conversation:conversation-1",
            content="explicit product action",
        )
        self.assertEqual(
            [body["kind"] for body in bodies],
            ["progress_observed", "conversation_publication_observed"],
        )
        self.assertIsNone(binding.terminal_action_reason())
        self.assertEqual(binding.publication_count, 1)

    def test_terminal_action_does_not_repeat_host_publish_after_applied_outcome(self) -> None:
        bodies: list[dict] = []
        calls: list[dict] = []

        def rpc(request: dict) -> dict:
            calls.append(request)
            if len(calls) > 1:
                return _result(
                    request,
                    status="conflict",
                    output={"error": "duplicate publication"},
                    errorCode="IDEMPOTENCY_CONFLICT",
                    errorMessage="the outcome was already published",
                )
            return _result(
                request,
                output={"published": True},
                publication={
                    "action": "applied",
                    "productKind": "outcome_submission",
                    "productRef": "outcome-submission:test",
                    "operationAttemptRef": "operation-attempt:attempt-1",
                    "operationRef": PLANE_OUTCOME_PUBLISH_OPERATION,
                    "applicationServiceRef": "application-service:conversation",
                    "gatewayReceiptRef": "gateway-receipt:receipt-1",
                    "receiptRef": "receipt:receipt-1",
                    "auditReceiptRef": "audit-receipt:audit-1",
                    "productEventRef": "product-event:event-1",
                },
            )

        binding = PlaneHostBinding(
            port=CallablePlaneHostPort(rpc),
            run_id="run:test",
            invocation_id="invocation:test",
            correlation_id="correlation:test",
            cancellation=lambda: False,
            emit_body=bodies.append,
        )
        first = binding.publish(
            kind="outcome",
            operation_ref=PLANE_OUTCOME_PUBLISH_OPERATION,
            resource_ref="outcome-submission:test",
            content="bounded outcome",
        )
        duplicate = binding.publish(
            kind="outcome",
            operation_ref=PLANE_OUTCOME_PUBLISH_OPERATION,
            resource_ref="outcome-submission:test",
            content="bounded outcome",
        )

        self.assertEqual(binding.terminal_action_reason(), "product_outcome_published")
        self.assertEqual(duplicate, first)
        self.assertEqual(len(calls), 1)
        self.assertIsNone(binding.fatal_error)
        self.assertEqual(
            [body["kind"] for body in bodies],
            ["progress_observed", "outcome_submission_observed"],
        )

    def test_replayed_outcome_publication_retains_nonarming_metadata(self) -> None:
        def rpc(request: dict) -> dict:
            return _result(
                request,
                status="replayed",
                publication=_applied_outcome_publication(),
            )

        binding = PlaneHostBinding(
            port=CallablePlaneHostPort(rpc),
            run_id="run:test",
            invocation_id="invocation:test",
            correlation_id="correlation:test",
            cancellation=lambda: False,
            emit_body=lambda _body: None,
        )
        binding.publish(
            kind="outcome",
            operation_ref=PLANE_OUTCOME_PUBLISH_OPERATION,
            resource_ref="outcome-submission:test",
            content="replayed outcome",
        )

        self.assertIsNone(binding.terminal_action_reason())
        self.assertEqual(
            binding.outcome_publication_metadata(),
            {
                "status": "replayed",
                "replayed": True,
                "publication_action": "applied",
                "operation_ref": PLANE_OUTCOME_PUBLISH_OPERATION,
                "terminal_armed": False,
            },
        )

    def test_generic_outcome_publication_arms_and_dedicated_route_reuses_receipt(self) -> None:
        bodies: list[dict] = []
        calls: list[dict] = []

        def rpc(request: dict) -> dict:
            calls.append(request)
            return _result(
                request,
                output={"published": True},
                publication=_applied_outcome_publication(),
            )

        binding = PlaneHostBinding(
            port=CallablePlaneHostPort(rpc),
            run_id="run:test",
            invocation_id="invocation:test",
            correlation_id="correlation:test",
            cancellation=lambda: False,
            emit_body=bodies.append,
        )
        install_plane_tools()
        with bind_plane_host(binding):
            generic_payload = registry.dispatch(
                "plane_operation",
                {
                    "action": "mutate",
                    "operationRef": PLANE_OUTCOME_PUBLISH_OPERATION,
                    "input": {
                        "run_ref": "run:test",
                        "outcome_ref": "outcome-submission:test",
                        "content": "generic outcome",
                    },
                },
            )

        self.assertEqual(json.loads(generic_payload)["status"], "ok")
        original = binding.records[0].result
        dedicated = binding.publish(
            kind="outcome",
            operation_ref=PLANE_OUTCOME_PUBLISH_OPERATION,
            resource_ref="outcome-submission:test",
            content="generic outcome",
        )
        self.assertIs(dedicated, original)
        self.assertEqual(calls[0]["action"], "mutate")
        self.assertEqual(len(calls), 1)
        self.assertEqual(binding.terminal_action_reason(), "product_outcome_published")
        self.assertEqual(
            [body["kind"] for body in bodies],
            ["progress_observed", "outcome_submission_observed"],
        )

    def test_generic_outcome_receipt_in_output_arms_terminal_publication(self) -> None:
        bodies: list[dict] = []
        calls: list[dict] = []

        def rpc(request: dict) -> dict:
            calls.append(request)
            return _result(
                request,
                output={
                    "ok": True,
                    "replayed": False,
                    "operationRef": PLANE_OUTCOME_PUBLISH_OPERATION,
                    "requestId": "request-1",
                    "gatewayReceipt": "gateway-1",
                    "auditReceipt": "audit-1",
                    "result": {
                        "outcome": {
                            "productEventRef": "product-event:event-1",
                        }
                    },
                },
            )

        binding = PlaneHostBinding(
            port=CallablePlaneHostPort(rpc),
            run_id="run:test",
            invocation_id="invocation:test",
            correlation_id="correlation:test",
            cancellation=lambda: False,
            emit_body=bodies.append,
        )
        generic = binding.call(
            action="mutate",
            operation_ref=PLANE_OUTCOME_PUBLISH_OPERATION,
            input={
                "run_ref": "run:test",
                "outcome_ref": "outcome-submission:test",
                "content": "generic outcome",
            },
            source="model",
        )
        dedicated = binding.publish(
            kind="outcome",
            operation_ref=PLANE_OUTCOME_PUBLISH_OPERATION,
            resource_ref="outcome-submission:test",
            content="generic outcome",
        )

        self.assertIs(dedicated, generic)
        self.assertEqual(len(calls), 1)
        self.assertEqual(binding.terminal_action_reason(), "product_outcome_published")
        self.assertIsNone(binding.fatal_error)
        self.assertEqual(
            [body["kind"] for body in bodies],
            ["progress_observed", "outcome_submission_observed"],
        )
        self.assertEqual(
            bodies[1]["publication"],
            {
                "action": "applied",
                "productKind": "outcome_submission",
                "productRef": "outcome-submission:test",
                "operationAttemptRef": "operation-attempt:request-1",
                "operationRef": PLANE_OUTCOME_PUBLISH_OPERATION,
                "applicationServiceRef": "application-service:agent-lifecycle",
                "gatewayReceiptRef": "gateway-receipt:gateway-1",
                "receiptRef": "receipt:request-1",
                "auditReceiptRef": "audit-receipt:audit-1",
                "productEventRef": "product-event:event-1",
            },
        )

    def test_code_mode_outcome_publication_uses_the_same_observation_seam(self) -> None:
        bodies: list[dict] = []
        calls: list[dict] = []

        def rpc(request: dict) -> dict:
            calls.append(request)
            return _result(
                request,
                publication=_applied_outcome_publication(),
            )

        binding = PlaneHostBinding(
            port=CallablePlaneHostPort(rpc),
            run_id="run:test",
            invocation_id="invocation:test",
            correlation_id="correlation:test",
            cancellation=lambda: False,
            emit_body=bodies.append,
        )
        install_plane_tools()
        with bind_plane_host(binding), plane_code_mode():
            code_payload = registry.dispatch(
                "plane_operation",
                {
                    "action": "code",
                    "operationRef": PLANE_OUTCOME_PUBLISH_OPERATION,
                    "input": {
                        "run_ref": "run:test",
                        "outcome_ref": "outcome-submission:test",
                        "content": "code mode outcome",
                    },
                },
            )

        self.assertEqual(json.loads(code_payload)["status"], "ok")
        self.assertEqual([(request["action"], request["source"]) for request in calls], [("code", "code")])
        self.assertEqual(binding.terminal_action_reason(), "product_outcome_published")
        self.assertEqual(
            [body["kind"] for body in bodies],
            ["progress_observed", "outcome_submission_observed"],
        )

    def test_generic_replay_denial_failure_and_other_operation_never_arm(self) -> None:
        cases = (
            ("replayed", None, _applied_outcome_publication()),
            ("denied", "NOT_AUTHORIZED", None),
            ("unavailable", "OUTCOME_UNKNOWN", None),
        )
        for status, error_code, publication in cases:
            with self.subTest(status=status):
                def rpc(request: dict, *, status=status, error_code=error_code, publication=publication) -> dict:
                    extra: dict[str, object] = {}
                    if error_code is not None:
                        extra.update(
                            errorCode=error_code,
                            errorMessage="test host result",
                        )
                    if publication is not None:
                        extra["publication"] = publication
                    return _result(request, status=status, **extra)

                binding = PlaneHostBinding(
                    port=CallablePlaneHostPort(rpc),
                    run_id="run:test",
                    invocation_id="invocation:test",
                    correlation_id="correlation:test",
                    cancellation=lambda: False,
                    emit_body=lambda _body: None,
                )
                binding.call(
                    action="mutate",
                    operation_ref=PLANE_OUTCOME_PUBLISH_OPERATION,
                    input={
                        "run_ref": "run:test",
                        "outcome_ref": "outcome-submission:test",
                        "content": "outcome",
                    },
                    source="model",
                )
                self.assertIsNone(binding.terminal_action_reason())

        def other_rpc(request: dict) -> dict:
            return _result(
                request,
                publication=_applied_outcome_publication(),
            )

        other = PlaneHostBinding(
            port=CallablePlaneHostPort(other_rpc),
            run_id="run:test",
            invocation_id="invocation:test",
            correlation_id="correlation:test",
            cancellation=lambda: False,
            eager_operation_refs=_TEST_EAGER_OPERATION_REFS | {"operation:other"},
            emit_body=lambda _body: None,
        )
        other.call(
            action="mutate",
            operation_ref="operation:other",
            input={"content": "not an outcome operation"},
            source="model",
        )
        self.assertIsNone(other.terminal_action_reason())

    def test_generic_and_dedicated_missing_or_malformed_publication_fail_closed(self) -> None:
        for route in ("generic", "dedicated"):
            with self.subTest(route=route):
                def missing_rpc(request: dict) -> dict:
                    return _result(request)

                binding = PlaneHostBinding(
                    port=CallablePlaneHostPort(missing_rpc),
                    run_id="run:test",
                    invocation_id="invocation:test",
                    correlation_id="correlation:test",
                    cancellation=lambda: False,
                    emit_body=lambda _body: None,
                )
                with self.assertRaises(PlaneHostUnavailable):
                    if route == "generic":
                        binding.call(
                            action="mutate",
                            operation_ref=PLANE_OUTCOME_PUBLISH_OPERATION,
                            input={
                                "run_ref": "run:test",
                                "outcome_ref": "outcome-submission:test",
                                "content": "outcome",
                            },
                            source="model",
                        )
                    else:
                        binding.publish(
                            kind="outcome",
                            operation_ref=PLANE_OUTCOME_PUBLISH_OPERATION,
                            resource_ref="outcome-submission:test",
                            content="outcome",
                        )
                self.assertIsNone(binding.terminal_action_reason())
                self.assertIsNotNone(binding.fatal_error)

        def malformed_rpc(request: dict) -> dict:
            publication = _applied_outcome_publication()
            publication["productKind"] = "conversation"
            return _result(request, publication=publication)

        malformed = PlaneHostBinding(
            port=CallablePlaneHostPort(malformed_rpc),
            run_id="run:test",
            invocation_id="invocation:test",
            correlation_id="correlation:test",
            cancellation=lambda: False,
            emit_body=lambda _body: None,
        )
        with self.assertRaises(PlaneHostUnavailable):
            malformed.call(
                action="mutate",
                operation_ref=PLANE_OUTCOME_PUBLISH_OPERATION,
                input={
                    "run_ref": "run:test",
                    "outcome_ref": "outcome-submission:test",
                    "content": "outcome",
                },
                source="model",
            )
        self.assertIsNone(malformed.terminal_action_reason())

    def test_concurrent_generic_and_dedicated_publications_share_one_receipt(self) -> None:
        calls: list[dict] = []
        calls_lock = threading.Lock()
        bodies: list[dict] = []

        def rpc(request: dict) -> dict:
            with calls_lock:
                calls.append(request)
            time.sleep(0.01)
            return _result(request, publication=_applied_outcome_publication())

        binding = PlaneHostBinding(
            port=CallablePlaneHostPort(rpc),
            run_id="run:test",
            invocation_id="invocation:test",
            correlation_id="correlation:test",
            cancellation=lambda: False,
            emit_body=bodies.append,
        )

        def generic() -> object:
            return binding.call(
                action="mutate",
                operation_ref=PLANE_OUTCOME_PUBLISH_OPERATION,
                input={
                    "run_ref": "run:test",
                    "outcome_ref": "outcome-submission:test",
                    "content": "concurrent outcome",
                },
                source="model",
            )

        def dedicated() -> object:
            return binding.publish(
                kind="outcome",
                operation_ref=PLANE_OUTCOME_PUBLISH_OPERATION,
                resource_ref="outcome-submission:test",
                content="concurrent outcome",
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = [executor.submit(generic), executor.submit(dedicated)]
            returned = [future.result() for future in results]

        self.assertEqual(len(calls), 1)
        self.assertIs(returned[0], returned[1])
        self.assertIs(returned[0], binding.records[0].result)
        self.assertEqual(binding.terminal_action_reason(), "product_outcome_published")
        self.assertEqual(
            sum(body["kind"] == "outcome_submission_observed" for body in bodies),
            1,
        )

    def test_pre_terminal_host_failure_remains_fatal_after_later_terminal(self) -> None:
        calls: list[dict] = []

        def rpc(request: dict) -> dict:
            calls.append(request)
            if request["action"] == "read":
                return _result(
                    request,
                    status="unavailable",
                    errorCode="OPERATION_UNAVAILABLE",
                    errorMessage="required operation is unavailable",
                )
            return _result(
                request,
                output={"published": True},
                publication={
                    "action": "applied",
                    "productKind": "outcome_submission",
                    "productRef": "outcome-submission:test",
                    "operationAttemptRef": "operation-attempt:attempt-1",
                    "operationRef": PLANE_OUTCOME_PUBLISH_OPERATION,
                    "applicationServiceRef": "application-service:conversation",
                    "gatewayReceiptRef": "gateway-receipt:receipt-1",
                    "receiptRef": "receipt:receipt-1",
                    "auditReceiptRef": "audit-receipt:audit-1",
                    "productEventRef": "product-event:event-1",
                },
            )

        binding = PlaneHostBinding(
            port=CallablePlaneHostPort(rpc),
            run_id="run:test",
            invocation_id="invocation:test",
            correlation_id="correlation:test",
            cancellation=lambda: False,
            emit_body=lambda _body: None,
        )
        failed = binding.call(
            action="read",
            operation_ref="operation:read@1",
            input={},
            source="model",
        )
        binding.publish(
            kind="outcome",
            operation_ref=PLANE_OUTCOME_PUBLISH_OPERATION,
            resource_ref="outcome-submission:test",
            content="bounded outcome",
        )

        self.assertEqual(failed.status, "unavailable")
        self.assertEqual(binding.terminal_action_reason(), "product_outcome_published")
        self.assertFalse(binding.fatal_error_after_terminal)
        self.assertIsNotNone(binding.fatal_error)
        self.assertEqual([request["action"] for request in calls], ["read", "publish"])

    def test_replayed_applied_outcome_on_fresh_binding_does_not_signal(self) -> None:
        bodies: list[dict] = []

        def rpc(request: dict) -> dict:
            return _result(
                request,
                status="replayed",
                output={"published": True},
                publication={
                    "action": "applied",
                    "productKind": "outcome_submission",
                    "productRef": "outcome-submission:test",
                    "operationAttemptRef": "operation-attempt:attempt-1",
                    "operationRef": PLANE_OUTCOME_PUBLISH_OPERATION,
                    "applicationServiceRef": "application-service:conversation",
                    "gatewayReceiptRef": "gateway-receipt:receipt-1",
                    "receiptRef": "receipt:receipt-1",
                    "auditReceiptRef": "audit-receipt:audit-1",
                    "productEventRef": "product-event:event-1",
                },
            )

        binding = PlaneHostBinding(
            port=CallablePlaneHostPort(rpc),
            run_id="run:test",
            invocation_id="invocation:test",
            correlation_id="correlation:test",
            cancellation=lambda: False,
            emit_body=bodies.append,
        )
        binding.publish(
            kind="outcome",
            operation_ref=PLANE_OUTCOME_PUBLISH_OPERATION,
            resource_ref="outcome-submission:test",
            content="replayed outcome",
        )

        self.assertIsNone(binding.terminal_action_reason())
        self.assertEqual(
            [body["kind"] for body in bodies],
            ["progress_observed", "outcome_submission_observed"],
        )

    def test_nonterminal_publications_and_plain_calls_do_not_signal(self) -> None:
        def conversation_rpc(request: dict) -> dict:
            return _result(
                request,
                publication={
                    "action": "proposal",
                    "productKind": "conversation",
                    "productRef": "conversation:test",
                    "operationAttemptRef": "operation-attempt:attempt-1",
                },
            )

        binding = PlaneHostBinding(
            port=CallablePlaneHostPort(conversation_rpc),
            run_id="run:test",
            invocation_id="invocation:test",
            correlation_id="correlation:test",
            cancellation=lambda: False,
            emit_body=lambda _body: None,
        )
        binding.publish(
            kind="conversation",
            operation_ref="operation:conversation-publish@1",
            resource_ref="conversation:test",
            content="conversation evidence",
        )
        binding.call(
            action="read",
            operation_ref="operation:read@1",
            input={},
            source="model",
        )
        self.assertIsNone(binding.terminal_action_reason())

        def denied_rpc(request: dict) -> dict:
            return _result(
                request,
                status="denied",
                errorCode="NOT_AUTHORIZED",
                errorMessage="publication is not authorized",
            )

        denied = PlaneHostBinding(
            port=CallablePlaneHostPort(denied_rpc),
            run_id="run:test",
            invocation_id="invocation:test",
            correlation_id="correlation:test",
            cancellation=lambda: False,
            emit_body=lambda _body: None,
        )
        denied.publish(
            kind="outcome",
            operation_ref=PLANE_OUTCOME_PUBLISH_OPERATION,
            resource_ref="outcome-submission:test",
            content="denied outcome",
        )
        self.assertIsNone(denied.terminal_action_reason())
        self.assertIsNotNone(denied.fatal_error)

        failed = PlaneHostBinding(
            port=CallablePlaneHostPort(
                lambda request: _result(
                    request,
                    status="unavailable",
                    errorCode="OUTCOME_UNKNOWN",
                    errorMessage="publication outcome is unknown",
                )
            ),
            run_id="run:test",
            invocation_id="invocation:test",
            correlation_id="correlation:test",
            cancellation=lambda: False,
            emit_body=lambda _body: None,
        )
        failed.publish(
            kind="outcome",
            operation_ref=PLANE_OUTCOME_PUBLISH_OPERATION,
            resource_ref="outcome-submission:test",
            content="failed outcome",
        )
        self.assertIsNone(failed.terminal_action_reason())
        self.assertIsNotNone(failed.fatal_error)

        def applied_rpc(request: dict) -> dict:
            return _result(
                request,
                publication={
                    "action": "applied",
                    "productKind": "outcome_submission",
                    "productRef": "outcome-submission:test",
                    "operationAttemptRef": "operation-attempt:attempt-1",
                    "operationRef": PLANE_OUTCOME_PUBLISH_OPERATION,
                    "applicationServiceRef": "application-service:conversation",
                    "gatewayReceiptRef": "gateway-receipt:receipt-1",
                    "receiptRef": "receipt:receipt-1",
                    "auditReceiptRef": "audit-receipt:audit-1",
                    "productEventRef": "product-event:event-1",
                },
            )

        emission_failed = PlaneHostBinding(
            port=CallablePlaneHostPort(applied_rpc),
            run_id="run:test",
            invocation_id="invocation:test",
            correlation_id="correlation:test",
            cancellation=lambda: False,
            emit_body=mock.Mock(side_effect=RuntimeError("event sink failed")),
        )
        with self.assertRaises(PlaneHostUnavailable):
            emission_failed.publish(
                kind="outcome",
                operation_ref=PLANE_OUTCOME_PUBLISH_OPERATION,
                resource_ref="outcome-submission:test",
                content="unobserved outcome",
            )
        self.assertIsNone(emission_failed.terminal_action_reason())

        def non_applied_replay(request: dict) -> dict:
            return _result(
                request,
                status="replayed",
                publication={
                    "action": "proposal",
                    "productKind": "conversation",
                    "productRef": "conversation:test",
                    "operationAttemptRef": "operation-attempt:attempt-1",
                },
            )

        replayed = PlaneHostBinding(
            port=CallablePlaneHostPort(non_applied_replay),
            run_id="run:test",
            invocation_id="invocation:test",
            correlation_id="correlation:test",
            cancellation=lambda: False,
            emit_body=lambda _body: None,
        )
        replayed.publish(
            kind="conversation",
            operation_ref="operation:conversation-publish@1",
            resource_ref="conversation:test",
            content="replayed conversation",
        )
        self.assertIsNone(replayed.terminal_action_reason())

    def test_tools_are_registry_dispatchable_only_inside_a_bound_context(self) -> None:
        install_plane_tools()
        self.assertIsNotNone(registry.get_entry("plane_operation"))
        self.assertIsNotNone(registry.get_entry("plane_publish"))
        unbound = registry.dispatch(
            "plane_operation",
            {"action": "read", "operationRef": "operation:read@1", "input": {}},
        )
        self.assertIn("error", unbound)

        requests: list[dict] = []

        def rpc(request: dict) -> dict:
            requests.append(request)
            return _result(request, output={"read": True})

        binding = PlaneHostBinding(
            port=CallablePlaneHostPort(rpc),
            run_id="run:test",
            invocation_id="invocation:test",
            correlation_id="correlation:test",
            cancellation=lambda: False,
        )
        with bind_plane_host(binding):
            rejected_code = registry.dispatch(
                "plane_operation",
                {"action": "code", "operationRef": "operation:compose@1", "input": {}},
            )
            self.assertIn("restricted to plane_execute_typescript", rejected_code)
            self.assertEqual(len(requests), 0)
            with plane_code_mode():
                code_result = registry.dispatch(
                    "plane_operation",
                    {"action": "code", "operationRef": "operation:compose@1", "input": {}},
                )
        self.assertIn('"read":true', code_result)
        self.assertEqual(len(requests), 1)

    def test_registry_normalizes_exact_ready_to_call_prepared_read_envelope(self) -> None:
        install_plane_tools()
        requests: list[dict] = []

        def rpc(request: dict) -> dict:
            requests.append(request)
            return _result(request, output={"work_item": {"title": "assigned"}})

        binding = PlaneHostBinding(
            port=CallablePlaneHostPort(rpc),
            run_id="run:prepared-envelope",
            invocation_id="invocation:prepared-envelope",
            correlation_id="correlation:prepared-envelope",
            cancellation=lambda: False,
        )
        ready_to_call = {
            "action": "read",
            "operationRef": "operation:work_item.read",
            "input": {"preparedCallRef": "prepared-call:opaque"},
        }
        with bind_plane_host(binding):
            result = registry.dispatch(
                "plane_operation",
                {
                    "action": "read",
                    "operationRef": "operation:work_item.read",
                    "input": ready_to_call,
                },
            )

        self.assertIn('"status":"ok"', result)
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]["input"], {"preparedCallRef": "prepared-call:opaque"})

    def test_prepared_read_normalization_fails_closed_for_tampered_envelope(self) -> None:
        install_plane_tools()
        requests: list[dict] = []

        def rpc(request: dict) -> dict:
            requests.append(request)
            return _result(request, output={"accepted": True})

        binding = PlaneHostBinding(
            port=CallablePlaneHostPort(rpc),
            run_id="run:prepared-envelope-tamper",
            invocation_id="invocation:prepared-envelope-tamper",
            correlation_id="correlation:prepared-envelope-tamper",
            cancellation=lambda: False,
        )
        tampered = {
            "action": "read",
            "operationRef": "operation:work_item.read",
            "input": {
                "preparedCallRef": "prepared-call:opaque",
                "issue_id": "should-not-be-forwarded",
            },
        }
        with bind_plane_host(binding):
            result = registry.dispatch(
                "plane_operation",
                {
                    "action": "read",
                    "operationRef": "operation:work_item.read",
                    "input": tampered,
                },
            )

        self.assertIn('"status":"ok"', result)
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]["input"], tampered)

    def test_real_aiagent_loop_reaches_read_mutation_code_and_explicit_publication(self) -> None:
        """Use a deterministic provider boundary, not a fake Hermes agent."""

        from tests.plane_runtime.test_g1_runtime_process import (
            G1InvocationEnvelope,
            G1RunSnapshot,
            _digest,
            make_invocation,
            make_snapshot,
        )
        from plane_runtime.hermes_adapter import HermesKernelAdapter

        snapshot_raw = make_snapshot()
        snapshot_raw["runtimePolicy"] = dict(snapshot_raw["runtimePolicy"])  # type: ignore[index]
        snapshot_raw["runtimePolicy"]["adapter"] = "hermes"  # type: ignore[index]
        snapshot_raw["runtimePolicy"]["model"] = {  # type: ignore[index]
            "provider": "openai",
            "model": "deterministic-local",
        }
        snapshot_raw["runtimePolicy"].update(  # type: ignore[union-attr]
            {
                "maxCodeModeInputBytes": 65_536,
                "maxCodeModeOutputBytes": 65_536,
                "maxCodeModeCalls": 4,
            }
        )
        snapshot_raw["contentDigest"] = _digest(  # type: ignore[assignment]
            "snapshot",
            {key: value for key, value in snapshot_raw.items() if key != "contentDigest"},
        )
        snapshot = G1RunSnapshot.from_dict(snapshot_raw)
        invocation = G1InvocationEnvelope.from_dict(make_invocation(snapshot.to_dict()))

        model_tool_names: list[list[str]] = []

        class DeterministicCompletions:
            def __init__(self) -> None:
                self.calls = 0

            @staticmethod
            def tool_call(name: str, arguments: dict[str, object], call_id: str):
                return SimpleNamespace(
                    id=call_id,
                    function=SimpleNamespace(
                        name=name,
                        arguments=json.dumps(arguments),
                    ),
                    extra_content=None,
                )

            def create(self, **kwargs: object):
                self.calls += 1
                model_tool_names.append(
                    [
                        tool["function"]["name"]
                        for tool in kwargs.get("tools", [])  # type: ignore[union-attr]
                    ]
                )
                if self.calls == 1:
                    tool_calls = [
                        self.tool_call(
                            "tool_call",
                            {
                                "name": "plane_operation",
                                "arguments": {"action": "discover", "input": {}},
                            },
                            "call-discover",
                        )
                    ]
                elif self.calls == 2:
                    tool_calls = [
                        self.tool_call(
                            "tool_call",
                            {
                                "name": "plane_operation",
                                "arguments": {
                                    "action": "read",
                                    "operationRef": "operation:work-item-get",
                                    "input": {"workItemRef": "work-item:test"},
                                },
                            },
                            "call-read",
                        )
                    ]
                elif self.calls == 3:
                    tool_calls = [
                        self.tool_call(
                            "plane_execute_typescript",
                            {
                                "typescript_source": (
                                    "export default ({ input }: { input: Record<string, unknown> }) => ({"
                                    " accepted: true, input"
                                    "});"
                                )
                            },
                            "call-code",
                        )
                    ]
                elif self.calls == 4:
                    tool_calls = [
                        self.tool_call(
                            "tool_call",
                            {
                                "name": "plane_operation",
                                "arguments": {
                                    "action": "mutate",
                                    "operationRef": "operation:work-item-update",
                                    "input": {
                                        "workItemRef": "work-item:test",
                                        "title": "updated",
                                    },
                                },
                            },
                            "call-mutate",
                        )
                    ]
                elif self.calls == 5:
                    tool_calls = [
                        self.tool_call(
                            "tool_call",
                            {
                                "name": "plane_publish",
                                "arguments": {
                                    "kind": "conversation",
                                    "operationRef": "operation:conversation-publish",
                                    "resourceRef": "conversation:test",
                                    "content": "publish only through this explicit action",
                                },
                            },
                            "call-publish",
                        )
                    ]
                else:
                    tool_calls = []
                if tool_calls:
                    message = SimpleNamespace(
                        content=None,
                        tool_calls=tool_calls,
                        reasoning=None,
                        reasoning_content=None,
                        refusal=None,
                    )
                    choice = SimpleNamespace(finish_reason="tool_calls", message=message)
                else:
                    message = SimpleNamespace(
                        content="ordinary final evidence",
                        tool_calls=None,
                        reasoning=None,
                        reasoning_content=None,
                        refusal=None,
                    )
                    choice = SimpleNamespace(finish_reason="stop", message=message)
                return SimpleNamespace(
                    choices=[choice],
                    usage=SimpleNamespace(
                        prompt_tokens=1,
                        completion_tokens=1,
                        total_tokens=2,
                    ),
                )

        class DeterministicClient:
            def __init__(self) -> None:
                self.chat = SimpleNamespace(completions=DeterministicCompletions())

        class ModelCredentials:
            def resolve(self, provider: str) -> dict[str, str]:
                self.provider = provider
                return {
                    "api_key": "model-only-secret",
                    "base_url": "http://127.0.0.1",
                    "api_mode": "chat_completions",
                }

        requests: list[dict] = []

        def rpc(request: dict) -> dict:
            requests.append(request)
            response: dict[str, object] = _result(
                request,
                output={"operationRef": request["operationRef"], "accepted": True},
            )
            if request["action"] == "publish":
                response["publication"] = {
                    "action": "applied",
                    "productKind": "conversation",
                    "productRef": "conversation:test",
                    "operationAttemptRef": "operation-attempt:attempt-1",
                    "operationRef": "operation:conversation-publish",
                    "applicationServiceRef": "application-service:conversation",
                    "gatewayReceiptRef": "gateway-receipt:receipt-1",
                    "receiptRef": "receipt:receipt-1",
                    "auditReceiptRef": "audit-receipt:audit-1",
                    "productEventRef": "product-event:event-1",
                }
            return response

        from run_agent import AIAgent

        def agent_factory(**kwargs: object) -> AIAgent:
            agent = AIAgent(**kwargs)
            client = DeterministicClient()
            # This replaces only the model/provider boundary. The real Hermes
            # AIAgent constructor, tool loop, registry, and Plane Code Mode path
            # remain active.
            agent._create_request_openai_client = lambda **_: client  # type: ignore[method-assign]
            return agent

        bodies: list[dict] = []
        with mock.patch.dict(os.environ, {"HERMES_HOME": _TEST_HERMES_HOME.name}):
            import run_agent

            run_agent._hermes_home = Path(_TEST_HERMES_HOME.name)
            result = HermesKernelAdapter(
                agent_factory=agent_factory,
                credential_source=ModelCredentials(),
                host_port=CallablePlaneHostPort(rpc),
            ).dispatch(
                snapshot,
                invocation,
                lambda: False,
                bodies.append,
                model_call_allowance=8,
            )

        self.assertEqual(result.kind, "completed")
        self.assertFalse(any("execute_code" in names for names in model_tool_names))
        self.assertEqual(
            [(request["action"], request["source"]) for request in requests],
            [
                ("discover", "model"),
                ("read", "model"),
                ("code", "code"),
                ("mutate", "model"),
                ("publish", "model"),
            ],
        )
        self.assertEqual(len({request["correlationId"] for request in requests}), 1)
        self.assertNotIn("model-only-secret", json.dumps(requests + bodies))
        self.assertNotIn("plane-secret", json.dumps(requests + bodies))
        self.assertEqual(
            [body["kind"] for body in bodies if "publication" in body and body["publication"].get("action") != "observation_only"],
            ["conversation_publication_observed"],
        )
        self.assertEqual(
            [body["payload"]["text"] for body in bodies if body["kind"] == "transcript_evidence_observed"],
            ["ordinary final evidence"],
        )

    def test_real_hermes_adapter_stops_after_generic_applied_outcome_without_fake_final_text(self) -> None:
        from tests.plane_runtime.test_g1_runtime_process import (
            G1InvocationEnvelope,
            G1RunSnapshot,
            _digest,
            make_invocation,
            make_snapshot,
        )
        from plane_runtime.hermes_adapter import HermesKernelAdapter
        from run_agent import AIAgent

        snapshot_raw = make_snapshot()
        snapshot_raw["runtimePolicy"] = dict(snapshot_raw["runtimePolicy"])  # type: ignore[index]
        snapshot_raw["runtimePolicy"]["adapter"] = "hermes"  # type: ignore[index]
        snapshot_raw["runtimePolicy"]["model"] = {  # type: ignore[index]
            "provider": "openai",
            "model": "deterministic-local",
        }
        snapshot_raw["contentDigest"] = _digest(  # type: ignore[assignment]
            "snapshot",
            {key: value for key, value in snapshot_raw.items() if key != "contentDigest"},
        )
        snapshot = G1RunSnapshot.from_dict(snapshot_raw)
        invocation = G1InvocationEnvelope.from_dict(make_invocation(snapshot.to_dict()))

        class Completions:
            def __init__(self) -> None:
                self.calls = 0

            def create(self, **_kwargs: object):
                self.calls += 1
                def tool_call(call_id: str) -> SimpleNamespace:
                    return SimpleNamespace(
                        id=call_id,
                        function=SimpleNamespace(
                            name="tool_call",
                            arguments=json.dumps(
                                {
                                    "name": "plane_operation",
                                    "arguments": {
                                        "action": "mutate",
                                        "operationRef": PLANE_OUTCOME_PUBLISH_OPERATION,
                                        "input": {
                                            "run_ref": "run:test",
                                            "outcome_ref": "outcome-submission:test",
                                            "content": "bounded outcome",
                                        },
                                    },
                                }
                            ),
                        ),
                        extra_content=None,
                    )

                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            finish_reason="tool_calls",
                            message=SimpleNamespace(
                                content=None,
                                tool_calls=[
                                    tool_call("call-outcome"),
                                    tool_call("call-duplicate-outcome"),
                                ],
                                reasoning=None,
                                reasoning_content=None,
                                refusal=None,
                            ),
                        )
                    ],
                    usage=SimpleNamespace(
                        prompt_tokens=1,
                        completion_tokens=1,
                        total_tokens=2,
                    ),
                )

        class Client:
            def __init__(self) -> None:
                self.completions = Completions()
                self.chat = SimpleNamespace(completions=self.completions)

        class Credentials:
            def resolve(self, provider: str) -> dict[str, str]:
                del provider
                return {
                    "api_key": "model-only-secret",
                    "base_url": "http://127.0.0.1",
                    "api_mode": "chat_completions",
                }

        client = Client()

        def agent_factory(**kwargs: object) -> AIAgent:
            agent = AIAgent(**kwargs)
            agent._create_request_openai_client = lambda **_: client  # type: ignore[method-assign]
            return agent

        requests: list[dict] = []

        def rpc(request: dict) -> dict:
            requests.append(request)
            if len(requests) > 1:
                return _result(
                    request,
                    status="conflict",
                    output={"error": "duplicate publication"},
                    errorCode="IDEMPOTENCY_CONFLICT",
                    errorMessage="the outcome was already published",
                )
            return _result(
                request,
                output={"published": True},
                publication={
                    "action": "applied",
                    "productKind": "outcome_submission",
                    "productRef": "outcome-submission:test",
                    "operationAttemptRef": "operation-attempt:attempt-1",
                    "operationRef": PLANE_OUTCOME_PUBLISH_OPERATION,
                    "applicationServiceRef": "application-service:conversation",
                    "gatewayReceiptRef": "gateway-receipt:receipt-1",
                    "receiptRef": "receipt:receipt-1",
                    "auditReceiptRef": "audit-receipt:audit-1",
                    "productEventRef": "product-event:event-1",
                },
            )

        bodies: list[dict] = []
        with mock.patch.dict(os.environ, {"HERMES_HOME": _TEST_HERMES_HOME.name}):
            import run_agent

            run_agent._hermes_home = Path(_TEST_HERMES_HOME.name)
            result = HermesKernelAdapter(
                agent_factory=agent_factory,
                credential_source=Credentials(),
                host_port=CallablePlaneHostPort(rpc),
            ).dispatch(
                snapshot,
                invocation,
                lambda: False,
                bodies.append,
                model_call_allowance=3,
            )

        self.assertEqual(result.kind, "completed")
        self.assertEqual(result.output_text, "")
        self.assertEqual(result.model_calls, 1)
        self.assertEqual(client.completions.calls, 1)
        self.assertEqual(len(requests), 1)
        self.assertEqual(
            sum(body["kind"] == "outcome_submission_observed" for body in bodies),
            1,
        )
        self.assertEqual(sum(body["kind"] == "usage_observed" for body in bodies), 1)
        self.assertFalse(any(body["kind"] == "transcript_evidence_observed" for body in bodies))
        self.assertNotIn("Hermes invocation completed.", json.dumps(bodies))
        lifecycle_events = [
            json.loads(body["payload"]["text"])
            for body in bodies
            if body["kind"] == "progress_observed"
            and body.get("payload", {}).get("kind") == "inline_text"
            and body["payload"]["text"].startswith('{"category":"terminal_lifecycle"')
        ]
        self.assertEqual(len(lifecycle_events), 1)
        self.assertEqual(
            lifecycle_events[0],
            {
                "category": "terminal_lifecycle",
                "hook_installed": True,
                "protocol": "hermes.terminal-lifecycle/v1",
                "terminal_action_observed": True,
                "terminal_reason": "product_outcome_published",
                "terminal_action": {
                    "api_call_count": 1,
                    "iteration_budget_remaining": 2,
                    "iteration_budget_used": 1,
                    "provider_responses": 1,
                    "observed_at": "post_tool_batch",
                    "reason": "product_outcome_published",
                },
                "outcome_publication": {
                    "operation_ref": PLANE_OUTCOME_PUBLISH_OPERATION,
                    "publication_action": "applied",
                    "replayed": False,
                    "status": "ok",
                    "terminal_armed": True,
                },
                "finalization": {
                    "api_call_count": 1,
                    "exit_reason_after_mapping": "terminal_action",
                    "exit_reason_before_mapping": "terminal_action",
                    "iteration_budget_max_total": 3,
                    "iteration_budget_remaining": 2,
                    "iteration_budget_used": 1,
                    "max_iterations": 3,
                    "provider_responses": 1,
                },
            },
        )
        self.assertNotIn("bounded outcome", json.dumps(lifecycle_events))

    def test_terminal_action_emits_existing_tool_turn_text_as_transcript_evidence(self) -> None:
        from tests.plane_runtime.test_g1_runtime_process import (
            G1InvocationEnvelope,
            G1RunSnapshot,
            make_invocation,
            make_snapshot,
        )
        from plane_runtime.hermes_adapter import HermesKernelAdapter

        snapshot = G1RunSnapshot.from_dict(make_snapshot())
        invocation = G1InvocationEnvelope.from_dict(make_invocation(snapshot.to_dict()))

        class TerminalAgent:
            session_input_tokens = 1
            session_output_tokens = 1
            session_api_calls = 1

            def run_conversation(self, *_args: object, **_kwargs: object) -> dict[str, object]:
                return {
                    # This is the assistant message persisted before tool
                    # execution.  The terminal action leaves final_response
                    # unset; the adapter must not invent replacement prose.
                    "final_response": None,
                    "turn_exit_reason": "terminal_action(product_outcome_published)",
                    "messages": [
                        {
                            "role": "assistant",
                            "content": "ordinary final evidence",
                            "tool_calls": [{"function": {"name": "plane_publish"}}],
                        },
                        {"role": "tool", "name": "plane_publish", "content": "published"},
                    ],
                }

        class Credentials:
            def resolve(self, provider: str) -> dict[str, str]:
                del provider
                return {"api_key": "model-only-secret"}

        bodies: list[dict] = []
        result = HermesKernelAdapter(
            agent_factory=lambda **_kwargs: TerminalAgent(),
            credential_source=Credentials(),
        ).dispatch(
            snapshot,
            invocation,
            lambda: False,
            bodies.append,
            model_call_allowance=1,
        )

        self.assertEqual(result.kind, "completed")
        self.assertEqual(result.output_text, "ordinary final evidence")
        self.assertEqual(
            [body["payload"]["text"] for body in bodies if body["kind"] == "transcript_evidence_observed"],
            ["ordinary final evidence"],
        )

    def test_terminal_action_barrier_skips_later_host_mutation(self) -> None:
        from tests.plane_runtime.test_g1_runtime_process import (
            G1InvocationEnvelope,
            G1RunSnapshot,
            _digest,
            make_invocation,
            make_snapshot,
        )
        from plane_runtime.hermes_adapter import HermesKernelAdapter
        from run_agent import AIAgent

        snapshot_raw = make_snapshot()
        snapshot_raw["runtimePolicy"] = dict(snapshot_raw["runtimePolicy"])  # type: ignore[index]
        snapshot_raw["runtimePolicy"]["adapter"] = "hermes"  # type: ignore[index]
        snapshot_raw["runtimePolicy"]["model"] = {  # type: ignore[index]
            "provider": "openai",
            "model": "deterministic-local",
        }
        snapshot_raw["contentDigest"] = _digest(  # type: ignore[assignment]
            "snapshot",
            {key: value for key, value in snapshot_raw.items() if key != "contentDigest"},
        )
        snapshot = G1RunSnapshot.from_dict(snapshot_raw)
        invocation = G1InvocationEnvelope.from_dict(make_invocation(snapshot.to_dict()))

        class Completions:
            calls = 0

            @staticmethod
            def tool_call(name: str, arguments: dict[str, object], call_id: str) -> SimpleNamespace:
                return SimpleNamespace(
                    id=call_id,
                    function=SimpleNamespace(
                        name=name,
                        arguments=json.dumps(arguments),
                    ),
                    extra_content=None,
                )

            def create(self, **_kwargs: object):
                self.calls += 1
                if self.calls != 1:
                    raise AssertionError("terminal publication must stop before provider call N+1")
                tool_calls = [
                    self.tool_call(
                        "tool_call",
                        {
                            "name": "plane_operation",
                            "arguments": {
                                "action": "mutate",
                                "operationRef": "operation:work-item-update",
                                "input": {
                                    "workItemRef": "work-item:test",
                                    "title": "submitted",
                                },
                            },
                        },
                        "call-submit",
                    ),
                    self.tool_call(
                        "tool_call",
                        {
                            "name": "plane_publish",
                            "arguments": {
                                "kind": "outcome",
                                "operationRef": PLANE_OUTCOME_PUBLISH_OPERATION,
                                "resourceRef": "outcome-submission:test",
                                "content": "bounded outcome",
                            },
                        },
                        "call-outcome",
                    ),
                    self.tool_call(
                        "tool_call",
                        {
                            "name": "plane_operation",
                            "arguments": {
                                "action": "mutate",
                                "operationRef": "operation:work-item-update",
                                "input": {
                                    "workItemRef": "work-item:test",
                                    "title": "must-not-run",
                                },
                            },
                        },
                        "call-late-mutation",
                    ),
                ]
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            finish_reason="tool_calls",
                            message=SimpleNamespace(
                                content=None,
                                tool_calls=tool_calls,
                                reasoning=None,
                                reasoning_content=None,
                                refusal=None,
                            ),
                        )
                    ],
                    usage=SimpleNamespace(
                        prompt_tokens=1,
                        completion_tokens=1,
                        total_tokens=2,
                    ),
                )

        class Client:
            def __init__(self) -> None:
                self.completions = Completions()
                self.chat = SimpleNamespace(completions=self.completions)

        class Credentials:
            def resolve(self, provider: str) -> dict[str, str]:
                del provider
                return {
                    "api_key": "model-only-secret",
                    "base_url": "http://127.0.0.1",
                    "api_mode": "chat_completions",
                }

        client = Client()

        def agent_factory(**kwargs: object) -> AIAgent:
            agent = AIAgent(**kwargs)
            agent._create_request_openai_client = lambda **_: client  # type: ignore[method-assign]
            return agent

        requests: list[dict] = []

        def rpc(request: dict) -> dict:
            requests.append(request)
            if request["action"] == "publish":
                return _result(
                    request,
                    output={"published": True},
                    publication={
                        "action": "applied",
                        "productKind": "outcome_submission",
                        "productRef": "outcome-submission:test",
                        "operationAttemptRef": "operation-attempt:attempt-1",
                        "operationRef": PLANE_OUTCOME_PUBLISH_OPERATION,
                        "applicationServiceRef": "application-service:conversation",
                        "gatewayReceiptRef": "gateway-receipt:receipt-1",
                        "receiptRef": "receipt:receipt-1",
                        "auditReceiptRef": "audit-receipt:audit-1",
                        "productEventRef": "product-event:event-1",
                    },
                )
            return _result(request, output={"accepted": True})

        bodies: list[dict] = []
        with mock.patch.dict(os.environ, {"HERMES_HOME": _TEST_HERMES_HOME.name}):
            import run_agent

            run_agent._hermes_home = Path(_TEST_HERMES_HOME.name)
            result = HermesKernelAdapter(
                agent_factory=agent_factory,
                credential_source=Credentials(),
                host_port=CallablePlaneHostPort(rpc),
            ).dispatch(
                snapshot,
                invocation,
                lambda: False,
                bodies.append,
                model_call_allowance=2,
            )

        self.assertEqual(result.kind, "completed")
        self.assertIsNone(result.failure_code)
        self.assertEqual(result.output_text, "")
        self.assertEqual(result.model_calls, 1)
        self.assertEqual(client.completions.calls, 1)
        self.assertEqual([request["action"] for request in requests], ["mutate", "publish"])
        self.assertEqual(
            [request["operationRef"] for request in requests],
            ["operation:work-item-update", PLANE_OUTCOME_PUBLISH_OPERATION],
        )
        self.assertEqual(
            [
                body["payload"]["text"]
                for body in bodies
                if body["kind"] == "progress_observed"
                and body["payload"]["text"].startswith("Plane host ")
            ],
            [
                "Plane host model mutate operation:work-item-update -> ok",
                "Plane host model publish operation:agent.outcome.publish -> ok",
            ],
        )
        self.assertFalse(any(body["kind"] == "transcript_evidence_observed" for body in bodies))
        self.assertNotIn("Hermes invocation completed.", json.dumps(bodies))

    def test_real_hermes_adapter_recovers_from_validation_result(self) -> None:
        from tests.plane_runtime.test_g1_runtime_process import (
            G1InvocationEnvelope,
            G1RunSnapshot,
            _digest,
            make_invocation,
            make_snapshot,
        )
        from plane_runtime.hermes_adapter import HermesKernelAdapter
        from run_agent import AIAgent

        snapshot_raw = make_snapshot()
        snapshot_raw["runtimePolicy"] = dict(snapshot_raw["runtimePolicy"])  # type: ignore[index]
        snapshot_raw["runtimePolicy"]["adapter"] = "hermes"  # type: ignore[index]
        snapshot_raw["runtimePolicy"]["model"] = {  # type: ignore[index]
            "provider": "openai",
            "model": "deterministic-local",
        }
        snapshot_raw["contentDigest"] = _digest(  # type: ignore[assignment]
            "snapshot",
            {key: value for key, value in snapshot_raw.items() if key != "contentDigest"},
        )
        snapshot = G1RunSnapshot.from_dict(snapshot_raw)
        invocation = G1InvocationEnvelope.from_dict(make_invocation(snapshot.to_dict()))

        class Completions:
            def __init__(self) -> None:
                self.calls = 0

            @staticmethod
            def tool_call(arguments: dict[str, object], call_id: str):
                return SimpleNamespace(
                    id=call_id,
                    function=SimpleNamespace(
                        name="tool_call",
                        arguments=json.dumps(
                            {"name": "plane_operation", "arguments": arguments}
                        ),
                    ),
                    extra_content=None,
                )

            def create(self, **_kwargs: object):
                self.calls += 1
                if self.calls == 1:
                    tool_calls = [
                        self.tool_call(
                            {
                                "action": "read",
                                "operationRef": "operation:work_item.read",
                                "input": {"issue_ref": "issue:test"},
                            },
                            "call-invalid-read",
                        )
                    ]
                elif self.calls == 2:
                    tool_calls = [
                        self.tool_call(
                            {
                                "action": "read",
                                "operationRef": "operation:work_item.read",
                                "input": {
                                    "project_id": "00000000-0000-0000-0000-000000000001",
                                    "issue_id": "00000000-0000-0000-0000-000000000002",
                                },
                            },
                            "call-valid-read",
                        )
                    ]
                else:
                    tool_calls = []
                if tool_calls:
                    message = SimpleNamespace(
                        content=None,
                        tool_calls=tool_calls,
                        reasoning=None,
                        reasoning_content=None,
                        refusal=None,
                    )
                    finish_reason = "tool_calls"
                else:
                    message = SimpleNamespace(
                        content="ordinary final evidence",
                        tool_calls=None,
                        reasoning=None,
                        reasoning_content=None,
                        refusal=None,
                    )
                    finish_reason = "stop"
                return SimpleNamespace(
                    choices=[SimpleNamespace(finish_reason=finish_reason, message=message)],
                    usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
                )

        completions = Completions()

        class Client:
            chat = SimpleNamespace(completions=completions)

        class Credentials:
            def resolve(self, provider: str) -> dict[str, str]:
                del provider
                return {"api_key": "model-only-secret", "base_url": "http://127.0.0.1"}

        def agent_factory(**kwargs: object) -> AIAgent:
            agent = AIAgent(**kwargs)
            agent._create_request_openai_client = lambda **_: Client()  # type: ignore[method-assign]
            return agent

        requests: list[dict] = []

        def rpc(request: dict) -> dict:
            requests.append(request)
            if len(requests) == 1:
                return _result(
                    request,
                    status="invalid",
                    output={"error": "bounded validation result"},
                    errorCode="VALIDATION_ERROR",
                    errorMessage="project_id and issue_id are required",
                )
            return _result(request, output={"work_item": {"id": request["input"]["issue_id"]}})

        bodies: list[dict] = []
        with mock.patch.dict(os.environ, {"HERMES_HOME": _TEST_HERMES_HOME.name}):
            import run_agent

            run_agent._hermes_home = Path(_TEST_HERMES_HOME.name)
            result = HermesKernelAdapter(
                agent_factory=agent_factory,
                credential_source=Credentials(),
                host_port=CallablePlaneHostPort(rpc),
            ).dispatch(
                snapshot,
                invocation,
                lambda: False,
                bodies.append,
                model_call_allowance=3,
            )

        self.assertEqual(result.kind, "completed")
        self.assertEqual(completions.calls, 3)
        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[0]["input"], {"issue_ref": "issue:test"})
        self.assertEqual(
            requests[1]["input"],
            {
                "project_id": "00000000-0000-0000-0000-000000000001",
                "issue_id": "00000000-0000-0000-0000-000000000002",
            },
        )
        self.assertEqual(
            [body["payload"]["text"] for body in bodies if body["kind"] == "transcript_evidence_observed"],
            ["ordinary final evidence"],
        )

    def test_real_hermes_adapter_recovers_from_early_publish_before_code_mode(self) -> None:
        """A validation result must let the model correct its publish ordering."""

        from tests.plane_runtime.test_g1_runtime_process import (
            G1InvocationEnvelope,
            G1RunSnapshot,
            _digest,
            make_invocation,
            make_snapshot,
        )
        from plane_runtime.hermes_adapter import HermesKernelAdapter
        from run_agent import AIAgent

        snapshot_raw = make_snapshot()
        snapshot_raw["runtimePolicy"] = dict(snapshot_raw["runtimePolicy"])  # type: ignore[index]
        snapshot_raw["runtimePolicy"]["adapter"] = "hermes"  # type: ignore[index]
        snapshot_raw["runtimePolicy"]["model"] = {  # type: ignore[index]
            "provider": "openai",
            "model": "deterministic-local",
        }
        snapshot_raw["runtimePolicy"].update(  # type: ignore[union-attr]
            {
                "maxCodeModeInputBytes": 65_536,
                "maxCodeModeOutputBytes": 65_536,
                "maxCodeModeCalls": 4,
            }
        )
        snapshot_raw["contentDigest"] = _digest(  # type: ignore[assignment]
            "snapshot",
            {key: value for key, value in snapshot_raw.items() if key != "contentDigest"},
        )
        snapshot = G1RunSnapshot.from_dict(snapshot_raw)
        invocation = G1InvocationEnvelope.from_dict(make_invocation(snapshot.to_dict()))

        class Completions:
            def __init__(self) -> None:
                self.calls = 0

            @staticmethod
            def tool_call(name: str, arguments: dict[str, object], call_id: str):
                return SimpleNamespace(
                    id=call_id,
                    function=SimpleNamespace(
                        name=name,
                        arguments=json.dumps(arguments),
                    ),
                    extra_content=None,
                )

            def create(self, **_kwargs: object):
                self.calls += 1
                if self.calls == 1:
                    tool_calls = [
                        self.tool_call(
                            "tool_call",
                            {
                                "name": "plane_publish",
                                "arguments": {
                                    "kind": "outcome",
                                    "operationRef": PLANE_OUTCOME_PUBLISH_OPERATION,
                                    "resourceRef": "outcome-submission:test",
                                    "content": "premature publication",
                                },
                            },
                            "call-early-publish",
                        )
                    ]
                elif self.calls == 2:
                    tool_calls = [
                        self.tool_call(
                            "tool_call",
                            {
                                "name": "plane_execute_typescript",
                                "arguments": {
                                    "typescript_source": (
                                        "export default async ({ host, input }) => ({"
                                        " accepted: true, input"
                                        "});"
                                    )
                                },
                            },
                            "call-code-mode",
                        )
                    ]
                elif self.calls == 3:
                    tool_calls = [
                        self.tool_call(
                            "tool_call",
                            {
                                "name": "plane_publish",
                                "arguments": {
                                    "kind": "outcome",
                                    "operationRef": PLANE_OUTCOME_PUBLISH_OPERATION,
                                    "resourceRef": "outcome-submission:test",
                                    "content": "corrected publication",
                                },
                            },
                            "call-corrected-publish",
                        )
                    ]
                else:
                    tool_calls = []
                if tool_calls:
                    message = SimpleNamespace(
                        content=None,
                        tool_calls=tool_calls,
                        reasoning=None,
                        reasoning_content=None,
                        refusal=None,
                    )
                    finish_reason = "tool_calls"
                else:
                    message = SimpleNamespace(
                        content="ordinary final evidence",
                        tool_calls=None,
                        reasoning=None,
                        reasoning_content=None,
                        refusal=None,
                    )
                    finish_reason = "stop"
                return SimpleNamespace(
                    choices=[SimpleNamespace(finish_reason=finish_reason, message=message)],
                    usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
                )

        completions = Completions()

        class Client:
            chat = SimpleNamespace(completions=completions)

        class Credentials:
            def resolve(self, provider: str) -> dict[str, str]:
                del provider
                return {"api_key": "model-only-secret", "base_url": "http://127.0.0.1"}

        def agent_factory(**kwargs: object) -> AIAgent:
            agent = AIAgent(**kwargs)
            agent._create_request_openai_client = lambda **_: Client()  # type: ignore[method-assign]
            return agent

        requests: list[dict] = []
        code_completed = False

        def rpc(request: dict) -> dict:
            nonlocal code_completed
            requests.append(request)
            if request["action"] == "publish" and not code_completed:
                return _result(
                    request,
                    status="invalid",
                    output={"error": "bounded validation result"},
                    errorCode="VALIDATION_ERROR",
                    errorMessage="outcome publication requires the code-mode step",
                )
            if request["action"] == "code":
                code_completed = True
                return _result(request, output={"accepted": True})
            if request["action"] == "publish":
                return _result(
                    request,
                    publication=_applied_outcome_publication(),
                    output={"published": True},
                )
            raise AssertionError(
                "event=plane_publish_order expected=publish_or_code "
                f"actual={request['action']}"
            )

        bodies: list[dict] = []
        with mock.patch.dict(os.environ, {"HERMES_HOME": _TEST_HERMES_HOME.name}):
            import run_agent

            run_agent._hermes_home = Path(_TEST_HERMES_HOME.name)
            result = HermesKernelAdapter(
                agent_factory=agent_factory,
                credential_source=Credentials(),
                host_port=CallablePlaneHostPort(rpc),
            ).dispatch(
                snapshot,
                invocation,
                lambda: False,
                bodies.append,
                model_call_allowance=4,
            )

        self.assertEqual(result.kind, "completed")
        self.assertIsNone(result.failure_code)
        self.assertEqual(completions.calls, 3)
        self.assertEqual(
            [(request["action"], request["source"]) for request in requests],
            [("publish", "model"), ("code", "code"), ("publish", "model")],
            json.dumps({"requests": requests, "bodies": bodies}),
        )
        self.assertTrue(code_completed)
        self.assertEqual(
            sum(body["kind"] == "outcome_submission_observed" for body in bodies),
            1,
        )
        self.assertFalse(any(body["kind"] == "transcript_evidence_observed" for body in bodies))

    def test_code_mode_requires_first_tool_after_final_text_and_releases_guard(self) -> None:
        """A final-text first response is recalled before Code Mode can publish."""

        from tests.plane_runtime.test_g1_runtime_process import (
            G1InvocationEnvelope,
            G1RunSnapshot,
            _digest,
            make_invocation,
            make_snapshot,
        )
        from plane_runtime.hermes_adapter import HermesKernelAdapter
        from run_agent import AIAgent

        snapshot_raw = make_snapshot()
        snapshot_raw["toolCatalog"] = dict(snapshot_raw["toolCatalog"])  # type: ignore[index]
        snapshot_raw["toolCatalog"]["modelToolset"] = "code_mode_only"  # type: ignore[index]
        snapshot_raw["runtimePolicy"] = dict(snapshot_raw["runtimePolicy"])  # type: ignore[index]
        snapshot_raw["runtimePolicy"].update(  # type: ignore[union-attr]
            {
                "adapter": "hermes",
                "model": {"provider": "openai", "model": "deterministic-local"},
                "maxCodeModeInputBytes": 65_536,
                "maxCodeModeOutputBytes": 65_536,
                "maxCodeModeCalls": 4,
            }
        )
        snapshot_raw["contentDigest"] = _digest(
            "snapshot",
            {key: value for key, value in snapshot_raw.items() if key != "contentDigest"},
        )
        snapshot = G1RunSnapshot.from_dict(snapshot_raw)
        invocation = G1InvocationEnvelope.from_dict(make_invocation(snapshot.to_dict()))

        class Completions:
            def __init__(self) -> None:
                self.calls = 0
                self.tool_choices: list[object] = []

            @staticmethod
            def tool_call(arguments: dict[str, object], call_id: str):
                return SimpleNamespace(
                    id=call_id,
                    function=SimpleNamespace(
                        name="tool_call", arguments=json.dumps(arguments)
                    ),
                    extra_content=None,
                )

            def create(self, **kwargs: object):
                self.calls += 1
                self.tool_choices.append(kwargs.get("tool_choice"))
                if self.calls == 1:
                    message = SimpleNamespace(
                        content="ordinary final text before the required action",
                        tool_calls=None,
                        reasoning=None,
                        reasoning_content=None,
                        refusal=None,
                    )
                    finish_reason = "stop"
                elif self.calls == 2:
                    tool_calls = [
                        self.tool_call(
                            {
                                "name": "plane_execute_typescript",
                                "arguments": {
                                    "typescript_source": "export default async ({ host }) => ({ accepted: true });"
                                },
                            },
                            "call-code-mode",
                        )
                    ]
                    message = SimpleNamespace(
                        content=None,
                        tool_calls=tool_calls,
                        reasoning=None,
                        reasoning_content=None,
                        refusal=None,
                    )
                    finish_reason = "tool_calls"
                elif self.calls == 3:
                    tool_calls = [
                        self.tool_call(
                            {
                                "name": "plane_publish",
                                "arguments": {
                                    "kind": "conversation",
                                    "operationRef": "operation:conversation-publish",
                                    "resourceRef": "conversation:test",
                                    "content": "explicit publication",
                                },
                            },
                            "call-publish",
                        )
                    ]
                    message = SimpleNamespace(
                        content=None,
                        tool_calls=tool_calls,
                        reasoning=None,
                        reasoning_content=None,
                        refusal=None,
                    )
                    finish_reason = "tool_calls"
                else:
                    message = SimpleNamespace(
                        content="ordinary final evidence",
                        tool_calls=None,
                        reasoning=None,
                        reasoning_content=None,
                        refusal=None,
                    )
                    finish_reason = "stop"
                return SimpleNamespace(
                    choices=[SimpleNamespace(finish_reason=finish_reason, message=message)],
                    usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
                )

        completions = Completions()

        class Client:
            chat = SimpleNamespace(completions=completions)

        class Credentials:
            def resolve(self, provider: str) -> dict[str, str]:
                del provider
                return {
                    "api_key": "model-only-secret",
                    "base_url": "http://127.0.0.1",
                    "api_mode": "chat_completions",
                }

        def agent_factory(**kwargs: object) -> AIAgent:
            agent = AIAgent(**kwargs)
            agent._create_request_openai_client = lambda **_: Client()  # type: ignore[method-assign]
            return agent

        requests: list[dict] = []

        def rpc(request: dict) -> dict:
            requests.append(request)
            if request["action"] == "publish":
                publication = _applied_outcome_publication(
                    operation_ref="operation:conversation-publish"
                )
                publication.update(
                    productKind="conversation", productRef="conversation:test"
                )
                return _result(request, output={"published": True}, publication=publication)
            return _result(request, output={"accepted": True})

        with mock.patch.dict(os.environ, {"HERMES_HOME": _TEST_HERMES_HOME.name}):
            import run_agent

            run_agent._hermes_home = Path(_TEST_HERMES_HOME.name)
            result = HermesKernelAdapter(
                agent_factory=agent_factory,
                credential_source=Credentials(),
                host_port=CallablePlaneHostPort(rpc),
            ).dispatch(
                snapshot,
                invocation,
                lambda: False,
                lambda body: None,
                model_call_allowance=4,
            )

        self.assertEqual(result.kind, "completed", result)
        self.assertEqual(
            [request["action"] for request in requests], ["code", "publish"]
        )
        self.assertEqual(completions.tool_choices, [None, None, None, None])

    def test_code_mode_fails_closed_when_first_tool_is_not_registered(self) -> None:
        from tests.plane_runtime.test_g1_runtime_process import (
            G1InvocationEnvelope,
            G1RunSnapshot,
            _digest,
            make_invocation,
            make_snapshot,
        )
        from plane_runtime.hermes_adapter import HermesKernelAdapter

        raw = make_snapshot()
        raw["toolCatalog"] = dict(raw["toolCatalog"])  # type: ignore[index]
        raw["toolCatalog"]["modelToolset"] = "code_mode_only"  # type: ignore[index]
        raw["contentDigest"] = _digest(
            "snapshot", {key: value for key, value in raw.items() if key != "contentDigest"}
        )
        snapshot = G1RunSnapshot.from_dict(raw)
        invocation = G1InvocationEnvelope.from_dict(make_invocation(snapshot.to_dict()))

        class Agent:
            valid_tool_names = {"plane_publish"}
            request_overrides: dict[str, object] = {}

            def run_conversation(self, *args: object, **kwargs: object) -> dict[str, object]:
                raise AssertionError("agent must not run without the required tool")

        class Credentials:
            def resolve(self, provider: str) -> dict[str, str]:
                del provider
                return {"api_key": "secret"}

        with mock.patch.object(registry, "get_entry", return_value=None):
            result = HermesKernelAdapter(
                agent_factory=lambda **kwargs: Agent(),
                credential_source=Credentials(),
                host_port=CallablePlaneHostPort(lambda request: _result(request)),
            ).dispatch(snapshot, invocation, lambda: False, lambda body: None, model_call_allowance=1)

        self.assertEqual(result.kind, "failed")
        self.assertEqual(result.failure_cause, "static_configuration_failure")

    def test_ungranted_snapshot_cannot_enable_plane_code_mode_from_adapter_override(self) -> None:
        from tests.plane_runtime.test_g1_runtime_process import (
            G1InvocationEnvelope,
            G1RunSnapshot,
            make_invocation,
            make_snapshot,
        )
        from plane_runtime.hermes_adapter import HermesKernelAdapter

        snapshot = G1RunSnapshot.from_dict(make_snapshot())
        invocation = G1InvocationEnvelope.from_dict(make_invocation(snapshot.to_dict()))
        captured: dict[str, object] = {}

        class Agent:
            session_input_tokens = 1
            session_output_tokens = 1
            session_api_calls = 1

            def run_conversation(self, *_args: object, **_kwargs: object) -> dict[str, str]:
                return {"final_response": "ordinary final evidence"}

            def interrupt(self, _reason: str) -> None:
                return

        def agent_factory(**kwargs: object) -> Agent:
            captured.update(kwargs)
            return Agent()

        result = HermesKernelAdapter(
            agent_factory=agent_factory,
            credential_source=type("Credentials", (), {"resolve": lambda _self, _provider: {"api_key": "secret"}})(),
            enabled_toolsets=("code_execution",),
        ).dispatch(
            snapshot,
            invocation,
            lambda: False,
            lambda _body: None,
            model_call_allowance=1,
        )

        self.assertEqual(result.kind, "completed")
        self.assertNotIn(
            "code_execution",
            captured["enabled_toolsets"],
            "event=plane.code_mode.prevention actor=ungranted-invocation operation=adapter_dispatch "
            "risk=caller-controlled_toolset_grants_code_mode expected=code_execution_filtered "
            "actual=code_execution_present suggestion=keep_code_mode_activation_snapshot_bound",
        )

    def test_ordinary_final_text_never_publishes_and_host_failure_fails_closed(self) -> None:
        from tests.plane_runtime.test_g1_runtime_process import (
            G1InvocationEnvelope,
            G1RunSnapshot,
            _digest,
            make_invocation,
            make_snapshot,
        )
        from plane_runtime.hermes_adapter import HermesKernelAdapter
        from run_agent import AIAgent

        snapshot_raw = make_snapshot()
        snapshot_raw["runtimePolicy"] = dict(snapshot_raw["runtimePolicy"])  # type: ignore[index]
        snapshot_raw["runtimePolicy"]["adapter"] = "hermes"  # type: ignore[index]
        snapshot_raw["runtimePolicy"]["model"] = {  # type: ignore[index]
            "provider": "openai",
            "model": "deterministic-local",
        }
        snapshot_raw["contentDigest"] = _digest(  # type: ignore[assignment]
            "snapshot",
            {key: value for key, value in snapshot_raw.items() if key != "contentDigest"},
        )
        snapshot = G1RunSnapshot.from_dict(snapshot_raw)
        invocation = G1InvocationEnvelope.from_dict(make_invocation(snapshot.to_dict()))

        class FinalClient:
            class Completions:
                def create(self, **_: object):
                    return SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                finish_reason="stop",
                                message=SimpleNamespace(
                                    content="ordinary final only",
                                    tool_calls=None,
                                    reasoning=None,
                                    reasoning_content=None,
                                    refusal=None,
                                ),
                            )
                        ],
                        usage=SimpleNamespace(
                            prompt_tokens=1,
                            completion_tokens=1,
                            total_tokens=2,
                        ),
                    )

            def __init__(self) -> None:
                self.chat = SimpleNamespace(completions=self.Completions())

        class Credentials:
            def resolve(self, provider: str) -> dict[str, str]:
                return {
                    "api_key": "model-only-secret",
                    "base_url": "http://127.0.0.1",
                    "api_mode": "chat_completions",
                }

        def factory(**kwargs: object) -> AIAgent:
            agent = AIAgent(**kwargs)
            client = FinalClient()
            agent._create_request_openai_client = lambda **_: client  # type: ignore[method-assign]
            return agent

        bodies: list[dict] = []
        with mock.patch.dict(os.environ, {"HERMES_HOME": _TEST_HERMES_HOME.name}):
            import run_agent

            run_agent._hermes_home = Path(_TEST_HERMES_HOME.name)
            result = HermesKernelAdapter(
                agent_factory=factory,
                credential_source=Credentials(),
                host_port=CallablePlaneHostPort(
                    lambda request: (_ for _ in ()).throw(RuntimeError("host unavailable"))
                ),
            ).dispatch(
                snapshot,
                invocation,
                lambda: False,
                bodies.append,
                model_call_allowance=2,
            )
        self.assertEqual(result.kind, "completed")
        self.assertFalse(any(body["kind"] == "conversation_publication_observed" for body in bodies))
        self.assertTrue(any(body["kind"] == "transcript_evidence_observed" for body in bodies))

        class FailingClient:
            class Completions:
                def __init__(self) -> None:
                    self.calls = 0

                def create(self, **_: object):
                    self.calls += 1
                    if self.calls == 1:
                        tool_call = SimpleNamespace(
                            id="call-failure",
                            function=SimpleNamespace(
                                name="tool_call",
                                arguments=json.dumps(
                                    {
                                        "name": "plane_operation",
                                        "arguments": {
                                            "action": "read",
                                            "operationRef": "operation:read",
                                            "input": {},
                                        },
                                    }
                                ),
                            ),
                            extra_content=None,
                        )
                        message = SimpleNamespace(
                            content=None,
                            tool_calls=[tool_call],
                            reasoning=None,
                            reasoning_content=None,
                            refusal=None,
                        )
                        finish_reason = "tool_calls"
                    else:
                        message = SimpleNamespace(
                            content="should not become a successful runtime result",
                            tool_calls=None,
                            reasoning=None,
                            reasoning_content=None,
                            refusal=None,
                        )
                        finish_reason = "stop"
                    return SimpleNamespace(
                        choices=[SimpleNamespace(finish_reason=finish_reason, message=message)],
                        usage=SimpleNamespace(
                            prompt_tokens=1,
                            completion_tokens=1,
                            total_tokens=2,
                        ),
                    )

            def __init__(self) -> None:
                self.chat = SimpleNamespace(completions=self.Completions())

        def failing_factory(**kwargs: object) -> AIAgent:
            agent = AIAgent(**kwargs)
            client = FailingClient()
            agent._create_request_openai_client = lambda **_: client  # type: ignore[method-assign]
            return agent

        failed_bodies: list[dict] = []
        with mock.patch.dict(os.environ, {"HERMES_HOME": _TEST_HERMES_HOME.name}):
            import run_agent

            run_agent._hermes_home = Path(_TEST_HERMES_HOME.name)
            failed = HermesKernelAdapter(
                agent_factory=failing_factory,
                credential_source=Credentials(),
                host_port=CallablePlaneHostPort(
                    lambda request: (_ for _ in ()).throw(RuntimeError("host unavailable"))
                ),
            ).dispatch(
                snapshot,
                invocation,
                lambda: False,
                failed_bodies.append,
                model_call_allowance=2,
            )
        self.assertEqual(failed.kind, "failed")
        self.assertEqual(failed.failure_code, "runtime_error")


if __name__ == "__main__":
    unittest.main()
