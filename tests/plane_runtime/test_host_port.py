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
    PLANE_CATALOG_DESCRIBE_OPERATION,
    PLANE_CATALOG_SEARCH_OPERATION,
    PLANE_OUTCOME_SUBMIT_OPERATION,
    PLANE_OUTCOME_PUBLISH_OPERATION,
    UnixSocketPlaneHostPort,
    bind_plane_host,
    install_plane_tools,
    plane_code_mode,
    _normalize_prepared_read_input,
    _prepared_read_refs_from_search_result,
    _prepared_read_refs_from_code_mode_result,
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
        "operation:agent.outcome.submit",
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


def _submitted_result(
    request: dict,
    *,
    outcome_ref: str = "outcome-submission:test",
    status: str = "ok",
) -> dict:
    return _result(
        request,
        status=status,
        output={"result": {"outcome": {"outcomeRef": outcome_ref}}},
    )


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
    def test_submit_arms_explicit_publish_and_rejected_publish_stays_recoverable(self) -> None:
        def submitted_port(request: dict) -> dict:
            if request["action"] == "code":
                return _result(
                    request,
                    output={
                        "result": {
                            "ok": True,
                            "replayed": False,
                            "result": {
                                "outcome": {"outcomeRef": "outcome-submission:test"}
                            },
                        },
                        "observations": [
                            {
                                "source": "code",
                                "action": "code",
                                "operationRef": "operation:agent.outcome.submit",
                                "status": "ok",
                            }
                        ]
                    },
                )
            return _result(request, publication=_applied_outcome_publication())

        binding = PlaneHostBinding(
            port=CallablePlaneHostPort(submitted_port),
            run_id="run:publication-continuation",
            invocation_id="invocation:publication-continuation",
            correlation_id="correlation:publication-continuation",
            cancellation=lambda: False,
        )
        result = binding.call(
            action="code",
            operation_ref="plane.code-mode.execute@1",
            input={"source": "export default async () => ({})"},
            source="code",
        )
        self.assertTrue(binding.outcome_submission_pending())

        binding.publish(
            kind="outcome",
            operation_ref=PLANE_OUTCOME_PUBLISH_OPERATION,
            resource_ref="outcome-submission:test",
            content="explicit publication",
        )
        self.assertFalse(binding.outcome_submission_pending())

        rejected = PlaneHostBinding(
            port=CallablePlaneHostPort(
                lambda request: _result(
                    request,
                    status="invalid",
                    errorCode="OPERATION_REJECTED",
                    errorMessage="submit the outcome first",
                )
            ),
            run_id="run:publication-recoverable",
            invocation_id="invocation:publication-recoverable",
            correlation_id="correlation:publication-recoverable",
            cancellation=lambda: False,
        )
        with self.assertRaises(PlaneHostError):
            rejected.publish(
                kind="outcome",
                operation_ref=PLANE_OUTCOME_PUBLISH_OPERATION,
                resource_ref="outcome-submission:missing",
                content="too early",
            )
        self.assertEqual(rejected.records, [])

        native = PlaneHostBinding(
            port=CallablePlaneHostPort(
                lambda request: _result(
                    request,
                    output={
                        "result": {
                            "outcome": {"outcomeRef": "outcome-submission:test"}
                        }
                    },
                )
            ),
            run_id="run:native-submit",
            invocation_id="invocation:native-submit",
            correlation_id="correlation:native-submit",
            cancellation=lambda: False,
        )
        native.call(
            action="mutate",
            operation_ref=PLANE_OUTCOME_SUBMIT_OPERATION,
            input={"summary": "bounded"},
            source="model",
        )
        self.assertTrue(native.outcome_submission_pending())
        self.assertEqual(native.outcome_submission_ref(), "outcome-submission:test")

        unrelated = PlaneHostBinding(
            port=CallablePlaneHostPort(
                lambda request: _result(
                    request,
                    status="invalid",
                    errorCode="OPERATION_REJECTED",
                    errorMessage="unrelated mutation rejected",
                )
            ),
            run_id="run:unrelated-rejection",
            invocation_id="invocation:unrelated-rejection",
            correlation_id="correlation:unrelated-rejection",
            cancellation=lambda: False,
        )
        unrelated.call(
            action="mutate",
            operation_ref="operation:mutate@1",
            input={"name": "not applied"},
            source="model",
        )
        self.assertEqual(unrelated.fatal_error, "unrelated mutation rejected")

    def test_outcome_publish_tool_binds_submit_ref_and_hides_legacy_fields(self) -> None:
        calls: list[dict] = []

        def rpc(request: dict) -> dict:
            calls.append(request)
            if request["operationRef"] == PLANE_OUTCOME_SUBMIT_OPERATION:
                return _result(
                    request,
                    output={
                        "result": {
                            "outcome": {"outcomeRef": "outcome-submission:trusted"}
                        }
                    },
                )
            return _result(
                request,
                publication={
                    **_applied_outcome_publication(),
                    "productRef": request["input"]["resourceRef"],
                },
            )

        binding = PlaneHostBinding(
            port=CallablePlaneHostPort(rpc),
            run_id="run:publish-tool",
            invocation_id="invocation:publish-tool",
            correlation_id="correlation:publish-tool",
            cancellation=lambda: False,
        )
        install_plane_tools()
        with bind_plane_host(binding):
            registry.dispatch(
                "plane_operation",
                {
                    "action": "mutate",
                    "operationRef": PLANE_OUTCOME_SUBMIT_OPERATION,
                    "input": {"summary": "submitted"},
                },
            )
            published = registry.dispatch(
                "plane_publish",
                {"kind": "outcome", "content": "explicit publication"},
            )

        self.assertEqual(json.loads(published)["status"], "ok")
        self.assertEqual(
            [(call["action"], call["operationRef"], call["input"].get("resourceRef")) for call in calls],
            [
                ("mutate", PLANE_OUTCOME_SUBMIT_OPERATION, None),
                ("publish", PLANE_OUTCOME_PUBLISH_OPERATION, "outcome-submission:trusted"),
            ],
        )
        publish_definition = next(
            definition
            for definition in registry.get_definitions({"plane_publish"}, quiet=True)
            if definition["function"]["name"] == "plane_publish"
        )
        outcome_schema = publish_definition["function"]["parameters"]["oneOf"][0]
        self.assertEqual(outcome_schema["required"], ["kind", "content"])
        self.assertNotIn("operationRef", outcome_schema["properties"])
        self.assertNotIn("resourceRef", outcome_schema["properties"])

    def test_outcome_publish_tool_rejects_early_and_tampered_refs_without_host_call(self) -> None:
        calls: list[dict] = []

        def rpc(request: dict) -> dict:
            calls.append(request)
            if request["operationRef"] == PLANE_OUTCOME_SUBMIT_OPERATION:
                return _result(
                    request,
                    output={
                        "result": {
                            "outcome": {"outcomeRef": "outcome-submission:trusted"}
                        }
                    },
                )
            return _result(request, publication=_applied_outcome_publication())

        install_plane_tools()
        early = PlaneHostBinding(
            port=CallablePlaneHostPort(rpc),
            run_id="run:early-publish",
            invocation_id="invocation:early-publish",
            correlation_id="correlation:early-publish",
            cancellation=lambda: False,
        )
        with bind_plane_host(early):
            early_payload = registry.dispatch(
                "plane_publish",
                {"kind": "outcome", "content": "too early"},
            )
        self.assertEqual(json.loads(early_payload)["status"], "error")
        self.assertEqual(calls, [])

        trusted = PlaneHostBinding(
            port=CallablePlaneHostPort(rpc),
            run_id="run:tampered-publish",
            invocation_id="invocation:tampered-publish",
            correlation_id="correlation:tampered-publish",
            cancellation=lambda: False,
        )
        with bind_plane_host(trusted):
            registry.dispatch(
                "plane_operation",
                {
                    "action": "mutate",
                    "operationRef": PLANE_OUTCOME_SUBMIT_OPERATION,
                    "input": {"summary": "submitted"},
                },
            )
            tampered_payload = registry.dispatch(
                "plane_publish",
                {
                    "kind": "outcome",
                    "operationRef": PLANE_OUTCOME_PUBLISH_OPERATION,
                    "resourceRef": "outcome-submission:other-run",
                    "content": "tampered",
                },
            )
        self.assertEqual(json.loads(tampered_payload)["status"], "error")
        self.assertEqual(
            [call["action"] for call in calls],
            ["mutate"],
        )

    def test_replayed_submit_binds_ref_and_replayed_publish_does_not_terminalize(self) -> None:
        calls: list[dict] = []

        def rpc(request: dict) -> dict:
            calls.append(request)
            if request["operationRef"] == PLANE_OUTCOME_SUBMIT_OPERATION:
                return _result(
                    request,
                    status="replayed",
                    output={
                        "result": {
                            "outcome": {"outcomeRef": "outcome-submission:replayed"}
                        }
                    },
                )
            return _result(
                request,
                status="replayed",
                publication={
                    **_applied_outcome_publication(),
                    "productRef": "outcome-submission:replayed",
                },
            )

        binding = PlaneHostBinding(
            port=CallablePlaneHostPort(rpc),
            run_id="run:replayed-submit",
            invocation_id="invocation:replayed-submit",
            correlation_id="correlation:replayed-submit",
            cancellation=lambda: False,
        )
        binding.call(
            action="mutate",
            operation_ref=PLANE_OUTCOME_SUBMIT_OPERATION,
            input={"summary": "replayed"},
            source="model",
        )
        result = binding.publish_outcome(content="replayed publication")

        self.assertEqual(result.status, "replayed")
        self.assertEqual(binding.outcome_submission_ref(), "outcome-submission:replayed")
        self.assertIsNone(binding.terminal_action_reason())
        self.assertEqual([call["action"] for call in calls], ["mutate", "publish"])

    def test_untrusted_submit_observation_does_not_arm_publication(self) -> None:
        binding = PlaneHostBinding(
            port=CallablePlaneHostPort(
                lambda request: _result(
                    request,
                    output={
                        "observations": [
                            {
                                "source": "model",
                                "action": "mutate",
                                "operationRef": "operation:agent.outcome.submit",
                                "status": "ok",
                            }
                        ]
                    },
                )
            ),
            run_id="run:untrusted-submit",
            invocation_id="invocation:untrusted-submit",
            correlation_id="correlation:untrusted-submit",
            cancellation=lambda: False,
        )
        binding.call(
            action="code",
            operation_ref="plane.code-mode.execute@1",
            input={"source": "export default async () => ({})"},
            source="code",
        )
        self.assertFalse(binding.outcome_submission_pending())

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
                        "results": [],
                        "assignmentWorkItemReadDecision": {
                            "schemaVersion": "plane.assignment-read-handoff/v1",
                            "recognizedCount": 2,
                            "acceptedForm": "unrecognized",
                            "failureClass": "multiple",
                            "shape": {"nestingDepth": 0, "sizeClass": "large"},
                        },
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

    def test_post_consume_stray_prepared_read_is_bounded_local_replay(self) -> None:
        requests: list[dict] = []

        def respond(request: dict) -> dict:
            requests.append(request)
            if len(requests) == 1:
                return _result(
                    request,
                    output={"ok": True, "result": {"work_item": {"title": "assigned"}}},
                )
            return _result(
                request,
                status="invalid",
                output={"shapeDiagnostic": {"acceptedForm": "unrecognized"}},
                errorCode="PREPARED_CALL_INVALID",
                errorMessage="prepared work-item read reference is invalid",
            )

        binding = PlaneHostBinding(
            port=CallablePlaneHostPort(respond),
            run_id="run:duplicate-read",
            invocation_id="invocation:duplicate-read",
            correlation_id="correlation:duplicate-read",
            cancellation=lambda: False,
        )
        first = binding.call(
            action="read",
            operation_ref="operation:work_item.read",
            input={"preparedCallRef": "prepared-call:opaque"},
            source="model",
        )
        duplicate = binding.call(
            action="read",
            operation_ref="operation:work_item.read",
            input={
                "preparedCallRef": {
                    "acceptedForm": "unrecognized",
                    "keyNames": ["preparedCallRef"],
                    "valueTypes": ["object", "string"],
                    "nestingDepth": 1,
                    "sizeClass": "small",
                }
            },
            source="model",
        )

        self.assertEqual(first.status, "ok")
        self.assertEqual(duplicate.status, "invalid")
        self.assertFalse(duplicate.replayed)
        self.assertEqual(duplicate.error_code, "READ_ALREADY_CONSUMED")
        self.assertEqual(
            duplicate.error_message,
            "the invocation already consumed its prepared work-item read",
        )
        self.assertNotIn("prepared-call:opaque", json.dumps(duplicate.to_dict()))
        self.assertEqual(len(requests), 1)
        self.assertIsNone(binding.fatal_error)

        tampered = binding.call(
            action="read",
            operation_ref="operation:work_item.read",
            input={"preparedCallRef": "prepared-call:opaque-tampered"},
            source="model",
        )
        self.assertEqual(tampered.status, "invalid")
        self.assertEqual(tampered.error_code, "PREPARED_CALL_INVALID")
        self.assertEqual(len(requests), 2)

        fresh_requests: list[dict] = []

        def fresh_respond(request: dict) -> dict:
            fresh_requests.append(request)
            return _result(
                request,
                status="invalid",
                errorCode="PREPARED_CALL_INVALID",
                errorMessage="prepared work-item read reference is invalid",
            )

        fresh = PlaneHostBinding(
            port=CallablePlaneHostPort(fresh_respond),
            run_id="run:first-use-malformed",
            invocation_id="invocation:first-use-malformed",
            correlation_id="correlation:first-use-malformed",
            cancellation=lambda: False,
        )
        first_use = fresh.call(
            action="read",
            operation_ref="operation:work_item.read",
            input={"preparedCallRef": {"unexpected": True}},
            source="model",
        )
        self.assertEqual(first_use.status, "invalid")
        self.assertEqual(len(fresh_requests), 1)

    def test_catalog_search_replays_locally_after_successful_discovery(self) -> None:
        requests: list[dict] = []

        def respond(request: dict) -> dict:
            requests.append(request)
            if request["operationRef"] == PLANE_CATALOG_SEARCH_OPERATION:
                return _result(
                    request,
                    output={"operations": [{"operationId": "work_item.read"}]},
                )
            if request["operationRef"] == PLANE_CATALOG_DESCRIBE_OPERATION:
                return _result(
                    request,
                    output={
                        "operation": {
                            "operationId": request["input"]["operation_id"],
                            "operationRef": "operation:" + request["input"]["operation_id"],
                            "inputSchema": {"type": "object"},
                        }
                    },
                )
            self.assertEqual(request["operationRef"], "operation:mutate@1")
            return _result(
                request,
                status="denied",
                errorCode="NOT_AUTHORIZED",
                errorMessage="operation is not authorized",
            )

        binding = PlaneHostBinding(
            port=CallablePlaneHostPort(respond),
            run_id="run:catalog-replay",
            invocation_id="invocation:catalog-replay",
            correlation_id="correlation:catalog-replay",
            cancellation=lambda: False,
        )
        first_search = binding.call(
            action="read",
            operation_ref=PLANE_CATALOG_SEARCH_OPERATION,
            input={"query": "assigned", "limit": 8},
            source="model",
        )
        description = binding.call(
            action="read",
            operation_ref=PLANE_CATALOG_DESCRIBE_OPERATION,
            input={"operation_id": "work_item.read"},
            source="model",
        )
        different_description = binding.call(
            action="read",
            operation_ref=PLANE_CATALOG_DESCRIBE_OPERATION,
            input={"operation_id": "work_item.rename"},
            source="model",
        )
        repeated_search = binding.call(
            action="read",
            operation_ref=PLANE_CATALOG_SEARCH_OPERATION,
            input={"query": "assigned", "limit": 8},
            source="model",
        )
        denied_mutation = binding.call(
            action="mutate",
            operation_ref="operation:mutate@1",
            input={"name": "assigned"},
            source="model",
        )

        self.assertEqual(first_search.status, "ok")
        self.assertEqual(description.status, "ok")
        self.assertEqual(different_description.status, "ok")
        self.assertEqual(repeated_search.status, "replayed")
        self.assertTrue(repeated_search.replayed)
        self.assertEqual(
            repeated_search.output,
            {
                "alreadyDiscovered": True,
                "operationRef": PLANE_CATALOG_SEARCH_OPERATION,
            },
        )
        self.assertEqual(denied_mutation.status, "denied")
        self.assertIsNone(binding.fatal_error)
        self.assertEqual(
            [request["operationRef"] for request in requests],
            [
                PLANE_CATALOG_SEARCH_OPERATION,
                PLANE_CATALOG_DESCRIBE_OPERATION,
                PLANE_CATALOG_DESCRIBE_OPERATION,
                "operation:mutate@1",
            ],
        )
        self.assertEqual(len(binding.records), 4)

    def test_catalog_search_guard_does_not_arm_on_incomplete_discovery(self) -> None:
        requests: list[dict] = []

        def respond(request: dict) -> dict:
            requests.append(request)
            if request["operationRef"] == PLANE_CATALOG_DESCRIBE_OPERATION:
                return _result(
                    request,
                    status="invalid",
                    errorCode="VALIDATION_ERROR",
                    errorMessage="operation_id is unknown",
                )
            return _result(request, output={"operations": []})

        binding = PlaneHostBinding(
            port=CallablePlaneHostPort(respond),
            run_id="run:catalog-incomplete",
            invocation_id="invocation:catalog-incomplete",
            correlation_id="correlation:catalog-incomplete",
            cancellation=lambda: False,
        )
        binding.call(
            action="read",
            operation_ref=PLANE_CATALOG_SEARCH_OPERATION,
            input={"query": "assigned", "limit": 8},
            source="model",
        )
        binding.call(
            action="read",
            operation_ref=PLANE_CATALOG_DESCRIBE_OPERATION,
            input={"operation_id": "missing"},
            source="model",
        )
        second_search = binding.call(
            action="read",
            operation_ref=PLANE_CATALOG_SEARCH_OPERATION,
            input={"query": "assigned", "limit": 8},
            source="model",
        )

        self.assertEqual(second_search.status, "ok")
        self.assertFalse(second_search.replayed)
        self.assertEqual(
            [request["operationRef"] for request in requests],
            [
                PLANE_CATALOG_SEARCH_OPERATION,
                PLANE_CATALOG_DESCRIBE_OPERATION,
                PLANE_CATALOG_SEARCH_OPERATION,
            ],
        )

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

    def test_direct_search_auto_reads_opaque_ref_without_schema_redisclosure(self) -> None:
        """The trusted prepared continuation reaches the gateway before text exit."""

        requests: list[dict] = []

        def respond(request: dict) -> dict:
            requests.append(request)
            if request["operationRef"] == "operation:search_workspace":
                output = {
                    "ok": True,
                    "result": {
                        "results": [
                            {
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
                self.assertEqual(
                    request["input"], {"preparedCallRef": "prepared-call:opaque"}
                )
                output = {"ok": True, "result": {"work_item": {"title": "assigned"}}}
            return _result(request, output=output)

        binding = _RuntimePlaneHostBinding(
            port=CallablePlaneHostPort(respond),
            run_id="run:direct-search",
            invocation_id="invocation:direct-search",
            correlation_id="correlation:direct-search",
            cancellation=lambda: False,
            eager_operation_refs=frozenset({"operation:search_workspace"}),
        )

        search = binding.call(
            action="read",
            operation_ref="operation:search_workspace",
            input={"query": "assigned", "limit": 1},
            source="model",
        )

        self.assertEqual(search.status, "ok")
        self.assertEqual(
            [request["operationRef"] for request in requests],
            ["operation:search_workspace", "operation:work_item.read"],
        )
        self.assertFalse(binding.prepared_read_handoff_pending())
        self.assertEqual(
            search.output["preparedReadResult"]["output"]["result"]["work_item"]["title"],
            "assigned",
        )

    def test_direct_search_auto_reads_bare_opaque_ref_before_text_exit(self) -> None:
        requests: list[dict] = []

        def respond(request: dict) -> dict:
            requests.append(request)
            if request["operationRef"] == "operation:search_workspace":
                output = {
                    "ok": True,
                    "result": {
                        "results": [
                            {
                                "objectType": "work_item",
                                "workItemReadCall": "prepared-call:bare",
                            }
                        ]
                    },
                }
            else:
                self.assertEqual(request["operationRef"], "operation:work_item.read")
                self.assertEqual(
                    request["input"], {"preparedCallRef": "prepared-call:bare"}
                )
                output = {"ok": True, "result": {"work_item": {"title": "assigned"}}}
            return _result(request, output=output)

        binding = _RuntimePlaneHostBinding(
            port=CallablePlaneHostPort(respond),
            run_id="run:bare-search",
            invocation_id="invocation:bare-search",
            correlation_id="correlation:bare-search",
            cancellation=lambda: False,
            eager_operation_refs=frozenset({"operation:search_workspace"}),
        )

        search = binding.call(
            action="read",
            operation_ref="operation:search_workspace",
            input={"query": "assigned", "limit": 1},
            source="model",
        )

        self.assertEqual(
            [request["operationRef"] for request in requests],
            ["operation:search_workspace", "operation:work_item.read"],
        )
        self.assertFalse(binding.prepared_read_handoff_pending())
        self.assertEqual(
            search.output["preparedReadResult"]["output"]["result"]["work_item"]["title"],
            "assigned",
        )

    def test_prepared_search_handoff_is_fail_closed_for_no_multiple_or_tampered_refs(self) -> None:
        def search_output(items: list[dict]) -> dict:
            return {
                "ok": True,
                "result": {"results": items},
            }

        self.assertEqual(
            _prepared_read_refs_from_code_mode_result(
                {
                    "result": search_output([]),
                    "observations": [
                        {
                            "source": "code",
                            "action": "code",
                            "operationRef": "operation:search_workspace",
                            "status": "ok",
                        }
                    ],
                }
            ),
            (),
        )

        self.assertEqual(
            _prepared_read_refs_from_code_mode_result(
                {
                    "result": search_output(
                        [
                            {
                                "objectType": "work_item",
                                "workItemReadCall": "prepared-call:first",
                            },
                            {
                                "objectType": "work_item",
                                "workItemReadCall": "prepared-call:second",
                            },
                        ]
                    ),
                    "observations": [
                        {
                            "source": "code",
                            "action": "code",
                            "operationRef": "operation:search_workspace",
                            "status": "ok",
                        }
                    ],
                }
            ),
            ("prepared-call:first", "prepared-call:second"),
        )
        self.assertEqual(
            _prepared_read_refs_from_code_mode_result(
                {
                    "result": search_output(
                        [
                            {
                                "objectType": "work_item",
                                "workItemReadCall": {
                                    "preparedCallRef": {
                                        "preparedCallRef": "prepared-call:tampered"
                                    }
                                },
                            }
                        ]
                    ),
                    "observations": [
                        {
                            "source": "code",
                            "action": "code",
                            "operationRef": "operation:search_workspace",
                            "status": "ok",
                        }
                    ],
                }
            ),
            (),
        )

    def test_direct_search_handoff_rejects_extra_invalid_and_oversized_refs(self) -> None:
        for work_item_read_call in (
            {"preparedCallRef": "prepared-call:opaque", "extra": True},
            "not-a-prepared-call",
            "prepared-call:" + ("x" * 256),
        ):
            output = {
                "ok": True,
                "result": {
                    "results": [
                        {
                            "objectType": "work_item",
                            "workItemReadCall": work_item_read_call,
                        }
                    ]
                },
            }
            self.assertEqual(_prepared_read_refs_from_search_result(output), ())

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

    def test_post_search_code_mode_hint_arms_after_prepared_read_and_is_consumed(self) -> None:
        def respond(request: dict) -> dict:
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
                                    "input": {"preparedCallRef": "prepared-call:post-search"},
                                },
                            }
                        ]
                    },
                }
            elif request["operationRef"] == "operation:work_item.read":
                output = {"ok": True, "result": {"work_item": {"title": "assigned"}}}
            else:
                self.assertEqual(request["action"], "code")
                output = {"ok": True, "result": {"completed": True}}
            return _result(request, output=output)

        binding = PlaneHostBinding(
            port=CallablePlaneHostPort(respond),
            run_id="run:post-search",
            invocation_id="invocation:post-search",
            correlation_id="correlation:post-search",
            cancellation=lambda: False,
            code_mode_phase="post_search",
        )
        binding.call(
            action="read",
            operation_ref="operation:search_workspace",
            input={"query": "assigned"},
            source="model",
        )

        self.assertEqual(binding.code_mode_phase_hint(), "post_search")
        result = binding.call(
            action="code",
            operation_ref="plane.code-mode.execute@1",
            input={"source": "export default async () => ({})"},
            source="code",
        )
        self.assertIsNone(binding.code_mode_phase_hint())

    def test_code_mode_claim_is_atomic_and_duplicate_or_missing_tool_fails_closed(self) -> None:
        calls: list[dict] = []
        binding = PlaneHostBinding(
            port=CallablePlaneHostPort(
                lambda request: calls.append(request) or _result(request)
            ),
            run_id="run:claim",
            invocation_id="invocation:claim",
            correlation_id="correlation:claim",
            cancellation=lambda: False,
            code_mode_phase="post_search",
        )
        binding._code_mode_phase_hint = "post_search"

        self.assertEqual(binding.consume_code_mode_phase(tool_available=True), "post_search")
        with self.assertRaises(PlaneHostError):
            binding.consume_code_mode_phase(tool_available=True)
        self.assertEqual(calls, [])

        missing_tool = PlaneHostBinding(
            port=CallablePlaneHostPort(
                lambda request: calls.append(request) or _result(request)
            ),
            run_id="run:missing-tool",
            invocation_id="invocation:missing-tool",
            correlation_id="correlation:missing-tool",
            cancellation=lambda: False,
            code_mode_phase="post_search",
        )
        missing_tool._code_mode_phase_hint = "post_search"
        with self.assertRaises(PlaneHostError):
            missing_tool.consume_code_mode_phase(tool_available=False)
        self.assertEqual(calls, [])

    def test_code_mode_claim_clears_after_callback_success(self) -> None:
        binding = PlaneHostBinding(
            port=CallablePlaneHostPort(lambda request: _result(request)),
            run_id="run:claim-success",
            invocation_id="invocation:claim-success",
            correlation_id="correlation:claim-success",
            cancellation=lambda: False,
            code_mode_phase="post_search",
        )
        binding._code_mode_phase_hint = "post_search"
        self.assertEqual(binding.consume_code_mode_phase(tool_available=True), "post_search")
        binding.call(
            action="code",
            operation_ref="plane.code-mode.execute@1",
            input={"source": "return 1"},
            source="code",
        )
        self.assertIsNone(binding.code_mode_phase_hint())
        self.assertFalse(binding._code_mode_phase_claimed)

    def test_code_mode_claim_clears_after_callback_failure(self) -> None:
        def fail(_request: dict) -> dict:
            raise RuntimeError("callback failed")

        binding = PlaneHostBinding(
            port=CallablePlaneHostPort(fail),
            run_id="run:claim-failure",
            invocation_id="invocation:claim-failure",
            correlation_id="correlation:claim-failure",
            cancellation=lambda: False,
            code_mode_phase="post_search",
        )
        binding._code_mode_phase_hint = "post_search"
        self.assertEqual(binding.consume_code_mode_phase(tool_available=True), "post_search")
        with self.assertRaises(PlaneHostUnavailable):
            binding.call(
                action="code",
                operation_ref="plane.code-mode.execute@1",
                input={"source": "return 1"},
                source="code",
            )
        self.assertIsNone(binding.code_mode_phase_hint())
        self.assertFalse(binding._code_mode_phase_claimed)

    def test_code_mode_search_result_auto_consumes_one_prepared_read(self) -> None:
        def respond(request: dict) -> dict:
            if request["action"] == "code":
                output = {
                    "schemaVersion": "plane.code-mode/v1",
                    "result": {
                        "ok": True,
                        "result": {
                            "results": [
                                {
                                    "objectType": "work_item",
                                    "workItemReadCall": {
                                        "action": "read",
                                        "operationRef": "operation:work_item.read",
                                        "input": {"preparedCallRef": "prepared-call:code-search"},
                                    },
                                }
                            ]
                        },
                    },
                    "observations": [
                        {
                            "source": "code",
                            "action": "code",
                            "operationRef": "operation:search_workspace",
                            "status": "ok",
                        }
                    ],
                }
                return _result(request, output=output)
            self.assertEqual(request["operationRef"], "operation:work_item.read")
            self.assertEqual(
                request["input"], {"preparedCallRef": "prepared-call:code-search"}
            )
            return _result(
                request,
                output={"ok": True, "result": {"work_item": {"title": "assigned"}}},
            )

        binding = PlaneHostBinding(
            port=CallablePlaneHostPort(respond),
            run_id="run:code-search",
            invocation_id="invocation:code-search",
            correlation_id="correlation:code-search",
            cancellation=lambda: False,
            code_mode_phase="post_search",
        )
        result = binding.call(
            action="code",
            operation_ref="plane.code-mode.execute@1",
            input={"source": "export default async () => ({})"},
            source="code",
        )

        self.assertEqual(
            [record.request.operation_ref for record in binding.records],
            [
                "plane.code-mode.execute@1",
                "operation:work_item.read",
            ],
        )
        self.assertEqual(
            binding.records[1].request.input,
            {"preparedCallRef": "prepared-call:code-search"},
        )
        self.assertEqual(
            result.output["preparedReadResult"]["output"]["result"]["work_item"]["title"],
            "assigned",
        )
        self.assertEqual(binding.code_mode_phase_hint(), "post_search")
        self.assertEqual(binding.take_code_mode_phase_hint(), "post_search")
        self.assertIsNone(binding.take_code_mode_phase_hint())

    def test_code_mode_host_consumed_read_is_not_replayed_by_hermes(self) -> None:
        requests: list[dict] = []

        def respond(request: dict) -> dict:
            requests.append(request)
            return _result(
                request,
                output={
                    "schemaVersion": "plane.code-mode/v1",
                    "result": {
                        "ok": True,
                        "result": {
                            "results": [
                                {
                                    "objectType": "work_item",
                                    "workItemReadCall": "prepared-call:already-consumed",
                                }
                            ]
                        },
                    },
                    "observations": [
                        {
                            "source": "code",
                            "action": "code",
                            "operationRef": "operation:search_workspace",
                            "status": "ok",
                        }
                    ],
                    "preparedReadResult": {
                        "status": "ok",
                        "replayed": False,
                        "output": {
                            "ok": True,
                            "result": {"work_item": {"title": "assigned"}},
                        },
                    },
                },
            )

        binding = PlaneHostBinding(
            port=CallablePlaneHostPort(respond),
            run_id="run:code-host-consumed",
            invocation_id="invocation:code-host-consumed",
            correlation_id="correlation:code-host-consumed",
            cancellation=lambda: False,
            code_mode_phase="post_search",
        )
        result = binding.call(
            action="code",
            operation_ref="plane.code-mode.execute@1",
            input={"source": "export default async () => ({})"},
            source="code",
        )

        self.assertEqual(result.status, "ok")
        self.assertEqual(
            [request["operationRef"] for request in requests],
            ["plane.code-mode.execute@1"],
        )
        self.assertFalse(binding.prepared_read_handoff_pending())

    def test_code_mode_multi_prepared_reads_stay_pending_before_text_exit(self) -> None:
        output = {
            "result": {
                "ok": True,
                "result": {
                    "results": [],
                    "assignmentWorkItemReadDecision": {
                        "schemaVersion": "plane.assignment-read-handoff/v1",
                        "recognizedCount": 2,
                        "acceptedForm": "unrecognized",
                        "failureClass": "multiple",
                        "shape": {"nestingDepth": 0, "sizeClass": "large"},
                    },
                },
            },
            "observations": [
                {
                    "source": "code",
                    "action": "code",
                    "operationRef": "operation:search_workspace",
                    "status": "ok",
                }
            ],
        }
        requests: list[dict] = []
        binding = PlaneHostBinding(
            port=CallablePlaneHostPort(
                lambda request: requests.append(request) or _result(request, output=output)
            ),
            run_id="run:code-search-multi",
            invocation_id="invocation:code-search-multi",
            correlation_id="correlation:code-search-multi",
            cancellation=lambda: False,
            code_mode_phase="post_search",
        )

        binding.call(
            action="code",
            operation_ref="plane.code-mode.execute@1",
            input={"source": "export default async () => ({})"},
            source="code",
        )

        self.assertEqual(len(requests), 1)
        self.assertTrue(binding.prepared_read_handoff_pending())
        self.assertEqual(binding.take_code_mode_phase_hint(), "post_search")

    def test_non_execute_outer_code_operation_does_not_arm_continuation(self) -> None:
        output = {
            "result": {
                "ok": True,
                "result": {
                    "results": [
                        {
                            "objectType": "work_item",
                            "workItemReadCall": "prepared-call:guard",
                        }
                    ]
                },
            }
        }
        binding = PlaneHostBinding(
            port=CallablePlaneHostPort(
                lambda request: _result(request, output=output)
            ),
            run_id="run:outer-guard",
            invocation_id="invocation:outer-guard",
            correlation_id="correlation:outer-guard",
            cancellation=lambda: False,
            code_mode_phase="post_search",
            eager_operation_refs=frozenset({"operation:other"}),
        )

        binding.call(
            action="code",
            operation_ref="operation:other",
            input={"source": "export default async () => ({})"},
            source="code",
        )
        self.assertIsNone(binding.code_mode_phase_hint())

    def test_code_mode_search_result_accepts_v72_opaque_read_call(self) -> None:
        output = {
            "result": {
                "ok": True,
                "result": {
                    "results": [
                        {
                            "objectType": "work_item",
                            "workItemReadCall": "prepared-call:v72",
                        }
                    ]
                },
            },
            "observations": [
                {
                    "source": "code",
                    "action": "code",
                    "operationRef": "operation:search_workspace",
                    "status": "ok",
                }
            ],
        }
        self.assertEqual(
            _prepared_read_refs_from_code_mode_result(output),
            ("prepared-call:v72",),
        )

    def test_code_mode_search_result_accepts_bare_prepared_ref_object(self) -> None:
        output = {
            "result": {
                "ok": True,
                "result": {
                    "results": [
                        {
                            "objectType": "work_item",
                            "workItemReadCall": {"preparedCallRef": "prepared-call:object"},
                        }
                    ]
                },
            },
            "observations": [
                {
                    "source": "code",
                    "action": "code",
                    "operationRef": "operation:search_workspace",
                    "status": "ok",
                }
            ],
        }
        self.assertEqual(
            _prepared_read_refs_from_code_mode_result(output),
            ("prepared-call:object",),
        )

    def test_code_mode_search_result_ignores_prepared_ref_on_non_work_item(self) -> None:
        output = {
            "result": {
                "ok": True,
                "result": {
                    "results": [
                        {
                            "objectType": "project",
                            "workItemReadCall": "prepared-call:not-a-work-item",
                        }
                    ]
                },
            },
            "observations": [
                {
                    "source": "code",
                    "action": "code",
                    "operationRef": "operation:search_workspace",
                    "status": "ok",
                }
            ],
        }
        self.assertEqual(_prepared_read_refs_from_code_mode_result(output), ())

    def test_code_mode_malformed_prepared_search_result_stays_fail_closed(self) -> None:
        requests: list[dict] = []

        def respond(request: dict) -> dict:
            requests.append(request)
            return _result(
                request,
                output={
                    "schemaVersion": "plane.code-mode/v1",
                    "result": {
                        "ok": True,
                        "result": {
                            "results": [
                                {
                                    "objectType": "work_item",
                                    "workItemReadCall": {
                                        "preparedCallRef": {
                                            "preparedCallRef": "prepared-call:tampered"
                                        }
                                    },
                                }
                            ]
                        },
                    },
                    "observations": [
                        {
                            "source": "code",
                            "action": "code",
                            "operationRef": "operation:search_workspace",
                            "status": "ok",
                        }
                    ],
                },
            )

        binding = PlaneHostBinding(
            port=CallablePlaneHostPort(respond),
            run_id="run:code-search-tamper",
            invocation_id="invocation:code-search-tamper",
            correlation_id="correlation:code-search-tamper",
            cancellation=lambda: False,
            code_mode_phase="post_search",
        )
        binding.call(
            action="code",
            operation_ref="plane.code-mode.execute@1",
            input={"source": "export default async () => ({})"},
            source="code",
        )

        assert [request["operationRef"] for request in requests] == [
            "plane.code-mode.execute@1"
        ]
        assert binding.code_mode_phase_hint() is None

    def test_code_mode_read_consumption_and_non_code_operations_do_not_arm(self) -> None:
        def respond(request: dict) -> dict:
            return _result(request, output={"accepted": True})

        binding = PlaneHostBinding(
            port=CallablePlaneHostPort(respond),
            run_id="run:no-arm",
            invocation_id="invocation:no-arm",
            correlation_id="correlation:no-arm",
            cancellation=lambda: False,
            code_mode_phase="post_search",
            eager_operation_refs=frozenset(
                {
                    "operation:work_item.read",
                    "operation:work_item.rename",
                    "operation:agent.outcome.submit",
                }
            ),
        )
        for action, operation_ref in (
            ("read", "operation:work_item.read"),
            ("mutate", "operation:work_item.rename"),
            ("mutate", "operation:agent.outcome.submit"),
        ):
            binding.call(
                action=action,
                operation_ref=operation_ref,
                input={},
                source="model",
            )
        self.assertIsNone(binding.code_mode_phase_hint())

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

    def test_code_mode_diagnostic_classification_is_finite_and_redacted(self) -> None:
        cases = (
            ("invalid", "CODE_MODE_FAILED", "code_mode"),
            ("invalid", "VALIDATION_ERROR", "contract"),
            ("unavailable", "CALLBACK_FAILED", "callback"),
        )
        for status, error_code, failure_class in cases:
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
                    run_id="run:diagnostic",
                    invocation_id="invocation:diagnostic",
                    correlation_id="correlation:diagnostic",
                    cancellation=lambda: False,
                )
                binding.call(
                    action="code",
                    operation_ref="plane.code-mode.execute@1",
                    input={"source": "opaque-module"},
                    source="code",
                )
                diagnostic = binding.host_operation_diagnostic
                self.assertIsNotNone(diagnostic)
                assert diagnostic is not None
                self.assertEqual(diagnostic["codeModeHostStatus"], status)
                self.assertEqual(diagnostic["codeModeFailureClass"], failure_class)
                encoded = json.dumps(diagnostic, sort_keys=True)
                self.assertNotIn("opaque-module", encoded)
                self.assertNotIn("bounded host failure", encoded)

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
            if request["operationRef"] == PLANE_OUTCOME_SUBMIT_OPERATION:
                return _submitted_result(request)
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
        binding.call(
            action="mutate",
            operation_ref=PLANE_OUTCOME_SUBMIT_OPERATION,
            input={"summary": "submitted"},
            source="model",
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
        self.assertEqual(len(calls), 2)
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
            if request["operationRef"] == PLANE_OUTCOME_SUBMIT_OPERATION:
                return _submitted_result(request, status="replayed")
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
        binding.call(
            action="mutate",
            operation_ref=PLANE_OUTCOME_SUBMIT_OPERATION,
            input={"summary": "replayed"},
            source="model",
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
            if request["operationRef"] == PLANE_OUTCOME_SUBMIT_OPERATION:
                return _result(
                    request,
                    output={"result": {"outcome": {"outcomeRef": "outcome-submission:test"}}},
                )
            if len(calls) > 2:
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
        binding.call(
            action="mutate",
            operation_ref=PLANE_OUTCOME_SUBMIT_OPERATION,
            input={"summary": "submitted"},
            source="model",
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
        self.assertEqual(binding.outcome_submission_ref(), "outcome-submission:test")
        self.assertEqual(duplicate, first)
        self.assertEqual(len(calls), 2)
        self.assertIsNone(binding.fatal_error)
        self.assertEqual(
            [body["kind"] for body in bodies],
            [
                "progress_observed",
                "progress_observed",
                "outcome_submission_observed",
            ],
        )

    def test_replayed_outcome_publication_retains_nonarming_metadata(self) -> None:
        def rpc(request: dict) -> dict:
            if request["operationRef"] == PLANE_OUTCOME_SUBMIT_OPERATION:
                return _submitted_result(request, status="replayed")
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
        binding.call(
            action="mutate",
            operation_ref=PLANE_OUTCOME_SUBMIT_OPERATION,
            input={"summary": "replayed"},
            source="model",
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
            if request["operationRef"] == PLANE_OUTCOME_SUBMIT_OPERATION:
                return _submitted_result(request)
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
        binding.call(
            action="mutate",
            operation_ref=PLANE_OUTCOME_SUBMIT_OPERATION,
            input={"summary": "submitted"},
            source="model",
        )
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
        original = binding.records[1].result
        dedicated = binding.publish(
            kind="outcome",
            operation_ref=PLANE_OUTCOME_PUBLISH_OPERATION,
            resource_ref="outcome-submission:test",
            content="generic outcome",
        )
        self.assertIs(dedicated, original)
        self.assertEqual(calls[1]["action"], "mutate")
        self.assertEqual(len(calls), 2)
        self.assertEqual(binding.terminal_action_reason(), "product_outcome_published")
        self.assertEqual(
            [body["kind"] for body in bodies],
            [
                "progress_observed",
                "progress_observed",
                "outcome_submission_observed",
            ],
        )

    def test_generic_outcome_receipt_in_output_arms_terminal_publication(self) -> None:
        bodies: list[dict] = []
        calls: list[dict] = []

        def rpc(request: dict) -> dict:
            calls.append(request)
            if request["operationRef"] == PLANE_OUTCOME_SUBMIT_OPERATION:
                return _submitted_result(request)
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
        binding.call(
            action="mutate",
            operation_ref=PLANE_OUTCOME_SUBMIT_OPERATION,
            input={"summary": "submitted"},
            source="model",
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
        self.assertEqual(len(calls), 2)
        self.assertEqual(binding.terminal_action_reason(), "product_outcome_published")
        self.assertIsNone(binding.fatal_error)
        self.assertEqual(
            [body["kind"] for body in bodies],
            [
                "progress_observed",
                "progress_observed",
                "outcome_submission_observed",
            ],
        )
        self.assertEqual(
            bodies[2]["publication"],
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
            if request["operationRef"] == PLANE_OUTCOME_SUBMIT_OPERATION:
                return _submitted_result(request)
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
        binding.call(
            action="mutate",
            operation_ref=PLANE_OUTCOME_SUBMIT_OPERATION,
            input={"summary": "submitted"},
            source="model",
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

        self.assertEqual(len(calls), 2)
        self.assertIs(returned[0], returned[1])
        self.assertIs(returned[0], binding.records[1].result)
        self.assertEqual(binding.terminal_action_reason(), "product_outcome_published")
        self.assertEqual(
            sum(body["kind"] == "outcome_submission_observed" for body in bodies),
            1,
        )

    def test_pre_terminal_host_failure_remains_fatal_after_later_terminal(self) -> None:
        calls: list[dict] = []

        def rpc(request: dict) -> dict:
            calls.append(request)
            if request["operationRef"] == PLANE_OUTCOME_SUBMIT_OPERATION:
                return _submitted_result(request)
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
        binding.call(
            action="mutate",
            operation_ref=PLANE_OUTCOME_SUBMIT_OPERATION,
            input={"summary": "submitted"},
            source="model",
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
        self.assertEqual([request["action"] for request in calls], ["mutate", "read", "publish"])

    def test_replayed_applied_outcome_on_fresh_binding_does_not_signal(self) -> None:
        bodies: list[dict] = []

        def rpc(request: dict) -> dict:
            if request["operationRef"] == PLANE_OUTCOME_SUBMIT_OPERATION:
                return _submitted_result(request, status="replayed")
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
        binding.call(
            action="mutate",
            operation_ref=PLANE_OUTCOME_SUBMIT_OPERATION,
            input={"summary": "replayed"},
            source="model",
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
            [
                "progress_observed",
                "progress_observed",
                "outcome_submission_observed",
            ],
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
            if request["operationRef"] == PLANE_OUTCOME_SUBMIT_OPERATION:
                return _submitted_result(request)
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
        denied.call(
            action="mutate",
            operation_ref=PLANE_OUTCOME_SUBMIT_OPERATION,
            input={"summary": "submitted"},
            source="model",
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
                lambda request: (
                    _submitted_result(request)
                    if request["operationRef"] == PLANE_OUTCOME_SUBMIT_OPERATION
                    else _result(
                        request,
                        status="unavailable",
                        errorCode="OUTCOME_UNKNOWN",
                        errorMessage="publication outcome is unknown",
                    )
                )
            ),
            run_id="run:test",
            invocation_id="invocation:test",
            correlation_id="correlation:test",
            cancellation=lambda: False,
            emit_body=lambda _body: None,
        )
        failed.call(
            action="mutate",
            operation_ref=PLANE_OUTCOME_SUBMIT_OPERATION,
            input={"summary": "submitted"},
            source="model",
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
            if request["operationRef"] == PLANE_OUTCOME_SUBMIT_OPERATION:
                return _submitted_result(request)
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
        emission_failed.call(
            action="mutate",
            operation_ref=PLANE_OUTCOME_SUBMIT_OPERATION,
            input={"summary": "submitted"},
            source="model",
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

    def test_registry_normalizes_exact_prepared_read_wrappers(self) -> None:
        install_plane_tools()
        requests: list[dict] = []

        def rpc(request: dict) -> dict:
            requests.append(request)
            return _result(request, output={"work_item": {"title": "assigned"}})

        binding = PlaneHostBinding(
            port=CallablePlaneHostPort(rpc),
            run_id="run:named-prepared-envelope",
            invocation_id="invocation:named-prepared-envelope",
            correlation_id="correlation:named-prepared-envelope",
            cancellation=lambda: False,
        )
        ready_to_call = {
            "action": "read",
            "operationRef": "operation:work_item.read",
            "input": {"preparedCallRef": "prepared-call:opaque"},
        }
        wrapped_forms = (
            {"workItemReadCall": ready_to_call},
            {"preparedCallRef": ready_to_call},
            {
                "preparedCallRef": json.dumps(
                    ready_to_call, sort_keys=True, separators=(",", ":")
                )
            },
        )
        with bind_plane_host(binding):
            for wrapped in wrapped_forms:
                result = registry.dispatch(
                    "plane_operation",
                    {
                        "action": "read",
                        "operationRef": "operation:work_item.read",
                        "input": wrapped,
                    },
                )
                self.assertIn('"status":"ok"', result)

        self.assertEqual(
            [request["input"] for request in requests],
            [{"preparedCallRef": "prepared-call:opaque"}] * len(wrapped_forms),
        )

    def test_registry_normalizes_sparse_prepared_ref_and_rejects_malformed_variants(self) -> None:
        prepared_ref = "prepared-call:sparse"
        sparse = {"preparedCallRef": {"preparedCallRef": prepared_ref}}
        self.assertEqual(
            _normalize_prepared_read_input(
                "read", "operation:work_item.read", sparse
            ),
            {"preparedCallRef": prepared_ref},
        )

        malformed = (
            # Tampered opaque value.
            {"preparedCallRef": {"preparedCallRef": "not-a-prepared-call"}},
            # Extra fields must not be stripped or forwarded as a valid ref.
            {
                "preparedCallRef": {
                    "preparedCallRef": prepared_ref,
                    "issue_id": "must-not-cross-the-seam",
                }
            },
            # The sparse adapter is for the work-item read operation only.
            {
                "preparedCallRef": {
                    "action": "mutate",
                    "operationRef": "operation:work_item.read",
                    "input": {"preparedCallRef": prepared_ref},
                }
            },
            # Oversized opaque values remain untouched for gateway rejection.
            {
                "preparedCallRef": {
                    "preparedCallRef": "prepared-call:" + "x" * 256,
                }
            },
        )
        for input_value in malformed:
            self.assertIs(
                _normalize_prepared_read_input(
                    "read", "operation:work_item.read", input_value
                ),
                input_value,
            )

    def test_registry_normalizes_json_stringified_sparse_ref_only(self) -> None:
        prepared_ref = "prepared-call:json-stringified"
        stringified = json.dumps(
            {"preparedCallRef": prepared_ref},
            sort_keys=True,
            separators=(",", ":"),
        )
        self.assertEqual(
            _normalize_prepared_read_input(
                "read",
                "operation:work_item.read",
                {"preparedCallRef": stringified},
            ),
            {"preparedCallRef": prepared_ref},
        )

        malformed = (
            # Duplicate keys must not be reduced to an accepted canonical ref.
            {
                "preparedCallRef": (
                    '{"preparedCallRef":"prepared-call:first",'
                    '"preparedCallRef":"prepared-call:second"}'
                )
            },
            # Extra fields are outside the finite canonical-ref shape.
            {
                "preparedCallRef": json.dumps(
                    {"preparedCallRef": prepared_ref, "extra": True},
                    separators=(",", ":"),
                )
            },
            # The opaque reference prefix remains gateway-owned validation.
            {
                "preparedCallRef": json.dumps(
                    {"preparedCallRef": "not-a-prepared-call"},
                    separators=(",", ":"),
                )
            },
            # Oversized opaque values are not normalized.
            {
                "preparedCallRef": json.dumps(
                    {"preparedCallRef": "prepared-call:" + "x" * 256},
                    separators=(",", ":"),
                )
            },
            # The accepted branch is exactly one JSON object layer, not recursive.
            {
                "preparedCallRef": json.dumps(
                    {"preparedCallRef": {"preparedCallRef": prepared_ref}},
                    separators=(",", ":"),
                )
            },
        )
        for input_value in malformed:
            self.assertIs(
                _normalize_prepared_read_input(
                    "read", "operation:work_item.read", input_value
                ),
                input_value,
            )

    def test_registry_normalizes_sparse_prepared_ref_and_rejects_malformed_variants(self) -> None:
        prepared_ref = "prepared-call:sparse"
        sparse = {"preparedCallRef": {"preparedCallRef": prepared_ref}}
        self.assertEqual(
            _normalize_prepared_read_input(
                "read", "operation:work_item.read", sparse
            ),
            {"preparedCallRef": prepared_ref},
        )

        malformed = (
            {"preparedCallRef": {"preparedCallRef": "not-a-prepared-call"}},
            {
                "preparedCallRef": {
                    "preparedCallRef": prepared_ref,
                    "issue_id": "must-not-cross-the-seam",
                }
            },
            {
                "preparedCallRef": {
                    "action": "mutate",
                    "operationRef": "operation:work_item.read",
                    "input": {"preparedCallRef": prepared_ref},
                }
            },
            {
                "preparedCallRef": {
                    "preparedCallRef": "prepared-call:" + "x" * 256,
                }
            },
        )
        for input_value in malformed:
            self.assertIs(
                _normalize_prepared_read_input(
                    "read", "operation:work_item.read", input_value
                ),
                input_value,
            )

    def test_registry_normalizes_bare_and_input_wrapped_prepared_read_refs(self) -> None:
        install_plane_tools()
        requests: list[dict] = []

        def rpc(request: dict) -> dict:
            requests.append(request)
            return _result(request, output={"work_item": {"title": "assigned"}})

        binding = PlaneHostBinding(
            port=CallablePlaneHostPort(rpc),
            run_id="run:bare-prepared-envelope",
            invocation_id="invocation:bare-prepared-envelope",
            correlation_id="correlation:bare-prepared-envelope",
            cancellation=lambda: False,
        )
        with bind_plane_host(binding):
            for input_value in (
                {"preparedCallRef": "prepared-call:bare"},
                {"input": {"preparedCallRef": "prepared-call:input-wrapper"}},
            ):
                result = registry.dispatch(
                    "plane_operation",
                    {
                        "action": "read",
                        "operationRef": "operation:work_item.read",
                        "input": input_value,
                    },
                )
                self.assertIn('"status":"ok"', result)

        self.assertEqual(
            [request["input"] for request in requests],
            [
                {"preparedCallRef": "prepared-call:bare"},
                {"preparedCallRef": "prepared-call:input-wrapper"},
            ],
        )

    def test_named_prepared_read_wrapper_rejects_tamper_shapes(self) -> None:
        install_plane_tools()
        requests: list[dict] = []

        def rpc(request: dict) -> dict:
            requests.append(request)
            return _result(request, output={"accepted": True})

        binding = PlaneHostBinding(
            port=CallablePlaneHostPort(rpc),
            run_id="run:named-prepared-envelope-tamper",
            invocation_id="invocation:named-prepared-envelope-tamper",
            correlation_id="correlation:named-prepared-envelope-tamper",
            cancellation=lambda: False,
        )
        shapes = (
            {
                "workItemReadCall": {
                    "action": "read",
                    "operationRef": "operation:work_item.read",
                    "input": {
                        "preparedCallRef": "prepared-call:opaque",
                        "issue_id": "should-not-be-forwarded",
                    },
                }
            },
            {
                "workItemReadCall": {
                    "action": "read",
                    "operationRef": "operation:work_item.rename",
                    "input": {"preparedCallRef": "prepared-call:opaque"},
                }
            },
            {
                "workItemReadCall": {
                    "action": "read",
                    "operationRef": "operation:work_item.read",
                    "input": {"preparedCallRef": "prepared-call:" + "x" * 256},
                }
            },
        )
        with bind_plane_host(binding):
            for shape in shapes:
                result = registry.dispatch(
                    "plane_operation",
                    {
                        "action": "read",
                        "operationRef": "operation:work_item.read",
                        "input": shape,
                    },
                )
                self.assertIn('"status":"ok"', result)

        self.assertEqual([request["input"] for request in requests], list(shapes))

    def test_prepared_read_unwrap_rejects_extra_and_deep_wrappers(self) -> None:
        install_plane_tools()
        requests: list[dict] = []

        def rpc(request: dict) -> dict:
            requests.append(request)
            return _result(request, output={"accepted": True})

        binding = PlaneHostBinding(
            port=CallablePlaneHostPort(rpc),
            run_id="run:prepared-envelope-depth",
            invocation_id="invocation:prepared-envelope-depth",
            correlation_id="correlation:prepared-envelope-depth",
            cancellation=lambda: False,
        )
        shapes = (
            {"preparedCallRef": "prepared-call:opaque", "extra": True},
            {"preparedCallRef": "not-a-prepared-call"},
            {"preparedCallRef": "prepared-call:" + "x" * 256},
            {"input": {"input": {"preparedCallRef": "prepared-call:opaque"}}},
            {"workItemReadCall": {"preparedCallRef": "prepared-call:opaque"}},
            {
                "action": "read",
                "operationRef": "operation:work_item.read",
                "input": {
                    "action": "mutate",
                    "operationRef": "operation:work_item.read",
                    "input": {"preparedCallRef": "prepared-call:opaque"},
                },
            },
        )
        with bind_plane_host(binding):
            for shape in shapes:
                result = registry.dispatch(
                    "plane_operation",
                    {
                        "action": "read",
                        "operationRef": "operation:work_item.read",
                        "input": shape,
                    },
                )
                self.assertIn('"status":"ok"', result)

        self.assertEqual([request["input"] for request in requests], list(shapes))

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
                and body["payload"].get("kind") == "inline_text"
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

    def test_real_hermes_route_reads_search_ref_before_ordinary_final_text(self) -> None:
        """The actual model turn cannot finish before its opaque read continuation."""

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
        snapshot_raw["runtimePolicy"].update(  # type: ignore[union-attr]
            {
                "adapter": "hermes",
                "model": {"provider": "openai", "model": "deterministic-local"},
            }
        )
        snapshot_raw["toolCatalog"] = {  # type: ignore[index]
            "catalogDigest": "content:" + "c" * 64,
            "modelToolset": "standard",
            "eagerOperations": [
                {
                    "operationRef": "operation:search_workspace",
                    "schemaDigest": "content:" + "d" * 64,
                    "inputSchema": {"type": "object"},
                    "disclosure": "eager",
                }
            ],
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
                    message = SimpleNamespace(
                        content=None,
                        tool_calls=[
                            self.tool_call(
                                {
                                    "action": "read",
                                    "operationRef": "operation:search_workspace",
                                    "input": {"query": "assigned", "limit": 1},
                                },
                                "call-search",
                            )
                        ],
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
                self.completions = Completions()
                self.chat = SimpleNamespace(completions=self.completions)

        class Credentials:
            def resolve(self, provider: str) -> dict[str, str]:
                del provider
                return {"api_key": "model-only-secret", "base_url": "http://127.0.0.1"}

        client = Client()

        def agent_factory(**kwargs: object) -> AIAgent:
            agent = AIAgent(**kwargs)
            agent._create_request_openai_client = lambda **_: client  # type: ignore[method-assign]
            return agent

        requests: list[dict] = []

        def rpc(request: dict) -> dict:
            requests.append(request)
            if request["operationRef"] == "operation:search_workspace":
                output = {
                    "ok": True,
                    "result": {
                        "results": [
                            {
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
                self.assertEqual(
                    request["input"], {"preparedCallRef": "prepared-call:opaque"}
                )
                output = {"ok": True, "result": {"work_item": {"title": "assigned"}}}
            return _result(request, output=output)

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
        self.assertEqual(result.output_text, "ordinary final evidence")
        self.assertEqual(client.completions.calls, 2)
        self.assertEqual(
            [request["operationRef"] for request in requests],
            ["operation:search_workspace", "operation:work_item.read"],
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
