"""Provider-free adapter proof for one canonical Plane work-item lifecycle."""

from __future__ import annotations

import copy
import json
import unittest

from plane_runtime.g1_contract import G1InvocationEnvelope, G1RunSnapshot
from plane_runtime.hermes_adapter import HermesKernelAdapter
from plane_runtime.host_port import CallablePlaneHostPort, current_plane_host
from tests.plane_runtime.test_g1_runtime_process import _digest, make_invocation, make_snapshot
from tools.registry import registry


class AdapterPresentationTests(unittest.TestCase):
    def test_manager_route_uses_search_then_canonical_work_item_read_input(self) -> None:
        raw = copy.deepcopy(make_snapshot())
        raw["assignment"] = {
            "assignmentRef": "assignment:manager",
            "revision": "revision:one",
            "targetRef": "target:assigned-work-item",
            "objective": "Coordinate a bounded Manager objective for the assigned work item.",
            "acceptanceCriteria": ["Publish one reviewable result."],
        }
        raw["profile"]["role"] = "delegator"  # type: ignore[index]
        raw["toolCatalog"] = {
            "catalogDigest": "content:" + "c" * 64,
            "eagerOperations": [
                {
                    "operationRef": "operation:catalog.search",
                    "schemaDigest": "content:" + "b" * 64,
                    "inputSchema": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["query"],
                        "properties": {"query": {"type": "string"}},
                    },
                    "disclosure": "eager",
                },
                {
                    "operationRef": "operation:catalog.describe",
                    "schemaDigest": "content:" + "a" * 64,
                    "inputSchema": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["operation_id"],
                        "properties": {"operation_id": {"type": "string"}},
                    },
                    "disclosure": "eager",
                },
                {
                    "operationRef": "operation:search_workspace",
                    "schemaDigest": "content:" + "d" * 64,
                    "inputSchema": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["query"],
                        "properties": {"query": {"type": "string"}},
                    },
                    "disclosure": "eager",
                },
                {
                    "operationRef": "operation:work_item.read",
                    "schemaDigest": "content:" + "e" * 64,
                    "inputSchema": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["project_id", "issue_id"],
                        "properties": {
                            "project_id": {"type": "string"},
                            "issue_id": {"type": "string"},
                        },
                    },
                    "disclosure": "eager",
                },
            ],
        }
        raw["contentDigest"] = _digest(
            "snapshot", {key: value for key, value in raw.items() if key != "contentDigest"}
        )
        snapshot = G1RunSnapshot.from_dict(raw)
        invocation = G1InvocationEnvelope.from_dict(make_invocation(snapshot.to_dict()))
        requests: list[dict] = []

        def rpc(request: dict) -> dict:
            requests.append(request)
            if request["operationRef"] == "operation:search_workspace":
                return {
                    "protocol": "plane.agent-runtime/v1",
                    "requestRef": request["requestRef"],
                    "correlationId": request["correlationId"],
                    "idempotencyKey": request["idempotencyKey"],
                    "status": "ok",
                    "replayed": False,
                    "output": {
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
                    },
                }
            if request["operationRef"] == "operation:work_item.read":
                self.assertEqual(
                    request["input"],
                    {"preparedCallRef": "prepared-call:opaque"},
                )
                return {
                    "protocol": "plane.agent-runtime/v1",
                    "requestRef": request["requestRef"],
                    "correlationId": request["correlationId"],
                    "idempotencyKey": request["idempotencyKey"],
                    "status": "ok",
                    "replayed": False,
                    "output": {"ok": True, "result": {"work_item": {"title": "assigned"}}},
                }
            raise AssertionError(f"unexpected operation {request['operationRef']}")

        captured: dict[str, object] = {}

        class FakeAgent:
            session_input_tokens = 1
            session_output_tokens = 1
            session_api_calls = 1

            def run_conversation(self, message: str, *, system_message: str) -> dict[str, str]:
                captured["message"] = message
                captured["system_message"] = system_message
                if "operation:search_workspace the first Plane call" not in system_message:
                    raise AssertionError("Manager guidance did not prioritize search_workspace")
                binding = current_plane_host()
                self.assert_binding(binding)
                search = json.loads(
                    registry.dispatch(
                        "plane_operation",
                        {
                            "action": "read",
                            "operationRef": "operation:search_workspace",
                            "input": {"query": "assigned work item"},
                        },
                    )
                )
                return {"final_response": search}

            @staticmethod
            def assert_binding(binding: object) -> None:
                if binding is None:
                    raise AssertionError("Plane host was not bound for the invocation")

        class Credentials:
            def resolve(self, provider: str) -> dict[str, str]:
                del provider
                return {"api_key": "provider-free-test-secret"}

        result = HermesKernelAdapter(
            agent_factory=lambda **kwargs: FakeAgent(),
            credential_source=Credentials(),
            host_port=CallablePlaneHostPort(rpc),
        ).dispatch(snapshot, invocation, lambda: False, lambda body: None, model_call_allowance=1)

        self.assertEqual(result.kind, "completed")
        self.assertEqual(
            [request["operationRef"] for request in requests],
            ["operation:search_workspace", "operation:work_item.read"],
        )
        self.assertNotIn("operation:catalog.describe", [request["operationRef"] for request in requests])
        self.assertIn('"project_id"', str(captured["system_message"]))
        self.assertIn('"issue_id"', str(captured["system_message"]))


if __name__ == "__main__":
    unittest.main()
