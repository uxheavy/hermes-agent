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
    def test_fake_agent_uses_search_then_canonical_work_item_read_input(self) -> None:
        raw = copy.deepcopy(make_snapshot())
        raw["toolCatalog"] = {
            "catalogDigest": "content:" + "c" * 64,
            "eagerOperations": [
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

        project_id = "project-7f8e9d0c"
        issue_id = "issue-1a2b3c4d"

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
                        "results": [
                            {
                                "objectType": "work_item",
                                "project_id": project_id,
                                "issue_id": issue_id,
                            }
                        ]
                    },
                }
            if request["operationRef"] == "operation:work_item.read":
                self.assertEqual(
                    request["input"],
                    {"project_id": project_id, "issue_id": issue_id},
                )
                return {
                    "protocol": "plane.agent-runtime/v1",
                    "requestRef": request["requestRef"],
                    "correlationId": request["correlationId"],
                    "idempotencyKey": request["idempotencyKey"],
                    "status": "ok",
                    "replayed": False,
                    "output": {"work_item": {"project_id": project_id, "issue_id": issue_id}},
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
                found = search["output"]["results"][0]
                read = registry.dispatch(
                    "plane_operation",
                    {
                        "action": "read",
                        "operationRef": "operation:work_item.read",
                        "input": {
                            "project_id": found["project_id"],
                            "issue_id": found["issue_id"],
                        },
                    },
                )
                return {"final_response": read}

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
        self.assertIn('"project_id"', str(captured["system_message"]))
        self.assertIn('"issue_id"', str(captured["system_message"]))


if __name__ == "__main__":
    unittest.main()
