"""Invocation-local progressive disclosure tests for the Plane host seam."""

from __future__ import annotations

import json
import unittest

from plane_runtime.host_port import (
    CallablePlaneHostPort,
    PlaneHostBinding,
    PlaneHostSchemaNotDisclosed,
    PlaneHostUnavailable,
    bind_plane_host,
    install_plane_tools,
    plane_code_mode,
)
from tools.registry import registry


_INPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["project_id", "issue_id"],
    "properties": {
        "project_id": {"type": "string"},
        "issue_id": {"type": "string"},
    },
}


def _result(request: dict, *, output: object = None, status: str = "ok", **extra: object) -> dict:
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


class HostDisclosureTests(unittest.TestCase):
    def test_eager_invocation_and_discovery_are_allowed_but_progressive_is_recoverable(self) -> None:
        requests: list[dict] = []
        binding = PlaneHostBinding(
            port=CallablePlaneHostPort(
                lambda request: requests.append(request) or _result(request, output={"accepted": True})
            ),
            run_id="run:test",
            invocation_id="invocation:test",
            correlation_id="correlation:test",
            cancellation=lambda: False,
            eager_operation_refs=frozenset({"operation:work_item.read"}),
        )

        install_plane_tools()
        with bind_plane_host(binding):
            eager = registry.dispatch(
                "plane_operation",
                {
                    "action": "read",
                    "operationRef": "operation:work_item.read",
                    "input": {"project_id": "project", "issue_id": "issue"},
                },
            )
            blocked = registry.dispatch(
                "plane_operation",
                {
                    "action": "read",
                    "operationRef": "operation:work_item.rename",
                    "input": {"project_id": "project", "issue_id": "issue"},
                },
            )
            discovered = registry.dispatch(
                "plane_operation",
                {"action": "discover", "input": {}},
            )

        self.assertIn('"accepted":true', eager)
        self.assertEqual(json.loads(blocked)["error"]["code"], "SCHEMA_NOT_DISCLOSED")
        self.assertIn('"accepted":true', discovered)
        self.assertEqual(
            [request["operationRef"] for request in requests],
            ["operation:work_item.read", "plane.operations.discover@1"],
        )
        self.assertIsNone(binding.fatal_error)

    def test_describe_unlocks_only_the_exact_operation_and_current_binding(self) -> None:
        requests: list[dict] = []

        def rpc(request: dict) -> dict:
            requests.append(request)
            if request["operationRef"] == "operation:catalog.describe":
                operation_id = request["input"]["operation_id"]
                return _result(
                    request,
                    output={
                        "operation": {
                            "operationId": operation_id,
                            "operationRef": f"operation:{operation_id}",
                            "inputSchema": _INPUT_SCHEMA,
                        }
                    },
                )
            return _result(request, output={"accepted": True})

        port = CallablePlaneHostPort(rpc)
        binding = PlaneHostBinding(
            port=port,
            run_id="run:test",
            invocation_id="invocation:test",
            correlation_id="correlation:test",
            cancellation=lambda: False,
        )
        binding.call(
            action="read",
            operation_ref="operation:catalog.search",
            input={"query": "work item"},
            source="model",
        )
        binding.call(
            action="read",
            operation_ref="operation:catalog.describe",
            input={"operation_id": "work_item.read"},
            source="model",
        )
        binding.call(
            action="read",
            operation_ref="operation:work_item.read",
            input={"project_id": "project", "issue_id": "issue"},
            source="model",
        )
        with self.assertRaises(PlaneHostSchemaNotDisclosed):
            binding.call(
                action="read",
                operation_ref="operation:work_item.rename",
                input={"project_id": "project", "issue_id": "issue"},
                source="model",
            )

        other_binding = PlaneHostBinding(
            port=port,
            run_id="run:test",
            invocation_id="invocation:other",
            correlation_id="correlation:other",
            cancellation=lambda: False,
        )
        with self.assertRaises(PlaneHostSchemaNotDisclosed):
            other_binding.call(
                action="read",
                operation_ref="operation:work_item.read",
                input={"project_id": "project", "issue_id": "issue"},
                source="model",
            )
        self.assertEqual(
            [request["operationRef"] for request in requests],
            [
                "operation:catalog.search",
                "operation:catalog.describe",
                "operation:work_item.read",
            ],
        )

    def test_malformed_describe_result_fails_closed_without_disclosing_schema(self) -> None:
        def mismatched(request: dict) -> dict:
            return _result(
                request,
                output={
                    "operation": {
                        "operationId": "work_item.read",
                        "operationRef": "operation:work_item.rename",
                        "inputSchema": _INPUT_SCHEMA,
                    }
                },
            )

        binding = PlaneHostBinding(
            port=CallablePlaneHostPort(mismatched),
            run_id="run:test",
            invocation_id="invocation:test",
            correlation_id="correlation:test",
            cancellation=lambda: False,
        )
        with self.assertRaises(PlaneHostUnavailable) as raised:
            binding.call(
                action="read",
                operation_ref="operation:catalog.describe",
                input={"operation_id": "work_item.read"},
                source="model",
            )
        self.assertNotIn("project_id", str(raised.exception))
        self.assertIsNotNone(binding.fatal_error)
        self.assertEqual(binding.described_operation_refs, set())

    def test_code_mode_and_publication_use_the_same_disclosure_precondition(self) -> None:
        requests: list[dict] = []
        binding = PlaneHostBinding(
            port=CallablePlaneHostPort(
                lambda request: requests.append(request) or _result(request, output={"accepted": True})
            ),
            run_id="run:test",
            invocation_id="invocation:test",
            correlation_id="correlation:test",
            cancellation=lambda: False,
        )
        install_plane_tools()
        with bind_plane_host(binding), plane_code_mode():
            code_result = registry.dispatch(
                "plane_operation",
                {
                    "action": "code",
                    "operationRef": "operation:compose",
                    "input": {},
                },
            )
        with self.assertRaises(PlaneHostSchemaNotDisclosed):
            binding.publish(
                kind="conversation",
                operation_ref="operation:conversation.publish",
                resource_ref="conversation:test",
                content="explicit publication",
            )
        self.assertEqual(json.loads(code_result)["error"]["code"], "SCHEMA_NOT_DISCLOSED")
        self.assertEqual(requests, [])
        self.assertIsNone(binding.fatal_error)


if __name__ == "__main__":
    unittest.main()
