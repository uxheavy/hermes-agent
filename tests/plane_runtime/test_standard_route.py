from __future__ import annotations

import pytest

from plane_runtime.host_port import (
    CallablePlaneHostPort,
    PlaneHostBinding,
    PlaneHostUnavailable,
    PLANE_OUTCOME_PUBLISH_OPERATION,
    PLANE_OUTCOME_SUBMIT_OPERATION,
)


ROUTE = {
    "schemaVersion": "plane.standard-route/v1",
    "steps": [
        {"operationRef": "operation:catalog.search"},
        {"operationRef": "operation:catalog.describe"},
        {"operationRef": "operation:search_workspace"},
        {"operationRef": "operation:work_item.read"},
        {
            "operationRef": "operation:agent.outcome.evaluate",
            "expectedStatus": "denied",
            "expectedErrorCode": "NOT_AUTHORIZED",
        },
        {"operationRef": PLANE_OUTCOME_SUBMIT_OPERATION},
        {"operationRef": PLANE_OUTCOME_PUBLISH_OPERATION},
    ],
}


def _result(request, *, status="ok", output=None, **extra):
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


def _binding(responder):
    return PlaneHostBinding(
        port=CallablePlaneHostPort(responder),
        run_id="run:route",
        invocation_id="invocation:route",
        correlation_id="correlation:route",
        cancellation=lambda: False,
        standard_route=True,
        standard_route_contract=ROUTE,
        eager_operation_refs=frozenset(step["operationRef"] for step in ROUTE["steps"]),
    )


def test_bound_standard_route_advances_denial_then_uses_existing_publish_guard():
    requests = []

    def respond(request):
        requests.append(request)
        if request["operationRef"] == "operation:search_workspace":
            return _result(request, output={"result": {"results": [{"workItemReadCall": "prepared-call:route"}]}})
        if request["operationRef"] == "operation:catalog.describe":
            return _result(
                request,
                output={
                    "operation": {
                        "operationRef": "operation:search_workspace",
                        "operationId": "search_workspace",
                        "inputSchema": {"type": "object"},
                    }
                },
            )
        if request["operationRef"] == "operation:agent.outcome.evaluate":
            return _result(request, status="denied", errorCode="NOT_AUTHORIZED", errorMessage="not authorized")
        if request["operationRef"] == PLANE_OUTCOME_SUBMIT_OPERATION:
            return _result(request, output={"result": {"outcome": {"outcomeRef": "outcome-submission:route"}}})
        if request["action"] == "publish":
            return _result(
                request,
                output={"published": True},
                publication={
                    "action": "applied",
                    "productKind": "outcome_submission",
                    "productRef": "outcome-submission:route",
                    "operationAttemptRef": "operation-attempt:route",
                    "operationRef": PLANE_OUTCOME_PUBLISH_OPERATION,
                    "applicationServiceRef": "application-service:conversation",
                    "gatewayReceiptRef": "gateway-receipt:route",
                    "receiptRef": "receipt:route",
                    "auditReceiptRef": "audit-receipt:route",
                    "productEventRef": "product-event:route",
                },
            )
        return _result(request)

    binding = _binding(respond)
    binding.call(action="read", operation_ref="operation:catalog.search", input={}, source="model")
    binding.call(action="read", operation_ref="operation:catalog.describe", input={"operation_id": "search_workspace"}, source="model")
    binding.call(action="read", operation_ref="operation:search_workspace", input={}, source="model")
    denial = binding.call(action="mutate", operation_ref="operation:agent.outcome.evaluate", input={}, source="model")
    assert denial.status == "denied"
    binding.call(action="mutate", operation_ref=PLANE_OUTCOME_SUBMIT_OPERATION, input={}, source="model")
    binding.publish(kind="outcome", operation_ref=PLANE_OUTCOME_PUBLISH_OPERATION, resource_ref="outcome-submission:route", content="published")

    assert [request["operationRef"] for request in requests] == [
        "operation:catalog.search",
        "operation:catalog.describe",
        "operation:search_workspace",
        "operation:work_item.read",
        "operation:agent.outcome.evaluate",
        PLANE_OUTCOME_SUBMIT_OPERATION,
        PLANE_OUTCOME_PUBLISH_OPERATION,
    ]
    assert binding.publication_count == 1
    assert binding.standard_route_required_tool() is None


def test_wrong_step_does_not_reach_host():
    calls = []

    def respond(request):
        calls.append(request)
        return _result(request, status="denied", errorCode="WRONG_DENIAL")

    binding = _binding(respond)
    with pytest.raises(PlaneHostUnavailable):
        binding.call(action="read", operation_ref="operation:search_workspace", input={}, source="model")
    assert calls == []
