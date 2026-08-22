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


def test_standard_route_auto_reads_one_trusted_ref_and_advances_route():
    route = {
        "schemaVersion": "plane.standard-route/v1",
        "steps": [
            {"operationRef": "operation:search_workspace"},
            {"operationRef": "operation:work_item.read"},
            {"operationRef": "operation:agent.outcome.evaluate"},
        ],
    }
    requests = []

    def respond(request):
        requests.append(request)
        if request["operationRef"] == "operation:search_workspace":
            return _result(
                request,
                output={
                    "result": {
                        "results": [
                            {
                                "objectType": "work_item",
                                "workItemReadCall": "prepared-call:trusted",
                            }
                        ]
                    }
                },
            )
        assert request["operationRef"] == "operation:work_item.read"
        assert request["input"] == {"preparedCallRef": "prepared-call:trusted"}
        return _result(request, output={"work_item": {"title": "assigned"}})

    binding = PlaneHostBinding(
        port=CallablePlaneHostPort(respond),
        run_id="run:trusted-auto-read",
        invocation_id="invocation:trusted-auto-read",
        correlation_id="correlation:trusted-auto-read",
        cancellation=lambda: False,
        standard_route=True,
        standard_route_contract=route,
        eager_operation_refs=frozenset(step["operationRef"] for step in route["steps"]),
    )
    result = binding.call(
        action="read",
        operation_ref="operation:search_workspace",
        input={},
        source="model",
    )

    assert result.status == "ok"
    assert [request["operationRef"] for request in requests] == [
        "operation:search_workspace",
        "operation:work_item.read",
    ]
    assert result.output["preparedReadResult"]["output"]["work_item"]["title"] == "assigned"
    assert binding.standard_route_required_tool() == "plane_operation"
    assert binding._prepared_call_registry == {"prepared-call:trusted": True}


def test_standard_route_read_failure_is_attached_and_stays_pending():
    route = {
        "schemaVersion": "plane.standard-route/v1",
        "steps": [
            {"operationRef": "operation:search_workspace"},
            {"operationRef": "operation:work_item.read"},
        ],
    }
    requests = []

    def respond(request):
        requests.append(request)
        if request["operationRef"] == "operation:search_workspace":
            output = {"result": {"results": [{"workItemReadCall": "prepared-call:failure"}]}}
            return _result(request, output=output)
        return _result(
            request,
            status="invalid",
            errorCode="NOT_AUTHORIZED",
            errorMessage="read denied",
        )

    binding = PlaneHostBinding(
        port=CallablePlaneHostPort(respond),
        run_id="run:read-failure",
        invocation_id="invocation:read-failure",
        correlation_id="correlation:read-failure",
        cancellation=lambda: False,
        standard_route=True,
        standard_route_contract=route,
        eager_operation_refs=frozenset(step["operationRef"] for step in route["steps"]),
    )

    result = binding.call(
        action="read",
        operation_ref="operation:search_workspace",
        input={},
        source="model",
    )

    assert result.status == "ok"
    assert result.output["preparedReadResult"]["errorCode"] == "NOT_AUTHORIZED"
    assert binding.prepared_read_handoff_pending() is True
    assert [request["operationRef"] for request in requests] == [
        "operation:search_workspace",
        "operation:work_item.read",
    ]
    with pytest.raises(PlaneHostUnavailable):
        binding.call(
            action="read",
            operation_ref="operation:search_workspace",
            input={},
            source="model",
        )
    assert [request["operationRef"] for request in requests] == [
        "operation:search_workspace",
        "operation:work_item.read",
    ]


def test_standard_route_replays_only_a_completed_workspace_search():
    route = {
        "schemaVersion": "plane.standard-route/v1",
        "steps": [
            {"operationRef": "operation:search_workspace"},
            {"operationRef": "operation:work_item.read"},
        ],
    }
    requests = []

    def respond(request):
        requests.append(request)
        if request["operationRef"] == "operation:search_workspace":
            if len(requests) == 1:
                return _result(
                    request,
                    status="invalid",
                    errorCode="SEARCH_FAILED",
                    errorMessage="search failed",
                )
            return _result(
                request,
                output={
                    "result": {
                        "results": [{"workItemReadCall": "prepared-call:assigned"}]
                    }
                },
            )
        assert request["operationRef"] == "operation:work_item.read"
        return _result(request, output={"work_item": {"title": "assigned"}})

    binding = PlaneHostBinding(
        port=CallablePlaneHostPort(respond),
        run_id="run:workspace-search-replay",
        invocation_id="invocation:workspace-search-replay",
        correlation_id="correlation:workspace-search-replay",
        cancellation=lambda: False,
        standard_route=True,
        standard_route_contract=route,
        eager_operation_refs=frozenset(step["operationRef"] for step in route["steps"]),
    )

    failed = binding.call(
        action="read",
        operation_ref="operation:search_workspace",
        input={"query": "assigned"},
        source="model",
    )
    completed = binding.call(
        action="read",
        operation_ref="operation:search_workspace",
        input={"query": "assigned"},
        source="model",
    )
    repeated = binding.call(
        action="read",
        operation_ref="operation:search_workspace",
        input={"query": "assigned"},
        source="model",
    )

    assert failed.status == "invalid"
    assert completed.status == "ok"
    assert repeated.status == "replayed"
    assert repeated.output == {
        "alreadySearched": True,
        "operationRef": "operation:search_workspace",
    }
    with pytest.raises(PlaneHostUnavailable):
        binding.call(
            action="read",
            operation_ref="operation:search_workspace",
            input={"query": "new-target"},
            source="model",
        )
    assert [request["operationRef"] for request in requests] == [
        "operation:search_workspace",
        "operation:search_workspace",
        "operation:work_item.read",
    ]


def test_standard_route_does_not_auto_read_ambiguous_refs():
    route = {
        "schemaVersion": "plane.standard-route/v1",
        "steps": [
            {"operationRef": "operation:search_workspace"},
            {"operationRef": "operation:work_item.read"},
        ],
    }
    requests = []

    def respond(request):
        requests.append(request)
        return _result(
            request,
            output={
                "result": {
                    "results": [
                        {"workItemReadCall": "prepared-call:first"},
                        {"workItemReadCall": "prepared-call:second"},
                    ]
                }
            },
        )

    binding = PlaneHostBinding(
        port=CallablePlaneHostPort(respond),
        run_id="run:ambiguous-auto-read",
        invocation_id="invocation:ambiguous-auto-read",
        correlation_id="correlation:ambiguous-auto-read",
        cancellation=lambda: False,
        standard_route=True,
        standard_route_contract=route,
        eager_operation_refs=frozenset(step["operationRef"] for step in route["steps"]),
    )

    result = binding.call(
        action="read",
        operation_ref="operation:search_workspace",
        input={},
        source="model",
    )

    assert result.status == "ok"
    assert "preparedReadResult" not in result.output
    assert binding.prepared_read_handoff_pending() is True
    assert len(requests) == 1
    with pytest.raises(PlaneHostUnavailable):
        binding.call(
            action="read",
            operation_ref="operation:search_workspace",
            input={},
            source="model",
        )
    assert len(requests) == 1


def test_non_standard_search_does_not_auto_read_prepared_ref():
    requests = []

    def respond(request):
        requests.append(request)
        return _result(
            request,
            output={"result": {"results": [{"workItemReadCall": "prepared-call:non-standard"}]}},
        )

    binding = PlaneHostBinding(
        port=CallablePlaneHostPort(respond),
        run_id="run:non-standard",
        invocation_id="invocation:non-standard",
        correlation_id="correlation:non-standard",
        cancellation=lambda: False,
        eager_operation_refs=frozenset({"operation:search_workspace"}),
    )

    result = binding.call(
        action="read",
        operation_ref="operation:search_workspace",
        input={},
        source="model",
    )

    assert result.status == "ok"
    assert "preparedReadResult" not in result.output
    assert [request["operationRef"] for request in requests] == [
        "operation:search_workspace"
    ]


@pytest.mark.parametrize(
    "registry",
    [
        {},
        {"prepared-call:consumed": True},
        {"prepared-call:first": False, "prepared-call:second": False},
    ],
)
def test_standard_route_does_not_select_an_ambiguous_or_unbound_ref(registry):
    route = {
        "schemaVersion": "plane.standard-route/v1",
        "steps": [{"operationRef": "operation:work_item.read"}],
    }
    requests = []

    def respond(request):
        requests.append(request)
        return _result(
            request,
            status="invalid",
            errorCode="VALIDATION_ERROR",
            errorMessage="prepared ref required",
        )

    binding = PlaneHostBinding(
        port=CallablePlaneHostPort(respond),
        run_id="run:fail-closed",
        invocation_id="invocation:fail-closed",
        correlation_id="correlation:fail-closed",
        cancellation=lambda: False,
        standard_route=True,
        standard_route_contract=route,
        eager_operation_refs=frozenset({"operation:work_item.read"}),
    )
    binding._prepared_call_registry.update(registry)
    result = binding.call(
        action="read",
        operation_ref="operation:work_item.read",
        input={"issue_id": "must-not-be-forwarded-as-a-ref"},
        source="model",
    )

    assert result.error_code == "PREPARED_CALL_INVALID"
    assert requests == []


def test_standard_route_rejects_wrong_read_action_before_host_dispatch():
    route = {
        "schemaVersion": "plane.standard-route/v1",
        "steps": [{"operationRef": "operation:work_item.read"}],
    }
    requests = []
    binding = PlaneHostBinding(
        port=CallablePlaneHostPort(lambda request: requests.append(request) or _result(request)),
        run_id="run:wrong-action",
        invocation_id="invocation:wrong-action",
        correlation_id="correlation:wrong-action",
        cancellation=lambda: False,
        standard_route=True,
        standard_route_contract=route,
        eager_operation_refs=frozenset({"operation:work_item.read"}),
    )

    with pytest.raises(PlaneHostUnavailable):
        binding.call(
            action="mutate",
            operation_ref="operation:work_item.read",
            input={},
            source="model",
        )
    assert requests == []


def test_standard_route_does_not_replace_an_explicit_tampered_ref():
    route = {
        "schemaVersion": "plane.standard-route/v1",
        "steps": [{"operationRef": "operation:work_item.read"}],
    }
    requests = []
    binding = PlaneHostBinding(
        port=CallablePlaneHostPort(lambda request: requests.append(request) or _result(request)),
        run_id="run:tampered-ref",
        invocation_id="invocation:tampered-ref",
        correlation_id="correlation:tampered-ref",
        cancellation=lambda: False,
        standard_route=True,
        standard_route_contract=route,
        eager_operation_refs=frozenset({"operation:work_item.read"}),
    )
    binding._prepared_call_registry["prepared-call:trusted"] = False

    result = binding.call(
        action="read",
        operation_ref="operation:work_item.read",
        input={"preparedCallRef": "prepared-call:tampered"},
        source="model",
    )

    assert result.error_code == "PREPARED_CALL_INVALID"
    assert requests == []


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
