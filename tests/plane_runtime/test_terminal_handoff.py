"""Provider-free regressions for the Hermes terminal handoff seam."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from plane_runtime.host_port import (
    CallablePlaneHostPort,
    PLANE_CODE_MODE_EXECUTE_OPERATION,
    PLANE_OUTCOME_PUBLISH_OPERATION,
)


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


def _tool_call(
    name: str,
    arguments: dict[str, object],
    call_id: str,
) -> SimpleNamespace:
    if name == "plane_execute_typescript":
        function_name = name
        function_arguments = arguments
    else:
        function_name = "tool_call"
        function_arguments = {"name": name, "arguments": arguments}
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(
            name=function_name,
            arguments=json.dumps(function_arguments),
        ),
        extra_content=None,
    )


def test_real_hermes_long_route_preserves_applied_publication_after_observation_handoff_failure() -> None:
    """An applied product publication owns terminalization after its observation is attempted."""

    from plane_runtime.hermes_adapter import HermesKernelAdapter
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
    snapshot_raw["contentDigest"] = _digest(  # type: ignore[assignment]
        "snapshot",
        {key: value for key, value in snapshot_raw.items() if key != "contentDigest"},
    )
    snapshot = G1RunSnapshot.from_dict(snapshot_raw)
    invocation = G1InvocationEnvelope.from_dict(make_invocation(snapshot.to_dict()))

    plan = [
        (
            "plane_operation",
            {
                "action": "read",
                "operationRef": "operation:catalog.search",
                "input": {"query": "work item", "limit": 8},
            },
        ),
        (
            "plane_operation",
            {
                "action": "read",
                "operationRef": "operation:catalog.describe",
                "input": {"operation_id": "work-item.read"},
            },
        ),
        (
            "plane_execute_typescript",
            {"typescript_source": "export default ({ input }) => ({ ok: true, input });"},
        ),
        (
            "plane_operation",
            {
                "action": "read",
                "operationRef": "operation:work-item.read",
                "input": {"workItemRef": "work-item:test"},
            },
        ),
        (
            "plane_operation",
            {
                "action": "mutate",
                "operationRef": "operation:work-item-update",
                "input": {"workItemRef": "work-item:test", "title": "updated"},
            },
        ),
        (
            "plane_operation",
            {
                "action": "read",
                "operationRef": "operation:work-item.read",
                "input": {"workItemRef": "work-item:test", "forbidden": True},
            },
        ),
        (
            "plane_operation",
            {
                "action": "read",
                "operationRef": "operation:catalog.describe",
                "input": {"operation_id": "agent.outcome.submit"},
            },
        ),
        (
            "plane_operation",
            {
                "action": "mutate",
                "operationRef": "operation:agent.outcome.submit",
                "input": {
                    "run_ref": "run:test",
                    "summary": "bounded outcome",
                    "artifacts": [],
                    "evidence": [],
                },
            },
        ),
        (
            "plane_publish",
            {
                "kind": "outcome",
                "operationRef": PLANE_OUTCOME_PUBLISH_OPERATION,
                "resourceRef": "outcome-submission:test",
                "content": "explicit outcome publication",
            },
        ),
    ]

    class Completions:
        def __init__(self) -> None:
            self.calls = 0

        def create(self, **_kwargs: object):
            if self.calls >= len(plan):
                raise AssertionError("Hermes made a provider call after terminal publication")
            name, arguments = plan[self.calls]
            self.calls += 1
            content = (
                "ordinary final transcript evidence"
                if self.calls == len(plan)
                else None
            )
            message = SimpleNamespace(
                content=content,
                tool_calls=[_tool_call(name, arguments, f"call-{self.calls}")],
                reasoning=None,
                reasoning_content=None,
                refusal=None,
            )
            return SimpleNamespace(
                choices=[SimpleNamespace(finish_reason="tool_calls", message=message)],
                usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            )

    class Client:
        def __init__(self) -> None:
            self.chat = SimpleNamespace(completions=Completions())

    class Credentials:
        def resolve(self, provider: str) -> dict[str, str]:
            del provider
            return {
                "api_key": "provider-test-marker",
                "base_url": "http://127.0.0.1",
                "api_mode": "chat_completions",
            }

    client = Client()

    def agent_factory(**kwargs: object) -> AIAgent:
        agent = AIAgent(**kwargs)
        agent._create_request_openai_client = lambda **_: client  # type: ignore[method-assign]
        return agent

    requests: list[dict] = []
    outcome_ref = "outcome-submission:test"

    def rpc(request: dict) -> dict:
        requests.append(request)
        operation_ref = request["operationRef"]
        if operation_ref == "operation:catalog.describe":
            operation_id = request["input"]["operation_id"]
            return _result(
                request,
                output={
                    "operation": {
                        "operationId": operation_id,
                        "operationRef": f"operation:{operation_id}",
                        "inputSchema": {"type": "object"},
                    }
                },
            )
        if request["action"] == "code":
            assert operation_ref == PLANE_CODE_MODE_EXECUTE_OPERATION
            return _result(request, output={"accepted": True})
        if (
            operation_ref == "operation:work-item.read"
            and request["input"].get("forbidden") is True
        ):
            return _result(
                request,
                status="denied",
                errorCode="NOT_AUTHORIZED",
                errorMessage="policy denied this read",
            )
        if operation_ref == "operation:agent.outcome.submit":
            return _result(
                request,
                output={
                    "result": {
                        "outcome": {"outcomeRef": outcome_ref}
                    }
                },
            )
        if request["action"] == "publish":
            return _result(
                request,
                output={"published": True},
                publication={
                    "action": "applied",
                    "productKind": "outcome_submission",
                    "productRef": outcome_ref,
                    "operationAttemptRef": "operation-attempt:test",
                    "operationRef": PLANE_OUTCOME_PUBLISH_OPERATION,
                    "applicationServiceRef": "application-service:agent-lifecycle",
                    "gatewayReceiptRef": "gateway-receipt:test",
                    "receiptRef": "receipt:test",
                    "auditReceiptRef": "audit-receipt:test",
                    "productEventRef": "product-event:test",
                },
            )
        return _result(request, output={"accepted": True})

    bodies: list[dict] = []

    def emit_body(body: dict) -> None:
        bodies.append(body)
        if body["kind"] == "outcome_submission_observed":
            raise RuntimeError("terminal observation handoff failed")

    with tempfile.TemporaryDirectory(prefix="hermes-plane-terminal-handoff-") as hermes_home_raw:
        hermes_home = Path(hermes_home_raw)
        with mock.patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}):
            import run_agent

            run_agent._hermes_home = hermes_home
            result = HermesKernelAdapter(
                agent_factory=agent_factory,
                credential_source=Credentials(),
                host_port=CallablePlaneHostPort(rpc),
            ).dispatch(
                snapshot,
                invocation,
                lambda: False,
                emit_body,
                model_call_allowance=12,
            )

    assert result.kind == "completed"
    assert result.output_text == "ordinary final transcript evidence"
    assert client.chat.completions.calls == len(plan)
    assert [request["action"] for request in requests].count("publish") == 1
    assert [request["operationRef"] for request in requests].count(
        PLANE_OUTCOME_PUBLISH_OPERATION
    ) == 1
    assert sum(body["kind"] == "outcome_submission_observed" for body in bodies) == 1
    assert sum(body["kind"] == "transcript_evidence_observed" for body in bodies) == 1
    assert not any(
        body.get("publication", {}).get("action") == "terminal"
        for body in bodies
    )
    assert "provider-test-marker" not in json.dumps(requests + bodies)


def test_post_terminal_code_mode_host_exception_preserves_applied_publication() -> None:
    """A late Code Mode host exception cannot overturn the applied terminal."""

    from plane_runtime import current_plane_host
    from plane_runtime.hermes_adapter import HermesKernelAdapter
    from plane_runtime.host_port import PlaneHostError
    from tests.plane_runtime.test_g1_runtime_process import (
        G1InvocationEnvelope,
        G1RunSnapshot,
        make_invocation,
        make_snapshot,
    )

    snapshot = G1RunSnapshot.from_dict(make_snapshot())
    invocation = G1InvocationEnvelope.from_dict(make_invocation(snapshot.to_dict()))
    host_requests: list[dict] = []
    model_calls = 0

    def rpc(request: dict) -> dict:
        host_requests.append(request)
        assert request["action"] == "publish"
        return _result(
            request,
            output={"published": True},
            publication={
                "action": "applied",
                "productKind": "outcome_submission",
                "productRef": "outcome-submission:test",
                "operationAttemptRef": "operation-attempt:test",
                "operationRef": PLANE_OUTCOME_PUBLISH_OPERATION,
                "applicationServiceRef": "application-service:agent-lifecycle",
                "gatewayReceiptRef": "gateway-receipt:test",
                "receiptRef": "receipt:test",
                "auditReceiptRef": "audit-receipt:test",
                "productEventRef": "product-event:test",
            },
        )

    class FakeAgent:
        session_input_tokens = 1
        session_output_tokens = 1
        session_api_calls = 0

        def __init__(self, **_kwargs: object) -> None:
            self._session_messages = [
                {
                    "role": "assistant",
                    "content": "ordinary final transcript evidence",
                    "tool_calls": [{"function": {"name": "plane_publish"}}],
                }
            ]

        def interrupt(self, _reason: str) -> None:
            return

        def run_conversation(self, *_args: object, **_kwargs: object) -> dict[str, object]:
            nonlocal model_calls
            model_calls += 1
            self.session_api_calls = model_calls
            binding = current_plane_host()
            assert binding is not None
            binding.publish(
                kind="outcome",
                operation_ref=PLANE_OUTCOME_PUBLISH_OPERATION,
                resource_ref="outcome-submission:test",
                content="bounded outcome",
            )
            try:
                # This is the late Code Mode host callback. The invalid
                # capsule is rejected at the binding seam after publication,
                # so no second host mutation can occur.
                binding.call(
                    action="code",
                    operation_ref=PLANE_CODE_MODE_EXECUTE_OPERATION,
                    input={"code": object()},
                    source="code",
                )
            except PlaneHostError:
                assert binding.fatal_error_after_terminal is True
                raise
            raise AssertionError("late host exception did not escape the fake agent")

    bodies: list[dict] = []
    result = HermesKernelAdapter(
        agent_factory=lambda **kwargs: FakeAgent(**kwargs),
        credential_source=type(
            "Credentials",
            (),
            {"resolve": lambda _self, _provider: {"api_key": "provider-test-marker"}},
        )(),
        host_port=CallablePlaneHostPort(rpc),
    ).dispatch(
        snapshot,
        invocation,
        lambda: False,
        bodies.append,
        model_call_allowance=2,
    )

    assert result.kind == "completed", (
        "event=terminal_handoff.post_terminal_exception actor=hermes "
        "operation=code_mode_host_callback risk=applied_outcome_relabeled "
        "expected=completed_product_outcome_published "
        f"actual={result.kind} suggestion=inspect_adapter_exception_path"
    )
    assert result.output_text == "ordinary final transcript evidence"
    assert result.model_calls == 1
    assert model_calls == 1
    assert len(host_requests) == 1
    assert host_requests[0]["action"] == "publish"
    assert sum(body["kind"] == "outcome_submission_observed" for body in bodies) == 1
    assert sum(
        body["kind"] == "progress_observed"
        and body["payload"]["text"]
        == "Hermes preserved the applied Plane terminal after a late host callback failure."
        for body in bodies
    ) == 1
    assert sum(body["kind"] == "transcript_evidence_observed" for body in bodies) == 1
    assert not any(body.get("publication", {}).get("action") == "terminal" for body in bodies)
    assert "provider-test-marker" not in json.dumps(host_requests + bodies)


def test_diagnostics_projection_failure_cannot_relabel_applied_publication() -> None:
    """Diagnostics are observation-only and must not turn a completed product into a runtime failure."""

    from plane_runtime import current_plane_host
    from plane_runtime.hermes_adapter import HermesKernelAdapter
    from tests.plane_runtime.test_g1_runtime_process import (
        G1InvocationEnvelope,
        G1RunSnapshot,
        make_invocation,
        make_snapshot,
    )

    snapshot = G1RunSnapshot.from_dict(make_snapshot())
    invocation = G1InvocationEnvelope.from_dict(make_invocation(snapshot.to_dict()))

    def rpc(request: dict) -> dict:
        return _result(
            request,
            output={"published": True},
            publication={
                "action": "applied",
                "productKind": "outcome_submission",
                "productRef": "outcome-submission:test",
                "operationAttemptRef": "operation-attempt:test",
                "operationRef": PLANE_OUTCOME_PUBLISH_OPERATION,
                "applicationServiceRef": "application-service:agent-lifecycle",
                "gatewayReceiptRef": "gateway-receipt:test",
                "receiptRef": "receipt:test",
                "auditReceiptRef": "audit-receipt:test",
                "productEventRef": "product-event:test",
            },
        )

    class FakeAgent:
        session_input_tokens = 1
        session_output_tokens = 1
        session_api_calls = 1

        def __init__(self, **_kwargs: object) -> None:
            self._session_messages = []

        def interrupt(self, _reason: str) -> None:
            return

        def run_conversation(self, *_args: object, **_kwargs: object) -> dict[str, object]:
            binding = current_plane_host()
            assert binding is not None
            binding.publish(
                kind="outcome",
                operation_ref=PLANE_OUTCOME_PUBLISH_OPERATION,
                resource_ref="outcome-submission:test",
                content="bounded outcome",
            )
            return {
                "final_response": "ordinary final transcript evidence",
                "failed": False,
                "turn_exit_reason": "terminal_action(product_outcome_published)",
            }

    bodies: list[dict] = []

    def emit_body(body: dict) -> None:
        bodies.append(body)
        if body.get("payload", {}).get("kind") == "runtime_diagnostics":
            raise RuntimeError("synthetic observation sink failure")

    result = HermesKernelAdapter(
        agent_factory=lambda **kwargs: FakeAgent(**kwargs),
        credential_source=type(
            "Credentials",
            (),
            {"resolve": lambda _self, _provider: {"api_key": "provider-test-marker"}},
        )(),
        host_port=CallablePlaneHostPort(rpc),
    ).dispatch(
        snapshot,
        invocation,
        lambda: False,
        emit_body,
        model_call_allowance=2,
    )

    assert result.kind == "completed"
    assert result.output_text == "ordinary final transcript evidence"
    assert sum(body["kind"] == "outcome_submission_observed" for body in bodies) == 1
    assert sum(body.get("payload", {}).get("kind") == "runtime_diagnostics" for body in bodies) == 1


def test_pre_terminal_code_mode_host_exception_remains_failed() -> None:
    """The terminal-preserving exception branch must not swallow pre-terminal errors."""

    from plane_runtime import current_plane_host
    from plane_runtime.hermes_adapter import HermesKernelAdapter
    from plane_runtime.host_port import PlaneHostError
    from tests.plane_runtime.test_g1_runtime_process import (
        G1InvocationEnvelope,
        G1RunSnapshot,
        make_invocation,
        make_snapshot,
    )

    snapshot = G1RunSnapshot.from_dict(make_snapshot())
    invocation = G1InvocationEnvelope.from_dict(make_invocation(snapshot.to_dict()))

    class FakeAgent:
        session_input_tokens = 0
        session_output_tokens = 0
        session_api_calls = 1

        def interrupt(self, _reason: str) -> None:
            return

        def run_conversation(self, *_args: object, **_kwargs: object) -> dict[str, object]:
            binding = current_plane_host()
            assert binding is not None
            try:
                binding.call(
                    action="code",
                    operation_ref=PLANE_CODE_MODE_EXECUTE_OPERATION,
                    input={"code": object()},
                    source="code",
                )
            except PlaneHostError:
                assert binding.fatal_error_after_terminal is False
                raise
            raise AssertionError("pre-terminal host exception did not escape the fake agent")

    result = HermesKernelAdapter(
        agent_factory=lambda **kwargs: FakeAgent(),
        credential_source=type(
            "Credentials",
            (),
            {"resolve": lambda _self, _provider: {"api_key": "provider-test-marker"}},
        )(),
        host_port=CallablePlaneHostPort(
            lambda _request: (_ for _ in ()).throw(AssertionError("host mutation was attempted"))
        ),
    ).dispatch(
        snapshot,
        invocation,
        lambda: False,
        lambda _body: None,
        model_call_allowance=1,
    )

    assert result.kind == "failed"
    assert result.failure_code == "runtime_error"
