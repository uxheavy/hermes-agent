"""Focused contract, adapter, and process-boundary proof for plane_runtime."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from plane_runtime import (
    PROTOCOL,
    AssignmentSnapshot,
    BindingError,
    BoundsError,
    ContractDigests,
    ContractError,
    EventCollector,
    FakeKernel,
    FakeKernelPlan,
    InvocationEnvelope,
    InvocationTrigger,
    MutableCancellation,
    OperationDescriptor,
    ProgressObserved,
    RecordingHost,
    RuntimeBudget,
    RuntimeBudgetPolicy,
    RuntimeEvent,
    RuntimeExit,
    RuntimeFailure,
    RuntimeLease,
    RuntimeModelRoute,
    RunSnapshot,
    SequenceError,
    ToolPresentation,
    TranscriptObserved,
    UsageObserved,
    VersionedContextRef,
    classify_process_death,
    execute,
)


def make_snapshot(*, total: RuntimeBudget | None = None) -> RunSnapshot:
    return RunSnapshot(
        protocol=PROTOCOL,
        run_id="run:one",
        assignment=AssignmentSnapshot(
            version="assignment:v1",
            target_ref="issue:one",
            objective="Produce the assigned result",
            acceptance_criteria=("The result is deterministic",),
        ),
        actor_ref="agent:one",
        profile_version="profile:v1",
        behavioral_prompt="Use the runtime port.",
        context=(VersionedContextRef("context:one", "sha256:context"),),
        tool_presentation=ToolPresentation(
            eager_operations=(OperationDescriptor("operation:read", "sha256:operation"),),
            catalog_digest="sha256:catalog",
        ),
        model=RuntimeModelRoute(model="fake-model", route_ref="route:fake"),
        total_budget_policy=RuntimeBudgetPolicy(total or RuntimeBudget(5, 100, 100)),
        contract_digests=ContractDigests("snapshot:v1", "invocation:v1", "event:v1", "exit:v1"),
    )


def make_invocation(
    snapshot: RunSnapshot,
    *,
    invocation_id: str = "invocation:one",
    trigger: InvocationTrigger | None = None,
    remaining: RuntimeBudget | None = None,
    checkpoint_ref: str | None = None,
    context_refs: tuple[str, ...] = (),
) -> InvocationEnvelope:
    return InvocationEnvelope(
        protocol=PROTOCOL,
        invocation_id=invocation_id,
        run_id=snapshot.run_id,
        run_snapshot_digest=snapshot.digest(),
        trigger=trigger or InvocationTrigger("initial"),
        new_context_event_refs=context_refs,
        checkpoint_ref=checkpoint_ref,
        remaining_budget=remaining or snapshot.total_budget_policy.total,
        lease=RuntimeLease("lease:one", "host:one", "2099-01-01T00:00:00Z"),
        causation_ref="cause:one",
        cancellation_ref="cancel:one",
    )


def event(*, sequence: int, event_id: str = "event:one", run_id: str = "run:one") -> RuntimeEvent:
    return RuntimeEvent(
        protocol=PROTOCOL,
        run_id=run_id,
        invocation_id="invocation:one",
        sequence=sequence,
        event_id=event_id,
        correlation_ref="cause:one",
        idempotency_key=event_id,
        body=ProgressObserved("observed"),
    )


def invoke_service(request: dict[str, object]) -> list[dict[str, object]]:
    completed = subprocess.run(
        [sys.executable, "-m", "plane_runtime.service", "--once"],
        input=json.dumps(request) + "\n",
        text=True,
        capture_output=True,
        check=True,
    )
    return [json.loads(line) for line in completed.stdout.splitlines()]


def test_run_snapshot_is_deeply_immutable() -> None:
    criteria = ["one"]
    context = [VersionedContextRef("context:one", "digest")]
    snapshot = make_snapshot()
    snapshot = RunSnapshot(
        protocol=snapshot.protocol,
        run_id=snapshot.run_id,
        assignment=AssignmentSnapshot("v1", "target", "objective", criteria),
        actor_ref=snapshot.actor_ref,
        profile_version=snapshot.profile_version,
        behavioral_prompt=snapshot.behavioral_prompt,
        context=context,
        tool_presentation=snapshot.tool_presentation,
        model=snapshot.model,
        total_budget_policy=snapshot.total_budget_policy,
        contract_digests=snapshot.contract_digests,
    )

    criteria.append("mutated outside")
    context.append(VersionedContextRef("context:two", "digest"))
    assert snapshot.assignment.acceptance_criteria == ("one",)
    assert snapshot.context == (VersionedContextRef("context:one", "digest"),)
    with pytest.raises((AttributeError, TypeError)):
        snapshot.run_id = "mutated"  # type: ignore[misc]
    wire = snapshot.to_dict()
    wire["assignment"]["acceptanceCriteria"].append("mutated wire")
    assert snapshot.assignment.acceptance_criteria == ("one",)


def test_contracts_round_trip_through_json() -> None:
    snapshot = make_snapshot()
    invocation = make_invocation(snapshot)
    transcript_event = RuntimeEvent(
        PROTOCOL,
        snapshot.run_id,
        invocation.invocation_id,
        1,
        "event:transcript",
        invocation.causation_ref,
        "event:transcript",
        TranscriptObserved("transcript:one", "final text is evidence"),
    )
    exit_value = RuntimeExit(
        "failed",
        1,
        RuntimeFailure("process_died", "the replaceable process exited", True),
    )

    assert RunSnapshot.from_json(snapshot.to_json()) == snapshot
    assert InvocationEnvelope.from_json(invocation.to_json()) == invocation
    assert RuntimeEvent.from_json(transcript_event.to_json()) == transcript_event
    assert RuntimeExit.from_json(exit_value.to_json()) == exit_value


def test_later_input_is_an_event_reference_and_does_not_mutate_snapshot() -> None:
    snapshot = make_snapshot()
    before = snapshot.to_json()
    invocation = make_invocation(
        snapshot,
        trigger=InvocationTrigger("human_input", "event:answer"),
        context_refs=("event:answer",),
        remaining=RuntimeBudget(4, 90, 90),
    )
    kernel = FakeKernel()
    execute(run=snapshot, invocation=invocation, host=RecordingHost(), kernel=kernel)

    assert snapshot.to_json() == before
    assert invocation.trigger.event_ref == "event:answer"
    assert "answer content" not in invocation.to_json()
    assert kernel.requests[0].new_context_event_refs == ("event:answer",)


def test_remaining_budget_is_cumulative_across_invocations() -> None:
    snapshot = make_snapshot(total=RuntimeBudget(5, 100, 100))
    first = make_invocation(snapshot, remaining=RuntimeBudget(5, 100, 100))
    first_kernel = FakeKernel(FakeKernelPlan(usage=RuntimeBudget(2, 10, 20)))
    assert execute(run=snapshot, invocation=first, host=RecordingHost(), kernel=first_kernel).kind == "completed"

    second = make_invocation(
        snapshot,
        invocation_id="invocation:two",
        trigger=InvocationTrigger("continuation", "event:continuation"),
        remaining=RuntimeBudget(3, 90, 80),
    )
    second_kernel = FakeKernel(FakeKernelPlan(usage=RuntimeBudget(3, 90, 80)))
    assert execute(run=snapshot, invocation=second, host=RecordingHost(), kernel=second_kernel).kind == "completed"
    assert second_kernel.requests[0].remaining_budget == RuntimeBudget(3, 90, 80)
    with pytest.raises(ContractError):
        execute(
            run=snapshot,
            invocation=make_invocation(snapshot, remaining=RuntimeBudget(6, 100, 100)),
            host=RecordingHost(),
        )


def test_adapter_rejects_binding_mismatch() -> None:
    snapshot = make_snapshot()
    wrong_run = replace(make_invocation(snapshot), run_id="run:other")
    with pytest.raises(BindingError):
        execute(run=snapshot, invocation=wrong_run, host=RecordingHost())

    collector = EventCollector(run_id=snapshot.run_id, invocation_id="invocation:one")
    with pytest.raises(BindingError):
        collector.accept(event(sequence=1, run_id="run:other"))


def test_event_stream_is_duplicate_safe_but_strictly_ordered() -> None:
    collector = EventCollector(run_id="run:one", invocation_id="invocation:one")
    first = event(sequence=1)
    assert collector.accept(first) is True
    assert collector.accept(first) is False
    with pytest.raises(SequenceError):
        collector.accept(event(sequence=3, event_id="event:three"))
    with pytest.raises(SequenceError):
        collector.accept(event(sequence=2, event_id="event:one"))


def test_cancellation_is_invocation_scoped() -> None:
    cancellation = MutableCancellation()
    cancellation.cancel()
    snapshot = make_snapshot()
    events: list[RuntimeEvent] = []
    exit_value = execute(
        run=snapshot,
        invocation=make_invocation(snapshot),
        host=RecordingHost(),
        emit=events.append,
        cancellation=cancellation,
    )
    assert exit_value == RuntimeExit("cancelled", 0)
    assert events == []

    cancellation = MutableCancellation()
    kernel = FakeKernel(on_step=cancellation.cancel)
    exit_value = execute(
        run=snapshot,
        invocation=make_invocation(snapshot, invocation_id="invocation:cancelled"),
        host=RecordingHost(),
        cancellation=cancellation,
        kernel=kernel,
    )
    assert exit_value.kind == "cancelled"
    assert exit_value.final_sequence == 1


def test_checkpoint_continuation_is_passed_without_kernel_durable_state() -> None:
    snapshot = make_snapshot()
    invocation = make_invocation(
        snapshot,
        trigger=InvocationTrigger("recoverable_restart", "event:restart"),
        checkpoint_ref="checkpoint:one",
        context_refs=("event:restart",),
        remaining=RuntimeBudget(3, 80, 80),
    )
    kernel = FakeKernel()
    execute(run=snapshot, invocation=invocation, host=RecordingHost(), kernel=kernel)
    request = kernel.requests[0]
    assert request.checkpoint_ref == "checkpoint:one"
    assert request.trigger_kind == "recoverable_restart"
    assert request.new_context_event_refs == ("event:restart",)


def test_transcript_is_not_published_without_explicit_action() -> None:
    host = RecordingHost()
    snapshot = make_snapshot()
    events: list[RuntimeEvent] = []
    exit_value = execute(
        run=snapshot,
        invocation=make_invocation(snapshot),
        host=host,
        emit=events.append,
        kernel=FakeKernel(FakeKernelPlan(publication_requested=False)),
    )
    assert exit_value.kind == "completed"
    assert any(isinstance(item.body, TranscriptObserved) for item in events)
    assert not any(item.body.to_dict()["kind"] == "conversation_publication" for item in events)
    assert host.publications == []


def test_explicit_publication_requires_and_correlates_a_receipt() -> None:
    snapshot = make_snapshot()
    host = RecordingHost()
    events: list[RuntimeEvent] = []
    exit_value = execute(
        run=snapshot,
        invocation=make_invocation(snapshot),
        host=host,
        emit=events.append,
        kernel=FakeKernel(FakeKernelPlan(publication_requested=True)),
    )
    assert exit_value.kind == "completed"
    assert len(host.publications) == 1
    publication = [item for item in events if item.body.to_dict()["kind"] == "conversation_publication"]
    assert len(publication) == 1
    assert publication[0].body.to_dict()["receiptRef"].startswith("receipt:")

    with pytest.raises(ContractError):
        execute(
            run=snapshot,
            invocation=make_invocation(snapshot, invocation_id="invocation:no-host"),
            host=None,
            kernel=FakeKernel(FakeKernelPlan(publication_requested=True)),
        )


def test_payloads_are_bounded_at_the_event_boundary() -> None:
    with pytest.raises(BoundsError):
        RuntimeEvent(
            PROTOCOL,
            "run:one",
            "invocation:one",
            1,
            "event:large",
            "cause:one",
            "event:large",
            TranscriptObserved("transcript:large", "x" * 40_000),
        )


def test_replaced_processes_round_trip_serialized_invocations() -> None:
    snapshot = make_snapshot()
    first = make_invocation(snapshot, remaining=RuntimeBudget(5, 100, 100))
    first_lines = invoke_service(
        {
            "run": snapshot.to_dict(),
            "invocation": first.to_dict(),
            "fakePlan": {"terminalKind": "waiting_for_input", "inputRequest": "Need one answer"},
        }
    )
    assert [line["type"] for line in first_lines].count("exit") == 1
    assert first_lines[-1]["exit"]["kind"] == "waiting_for_input"  # type: ignore[index]
    first_events = [
        RuntimeEvent.from_dict(line["event"])
        for line in first_lines
        if line["type"] == "event"
    ]
    first_stream = EventCollector(run_id=snapshot.run_id, invocation_id=first.invocation_id)
    for item in first_events:
        first_stream.accept(item)
    assert RuntimeExit.from_dict(first_lines[-1]["exit"]).kind == "waiting_for_input"  # type: ignore[arg-type,index]

    second = make_invocation(
        snapshot,
        invocation_id="invocation:replacement",
        trigger=InvocationTrigger("continuation", "event:answer"),
        context_refs=("event:answer",),
        checkpoint_ref="checkpoint:one",
        remaining=RuntimeBudget(4, 90, 90),
    )
    second_lines = invoke_service(
        {
            "run": snapshot.to_dict(),
            "invocation": second.to_dict(),
            "fakePlan": {"terminalKind": "completed", "transcript": "continued"},
        }
    )
    assert [line["type"] for line in second_lines].count("exit") == 1
    assert second_lines[-1]["exit"]["kind"] == "completed"  # type: ignore[index]
    assert sum(line["type"] == "event" for line in second_lines) >= 3


def test_supervisor_classifies_process_death_as_one_terminal_exit() -> None:
    exit_value = classify_process_death(final_sequence=2)
    assert exit_value.kind == "failed"
    assert exit_value.failure is not None
    assert exit_value.failure.code == "process_died"
    assert RuntimeExit.from_json(exit_value.to_json()) == exit_value


def test_runtime_package_has_no_product_or_network_imports() -> None:
    package = Path(__file__).parents[2] / "plane_runtime"
    forbidden = {"run_agent", "model_tools", "gateway", "hermes_state", "requests", "httpx"}
    for path in package.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module.split(".")[0]]
            else:
                continue
            assert not forbidden.intersection(names), f"forbidden import in {path}: {names}"
