"""The narrow runtime adapter and the deterministic kernel port.

The adapter is intentionally the only module that knows how to translate the
versioned Plane contract into a kernel request.  A future Hermes adapter can
implement :class:`KernelPort` without adding Plane vocabulary to Hermes core.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Protocol

from .contract import (
    ArtifactObserved,
    BindingError,
    ContractError,
    ConversationPublicationObserved,
    InputRequestObserved,
    ProgressObserved,
    PublicationReceipt,
    RuntimeBudget,
    RuntimeEvent,
    RuntimeExit,
    RuntimeFailure,
    RunSnapshot,
    SequenceError,
    TranscriptObserved,
    UsageObserved,
)
from .contract import InvocationEnvelope, PROTOCOL


EventSink = Callable[[RuntimeEvent], None]


class CancellationSignal(Protocol):
    """Invocation-scoped cancellation, not durable run state."""

    def is_cancelled(self) -> bool:
        ...


class RuntimeHost(Protocol):
    """Trusted host actions available to the adapter.

    The host returns a Plane receipt for an explicit publication action.  The
    adapter never treats a final transcript as publication by itself.
    """

    def publish_transcript(
        self,
        *,
        run_id: str,
        invocation_id: str,
        transcript: TranscriptObserved,
        idempotency_key: str,
    ) -> PublicationReceipt:
        ...


class KernelPort(Protocol):
    """Small seam for a replaceable execution kernel."""

    def dispatch(
        self,
        request: "KernelRequest",
        emit: Callable[["KernelObservation"], None],
        cancellation: CancellationSignal,
    ) -> "KernelResult":
        ...


@dataclass(frozen=True)
class KernelRequest:
    """Kernel-neutral request assembled from one immutable invocation."""

    run_id: str
    invocation_id: str
    objective: str
    behavioral_prompt: str
    context_refs: tuple[str, ...]
    new_context_event_refs: tuple[str, ...]
    checkpoint_ref: str | None
    remaining_budget: RuntimeBudget
    model: str
    route_ref: str
    trigger_kind: str


@dataclass(frozen=True)
class KernelObservation:
    """A bounded, kernel-neutral observation before adapter translation."""

    kind: str
    message: str | None = None
    payload: Mapping[str, object] = field(default_factory=dict)
    transcript_ref: str | None = None
    text: str | None = None
    request_ref: str | None = None
    artifact_ref: str | None = None
    artifact_digest: str | None = None
    usage: RuntimeBudget | None = None


@dataclass(frozen=True)
class KernelResult:
    terminal_kind: str
    failure: RuntimeFailure | None = None

    def __post_init__(self) -> None:
        if self.terminal_kind not in {"completed", "waiting_for_input", "failed", "blocked", "cancelled"}:
            raise ContractError(f"unsupported kernel exit kind: {self.terminal_kind!r}")
        if self.terminal_kind in {"failed", "blocked"} and self.failure is None:
            raise ContractError(f"{self.terminal_kind} kernel result requires failure details")
        if self.terminal_kind not in {"failed", "blocked"} and self.failure is not None:
            raise ContractError(f"{self.terminal_kind} kernel result cannot carry failure details")


class CancellationRequested(Exception):
    """Internal control flow for an invocation cancelled during dispatch."""


class NeverCancelled:
    def is_cancelled(self) -> bool:
        return False


class MutableCancellation:
    """Small test/host implementation of invocation cancellation."""

    def __init__(self) -> None:
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def is_cancelled(self) -> bool:
        return self._cancelled


class EventStream:
    """Validate and forward a bound, ordered, duplicate-safe event stream."""

    def __init__(
        self,
        *,
        run_id: str,
        invocation_id: str,
        sink: EventSink | None = None,
    ) -> None:
        self.run_id = run_id
        self.invocation_id = invocation_id
        self._sink = sink
        self._events: list[RuntimeEvent] = []
        self._seen: dict[str, RuntimeEvent] = {}

    @property
    def events(self) -> tuple[RuntimeEvent, ...]:
        return tuple(self._events)

    @property
    def last_sequence(self) -> int:
        return self._events[-1].sequence if self._events else 0

    def accept(self, event: RuntimeEvent) -> bool:
        if event.run_id != self.run_id or event.invocation_id != self.invocation_id:
            raise BindingError(
                f"event {event.event_id!r} is bound to {event.run_id}/{event.invocation_id}, "
                f"expected {self.run_id}/{self.invocation_id}"
            )
        previous = self._seen.get(event.event_id)
        if previous is not None:
            if previous == event:
                return False
            raise SequenceError(f"event id {event.event_id!r} was reused with different content")
        if event.sequence != self.last_sequence + 1:
            raise SequenceError(
                f"expected sequence {self.last_sequence + 1}, received {event.sequence}"
            )
        if self._sink is not None:
            self._sink(event)
        self._seen[event.event_id] = event
        self._events.append(event)
        return True


class EventCollector(EventStream):
    """An event sink that also retains the validated observations."""

    def emit(self, event: RuntimeEvent) -> bool:
        return self.accept(event)


def _next_event(
    stream: EventStream,
    invocation: InvocationEnvelope,
    body: object,
) -> RuntimeEvent:
    event_id = f"{invocation.invocation_id}:event:{stream.last_sequence + 1}"
    event = RuntimeEvent(
        protocol=PROTOCOL,
        run_id=invocation.run_id,
        invocation_id=invocation.invocation_id,
        sequence=stream.last_sequence + 1,
        event_id=event_id,
        correlation_ref=invocation.causation_ref,
        idempotency_key=event_id,
        body=body,  # type: ignore[arg-type]
    )
    stream.accept(event)
    return event


def _budget_add(left: RuntimeBudget, right: RuntimeBudget) -> RuntimeBudget:
    return RuntimeBudget(
        iterations=left.iterations + right.iterations,
        input_tokens=left.input_tokens + right.input_tokens,
        output_tokens=left.output_tokens + right.output_tokens,
    )


def _validate_inputs(run: RunSnapshot, invocation: InvocationEnvelope) -> None:
    if invocation.protocol != PROTOCOL or run.protocol != PROTOCOL:
        raise ContractError("runtime contract protocol mismatch")
    if invocation.run_id != run.run_id:
        raise BindingError(
            f"invocation {invocation.invocation_id!r} belongs to {invocation.run_id!r}, expected {run.run_id!r}"
        )
    if invocation.run_snapshot_digest != run.digest():
        raise BindingError("invocation does not reference the supplied immutable run snapshot")
    if not invocation.remaining_budget.within(run.total_budget_policy.total):
        raise ContractError("invocation remaining budget exceeds the run total budget")


def execute(
    *,
    run: RunSnapshot,
    invocation: InvocationEnvelope,
    host: RuntimeHost | None,
    emit: EventSink | None = None,
    cancellation: CancellationSignal | None = None,
    kernel: KernelPort | None = None,
) -> RuntimeExit:
    """Execute exactly one invocation through the replaceable kernel port.

    The adapter validates binding and cumulative budget at entry, translates
    kernel-neutral observations into bounded runtime events, and returns one
    immutable terminal exit.  It does not persist state or mutate ``run``.
    """

    _validate_inputs(run, invocation)
    cancellation = cancellation or NeverCancelled()
    stream = EventStream(run_id=run.run_id, invocation_id=invocation.invocation_id, sink=emit)
    if cancellation.is_cancelled():
        return RuntimeExit(kind="cancelled", final_sequence=stream.last_sequence)
    kernel = kernel or FakeKernel()
    request = KernelRequest(
        run_id=run.run_id,
        invocation_id=invocation.invocation_id,
        objective=run.assignment.objective,
        behavioral_prompt=run.behavioral_prompt,
        context_refs=tuple(item.ref for item in run.context),
        new_context_event_refs=invocation.new_context_event_refs,
        checkpoint_ref=invocation.checkpoint_ref,
        remaining_budget=invocation.remaining_budget,
        model=run.model.model,
        route_ref=run.model.route_ref,
        trigger_kind=invocation.trigger.kind,
    )
    transcripts: dict[str, TranscriptObserved] = {}
    used = RuntimeBudget()

    def on_observation(observation: KernelObservation) -> None:
        nonlocal used
        if cancellation.is_cancelled():
            raise CancellationRequested
        if observation.kind == "progress":
            if observation.message is None:
                raise ContractError("progress observation requires a message")
            _next_event(stream, invocation, ProgressObserved(observation.message, observation.payload))
            return
        if observation.kind == "usage":
            if observation.usage is None:
                raise ContractError("usage observation requires usage")
            used = _budget_add(used, observation.usage)
            if not used.within(invocation.remaining_budget):
                raise ContractError("kernel usage exceeds invocation remaining budget")
            _next_event(stream, invocation, UsageObserved(observation.usage))
            return
        if observation.kind == "transcript":
            if observation.transcript_ref is None or observation.text is None:
                raise ContractError("transcript observation requires a reference and text")
            transcript = TranscriptObserved(observation.transcript_ref, observation.text)
            transcripts[transcript.transcript_ref] = transcript
            _next_event(stream, invocation, transcript)
            return
        if observation.kind == "input_request":
            if observation.request_ref is None or observation.message is None:
                raise ContractError("input request requires a reference and prompt")
            _next_event(stream, invocation, InputRequestObserved(observation.request_ref, observation.message))
            return
        if observation.kind == "artifact":
            if observation.artifact_ref is None or observation.artifact_digest is None:
                raise ContractError("artifact observation requires a reference and digest")
            _next_event(
                stream,
                invocation,
                ArtifactObserved(observation.artifact_ref, observation.artifact_digest),
            )
            return
        if observation.kind == "publication_request":
            if host is None:
                raise ContractError("explicit publication requires a trusted runtime host")
            if observation.transcript_ref is None:
                raise ContractError("publication request requires a transcript reference")
            try:
                transcript = transcripts[observation.transcript_ref]
            except KeyError as exc:
                raise ContractError("publication request must reference prior transcript evidence") from exc
            event_id = f"{invocation.invocation_id}:publication:{observation.transcript_ref}"
            receipt = host.publish_transcript(
                run_id=run.run_id,
                invocation_id=invocation.invocation_id,
                transcript=transcript,
                idempotency_key=event_id,
            )
            if receipt.transcript_ref != transcript.transcript_ref:
                raise BindingError("publication receipt references a different transcript")
            _next_event(
                stream,
                invocation,
                ConversationPublicationObserved(
                    transcript_ref=transcript.transcript_ref,
                    publication_ref=receipt.publication_ref,
                    receipt_ref=receipt.receipt_ref,
                ),
            )
            return
        raise ContractError(f"unsupported kernel observation: {observation.kind!r}")

    try:
        result = kernel.dispatch(request, on_observation, cancellation)
    except CancellationRequested:
        return RuntimeExit(kind="cancelled", final_sequence=stream.last_sequence)
    if cancellation.is_cancelled():
        return RuntimeExit(kind="cancelled", final_sequence=stream.last_sequence)
    return RuntimeExit(
        kind=result.terminal_kind,
        final_sequence=stream.last_sequence,
        failure=result.failure,
    )


@dataclass(frozen=True)
class FakeKernelPlan:
    """Deterministic fake behavior used by contract and service tests."""

    transcript: str = "deterministic fake transcript"
    transcript_ref: str = "transcript:fake"
    usage: RuntimeBudget = field(default_factory=lambda: RuntimeBudget(iterations=1, output_tokens=4))
    terminal_kind: str = "completed"
    input_request: str | None = None
    input_request_ref: str = "input:fake"
    publication_requested: bool = False


class FakeKernel:
    """A stateless, deterministic KernelPort implementation."""

    def __init__(self, plan: FakeKernelPlan | None = None, *, on_step: Callable[[], None] | None = None) -> None:
        self.plan = plan or FakeKernelPlan()
        self.on_step = on_step
        self.requests: list[KernelRequest] = []

    def dispatch(
        self,
        request: KernelRequest,
        emit: Callable[[KernelObservation], None],
        cancellation: CancellationSignal,
    ) -> KernelResult:
        self.requests.append(request)
        observations = (
            KernelObservation(kind="progress", message="fake kernel dispatched"),
            KernelObservation(kind="usage", usage=self.plan.usage),
            KernelObservation(
                kind="transcript",
                transcript_ref=self.plan.transcript_ref,
                text=self.plan.transcript,
            ),
        )
        for observation in observations:
            if cancellation.is_cancelled():
                return KernelResult(terminal_kind="cancelled")
            emit(observation)
            if self.on_step is not None:
                self.on_step()
        if self.plan.input_request is not None:
            emit(
                KernelObservation(
                    kind="input_request",
                    request_ref=self.plan.input_request_ref,
                    message=self.plan.input_request,
                )
            )
        if self.plan.publication_requested:
            emit(KernelObservation(kind="publication_request", transcript_ref=self.plan.transcript_ref))
        if cancellation.is_cancelled():
            return KernelResult(terminal_kind="cancelled")
        return KernelResult(terminal_kind=self.plan.terminal_kind)


class RecordingHost:
    """Deterministic trusted host used by the fake service and tests."""

    def __init__(self) -> None:
        self.publications: list[tuple[str, str, str]] = []

    def publish_transcript(
        self,
        *,
        run_id: str,
        invocation_id: str,
        transcript: TranscriptObserved,
        idempotency_key: str,
    ) -> PublicationReceipt:
        self.publications.append((run_id, invocation_id, idempotency_key))
        return PublicationReceipt(
            publication_ref=f"publication:{transcript.transcript_ref}",
            receipt_ref=f"receipt:{idempotency_key}",
            transcript_ref=transcript.transcript_ref,
        )


def classify_process_death(
    *,
    final_sequence: int,
    reason: str = "runtime process exited before returning an exit",
) -> RuntimeExit:
    """Supervisor evidence for a dead replaceable process/container."""

    return RuntimeExit(
        kind="failed",
        final_sequence=final_sequence,
        failure=RuntimeFailure(code="process_died", message=reason, retryable=True),
    )
