"""The narrow runtime adapter and the deterministic kernel port.

The adapter is intentionally the only module that knows how to translate the
versioned Plane contract into a kernel request.  A future Hermes adapter can
implement :class:`KernelPort` without adding Plane vocabulary to Hermes core.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from threading import Event
from typing import Callable, Mapping, Protocol

from .contract import (
    ArtifactObserved,
    BindingError,
    ContractError,
    ConversationPublicationObserved,
    InputRequestObserved,
    LeaseError,
    ProgressObserved,
    ProductReceipt,
    PublicationReceipt,
    RuntimeBudget,
    RuntimeEvent,
    RuntimeExit,
    RuntimeFailure,
    RunSnapshot,
    SequenceError,
    TranscriptObserved,
    UsageObserved,
    parse_utc_timestamp,
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

    def request_input(
        self,
        *,
        run_id: str,
        invocation_id: str,
        request_ref: str,
        prompt: str,
        idempotency_key: str,
    ) -> ProductReceipt:
        ...

    def record_artifact(
        self,
        *,
        run_id: str,
        invocation_id: str,
        artifact_ref: str,
        digest: str,
        idempotency_key: str,
    ) -> ProductReceipt:
        ...


class RuntimeSupervisor(Protocol):
    """Trusted host seam for lease and checkpoint authority."""

    def validate_lease(self, *, run: RunSnapshot, invocation: InvocationEnvelope) -> None:
        ...

    def validate_checkpoint(self, *, run: RunSnapshot, invocation: InvocationEnvelope) -> None:
        ...


class TrustedClock(Protocol):
    def now(self) -> datetime:
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
        expected_correlation_ref: str | None = None,
        expected_causation_ref: str | None = None,
    ) -> None:
        if (
            expected_correlation_ref is not None
            and expected_causation_ref is not None
            and expected_correlation_ref != expected_causation_ref
        ):
            raise BindingError("expected correlation and causation references disagree")
        self.run_id = run_id
        self.invocation_id = invocation_id
        self._sink = sink
        self.expected_correlation_ref = expected_correlation_ref or expected_causation_ref
        self._events: list[RuntimeEvent] = []
        self._seen: dict[str, RuntimeEvent] = {}
        self._seen_idempotency: dict[str, RuntimeEvent] = {}

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
        if (
            self.expected_correlation_ref is not None
            and event.correlation_ref != self.expected_correlation_ref
        ):
            raise BindingError(
                f"event {event.event_id!r} has correlation {event.correlation_ref!r}, "
                f"expected {self.expected_correlation_ref!r}"
            )
        previous = self._seen.get(event.event_id)
        if previous is not None:
            if previous == event:
                return False
            raise SequenceError(f"event id {event.event_id!r} was reused with different content")
        previous_idempotency = self._seen_idempotency.get(event.idempotency_key)
        if previous_idempotency is not None:
            raise SequenceError(
                f"idempotency key {event.idempotency_key!r} was reused by "
                f"event {previous_idempotency.event_id!r}"
            )
        if event.sequence != self.last_sequence + 1:
            raise SequenceError(
                f"expected sequence {self.last_sequence + 1}, received {event.sequence}"
            )
        if self._sink is not None:
            self._sink(event)
        self._seen[event.event_id] = event
        self._seen_idempotency[event.idempotency_key] = event
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


def _trusted_now(clock: TrustedClock | Callable[[], datetime]) -> datetime:
    value = clock.now() if hasattr(clock, "now") else clock()
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise LeaseError("trusted clock must return a timezone-aware datetime")
    if value.utcoffset() != timedelta(0):
        raise LeaseError("trusted clock must return UTC time")
    value = value.astimezone(timezone.utc)
    return value


def _validate_lease_with_clock(invocation: InvocationEnvelope, clock: TrustedClock | Callable[[], datetime]) -> None:
    expires_at = parse_utc_timestamp(invocation.lease.expires_at, "lease.expiresAt")
    if expires_at <= _trusted_now(clock):
        raise LeaseError("invocation lease is expired")


def _validate_publication_receipt(
    receipt: PublicationReceipt,
    *,
    run_id: str,
    invocation_id: str,
    transcript_ref: str,
    idempotency_key: str,
) -> None:
    if not isinstance(receipt, PublicationReceipt):
        raise ContractError("publication operation returned an invalid receipt")
    if receipt.transcript_ref != transcript_ref:
        raise BindingError("publication receipt references a different transcript")
    if (
        receipt.run_id != run_id
        or receipt.invocation_id != invocation_id
        or receipt.idempotency_key != idempotency_key
    ):
        raise BindingError("publication receipt is not bound to this invocation")


def _validate_product_receipt(
    receipt: ProductReceipt,
    *,
    run_id: str,
    invocation_id: str,
    resource_ref: str,
    idempotency_key: str,
) -> None:
    if not isinstance(receipt, ProductReceipt):
        raise ContractError("product operation returned an invalid receipt")
    if (
        receipt.run_id != run_id
        or receipt.invocation_id != invocation_id
        or receipt.resource_ref != resource_ref
        or receipt.idempotency_key != idempotency_key
    ):
        raise BindingError("product receipt is not bound to this invocation and resource")


def execute(
    *,
    run: RunSnapshot,
    invocation: InvocationEnvelope,
    host: RuntimeHost | None,
    emit: EventSink | None = None,
    cancellation: CancellationSignal | None = None,
    kernel: KernelPort | None = None,
    clock: TrustedClock | Callable[[], datetime] | None = None,
    supervisor: RuntimeSupervisor | None = None,
) -> RuntimeExit:
    """Execute exactly one invocation through the replaceable kernel port.

    The adapter validates binding and cumulative budget at entry, translates
    kernel-neutral observations into bounded runtime events, and returns one
    immutable terminal exit.  It does not persist state or mutate ``run``.
    """

    _validate_inputs(run, invocation)
    if supervisor is not None:
        supervisor.validate_lease(run=run, invocation=invocation)
        if invocation.checkpoint_ref is not None:
            supervisor.validate_checkpoint(run=run, invocation=invocation)
    elif clock is not None:
        _validate_lease_with_clock(invocation, clock)
        if invocation.checkpoint_ref is not None:
            raise ContractError("checkpoint continuation requires a trusted supervisor")
    else:
        raise LeaseError("execution requires an injected trusted clock or supervisor")
    cancellation = cancellation or NeverCancelled()
    stream = EventStream(
        run_id=run.run_id,
        invocation_id=invocation.invocation_id,
        sink=emit,
        expected_causation_ref=invocation.causation_ref,
    )
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
    pending_input_requests: set[str] = set()
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
            if host is None:
                raise ContractError("input requests require a trusted runtime host")
            if observation.request_ref is None or observation.message is None:
                raise ContractError("input request requires a reference and prompt")
            if observation.request_ref in pending_input_requests:
                raise SequenceError("input request reference was reused in one invocation")
            event_id = f"{invocation.invocation_id}:input:{observation.request_ref}"
            receipt = host.request_input(
                run_id=run.run_id,
                invocation_id=invocation.invocation_id,
                request_ref=observation.request_ref,
                prompt=observation.message,
                idempotency_key=event_id,
            )
            _validate_product_receipt(
                receipt,
                run_id=run.run_id,
                invocation_id=invocation.invocation_id,
                resource_ref=observation.request_ref,
                idempotency_key=event_id,
            )
            pending_input_requests.add(observation.request_ref)
            _next_event(
                stream,
                invocation,
                InputRequestObserved(observation.request_ref, observation.message, receipt.receipt_ref),
            )
            return
        if observation.kind == "artifact":
            if host is None:
                raise ContractError("artifacts require a trusted runtime host")
            if observation.artifact_ref is None or observation.artifact_digest is None:
                raise ContractError("artifact observation requires a reference and digest")
            event_id = f"{invocation.invocation_id}:artifact:{observation.artifact_ref}"
            receipt = host.record_artifact(
                run_id=run.run_id,
                invocation_id=invocation.invocation_id,
                artifact_ref=observation.artifact_ref,
                digest=observation.artifact_digest,
                idempotency_key=event_id,
            )
            _validate_product_receipt(
                receipt,
                run_id=run.run_id,
                invocation_id=invocation.invocation_id,
                resource_ref=observation.artifact_ref,
                idempotency_key=event_id,
            )
            _next_event(
                stream,
                invocation,
                ArtifactObserved(observation.artifact_ref, observation.artifact_digest, receipt.receipt_ref),
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
            _validate_publication_receipt(
                receipt,
                run_id=run.run_id,
                invocation_id=invocation.invocation_id,
                transcript_ref=transcript.transcript_ref,
                idempotency_key=event_id,
            )
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
    if result.terminal_kind == "waiting_for_input" and not pending_input_requests:
        raise ContractError("waiting_for_input requires a visible authorized input request")
    if result.terminal_kind != "waiting_for_input" and pending_input_requests:
        raise ContractError("terminal exit cannot leave an unresolved input request")
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
    hold_after_observations: int | None = None


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
        for index, observation in enumerate(observations, start=1):
            if cancellation.is_cancelled():
                return KernelResult(terminal_kind="cancelled")
            emit(observation)
            if self.on_step is not None:
                self.on_step()
            if self.plan.hold_after_observations == index:
                Event().wait()
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
        self.input_requests: list[tuple[str, str, str]] = []
        self.artifacts: list[tuple[str, str, str]] = []
        self._receipts: dict[str, PublicationReceipt | ProductReceipt] = {}

    def publish_transcript(
        self,
        *,
        run_id: str,
        invocation_id: str,
        transcript: TranscriptObserved,
        idempotency_key: str,
    ) -> PublicationReceipt:
        prior = self._receipts.get(idempotency_key)
        if prior is not None:
            if not isinstance(prior, PublicationReceipt) or prior.transcript_ref != transcript.transcript_ref:
                raise BindingError("publication idempotency key was reused for another operation")
            return prior
        self.publications.append((run_id, invocation_id, idempotency_key))
        receipt = PublicationReceipt(
            publication_ref=f"publication:{transcript.transcript_ref}",
            receipt_ref=f"receipt:{idempotency_key}",
            transcript_ref=transcript.transcript_ref,
            run_id=run_id,
            invocation_id=invocation_id,
            idempotency_key=idempotency_key,
        )
        self._receipts[idempotency_key] = receipt
        return receipt

    def request_input(
        self,
        *,
        run_id: str,
        invocation_id: str,
        request_ref: str,
        prompt: str,
        idempotency_key: str,
    ) -> ProductReceipt:
        prior = self._receipts.get(idempotency_key)
        if prior is not None:
            if not isinstance(prior, ProductReceipt) or prior.resource_ref != request_ref:
                raise BindingError("input idempotency key was reused for another operation")
            return prior
        self.input_requests.append((run_id, request_ref, idempotency_key))
        receipt = ProductReceipt(
            resource_ref=request_ref,
            receipt_ref=f"receipt:{idempotency_key}",
            run_id=run_id,
            invocation_id=invocation_id,
            idempotency_key=idempotency_key,
        )
        self._receipts[idempotency_key] = receipt
        return receipt

    def record_artifact(
        self,
        *,
        run_id: str,
        invocation_id: str,
        artifact_ref: str,
        digest: str,
        idempotency_key: str,
    ) -> ProductReceipt:
        prior = self._receipts.get(idempotency_key)
        if prior is not None:
            if not isinstance(prior, ProductReceipt) or prior.resource_ref != artifact_ref:
                raise BindingError("artifact idempotency key was reused for another operation")
            return prior
        self.artifacts.append((run_id, artifact_ref, idempotency_key))
        receipt = ProductReceipt(
            resource_ref=artifact_ref,
            receipt_ref=f"receipt:{idempotency_key}",
            run_id=run_id,
            invocation_id=invocation_id,
            idempotency_key=idempotency_key,
        )
        self._receipts[idempotency_key] = receipt
        return receipt


class TrustedRuntimeSupervisor:
    """Small deterministic supervisor adapter used by the service and tests."""

    def __init__(
        self,
        *,
        clock: TrustedClock | Callable[[], datetime],
        checkpoint_refs: set[str] | frozenset[str] = frozenset(),
    ) -> None:
        self.clock = clock
        self.checkpoint_refs = frozenset(checkpoint_refs)

    def validate_lease(self, *, run: RunSnapshot, invocation: InvocationEnvelope) -> None:
        del run
        _validate_lease_with_clock(invocation, self.clock)

    def validate_checkpoint(self, *, run: RunSnapshot, invocation: InvocationEnvelope) -> None:
        del run
        checkpoint_ref = invocation.checkpoint_ref
        if checkpoint_ref is None or checkpoint_ref not in self.checkpoint_refs:
            raise BindingError("checkpoint is not trusted for this continuation")


class SupervisorReconciler:
    """Idempotently records the one supervisor-classified terminal result."""

    def __init__(self) -> None:
        self._terminal: RuntimeExit | None = None
        self._reconciliation_count = 0

    @property
    def terminal(self) -> RuntimeExit | None:
        return self._terminal

    @property
    def reconciliation_count(self) -> int:
        return self._reconciliation_count

    def reconcile(self, exit_value: RuntimeExit) -> RuntimeExit:
        if self._terminal is None:
            self._terminal = exit_value
            self._reconciliation_count = 1
            return exit_value
        if self._terminal != exit_value:
            raise SequenceError("terminal reconciliation was attempted with different content")
        return self._terminal

    def reconcile_process_death(
        self,
        *,
        final_sequence: int,
        reason: str = "runtime process exited before returning an exit",
    ) -> RuntimeExit:
        return self.reconcile(classify_process_death(final_sequence=final_sequence, reason=reason))


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
