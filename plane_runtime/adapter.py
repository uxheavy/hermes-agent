"""The narrow runtime adapter and the deterministic kernel port.

The adapter is intentionally the only module that knows how to translate the
versioned Plane contract into a kernel request.  A future Hermes adapter can
implement :class:`KernelPort` without adding Plane vocabulary to Hermes core.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from threading import Event, RLock
from typing import Callable, Iterable, Mapping, Protocol

from .contract import (
    ArtifactObserved,
    BindingError,
    BoundsError,
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
    TerminalProposal,
    TerminalReconciliationReceipt,
    TERMINAL_KINDS,
    TranscriptObserved,
    UsageObserved,
    RuntimeLease,
    MAX_REFERENCE_LENGTH,
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


@dataclass(frozen=True)
class CanonicalLeaseBinding:
    """Host-owned lease state kept separate from the untrusted envelope."""

    run_id: str
    invocation_id: str
    lease_id: str
    holder_ref: str
    active: bool
    expires_at: str

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ContractError("lease binding runId must be a non-empty string")
        if len(self.run_id) > MAX_REFERENCE_LENGTH:
            raise BoundsError("lease binding runId exceeds the reference limit")
        if not isinstance(self.invocation_id, str) or not self.invocation_id:
            raise ContractError("lease binding invocationId must be a non-empty string")
        if len(self.invocation_id) > MAX_REFERENCE_LENGTH:
            raise BoundsError("lease binding invocationId exceeds the reference limit")
        lease = RuntimeLease(self.lease_id, self.holder_ref, self.expires_at)
        object.__setattr__(self, "lease_id", lease.lease_id)
        object.__setattr__(self, "holder_ref", lease.holder_ref)
        object.__setattr__(self, "expires_at", lease.expires_at)
        if not isinstance(self.active, bool):
            raise ContractError("lease binding active must be a boolean")

    def matches(self, invocation: InvocationEnvelope) -> bool:
        lease = invocation.lease
        return (
            self.run_id == invocation.run_id
            and self.invocation_id == invocation.invocation_id
            and self.lease_id == lease.lease_id
            and self.holder_ref == lease.holder_ref
            and self.expires_at == lease.expires_at
        )


class CanonicalLeaseAuthority(Protocol):
    """Host authority that atomically validates the complete lease binding."""

    def validate_lease(
        self,
        *,
        run: RunSnapshot,
        invocation: InvocationEnvelope,
        binding: CanonicalLeaseBinding,
    ) -> None:
        ...


@dataclass(frozen=True)
class CheckpointAttestation:
    """Host attestation for one safe, invocation-specific continuation."""

    checkpoint_ref: str
    source_run_id: str
    source_invocation_id: str
    snapshot_digest: str
    actor_ref: str
    profile_version: str
    continuation_event_ref: str
    continuation_trigger_kind: str
    allowed_target_invocation_id: str

    def __post_init__(self) -> None:
        fields = (
            "checkpoint_ref",
            "source_run_id",
            "source_invocation_id",
            "snapshot_digest",
            "actor_ref",
            "profile_version",
            "continuation_event_ref",
            "continuation_trigger_kind",
            "allowed_target_invocation_id",
        )
        for name in fields:
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ContractError(f"checkpoint attestation {name} must be a non-empty string")
            if len(value) > MAX_REFERENCE_LENGTH:
                raise BoundsError(f"checkpoint attestation {name} exceeds the reference limit")


class CheckpointAuthority(Protocol):
    """Host authority for complete, immutable checkpoint attestations."""

    def validate_checkpoint(
        self,
        *,
        run: RunSnapshot,
        invocation: InvocationEnvelope,
        attestation: CheckpointAttestation,
    ) -> None:
        ...


class TerminalReconciliationPort(Protocol):
    """Plane application-service seam for terminal lifecycle proposals."""

    def reconcile_terminal(
        self, proposal: TerminalProposal
    ) -> TerminalReconciliationReceipt:
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
        if self.terminal_kind not in TERMINAL_KINDS:
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


def _authority_now(clock: Callable[[], datetime]) -> datetime:
    value = clock()
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise LeaseError("trusted clock must return a timezone-aware datetime")
    if value.utcoffset() != timedelta(0):
        raise LeaseError("trusted clock must return UTC time")
    value = value.astimezone(timezone.utc)
    return value


class FixtureCanonicalLeaseAuthority:
    """Deterministic host authority used by tests and the demo service only."""

    def __init__(
        self,
        bindings: Iterable[CanonicalLeaseBinding],
        *,
        clock: Callable[[], datetime],
    ) -> None:
        self._clock = clock
        self._lock = RLock()
        self._bindings: dict[tuple[str, str], CanonicalLeaseBinding] = {}
        for binding in bindings:
            key = (binding.run_id, binding.invocation_id)
            if key in self._bindings:
                raise ContractError("duplicate canonical lease binding")
            self._bindings[key] = binding

    def validate_lease(
        self,
        *,
        run: RunSnapshot,
        invocation: InvocationEnvelope,
        binding: CanonicalLeaseBinding,
    ) -> None:
        """Check every lease field under one authority lock."""

        with self._lock:
            canonical = self._bindings.get((run.run_id, invocation.invocation_id))
            if canonical is None:
                raise LeaseError("no canonical lease binding exists for this invocation")
            if canonical != binding:
                raise BindingError("host lease binding does not match canonical authority state")
            if not canonical.matches(invocation):
                raise BindingError("invocation lease does not match canonical authority state")
            if not canonical.active:
                raise LeaseError("invocation lease is not active")
            if parse_utc_timestamp(canonical.expires_at, "lease binding expiresAt") <= _authority_now(
                self._clock
            ):
                raise LeaseError("invocation lease is expired")


class FixtureCheckpointAuthority:
    """Deterministic host authority for complete checkpoint attestations."""

    def __init__(self, attestations: Iterable[CheckpointAttestation]) -> None:
        self._lock = RLock()
        self._attestations: dict[tuple[str, str, str], CheckpointAttestation] = {}
        for attestation in attestations:
            key = (
                attestation.source_run_id,
                attestation.checkpoint_ref,
                attestation.allowed_target_invocation_id,
            )
            if key in self._attestations:
                raise ContractError("duplicate checkpoint attestation")
            self._attestations[key] = attestation

    def validate_checkpoint(
        self,
        *,
        run: RunSnapshot,
        invocation: InvocationEnvelope,
        attestation: CheckpointAttestation,
    ) -> None:
        with self._lock:
            if invocation.checkpoint_ref is None:
                raise BindingError("checkpoint attestation requires a checkpoint reference")
            canonical = self._attestations.get(
                (run.run_id, invocation.checkpoint_ref, invocation.invocation_id)
            )
            if canonical is None or canonical != attestation:
                raise BindingError("checkpoint attestation is not canonical for this continuation")
            if (
                attestation.source_run_id != run.run_id
                or attestation.snapshot_digest != run.digest()
                or attestation.actor_ref != run.actor_ref
                or attestation.profile_version != run.profile_version
                or attestation.allowed_target_invocation_id != invocation.invocation_id
                or attestation.checkpoint_ref != invocation.checkpoint_ref
                or attestation.continuation_trigger_kind != invocation.trigger.kind
                or attestation.continuation_event_ref != invocation.trigger.event_ref
            ):
                raise BindingError("checkpoint attestation is bound to different run continuation state")


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


def _terminal_proposal(
    *,
    run: RunSnapshot,
    invocation: InvocationEnvelope,
    exit_value: RuntimeExit,
    stream: EventStream,
    source: str = "runtime",
) -> TerminalProposal:
    receipt_refs = tuple(
        receipt_ref
        for event in stream.events
        for receipt_ref in (getattr(event.body, "receipt_ref", None),)
        if receipt_ref is not None
    )
    return TerminalProposal(
        run_id=run.run_id,
        invocation_id=invocation.invocation_id,
        kind=exit_value.kind,
        final_sequence=exit_value.final_sequence,
        evidence_event_ids=tuple(event.event_id for event in stream.events),
        evidence_receipt_refs=receipt_refs,
        failure=exit_value.failure,
        source=source,
    )


def _reconcile_terminal(
    port: TerminalReconciliationPort | None,
    proposal: TerminalProposal,
) -> TerminalReconciliationReceipt:
    if port is None:
        raise ContractError("execution requires an injected terminal reconciliation port")
    receipt = port.reconcile_terminal(proposal)
    if not isinstance(receipt, TerminalReconciliationReceipt):
        raise ContractError("terminal reconciliation returned an invalid receipt")
    if (
        receipt.run_id != proposal.run_id
        or receipt.invocation_id != proposal.invocation_id
        or receipt.kind != proposal.kind
        or receipt.idempotency_key != proposal.idempotency_key
    ):
        raise BindingError("terminal reconciliation receipt is not bound to this proposal")
    if not receipt.accepted or not receipt.legal_transition:
        raise BindingError("Plane rejected the terminal proposal or reported an illegal transition")
    return receipt


def _return_terminal(
    *,
    port: TerminalReconciliationPort | None,
    run: RunSnapshot,
    invocation: InvocationEnvelope,
    stream: EventStream,
    exit_value: RuntimeExit,
) -> RuntimeExit:
    _reconcile_terminal(
        port,
        _terminal_proposal(
            run=run,
            invocation=invocation,
            exit_value=exit_value,
            stream=stream,
        ),
    )
    return exit_value


def execute(
    *,
    run: RunSnapshot,
    invocation: InvocationEnvelope,
    host: RuntimeHost | None,
    emit: EventSink | None = None,
    cancellation: CancellationSignal | None = None,
    kernel: KernelPort | None = None,
    lease_authority: CanonicalLeaseAuthority | None = None,
    lease_binding: CanonicalLeaseBinding | None = None,
    checkpoint_authority: CheckpointAuthority | None = None,
    checkpoint_attestation: CheckpointAttestation | None = None,
    terminal_port: TerminalReconciliationPort | None = None,
) -> RuntimeExit:
    """Execute exactly one invocation through the replaceable kernel port.

    The adapter validates binding and cumulative budget at entry, translates
    kernel-neutral observations into bounded runtime events, and returns one
    immutable terminal exit.  It does not persist state or mutate ``run``.
    """

    _validate_inputs(run, invocation)
    if lease_authority is None or lease_binding is None:
        raise LeaseError("execution requires an injected canonical lease authority and host binding")
    lease_authority.validate_lease(run=run, invocation=invocation, binding=lease_binding)
    if invocation.checkpoint_ref is not None:
        if checkpoint_authority is None or checkpoint_attestation is None:
            raise ContractError(
                "checkpoint continuation requires an injected checkpoint authority and attestation"
            )
        checkpoint_authority.validate_checkpoint(
            run=run,
            invocation=invocation,
            attestation=checkpoint_attestation,
        )
    elif checkpoint_authority is not None or checkpoint_attestation is not None:
        raise ContractError("checkpoint authority data is not allowed for an initial invocation")
    if terminal_port is None:
        raise ContractError("execution requires an injected terminal reconciliation port")
    cancellation = cancellation or NeverCancelled()
    stream = EventStream(
        run_id=run.run_id,
        invocation_id=invocation.invocation_id,
        sink=emit,
        expected_causation_ref=invocation.causation_ref,
    )
    if cancellation.is_cancelled():
        return _return_terminal(
            port=terminal_port,
            run=run,
            invocation=invocation,
            stream=stream,
            exit_value=RuntimeExit(kind="cancelled", final_sequence=stream.last_sequence),
        )
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
        return _return_terminal(
            port=terminal_port,
            run=run,
            invocation=invocation,
            stream=stream,
            exit_value=RuntimeExit(kind="cancelled", final_sequence=stream.last_sequence),
        )
    if cancellation.is_cancelled():
        return _return_terminal(
            port=terminal_port,
            run=run,
            invocation=invocation,
            stream=stream,
            exit_value=RuntimeExit(kind="cancelled", final_sequence=stream.last_sequence),
        )
    if result.terminal_kind == "waiting_for_input" and not pending_input_requests:
        raise ContractError("waiting_for_input requires a visible authorized input request")
    if result.terminal_kind != "waiting_for_input" and pending_input_requests:
        raise ContractError("terminal exit cannot leave an unresolved input request")
    return _return_terminal(
        port=terminal_port,
        run=run,
        invocation=invocation,
        stream=stream,
        exit_value=RuntimeExit(
            kind=result.terminal_kind,
            final_sequence=stream.last_sequence,
            failure=result.failure,
        ),
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


def reconcile_process_death(
    *,
    port: TerminalReconciliationPort,
    run_id: str,
    invocation_id: str,
    final_sequence: int,
    reason: str = "runtime process exited before returning an exit",
) -> TerminalReconciliationReceipt:
    """Submit trusted supervisor process-death evidence through the Plane port."""

    exit_value = classify_process_death(final_sequence=final_sequence, reason=reason)
    proposal = TerminalProposal(
        run_id=run_id,
        invocation_id=invocation_id,
        kind=exit_value.kind,
        final_sequence=exit_value.final_sequence,
        failure=exit_value.failure,
        source="supervisor",
    )
    return _reconcile_terminal(port, proposal)
