"""The narrow runtime adapter and the deterministic kernel port.

The adapter is intentionally the only module that knows how to translate the
versioned Plane contract into a kernel request.  A future Hermes adapter can
implement :class:`KernelPort` without adding Plane vocabulary to Hermes core.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from collections import deque
from threading import Event, RLock
from typing import Callable, Iterable, Mapping, Protocol

from .contract import (
    ArtifactObserved,
    ArtifactProposal,
    BindingError,
    BoundsError,
    CancellationAuthorityReceipt,
    ContractError,
    InputRequestObserved,
    MessageProposal,
    MessageProposalObserved,
    LeaseError,
    OutcomeSubmissionObserved,
    OutcomeProposal,
    InputRequestProposal,
    ProgressObserved,
    ProductReceipt,
    RuntimeBudget,
    RuntimeEvent,
    RuntimeExit,
    RuntimeFailure,
    RuntimeConfigurationError,
    RunSnapshot,
    SequenceError,
    TerminalProposal,
    TerminalProof,
    TerminalReconciliationReceipt,
    TERMINAL_KINDS,
    TranscriptObserved,
    UsageObserved,
    RuntimeLease,
    MAX_REFERENCE_LENGTH,
    MAX_ARTIFACT_PROPOSALS,
    MAX_ARTIFACT_PROPOSAL_BYTES,
    MAX_INPUT_PROPOSALS,
    MAX_INPUT_PROPOSAL_BYTES,
    MAX_MESSAGE_PROPOSALS,
    MAX_MESSAGE_PROPOSAL_BYTES,
    MAX_OUTCOME_PROPOSALS,
    MAX_OUTCOME_PROPOSAL_BYTES,
    MAX_TRANSCRIPT_BYTES,
    MAX_TRANSCRIPT_OBSERVATIONS,
    MAX_EVENTS_PER_INVOCATION,
    MAX_EVENT_STREAM_BYTES,
    MAX_OPTIONAL_EVENT_TAIL,
    MAX_TERMINAL_EVIDENCE,
    canonical_json_bytes,
    parse_utc_timestamp,
    product_proof_identity,
)
from .contract import InvocationEnvelope, PROTOCOL


EventSink = Callable[[RuntimeEvent], None]
TerminalProposalSink = Callable[[TerminalProposal], None]


class CancellationSignal(Protocol):
    """Invocation-scoped cancellation, not durable run state."""

    def is_cancelled(self) -> bool:
        ...


class RuntimeHost(Protocol):
    """Compatibility marker for non-authoritative host context.

    RuntimeHost intentionally has no callable product operation.  Runtime
    observations and message proposals cross the event sink only; publication
    belongs to a later explicit Plane gateway operation.
    """

    pass

class CancellationAuthority(Protocol):
    """Plane-owned cancellation correlation authority."""

    def validate_cancellation(
        self,
        *,
        run: RunSnapshot,
        invocation: InvocationEnvelope,
    ) -> CancellationAuthorityReceipt:
        ...


@dataclass(frozen=True)
class CanonicalCancellationBinding:
    """Immutable host cancellation state bound to the complete invocation lease."""

    run_id: str
    invocation_id: str
    actor_ref: str
    workspace_ref: str
    snapshot_digest: str
    lease_id: str
    lease_holder_ref: str
    lease_expires_at: str
    cancellation_ref: str
    idempotency_key: str
    gateway_receipt_ref: str
    audit_ref: str
    receipt_ref: str
    active: bool = True

    def __post_init__(self) -> None:
        RuntimeLease(self.lease_id, self.lease_holder_ref, self.lease_expires_at)
        CancellationAuthorityReceipt(
            resource_ref=self.cancellation_ref,
            receipt_ref=self.receipt_ref,
            run_id=self.run_id,
            invocation_id=self.invocation_id,
            actor_ref=self.actor_ref,
            workspace_ref=self.workspace_ref,
            snapshot_digest=self.snapshot_digest,
            idempotency_key=self.idempotency_key,
            gateway_receipt_ref=self.gateway_receipt_ref,
            audit_ref=self.audit_ref,
        )
        if not isinstance(self.active, bool):
            raise ContractError("cancellation binding active must be a boolean")

    def matches(self, run: RunSnapshot, invocation: InvocationEnvelope) -> bool:
        lease = invocation.lease
        return (
            self.run_id == run.run_id
            and self.invocation_id == invocation.invocation_id
            and self.actor_ref == run.actor_ref
            and self.workspace_ref == run.workspace_ref
            and self.snapshot_digest == run.digest()
            and self.lease_id == lease.lease_id
            and self.lease_holder_ref == lease.holder_ref
            and self.lease_expires_at == lease.expires_at
            and self.cancellation_ref == invocation.cancellation_ref
            and self.idempotency_key == f"cancel:{run.run_id}:{invocation.invocation_id}"
            and self.gateway_receipt_ref == f"gateway:{self.idempotency_key}"
            and self.audit_ref == f"audit:{self.idempotency_key}"
            and self.receipt_ref == f"cancel-receipt:{invocation.invocation_id}"
        )

    def receipt(self) -> CancellationAuthorityReceipt:
        return CancellationAuthorityReceipt(
            resource_ref=self.cancellation_ref,
            receipt_ref=self.receipt_ref,
            run_id=self.run_id,
            invocation_id=self.invocation_id,
            actor_ref=self.actor_ref,
            workspace_ref=self.workspace_ref,
            snapshot_digest=self.snapshot_digest,
            idempotency_key=self.idempotency_key,
            gateway_receipt_ref=self.gateway_receipt_ref,
            audit_ref=self.audit_ref,
        )


class CanonicalCancellationAuthority:
    """Trusted host port for exact, idempotent cancellation authority state."""

    def __init__(self, bindings: Iterable[CanonicalCancellationBinding]) -> None:
        self._lock = RLock()
        self._bindings: dict[tuple[str, str], CanonicalCancellationBinding] = {}
        for binding in bindings:
            key = (binding.run_id, binding.invocation_id)
            if key in self._bindings:
                raise ContractError("duplicate canonical cancellation binding")
            self._bindings[key] = binding

    def validate_cancellation(
        self,
        *,
        run: RunSnapshot,
        invocation: InvocationEnvelope,
    ) -> CancellationAuthorityReceipt:
        with self._lock:
            binding = self._bindings.get((run.run_id, invocation.invocation_id))
            if binding is None:
                raise BindingError("no canonical cancellation binding exists for this invocation")
            if not binding.active:
                raise BindingError("canonical cancellation authority is inactive")
            if not binding.matches(run, invocation):
                raise BindingError("canonical cancellation binding does not match the invocation")
            return binding.receipt()


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

    def claim_checkpoint(
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
    submission_ref: str | None = None
    outcome_content: str | None = None
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


@dataclass
class ExecutionPhase:
    """Observable phase marker used only by the process boundary classifier."""

    execution_started: bool = False


class TerminalReconciliationError(ContractError):
    """Raised when terminal reconciliation cannot be trusted or reached."""

    def __init__(
        self,
        receipt: TerminalReconciliationReceipt | None = None,
        message: str = "Plane did not accept the terminal reconciliation proposal",
    ) -> None:
        self.receipt = receipt
        super().__init__(message)


class TerminalReconciliationRejected(Exception):
    """Raised only for a fully validated, proofless legal terminal rejection."""

    def __init__(self, receipt: TerminalReconciliationReceipt) -> None:
        if not isinstance(receipt, TerminalReconciliationReceipt):
            raise TypeError("terminal rejection requires a terminal reconciliation receipt")
        if receipt.accepted or receipt.legal_transition:
            raise ValueError("terminal rejection receipt must be rejected and illegal")
        if receipt.proofs or receipt.product_receipts:
            raise ValueError("terminal rejection receipt must be proofless")
        self.receipt = receipt
        super().__init__("Plane legally rejected the terminal reconciliation proposal")


class ChildCancellationProposalRejected(BindingError):
    """A child tried to provide authority for a host-owned cancellation."""


_TerminalRejectionSink = Callable[[TerminalReconciliationRejected], None]


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
        self._event_count = 0
        self._stream_bytes = 0
        self._seen: dict[str, RuntimeEvent] = {}
        self._seen_idempotency: dict[str, RuntimeEvent] = {}
        self._required_events: dict[str, RuntimeEvent] = {}
        self._optional_events: deque[RuntimeEvent] = deque(maxlen=MAX_OPTIONAL_EVENT_TAIL)
        self._category_counts: dict[str, int] = {}
        self._category_bytes: dict[str, int] = {}

    @property
    def events(self) -> tuple[RuntimeEvent, ...]:
        events = tuple(self._required_events.values()) + tuple(self._optional_events)
        return tuple(sorted(events, key=lambda item: item.sequence))

    @property
    def last_sequence(self) -> int:
        return max((event.sequence for event in self.events), default=0)

    @property
    def event_count(self) -> int:
        return self._event_count

    @property
    def retained_event_count(self) -> int:
        return len(self._required_events) + len(self._optional_events)

    @property
    def stream_bytes(self) -> int:
        return self._stream_bytes

    @property
    def retained_sequence_entries(self) -> int:
        return len(self._seen)

    @property
    def retained_idempotency_entries(self) -> int:
        return len(self._seen_idempotency)

    @property
    def category_counts(self) -> Mapping[str, int]:
        """Return bounded observation counts by proposal category."""

        return dict(self._category_counts)

    @property
    def category_bytes(self) -> Mapping[str, int]:
        """Return bounded canonical event bytes by proposal category."""

        return dict(self._category_bytes)

    @staticmethod
    def _category(event: RuntimeEvent) -> str:
        body = event.body
        if isinstance(body, TranscriptObserved):
            return "transcript"
        if isinstance(body, ArtifactObserved):
            return "artifact"
        if isinstance(body, InputRequestObserved):
            return "input"
        if isinstance(body, OutcomeSubmissionObserved):
            return "outcome"
        if isinstance(body, MessageProposalObserved):
            return "message"
        return "event"

    @staticmethod
    def _category_limits(category: str) -> tuple[int | None, int | None]:
        return {
            "transcript": (MAX_TRANSCRIPT_OBSERVATIONS, MAX_TRANSCRIPT_BYTES),
            "artifact": (MAX_ARTIFACT_PROPOSALS, MAX_ARTIFACT_PROPOSAL_BYTES),
            "input": (MAX_INPUT_PROPOSALS, MAX_INPUT_PROPOSAL_BYTES),
            "outcome": (MAX_OUTCOME_PROPOSALS, MAX_OUTCOME_PROPOSAL_BYTES),
            "message": (MAX_MESSAGE_PROPOSALS, MAX_MESSAGE_PROPOSAL_BYTES),
            "event": (None, None),
        }[category]

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
        if self._event_count >= MAX_EVENTS_PER_INVOCATION:
            raise BoundsError(
                f"invocation event count exceeds {MAX_EVENTS_PER_INVOCATION}"
            )
        event_bytes = len(canonical_json_bytes(event.to_dict()))
        if self._stream_bytes + event_bytes > MAX_EVENT_STREAM_BYTES:
            raise BoundsError(
                f"invocation event stream exceeds {MAX_EVENT_STREAM_BYTES} canonical UTF-8 bytes"
            )
        category = self._category(event)
        category_count, category_limit_bytes = self._category_limits(category)
        current_count = self._category_counts.get(category, 0)
        current_bytes = self._category_bytes.get(category, 0)
        if category_count is not None and current_count >= category_count:
            raise BoundsError(f"{category} proposal count exceeds {category_count}")
        if category_limit_bytes is not None and current_bytes + event_bytes > category_limit_bytes:
            raise BoundsError(f"{category} proposal bytes exceed {category_limit_bytes}")
        if self._sink is not None:
            self._sink(event)
        self._seen[event.event_id] = event
        self._seen_idempotency[event.idempotency_key] = event
        self._event_count += 1
        self._stream_bytes += event_bytes
        self._category_counts[category] = current_count + 1
        self._category_bytes[category] = current_bytes + event_bytes
        if isinstance(
            event.body,
            (
                ArtifactObserved,
                InputRequestObserved,
                OutcomeSubmissionObserved,
                MessageProposalObserved,
            ),
        ):
            self._required_events[event.event_id] = event
        else:
            self._optional_events.append(event)
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
    """Deterministic host authority used by explicit tests only."""

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
        self._attestations: dict[tuple[str, ...], CheckpointAttestation] = {}
        self._consumed: set[tuple[str, ...]] = set()
        for attestation in attestations:
            key = self._key(attestation)
            if key in self._attestations:
                raise ContractError("duplicate checkpoint attestation")
            self._attestations[key] = attestation

    @staticmethod
    def _key(attestation: CheckpointAttestation) -> tuple[str, ...]:
        return (
            attestation.source_run_id,
            attestation.source_invocation_id,
            attestation.checkpoint_ref,
            attestation.snapshot_digest,
            attestation.actor_ref,
            attestation.profile_version,
            attestation.continuation_event_ref,
            attestation.continuation_trigger_kind,
            attestation.allowed_target_invocation_id,
        )

    def claim_checkpoint(
        self,
        *,
        run: RunSnapshot,
        invocation: InvocationEnvelope,
        attestation: CheckpointAttestation,
    ) -> None:
        with self._lock:
            if invocation.checkpoint_ref is None:
                raise BindingError("checkpoint attestation requires a checkpoint reference")
            key = self._key(attestation)
            canonical = self._attestations.get(key)
            if canonical is None or canonical != attestation:
                raise BindingError("checkpoint attestation is not canonical for this continuation")
            if (
                attestation.source_run_id != run.run_id
                or not attestation.source_invocation_id
                or attestation.snapshot_digest != run.digest()
                or attestation.actor_ref != run.actor_ref
                or attestation.profile_version != run.profile_version
                or attestation.allowed_target_invocation_id != invocation.invocation_id
                or attestation.checkpoint_ref != invocation.checkpoint_ref
                or attestation.continuation_trigger_kind != invocation.trigger.kind
                or attestation.continuation_event_ref != invocation.trigger.event_ref
            ):
                raise BindingError("checkpoint attestation is bound to different run continuation state")
            if key in self._consumed:
                raise SequenceError("checkpoint attestation was already consumed")
            self._consumed.add(key)


class FixtureCancellationAuthority:
    """Deterministic host authority used by tests and the demo service only."""

    def __init__(self, receipts: Iterable[CancellationAuthorityReceipt]) -> None:
        self._lock = RLock()
        self._receipts: dict[tuple[str, str], CancellationAuthorityReceipt] = {}
        for receipt in receipts:
            key = (receipt.run_id, receipt.invocation_id)
            if key in self._receipts:
                raise ContractError("duplicate canonical cancellation receipt")
            self._receipts[key] = receipt

    def validate_cancellation(
        self,
        *,
        run: RunSnapshot,
        invocation: InvocationEnvelope,
    ) -> CancellationAuthorityReceipt:
        with self._lock:
            receipt = self._receipts.get((run.run_id, invocation.invocation_id))
            if receipt is None:
                raise BindingError("no canonical cancellation receipt exists for this invocation")
            if (
                receipt.run_id != run.run_id
                or receipt.invocation_id != invocation.invocation_id
                or receipt.resource_ref != invocation.cancellation_ref
            ):
                raise BindingError("cancellation receipt is not bound to this invocation")
            return receipt


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


def _terminal_proof_resource(proof_kind: str, terminal_slot: str) -> str:
    prefixes = {
        "operation_attempt": "operation",
        "application": "application",
        "gateway": "gateway",
        "audit": "audit",
        "product_event": "product-event",
    }
    try:
        prefix = prefixes[proof_kind]
    except KeyError as exc:  # pragma: no cover - TerminalProof rejects this first
        raise ContractError(f"unsupported terminal proof kind: {proof_kind!r}") from exc
    return f"{prefix}:{terminal_slot}"


def _terminal_proof_ref(proof_kind: str, terminal_slot: str) -> str:
    return f"terminal-proof:{proof_kind}:{terminal_slot}"


def _proofs_by_kind(receipt: TerminalReconciliationReceipt) -> dict[str, TerminalProof]:
    return {proof.proof_kind: proof for proof in receipt.proofs}


def _terminal_proposal(
    *,
    run: RunSnapshot,
    invocation: InvocationEnvelope,
    exit_value: RuntimeExit,
    stream: EventStream,
    source: str = "runtime",
    outcome_proposal: OutcomeProposal | None = None,
    input_request_proposal: InputRequestProposal | None = None,
    cancellation_receipt: CancellationAuthorityReceipt | None = None,
    artifact_proposals: Iterable[ArtifactProposal] = (),
    message_proposals: Iterable[MessageProposal] = (),
) -> TerminalProposal:
    all_receipt_refs = tuple(
        receipt_ref
        for event in stream.events
        for receipt_ref in (getattr(event.body, "receipt_ref", None),)
        if receipt_ref is not None
    )
    required_event_ids = tuple(
        item.event_id
        for item in (outcome_proposal, input_request_proposal)
        if item is not None
    )
    required_receipt_refs = tuple(
        item.proposal_receipt_ref
        for item in (outcome_proposal, input_request_proposal)
        if item is not None
    )
    if cancellation_receipt is not None:
        required_receipt_refs += (cancellation_receipt.receipt_ref,)

    def preserve_required(required: Iterable[str], optional: Iterable[str]) -> tuple[str, ...]:
        required_values = tuple(dict.fromkeys(required))
        if len(required_values) > MAX_TERMINAL_EVIDENCE:
            raise BoundsError("terminal required evidence exceeds the bounded terminal surface")
        optional_values = tuple(
            value for value in optional if value not in set(required_values)
        )
        room = MAX_TERMINAL_EVIDENCE - len(required_values)
        return required_values + optional_values[-room:]

    artifacts = tuple(artifact_proposals)
    messages = tuple(message_proposals)
    return TerminalProposal(
        run_id=run.run_id,
        invocation_id=invocation.invocation_id,
        actor_ref=run.actor_ref,
        workspace_ref=run.workspace_ref,
        snapshot_digest=run.digest(),
        kind=exit_value.kind,
        final_sequence=exit_value.final_sequence,
        evidence_event_ids=preserve_required(
            required_event_ids, (event.event_id for event in stream.events)
        ),
        evidence_receipt_refs=preserve_required(required_receipt_refs, all_receipt_refs),
        failure=exit_value.failure,
        source=source,
        outcome_proposal=outcome_proposal,
        input_request_proposal=input_request_proposal,
        cancellation_receipt=cancellation_receipt,
        artifact_proposals=artifacts,
        message_proposals=messages,
    )


def _reconcile_terminal(
    port: TerminalReconciliationPort | None,
    proposal: TerminalProposal,
) -> TerminalReconciliationReceipt:
    if port is None:
        raise ContractError("execution requires an injected terminal reconciliation port")
    if proposal.kind == "cancelled" and proposal.cancellation_receipt is None:
        raise TerminalReconciliationError(
            message="host cancellation authority is required before terminal reconciliation",
        )
    try:
        receipt = port.reconcile_terminal(proposal)
    except Exception as exc:
        raise TerminalReconciliationError(
            message="terminal reconciliation was unavailable",
        ) from exc
    if not isinstance(receipt, TerminalReconciliationReceipt):
        raise TerminalReconciliationError(
            message="terminal reconciliation returned an invalid receipt",
        )
    try:
        receipt.to_json()
    except Exception as exc:
        raise TerminalReconciliationError(
            message="terminal reconciliation returned an oversized receipt",
        ) from exc
    if (
        receipt.run_id != proposal.run_id
        or receipt.invocation_id != proposal.invocation_id
        or receipt.kind != proposal.kind
        or receipt.idempotency_key != proposal.idempotency_key
    ):
        raise TerminalReconciliationError(
            message="terminal reconciliation receipt is not bound to this proposal",
        )
    if not receipt.accepted or not receipt.legal_transition:
        if (
            receipt.accepted
            or receipt.legal_transition
            or receipt.proofs
            or receipt.product_receipts
        ):
            raise TerminalReconciliationError(
                message="terminal rejection receipt is not a proofless legal rejection",
            )
        return receipt
    if len(receipt.proofs) != 5:
        raise TerminalReconciliationError(
            message="terminal reconciliation receipt requires exactly five typed proofs",
        )
    if (
        len({proof.proof_ref for proof in receipt.proofs}) != 5
        or len({proof.resource_ref for proof in receipt.proofs}) != 5
    ):
        raise TerminalReconciliationError(
            message="terminal reconciliation receipt contains duplicate proof identities",
        )
    proofs = _proofs_by_kind(receipt)
    if set(proofs) != {
        "operation_attempt",
        "application",
        "gateway",
        "audit",
        "product_event",
    }:
        raise TerminalReconciliationError(
            message="terminal reconciliation receipt has missing or extra typed proofs",
        )
    for proof_kind, proof in proofs.items():
        if (
            proof.proof_ref != _terminal_proof_ref(proof_kind, proposal.idempotency_key)
            or proof.resource_ref != _terminal_proof_resource(proof_kind, proposal.idempotency_key)
            or proof.run_id != proposal.run_id
            or proof.invocation_id != proposal.invocation_id
            or proof.actor_ref != proposal.actor_ref
            or proof.workspace_ref != proposal.workspace_ref
            or proof.snapshot_digest != proposal.snapshot_digest
            or proof.terminal_slot != proposal.idempotency_key
            or proof.terminal_kind != proposal.kind
            or proof.proposal_digest != proposal.digest()
        ):
            raise TerminalReconciliationError(
                message="terminal reconciliation receipt contains an unbound typed proof",
            )
    expected: list[tuple[str, str]] = []
    if proposal.kind == "completed":
        assert proposal.outcome_proposal is not None
        expected.append(("outcome_submission", proposal.outcome_proposal.submission_ref))
    elif proposal.kind == "waiting_for_input":
        assert proposal.input_request_proposal is not None
        expected.append(("input_request", proposal.input_request_proposal.request_ref))
    else:
        expected.append(("terminal_event", proofs["product_event"].resource_ref))
    expected.extend(("artifact", item.artifact_ref) for item in proposal.artifact_proposals)
    expected.extend(("message", item.message_ref) for item in proposal.message_proposals)
    if len(receipt.product_receipts) != len(expected):
        raise TerminalReconciliationError(
            message="terminal reconciliation receipt has missing or extra product receipts",
        )
    seen_receipts: set[str] = set()
    seen_resources: set[str] = set()
    seen_idempotency_keys: set[str] = set()
    for product_receipt, (expected_kind, expected_resource) in zip(
        receipt.product_receipts, expected
    ):
        if (
            product_receipt.receipt_ref in seen_receipts
            or product_receipt.resource_ref in seen_resources
            or product_receipt.idempotency_key in seen_idempotency_keys
        ):
            raise TerminalReconciliationError(
                message="terminal reconciliation receipt contains duplicate product proof",
            )
        seen_receipts.add(product_receipt.receipt_ref)
        seen_resources.add(product_receipt.resource_ref)
        seen_idempotency_keys.add(product_receipt.idempotency_key)
        expected_receipt_ref, expected_idempotency_key = product_proof_identity(
            proof_kind=expected_kind,
            product_kind=expected_kind,
            resource_ref=expected_resource,
            run_id=proposal.run_id,
            invocation_id=proposal.invocation_id,
            actor_ref=proposal.actor_ref,
            workspace_ref=proposal.workspace_ref,
            snapshot_digest=proposal.snapshot_digest,
            terminal_slot=proposal.idempotency_key,
            terminal_kind=proposal.kind,
            proposal_digest=proposal.digest(),
        )
        if (
            product_receipt.run_id != proposal.run_id
            or product_receipt.invocation_id != proposal.invocation_id
            or product_receipt.kind != expected_kind
            or product_receipt.terminal_kind != proposal.kind
            or product_receipt.resource_ref != expected_resource
            or product_receipt.actor_ref != proposal.actor_ref
            or product_receipt.workspace_ref != proposal.workspace_ref
            or product_receipt.snapshot_digest != proposal.snapshot_digest
            or product_receipt.terminal_slot != proposal.idempotency_key
            or product_receipt.proposal_digest != proposal.digest()
            or product_receipt.receipt_ref != expected_receipt_ref
            or product_receipt.idempotency_key != expected_idempotency_key
        ):
            raise TerminalReconciliationError(
                message="terminal reconciliation receipt contains wrong product proof",
            )
    return receipt


def reconcile_terminal_proposal(
    *,
    port: TerminalReconciliationPort,
    proposal: TerminalProposal,
) -> TerminalReconciliationReceipt:
    """Submit one bounded proposal through the injected Plane seam."""

    return _reconcile_terminal(port, proposal)


def _validate_terminal_evidence(
    *,
    run: RunSnapshot,
    invocation: InvocationEnvelope,
    kind: str,
    proposal: TerminalProposal,
    stream: EventStream,
    cancellation_receipt: CancellationAuthorityReceipt | None,
    allow_untrusted_cancellation: bool = False,
) -> None:
    if (
        proposal.run_id != run.run_id
        or proposal.invocation_id != invocation.invocation_id
        or proposal.actor_ref != run.actor_ref
        or proposal.workspace_ref != run.workspace_ref
        or proposal.snapshot_digest != run.digest()
    ):
        raise BindingError("terminal proposal is not bound to the runtime snapshot")
    if proposal.kind != kind:
        raise BindingError("terminal proposal kind does not match the exit")
    if proposal.final_sequence != stream.last_sequence:
        raise ContractError("terminal proposal final sequence does not match finalized evidence")
    known_events = {event.event_id: event for event in stream.events}
    if any(event_id not in known_events for event_id in proposal.evidence_event_ids):
        raise ContractError("terminal proposal contains forged event evidence")
    known_receipts = {
        receipt_ref
        for event in stream.events
        for receipt_ref in (getattr(event.body, "receipt_ref", None),)
        if receipt_ref is not None
    }
    allowed_receipts = set(known_receipts)
    if cancellation_receipt is not None:
        allowed_receipts.add(cancellation_receipt.receipt_ref)
    if any(receipt_ref not in allowed_receipts for receipt_ref in proposal.evidence_receipt_refs):
        raise ContractError("terminal proposal contains forged receipt evidence")
    if kind == "completed" and proposal.outcome_proposal is None:
        raise ContractError("completed requires one finalized outcome proposal")
    if kind == "completed":
        outcome = proposal.outcome_proposal
        assert outcome is not None
        outcome_event = known_events.get(outcome.event_id)
        if outcome_event is None or not isinstance(outcome_event.body, OutcomeSubmissionObserved):
            raise ContractError("completed outcome evidence does not bind to an outcome event")
        if (
            outcome_event.body.submission_ref != outcome.submission_ref
            or outcome_event.body.content != outcome.content
            or outcome_event.body.receipt_ref != outcome.proposal_receipt_ref
        ):
            raise ContractError("completed outcome evidence was forged")
    if kind == "waiting_for_input" and proposal.input_request_proposal is None:
        raise ContractError("waiting_for_input requires one finalized input proposal")
    if kind == "waiting_for_input":
        input_request = proposal.input_request_proposal
        assert input_request is not None
        input_event = known_events.get(input_request.event_id)
        if input_event is None or not isinstance(input_event.body, InputRequestObserved):
            raise ContractError("waiting evidence does not bind to an input event")
        if (
            input_event.body.request_ref != input_request.request_ref
            or input_event.body.prompt != input_request.prompt
            or input_event.body.receipt_ref != input_request.proposal_receipt_ref
        ):
            raise ContractError("waiting evidence was forged")
    if kind == "cancelled":
        if cancellation_receipt is None:
            if not allow_untrusted_cancellation or proposal.cancellation_receipt is not None:
                raise ContractError("cancelled requires host-owned cancellation state")
        elif proposal.cancellation_receipt != cancellation_receipt:
            raise BindingError("terminal cancellation authority does not match host state")
    for artifact in proposal.artifact_proposals:
        artifact_event = known_events.get(artifact.event_id)
        if artifact_event is None or not isinstance(artifact_event.body, ArtifactObserved):
            raise ContractError("artifact proposal does not bind to an artifact event")
        if (
            artifact_event.body.artifact_ref != artifact.artifact_ref
            or artifact_event.body.digest != artifact.digest
        ):
            raise ContractError("artifact evidence was forged")
    for message in proposal.message_proposals:
        message_event = known_events.get(message.event_id)
        if message_event is None or not isinstance(message_event.body, MessageProposalObserved):
            raise ContractError("message proposal does not bind to a message event")
        if (
            message_event.body.message_ref != message.message_ref
            or message_event.body.content != message.content
            or message_event.body.proposal_receipt_ref != message.proposal_receipt_ref
        ):
            raise ContractError("message evidence was forged")


def validate_terminal_proposal(
    *,
    run: RunSnapshot,
    invocation: InvocationEnvelope,
    kind: str,
    proposal: TerminalProposal,
    stream: EventStream,
) -> None:
    """Validate a child proposal without giving it a reconciliation port."""

    if proposal.kind == "cancelled" and proposal.cancellation_receipt is not None:
        raise ChildCancellationProposalRejected(
            "child cancellation proposals cannot carry cancellation authority"
        )

    _validate_terminal_evidence(
        run=run,
        invocation=invocation,
        kind=kind,
        proposal=proposal,
        stream=stream,
        cancellation_receipt=None,
        allow_untrusted_cancellation=proposal.kind == "cancelled",
    )


def _return_terminal(
    *,
    port: TerminalReconciliationPort | None,
    run: RunSnapshot,
    invocation: InvocationEnvelope,
    stream: EventStream,
    exit_value: RuntimeExit,
    cancellation_receipt: CancellationAuthorityReceipt | None = None,
    outcome_proposal: OutcomeProposal | None = None,
    input_request_proposal: InputRequestProposal | None = None,
    artifact_proposals: Iterable[ArtifactProposal] = (),
    message_proposals: Iterable[MessageProposal] = (),
    proposal_sink: TerminalProposalSink | None = None,
) -> RuntimeExit:
    proposal = _terminal_proposal(
        run=run,
        invocation=invocation,
        exit_value=exit_value,
        stream=stream,
        outcome_proposal=outcome_proposal,
        input_request_proposal=input_request_proposal,
        cancellation_receipt=cancellation_receipt,
        artifact_proposals=artifact_proposals,
        message_proposals=message_proposals,
    )
    _validate_terminal_evidence(
        run=run,
        invocation=invocation,
        kind=exit_value.kind,
        proposal=proposal,
        stream=stream,
        cancellation_receipt=cancellation_receipt,
        allow_untrusted_cancellation=(
            proposal_sink is not None and proposal.kind == "cancelled"
        ),
    )
    if proposal_sink is not None:
        if port is not None:
            raise ContractError("terminal proposal cannot use host reconciliation and proposal-only output")
        proposal_sink(proposal)
        return exit_value
    receipt = _reconcile_terminal(port, proposal)
    if not receipt.accepted or not receipt.legal_transition:
        raise TerminalReconciliationRejected(receipt)
    return exit_value


def _return_terminal_safely(
    *,
    port: TerminalReconciliationPort | None,
    run: RunSnapshot,
    invocation: InvocationEnvelope,
    stream: EventStream,
    exit_value: RuntimeExit,
    cancellation_receipt: CancellationAuthorityReceipt | None = None,
    outcome_proposal: OutcomeProposal | None = None,
    input_request_proposal: InputRequestProposal | None = None,
    artifact_proposals: Iterable[ArtifactProposal] = (),
    message_proposals: Iterable[MessageProposal] = (),
    terminal_rejection_sink: _TerminalRejectionSink | None = None,
    proposal_sink: TerminalProposalSink | None = None,
) -> RuntimeExit:
    """Handle validated terminal rejection or convert wire overflow to failure."""

    try:
        return _return_terminal(
            port=port,
            run=run,
            invocation=invocation,
            stream=stream,
            exit_value=exit_value,
            cancellation_receipt=cancellation_receipt,
            outcome_proposal=outcome_proposal,
            input_request_proposal=input_request_proposal,
            artifact_proposals=artifact_proposals,
            message_proposals=message_proposals,
            proposal_sink=proposal_sink,
        )
    except TerminalReconciliationRejected as exc:
        if terminal_rejection_sink is None:
            raise
        terminal_rejection_sink(exc)
        return exit_value
    except BoundsError:
        return _return_terminal(
            port=port,
            run=run,
            invocation=invocation,
            stream=stream,
            exit_value=RuntimeExit(
                kind="failed",
                final_sequence=stream.last_sequence,
                failure=RuntimeFailure(
                    code="ingestion_bounds",
                    message="runtime observation limits exceeded; reconciliation is required",
                    retryable=False,
                ),
            ),
            terminal_rejection_sink=terminal_rejection_sink,
            proposal_sink=proposal_sink,
        )


def _cancellation_receipt(
    *,
    authority: CancellationAuthority | None,
    run: RunSnapshot,
    invocation: InvocationEnvelope,
) -> CancellationAuthorityReceipt:
    if authority is None:
        raise ContractError("cancelled execution requires an injected cancellation authority")
    receipt = authority.validate_cancellation(run=run, invocation=invocation)
    if not isinstance(receipt, CancellationAuthorityReceipt):
        raise ContractError("cancellation authority returned an invalid receipt")
    expected_idempotency_key = f"cancel:{run.run_id}:{invocation.invocation_id}"
    if (
        receipt.run_id != run.run_id
        or receipt.invocation_id != invocation.invocation_id
        or receipt.resource_ref != invocation.cancellation_ref
        or receipt.actor_ref != run.actor_ref
        or receipt.workspace_ref != run.workspace_ref
        or receipt.snapshot_digest != run.digest()
        or receipt.kind != "cancellation"
        or receipt.idempotency_key != expected_idempotency_key
        or receipt.receipt_ref != f"cancel-receipt:{invocation.invocation_id}"
        or receipt.gateway_receipt_ref != f"gateway:{expected_idempotency_key}"
        or receipt.audit_ref != f"audit:{expected_idempotency_key}"
    ):
        raise BindingError("cancellation receipt is not fully bound to this invocation")
    return receipt


def build_host_cancellation_proposal(
    *,
    run: RunSnapshot,
    invocation: InvocationEnvelope,
    cancellation_authority: CancellationAuthority,
    final_sequence: int = 0,
) -> TerminalProposal:
    """Synthesize a cancellation proposal only from trusted host state."""

    if isinstance(final_sequence, bool) or not isinstance(final_sequence, int) or final_sequence < 0:
        raise ContractError("host cancellation final sequence must be a non-negative integer")
    receipt = _cancellation_receipt(
        authority=cancellation_authority,
        run=run,
        invocation=invocation,
    )
    stream = EventStream(run_id=run.run_id, invocation_id=invocation.invocation_id)
    proposal = _terminal_proposal(
        run=run,
        invocation=invocation,
        exit_value=RuntimeExit(kind="cancelled", final_sequence=final_sequence),
        stream=stream,
        cancellation_receipt=receipt,
    )
    _validate_terminal_evidence(
        run=run,
        invocation=invocation,
        kind="cancelled",
        proposal=proposal,
        stream=stream,
        cancellation_receipt=receipt,
    )
    return proposal


def execute(
    *,
    run: RunSnapshot,
    invocation: InvocationEnvelope,
    host: RuntimeHost | None = None,
    emit: EventSink | None = None,
    cancellation: CancellationSignal | None = None,
    cancellation_authority: CancellationAuthority | None = None,
    kernel: KernelPort | None = None,
    lease_authority: CanonicalLeaseAuthority | None = None,
    lease_binding: CanonicalLeaseBinding | None = None,
    checkpoint_authority: CheckpointAuthority | None = None,
    checkpoint_attestation: CheckpointAttestation | None = None,
    terminal_port: TerminalReconciliationPort | None = None,
    execution_phase: ExecutionPhase | None = None,
    _terminal_rejection_sink: _TerminalRejectionSink | None = None,
    terminal_proposal_sink: TerminalProposalSink | None = None,
) -> RuntimeExit:
    """Execute exactly one invocation through the replaceable kernel port.

    The adapter validates binding and cumulative budget at entry, translates
    kernel-neutral observations into bounded runtime events, and returns one
    immutable terminal exit.  It does not persist state or mutate ``run``.
    """

    _validate_inputs(run, invocation)
    cancellation = cancellation if cancellation is not None else NeverCancelled()
    try:
        initially_cancelled = cancellation.is_cancelled()
    except Exception as exc:
        # A broken injected signal is a dependency failure.  Keep the original
        # exception private while preserving the existing execution-failure
        # classification at the service boundary.
        raise ContractError("cancellation signal could not be evaluated") from exc
    if not isinstance(initially_cancelled, bool):
        raise ContractError("cancellation signal must return a boolean")
    if lease_authority is None or lease_binding is None:
        raise LeaseError("execution requires an injected canonical lease authority and host binding")
    if terminal_port is None and terminal_proposal_sink is None:
        raise ContractError(
            "execution requires an injected terminal reconciliation port or proposal sink"
        )
    if terminal_port is not None and terminal_proposal_sink is not None:
        raise ContractError("execution cannot reconcile and emit a proposal in the same path")
    if terminal_proposal_sink is not None and cancellation_authority is not None:
        raise RuntimeConfigurationError(
            "proposal-only execution cannot accept a cancellation authority"
        )
    if terminal_proposal_sink is None and initially_cancelled and cancellation_authority is None:
        raise RuntimeConfigurationError(
            "signalled cancellation requires an injected cancellation authority"
        )
    if invocation.checkpoint_ref is not None and (
        checkpoint_authority is None or checkpoint_attestation is None
    ):
        raise ContractError(
            "checkpoint continuation requires an injected checkpoint authority and attestation"
        )
    if invocation.checkpoint_ref is None and (
        checkpoint_authority is not None or checkpoint_attestation is not None
    ):
        raise ContractError("checkpoint authority data is not allowed for an initial invocation")
    lease_authority.validate_lease(run=run, invocation=invocation, binding=lease_binding)
    if invocation.checkpoint_ref is not None:
        assert checkpoint_authority is not None
        assert checkpoint_attestation is not None
        checkpoint_authority.claim_checkpoint(
            run=run,
            invocation=invocation,
            attestation=checkpoint_attestation,
        )
    if execution_phase is not None:
        execution_phase.execution_started = True
    stream = EventStream(
        run_id=run.run_id,
        invocation_id=invocation.invocation_id,
        sink=emit,
        expected_causation_ref=invocation.causation_ref,
    )
    if initially_cancelled:
        cancellation_receipt = (
            None
            if terminal_proposal_sink is not None
            else _cancellation_receipt(
                authority=cancellation_authority,
                run=run,
                invocation=invocation,
            )
        )
        return _return_terminal_safely(
            port=terminal_port,
            run=run,
            invocation=invocation,
            stream=stream,
            exit_value=RuntimeExit(kind="cancelled", final_sequence=stream.last_sequence),
            cancellation_receipt=cancellation_receipt,
            terminal_rejection_sink=_terminal_rejection_sink,
            proposal_sink=terminal_proposal_sink,
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
    transcript_bytes = 0
    pending_input_requests: set[str] = set()
    outcome_submissions: set[str] = set()
    artifact_proposals: list[ArtifactProposal] = []
    message_proposals: list[MessageProposal] = []
    outcome_proposal: OutcomeProposal | None = None
    input_request_proposal: InputRequestProposal | None = None
    used = RuntimeBudget()

    def on_observation(observation: KernelObservation) -> None:
        nonlocal used, outcome_proposal, input_request_proposal, transcript_bytes
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
            if transcript.transcript_ref not in transcripts:
                if len(transcripts) >= MAX_TRANSCRIPT_OBSERVATIONS:
                    raise BoundsError("transcript proposal count exceeded")
                transcript_size = len(transcript.text.encode("utf-8"))
                if transcript_bytes + transcript_size > MAX_TRANSCRIPT_BYTES:
                    raise BoundsError("transcript proposal bytes exceeded")
                transcript_bytes += transcript_size
            transcripts[transcript.transcript_ref] = transcript
            _next_event(stream, invocation, transcript)
            return
        if observation.kind == "input_request":
            if observation.request_ref is None or observation.message is None:
                raise ContractError("input request requires a reference and prompt")
            if observation.request_ref in pending_input_requests:
                raise SequenceError("input request reference was reused in one invocation")
            if len(pending_input_requests) >= MAX_INPUT_PROPOSALS:
                raise BoundsError("input proposal count exceeded")
            pending_input_requests.add(observation.request_ref)
            event = _next_event(
                stream,
                invocation,
                InputRequestObserved(
                    observation.request_ref,
                    observation.message,
                    f"proposal:{invocation.invocation_id}:input:{observation.request_ref}",
                ),
            )
            input_request_proposal = InputRequestProposal(
                request_ref=observation.request_ref,
                prompt=observation.message,
                event_id=event.event_id,
                proposal_receipt_ref=event.body.receipt_ref,  # type: ignore[union-attr]
            )
            return
        if observation.kind == "artifact":
            if observation.artifact_ref is None or observation.artifact_digest is None:
                raise ContractError("artifact observation requires a reference and digest")
            event = _next_event(
                stream,
                invocation,
                ArtifactObserved(
                    observation.artifact_ref,
                    observation.artifact_digest,
                    f"proposal:{invocation.invocation_id}:artifact:{observation.artifact_ref}",
                ),
            )
            artifact_proposals.append(
                ArtifactProposal(
                    artifact_ref=observation.artifact_ref,
                    digest=observation.artifact_digest,
                    event_id=event.event_id,
                )
            )
            return
        if observation.kind == "outcome_submission":
            if observation.submission_ref is None:
                raise ContractError("outcome submission requires a submission reference")
            if observation.submission_ref in outcome_submissions:
                raise SequenceError("outcome submission reference was reused in one invocation")
            outcome_submissions.add(observation.submission_ref)
            content = observation.outcome_content
            matching: TranscriptObserved | None = None
            if not content:
                matching = transcripts.get(observation.transcript_ref or "")
            if matching is None and transcripts:
                matching = next(reversed(transcripts.values()))
            if not content:
                content = matching.text if matching is not None else None
            if not content:
                raise ContractError("outcome submission requires explicit outcome content")
            event = _next_event(
                stream,
                invocation,
                OutcomeSubmissionObserved(
                    submission_ref=observation.submission_ref,
                    receipt_ref=(
                        f"proposal:{invocation.invocation_id}:outcome:{observation.submission_ref}"
                    ),
                    content=content,
                ),
            )
            outcome_proposal = OutcomeProposal(
                submission_ref=observation.submission_ref,
                content=content,
                event_id=event.event_id,
                proposal_receipt_ref=event.body.receipt_ref,  # type: ignore[union-attr]
            )
            return
        if observation.kind == "publication_request":
            if observation.transcript_ref is None:
                raise ContractError("publication request requires a transcript reference")
            try:
                transcript = transcripts[observation.transcript_ref]
            except KeyError as exc:
                raise ContractError("publication request must reference prior transcript evidence") from exc
            if len(message_proposals) >= MAX_MESSAGE_PROPOSALS:
                raise BoundsError("message proposal count exceeded")
            event = _next_event(
                stream,
                invocation,
                MessageProposalObserved(
                    message_ref=transcript.transcript_ref,
                    transcript_ref=transcript.transcript_ref,
                    content=transcript.text,
                    proposal_receipt_ref=(
                        f"proposal:{invocation.invocation_id}:message:{transcript.transcript_ref}"
                    ),
                ),
            )
            body = event.body
            assert isinstance(body, MessageProposalObserved)
            message_proposals.append(
                MessageProposal(
                    message_ref=body.message_ref,
                    content=body.content,
                    event_id=event.event_id,
                    proposal_receipt_ref=body.proposal_receipt_ref,
                )
            )
            return
        raise ContractError(f"unsupported kernel observation: {observation.kind!r}")

    try:
        result = kernel.dispatch(request, on_observation, cancellation)
    except CancellationRequested:
        cancellation_receipt = (
            None
            if terminal_proposal_sink is not None
            else _cancellation_receipt(
                authority=cancellation_authority,
                run=run,
                invocation=invocation,
            )
        )
        return _return_terminal_safely(
            port=terminal_port,
            run=run,
            invocation=invocation,
            stream=stream,
            exit_value=RuntimeExit(kind="cancelled", final_sequence=stream.last_sequence),
            cancellation_receipt=cancellation_receipt,
            artifact_proposals=artifact_proposals,
            message_proposals=message_proposals,
            terminal_rejection_sink=_terminal_rejection_sink,
            proposal_sink=terminal_proposal_sink,
        )
    except BoundsError:
        # Ingestion limits are an invocation failure, not permission to keep
        # consuming kernel output.  Reconcile one minimal bounded failure
        # proposal through the same atomic terminal slot.
        return _return_terminal_safely(
            port=terminal_port,
            run=run,
            invocation=invocation,
            stream=stream,
            exit_value=RuntimeExit(
                kind="failed",
                final_sequence=stream.last_sequence,
                failure=RuntimeFailure(
                    code="ingestion_bounds",
                    message="runtime observation limits exceeded; reconciliation is required",
                    retryable=False,
                ),
            ),
            terminal_rejection_sink=_terminal_rejection_sink,
            proposal_sink=terminal_proposal_sink,
        )
    if cancellation.is_cancelled():
        cancellation_receipt = (
            None
            if terminal_proposal_sink is not None
            else _cancellation_receipt(
                authority=cancellation_authority,
                run=run,
                invocation=invocation,
            )
        )
        return _return_terminal_safely(
            port=terminal_port,
            run=run,
            invocation=invocation,
            stream=stream,
            exit_value=RuntimeExit(kind="cancelled", final_sequence=stream.last_sequence),
            cancellation_receipt=cancellation_receipt,
            artifact_proposals=artifact_proposals,
            message_proposals=message_proposals,
            terminal_rejection_sink=_terminal_rejection_sink,
            proposal_sink=terminal_proposal_sink,
        )
    if result.terminal_kind == "waiting_for_input" and not pending_input_requests:
        raise ContractError("waiting_for_input requires a visible authorized input request")
    if result.terminal_kind != "waiting_for_input" and pending_input_requests:
        raise ContractError("terminal exit cannot leave an unresolved input request")
    if result.terminal_kind == "completed" and len(outcome_submissions) != 1:
        raise ContractError("completed requires exactly one untrusted outcome proposal")
    if result.terminal_kind == "cancelled":
        cancellation_receipt = (
            None
            if terminal_proposal_sink is not None
            else _cancellation_receipt(
                authority=cancellation_authority,
                run=run,
                invocation=invocation,
            )
        )
        return _return_terminal_safely(
            port=terminal_port,
            run=run,
            invocation=invocation,
            stream=stream,
            exit_value=RuntimeExit(kind="cancelled", final_sequence=stream.last_sequence),
            cancellation_receipt=cancellation_receipt,
            artifact_proposals=artifact_proposals,
            message_proposals=message_proposals,
            terminal_rejection_sink=_terminal_rejection_sink,
            proposal_sink=terminal_proposal_sink,
        )
    return _return_terminal_safely(
        port=terminal_port,
        run=run,
        invocation=invocation,
        stream=stream,
        exit_value=RuntimeExit(
            kind=result.terminal_kind,
            final_sequence=stream.last_sequence,
            failure=result.failure,
        ),
        outcome_proposal=outcome_proposal,
        input_request_proposal=input_request_proposal,
        artifact_proposals=artifact_proposals,
        message_proposals=message_proposals,
        terminal_rejection_sink=_terminal_rejection_sink,
        proposal_sink=terminal_proposal_sink,
    )


def execute_proposal_only(
    *,
    run: RunSnapshot,
    invocation: InvocationEnvelope,
    emit: EventSink | None = None,
    cancellation: CancellationSignal | None = None,
    kernel: KernelPort,
    lease_authority: CanonicalLeaseAuthority,
    lease_binding: CanonicalLeaseBinding,
    checkpoint_authority: CheckpointAuthority | None = None,
    checkpoint_attestation: CheckpointAttestation | None = None,
    execution_phase: ExecutionPhase | None = None,
    proposal_sink: TerminalProposalSink,
) -> RuntimeExit:
    """Run the untrusted child path and emit one terminal proposal only.

    The child can validate host-supplied lease/checkpoint inputs and translate
    bounded kernel observations, but it has no terminal port.  Consequently it
    cannot construct or return a Plane reconciliation receipt.
    """

    if kernel is None or proposal_sink is None:
        raise RuntimeConfigurationError(
            "proposal-only execution requires an injected kernel and proposal sink"
        )
    return execute(
        run=run,
        invocation=invocation,
        emit=emit,
        cancellation=cancellation,
        kernel=kernel,
        lease_authority=lease_authority,
        lease_binding=lease_binding,
        checkpoint_authority=checkpoint_authority,
        checkpoint_attestation=checkpoint_attestation,
        terminal_port=None,
        execution_phase=execution_phase,
        terminal_proposal_sink=proposal_sink,
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
    outcome_submission_requested: bool = True
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
        if self.plan.outcome_submission_requested and self.plan.terminal_kind == "completed":
            emit(
                KernelObservation(
                    kind="outcome_submission",
                    submission_ref="submission:fake",
                )
            )
        if self.plan.publication_requested:
            emit(KernelObservation(kind="publication_request", transcript_ref=self.plan.transcript_ref))
        if cancellation.is_cancelled():
            return KernelResult(terminal_kind="cancelled")
        return KernelResult(terminal_kind=self.plan.terminal_kind)


class FixtureTerminalReconciliationPort:
    """Explicit atomic fixture for tests only.

    The fixture intentionally has no separate outcome, artifact, or input
    mutation methods.  All visible terminal product events are applied while
    holding the same slot lock as the idempotency decision.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._slots: dict[str, tuple[TerminalProposal, TerminalReconciliationReceipt]] = {}
        self.accepted: list[str] = []
        self.product_events: list[tuple[str, str]] = []
        self.product_event_payloads: list[dict[str, object]] = []
        self.proposals: list[TerminalProposal] = []
        self.receipts: list[TerminalReconciliationReceipt] = []

    def _receipt(
        self,
        proposal: TerminalProposal,
        *,
        accepted: bool,
        legal_transition: bool,
        product_receipts: tuple[ProductReceipt, ...] = (),
    ) -> TerminalReconciliationReceipt:
        key = proposal.idempotency_key
        return TerminalReconciliationReceipt(
            receipt_ref=f"terminal-receipt:{key}",
            run_id=proposal.run_id,
            invocation_id=proposal.invocation_id,
            kind=proposal.kind,
            idempotency_key=key,
            accepted=accepted,
            legal_transition=legal_transition,
            proofs=(
                tuple(
                    TerminalProof(
                        proof_kind=proof_kind,
                        proof_ref=_terminal_proof_ref(proof_kind, key),
                        resource_ref=_terminal_proof_resource(proof_kind, key),
                        run_id=proposal.run_id,
                        invocation_id=proposal.invocation_id,
                        actor_ref=proposal.actor_ref,
                        workspace_ref=proposal.workspace_ref,
                        snapshot_digest=proposal.snapshot_digest,
                        terminal_slot=key,
                        terminal_kind=proposal.kind,
                        proposal_digest=proposal.digest(),
                    )
                    for proof_kind in (
                        "operation_attempt",
                        "application",
                        "gateway",
                        "audit",
                        "product_event",
                    )
                )
                if accepted and legal_transition
                else ()
            ),
            product_receipts=product_receipts if accepted and legal_transition else (),
        )

    def _product_receipt(
        self,
        proposal: TerminalProposal,
        resource_ref: str,
        kind: str,
    ) -> ProductReceipt:
        receipt_ref, idempotency_key = product_proof_identity(
            proof_kind=kind,
            product_kind=kind,
            resource_ref=resource_ref,
            run_id=proposal.run_id,
            invocation_id=proposal.invocation_id,
            actor_ref=proposal.actor_ref,
            workspace_ref=proposal.workspace_ref,
            snapshot_digest=proposal.snapshot_digest,
            terminal_slot=proposal.idempotency_key,
            terminal_kind=proposal.kind,
            proposal_digest=proposal.digest(),
        )
        return ProductReceipt(
            resource_ref=resource_ref,
            receipt_ref=receipt_ref,
            run_id=proposal.run_id,
            invocation_id=proposal.invocation_id,
            idempotency_key=idempotency_key,
            kind=kind,
            actor_ref=proposal.actor_ref,
            workspace_ref=proposal.workspace_ref,
            snapshot_digest=proposal.snapshot_digest,
            terminal_slot=proposal.idempotency_key,
            proposal_digest=proposal.digest(),
            terminal_kind=proposal.kind,
        )

    def reconcile_terminal(self, proposal: TerminalProposal) -> TerminalReconciliationReceipt:
        with self._lock:
            key = proposal.idempotency_key
            prior = self._slots.get(key)
            if prior is not None:
                prior_proposal, prior_receipt = prior
                if prior_proposal == proposal:
                    return prior_receipt
                return self._receipt(proposal, accepted=False, legal_transition=False)

            product_receipts: list[ProductReceipt] = []
            if proposal.kind == "completed":
                assert proposal.outcome_proposal is not None
                product_receipts.append(
                    self._product_receipt(
                        proposal,
                        proposal.outcome_proposal.submission_ref,
                        "outcome_submission",
                    )
                )
            elif proposal.kind == "waiting_for_input":
                assert proposal.input_request_proposal is not None
                product_receipts.append(
                    self._product_receipt(
                        proposal,
                        proposal.input_request_proposal.request_ref,
                        "input_request",
                    )
                )
            elif proposal.kind == "cancelled":
                assert proposal.cancellation_receipt is not None
                product_receipts.append(
                    self._product_receipt(
                        proposal,
                        f"product-event:{key}",
                        "terminal_event",
                    )
                )
            else:
                product_receipts.append(
                    self._product_receipt(
                        proposal,
                        f"product-event:{key}",
                        "terminal_event",
                    )
                )
            artifact_items = proposal.artifact_proposals
            product_receipts.extend(
                self._product_receipt(
                    proposal, item.artifact_ref, "artifact"
                )
                for item in artifact_items
            )
            product_receipts.extend(
                self._product_receipt(proposal, item.message_ref, "message")
                for item in proposal.message_proposals
            )
            product_event_kind = {
                "completed": "OutcomeSubmission",
                "waiting_for_input": "InputRequest",
                "cancelled": "Cancellation",
                "failed": "TerminalFailure",
                "blocked": "TerminalBlock",
            }[proposal.kind]
            product_event_payload: dict[str, object] = {
                "kind": product_event_kind,
                "runId": proposal.run_id,
                "invocationId": proposal.invocation_id,
                "artifactRefs": [item.artifact_ref for item in artifact_items],
                "messageRefs": [item.message_ref for item in proposal.message_proposals],
            }
            if proposal.outcome_proposal is not None:
                product_event_payload.update(
                    {
                        "submissionRef": proposal.outcome_proposal.submission_ref,
                        "content": proposal.outcome_proposal.content,
                    }
                )
            elif proposal.input_request_proposal is not None:
                product_event_payload.update(
                    {
                        "requestRef": proposal.input_request_proposal.request_ref,
                        "prompt": proposal.input_request_proposal.prompt,
                    }
                )
            elif proposal.cancellation_receipt is not None:
                product_event_payload["cancellationRef"] = proposal.cancellation_receipt.resource_ref
            receipt = self._receipt(
                proposal,
                accepted=True,
                legal_transition=True,
                product_receipts=tuple(product_receipts),
            )
            self._slots[key] = (proposal, receipt)
            self.accepted.append(key)
            self.product_events.append((key, proposal.kind))
            self.product_event_payloads.append(product_event_payload)
            self.proposals.append(proposal)
            self.receipts.append(receipt)
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
    run: RunSnapshot | None = None,
    run_id: str | None = None,
    invocation_id: str | None = None,
    actor_ref: str | None = None,
    workspace_ref: str | None = None,
    snapshot_digest: str | None = None,
    final_sequence: int,
    reason: str = "runtime process exited before returning an exit",
) -> TerminalReconciliationReceipt:
    """Submit trusted supervisor process-death evidence through the Plane port."""

    if run is not None:
        run_id = run.run_id
        actor_ref = run.actor_ref
        workspace_ref = run.workspace_ref
        snapshot_digest = run.digest()
    if not all((run_id, invocation_id, actor_ref, workspace_ref, snapshot_digest)):
        raise ContractError(
            "process-death reconciliation requires run, invocation, actor, workspace, and snapshot binding"
        )
    exit_value = classify_process_death(final_sequence=final_sequence, reason=reason)
    proposal = TerminalProposal(
        run_id=run_id,
        invocation_id=invocation_id,
        actor_ref=actor_ref,
        workspace_ref=workspace_ref,
        snapshot_digest=snapshot_digest,
        kind=exit_value.kind,
        final_sequence=exit_value.final_sequence,
        failure=exit_value.failure,
        source="supervisor",
    )
    return _reconcile_terminal(port, proposal)
