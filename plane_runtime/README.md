# Plane runtime adapter

`plane_runtime/` is the Hermes-side package boundary for the separately
deployed Plane Agent runtime service. The logical adapter interface is
`plane_runtime.execute`.

The adapter is the only place that may translate Plane's versioned
`RunSnapshot` and `InvocationEnvelope` into Hermes kernel calls and translate
bounded runtime observations back. Plane remains authoritative for product
state, lifecycle, publication, receipts, credentials, terminal reconciliation,
and durable recovery.

Execution requires host-injected `CanonicalLeaseAuthority` and
`CanonicalLeaseBinding` values. Continuations additionally require a complete,
single-use host `CheckpointAttestation` claim through `CheckpointAuthority`;
the envelope's lease and checkpoint references are never authorities. Runtime
observations, outcome content, artifact references, and evidence are proposal
inputs only. A completed exit carries typed outcome content/evidence, waiting
carries typed input evidence, and cancellation carries an authoritative
cancellation correlation. Every runtime exit is submitted through the
injected `TerminalReconciliationPort`, whose durable implementation must claim
one atomic `(run, invocation)` terminal slot, apply the kind-specific visible
product event (including the outcome submission), and return operation, audit,
and product receipts. A supervisor classifying process death uses the same port
through `reconcile_process_death()`.

The v1 contract contains immutable `RunSnapshot` and `InvocationEnvelope`
values, bounded `RuntimeEvent` observations/proposals, typed `TerminalProposal`
values, and one `RuntimeExit`.
Later Plane input is represented only by event references in an invocation;
budgets are supplied as remaining cumulative values. Transcript and message
proposal observations are operational evidence only: this adapter has no
product-publication host method. A typed message proposal may cross the
terminal proposal boundary, where the atomic terminal port applies it only as
part of an accepted terminal transition; any future non-terminal publication
is an explicit Plane gateway operation outside this adapter.
Terminal product mutations never use a runtime host seam.

The wire limits are deliberately finite: acceptance criteria are capped at 64
items, context/eager-operation/new-event-reference lists at 128 items, event
JSON at 16 KiB, invocation JSON at 16 KiB, and snapshot JSON at 128 KiB. The
snapshot and invocation limits measure canonical UTF-8 JSON bytes, so the
boundary is stable across transport implementations.

Per invocation, event ingestion is capped at 512 events and 256 KiB of
canonical event bytes. Transcript, artifact, input, outcome, and message
proposal categories each have bounded counts and bytes; sequence and
idempotency indexes and the optional observation tail are bounded as well.
Terminal proposals and reconciliation receipts are each capped at 128 KiB.
An accepted receipt carries exactly five typed `TerminalProof` values—operation
attempt, application, gateway, audit, and terminal product event. Each proof
binds its kind/resource and proof identity to the exact run, invocation, actor,
workspace, snapshot, terminal slot, terminal kind, and proposal digest. The
receipt also carries exact kind-specific product receipts with the same
binding. Cancellation uses a dedicated `CancellationAuthorityReceipt`; its
resource, principal, snapshot, idempotency, gateway, and audit fields are
serialized losslessly and validated before the terminal slot is reconciled.
The proposal-only child path never receives this authority and emits a
cancelled proposal without a cancellation receipt; the trusted host or
supervisor must synthesize the receipt from host-owned cancellation state.

`adapter.py` exposes the narrow `KernelPort` seam. `FakeKernel` is a
deterministic implementation for contract tests; a real Hermes implementation
can be added behind the same seam without changing Hermes core modules.
`service.py` provides a deliberately minimal one-invocation JSON-lines host,
with a finite composite request bound and a bounded byte/frame reader. Its
proposal-only child entrypoint accepts only an invocation-scoped cancellation
signal; the signal is an observation and is never treated as authority. The
trusted `serve_once` convenience path may additionally receive an independent
canonical cancellation authority and is the only service path that can emit a
host-authorized cancellation receipt.
Unexpected internal exceptions are retained only by the protected failure hook
and become a bounded generic failed proposal; if that proposal is not
accepted, the service returns status 1 so a trusted supervisor can reconcile
the visible failure:

```bash
printf '%s\n' '<request JSON>' | python3 -m plane_runtime.service --once
```

This package intentionally has no Hermes core wiring, dependencies,
configuration, Buzz integration, or chat UI. Generated code must receive no
credentials or ambient Plane authority; invocation processes and containers
are replaceable infrastructure, never run-owned state. A supervisor may use
`classify_process_death()` for local evidence and must submit the resulting
process-death proposal through the reconciliation port when a replaceable
process dies before returning an exit.

`invocation_supervisor.py` is a supervisor policy and host-reconciliation
foundation. The child protocol emits bounded observations, one untrusted
terminal proposal, and one exit; it never emits a reconciliation receipt or
Plane proof. The trusted supervisor validates the exact run, invocation,
snapshot, lease, sequence, and proposal bindings, then submits the proposal
once through its injected `TerminalReconciliationPort`. Product status is
derived only from the accepted host receipt. The current foundation has no
trusted production entrypoint, so every accepted runner attestation is
explicitly test-only and `production_completed` is permanently false.

The fixed child command uses `docker create`, `start`, and `attach` with
module-private network, namespace, privilege, mount, environment, logging,
storage, and cleanup controls. `SubprocessDockerRunner` performs bounded
daemon, image, and post-launch inspections and rejects ambiguity, but those
checks are test/integration evidence only until the real kernel/service
binding exists. Caller-owned runners are explicit test seams and cannot
produce a `production_completed` result; there is deliberately no mutable
closed-runner production path to replace. The attach client is terminated,
killed if necessary, and reaped on every collector exit before container
cleanup. Results are retained under a fixed bound only as validated private
canonical bytes in an invocation-local serialized store; caller-supplied or
mutated result objects are never trusted. Each lookup validates the binding
and integrity record, then returns a fresh immutable non-production value
view. The store is ephemeral Hermes execution state, not Plane product or
database authority.
Overflow is returned as supervisor action required without inserting past the
cap. Cleanup proves deterministic container absence after stop/kill/remove-volume
attempts.

The package does not claim a real Hermes `KernelPort`, provider credentials,
Operation Gateway, deployment, checkpoint service, or generated TypeScript
isolation. Real Docker enforcement is unproven unless a local daemon and
already-present digest-pinned image pass the inspection path; this slice does
not pull images or contact a registry. More importantly, a passing Docker
inspection alone is not the missing production entrypoint: a real Hermes
kernel/service binding and its trusted host-owned evidence path are still
absent. Current supervisor executions remain explicitly non-production and
fail closed on production-shaped runner attestations. The trusted
`serve_once` function remains a host/test convenience; fixture/demo
authorities are not command-line runtime entrypoints.

Verify the package from the repository root:

```bash
python3 -c "import plane_runtime"
python3 -m compileall -q plane_runtime
scripts/run_tests.sh tests/plane_runtime
git diff --check -- plane_runtime tests/plane_runtime
```
