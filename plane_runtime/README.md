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

`adapter.py` exposes the narrow `KernelPort` seam. `FakeKernel` is a
deterministic implementation for contract tests; a real Hermes implementation
can be added behind the same seam without changing Hermes core modules.
`service.py` provides a deliberately minimal one-invocation JSON-lines host,
with a finite composite request bound and a bounded byte/frame reader, and
accepts an injected cancellation signal plus an independent canonical
cancellation authority. The signal is never treated as authority.
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

`invocation_supervisor.py` adds the first production-directed isolation slice.
`InvocationPolicy` accepts only a digest-pinned image and emits one fixed
`python3 -m plane_runtime.service --once` Docker command with network disabled,
read-only rootfs, no-new-privileges, all capabilities dropped, a non-root
user, bounded CPU/memory/PIDs/tmpfs, and no host mounts or ambient environment.
The injected `DockerRunner` must explicitly advertise every required
enforcement and cleanup capability; a missing capability rejects launch before
Docker is called. `InvocationSupervisor.run_once()` bounds request, frame,
stdout, stderr, and wall-time handling, always attempts stop/kill/remove for
the derived invocation container, and requires both a valid `RuntimeExit` and
an accepted terminal receipt before returning `completed`. Process death,
malformed output, timeout, cancellation, and cleanup failure are non-success
outcomes; death reconciliation is idempotent and never replays an
`outcome_unknown` result.

Verify the package from the repository root:

```bash
python3 -c "import plane_runtime"
python3 -m compileall -q plane_runtime
scripts/run_tests.sh tests/plane_runtime
git diff --check -- plane_runtime tests/plane_runtime
```
