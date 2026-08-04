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
`CanonicalLeaseBinding` values. Continuations additionally require a complete
host `CheckpointAttestation` and `CheckpointAuthority`; the envelope's lease
and checkpoint references are never authorities. Every runtime exit is
submitted through the injected `TerminalReconciliationPort`, which returns a
receipt, audit reference, and legal-transition result. A supervisor classifying
process death uses the same port through `reconcile_process_death()`.

The v1 contract contains immutable `RunSnapshot` and `InvocationEnvelope`
values, bounded `RuntimeEvent` observations/proposals, and one `RuntimeExit`.
Later Plane input is represented only by event references in an invocation;
budgets are supplied as remaining cumulative values. Transcript evidence is
not publication: a publication observation is emitted only after a trusted
host returns a receipt for an explicit publication action.

The wire limits are deliberately finite: acceptance criteria are capped at 64
items, context/eager-operation/new-event-reference lists at 128 items, event
JSON at 16 KiB, invocation JSON at 16 KiB, and snapshot JSON at 128 KiB. The
snapshot and invocation limits measure canonical UTF-8 JSON bytes, so the
boundary is stable across transport implementations.

`adapter.py` exposes the narrow `KernelPort` seam. `FakeKernel` is a
deterministic implementation for contract tests; a real Hermes implementation
can be added behind the same seam without changing Hermes core modules.
`service.py` provides a deliberately minimal one-invocation JSON-lines host:

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

Verify the package from the repository root:

```bash
python3 -c "import plane_runtime"
python3 -m compileall -q plane_runtime
scripts/run_tests.sh tests/plane_runtime
git diff --check -- plane_runtime tests/plane_runtime
```
