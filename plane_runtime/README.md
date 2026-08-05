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
host-authorized cancellation receipt. Production G1 has one authority:
`plane_runtime.g1_runtime_image.bootstrap`; `service --g1-production` is
rejected unless invoked with the bootstrap's private child marker.
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
derived only from the accepted host receipt. G1 has no product-completion
claim: it returns runtime evidence frames only. `ProductionG1RuntimeRunner`
is the separate G1 production seam and uses the same fixed Docker supervisor
policy, with a host-owned credential source delivered through the bootstrap's
private three-frame handoff. `G1LocalTestRunner` is explicitly test-only.

The fixed child command uses `docker create`, `start`, and `attach` with
module-private network, namespace, privilege, mount, environment, logging,
storage, and cleanup controls. `SubprocessDockerRunner` performs bounded
daemon, image, and post-launch inspections and rejects ambiguity, but those
checks are bounded Docker enforcement evidence. The G1 production runner adds
the fixed `--g1-production` bootstrap command; the child reaches
`HermesKernelAdapter` only through the marked bootstrap child path.
Caller-owned runners remain
explicit test seams and cannot produce a product completion claim. The attach
client is terminated,
killed if necessary, and reaped on every collector exit before container
cleanup. Results are retained under a fixed bound only as validated private
canonical values inside one invocation-scoped authority process. The parent
can request only typed create/read/close operations; it receives no database,
URI, connection, digest key, row replacement, or delete path. The authority
performs canonical binding and keyed-integrity checks atomically, converges
same replays, rejects conflicting replays, and returns fresh immutable
non-production value views. Killing or replacing the authority fails closed.
This state is ephemeral Hermes execution state, not Plane product or database
authority. The parent authenticates the exact local authority source and
executable with a fresh per-authority secret, a nonce-bound challenge, and
parent/child process identities before accepting responses; every
request has a bounded incremental frame deadline and sequence-correlated
authenticated response. Callers should invoke `InvocationSupervisor.close()`
explicitly; the structural finalizer owns the original process resources and
reaps them if public handles are replaced or the supervisor is abandoned.
Overflow is returned as supervisor action required without inserting past the
cap. Cleanup proves deterministic container absence after stop/kill/remove-volume
attempts.

The `HermesKernelAdapter` now exposes one optional, invocation-scoped
`PlaneHostPort`.  Its dynamic `plane_runtime` toolset reuses Hermes' registry,
tool dispatch, bounded result handling, and `execute_code` parent-RPC sandbox:
`plane_operation` covers discovery/read/mutation and the same callback from
code execution, while `plane_publish` is the only explicit publication/outcome
action.  The port accepts only the versioned, credential-free host request and
result vocabulary; it derives correlation/idempotency references from the
trusted invocation and fails closed on malformed, conflicting, unavailable,
cancelled, or over-budget calls.  Ordinary final text still emits only
`transcript_evidence_observed`; it never invokes publication.  Plane remains
the authority for catalog disclosure, authorization, application, receipts,
and durable product state.

The production one-shot service binds the concrete `UnixSocketPlaneHostPort`
when the trusted supervisor supplies `--plane-host-socket <absolute-path>`.
The single production entrypoint is:

```bash
python3 -m plane_runtime.g1_runtime_image.bootstrap --once --g1-production \
  [--plane-host-socket <absolute-path>]
```

Its stdin is exactly three canonical JSON-lines frames followed by EOF, in
this order: dispatch control (`dispatch-control/v1`, bounded model-call
allowance), credential control (`credential-control/v1`, bounded host-only
credential map), and the ordinary `{"invocation","run"}` G1 request. The
bootstrap validates the order and bounds before spawning the child, then
forwards the same private frames to `service --once --g1-production
--g1-bootstrap-child`. The child clears the in-memory credential map after
the invocation. The dispatch and credential frames never appear in stdout,
stderr, the model prompt, tool input/results, transcript evidence, or the
request fingerprint.

The bootstrap forwards this dedicated host-socket argument to the child; it is
not read from the runtime request, environment, model prompt, tool result,
transcript, event, or generated code. The client opens a fresh local `AF_UNIX`
stream for each callback and closes it after the one canonical JSON-lines
response, so the endpoint and connection are invocation-scoped.

The trusted parent requests in-process cancellation by sending `SIGUSR1` to
the bootstrap process. The bootstrap forwards it to the child, whose bounded
adapter watcher calls `AIAgent.interrupt()` within its 50 ms polling cadence.
Hermes emits a typed `cancelled` runtime exit; process termination remains the
supervisor fallback and no cancellation receipt is authored by Hermes.

The host callback wire contract is exactly `plane.agent-runtime/v1`:

```text
request:  protocol, runId, invocationId, correlationId, action,
          operationRef, input, source, requestRef, idempotencyKey
response: protocol, requestRef, correlationId, idempotencyKey, status,
          replayed, output, optional errorCode/errorMessage, optional publication
```

Both sides use sorted-key, compact UTF-8 JSON with one object and one trailing
newline per connection. Hermes rejects duplicate or unknown response keys,
non-canonical JSON, missing/truncated/oversized frames, peer closure,
timeouts, cancellation, malformed status/error shapes, and any response whose
request, correlation, or idempotency binding differs. It never retries or
falls back to Plane REST/MCP or ambient credentials. A successful explicit
publication must carry the existing bounded Plane gateway receipt shape;
ordinary final model text remains transcript evidence only.

This seam does not claim a TypeScript/Deno runner: Hermes' existing restricted
code execution is Python PTC behind the same parent-RPC callback.  Plane's
`plane.typescript.compose@1` supervisor can use the same logical host request
shape when its runtime service is wired to the host port; generated code still
receives no credential, actor, workspace, or idempotency controls.

The package does not claim an Operation Gateway deployment, checkpoint
service, or generated TypeScript isolation. Real Docker enforcement is
unproven unless a local daemon and already-present digest-pinned image pass
the inspection path; this slice does not pull images or contact a registry.
The production G1 seam is wired to Hermes, but successful model execution
also requires an existing digest-pinned image containing this service and its
Hermes dependencies. The private bootstrap handoff keeps provider credentials
out of the snapshot, envelope, child environment, command arguments, logs,
events, and artifacts. The trusted `serve_once` function remains a host/test
convenience; fixture/demo authorities are not command-line runtime entrypoints.
The bootstrap frames are not part of the request fingerprint, ledger, retained
frames, or event stream; malformed, duplicate, oversized, and post-request
control frames fail before the Hermes service starts.

For a local no-install G1 execution probe, build
`plane_runtime/Dockerfile.g1` with the already-cached digest-pinned
`company-runner-upstream-runtime` base. Its scratch final stage strips the
base image's ambient environment while retaining its Python/Hermes files;
the resulting local image digest must be passed unchanged as the production
`InvocationPolicy.image` value and attested before launch. The image's
dotenv shim is deliberately a no-op: the only credential path is the private
bootstrap control frame.

Verify the package from the repository root:

```bash
python3 -c "import plane_runtime"
python3 -m compileall -q plane_runtime
scripts/run_tests.sh tests/plane_runtime
git diff --check -- plane_runtime tests/plane_runtime
```
