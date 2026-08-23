# Plane runtime adapter

`plane_runtime/` is the Hermes-side package boundary for the separately
deployed Plane Agent runtime service. The adapter execution seam is
`plane_runtime.adapter.execute`.

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
`service.py` accepts only the private `--g1-bootstrap-child` entrypoint used by
`plane_runtime.g1_runtime_image.bootstrap`; direct and proposal-only service
entrypoints are intentionally unsupported.

This package intentionally has no Hermes core wiring, dependencies,
configuration, Buzz integration, or chat UI. Generated code must receive no
credentials or ambient Plane authority; invocation processes and containers
are replaceable infrastructure, never run-owned state. A supervisor may use
`classify_process_death()` for local evidence and must submit the resulting
process-death proposal through the reconciliation port when a replaceable
process dies before returning an exit.

The `HermesKernelAdapter` now exposes one optional, invocation-scoped
`PlaneHostPort`.  Its dynamic `plane_runtime` toolset reuses Hermes' registry,
tool dispatch, bounded result handling, and `plane_execute_typescript` restricted TypeScript host sandbox. Hermes's native `execute_code` remains a separate non-Plane tool:
`plane_operation` covers discovery/read/mutation and the same callback from
`plane_execute_typescript`, while `plane_publish` is the only explicit publication/outcome
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
  [--plane-host-socket <absolute-path>] \
  [--provider-relay-socket <absolute-path>]
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

Provider-relay mode additionally requires the separately forwarded absolute
`--provider-relay-socket` and an exact private credential-control map containing
only `host=api.x.ai`, `path=/v1/chat/completions`, `provider`, `relayToken`, and
`invocationSocket` (which must match the argument). The child consumes those
controls before creating `InlineCredentialSource`; Hermes receives only the
dummy API key `plane-provider-relay`, logical base URL
`http://plane-provider-relay.invalid/v1`, and `api_mode=chat_completions`.
Each injected client uses a fresh `httpx.HTTPTransport(uds=..., retries=0)` and
SDK-owned `httpx.Client`; missing, invalid, or mismatched relay controls fail
closed. The relay token is never placed in the runtime request, model input,
trajectory, hook state, environment, or child command.

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
