# AGENTS.md

## Scope

This file governs `plane_runtime/`, the Hermes-side package boundary for the
separately deployed Plane Agent runtime service.

## Local Responsibility

Keep Plane-specific translation local to this package. The adapter execution
seam is `plane_runtime.adapter.execute`: it accepts versioned Plane
`RunSnapshot` and `InvocationEnvelope` inputs through the runtime service and
adapts them to Hermes kernel execution.

## Architecture Rules

- Plane owns product vocabulary, durable state, lifecycle transitions,
  publication, gateway receipts, and recovery authority. Do not make Hermes
  sessions, transcripts, checkpoints, or profile state the Plane record.
- Only the narrow adapter may import or call Hermes kernel mechanisms. Do not
  add Plane-specific imports or policy to `AIAgent`, `run_agent.py`,
  `model_tools.py`, `gateway/`, or other core modules.
- Runtime events are bounded observations, not product mutations. Explicit
  publication must correlate an authorized, idempotent Plane Operation Gateway
  receipt; an observation alone cannot publish product state.
- Keep credentials in trusted host state. Generated code receives no Plane,
  provider, storage, or host credentials and no ambient authority.
- Treat a process/container as replaceable invocation-scoped infrastructure.
  A durable run must not own a container lifetime.
- Do not add a Buzz dependency or a chat/composer UI here. Generic reusable
  Hermes hooks may be contributed separately, but Plane policy stays in this
  adapter.

## Working Method

| Situation | Required method |
| --- | --- |
| Adding runtime behavior | First pin the versioned cross-process contract; keep all Plane translation and policy in this package. |
| Reusing Hermes behavior | Depend on the smallest existing kernel seam and keep the Hermes import behind the adapter. |
| Handling observations | Validate defensively here, then send them across the service seam; never treat them as authoritative product writes. |
| Handling restart or waiting | Rebuild invocation-scoped infrastructure from Plane-owned snapshot, events, permitted checkpoints, and remaining budget. |

## Package Boundary

- The `plane_runtime.adapter.execute` interface is internal to the separate
  runtime service, not a Python import for Plane API modules.

## Local Verification

Run these commands from the Hermes repository root after changing this
package:

```bash
python3 -c "import plane_runtime"
python3 -m compileall -q plane_runtime
git diff --check -- plane_runtime
```
