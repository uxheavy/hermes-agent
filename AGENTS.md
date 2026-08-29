# Repository Map

## Scope

This file governs the repository unless a closer `AGENTS.md` applies.

## Canonical Commands

- `scripts/run_tests.sh <path>` — hermetic Python tests; do not call pytest directly.
- `uv run ruff check <path>` — Python lint.
- `uv run ty check` — Python type check.
- `npm run check` — JavaScript/TypeScript workspace checks.
- `python3 scripts/ci/repository_policy.py --base <ref>` — repository placement policy.

## Where Truth Lives

| Concern | Owner |
| --- | --- |
| Contributor workflow | `CONTRIBUTING.md` |
| CI routing | `scripts/ci/classify_changes.py`, `.github/workflows/ci.yml` |
| Repository placement policy | `scripts/ci/repository_policy.py` |
| Python tests | `tests/`, run through `scripts/run_tests.sh` |
| JavaScript/TypeScript checks | Root and workspace `package.json` files |
| Dependency versions | `pyproject.toml`, `uv.lock`, `package-lock.json` |

## Canonical Owners

- Core agent behavior: `agent/` and the existing narrow-waist entry points.
- Built-in tools: `tools/`, with registration colocated and exposure in `toolsets.py`.
- Provider-specific behavior: `plugins/model-providers/`; do not add new legacy `providers/*.py` modules.
- Platform transport: `gateway/` adapters.
- Scheduled lifecycle: `cron/`.
- CLI parsing and delegation: `hermes_cli/`; reusable behavior stays with its domain owner.
- Desktop-only state and UI: read `apps/desktop/AGENTS.md` first.

## Evidence and Authority

- Run the narrowest relevant check and report anything skipped or unavailable.
- CI's `All required checks pass` status is merge evidence; local hooks are feedback only.
- Requirement, architectural ownership, abstraction, and proof strength remain maintainer-review decisions.
