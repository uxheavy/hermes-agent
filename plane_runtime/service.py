"""Minimal JSON-lines host for a separately spawned runtime process.

This is deliberately a transport-shaped seam, not a queue or RPC decision.
One process accepts one invocation request, streams validated events, and
returns one exit.  A supervisor can replace the process with a new invocation.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from typing import Any, TextIO

from .adapter import (
    EventCollector,
    FakeKernel,
    FakeKernelPlan,
    RecordingHost,
    TrustedRuntimeSupervisor,
    execute,
)
from .contract import ContractError, InvocationEnvelope, RunSnapshot, RuntimeBudget


def _fake_plan(raw: Any) -> FakeKernelPlan:
    if raw is None:
        return FakeKernelPlan()
    if not isinstance(raw, dict):
        raise ContractError("fakePlan must be an object")
    unknown = sorted(set(raw).difference({
        "transcript",
        "transcriptRef",
        "usage",
        "terminalKind",
        "inputRequest",
        "inputRequestRef",
        "publicationRequested",
        "holdAfterObservations",
    }))
    if unknown:
        raise ContractError(f"fakePlan has unknown field(s): {', '.join(unknown)}")
    defaults = FakeKernelPlan()
    usage = raw.get("usage")
    transcript = raw.get("transcript", defaults.transcript)
    transcript_ref = raw.get("transcriptRef", defaults.transcript_ref)
    terminal_kind = raw.get("terminalKind", defaults.terminal_kind)
    input_request = raw.get("inputRequest")
    input_request_ref = raw.get("inputRequestRef", defaults.input_request_ref)
    publication_requested = raw.get("publicationRequested", False)
    hold_after_observations = raw.get("holdAfterObservations")
    if not isinstance(transcript, str) or not transcript:
        raise ContractError("fakePlan.transcript must be a non-empty string")
    if not isinstance(transcript_ref, str) or not transcript_ref:
        raise ContractError("fakePlan.transcriptRef must be a non-empty string")
    if not isinstance(terminal_kind, str):
        raise ContractError("fakePlan.terminalKind must be a string")
    if input_request is not None and (not isinstance(input_request, str) or not input_request):
        raise ContractError("fakePlan.inputRequest must be a non-empty string")
    if not isinstance(input_request_ref, str) or not input_request_ref:
        raise ContractError("fakePlan.inputRequestRef must be a non-empty string")
    if not isinstance(publication_requested, bool):
        raise ContractError("fakePlan.publicationRequested must be a boolean")
    if hold_after_observations is not None and (
        isinstance(hold_after_observations, bool)
        or not isinstance(hold_after_observations, int)
        or hold_after_observations < 1
    ):
        raise ContractError("fakePlan.holdAfterObservations must be an integer >= 1")
    return FakeKernelPlan(
        transcript=transcript,
        transcript_ref=transcript_ref,
        usage=RuntimeBudget.from_dict(
            usage
            if usage is not None
            else {"iterations": 1, "inputTokens": 0, "outputTokens": 4},
            "fakePlan.usage",
        ),
        terminal_kind=terminal_kind,
        input_request=input_request,
        input_request_ref=input_request_ref,
        publication_requested=publication_requested,
        hold_after_observations=hold_after_observations,
    )


def serve_once(request_line: str, output: TextIO) -> None:
    """Read one serialized invocation and write event/exit JSON lines."""

    try:
        request = json.loads(request_line)
        if not isinstance(request, dict):
            raise ContractError("service request must be an object")
        unknown = sorted(set(request).difference({"run", "invocation", "fakePlan"}))
        if unknown:
            raise ContractError(f"service request has unknown field(s): {', '.join(unknown)}")
        run = RunSnapshot.from_dict(request.get("run"))
        invocation = InvocationEnvelope.from_dict(request.get("invocation"))
        plan = _fake_plan(request.get("fakePlan"))
        collector = EventCollector(
            run_id=run.run_id,
            invocation_id=invocation.invocation_id,
            expected_causation_ref=invocation.causation_ref,
        )
        host = RecordingHost()

        def emit(event) -> None:
            collector.emit(event)
            output.write(json.dumps({"type": "event", "event": event.to_dict()}, sort_keys=True) + "\n")
            output.flush()

        exit_value = execute(
            run=run,
            invocation=invocation,
            host=host,
            emit=emit,
            kernel=FakeKernel(plan),
            supervisor=TrustedRuntimeSupervisor(
                clock=lambda: datetime.now(timezone.utc),
                checkpoint_refs={"checkpoint:one"},
            ),
        )
        output.write(json.dumps({"type": "exit", "exit": exit_value.to_dict()}, sort_keys=True) + "\n")
        output.flush()
    except (ContractError, TypeError, ValueError, json.JSONDecodeError) as exc:
        output.write(
            json.dumps(
                {"type": "error", "error": {"code": "invalid_request", "message": str(exc)}},
                sort_keys=True,
            )
            + "\n"
        )
        output.flush()
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one Plane Agent runtime invocation")
    parser.add_argument("--once", action="store_true", help="accept one JSON-lines invocation (the default)")
    args = parser.parse_args(argv)
    del args
    request_line = sys.stdin.readline()
    if not request_line:
        return 2
    try:
        serve_once(request_line, sys.stdout)
    except (ContractError, TypeError, ValueError, json.JSONDecodeError):
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess tests
    raise SystemExit(main())
