"""Minimal JSON-lines host for a separately spawned runtime process.

This is deliberately a transport-shaped seam, not a queue or RPC decision.
One process accepts one invocation request, streams validated events, and
returns one exit.  A supervisor can replace the process with a new invocation.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, TextIO

from .adapter import EventCollector, FakeKernel, FakeKernelPlan, RecordingHost, execute
from .contract import ContractError, InvocationEnvelope, RunSnapshot, RuntimeBudget


def _fake_plan(raw: Any) -> FakeKernelPlan:
    if raw is None:
        return FakeKernelPlan()
    if not isinstance(raw, dict):
        raise ContractError("fakePlan must be an object")
    defaults = FakeKernelPlan()
    usage = raw.get("usage")
    return FakeKernelPlan(
        transcript=raw.get("transcript", defaults.transcript),
        transcript_ref=raw.get("transcriptRef", defaults.transcript_ref),
        usage=RuntimeBudget.from_dict(
            usage
            if usage is not None
            else {"iterations": 1, "inputTokens": 0, "outputTokens": 4},
            "fakePlan.usage",
        ),
        terminal_kind=raw.get("terminalKind", defaults.terminal_kind),
        input_request=raw.get("inputRequest"),
        input_request_ref=raw.get("inputRequestRef", defaults.input_request_ref),
        publication_requested=raw.get("publicationRequested", False),
    )


def serve_once(request_line: str, output: TextIO) -> None:
    """Read one serialized invocation and write event/exit JSON lines."""

    try:
        request = json.loads(request_line)
        if not isinstance(request, dict):
            raise ContractError("service request must be an object")
        run = RunSnapshot.from_dict(request.get("run"))
        invocation = InvocationEnvelope.from_dict(request.get("invocation"))
        plan = _fake_plan(request.get("fakePlan"))
        streamed: list[dict[str, Any]] = []
        collector = EventCollector(run_id=run.run_id, invocation_id=invocation.invocation_id)
        host = RecordingHost()
        exit_value = execute(
            run=run,
            invocation=invocation,
            host=host,
            emit=lambda event: (collector.emit(event), streamed.append(event.to_dict())),
            kernel=FakeKernel(plan),
        )
        # ``collector`` is the cross-process ingress-side validation proof;
        # ``streamed`` is the transport representation sent to the caller.
        if len(streamed) != len(collector.events):
            raise ContractError("event collector and transport stream diverged")
        for event in streamed:
            output.write(json.dumps({"type": "event", "event": event}, sort_keys=True) + "\n")
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
