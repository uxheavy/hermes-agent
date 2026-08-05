"""Host-owned durable replay and cumulative-budget authority for G1.

The ledger is deliberately separate from Plane state and from Hermes session
state.  It stores only the immutable request binding, terminal evidence bytes,
and host-measured accounting needed to make a replaceable runtime process safe
across supervisor restarts.  A row left ``running`` by a crashed supervisor is
never relaunched: it becomes ``outcome_unknown`` and requires reconciliation.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


class G1LedgerError(RuntimeError):
    """The trusted host ledger cannot authorize a dispatch."""


@dataclass(frozen=True)
class G1LedgerClaim:
    state: str
    frames: tuple[str, ...] = ()

    @property
    def owns_dispatch(self) -> bool:
        return self.state == "owned"


class G1RuntimeLedger:
    """SQLite WAL ledger with atomic claim/finalize semantics."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        max_calls: int = 128,
        running_stale_seconds: float = 30.0,
        wait_seconds: float = 5.0,
    ) -> None:
        if isinstance(max_calls, bool) or not isinstance(max_calls, int) or max_calls < 1:
            raise ValueError("max_calls must be a positive integer")
        if running_stale_seconds <= 0 or wait_seconds <= 0:
            raise ValueError("ledger timeouts must be positive")
        self.path = Path(path).expanduser()
        self.max_calls = max_calls
        self.running_stale_seconds = float(running_stale_seconds)
        self.wait_seconds = float(wait_seconds)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path,
            timeout=max(1.0, self.wait_seconds),
            isolation_level=None,
            check_same_thread=False,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=5000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS g1_runs (
                    run_id TEXT PRIMARY KEY,
                    input_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    duration_ms INTEGER NOT NULL,
                    calls INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS g1_invocations (
                    run_id TEXT NOT NULL,
                    invocation_id TEXT NOT NULL,
                    request_fingerprint TEXT NOT NULL,
                    snapshot_digest TEXT NOT NULL,
                    state TEXT NOT NULL,
                    frames_json TEXT,
                    requested_input_tokens INTEGER NOT NULL,
                    requested_output_tokens INTEGER NOT NULL,
                    requested_duration_ms INTEGER NOT NULL,
                    host_duration_ms INTEGER NOT NULL DEFAULT 0,
                    started_at REAL NOT NULL,
                    PRIMARY KEY (run_id, invocation_id),
                    FOREIGN KEY (run_id) REFERENCES g1_runs(run_id)
                );
                CREATE INDEX IF NOT EXISTS g1_invocations_state_idx
                    ON g1_invocations(run_id, state);
                """
            )
            os.chmod(self.path, 0o600)

    @staticmethod
    def _frames(value: object) -> tuple[str, ...]:
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise G1LedgerError("durable G1 frames are malformed")
        return tuple(value)

    def _claim_once(
        self,
        *,
        run_id: str,
        invocation_id: str,
        request_fingerprint: str,
        snapshot_digest: str,
        total_budget: Mapping[str, int],
        remaining_budget: Mapping[str, int],
    ) -> G1LedgerClaim | None:
        now = time.time()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM g1_invocations WHERE run_id = ? AND invocation_id = ?",
                (run_id, invocation_id),
            ).fetchone()
            if row is not None:
                if row["request_fingerprint"] != request_fingerprint:
                    connection.execute("ROLLBACK")
                    return G1LedgerClaim("conflict")
                state = str(row["state"])
                if state == "completed":
                    frames = self._frames(json.loads(row["frames_json"] or "null"))
                    connection.execute("COMMIT")
                    return G1LedgerClaim("replay", frames)
                if state == "outcome_unknown":
                    connection.execute("COMMIT")
                    return G1LedgerClaim("outcome_unknown")
                if state == "running":
                    if now - float(row["started_at"]) > self.running_stale_seconds:
                        connection.execute(
                            "UPDATE g1_invocations SET state = 'outcome_unknown' WHERE run_id = ? AND invocation_id = ?",
                            (run_id, invocation_id),
                        )
                        connection.execute("COMMIT")
                        return G1LedgerClaim("outcome_unknown")
                    connection.execute("COMMIT")
                    return None
                connection.execute("ROLLBACK")
                raise G1LedgerError("durable G1 invocation state is invalid")

            run = connection.execute(
                "SELECT input_tokens, output_tokens, duration_ms, calls FROM g1_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if run is None:
                connection.execute(
                    "INSERT INTO g1_runs(run_id, input_tokens, output_tokens, duration_ms, calls) VALUES (?, ?, ?, ?, 0)",
                    (run_id, int(total_budget["inputTokens"]), int(total_budget["outputTokens"]), int(total_budget["durationMs"])),
                )
                run = connection.execute(
                    "SELECT input_tokens, output_tokens, duration_ms, calls FROM g1_runs WHERE run_id = ?",
                    (run_id,),
                ).fetchone()
            assert run is not None
            if (
                int(remaining_budget["inputTokens"]) > int(run["input_tokens"])
                or int(remaining_budget["outputTokens"]) > int(run["output_tokens"])
                or int(remaining_budget["durationMs"]) > int(run["duration_ms"])
                or int(run["calls"]) >= self.max_calls
            ):
                connection.execute("COMMIT")
                return G1LedgerClaim("budget_exhausted")
            connection.execute(
                "INSERT INTO g1_invocations(run_id, invocation_id, request_fingerprint, snapshot_digest, state, frames_json, requested_input_tokens, requested_output_tokens, requested_duration_ms, started_at) VALUES (?, ?, ?, ?, 'running', NULL, ?, ?, ?, ?)",
                (
                    run_id,
                    invocation_id,
                    request_fingerprint,
                    snapshot_digest,
                    int(remaining_budget["inputTokens"]),
                    int(remaining_budget["outputTokens"]),
                    int(remaining_budget["durationMs"]),
                    now,
                ),
            )
            connection.execute("UPDATE g1_runs SET calls = calls + 1 WHERE run_id = ?", (run_id,))
            connection.execute("COMMIT")
            return G1LedgerClaim("owned")

    def claim(
        self,
        *,
        run_id: str,
        invocation_id: str,
        request_fingerprint: str,
        snapshot_digest: str,
        total_budget: Mapping[str, int],
        remaining_budget: Mapping[str, int],
    ) -> G1LedgerClaim:
        deadline = time.monotonic() + self.wait_seconds
        while True:
            claim = self._claim_once(
                run_id=run_id,
                invocation_id=invocation_id,
                request_fingerprint=request_fingerprint,
                snapshot_digest=snapshot_digest,
                total_budget=total_budget,
                remaining_budget=remaining_budget,
            )
            if claim is not None:
                return claim
            if time.monotonic() >= deadline:
                return G1LedgerClaim("outcome_unknown")
            time.sleep(0.005)

    def finalize(
        self,
        *,
        run_id: str,
        invocation_id: str,
        request_fingerprint: str,
        frames: Sequence[str],
        requested_budget: Mapping[str, int],
        host_duration_ms: int,
    ) -> None:
        payload = json.dumps(list(frames), ensure_ascii=False, separators=(",", ":"))
        duration = max(0, int(host_duration_ms))
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT request_fingerprint, state, requested_duration_ms FROM g1_invocations WHERE run_id = ? AND invocation_id = ?",
                (run_id, invocation_id),
            ).fetchone()
            if row is None or row["request_fingerprint"] != request_fingerprint:
                connection.execute("ROLLBACK")
                raise G1LedgerError("G1 finalize binding is invalid")
            if row["state"] == "completed":
                connection.execute("COMMIT")
                return
            if row["state"] != "running":
                connection.execute("ROLLBACK")
                raise G1LedgerError("outcome_unknown G1 invocation cannot be finalized")
            connection.execute(
                "UPDATE g1_invocations SET state = 'completed', frames_json = ?, host_duration_ms = ? WHERE run_id = ? AND invocation_id = ?",
                (payload, duration, run_id, invocation_id),
            )
            connection.execute(
                "UPDATE g1_runs SET input_tokens = MIN(input_tokens, ?), output_tokens = MIN(output_tokens, ?), duration_ms = MAX(0, MIN(duration_ms, ?) - ?) WHERE run_id = ?",
                (
                    int(requested_budget["inputTokens"]),
                    int(requested_budget["outputTokens"]),
                    int(requested_budget["durationMs"]),
                    duration,
                    run_id,
                ),
            )
            connection.execute("COMMIT")

    def mark_outcome_unknown(self, *, run_id: str, invocation_id: str, request_fingerprint: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE g1_invocations SET state = 'outcome_unknown' WHERE run_id = ? AND invocation_id = ? AND request_fingerprint = ? AND state = 'running'",
                (run_id, invocation_id, request_fingerprint),
            )


__all__ = ["G1LedgerClaim", "G1LedgerError", "G1RuntimeLedger"]
