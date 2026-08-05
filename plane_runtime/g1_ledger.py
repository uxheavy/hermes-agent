"""Host-owned durable replay and cumulative-budget authority for G1.

This is intentionally a small operational store, not Plane state.  A run row
is bound to the immutable snapshot and to the first declared set of limits.
Claims reserve allowance under one SQLite transaction; finalization accounts
only for bounded evidence and host measurements.  A missing or invalid child
usage report consumes its reservation rather than creating allowance.
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


_SCHEMA_VERSION = 2
_BUDGET_KEYS = ("inputTokens", "outputTokens", "durationMs")


@dataclass(frozen=True)
class G1LedgerClaim:
    state: str
    frames: tuple[str, ...] = ()
    model_call_allowance: int = 0
    retained_bytes_allowance: int = 0

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
        max_model_calls: int = 128,
        running_stale_seconds: float = 30.0,
        wait_seconds: float = 5.0,
    ) -> None:
        if isinstance(max_calls, bool) or not isinstance(max_calls, int) or max_calls < 1:
            raise ValueError("max_calls must be a positive integer")
        if isinstance(max_model_calls, bool) or not isinstance(max_model_calls, int) or max_model_calls < 0:
            raise ValueError("max_model_calls must be a non-negative integer")
        if running_stale_seconds <= 0 or wait_seconds <= 0:
            raise ValueError("ledger timeouts must be positive")
        self.path = Path(path).expanduser()
        if not self.path.is_absolute():
            raise ValueError("durable G1 ledger path must be absolute")
        self.max_calls = max_calls
        self.max_model_calls = max_model_calls
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
            existing = connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('g1_meta','g1_runs','g1_invocations')"
            ).fetchall()
            names = {str(row[0]) for row in existing}
            if "g1_meta" in names:
                version_row = connection.execute("SELECT schema_version FROM g1_meta WHERE id = 1").fetchone()
                if version_row is None or int(version_row[0]) != _SCHEMA_VERSION:
                    raise G1LedgerError("durable G1 ledger schema version is unsupported")
            elif names:
                # Never silently reinterpret an older durable ledger.  An
                # operator must rotate/migrate it explicitly.
                raise G1LedgerError("durable G1 ledger requires an explicit schema migration")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS g1_meta (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    schema_version INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS g1_runs (
                    run_id TEXT PRIMARY KEY,
                    snapshot_digest TEXT NOT NULL,
                    runtime_invocation_limit INTEGER NOT NULL,
                    runtime_invocations_used INTEGER NOT NULL DEFAULT 0,
                    model_call_limit INTEGER NOT NULL,
                    model_calls_used INTEGER NOT NULL DEFAULT 0,
                    input_token_limit INTEGER NOT NULL,
                    highest_declared_input_tokens INTEGER NOT NULL DEFAULT 0,
                    input_tokens_used INTEGER NOT NULL DEFAULT 0,
                    output_token_limit INTEGER NOT NULL,
                    highest_declared_output_tokens INTEGER NOT NULL DEFAULT 0,
                    output_tokens_used INTEGER NOT NULL DEFAULT 0,
                    retained_bytes_limit INTEGER NOT NULL,
                    retained_bytes_used INTEGER NOT NULL DEFAULT 0,
                    host_duration_limit_ms INTEGER NOT NULL,
                    highest_declared_duration_ms INTEGER NOT NULL DEFAULT 0,
                    host_duration_used_ms INTEGER NOT NULL DEFAULT 0,
                    reserved_model_calls INTEGER NOT NULL DEFAULT 0,
                    reserved_input_tokens INTEGER NOT NULL DEFAULT 0,
                    reserved_output_tokens INTEGER NOT NULL DEFAULT 0,
                    reserved_retained_bytes INTEGER NOT NULL DEFAULT 0,
                    reserved_duration_ms INTEGER NOT NULL DEFAULT 0
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
                    reserved_model_calls INTEGER NOT NULL,
                    reserved_retained_bytes INTEGER NOT NULL,
                    actual_model_calls INTEGER,
                    actual_input_tokens INTEGER,
                    actual_output_tokens INTEGER,
                    actual_retained_bytes INTEGER,
                    host_duration_ms INTEGER,
                    usage_valid INTEGER,
                    started_at REAL NOT NULL,
                    PRIMARY KEY (run_id, invocation_id),
                    FOREIGN KEY (run_id) REFERENCES g1_runs(run_id)
                );
                CREATE INDEX IF NOT EXISTS g1_invocations_state_idx
                    ON g1_invocations(run_id, state);
                """
            )
            connection.execute(
                "INSERT OR IGNORE INTO g1_meta(id, schema_version) VALUES (1, ?)",
                (_SCHEMA_VERSION,),
            )
            os.chmod(self.path, 0o600)

    @staticmethod
    def _frames(value: object) -> tuple[str, ...]:
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise G1LedgerError("durable G1 frames are malformed")
        return tuple(value)

    @staticmethod
    def _budget(value: Mapping[str, int], name: str) -> dict[str, int]:
        if not isinstance(value, Mapping) or set(value) != set(_BUDGET_KEYS):
            raise G1LedgerError(f"{name} has an invalid shape")
        result: dict[str, int] = {}
        for key in _BUDGET_KEYS:
            item = value[key]
            if isinstance(item, bool) or not isinstance(item, int) or item < 0:
                raise G1LedgerError(f"{name}.{key} is invalid")
            result[key] = item
        return result

    @staticmethod
    def _nonnegative(value: object, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise G1LedgerError(f"{name} is invalid")
        return value

    def _claim_once(
        self,
        *,
        run_id: str,
        invocation_id: str,
        request_fingerprint: str,
        snapshot_digest: str,
        total_budget: Mapping[str, int],
        remaining_budget: Mapping[str, int],
        retained_bytes_limit: int,
    ) -> G1LedgerClaim | None:
        total = self._budget(total_budget, "total_budget")
        remaining = self._budget(remaining_budget, "remaining_budget")
        if any(remaining[key] > total[key] for key in _BUDGET_KEYS):
            return G1LedgerClaim("conflict")
        retained_limit = self._nonnegative(retained_bytes_limit, "retained_bytes_limit")
        now = time.time()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM g1_invocations WHERE run_id = ? AND invocation_id = ?",
                (run_id, invocation_id),
            ).fetchone()
            if row is not None:
                if row["request_fingerprint"] != request_fingerprint or row["snapshot_digest"] != snapshot_digest:
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
                        self._consume_unknown(connection, row)
                        connection.execute("COMMIT")
                        return G1LedgerClaim("outcome_unknown")
                    connection.execute("COMMIT")
                    return None
                connection.execute("ROLLBACK")
                raise G1LedgerError("durable G1 invocation state is invalid")

            run = connection.execute("SELECT * FROM g1_runs WHERE run_id = ?", (run_id,)).fetchone()
            if run is None:
                connection.execute(
                    """INSERT INTO g1_runs(
                        run_id, snapshot_digest, runtime_invocation_limit,
                        model_call_limit, input_token_limit, highest_declared_input_tokens,
                        output_token_limit, highest_declared_output_tokens,
                        retained_bytes_limit, host_duration_limit_ms, highest_declared_duration_ms
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        run_id,
                        snapshot_digest,
                        self.max_calls,
                        self.max_model_calls,
                        total["inputTokens"],
                        remaining["inputTokens"],
                        total["outputTokens"],
                        remaining["outputTokens"],
                        retained_limit,
                        total["durationMs"],
                        remaining["durationMs"],
                    ),
                )
                run = connection.execute("SELECT * FROM g1_runs WHERE run_id = ?", (run_id,)).fetchone()
            assert run is not None
            if (
                str(run["snapshot_digest"]) != snapshot_digest
                or int(run["runtime_invocation_limit"]) != self.max_calls
                or int(run["model_call_limit"]) != self.max_model_calls
                or int(run["input_token_limit"]) != total["inputTokens"]
                or int(run["output_token_limit"]) != total["outputTokens"]
                or int(run["retained_bytes_limit"]) != retained_limit
                or int(run["host_duration_limit_ms"]) != total["durationMs"]
                or remaining["inputTokens"] > int(run["highest_declared_input_tokens"])
                or remaining["outputTokens"] > int(run["highest_declared_output_tokens"])
                or remaining["durationMs"] > int(run["highest_declared_duration_ms"])
            ):
                connection.execute("COMMIT")
                return G1LedgerClaim("conflict")

            available = {
                "inputTokens": int(run["input_token_limit"]) - int(run["input_tokens_used"]) - int(run["reserved_input_tokens"]),
                "outputTokens": int(run["output_token_limit"]) - int(run["output_tokens_used"]) - int(run["reserved_output_tokens"]),
                "durationMs": int(run["host_duration_limit_ms"]) - int(run["host_duration_used_ms"]) - int(run["reserved_duration_ms"]),
            }
            retained_available = int(run["retained_bytes_limit"]) - int(run["retained_bytes_used"]) - int(run["reserved_retained_bytes"])
            model_available = int(run["model_call_limit"]) - int(run["model_calls_used"]) - int(run["reserved_model_calls"])
            if (
                int(run["runtime_invocations_used"]) >= int(run["runtime_invocation_limit"])
                or any(remaining[key] > available[key] for key in _BUDGET_KEYS)
                or retained_available <= 0
                or model_available < 0
            ):
                connection.execute("COMMIT")
                return G1LedgerClaim("budget_exhausted")
            # A zero-call invocation is valid only when the caller explicitly
            # has no model work; reserve no model call and still retain its
            # bounded terminal evidence.
            reserve_model_calls = model_available
            reserve_retained = retained_available
            connection.execute(
                """INSERT INTO g1_invocations(
                    run_id, invocation_id, request_fingerprint, snapshot_digest,
                    state, frames_json, requested_input_tokens,
                    requested_output_tokens, requested_duration_ms,
                    reserved_model_calls, reserved_retained_bytes, started_at
                ) VALUES (?, ?, ?, ?, 'running', NULL, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    invocation_id,
                    request_fingerprint,
                    snapshot_digest,
                    remaining["inputTokens"],
                    remaining["outputTokens"],
                    remaining["durationMs"],
                    reserve_model_calls,
                    reserve_retained,
                    now,
                ),
            )
            connection.execute(
                """UPDATE g1_runs SET
                    runtime_invocations_used = runtime_invocations_used + 1,
                    highest_declared_input_tokens = MAX(highest_declared_input_tokens, ?),
                    highest_declared_output_tokens = MAX(highest_declared_output_tokens, ?),
                    highest_declared_duration_ms = MAX(highest_declared_duration_ms, ?),
                    reserved_model_calls = reserved_model_calls + ?,
                    reserved_input_tokens = reserved_input_tokens + ?,
                    reserved_output_tokens = reserved_output_tokens + ?,
                    reserved_retained_bytes = reserved_retained_bytes + ?,
                    reserved_duration_ms = reserved_duration_ms + ?
                WHERE run_id = ?""",
                (
                    remaining["inputTokens"],
                    remaining["outputTokens"],
                    remaining["durationMs"],
                    reserve_model_calls,
                    remaining["inputTokens"],
                    remaining["outputTokens"],
                    reserve_retained,
                    remaining["durationMs"],
                    run_id,
                ),
            )
            connection.execute("COMMIT")
            return G1LedgerClaim(
                "owned",
                model_call_allowance=reserve_model_calls,
                retained_bytes_allowance=reserve_retained,
            )

    @staticmethod
    def _consume_unknown(connection: sqlite3.Connection, row: sqlite3.Row) -> None:
        connection.execute(
            """UPDATE g1_runs SET
                model_calls_used = model_calls_used + reserved_model_calls,
                input_tokens_used = input_tokens_used + reserved_input_tokens,
                output_tokens_used = output_tokens_used + reserved_output_tokens,
                retained_bytes_used = retained_bytes_used + reserved_retained_bytes,
                host_duration_used_ms = host_duration_used_ms + reserved_duration_ms,
                reserved_model_calls = reserved_model_calls - ?,
                reserved_input_tokens = reserved_input_tokens - ?,
                reserved_output_tokens = reserved_output_tokens - ?,
                reserved_retained_bytes = reserved_retained_bytes - ?,
                reserved_duration_ms = reserved_duration_ms - ?
            WHERE run_id = ?""",
            (
                int(row["reserved_model_calls"]),
                int(row["requested_input_tokens"]),
                int(row["requested_output_tokens"]),
                int(row["reserved_retained_bytes"]),
                int(row["requested_duration_ms"]),
                row["run_id"],
            ),
        )
        connection.execute(
            "UPDATE g1_invocations SET state = 'outcome_unknown', usage_valid = 0 WHERE run_id = ? AND invocation_id = ?",
            (row["run_id"], row["invocation_id"]),
        )

    def claim(
        self,
        *,
        run_id: str,
        invocation_id: str,
        request_fingerprint: str,
        snapshot_digest: str,
        total_budget: Mapping[str, int],
        remaining_budget: Mapping[str, int],
        retained_bytes_limit: int = 512 * 1024,
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
                retained_bytes_limit=retained_bytes_limit,
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
        requested_budget: Mapping[str, int] | None = None,
        host_duration_ms: int,
        usage: Mapping[str, int] | None = None,
        usage_valid: bool = False,
        model_calls: int | None = None,
        retained_bytes: int | None = None,
    ) -> None:
        del requested_budget  # The reservation in the durable row is authoritative.
        payload = json.dumps(list(frames), ensure_ascii=False, separators=(",", ":"))
        duration = self._nonnegative(host_duration_ms, "host_duration_ms")
        if not isinstance(usage_valid, bool):
            raise G1LedgerError("usage_valid is invalid")
        usage_map = dict(usage or {})
        valid_usage = usage_valid and set(usage_map) == set(_BUDGET_KEYS) and all(
            isinstance(usage_map[key], int) and not isinstance(usage_map[key], bool) and usage_map[key] >= 0
            for key in _BUDGET_KEYS
        )
        if valid_usage:
            child_usage = {key: int(usage_map[key]) for key in _BUDGET_KEYS}
        else:
            child_usage = None
        valid_calls = isinstance(model_calls, int) and not isinstance(model_calls, bool) and model_calls >= 0
        valid_retained = isinstance(retained_bytes, int) and not isinstance(retained_bytes, bool) and retained_bytes >= 0
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM g1_invocations WHERE run_id = ? AND invocation_id = ?",
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
            reserved = {
                "inputTokens": int(row["requested_input_tokens"]),
                "outputTokens": int(row["requested_output_tokens"]),
                "durationMs": int(row["requested_duration_ms"]),
            }
            if child_usage is None or any(child_usage[key] > reserved[key] for key in _BUDGET_KEYS):
                actual_usage = reserved
                usage_flag = 0
            else:
                actual_usage = child_usage
                usage_flag = 1
            reserved_calls = int(row["reserved_model_calls"])
            actual_calls = int(model_calls) if valid_calls and int(model_calls) <= reserved_calls else reserved_calls
            calls_flag = int(valid_calls and int(model_calls or 0) <= reserved_calls)
            reserved_retained = int(row["reserved_retained_bytes"])
            actual_retained = int(retained_bytes) if valid_retained and int(retained_bytes) <= reserved_retained else reserved_retained
            retained_flag = int(valid_retained and int(retained_bytes or 0) <= reserved_retained)
            reserved_duration = reserved["durationMs"]
            actual_duration = min(duration, reserved_duration)
            connection.execute(
                """UPDATE g1_invocations SET
                    state = 'completed', frames_json = ?,
                    actual_model_calls = ?, actual_input_tokens = ?,
                    actual_output_tokens = ?, actual_retained_bytes = ?,
                    host_duration_ms = ?, usage_valid = ?
                WHERE run_id = ? AND invocation_id = ?""",
                (
                    payload,
                    actual_calls,
                    actual_usage["inputTokens"],
                    actual_usage["outputTokens"],
                    actual_retained,
                    actual_duration,
                    int(bool(usage_flag and calls_flag and retained_flag)),
                    run_id,
                    invocation_id,
                ),
            )
            connection.execute(
                """UPDATE g1_runs SET
                    model_calls_used = model_calls_used + ?,
                    input_tokens_used = input_tokens_used + ?,
                    output_tokens_used = output_tokens_used + ?,
                    retained_bytes_used = retained_bytes_used + ?,
                    host_duration_used_ms = host_duration_used_ms + ?,
                    reserved_model_calls = reserved_model_calls - ?,
                    reserved_input_tokens = reserved_input_tokens - ?,
                    reserved_output_tokens = reserved_output_tokens - ?,
                    reserved_retained_bytes = reserved_retained_bytes - ?,
                    reserved_duration_ms = reserved_duration_ms - ?
                WHERE run_id = ?""",
                (
                    actual_calls,
                    actual_usage["inputTokens"],
                    actual_usage["outputTokens"],
                    actual_retained,
                    actual_duration,
                    reserved_calls,
                    reserved["inputTokens"],
                    reserved["outputTokens"],
                    reserved_retained,
                    reserved_duration,
                    run_id,
                ),
            )
            connection.execute("COMMIT")

    def mark_outcome_unknown(self, *, run_id: str, invocation_id: str, request_fingerprint: str) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM g1_invocations WHERE run_id = ? AND invocation_id = ? AND request_fingerprint = ? AND state = 'running'",
                (run_id, invocation_id, request_fingerprint),
            ).fetchone()
            if row is not None:
                self._consume_unknown(connection, row)
            connection.execute("COMMIT")

    def summary(self, run_id: str) -> dict[str, int | str]:
        """Return trusted accounting for tests/host diagnostics only."""
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM g1_runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            raise G1LedgerError("unknown G1 run")
        return {
            "snapshotDigest": str(row["snapshot_digest"]),
            "runtimeInvocationsUsed": int(row["runtime_invocations_used"]),
            "runtimeInvocationLimit": int(row["runtime_invocation_limit"]),
            "modelCallsUsed": int(row["model_calls_used"]),
            "modelCallLimit": int(row["model_call_limit"]),
            "inputTokensUsed": int(row["input_tokens_used"]),
            "inputTokenLimit": int(row["input_token_limit"]),
            "outputTokensUsed": int(row["output_tokens_used"]),
            "outputTokenLimit": int(row["output_token_limit"]),
            "retainedBytesUsed": int(row["retained_bytes_used"]),
            "retainedBytesLimit": int(row["retained_bytes_limit"]),
            "hostDurationUsedMs": int(row["host_duration_used_ms"]),
            "hostDurationLimitMs": int(row["host_duration_limit_ms"]),
        }


__all__ = ["G1LedgerClaim", "G1LedgerError", "G1RuntimeLedger"]
