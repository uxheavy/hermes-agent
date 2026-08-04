"""Fail-closed supervision for one disposable Plane runtime invocation.

This module deliberately does not reuse Hermes' configurable terminal or Docker
environment implementations.  Those implementations serve a wider product
surface and intentionally support persistence, mounts, ambient configuration,
and interactive processes.  An invocation needs the opposite: one immutable
command, one bounded child, and one authoritative terminal handoff.

The real runner is optional and capability-gated.  Tests and a future runtime
service inject a runner implementing :class:`DockerRunner`; the supervisor
never creates a Docker client or reads ambient environment on its own.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import selectors
import subprocess
import time
from dataclasses import dataclass
from threading import RLock
from typing import Callable, Mapping, Protocol, Sequence

from .adapter import (
    CancellationSignal,
    EventCollector,
    TerminalReconciliationPort,
    classify_process_death,
    reconcile_process_death,
)
from .contract import (
    BindingError,
    BoundsError,
    ContractError,
    InvocationEnvelope,
    MAX_EVENT_BYTES,
    MAX_INVOCATION_BYTES,
    MAX_RUN_SNAPSHOT_BYTES,
    MAX_TERMINAL_RECEIPT_BYTES,
    RuntimeExit,
    RuntimeConfigurationError,
    RunSnapshot,
    TerminalReconciliationReceipt,
    canonical_json_bytes,
)


_IMAGE_DIGEST = re.compile(
    r"[a-z0-9][a-z0-9._:/-]*[a-z0-9]@sha256:[0-9a-f]{64}"
)
_CONTAINER_NAME = re.compile(r"[a-z0-9][a-z0-9-]{0,62}")
_FIXED_ENTRYPOINT = "python3"
_FIXED_SERVICE_MODULE = "plane_runtime.service"
_FIXED_SERVICE_ARGS = ("--once",)
_FIXED_NETWORK = "none"
_FIXED_USER = "65532:65532"
_FIXED_TMPFS_TARGET = "/tmp"
_FIXED_TMPFS_OPTIONS = "rw,noexec,nosuid,nodev"
_FIXED_PULL_POLICY = "never"
_FIXED_CONTAINER_PREFIX = "plane-invocation"
_MAX_RETAINED_INVOCATIONS = 1024

# This is intentionally a literal allowlist.  It is not derived from
# os.environ, and it contains no HOME, HERMES_HOME, credential, proxy, cloud,
# provider, Plane, or database setting.
_CHILD_ENV: tuple[tuple[str, str], ...] = (
    ("LANG", "C.UTF-8"),
    ("LC_ALL", "C.UTF-8"),
    ("PATH", "/usr/local/bin:/usr/bin:/bin"),
    ("PYTHONDONTWRITEBYTECODE", "1"),
    ("PYTHONUNBUFFERED", "1"),
)
_DOCKER_CLIENT_ENV: Mapping[str, str] = {
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "PATH": "/usr/local/bin:/usr/bin:/bin",
}
_SAFE_DEATH_REASON = "runtime process exited before terminal evidence"
_SAFE_DEATH_REASONS = frozenset(
    {
        _SAFE_DEATH_REASON,
        "runtime output was malformed or lacked terminal evidence",
        "runtime output exceeded supervisor bounds",
        "runtime invocation exceeded its wall-time bound",
        "runtime invocation was stopped by a cancellation signal",
        "runtime process died before terminal evidence",
    }
)


def _positive_int(value: object, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ContractError(f"{name} must be an integer >= {minimum}")
    return value


def _positive_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ContractError(f"{name} must be a positive number")
    return float(value)


@dataclass(frozen=True)
class InvocationPolicy:
    """Immutable, explicit bounds and isolation requirements for one launch."""

    image: str
    cpu_millicores: int = 500
    memory_bytes: int = 256 * 1024 * 1024
    pids_limit: int = 64
    wall_time_seconds: float = 120.0
    stdout_limit_bytes: int = 256 * 1024
    stderr_limit_bytes: int = 64 * 1024
    frame_limit_bytes: int = MAX_TERMINAL_RECEIPT_BYTES + 4096
    request_limit_bytes: int = 192 * 1024
    max_output_frames: int = 514
    tmpfs_bytes: int = 16 * 1024 * 1024
    stop_timeout_seconds: float = 2.0
    kill_timeout_seconds: float = 2.0
    remove_timeout_seconds: float = 2.0
    entrypoint: str = _FIXED_ENTRYPOINT
    service_module: str = _FIXED_SERVICE_MODULE
    service_args: tuple[str, ...] = _FIXED_SERVICE_ARGS
    network_mode: str = _FIXED_NETWORK
    read_only_rootfs: bool = True
    no_new_privileges: bool = True
    drop_all_capabilities: bool = True
    user: str = _FIXED_USER
    pull_policy: str = _FIXED_PULL_POLICY

    def __post_init__(self) -> None:
        if not isinstance(self.image, str) or not _IMAGE_DIGEST.fullmatch(self.image):
            raise ContractError("invocation image must be an immutable sha256 digest reference")
        if not isinstance(self.service_args, (tuple, list)):
            raise ContractError("service_args must be immutable fixed command arguments")
        object.__setattr__(self, "service_args", tuple(self.service_args))
        _positive_int(self.cpu_millicores, "cpu_millicores")
        if self.cpu_millicores > 4000:
            raise ContractError("invocation CPU bound is outside the permitted range")
        _positive_int(self.memory_bytes, "memory_bytes", minimum=16 * 1024 * 1024)
        if self.memory_bytes > 4 * 1024 * 1024 * 1024:
            raise ContractError("invocation memory bound is outside the permitted range")
        _positive_int(self.pids_limit, "pids_limit", minimum=4)
        if self.pids_limit > 4096:
            raise ContractError("invocation PID bound is outside the permitted range")
        _positive_float(self.wall_time_seconds, "wall_time_seconds")
        if self.wall_time_seconds > 3600:
            raise ContractError("invocation wall-time bound is outside the permitted range")
        _positive_int(self.stdout_limit_bytes, "stdout_limit_bytes")
        _positive_int(self.stderr_limit_bytes, "stderr_limit_bytes")
        if self.stdout_limit_bytes > 16 * 1024 * 1024 or self.stderr_limit_bytes > 16 * 1024 * 1024:
            raise ContractError("invocation output bound is outside the permitted range")
        _positive_int(self.frame_limit_bytes, "frame_limit_bytes")
        if self.frame_limit_bytes < MAX_EVENT_BYTES:
            raise ContractError("frame_limit_bytes must cover the runtime event bound")
        if self.frame_limit_bytes > MAX_TERMINAL_RECEIPT_BYTES + MAX_EVENT_BYTES:
            raise ContractError("frame_limit_bytes exceeds the terminal receipt bound")
        _positive_int(self.request_limit_bytes, "request_limit_bytes")
        if self.request_limit_bytes < MAX_RUN_SNAPSHOT_BYTES + MAX_INVOCATION_BYTES:
            raise ContractError("request_limit_bytes cannot contain the two runtime contracts")
        if self.request_limit_bytes > 512 * 1024:
            raise ContractError("invocation request bound is outside the permitted range")
        _positive_int(self.max_output_frames, "max_output_frames", minimum=2)
        if self.max_output_frames > 4096:
            raise ContractError("invocation frame-count bound is outside the permitted range")
        _positive_int(self.tmpfs_bytes, "tmpfs_bytes", minimum=1024 * 1024)
        if self.tmpfs_bytes > 1024 * 1024 * 1024:
            raise ContractError("invocation tmpfs bound is outside the permitted range")
        for name, value in (
            ("stop_timeout_seconds", self.stop_timeout_seconds),
            ("kill_timeout_seconds", self.kill_timeout_seconds),
            ("remove_timeout_seconds", self.remove_timeout_seconds),
        ):
            _positive_float(value, name)
            if float(value) > 60:
                raise ContractError("invocation cleanup deadline is outside the permitted range")
        if (
            self.entrypoint != _FIXED_ENTRYPOINT
            or self.service_module != _FIXED_SERVICE_MODULE
            or self.service_args != _FIXED_SERVICE_ARGS
        ):
            raise ContractError("invocation command is fixed to plane_runtime.service --once")
        if (
            self.network_mode != _FIXED_NETWORK
            or self.read_only_rootfs is not True
            or self.no_new_privileges is not True
            or self.drop_all_capabilities is not True
            or self.user != _FIXED_USER
            or self.pull_policy != _FIXED_PULL_POLICY
        ):
            raise ContractError("invocation isolation requirements cannot be relaxed")


@dataclass(frozen=True)
class DockerRunnerCapabilities:
    """Capabilities a runner must prove before a container may be launched."""

    immutable_digest: bool = False
    no_pull: bool = False
    network_none: bool = False
    read_only_rootfs: bool = False
    no_new_privileges: bool = False
    cap_drop_all: bool = False
    non_root_user: bool = False
    bounded_tmpfs: bool = False
    cpu_limit: bool = False
    memory_limit: bool = False
    pid_limit: bool = False
    bounded_output: bool = False
    bounded_wall_time: bool = False
    stop: bool = False
    kill: bool = False
    remove: bool = False

    @classmethod
    def fully_supported(cls) -> "DockerRunnerCapabilities":
        return cls(*(True for _ in cls.__dataclass_fields__))

    def missing(self) -> tuple[str, ...]:
        return tuple(
            name for name, supported in (
                ("immutable_digest", self.immutable_digest),
                ("no_pull", self.no_pull),
                ("network_none", self.network_none),
                ("read_only_rootfs", self.read_only_rootfs),
                ("no_new_privileges", self.no_new_privileges),
                ("cap_drop_all", self.cap_drop_all),
                ("non_root_user", self.non_root_user),
                ("bounded_tmpfs", self.bounded_tmpfs),
                ("cpu_limit", self.cpu_limit),
                ("memory_limit", self.memory_limit),
                ("pid_limit", self.pid_limit),
                ("bounded_output", self.bounded_output),
                ("bounded_wall_time", self.bounded_wall_time),
                ("stop", self.stop),
                ("kill", self.kill),
                ("remove", self.remove),
            )
            if not supported
        )


@dataclass(frozen=True)
class ProcessCapture:
    """Bounded process output returned by a runner."""

    returncode: int | None
    stdout: bytes = b""
    stderr: bytes = b""
    timed_out: bool = False
    cancelled: bool = False
    output_exceeded: bool = False


class InvocationProcess(Protocol):
    def collect(
        self,
        *,
        input_bytes: bytes,
        deadline: float,
        stdout_limit_bytes: int,
        stderr_limit_bytes: int,
        is_cancelled: Callable[[], bool],
    ) -> ProcessCapture:
        ...


class DockerRunner(Protocol):
    @property
    def capabilities(self) -> DockerRunnerCapabilities:
        ...

    def launch(
        self,
        argv: Sequence[str],
        *,
        client_env: Mapping[str, str],
        input_bytes: bytes,
    ) -> InvocationProcess:
        ...

    def stop(self, container_name: str, *, timeout_seconds: float) -> None:
        ...

    def kill(self, container_name: str, *, timeout_seconds: float) -> None:
        ...

    def remove(self, container_name: str, *, timeout_seconds: float) -> None:
        ...


@dataclass(frozen=True)
class CleanupReport:
    """Exact cleanup attempts and safe operational failure codes."""

    container_name: str
    stop_attempted: bool
    kill_attempted: bool
    remove_attempted: bool
    failures: tuple[str, ...] = ()

    @property
    def succeeded(self) -> bool:
        return not self.failures


@dataclass(frozen=True)
class InvocationResult:
    """Supervisor result; ``completed`` is true only with trusted evidence."""

    status: str
    container_name: str
    exit: RuntimeExit | None = None
    receipt: TerminalReconciliationReceipt | None = None
    cleanup: CleanupReport | None = None
    evidence: tuple[str, ...] = ()

    @property
    def completed(self) -> bool:
        return self.status == "completed"


def build_invocation_env(policy: InvocationPolicy | None = None) -> dict[str, str]:
    """Return the literal child environment; ambient process state is ignored."""

    if policy is not None:
        if not isinstance(policy, InvocationPolicy):
            raise ContractError("invocation policy is invalid")
    return dict(_CHILD_ENV)


def _validate_binding(run: RunSnapshot, invocation: InvocationEnvelope) -> None:
    if not isinstance(run, RunSnapshot) or not isinstance(invocation, InvocationEnvelope):
        raise ContractError("invocation binding requires runtime contract values")
    if run.protocol != "plane.agent-runtime/v1" or invocation.protocol != run.protocol:
        raise ContractError("invocation protocol is invalid")
    if invocation.run_id != run.run_id or invocation.run_snapshot_digest != run.digest():
        raise BindingError("invocation is not bound to the supplied run snapshot")


def _binding_digest(run: RunSnapshot, invocation: InvocationEnvelope) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "invocationId": invocation.invocation_id,
                "runId": run.run_id,
                "snapshotDigest": run.digest(),
            }
        )
    ).hexdigest()


def invocation_container_name(run: RunSnapshot, invocation: InvocationEnvelope) -> str:
    """Derive one opaque, invocation-bound Docker name without user text."""

    _validate_binding(run, invocation)
    name = f"{_FIXED_CONTAINER_PREFIX}-{_binding_digest(run, invocation)[:32]}"
    if not _CONTAINER_NAME.fullmatch(name):  # pragma: no cover - fixed construction
        raise ContractError("derived container name is invalid")
    return name


def _size(value: int) -> str:
    if value % (1024 * 1024) == 0:
        return f"{value // (1024 * 1024)}m"
    if value % 1024 == 0:
        return f"{value // 1024}k"
    return f"{value}b"


def build_invocation_argv(
    run: RunSnapshot,
    invocation: InvocationEnvelope,
    policy: InvocationPolicy,
) -> tuple[str, ...]:
    """Build the complete fixed Docker argv for one invocation."""

    if not isinstance(policy, InvocationPolicy):
        raise ContractError("invocation policy is invalid")
    _validate_binding(run, invocation)
    name = invocation_container_name(run, invocation)
    binding = _binding_digest(run, invocation)
    argv: list[str] = [
        "docker",
        "run",
        "--name",
        name,
        "--label",
        "plane.agent-runtime/protocol=plane.agent-runtime/v1",
        "--label",
        f"plane.agent-runtime/invocation-binding=sha256:{binding}",
        "--pull",
        policy.pull_policy,
        "--network",
        policy.network_mode,
        "--read-only",
        "--security-opt",
        "no-new-privileges",
        "--cap-drop",
        "ALL",
        "--user",
        policy.user,
        "--cpus",
        f"{policy.cpu_millicores / 1000:.3f}",
        "--memory",
        _size(policy.memory_bytes),
        "--pids-limit",
        str(policy.pids_limit),
        "--stop-timeout",
        str(max(1, int(policy.stop_timeout_seconds))),
        "--tmpfs",
        f"{_FIXED_TMPFS_TARGET}:{_FIXED_TMPFS_OPTIONS},size={_size(policy.tmpfs_bytes)}",
    ]
    for key, value in _CHILD_ENV:
        argv.extend(("--env", f"{key}={value}"))
    argv.extend(
        (
            "--entrypoint",
            policy.entrypoint,
            policy.image,
            "-m",
            policy.service_module,
            *policy.service_args,
        )
    )
    return tuple(argv)


def _request_bytes(
    run: RunSnapshot,
    invocation: InvocationEnvelope,
    policy: InvocationPolicy,
) -> bytes:
    _validate_binding(run, invocation)
    payload = json.dumps(
        {"invocation": invocation.to_dict(), "run": run.to_dict()},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    if len(payload) + 1 > policy.request_limit_bytes:
        raise BoundsError("invocation request exceeds the supervisor request bound")
    return payload + b"\n"


class _NeverCancelled:
    def is_cancelled(self) -> bool:
        return False


def _safe_cancel_check(signal: CancellationSignal) -> tuple[bool, bool]:
    try:
        value = signal.is_cancelled()
    except Exception:
        return True, True
    if not isinstance(value, bool):
        return True, True
    return value, False


@dataclass(frozen=True)
class _ParsedOutput:
    exit: RuntimeExit
    receipt: TerminalReconciliationReceipt
    final_sequence: int


def _parse_child_output(
    stdout: bytes,
    *,
    run: RunSnapshot,
    invocation: InvocationEnvelope,
    policy: InvocationPolicy,
) -> _ParsedOutput:
    if not stdout or len(stdout) > policy.stdout_limit_bytes:
        raise BoundsError("child stdout exceeded its supervisor bound")
    if not stdout.endswith(b"\n"):
        raise ContractError("child stdout ended with a truncated frame")
    collector = EventCollector(
        run_id=run.run_id,
        invocation_id=invocation.invocation_id,
        expected_causation_ref=invocation.causation_ref,
    )
    exit_value: RuntimeExit | None = None
    receipt: TerminalReconciliationReceipt | None = None
    frames = 0
    for raw_line in stdout.splitlines():
        frames += 1
        if frames > policy.max_output_frames:
            raise BoundsError("child frame count exceeded its supervisor bound")
        if not raw_line or len(raw_line) > policy.frame_limit_bytes:
            raise BoundsError("child frame exceeded its supervisor bound")
        try:
            decoded = raw_line.decode("utf-8")
            frame = json.loads(decoded)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ContractError("child emitted malformed JSON output") from exc
        if not isinstance(frame, dict):
            raise ContractError("child emitted a non-object frame")
        kind = frame.get("type")
        if kind == "event":
            if exit_value is not None or receipt is not None:
                raise ContractError("child emitted an event after terminal evidence")
            event = frame.get("event")
            from .contract import RuntimeEvent

            collector.accept(RuntimeEvent.from_dict(event))
        elif kind == "exit":
            if exit_value is not None:
                raise ContractError("child emitted duplicate exit evidence")
            exit_value = RuntimeExit.from_dict(frame.get("exit"))
        elif kind == "reconciliation":
            if receipt is not None:
                raise ContractError("child emitted duplicate terminal receipt")
            receipt = TerminalReconciliationReceipt.from_dict(frame.get("receipt"))
        else:
            # Error and reconciliation-request frames intentionally do not
            # cross this boundary as successful terminal evidence.
            raise ContractError("child emitted an unsupported service frame")
    if exit_value is None or receipt is None:
        raise ContractError("child did not return both exit and terminal receipt evidence")
    if exit_value.final_sequence != collector.last_sequence:
        raise ContractError("child exit sequence does not match bounded event evidence")
    _validate_child_receipt(receipt, exit_value, run, invocation)
    return _ParsedOutput(exit_value, receipt, collector.last_sequence)


def _validate_child_receipt(
    receipt: TerminalReconciliationReceipt,
    exit_value: RuntimeExit,
    run: RunSnapshot,
    invocation: InvocationEnvelope,
) -> None:
    if (
        not receipt.accepted
        or not receipt.legal_transition
        or receipt.run_id != run.run_id
        or receipt.invocation_id != invocation.invocation_id
        or receipt.kind != exit_value.kind
        or receipt.idempotency_key != f"terminal:{run.run_id}:{invocation.invocation_id}"
    ):
        raise BindingError("child terminal receipt is not an accepted invocation receipt")
    for proof in receipt.proofs:
        if (
            proof.run_id != run.run_id
            or proof.invocation_id != invocation.invocation_id
            or proof.actor_ref != run.actor_ref
            or proof.workspace_ref != run.workspace_ref
            or proof.snapshot_digest != run.digest()
            or proof.terminal_slot != receipt.idempotency_key
            or proof.terminal_kind != exit_value.kind
        ):
            raise BindingError("child terminal proof is not bound to the invocation")
    for product in receipt.product_receipts:
        if (
            product.run_id != run.run_id
            or product.invocation_id != invocation.invocation_id
            or product.actor_ref != run.actor_ref
            or product.workspace_ref != run.workspace_ref
            or product.snapshot_digest != run.digest()
            or product.terminal_slot != receipt.idempotency_key
            or product.terminal_kind != exit_value.kind
        ):
            raise BindingError("child product receipt is not bound to the invocation")


class _SubprocessDockerProcess:
    """Selector-driven bounded I/O for the Docker CLI launcher."""

    def __init__(self, process: subprocess.Popen[bytes]) -> None:
        self._process = process

    def collect(
        self,
        *,
        input_bytes: bytes,
        deadline: float,
        stdout_limit_bytes: int,
        stderr_limit_bytes: int,
        is_cancelled: Callable[[], bool],
    ) -> ProcessCapture:
        selector = selectors.DefaultSelector()
        stdout = bytearray()
        stderr = bytearray()
        pending = memoryview(input_bytes)
        try:
            if self._process.stdin is not None:
                selector.register(self._process.stdin, selectors.EVENT_WRITE, "stdin")
            if self._process.stdout is not None:
                selector.register(self._process.stdout, selectors.EVENT_READ, "stdout")
            if self._process.stderr is not None:
                selector.register(self._process.stderr, selectors.EVENT_READ, "stderr")
            while selector.get_map():
                try:
                    cancelled = is_cancelled()
                except Exception:
                    cancelled = True
                if cancelled:
                    return ProcessCapture(self._process.poll(), bytes(stdout), bytes(stderr), cancelled=True)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return ProcessCapture(self._process.poll(), bytes(stdout), bytes(stderr), timed_out=True)
                for key, mask in selector.select(min(remaining, 0.1)):
                    if key.data == "stdin" and mask & selectors.EVENT_WRITE:
                        if not pending:
                            selector.unregister(key.fileobj)
                            key.fileobj.close()
                            continue
                        try:
                            written = os.write(key.fileobj.fileno(), pending[:16 * 1024])
                        except (BrokenPipeError, OSError):
                            selector.unregister(key.fileobj)
                            key.fileobj.close()
                        else:
                            pending = pending[written:]
                            if not pending:
                                selector.unregister(key.fileobj)
                                key.fileobj.close()
                    elif mask & selectors.EVENT_READ:
                        try:
                            chunk = os.read(key.fileobj.fileno(), 16 * 1024)
                        except OSError:
                            chunk = b""
                        if not chunk:
                            selector.unregister(key.fileobj)
                            key.fileobj.close()
                            continue
                        target = stdout if key.data == "stdout" else stderr
                        limit = stdout_limit_bytes if key.data == "stdout" else stderr_limit_bytes
                        if len(target) + len(chunk) > limit:
                            return ProcessCapture(
                                self._process.poll(),
                                bytes(stdout),
                                bytes(stderr),
                                output_exceeded=True,
                            )
                        target.extend(chunk)
                if self._process.poll() is not None and "stdin" in {
                    item.data for item in selector.get_map().values()
                }:
                    key = next(item for item in selector.get_map().values() if item.data == "stdin")
                    selector.unregister(key.fileobj)
                    key.fileobj.close()
            return ProcessCapture(self._process.poll(), bytes(stdout), bytes(stderr))
        finally:
            selector.close()
            for stream in (self._process.stdin, self._process.stdout, self._process.stderr):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass


class SubprocessDockerRunner:
    """Small real Docker CLI runner, usable only with explicit capabilities."""

    def __init__(self, capabilities: DockerRunnerCapabilities) -> None:
        self.capabilities = capabilities

    def launch(
        self,
        argv: Sequence[str],
        *,
        client_env: Mapping[str, str],
        input_bytes: bytes,
    ) -> InvocationProcess:
        del input_bytes
        if dict(client_env) != dict(_DOCKER_CLIENT_ENV):
            raise ContractError("Docker client environment is not the fixed allowlist")
        try:
            name_index = tuple(argv).index("--name") + 1
            name = tuple(argv)[name_index]
        except (ValueError, IndexError) as exc:
            raise ContractError("Docker argv has no valid invocation name") from exc
        process = _SubprocessDockerProcess(
            subprocess.Popen(
                tuple(argv),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=dict(client_env),
                cwd="/",
                close_fds=True,
            )
        )
        return process

    def _control(self, action: str, name: str, timeout_seconds: float) -> None:
        if action == "rm":
            command = ("docker", "rm", "--force", name)
        elif action == "stop":
            command = ("docker", "stop", "--time", "1", name)
        elif action == "kill":
            command = ("docker", "kill", name)
        else:  # pragma: no cover - only called by the fixed cleanup methods
            raise ContractError("unsupported Docker cleanup action")
        completed = subprocess.run(
            command,
            env=dict(_DOCKER_CLIENT_ENV),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=max(0.1, timeout_seconds),
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError("docker cleanup command failed")

    def stop(self, container_name: str, *, timeout_seconds: float) -> None:
        self._control("stop", container_name, timeout_seconds)

    def kill(self, container_name: str, *, timeout_seconds: float) -> None:
        self._control("kill", container_name, timeout_seconds)

    def remove(self, container_name: str, *, timeout_seconds: float) -> None:
        self._control("rm", container_name, timeout_seconds)


class InvocationSupervisor:
    """Run, bound, and dispose of exactly one invocation process."""

    def __init__(
        self,
        *,
        policy: InvocationPolicy,
        runner: DockerRunner,
        terminal_port: TerminalReconciliationPort,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(policy, InvocationPolicy):
            raise ContractError("invocation policy is invalid")
        if terminal_port is None:
            raise RuntimeConfigurationError("invocation supervisor requires terminal reconciliation")
        self.policy = policy
        self.runner = runner
        self.terminal_port = terminal_port
        self._clock = clock
        self._lock = RLock()
        self._results: dict[tuple[str, str], InvocationResult] = {}
        self._death_receipts: dict[tuple[str, str], TerminalReconciliationReceipt | None] = {}

    def _capability_failure(self) -> bool:
        capabilities = getattr(self.runner, "capabilities", None)
        return not isinstance(capabilities, DockerRunnerCapabilities) or bool(capabilities.missing())

    def _cleanup(self, name: str) -> CleanupReport:
        failures: list[str] = []
        for action, timeout in (
            ("stop", self.policy.stop_timeout_seconds),
            ("kill", self.policy.kill_timeout_seconds),
            ("remove", self.policy.remove_timeout_seconds),
        ):
            try:
                getattr(self.runner, action)(name, timeout_seconds=timeout)
            except Exception:
                failures.append(f"{action}_failed")
        return CleanupReport(name, True, True, True, tuple(failures))

    def reconcile_death(
        self,
        run: RunSnapshot,
        invocation: InvocationEnvelope,
        *,
        final_sequence: int = 0,
        reason: str = "runtime process exited before terminal evidence",
    ) -> TerminalReconciliationReceipt | None:
        """Reconcile one process death once; repeated races reuse its receipt."""

        _validate_binding(run, invocation)
        if reason not in _SAFE_DEATH_REASONS:
            reason = _SAFE_DEATH_REASON
        key = (run.run_id, invocation.invocation_id)
        with self._lock:
            if key in self._death_receipts:
                return self._death_receipts[key]
            if len(self._death_receipts) >= _MAX_RETAINED_INVOCATIONS:
                return None
            try:
                receipt = reconcile_process_death(
                    port=self.terminal_port,
                    run=run,
                    invocation_id=invocation.invocation_id,
                    final_sequence=final_sequence,
                    reason=reason,
                )
            except Exception:
                receipt = None
            self._death_receipts[key] = receipt
            return receipt

    def _result_for_death(
        self,
        run: RunSnapshot,
        invocation: InvocationEnvelope,
        *,
        name: str,
        cleanup: CleanupReport,
        final_sequence: int,
        reason: str,
        evidence: tuple[str, ...],
    ) -> InvocationResult:
        receipt = self.reconcile_death(
            run,
            invocation,
            final_sequence=final_sequence,
            reason=reason,
        )
        status = (
            "failed"
            if receipt is not None and receipt.accepted and cleanup.succeeded
            else "supervisor_action_required"
        )
        codes = list(evidence)
        if not cleanup.succeeded or receipt is None:
            codes.append("supervisor_action_required")
        if not cleanup.succeeded:
            codes.append("cleanup_failed")
        return InvocationResult(
            status=status,
            container_name=name,
            exit=classify_process_death(final_sequence=final_sequence, reason=reason),
            receipt=receipt,
            cleanup=cleanup,
            evidence=tuple(dict.fromkeys(codes)),
        )

    def run_once(
        self,
        run: RunSnapshot,
        invocation: InvocationEnvelope,
        *,
        cancellation: CancellationSignal | None = None,
    ) -> InvocationResult:
        """Launch and dispose of one invocation; never replay ``outcome_unknown``."""

        _validate_binding(run, invocation)
        key = (run.run_id, invocation.invocation_id)
        name = invocation_container_name(run, invocation)
        with self._lock:
            prior = self._results.get(key)
            if prior is not None:
                return prior
            if len(self._results) >= _MAX_RETAINED_INVOCATIONS:
                result = InvocationResult(
                    status="supervisor_action_required",
                    container_name=name,
                    evidence=("supervisor_action_required",),
                )
                return result
            if self._capability_failure():
                result = InvocationResult(
                    status="rejected",
                    container_name=name,
                    evidence=("policy_rejected", "runner_capability_missing"),
                )
                self._results[key] = result
                return result
            signal = cancellation or _NeverCancelled()
            cancelled, signal_error = _safe_cancel_check(signal)
            if cancelled:
                evidence = ("cancellation_signal_unavailable",) if signal_error else ("cancellation_signal",)
                result = InvocationResult(
                    status="supervisor_action_required",
                    container_name=name,
                    evidence=evidence + ("supervisor_action_required",),
                )
                self._results[key] = result
                return result
            try:
                argv = build_invocation_argv(run, invocation, self.policy)
                request = _request_bytes(run, invocation, self.policy)
                process = self.runner.launch(
                    argv,
                    client_env=dict(_DOCKER_CLIENT_ENV),
                    input_bytes=request,
                )
            except (ContractError, BoundsError):
                result = InvocationResult(
                    status="rejected",
                    container_name=name,
                    evidence=("policy_rejected",),
                )
                self._results[key] = result
                return result
            except Exception:
                result = InvocationResult(
                    status="supervisor_action_required",
                    container_name=name,
                    evidence=("launch_failed", "supervisor_action_required"),
                )
                self._results[key] = result
                return result

            deadline = self._clock() + self.policy.wall_time_seconds
            try:
                capture = process.collect(
                    input_bytes=request,
                    deadline=deadline,
                    stdout_limit_bytes=self.policy.stdout_limit_bytes,
                    stderr_limit_bytes=self.policy.stderr_limit_bytes,
                    is_cancelled=lambda: _safe_cancel_check(signal)[0],
                )
            except Exception:
                capture = ProcessCapture(None)
            cleanup = self._cleanup(name)
            final_sequence = 0
            if (
                capture.output_exceeded
                or len(capture.stdout) > self.policy.stdout_limit_bytes
                or len(capture.stderr) > self.policy.stderr_limit_bytes
            ):
                reason = "runtime output exceeded supervisor bounds"
                evidence = ("output_limit_exceeded",)
            elif capture.timed_out:
                reason = "runtime invocation exceeded its wall-time bound"
                evidence = ("timeout",)
            elif capture.cancelled:
                reason = "runtime invocation was stopped by a cancellation signal"
                evidence = ("cancellation_signal",)
            elif capture.returncode not in (0, None):
                reason = "runtime process exited before terminal evidence"
                evidence = ("child_nonzero_exit",)
            else:
                try:
                    parsed = _parse_child_output(
                        capture.stdout,
                        run=run,
                        invocation=invocation,
                        policy=self.policy,
                    )
                except Exception:
                    reason = "runtime output was malformed or lacked terminal evidence"
                    evidence = ("invalid_child_output", "missing_terminal_receipt")
                else:
                    final_sequence = parsed.final_sequence
                    if capture.returncode is None:
                        reason = "runtime process died before terminal evidence"
                        evidence = ("child_process_died",)
                    elif not cleanup.succeeded:
                        result = InvocationResult(
                            status="supervisor_action_required",
                            container_name=name,
                            exit=parsed.exit,
                            receipt=parsed.receipt,
                            cleanup=cleanup,
                            evidence=("cleanup_failed", "supervisor_action_required"),
                        )
                        self._results[key] = result
                        return result
                    else:
                        result = InvocationResult(
                            status=parsed.exit.kind,
                            container_name=name,
                            exit=parsed.exit,
                            receipt=parsed.receipt,
                            cleanup=cleanup,
                            evidence=("terminal_receipt",),
                        )
                        self._results[key] = result
                        return result
            result = self._result_for_death(
                run,
                invocation,
                name=name,
                cleanup=cleanup,
                final_sequence=final_sequence,
                reason=reason,
                evidence=evidence,
            )
            self._results[key] = result
            return result


__all__ = [
    "CleanupReport",
    "DockerRunnerCapabilities",
    "InvocationPolicy",
    "InvocationResult",
    "InvocationSupervisor",
    "ProcessCapture",
    "SubprocessDockerRunner",
    "build_invocation_argv",
    "build_invocation_env",
    "invocation_container_name",
]
