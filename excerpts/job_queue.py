"""Durable job queue: one ``<jobs>/<id>/state.json`` per job, advanced by an idempotent tick.

Excerpt from the private lab's queue module. Sanitized and trimmed:
  * kept: the state enum and transition table, atomic state writes, the job
    record, the non-blocking queue lock, and the remote-kernel tick (budget gate,
    slug-before-push ordering, refused vs unknown push outcomes, bounded retries,
    settling actual hours);
  * removed: the detached local-run backend (pid / boot-id tracking, relaunch
    after a power cut, max_hours kill), the pull-and-evaluate finisher, and the
    full job-spec validation (see ``jobspec.py``). The spec here is a plain dict
    with ``name``, ``max_hours`` and an optional ``needs_gpu``.

States run queued -> submitted -> running -> complete -> pulled -> evaluated, with
failed and cancelled as the other terminal states. Every move is checked against
``TRANSITIONS``; anything else raises ``IllegalTransition``.

Power-cut safety rests on one ordering rule: the kernel slug is written to disk
(state ``submitted``) BEFORE the push. So after any crash a job either has no slug
(it was never pushed, push it) or has one (it may have been pushed, poll it, never
push again). The remote names a kernel by its slug, so the slug is known in advance.

Retries (at most ``MAX_RETRIES`` after the first try) cover lab-side and transport
failures: a refused push, a failed or unmappable status poll. A kernel that ran and
reported ERROR is failed at once, because rerunning broken code only burns GPU-hours.

In the lab, a systemd user timer runs the tick every few minutes.
"""

from __future__ import annotations

import contextlib
import enum
import fcntl
import json
import os
import re
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

MAX_RETRIES = 2
STATE_FILE = "state.json"
JOB_ID_RE = re.compile(r"^\d{8}T\d{6}Z-[a-z0-9][a-z0-9-]*$")
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,48}[a-z0-9]$")


class State(enum.Enum):
    """Every state a job can be in; anything else on disk is an error."""

    QUEUED = "queued"
    SUBMITTED = "submitted"
    RUNNING = "running"
    COMPLETE = "complete"
    PULLED = "pulled"
    EVALUATED = "evaluated"
    FAILED = "failed"
    CANCELLED = "cancelled"


class KernelStatus(enum.Enum):
    """What the remote compute reports about a kernel."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETE = "complete"
    ERROR = "error"
    CANCELLED = "cancelled"


S = State
TRANSITIONS: dict[State, frozenset[State]] = {
    S.QUEUED: frozenset({S.SUBMITTED, S.FAILED, S.CANCELLED}),
    # submitted -> queued only when the remote explicitly refused the push (nothing runs).
    # submitted -> complete: a short kernel can finish between two polls.
    S.SUBMITTED: frozenset({S.QUEUED, S.RUNNING, S.COMPLETE, S.FAILED, S.CANCELLED}),
    S.RUNNING: frozenset({S.COMPLETE, S.FAILED, S.CANCELLED}),
    S.COMPLETE: frozenset({S.PULLED, S.FAILED}),
    S.PULLED: frozenset({S.EVALUATED, S.FAILED}),
    S.EVALUATED: frozenset(),
    S.FAILED: frozenset(),
    S.CANCELLED: frozenset(),
}
POLLED = frozenset({S.SUBMITTED, S.RUNNING})
# The remote's view of a kernel -> the job state it moves to (None = no change).
POLL_TARGET = {
    KernelStatus.QUEUED: None,
    KernelStatus.RUNNING: S.RUNNING,
    KernelStatus.COMPLETE: S.COMPLETE,
    KernelStatus.ERROR: S.FAILED,
    KernelStatus.CANCELLED: S.CANCELLED,
}


class JobStateError(RuntimeError):
    """A job id or state file is invalid, missing or corrupt."""


class IllegalTransition(JobStateError):
    """A state change not in ``TRANSITIONS`` was requested."""


class QueueBusy(RuntimeError):
    """Another process holds the queue lock (e.g. a timer tick overlapping the CLI)."""


class PushRefused(RuntimeError):
    """The remote explicitly rejected a push: nothing is running."""


class KernelBackend(Protocol):
    """The part of the remote backend the queue drives."""

    def kernel_slug(self, job_id: str, spec: dict) -> str:
        """Return the slug the kernel will have, before anything is pushed."""
        ...

    def push(self, job_id: str, spec: dict) -> str:
        """Push and start the kernel; raise ``PushRefused`` if the remote says no.

        Any ``OSError``/``TimeoutError`` means the outcome is unknown.
        """
        ...

    def status(self, slug: str) -> KernelStatus:
        """Return the kernel's current status."""
        ...


class GpuLedger(Protocol):
    """The part of ``budget_ledger.Ledger`` the queue uses."""

    cap_hours: float

    def ensure(self) -> None:
        """Create the ledger file if absent."""
        ...

    def can_spend(self, hours: float, job_id: str | None = None) -> bool:
        """Return whether ``hours`` more fit under the cap."""
        ...

    def reserve(self, job_id: str, hours: float) -> None:
        """Charge ``hours`` before a push."""
        ...

    def settle(self, job_id: str, hours: float) -> None:
        """Replace a job's charge with the hours it used."""
        ...


def utc_now() -> datetime:
    """Return the current time as an aware UTC datetime."""
    return datetime.now(UTC)


def check_transition(current: State, target: State) -> None:
    """Raise ``IllegalTransition`` unless ``current -> target`` is in the table."""
    if target not in TRANSITIONS[current]:
        raise IllegalTransition(f"{current.value} -> {target.value} is not allowed")


def atomic_write_json(path: Path, data: dict) -> None:
    """Replace ``path`` with ``data`` so a reader sees the old or the new file, never half.

    Writes ``<path>.tmp``, fsyncs it, renames it over ``path`` and fsyncs the
    directory so the rename itself survives a power cut.

    Args:
        path: the destination file.
        data: a JSON-serialisable mapping.
    """
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(data, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    dir_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


@dataclass
class Job:
    """One job's durable record (the content of its state.json)."""

    id: str
    state: State
    spec: dict
    kernel_slug: str | None = None
    attempts: int = 0  # failed submit attempts
    poll_failures: int = 0  # consecutive failed polls; reset by a good one
    pushed_at: str | None = None
    reason: str | None = None
    history: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Return the JSON form written to disk."""
        return asdict(self) | {"state": self.state.value}

    @classmethod
    def from_dict(cls, data: object) -> Job:
        """Rebuild a job from its JSON form.

        Args:
            data: the parsed state file.

        Returns:
            The job.

        Raises:
            JobStateError: unknown state, missing or extra fields.
        """
        if not isinstance(data, dict):
            raise JobStateError("state file must hold a JSON object")
        try:
            state = State(data.get("state"))
        except ValueError:
            raise JobStateError(f"unknown job state {data.get('state')!r}") from None
        try:
            return cls(**(data | {"state": state}))
        except TypeError as exc:
            raise JobStateError(f"bad state file fields: {exc}") from None


def check_spec(spec: object) -> dict:
    """Return a copy of the minimal spec this excerpt needs, or raise ``ValueError``.

    Args:
        spec: ``{"name": str, "max_hours": positive number, "needs_gpu": bool (optional)}``.

    Returns:
        A new dict with ``needs_gpu`` defaulted to True (unknown means it costs GPU).
    """
    if not isinstance(spec, dict) or not NAME_RE.fullmatch(str(spec.get("name", ""))):
        raise ValueError("spec needs a lowercase-hyphen name")
    hours = spec.get("max_hours")
    if isinstance(hours, bool) or not isinstance(hours, int | float) or not hours > 0:
        raise ValueError(f"max_hours must be a positive number, got {hours!r}")
    return {"needs_gpu": True} | spec


class JobQueue:
    """The jobs directory, its state machine, and the tick that drives remote kernels."""

    def __init__(
        self,
        root: str | Path,
        backend: KernelBackend,
        ledger: GpuLedger,
        clock: Callable[[], datetime] = utc_now,
    ):
        """Wire the queue.

        Args:
            root: jobs directory.
            backend: the remote kernel backend.
            ledger: GPU-hour ledger.
            clock: returns the current aware datetime (injectable for tests).
        """
        self.root = Path(root)
        self.backend = backend
        self.ledger = ledger
        self.clock = clock

    # --- storage ---------------------------------------------------------------

    def _path(self, job_id: str) -> Path:
        """Return the state file of ``job_id``, refusing ids that could escape the root."""
        if not isinstance(job_id, str) or not JOB_ID_RE.fullmatch(job_id):
            raise JobStateError(f"malformed job id {job_id!r}")
        return self.root / job_id / STATE_FILE

    def add(self, spec: dict) -> str:
        """Queue a spec and return its new job id (``<UTC stamp>-<name>[-n]``).

        Raises:
            ValueError: if the spec is malformed (see ``check_spec``).
        """
        spec = check_spec(spec)
        stem = f"{self.clock():%Y%m%dT%H%M%SZ}-{spec['name']}"
        job_id, n = stem, 1
        self.root.mkdir(parents=True, exist_ok=True)
        while (self.root / job_id).exists():
            n += 1
            job_id = f"{stem}-{n}"
        (self.root / job_id).mkdir()
        self._save(Job(id=job_id, state=S.QUEUED, spec=spec), note=None)
        return job_id

    def get(self, job_id: str) -> Job:
        """Load one job from disk.

        Raises:
            JobStateError: bad id, missing or corrupt file, unknown state.
        """
        path = self._path(job_id)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise JobStateError(f"cannot read {path}") from None
        job = Job.from_dict(data)
        if job.id != job_id:
            raise JobStateError(f"{path} holds job {job.id!r}")
        return job

    def ids(self) -> list[str]:
        """Return every job id in submission order."""
        if not self.root.is_dir():
            return []
        return sorted(p.name for p in self.root.iterdir() if JOB_ID_RE.fullmatch(p.name))

    def _save(self, job: Job, note: str | None) -> None:
        """Record ``note`` as the reason, append history and write the file atomically."""
        job.reason = note
        job.history.append({"at": self.clock().isoformat(), "state": job.state.value, "note": note})
        atomic_write_json(self._path(job.id), job.to_dict())

    def _move(self, job: Job, target: State, note: str | None) -> None:
        """Check and apply ``job -> target``, then persist it."""
        check_transition(job.state, target)
        job.state = target
        self._save(job, note)

    def advance(self, job_id: str, target: State, note: str | None = None) -> Job:
        """Move a job to ``target`` (used by pull/eval) and return it.

        Raises:
            IllegalTransition: if the move is not in the table; disk is unchanged.
        """
        job = self.get(job_id)
        self._move(job, target, note)
        return job

    def cancel(self, job_id: str) -> Job:
        """Cancel a job that has not been pushed.

        A pushed kernel cannot be stopped from here, so cancelling one would only
        hide GPU time that is still being spent.

        Raises:
            IllegalTransition: unless the job is queued.
        """
        job = self.get(job_id)
        if job.state is not S.QUEUED:
            raise IllegalTransition(
                f"only queued jobs can be cancelled, {job_id} is {job.state.value}"
            )
        self._move(job, S.CANCELLED, "cancelled by operator")
        return job

    @contextlib.contextmanager
    def locked(self) -> Iterator[None]:
        """Hold the queue's exclusive lock, or raise ``QueueBusy`` at once."""
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / ".lock").open("w") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise QueueBusy(f"{self.root} is locked by another tick") from None
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    # --- tick ------------------------------------------------------------------

    def tick(self) -> list[str]:
        """Advance every live job by at most one step; safe to run any number of times.

        Returns:
            One ``"<job id>: <what happened>"`` line per job acted on. An error in
            one job is reported here and does not stop the others.

        Raises:
            QueueBusy: if another tick holds the lock.
        """
        report = []
        with self.locked():
            self.ledger.ensure()
            for job_id in self.ids():
                try:
                    line = self._step(self.get(job_id))
                except Exception as exc:  # noqa: BLE001 - one bad job must not stall the queue
                    line = f"error: {type(exc).__name__}: {exc}"
                if line:
                    report.append(f"{job_id}: {line}")
        return report

    def _step(self, job: Job) -> str | None:
        """Submit a queued job or poll a submitted/running one."""
        if job.state is S.QUEUED:
            return self._submit(job)
        if job.state in POLLED:
            return self._poll(job)
        return None

    def _wait(self, job: Job, note: str) -> str:
        """Keep a job queued with ``note``; write only if the note changed."""
        if job.reason != note:
            self._save(job, note)
        return note

    def _submit_blocker(self, job: Job, slug: str) -> str | None:
        """Return why ``job`` must stay queued for now, or None if it may be pushed."""
        for other_id in self.ids():
            if other_id == job.id:
                continue
            with contextlib.suppress(JobStateError):
                other = self.get(other_id)
                if other.state in POLLED and other.kernel_slug == slug:
                    return f"waiting: kernel slug {slug} is in use by another job"
        hours = job.spec["max_hours"]
        if job.spec["needs_gpu"] and not self.ledger.can_spend(hours, job.id):
            return (
                f"waiting for budget: {hours:g} h would cross the "
                f"{self.ledger.cap_hours:g} GPU-h weekly cap"
            )
        return None

    def _submit(self, job: Job) -> str:
        """Reserve budget, record the slug, then push, in that order (see module doc)."""
        try:
            slug = self.backend.kernel_slug(job.id, job.spec)
        except Exception as exc:  # noqa: BLE001 - any build failure is a counted attempt
            return self._attempt_failed(job, f"kernel build failed: {exc}")
        blocker = self._submit_blocker(job, slug)
        if blocker:
            return self._wait(job, blocker)
        if job.spec["needs_gpu"]:
            self.ledger.reserve(job.id, job.spec["max_hours"])
        job.kernel_slug, job.pushed_at = slug, self.clock().isoformat()
        self._move(job, S.SUBMITTED, f"pushing {slug}")
        return self._push(job)

    def _push(self, job: Job) -> str:
        """Push the kernel of a job already recorded as submitted."""
        try:
            out = self.backend.push(job.id, job.spec)
        except PushRefused as exc:
            # The remote said no: nothing is running, so un-record the slug and charge.
            if job.spec["needs_gpu"]:
                self.ledger.settle(job.id, 0.0)
            job.kernel_slug = job.pushed_at = None
            return self._attempt_failed(job, f"push refused: {exc}")
        except (OSError, TimeoutError) as exc:
            # The push may or may not have arrived: keep the slug and poll it.
            note = f"push outcome unknown ({type(exc).__name__}); will poll {job.kernel_slug}"
            self._save(job, note)
            return note
        self._save(job, f"pushed: {out}")
        return f"pushed {job.kernel_slug}"

    def _attempt_failed(self, job: Job, why: str) -> str:
        """Count a failed submit; requeue it, or fail it once retries are spent."""
        job.attempts += 1
        if job.attempts > MAX_RETRIES:
            self._move(job, S.FAILED, f"gave up after {job.attempts} attempts: {why}")
            return f"failed: {why}"
        target = S.QUEUED if job.state is S.SUBMITTED else job.state
        if target is job.state:
            self._save(job, f"attempt {job.attempts} failed: {why}")
        else:
            self._move(job, target, f"attempt {job.attempts} failed: {why}")
        return f"retry later: {why}"

    def _poll(self, job: Job) -> str | None:
        """Ask the remote about a submitted/running job's stored slug and follow its answer."""
        try:
            status = self.backend.status(job.kernel_slug)
            if status not in POLL_TARGET:
                raise ValueError(f"kernel status {status!r} cannot be mapped")
        except Exception as exc:  # noqa: BLE001 - any poll failure is counted
            return self._poll_failed(job, f"{type(exc).__name__}: {exc}")
        if job.poll_failures:
            job.poll_failures = 0
            self._save(job, f"poll recovered: remote says {status.value}")
        target = POLL_TARGET[status]
        if target is None or target is job.state:
            return None
        if target in (S.COMPLETE, S.FAILED, S.CANCELLED):
            self._settle(job)
        note = f"kernel reported {status.value}" + (" (error)" if target is S.FAILED else "")
        self._move(job, target, note)
        return note

    def _poll_failed(self, job: Job, why: str) -> str:
        """Count a failed poll; fail the job after ``MAX_RETRIES`` retries in a row.

        The ledger keeps the reservation: the kernel may well have run.
        """
        job.poll_failures += 1
        if job.poll_failures > MAX_RETRIES:
            self._move(job, S.FAILED, f"gave up polling after {job.poll_failures} tries: {why}")
            return f"failed: {why}"
        self._save(job, f"poll {job.poll_failures} failed: {why}")
        return f"poll failed: {why}"

    def _settle(self, job: Job) -> None:
        """Charge a finished kernel's elapsed time, capped by its timeout (max_hours)."""
        if not job.spec["needs_gpu"] or job.pushed_at is None:
            return
        elapsed = (self.clock() - datetime.fromisoformat(job.pushed_at)).total_seconds() / 3600
        self.ledger.settle(job.id, min(max(elapsed, 0.0), job.spec["max_hours"]))
