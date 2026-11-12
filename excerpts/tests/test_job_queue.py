"""Tests for the durable job queue and its tick (AAA: Arrange, Act, Assert)."""
import json
from datetime import UTC, datetime, timedelta

import pytest

from budget_ledger import Ledger
from job_queue import (
    TRANSITIONS,
    IllegalTransition,
    Job,
    JobQueue,
    JobStateError,
    KernelStatus,
    PushRefused,
    QueueBusy,
    State,
    check_transition,
)

START = datetime(2026, 10, 14, 12, 0, tzinfo=UTC)


class Clock:
    """A settable clock shared by the queue and the ledger."""

    def __init__(self, now: datetime):
        self.now = now

    def __call__(self) -> datetime:
        return self.now


class FakeBackend:
    """Records pushes; push outcomes and statuses are scripted per test."""

    def __init__(self):
        self.pushes = []
        self.push_error = None
        self.statuses = []

    def kernel_slug(self, job_id, spec):
        return f"lab/{spec['name']}"

    def push(self, job_id, spec):
        self.pushes.append(job_id)
        if self.push_error:
            raise self.push_error
        return "kernel version 1"

    def status(self, slug):
        result = self.statuses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


@pytest.fixture
def setup(tmp_path):
    clock = Clock(START)
    ledger = Ledger(tmp_path / "budget.jsonl", cap_hours=10, clock=clock)
    backend = FakeBackend()
    queue = JobQueue(tmp_path / "jobs", backend, ledger, clock=clock)
    return queue, backend, ledger, clock


def spec(name="qlora-smoke", hours=3, needs_gpu=True):
    return {"name": name, "max_hours": hours, "needs_gpu": needs_gpu}


def test_terminal_states_have_no_exits():
    # Arrange
    terminal = (State.EVALUATED, State.FAILED, State.CANCELLED)
    # Act
    exits = [TRANSITIONS[state] for state in terminal]
    # Assert
    assert all(not e for e in exits)


def test_illegal_transition_raises_and_leaves_disk_unchanged(setup):
    # Arrange
    queue, *_ = setup
    job_id = queue.add(spec())
    before = queue.get(job_id).to_dict()
    # Act / Assert
    with pytest.raises(IllegalTransition):
        queue.advance(job_id, State.EVALUATED)
    assert queue.get(job_id).to_dict() == before


def test_check_transition_allows_short_kernel_to_finish_between_polls():
    # Arrange / Act / Assert: no exception
    check_transition(State.SUBMITTED, State.COMPLETE)


def test_add_writes_queued_state_file_with_history(setup):
    # Arrange
    queue, *_ = setup
    # Act
    job_id = queue.add(spec())
    # Assert
    data = json.loads((queue.root / job_id / "state.json").read_text())
    assert job_id == "20261014T120000Z-qlora-smoke"
    assert data["state"] == "queued"
    assert len(data["history"]) == 1


def test_same_second_submissions_get_distinct_ids(setup):
    # Arrange
    queue, *_ = setup
    # Act
    first, second = queue.add(spec()), queue.add(spec())
    # Assert
    assert second == first + "-2"


def test_add_rejects_malformed_spec(setup):
    # Arrange
    queue, *_ = setup
    # Act / Assert
    with pytest.raises(ValueError):
        queue.add({"name": "Bad Name", "max_hours": 1})


def test_get_refuses_path_traversal_ids(setup):
    # Arrange
    queue, *_ = setup
    # Act / Assert
    with pytest.raises(JobStateError):
        queue.get("../../etc")


def test_unknown_state_on_disk_is_an_error():
    # Arrange
    data = {"id": "x", "state": "half-done", "spec": {}}
    # Act / Assert
    with pytest.raises(JobStateError, match="unknown job state"):
        Job.from_dict(data)


def test_tick_reserves_budget_and_records_slug_before_push(setup):
    # Arrange
    queue, backend, ledger, _ = setup
    job_id = queue.add(spec(hours=3))
    # Act
    report = queue.tick()
    # Assert
    job = queue.get(job_id)
    assert report == [f"{job_id}: pushed lab/qlora-smoke"]
    assert job.state is State.SUBMITTED
    assert job.kernel_slug == "lab/qlora-smoke"
    assert ledger.spent() == 3
    assert [h["note"] for h in job.history][1] == "pushing lab/qlora-smoke"


def test_job_waits_when_budget_would_cross_cap(setup):
    # Arrange
    queue, backend, ledger, _ = setup
    ledger.reserve("other", 9)
    job_id = queue.add(spec(hours=3))
    # Act
    queue.tick()
    # Assert
    job = queue.get(job_id)
    assert job.state is State.QUEUED
    assert job.reason.startswith("waiting for budget")
    assert backend.pushes == []


def test_cpu_job_ignores_the_gpu_budget(setup):
    # Arrange
    queue, backend, ledger, _ = setup
    ledger.reserve("other", 10)
    job_id = queue.add(spec(hours=3, needs_gpu=False))
    # Act
    queue.tick()
    # Assert
    assert queue.get(job_id).state is State.SUBMITTED
    assert ledger.spent() == 10


def test_refused_push_requeues_and_refunds_then_fails_after_retries(setup):
    # Arrange
    queue, backend, ledger, _ = setup
    backend.push_error = PushRefused("quota")
    job_id = queue.add(spec(hours=3))
    # Act
    for _ in range(3):
        queue.tick()
    # Assert
    job = queue.get(job_id)
    assert job.state is State.FAILED
    assert job.attempts == 3
    assert job.kernel_slug is None
    assert ledger.spent() == 0


def test_unknown_push_outcome_keeps_slug_and_never_pushes_again(setup):
    # Arrange
    queue, backend, ledger, _ = setup
    backend.push_error = TimeoutError("network")
    job_id = queue.add(spec(hours=3))
    queue.tick()
    backend.statuses = [KernelStatus.RUNNING]
    # Act
    queue.tick()
    # Assert
    job = queue.get(job_id)
    assert backend.pushes == [job_id]
    assert job.state is State.RUNNING
    assert job.kernel_slug == "lab/qlora-smoke"
    assert ledger.spent() == 3


def test_complete_kernel_settles_actual_hours(setup):
    # Arrange
    queue, backend, ledger, clock = setup
    job_id = queue.add(spec(hours=3))
    queue.tick()
    clock.now = START + timedelta(hours=1)
    backend.statuses = [KernelStatus.COMPLETE]
    # Act
    queue.tick()
    # Assert
    assert queue.get(job_id).state is State.COMPLETE
    assert ledger.spent() == pytest.approx(1.0)


def test_settled_hours_are_capped_at_max_hours(setup):
    # Arrange
    queue, backend, ledger, clock = setup
    job_id = queue.add(spec(hours=3))
    queue.tick()
    clock.now = START + timedelta(hours=7)
    backend.statuses = [KernelStatus.COMPLETE]
    # Act
    queue.tick()
    # Assert
    assert ledger.spent_trailing() == 3


def test_kernel_error_fails_at_once(setup):
    # Arrange
    queue, backend, _, _ = setup
    job_id = queue.add(spec())
    queue.tick()
    backend.statuses = [KernelStatus.ERROR]
    # Act
    queue.tick()
    # Assert
    job = queue.get(job_id)
    assert job.state is State.FAILED
    assert job.reason == "kernel reported error (error)"


def test_poll_failures_retry_then_fail_but_keep_the_charge(setup):
    # Arrange
    queue, backend, ledger, _ = setup
    job_id = queue.add(spec(hours=3))
    queue.tick()
    backend.statuses = [OSError("dns")] * 3
    # Act
    for _ in range(3):
        queue.tick()
    # Assert
    assert queue.get(job_id).state is State.FAILED
    assert ledger.spent() == 3


def test_poll_recovery_resets_failure_count(setup):
    # Arrange
    queue, backend, _, _ = setup
    job_id = queue.add(spec())
    queue.tick()
    backend.statuses = [OSError("dns"), KernelStatus.QUEUED]
    # Act
    queue.tick()
    queue.tick()
    # Assert
    job = queue.get(job_id)
    assert job.poll_failures == 0
    assert job.state is State.SUBMITTED


def test_second_job_with_same_slug_waits(setup):
    # Arrange
    queue, backend, _, _ = setup
    first = queue.add(spec())
    queue.tick()
    second = queue.add(spec())
    backend.statuses = [KernelStatus.RUNNING]
    # Act
    queue.tick()
    # Assert
    assert queue.get(first).state is State.RUNNING
    assert queue.get(second).reason.startswith("waiting: kernel slug")


def test_corrupt_job_is_reported_and_others_still_advance(setup):
    # Arrange
    queue, _, _, _ = setup
    bad = queue.add(spec(name="broken-job"))
    (queue.root / bad / "state.json").write_text("{")
    good = queue.add(spec(name="good-job"))
    # Act
    report = queue.tick()
    # Assert
    assert any(line.startswith(f"{bad}: error: JobStateError") for line in report)
    assert queue.get(good).state is State.SUBMITTED


def test_only_queued_jobs_can_be_cancelled(setup):
    # Arrange
    queue, *_ = setup
    job_id = queue.add(spec())
    queue.tick()
    # Act / Assert
    with pytest.raises(IllegalTransition, match="only queued jobs"):
        queue.cancel(job_id)


def test_overlapping_tick_raises_queue_busy(setup):
    # Arrange
    queue, *_ = setup
    # Act / Assert
    with queue.locked(), pytest.raises(QueueBusy):
        queue.tick()
