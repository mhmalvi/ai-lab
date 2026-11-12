"""Tests for the weekly GPU-hour ledger (AAA: Arrange, Act, Assert)."""
from datetime import UTC, datetime, timedelta

import pytest

from budget_ledger import BudgetError, Ledger, check_hours, week_start

# A Wednesday; the quota week started on the Saturday before.
WEDNESDAY = datetime(2026, 10, 14, 12, 0, tzinfo=UTC)


class Clock:
    """A settable clock for the ledger."""

    def __init__(self, now: datetime):
        self.now = now

    def __call__(self) -> datetime:
        return self.now


def make_ledger(tmp_path, now=WEDNESDAY, cap=25.0):
    clock = Clock(now)
    return Ledger(tmp_path / "budget.jsonl", cap_hours=cap, clock=clock), clock


def test_week_start_is_previous_saturday_midnight_utc():
    # Arrange
    now = WEDNESDAY
    # Act
    start = week_start(now)
    # Assert
    assert start == datetime(2026, 10, 10, 0, 0, tzinfo=UTC)
    assert start.weekday() == 5


def test_week_start_on_saturday_midnight_is_that_instant():
    # Arrange
    saturday = datetime(2026, 10, 10, 0, 0, tzinfo=UTC)
    # Act
    start = week_start(saturday)
    # Assert
    assert start == saturday


def test_week_start_rejects_naive_datetime():
    # Arrange
    naive = datetime(2026, 10, 14, 12, 0)
    # Act / Assert
    with pytest.raises(BudgetError):
        week_start(naive)


@pytest.mark.parametrize("bad", [True, "3", float("nan"), float("inf"), -1, None])
def test_check_hours_rejects_non_finite_negative_and_non_numbers(bad):
    # Arrange: bad value from the parameter list
    # Act / Assert
    with pytest.raises(BudgetError):
        check_hours(bad)


def test_settle_replaces_reservation_instead_of_adding(tmp_path):
    # Arrange
    ledger, _ = make_ledger(tmp_path)
    ledger.reserve("job-a", 10)
    # Act
    ledger.settle("job-a", 2.5)
    # Assert
    assert ledger.spent() == 2.5


def test_repeated_reservation_after_crash_is_not_double_counted(tmp_path):
    # Arrange
    ledger, _ = make_ledger(tmp_path)
    ledger.reserve("job-a", 6)
    # Act
    ledger.reserve("job-a", 6)
    # Assert
    assert ledger.spent() == 6


def test_can_spend_refuses_when_cap_would_be_crossed(tmp_path):
    # Arrange
    ledger, _ = make_ledger(tmp_path, cap=10)
    ledger.reserve("job-a", 8)
    # Act
    fits, too_much = ledger.can_spend(2), ledger.can_spend(2.5)
    # Assert
    assert fits is True
    assert too_much is False


def test_can_spend_does_not_count_the_candidates_own_reservation(tmp_path):
    # Arrange
    ledger, _ = make_ledger(tmp_path, cap=10)
    ledger.reserve("job-a", 8)
    # Act
    allowed = ledger.can_spend(8, job_id="job-a")
    # Assert
    assert allowed is True


def test_trailing_window_still_counts_hours_from_before_the_reset(tmp_path):
    # Arrange: hours charged on Friday, checked on Sunday (a new calendar week)
    friday = datetime(2026, 10, 16, 20, 0, tzinfo=UTC)
    ledger, clock = make_ledger(tmp_path, now=friday, cap=10)
    ledger.reserve("job-a", 9)
    clock.now = friday + timedelta(days=2)
    # Act
    allowed = ledger.can_spend(2)
    # Assert
    assert ledger.spent() == 0
    assert ledger.spent_trailing() == 9
    assert allowed is False


def test_settle_keeps_job_in_the_week_it_was_first_charged(tmp_path):
    # Arrange
    friday = datetime(2026, 10, 16, 20, 0, tzinfo=UTC)
    ledger, clock = make_ledger(tmp_path, now=friday)
    ledger.reserve("job-a", 5)
    clock.now = friday + timedelta(days=2)
    # Act
    ledger.settle("job-a", 4)
    # Assert
    assert ledger.spent() == 0
    assert ledger.charges()["job-a"][0] == friday


def test_malformed_line_fails_closed(tmp_path):
    # Arrange
    ledger, _ = make_ledger(tmp_path)
    ledger.reserve("job-a", 1)
    with ledger.path.open("a", encoding="utf-8") as handle:
        handle.write("{not json\n")
    # Act / Assert
    with pytest.raises(BudgetError, match=":2: malformed"):
        ledger.can_spend(1)


def test_ensure_creates_owner_only_file_and_never_truncates(tmp_path):
    # Arrange
    ledger, _ = make_ledger(tmp_path)
    ledger.ensure()
    ledger.reserve("job-a", 1)
    # Act
    ledger.ensure()
    # Assert
    assert ledger.path.stat().st_mode & 0o777 == 0o600
    assert ledger.spent() == 1


def test_ensure_refuses_a_directory_in_place_of_the_ledger(tmp_path):
    # Arrange
    ledger, _ = make_ledger(tmp_path)
    ledger.path.mkdir()
    # Act / Assert
    with pytest.raises(BudgetError, match="not a regular file"):
        ledger.ensure()


def test_empty_job_id_is_refused(tmp_path):
    # Arrange
    ledger, _ = make_ledger(tmp_path)
    # Act / Assert
    with pytest.raises(BudgetError):
        ledger.reserve("", 1)
