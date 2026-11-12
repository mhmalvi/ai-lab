"""Weekly GPU-hour budget: an append-only JSON-lines ledger of kernel hours.

Excerpt from the private lab's budget module. Sanitized and trimmed:
  * kept: the reserve/settle event model, the two-window cap check, the
    fail-closed parser and the fsync'd append;
  * removed: the lab's config lookup for the default ledger path (the path is
    a required argument here).

Each line is one event for one job: ``reserve`` (the job's ``max_hours``, written
before its kernel is pushed) or ``settle`` (the hours it actually used, written when
it finishes). A job's charge is its LATEST event, so a reservation repeated after a
power cut replaces itself instead of double-counting, and the file is only appended.

The cap is checked two ways and both must pass:

- the calendar week, starting at the provider's assumed quota reset
  (Saturday 00:00 UTC; the provider does not document it);
- the trailing 7 days. Any real quota week is a 7-day window, so this second check
  keeps the lab under the cap even if the assumed reset day is wrong.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

CAP_HOURS = 25.0  # the lab's own weekly limit, set below the provider's quota as headroom
RESET_WEEKDAY = 5  # Saturday (Monday == 0)
RESET_HOUR_UTC = 0
WEEK = timedelta(days=7)
EVENTS = frozenset({"reserve", "settle"})


class BudgetError(RuntimeError):
    """The ledger is unreadable or a request is malformed; never treated as 'budget free'."""


def utc_now() -> datetime:
    """Return the current time as an aware UTC datetime."""
    return datetime.now(UTC)


def week_start(now: datetime) -> datetime:
    """Return the start of the quota week containing ``now``.

    Args:
        now: an aware datetime.

    Returns:
        The most recent Saturday 00:00 UTC at or before ``now``.

    Raises:
        BudgetError: if ``now`` is naive (its UTC instant would be a guess).
    """
    if now.tzinfo is None:
        raise BudgetError("naive datetime: the budget week is defined in UTC")
    now = now.astimezone(UTC)
    start = now.replace(hour=RESET_HOUR_UTC, minute=0, second=0, microsecond=0)
    start -= timedelta(days=(now.weekday() - RESET_WEEKDAY) % 7)
    return start if start <= now else start - WEEK


def check_hours(hours: object) -> float:
    """Return ``hours`` as a float if it is a finite, non-negative number.

    Args:
        hours: the candidate value.

    Returns:
        The value as a float.

    Raises:
        BudgetError: for bools, strings, NaN, infinities and negatives.
    """
    if isinstance(hours, bool) or not isinstance(hours, int | float):
        raise BudgetError(f"hours must be a number, got {hours!r}")
    if not math.isfinite(hours) or hours < 0:
        raise BudgetError(f"hours must be finite and >= 0, got {hours!r}")
    return float(hours)


class Ledger:
    """The GPU-hour ledger file plus the cap it enforces."""

    def __init__(
        self,
        path: str | Path,
        cap_hours: float = CAP_HOURS,
        clock: Callable[[], datetime] = utc_now,
    ):
        """Wire the ledger.

        Args:
            path: JSON-lines ledger file (created by ``ensure`` or the first write).
            cap_hours: the weekly limit.
            clock: returns the current aware datetime (injectable for tests).
        """
        self.path = Path(path)
        self.cap_hours = check_hours(cap_hours)
        self.clock = clock

    def ensure(self) -> None:
        """Create the ledger as an empty owner-only file if it is absent; never truncate it.

        Raises:
            BudgetError: if the path exists but is not a regular file.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            # O_EXCL: only a brand-new file is created; an existing ledger is left as is.
            os.close(os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600))
        except FileExistsError:
            if not self.path.is_file():
                raise BudgetError(f"{self.path} is not a regular file") from None

    def reserve(self, job_id: str, hours: object) -> None:
        """Charge ``hours`` to ``job_id`` before its kernel starts (replaces earlier events)."""
        self._append("reserve", job_id, check_hours(hours))

    def settle(self, job_id: str, hours: object) -> None:
        """Replace ``job_id``'s charge with the hours it actually used."""
        self._append("settle", job_id, check_hours(hours))

    def _append(self, event: str, job_id: str, hours: float) -> None:
        """Append one event and fsync it, so a charge survives a power cut."""
        if not isinstance(job_id, str) or not job_id:
            raise BudgetError(f"job_id must be a non-empty string, got {job_id!r}")
        line = {"at": self.clock().isoformat(), "event": event, "job_id": job_id, "hours": hours}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(line, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def charges(self) -> dict[str, tuple[datetime, float]]:
        """Return job_id -> (time of its first event, hours of its latest event).

        A job is dated by its first event so that settling it later does not move
        its hours into a newer week.

        Raises:
            BudgetError: on any unreadable or malformed line (fail closed).
        """
        if not self.path.exists():
            return {}
        charges: dict[str, tuple[datetime, float]] = {}
        for number, raw in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            at, job_id, hours = self._parse(number, raw)
            first = charges[job_id][0] if job_id in charges else at
            charges[job_id] = (first, hours)
        return charges

    def _parse(self, number: int, raw: str) -> tuple[datetime, str, float]:
        """Parse ledger line ``number`` into (time, job_id, hours)."""
        try:
            entry = json.loads(raw)
            at = datetime.fromisoformat(entry["at"])
            if entry["event"] not in EVENTS or at.tzinfo is None:
                raise ValueError(entry["event"])
            return at, str(entry["job_id"]), check_hours(entry["hours"])
        except (ValueError, KeyError, TypeError, BudgetError):
            raise BudgetError(f"{self.path}:{number}: malformed ledger line") from None

    def _sum_since(self, start: datetime, exclude: str | None) -> float:
        """Return hours charged to jobs first seen at or after ``start``."""
        return sum(
            hours
            for job_id, (at, hours) in self.charges().items()
            if at >= start and job_id != exclude
        )

    def spent(self, exclude: str | None = None) -> float:
        """Return hours charged in the current quota week (optionally minus one job)."""
        return self._sum_since(week_start(self.clock()), exclude)

    def spent_trailing(self, exclude: str | None = None) -> float:
        """Return hours charged in the last 7 days (optionally minus one job)."""
        return self._sum_since(self.clock() - WEEK, exclude)

    def can_spend(self, hours: object, job_id: str | None = None) -> bool:
        """Return whether ``hours`` more fit under the cap in both windows.

        Args:
            hours: the candidate job's ``max_hours``.
            job_id: the candidate itself, whose own earlier reservation is not counted
                twice (it is re-reserved, which replaces it).

        Raises:
            BudgetError: malformed ``hours`` or an unreadable ledger.
        """
        hours = check_hours(hours)
        used = max(self.spent(job_id), self.spent_trailing(job_id))
        return used + hours <= self.cap_hours + 1e-9
