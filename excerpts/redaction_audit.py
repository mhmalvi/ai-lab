"""Independent check that redacted email text carries no personal data.

Excerpt from the private lab's redaction audit. Sanitized and trimmed:
  * kept: every detector, the Luhn and IBAN checks, name-part matching and the
    fail-closed ``audit_or_raise``;
  * changed: the list of words that mark a display name as a system or team
    (not a person) is shortened.

A separate redaction step (not in this repo) masks the text first. This module
is the second key: it scans the result with its OWN detectors and reports what
is left. It never returns or logs the matched text. A ``Hit`` holds only the kind
and the character span, so hit counts can go into reports and logs safely.

Detectors (placeholders such as ``[EMAIL]``, ``[PHONE]``, ``[NAME]`` never match):

- ``email``: any ``local@domain`` with a real domain; ``local@[DOMAIN]`` unless
  the local part is a role mailbox (mailer-daemon, postmaster, no-reply, ...).
- ``phone``: 9-15 digits written like a phone (``+``/``00``/``(`` prefix or
  separators), a solid ``0``-prefixed mobile run, or a ``tel:`` link. Dates,
  times, SMTP status codes and IPs are not phones.
- ``url_token``: a URL with a query, fragment, ``=`` or a long opaque path
  segment, and orphan ``?key=value`` / ``&key=value`` tails.
- ``card``: a Luhn-valid run of 13-19 digits (spaces/dashes allowed).
- ``iban``: country code + check digits + BBAN that passes the mod-97 check.
- ``name``: any part (>= 3 letters) of a name the caller supplies, usually
  taken from the item's ``sender_name`` and the local part of ``sender``.

``audit_or_raise`` fails closed: any hit raises ``RedactionAuditError``, whose
message carries counts per kind and the item ids, never text.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

KINDS = ("email", "phone", "url_token", "card", "iban", "name")

ROLE_MAILBOXES = frozenset(
    {
        "mailer-daemon", "postmaster", "no-reply", "noreply", "do-not-reply", "donotreply",
        "do_not_reply", "notifications", "notification", "bounce", "bounces",
    }
)  # fmt: skip
# Words that make a display name a system or a team, not a person. Such names
# are signal for the classifier (e.g. "Mail Delivery Subsystem"), not personal data.
NON_PERSON_WORDS = frozenset(
    {
        "mail", "mailer", "delivery", "subsystem", "system", "systems", "postmaster", "daemon",
        "team", "support", "notification", "notifications", "noreply", "no-reply", "reply",
        "service", "services", "help", "desk", "helpdesk", "admin", "info", "office",
        "department", "group", "inc", "ltd", "llc", "gmbh", "bounce", "bounces", "contact",
        "hello", "sales", "news", "newsletter",
    }
)  # fmt: skip

_EMAIL = re.compile(r"(?<![\w.+%-])([A-Za-z0-9._%+-]+)@([A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+)")
_EMAIL_MASKED_DOMAIN = re.compile(r"(?<![\w.+%-])([A-Za-z0-9._%+-]+)@\[DOMAIN\]")
_URL = re.compile(r"\bhttps?://[^\s<>\"')\]]+", re.I)
_ORPHAN_PARAM = re.compile(r"[?&][\w.-]+=[^\s&)\]>\"']+")
_TEL = re.compile(r"(?i)\btel:\s*\+?[\d\s().-]{6,}\d")
_PHONE = re.compile(r"(?<![\w.+/:-])(?:\+|00|\()?\d[\d\s().-]{6,22}\d(?![\w/-]|\.\d)")
_PHONE_SOLID = re.compile(r"(?<![\w.+-])(?:\+\d{9,14}|0\d{9,12})(?![\w-])")
_DIGIT_RUN = re.compile(r"(?<![\w-])\d(?:[ -]?\d){12,18}(?![\w-])")
_IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]){11,30}\b")
_DATE = re.compile(r"\d{4}[-./]\d{1,2}[-./]\d{1,2}|\d{1,2}[-./]\d{1,2}[-./]\d{2,4}")
_TIME = re.compile(r"\d{1,2}:\d{2}")
_SMTP_CODE = re.compile(r"^[245]\d\d[\s-][245]\.\d{1,3}\.\d{1,3}$")
_IPV4 = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")
_NAME_SPLIT = re.compile(r"[\s,.()_\-+\"']+")


class RedactionAuditError(RuntimeError):
    """Redacted text still carries personal data. The message holds counts and ids, never text."""


@dataclass(frozen=True)
class Hit:
    """One finding: the detector kind and the span in the audited text (no text)."""

    kind: str
    start: int
    end: int
    item_id: str | None = None


def _email_hits(text: str) -> list[Hit]:
    """Real-domain addresses, and masked-domain ones with a personal local part."""
    hits = [Hit("email", m.start(), m.end()) for m in _EMAIL.finditer(text)]
    hits += [
        Hit("email", m.start(), m.end())
        for m in _EMAIL_MASKED_DOMAIN.finditer(text)
        if m.group(1).lower() not in ROLE_MAILBOXES
    ]
    return hits


def url_is_tokenised(url: str) -> bool:
    """Return True when a URL carries a query, fragment, ``=`` or a long opaque path segment.

    Args:
        url: an ``http(s)://`` URL.

    Returns:
        Whether the URL may carry a per-recipient token.
    """
    _, _, rest = url.partition("://")
    _, _, path = rest.partition("/")
    if path == "[URL-TOKEN]":
        return False
    if any(ch in url for ch in "?#="):
        return True
    return any(
        len(segment) >= 16 and re.search(r"\d", segment) for segment in re.split(r"[/&;]", path)
    )


def _url_hits(text: str) -> list[Hit]:
    """Token-carrying URLs plus orphan query-parameter tails outside a URL."""
    hits = [
        Hit("url_token", m.start(), m.end())
        for m in _URL.finditer(text)
        if url_is_tokenised(m.group(0))
    ]
    url_spans = [(m.start(), m.end()) for m in _URL.finditer(text)]
    for m in _ORPHAN_PARAM.finditer(text):
        if not any(s <= m.start() < e for s, e in url_spans):
            hits.append(Hit("url_token", m.start(), m.end()))
    return hits


def _looks_like_phone(candidate: str) -> bool:
    """A 9-15 digit number written like a phone; never a date, time, SMTP code or IP."""
    stripped = candidate.strip(" .-()")
    digits = re.sub(r"\D", "", stripped)
    if not 9 <= len(digits) <= 15:
        return False
    if _DATE.search(stripped) or _TIME.search(stripped):
        return False
    if _SMTP_CODE.match(stripped) or _IPV4.match(stripped):
        return False
    return candidate.startswith(("+", "00", "(")) or bool(re.search(r"[\s.()-]", stripped))


def _phone_hits(text: str) -> list[Hit]:
    """Separated or prefixed phone numbers, solid 0-prefixed mobiles and tel: links."""
    hits = [Hit("phone", m.start(), m.end()) for m in _TEL.finditer(text)]
    hits += [
        Hit("phone", m.start(), m.end())
        for m in _PHONE.finditer(text)
        if _looks_like_phone(m.group(0))
    ]
    hits += [Hit("phone", m.start(), m.end()) for m in _PHONE_SOLID.finditer(text)]
    return _dedupe(hits)


def luhn_valid(digits: str) -> bool:
    """Return True when ``digits`` passes the Luhn checksum used by payment cards.

    Args:
        digits: a string of decimal digits only.

    Returns:
        Whether the checksum holds.
    """
    total = 0
    for index, char in enumerate(reversed(digits)):
        value = int(char)
        if index % 2:
            value = value * 2 - 9 if value > 4 else value * 2
        total += value
    return total % 10 == 0


def iban_valid(candidate: str) -> bool:
    """Return True when ``candidate`` passes the ISO 13616 mod-97 check.

    Args:
        candidate: an IBAN-shaped string; spaces are ignored.

    Returns:
        Whether the remainder is 1.
    """
    compact = candidate.replace(" ", "").upper()
    if not 15 <= len(compact) <= 34:
        return False
    rotated = compact[4:] + compact[:4]
    number = "".join(str(int(ch, 36)) for ch in rotated)
    return int(number) % 97 == 1


def _number_hits(text: str) -> list[Hit]:
    """Luhn-valid 13-19 digit runs (card-like) and mod-97-valid IBANs."""
    hits = [
        Hit("card", m.start(), m.end())
        for m in _DIGIT_RUN.finditer(text)
        if luhn_valid(re.sub(r"\D", "", m.group(0)))
    ]
    hits += [
        Hit("iban", m.start(), m.end()) for m in _IBAN.finditer(text) if iban_valid(m.group(0))
    ]
    return hits


def name_parts(names: Iterable[str]) -> set[str]:
    """Return the maskable parts of the given names and sender addresses.

    A full name and each alphabetic part of at least 3 letters is kept. An
    address contributes its local part (``jane.doe@x`` -> ``jane``, ``doe``).
    Names containing a system/team word, and role mailboxes, contribute nothing.

    Args:
        names: display names and/or email addresses.

    Returns:
        Lower-cased parts to search for as whole words.
    """
    parts: set[str] = set()
    for raw in names:
        name = (raw or "").strip().strip('"').strip()
        if "@" in name:
            name = name.split("@", 1)[0]
            if name.lower() in ROLE_MAILBOXES:
                continue
        tokens = [t for t in _NAME_SPLIT.split(name.lower()) if t]
        if not tokens or NON_PERSON_WORDS & set(tokens):
            continue
        if "@" not in (raw or "") and len(tokens) > 1:
            parts.add(" ".join(tokens))
        parts.update(t for t in tokens if len(t) >= 3 and t.isalpha())
    return parts


def _name_hits(text: str, names: Iterable[str]) -> list[Hit]:
    """Whole-word, case-insensitive occurrences of any supplied name part."""
    hits = []
    for part in sorted(name_parts(names), key=len, reverse=True):
        pattern = r"(?<!\w)" + r"\s+".join(map(re.escape, part.split())) + r"(?!\w)"
        hits += [Hit("name", m.start(), m.end()) for m in re.finditer(pattern, text, re.I)]
    return _dedupe(hits)


def _dedupe(hits: list[Hit]) -> list[Hit]:
    """Drop hits of the same kind whose span sits inside an earlier one."""
    kept: list[Hit] = []
    for hit in sorted(hits, key=lambda h: (h.start, -(h.end - h.start))):
        if not any(k.kind == hit.kind and k.start <= hit.start and hit.end <= k.end for k in kept):
            kept.append(hit)
    return kept


def audit(text: str, names: Iterable[str] = ()) -> list[Hit]:
    """Scan one text for personal data that survived redaction.

    Args:
        text: redacted text (subject, body or the whole state text).
        names: names and addresses to look for (sender_name, sender, extra names).

    Returns:
        Every hit, ordered by position. Empty means clean.
    """
    text = text or ""
    hits = _email_hits(text) + _url_hits(text) + _phone_hits(text) + _number_hits(text)
    hits += _name_hits(text, names)
    return sorted(hits, key=lambda h: (h.start, h.kind))


def audit_items(items: Iterable[tuple[str, str, Iterable[str]]]) -> list[Hit]:
    """Audit many texts; each hit carries the id of the item it came from.

    Args:
        items: ``(item_id, text, names)`` triples.

    Returns:
        All hits across the items.
    """
    found: list[Hit] = []
    for item_id, text, names in items:
        found += [Hit(h.kind, h.start, h.end, item_id) for h in audit(text, names)]
    return found


def hit_counts(hits: Iterable[Hit]) -> dict[str, int]:
    """Return ``{kind: count}`` for every kind (zeros included), safe to log."""
    counts = Counter(h.kind for h in hits)
    return {kind: counts.get(kind, 0) for kind in KINDS}


def raise_on_hits(hits: Iterable[Hit]) -> dict[str, int]:
    """Fail closed on any hit.

    Args:
        hits: output of ``audit`` / ``audit_items``.

    Returns:
        The all-zero hit counts when there are no hits.

    Raises:
        RedactionAuditError: any hit; the message lists counts per kind and the
            affected item ids, never the matched text.
    """
    hits = list(hits)
    if hits:
        counts = {k: v for k, v in hit_counts(hits).items() if v}
        ids = sorted({h.item_id for h in hits if h.item_id is not None})
        raise RedactionAuditError(
            f"redaction audit failed: {len(hits)} hits {counts} in items {ids}"
        )
    return hit_counts(hits)


def audit_or_raise(items: Iterable[tuple[str, str, Iterable[str]]]) -> dict[str, int]:
    """Audit many texts and fail closed.

    Args:
        items: ``(item_id, text, names)`` triples.

    Returns:
        The all-zero hit counts, when the texts are clean.

    Raises:
        RedactionAuditError: any hit (see ``raise_on_hits``).
    """
    return raise_on_hits(audit_items(items))
