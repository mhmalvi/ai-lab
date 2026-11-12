"""Tests for the fail-closed redaction audit (AAA: Arrange, Act, Assert).

All personal data below is invented: example.com addresses, the UK drama-reserved
phone range, the standard Visa test card and the textbook example IBAN.
"""
import pytest

from redaction_audit import (
    RedactionAuditError,
    audit,
    audit_or_raise,
    hit_counts,
    iban_valid,
    luhn_valid,
    name_parts,
    url_is_tokenised,
)


def kinds(text, names=()):
    return [hit.kind for hit in audit(text, names)]


def test_clean_placeholder_text_has_no_hits():
    # Arrange
    text = "Hi [NAME], reply to [EMAIL] or call [PHONE]. See https://example.com/pricing"
    # Act
    hits = audit(text)
    # Assert
    assert hits == []


def test_real_email_address_is_found():
    # Arrange
    text = "write to jane.doe@example.com today"
    # Act / Assert
    assert kinds(text) == ["email"]


def test_masked_domain_with_role_mailbox_is_allowed_but_personal_is_not():
    # Arrange
    role, personal = "from mailer-daemon@[DOMAIN]", "from jane@[DOMAIN]"
    # Act / Assert
    assert kinds(role) == []
    assert kinds(personal) == ["email"]


@pytest.mark.parametrize(
    "text", ["call +44 20 7946 0958", "mobile 07700 900123", "tel: 020 7946 0958"]
)
def test_phone_numbers_are_found(text):
    # Arrange: text from the parameter list
    # Act / Assert
    assert kinds(text) == ["phone"]


@pytest.mark.parametrize(
    "text", ["on 2026-10-14 at 12:30", "smtp said 550 5.1.1", "host 192.168.10.20"]
)
def test_dates_times_smtp_codes_and_ips_are_not_phones(text):
    # Arrange: text from the parameter list
    # Act / Assert
    assert "phone" not in kinds(text)


def test_tokenised_url_and_orphan_query_tail_are_found():
    # Arrange
    text = "click https://example.com/unsub?u=abc123 or append &token=xyz"
    # Act
    found = kinds(text)
    # Assert
    assert found == ["url_token", "url_token"]


def test_url_with_long_opaque_segment_counts_as_tokenised():
    # Arrange
    url = "https://example.com/t/a8f3k29dk3l2m4n5p6"
    # Act / Assert
    assert url_is_tokenised(url) is True
    assert url_is_tokenised("https://example.com/about-us") is False


def test_luhn_valid_card_number_is_found_and_invalid_is_not():
    # Arrange
    good, bad = "card 4111 1111 1111 1111", "ref 4111 1111 1111 1112"
    # Act / Assert
    assert luhn_valid("4111111111111111")
    assert kinds(good) == ["card"]
    assert "card" not in kinds(bad)


def test_iban_passing_mod_97_is_found():
    # Arrange
    text = "pay GB82 WEST 1234 5698 7654 32 please"
    # Act / Assert
    assert iban_valid("GB82WEST12345698765432")
    assert not iban_valid("GB00WEST12345698765432")
    assert "iban" in kinds(text)


def test_name_parts_come_from_display_names_and_address_local_parts():
    # Arrange
    names = ["Jane Doe", "jdoe.smith@example.com"]
    # Act
    parts = name_parts(names)
    # Assert
    assert {"jane doe", "jane", "doe", "jdoe", "smith"} <= parts


def test_system_and_team_names_contribute_no_parts():
    # Arrange
    names = ["Mail Delivery Subsystem", "Support Team", "no-reply@example.com"]
    # Act
    parts = name_parts(names)
    # Assert
    assert parts == set()


def test_surviving_sender_name_is_found_case_insensitively():
    # Arrange
    text = "Thanks, JANE"
    # Act / Assert
    assert kinds(text, names=["Jane Doe"]) == ["name"]


def test_audit_or_raise_reports_counts_and_ids_never_text():
    # Arrange
    items = [
        ("msg-1", "fine text", []),
        ("msg-2", "reach me at jane.doe@example.com", []),
    ]
    # Act
    with pytest.raises(RedactionAuditError) as caught:
        audit_or_raise(items)
    # Assert
    message = str(caught.value)
    assert "msg-2" in message
    assert "'email': 1" in message
    assert "jane.doe" not in message


def test_audit_or_raise_returns_zero_counts_when_clean():
    # Arrange
    items = [("msg-1", "Hello [NAME]", ["Jane Doe"])]
    # Act
    counts = audit_or_raise(items)
    # Assert
    assert counts == {kind: 0 for kind in hit_counts([])}
    assert set(counts) == {"email", "phone", "url_token", "card", "iban", "name"}
