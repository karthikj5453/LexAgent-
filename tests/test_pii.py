import pytest
from api.pii_redactor import PIIRedactor

def test_pii_redactor_all_types():
    redactor = PIIRedactor()
    original_text = (
        "Hello, my email is karthik@example.com and my phone number is +91 9876543210. "
        "For compliance, my Aadhaar card is 9876 5432 1098 and my PAN card is ABCDE1234F."
    )
    
    redacted = redactor.redact(original_text)
    
    # Assertions on redacted text (no raw sensitive data present)
    assert "karthik@example.com" not in redacted
    assert "+91 9876543210" not in redacted
    assert "9876 5432 1098" not in redacted
    assert "ABCDE1234F" not in redacted
    
    # Assertions on placeholder presence
    assert "[EMAIL_0]" in redacted
    assert "[PHONE_" in redacted
    assert "[AADHAAR_" in redacted
    assert "[PAN_" in redacted
    
    # Unredact back to original
    unredacted = redactor.unredact(redacted)
    assert unredacted == original_text

def test_pii_redactor_empty_text():
    redactor = PIIRedactor()
    assert redactor.redact("") == ""
    assert redactor.unredact("") == ""

def test_pii_redactor_no_pii():
    redactor = PIIRedactor()
    text = "This is a standard contract clause with no PII data."
    assert redactor.redact(text) == text
    assert redactor.unredact(text) == text
