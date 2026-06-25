import re

class PIIRedactor:
    """
    Enterprise-ready PII Redactor for compliance with data privacy regulations.
    Masks and restores sensitive information like Emails, Phone Numbers, Aadhaar, and PAN Cards.
    """
    def __init__(self):
        self.mappings = {}

    def redact(self, text: str) -> str:
        if not text:
            return text

        # 1. Email Redaction
        email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
        for match in re.findall(email_pattern, text):
            if match not in self.mappings.values():
                placeholder = f"[EMAIL_{len(self.mappings)}]"
                self.mappings[placeholder] = match
                text = text.replace(match, placeholder)
            else:
                # Reuse existing mapping
                placeholder = [k for k, v in self.mappings.items() if v == match][0]
                text = text.replace(match, placeholder)

        # 2. Aadhaar Numbers (e.g., 1234 5678 9012, 1234-5678-9012, 123456789012)
        aadhaar_pattern = r'\b[2-9]{1}[0-9]{3}[-\s]?[0-9]{4}[-\s]?[0-9]{4}\b'
        for match in re.findall(aadhaar_pattern, text):
            if match not in self.mappings.values():
                placeholder = f"[AADHAAR_{len(self.mappings)}]"
                self.mappings[placeholder] = match
                text = text.replace(match, placeholder)
            else:
                placeholder = [k for k, v in self.mappings.items() if v == match][0]
                text = text.replace(match, placeholder)

        # 3. PAN Card Numbers (Indian Tax ID: 5 letters, 4 digits, 1 letter)
        pan_pattern = r'\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b'
        for match in re.findall(pan_pattern, text):
            if match not in self.mappings.values():
                placeholder = f"[PAN_{len(self.mappings)}]"
                self.mappings[placeholder] = match
                text = text.replace(match, placeholder)
            else:
                placeholder = [k for k, v in self.mappings.items() if v == match][0]
                text = text.replace(match, placeholder)

        # 4. Indian Phone Numbers (e.g., +91 9876543210, 9876543210)
        phone_pattern = r'\b(?:\+91|91)?[-\s]?[6-9]\d{9}\b'
        for match in re.findall(phone_pattern, text):
            # Check if this isn't already inside another redacted match (e.g., Aadhaar)
            already_redacted = False
            for val in self.mappings.values():
                if match in val and len(match) < len(val):
                    already_redacted = True
                    break
            if already_redacted:
                continue

            if match not in self.mappings.values():
                placeholder = f"[PHONE_{len(self.mappings)}]"
                self.mappings[placeholder] = match
                text = text.replace(match, placeholder)
            else:
                placeholder = [k for k, v in self.mappings.items() if v == match][0]
                text = text.replace(match, placeholder)

        return text

    def unredact(self, text: str) -> str:
        if not text:
            return text
        # Restore mappings in reverse order of placeholders to handle nested/partial matches safely
        for placeholder, original in sorted(self.mappings.items(), key=lambda x: len(x[0]), reverse=True):
            text = text.replace(placeholder, original)
        return text
