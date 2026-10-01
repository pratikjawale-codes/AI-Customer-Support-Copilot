import re
from typing import List, Tuple

class PIIMasker:
    """
    PII Detection and Masking Engine
    Supports Regex-based rules + Presidio fallback for identifying sensitive personal data:
    - Phone Numbers (e.g., 9876543210 -> ******3210)
    - Credit Card Numbers (e.g., 4532123456789101 -> ************9101)
    - Email Addresses (e.g., john.doe@example.com -> j***e@example.com)
    - Account Numbers / SSN (e.g., SSN 123-45-6789 -> ***-**-6789)
    """

    PHONE_REGEX = re.compile(r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b')
    CREDIT_CARD_REGEX = re.compile(r'\b(?:\d[ -]*?){13,19}\b')
    EMAIL_REGEX = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    SSN_REGEX = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')

    @classmethod
    def mask_phone(cls, match: re.Match) -> str:
        phone = match.group(0)
        digits = re.sub(r'\D', '', phone)
        if len(digits) >= 10:
            masked_digits = '*' * (len(digits) - 4) + digits[-4:]
            return masked_digits
        return '******' + phone[-4:]

    @classmethod
    def mask_credit_card(cls, match: re.Match) -> str:
        card = match.group(0)
        digits = re.sub(r'\D', '', card)
        if len(digits) >= 13:
            return '*' * (len(digits) - 4) + digits[-4:]
        return '************' + card[-4:]

    @classmethod
    def mask_email(cls, match: re.Match) -> str:
        email = match.group(0)
        parts = email.split('@')
        if len(parts) == 2:
            name, domain = parts
            if len(name) <= 2:
                masked_name = name[0] + '*'
            else:
                masked_name = name[0] + '*' * (len(name) - 2) + name[-1]
            return f"{masked_name}@{domain}"
        return "****@****.com"

    @classmethod
    def mask_ssn(cls, match: re.Match) -> str:
        return "***-**-" + match.group(0)[-4:]

    @classmethod
    def mask_text(cls, text: str) -> Tuple[str, List[str]]:
        masked = text
        detected_types = []

        if cls.CREDIT_CARD_REGEX.search(masked):
            detected_types.append("Credit Card Number")
            masked = cls.CREDIT_CARD_REGEX.sub(cls.mask_credit_card, masked)

        if cls.SSN_REGEX.search(masked):
            detected_types.append("Social Security / Account Number")
            masked = cls.SSN_REGEX.sub(cls.mask_ssn, masked)

        if cls.EMAIL_REGEX.search(masked):
            detected_types.append("Email Address")
            masked = cls.EMAIL_REGEX.sub(cls.mask_email, masked)

        if cls.PHONE_REGEX.search(masked):
            detected_types.append("Phone Number")
            masked = cls.PHONE_REGEX.sub(cls.mask_phone, masked)

        return masked, detected_types
