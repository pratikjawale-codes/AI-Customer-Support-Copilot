import re
import unicodedata
from typing import List, Tuple, Dict, Any

class PIIMasker:
    """
    PII Detection and Masking Engine
    Fulfills Layer 2 (Input & Security):
    - Phone Numbers (Indian mobile + international)
    - Credit Cards with Luhn checksum validation (cuts false positives like 16-digit order numbers)
    - Indian Aadhaar with Verhoeff checksum validation
    - Indian PAN (Permanent Account Number)
    - Email Addresses
    - SSN / Account numbers
    - Two masking styles:
        1. Human-facing partial mask (******3210, j***e@domain.com)
        2. LLM placeholder tokenization (<PHONE_1>, <CARD_1>) with server-side reversible mapping
    - Unicode NFKC normalization
    """

    # Verhoeff tables for Aadhaar checksum validation
    _VERHOEFF_D = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
        [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
        [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
        [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
        [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
        [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
        [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
        [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
        [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
    ]
    _VERHOEFF_P = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
        [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
        [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
        [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
        [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
        [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
        [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
    ]

    PHONE_REGEX = re.compile(r'(?<!\d)(?:\+91[- ]?)?[6-9]\d{9}(?!\d)|\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b')
    CREDIT_CARD_REGEX = re.compile(r'\b(?:\d[ -]*?){13,19}\b')
    AADHAAR_REGEX = re.compile(r'\b[2-9]\d{3}[ -]?\d{4}[ -]?\d{4}\b')
    PAN_REGEX = re.compile(r'\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b')
    EMAIL_REGEX = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    SSN_REGEX = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')

    @classmethod
    def validate_luhn(cls, card_number: str) -> bool:
        """Validate credit card using Luhn algorithm to prevent false positives on order IDs."""
        digits = [int(c) for c in card_number if c.isdigit()]
        if len(digits) < 13 or len(digits) > 19:
            return False
        total = 0
        reverse_digits = digits[::-1]
        for i, digit in enumerate(reverse_digits):
            if i % 2 == 1:
                doubled = digit * 2
                total += doubled - 9 if doubled > 9 else doubled
            else:
                total += digit
        return total % 10 == 0

    @classmethod
    def validate_verhoeff(cls, number_str: str) -> bool:
        """Validate 12-digit Indian Aadhaar using Verhoeff algorithm."""
        digits = [int(c) for c in number_str if c.isdigit()]
        if len(digits) != 12:
            return False
        c = 0
        for i, item in enumerate(reversed(digits)):
            c = cls._VERHOEFF_D[c][cls._VERHOEFF_P[i % 8][item]]
        return c == 0

    @classmethod
    def normalize_text(cls, text: str) -> str:
        """Applies Unicode NFKC normalization to resolve homoglyphs and styled characters."""
        return unicodedata.normalize('NFKC', text)

    @classmethod
    def mask_phone(cls, match: re.Match) -> str:
        phone = match.group(0)
        digits = re.sub(r'\D', '', phone)
        if len(digits) >= 10:
            return '*' * (len(digits) - 4) + digits[-4:]
        return '******' + phone[-4:]

    @classmethod
    def mask_credit_card(cls, match: re.Match) -> str:
        card = match.group(0)
        digits = re.sub(r'\D', '', card)
        if cls.validate_luhn(digits):
            return '*' * (len(digits) - 4) + digits[-4:]
        return card  # Not a valid credit card, don't mask (e.g., standard order ID)

    @classmethod
    def mask_aadhaar(cls, match: re.Match) -> str:
        raw = match.group(0)
        digits = re.sub(r'\D', '', raw)
        if cls.validate_verhoeff(digits):
            return "XXXXXXXX" + digits[-4:]
        return raw

    @classmethod
    def mask_pan(cls, match: re.Match) -> str:
        pan = match.group(0)
        return pan[:2] + "XXXXX" + pan[-3:]

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
        """Human-facing partial masking (asterisks) for UI display and audit logging."""
        norm = cls.normalize_text(text)
        masked = norm
        detected_types = []

        # Check Aadhaar
        aadhaar_matches = [m.group(0) for m in cls.AADHAAR_REGEX.finditer(masked) if cls.validate_verhoeff(re.sub(r'\D', '', m.group(0)))]
        if aadhaar_matches:
            detected_types.append("Aadhaar Number")
            masked = cls.AADHAAR_REGEX.sub(cls.mask_aadhaar, masked)

        # Check Credit Card (with Luhn)
        card_matches = [m.group(0) for m in cls.CREDIT_CARD_REGEX.finditer(masked) if cls.validate_luhn(re.sub(r'\D', '', m.group(0)))]
        if card_matches:
            detected_types.append("Credit Card Number")
            masked = cls.CREDIT_CARD_REGEX.sub(cls.mask_credit_card, masked)

        # Check PAN
        if cls.PAN_REGEX.search(masked):
            detected_types.append("PAN Number")
            masked = cls.PAN_REGEX.sub(cls.mask_pan, masked)

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

    @classmethod
    def mask_with_placeholders(cls, text: str) -> Tuple[str, Dict[str, str], List[str]]:
        """
        Reversible tokenization masking for LLM prompts.
        Replaces sensitive values with tokens (<PHONE_1>, <CARD_1>) and maintains
        a server-side mapping dictionary that never leaves the system.
        """
        norm = cls.normalize_text(text)
        placeholder_map: Dict[str, str] = {}
        detected_types: List[str] = []
        counters: Dict[str, int] = {
            "CARD": 0, "AADHAAR": 0, "PAN": 0, "SSN": 0, "EMAIL": 0, "PHONE": 0
        }

        result = norm

        # 1. Aadhaar
        def repl_aadhaar(m):
            raw = m.group(0)
            if cls.validate_verhoeff(re.sub(r'\D', '', raw)):
                counters["AADHAAR"] += 1
                token = f"<AADHAAR_{counters['AADHAAR']}>"
                placeholder_map[token] = raw
                if "Aadhaar Number" not in detected_types:
                    detected_types.append("Aadhaar Number")
                return token
            return raw
        result = cls.AADHAAR_REGEX.sub(repl_aadhaar, result)

        # 2. Credit Card
        def repl_card(m):
            raw = m.group(0)
            if cls.validate_luhn(re.sub(r'\D', '', raw)):
                counters["CARD"] += 1
                token = f"<CARD_{counters['CARD']}>"
                placeholder_map[token] = raw
                if "Credit Card Number" not in detected_types:
                    detected_types.append("Credit Card Number")
                return token
            return raw
        result = cls.CREDIT_CARD_REGEX.sub(repl_card, result)

        # 3. PAN
        def repl_pan(m):
            counters["PAN"] += 1
            token = f"<PAN_{counters['PAN']}>"
            placeholder_map[token] = m.group(0)
            if "PAN Number" not in detected_types:
                detected_types.append("PAN Number")
            return token
        result = cls.PAN_REGEX.sub(repl_pan, result)

        # 4. SSN
        def repl_ssn(m):
            counters["SSN"] += 1
            token = f"<SSN_{counters['SSN']}>"
            placeholder_map[token] = m.group(0)
            if "Social Security / Account Number" not in detected_types:
                detected_types.append("Social Security / Account Number")
            return token
        result = cls.SSN_REGEX.sub(repl_ssn, result)

        # 5. Email
        def repl_email(m):
            counters["EMAIL"] += 1
            token = f"<EMAIL_{counters['EMAIL']}>"
            placeholder_map[token] = m.group(0)
            if "Email Address" not in detected_types:
                detected_types.append("Email Address")
            return token
        result = cls.EMAIL_REGEX.sub(repl_email, result)

        # 6. Phone
        def repl_phone(m):
            counters["PHONE"] += 1
            token = f"<PHONE_{counters['PHONE']}>"
            placeholder_map[token] = m.group(0)
            if "Phone Number" not in detected_types:
                detected_types.append("Phone Number")
            return token
        result = cls.PHONE_REGEX.sub(repl_phone, result)

        return result, placeholder_map, detected_types

    @classmethod
    def restore_placeholders(cls, text: str, placeholder_map: Dict[str, str]) -> str:
        """Restores placeholders in draft responses back to real values for human reviewer screen."""
        if not placeholder_map:
            return text
        restored = text
        for token, original in placeholder_map.items():
            restored = restored.replace(token, original)
        return restored
