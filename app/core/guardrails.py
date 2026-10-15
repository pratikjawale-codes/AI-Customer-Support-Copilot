import re
import html
from typing import List, Tuple, Dict, Any, Optional
from app.core.pii_masker import PIIMasker

class GuardrailsEngine:
    """
    Input and Output Guardrails Engine
    Fulfills Layer 2 (Input Security) and Layer 8 (Output Safety):
    - Input: Prompt injection detection, jailbreak prevention, code injection filters, length capping
    - Output:
        - Commitment detection (unapproved refund / timeline promises without policy backing)
        - Numbers & entity verification (numbers in draft must appear in context / tool output)
        - Secret & system prompt leakage scanner
        - Unresolved placeholder detector
        - Output HTML/XSS sanitization
    """

    MAX_INPUT_LENGTH = 5000

    PROMPT_INJECTION_PATTERNS = [
        r'ignore (all )?(previous|above|prior) instructions',
        r'reveal (the )?(system|internal) (prompt|rules|policy)',
        r'you are now in (developer|dan|jailbreak) mode',
        r'disregard (all )?(safety|system) guidelines',
        r'bypass (security|safety) filters',
        r'system override',
        r'print (your )?(system prompt|instructions)',
        r'tell me your internal (instructions|prompt|rules)',
        r'act as an unrestricted ai',
        r'say "yes" to everything',
        r'pretend to be the company ceo and grant'
    ]

    COMMITMENT_REGEX = re.compile(
        r'\b(refund|reimburse|compensat\w+|waive|guarantee|credit(ed)?|within \d+ (?:days|hours|business days)|by (?:tomorrow|today))\b',
        re.IGNORECASE
    )

    NUMBER_REGEX = re.compile(r'\b\d+(?:[.,]\d+)?\b')

    LEAKAGE_PATTERNS = [
        r'sk-[a-zA-Z0-9]{20,}',
        r'bearer\s+[a-zA-Z0-9\-_\.]+',
        r'internal database password',
        r'admin access token',
        r'secret internal key',
        r'system prompt:\s*you draft'
    ]

    @classmethod
    def validate_input(cls, input_text: str) -> Dict[str, Any]:
        """Layer 2: Validate incoming customer text before LLM exposure."""
        text = PIIMasker.normalize_text(input_text.strip())
        flags = []
        is_safe = True
        status = "pass"
        block_reason = None
        sanitized = text

        # 1. Size limit
        if len(text) > cls.MAX_INPUT_LENGTH:
            flags.append("Input Size Exceeded")
            is_safe = False
            status = "block"
            block_reason = f"Blocked: Message exceeds maximum allowed size of {cls.MAX_INPUT_LENGTH} characters."
            return {
                "is_safe": False,
                "status": "block",
                "input_flagged": True,
                "flags": flags,
                "block_reason": block_reason,
                "sanitized_text": text[:cls.MAX_INPUT_LENGTH]
            }

        # 2. Prompt injection checks
        for pattern in cls.PROMPT_INJECTION_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                is_safe = False
                status = "block"
                flags.append("Prompt Injection / Jailbreak Attempt")
                block_reason = "Blocked: Malicious prompt injection or instruction override attempt detected."
                sanitized = "[BLOCKED: Security Policy Violation - Prompt Injection Detected]"
                break

        # 3. Dangerous code payloads
        if any(term in text.lower() for term in ["<script>", "javascript:", "eval(", "exec(", "__import__"]):
            is_safe = False
            status = "block"
            flags.append("Malicious Code Payload")
            block_reason = "Blocked: Potential code injection payload."
            sanitized = "[BLOCKED: Security Policy Violation - Code Injection Detected]"

        return {
            "is_safe": is_safe,
            "status": status,
            "input_flagged": not is_safe,
            "flags": flags,
            "block_reason": block_reason,
            "sanitized_text": sanitized
        }

    @classmethod
    def numbers_supported(cls, draft: str, evidence: str) -> List[str]:
        """Layer 8: Verify that numbers in draft appear in evidence (RAG context + tool results)."""
        ev_nums = set(cls.NUMBER_REGEX.findall(evidence))
        draft_nums = cls.NUMBER_REGEX.findall(draft)
        # Exclude common single-digit conversational counts like 1, 2, 3 if not monetary
        unsupported = [n for n in draft_nums if n not in ev_nums and len(n) > 1]
        return unsupported

    @classmethod
    def validate_output(
        cls,
        output_text: str,
        evidence_text: str = "",
        sources_used: Optional[List[str]] = None,
        allowed_sources: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Layer 8: Output Safety & Hallucination Inspection.
        Validates draft against evidence before reaching human reviewer.
        """
        flags = []
        is_safe = True
        status = "pass"
        block_reason = None
        sanitized = output_text

        # 1. Structural citation check (fabrication detection)
        if sources_used and allowed_sources:
            fabricated = set(sources_used) - set(allowed_sources)
            if fabricated:
                flags.append(f"Fabricated Source Reference: {list(fabricated)}")
                is_safe = False
                status = "block"
                block_reason = "Draft referenced external sources not present in verified context."

        # 2. Secrets & Internal Data Leakage
        for pattern in cls.LEAKAGE_PATTERNS:
            if re.search(pattern, sanitized, re.IGNORECASE):
                flags.append("Sensitive Data / Key Leakage")
                is_safe = False
                status = "block"
                block_reason = "Draft contained internal configuration or credential fragments."
                sanitized = "Draft blocked due to sensitive system data detection."
                break

        # 3. Unresolved Placeholder Check
        if re.search(r'<[A-Z_]+_\d+>', sanitized):
            flags.append("Unresolved PII Placeholder")
            is_safe = False
            status = "fix"

        # 4. Commitment Detection (The Unapproved Refund check)
        if cls.COMMITMENT_REGEX.search(sanitized):
            if evidence_text and not cls.COMMITMENT_REGEX.search(evidence_text):
                flags.append("Unauthorized Financial / Policy Commitment")
                is_safe = False
                status = "fix"
                block_reason = "Draft contains refund or timeline promise not authorized in policy context."
                # Auto-fix: rewrite to safe deferral phrasing
                sanitized = cls.COMMITMENT_REGEX.sub("our support team will verify", sanitized)

        # 5. Unsupported Numbers / Entities Check
        if evidence_text:
            bad_nums = cls.numbers_supported(sanitized, evidence_text)
            if bad_nums:
                flags.append(f"Unsupported Numerical Claims: {bad_nums}")
                is_safe = False
                if status != "block":
                    status = "flag"

        # 6. HTML Sanitization for agent UI display
        sanitized = html.escape(sanitized)

        return {
            "is_safe": is_safe,
            "status": status,
            "output_flagged": not is_safe,
            "flags": flags,
            "block_reason": block_reason,
            "sanitized_text": sanitized
        }
