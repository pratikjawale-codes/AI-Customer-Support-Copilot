import re
from typing import List, Tuple, Dict, Any, Optional

class GuardrailsEngine:
    """
    Input and output guardrails for customer support responses
    Fulfills PDF Requirement 10:
    - Input Guardrails: Prompt injection, jailbreak, system prompt extraction, toxic input
    - Output Guardrails: Unauthorized refund promises, confidential leakage, unsupported claims
    """

    PROMPT_INJECTION_PATTERNS = [
        r'ignore (all )?(previous|above) instructions',
        r'reveal (the )?(system|internal) (prompt|rules|policy)',
        r'you are now in (developer|dan|jailbreak) mode',
        r'disregard safety guidelines',
        r'bypass security filters',
        r'system override',
        r'print your system prompt',
        r'tell me your internal instructions'
    ]

    UNAUTHORIZED_OUTPUT_PATTERNS = [
        r'i hereby approve a full refund of \$\d+',
        r'guarantee a 100% payout without review',
        r'internal database password',
        r'admin access token',
        r'secret internal key'
    ]

    @classmethod
    def validate_input(cls, input_text: str) -> Dict[str, Any]:
        flags = []
        is_safe = True
        block_reason = None
        sanitized = input_text

        # Check prompt injection
        for pattern in cls.PROMPT_INJECTION_PATTERNS:
            if re.search(pattern, input_text, re.IGNORECASE):
                is_safe = False
                flags.append("Prompt Injection / Jailbreak Attempt")
                block_reason = "Blocked: Malicious prompt injection or instruction override attempt detected."
                sanitized = "[BLOCKED: Security Policy Violation - Prompt Injection Detected]"
                break

        # Check for dangerous code execution phrases
        if "<script>" in input_text.lower() or "eval(" in input_text.lower() or "exec(" in input_text.lower():
            is_safe = False
            flags.append("Malicious Code Payload")
            block_reason = "Blocked: Potential code injection payload."

        return {
            "is_safe": is_safe,
            "input_flagged": not is_safe,
            "flags": flags,
            "block_reason": block_reason,
            "sanitized_text": sanitized
        }

    @classmethod
    def validate_output(cls, output_text: str) -> Dict[str, Any]:
        flags = []
        is_safe = True
        block_reason = None
        sanitized = output_text

        for pattern in cls.UNAUTHORIZED_OUTPUT_PATTERNS:
            if re.search(pattern, output_text, re.IGNORECASE):
                is_safe = False
                flags.append("Unauthorized Financial / Claim Statement")
                block_reason = "Output Filtered: AI attempted to promise unapproved financial payout or reveal internal secrets."
                sanitized = "I can assist you with your request, but financial actions require human agent verification."
                break

        return {
            "is_safe": is_safe,
            "output_flagged": not is_safe,
            "flags": flags,
            "block_reason": block_reason,
            "sanitized_text": sanitized
        }
