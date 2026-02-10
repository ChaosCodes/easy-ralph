"""
Shared parsing utilities for Ralph SDK agents.

Provides:
- extract_json: Extract JSON from LLM output (code fence or bare)
"""

import json
import re
from typing import Optional


def extract_json(text: str) -> Optional[dict]:
    """Extract a JSON object from text, handling common LLM output patterns.

    Tries two patterns:
    1. JSON in markdown code fence (```json ... ```)
    2. Bare JSON object (first { to last })

    Returns None if no valid JSON found.
    """
    # Pattern 1: JSON in markdown code fence
    fence_match = re.search(r"```(?:json)?\s*\n(\{.*?\})\s*\n```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # Pattern 2: Bare JSON object
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end > brace_start:
        try:
            return json.loads(text[brace_start:brace_end + 1])
        except json.JSONDecodeError:
            pass

    return None
