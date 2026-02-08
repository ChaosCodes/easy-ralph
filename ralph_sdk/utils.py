"""
Shared parsing utilities for Ralph SDK agents.

Provides:
- extract_json: Extract JSON from LLM output (code fence or bare)
- parse_agent_output: Unified "JSON first → regex fallback" pattern
"""

import json
import re
from typing import Callable, Optional, TypeVar

T = TypeVar("T")


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


def parse_agent_output(
    text: str,
    json_parser: Callable[[dict], T],
    regex_parser: Callable[[str], T],
    json_key: str = "action",
) -> T:
    """Unified "JSON first → regex fallback" parsing pattern.

    Args:
        text: Raw agent output text
        json_parser: Function to convert a JSON dict into the result type
        regex_parser: Fallback function to parse text with regex
        json_key: Key that must exist in JSON for it to be valid (default: "action")

    Returns:
        Parsed result of type T
    """
    json_obj = extract_json(text)
    if json_obj and json_key in json_obj:
        return json_parser(json_obj)

    return regex_parser(text)
