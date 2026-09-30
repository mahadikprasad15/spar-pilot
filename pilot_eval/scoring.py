"""Public scoring interface for saved benchmark responses."""

import re
from decimal import Decimal
from fractions import Fraction


_DECIMAL = r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)"
_NUMBER = rf"(?:{_DECIMAL}|[+-]?\d+/\d+)"
_FINAL = re.compile(rf"^####\s+({_NUMBER})$")
_ANSWER_PHRASE = re.compile(rf"(?:final answer:|the answer is)\s*({_NUMBER})[.!]?\s*$", re.I)


def _numeric_value(value: str) -> Fraction:
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        return Fraction(int(numerator), int(denominator))
    return Fraction(Decimal(value.replace(",", "")))


def score_gsm8k(response: str, gold_answer: str, capped: bool = False) -> dict:
    """Score a saved GSM8K response under strict and flexible extraction."""
    gold = _numeric_value(gold_answer.rsplit("####", 1)[-1].strip())
    final_line = response.strip().splitlines()[-1] if response.strip() else ""
    match = _FINAL.fullmatch(final_line)
    extracted = match.group(1) if match else None
    scored = {
        "extracted": extracted,
        "status": "valid" if match and not capped else "invalid",
        "correct": bool(match and not capped and _numeric_value(extracted) == gold),
    }
    flexible_match = match or _ANSWER_PHRASE.search(final_line)
    flexible_answer = flexible_match.group(1) if flexible_match else None
    flexible = {
        "extracted": flexible_answer,
        "status": "valid" if flexible_match and not capped else "invalid",
        "correct": bool(flexible_match and not capped and _numeric_value(flexible_answer) == gold),
    }
    return {"strict": scored, "flexible": flexible}
