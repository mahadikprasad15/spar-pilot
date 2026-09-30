"""Public scoring interface for saved benchmark responses."""

import re
from decimal import Decimal
from fractions import Fraction


_DECIMAL = r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)"
_NUMBER = rf"(?:{_DECIMAL}|[+-]?\d+/\d+)"
_FINAL = re.compile(rf"^####\s+({_NUMBER})$")
_ANSWER_PHRASE = re.compile(rf"(?:final answer:|the answer is)\s*({_NUMBER})[.!]?\s*$", re.I)
_LAST_NUMBER_LINE = re.compile(rf"^({_NUMBER})[.!]?$")
_MMLU_ANSWER = re.compile(r"^(?:(?i:Answer:)\s*|(?i:The answer is)\s*)?([ABCD])(?=\b|[.)])")


def _numeric_value(value: str) -> Fraction:
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        return Fraction(int(numerator), int(denominator))
    return Fraction(Decimal(value.replace(",", "")))


def score_gsm8k(response: str, gold_answer: str, capped: bool = False) -> dict:
    """Score a saved GSM8K response under strict and flexible extraction."""
    if sum(line.lstrip().startswith("####") for line in response.splitlines()) > 1:
        invalid = {"extracted": None, "status": "invalid", "correct": False}
        return {"strict": invalid.copy(), "flexible": invalid.copy()}
    gold = _numeric_value(gold_answer.rsplit("####", 1)[-1].strip())
    final_line = response.strip().splitlines()[-1] if response.strip() else ""
    match = _FINAL.fullmatch(final_line)
    extracted = match.group(1) if match else None
    scored = {
        "extracted": extracted,
        "status": "valid" if match and not capped else "invalid",
        "correct": bool(match and not capped and _numeric_value(extracted) == gold),
    }
    flexible_match = match or _ANSWER_PHRASE.search(final_line) or _LAST_NUMBER_LINE.fullmatch(final_line)
    flexible_answer = flexible_match.group(1) if flexible_match else None
    flexible = {
        "extracted": flexible_answer,
        "status": "valid" if flexible_match and not capped else "invalid",
        "correct": bool(flexible_match and not capped and _numeric_value(flexible_answer) == gold),
    }
    return {"strict": scored, "flexible": flexible}


def score_mmlu_text(response: str, gold_choice: str, capped: bool = False) -> dict:
    """Score a generated MMLU letter response."""
    response = response.strip()
    match = _MMLU_ANSWER.match(response)
    choice = match.group(1) if match else None
    other_choices = re.findall(r"\b[ABCD]\b", response[match.end():]) if match else []
    valid = choice is not None and not other_choices and not capped
    return {
        "choice": choice if valid else None,
        "status": "valid" if valid else "invalid",
        "correct": valid and choice == gold_choice,
    }


def score_mmlu_logits(candidate_scores: dict[str, float], gold_choice: str) -> dict:
    """Score next-token logits over four MMLU choices."""
    if set(candidate_scores) != set("ABCD"):
        raise ValueError("candidate scores must contain A, B, C, and D")
    choice = max(candidate_scores, key=candidate_scores.get)
    return {"choice": choice, "status": "valid", "correct": choice == gold_choice}
