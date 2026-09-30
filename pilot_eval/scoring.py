"""Public scoring interface for saved benchmark responses."""

import re
import math
from decimal import Decimal
from fractions import Fraction


_DECIMAL = r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)"
_NUMBER = rf"(?:[+-]?\d+/\d+|{_DECIMAL})"
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
    markers = sum(line.lstrip().startswith("####") for line in response.splitlines())
    phrases = re.findall(r"(?:final answer:|the answer is)\s*(" + _NUMBER + r")", response, re.I)
    final_line = response.strip().splitlines()[-1] if response.strip() else ""
    if markers > 1 or len(phrases) > 1 or (markers and not _FINAL.fullmatch(final_line)):
        invalid = {"extracted": None, "status": "invalid", "correct": False}
        return {"strict": invalid.copy(), "flexible": invalid.copy()}
    gold = _numeric_value(gold_answer.rsplit("####", 1)[-1].strip())
    match = _FINAL.fullmatch(final_line)
    extracted = match.group(1) if match else None
    if match:
        try:
            value = _numeric_value(extracted)
            if phrases and any(_numeric_value(phrase) != value for phrase in phrases):
                raise ValueError("contradictory answer")
        except (ValueError, ZeroDivisionError):
            invalid = {"extracted": extracted, "status": "invalid", "correct": False}
            return {"strict": invalid.copy(), "flexible": invalid.copy()}
    scored = {
        "extracted": extracted,
        "status": "valid" if match and not capped else "invalid",
        "correct": bool(match and not capped and _numeric_value(extracted) == gold),
    }
    flexible_match = match or _ANSWER_PHRASE.search(final_line) or _LAST_NUMBER_LINE.fullmatch(final_line)
    flexible_answer = flexible_match.group(1) if flexible_match else None
    if flexible_match:
        try:
            _numeric_value(flexible_answer)
        except (ValueError, ZeroDivisionError):
            flexible_match = None
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


def score_mmlu_logits(candidate_scores: dict[str, float], gold_choice: str, tie_policy="stop") -> dict:
    """Score next-token logits over four MMLU choices."""
    if tie_policy not in ("stop", "invalid"):
        raise ValueError("unsupported logit tie policy")
    if set(candidate_scores) != set("ABCD"):
        raise ValueError("candidate scores must contain A, B, C, and D")
    if not all(math.isfinite(score) for score in candidate_scores.values()):
        raise ValueError("candidate logits must be finite")
    highest = max(candidate_scores.values())
    tied = sorted(choice for choice, score in candidate_scores.items() if score == highest)
    if len(tied) != 1:
        if tie_policy == "invalid":
            return {"choice": None, "status": "invalid", "correct": False, "tied_choices": tied}
        raise ValueError("top-logit tie makes the item invalid")
    choice = max(candidate_scores, key=candidate_scores.get)
    return {"choice": choice, "status": "valid", "correct": choice == gold_choice}
