"""Public scoring interface for saved benchmark responses."""

import re


_FINAL = re.compile(r"^####\s+([+-]?\d+)$")
_ANSWER_PHRASE = re.compile(r"(?:final answer:|the answer is)\s*([+-]?\d+)[.!]?\s*$", re.I)


def score_gsm8k(response: str, gold_answer: str, capped: bool = False) -> dict:
    """Score a saved GSM8K response under strict and flexible extraction."""
    gold = gold_answer.rsplit("####", 1)[-1].strip()
    final_line = response.strip().splitlines()[-1] if response.strip() else ""
    match = _FINAL.fullmatch(final_line)
    extracted = match.group(1) if match else None
    scored = {
        "extracted": extracted,
        "status": "valid" if match and not capped else "invalid",
        "correct": bool(match and not capped and extracted == gold),
    }
    flexible_match = match or _ANSWER_PHRASE.search(final_line)
    flexible_answer = flexible_match.group(1) if flexible_match else None
    flexible = {
        "extracted": flexible_answer,
        "status": "valid" if flexible_match and not capped else "invalid",
        "correct": bool(flexible_match and not capped and flexible_answer == gold),
    }
    return {"strict": scored, "flexible": flexible}
