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


_V2_CUE = re.compile(r"####|\\boxed\{|final answer\s*:|(?:the\s+)?answer\s+is\s*:?", re.I)
_V2_TOKEN = re.compile(r"(?<![\w.])[+-]?(?:\d[\d, ]*\d|\d+|\.\d+)(?:\.\d+)?(?:/\d+)?(?!\w)")


def extract_gsm8k_flexible_v2(response: str, capped: bool = False) -> dict:
    """Extract without gold access; v2 grammar is documented in ADR 0003."""
    result = {"scorer_version": "gsm8k-flexible-v2", "extracted": None,
              "extraction_rule": None, "status": "invalid", "invalid_reason": None}
    def invalid(reason):
        return {**result, "invalid_reason": reason}
    if capped:
        return invalid("capped_response")
    paragraphs = re.split(r"\n\s*\n", response.strip())
    region = paragraphs[-1]
    # Include contiguous preceding answer paragraphs to expose contradictions.
    for paragraph in reversed(paragraphs[:-1]):
        if not _V2_CUE.search(paragraph):
            break
        region = paragraph + "\n\n" + region
    cue = _V2_CUE.search(region)
    # A phrase delimits the conclusion from calculations in the same paragraph.
    if cue and cue.group(0).lower() not in ("####", "\\boxed{"):
        region = region[cue.start():]
    normalized = re.sub(r"(?<=\d),[ \t]+(?=\d{3}\b)", ",", region)
    for boxed in re.findall(r"\\boxed\{([^{}]*)\}", normalized):
        if not re.fullmatch(_NUMBER, boxed.strip()):
            return invalid("malformed_boxed_answer")
    if "\\boxed" in normalized and not re.search(r"\\boxed\{[^{}]*\}", normalized):
        return invalid("malformed_boxed_answer")
    tokens = list(_V2_TOKEN.finditer(normalized))
    if not tokens:
        return invalid("no_numeric_answer")
    values = []
    for token in tokens:
        raw = token.group(0).strip()
        if not re.fullmatch(_NUMBER, raw):
            return invalid("malformed_number")
        try:
            values.append(_numeric_value(raw))
        except (ValueError, ZeroDivisionError):
            return invalid("malformed_number")
    if len(set(values)) != 1:
        return invalid("conflicting_answers" if "####" in region or "\\boxed" in region else "ambiguous_numbers")
    remainder = _V2_TOKEN.sub("", normalized)
    if re.search(r"[+=*/]", remainder):
        return invalid("unsupported_expression")
    value = values[0]
    canonical = str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"
    rule = ("final_marker" if "####" in region else "boxed_answer" if "\\boxed" in region
            else "final_phrase_single_value" if cue else "final_paragraph_single_value")
    return {**result, "extracted": canonical, "extraction_rule": rule, "status": "valid"}


def score_gsm8k_flexible_v2(response: str, gold_answer: str, capped: bool = False) -> dict:
    extracted = extract_gsm8k_flexible_v2(response, capped)
    gold = _numeric_value(gold_answer.rsplit("####", 1)[-1].strip())
    return {**extracted, "correct": extracted["status"] == "valid" and _numeric_value(extracted["extracted"]) == gold}


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


def extract_gsm8k_flexible_v3(response: str, capped: bool = False) -> dict:
    """Extend v2 with the strict final-line contract; no gold access."""
    final_line = response.strip().splitlines()[-1] if response.strip() else ''
    match = _FINAL.fullmatch(final_line)
    markers = sum(line.lstrip().startswith('####') for line in response.splitlines())
    phrases = re.findall(r'(?:final answer:|the answer is)\s*(' + _NUMBER + r')', response, re.I)
    if not capped and match and markers == 1 and len(phrases) <= 1:
        try:
            value = _numeric_value(match.group(1))
            if all(_numeric_value(phrase) == value for phrase in phrases):
                answer = str(value.numerator) if value.denominator == 1 else f'{value.numerator}/{value.denominator}'
                return dict(scorer_version='gsm8k-flexible-v3', extracted=answer,
                            extraction_rule='strict_final_line', status='valid', invalid_reason=None)
        except (ValueError, ZeroDivisionError):
            pass
    result = extract_gsm8k_flexible_v2(response, capped)
    return {**result, 'scorer_version': 'gsm8k-flexible-v3'}


def score_gsm8k_flexible_v3(response: str, gold_answer: str, capped: bool = False) -> dict:
    extracted = extract_gsm8k_flexible_v3(response, capped)
    gold = _numeric_value(gold_answer.rsplit('####', 1)[-1].strip())
    return {**extracted, 'correct': extracted['status'] == 'valid' and _numeric_value(extracted['extracted']) == gold}
