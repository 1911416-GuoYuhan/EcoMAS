from __future__ import annotations

import re

FLOAT_TOLERANCE = 1e-6
INVALID_ANSWER_PREFIX = "invalid::"


def invalid_answer(task_name: str) -> str:
    return f"{INVALID_ANSWER_PREFIX}{task_name}"


def is_valid_answer(task_name: str, answer: str) -> bool:
    text = str(answer or "").strip()
    if task_name == "mmlu_pro":
        return re.fullmatch(r"[A-J]", text.upper()) is not None
    if task_name == "chaosnli":
        return text.lower() in {"entailment", "neutral", "contradiction"}
    if task_name == "math500":
        if not text or len(text) > 160:
            return False
        forbidden = (
            r"\[\s*YOUR\s+(?:FINAL\s+)?ANSWER\s*\]",
            r"\bgo\s+forward\b",
            r"\bfollow\s+the\s+direction\b",
            r"\bcontinue\s+the\s+reasoning\b",
            r"\bREASONING\s+RESULT\b",
            r"\bFINAL\s+ANSWER\s*:",
        )
        return not any(re.search(pattern, text, re.IGNORECASE) for pattern in forbidden)
    return bool(text)


def normalize_answer(task_name: str, answer: str) -> str:
    text = str(answer).strip()
    if task_name == "mmlu_pro":
        return text.upper() if is_valid_answer(task_name, text) else invalid_answer(task_name)
    if task_name == "chaosnli":
        low = text.lower()
        return low if is_valid_answer(task_name, low) else invalid_answer(task_name)
    if task_name == "math500":
        return normalize_math_text(text) if is_valid_answer(task_name, text) else invalid_answer(task_name)
    return text


def is_correct(task_name: str, prediction: str, gold: str) -> bool:
    return answer_match(task_name, prediction, gold)[0]


def answer_match(task_name: str, prediction: str, gold: str) -> tuple[bool, str]:
    if not is_valid_answer(task_name, prediction):
        return False, "invalid_prediction"
    if task_name == "math500":
        return check_math500_with_method(prediction, gold)
    matched = normalize_answer(task_name, prediction) == normalize_answer(task_name, gold)
    return matched, "exact_normalized" if matched else "not_equal"


def normalize_math_text(text: str) -> str:
    replacements = {
        "\\left": "",
        "\\right": "",
        " ": "",
        "$": "",
        "°": "",
    }
    out = text
    boxed = re.search(r"\\boxed\{(.+)\}", out)
    if boxed:
        out = boxed.group(1)
    out = re.sub(r"\\text\{([^{}]*)\}", r"\1", out)
    out = re.sub(r"\^?\\circ\b", "", out)
    out = re.sub(r"\bdegrees?\b", "", out, flags=re.IGNORECASE)
    for src, dst in replacements.items():
        out = out.replace(src, dst)
    return out.strip().rstrip(".")


def _math_scalar(text: str) -> str | None:
    normalized = normalize_math_text(text)
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", normalized):
        return normalized
    match = re.search(r"(?:=|answer\s*(?:is|:))\s*[-+]?\d+(?:\.\d+)?\s*\}?(?:\.|$)", normalized, re.IGNORECASE)
    if match:
        number = re.search(r"[-+]?\d+(?:\.\d+)?", match.group(0))
        return number.group(0) if number else None
    return None


def check_math500(prediction: str, gold: str) -> bool:
    return check_math500_with_method(prediction, gold)[0]


def check_math500_with_method(prediction: str, gold: str) -> tuple[bool, str]:
    pred_readable = _readable_math_text(prediction)
    expected_readable = _readable_math_text(gold)
    if re.fullmatch(r"[A-Za-z][A-Za-z .'-]*", expected_readable):
        textual = re.fullmatch(
            rf"{re.escape(expected_readable)}(?:\s+(?:has|is|was|were)\b.*)?[.!]?",
            pred_readable,
            re.IGNORECASE,
        )
        if textual:
            method = "exact_normalized" if pred_readable.lower() == expected_readable.lower() else "text_answer_in_sentence"
            return True, method
    pred = normalize_math_text(prediction)
    expected = normalize_math_text(gold)
    if pred == expected:
        return True, "exact_normalized"
    pred_scalar = _math_scalar(pred)
    gold_scalar = _math_scalar(expected)
    if pred_scalar is not None and gold_scalar is not None:
        try:
            matched = abs(float(pred_scalar) - float(gold_scalar)) < FLOAT_TOLERANCE
            return matched, "numeric_tolerance" if matched else "not_equal"
        except ValueError:
            pass
    try:
        from sympy import simplify, sympify

        matched = simplify(sympify(pred.replace("^", "**")) - sympify(expected.replace("^", "**"))) == 0
        return matched, "symbolic_equivalence" if matched else "not_equal"
    except Exception:
        return False, "not_equal"


def _readable_math_text(text: str) -> str:
    out = str(text or "").strip()
    out = re.sub(r"\\boxed\{(.+)\}", r"\1", out)
    out = re.sub(r"\\text\{([^{}]*)\}", r"\1", out)
    out = out.replace("\\left", "").replace("\\right", "")
    out = re.sub(r"\\[()\[\]]", "", out).replace("$", "")
    return re.sub(r"\s+", " ", out).strip().rstrip(".")
