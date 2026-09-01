from __future__ import annotations

import re


def normalize_answer(task_name: str, answer: str) -> str:
    text = str(answer).strip()
    if task_name == "mmlu_pro":
        match = re.search(r"\b([A-J])\b", text.upper())
        return match.group(1) if match else text.upper()
    if task_name == "chaosnli":
        low = text.lower()
        if low.startswith("entail") or low == "e":
            return "entailment"
        if low.startswith("neutral") or low == "n":
            return "neutral"
        if low.startswith("contrad") or low == "c":
            return "contradiction"
        return low
    return normalize_math_text(text)


def is_correct(task_name: str, prediction: str, gold: str) -> bool:
    return normalize_answer(task_name, prediction) == normalize_answer(task_name, gold)


def normalize_math_text(text: str) -> str:
    replacements = {
        "\\left": "",
        "\\right": "",
        " ": "",
        "$": "",
    }
    out = text
    boxed = re.search(r"\\boxed\{(.+)\}", out)
    if boxed:
        out = boxed.group(1)
    for src, dst in replacements.items():
        out = out.replace(src, dst)
    return out.strip().rstrip(".")
