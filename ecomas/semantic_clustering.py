from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

from ecomas.evaluation import invalid_answer, normalize_answer, normalize_math_text


@dataclass(frozen=True)
class SemanticAnswer:
    output_id: str
    question_uid: str
    task_name: str
    raw_answer: str
    parsed_answer: str
    parse_valid: bool
    answer_type: str
    canonical_key: str
    resolution_method: str


@dataclass(frozen=True)
class PairwiseSemanticDecision:
    left_id: str
    right_id: str
    entailment_lr: bool
    entailment_rl: bool
    contradiction_lr: bool
    contradiction_rl: bool
    equivalent: bool
    method: str


@dataclass
class SemanticClusteringResult:
    question_uid: str
    task_name: str
    answers: list[SemanticAnswer]
    cluster_by_output_id: dict[str, int]
    members_by_cluster: dict[int, list[str]]
    representative_by_cluster: dict[int, str]
    entailment_matrix: list[list[bool]]
    equivalence_matrix: list[list[bool]]
    pairwise_decisions: list[PairwiseSemanticDecision]
    non_transitive_relation: bool
    conflicting_triples: list[tuple[str, str, str]]
    invalid_rate: float
    unresolved_rate: float
    configuration: dict[str, Any]

    def same_cluster(self, left_id: str, right_id: str) -> int:
        return int(self.cluster_by_output_id[left_id] == self.cluster_by_output_id[right_id])

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["answers"] = [asdict(answer) for answer in self.answers]
        payload["pairwise_decisions"] = [asdict(item) for item in self.pairwise_decisions]
        return payload


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _replace_latex_command(text: str, command: str, arity: int, replacement) -> str:
    marker = f"\\{command}"
    while marker in text:
        start = text.find(marker)
        cursor = start + len(marker)
        arguments: list[str] = []
        valid = True
        for _ in range(arity):
            while cursor < len(text) and text[cursor].isspace():
                cursor += 1
            if cursor >= len(text) or text[cursor] != "{":
                valid = False
                break
            depth = 1
            end = cursor + 1
            while end < len(text) and depth:
                depth += (text[end] == "{") - (text[end] == "}")
                end += 1
            if depth:
                valid = False
                break
            arguments.append(text[cursor + 1:end - 1])
            cursor = end
        if not valid:
            break
        text = text[:start] + replacement(*arguments) + text[cursor:]
    return text


def _latex_to_sympy(text: str) -> str:
    out = normalize_math_text(text)
    out = _replace_latex_command(out, "frac", 2, lambda left, right: f"(({left})/({right}))")
    out = _replace_latex_command(out, "dfrac", 2, lambda left, right: f"(({left})/({right}))")
    out = _replace_latex_command(out, "tfrac", 2, lambda left, right: f"(({left})/({right}))")
    out = _replace_latex_command(out, "sqrt", 1, lambda value: f"sqrt({value})")
    out = out.replace("\\cdot", "*").replace("\\times", "*")
    out = out.replace("\\pi", "pi").replace("−", "-")
    out = re.sub(r"(?<=\d)(?=[A-Za-z(])", "*", out)
    out = re.sub(r"(?<=\))(?=[A-Za-z0-9(])", "*", out)
    out = re.sub(r"(?<=\d)i\b", "*I", out)
    out = re.sub(r"\bi\b", "I", out)
    out = re.sub(r"(?<=I)(?=\()", "*", out)
    return out.replace("^", "**")


def _split_top_level(text: str, separator: str = ",") -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    for index, char in enumerate(text):
        depth += char in "([{"
        depth -= char in ")]}"
        if char == separator and depth == 0:
            parts.append(text[start:index])
            start = index + 1
    parts.append(text[start:])
    return [part.strip() for part in parts]


def _sympy_expression(text: str):
    from sympy import sympify

    return sympify(_latex_to_sympy(text), evaluate=True)


def _canonical_srepr(text: str) -> str:
    from sympy import nsimplify, simplify, srepr

    return srepr(simplify(nsimplify(_sympy_expression(text))))


def _math_symbol(answer: str) -> tuple[str, str, str]:
    text = normalize_math_text(answer)
    if re.fullmatch(r"[A-Za-z]+(?:[ .'][A-Za-z]+)*", text):
        compact = re.sub(r"\s+", " ", text).strip().casefold()
        return "text", f"math_text::{compact}", "normalized_text"
    if re.fullmatch(r"\([^()]+(?:,[^()]+)+\)", text):
        answer_type = "tuple"
    elif re.fullmatch(r"\{.*\}", text):
        answer_type = "set"
    elif re.fullmatch(r"\[[^]]*\]", text):
        answer_type = "interval_or_list"
    else:
        answer_type = "expression"
    try:
        if answer_type in {"tuple", "set"}:
            body = text[1:-1]
            values = [_canonical_srepr(part) for part in _split_top_level(body)]
            if answer_type == "set":
                values.sort()
            return answer_type, f"math_{answer_type}::{values!r}", "sympy_structured"
        return answer_type, f"math::{_canonical_srepr(text)}", "sympy"
    except Exception:
        compact = re.sub(r"\s+", " ", text).strip().casefold()
        return answer_type, f"math_text::{_digest(compact)}", "normalized_text"


def _math_entails(left: SemanticAnswer, right: SemanticAnswer) -> tuple[bool, str]:
    if left.canonical_key == right.canonical_key:
        return True, left.resolution_method
    if left.answer_type != right.answer_type:
        return False, "type_mismatch"
    if left.answer_type == "text":
        left_text = re.sub(r"\s+", " ", left.parsed_answer).strip().casefold().rstrip(".")
        right_text = re.sub(r"\s+", " ", right.parsed_answer).strip().casefold().rstrip(".")
        if right_text.startswith(left_text + " ") and re.match(r"(?:has|is|was|were)\b", right_text[len(left_text) + 1:]):
            return True, "text_answer_in_sentence"
        if left_text.startswith(right_text + " ") and re.match(r"(?:has|is|was|were)\b", left_text[len(right_text) + 1:]):
            return True, "text_answer_in_sentence"
        return False, "normalized_text"
    try:
        from sympy import simplify

        left_text = normalize_math_text(left.parsed_answer)
        right_text = normalize_math_text(right.parsed_answer)
        if left.answer_type in {"tuple", "set"}:
            left_values = [_sympy_expression(part) for part in _split_top_level(left_text[1:-1])]
            right_values = [_sympy_expression(part) for part in _split_top_level(right_text[1:-1])]
            if len(left_values) != len(right_values):
                return False, "symbolic_structured"
            if left.answer_type == "tuple":
                return all(simplify(a - b) == 0 for a, b in zip(left_values, right_values)), "symbolic_tuple"
            unmatched = list(right_values)
            for value in left_values:
                match = next((index for index, candidate in enumerate(unmatched) if simplify(value - candidate) == 0), None)
                if match is None:
                    return False, "symbolic_set"
                unmatched.pop(match)
            return not unmatched, "symbolic_set"
        return simplify(_sympy_expression(left_text) - _sympy_expression(right_text)) == 0, "symbolic_equivalence"
    except Exception:
        return False, "normalized_text"


def _symbolize(task_name: str, parsed: str, parse_valid: bool) -> tuple[str, str, str]:
    task = task_name.lower()
    if not parse_valid:
        return "invalid", invalid_answer(task), "invalid_protocol"
    if "mmlu" in task:
        return "mmlu_option", f"mmlu_pro::{normalize_answer('mmlu_pro', parsed)}", "task_label"
    if "chaos" in task or "nli" in task:
        return "nli_label", f"chaosnli::{normalize_answer('chaosnli', parsed)}", "task_label"
    return _math_symbol(parsed)


def _answer_item(task_name: str, question_uid: str, item: Any, index: int) -> SemanticAnswer:
    if isinstance(item, Mapping):
        raw = str(item.get("raw_answer", item.get("answer", item.get("normalized_prediction", ""))))
        parsed = str(item.get("parsed_answer", item.get("normalized_prediction", item.get("answer", raw))))
        valid = bool(item.get("parse_valid", not parsed.startswith("invalid::")))
        output_id = str(item.get("output_id", f"answer-{index}"))
    else:
        raw = str(item)
        parsed = raw
        valid = not parsed.startswith("invalid::")
        output_id = f"answer-{index}"
    answer_type, key, method = _symbolize(task_name, parsed, valid)
    return SemanticAnswer(output_id, question_uid, task_name, raw, parsed, valid, answer_type, key, method)


def cluster_answers(
    task_name: str,
    question_uid: str,
    answers: Iterable[Any],
    *,
    question: str = "",
    mode: str = "task_symbolic",
) -> SemanticClusteringResult:
    if mode != "task_symbolic":
        raise ValueError("Only deterministic task_symbolic mode is currently supported")
    items = [_answer_item(task_name, question_uid, item, index) for index, item in enumerate(answers)]
    output_ids = [item.output_id for item in items]
    if len(output_ids) != len(set(output_ids)):
        raise ValueError("Semantic answer output_id values must be unique within a question")
    size = len(items)
    entailment = [[False] * size for _ in range(size)]
    equivalent = [[False] * size for _ in range(size)]
    decisions: list[PairwiseSemanticDecision] = []
    for i, left in enumerate(items):
        entailment[i][i] = equivalent[i][i] = True
        for j in range(i + 1, size):
            right = items[j]
            if left.answer_type == "invalid" or right.answer_type == "invalid":
                entail_lr = entail_rl = left.answer_type == right.answer_type
                method = "invalid_failure_symbol"
            elif task_name == "math500":
                entail_lr, method_lr = _math_entails(left, right)
                entail_rl, method_rl = _math_entails(right, left)
                method = method_lr if method_lr == method_rl else f"{method_lr}+{method_rl}"
            else:
                entail_lr = entail_rl = left.canonical_key == right.canonical_key
                method = "task_label"
            same = entail_lr and entail_rl
            contradiction_lr = contradiction_rl = (
                not same and left.answer_type in {"mmlu_option", "nli_label"}
            )
            entailment[i][j] = entail_lr
            entailment[j][i] = entail_rl
            equivalent[i][j] = equivalent[j][i] = same
            decisions.append(PairwiseSemanticDecision(
                left.output_id, right.output_id, entail_lr, entail_rl,
                contradiction_lr, contradiction_rl, same, method
            ))
    clusters: list[list[int]] = []
    for index in range(size):
        compatible = [cluster for cluster in clusters if all(equivalent[index][member] for member in cluster)]
        if compatible:
            compatible[0].append(index)
        else:
            clusters.append([index])
    cluster_by_id: dict[str, int] = {}
    members: dict[int, list[str]] = {}
    representatives: dict[int, str] = {}
    for cluster_id, cluster in enumerate(clusters):
        members[cluster_id] = [items[index].output_id for index in cluster]
        representatives[cluster_id] = items[cluster[0]].output_id
        for index in cluster:
            cluster_by_id[items[index].output_id] = cluster_id
    conflicts: list[tuple[str, str, str]] = []
    for i in range(size):
        for j in range(size):
            if equivalent[i][j]:
                for k in range(size):
                    if equivalent[j][k] and not equivalent[i][k]:
                        conflicts.append((items[i].output_id, items[j].output_id, items[k].output_id))
    invalid_count = sum(not item.parse_valid for item in items)
    return SemanticClusteringResult(
        question_uid, task_name, items, cluster_by_id, members, representatives,
        entailment, equivalent, decisions, bool(conflicts), conflicts,
        invalid_count / size if size else 0.0, 0.0,
        {"mode": mode, "question_conditioned": bool(question), "cluster_algorithm": "complete_link"},
    )


def semantic_kernel(result: SemanticClusteringResult):
    return result.same_cluster
