from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import re
from typing import Protocol

from ecomas.config import METRIC_CONFIG, PROMPT_CONFIG
from ecomas.evaluation import is_valid_answer


class LLMClient(Protocol):
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        ...

    def generate_messages(self, messages: list[dict[str, str]]) -> str:
        ...


@dataclass(frozen=True)
class AgentSpec:
    name: str
    role: str
    prompt: str
    example: str


@dataclass
class AgentOutput:
    agent_name: str
    raw_text: str
    analysis: str
    candidate_answer: str
    parse_valid: bool
    protocol_valid: bool
    parse_error: str = ""


class SpecialistAgent:
    def __init__(self, spec: AgentSpec, llm: LLMClient) -> None:
        self.spec = spec
        self.llm = llm
        self.dialog_history: list[dict[str, str]] = []

    def _system_prompt(self, task_text: str, task_name: str) -> str:
        role_prompt = self.spec.prompt
        if self.spec.example and self.spec.example not in role_prompt:
            role_prompt = PROMPT_CONFIG["role_example_template"].format(
                role_prompt=role_prompt,
                example=self.spec.example,
            )
        return PROMPT_CONFIG["system_template"].format(
            role_prompt=role_prompt,
            task_text=task_text,
        )

    def run(
        self,
        task_text: str,
        previous_results: list[str],
        task_name: str,
        generation_seed: int | None = None,
        is_final_step: bool = False,
    ) -> AgentOutput:
        if not self.dialog_history:
            self.dialog_history = [
                {"role": "system", "content": self._system_prompt(task_text, task_name)}
            ]
        else:
            self.dialog_history[0] = {
                "role": "system",
                "content": self._system_prompt(task_text, task_name),
            }

        protocols = PROMPT_CONFIG["protocols"]
        history_limit = PROMPT_CONFIG["history_limit"]
        prior_context = "\n\n".join(previous_results[-history_limit:]) if previous_results else PROMPT_CONFIG["empty_history"]
        user_prompt = PROMPT_CONFIG["user_template"].format(
            prior_context=prior_context,
            protocol=protocols[task_name],
        )
        if is_final_step:
            user_prompt += f"\n\n{PROMPT_CONFIG['final_instructions'][task_name]}"
        user_message = {"role": "user", "content": user_prompt}
        if self.dialog_history[-1] != user_message:
            self.dialog_history.append(user_message)
        response_messages = [deepcopy(self.dialog_history[0]), deepcopy(user_message)]
        try:
            raw_text = self.llm.generate_messages(response_messages, seed=generation_seed)
        except TypeError:
            raw_text = self.llm.generate_messages(response_messages)
        self.dialog_history.append({"role": "assistant", "content": str(raw_text)})
        analysis, candidate = parse_agent_response(raw_text, task_name)
        task_key = _task_key(task_name)
        parse_valid = is_valid_answer(task_key, candidate)
        protocol_valid = is_protocol_compliant(raw_text, task_name)
        return AgentOutput(
            agent_name=self.spec.name,
            raw_text=raw_text,
            analysis=analysis,
            candidate_answer=candidate,
            parse_valid=parse_valid,
            protocol_valid=protocol_valid,
            parse_error="" if parse_valid else "no_valid_task_answer",
        )

    def state_messages(self) -> list[dict[str, str]]:
        messages = deepcopy(self.dialog_history)
        for message in messages:
            if message.get("role") == "user":
                message["content"] = re.sub(r"\*.*?\*", "", message["content"])
        return messages

    def reset(self) -> None:
        self.dialog_history = []


def parse_agent_response(text: str, task_name: str | None = None) -> tuple[str, str]:
    import re

    raw = str(text or "").strip()
    if not raw:
        return "", ""
    marker_pattern = re.compile(
        r"^\s*(?:FINAL\s+ANSWER|CANDIDATE[_ ]ANSWER)\s*[:：=]?\s*",
        re.IGNORECASE | re.MULTILINE,
    )
    markers = list(marker_pattern.finditer(raw))
    marker = markers[-1] if markers else None
    if marker:
        analysis = raw[: marker.start()]
        answer_text = raw[marker.end() :].strip()
    else:
        analysis = raw
        answer_text = raw

    task = (task_name or "").lower()
    if "mmlu" in task:
        candidate = _extract_mmlu_answer(answer_text, require_terminal=not marker)
    elif "chaos" in task or "nli" in task:
        candidate = _extract_chaos_answer(answer_text, require_terminal=not marker)
    else:
        candidate = _extract_math_answer(answer_text) if marker else _extract_boxed_answer(answer_text)
    analysis = re.sub(r"^\s*ANALYSIS\s*[:：]?\s*", "", analysis, flags=re.IGNORECASE).strip()
    return analysis, candidate


def _extract_mmlu_answer(text: str, require_terminal: bool = False) -> str:
    import re

    text = text.replace("\\\\", "\\")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        match = re.fullmatch(r"(?:FINAL ANSWER\s*:\s*)?[\(\[]?([A-J])[\)\]]?[.!]?$", line, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    if not require_terminal and lines:
        match = re.match(r"^[`'\"\s]*[\(\[]?([A-J])[\)\]]?(?:\s*[:.)-]\s+.*)?$", lines[0], re.IGNORECASE)
        if match:
            return match.group(1).upper()
    patterns = [r"(?:final\s+answer|candidate\s+answer|answer|choice|option)\s*(?:is|:|=)?\s*[\(\[]?\s*([A-J])\b"]
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            match = matches[-1].upper()
            if not require_terminal or re.search(rf"{match}[\)\].!\s]*$", text, re.IGNORECASE):
                return match
    return ""


def _extract_chaos_answer(text: str, require_terminal: bool = False) -> str:
    import re

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        match = re.fullmatch(
            r"(?:FINAL ANSWER\s*:\s*)?(entailment|neutral|contradiction)[.!]?",
            line,
            re.IGNORECASE,
        )
        if match:
            return match.group(1).lower()
    matches = list(re.finditer(r"\b(entailment|neutral|contradiction)\b", text, re.IGNORECASE))
    if matches and (not require_terminal or not text[matches[-1].end():].strip(" .!`'\"")):
        return matches[-1].group(1).lower()
    return ""


def _extract_math_answer(text: str) -> str:
    import re

    text = text.replace("\\\\", "\\")
    text = re.split(r"(?:\]\s*(?:\]\s*)?\.\*|\*Your previous reasoning|You need to follow|Please continue|REASONING RESULT:)", text, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    boxed = re.findall(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", text)
    if boxed:
        return boxed[-1].strip()
    terminal = re.search(r"\b(?:is|equals|equal to)\s+([^\n.]+)\.?\s*$", text, re.IGNORECASE)
    if terminal and len(terminal.group(1).strip()) <= METRIC_CONFIG["math_answer_max_characters"]:
        return terminal.group(1).strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    candidate = lines[-1]
    candidate = re.sub(r"\\[\(\[|\\[\)\]]", "", candidate).replace("$", "").strip(" `'\"")
    if "=" in candidate:
        rhs = re.search(r"=\s*([^=\n]+?)\s*$", candidate)
        if rhs:
            candidate = rhs.group(1).strip()
    return candidate if is_valid_answer("math500", candidate) else ""


def _extract_boxed_answer(text: str) -> str:
    import re

    boxed_start = text.rfind(r"\boxed{")
    if boxed_start < 0 or not text.rstrip().endswith("}"):
        return ""
    candidate = text[boxed_start + len(r"\boxed{") :].rstrip()[:-1].strip()
    return candidate if is_valid_answer("math500", candidate) else ""


def _task_key(task_name: str | None) -> str:
    task = str(task_name or "").lower()
    if "mmlu" in task:
        return "mmlu_pro"
    if "chaos" in task or "nli" in task:
        return "chaosnli"
    return "math500"


def is_protocol_compliant(text: str, task_name: str | None) -> bool:
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    if not lines:
        return False
    final_line = lines[-1]
    task = _task_key(task_name)
    if task == "mmlu_pro":
        return re.fullmatch(r"FINAL\s+ANSWER\s*:\s*[A-J]", final_line, re.IGNORECASE) is not None
    if task == "chaosnli":
        return re.fullmatch(
            r"FINAL\s+ANSWER\s*:\s*(?:entailment|neutral|contradiction)",
            final_line,
            re.IGNORECASE,
        ) is not None
    match = re.fullmatch(r"FINAL\s+ANSWER\s*:\s*(.+)", final_line, re.IGNORECASE)
    if not match:
        return False
    candidate = _extract_math_answer(match.group(1))
    return is_valid_answer("math500", candidate)
