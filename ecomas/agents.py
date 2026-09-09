from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import re
from typing import Protocol


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


class SpecialistAgent:
    def __init__(self, spec: AgentSpec, llm: LLMClient) -> None:
        self.spec = spec
        self.llm = llm
        self.dialog_history: list[dict[str, str]] = []

    def _system_prompt(self, task_text: str, task_name: str) -> str:
        suffixes = {
            "MMLU-Pro": "For MMLU-Pro, return exactly one option letter A-J at the end.",
            "MATH-500": "For MATH-500, return a concise exact mathematical answer, preferably LaTeX.",
            "ChaosNLI": "For ChaosNLI, return exactly one label: entailment, neutral, or contradiction.",
        }
        role_prompt = f"{self.spec.prompt} {suffixes[task_name]}"
        return (
            f"{role_prompt}, and You work as a helpful AI assistant. \n"
            "I will ask you a question. Answer this question using your coding and language skills.\n"
            f"Now your question is: {task_text}\n"
            "Previously, you collected the some information about this question from some actions: []"
        )

    def run(
        self,
        task_text: str,
        previous_results: list[str],
        task_name: str,
        generation_seed: int | None = None,
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

        user_prompt = (
            "Return the answer before any explanation. Your first non-empty line MUST be exactly "
            "FINAL ANSWER: [one answer]. Do not put reasoning, option lists, or a preamble before it. "
            "Then provide at most a short explanation starting with REASONING RESULT:. "
            "For MMLU-Pro, [one answer] must be one letter A-J; for ChaosNLI it must be "
            "entailment, neutral, or contradiction. For MATH-500, provide the exact answer. "
            "Now continue the reasoning to get closer to the correct answer. "
            f"*Your previous reasoning was: {previous_results[-4:]}.* "
            "You need to follow the direction of the reasoning path and go forward:"
        )
        user_message = {"role": "user", "content": user_prompt}
        if self.dialog_history[-1] != user_message:
            self.dialog_history.append(user_message)
        if len(self.dialog_history) > 5:
            response_messages = self.dialog_history[:1] + self.dialog_history[-4:]
        else:
            response_messages = deepcopy(self.dialog_history)
        try:
            raw_text = self.llm.generate_messages(response_messages, seed=generation_seed)
        except TypeError:
            raw_text = self.llm.generate_messages(response_messages)
        self.dialog_history.append({"role": "assistant", "content": str(raw_text)})
        analysis, candidate = parse_agent_response(raw_text, task_name)
        return AgentOutput(
            agent_name=self.spec.name,
            raw_text=raw_text,
            analysis=analysis,
            candidate_answer=candidate,
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
    """Extract reasoning and a task-compatible answer from imperfect LLM output."""
    import re

    raw = str(text or "").strip()
    if not raw:
        return "", ""
    marker_pattern = re.compile(
        r"^\s*(?:FINAL\s+ANSWER|CANDIDATE[_ ]ANSWER)\s*[:：=]?\s*",
        re.IGNORECASE | re.MULTILINE,
    )
    marker = marker_pattern.search(raw)
    if marker:
        analysis = raw[: marker.start()]
        answer_text = raw[marker.end() :].strip()
    else:
        analysis = raw
        answer_text = raw

    task = (task_name or "").lower()
    if "mmlu" in task:
        candidate = _extract_mmlu_answer(answer_text) or _extract_mmlu_answer(raw)
    elif "chaos" in task or "nli" in task:
        candidate = _extract_chaos_answer(answer_text) or _extract_chaos_answer(raw)
    else:
        candidate = _extract_math_answer(answer_text) if marker else _extract_boxed_answer(answer_text)
    if not candidate and "mmlu" not in task and ("chaos" not in task and "nli" not in task):
        lines = [line.strip() for line in answer_text.splitlines() if line.strip()]
        candidate = lines[0] if lines else ""
    analysis = re.sub(r"^\s*ANALYSIS\s*[:：]?\s*", "", analysis, flags=re.IGNORECASE).strip()
    return analysis, candidate


def _extract_mmlu_answer(text: str) -> str:
    import re

    patterns = [
        r"(?:final\s+answer|candidate\s+answer|answer|choice|option)\s*(?:is|:|=)?\s*[\(\[]?\s*([A-J])\b",
        r"(?:^|\n)\s*[\(\[]?([A-J])[\)\]]?\s*(?:\n|$)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return matches[-1].upper()
    return ""


def _extract_chaos_answer(text: str) -> str:
    import re

    matches = re.findall(r"\b(entailment|neutral|contradiction)\b", text, flags=re.IGNORECASE)
    return matches[-1].lower() if matches else ""


def _extract_math_answer(text: str) -> str:
    import re

    text = text.split("REASONING RESULT:", 1)[0].strip()
    boxed_start = text.rfind(r"\boxed{")
    if boxed_start >= 0:
        boxed = text[boxed_start + len(r"\boxed{") :].strip()
        if boxed.endswith("}"):
            return boxed[:-1].strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    candidate = lines[0]
    return candidate if len(candidate) <= 160 else ""


def _extract_boxed_answer(text: str) -> str:
    import re

    boxed_start = text.rfind(r"\boxed{")
    if boxed_start < 0 or not text.rstrip().endswith("}"):
        return ""
    return text[boxed_start + len(r"\boxed{") :].rstrip()[:-1].strip()
