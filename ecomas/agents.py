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
            "Now, you need to continue the reasoning to get closer to the correct answer. "
            "You should finish your reasoning with the following template: "
            "REASONING RESULT: [YOUR REASONING RESULT]. "
            "Finish your answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER]. "
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
        raw_text = self.llm.generate_messages(response_messages)
        self.dialog_history.append({"role": "assistant", "content": str(raw_text)})
        analysis, candidate = parse_agent_response(raw_text)
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


def parse_agent_response(text: str) -> tuple[str, str]:
    analysis = ""
    candidate = ""
    marker = "FINAL ANSWER:"
    if marker in text:
        before, after = text.split(marker, 1)
        analysis = before.replace("ANALYSIS:", "", 1).strip()
        candidate = after.strip().splitlines()[0].strip()
    elif "CANDIDATE_ANSWER:" in text:
        before, after = text.split("CANDIDATE_ANSWER:", 1)
        analysis = before.replace("ANALYSIS:", "", 1).strip()
        candidate = after.strip().splitlines()[0].strip()
    else:
        analysis = text.strip()
        candidate = text.strip().splitlines()[-1].strip() if text.strip() else ""
    return analysis, candidate
