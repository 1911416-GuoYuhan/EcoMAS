from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class LLMClient(Protocol):
    def generate(self, system_prompt: str, user_prompt: str) -> str:
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

    def run(self, task_text: str, history: str, answer_format: str) -> AgentOutput:
        system_prompt = (
            f"{self.spec.prompt}\n\n"
            "You are one specialist in a multi-agent benchmark system. "
            "Do not call tools, browse, cite external APIs, or ask for more input. "
            "You must reason from the problem text and the prior agent history. "
            "You may explicitly correct previous candidates when needed.\n\n"
            "Every response must use exactly this format:\n"
            "ANALYSIS: <your concise analysis>\n"
            "CANDIDATE_ANSWER: <current best answer>\n\n"
            f"Answer format requirement: {answer_format}\n\n"
            f"Specialist example:\n{self.spec.example}"
        )
        user_prompt = (
            f"Task:\n{task_text}\n\n"
            f"Prior agent history:\n{history if history else '(none)'}\n\n"
            "Return your analysis and current candidate answer now."
        )
        raw_text = self.llm.generate(system_prompt, user_prompt)
        analysis, candidate = parse_agent_response(raw_text)
        return AgentOutput(
            agent_name=self.spec.name,
            raw_text=raw_text,
            analysis=analysis,
            candidate_answer=candidate,
        )


def parse_agent_response(text: str) -> tuple[str, str]:
    analysis = ""
    candidate = ""
    marker = "CANDIDATE_ANSWER:"
    if marker in text:
        before, after = text.split(marker, 1)
        analysis = before.replace("ANALYSIS:", "", 1).strip()
        candidate = after.strip().splitlines()[0].strip()
    else:
        analysis = text.strip()
        candidate = text.strip().splitlines()[-1].strip() if text.strip() else ""
    return analysis, candidate
