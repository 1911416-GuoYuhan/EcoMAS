from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import torch

from ecomas.agents import AgentOutput, SpecialistAgent, is_protocol_compliant, parse_agent_response
from ecomas.datasets import BenchmarkSample
from ecomas.encoder import FrozenTextEncoder
from ecomas.evaluation import answer_match, is_valid_answer, normalize_answer
from ecomas.registry import TASK_REGISTRY
from ecomas.router import ArgmaxRouter


@dataclass
class StepRecord:
    step: int
    agent_name: str
    probabilities: list[float]
    analysis: str
    candidate_answer: str
    raw_text: str
    route_mode: str = "argmax"
    generation_seed: int | None = None
    route_seed: int | None = None
    state_digest: str = ""
    parse_valid: bool = False
    protocol_valid: bool = False
    parse_error: str = ""


@dataclass
class RunRecord:
    uid: str
    task_name: str
    gold_answer: str
    prediction: str
    normalized_gold: str
    normalized_prediction: str
    correct: bool
    steps: list[StepRecord]
    metadata: dict
    match_method: str = ""
    path: list[str] = field(default_factory=list)
    output_parse_valid: bool = False
    terminal_step_parse_valid: bool = False
    terminal_step_protocol_valid: bool = False
    answer_source_step: int | None = None
    used_answer_fallback: bool = False


@dataclass
class ContextSnapshot:
    sample: BenchmarkSample
    step: int
    state_messages: list[dict[str, str]]
    previous_results: list[str]
    agent_histories: dict[str, list[dict[str, str]]]


class MASRunner:
    def __init__(
        self,
        task_name: str,
        agents: list[SpecialistAgent],
        encoder: FrozenTextEncoder,
        router: ArgmaxRouter,
    ) -> None:
        self.task_name = task_name
        self.agents = {agent.spec.name: agent for agent in agents}
        self.encoder = encoder
        self.router = router
        self.task_config = TASK_REGISTRY[task_name]
        self.branch_cache: dict[tuple, str] = {}

    def snapshot(self, sample: BenchmarkSample, step: int, state_messages: list[dict[str, str]],
                 previous_results: list[str]) -> ContextSnapshot:
        return ContextSnapshot(sample, step, deepcopy(state_messages), deepcopy(previous_results),
                               {name: deepcopy(agent.dialog_history) for name, agent in self.agents.items()})

    def continue_from_snapshot(
        self,
        snapshot: ContextSnapshot,
        forced_agent: str | None = None,
        forced_output: str | None = None,
        seed: int | None = None,
        *,
        route_mode: str = "sample",
        route_seed: int | None = None,
        generation_seed: int | None = None,
    ) -> str:
        """Run only the suffix under an explicit conditional policy.

        ``route_mode`` selects either categorical sampling or argmax for every
        unfixed route in the suffix. ``seed`` remains a backwards-compatible
        shorthand for both seed roots.
        """
        if seed is not None:
            route_seed = seed if route_seed is None else route_seed
            generation_seed = seed if generation_seed is None else generation_seed
        if route_mode not in {"argmax", "sample"}:
            raise ValueError(f"Unknown route mode: {route_mode}")
        snapshot_payload = (
            snapshot.state_messages,
            snapshot.previous_results,
            snapshot.agent_histories,
        )
        key = (
            snapshot.sample.uid, snapshot.step, forced_agent, forced_output,
            route_mode, route_seed, generation_seed,
            _digest_object(snapshot_payload),
        )
        if key in self.branch_cache:
            return self.branch_cache[key]
        for name, history in snapshot.agent_histories.items():
            self.agents[name].dialog_history = deepcopy(history)
        state_messages = deepcopy(snapshot.state_messages)
        previous_results = deepcopy(snapshot.previous_results)
        final_candidate = ""
        for step in range(snapshot.step, int(self.task_config["steps"]) + 1):
            offset = step - snapshot.step
            decision = self.router.decide(
                self.encoder([state_messages]),
                mode=route_mode,
                seed=(route_seed + offset) if route_seed is not None else None,
            )
            agent = self.agents[forced_agent] if step == snapshot.step and forced_agent else self.agents[decision.agent_name]
            if step == snapshot.step and forced_output is not None:
                raw_text = forced_output
                analysis, candidate = parse_agent_response(raw_text, self.task_config["display_name"])
                parse_valid = is_valid_answer(self.task_name, candidate)
                output = AgentOutput(
                    agent.spec.name,
                    raw_text,
                    analysis,
                    candidate,
                    parse_valid,
                    is_protocol_compliant(raw_text, self.task_config["display_name"]),
                    "" if parse_valid else "no_valid_task_answer",
                )
                agent.dialog_history.append({"role": "assistant", "content": raw_text})
            else:
                output = agent.run(snapshot.sample.input_text, previous_results,
                                   str(self.task_config["display_name"]),
                                   generation_seed=(generation_seed + offset) if generation_seed is not None else None,
                                   is_final_step=(step == int(self.task_config["steps"])))
            if output.parse_valid:
                final_candidate = output.candidate_answer
            previous_results.append(_summarize_agent_output(output))
            state_messages = agent.state_messages()
        self.branch_cache[key] = final_candidate
        return final_candidate

    def run_sample(
        self,
        sample: BenchmarkSample,
        training: bool = False,
        loss_fn: Callable[[str, str, str], float] | None = None,
        route_mode: str = "argmax",
        generation_seed: int | None = None,
        route_seed: int | None = None,
    ) -> tuple[RunRecord, torch.Tensor | None]:
        if generation_seed is not None:
            torch.manual_seed(generation_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(generation_seed)
        for agent in self.agents.values():
            agent.reset()

        previous_results: list[str] = []
        state_messages = [
            {
                "role": "system",
                "content": f"You are an assistant. Your task is to {sample.input_text}",
            }
        ]
        step_records: list[StepRecord] = []
        log_probs: list[torch.Tensor] = []
        final_candidate = ""
        answer_source_step: int | None = None
        self.last_snapshots: list[ContextSnapshot] = []

        for step in range(1, int(self.task_config["steps"]) + 1):
            self.last_snapshots.append(self.snapshot(sample, step, state_messages, previous_results))
            encoded = self.encoder([state_messages])
            current_route_seed = (route_seed + step - 1) if route_seed is not None else None
            decision = self.router.decide(encoded, mode=route_mode, seed=current_route_seed)
            agent = self.agents[decision.agent_name]
            output: AgentOutput = agent.run(
                sample.input_text,
                previous_results,
                str(self.task_config["display_name"]),
                generation_seed=(generation_seed + step - 1) if generation_seed is not None else None,
                is_final_step=(step == int(self.task_config["steps"])),
            )
            if output.parse_valid:
                final_candidate = output.candidate_answer
                answer_source_step = step
            log_probs.append(decision.log_prob)
            step_records.append(
                StepRecord(
                    step=step,
                    agent_name=decision.agent_name,
                    probabilities=decision.probabilities,
                    analysis=output.analysis,
                    candidate_answer=output.candidate_answer,
                    raw_text=output.raw_text,
                    route_mode=route_mode,
                    generation_seed=(generation_seed + step - 1) if generation_seed is not None else None,
                    route_seed=current_route_seed,
                    state_digest=_digest_messages(state_messages),
                    parse_valid=output.parse_valid,
                    protocol_valid=output.protocol_valid,
                    parse_error=output.parse_error,
                )
            )
            previous_results.append(_summarize_agent_output(output))
            state_messages = agent.state_messages()

        correct, match_method = answer_match(self.task_name, final_candidate, sample.gold_answer)
        record = RunRecord(
            uid=sample.uid,
            task_name=self.task_name,
            gold_answer=sample.gold_answer,
            prediction=final_candidate,
            normalized_gold=normalize_answer(self.task_name, sample.gold_answer),
            normalized_prediction=normalize_answer(self.task_name, final_candidate),
            correct=correct,
            match_method=match_method,
            steps=step_records,
            metadata={
                **sample.metadata,
                "route_mode": route_mode,
                "generation_seed": generation_seed,
                "output_parse_valid": is_valid_answer(self.task_name, final_candidate),
                "terminal_step_parse_valid": step_records[-1].parse_valid if step_records else False,
                "terminal_step_protocol_valid": step_records[-1].protocol_valid if step_records else False,
                "answer_source_step": answer_source_step,
                "used_answer_fallback": bool(answer_source_step and answer_source_step != len(step_records)),
            },
            path=[step.agent_name for step in step_records],
            output_parse_valid=is_valid_answer(self.task_name, final_candidate),
            terminal_step_parse_valid=step_records[-1].parse_valid if step_records else False,
            terminal_step_protocol_valid=step_records[-1].protocol_valid if step_records else False,
            answer_source_step=answer_source_step,
            used_answer_fallback=bool(answer_source_step and answer_source_step != len(step_records)),
        )
        loss = None
        if training:
            score = loss_fn(self.task_name, final_candidate, sample.gold_answer) if loss_fn else float(correct)
            reward = torch.tensor(float(score), device=self.router.device)
            loss = -reward * torch.stack(log_probs).sum()
        return record, loss

def write_jsonl(path: Path, records: list[RunRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def _digest_messages(messages: list[dict[str, str]]) -> str:
    import hashlib

    payload = json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _digest_object(value: object) -> str:
    import hashlib

    payload = repr(value).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _summarize_agent_output(output: AgentOutput) -> str:
    reasoning = output.analysis.strip()
    reasoning = re.sub(
        r"(?:You need to follow the direction|follow the direction of the reasoning path|go forward:?).*$",
        "",
        reasoning,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()
    reasoning = re.sub(r"\b(?:FINAL\s+ANSWER|CANDIDATE[_ ]ANSWER)\s*:", "Reported answer:", reasoning,
                       flags=re.IGNORECASE)
    if len(reasoning) > 2400:
        reasoning = reasoning[-2400:]
    candidate = output.candidate_answer if output.parse_valid else "(no valid task answer)"
    return f"Agent: {output.agent_name}\nReasoning: {reasoning}\nCandidate answer: {candidate}"
