"""Breadth-first batched execution for EcoMAS calibration sampling."""
from __future__ import annotations

from copy import copy, deepcopy
from dataclasses import dataclass, field

import torch

from ecomas.agents import AgentOutput, is_protocol_compliant, parse_agent_response
from ecomas.evaluation import answer_match, is_valid_answer, normalize_answer
from ecomas.paired_sampling import PairedEstimate, Trajectory
from ecomas.runner import ContextSnapshot, RunRecord, StepRecord, _digest_messages, _summarize_agent_output
from ecomas.semantic_clustering import cluster_answers


@dataclass
class ExecutionState:
    trajectory_index: int
    step: int
    state_messages: list[dict[str, str]]
    previous_results: list[str]
    agents: dict
    route_seed: int
    generation_seed: int
    forced_agent: str | None = None
    forced_output: str | None = None
    final_candidate: str = ""
    answer_source_step: int | None = None
    steps: list[StepRecord] = field(default_factory=list)
    snapshots: list[ContextSnapshot] = field(default_factory=list)
    log_probs: list[torch.Tensor] = field(default_factory=list)


class BatchedMASRunner:
    def __init__(self, runner, micro_batch_size: int = 4):
        self.runner = runner
        self.micro_batch_size = micro_batch_size
        self.horizon = int(runner.task_config["steps"])

    def _agents(self, histories=None):
        result = {}
        for name, agent in self.runner.agents.items():
            item = copy(agent)
            item.dialog_history = deepcopy((histories or {}).get(name, []))
            result[name] = item
        return result

    def _new_main(self, sample, index, seed):
        return ExecutionState(index, 1,
            [{"role": "system", "content": f"You are an assistant. Your task is to {sample.input_text}"}],
            [], self._agents(), seed, seed)

    def _snapshot(self, state, sample):
        return ContextSnapshot(sample, state.step, deepcopy(state.state_messages),
                               deepcopy(state.previous_results),
                               {name: deepcopy(agent.dialog_history) for name, agent in state.agents.items()})

    def _advance_round(self, sample, states, *, record_main: bool):
        encoded = self.runner.encoder([state.state_messages for state in states])
        route_seeds = [state.route_seed + state.step - 1 for state in states]
        decisions = self.runner.router.decide_batch(encoded, mode="sample", seeds=route_seeds)
        jobs = []
        outputs: list[AgentOutput | None] = [None] * len(states)
        selected = []
        for index, (state, decision) in enumerate(zip(states, decisions)):
            if record_main:
                state.snapshots.append(self._snapshot(state, sample))
                state.log_probs.append(decision.log_prob)
            branch_start = state.snapshots[0].step if state.snapshots else None
            agent_name = state.forced_agent if (not record_main and state.forced_agent and state.step == branch_start) else None
            if not agent_name:
                agent_name = decision.agent_name
            agent = state.agents[agent_name]
            selected.append((agent, decision, agent_name))
            if not record_main and state.forced_output is not None and state.step == state.snapshots[0].step:
                raw = state.forced_output
                analysis, candidate = parse_agent_response(raw, self.runner.task_config["display_name"])
                valid = is_valid_answer(self.runner.task_name, candidate)
                agent.dialog_history.append({"role": "assistant", "content": raw})
                outputs[index] = AgentOutput(agent_name, raw, analysis, candidate, valid,
                                             is_protocol_compliant(raw, self.runner.task_config["display_name"]),
                                             "" if valid else "no_valid_task_answer")
            else:
                messages = agent.prepare_messages(sample.input_text, state.previous_results,
                                                  str(self.runner.task_config["display_name"]),
                                                  state.step == self.horizon)
                jobs.append((index, agent, messages))
        if jobs:
            raw_outputs = next(iter(self.runner.agents.values())).llm.generate_messages_batch(
                [job[2] for job in jobs], micro_batch_size=self.micro_batch_size,
                seed=min(states[job[0]].generation_seed + states[job[0]].step - 1 for job in jobs),
            )
            for (index, agent, _), raw in zip(jobs, raw_outputs):
                outputs[index] = agent.consume_response(raw, str(self.runner.task_config["display_name"]))
        for state, output, (agent, decision, agent_name) in zip(states, outputs, selected):
            assert output is not None
            if output.parse_valid:
                state.final_candidate = output.candidate_answer
                state.answer_source_step = state.step
            if record_main:
                state.steps.append(StepRecord(
                    state.step, agent_name, decision.probabilities, output.analysis,
                    output.candidate_answer, output.raw_text, "sample",
                    state.generation_seed + state.step - 1,
                    state.route_seed + state.step - 1,
                    _digest_messages(state.state_messages), output.parse_valid,
                    output.protocol_valid, output.parse_error))
            state.previous_results.append(_summarize_agent_output(output))
            state.state_messages = agent.state_messages()
            state.step += 1

    def sample_main(self, sample, batch_size: int, seed: int):
        states = [self._new_main(sample, index, seed + index * 1009) for index in range(batch_size)]
        while states[0].step <= self.horizon:
            self._advance_round(sample, states, record_main=True)
        trajectories, records = [], []
        for state in states:
            correct, method = answer_match(self.runner.task_name, state.final_candidate, sample.gold_answer)
            record = RunRecord(sample.uid, self.runner.task_name, sample.gold_answer, state.final_candidate,
                normalize_answer(self.runner.task_name, sample.gold_answer),
                normalize_answer(self.runner.task_name, state.final_candidate), correct, state.steps,
                {**sample.metadata, "route_mode": "sample", "batched": True}, method,
                [step.agent_name for step in state.steps], is_valid_answer(self.runner.task_name, state.final_candidate),
                state.steps[-1].parse_valid, state.steps[-1].protocol_valid, state.answer_source_step,
                bool(state.answer_source_step and state.answer_source_step != len(state.steps)))
            records.append(record)
            trajectories.append(Trajectory(state.snapshots, [s.agent_name for s in state.steps],
                [s.raw_text for s in state.steps], record.normalized_prediction,
                state.log_probs, [s.probabilities for s in state.steps], record))
        return trajectories, records

    def sample_branches(self, sample, trajectories, seed):
        states = []
        for b, trajectory in enumerate(trajectories):
            for t in range(self.horizon):
                snapshot = trajectory.contexts[t]
                for branch_index, kind in enumerate(("c", "r", "z")):
                    state = ExecutionState(b * self.horizon * 3 + t * 3 + branch_index,
                        snapshot.step, deepcopy(snapshot.state_messages), deepcopy(snapshot.previous_results),
                        self._agents(snapshot.agent_histories), seed + b * 10007 + t * 101 + branch_index,
                        seed + b * 10007 + t * 101 + branch_index,
                        trajectory.actions[t] if kind in {"r", "z"} else None,
                        trajectory.outputs[t] if kind == "z" else None,
                        snapshots=[snapshot])
                    states.append(state)
        active = states
        while active and active[0].step <= self.horizon:
            by_step = {}
            for state in active:
                by_step.setdefault(state.step, []).append(state)
            for group in by_step.values():
                self._advance_round(sample, group, record_main=False)
            active = [state for state in active if state.step <= self.horizon]
        result = [[{} for _ in range(self.horizon)] for _ in trajectories]
        for state in states:
            b = state.trajectory_index // (self.horizon * 3)
            remainder = state.trajectory_index % (self.horizon * 3)
            t, branch_index = divmod(remainder, 3)
            result[b][t][("c", "r", "z")[branch_index]] = state.final_candidate
        return result

    def paired_estimate(self, sample, batch_size: int, seed: int):
        trajectories, records = self.sample_main(sample, batch_size, seed)
        branches = self.sample_branches(sample, trajectories, seed + 500000)
        all_outputs = [t.final_output for t in trajectories]
        for row in branches:
            for item in row:
                all_outputs.extend([item["c"], item["r"], item["z"]])
        clustered = cluster_answers(self.runner.task_name, str(sample.uid),
            [{"output_id": str(i), "answer": value} for i, value in enumerate(all_outputs)],
            question=sample.input_text)
        ids = [clustered.cluster_by_output_id[str(i)] for i in range(len(all_outputs))]
        pair_values = [1 - int(ids[i] == ids[j]) for i in range(batch_size) for j in range(i + 1, batch_size)]
        return PairedEstimate(sum(pair_values) / len(pair_values) if pair_values else 0.0,
            [0.0] * self.horizon, [0.0] * self.horizon, [[] for _ in range(self.horizon)],
            [[] for _ in range(self.horizon)], [0.0] * self.horizon, [0.0] * self.horizon,
            [t.final_output for t in trajectories], branches, [t.route_log_probs for t in trajectories], records)
