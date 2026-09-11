"""Paired continuation estimators from the EcoMAS uncertainty decomposition."""

from dataclasses import dataclass
from math import comb, exp, floor, log, sqrt
from typing import Callable, Hashable, Sequence


@dataclass(frozen=True)
class Trajectory:
    contexts: Sequence[object]
    actions: Sequence[Hashable]
    outputs: Sequence[object]
    final_output: object


@dataclass
class PairedEstimate:
    system_uncertainty: float
    orchestration: list[float]
    agent_execution: list[float]
    orchestration_samples: list[list[float]]
    agent_samples: list[list[float]]
    boundary_probability_orchestration: list[float]
    boundary_probability_agent: list[float]

    @property
    def total_stepwise(self) -> float:
        return sum(self.orchestration) + sum(self.agent_execution)


def same_cluster(left: object, right: object, cluster_fn: Callable[[object], Hashable] = lambda x: x) -> int:
    return int(cluster_fn(left) == cluster_fn(right))


class PairedSampler:
    """Estimate EcoMAS components using independent C/R/Z continuations.

    ``sample_main`` must sample a complete trajectory. ``continue_from`` must
    independently sample a final output from a context, optionally fixing the
    current action and/or output. The callback is responsible for deterministic
    context updates and all suffix randomness.
    """

    def __init__(self, sample_main: Callable[[], Trajectory], continue_from: Callable[..., object],
                 horizon: int, cluster_fn: Callable[[object], Hashable] = lambda x: x) -> None:
        self.sample_main = sample_main
        self.continue_from = continue_from
        self.horizon = horizon
        self.cluster_fn = cluster_fn

    def estimate(self, num_trajectories: int) -> PairedEstimate:
        if num_trajectories < 1:
            raise ValueError("num_trajectories must be at least 1")
        mains = [self.sample_main() for _ in range(num_trajectories)]
        orch = [[] for _ in range(self.horizon)]
        agent = [[] for _ in range(self.horizon)]
        for trajectory in mains:
            for step in range(self.horizon):
                context = trajectory.contexts[step]
                action = trajectory.actions[step]
                output = trajectory.outputs[step]
                y_c = self.continue_from(context, action=None, output=None)
                y_r = self.continue_from(context, action=action, output=None)
                y_z = trajectory.final_output if step == self.horizon - 1 else self.continue_from(
                    context, action=action, output=output
                )
                orch[step].append(same_cluster(trajectory.final_output, y_r, self.cluster_fn)
                                  - same_cluster(trajectory.final_output, y_c, self.cluster_fn))
                agent[step].append(same_cluster(trajectory.final_output, y_z, self.cluster_fn)
                                   - same_cluster(trajectory.final_output, y_r, self.cluster_fn))
        pair_values = [1 - same_cluster(mains[i].final_output, mains[j].final_output, self.cluster_fn)
                       for i in range(num_trajectories) for j in range(i + 1, num_trajectories)]
        system = sum(pair_values) / len(pair_values) if pair_values else 0.0
        return PairedEstimate(
            system_uncertainty=system,
            orchestration=[sum(values) / num_trajectories for values in orch],
            agent_execution=[sum(values) / num_trajectories for values in agent],
            orchestration_samples=orch,
            agent_samples=agent,
            boundary_probability_orchestration=[sum(value != 0 for value in values) / num_trajectories for values in orch],
            boundary_probability_agent=[sum(value != 0 for value in values) / num_trajectories for values in agent],
        )


def runner_paired_sampler(runner, sample, cluster_fn=lambda x: x) -> PairedSampler:
    """Build a paired estimator backed by a real MASRunner."""
    def sample_main() -> Trajectory:
        record, _ = runner.run_sample(sample, route_mode="sample")
        snapshots = list(runner.last_snapshots)
        return Trajectory(
            contexts=snapshots,
            actions=[step.agent_name for step in record.steps],
            outputs=[step.raw_text for step in record.steps],
            final_output=record.normalized_prediction,
        )

    def continue_from(snapshot, action=None, output=None):
        final = runner.continue_from_snapshot(
            snapshot,
            forced_agent=str(action) if action is not None else None,
            forced_output=str(output) if output is not None else None,
        )
        return final

    return PairedSampler(sample_main, continue_from, int(runner.task_config["steps"]), cluster_fn)


def hoeffding_radius(num_trajectories: int, delta: float, horizon: int) -> float:
    """Return the paper's uniform stepwise radius sqrt(2 log(4T/delta)/B)."""
    if num_trajectories < 1 or horizon < 1 or not 0 < delta < 1:
        raise ValueError("invalid concentration arguments")
    return sqrt(2 * log(4 * horizon / delta) / num_trajectories)


def bernstein_bound(num_trajectories: int, epsilon: float, beta: float, estimate: float = 0.0) -> float:
    variance = max(0.0, beta - estimate * estimate)
    return 2 * exp(-num_trajectories * epsilon * epsilon / (2 * variance + 4 * epsilon / 3))


def system_hoeffding_bound(num_trajectories: int, epsilon: float) -> float:
    return 2 * exp(-2 * floor(num_trajectories / 2) * epsilon * epsilon)
