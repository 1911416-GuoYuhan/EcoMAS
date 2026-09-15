"""Paired continuation estimators from the EcoMAS uncertainty decomposition."""

from dataclasses import dataclass
from typing import Callable, Hashable, Sequence
from concurrent.futures import ThreadPoolExecutor
from enum import Enum


@dataclass(frozen=True)
class Trajectory:
    contexts: Sequence[object]
    actions: Sequence[Hashable]
    outputs: Sequence[object]
    final_output: object
    route_log_probs: Sequence[object] = ()
    route_probabilities: Sequence[Sequence[float]] = ()
    record: object | None = None


@dataclass
class PairedEstimate:
    system_uncertainty: float
    orchestration: list[float]
    agent_execution: list[float]
    orchestration_samples: list[list[float]]
    agent_samples: list[list[float]]
    boundary_probability_orchestration: list[float]
    boundary_probability_agent: list[float]
    main_outputs: list[object] | None = None
    branch_outputs: list[list[dict[str, object]]] | None = None
    route_log_probs: list[Sequence[object]] | None = None
    records: list[object] | None = None

    @property
    def total_stepwise(self) -> float:
        return sum(self.orchestration) + sum(self.agent_execution)


class BranchType(str, Enum):
    C = "c"
    R = "r"
    Z = "z"


@dataclass(frozen=True)
class BranchRequest:
    trajectory_index: int
    step: int
    branch_type: BranchType
    context: object
    action: Hashable | None
    output: object | None


def build_branch_requests(trajectories: Sequence[Trajectory], horizon: int) -> list[BranchRequest]:
    requests = []
    for trajectory_index, trajectory in enumerate(trajectories):
        for step in range(horizon):
            requests.extend([
                BranchRequest(trajectory_index, step, BranchType.C, trajectory.contexts[step], None, None),
                BranchRequest(trajectory_index, step, BranchType.R, trajectory.contexts[step], trajectory.actions[step], None),
                BranchRequest(trajectory_index, step, BranchType.Z, trajectory.contexts[step], trajectory.actions[step], trajectory.outputs[step]),
            ])
    return requests


def execute_branch_requests(
    requests: Sequence[BranchRequest],
    continue_fn: Callable[[BranchRequest], object],
    *,
    batch_fn: Callable[[Sequence[BranchRequest]], Sequence[object]] | None = None,
    micro_batch_size: int = 1,
) -> list[object]:
    """Execute branch requests in stable order with bounded micro-batches.

    ``batch_fn`` is intentionally injected: a real runner can provide a
    single-model batched generation implementation, while tests and legacy
    callers use ``continue_fn``.  No background threads are used here, so CUDA
    model state and paired request ordering remain deterministic.
    """
    if micro_batch_size < 1:
        raise ValueError("micro_batch_size must be positive")
    requests = list(requests)
    if batch_fn is None or micro_batch_size == 1:
        return [continue_fn(request) for request in requests]
    values: list[object] = []
    for start in range(0, len(requests), micro_batch_size):
        chunk = requests[start:start + micro_batch_size]
        result = list(batch_fn(chunk))
        if len(result) != len(chunk):
            raise ValueError("batch_fn must return one value per request")
        values.extend(result)
    return values


def same_cluster(
    left: object,
    right: object,
    cluster_fn: Callable[[object], Hashable] = lambda x: x,
    cluster_kernel: Callable[[object, object], bool] | None = None,
) -> int:
    """Return the binary semantic kernel used by the estimators.

    ``cluster_kernel`` is preferred for question-conditioned semantic
    clustering. ``cluster_fn`` remains as a compatibility path for exact
    symbolic keys and existing callers.
    """
    if cluster_kernel is not None:
        return int(bool(cluster_kernel(left, right)))
    return int(cluster_fn(left) == cluster_fn(right))


class PairedSampler:
    """Estimate EcoMAS components using independent C/R/Z continuations.

    ``sample_main`` must sample a complete trajectory. ``continue_from`` must
    independently sample a final output from a context, optionally fixing the
    current action and/or output. The callback is responsible for deterministic
    context updates and all suffix randomness.
    """

    def __init__(self, sample_main: Callable[[], Trajectory], continue_from: Callable[..., object],
                 horizon: int, cluster_fn: Callable[[object], Hashable] = lambda x: x,
                 cluster_kernel: Callable[[object, object], bool] | None = None,
                 batch_cluster_fn: Callable[[Sequence[object]], Sequence[Hashable]] | None = None,
                 parallel_workers: int = 1) -> None:
        self.sample_main = sample_main
        self.continue_from = continue_from
        self.horizon = horizon
        self.cluster_fn = cluster_fn
        self.cluster_kernel = cluster_kernel
        self.batch_cluster_fn = batch_cluster_fn
        self.parallel_workers = max(1, int(parallel_workers))

    def estimate(self, num_trajectories: int) -> PairedEstimate:
        if num_trajectories < 1:
            raise ValueError("num_trajectories must be at least 1")
        mains = [self.sample_main() for _ in range(num_trajectories)]
        orch = [[] for _ in range(self.horizon)]
        agent = [[] for _ in range(self.horizon)]
        all_outputs: list[object] = [trajectory.final_output for trajectory in mains]
        branch_indices: list[list[tuple[int, int, int]]] = []
        branch_outputs: list[list[dict[str, object]]] = []
        for trajectory in mains:
            trajectory_branches: list[tuple[int, int, int]] = []
            trajectory_outputs: list[dict[str, object]] = []
            def branch_values(step):
                context = trajectory.contexts[step]
                action = trajectory.actions[step]
                output = trajectory.outputs[step]
                calls = [
                    (context, None, None),
                    (context, action, None),
                ]
                if step == self.horizon - 1:
                    calls.append((context, action, output))
                else:
                    calls.append((context, action, output))
                if self.parallel_workers == 1:
                    values = [self.continue_from(*call) for call in calls]
                else:
                    with ThreadPoolExecutor(max_workers=3) as pool:
                        futures = [pool.submit(self.continue_from, *call) for call in calls]
                        values = [future.result() for future in futures]
                return values

            def collect_step(step):
                context = trajectory.contexts[step]
                action = trajectory.actions[step]
                output = trajectory.outputs[step]
                y_c, y_r, y_z = branch_values(step)
                if step == self.horizon - 1:
                    y_z = trajectory.final_output
                return step, (y_c, y_r, y_z)

            if self.parallel_workers == 1:
                step_results = [collect_step(step) for step in range(self.horizon)]
            else:
                with ThreadPoolExecutor(max_workers=self.parallel_workers) as pool:
                    step_results = [future.result() for future in [pool.submit(collect_step, step) for step in range(self.horizon)]]
            for step, (y_c, y_r, y_z) in sorted(step_results):
                indices = []
                for value in (y_c, y_r, y_z):
                    indices.append(len(all_outputs))
                    all_outputs.append(value)
                trajectory_branches.append(tuple(indices))
                trajectory_outputs.append({"c": y_c, "r": y_r, "z": y_z})
            branch_indices.append(trajectory_branches)
            branch_outputs.append(trajectory_outputs)

        cluster_ids = list(self.batch_cluster_fn(all_outputs)) if self.batch_cluster_fn else None
        if cluster_ids is not None and len(cluster_ids) != len(all_outputs):
            raise ValueError("batch_cluster_fn must return one cluster ID per collected output")

        def kernel(left_index: int, right_index: int) -> int:
            if cluster_ids is not None:
                return int(cluster_ids[left_index] == cluster_ids[right_index])
            return same_cluster(all_outputs[left_index], all_outputs[right_index], self.cluster_fn, self.cluster_kernel)

        for main_index in range(num_trajectories):
            for step, (c_index, r_index, z_index) in enumerate(branch_indices[main_index]):
                orch[step].append(kernel(main_index, r_index) - kernel(main_index, c_index))
                agent[step].append(kernel(main_index, z_index) - kernel(main_index, r_index))
        pair_values = [1 - kernel(i, j)
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
            main_outputs=[trajectory.final_output for trajectory in mains],
            branch_outputs=branch_outputs,
            route_log_probs=[trajectory.route_log_probs for trajectory in mains],
            records=[trajectory.record for trajectory in mains],
        )


def runner_paired_sampler(
    runner,
    sample,
    cluster_fn=lambda x: x,
    *,
    route_mode: str = "sample",
    seed: int = 0,
    parallel_workers: int = 4,
) -> PairedSampler:
    """Build a paired estimator backed by a real MASRunner."""
    call_index = 0

    def next_seed() -> int:
        nonlocal call_index
        value = seed + call_index
        call_index += 1
        return value

    def sample_main() -> Trajectory:
        root = next_seed()
        record, _ = runner.run_sample(
            sample,
            training=True,
            loss_fn=lambda *_: 1.0,
            route_mode=route_mode,
            generation_seed=root,
            route_seed=root,
        )
        snapshots = list(runner.last_snapshots)
        return Trajectory(
            contexts=snapshots,
            actions=[step.agent_name for step in record.steps],
            outputs=[step.raw_text for step in record.steps],
            final_output=record.normalized_prediction,
            route_log_probs=list(getattr(runner, "last_route_log_probs", [])),
            route_probabilities=[step.probabilities for step in getattr(record, "steps", [])],
            record=record,
        )

    def continue_from(snapshot, action=None, output=None):
        root = next_seed()
        final = runner.continue_from_snapshot(
            snapshot,
            forced_agent=str(action) if action is not None else None,
            forced_output=str(output) if output is not None else None,
            route_mode=route_mode,
            route_seed=root,
            generation_seed=root,
        )
        return final

    def batch_cluster(outputs):
        from ecomas.semantic_clustering import cluster_answers

        result = cluster_answers(
            runner.task_name,
            str(sample.uid),
            [{"output_id": str(index), "answer": value} for index, value in enumerate(outputs)],
            question=sample.input_text,
        )
        return [result.cluster_by_output_id[str(index)] for index in range(len(outputs))]

    return PairedSampler(
        sample_main,
        continue_from,
        int(runner.task_config["steps"]),
        cluster_fn,
        batch_cluster_fn=batch_cluster,
        parallel_workers=parallel_workers,
    )
