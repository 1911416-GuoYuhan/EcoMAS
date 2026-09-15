import torch

from ecomas.llm import MockLLM
from ecomas.paired_sampling import (
    BranchType,
    Trajectory,
    build_branch_requests,
    execute_branch_requests,
)
from ecomas.router import ArgmaxRouter


def test_router_batch_matches_serial_sampling():
    router = ArgmaxRouter("toy", ["a", "b", "c"], 4)
    states = torch.randn(6, 4)
    seeds = list(range(6))
    batch = router.decide_batch(states, mode="sample", seeds=seeds)
    serial = [router.decide(states[i:i + 1], mode="sample", seed=seeds[i]) for i in range(6)]
    assert [x.agent_index for x in batch] == [x.agent_index for x in serial]


def test_branch_requests_and_bounded_executor_preserve_order():
    trajectory = Trajectory(["c0", "c1"], ["a", "b"], ["o0", "o1"], "final")
    requests = build_branch_requests([trajectory], 2)
    assert [request.branch_type for request in requests] == [
        BranchType.C, BranchType.R, BranchType.Z,
        BranchType.C, BranchType.R, BranchType.Z,
    ]
    values = execute_branch_requests(
        requests,
        lambda request: f"{request.step}:{request.branch_type.value}",
        batch_fn=lambda chunk: [f"{r.step}:{r.branch_type.value}" for r in chunk],
        micro_batch_size=2,
    )
    assert values == ["0:c", "0:r", "0:z", "1:c", "1:r", "1:z"]


def test_mock_batch_generation_keeps_cardinality():
    llm = MockLLM()
    messages = [[{"role": "user", "content": f"A: x {i} B: y"}] for i in range(5)]
    outputs = llm.generate_messages_batch(messages, micro_batch_size=2, seed=10)
    assert len(outputs) == len(messages)
