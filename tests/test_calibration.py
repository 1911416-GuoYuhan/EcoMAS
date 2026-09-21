from types import SimpleNamespace

import pytest
import torch

from ecomas.agents import SpecialistAgent
from ecomas.calibration import (
    build_semantic_space,
    estimate_paired_calibration_gradient,
    gold_distribution,
    output_distribution,
    semantic_key,
)
from ecomas.datasets import BenchmarkSample
from ecomas.llm import MockLLM
from ecomas.registry import TASK_REGISTRY
from ecomas.router import ArgmaxRouter
from ecomas.runner import ContextSnapshot, MASRunner
from ecomas.training import evaluate_calibration
from ecomas.paired_sampling import PairedEstimate


def test_paired_calibration_gradient_uses_paper_two_over_b_scaling():
    main = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    branches = torch.zeros(2, 2, 3, 2)
    branches[0, :, 1] = torch.tensor([-1.0, 1.0])
    route_log_probs = torch.ones(2, 2, requires_grad=True)

    _, surrogate = estimate_paired_calibration_gradient(
        main, branches, route_log_probs, torch.tensor([1.0, 0.0])
    )
    surrogate.backward()

    assert surrogate.item() == pytest.approx(4.0)
    assert torch.allclose(route_log_probs.grad, torch.tensor([[2.0, 2.0], [0.0, 0.0]]))


def test_paired_calibration_requires_leave_one_out_batch():
    with pytest.raises(ValueError):
        estimate_paired_calibration_gradient(
            torch.tensor([[1.0, 0.0]]),
            torch.zeros(1, 1, 3, 2),
            torch.zeros(1, 1),
            torch.tensor([1.0, 0.0]),
        )


def test_semantic_space_handles_invalid_and_equivalent_math_answers():
    chaos_space = build_semantic_space("chaosnli")
    distribution = output_distribution(
        "chaosnli", ["entailment", "invalid::chaosnli"], chaos_space
    )
    assert sum(distribution) == pytest.approx(1.0)
    assert distribution[chaos_space.index("invalid::chaosnli")] == pytest.approx(0.5)
    assert semantic_key("math500", "1/2") == semantic_key("math500", "0.5")


def test_chaos_true_distribution_keeps_invalid_coordinate_empty():
    sample = BenchmarkSample(
        "c", "chaosnli", "premise", "neutral", {"label_dist": [0.2, 0.7, 0.1]}
    )
    target = gold_distribution("chaosnli", sample, build_semantic_space("chaosnli"))
    assert target == pytest.approx([0.2, 0.7, 0.1, 0.0])


def test_calibration_evaluation_averages_per_input_objectives():
    samples = [
        BenchmarkSample("a", "mmlu_pro", "q1", "A", {}),
        BenchmarkSample("b", "mmlu_pro", "q2", "B", {}),
    ]

    class FixedRunner:
        task_name = "mmlu_pro"

        def run_sample(self, sample, **kwargs):
            return SimpleNamespace(
                normalized_prediction="A",
                prediction="A",
                correct=sample.gold_answer == "A",
                output_parse_valid=True,
            ), None

    metrics = evaluate_calibration(FixedRunner(), samples, num_samples=2)
    assert metrics["per_sample_calibration_error"] == pytest.approx([0.0, 2.0])
    assert metrics["calibration_error"] == pytest.approx(1.0)
    assert metrics["accuracy"] == pytest.approx(0.5)


def test_continuation_inherits_last_valid_answer_from_context():
    task = "mmlu_pro"
    spec = TASK_REGISTRY[task]["agents"][0]
    agent = SpecialistAgent(spec, MockLLM())
    router = ArgmaxRouter(task, [spec.name], 2)

    class Encoder:
        def __call__(self, batches):
            return torch.zeros(len(batches), 2)

    runner = MASRunner(task, [agent], Encoder(), router)
    sample = BenchmarkSample("m", task, "A: yes B: no", "A", {})
    snapshot = ContextSnapshot(
        sample,
        int(TASK_REGISTRY[task]["steps"]),
        [{"role": "system", "content": sample.input_text}],
        [],
        {spec.name: []},
        "A",
    )
    result = runner.continue_from_snapshot(
        snapshot,
        forced_agent=spec.name,
        forced_output="unparseable output",
        route_mode="argmax",
    )
    assert result == "A"


def test_explanation_writes_paper_estimators(monkeypatch, tmp_path):
    from ecomas import training

    estimate = PairedEstimate(
        0.5,
        [0.25],
        [0.25],
        [[1.0, -0.5]],
        [[0.0, 0.5]],
        [1.0],
        [0.5],
        ["A", "B"],
        [[{"c": "B", "r": "A", "z": "A"}]],
        [[], []],
        [],
    )

    class Sampler:
        def estimate(self, budget):
            assert budget == 2
            return estimate

    monkeypatch.setattr(training, "runner_paired_sampler", lambda *args, **kwargs: Sampler())
    runner = SimpleNamespace(
        task_name="mmlu_pro",
        task_config={"steps": 1},
        branch_cache={},
    )
    sample = BenchmarkSample("m", "mmlu_pro", "question", "A", {})
    report_path = tmp_path / "explanation.json"
    records = training.run_explanation(
        runner,
        [sample],
        tmp_path / "results.jsonl",
        report_path,
        sampling_budget=2,
    )
    payload = __import__("json").loads(report_path.read_text(encoding="utf-8"))
    assert records == []
    assert payload["samples"][0]["system_uncertainty"] == pytest.approx(0.5)
    assert payload["samples"][0]["closure_difference"] == pytest.approx(0.0)
    assert payload["samples"][0]["branch_outputs"] == estimate.branch_outputs
