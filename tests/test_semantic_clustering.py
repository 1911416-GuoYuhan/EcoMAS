import unittest

from ecomas.paired_sampling import PairedSampler, Trajectory
from ecomas.semantic_clustering import cluster_answers
from ecomas.uncertainty import estimate_system_entropy, semantic_cluster_report


class SemanticClusteringTest(unittest.TestCase):
    def test_task_label_adapters(self):
        mmlu = cluster_answers("mmlu_pro", "m", ["A", "A", "B"])
        chaos = cluster_answers("chaosnli", "c", ["neutral", "entailment", "neutral"])
        self.assertEqual([0, 0, 1], [mmlu.cluster_by_output_id[f"answer-{i}"] for i in range(3)])
        self.assertEqual([0, 1, 0], [chaos.cluster_by_output_id[f"answer-{i}"] for i in range(3)])

    def test_math_bidirectional_symbolic_entailment(self):
        result = cluster_answers("math500", "q", ["1/2", "0.5", r"\frac{2}{4}", "2"])
        ids = list(result.cluster_by_output_id.values())
        self.assertEqual(ids[0], ids[1])
        self.assertEqual(ids[1], ids[2])
        self.assertNotEqual(ids[2], ids[3])
        self.assertTrue(result.entailment_matrix[0][2])
        self.assertTrue(result.entailment_matrix[2][0])

    def test_structured_math_types(self):
        result = cluster_answers("math500", "q", ["(1,2)", "(1.0,2)", "(2,1)", "{1,2}", "{2,1}"])
        ids = list(result.cluster_by_output_id.values())
        self.assertEqual(ids[0], ids[1])
        self.assertNotEqual(ids[0], ids[2])
        self.assertEqual(ids[3], ids[4])
        self.assertNotEqual(ids[0], ids[3])

    def test_invalid_is_one_auditable_failure_symbol(self):
        answers = [
            {"output_id": "a", "answer": "", "parse_valid": False},
            {"output_id": "b", "answer": "polluted", "parse_valid": False},
        ]
        result = cluster_answers("math500", "q", answers)
        self.assertEqual(1, result.same_cluster("a", "b"))
        self.assertEqual(1.0, result.invalid_rate)
        self.assertEqual(0.0, result.unresolved_rate)

    def test_report_uses_semantic_clusters(self):
        report = semantic_cluster_report("math500", "q", ["1/2", "0.5", "2"])
        self.assertEqual(2, report["observed_cluster_count"])
        self.assertAlmostEqual(4 / 9, report["second_order_tsallis"])

    def test_system_entropy_exposes_plugin_and_u_statistic(self):
        estimate = estimate_system_entropy("mmlu_pro", "q", ["A", "A", "B", "B"])
        self.assertEqual(6, estimate.pair_count)
        self.assertAlmostEqual(0.5, estimate.plugin_tsallis)
        self.assertAlmostEqual(4 / 6, estimate.u_statistic_tsallis)

    def test_paired_sampler_clusters_after_collection(self):
        trajectories = iter([
            Trajectory(["x"], ["a"], ["o"], "1/2"),
            Trajectory(["x"], ["a"], ["o"], "0.5"),
        ])

        def continuation(context, action=None, output=None):
            return "2"

        def batch_cluster(outputs):
            result = cluster_answers("math500", "q", [
                {"output_id": str(index), "answer": answer} for index, answer in enumerate(outputs)
            ])
            return [result.cluster_by_output_id[str(index)] for index in range(len(outputs))]

        estimate = PairedSampler(
            lambda: next(trajectories), continuation, 1, batch_cluster_fn=batch_cluster
        ).estimate(2)
        self.assertEqual(0.0, estimate.system_uncertainty)

    def test_paired_sampler_batch_is_reproducible_for_seeded_source(self):
        def make_sampler(seed):
            import random

            source = random.Random(seed)

            def main():
                value = "A" if source.random() < 0.5 else "B"
                return Trajectory(["context"], ["agent"], ["output"], value)

            def continuation(context, action=None, output=None):
                return "A" if source.random() < 0.5 else "B"

            return PairedSampler(main, continuation, 1)

        left = make_sampler(7).estimate(5)
        right = make_sampler(7).estimate(5)
        self.assertEqual(left.system_uncertainty, right.system_uncertainty)
        self.assertEqual(left.orchestration, right.orchestration)
        self.assertEqual(left.agent_execution, right.agent_execution)


if __name__ == "__main__":
    unittest.main()
