import unittest
from importlib import import_module
from unittest import mock

from aethereval.core.task_defaults import resolve_task_default_metrics
from aethereval.core.types import GenerationOutput, Sample
metrics = import_module("benchmarks.safe-alignment.metrics")
build_prompt = import_module("benchmarks.safe-alignment.task").build_prompt


class _FakeBackend:
    name = "fake"

    def __init__(self, scores_by_model: dict[str, list[float]]) -> None:
        self.scores_by_model = scores_by_model
        self.calls: list[dict] = []

    def score_reward_models(self, model_paths, conversations, scorer_kwargs):
        self.calls.append(
            {
                "model_paths": list(model_paths),
                "num_conversations": len(conversations),
                "scorer_kwargs": dict(scorer_kwargs),
            }
        )
        return {path: self.scores_by_model[path] for path in model_paths}


class SafeAlignmentTests(unittest.TestCase):
    def test_reward_model_defaults_come_from_task_config(self) -> None:
        configured = resolve_task_default_metrics("safe_alignment")

        self.assertEqual(configured["rm_model_path"], metrics.DEFAULT_RM_MODEL_PATH)
        self.assertEqual(configured["cm_model_path"], metrics.DEFAULT_CM_MODEL_PATH)

    def test_eval_only_backend_uses_requested_sglang_topology(self) -> None:
        with mock.patch.object(
            metrics,
            "SGLangRewardModelBackend",
        ) as backend_cls:
            metrics.create_evaluation_backend(
                {},
                dp_size=4,
                tensor_parallel_size=2,
            )
            backend_cls.assert_called_once_with(
                dp_size=4,
                tensor_parallel_size=2,
            )

    def test_rm_parallelism_can_differ_from_generation(self) -> None:
        for options, expected in (
            ({"rm_dp_size": 8}, (8, 1)),
            ({"rm_tp_size": 2}, (1, 2)),
            ({"rm_dp_size": 4, "rm_tp_size": 2}, (4, 2)),
        ):
            with self.subTest(options=options), mock.patch.object(metrics, "SGLangRewardModelBackend") as backend_cls:
                metrics.create_evaluation_backend(options, dp_size=1, tensor_parallel_size=8)
                backend_cls.assert_called_once_with(dp_size=expected[0], tensor_parallel_size=expected[1])

    def test_batch_scoring_combines_reward_models(self) -> None:
        samples = [
            Sample(
                id="s1",
                data={
                    "prompt": [{"role": "user", "content": "hello"}],
                    "data_source": "Stanford Alpaca",
                },
            ),
            Sample(
                id="s2",
                data={
                    "prompt": [{"role": "user", "content": "stay safe"}],
                    "data_source": "PKU-Alignment/PKU-SafeRLHF",
                },
            ),
        ]
        outputs = [
            GenerationOutput(
                sample_id="s1",
                prompt=build_prompt(samples[0]),
                generations=["a"],
            ),
            GenerationOutput(
                sample_id="s2",
                prompt=build_prompt(samples[1]),
                generations=["b"],
            ),
        ]

        backend = _FakeBackend({"rm": [1.0, 2.0], "cm": [0.5, -0.25]})
        results = metrics.score_generations_batch(
            samples,
            outputs,
            {
                "rm_model_path": "rm",
                "cm_model_path": "cm",
                "_backend": backend,
            },
        )

        self.assertEqual(len(results), 2)
        self.assertAlmostEqual(results[0][0]["score"], 0.75)
        self.assertAlmostEqual(results[1][0]["meta"]["helpful_harmless_average"], 0.875)
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(backend.calls[0]["model_paths"], ["rm", "cm"])
        self.assertEqual(backend.calls[0]["num_conversations"], 2)
        self.assertNotIn("max_length", backend.calls[0]["scorer_kwargs"])

    def test_batch_scoring_uses_default_rllab_models(self) -> None:
        sample = Sample(
            id="s1",
            data={
                "prompt": [{"role": "user", "content": "hello"}],
                "data_source": "Stanford Alpaca",
            },
        )
        output = GenerationOutput(
            sample_id="s1",
            prompt=build_prompt(sample),
            generations=["a"],
        )
        backend = _FakeBackend(
            {
                "RLLab/Qwen2.5-7B-SafeRLHF-RM": [1.0],
                "RLLab/Qwen2.5-7B-SafeRLHF-CM": [1.0],
            }
        )
        metrics.score_generations_batch([sample], [output], {"_backend": backend})

        self.assertEqual(
            backend.calls[0]["model_paths"],
            [
                "RLLab/Qwen2.5-7B-SafeRLHF-RM",
                "RLLab/Qwen2.5-7B-SafeRLHF-CM",
            ],
        )

    def test_aggregate_safe_alignment_metrics(self) -> None:
        sample_results = [
            {
                "sample_id": "s1",
                "gold": None,
                "meta": {"data_source": "Stanford Alpaca"},
                "scores": [0.75],
                "passes": [True],
                "records": [
                    _record(
                        "s1",
                        helpful=1.0,
                        harmless=0.5,
                        helpful_harmless_average=0.75,
                    )
                ],
            },
            {
                "sample_id": "s2",
                "gold": None,
                "meta": {"data_source": "Stanford Alpaca"},
                "scores": [1.0],
                "passes": [True],
                "records": [
                    _record(
                        "s2",
                        helpful=1.5,
                        harmless=0.5,
                        helpful_harmless_average=1.0,
                    )
                ],
            },
            {
                "sample_id": "s3",
                "gold": None,
                "meta": {"data_source": "Anthropic/hh-rlhf"},
                "scores": [2.0],
                "passes": [True],
                "records": [
                    _record(
                        "s3",
                        helpful=3.0,
                        harmless=1.0,
                        helpful_harmless_average=2.0,
                    )
                ],
            },
            {
                "sample_id": "s4",
                "gold": None,
                "meta": {"data_source": "PKU-Alignment/PKU-SafeRLHF"},
                "scores": [3.0],
                "passes": [True],
                "records": [
                    _record(
                        "s4",
                        helpful=4.0,
                        harmless=2.0,
                        helpful_harmless_average=3.0,
                    )
                ],
            },
        ]

        result = metrics.aggregate(sample_results, {})

        self.assertAlmostEqual(result["alpaca/helpful"], 1.25)
        self.assertAlmostEqual(result["alpaca/harmless"], 0.5)
        self.assertAlmostEqual(result["alpaca/helpful_harmless_average"], 0.875)
        self.assertAlmostEqual(result["hh_rlhf/helpful_harmless_average"], 2.0)
        self.assertAlmostEqual(result["pku/helpful_harmless_average"], 3.0)
        self.assertAlmostEqual(result["overall/average"], (0.875 + 2.0 + 3.0) / 3.0)


def _record(
    sample_id: str,
    *,
    helpful: float,
    harmless: float,
    helpful_harmless_average: float,
) -> dict:
    return {
        "sample_id": sample_id,
        "gen_idx": 0,
        "prompt": [{"role": "user", "content": sample_id}],
        "generation": "answer",
        "score": helpful_harmless_average,
        "is_pass": True,
        "parsed": None,
        "gold": None,
        "error": None,
        "meta": {
            "helpful": helpful,
            "harmless": harmless,
            "helpful_harmless_average": helpful_harmless_average,
        },
    }


if __name__ == "__main__":
    unittest.main()
