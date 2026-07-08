import unittest
from unittest.mock import patch

from aethereval.core.types import GenerationOutput, Sample
from benchmarks.safe_alignment import metrics
from benchmarks.safe_alignment.task import build_prompt


class SafeAlignmentTests(unittest.TestCase):
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

        def fake_score(model_path, conversations, options):  # noqa: ANN001
            self.assertEqual(len(conversations), 2)
            self.assertEqual(options["rm_batch_size"], 2)
            if model_path == "rm":
                return [1.0, 2.0]
            if model_path == "cm":
                return [0.5, -0.25]
            raise AssertionError(model_path)

        with patch(
            "benchmarks.safe_alignment.metrics._score_with_reward_model",
            side_effect=fake_score,
        ):
            results = metrics.score_generations_batch(
                samples,
                outputs,
                {
                    "rm_model_path": "rm",
                    "cm_model_path": "cm",
                    "rm_batch_size": 2,
                },
            )

        self.assertEqual(len(results), 2)
        self.assertAlmostEqual(results[0][0]["score"], 0.75)
        self.assertAlmostEqual(results[1][0]["meta"]["helpful_harmless_average"], 0.875)

    def test_batch_scoring_uses_default_rihong_models(self) -> None:
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
        seen_paths: list[str] = []

        def fake_score(model_path, conversations, options):  # noqa: ANN001
            del conversations, options
            seen_paths.append(model_path)
            return [1.0]

        with patch(
            "benchmarks.safe_alignment.metrics._score_with_reward_model",
            side_effect=fake_score,
        ):
            metrics.score_generations_batch([sample], [output], {})

        self.assertEqual(
            seen_paths,
            [
                "Rihong/Qwen2.5-7B-SafeRLHF-RM",
                "Rihong/Qwen2.5-7B-SafeRLHF-CM",
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
