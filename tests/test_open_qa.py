import unittest

from aethereval.core.types import Sample
from benchmark_utils.open_qa import (
    aggregate_open_qa,
    normalize_answer,
    score_open_qa,
)


class OpenQAMetricTests(unittest.TestCase):
    def test_normalized_alias_match(self) -> None:
        sample = Sample(id="nq", gold=["The Beatles", "Beatles"])
        result = score_open_qa(sample, "the beatles.")
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(normalize_answer("The Beatles!"), "beatles")

    def test_open_qa_scores_the_full_submitted_prediction(self) -> None:
        sample = Sample(id="nq", gold=["Paris"])
        result = score_open_qa(sample, "Some reasoning.\nFinal answer: Paris")
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(score_open_qa(sample, "Paris")["score"], 1.0)

    def test_open_qa_aggregate_averages_per_sample(self) -> None:
        records = [
            {
                "sample_id": "q",
                "gen_idx": index,
                "prompt": "question",
                "generation": "answer",
                "score": score,
                "is_pass": bool(score),
                "parsed": {"token_f1": score, "prediction_normalized": "answer"},
                "meta": {},
            }
            for index, score in enumerate((1.0, 0.0))
        ]
        metrics = aggregate_open_qa([{"records": records}])
        self.assertEqual(metrics["exact_match"], 0.5)
        self.assertEqual(metrics["token_f1"], 0.5)


if __name__ == "__main__":
    unittest.main()
