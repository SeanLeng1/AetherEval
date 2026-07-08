import json
import tempfile
import unittest
from pathlib import Path

from aethereval.core.types import Sample
from benchmark_utils.eval_set_math import (
    build_eval_set_math_prompt,
    load_eval_set_math_samples,
    score_generation,
)


class EvalSetMathTests(unittest.TestCase):
    def test_full_solution_gold_is_scored_directly(self) -> None:
        sample = Sample(
            id="math500_0",
            gold="We compute the value and obtain $\\boxed{2}$.",
            data={"problem": "What is 1+1?"},
        )

        result = score_generation(sample, "The final answer is \\boxed{2}.")

        self.assertEqual(result["score"], 1.0)
        self.assertTrue(result["is_pass"])
        self.assertIn("2", result["parsed"]["gold_extracted"])

    def test_numeric_solution_gold_is_scored_directly(self) -> None:
        sample = Sample(
            id="amc23_0",
            gold="27.0",
            data={"problem": "Compute the answer."},
        )

        result = score_generation(sample, "Therefore the answer is \\boxed{27}.")

        self.assertEqual(result["score"], 1.0)
        self.assertTrue(result["is_pass"])

    def test_load_samples_preserves_source_solution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task_dir = Path(tmp)
            data_dir = task_dir / "data"
            data_dir.mkdir()
            row = {
                "id": "minervamath_0",
                "problem": "Find x.\\n\\nPlease think step by step.",
                "solution": "The answer is $\\boxed{1.6}$.",
                "source": "RLLab/eval-set",
                "subset": "minervamath",
            }
            with (data_dir / "eval.jsonl").open("w", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")

            samples = load_eval_set_math_samples(task_dir)

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].gold, row["solution"])
        self.assertEqual(samples[0].meta["source"], "RLLab/eval-set")
        self.assertEqual(build_eval_set_math_prompt(samples[0]), row["problem"])


if __name__ == "__main__":
    unittest.main()
