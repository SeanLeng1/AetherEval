import json
import tempfile
import threading
import unittest
from pathlib import Path

from aethereval.core.task_register import load_task
from aethereval.core.types import Sample
from benchmark_utils.eval_set_math import (
    MATH_PROMPT_SUFFIX,
    aggregate,
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

    def test_aime_rows_keep_the_bare_answer_and_add_the_instruction_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task_dir = Path(tmp)
            data_dir = task_dir / "data"
            data_dir.mkdir()
            row = {
                "id": "aime24_0",
                "problem": "Find x.",
                "answer": "025",
                "year": 2024,
                "source": "RLLab/eval-set",
                "subset": "aime24",
            }
            with (data_dir / "eval.jsonl").open("w", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")

            bundle = load_task("aime24")
            samples = bundle.task_module.load_samples(task_dir)

        self.assertEqual(samples[0].gold, "025")
        self.assertEqual(
            bundle.task_module.build_prompt(samples[0]), "Find x." + MATH_PROMPT_SUFFIX
        )
        result = bundle.metrics_module.score_generation(samples[0], "\\boxed{25}")
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["parsed"]["gold_extracted"], ["25", "025"])

    def test_aggregate_reports_samples_whose_gold_does_not_extract(self) -> None:
        sample_results = []
        for sample in (
            Sample(id="ok", gold="so $\\boxed{2}$", data={}),
            Sample(id="bad", gold="see figure", data={}),
        ):
            records = []
            for gen_idx, text in enumerate(("\\boxed{2}", "\\boxed{3}")):
                scored = score_generation(sample, text)
                records.append(
                    {"sample_id": sample.id, "gen_idx": gen_idx, "prompt": "p",
                     "generation": text, **scored}
                )
            sample_results.append({"sample_id": sample.id, "records": records})
        without_warnings = [
            {**item, "records": [{**r, "meta": {}} for r in item["records"]]}
            for item in sample_results
        ]

        metrics = aggregate(sample_results, {"n": 2})

        self.assertEqual(
            metrics.pop("__warnings__"),
            ["1 samples scored 0 with warning 'no gold extraction' (e.g. bad)"],
        )
        self.assertEqual(metrics, aggregate(without_warnings, {"n": 2}))
        self.assertNotIn("__warnings__", aggregate(without_warnings, {"n": 2}))

    def test_scoring_off_the_main_thread_raises_instead_of_scoring_zero(self) -> None:
        sample = Sample(id="amc23_0", gold="2", data={})
        errors: list[BaseException] = []

        def score() -> None:
            try:
                score_generation(sample, "\\boxed{2}")
            except ValueError as exc:
                errors.append(exc)

        thread = threading.Thread(target=score)
        thread.start()
        thread.join()

        self.assertEqual(len(errors), 1)
        self.assertIn("threaded", str(errors[0]))


if __name__ == "__main__":
    unittest.main()
