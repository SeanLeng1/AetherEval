import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from aethereval.core.task_defaults import resolve_task_default_metrics
from aethereval.core.task_register import list_tasks, load_task
from aethereval.core.types import GenerationOutput, Sample
from benchmark_utils.rar import (
    aggregate_rar,
    build_grader_prompt,
    load_rar_samples,
    parse_presence_response,
    score_rar_generations_batch,
)
from benchmark_utils.rar_data import (
    FAMILY_NAMES,
    FAMILY_POINTS,
    normalize_rubrics,
    rubric_family,
    transform_eval_row,
)


def _source_row() -> dict:
    rubrics = [
        {
            "description": "Essential Criteria: Gives the correct answer.",
            "title": "Answer",
            "weight": 5,
        },
        {
            "description": "Pitfall Criteria: Does not invent facts.",
            "title": "No invention",
            "weight": -1,
        },
        {
            "description": "Optional Criteria: Is concise.",
            "title": "Concise",
            "weight": 2,
        },
    ]
    return {
        "question": "What is the answer?",
        "reference_answer": "The reference.",
        "question_source": "unit-test",
        "rubric": rubrics,
        "rubric_list": [item["description"] for item in rubrics],
        "rubric_count": len(rubrics),
    }


def _sample() -> Sample:
    row = transform_eval_row(
        _source_row(), 7, domain="Medical", dataset_id="ScaleAI/RaR-Medicine"
    )
    return Sample(
        id=str(row["id"]),
        gold=row["reference_answer"],
        data={"prompt": row["prompt"], "rubrics": row["rubrics"]},
        meta=row["meta"],
    )


class RarTests(unittest.TestCase):
    def test_tasks_use_standard_native_contract(self) -> None:
        self.assertIn("rar-medical", list_tasks())
        self.assertIn("rar-science", list_tasks())
        for name in ("rar-medical", "rar-science"):
            bundle = load_task(name)
            self.assertEqual(bundle.task_module.DATA_FILE, "data/eval.jsonl")
            self.assertEqual(bundle.metrics_module.PRIMARY_METRIC, "score")
            defaults = resolve_task_default_metrics(name)
            self.assertEqual(
                defaults["judge_model"], "google/gemma-4-26B-A4B-it"
            )
            self.assertEqual(defaults["judge_temperature"], 1.0)
            self.assertEqual(defaults["judge_max_new_tokens"], 4096)

    def test_family_weights_are_positive_and_importance_ordered(self) -> None:
        self.assertEqual(
            FAMILY_NAMES, ("Essential", "Pitfall", "Important", "Optional")
        )
        self.assertEqual(
            [FAMILY_POINTS[name] for name in FAMILY_NAMES], [1.0, 0.9, 0.7, 0.3]
        )
        rubrics = normalize_rubrics(_source_row(), domain="Medical")
        self.assertEqual(
            [rubric["points"] for rubric in rubrics], [1.0, 0.9, 0.3]
        )
        self.assertEqual(
            [rubric["family"] for rubric in rubrics],
            ["Essential", "Pitfall", "Optional"],
        )

    def test_science_category_comes_from_text_not_positive_weight(self) -> None:
        family, criterion = rubric_family(
            {
                "description": "Important Criteria: Includes the derivation.",
                "title": "Derivation",
                "weight": 5,
            },
            domain="Science",
        )
        self.assertEqual(family, "Important")
        self.assertEqual(criterion, "Includes the derivation.")

    def test_loader_materializes_a_standard_sample(self) -> None:
        row = transform_eval_row(
            _source_row(), 7, domain="Medical", dataset_id="ScaleAI/RaR-Medicine"
        )
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data"
            data_dir.mkdir()
            (data_dir / "eval.jsonl").write_text(
                json.dumps(row) + "\n", encoding="utf-8"
            )
            samples = load_rar_samples(
                Path(tmp), "data/eval.jsonl", expected_domain="Medical"
            )
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].id, "rar-medical-000007")
        self.assertEqual(samples[0].gold, "The reference.")
        self.assertEqual(samples[0].data["prompt"][0]["role"], "user")
        self.assertEqual(samples[0].meta["split"], "test")

    def test_grader_prompt_hides_importance_and_family_names(self) -> None:
        sample = _sample()
        prompt = build_grader_prompt(
            sample.data["prompt"], "Candidate.", sample.data["rubrics"]
        )
        self.assertIn("evaluate the response against EACH rubric", prompt)
        self.assertIn("1. Gives the correct answer.", prompt)
        self.assertIn("2. Does not invent facts.", prompt)
        self.assertNotIn("points:", prompt)
        self.assertNotIn("Essential Criteria", prompt)
        self.assertNotIn("Pitfall Criteria", prompt)
        self.assertIn("<Response>\nCandidate.\n</Response>", prompt)

    def test_presence_parser_is_strict_and_removes_reasoning(self) -> None:
        self.assertEqual(
            parse_presence_response(
                '<|channel>thought\nignore {"x":1}<channel|>'
                '```json\n{"1":"PRESENT","2":"NOT_PRESENT"}\n```',
                2,
            ),
            [True, False],
        )
        self.assertEqual(
            parse_presence_response('{"1":true,"2":false}', 2),
            [True, False],
        )
        with self.assertRaises(ValueError):
            parse_presence_response('{"1":"PRESENT"}', 2)

    def test_one_judge_call_scores_all_rubrics(self) -> None:
        sample = _sample()
        output = GenerationOutput(
            sample_id=sample.id,
            prompt=sample.data["prompt"],
            generations=["Candidate."],
        )
        verdict = '{"1":"PRESENT","2":"NOT_PRESENT","3":"PRESENT"}'
        with mock.patch(
            "benchmark_utils.rar.chat_completion", return_value=verdict
        ) as completion:
            results = score_rar_generations_batch(
                [sample],
                [output],
                {"judge_workers": 1},
                default_model="judge",
            )
        self.assertEqual(completion.call_count, 1)
        record = results[0][0]
        self.assertAlmostEqual(record["score"], 1.3 / 2.2)
        self.assertFalse(record["meta"]["judge_failed"])
        self.assertEqual(record["meta"]["criterion_count"], 3)

    def test_exhausted_judge_failure_zero_fills_without_raising(self) -> None:
        sample = _sample()
        output = GenerationOutput(
            sample_id=sample.id,
            prompt=sample.data["prompt"],
            generations=["Candidate."],
        )
        with mock.patch(
            "benchmark_utils.rar.chat_completion",
            side_effect=RuntimeError("judge unavailable"),
        ):
            results = score_rar_generations_batch(
                [sample],
                [output],
                {"judge_workers": 1},
                default_model="judge",
            )
        record = results[0][0]
        self.assertEqual(record["score"], 0.0)
        self.assertTrue(record["meta"]["judge_failed"])
        aggregate = aggregate_rar(
            [
                {
                    "records": [
                        {
                            "score": record["score"],
                            "meta": record["meta"],
                        }
                    ]
                }
            ]
        )
        self.assertEqual(aggregate["score"], 0.0)
        self.assertEqual(aggregate["judge_failure_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
