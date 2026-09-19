import copy
import unittest

from aethereval.core.task_register import load_task
from aethereval.core.types import Sample


class MetricsTests(unittest.TestCase):
    def _complete_sample_results(
        self,
        sample_results: list[dict],
    ) -> list[dict]:
        complete = copy.deepcopy(sample_results)
        for item in complete:
            item.setdefault("meta", {})
            for record in item["records"]:
                record.setdefault("prompt", "")
                record.setdefault("generation", "")
                record.setdefault("meta", {})
        return complete

    def _aggregate(
        self,
        metrics_module,
        sample_results: list[dict],
        metric_options: dict,
    ) -> dict:
        return metrics_module.aggregate(
            self._complete_sample_results(sample_results), metric_options
        )

    def _assert_instruction_following_micro_aggregation(self, task_name: str) -> None:
        bundle = load_task(task_name)
        metrics_module = bundle.metrics_module

        sample_results = [
            {
                "sample_id": "s1",
                "meta": {"instruction_id_list": ["i1"]},
                "records": [
                    {
                        "sample_id": "s1",
                        "gen_idx": 0,
                        "score": 1.0,
                        "is_pass": True,
                        "parsed": {
                            "prompt_level_strict_acc": 1.0,
                            "prompt_level_loose_acc": 1.0,
                            "inst_level_strict_acc": [True],
                            "inst_level_loose_acc": [True],
                        },
                    },
                ],
            },
            {
                "sample_id": "s2",
                "meta": {"instruction_id_list": ["i1", "i2", "i3"]},
                "records": [
                    {
                        "sample_id": "s2",
                        "gen_idx": 0,
                        "score": 0.0,
                        "is_pass": False,
                        "parsed": {
                            "prompt_level_strict_acc": 0.0,
                            "prompt_level_loose_acc": 0.0,
                            "inst_level_strict_acc": [False, False, False],
                            "inst_level_loose_acc": [True, False, False],
                        },
                    },
                ],
            },
        ]

        result = self._aggregate(metrics_module, sample_results, {})
        self.assertAlmostEqual(result["prompt_level_strict_acc"], 0.5, places=6)
        self.assertAlmostEqual(result["prompt_level_loose_acc"], 0.5, places=6)
        # Instruction-level should be micro-averaged over all instruction instances.
        self.assertAlmostEqual(result["inst_level_strict_acc"], 0.25, places=6)
        self.assertAlmostEqual(result["inst_level_loose_acc"], 0.5, places=6)

    def test_ifeval_score_generation(self) -> None:
        bundle = load_task("ifeval")
        metrics_module = bundle.metrics_module

        sample = Sample(
            id="s1",
            gold=None,
            meta={
                "instruction_id_list": ["punctuation:no_comma"],
                "kwargs": [{}],
            },
            data={"prompt": "Respond briefly."},
        )

        result = metrics_module.score_generation(sample, "This answer has no comma")
        self.assertIn("score", result)
        self.assertIn("parsed", result)
        self.assertEqual(result["score"], 1.0)

    def test_ifeval_aggregate_task_defined_metrics(self) -> None:
        bundle = load_task("ifeval")
        metrics_module = bundle.metrics_module

        sample_results = [
            {
                "sample_id": "q1",
                "records": [
                    {
                        "sample_id": "q1",
                        "gen_idx": 0,
                        "score": 1.0,
                        "is_pass": True,
                        "parsed": {
                            "prompt_level_strict_acc": 1.0,
                            "prompt_level_loose_acc": 1.0,
                            "inst_level_strict_acc": [True, False],
                            "inst_level_loose_acc": [True, True],
                        },
                    },
                    {
                        "sample_id": "q1",
                        "gen_idx": 1,
                        "score": 0.0,
                        "is_pass": False,
                        "parsed": {
                            "prompt_level_strict_acc": 0.0,
                            "prompt_level_loose_acc": 0.0,
                            "inst_level_strict_acc": [False, False],
                            "inst_level_loose_acc": [False, False],
                        },
                    },
                ],
            },
            {
                "sample_id": "q2",
                "records": [
                    {
                        "sample_id": "q2",
                        "gen_idx": 0,
                        "score": 0.0,
                        "is_pass": False,
                        "parsed": {
                            "prompt_level_strict_acc": 0.0,
                            "prompt_level_loose_acc": 1.0,
                            "inst_level_strict_acc": [False, False],
                            "inst_level_loose_acc": [True, False],
                        },
                    },
                    {
                        "sample_id": "q2",
                        "gen_idx": 1,
                        "score": 1.0,
                        "is_pass": True,
                        "parsed": {
                            "prompt_level_strict_acc": 1.0,
                            "prompt_level_loose_acc": 1.0,
                            "inst_level_strict_acc": [True, True],
                            "inst_level_loose_acc": [True, True],
                        },
                    },
                ],
            },
        ]

        result = self._aggregate(metrics_module, sample_results, {})

        self.assertAlmostEqual(result["prompt_level_strict_acc"], 0.5, places=6)
        self.assertAlmostEqual(result["inst_level_strict_acc"], 0.375, places=6)
        self.assertAlmostEqual(result["prompt_level_loose_acc"], 0.75, places=6)
        self.assertAlmostEqual(result["inst_level_loose_acc"], 0.625, places=6)

        self.assertIn("prompt_level_strict_acc_stderr", result)
        self.assertIn("inst_level_strict_acc_stderr", result)
        self.assertIn("prompt_level_loose_acc_stderr", result)
        self.assertIn("inst_level_loose_acc_stderr", result)
        self.assertNotIn("pass@1", result)
        self.assertNotIn("mean@1", result)
        self.assertNotIn("prompt_level_strict_acc_ci_low", result)
        self.assertNotIn("prompt_level_strict_acc_ci_high", result)

    def test_ifbench_score_generation(self) -> None:
        bundle = load_task("ifbench")
        metrics_module = bundle.metrics_module

        sample = Sample(
            id="s1",
            gold=None,
            meta={
                "instruction_id_list": ["sentence:keyword"],
                "kwargs": [{"word": "hello", "N": 1}],
            },
            data={"prompt": "Write one sentence."},
        )

        result = metrics_module.score_generation(sample, "hello world.")
        self.assertIn("score", result)
        self.assertIn("parsed", result)
        self.assertEqual(result["score"], 1.0)

    def test_ifbench_aggregate_task_defined_metrics(self) -> None:
        bundle = load_task("ifbench")
        metrics_module = bundle.metrics_module

        sample_results = [
            {
                "sample_id": "q1",
                "records": [
                    {
                        "sample_id": "q1",
                        "gen_idx": 0,
                        "score": 1.0,
                        "is_pass": True,
                        "parsed": {
                            "prompt_level_strict_acc": 1.0,
                            "prompt_level_loose_acc": 1.0,
                            "inst_level_strict_acc": [True, False],
                            "inst_level_loose_acc": [True, True],
                        },
                    },
                    {
                        "sample_id": "q1",
                        "gen_idx": 1,
                        "score": 0.0,
                        "is_pass": False,
                        "parsed": {
                            "prompt_level_strict_acc": 0.0,
                            "prompt_level_loose_acc": 0.0,
                            "inst_level_strict_acc": [False, False],
                            "inst_level_loose_acc": [False, False],
                        },
                    },
                ],
            },
            {
                "sample_id": "q2",
                "records": [
                    {
                        "sample_id": "q2",
                        "gen_idx": 0,
                        "score": 0.0,
                        "is_pass": False,
                        "parsed": {
                            "prompt_level_strict_acc": 0.0,
                            "prompt_level_loose_acc": 1.0,
                            "inst_level_strict_acc": [False, False],
                            "inst_level_loose_acc": [True, False],
                        },
                    },
                    {
                        "sample_id": "q2",
                        "gen_idx": 1,
                        "score": 1.0,
                        "is_pass": True,
                        "parsed": {
                            "prompt_level_strict_acc": 1.0,
                            "prompt_level_loose_acc": 1.0,
                            "inst_level_strict_acc": [True, True],
                            "inst_level_loose_acc": [True, True],
                        },
                    },
                ],
            },
        ]

        result = self._aggregate(metrics_module, sample_results, {})
        self.assertAlmostEqual(result["prompt_level_strict_acc"], 0.5, places=6)
        self.assertAlmostEqual(result["inst_level_strict_acc"], 0.375, places=6)
        self.assertAlmostEqual(result["prompt_level_loose_acc"], 0.75, places=6)
        self.assertAlmostEqual(result["inst_level_loose_acc"], 0.625, places=6)

    def test_ifeval_instruction_level_metrics_use_micro_averaging(self) -> None:
        self._assert_instruction_following_micro_aggregation("ifeval")

    def test_ifbench_instruction_level_metrics_use_micro_averaging(self) -> None:
        self._assert_instruction_following_micro_aggregation("ifbench")

    def test_gpqa_score_generation_parsing(self) -> None:
        bundle = load_task("gpqa_diamond")
        metrics_module = bundle.metrics_module

        sample = Sample(
            id="g1",
            gold="C",
            meta={"domain": "Physics"},
            data={
                "question": "Dummy question",
                "choices": {
                    "A": "alpha option",
                    "B": "beta option",
                    "C": "gamma option",
                    "D": "delta option",
                },
            },
        )

        result1 = metrics_module.score_generation(sample, "Final answer: (C).")
        self.assertEqual(result1["score"], 1.0)
        self.assertTrue(result1["is_pass"])
        self.assertEqual(result1["parsed"]["prediction"], "C")

        result2 = metrics_module.score_generation(
            sample, "I think the correct option is B."
        )
        self.assertEqual(result2["score"], 0.0)
        self.assertFalse(result2["is_pass"])
        self.assertEqual(result2["parsed"]["prediction"], "B")

        # Do not parse letters embedded in normal words.
        result3 = metrics_module.score_generation(
            sample, "Answer: because of uncertainty."
        )
        self.assertEqual(result3["score"], 0.0)
        self.assertIsNone(result3["parsed"]["prediction"])

        # Do not fallback to option-text matching; only explicit choice letters count.
        result4 = metrics_module.score_generation(sample, "My answer is gamma option.")
        self.assertEqual(result4["score"], 0.0)
        self.assertIsNone(result4["parsed"]["prediction"])

    def test_gpqa_score_generation_parsing_long_output_window(self) -> None:
        bundle = load_task("gpqa_diamond")
        metrics_module = bundle.metrics_module

        sample = Sample(
            id="g2",
            gold="C",
            meta={"domain": "Physics"},
            data={
                "question": "Dummy question",
                "choices": {
                    "A": "alpha option",
                    "B": "beta option",
                    "C": "gamma option",
                    "D": "delta option",
                },
            },
        )

        long_reasoning = "because " * 20000
        result_tail = metrics_module.score_generation(
            sample,
            f"{long_reasoning}\nFinal answer: (C).",
        )
        self.assertEqual(result_tail["score"], 1.0)
        self.assertEqual(result_tail["parsed"]["prediction"], "C")

        result_head = metrics_module.score_generation(
            sample,
            f"Answer: C\n{long_reasoning}",
        )
        self.assertEqual(result_head["score"], 0.0)
        self.assertIsNone(result_head["parsed"]["prediction"])

    def test_gpqa_aggregate_task_defined_metrics(self) -> None:
        bundle = load_task("gpqa_diamond")
        metrics_module = bundle.metrics_module

        sample_results = [
            {
                "sample_id": "q1",
                "meta": {"domain": "Physics"},
                "records": [
                    {
                        "sample_id": "q1",
                        "gen_idx": 0,
                        "score": 1.0,
                        "is_pass": True,
                        "parsed": {"prediction": "A"},
                    },
                    {
                        "sample_id": "q1",
                        "gen_idx": 1,
                        "score": 0.0,
                        "is_pass": False,
                        "parsed": {"prediction": "B"},
                    },
                ],
            },
            {
                "sample_id": "q2",
                "meta": {"domain": "Chemistry"},
                "records": [
                    {
                        "sample_id": "q2",
                        "gen_idx": 0,
                        "score": 0.0,
                        "is_pass": False,
                        "parsed": {"prediction": None},
                    },
                    {
                        "sample_id": "q2",
                        "gen_idx": 1,
                        "score": 0.0,
                        "is_pass": False,
                        "parsed": {"prediction": None},
                    },
                ],
            },
        ]

        result = self._aggregate(metrics_module, sample_results, {"n": 2})
        self.assertAlmostEqual(result["accuracy"], 0.25, places=6)
        self.assertIn("accuracy_stderr", result)
        self.assertAlmostEqual(result["accuracy@2"], 0.25, places=6)
        self.assertAlmostEqual(result["parsed_rate"], 0.5, places=6)
        self.assertAlmostEqual(result["pass@1"], 0.25, places=6)
        self.assertAlmostEqual(result["pass@2"], 0.5, places=6)
        self.assertAlmostEqual(result["accuracy_physics"], 0.5, places=6)
        self.assertAlmostEqual(result["accuracy_chemistry"], 0.0, places=6)

    def test_aime_score_generation_math_verify(self) -> None:
        bundle = load_task("aime24")
        metrics_module = bundle.metrics_module

        sample = Sample(
            id="a1",
            gold="204",
            meta={},
            data={"problem": "Dummy"},
        )

        result = metrics_module.score_generation(
            sample,
            "Therefore, the final answer is: \\boxed{204}. I hope it is correct",
        )
        self.assertEqual(result["score"], 1.0)
        self.assertTrue(result["is_pass"])
        self.assertIn("prediction_extracted", result["parsed"])

    def test_aime_prompt_template_render(self) -> None:
        bundle = load_task("aime24")
        sample = Sample(
            id="a1",
            gold="42",
            meta={},
            data={"problem": "What is 6*7?"},
        )
        prompt = bundle.task_module.build_prompt(sample)
        self.assertIn("\\boxed{}", prompt)
        self.assertIn("What is 6*7?", prompt)

    def test_aime_aggregate_pass_at_k(self) -> None:
        bundle = load_task("aime24")
        metrics_module = bundle.metrics_module

        sample_results = [
            {
                "sample_id": "q1",
                "records": [
                    {
                        "sample_id": "q1",
                        "gen_idx": 0,
                        "score": 1.0,
                        "is_pass": True,
                        "parsed": {},
                    },
                    {
                        "sample_id": "q1",
                        "gen_idx": 1,
                        "score": 0.0,
                        "is_pass": False,
                        "parsed": {},
                    },
                    {
                        "sample_id": "q1",
                        "gen_idx": 2,
                        "score": 0.0,
                        "is_pass": False,
                        "parsed": {},
                    },
                    {
                        "sample_id": "q1",
                        "gen_idx": 3,
                        "score": 1.0,
                        "is_pass": True,
                        "parsed": {},
                    },
                ],
            },
            {
                "sample_id": "q2",
                "records": [
                    {
                        "sample_id": "q2",
                        "gen_idx": 0,
                        "score": 0.0,
                        "is_pass": False,
                        "parsed": {},
                    },
                    {
                        "sample_id": "q2",
                        "gen_idx": 1,
                        "score": 0.0,
                        "is_pass": False,
                        "parsed": {},
                    },
                    {
                        "sample_id": "q2",
                        "gen_idx": 2,
                        "score": 1.0,
                        "is_pass": True,
                        "parsed": {},
                    },
                    {
                        "sample_id": "q2",
                        "gen_idx": 3,
                        "score": 0.0,
                        "is_pass": False,
                        "parsed": {},
                    },
                ],
            },
        ]

        result = self._aggregate(metrics_module, sample_results, {"n": 4})
        self.assertAlmostEqual(result["accuracy"], 0.375, places=6)
        self.assertIn("accuracy_stderr", result)
        self.assertAlmostEqual(result["accuracy@4"], 0.375, places=6)
        self.assertAlmostEqual(result["pass@1"], 0.375, places=6)
        self.assertAlmostEqual(result["pass@2"], 0.6666666666666667, places=6)
        self.assertAlmostEqual(result["pass@4"], 1.0, places=6)

    def test_aime_default_pass_k_schedule_hits_n(self) -> None:
        bundle = load_task("aime24")
        metrics_module = bundle.metrics_module

        sample_results = [
            {
                "sample_id": "q1",
                "records": [
                    {
                        "sample_id": "q1",
                        "gen_idx": 0,
                        "score": 0.0,
                        "is_pass": False,
                        "parsed": {},
                    },
                    {
                        "sample_id": "q1",
                        "gen_idx": 1,
                        "score": 0.0,
                        "is_pass": False,
                        "parsed": {},
                    },
                    {
                        "sample_id": "q1",
                        "gen_idx": 2,
                        "score": 0.0,
                        "is_pass": False,
                        "parsed": {},
                    },
                    {
                        "sample_id": "q1",
                        "gen_idx": 3,
                        "score": 0.0,
                        "is_pass": False,
                        "parsed": {},
                    },
                    {
                        "sample_id": "q1",
                        "gen_idx": 4,
                        "score": 1.0,
                        "is_pass": True,
                        "parsed": {},
                    },
                ],
            },
        ]

        result = self._aggregate(metrics_module, sample_results, {"n": 5})
        self.assertIn("pass@1", result)
        self.assertIn("pass@2", result)
        self.assertIn("pass@4", result)
        self.assertIn("pass@5", result)

    def test_mmlu_pro_metrics(self) -> None:
        bundle = load_task("mmlu_pro")
        metrics_module = bundle.metrics_module

        sample = Sample(
            id="m1",
            gold="I",
            meta={"category": "business"},
            data={
                "question": "Dummy",
                "choices": {
                    "A": "a",
                    "B": "b",
                    "C": "c",
                    "D": "d",
                    "E": "e",
                    "F": "f",
                    "G": "g",
                    "H": "h",
                    "I": "i",
                },
            },
        )

        scored = metrics_module.score_generation(sample, "Answer: I")
        self.assertEqual(scored["score"], 1.0)
        self.assertEqual(scored["parsed"]["prediction"], "I")

        sample_results = [
            {
                "sample_id": "m1",
                "meta": {"category": "business"},
                "records": [
                    {
                        "sample_id": "m1",
                        "gen_idx": 0,
                        "score": 1.0,
                        "is_pass": True,
                        "parsed": {"prediction": "I"},
                    },
                    {
                        "sample_id": "m1",
                        "gen_idx": 1,
                        "score": 0.0,
                        "is_pass": False,
                        "parsed": {"prediction": "A"},
                    },
                ],
            },
            {
                "sample_id": "m2",
                "meta": {"category": "law"},
                "records": [
                    {
                        "sample_id": "m2",
                        "gen_idx": 0,
                        "score": 0.0,
                        "is_pass": False,
                        "parsed": {"prediction": "B"},
                    },
                    {
                        "sample_id": "m2",
                        "gen_idx": 1,
                        "score": 0.0,
                        "is_pass": False,
                        "parsed": {"prediction": "C"},
                    },
                ],
            },
        ]
        result = self._aggregate(metrics_module, sample_results, {"n": 2})
        self.assertAlmostEqual(result["accuracy"], 0.25, places=6)
        self.assertAlmostEqual(result["accuracy@2"], 0.25, places=6)
        self.assertAlmostEqual(result["pass@1"], 0.25, places=6)
        self.assertAlmostEqual(result["pass@2"], 0.5, places=6)
        self.assertAlmostEqual(result["accuracy_business"], 0.5, places=6)
        self.assertAlmostEqual(result["accuracy_law"], 0.0, places=6)

    def test_agieval_en_metrics(self) -> None:
        bundle = load_task("agieval_en")
        metrics_module = bundle.metrics_module

        sample = Sample(
            id="a1",
            gold="D",
            meta={"subset": "sat-en"},
            data={
                "question": "Dummy",
                "choices": {
                    "A": "opt a",
                    "B": "opt b",
                    "C": "opt c",
                    "D": "opt d",
                },
            },
        )
        scored = metrics_module.score_generation(sample, "The answer is (D).")
        self.assertEqual(scored["score"], 1.0)
        self.assertEqual(scored["parsed"]["prediction"], "D")

        sample_results = [
            {
                "sample_id": "a1",
                "meta": {"subset": "sat-en"},
                "records": [
                    {
                        "sample_id": "a1",
                        "gen_idx": 0,
                        "score": 1.0,
                        "is_pass": True,
                        "parsed": {"prediction": "D"},
                    },
                ],
            },
            {
                "sample_id": "a2",
                "meta": {"subset": "logiqa-en"},
                "records": [
                    {
                        "sample_id": "a2",
                        "gen_idx": 0,
                        "score": 0.0,
                        "is_pass": False,
                        "parsed": {"prediction": "A"},
                    },
                ],
            },
        ]
        result = self._aggregate(metrics_module, sample_results, {"n": 1})
        self.assertAlmostEqual(result["accuracy"], 0.5, places=6)
        self.assertAlmostEqual(result["pass@1"], 0.5, places=6)
        self.assertAlmostEqual(result["accuracy_sat_en"], 1.0, places=6)
        self.assertAlmostEqual(result["accuracy_logiqa_en"], 0.0, places=6)

        # Primary metric follows OLMES: macro average over subsets.
        self.assertEqual(metrics_module.PRIMARY_METRIC, "macro_accuracy")
        sample_results.append({**sample_results[1], "sample_id": "a3"})
        result = self._aggregate(metrics_module, sample_results, {"n": 1})
        self.assertAlmostEqual(result["accuracy"], 1 / 3, places=6)
        self.assertAlmostEqual(result["macro_accuracy"], 0.5, places=6)

    def test_agieval_score_generation_parsing_long_output_window(self) -> None:
        bundle = load_task("agieval_en")
        metrics_module = bundle.metrics_module

        sample = Sample(
            id="a2",
            gold="D",
            meta={"subset": "sat-en"},
            data={
                "question": "Dummy",
                "choices": {
                    "A": "opt a",
                    "B": "opt b",
                    "C": "opt c",
                    "D": "opt d",
                },
            },
        )

        long_reasoning = "because " * 20000
        result_tail = metrics_module.score_generation(
            sample,
            f"{long_reasoning}\nTherefore, the answer is (D).",
        )
        self.assertEqual(result_tail["score"], 1.0)
        self.assertEqual(result_tail["parsed"]["prediction"], "D")

        result_head = metrics_module.score_generation(
            sample,
            f"Therefore, the answer is (D).\n{long_reasoning}",
        )
        self.assertEqual(result_head["score"], 0.0)
        self.assertIsNone(result_head["parsed"]["prediction"])

    def test_bbh_metrics(self) -> None:
        bundle = load_task("bbh")
        metrics_module = bundle.metrics_module

        sample = Sample(
            id="bbh_1",
            gold="(B)",
            meta={"subset": "date_understanding"},
            data={
                "subset": "date_understanding",
                "input": "If today is Monday, what day comes after Tuesday?",
                "target": "Let's think step by step. So the answer is (B).",
                "answer": "(B)",
                "description": "Infer the date from context.",
            },
        )

        prompt = bundle.task_module.build_prompt(sample)
        self.assertIn("Question:", prompt)
        self.assertIn("Answer: Let's think step by step.", prompt)

        scored = metrics_module.score_generation(
            sample,
            "We reason it out. So the answer is B.",
        )
        self.assertEqual(scored["score"], 1.0)
        self.assertEqual(scored["parsed"]["prediction"], "(B)")

        # Malformed MC gold labels must not become gold-conditioned search patterns.
        sample_free_form = Sample(
            id="bbh_2",
            gold="dearth, wind, & fire",
            meta={"subset": "ruin_names"},
            data={
                "subset": "ruin_names",
                "input": "Dummy",
                "target": "dearth, wind, & fire",
                "answer": "dearth, wind, & fire",
                "description": "Dummy",
            },
        )
        scored_free_form = metrics_module.score_generation(
            sample_free_form,
            "dearth, wind, & fire",
        )
        self.assertEqual(scored_free_form["score"], 0.0)
        denied = metrics_module.score_generation(
            sample_free_form,
            "dearth, wind, & fire is not my answer. The answer is (A).",
        )
        self.assertEqual(denied["score"], 0.0)
        self.assertEqual(denied["parsed"]["prediction"], "(A)")

        sample_results = [
            {
                "sample_id": "bbh_1",
                "meta": {"subset": "date_understanding"},
                "records": [
                    {
                        "sample_id": "bbh_1",
                        "gen_idx": 0,
                        "score": 1.0,
                        "is_pass": True,
                        "parsed": {"prediction_norm": "b"},
                    },
                    {
                        "sample_id": "bbh_1",
                        "gen_idx": 1,
                        "score": 0.0,
                        "is_pass": False,
                        "parsed": {"prediction_norm": ""},
                    },
                ],
            },
            {
                "sample_id": "bbh_2",
                "meta": {"subset": "boolean_expressions"},
                "records": [
                    {
                        "sample_id": "bbh_2",
                        "gen_idx": 0,
                        "score": 0.0,
                        "is_pass": False,
                        "parsed": {"prediction_norm": "false"},
                    },
                    {
                        "sample_id": "bbh_2",
                        "gen_idx": 1,
                        "score": 0.0,
                        "is_pass": False,
                        "parsed": {"prediction_norm": ""},
                    },
                ],
            },
        ]
        result = self._aggregate(metrics_module, sample_results, {"n": 2})
        self.assertAlmostEqual(result["exact_match"], 0.25, places=6)
        self.assertAlmostEqual(result["accuracy"], 0.25, places=6)
        self.assertAlmostEqual(result["pass@1"], 0.25, places=6)
        self.assertAlmostEqual(result["pass@2"], 0.5, places=6)
        self.assertAlmostEqual(result["exact_match_date_understanding"], 0.5, places=6)
        self.assertAlmostEqual(result["exact_match_boolean_expressions"], 0.0, places=6)

    def test_zebralogic_metrics(self) -> None:
        bundle = load_task("zebralogic")
        metrics_module = bundle.metrics_module

        sample = Sample(
            id="z1",
            gold={"House 1": {"Name": "Alice"}},
            meta={},
            data={"total_cells": 1},
        )
        scored = metrics_module.score_generation(
            sample,
            ('{"reasoning":"dummy","solution":{"House 1":{"Name":"Alice"}}}'),
        )
        self.assertEqual(scored["score"], 1.0)
        self.assertAlmostEqual(scored["parsed"]["cell_accuracy"], 1.0, places=6)
        self.assertEqual(scored["parsed"]["correct_cells"], 1)

        sample_results = [
            {
                "sample_id": "z1",
                "records": [
                    {
                        "sample_id": "z1",
                        "gen_idx": 0,
                        "score": 1.0,
                        "is_pass": True,
                        "parsed": {
                            "parsed": 1.0,
                            "cell_accuracy": 1.0,
                            "correct_cells": 4,
                            "total_cells": 4,
                        },
                    },
                    {
                        "sample_id": "z1",
                        "gen_idx": 1,
                        "score": 0.0,
                        "is_pass": False,
                        "parsed": {"parsed": 1.0, "cell_accuracy": 0.0},
                    },
                ],
            },
            {
                "sample_id": "z2",
                "records": [
                    {
                        "sample_id": "z2",
                        "gen_idx": 0,
                        "score": 0.0,
                        "is_pass": False,
                        "parsed": {
                            "parsed": 1.0,
                            "cell_accuracy": 0.0,
                            "correct_cells": 0,
                            "total_cells": 12,
                        },
                    },
                    {
                        "sample_id": "z2",
                        "gen_idx": 1,
                        "score": 0.0,
                        "is_pass": False,
                        "parsed": {"parsed": 1.0, "cell_accuracy": 0.0},
                    },
                ],
            },
        ]
        result = self._aggregate(metrics_module, sample_results, {"n": 2})
        self.assertAlmostEqual(result["puzzle_accuracy"], 0.5, places=6)
        # ZeroEval micro-averages cells over puzzles: 4 of 4 + 12 cells.
        self.assertAlmostEqual(result["cell_accuracy"], 0.25, places=6)
        self.assertAlmostEqual(result["parsed"], 1.0, places=6)

    def test_zebralogic_score_generation_json_with_braces_in_string(self) -> None:
        bundle = load_task("zebralogic")
        metrics_module = bundle.metrics_module

        sample = Sample(
            id="z_braces",
            gold={"House 1": {"Name": "Alice"}},
            meta={},
            data={"total_cells": 1},
        )
        generation = (
            "Some notes before the answer.\n"
            '{"reasoning":"Use clue {A} then {B}.","solution":{"House 1":{"Name":"Alice"}}}\n'
        )

        scored = metrics_module.score_generation(sample, generation)
        self.assertEqual(scored["score"], 1.0)
        self.assertAlmostEqual(scored["parsed"]["cell_accuracy"], 1.0, places=6)

    def test_livecodebench_score_generation(self) -> None:
        bundle = load_task("livecodebench")
        metrics_module = bundle.metrics_module

        sample = Sample(
            id="lcb_demo",
            gold=None,
            meta={"platform": "atcoder"},
            data={
                "question_content": "Print 42.",
                "starter_code": "",
                "fn_name": None,
                "inputs": [""],
                "outputs": ["42\n"],
                "timeout_sec": 6,
            },
        )
        scored = metrics_module.score_generation(
            sample,
            "```python\nprint(42)\n```",
        )
        self.assertEqual(scored["score"], 1.0)
        self.assertTrue(scored["is_pass"])
        self.assertEqual(scored["parsed"]["passed_tests"], 1)

    def test_livecodebench_aggregate(self) -> None:
        bundle = load_task("livecodebench")
        metrics_module = bundle.metrics_module

        sample_results = [
            {
                "sample_id": "l1",
                "meta": {"platform": "atcoder"},
                "records": [
                    {
                        "sample_id": "l1",
                        "gen_idx": 0,
                        "score": 1.0,
                        "is_pass": True,
                        "parsed": {"had_code": True},
                    },
                    {
                        "sample_id": "l1",
                        "gen_idx": 1,
                        "score": 0.0,
                        "is_pass": False,
                        "parsed": {"had_code": True},
                    },
                ],
            },
            {
                "sample_id": "l2",
                "meta": {"platform": "leetcode"},
                "records": [
                    {
                        "sample_id": "l2",
                        "gen_idx": 0,
                        "score": 0.0,
                        "is_pass": False,
                        "parsed": {"had_code": False},
                    },
                    {
                        "sample_id": "l2",
                        "gen_idx": 1,
                        "score": 0.0,
                        "is_pass": False,
                        "parsed": {"had_code": True},
                    },
                ],
            },
        ]

        result = self._aggregate(metrics_module, sample_results, {"n": 2})
        self.assertAlmostEqual(result["accuracy"], 0.25, places=6)
        self.assertAlmostEqual(result["accuracy@2"], 0.25, places=6)
        self.assertAlmostEqual(result["pass@1"], 0.25, places=6)
        self.assertAlmostEqual(result["pass@2"], 0.5, places=6)
        self.assertAlmostEqual(result["parsed_rate"], 0.75, places=6)
        self.assertAlmostEqual(result["accuracy_atcoder"], 0.5, places=6)
        self.assertAlmostEqual(result["accuracy_leetcode"], 0.0, places=6)

    def test_livecodebench_memory_limit(self) -> None:
        import resource

        from benchmarks.livecodebench.lcb_eval_runtime import MAXIMUM_MEMORY_BYTES

        metrics = load_task("livecodebench").metrics_module
        parent_limit = resource.getrlimit(resource.RLIMIT_AS)
        allocation = f"bytearray({2 * MAXIMUM_MEMORY_BYTES})"
        for fn_name, code in (
            (None, allocation),
            ("solve", f"def solve(x):\n    return {allocation}"),
            ("solve", f"payload = {allocation}\ndef solve(x):\n    return 42"),
        ):
            with self.subTest(fn_name=fn_name, code=code):
                sample = Sample(id="memory", data={
                    "fn_name": fn_name, "inputs": ["0"], "outputs": ["42"], "timeout_sec": 2,
                })
                scored = metrics.score_generation(sample, f"```python\n{code}\n```")
                self.assertEqual(scored["score"], 0.0)
                self.assertIn("Memory Limit Exceeded", scored["meta"]["runtime_error"])
        self.assertEqual(resource.getrlimit(resource.RLIMIT_AS), parent_limit)
        # An oversized candidate must not poison the next execution.
        sample = Sample(id="normal", data={"inputs": [""], "outputs": ["42"]})
        self.assertEqual(metrics.score_generation(sample, "```python\nprint(42)\n```")["score"], 1.0)

    def test_livecodebench_score_generation_requires_fenced_code(self) -> None:
        bundle = load_task("livecodebench")
        metrics_module = bundle.metrics_module

        sample = Sample(
            id="lcb_fenced_only",
            gold=None,
            meta={"platform": "atcoder"},
            data={
                "question_content": "Print 42.",
                "starter_code": "",
                "fn_name": None,
                "inputs": [""],
                "outputs": ["42\n"],
                "timeout_sec": 6,
            },
        )
        scored = metrics_module.score_generation(sample, "print(42)")
        self.assertEqual(scored["score"], 0.0)
        self.assertFalse(scored["is_pass"])
        self.assertFalse(scored["parsed"]["had_code"])
        self.assertEqual(scored["parsed"]["extract_method"], "no_fenced_code")

    def test_livecodebench_prompt_template_alignment(self) -> None:
        bundle = load_task("livecodebench")

        sample_no_starter = Sample(
            id="lcb_prompt_std",
            gold=None,
            meta={"platform": "atcoder"},
            data={
                "question_content": "Given n, print n.",
                "starter_code": "",
                "fn_name": None,
                "inputs": ["1"],
                "outputs": ["1"],
            },
        )
        prompt_no_starter = bundle.task_module.build_prompt(sample_no_starter)
        self.assertEqual(prompt_no_starter[0]["role"], "system")
        self.assertIn(
            "question (problem specification) and will generate a correct Python program",
            prompt_no_starter[0]["content"],
        )
        self.assertIn("### Question:", prompt_no_starter[1]["content"])
        self.assertIn("### Format:", prompt_no_starter[1]["content"])
        self.assertIn(
            "Provide CONCISE reasoning on how to arrive at the answer.",
            prompt_no_starter[1]["content"],
        )
        self.assertIn(
            "Read the inputs from stdin solve the problem",
            prompt_no_starter[1]["content"],
        )
        self.assertIn(
            "### Answer: (use the provided format with backticks)",
            prompt_no_starter[1]["content"],
        )

        sample_with_starter = Sample(
            id="lcb_prompt_func",
            gold=None,
            meta={"platform": "leetcode"},
            data={
                "question_content": "Implement add.",
                "starter_code": "class Solution:\n    def add(self, a, b):\n        pass",
                "fn_name": "add",
                "inputs": ["1\n2"],
                "outputs": ["3"],
            },
        )
        prompt_with_starter = bundle.task_module.build_prompt(sample_with_starter)
        self.assertIn(
            "You will use the following starter code to write the solution to the problem",
            prompt_with_starter[1]["content"],
        )
        self.assertIn(
            "Provide CONCISE reasoning on how to arrive at the answer.",
            prompt_with_starter[1]["content"],
        )
        self.assertIn("class Solution:", prompt_with_starter[1]["content"])

    def test_mbpp_plus_offline_scoring(self) -> None:
        import json
        from unittest.mock import patch

        bundle = load_task("mbpp-plus")
        with patch("urllib.request.urlopen", side_effect=AssertionError("offline")):
            samples = bundle.task_module.load_samples(bundle.spec.task_dir)
            self.assertEqual(len(samples), 378)
            json.dumps([sample.data for sample in samples])
            # Real release inputs: tuples, complex numbers, and a nonserializable
            # reference output (regex match) handled by the official special oracle.
            selected = [s for s in samples if s.id in ("Mbpp/2", "Mbpp/124", "Mbpp/252", "Mbpp/793")
                        or s.data["entry_point"] == "check_str"]
            self.assertEqual(len(selected), 5)
            for sample in selected:
                with self.subTest(task=sample.id):
                    result = bundle.metrics_module.score_generation(
                        sample, "```python\n" + sample.data["canonical_solution"] + "\n```"
                    )
                    self.assertTrue(result["is_pass"])
            prompt = bundle.task_module.build_prompt(samples[0])
            humaneval = load_task("humaneval-plus")
            expected = humaneval.task_module.build_prompt(samples[0])
            self.assertEqual(prompt, expected)
            self.assertIn("Provide a SHORT reasoning", prompt[1]["content"])
            self.assertEqual([message["role"] for message in prompt], ["system", "user"])

        sample = Sample(id="Mbpp/99999", data={
            "prompt": "", "entry_point": "f", "canonical_solution": "def f(x): return x",
            "base_input": [[1]], "plus_input": [[2]], "atol": 0,
        })
        for trailing_block in ("```text\n2\n```", "```python\nassert f(1) == 1\n```"):
            generation = (
                "```python\ndef f(x):\n    return x\n```\nExample:\n" + trailing_block
            )
            self.assertTrue(bundle.metrics_module.score_generation(sample, generation)["is_pass"])
        scored = bundle.metrics_module.score_generation(sample, "def f(x): return 1")
        self.assertTrue(scored["parsed"]["base_pass"])
        self.assertFalse(scored["parsed"]["plus_pass"])
        self.assertFalse(scored["is_pass"])
        summary = self._aggregate(bundle.metrics_module, [{
            "sample_id": sample.id,
            "records": [{"sample_id": sample.id, "gen_idx": 0, **scored}],
        }], {})
        self.assertEqual(summary["accuracy_base"], 1.0)
        self.assertEqual(summary["pass@1"], 0.0)

    def test_humaneval_plus_score_generation(self) -> None:
        bundle = load_task("humaneval_plus")
        metrics_module = bundle.metrics_module

        sample = Sample(
            id="HumanEval/test_add",
            gold=None,
            meta={"entry_point": "add"},
            data={
                "task_id": "HumanEval/test_add",
                "prompt": 'def add(a, b):\n    """Return sum of two numbers."""\n',
                "entry_point": "add",
                "test": (
                    "def check(candidate):\n"
                    "    assert candidate(1, 2) == 3\n"
                    "    assert candidate(3, 4) == 7\n"
                ),
                "canonical_solution": "    return a + b\n",
                "base_input": [[1, 2], [3, 4]],
                "plus_input": [[-1, 1], [10, -3]],
                "atol": 0.0,
            },
        )

        prompt = bundle.task_module.build_prompt(sample)
        self.assertIsInstance(prompt, list)
        self.assertEqual(prompt[0]["role"], "system")
        self.assertEqual(prompt[1]["role"], "user")
        self.assertIn("### Question:", prompt[1]["content"])
        self.assertIn("### Format:", prompt[1]["content"])
        self.assertIn(
            "Provide a SHORT reasoning on how to solve the task", prompt[1]["content"]
        )
        self.assertIn("# YOUR CODE HERE", prompt[1]["content"])
        self.assertIn(
            "### Answer: (use the provided format with backticks)", prompt[1]["content"]
        )
        self.assertIn(sample.data["prompt"], prompt[1]["content"])

        scored = metrics_module.score_generation(
            sample,
            "```python\ndef add(a, b):\n    return a + b\n```",
        )
        self.assertEqual(scored["score"], 1.0)
        self.assertTrue(scored["parsed"]["base_pass"])
        self.assertTrue(scored["parsed"]["plus_pass"])

        # Ordinary examples pass, but a negative-number Plus case catches the bug.
        plus_failure = metrics_module.score_generation(
            sample, "    return abs(a) + abs(b)\n"
        )
        self.assertTrue(plus_failure["parsed"]["base_pass"])
        self.assertFalse(plus_failure["parsed"]["plus_pass"])
        self.assertEqual(plus_failure["score"], 0.0)
        self.assertEqual(plus_failure["meta"]["scoring_protocol"], "evalplus-26d6d00")

        base_failure = metrics_module.score_generation(sample, "    return 0\n")
        self.assertFalse(base_failure["parsed"]["base_pass"])
        self.assertEqual(base_failure["parsed"]["plus_status"], "skipped")
        self.assertEqual(base_failure["score"], 0.0)

        # Changing reference contents under the same ID must invalidate the oracle.
        changed = copy.deepcopy(sample)
        changed.data["canonical_solution"] = "    return a - b\n"
        self.assertEqual(
            metrics_module.score_generation(changed, "    return a - b\n")["score"],
            1.0,
        )

        # Regression guard: if generation includes a full function with typing annotations,
        # evaluator must still execute with the original prompt imports.
        sample_with_import = Sample(
            id="HumanEval/test_import",
            gold=None,
            meta={"entry_point": "first_item"},
            data={
                "task_id": "HumanEval/test_import",
                "prompt": (
                    "from typing import List\n\n"
                    "def first_item(items: List[int]) -> int:\n"
                    '    """Return the first item."""\n'
                ),
                "entry_point": "first_item",
                "test": (
                    "def check(candidate):\n"
                    "    assert candidate([1, 2, 3]) == 1\n"
                    "    assert candidate([9, 8, 7]) == 9\n"
                ),
                "canonical_solution": "    return items[0]\n",
                "base_input": [[[1, 2, 3]]],
                "plus_input": [[[9, 8, 7]]],
                "atol": 0.0,
            },
        )
        scored_full = metrics_module.score_generation(
            sample_with_import,
            "```python\ndef first_item(items: List[int]) -> int:\n    return items[0]\n```",
        )
        self.assertEqual(scored_full["score"], 1.0)
        self.assertTrue(scored_full["parsed"]["base_pass"])
        self.assertTrue(scored_full["parsed"]["plus_pass"])

    def test_humaneval_plus_score_generation_preserves_first_line_indentation(
        self,
    ) -> None:
        bundle = load_task("humaneval_plus")
        metrics_module = bundle.metrics_module

        sample = Sample(
            id="HumanEval/test_indent",
            gold=None,
            meta={"entry_point": "add"},
            data={
                "task_id": "HumanEval/test_indent",
                "prompt": 'def add(a, b):\n    """Return sum of two numbers."""\n',
                "entry_point": "add",
                "test": (
                    "def check(candidate):\n"
                    "    assert candidate(1, 2) == 3\n"
                    "    assert candidate(3, 4) == 7\n"
                ),
                "canonical_solution": "    return a + b\n",
                "base_input": [[1, 2], [3, 4]],
                "plus_input": [[-1, 1], [10, -3]],
                "atol": 0.0,
            },
        )

        scored = metrics_module.score_generation(
            sample,
            "```python\n    return a + b\n```",
        )
        self.assertEqual(scored["score"], 1.0)
        self.assertTrue(scored["parsed"]["base_pass"])
        self.assertTrue(scored["parsed"]["plus_pass"])

    def test_humaneval_plus_ignores_usage_example_blocks(self) -> None:
        metrics_module = load_task("humaneval_plus").metrics_module
        sample = Sample(
            id="HumanEval/test_blocks",
            gold=None,
            meta={"entry_point": "add"},
            data={
                "task_id": "HumanEval/test_blocks",
                "prompt": 'def add(a, b):\n    """Return sum of two numbers."""\n',
                "entry_point": "add",
                "canonical_solution": "    return a + b\n",
                "base_input": [[1, 2], [3, 4]],
                "plus_input": [[-1, 1], [10, -3]],
                "atol": 0.0,
            },
        )
        example = (
            "values = [(1, 2), (3, 4), (5, 6), (7, 8)]\n"
            "for left, right in values:\n"
            "    result = add(left, right)\n"
            "    print(left, right, result)\n"
        )
        full = "```python\ndef add(a, b):\n    return a + b\n```"
        cases = {
            "full function, longer example": f"{full}\nUsage:\n```python\n{example}```",
            "body, then example": f"```python\n    return a + b\n```\n```python\n{example}```",
            "draft, then final": f"```python\ndef add(a, b):\n    return 0\n```\n{full}",
            "draft, then final body": "```python\ndef add(a, b):\n    return 0\n```\n```python\n    return a + b\n```",
            "separate import": "```python\nimport operator\n```\n```python\ndef add(a, b):\n    return operator.add(a, b)\n```",
            "separate helper": "```python\ndef helper(a, b):\n    return a + b\n```\n```python\ndef add(a, b):\n    return helper(a, b)\n```",
            "helper after entry point": "```python\ndef add(a, b):\n    return helper(a, b)\n```\n```python\ndef helper(a, b):\n    return a + b\n```",
        }
        for name, generation in cases.items():
            with self.subTest(name):
                scored = metrics_module.score_generation(sample, generation)
                self.assertEqual(scored["score"], 1.0)

        for generation in (
            "",
            f"{full}\n```python\ndef add(a, b):\n    return 0\n```",
            f"{full}\n```python\n    return 0\n```",
        ):
            with self.subTest(generation=generation):
                self.assertEqual(metrics_module.score_generation(sample, generation)["score"], 0.0)

    def test_humaneval_plus_aggregate(self) -> None:
        bundle = load_task("humaneval_plus")
        metrics_module = bundle.metrics_module

        sample_results = [
            {
                "sample_id": "h1",
                "records": [
                    {
                        "sample_id": "h1",
                        "gen_idx": 0,
                        "score": 1.0,
                        "is_pass": True,
                        "parsed": {"base_pass": True, "plus_pass": True},
                    },
                    {
                        "sample_id": "h1",
                        "gen_idx": 1,
                        "score": 0.0,
                        "is_pass": False,
                        "parsed": {"base_pass": False, "plus_pass": False},
                    },
                ],
            },
            {
                "sample_id": "h2",
                "records": [
                    {
                        "sample_id": "h2",
                        "gen_idx": 0,
                        "score": 0.0,
                        "is_pass": False,
                        "parsed": {"base_pass": True, "plus_pass": False},
                    },
                    {
                        "sample_id": "h2",
                        "gen_idx": 1,
                        "score": 0.0,
                        "is_pass": False,
                        "parsed": {"base_pass": False, "plus_pass": False},
                    },
                ],
            },
        ]
        result = self._aggregate(metrics_module, sample_results, {"n": 2})
        self.assertAlmostEqual(result["accuracy"], 0.25, places=6)
        self.assertAlmostEqual(result["accuracy_plus"], 0.25, places=6)
        self.assertAlmostEqual(result["accuracy_base"], 0.5, places=6)
        self.assertAlmostEqual(result["accuracy@2"], 0.25, places=6)
        self.assertAlmostEqual(result["pass@1"], 0.25, places=6)
        self.assertAlmostEqual(result["pass@2"], 0.5, places=6)

    def test_humaneval_plus_score_generation_does_not_mutate_inputs(self) -> None:
        bundle = load_task("humaneval_plus")
        metrics_module = bundle.metrics_module

        sample = Sample(
            id="HumanEval/mutation_guard",
            gold=None,
            meta={"entry_point": "find_closest_elements"},
            data={
                "task_id": "HumanEval/mutation_guard",
                "prompt": "def find_closest_elements(numbers):\n",
                "entry_point": "find_closest_elements",
                "test": (
                    "def check(candidate):\n"
                    "    assert candidate([3.0, 1.0, 2.0]) == (1.0, 2.0)\n"
                    "    assert candidate([5.0, 4.0, 6.0]) == (4.0, 5.0)\n"
                ),
                "canonical_solution": (
                    "    numbers.sort()\n    return (numbers[0], numbers[1])\n"
                ),
                "base_input": [[[3.0, 1.0, 2.0]], [[5.0, 4.0, 6.0]]],
                "plus_input": [[[9.0, 7.0, 8.0]]],
                "atol": 0.0,
            },
        )

        before_base = copy.deepcopy(sample.data["base_input"])
        before_plus = copy.deepcopy(sample.data["plus_input"])

        metrics_module.score_generation(
            sample,
            "```python\ndef find_closest_elements(numbers):\n    return (numbers[0], numbers[1])\n```",
        )

        self.assertEqual(sample.data["base_input"], before_base)
        self.assertEqual(sample.data["plus_input"], before_plus)

    def test_strip_reasoning_keeps_only_final_answer(self) -> None:
        from aethereval.backends.prompt import prefilled_reasoning_prefix
        from aethereval.metrics.common import strip_reasoning

        self.assertEqual(strip_reasoning("<think>draft</think>\n\nfinal"), "final")
        self.assertEqual(strip_reasoning("plain answer"), "plain answer")
        # Budget exhausted while thinking: there is no answer to grade.
        self.assertEqual(strip_reasoning("<think>so it is \\boxed{204}"), "")
        # A literal closing tag in the answer is not a reasoning boundary.
        code = 'def f():\n    return "</think>"'
        self.assertEqual(strip_reasoning(code), code)
        self.assertEqual(strip_reasoning("<think>x</think>" + code), code)
        # Templates that pre-fill the opener: the backend restores it.
        self.assertEqual(prefilled_reasoning_prefix("<|im_start|>assistant\n<think>\n"), "<think>\n")
        self.assertEqual(prefilled_reasoning_prefix("<|im_start|>assistant\n"), "")
        self.assertEqual(
            prefilled_reasoning_prefix("assistant\n<think>\n\n</think>\n\n"), ""
        )

    def test_mcq_extractor_ignores_article_a(self) -> None:
        from aethereval.metrics.common import extract_choice

        letters = ["A", "B", "C", "D"]
        for text in ("The answer depends on a catalyst.", "Answer: a catalyst."):
            self.assertIsNone(extract_choice(text, {}, letters)[0])
        self.assertEqual(extract_choice("Answer: (c)", {}, letters)[0], "C")
        self.assertEqual(extract_choice("The final answer is C.", {}, letters)[0], "C")
        for text in ("Answer: c", "c", "**c**", "The final answer is c.\nExplanation follows."):
            self.assertEqual(extract_choice(text, {}, letters)[0], "C")
        self.assertEqual(extract_choice("Answer: a", {}, letters)[0], "A")

    def test_math_scoring_preserves_text_and_compares_tiny_values(self) -> None:
        from benchmark_utils.math_scoring import score_with_math_verify

        # Gold notation is repaired in the dataset, not rewritten by the scorer.
        gold = "so the mass is $\\boxed{4.5 \\times 10^{33}}$ g."
        self.assertEqual(
            score_with_math_verify(gold, "\\boxed{4.5 \\times 10^{33}}")[0], 1.0
        )
        self.assertEqual(score_with_math_verify("\\boxed{10}", "2.88e-19")[0], 0.0)
        tiny = "energy is $\\boxed{2.88 \\times 10^{-19}}$"
        self.assertEqual(
            score_with_math_verify(tiny, "\\boxed{2.88 \\times 10^{-19}}")[0], 1.0
        )
        self.assertEqual(
            score_with_math_verify(tiny, "\\boxed{5.76 \\times 10^{-19}}")[0], 0.0
        )

    def test_bbh_extraction_edge_cases(self) -> None:
        extract = load_task("bbh").metrics_module._extract_answer
        cases = [
            ("web_of_lies", "I know the answer now. Yes", "Yes"),
            ("word_sorting", "So the answer is: apple banana", "apple banana"),
            ("word_sorting", "So the answer is **apple banana**.", "apple banana"),
            ("multistep_arithmetic_two", "So the answer is 1,234.", "1,234"),
            ("dyck_languages", "So the answer is `] ) >`", "] ) >"),
            # Bare letters at an answer position beat the position-free fallbacks.
            ("date_understanding", "Check (A).. (C) is off. So the answer is B.", "(B)"),
            ("date_understanding", "The answer is B. I hope this helps.", "(B)"),
            ("date_understanding", "So the answer is **B**. I checked it.", "(B)"),
            ("date_understanding", "The answer is \\boxed{B}. A quick check.", "(B)"),
            ("date_understanding", "**Answer:** D. I am sure", "(D)"),
            ("date_understanding", "The answer is I think (B)", "(B)"),
            ("date_understanding", "the answer is a bit tricky. (D)", "(D)"),
            ("date_understanding", "So the answer is (C).", "(C)"),
            # The position-free fallback skips prose "I" / "A" as well.
            ("date_understanding", "B. I hope this helps.", "(B)"),
            ("date_understanding", "I think", ""),
            ("date_understanding", "A quick check", ""),
        ]
        for subset, text, expected in cases:
            with self.subTest(subset=subset, text=text):
                self.assertEqual(extract(text, subset)[0], expected)


if __name__ == "__main__":
    unittest.main()
