from __future__ import annotations

import copy
import unittest

from aethereval.core.task_register import load_task
from aethereval.core.types import Sample


class MetricsTests(unittest.TestCase):
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

        result = metrics_module.aggregate(sample_results, {})
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

        result = metrics_module.aggregate(sample_results, {})

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

        result = metrics_module.aggregate(sample_results, {})
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

        result2 = metrics_module.score_generation(sample, "I think the correct option is B.")
        self.assertEqual(result2["score"], 0.0)
        self.assertFalse(result2["is_pass"])
        self.assertEqual(result2["parsed"]["prediction"], "B")

        # Do not parse letters embedded in normal words.
        result3 = metrics_module.score_generation(sample, "Answer: because of uncertainty.")
        self.assertEqual(result3["score"], 0.0)
        self.assertIsNone(result3["parsed"]["prediction"])

        # Do not fallback to option-text matching; only explicit choice letters count.
        result4 = metrics_module.score_generation(sample, "My answer is gamma option.")
        self.assertEqual(result4["score"], 0.0)
        self.assertIsNone(result4["parsed"]["prediction"])

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

        result = metrics_module.aggregate(sample_results, {"n": 2})
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
                    {"sample_id": "q1", "gen_idx": 0, "score": 1.0, "is_pass": True, "parsed": {}},
                    {"sample_id": "q1", "gen_idx": 1, "score": 0.0, "is_pass": False, "parsed": {}},
                    {"sample_id": "q1", "gen_idx": 2, "score": 0.0, "is_pass": False, "parsed": {}},
                    {"sample_id": "q1", "gen_idx": 3, "score": 1.0, "is_pass": True, "parsed": {}},
                ],
            },
            {
                "sample_id": "q2",
                "records": [
                    {"sample_id": "q2", "gen_idx": 0, "score": 0.0, "is_pass": False, "parsed": {}},
                    {"sample_id": "q2", "gen_idx": 1, "score": 0.0, "is_pass": False, "parsed": {}},
                    {"sample_id": "q2", "gen_idx": 2, "score": 1.0, "is_pass": True, "parsed": {}},
                    {"sample_id": "q2", "gen_idx": 3, "score": 0.0, "is_pass": False, "parsed": {}},
                ],
            },
        ]

        result = metrics_module.aggregate(sample_results, {"n": 4})
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
                    {"sample_id": "q1", "gen_idx": 0, "score": 0.0, "is_pass": False, "parsed": {}},
                    {"sample_id": "q1", "gen_idx": 1, "score": 0.0, "is_pass": False, "parsed": {}},
                    {"sample_id": "q1", "gen_idx": 2, "score": 0.0, "is_pass": False, "parsed": {}},
                    {"sample_id": "q1", "gen_idx": 3, "score": 0.0, "is_pass": False, "parsed": {}},
                    {"sample_id": "q1", "gen_idx": 4, "score": 1.0, "is_pass": True, "parsed": {}},
                ],
            },
        ]

        result = metrics_module.aggregate(sample_results, {"n": 5})
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
                    {"sample_id": "m1", "gen_idx": 0, "score": 1.0, "is_pass": True, "parsed": {"prediction": "I"}},
                    {"sample_id": "m1", "gen_idx": 1, "score": 0.0, "is_pass": False, "parsed": {"prediction": "A"}},
                ],
            },
            {
                "sample_id": "m2",
                "meta": {"category": "law"},
                "records": [
                    {"sample_id": "m2", "gen_idx": 0, "score": 0.0, "is_pass": False, "parsed": {"prediction": "B"}},
                    {"sample_id": "m2", "gen_idx": 1, "score": 0.0, "is_pass": False, "parsed": {"prediction": "C"}},
                ],
            },
        ]
        result = metrics_module.aggregate(sample_results, {"n": 2})
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
                    {"sample_id": "a1", "gen_idx": 0, "score": 1.0, "is_pass": True, "parsed": {"prediction": "D"}},
                ],
            },
            {
                "sample_id": "a2",
                "meta": {"subset": "logiqa-en"},
                "records": [
                    {"sample_id": "a2", "gen_idx": 0, "score": 0.0, "is_pass": False, "parsed": {"prediction": "A"}},
                ],
            },
        ]
        result = metrics_module.aggregate(sample_results, {"n": 1})
        self.assertAlmostEqual(result["accuracy"], 0.5, places=6)
        self.assertAlmostEqual(result["pass@1"], 0.5, places=6)
        self.assertAlmostEqual(result["accuracy_sat_en"], 1.0, places=6)
        self.assertAlmostEqual(result["accuracy_logiqa_en"], 0.0, places=6)

    def test_zebralogic_metrics(self) -> None:
        bundle = load_task("zebralogic")
        metrics_module = bundle.metrics_module

        sample = Sample(
            id="z1",
            gold="6",
            meta={},
            data={"question": "Dummy"},
        )
        scored = metrics_module.score_generation(sample, "ANSWER: 6")
        self.assertEqual(scored["score"], 1.0)
        self.assertEqual(scored["parsed"]["prediction_norm"], "6")

        sample_results = [
            {
                "sample_id": "z1",
                "records": [
                    {"sample_id": "z1", "gen_idx": 0, "score": 1.0, "is_pass": True, "parsed": {"prediction": "6"}},
                    {"sample_id": "z1", "gen_idx": 1, "score": 0.0, "is_pass": False, "parsed": {"prediction": "5"}},
                ],
            },
            {
                "sample_id": "z2",
                "records": [
                    {"sample_id": "z2", "gen_idx": 0, "score": 0.0, "is_pass": False, "parsed": {"prediction": "1"}},
                    {"sample_id": "z2", "gen_idx": 1, "score": 0.0, "is_pass": False, "parsed": {"prediction": "2"}},
                ],
            },
        ]
        result = metrics_module.aggregate(sample_results, {"n": 2})
        self.assertAlmostEqual(result["accuracy"], 0.25, places=6)
        self.assertAlmostEqual(result["accuracy@2"], 0.25, places=6)
        self.assertAlmostEqual(result["pass@1"], 0.25, places=6)
        self.assertAlmostEqual(result["pass@2"], 0.5, places=6)

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
                    {"sample_id": "l1", "gen_idx": 0, "score": 1.0, "is_pass": True, "parsed": {"had_code": True}},
                    {"sample_id": "l1", "gen_idx": 1, "score": 0.0, "is_pass": False, "parsed": {"had_code": True}},
                ],
            },
            {
                "sample_id": "l2",
                "meta": {"platform": "leetcode"},
                "records": [
                    {"sample_id": "l2", "gen_idx": 0, "score": 0.0, "is_pass": False, "parsed": {"had_code": False}},
                    {"sample_id": "l2", "gen_idx": 1, "score": 0.0, "is_pass": False, "parsed": {"had_code": True}},
                ],
            },
        ]

        result = metrics_module.aggregate(sample_results, {"n": 2})
        self.assertAlmostEqual(result["accuracy"], 0.25, places=6)
        self.assertAlmostEqual(result["accuracy@2"], 0.25, places=6)
        self.assertAlmostEqual(result["pass@1"], 0.25, places=6)
        self.assertAlmostEqual(result["pass@2"], 0.5, places=6)
        self.assertAlmostEqual(result["parsed_rate"], 0.75, places=6)
        self.assertAlmostEqual(result["accuracy_atcoder"], 0.5, places=6)
        self.assertAlmostEqual(result["accuracy_leetcode"], 0.0, places=6)

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
        self.assertIn("Read the inputs from stdin solve the problem", prompt_no_starter[1]["content"])
        self.assertIn("### Answer: (use the provided format with backticks)", prompt_no_starter[1]["content"])

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
        self.assertIn("class Solution:", prompt_with_starter[1]["content"])

    def test_humaneval_plus_score_generation(self) -> None:
        bundle = load_task("humaneval_plus")
        metrics_module = bundle.metrics_module

        sample = Sample(
            id="HumanEval/test_add",
            gold=None,
            meta={"entry_point": "add"},
            data={
                "task_id": "HumanEval/test_add",
                "prompt": "def add(a, b):\n    \"\"\"Return sum of two numbers.\"\"\"\n",
                "entry_point": "add",
                "canonical_solution": "    return a + b\n",
                "base_input": [[1, 2], [3, 4]],
                "plus_input": [[-1, 1], [10, -3]],
                "atol": 0.0,
            },
        )

        scored = metrics_module.score_generation(
            sample,
            "```python\ndef add(a, b):\n    return a + b\n```",
        )
        self.assertEqual(scored["score"], 1.0)
        self.assertTrue(scored["parsed"]["base_pass"])
        self.assertTrue(scored["parsed"]["plus_pass"])

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
                        "parsed": {"base_pass": False, "plus_pass": False},
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
        result = metrics_module.aggregate(sample_results, {"n": 2})
        self.assertAlmostEqual(result["accuracy"], 0.25, places=6)
        self.assertAlmostEqual(result["accuracy_plus"], 0.25, places=6)
        self.assertAlmostEqual(result["accuracy_base"], 0.25, places=6)
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
                "canonical_solution": (
                    "    numbers.sort()\n"
                    "    return (numbers[0], numbers[1])\n"
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


if __name__ == "__main__":
    unittest.main()
