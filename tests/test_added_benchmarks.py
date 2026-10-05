import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

from aethereval.cli import build_parser
from aethereval.config import resolve_run_arguments
from aethereval.core import runner
from aethereval.core.io import write_jsonl
from aethereval.core.task_defaults import resolve_task_default_gen
from aethereval.core.task_register import load_task
from aethereval.core.types import GenerationOutput, Sample
from benchmarks.guidebench import metrics as guide
from benchmarks.guidebench import task as guide_task
from tests.test_runner import FakeBackend

ROOT = Path(__file__).resolve().parents[1]
LONG = load_task("longbench-write").metrics_module


class AddedBenchmarkTests(unittest.TestCase):
    def test_registered_tasks_and_reference_decoder_settings(self):
        for name, cap, temperature in (
            ("guidebench", 4096, 0),
            ("longbench-write", 32768, 0.5),
        ):
            bundle = load_task(name)
            self.assertEqual(bundle.task_module.TASK_NAME, name)
            gen = resolve_task_default_gen(name)
            self.assertEqual(
                (gen["n"], gen["max_new_tokens"], gen["temperature"]),
                (1, cap, temperature),
            )

    def test_prepared_public_data_counts_and_prompt_gold_separation(self):
        # Actual pinned releases, when prepared; unit-only environments may omit data.
        for name, expected in (
            ("guidebench", 1042),
            ("longbench-write", 120),
        ):
            bundle = load_task(name)
            if not (bundle.spec.task_dir / bundle.task_module.DATA_FILE).exists():
                continue
            samples = bundle.task_module.load_samples(bundle.spec.task_dir)
            self.assertEqual(len(samples), expected)
            self.assertEqual(len({sample.id for sample in samples}), expected)
            for sample in samples:
                self.assertTrue(bundle.task_module.build_prompt(sample).strip())
            if name == "guidebench":
                self.assertEqual(
                    Counter(sample.meta["category"] for sample in samples),
                    {
                        "price": 442,
                        "relevance": 192,
                        "math": 52,
                        "chat": 180,
                        "summary": 58,
                        "hallu": 118,
                    },
                )
                self.assertNotIn("Groundtruth", samples[0].data)
            else:
                self.assertEqual(
                    Counter(sample.meta["language"] for sample in samples),
                    {"en": 60, "zh": 60},
                )

    def test_guide_exact_json_scoring_and_no_credit_for_malformed_outputs(self):
        samples = [
            ("price", 1, "CandidateAnswer"),
            ("relevance", "2（强相关）", "CandidateAnswer"),
            ("math", "170", "CandidateAnswer"),
            ("chat", "A", "OptimalOption"),
        ]
        for category, gold, field in samples:
            sample = Sample("x", gold=gold, meta={"category": category})
            answer = json.dumps({field: gold}, ensure_ascii=False)
            for response in (answer, "```json\n" + answer + "\n```"):
                self.assertEqual(guide.score_generation(sample, response)["score"], 1)
            for response in (
                "No answer",
                "[]",
                "null",
                "{}",
                '{"CandidateAnswer": true}',
                "prefix " + answer,
            ):
                self.assertEqual(guide.score_generation(sample, response)["score"], 0)
        sample = Sample("x", gold="170", meta={"category": "math"})
        self.assertEqual(
            guide.score_generation(sample, '{"CandidateAnswer": 170}')["score"], 0
        )

    def test_guide_count_weighted_aggregation_and_parse_failure_denominator(self):
        items = []
        for category, gold, answer in (
            ("price", 1, '{"CandidateAnswer":1}'),
            ("price", 1, "{}"),
            ("chat", "A", '{"OptimalOption":"A"}'),
        ):
            result = guide.score_generation(
                Sample("x", gold=gold, meta={"category": category}), answer
            )
            items.append({"meta": {"category": category}, "records": [result]})
        metrics = guide.aggregate(items)
        self.assertAlmostEqual(metrics["accuracy"], 2 / 3)
        self.assertEqual(metrics["category/price"], 0.5)
        self.assertAlmostEqual(metrics["parsed_rate"], 2 / 3)

    def test_guide_prompt_preserves_reference_template_without_groundtruth(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "data").mkdir()
            row = {
                "id": "math:0",
                "category": "math",
                "Instruction": "task",
                "Guidelines": [{"rule_text": "rule"}],
                "Context": "context",
                "Groundtruth": {
                    "ReferenceAnswer": "12345",
                    "ReferenceAnalysis": "SECRET ANALYSIS",
                },
            }
            write_jsonl(directory / "data/eval.jsonl", [row])
            sample = guide_task.load_samples(directory)[0]
            prompt = guide_task.build_prompt(sample)
            self.assertIn(str(row["Guidelines"]), prompt)
            self.assertNotIn("12345", prompt)
            self.assertNotIn("SECRET ANALYSIS", prompt)

    def test_longbench_word_count_and_official_piecewise_formula(self):
        self.assertEqual(LONG.count_words("Hello-world 123 你好 café A_b"), 4)
        for requested, actual, expected in (
            (100, 100, 100),
            (100, 0, 0),
            (100, 50, 50),
            (100, 25, 0),
            (100, 400, 0),
            (100, 200, 200 / 3),
        ):
            self.assertAlmostEqual(LONG.length_score(requested, actual), expected)
        with self.assertRaises(ValueError):
            LONG.length_score(0, 1)

    def test_longbench_all_six_dimensions_required_and_bounded_integers(self):
        grade = {name: 3 for name in LONG.DIMENSIONS}
        self.assertEqual(LONG.parse_grade(json.dumps(grade)), grade)
        for bad in (
            {},
            {**grade, "Clarity": 6},
            {**grade, "Clarity": True},
            {**grade, "Clarity": 3.5},
            {**grade, "Clarity": "3"},
        ):
            self.assertIsNone(LONG.parse_grade(json.dumps(bad)))

    def test_longbench_separate_quality_length_and_failed_judgment(self):
        sample = Sample(
            "x", data={"prompt": "write 4 words", "length": 4}, meta={"language": "en"}
        )
        output = GenerationOutput("x", "write 4 words", ["one two three four"])
        grade = {"Analysis": "analysis", **dict.fromkeys(LONG.DIMENSIONS, 3)}
        with mock.patch(
            "benchmark_utils.llm_judge.chat_completion", return_value=json.dumps(grade)
        ) as chat:
            result = LONG.score_generations_batch(
                [sample], [output], {"judge_workers": 1}
            )[0][0]
        self.assertEqual(
            chat.call_args.args[1],
            [
                {
                    "role": "user",
                    "content": LONG.PROMPT.replace(
                        "$INST$", sample.data["prompt"]
                    ).replace("$RESPONSE$", output.generations[0]),
                }
            ],
        )
        self.assertEqual(result["parsed"]["quality_score"], 50)
        self.assertEqual(result["parsed"]["length_score"], 100)
        self.assertEqual(result["score"], 75)
        metrics = LONG.aggregate([{"meta": sample.meta, "records": [result]}])
        self.assertEqual(
            (
                metrics["quality_score"],
                metrics["length_score"],
                metrics["overall_score"],
            ),
            (50, 100, 75),
        )
        with mock.patch(
            "benchmark_utils.llm_judge.chat_completion", return_value="invalid"
        ):
            failed = LONG.score_generations_batch(
                [sample], [output], {"judge_workers": 1}
            )[0][0]
        self.assertTrue(failed["meta"]["_aethereval_unscored"])
        self.assertIsNone(failed["parsed"]["quality_score"])
        metrics = LONG.aggregate([{"meta": sample.meta, "records": [result, failed]}])
        self.assertEqual(metrics["quality_score"], 50)  # No invented zero included.
        self.assertEqual(metrics["failed_judgments"], 1)

    def test_csv_judges_shared_runtime_options(self):
        args = build_parser().parse_args(
            [
                "--model",
                "candidate",
                "--judge-models",
                " a , b ",
                "--judge-backend",
                "local",
                "--judge-tp-size",
                "2",
                "--no-judge-enable-thinking",
                "--judge-sglang-arg",
                "context_length=65536",
            ]
        )
        options = resolve_run_arguments(args, {})["metric_options"]
        self.assertEqual(options["judge_models"], ["a", "b"])
        self.assertEqual(options["judge_tp_size"], 2)
        self.assertIs(options["judge_enable_thinking"], False)
        self.assertEqual(options["judge_sglang_args"], {"context_length": 65536})

    def test_longbench_end_to_end_two_local_judges_share_one_generation_and_settings(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "benchmarks" / "longbench-write" / "data").mkdir(parents=True)
            # Use the real adapter modules, only replace the data with two prompts.
            import shutil

            for name in ("task.py", "metrics.py", "judge.txt"):
                shutil.copy2(
                    ROOT / "benchmarks/longbench-write" / name,
                    root / "benchmarks/longbench-write" / name,
                )
            write_jsonl(
                root / "benchmarks/longbench-write/data/eval.jsonl",
                [{"prompt": "write", "length": 1, "type": "test", "language": "en"}],
            )
            backend = FakeBackend()
            configs, requests = [], []

            class Judge:
                def __init__(self, **kwargs):
                    self.model = kwargs["model"]
                    configs.append(kwargs)

                def complete(self, messages, **kwargs):
                    requests.append((self.model, kwargs))
                    grade = 3 if self.model == "a" else 5
                    return json.dumps(
                        {"Analysis": "test", **dict.fromkeys(LONG.DIMENSIONS, grade)}
                    )

                def close(self):
                    pass

            options = {
                "judge_models": "a,b",
                "judge_backend": "local",
                "judge_dp_size": 1,
                "judge_tp_size": 2,
                "judge_enable_thinking": False,
                "judge_sglang_args": {"context_length": 65536},
                "judge_workers": 1,
            }
            with (
                mock.patch.object(runner, "create_backend", return_value=backend),
                mock.patch.object(runner, "OfflineJudgeClient", Judge),
            ):
                result = runner.run_evaluation(
                    model="candidate",
                    tasks="longbench-write",
                    output_dir=root / "out",
                    benchmarks_dir=root / "benchmarks",
                    metric_options=options,
                )
            self.assertEqual(backend.calls, 1)
            self.assertEqual([entry["model"] for entry in configs], ["a", "b"])
            for entry in configs:
                self.assertEqual(
                    (entry["dp_size"], entry["tensor_parallel_size"]), (1, 2)
                )
                self.assertEqual(entry["model_kwargs"], {"context_length": 65536})
            for model, request in requests:
                self.assertIs(request["extra_body"]["enable_thinking"], False)
                self.assertIn(model, ("a", "b"))
            self.assertTrue(result["evaluation_complete"])
            self.assertEqual(
                result["judges"]["a"]["results"]["longbench-write"]["metrics"][
                    "quality_score"
                ],
                50,
            )
            self.assertEqual(
                result["judges"]["b"]["results"]["longbench-write"]["metrics"][
                    "quality_score"
                ],
                100,
            )
