import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from aethereval.cli import build_parser, run_selected_tasks
from aethereval.config import resolve_run_arguments
from aethereval.core import runner
from aethereval.core.multi_judge import judge_directory, validate_judge_models
from tests.test_runner import FakeBackend, _write_toy2_benchmark, _write_toy_benchmark


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class MultiJudgeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.benchmarks = self.root / "benchmarks"
        _write_toy_benchmark(self.benchmarks)
        _write_toy2_benchmark(self.benchmarks)
        (self.benchmarks / "toy" / "metrics.py").write_text(
            "USES_LLM_JUDGE=True\n"
            "PRESERVE_EXISTING_SCORES_ON_RESUME=True\n"
            "PRIMARY_METRIC='accuracy'\n"
            "def score_generation(sample, generation):\n"
            "    raise AssertionError('batch judging required')\n"
            "def score_generations_batch(samples, outputs, options):\n"
            "    client=options.get('_judge_client')\n"
            "    result=[]\n"
            "    for output in outputs:\n"
            "        scores=[]\n"
            "        for text in output.generations:\n"
            "            value=float(client.complete([{'role': 'user', 'content': text}])) if client else (0.25 if options['judge_model']=='judges/a' else 0.75)\n"
            "            scores.append({'score': max(value, 0), 'meta': {'_aethereval_unscored': value < 0}})\n"
            "        result.append(scores)\n"
            "    return result\n"
            "def aggregate(results, metric_options=None):\n"
            "    scores=[score for item in results for score in item['scores']]\n"
            "    return {'accuracy': sum(scores)/len(scores)}\n"
        )
        self.run_root = self.root / "outputs" / "candidate" / "dual"
        self.events = []
        self.fail_model = None
        self.unscored_model = None
        self.backend = FakeBackend()
        events = self.events
        original_generate = self.backend.generate

        def generate(inputs, gen_cfg):
            events.append("generate")
            return original_generate(inputs, gen_cfg)

        self.backend.generate = generate
        self.backend.close = lambda: events.append("candidate-close")
        test = self

        class Judge:
            def __init__(self, model, **kwargs):
                self.model = model
                events.append(("start", model))

            def complete(self, messages, **kwargs):
                events.append(("score", self.model, messages[0]["content"]))
                if test.fail_model == self.model:
                    raise RuntimeError("judge unavailable")
                if test.unscored_model == self.model:
                    return "-1"
                return "0.25" if self.model == "judges/a" else "0.75"

            def close(self):
                events.append(("close", self.model))

        self.create_backend = mock.patch.object(
            runner, "create_backend", return_value=self.backend
        ).start()
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(runner, "OfflineJudgeClient", Judge).start()
        self.scoring = mock.patch.object(
            runner, "_score_generation_outputs", wraps=runner._score_generation_outputs
        ).start()

    def evaluate(self, **kwargs):
        return runner.run_evaluation(
            **{
                "model": "candidate",
                "tasks": "toy,toy2",
                "output_dir": self.root / "outputs",
                "run_id": "dual",
                "benchmarks_dir": self.benchmarks,
                "metric_options": {
                    "judge_backend": "local",
                    "judge_models": ["judges/a", "judges/b"],
                },
                **kwargs,
            }
        )

    def test_generation_shared_and_scores_separate_with_common_tasks_scored_once(self):
        result = self.evaluate()
        self.assertTrue(result["evaluation_complete"])
        self.assertNotIn("primary_score_aggregate", result)
        self.assertNotIn("results", result)
        self.assertEqual(self.backend.calls, 2)  # One generation batch per task.
        self.assertEqual(self.create_backend.call_count, 1)
        ordinary_calls = [
            call
            for call in self.scoring.call_args_list
            if call.kwargs["metrics_module"].__file__.endswith("toy2/metrics.py")
        ]
        self.assertEqual(len(ordinary_calls), 1)
        source = _rows(self.run_root / "toy" / "predictions.jsonl")
        for model, score in (("judges/a", 0.25), ("judges/b", 0.75)):
            profile = result["judges"][model]
            path = judge_directory(self.run_root, model)
            judged = _rows(path / "toy" / "predictions.jsonl")
            self.assertEqual(
                [row["generation"] for row in judged],
                [row["generation"] for row in source],
            )
            self.assertEqual([row["score"] for row in judged], [score, score])
            self.assertEqual(profile["results"]["toy"]["primary_score"], score * 100)
            self.assertEqual(profile["results"]["toy2"]["primary_score"], 100)
            self.assertEqual(
                profile["primary_score_aggregate"], (score * 100 + 100) / 2
            )
            manifest = json.loads((path / "toy" / "run_config.json").read_text())
            self.assertEqual(manifest["metric_options"]["judge_model"], model)
            self.assertNotIn("judge_models", manifest["metric_options"])
        self.assertTrue(all(row["meta"]["_aethereval_unscored"] for row in source))
        self.assertLess(
            self.events.index("candidate-close"),
            self.events.index(("start", "judges/a")),
        )
        self.assertLess(
            self.events.index(("close", "judges/a")),
            self.events.index(("start", "judges/b")),
        )
        self.assertEqual(self.events[-1], ("close", "judges/b"))
        self.assertEqual(
            json.loads((self.run_root / "run_summary.json").read_text()), result
        )

    def test_resume_keeps_both_judgments_and_adding_judge_only_scores_new_model(self):
        self.evaluate()
        self.events.clear()
        result = self.evaluate()
        self.assertEqual(self.events, [])
        self.assertTrue(result["evaluation_complete"])
        self.assertEqual(self.backend.calls, 2)
        self.evaluate(
            metric_options={
                "judge_backend": "local",
                "judge_models": ["judges/a", "judges/b", "judges/c"],
            }
        )
        starts = [
            event
            for event in self.events
            if isinstance(event, tuple) and event[0] == "start"
        ]
        self.assertEqual(starts, [("start", "judges/c")])
        self.assertEqual(self.backend.calls, 2)

    def test_failure_keeps_first_judge_and_resume_does_not_regenerate_or_rescore_it(
        self,
    ):
        self.fail_model = "judges/b"
        with self.assertRaisesRegex(RuntimeError, "judge unavailable"):
            self.evaluate()
        saved = json.loads((self.run_root / "run_summary.json").read_text())
        self.assertEqual(list(saved["judges"]), ["judges/a"])
        self.assertFalse(saved["evaluation_complete"])
        self.assertEqual(
            saved["judges"]["judges/a"]["results"]["toy"]["primary_score"], 25
        )
        self.assertEqual(self.events[-1], ("close", "judges/b"))
        self.events.clear()
        self.fail_model = None
        self.assertTrue(self.evaluate()["evaluation_complete"])
        self.assertEqual(
            [event for event in self.events if event[0] == "start"],
            [("start", "judges/b")],
        )
        self.assertEqual(self.backend.calls, 2)

    def test_eval_only_rejudges_each_model_without_generation(self):
        self.evaluate(generate_only=True)
        self.assertFalse((self.run_root / "judges").exists())
        self.create_backend.reset_mock()
        self.events.clear()
        self.assertTrue(self.evaluate(eval_only=True)["evaluation_complete"])
        self.create_backend.assert_not_called()
        self.assertNotIn("generate", self.events)
        self.assertEqual(
            [event for event in self.events if event[0] == "start"],
            [("start", "judges/a"), ("start", "judges/b")],
        )
        self.events.clear()
        self.evaluate(eval_only=True)
        self.assertEqual(sum(event[0] == "score" for event in self.events), 4)

    def test_incomplete_judge_is_reported_and_only_its_unscored_rows_retried(self):
        self.unscored_model = "judges/b"
        result = self.evaluate()
        self.assertFalse(result["evaluation_complete"])
        self.assertEqual(result["judges"]["judges/a"]["primary_score_aggregate"], 62.5)
        self.assertIsNone(result["judges"]["judges/b"]["primary_score_aggregate"])
        self.events.clear()
        self.unscored_model = None
        self.assertTrue(self.evaluate()["evaluation_complete"])
        self.assertEqual(
            [event for event in self.events if event[0] == "start"],
            [("start", "judges/b")],
        )

    def test_api_judges_receive_independent_model_settings(self):
        result = self.evaluate(
            metric_options={"judge_models": ["judges/a", "judges/b"]}
        )
        self.assertTrue(result["evaluation_complete"])
        self.assertFalse(any(isinstance(event, tuple) for event in self.events))
        self.assertEqual(
            result["judges"]["judges/a"]["results"]["toy"]["primary_score"], 25
        )
        self.assertEqual(
            result["judges"]["judges/b"]["results"]["toy"]["primary_score"], 75
        )

    def test_repeats_share_all_generations_and_resume_independently(self):
        result = self.evaluate(num_repeats=2)
        self.assertEqual(self.backend.calls, 4)
        for model in result["judges"]:
            for repeat in ("run_01", "run_02"):
                source = self.run_root / "toy" / repeat / "predictions.jsonl"
                scored = (
                    judge_directory(self.run_root, model)
                    / "toy"
                    / repeat
                    / "predictions.jsonl"
                )
                self.assertEqual(
                    [row["generation"] for row in _rows(source)],
                    [row["generation"] for row in _rows(scored)],
                )
        self.events.clear()
        self.assertTrue(self.evaluate(num_repeats=2)["evaluation_complete"])
        self.assertEqual(self.events, [])

    def test_cached_different_generation_or_generation_settings_rejected(self):
        self.evaluate()
        directory = judge_directory(self.run_root, "judges/a") / "toy"
        predictions = directory / "predictions.jsonl"
        original = predictions.read_text()
        rows = _rows(predictions)
        rows[0]["generation"] = "a different answer"
        predictions.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        with self.assertRaisesRegex(ValueError, "cache generations differ"):
            self.evaluate()
        predictions.write_text(original)
        manifest = directory / "run_config.json"
        config = json.loads(manifest.read_text())
        config["generation_config"]["temperature"] = 0.123
        manifest.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "cache generation settings differ"):
            self.evaluate()
        self.assertTrue(self.evaluate(overwrite=True)["evaluation_complete"])

    def test_no_judged_tasks_rejected_before_generation(self):
        with self.assertRaisesRegex(ValueError, "at least one LLM-judge task"):
            self.evaluate(tasks="toy2")
        self.create_backend.assert_not_called()


class MultiJudgeConfigTests(unittest.TestCase):
    def resolve(self, flags=(), cfg=None):
        args = build_parser().parse_args(["--model", "candidate", *flags])
        return resolve_run_arguments(args, cfg or {})["metric_options"]

    def test_cli_yaml_and_cli_overrides(self):
        judges = ["judges/a", "judges/b"]
        self.assertEqual(
            self.resolve(["--judge-models", ",".join(judges)])["judge_models"], judges
        )
        self.assertEqual(
            self.resolve(cfg={"metrics": {"judge_models": judges}})["judge_models"],
            judges,
        )
        options = self.resolve(
            ["--judge-models", ",".join(judges)], {"metrics": {"judge_model": "old"}}
        )
        self.assertNotIn("judge_model", options)
        options = self.resolve(
            ["--judge-model", "new"], {"metrics": {"judge_models": judges}}
        )
        self.assertEqual(options["judge_model"], "new")
        self.assertNotIn("judge_models", options)

    def test_invalid_judge_lists_and_ambiguous_settings(self):
        for value in ([], "", "a,,b", "a,", ["a", "a"], [""], [None], ["a", " a "]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_judge_models(value)
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            self.resolve(["--judge-model", "a", "--judge-models", "a,b"])
        with self.assertRaisesRegex(ValueError, "Use either"):
            self.resolve(
                cfg={"metrics": {"judge_model": "a", "judge_models": ["a", "b"]}}
            )

    def test_directory_names_distinguish_full_model_identity(self):
        self.assertNotEqual(
            judge_directory(Path("out"), "group-a/model"),
            judge_directory(Path("out"), "group-b/model"),
        )

    def test_cli_returns_multi_judge_summary_unchanged(self):
        args = build_parser().parse_args(
            [
                "--model",
                "candidate",
                "--tasks",
                "writingbench",
                "--judge-models",
                "a,b",
            ]
        )
        resolved = resolve_run_arguments(args, {})
        result = {"judges": {"a": {}, "b": {}}, "evaluation_complete": True}
        with tempfile.TemporaryDirectory() as tmp:
            resolved["output_dir"] = tmp
            with mock.patch("aethereval.cli.run_evaluation", return_value=result):
                self.assertIs(run_selected_tasks(args, resolved), result)

    def test_multi_judge_rejects_external_tasks_before_starting(self):
        args = build_parser().parse_args(
            ["--model", "candidate", "--tasks", "bfcl", "--judge-models", "a,b"]
        )
        resolved = resolve_run_arguments(args, {})
        with mock.patch("aethereval.cli.run_evaluation") as run:
            with self.assertRaisesRegex(ValueError, "native tasks only"):
                run_selected_tasks(args, resolved)
            run.assert_not_called()
