import json
import tempfile
import unittest
from pathlib import Path

from aethereval.core.runner import run_evaluation
from tests.test_runner import FakeBackend, _write_toy_benchmark


class ResumeIdentityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "benchmarks"
        _write_toy_benchmark(self.root)
        self.backend = FakeBackend()
        self.options = {
            "model": "model-a", "model_name": "shared", "tasks": "toy",
            "output_dir": Path(self.tmp.name) / "outputs", "run_id": "identity",
            "backend": self.backend, "benchmarks_dir": self.root,
            "gen_overrides": {"n": 1, "temperature": 0.0},
        }
        self.task_output = self.options["output_dir"] / "shared" / "identity" / "toy"

    def generate(self, **overrides):
        return run_evaluation(**{**self.options, "generate_only": True, **overrides})

    def test_resume_rejects_a_different_model_or_sampling_configuration(self):
        self.generate()
        paths = [self.task_output / name for name in ("run_config.json", "predictions.jsonl")]
        originals = [path.read_bytes() for path in paths]
        for overrides in ({"model": "model-b"}, {"gen_overrides": {"n": 1, "temperature": 0.8}}):
            with self.subTest(overrides=overrides), self.assertRaisesRegex(ValueError, "settings differ"):
                self.generate(**overrides)
            self.assertEqual([path.read_bytes() for path in paths], originals)
        self.assertEqual(self.backend.calls, 1)

    def test_resume_and_eval_only_reject_changed_gold_or_prompt(self):
        self.generate()
        data_path = self.root / "toy" / "data" / "eval.jsonl"
        original_data = data_path.read_text()
        records_path = self.task_output / "predictions.jsonl"
        original_records = records_path.read_bytes()
        for key, value in (("answer", "28"), ("question", "7 * 4")):
            rows = [json.loads(line) for line in original_data.splitlines()]
            rows[0][key] = value
            data_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            for eval_only in (False, True):
                with self.subTest(key=key, eval_only=eval_only), self.assertRaisesRegex(ValueError, "data or prompts differ"):
                    run_evaluation(**self.options, generate_only=not eval_only, eval_only=eval_only)
            self.assertEqual(records_path.read_bytes(), original_records)
        self.assertEqual(self.backend.calls, 1)

    def test_same_configuration_resumes_without_regeneration(self):
        self.generate()
        result = run_evaluation(**self.options)
        self.assertEqual(self.backend.calls, 1)
        self.assertEqual(result["results"]["toy"]["primary_score"], 100.0)
        self.assertEqual(result["results"]["toy"]["raw_primary_score"], 1.0)

    def test_legacy_predictions_require_matching_prompt_and_gold(self):
        self.generate()
        config_path = self.task_output / "run_config.json"
        config = json.loads(config_path.read_text())
        config.pop("task_fingerprint")
        config_path.write_text(json.dumps(config))
        run_evaluation(**self.options, eval_only=True)
        config = json.loads(config_path.read_text())
        self.assertIn("task_fingerprint", config)
        config.pop("task_fingerprint")
        config_path.write_text(json.dumps(config))
        records_path = self.task_output / "predictions.jsonl"
        rows = [json.loads(line) for line in records_path.read_text().splitlines()]
        rows[0]["prompt"][0]["content"] = "Different problem"
        records_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        with self.assertRaisesRegex(ValueError, "prompt or gold differs"):
            run_evaluation(**self.options, eval_only=True)

    def test_partial_resume_rejects_a_different_generation_backend(self):
        self.generate()
        records_path = self.task_output / "predictions.jsonl"
        first_line = records_path.read_text().splitlines()[0]
        records_path.write_text(first_line + "\n")
        self.backend.name = "different-backend"
        with self.assertRaisesRegex(ValueError, "settings differ"):
            self.generate()
        self.assertEqual(self.backend.calls, 1)

    def test_interrupted_generation_keeps_its_manifest(self):
        class InterruptedBackend(FakeBackend):
            def generate(self, inputs, gen_cfg):
                raise RuntimeError("inference interrupted")

        with self.assertRaisesRegex(RuntimeError, "inference interrupted"):
            self.generate(backend=InterruptedBackend())
        config = json.loads((self.task_output / "run_config.json").read_text())
        self.assertEqual(config["model"], "model-a")
        self.assertEqual(config["generation_config"]["temperature"], 0.0)
        self.assertEqual(len(config["task_fingerprint"]), 64)


if __name__ == "__main__":
    unittest.main()
