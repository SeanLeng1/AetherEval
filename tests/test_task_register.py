import ast
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from aethereval.core.task_register import (
    discover_tasks,
    list_task_default_gens,
    list_tasks,
    load_task,
    _validate_metrics_contract,
)


class TaskRegisterTests(unittest.TestCase):
    def test_single_and_batch_scoring_contract(self):
        from types import SimpleNamespace

        for scoring in ("score_generation", "score_generations_batch"):
            with self.subTest(scoring=scoring):
                module = SimpleNamespace(__name__="test", aggregate=lambda *args: {},
                                         **{scoring: lambda *args: []})
                _validate_metrics_contract(module)
        with self.assertRaises(ValueError):
            _validate_metrics_contract(SimpleNamespace(__name__="test", aggregate=lambda: {}))

    def test_defaults_follow_benchmark_directory_order(self):
        from aethereval.cli import EXTERNAL_TASKS
        from aethereval.core.task_defaults import _load_task_default_overrides

        names = list(_load_task_default_overrides())
        self.assertEqual(names, sorted(names))
        # One YAML entry per benchmark directory (plus external adapters), no strays.
        self.assertEqual(set(names), set(list_tasks()) | set(EXTERNAL_TASKS))

    def test_canonical_names_and_legacy_aliases(self):
        from aethereval.core.task_register import parse_task_names

        names = list_tasks()
        self.assertTrue(all("_" not in name for name in names))
        self.assertTrue(all(spec.task_dir.name == name for name, spec in discover_tasks().items()))
        self.assertEqual(
            parse_task_names("safe_alignment,safe-alignment", names), ["safe-alignment"]
        )
        bundle = load_task("safe_alignment")
        self.assertEqual(bundle.spec.name, "safe-alignment")
        self.assertEqual(bundle.task_module.TASK_NAME, "safe-alignment")

    def _read_primary_metric(self, metrics_path: Path) -> str:
        tree = ast.parse(metrics_path.read_text(encoding="utf-8"))
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "PRIMARY_METRIC":
                    if isinstance(node.value, ast.Constant) and isinstance(
                        node.value.value, str
                    ):
                        return node.value.value
                    raise AssertionError(
                        f"{metrics_path} PRIMARY_METRIC must be a string literal"
                    )
        raise AssertionError(f"{metrics_path} does not define PRIMARY_METRIC")

    def test_ifeval_task_discoverable(self) -> None:
        tasks = list_tasks()
        self.assertIn("ifeval", tasks)
        self.assertIn("gpqa-diamond", tasks)
        self.assertIn("aime24", tasks)
        self.assertIn("aime25", tasks)
        self.assertIn("amc23", tasks)
        self.assertIn("math500", tasks)
        self.assertIn("minerva", tasks)
        self.assertIn("olympiad-bench", tasks)
        self.assertIn("safe-alignment", tasks)
        self.assertIn("mmlu-pro", tasks)
        self.assertIn("agieval-en", tasks)
        self.assertIn("bbh", tasks)
        self.assertIn("ifbench", tasks)
        self.assertIn("humaneval-plus", tasks)
        self.assertIn("mbpp-plus", tasks)
        self.assertIn("zebralogic", tasks)
        self.assertIn("livecodebench", tasks)
        self.assertNotIn("qampari-oracle5", tasks)

    def test_contract_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bad_task_dir = root / "bad-task"
            bad_task_dir.mkdir(parents=True, exist_ok=True)
            (bad_task_dir / "task.py").write_text(
                "TASK_NAME='bad_task'\n"
                "DATA_FILE='data.json'\n"
                "def load_samples(task_dir):\n"
                "    return []\n"
                "def build_prompt(sample):\n"
                "    return ''\n",
                encoding="utf-8",
            )
            (bad_task_dir / "metrics.py").write_text(
                "def score_generation(sample, generation):\n"
                "    return {'score': 1.0}\n"
                "def aggregate(sample_results, metric_options=None):\n"
                "    return {'x': 1.0}\n",
                encoding="utf-8",
            )

            tasks = discover_tasks(root)
            self.assertIn("bad-task", tasks)
            with self.assertRaises(ValueError):
                load_task("bad-task", root)

    def test_contract_rejects_default_gen(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_dir = root / "ok-task"
            task_dir.mkdir(parents=True, exist_ok=True)
            (task_dir / "task.py").write_text(
                "TASK_NAME='ok_task'\n"
                "DATA_FILE='data/eval.jsonl'\n"
                "def load_samples(task_dir):\n"
                "    return []\n"
                "def build_prompt(sample):\n"
                "    return ''\n",
                encoding="utf-8",
            )
            (task_dir / "metrics.py").write_text(
                "def score_generation(sample, generation):\n"
                "    return {'score': 1.0}\n"
                "def aggregate(sample_results, metric_options=None):\n"
                "    return {'x': 1.0}\n",
                encoding="utf-8",
            )

            bundle = load_task("ok-task", root)
            self.assertFalse(hasattr(bundle.task_module, "DEFAULT_GEN"))
            task_path = task_dir / "task.py"
            task_path.write_text(
                "DEFAULT_GEN={'n': 1}\n" + task_path.read_text(), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "configs/task_defaults.yaml"):
                load_task("ok-task", root)

    def test_task_defaults_use_known_generation_keys(self) -> None:
        from aethereval.core.task_defaults import (
            GENERATION_KEYS,
            resolve_task_default_gen,
        )

        # The YAML task set itself is checked in test_defaults_follow_benchmark_directory_order.
        for name in list_tasks():
            defaults = resolve_task_default_gen(name)
            self.assertLessEqual(set(defaults), GENERATION_KEYS, name)
            for key in ("n", "max_new_tokens", "temperature", "top_p"):
                self.assertIn(key, defaults, name)

        typo = {"math500": {"n": 16, "max_new_token": 16384, "temperature": 0.6}}
        with mock.patch(
            "aethereval.core.task_defaults._load_task_default_overrides",
            return_value=typo,
        ):
            with self.assertRaisesRegex(ValueError, r"'math500'.*\['max_new_token'\]"):
                load_task("math500")

    def test_list_task_default_gens_does_not_need_evalplus(self) -> None:
        # EvalPlus is a scoring-only install; task.py modules must load without it.
        code = (
            "import json, sys\n"
            "sys.modules['evalplus'] = None\n"
            "from aethereval.core.task_register import list_task_default_gens\n"
            "print(json.dumps(sorted(list_task_default_gens())))\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), list_tasks())

    def test_list_task_default_gens(self) -> None:
        # Values are user-owned in configs/task_defaults.yaml; test shape, not numbers.
        defaults = list_task_default_gens()
        self.assertEqual(set(defaults), set(list_tasks()))
        for task, gen in defaults.items():
            with self.subTest(task=task):
                self.assertIsInstance(gen["n"], int)
                self.assertGreaterEqual(gen["n"], 1)
                self.assertIsInstance(gen["max_new_tokens"], int)
                self.assertGreater(gen["max_new_tokens"], 0)
                self.assertGreaterEqual(gen["temperature"], 0.0)
                self.assertTrue(0.0 < gen.get("top_p", 1.0) <= 1.0)
                if gen["n"] > 1:
                    self.assertGreater(gen["temperature"], 0.0)
                self.assertNotIn("metrics", gen)
                self.assertNotIn("judge_model", gen)
        # Every math task shares one prompt, so one long-reasoning profile.
        for task in ("aime24", "aime25", "amc23", "math500", "minerva", "olympiad-bench"):
            self.assertEqual(defaults[task], defaults["aime24"], task)
        # Deliberate: few-shot/base-model stop strings would truncate chat answers.
        self.assertNotIn("stop", defaults["livecodebench"])
        self.assertNotIn("stop", defaults["mmlu-pro"])

    def test_readmes_state_configured_defaults(self) -> None:
        from aethereval.core.task_defaults import _load_task_default_overrides

        benchmarks = Path(__file__).resolve().parents[1] / "benchmarks"
        for task, defaults in _load_task_default_overrides().items():
            text = " ".join((benchmarks / task / "README.md").read_text("utf-8").split())
            with self.subTest(task=task):
                self.assertIn("## Official source and protocol", text)
            # A regex, not a substring: n=1 must not match n=16 or judge_max_new_tokens.
            for key in ("n", "max_new_tokens"):
                with self.subTest(task=task, key=key):
                    self.assertRegex(text, rf"(?<![\w]){key}={defaults[key]}(?![\w.])")

    def test_instruction_following_primary_metrics(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        ifeval_metric = self._read_primary_metric(
            repo_root / "benchmarks" / "ifeval" / "metrics.py"
        )
        ifbench_metric = self._read_primary_metric(
            repo_root / "benchmarks" / "ifbench" / "metrics.py"
        )

        self.assertEqual(ifeval_metric, "prompt_level_loose_acc")
        self.assertEqual(ifbench_metric, "prompt_level_loose_acc")


if __name__ == "__main__":
    unittest.main()
