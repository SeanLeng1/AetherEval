import ast
import tempfile
import unittest
from pathlib import Path

from aethereval.core.task_register import (
    discover_tasks,
    list_task_default_gens,
    list_tasks,
    load_task,
)


class TaskRegisterTests(unittest.TestCase):
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
        self.assertIn("apibank", tasks)
        self.assertIn("mmlu-pro", tasks)
        self.assertIn("agieval-en", tasks)
        self.assertIn("bbh", tasks)
        self.assertIn("ifbench", tasks)
        self.assertIn("humaneval-plus", tasks)
        self.assertIn("zebralogic", tasks)
        self.assertIn("livecodebench", tasks)
        self.assertNotIn("qampari-oracle5", tasks)
        self.assertIn("nq-open", tasks)
        self.assertIn("triviaqa", tasks)

    def test_contract_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bad_task_dir = root / "bad-task"
            bad_task_dir.mkdir(parents=True, exist_ok=True)
            (bad_task_dir / "task.py").write_text(
                "TASK_NAME='bad_task'\n"
                "DATA_FILE='data.json'\n"
                "DEFAULT_GEN={}\n"
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

    def test_contract_allows_missing_default_gen(self) -> None:
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
            self.assertEqual(bundle.task_module.DEFAULT_GEN, {})

    def test_list_task_default_gens(self) -> None:
        defaults = list_task_default_gens()
        self.assertIn("ifeval", defaults)
        self.assertIn("gpqa-diamond", defaults)
        self.assertIn("bbh", defaults)
        self.assertEqual(defaults["ifeval"]["n"], 1)
        self.assertEqual(defaults["bbh"]["n"], 1)
        self.assertEqual(defaults["aime24"]["n"], 16)
        self.assertEqual(defaults["amc23"]["n"], 16)
        self.assertEqual(defaults["math500"]["n"], 16)
        self.assertEqual(defaults["minerva"]["n"], 16)
        self.assertEqual(defaults["olympiad-bench"]["n"], 16)
        self.assertEqual(defaults["safe-alignment"]["n"], 4)
        self.assertEqual(defaults["safe-alignment"]["max_new_tokens"], 1024)
        self.assertEqual(defaults["apibank"]["n"], 1)
        self.assertEqual(defaults["apibank"]["max_new_tokens"], 4096)
        self.assertNotIn("metrics", defaults["healthbench"])
        self.assertNotIn("judge_model", defaults["healthbench"])
        self.assertIn("max_new_tokens", defaults["livecodebench"])
        self.assertEqual(defaults["nq-open"]["n"], 1)
        self.assertEqual(defaults["triviaqa"]["n"], 1)

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
