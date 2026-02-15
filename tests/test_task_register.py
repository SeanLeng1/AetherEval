from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from aethereval.task_register import discover_tasks, list_tasks, load_task


class TaskRegisterTests(unittest.TestCase):
    def test_ifeval_task_discoverable(self) -> None:
        tasks = list_tasks()
        self.assertIn("ifeval", tasks)
        self.assertIn("gpqa_diamond", tasks)
        self.assertIn("aime24", tasks)
        self.assertIn("aime25", tasks)
        self.assertIn("mmlu_pro", tasks)
        self.assertIn("agieval_en", tasks)
        self.assertIn("ifbench", tasks)
        self.assertIn("humaneval_plus", tasks)
        self.assertIn("zebralogic", tasks)
        self.assertIn("livecodebench", tasks)

    def test_contract_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bad_task_dir = root / "bad_task"
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
            self.assertIn("bad_task", tasks)
            with self.assertRaises(ValueError):
                load_task("bad_task", root)


if __name__ == "__main__":
    unittest.main()
