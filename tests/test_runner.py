from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from aethereval.core.runner import inspect_prompts, run_evaluation
from aethereval.core.types import GenerationInput, GenerationOutput


class FakeBackend:
    def __init__(self) -> None:
        self.calls = 0
        self.last_gen_cfg: dict | None = None

    def generate(self, inputs: list[GenerationInput], gen_cfg: dict) -> list[GenerationOutput]:
        self.calls += 1
        self.last_gen_cfg = dict(gen_cfg)
        outputs: list[GenerationOutput] = []
        for item in inputs:
            prompt = item.prompt if isinstance(item.prompt, str) else str(item.prompt)
            if "2 + 2" in prompt:
                answer = "4"
            elif "capital of France" in prompt:
                answer = "paris"
            else:
                answer = "unknown"
            outputs.append(
                GenerationOutput(
                    sample_id=item.sample_id,
                    prompt=item.prompt,
                    generations=[answer for _ in range(item.num_generations)],
                )
            )
        return outputs

    def close(self) -> None:
        return None


class NeverCalledBackend(FakeBackend):
    def generate(self, inputs: list[GenerationInput], gen_cfg: dict) -> list[GenerationOutput]:
        raise AssertionError("generate should not be called during full resume")


def _write_toy_benchmark(root: Path) -> None:
    task_dir = root / "toy"
    data_dir = task_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        {"id": "1", "question": "2 + 2", "answer": "4"},
        {"id": "2", "question": "capital of France", "answer": "paris"},
    ]
    with (data_dir / "eval.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    (task_dir / "task.py").write_text(
        "from __future__ import annotations\n"
        "import json\n"
        "from pathlib import Path\n"
        "from aethereval.core.types import Sample\n"
        "TASK_NAME='toy'\n"
        "DATA_FILE='data/eval.jsonl'\n"
        "DEFAULT_GEN={'n': 1, 'max_new_tokens': 16, 'temperature': 0.0, 'top_p': 1.0}\n"
        "def load_samples(task_dir: Path):\n"
        "    rows = []\n"
        "    with (task_dir / DATA_FILE).open('r', encoding='utf-8') as f:\n"
        "        for line in f:\n"
        "            line = line.strip()\n"
        "            if not line:\n"
        "                continue\n"
        "            rows.append(json.loads(line))\n"
        "    out = []\n"
        "    for row in rows:\n"
        "        out.append(Sample(id=str(row['id']), gold=row['answer'], meta={'question': row['question']}, data={'question': row['question']}))\n"
        "    return out\n"
        "def build_prompt(sample: Sample):\n"
        "    return f\"Question: {sample.data['question']}\\nAnswer:\"\n",
        encoding="utf-8",
    )

    (task_dir / "metrics.py").write_text(
        "from __future__ import annotations\n"
        "def score_generation(sample, generation):\n"
        "    pred = generation.strip().lower()\n"
        "    gold = str(sample.gold).strip().lower()\n"
        "    return {'score': 1.0 if pred == gold else 0.0}\n"
        "def aggregate(sample_results, metric_options=None):\n"
        "    first_scores = [float(item['scores'][0]) if item.get('scores') else 0.0 for item in sample_results]\n"
        "    return {'accuracy_first': sum(first_scores)/len(first_scores) if first_scores else 0.0}\n",
        encoding="utf-8",
    )


def _write_toy2_benchmark(root: Path) -> None:
    task_dir = root / "toy2"
    data_dir = task_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        {"id": "1", "question": "whoami", "answer": "unknown"},
        {"id": "2", "question": "name", "answer": "unknown"},
    ]
    with (data_dir / "eval.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    (task_dir / "task.py").write_text(
        "from __future__ import annotations\n"
        "import json\n"
        "from pathlib import Path\n"
        "from aethereval.core.types import Sample\n"
        "TASK_NAME='toy2'\n"
        "DATA_FILE='data/eval.jsonl'\n"
        "DEFAULT_GEN={'n': 1, 'max_new_tokens': 16, 'temperature': 0.0, 'top_p': 1.0}\n"
        "def load_samples(task_dir: Path):\n"
        "    rows = []\n"
        "    with (task_dir / DATA_FILE).open('r', encoding='utf-8') as f:\n"
        "        for line in f:\n"
        "            line = line.strip()\n"
        "            if not line:\n"
        "                continue\n"
        "            rows.append(json.loads(line))\n"
        "    out = []\n"
        "    for row in rows:\n"
        "        out.append(Sample(id=str(row['id']), gold=row['answer'], meta={'question': row['question']}, data={'question': row['question']}))\n"
        "    return out\n"
        "def build_prompt(sample: Sample):\n"
        "    return f\"Question: {sample.data['question']}\\nAnswer:\"\n",
        encoding="utf-8",
    )

    (task_dir / "metrics.py").write_text(
        "from __future__ import annotations\n"
        "def score_generation(sample, generation):\n"
        "    pred = generation.strip().lower()\n"
        "    gold = str(sample.gold).strip().lower()\n"
        "    return {'score': 1.0 if pred == gold else 0.0}\n"
        "def aggregate(sample_results, metric_options=None):\n"
        "    first_scores = [float(item['scores'][0]) if item.get('scores') else 0.0 for item in sample_results]\n"
        "    return {'accuracy_first': sum(first_scores)/len(first_scores) if first_scores else 0.0}\n",
        encoding="utf-8",
    )


class RunnerTests(unittest.TestCase):
    def test_end_to_end_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "benchmarks"
            _write_toy_benchmark(root)

            out = Path(tmp) / "outputs"
            backend = FakeBackend()
            first = run_evaluation(
                model="fake-model",
                tasks="toy",
                output_dir=out,
                run_id="run1",
                backend=backend,
                benchmarks_dir=root,
            )
            self.assertIn("toy", first["results"])
            summary = first["results"]["toy"]
            self.assertEqual(summary["new_records"], 2)
            self.assertAlmostEqual(summary["metrics"]["accuracy_first"], 1.0, places=6)
            self.assertEqual(summary["primary_metric"], "accuracy_first")
            self.assertAlmostEqual(float(summary["primary_score"]), 1.0, places=6)
            self.assertAlmostEqual(
                first["summary"]["metrics"]["accuracy_first"],
                1.0,
                places=6,
            )
            self.assertIn("primary_scores", first)
            self.assertEqual(first["primary_scores"]["toy"]["metric"], "accuracy_first")
            self.assertAlmostEqual(float(first["primary_scores"]["toy"]["score"]), 1.0, places=6)
            self.assertAlmostEqual(float(first["primary_score_aggregate"]), 1.0, places=6)
            predictions_path = out / "run1" / "toy" / "predictions.jsonl"
            with predictions_path.open("r", encoding="utf-8") as f:
                first_row = json.loads(f.readline())
            self.assertIsInstance(first_row["prompt"], list)
            self.assertEqual(first_row["prompt"][0]["role"], "user")
            self.assertIn("Question: 2 + 2", first_row["prompt"][0]["content"])

            resume_backend = NeverCalledBackend()
            second = run_evaluation(
                model="fake-model",
                tasks="toy",
                output_dir=out,
                run_id="run1",
                backend=resume_backend,
                benchmarks_dir=root,
            )
            summary2 = second["results"]["toy"]
            self.assertEqual(summary2["new_records"], 0)
            self.assertEqual(summary2["existing_records"], 2)

    def test_generation_overrides_take_precedence_over_task_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "benchmarks"
            _write_toy_benchmark(root)

            out = Path(tmp) / "outputs"
            backend = FakeBackend()
            run_evaluation(
                model="fake-model",
                tasks="toy",
                output_dir=out,
                run_id="run_override",
                backend=backend,
                gen_overrides={"max_new_tokens": 99, "top_p": 0.8},
                benchmarks_dir=root,
            )
            assert backend.last_gen_cfg is not None
            self.assertEqual(backend.last_gen_cfg["max_new_tokens"], 99)
            self.assertAlmostEqual(float(backend.last_gen_cfg["top_p"]), 0.8, places=6)
            self.assertEqual(backend.last_gen_cfg["n"], 1)
            self.assertEqual(int(backend.last_gen_cfg["top_k"]), -1)

    def test_default_run_id_uses_model_suffix_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "benchmarks"
            _write_toy_benchmark(root)

            out = Path(tmp) / "outputs"
            backend = FakeBackend()
            result = run_evaluation(
                model="Qwen/Qwen3-0.6B-Base",
                tasks="toy",
                output_dir=out,
                backend=backend,
                benchmarks_dir=root,
            )

            run_id = str(result["run_id"])
            self.assertEqual(run_id, "qwen3-0.6b-base")
            self.assertTrue((out / run_id / "run_summary.json").exists())

    def test_overwrite_rebuilds_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "benchmarks"
            _write_toy_benchmark(root)
            out = Path(tmp) / "outputs"

            run_evaluation(
                model="fake-model",
                tasks="toy",
                output_dir=out,
                run_id="same_run",
                backend=FakeBackend(),
                benchmarks_dir=root,
            )
            rebuilt = run_evaluation(
                model="fake-model",
                tasks="toy",
                output_dir=out,
                run_id="same_run",
                backend=FakeBackend(),
                overwrite=True,
                benchmarks_dir=root,
            )
            summary = rebuilt["results"]["toy"]
            self.assertEqual(summary["existing_records"], 0)
            self.assertEqual(summary["new_records"], 2)

    def test_n_gt_1_with_zero_temperature_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "benchmarks"
            _write_toy_benchmark(root)

            out = Path(tmp) / "outputs"
            with self.assertRaises(ValueError):
                run_evaluation(
                    model="fake-model",
                    tasks="toy",
                    output_dir=out,
                    run_id="run2",
                    backend=FakeBackend(),
                    gen_overrides={"n": 2, "temperature": 0.0},
                    benchmarks_dir=root,
                )

    def test_inspect_prompts_without_inference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "benchmarks"
            _write_toy_benchmark(root)

            def _render(prompt):  # noqa: ANN001
                if isinstance(prompt, list):
                    return "\n".join(f"{m['role']}: {m['content']}" for m in prompt)
                return str(prompt)

            inspected = inspect_prompts(
                model="fake-model",
                tasks="toy",
                benchmarks_dir=root,
                prompt_renderer=_render,
            )
            self.assertEqual(inspected["tasks"], ["toy"])
            rows = inspected["results"]["toy"]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["sample_id"], "1")
            self.assertIn("Question: 2 + 2", rows[0]["prompt"])

    def test_run_summary_includes_existing_tasks_under_same_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "benchmarks"
            _write_toy_benchmark(root)
            _write_toy2_benchmark(root)
            out = Path(tmp) / "outputs"

            run_evaluation(
                model="fake-model",
                tasks="toy",
                output_dir=out,
                run_id="run_merge",
                backend=FakeBackend(),
                benchmarks_dir=root,
            )
            second = run_evaluation(
                model="fake-model",
                tasks="toy2",
                output_dir=out,
                run_id="run_merge",
                backend=FakeBackend(),
                benchmarks_dir=root,
            )

            self.assertEqual(second["selected_tasks"], ["toy2"])
            self.assertEqual(second["tasks"], ["toy", "toy2"])
            self.assertIn("toy", second["results"])
            self.assertIn("toy2", second["results"])
            self.assertEqual(second["summary"]["num_tasks"], 2)
            self.assertAlmostEqual(
                second["summary"]["metrics"]["accuracy_first"],
                1.0,
                places=6,
            )
            self.assertAlmostEqual(float(second["primary_score_aggregate"]), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
