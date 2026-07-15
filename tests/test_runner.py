import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from aethereval.core.runner import inspect_prompts, run_evaluation
from aethereval.core.types import GenerationInput, GenerationOutput


class FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[str]:
        del add_special_tokens
        return text.split()

    def apply_chat_template(
        self,
        prompt: list[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ) -> str:
        del tokenize, add_generation_prompt
        return "\n".join(f"{message['role']}: {message['content']}" for message in prompt)


class FakeBackend:
    def __init__(self) -> None:
        self.calls = 0
        self.last_gen_cfg: dict | None = None
        self._tokenizer = FakeTokenizer()

    def generate(
        self, inputs: list[GenerationInput], gen_cfg: dict
    ) -> list[GenerationOutput]:
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
                    meta={
                        "prompt_token_count": len(str(item.prompt).split()),
                        "response_token_counts": [
                            len(answer.split()) for _ in range(item.num_generations)
                        ],
                    },
                )
            )
        return outputs

    def close(self) -> None:
        return None


class NeverCalledBackend(FakeBackend):
    def generate(
        self, inputs: list[GenerationInput], gen_cfg: dict
    ) -> list[GenerationOutput]:
        raise AssertionError("generate should not be called during full resume")


class ShortGenerationBackend(FakeBackend):
    def generate(
        self, inputs: list[GenerationInput], gen_cfg: dict
    ) -> list[GenerationOutput]:
        item = inputs[0]
        return [
            GenerationOutput(
                sample_id=item.sample_id,
                prompt=item.prompt,
                generations=[],
            )
        ]


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
        "def score_generation(sample, generation):\n"
        "    pred = generation.strip().lower()\n"
        "    gold = str(sample.gold).strip().lower()\n"
        "    return {'score': 1.0 if pred == gold else 0.0}\n"
        "def aggregate(sample_results, metric_options=None):\n"
        "    first_scores = [float(item['scores'][0]) if item.get('scores') else 0.0 for item in sample_results]\n"
        "    return {'accuracy_first': sum(first_scores)/len(first_scores) if first_scores else 0.0}\n",
        encoding="utf-8",
    )


def _write_batch_benchmark(root: Path) -> None:
    task_dir = root / "batch_toy"
    data_dir = task_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        {"id": "1", "question": "batch one"},
        {"id": "2", "question": "batch two"},
    ]
    with (data_dir / "eval.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    (task_dir / "task.py").write_text(
        "import json\n"
        "from pathlib import Path\n"
        "from aethereval.core.types import Sample\n"
        "TASK_NAME='batch_toy'\n"
        "DATA_FILE='data/eval.jsonl'\n"
        "DEFAULT_GEN={'n': 1, 'max_new_tokens': 16, 'temperature': 0.0, 'top_p': 1.0}\n"
        "def load_samples(task_dir: Path):\n"
        "    rows = []\n"
        "    with (task_dir / DATA_FILE).open('r', encoding='utf-8') as f:\n"
        "        for line in f:\n"
        "            line = line.strip()\n"
        "            if line:\n"
        "                rows.append(json.loads(line))\n"
        "    return [Sample(id=str(row['id']), data={'question': row['question']}) for row in rows]\n"
        "def build_prompt(sample: Sample):\n"
        "    return sample.data['question']\n",
        encoding="utf-8",
    )

    (task_dir / "metrics.py").write_text(
        "def score_generation(sample, generation):\n"
        "    raise AssertionError('single-generation scorer should not be called')\n"
        "def score_generations_batch(samples, generation_outputs, metric_options=None):\n"
        "    offset = float((metric_options or {}).get('batch_offset', 0.0))\n"
        "    results = []\n"
        "    for sample, output in zip(samples, generation_outputs):\n"
        "        if sample.id != output.sample_id:\n"
        "            raise ValueError('sample/output mismatch')\n"
        "        results.append([\n"
        "            {'score': len(text) + offset, 'is_pass': True, 'meta': {'batch': True}}\n"
        "            for text in output.generations\n"
        "        ])\n"
        "    return results\n"
        "def aggregate(sample_results, metric_options=None):\n"
        "    first_scores = [float(item['scores'][0]) for item in sample_results]\n"
        "    return {'batch_mean': sum(first_scores)/len(first_scores)}\n",
        encoding="utf-8",
    )


class RunnerTests(unittest.TestCase):
    def test_generate_only_then_eval_only_without_candidate_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "benchmarks"
            _write_toy_benchmark(root)
            metrics_path = root / "toy" / "metrics.py"
            metrics_source = metrics_path.read_text(encoding="utf-8")
            metrics_path.write_text(
                "def validate_metric_options(options):\n"
                "    raise AssertionError('generate-only must not validate metrics')\n"
                + metrics_source,
                encoding="utf-8",
            )
            out = Path(tmp) / "outputs"

            backend = FakeBackend()
            backend.name = "offline-test-backend"
            generated = run_evaluation(
                model="offline/model",
                model_name="candidate",
                tasks="toy",
                output_dir=out,
                run_id="split_run",
                backend=backend,
                benchmarks_dir=root,
                generate_only=True,
                gen_overrides={"n": 2, "temperature": 0.7},
            )

            generated_summary = generated["results"]["toy"]
            self.assertEqual(backend.calls, 1)
            self.assertEqual(generated["phase"], "generate_only")
            self.assertTrue(generated_summary["generation_complete"])
            self.assertFalse(generated_summary["evaluation_complete"])
            self.assertEqual(generated_summary["n"], 2)
            self.assertEqual(generated_summary["unscored_records"], 4)
            self.assertEqual(generated_summary["metrics"], {})

            predictions_path = (
                out / "candidate" / "split_run" / "toy" / "predictions.jsonl"
            )
            with predictions_path.open(encoding="utf-8") as f:
                raw_rows = [json.loads(line) for line in f if line.strip()]
            self.assertTrue(
                all(row["meta"]["_aethereval_unscored"] is True for row in raw_rows)
            )

            metrics_path.write_text(metrics_source, encoding="utf-8")
            with mock.patch(
                "aethereval.core.runner.create_backend",
                side_effect=AssertionError("eval-only must not create a backend"),
            ) as create_backend:
                evaluated = run_evaluation(
                    model="offline/model",
                    model_name="candidate",
                    tasks="toy",
                    output_dir=out,
                    run_id="split_run",
                    benchmarks_dir=root,
                    eval_only=True,
                )

            create_backend.assert_not_called()
            evaluated_summary = evaluated["results"]["toy"]
            self.assertEqual(evaluated["phase"], "eval_only")
            self.assertEqual(evaluated["backend"], "offline-test-backend")
            self.assertEqual(evaluated_summary["new_records"], 0)
            self.assertEqual(evaluated_summary["n"], 2)
            self.assertEqual(evaluated_summary["rescored_records"], 4)
            self.assertEqual(evaluated_summary["unscored_records"], 0)
            self.assertTrue(evaluated_summary["evaluation_complete"])
            self.assertAlmostEqual(
                evaluated_summary["metrics"]["accuracy_first"], 1.0
            )

            with predictions_path.open(encoding="utf-8") as f:
                scored_rows = [json.loads(line) for line in f if line.strip()]
            self.assertTrue(
                all("_aethereval_unscored" not in row["meta"] for row in scored_rows)
            )
            run_config_path = predictions_path.parent / "run_config.json"
            with run_config_path.open(encoding="utf-8") as f:
                run_config = json.load(f)
            self.assertEqual(run_config["generation_config"]["n"], 2)
            self.assertEqual(run_config["generation_config"]["temperature"], 0.7)

            with self.assertRaisesRegex(ValueError, "overrides conflict"):
                run_evaluation(
                    model="offline/model",
                    model_name="candidate",
                    tasks="toy",
                    output_dir=out,
                    run_id="split_run",
                    benchmarks_dir=root,
                    eval_only=True,
                    gen_overrides={"n": 1},
                )

    def test_eval_only_rejects_incomplete_predictions_before_scoring(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "benchmarks"
            _write_toy_benchmark(root)
            out = Path(tmp) / "outputs"

            run_evaluation(
                model="fake-model",
                tasks="toy",
                output_dir=out,
                run_id="incomplete",
                backend=FakeBackend(),
                benchmarks_dir=root,
                generate_only=True,
            )
            predictions_path = (
                out / "fake-model" / "incomplete" / "toy" / "predictions.jsonl"
            )
            rows = predictions_path.read_text(encoding="utf-8").splitlines()
            predictions_path.write_text(rows[0] + "\n", encoding="utf-8")

            metrics_path = root / "toy" / "metrics.py"
            metrics_path.write_text(
                "def validate_metric_options(options):\n"
                "    raise AssertionError('completeness must be checked first')\n"
                "def score_generation(sample, generation):\n"
                "    raise AssertionError('scoring must not start')\n"
                "def aggregate(sample_results, metric_options=None):\n"
                "    return {'accuracy': 0.0}\n",
                encoding="utf-8",
            )

            with mock.patch("aethereval.core.runner.create_backend") as create_backend:
                with self.assertRaisesRegex(
                    ValueError, "eval-only requires complete existing predictions"
                ):
                    run_evaluation(
                        model="fake-model",
                        tasks="toy",
                        output_dir=out,
                        run_id="incomplete",
                        benchmarks_dir=root,
                        eval_only=True,
                    )
            create_backend.assert_not_called()

    def test_eval_only_supports_batch_judge_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "benchmarks"
            _write_batch_benchmark(root)
            out = Path(tmp) / "outputs"

            run_evaluation(
                model="fake-model",
                tasks="batch_toy",
                output_dir=out,
                run_id="batch_split",
                backend=FakeBackend(),
                benchmarks_dir=root,
                generate_only=True,
            )
            with mock.patch(
                "aethereval.core.runner.create_backend",
                side_effect=AssertionError("eval-only must not create a backend"),
            ):
                result = run_evaluation(
                    model="fake-model",
                    tasks="batch_toy",
                    output_dir=out,
                    run_id="batch_split",
                    benchmarks_dir=root,
                    metric_options={"batch_offset": 1.0},
                    eval_only=True,
                )

            summary = result["results"]["batch_toy"]
            self.assertEqual(summary["rescored_records"], 2)
            self.assertAlmostEqual(summary["metrics"]["batch_mean"], 8.0)

    def test_eval_only_creates_and_closes_metric_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "benchmarks"
            _write_batch_benchmark(root)
            out = Path(tmp) / "outputs"
            close_marker = Path(tmp) / "metric_backend_closed"

            run_evaluation(
                model="fake-model",
                tasks="batch_toy",
                output_dir=out,
                run_id="backend_split",
                backend=FakeBackend(),
                benchmarks_dir=root,
                generate_only=True,
            )

            (root / "batch_toy" / "metrics.py").write_text(
                "from pathlib import Path\n"
                "PRIMARY_METRIC='metric_score'\n"
                "REQUIRES_BACKEND=True\n"
                "class MetricBackend:\n"
                "    name='metric-only'\n"
                "    def __init__(self, marker): self.marker=marker\n"
                "    def close(self): Path(self.marker).write_text('closed')\n"
                "def create_evaluation_backend(options, *, dp_size, "
                "tensor_parallel_size):\n"
                "    assert dp_size == 2 and tensor_parallel_size == 2\n"
                "    return MetricBackend(options['close_marker'])\n"
                "def score_generation(sample, generation):\n"
                "    raise AssertionError('batch scorer required')\n"
                "def score_generations_batch(samples, outputs, options=None):\n"
                "    assert options['_backend'].name == 'metric-only'\n"
                "    return [[{'score': 1.0}] for output in outputs "
                "for _ in [output.generations]]\n"
                "def aggregate(results, options=None):\n"
                "    return {'metric_score': sum(r['scores'][0] for r in results) "
                "/ len(results)}\n",
                encoding="utf-8",
            )

            with mock.patch(
                "aethereval.core.runner.create_backend",
                side_effect=AssertionError("candidate backend must not be created"),
            ):
                result = run_evaluation(
                    model="fake-model",
                    tasks="batch_toy",
                    output_dir=out,
                    run_id="backend_split",
                    dp_size=2,
                    tensor_parallel_size=2,
                    metric_options={"close_marker": str(close_marker)},
                    benchmarks_dir=root,
                    eval_only=True,
                )

            self.assertTrue(close_marker.exists())
            self.assertEqual(
                result["results"]["batch_toy"]["primary_score"],
                1.0,
            )

    def test_eval_only_rejects_overwrite(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be combined with overwrite"):
            run_evaluation(
                model="fake-model",
                tasks="toy",
                output_dir="unused",
                overwrite=True,
                eval_only=True,
            )

    def test_model_name_controls_output_without_changing_model_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "benchmarks"
            _write_toy_benchmark(root)
            out = Path(tmp) / "outputs"

            result = run_evaluation(
                model="qwen2.5/huggingface",
                model_name="Qwen2.5/custom_model",
                tasks="toy",
                output_dir=out,
                run_id="production-1",
                backend=FakeBackend(),
                benchmarks_dir=root,
            )

            self.assertEqual(result["model"], "qwen2.5/huggingface")
            self.assertEqual(result["model_name"], "qwen2.5-custom_model")
            task_dir = out / "qwen2.5-custom_model" / "production-1" / "toy"
            self.assertTrue((task_dir / "predictions.jsonl").exists())
            with (task_dir / "run_config.json").open(encoding="utf-8") as f:
                run_config = json.load(f)
            self.assertEqual(run_config["model"], "qwen2.5/huggingface")
            self.assertEqual(run_config["model_name"], "qwen2.5-custom_model")

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
            self.assertAlmostEqual(
                summary["metrics"]["avg_response_tokens"], 1.0, places=6
            )
            self.assertEqual(summary["primary_metric"], "accuracy_first")
            self.assertAlmostEqual(float(summary["primary_score"]), 1.0, places=6)
            self.assertAlmostEqual(
                first["summary"]["metrics"]["accuracy_first"],
                1.0,
                places=6,
            )
            self.assertIn("primary_scores", first)
            self.assertEqual(first["primary_scores"]["toy"]["metric"], "accuracy_first")
            self.assertAlmostEqual(
                float(first["primary_scores"]["toy"]["score"]), 1.0, places=6
            )
            self.assertAlmostEqual(
                float(first["primary_score_aggregate"]), 1.0, places=6
            )
            predictions_path = (
                out / "fake-model" / "run1" / "toy" / "predictions.jsonl"
            )
            with predictions_path.open("r", encoding="utf-8") as f:
                first_row = json.loads(f.readline())
            self.assertIsInstance(first_row["prompt"], list)
            self.assertEqual(first_row["prompt"][0]["role"], "user")
            self.assertIn("Question: 2 + 2", first_row["prompt"][0]["content"])
            self.assertEqual(first_row["meta"]["response_token_count"], 1)

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

    def test_batch_metric_hook_scores_generations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "benchmarks"
            _write_batch_benchmark(root)
            out = Path(tmp) / "outputs"

            result = run_evaluation(
                model="fake-model",
                tasks="batch_toy",
                output_dir=out,
                run_id="batch_run",
                backend=FakeBackend(),
                benchmarks_dir=root,
                metric_options={"batch_offset": 1.0},
            )

            summary = result["results"]["batch_toy"]
            self.assertAlmostEqual(summary["metrics"]["batch_mean"], 8.0, places=6)
            predictions_path = (
                out
                / "fake-model"
                / "batch_run"
                / "batch_toy"
                / "predictions.jsonl"
            )
            with predictions_path.open("r", encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(row["meta"]["batch"] is True for row in rows))
            self.assertTrue(
                all(row["meta"]["response_token_count"] == 1 for row in rows)
            )

    def test_resume_rescores_existing_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "benchmarks"
            _write_toy_benchmark(root)
            out = Path(tmp) / "outputs"

            run_evaluation(
                model="fake-model",
                tasks="toy",
                output_dir=out,
                run_id="run_rescore",
                backend=FakeBackend(),
                benchmarks_dir=root,
            )

            (root / "toy" / "metrics.py").write_text(
                "def score_generation(sample, generation):\n"
                "    return {'score': 0.0}\n"
                "def aggregate(sample_results, metric_options=None):\n"
                "    first_scores = [float(item['scores'][0]) if item.get('scores') else 0.0 for item in sample_results]\n"
                "    return {'accuracy_first': sum(first_scores)/len(first_scores) if first_scores else 0.0}\n",
                encoding="utf-8",
            )

            resumed = run_evaluation(
                model="fake-model",
                tasks="toy",
                output_dir=out,
                run_id="run_rescore",
                backend=NeverCalledBackend(),
                benchmarks_dir=root,
            )
            summary = resumed["results"]["toy"]
            self.assertEqual(summary["existing_records"], 2)
            self.assertEqual(summary["new_records"], 0)
            self.assertAlmostEqual(summary["metrics"]["accuracy_first"], 0.0, places=6)

            predictions_path = (
                out
                / "fake-model"
                / "run_rescore"
                / "toy"
                / "predictions.jsonl"
            )
            with predictions_path.open("r", encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(float(row["score"]) == 0.0 for row in rows))

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

    def test_backend_generation_count_mismatch_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "benchmarks"
            _write_toy_benchmark(root)

            out = Path(tmp) / "outputs"
            with self.assertRaises(ValueError):
                run_evaluation(
                    model="fake-model",
                    tasks="toy",
                    output_dir=out,
                    run_id="run_short",
                    backend=ShortGenerationBackend(),
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
            self.assertAlmostEqual(
                float(second["primary_score_aggregate"]), 1.0, places=6
            )


if __name__ == "__main__":
    unittest.main()
