import importlib.metadata
import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from aethereval.core.runner import (
    _aggregate_repeat_token_usage,
    _generation_token_meta,
    _merge_generation_config,
    _records_to_generation_outputs,
    _score_generation_outputs,
    _token_usage_summary,
    inspect_prompts,
    run_evaluation,
)
from aethereval.core.task_register import _load_module_from_path
from aethereval.core.types import (
    GenerationInput,
    GenerationOutput,
    GenerationRecord,
    Sample,
)
from tests._deps import requires


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
        return "\n".join(
            f"{message['role']}: {message['content']}" for message in prompt
        )


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
                        "finish_reasons": ["stop"] * item.num_generations,
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


class TokenUsageTests(unittest.TestCase):
    def test_completed_length_uses_end_reason_not_score_or_length(self) -> None:
        rows = [
            ("stop", 4, None),
            ("length", 100, None),
            ("abort", 30, None),
            (None, 9, None),
            ("stop", 10, None),
            ("stop", 20, "backend error"),
        ]
        records = [
            GenerationRecord(
                sample_id="a",
                gen_idx=index,
                prompt="q",
                generation="answer",
                score=0.0,
                is_pass=False,
                error=error,
                meta={
                    "prompt_token_count": 2,
                    "response_token_count": count,
                    **({"finish_reason": reason} if reason is not None else {}),
                },
            )
            for index, (reason, count, error) in enumerate(rows)
        ]
        usage = _token_usage_summary(records)
        self.assertEqual(usage["avg_response_tokens"], 173 / 6)
        self.assertEqual(usage["avg_completed_response_tokens"], 7)
        self.assertEqual(usage["num_completed_responses"], 2)
        self.assertEqual(usage["total_completed_response_tokens"], 14)
        for subset in ([], records[1:4], records[-1:]):
            self.assertIsNone(
                _token_usage_summary(subset)["avg_completed_response_tokens"]
            )
        outputs, _ = _records_to_generation_outputs(records)
        for index, (reason, _, _) in enumerate(rows):
            self.assertEqual(
                _generation_token_meta(outputs[0], index)["finish_reason"], reason
            )

    def test_completed_length_pools_repeats_by_completed_count(self) -> None:
        repeats = [
            {"total_records": 2, "token_usage": {
                "num_completed_responses": 1,
                "total_completed_response_tokens": 20,
            }},
            {"total_records": 2, "token_usage": {
                "num_completed_responses": 2,
                "total_completed_response_tokens": 10,
            }},
            {"total_records": 2, "token_usage": {}},
        ]
        usage = _aggregate_repeat_token_usage(repeats)
        self.assertEqual(usage["avg_completed_response_tokens"], 10)
        self.assertEqual(usage["num_completed_responses"], 3)
        self.assertIsNone(
            _aggregate_repeat_token_usage([])["avg_completed_response_tokens"]
        )


class GenerationConfigTests(unittest.TestCase):
    def test_greedy_override_preserves_multi_sample_task_protocol(self) -> None:
        resolved = _merge_generation_config(
            {"n": 16, "temperature": 1.0, "top_p": 0.7},
            {"n": None, "temperature": 0.0},
        )

        self.assertEqual(resolved["n"], 16)
        self.assertEqual(resolved["temperature"], 1.0)

    def test_greedy_override_applies_to_single_sample_task(self) -> None:
        resolved = _merge_generation_config(
            {"n": 1, "temperature": 0.7},
            {"n": None, "temperature": 0.0},
        )

        self.assertEqual(resolved["n"], 1)
        self.assertEqual(resolved["temperature"], 0.0)

    def test_explicit_n_allows_replacing_multi_sample_task_protocol(self) -> None:
        resolved = _merge_generation_config(
            {"n": 16, "temperature": 1.0},
            {"n": 1, "temperature": 0.0},
        )

        self.assertEqual(resolved["n"], 1)
        self.assertEqual(resolved["temperature"], 0.0)

    def test_explicit_multi_sample_greedy_remains_invalid(self) -> None:
        with self.assertRaisesRegex(ValueError, "n>1 requires temperature>0"):
            _merge_generation_config(
                {"n": 1, "temperature": 0.7},
                {"n": 4, "temperature": 0.0},
            )


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


# A judge metric that sends one offline-judge request per sample when available.
_JUDGE_METRIC_PREFIX = (
    "USES_LLM_JUDGE = True\n"
    "def score_generations_batch(samples, outputs, options):\n"
    "    client = options.get('_judge_client')\n"
    "    if client is not None:\n"
    "        for output in outputs:\n"
    "            client.complete([{'role': 'user', 'content': output.sample_id}])\n"
    "    return [[{'score': 1.0} for _ in output.generations] for output in outputs]\n"
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
    @requires("evalplus")
    def test_evalplus_find_zero_scoring_matches_across_process_counts(self) -> None:
        metrics = _load_module_from_path(
            "find_zero_test_metrics",
            Path(__file__).resolve().parents[1] / "benchmarks/humaneval-plus/metrics.py",
        )
        sample = Sample(
            id="HumanEval/32-linear-probe",
            data={
                "prompt": 'def find_zero(xs):\n    """Find the root."""\n',
                "entry_point": "find_zero",
                "canonical_solution": "    return -xs[0] / xs[1]\n",
                "base_input": [[[1, 2]]],
                "plus_input": [[[2, 1]]],
                "atol": 0.0001,
            },
        )
        kwargs = dict(
            metrics_module=metrics,
            samples_by_id={sample.id: sample},
            outputs=[GenerationOutput(
                sample.id, sample.data["prompt"],
                [sample.data["canonical_solution"], "    return 0\n"],
            )],
            total_records=2,
            progress_desc="find_zero spawn regression",
        )
        serial = _score_generation_outputs(**kwargs, metric_options={"num_proc": 1})
        parallel = _score_generation_outputs(**kwargs, metric_options={"num_proc": 2})
        self.assertEqual([result[0] for result in serial[sample.id]], [1.0, 0.0])
        self.assertEqual(parallel, serial)

    @requires("evalplus")
    def test_evalplus_candidate_memory_budget_matches_across_process_counts(self) -> None:
        # A per-core BLAS pool in the checker, or a checker forked from this process,
        # used to leave candidates under 1.5 GB of EvalPlus's 4 GiB RLIMIT_AS.
        metrics = _load_module_from_path(
            "memory_budget_test_metrics",
            Path(__file__).resolve().parents[1] / "benchmarks/humaneval-plus/metrics.py",
        )
        sample = Sample(
            id="HumanEval/memory-probe",
            data={
                "prompt": 'def f(x):\n    """Return x."""\n',
                "entry_point": "f",
                "canonical_solution": "    return x\n",
                "base_input": [[1]],
                "plus_input": [[2]],
                "atol": 0.0,
            },
        )
        # bytes(n) reserves address space (what RLIMIT_AS counts) without touching it.
        kwargs = dict(
            metrics_module=metrics,
            samples_by_id={sample.id: sample},
            outputs=[GenerationOutput(sample.id, "", [
                "    buffer = bytes(5 * 1024**3 // 2)\n    return x\n",
                "    buffer = bytes(5 * 1024**3)\n    return x\n",
            ])],
            total_records=2,
            progress_desc="EvalPlus memory budget",
        )
        for num_proc in (1, 2):
            with self.subTest(num_proc=num_proc):
                scored = _score_generation_outputs(**kwargs, metric_options={"num_proc": num_proc})
                self.assertEqual([result[0] for result in scored[sample.id]], [1.0, 0.0])

    def test_score_in_subprocess_uses_spawned_worker_with_single_thread_blas(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.py"
            path.write_text(
                "import os\n"
                "SCORE_IN_SUBPROCESS = True\n"
                "def score_generation(sample, generation):\n"
                "    return {'score': 1.0, 'meta': {\n"
                "        'pid': os.getpid(), 'blas': os.environ.get('OPENBLAS_NUM_THREADS')}}\n",
                encoding="utf-8",
            )
            with mock.patch.dict(os.environ):
                os.environ.pop("OPENBLAS_NUM_THREADS", None)
                scored = _score_generation_outputs(
                    metrics_module=_load_module_from_path("subprocess_test_metrics", path),
                    samples_by_id={"a": Sample(id="a")},
                    outputs=[GenerationOutput("a", "", ["x"])],
                    metric_options={"num_proc": 1},
                    total_records=1,
                    progress_desc="test scoring",
                )
                # Only the scoring worker is capped, not processes started from here.
                self.assertNotIn("OPENBLAS_NUM_THREADS", os.environ)
            meta = scored["a"][0][3]
            self.assertNotEqual(meta["pid"], os.getpid())
            self.assertEqual(meta["blas"], "1")

    def test_pooled_scoring_stops_queued_records_on_first_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.py"
            log = Path(tmp) / "started.log"
            path.write_text(
                "import time\n"
                "SCORE_IN_SUBPROCESS = True\n"
                "def score_generation(sample, generation):\n"
                f"    with open({str(log)!r}, 'a') as handle:\n"
                "        handle.write(generation + '\\n')\n"
                "    if generation == '0':\n"
                "        raise ValueError('bad record')\n"
                "    time.sleep(0.2)\n"
                "    return {'score': 1.0}\n",
                encoding="utf-8",
            )
            generations = [str(index) for index in range(31)]
            for num_proc in (1, 2):
                with self.subTest(num_proc=num_proc):
                    log.write_text("", encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "bad record"):
                        _score_generation_outputs(
                            metrics_module=_load_module_from_path("fail_fast_test_metrics", path),
                            samples_by_id={"a": Sample(id="a")},
                            outputs=[GenerationOutput("a", "", generations)],
                            metric_options={"num_proc": num_proc},
                            total_records=len(generations),
                            progress_desc="test scoring",
                        )
                    # Only records already handed to a worker run; queued ones are cancelled.
                    started = log.read_text(encoding="utf-8").splitlines()
                    self.assertLess(len(started), 10)

    def test_parallel_scoring_preserves_order_and_allows_test_subprocesses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.py"
            path.write_text(
                "import multiprocessing as mp\n"
                "import threading\n"
                "import time\n"
                "def score_generation(sample, generation):\n"
                "    assert threading.current_thread() is threading.main_thread()\n"
                "    child = mp.Process(target=int, args=('1',))\n"
                "    child.start()\n"
                "    child.join(10)\n"
                "    assert child.exitcode == 0\n"
                "    time.sleep(0.1 if generation == '1' else 0)\n"
                "    return {'score': float(generation), 'parsed': sample.id}\n",
                encoding="utf-8",
            )
            samples = {key: Sample(id=key) for key in ("a", "b")}
            outputs = [
                GenerationOutput("b", "", ["1", "2"]),
                GenerationOutput("a", "", ["3"]),
            ]
            kwargs = dict(
                metrics_module=_load_module_from_path("parallel_test_metrics", path),
                samples_by_id=samples,
                outputs=outputs,
                total_records=3,
                progress_desc="test scoring",
            )
            serial = _score_generation_outputs(**kwargs, metric_options={"num_proc": 1})
            parallel = _score_generation_outputs(**kwargs, metric_options={"num_proc": 2})
            self.assertEqual(parallel, serial)
            self.assertEqual(list(parallel), ["b", "a"])
            outputs[0].generations[0] = "invalid"
            with self.assertRaises(ValueError):
                _score_generation_outputs(**kwargs, metric_options={"num_proc": 2})

    def test_eval_only_manages_offline_judge_and_keeps_runtime_object_out_of_json(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "benchmarks"
            _write_toy_benchmark(root)
            metrics_path = root / "toy" / "metrics.py"
            metrics_path.write_text(
                _JUDGE_METRIC_PREFIX + metrics_path.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            out = Path(tmp) / "outputs"
            run_evaluation(
                model="candidate/model",
                tasks="toy",
                output_dir=out,
                run_id="local_judge",
                backend=FakeBackend(),
                benchmarks_dir=root,
                generate_only=True,
            )

            judge = mock.Mock(name="offline-judge")
            with mock.patch(
                "aethereval.core.runner.OfflineJudgeClient",
                return_value=judge,
            ) as judge_class:
                evaluated = run_evaluation(
                    model="candidate/model",
                    tasks="toy",
                    output_dir=out,
                    run_id="local_judge",
                    benchmarks_dir=root,
                    eval_only=True,
                    metric_options={
                        "judge_backend": "local",
                        "judge_model": "local/judge",
                        "judge_dp_size": 1,
                        "judge_tp_size": 2,
                        "judge_workers": 8,
                        "judge_max_new_tokens": 1024,
                        "judge_sglang_args": {"context_length": 8192},
                    },
                )

            self.assertTrue(evaluated["results"]["toy"]["evaluation_complete"])
            self.assertEqual(judge.complete.call_count, 2)
            judge_class.assert_called_once_with(
                model="local/judge",
                dp_size=1,
                tensor_parallel_size=2,
                model_kwargs={"context_length": 8192},
            )
            judge.close.assert_called_once_with()
            run_config_path = out / "model" / "local_judge" / "toy" / "run_config.json"
            with run_config_path.open(encoding="utf-8") as f:
                run_config = json.load(f)
            self.assertNotIn("_judge_client", run_config["metric_options"])

    def test_local_judge_groups_reuse_services_and_close_on_failure(self) -> None:
        from aethereval.core import runner

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "benchmarks"
            names = ["judge-a1", "cpu", "judge-b", "judge-a2", "judge-context"]
            defaults = {}
            for name in names:
                _write_toy_benchmark(root)
                (root / "toy").rename(root / name)
                if name == "cpu":
                    continue
                path = root / name / "metrics.py"
                path.write_text(_JUDGE_METRIC_PREFIX + path.read_text())
                defaults[name] = {"metrics": {
                    "judge_model": "local/b" if name == "judge-b" else "local/a",
                    "judge_sglang_args": {
                        "context_length": 16384 if name == "judge-context" else 8192,
                    },
                    "judge_temperature": 0.7 if name == "judge-a2" else 0.0,
                }}
            kwargs = dict(
                model="candidate/model", tasks=",".join(names),
                output_dir=Path(tmp) / "outputs", run_id="grouped", benchmarks_dir=root,
            )
            run_evaluation(**kwargs, backend=FakeBackend(), generate_only=True)
            original_run_task = runner._run_repeated_task

            for mode in ("local", "api", "failure"):
                with self.subTest(mode=mode):
                    events, clients, scored_options = [], [], {}

                    def create_judge(**config):
                        index = len(clients)
                        client = mock.Mock()
                        client.close.side_effect = lambda: events.append(("close", index))
                        clients.append(client)
                        events.append(("load", index))
                        return client

                    def score_task(**options):
                        name = options["task_name"]
                        events.append(("score", name))
                        scored_options[name] = options["metric_options"]
                        if mode == "failure" and name == "judge-a2":
                            raise RuntimeError("test scoring failure")
                        return original_run_task(**options)

                    with (
                        mock.patch("aethereval.core.task_defaults._load_task_default_overrides", return_value=defaults),
                        mock.patch.object(runner, "OfflineJudgeClient", side_effect=create_judge) as constructor,
                        mock.patch.object(runner, "_run_repeated_task", side_effect=score_task),
                    ):
                        options = {"judge_backend": "api" if mode == "api" else "local"}
                        if mode == "failure":
                            with self.assertRaisesRegex(RuntimeError, "test scoring failure"):
                                run_evaluation(**kwargs, eval_only=True, metric_options=options)
                        else:
                            result = run_evaluation(**kwargs, eval_only=True, metric_options=options)
                            self.assertTrue(all(item["evaluation_complete"] for item in result["results"].values()))

                    if mode == "api":
                        constructor.assert_not_called()
                        self.assertEqual(events, [("score", name) for name in names])
                        continue
                    # Judges start lazily on the first request of their group.
                    expected = [
                        ("score", "cpu"), ("score", "judge-a1"), ("load", 0),
                        ("score", "judge-a2"), ("close", 0),
                    ]
                    if mode == "local":
                        expected += [
                            ("score", "judge-b"), ("load", 1), ("close", 1),
                            ("score", "judge-context"), ("load", 2), ("close", 2),
                        ]
                    self.assertEqual(events, expected)
                    self.assertIs(scored_options["judge-a1"]["_judge_client"], scored_options["judge-a2"]["_judge_client"])
                    self.assertEqual(scored_options["judge-a1"]["judge_temperature"], 0.0)
                    self.assertEqual(scored_options["judge-a2"]["judge_temperature"], 0.7)
                    for client in clients:
                        client.close.assert_called_once_with()

    def test_lazy_judge_starts_once_caches_failures_and_stays_closed(self) -> None:
        from aethereval.core.runner import _LazyJudgeClient

        config = {"model": "local/judge", "dp_size": 1, "tensor_parallel_size": 2}
        judge = mock.Mock()

        def slow_start(**kwargs):
            time.sleep(0.05)
            return judge

        with mock.patch(
            "aethereval.core.runner.OfflineJudgeClient", side_effect=slow_start
        ) as constructor:
            client = _LazyJudgeClient(config)
            with self.assertRaisesRegex(RuntimeError, "not running"):
                client.complete([{"role": "user", "content": "q"}])
            threads = [threading.Thread(target=client.start) for _ in range(16)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            constructor.assert_called_once_with(**config)
            self.assertIs(
                client.complete([{"role": "user", "content": "q"}], temperature=0.0),
                judge.complete.return_value,
            )
            client.close()
            client.close()
            judge.close.assert_called_once_with()
            with self.assertRaisesRegex(RuntimeError, "not running"):
                client.complete([{"role": "user", "content": "q"}])
            with self.assertRaisesRegex(RuntimeError, "closed"):
                client.start()
            constructor.assert_called_once()

        with mock.patch(
            "aethereval.core.runner.OfflineJudgeClient",
            side_effect=RuntimeError("no GPUs"),
        ) as constructor:
            client = _LazyJudgeClient(config)
            with self.assertRaisesRegex(RuntimeError, "no GPUs"):
                client.start()
            with self.assertRaisesRegex(RuntimeError, "failed to start"):
                client.start()
            constructor.assert_called_once()
            client.close()

    def test_grouped_local_judge_starts_on_runner_thread(self) -> None:
        # Judge servers are guarded subprocesses that exit with the thread that
        # started them, so a judge started by a parallel_map worker would die
        # before the group's next task.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "benchmarks"
            names = ["judge-a", "judge-b"]
            for name in names:
                _write_toy_benchmark(root)
                (root / "toy").rename(root / name)
                path = root / name / "metrics.py"
                path.write_text(
                    "USES_LLM_JUDGE = True\n"
                    "from benchmark_utils.llm_judge import parallel_map\n"
                    "def score_generations_batch(samples, outputs, options):\n"
                    "    client = options['_judge_client']\n"
                    "    parallel_map(\n"
                    "        lambda o: client.complete([{'role': 'user', 'content': o.sample_id}]),\n"
                    "        outputs, workers=2, desc='judge',\n"
                    "    )\n"
                    "    return [[{'score': 1.0} for _ in o.generations] for o in outputs]\n"
                    + path.read_text()
                )
            kwargs = dict(
                model="candidate/model", tasks=",".join(names),
                output_dir=Path(tmp) / "outputs", run_id="threads", benchmarks_dir=root,
            )
            run_evaluation(**kwargs, backend=FakeBackend(), generate_only=True)
            started, requests, closed = [], [], []

            class ThreadCheckingJudge:
                def __init__(self, **config):
                    started.append(threading.current_thread())

                def complete(self, messages, **options):
                    requests.append(threading.current_thread())
                    return "ok"

                def close(self):
                    closed.append(threading.current_thread())

            with mock.patch(
                "aethereval.core.runner.OfflineJudgeClient", ThreadCheckingJudge
            ):
                result = run_evaluation(
                    **kwargs,
                    eval_only=True,
                    metric_options={"judge_backend": "local", "judge_model": "local/judge"},
                )

            self.assertTrue(
                all(item["evaluation_complete"] for item in result["results"].values())
            )
            self.assertEqual(started, [threading.main_thread()])
            self.assertEqual(len(requests), 4)
            self.assertNotIn(threading.main_thread(), requests)
            self.assertEqual(closed, [threading.main_thread()])

    def test_automatic_eval_phase_reuses_judgments_with_same_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "benchmarks"
            _write_toy_benchmark(root)
            _write_toy2_benchmark(root)
            metrics_path = root / "toy" / "metrics.py"
            metrics_path.write_text(
                "PRESERVE_EXISTING_SCORES_ON_RESUME = True\n"
                + _JUDGE_METRIC_PREFIX
                + metrics_path.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            kwargs = dict(
                model="fake-model",
                tasks="toy,toy2",
                output_dir=Path(tmp) / "outputs",
                run_id="reuse",
                benchmarks_dir=root,
            )
            task_dir = Path(tmp) / "outputs" / "fake-model" / "reuse" / "toy"
            predictions_path = task_dir / "predictions.jsonl"
            judge = mock.Mock()

            def evaluate(*, backend=None, eval_only=False, **options):
                judge.reset_mock()
                # The runner creates (and closes) the candidate backend, as for the CLI.
                with mock.patch(
                    "aethereval.core.runner.create_backend",
                    return_value=backend or NeverCalledBackend(),
                ), mock.patch(
                    "aethereval.core.runner.OfflineJudgeClient", return_value=judge
                ) as judge_class:
                    result = run_evaluation(
                        **kwargs,
                        eval_only=eval_only,
                        metric_options={
                            "judge_backend": "local",
                            "judge_model": "local/judge",
                            "judge_workers": 4,
                            **options,
                        },
                    )
                calls = judge.complete.call_count
                self.assertEqual(judge_class.call_count, min(calls, 1))
                return calls, result["results"]

            calls, first = evaluate(backend=FakeBackend())
            self.assertEqual(calls, 2)
            self.assertEqual(first["toy"]["rescored_records"], 2)
            lines = predictions_path.read_text().splitlines()
            rows = [json.loads(line) for line in lines]
            fingerprint = rows[0]["meta"]["_aethereval_judge"]
            self.assertTrue(
                all(row["meta"]["_aethereval_judge"] == fingerprint for row in rows)
            )

            # A rerun keeps judgments and never starts the judge; CPU metrics rescore.
            predictions = predictions_path.read_bytes()
            summary = (task_dir / "summary.json").read_bytes()
            run_evaluation(**kwargs, backend=NeverCalledBackend(), generate_only=True)
            self.assertEqual((task_dir / "summary.json").read_bytes(), summary)
            calls, rerun = evaluate(judge_workers=8)
            self.assertEqual(calls, 0)
            self.assertEqual(rerun["toy"]["rescored_records"], 0)
            self.assertEqual(rerun["toy"]["metrics"], first["toy"]["metrics"])
            self.assertEqual(rerun["toy2"]["rescored_records"], 2)
            self.assertEqual(predictions_path.read_bytes(), predictions)

            # Only records without a current judgment are judged.
            predictions_path.write_text(
                predictions.decode().splitlines()[0] + "\n", encoding="utf-8"
            )
            self.assertEqual(evaluate(backend=FakeBackend())[0], 1)
            self.assertEqual(evaluate(judge_temperature=0.5)[0], 2)
            self.assertEqual(evaluate(judge_temperature=0.5)[0], 0)
            self.assertEqual(evaluate(eval_only=True)[0], 2)
            predictions_path.write_text(
                predictions_path.read_text().replace(
                    f'"_aethereval_judge": "{fingerprint}"', '"legacy": true'
                ),
                encoding="utf-8",
            )
            self.assertEqual(evaluate()[0], 2)

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
                gen_overrides={
                    "n": 2,
                    "temperature": 0.7,
                    "enable_thinking": False,
                },
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
            run_config_path = predictions_path.parent / "run_config.json"
            generated_config = json.loads(run_config_path.read_text(encoding="utf-8"))
            self.assertNotIn("scoring_packages", generated_config)

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
            self.assertAlmostEqual(evaluated_summary["metrics"]["accuracy_first"], 1.0)

            with predictions_path.open(encoding="utf-8") as f:
                scored_rows = [json.loads(line) for line in f if line.strip()]
            self.assertTrue(
                all("_aethereval_unscored" not in row["meta"] for row in scored_rows)
            )
            with run_config_path.open(encoding="utf-8") as f:
                run_config = json.load(f)
            self.assertEqual(
                run_config["scoring_packages"]["math-verify"],
                importlib.metadata.version("math-verify"),
            )
            self.assertEqual(run_config["generation_config"]["n"], 2)
            self.assertEqual(run_config["generation_config"]["temperature"], 0.7)
            self.assertIs(
                run_config["generation_config"]["enable_thinking"],
                False,
            )

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

            summary = result["results"]["batch-toy"]
            self.assertEqual(summary["rescored_records"], 2)
            self.assertAlmostEqual(summary["metrics"]["batch_mean"], 8.0)

    def test_metric_unscored_record_is_excluded_without_aborting_later_tasks(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "benchmarks"
            _write_batch_benchmark(root)
            _write_toy_benchmark(root)
            out = Path(tmp) / "outputs"
            kwargs = dict(
                model="fake-model",
                tasks="batch_toy,toy",
                output_dir=out,
                run_id="unscored",
                benchmarks_dir=root,
            )
            run_evaluation(**kwargs, backend=FakeBackend(), generate_only=True)
            metrics_path = root / "batch_toy" / "metrics.py"

            def write_metric(failed_id: str) -> None:
                metrics_path.write_text(
                    "def score_generations_batch(samples, outputs, options=None):\n"
                    "    return [[{'score': 1.0, 'meta': {\n"
                    f"        'judge_failed': o.sample_id == {failed_id!r},\n"
                    f"        '_aethereval_unscored': o.sample_id == {failed_id!r},\n"
                    "    }} for _ in o.generations] for o in outputs]\n"
                    "def aggregate(results, options=None):\n"
                    "    kept = [r for r in results if not r['records'][0]['meta']['judge_failed']]\n"
                    "    return {'judged': float(len(kept))}\n",
                    encoding="utf-8",
                )

            write_metric("1")
            failed = run_evaluation(**kwargs, eval_only=True)["results"]
            self.assertEqual(failed["batch-toy"]["metrics"]["judged"], 1.0)
            self.assertEqual(failed["batch-toy"]["unscored_records"], 1)
            self.assertFalse(failed["batch-toy"]["evaluation_complete"])
            self.assertIn("scored again on resume", failed["batch-toy"]["warnings"][-1])
            self.assertTrue(failed["toy"]["evaluation_complete"])

            write_metric("none")
            retried = run_evaluation(**kwargs, eval_only=True)["results"]
            self.assertEqual(retried["batch-toy"]["metrics"]["judged"], 2.0)
            self.assertEqual(retried["batch-toy"]["unscored_records"], 0)
            self.assertTrue(retried["batch-toy"]["evaluation_complete"])
            self.assertEqual(retried["batch-toy"]["warnings"], [])

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
                result["results"]["batch-toy"]["primary_score"],
                1.0,
            )

            # Without a phase flag, a caller-supplied backend also serves the metric.
            close_marker.unlink()
            supplied = FakeBackend()
            supplied.name = "metric-only"
            result = run_evaluation(
                model="fake-model",
                tasks="batch_toy",
                output_dir=out,
                run_id="backend_split",
                dp_size=2,
                tensor_parallel_size=2,
                metric_options={"close_marker": str(close_marker)},
                backend=supplied,
                benchmarks_dir=root,
            )
            self.assertFalse(close_marker.exists())
            self.assertEqual(supplied.calls, 0)
            self.assertEqual(result["results"]["batch-toy"]["primary_score"], 1.0)

    def test_run_evaluation_without_phase_flag_generates_then_evaluates(self) -> None:
        from aethereval.core import runner

        backend = FakeBackend()
        kwargs = dict(model="fake-model", tasks="toy", output_dir="unused")
        with mock.patch.object(runner, "_run_phase", return_value={}) as run_phase:
            run_evaluation(**kwargs, backend=backend, overwrite=True)
            run_evaluation(**kwargs, eval_only=True)
            # A caller-supplied backend stays loaded, so it cannot meet a local judge.
            with self.assertRaisesRegex(ValueError, "disjoint candidate/judge"):
                run_evaluation(
                    **kwargs, backend=backend, metric_options={"judge_backend": "local"}
                )
        calls = [call.kwargs for call in run_phase.call_args_list]
        flags = ("generate_only", "eval_only", "overwrite", "rescore_existing")
        self.assertEqual(
            [tuple(call[flag] for flag in flags) for call in calls],
            [
                (True, False, True, True),
                (False, True, False, False),
                (False, True, False, True),
            ],
        )
        self.assertIs(calls[0]["backend"], backend)
        self.assertIs(calls[1]["backend"], backend)
        self.assertIsNone(calls[2]["backend"])

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
                metric_options={"num_proc": 2},
            )
            self.assertIn("toy", first["results"])
            summary = first["results"]["toy"]
            self.assertEqual(backend.calls, 1)
            self.assertEqual(first["phase"], "eval_only")
            self.assertEqual(summary["rescored_records"], 2)
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
            predictions_path = out / "fake-model" / "run1" / "toy" / "predictions.jsonl"
            with predictions_path.open("r", encoding="utf-8") as f:
                first_row = json.loads(f.readline())
            self.assertIsInstance(first_row["prompt"], list)
            self.assertEqual(first_row["prompt"][0]["role"], "user")
            self.assertIn("Question: 2 + 2", first_row["prompt"][0]["content"])
            self.assertEqual(first_row["meta"]["response_token_count"], 1)
            self.assertEqual(first_row["meta"]["finish_reason"], "stop")
            self.assertEqual(summary["metrics"]["avg_completed_response_tokens"], 1.0)
            self.assertEqual(summary["token_usage"]["num_completed_responses"], 2)

            resume_backend = NeverCalledBackend()
            original_predictions = predictions_path.read_text()
            second = run_evaluation(
                model="fake-model",
                tasks="toy",
                output_dir=out,
                run_id="run1",
                backend=resume_backend,
                benchmarks_dir=root,
                metric_options={"num_proc": 2},
            )
            summary2 = second["results"]["toy"]
            self.assertEqual(summary2["rescored_records"], 2)
            self.assertEqual(summary2["existing_records"], 2)
            self.assertEqual(summary2["metrics"]["avg_completed_response_tokens"], 1.0)
            self.assertEqual(predictions_path.read_text(), original_predictions)

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
                metric_options={"batch_offset": 1.0, "num_proc": 2},
            )

            summary = result["results"]["batch-toy"]
            self.assertAlmostEqual(summary["metrics"]["batch_mean"], 8.0, places=6)
            predictions_path = (
                out / "fake-model" / "batch_run" / "batch-toy" / "predictions.jsonl"
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
                out / "fake-model" / "run_rescore" / "toy" / "predictions.jsonl"
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

    def test_num_repeats_is_distinct_from_n_and_averages_complete_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "benchmarks"
            _write_toy_benchmark(root)

            out = Path(tmp) / "outputs"
            backend = FakeBackend()
            generated = run_evaluation(
                model="fake-model",
                tasks="toy",
                output_dir=out,
                run_id="repeated",
                backend=backend,
                benchmarks_dir=root,
                num_repeats=2,
                generate_only=True,
            )
            self.assertEqual(generated["results"]["toy"]["metrics"], {})

            result = run_evaluation(
                model="fake-model",
                tasks="toy",
                output_dir=out,
                run_id="repeated",
                benchmarks_dir=root,
                eval_only=True,
            )

            summary = result["results"]["toy"]
            task_dir = out / "fake-model" / "repeated" / "toy"
            self.assertEqual(backend.calls, 2)
            self.assertEqual(summary["n"], 1)
            self.assertEqual(summary["num_repeats"], 2)
            self.assertEqual(summary["total_records"], 4)
            self.assertEqual(summary["token_usage"]["num_completed_responses"], 4)
            self.assertEqual(summary["metrics"]["avg_completed_response_tokens"], 1.0)
            self.assertEqual(summary["metrics"]["accuracy_first"], 1.0)
            self.assertEqual(summary["primary_score"], 1.0)
            self.assertFalse((task_dir / "predictions.jsonl").exists())
            self.assertTrue((task_dir / "run_01" / "predictions.jsonl").exists())
            self.assertTrue((task_dir / "run_02" / "predictions.jsonl").exists())

            run_1 = json.loads(
                (task_dir / "run_01" / "run_config.json").read_text()
            )
            run_2 = json.loads(
                (task_dir / "run_02" / "run_config.json").read_text()
            )
            self.assertEqual(run_1["generation_config"]["seed"], 0)
            self.assertEqual(run_2["generation_config"]["seed"], 1)

            summaries = [
                (task_dir / name / "summary.json").read_bytes()
                for name in (".", "run_01", "run_02")
            ]
            run_evaluation(
                model="fake-model",
                tasks="toy",
                output_dir=out,
                run_id="repeated",
                backend=NeverCalledBackend(),
                benchmarks_dir=root,
                num_repeats=2,
                generate_only=True,
            )
            self.assertEqual(
                [
                    (task_dir / name / "summary.json").read_bytes()
                    for name in (".", "run_01", "run_02")
                ],
                summaries,
            )

            def generate_parent(backend) -> dict:
                run_evaluation(
                    model="fake-model",
                    tasks="toy",
                    output_dir=out,
                    run_id="repeated",
                    backend=backend,
                    benchmarks_dir=root,
                    num_repeats=2,
                    generate_only=True,
                )
                parent = json.loads((task_dir / "summary.json").read_text())
                self.assertEqual(parent["phase"], "generate_only")
                self.assertFalse(parent["evaluation_complete"])
                self.assertEqual(parent["metrics"], {})
                self.assertIsNone(parent["primary_metric"])
                self.assertEqual(parent["warnings"], [])
                self.assertEqual(
                    [item["metrics"] for item in parent["repeats"]], [{}, {}]
                )
                return parent

            # Without a kept parent, or with a regenerated repeat, the parent
            # summarizes generation only; kept repeat summaries stay evaluated.
            (task_dir / "summary.json").unlink()
            parent = generate_parent(NeverCalledBackend())
            self.assertEqual((parent["existing_records"], parent["new_records"]), (4, 0))
            self.assertEqual(
                [(task_dir / name / "summary.json").read_bytes() for name in ("run_01", "run_02")],
                summaries[1:],
            )
            predictions_path = task_dir / "run_02" / "predictions.jsonl"
            predictions_path.write_text(
                predictions_path.read_text().splitlines()[0] + "\n"
            )
            parent = generate_parent(FakeBackend())
            self.assertEqual((parent["existing_records"], parent["new_records"]), (3, 1))
            self.assertEqual(
                (task_dir / "run_01" / "summary.json").read_bytes(), summaries[1]
            )

            # An unreadable summary is regenerated rather than kept.
            (task_dir / "run_01" / "summary.json").write_text("{")
            generate_parent(NeverCalledBackend())
            run_1_summary = json.loads((task_dir / "run_01" / "summary.json").read_text())
            self.assertEqual(run_1_summary["phase"], "generate_only")

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
            backend = FakeBackend()
            rebuilt = run_evaluation(
                model="fake-model",
                tasks="toy",
                output_dir=out,
                run_id="same_run",
                backend=backend,
                overwrite=True,
                benchmarks_dir=root,
            )
            summary = rebuilt["results"]["toy"]
            self.assertEqual(backend.calls, 1)
            self.assertEqual(summary["total_records"], 2)
            self.assertTrue(summary["evaluation_complete"])

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

    def test_inspect_prompts_uses_explicit_thinking_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "benchmarks"
            _write_toy_benchmark(root)

            class ThinkingTokenizer(FakeTokenizer):
                def apply_chat_template(
                    self,
                    prompt: list[dict[str, str]],
                    tokenize: bool = False,
                    add_generation_prompt: bool = True,
                    enable_thinking: bool | None = None,
                ) -> str:
                    del tokenize, add_generation_prompt
                    return f"thinking={enable_thinking}:{prompt[-1]['content']}"

            with mock.patch(
                "aethereval.core.runner.load_chat_tokenizer",
                return_value=ThinkingTokenizer(),
            ):
                inspected = inspect_prompts(
                    model="fake-model",
                    tasks="toy",
                    benchmarks_dir=root,
                    gen_overrides={"enable_thinking": False},
                )

            self.assertTrue(
                inspected["results"]["toy"][0]["prompt"].startswith("thinking=False:")
            )

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


    def test_generation_backend_starts_only_when_needed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "benchmarks"
            _write_toy_benchmark(root)
            kwargs = dict(
                model="fake-model",
                tasks="toy",
                output_dir=Path(tmp) / "outputs",
                run_id="lazy",
                benchmarks_dir=root,
            )
            backend = FakeBackend()
            backend.close = mock.Mock()
            with mock.patch(
                "aethereval.core.runner.create_backend", return_value=backend
            ) as create_backend:
                first = run_evaluation(**kwargs)
            create_backend.assert_called_once()
            self.assertEqual(backend.calls, 1)
            backend.close.assert_called_once()

            # Nothing is pending on a rerun, so the candidate model is never loaded.
            with mock.patch(
                "aethereval.core.runner.create_backend",
                side_effect=AssertionError("nothing to generate"),
            ) as create_backend:
                rerun = run_evaluation(**kwargs)
            create_backend.assert_not_called()
            self.assertEqual(rerun["backend"], first["backend"])
            self.assertEqual(
                rerun["results"]["toy"]["metrics"], first["results"]["toy"]["metrics"]
            )

if __name__ == "__main__":
    unittest.main()
