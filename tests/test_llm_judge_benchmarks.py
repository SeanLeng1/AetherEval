import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from aethereval.cli import build_parser
from aethereval.config import resolve_run_arguments
from aethereval.core.runner import run_evaluation
from aethereval.core.task_defaults import resolve_task_default_metrics
from aethereval.core.task_register import load_task
from aethereval.core.types import GenerationInput, GenerationOutput
from benchmark_utils.llm_judge import (
    chat_completion,
    parse_json_object,
    resolve_judge_settings,
)

BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"


class SequenceBackend:
    def __init__(self, responses: list[str]) -> None:
        self.responses = iter(responses)
        self.calls: list[list[GenerationInput]] = []

    def generate(self, inputs, gen_cfg):  # noqa: ANN001
        del gen_cfg
        self.calls.append(inputs)
        outputs = []
        for item in inputs:
            response = next(self.responses)
            outputs.append(
                GenerationOutput(
                    sample_id=item.sample_id,
                    prompt=item.prompt,
                    generations=[response],
                    meta={
                        "prompt_token_count": 1,
                        "response_token_counts": [1],
                    },
                )
            )
        return outputs


class NeverGenerateBackend:
    name = "never-generate"

    def generate(self, inputs, gen_cfg):  # noqa: ANN001
        del inputs, gen_cfg
        raise AssertionError("the task-level generation hook should be used")


class LlmJudgeBenchmarkTests(unittest.TestCase):
    def test_official_judge_models_come_from_task_defaults(self) -> None:
        expected = {
            "llmeval_med": ("gpt-4o", None, None, None),
            "healthbench": ("gpt-4.1-2025-04-14", 0.5, None, 2048),
            "writingbench": ("claude-sonnet-4-5", 1.0, 0.95, 2048),
            "creative_writing_v3": ("claude-sonnet-4-6", 0.0, None, 4096),
            "researchqa": ("gpt-4.1-mini", 0.0, None, None),
            "arena_hard_v2": ("gpt-4.1", 0.0, None, 16000),
        }
        for task_name, judge_defaults in expected.items():
            with self.subTest(task=task_name):
                judge_model, temperature, top_p, max_new_tokens = judge_defaults
                defaults = resolve_task_default_metrics(task_name)
                bundle = load_task(task_name, BENCHMARKS)
                self.assertEqual(defaults["judge_model"], judge_model)
                self.assertEqual(defaults["judge_temperature"], temperature)
                self.assertEqual(defaults["judge_top_p"], top_p)
                self.assertEqual(defaults["judge_max_new_tokens"], max_new_tokens)
                self.assertEqual(
                    bundle.metrics_module.DEFAULT_JUDGE_MODEL,
                    judge_model,
                )
                self.assertNotIn("metrics", bundle.task_module.DEFAULT_GEN)
                self.assertNotIn("judge_model", bundle.task_module.DEFAULT_GEN)

    def test_native_tasks_load_expected_release_sizes_and_defaults(self) -> None:
        expected = {
            "llmeval_med": (667, 1, 2048, 1.0),
            "healthbench": (5000, 1, 2048, 0.5),
            "writingbench": (1000, 1, 16000, 0.7),
            "creative_writing_v3": (96, 1, 12000, 0.7),
            "researchqa": (3750, 1, 2048, 0.0),
            "arena_hard_v2": (500, 1, 8192, 0.0),
        }
        for name, (count, n, max_tokens, temperature) in expected.items():
            with self.subTest(task=name):
                bundle = load_task(name, BENCHMARKS)
                samples = bundle.task_module.load_samples(bundle.spec.task_dir)
                self.assertEqual(len(samples), count)
                self.assertEqual(bundle.task_module.DEFAULT_GEN["n"], n)
                self.assertEqual(
                    bundle.task_module.DEFAULT_GEN["max_new_tokens"], max_tokens
                )
                self.assertEqual(
                    bundle.task_module.DEFAULT_GEN["temperature"], temperature
                )

    def test_judge_settings_and_json_parser(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "AETHEREVAL_JUDGE_API_KEY": "secret",
                "AETHEREVAL_JUDGE_BASE_URL": "http://judge.test/v1/",
            },
            clear=True,
        ):
            settings = resolve_judge_settings({}, default_model="judge-default")
        self.assertEqual(settings.model, "judge-default")
        self.assertEqual(settings.base_url, "http://judge.test/v1")
        self.assertEqual(settings.api_key, "secret")
        self.assertEqual(
            parse_json_object('prefix {"score": 8, "reason": "ok"} suffix')["score"],
            8,
        )
        self.assertTrue(
            parse_json_object('```json\n{"criteria_met": true}\n```')["criteria_met"]
        )

    def test_api_judge_uses_resolved_sampling_and_thinking_defaults(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"AETHEREVAL_JUDGE_API_KEY": "secret"},
            clear=True,
        ):
            settings = resolve_judge_settings(
                {
                    "judge_temperature": 0.25,
                    "judge_max_new_tokens": 512,
                    "judge_top_p": 0.9,
                    "judge_enable_thinking": False,
                },
                default_model="judge-default",
            )

        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = (
            b'{"choices":[{"message":{"content":"ok"}}]}'
        )
        with mock.patch(
            "benchmark_utils.llm_judge.urllib.request.urlopen",
            return_value=response,
        ) as urlopen:
            result = chat_completion(
                settings,
                [{"role": "user", "content": "grade"}],
            )

        self.assertEqual(result, "ok")
        request = urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(payload["temperature"], 0.25)
        self.assertEqual(payload["max_tokens"], 512)
        self.assertEqual(payload["top_p"], 0.9)
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})

    def test_judge_cli_options_are_forwarded_to_metrics_only(self) -> None:
        args = build_parser().parse_args(
            [
                "--model",
                "candidate",
                "--tasks",
                "healthbench",
                "--judge-model",
                "judge-alias",
                "--judge-base-url",
                "http://judge/v1",
                "--judge-api-key-env",
                "CUSTOM_KEY",
                "--judge-workers",
                "12",
                "--judge-timeout",
                "45",
                "--judge-max-retries",
                "7",
                "--judge-repeats",
                "3",
                "--judge-max-new-tokens",
                "8192",
                "--judge-temperature",
                "0.25",
                "--judge-top-p",
                "0.9",
                "--no-judge-enable-thinking",
            ]
        )
        resolved = resolve_run_arguments(args, {})
        self.assertEqual(
            resolved["metric_options"],
            {
                "judge_model": "judge-alias",
                "judge_base_url": "http://judge/v1",
                "judge_api_key_env": "CUSTOM_KEY",
                "judge_workers": 12,
                "judge_timeout": 45.0,
                "judge_max_retries": 7,
                "judge_repeats": 3,
                "judge_max_new_tokens": 8192,
                "judge_temperature": 0.25,
                "judge_top_p": 0.9,
                "judge_enable_thinking": False,
            },
        )
        self.assertNotIn("judge_model", resolved["backend_kwargs"])

    def test_creative_writing_retries_short_generations(self) -> None:
        bundle = load_task("creative_writing_v3", BENCHMARKS)
        sample = bundle.task_module.load_samples(bundle.spec.task_dir)[0]
        backend = SequenceBackend(["short", "still short", "x" * 500])
        outputs = bundle.task_module.generate_outputs(
            backend=backend,
            samples=[sample],
            pending_indices={sample.id: [0]},
            existing_records=[],
            gen_cfg=bundle.task_module.DEFAULT_GEN,
        )
        self.assertEqual(len(backend.calls), 3)
        self.assertEqual(outputs[0].generations, ["x" * 500])
        self.assertFalse(outputs[0].meta["creative_generation_failed"])

    def test_runner_uses_task_generation_hook_and_metric_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_dir = root / "hooked"
            data_dir = task_dir / "data"
            data_dir.mkdir(parents=True)
            (data_dir / "eval.jsonl").write_text('{"id":"one"}\n', encoding="utf-8")
            (task_dir / "task.py").write_text(
                "from aethereval.core.types import GenerationOutput, Sample\n"
                "TASK_NAME='hooked'\n"
                "DATA_FILE='data/eval.jsonl'\n"
                "DEFAULT_GEN={'n':1,'max_new_tokens':8,'temperature':0.0}\n"
                "def load_samples(task_dir):\n"
                "    return [Sample(id='one')]\n"
                "def build_prompt(sample):\n"
                "    return 'normal path'\n"
                "def generate_outputs(**kwargs):\n"
                "    return [GenerationOutput(sample_id='one', prompt='hook path', "
                "generations=['hooked'], meta={'prompt_token_count':2,"
                "'response_token_counts':[1]})]\n",
                encoding="utf-8",
            )
            metrics_source = (
                "PRIMARY_METRIC='accuracy'\n"
                "PRESERVE_EXISTING_SCORES_ON_RESUME=True\n"
                "def validate_metric_options(options):\n"
                "    assert options['probe'] == 'ok' and options['n'] == 1\n"
                "def score_generation(sample, generation):\n"
                "    return {'score': float(generation == 'hooked')}\n"
                "def aggregate(results, options=None):\n"
                "    return {'accuracy': results[0]['scores'][0]}\n"
            )
            metrics_path = task_dir / "metrics.py"
            metrics_path.write_text(metrics_source, encoding="utf-8")
            result = run_evaluation(
                model="unused",
                tasks="hooked",
                output_dir=root / "outputs",
                run_id="run",
                backend=NeverGenerateBackend(),
                benchmarks_dir=root,
                metric_options={"probe": "ok"},
            )
            metrics_path.write_text(
                metrics_source.replace(
                    "    return {'score': float(generation == 'hooked')}\n",
                    "    raise AssertionError('existing judge score must be preserved')\n",
                ),
                encoding="utf-8",
            )
            resumed = run_evaluation(
                model="unused",
                tasks="hooked",
                output_dir=root / "outputs",
                run_id="run",
                backend=NeverGenerateBackend(),
                benchmarks_dir=root,
                metric_options={"probe": "ok"},
            )
        self.assertEqual(result["results"]["hooked"]["primary_score"], 1.0)
        self.assertEqual(resumed["results"]["hooked"]["primary_score"], 1.0)

    def test_llmeval_med_builds_prior_turn_history(self) -> None:
        bundle = load_task("llmeval_med", BENCHMARKS)
        all_samples = bundle.task_module.load_samples(bundle.spec.task_dir)
        groups: dict[tuple[str, str], list] = {}
        for sample in all_samples:
            key = (sample.data["category"], sample.data["group_code"])
            groups.setdefault(key, []).append(sample)
        turns = next(
            sorted(values, key=lambda item: item.data["round"])
            for values in groups.values()
            if len(values) >= 2
        )[:2]
        backend = SequenceBackend(["first answer", "second answer"])
        outputs = bundle.task_module.generate_outputs(
            backend=backend,
            samples=turns,
            pending_indices={sample.id: [0] for sample in turns},
            existing_records=[],
            gen_cfg=bundle.task_module.DEFAULT_GEN,
        )
        self.assertEqual(len(outputs), 2)
        second_prompt = backend.calls[1][0].prompt
        self.assertIn({"role": "assistant", "content": "first answer"}, second_prompt)
        self.assertIn(
            {"role": "user", "content": turns[0].data["problem"]}, second_prompt
        )

    def test_writingbench_requirement_aggregation_matches_upstream_formula(
        self,
    ) -> None:
        metrics = load_task("writingbench", BENCHMARKS).metrics_module
        sample_results = [
            {
                "sample_id": "1",
                "meta": {
                    "domain1": "D1",
                    "domain2": "D2",
                    "requirement_subsets": ["style"],
                    "requirement_criteria": {"style": ["c1", "c2"]},
                },
                "records": [
                    {
                        "score": 6.0,
                        "meta": {"criterion_scores": {"c1": 8.0, "c2": 4.0}},
                    }
                ],
            }
        ]
        result = metrics.aggregate(sample_results)
        self.assertEqual(result["overall_score"], 60.0)
        self.assertEqual(result["requirement/style_R"], 60.0)
        self.assertEqual(result["requirement/style_C"], 60.0)

    def test_all_six_batched_judge_protocols_reassemble_scores(self) -> None:
        options = {"judge_workers": 1, "judge_max_retries": 0}
        with mock.patch.dict(
            os.environ, {"AETHEREVAL_JUDGE_API_KEY": "test"}, clear=True
        ):
            health = load_task("healthbench", BENCHMARKS)
            health_sample = health.task_module.load_samples(health.spec.task_dir)[0]
            health_output = GenerationOutput(
                health_sample.id,
                health.task_module.build_prompt(health_sample),
                ["answer"],
            )
            with mock.patch.object(
                health.metrics_module,
                "chat_completion",
                return_value='{"criteria_met": true, "explanation": "ok"}',
            ):
                health_score = health.metrics_module.score_generations_batch(
                    [health_sample], [health_output], options
                )[0][0]["score"]
            rubrics = health_sample.data["rubrics"]
            expected_health = sum(float(item["points"]) for item in rubrics) / sum(
                float(item["points"]) for item in rubrics if float(item["points"]) > 0
            )
            self.assertEqual(health_score, expected_health)

            writing = load_task("writingbench", BENCHMARKS)
            writing_sample = writing.task_module.load_samples(writing.spec.task_dir)[0]
            writing_output = GenerationOutput(writing_sample.id, "prompt", ["response"])
            with mock.patch.object(
                writing.metrics_module,
                "chat_completion",
                return_value='{"score": 8, "reason": "ok"}',
            ):
                writing_score = writing.metrics_module.score_generations_batch(
                    [writing_sample], [writing_output], options
                )[0][0]["score"]
            self.assertEqual(writing_score, 8.0)

            creative = load_task("creative_writing_v3", BENCHMARKS)
            creative_sample = creative.task_module.load_samples(creative.spec.task_dir)[
                0
            ]
            creative_output = GenerationOutput(
                creative_sample.id, "prompt", ["x" * 500]
            )
            creative_judgment = "\n".join(
                f"{name}: 10" for name in creative.metrics_module.CRITERIA
            )
            with mock.patch.object(
                creative.metrics_module,
                "chat_completion",
                return_value=creative_judgment,
            ):
                creative_score = creative.metrics_module.score_generations_batch(
                    [creative_sample], [creative_output], options
                )[0][0]["score"]
            self.assertEqual(creative_score, 10.0)

            research = load_task("researchqa", BENCHMARKS)
            research_samples = research.task_module.load_samples(research.spec.task_dir)
            research_sample = next(
                sample for sample in research_samples if len(sample.data["rubric"]) <= 8
            )
            research_output = GenerationOutput(research_sample.id, "prompt", ["answer"])
            research_judgment = "\n".join(
                "Completely" for _ in research_sample.data["rubric"]
            )
            with mock.patch.object(
                research.metrics_module,
                "chat_completion",
                return_value=research_judgment,
            ):
                research_score = research.metrics_module.score_generations_batch(
                    [research_sample], [research_output], options
                )[0][0]["score"]
            self.assertEqual(research_score, 1.0)

            medical = load_task("llmeval_med", BENCHMARKS)
            medical_sample = medical.task_module.load_samples(medical.spec.task_dir)[0]
            medical_output = GenerationOutput(
                medical_sample.id,
                medical.task_module.build_prompt(medical_sample),
                ["answer"],
            )
            with mock.patch.object(
                medical.metrics_module,
                "chat_completion",
                return_value='{"得分":"[4]"}',
            ):
                medical_score = medical.metrics_module.score_generations_batch(
                    [medical_sample], [medical_output], options
                )[0][0]["score"]
            self.assertEqual(medical_score, 4.0)

            arena = load_task("arena_hard_v2", BENCHMARKS)
            arena_sample = arena.task_module.load_samples(arena.spec.task_dir)[0]
            arena_output = GenerationOutput(arena_sample.id, "prompt", ["answer"])
            with mock.patch.object(
                arena.metrics_module,
                "chat_completion",
                side_effect=["[[B>A]]", "[[A>B]]"],
            ):
                arena_score = arena.metrics_module.score_generations_batch(
                    [arena_sample], [arena_output], options
                )[0][0]["score"]
            self.assertEqual(arena_score, 1.0)

    def test_arena_style_metadata_matches_published_baseline_metadata(self) -> None:
        bundle = load_task("arena_hard_v2", BENCHMARKS)
        sample = bundle.task_module.load_samples(bundle.spec.task_dir)[0]
        actual = bundle.metrics_module._style_metadata(sample.data["baseline_answer"])
        self.assertEqual(actual, sample.meta["baseline_metadata"])


if __name__ == "__main__":
    unittest.main()
