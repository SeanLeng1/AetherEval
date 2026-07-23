import contextlib
import csv
import io
import json
import os
import unittest
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from aethereval.cli import (
    _split_native_external_tasks,
    build_parser,
    run_selected_tasks,
)
from aethereval.config import resolve_run_arguments
from aethereval.core.io import run_output_dir
from aethereval.core.task_defaults import resolve_task_default_gen
from benchmarks.bfcl._compat import _set_bfcl_project_root
from benchmarks.bfcl.cli import build_bfcl_spec
from benchmarks.bfcl.external import (
    ExternalRunSpec,
    _filter_bfcl_prints,
    _evaluation_run_paths,
    _gen_args,
    _is_allowed_zero_score_error,
    _require_web_search_key,
    _raise_on_inference_errors,
    _run_generation,
    _run_generations,
    _run_seed,
    _server_command_for_spec,
    _warn_memory_vector_requirements,
    add_comparison_metrics,
    average_run_metrics,
    compute_format_rates,
    parse_scores,
    run as run_bfcl_external,
    write_predictions_jsonl,
)
from benchmarks.bfcl.register import register_rlla_model


class ExternalCliTests(unittest.TestCase):
    def test_bfcl_v4_requires_serpapi_only_for_web_search_generation(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "SERPAPI_API_KEY"):
                _require_web_search_key(["all"])
            _require_web_search_key(["non_live", "live", "multi_turn"])

    def test_bfcl_warns_only_when_memory_vector_is_selected(self) -> None:
        with self.assertWarnsRegex(RuntimeWarning, "all-MiniLM-L6-v2"):
            _warn_memory_vector_requirements(["all"])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _warn_memory_vector_requirements(["live", "non_live", "multi_turn"])
        self.assertEqual(caught, [])

    def test_local_judge_automatically_splits_generation_and_evaluation(self) -> None:
        args = build_parser().parse_args(
            [
                "--model",
                "candidate/model",
                "--tasks",
                "healthbench",
                "--judge-backend",
                "local",
                "--judge-model",
                "local/judge",
                "--overwrite",
            ]
        )
        resolved = resolve_run_arguments(args, {})
        result = {"results": {"healthbench": {"metrics": {"score": 0.5}}}}

        with mock.patch(
            "aethereval.cli.run_evaluation",
            side_effect=[result, result],
        ) as run_evaluation:
            actual = run_selected_tasks(args, resolved)

        self.assertIs(actual, result)
        self.assertEqual(run_evaluation.call_count, 2)
        generate_call = run_evaluation.call_args_list[0].kwargs
        evaluate_call = run_evaluation.call_args_list[1].kwargs
        self.assertTrue(generate_call["generate_only"])
        self.assertFalse(generate_call["eval_only"])
        self.assertTrue(generate_call["overwrite"])
        self.assertFalse(evaluate_call["generate_only"])
        self.assertTrue(evaluate_call["eval_only"])
        self.assertFalse(evaluate_call["overwrite"])

    def test_api_judge_automatically_splits_generation_and_evaluation(self) -> None:
        args = build_parser().parse_args(
            [
                "--model",
                "candidate/model",
                "--tasks",
                "healthbench,llmeval_med",
                "--overwrite",
            ]
        )
        resolved = resolve_run_arguments(args, {})
        result = {
            "results": {
                "healthbench": {"metrics": {"score": 0.5}},
                "llmeval_med": {"metrics": {"OP": 30.0}},
            }
        }

        with mock.patch(
            "aethereval.cli.run_evaluation",
            side_effect=[result, result],
        ) as run_evaluation:
            actual = run_selected_tasks(args, resolved)

        self.assertIs(actual, result)
        self.assertEqual(run_evaluation.call_count, 2)
        generate_call = run_evaluation.call_args_list[0].kwargs
        evaluate_call = run_evaluation.call_args_list[1].kwargs
        self.assertEqual(generate_call["tasks"], "healthbench,llmeval_med")
        self.assertTrue(generate_call["generate_only"])
        self.assertFalse(generate_call["eval_only"])
        self.assertTrue(generate_call["overwrite"])
        self.assertEqual(evaluate_call["tasks"], "healthbench,llmeval_med")
        self.assertFalse(evaluate_call["generate_only"])
        self.assertTrue(evaluate_call["eval_only"])
        self.assertFalse(evaluate_call["overwrite"])

    def test_thinking_mode_flags_are_tri_state(self) -> None:
        parser = build_parser()

        self.assertIsNone(parser.parse_args([]).enable_thinking)
        self.assertIs(parser.parse_args(["--enable-thinking"]).enable_thinking, True)
        self.assertIs(
            parser.parse_args(["--no-enable-thinking"]).enable_thinking,
            False,
        )
        self.assertIsNone(parser.parse_args([]).judge_enable_thinking)
        self.assertIs(
            parser.parse_args(["--judge-enable-thinking"]).judge_enable_thinking,
            True,
        )
        self.assertIs(
            parser.parse_args(["--no-judge-enable-thinking"]).judge_enable_thinking,
            False,
        )

    def test_split_tasks_accepts_external_name(self) -> None:
        native_tasks, external_tasks = _split_native_external_tasks("ifeval,bfcl")

        self.assertEqual(native_tasks, ["ifeval"])
        self.assertEqual(external_tasks, ["bfcl"])

    def test_bfcl_external_spec(self) -> None:
        args = build_parser().parse_args(
            [
                "--tasks",
                "bfcl",
                "--model",
                "rlla-gdpo",
                "--backend",
                "sglang",
                "--output-dir",
                "outputs/bfcl",
                "--categories",
                "non_live,live",
                "--tp-size",
                "4",
                "--temperature",
                "0.2",
                "--max-new-tokens",
                "2048",
                "--context-length",
                "8192",
                "--top-p",
                "0.9",
                "--top-k",
                "50",
                "--seed",
                "123",
                "--num-threads",
                "8",
                "--bfcl-verbose",
                "--no-overwrite",
                "--generate-only",
            ]
        )

        resolved = resolve_run_arguments(args, {})
        spec = build_bfcl_spec(args, resolved, Path(args.output_dir))

        self.assertEqual(spec.categories, ["non_live", "live"])
        self.assertEqual(spec.num_gpus, 4)
        self.assertEqual(spec.dp_size, 1)
        self.assertEqual(spec.tp_size, 4)
        self.assertEqual(spec.num_threads, 8)
        self.assertEqual(spec.temperature, 0.2)
        self.assertEqual(spec.max_tokens, 2048)
        self.assertEqual(spec.max_context_length, 8192)
        self.assertEqual(spec.top_p, 0.9)
        self.assertEqual(spec.top_k, 50)
        self.assertEqual(spec.seed, 123)
        self.assertEqual(spec.num_runs, 4)
        self.assertTrue(spec.verbose)
        self.assertFalse(spec.allow_overwrite)
        self.assertTrue(spec.run_generation)
        self.assertFalse(spec.run_evaluation)

    def test_bfcl_external_spec_reads_generation_defaults_from_config(self) -> None:
        args = build_parser().parse_args(
            ["--tasks", "bfcl", "--model", "rlla-gdpo", "--backend", "sglang"]
        )
        resolved = resolve_run_arguments(args, {})

        with mock.patch(
            "benchmarks.bfcl.cli.resolve_task_default_gen",
            return_value={
                "n": 1,
                "max_new_tokens": 1234,
                "temperature": 0.25,
                "top_p": 0.8,
                "top_k": 17,
            },
        ):
            spec = build_bfcl_spec(args, resolved, Path("outputs"))

        self.assertEqual(spec.max_tokens, 1234)
        self.assertEqual(spec.temperature, 0.25)
        self.assertEqual(spec.top_p, 0.8)
        self.assertEqual(spec.top_k, 17)
        self.assertEqual(spec.num_runs, 1)

    def test_bfcl_python_spec_defaults_match_task_config(self) -> None:
        configured = resolve_task_default_gen("bfcl", {})
        spec = ExternalRunSpec(model="model", output_dir=Path("output"))

        self.assertEqual(spec.max_tokens, configured["max_new_tokens"])
        self.assertEqual(spec.temperature, configured["temperature"])
        self.assertEqual(spec.top_p, configured["top_p"])
        self.assertEqual(spec.top_k, configured["top_k"])
        self.assertEqual(spec.num_runs, configured["n"])
        self.assertEqual(spec.categories, ["live", "non_live", "multi_turn"])

    def test_bfcl_external_spec_supports_unified_phase_flags(self) -> None:
        generate_args = build_parser().parse_args(
            ["--tasks", "bfcl", "--model", "model", "--generate-only"]
        )
        generate_spec = build_bfcl_spec(
            generate_args,
            resolve_run_arguments(generate_args, {}),
            Path("outputs"),
        )
        self.assertTrue(generate_spec.run_generation)
        self.assertFalse(generate_spec.run_evaluation)

        eval_args = build_parser().parse_args(
            ["--tasks", "bfcl", "--model", "model", "--eval-only"]
        )
        eval_spec = build_bfcl_spec(
            eval_args,
            resolve_run_arguments(eval_args, {}),
            Path("outputs"),
        )
        self.assertFalse(eval_spec.run_generation)
        self.assertTrue(eval_spec.run_evaluation)

    def test_phase_flags_are_mutually_exclusive(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                build_parser().parse_args(["--generate-only", "--eval-only"])

    def test_bfcl_external_spec_prefers_dp_size_for_num_gpus(self) -> None:
        args = build_parser().parse_args(
            [
                "--tasks",
                "bfcl",
                "--model",
                "rlla-gdpo",
                "--backend",
                "sglang",
                "--output-dir",
                "outputs/bfcl",
                "--dp-size",
                "8",
                "--tp-size",
                "1",
            ]
        )

        spec = build_bfcl_spec(
            args,
            resolve_run_arguments(args, {}),
            Path(args.output_dir),
        )

        self.assertEqual(spec.num_gpus, 8)
        self.assertEqual(spec.dp_size, 8)
        self.assertEqual(spec.tp_size, 1)
        self.assertEqual(spec.router_policy, "cache_aware")
        self.assertNotIn("log_level", spec.sglang_server_args)
        self.assertNotIn("log_level_http", spec.sglang_server_args)
        self.assertNotIn("router_log_level", spec.sglang_server_args)

    def test_bfcl_generation_reuses_managed_sglang_service(self) -> None:
        spec = ExternalRunSpec(
            model="test/model",
            output_dir=Path("outputs"),
            backend="sglang",
            dp_size=8,
            tp_size=1,
            router_policy="cache_aware",
            sglang_server_args={
                "context_length": 131072,
                "router_log_level": "warn",
            },
        )
        service = mock.Mock(base_url="http://127.0.0.1:18443")
        observed = {}

        def generation_main(args):  # noqa: ANN001
            observed["args"] = args
            observed["host"] = os.environ.get("LOCAL_SERVER_ENDPOINT")
            observed["port"] = os.environ.get("LOCAL_SERVER_PORT")
            observed["generate_url"] = os.environ.get(
                "RLLA_BFCL_GENERATE_URL"
            )

        with (
            mock.patch(
                "benchmarks.bfcl.external.SGLangService",
                return_value=service,
            ) as service_cls,
            mock.patch(
                "benchmarks.bfcl.external._filter_bfcl_prints",
                return_value=contextlib.nullcontext(),
            ),
            mock.patch.dict(
                os.environ,
                {
                    "LOCAL_SERVER_ENDPOINT": "old-host",
                    "LOCAL_SERVER_PORT": "1234",
                    "RLLA_BFCL_GENERATE_URL": "http://old/generate",
                },
            ),
        ):
            _run_generation(spec, Path("outputs/result"), generation_main)
            self.assertEqual(os.environ["LOCAL_SERVER_ENDPOINT"], "old-host")
            self.assertEqual(os.environ["LOCAL_SERVER_PORT"], "1234")
            self.assertEqual(
                os.environ["RLLA_BFCL_GENERATE_URL"],
                "http://old/generate",
            )

        service_cls.assert_called_once_with(
            model="test/model",
            dp_size=8,
            tensor_parallel_size=1,
            model_kwargs={
                "context_length": 131072,
                "router_log_level": "warn",
            },
            router_policy="cache_aware",
        )
        service.close.assert_called_once_with()
        self.assertTrue(observed["args"].skip_server_setup)
        self.assertEqual(observed["args"].num_threads, spec.num_threads)
        self.assertEqual(observed["host"], "127.0.0.1")
        self.assertEqual(observed["port"], "18443")
        self.assertEqual(
            observed["generate_url"],
            "http://127.0.0.1:18443/generate",
        )

    def test_bfcl_repeated_generation_reuses_server_and_advances_seed(self) -> None:
        spec = ExternalRunSpec(
            model="test/model",
            output_dir=Path("outputs"),
            backend="sglang",
            seed=10,
        )
        service = mock.Mock(base_url="http://127.0.0.1:18443")
        observed = []

        def generation_main(args):  # noqa: ANN001
            observed.append(
                (args.result_dir, os.environ.get("RLLA_BFCL_SEED"))
            )

        with (
            mock.patch(
                "benchmarks.bfcl.external.SGLangService",
                return_value=service,
            ) as service_cls,
            mock.patch(
                "benchmarks.bfcl.external._filter_bfcl_prints",
                return_value=contextlib.nullcontext(),
            ),
        ):
            _run_generations(
                spec,
                [(0, Path("result/run_01")), (1, Path("result/run_02"))],
                generation_main,
            )

        service_cls.assert_called_once()
        service.close.assert_called_once()
        self.assertEqual(
            observed,
            [
                (Path("result/run_01"), "10"),
                (Path("result/run_02"), "11"),
            ],
        )

    def test_bfcl_four_run_paths_and_metric_average(self) -> None:
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            paths = _evaluation_run_paths(out, 4)

        self.assertEqual(paths[0], (out / "result/run_01", out / "score/run_01"))
        self.assertEqual(paths[-1], (out / "result/run_04", out / "score/run_04"))
        self.assertEqual(_run_seed(ExternalRunSpec("m", Path("o")), 3), 3)

        runs = [
            {
                "live_acc": 70.0 + 2 * index,
                "non_live_acc": 80.0 + 2 * index,
                "multi_turn_acc": 10.0 + 2 * index,
                "live_format": 90.0,
                "non_live_format": 80.0,
                "multi_turn_format": 70.0,
            }
            for index in range(4)
        ]
        averaged = average_run_metrics(runs)

        self.assertEqual(averaged["live_acc"], 73.0)
        self.assertEqual(averaged["non_live_acc"], 83.0)
        self.assertEqual(averaged["multi_turn_acc"], 13.0)
        self.assertEqual(averaged["avg_acc"], 56.33)
        self.assertEqual(averaged["avg_format"], 80.0)

    def test_bfcl_run_writes_four_run_average_summary(self) -> None:
        per_run_metrics = [
            {
                "live_acc": 70.0 + 2 * index,
                "non_live_acc": 80.0 + 2 * index,
                "multi_turn_acc": 10.0 + 2 * index,
            }
            for index in range(4)
        ]
        format_rates = {
            "live": 90.0,
            "non_live": 80.0,
            "multi_turn": 70.0,
        }
        with (
            TemporaryDirectory() as tmp,
            mock.patch("benchmarks.bfcl.external.register_rlla_model"),
            mock.patch("benchmarks.bfcl.external._warn_memory_vector_requirements"),
            mock.patch(
                "bfcl_eval.eval_checker.eval_runner.main"
            ) as evaluation_main,
            mock.patch(
                "benchmarks.bfcl.external.parse_scores",
                side_effect=per_run_metrics,
            ),
            mock.patch(
                "benchmarks.bfcl.external.compute_format_rates",
                return_value=format_rates,
            ),
        ):
            out = Path(tmp) / "bfcl"
            result = run_bfcl_external(
                ExternalRunSpec(
                    model="dry-model",
                    output_dir=out,
                    num_runs=4,
                    run_generation=False,
                    run_evaluation=True,
                )
            )
            summary = json.loads((out / "summary.json").read_text())

        self.assertEqual(evaluation_main.call_count, 4)
        self.assertEqual(result.metrics["avg_acc"], 56.33)
        self.assertEqual(result.metrics["avg_format"], 80.0)
        self.assertEqual(result.primary_metric, "avg_acc")
        self.assertEqual(summary["num_runs"], 4)
        self.assertEqual(summary["run_seeds"], [0, 1, 2, 3])
        self.assertEqual(len(summary["runs"]), 4)
        self.assertEqual(summary["metrics"]["live_acc"], 73.0)

    def test_bfcl_only_context_and_sglang_args_override_global_values(self) -> None:
        args = build_parser().parse_args(
            [
                "--tasks",
                "apibank,bfcl",
                "--model",
                "rlla-gdpo",
                "--backend",
                "sglang",
                "--context-length",
                "32768",
                "--sglang-arg",
                "chunked_prefill_size=4096",
                "--sglang-arg",
                "schedule_conservativeness=1.0",
                "--bfcl-context-length",
                "131072",
                "--bfcl-sglang-arg",
                "schedule_conservativeness=0.3",
                "--bfcl-sglang-arg",
                'json_model_override_args={"max_position_embeddings":131072,'
                '"rope_parameters":{"rope_theta":1000000.0,"rope_type":"yarn",'
                '"factor":4.0,"original_max_position_embeddings":32768}}',
            ]
        )
        resolved = resolve_run_arguments(args, {})
        spec = build_bfcl_spec(args, resolved, Path("outputs"))

        self.assertEqual(resolved["backend_kwargs"]["context_length"], 32768)
        self.assertNotIn("json_model_override_args", resolved["backend_kwargs"])
        self.assertEqual(spec.max_context_length, 131072)
        self.assertEqual(spec.sglang_server_args["context_length"], 131072)
        self.assertEqual(spec.sglang_server_args["chunked_prefill_size"], 4096)
        self.assertEqual(spec.sglang_server_args["schedule_conservativeness"], 0.3)
        self.assertEqual(
            spec.sglang_server_args["json_model_override_args"],
            {
                "max_position_embeddings": 131072,
                "rope_parameters": {
                    "rope_theta": 1000000.0,
                    "rope_type": "yarn",
                    "factor": 4.0,
                    "original_max_position_embeddings": 32768,
                },
            },
        )

    def test_bfcl_legacy_flags_are_removed(self) -> None:
        parser = build_parser()
        for flag in ("--num-gpus", "--skip-generation", "--skip-evaluation"):
            with self.subTest(flag=flag), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parser.parse_args([flag, "1"] if flag == "--num-gpus" else [flag])

    def test_external_benchmark_flag_is_removed(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                build_parser().parse_args(["--external-benchmark", "bfcl"])

    def test_model_path_flag_is_removed(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                build_parser().parse_args(["--model-path", "/tmp/model"])

    def test_bfcl_model_name_and_explicit_run_id_match_native_layout(self) -> None:
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "outputs"
            model = "qwen2.5/huggingface"
            model_name = "qwen2.5_huggingface"
            args = build_parser().parse_args(
                [
                    "--tasks",
                    "bfcl",
                    "--model",
                    model,
                    "--model-name",
                    model_name,
                    "--output-dir",
                    str(out),
                    "--run-id",
                    "production-1",
                ]
            )

            resolved = resolve_run_arguments(args, {})
            self.assertEqual(
                run_output_dir(out, model, "production-1", model_name),
                out / model_name / "production-1",
            )

            spec = build_bfcl_spec(args, resolved, out / model_name / "production-1")
            generation_args = _gen_args(spec, out / "raw")
            self.assertEqual(spec.model_name, model_name)
            self.assertEqual(generation_args.model, [model])
            self.assertIsNone(generation_args.local_model_path)

    def test_bfcl_registry_handles_slashes_and_underscores(self) -> None:
        model = "/scratch/checkpoints/my_model_v2"
        with TemporaryDirectory() as tmp:
            register_rlla_model(model, project_root=tmp)

        from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING

        # BFCL escapes '/' to '_' for its result folder, then its evaluator turns
        # every '_' back into '/'. Both keys must resolve to the same handler config.
        evaluator_key = model.replace("/", "_").replace("_", "/")
        self.assertIs(MODEL_CONFIG_MAPPING[model], MODEL_CONFIG_MAPPING[evaluator_key])
        self.assertEqual(MODEL_CONFIG_MAPPING[model].model_name, model)

    def test_bfcl_predictions_jsonl_uses_aethereval_schema(self) -> None:
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "bfcl"
            result_dir = out / "result"
            score_dir = out / "score"
            model_dir = result_dir / "dry-model"
            score_model_dir = score_dir / "dry-model"
            model_dir.mkdir(parents=True)
            score_model_dir.mkdir(parents=True)
            (model_dir / "BFCL_v4_simple_python_result.json").write_text(
                json.dumps(
                    {
                        "id": "simple_python_1",
                        "result": "<think>x</think>",
                        "inference_input_log": {"formatted_prompt": "prompt text"},
                    }
                )
                + "\n"
                + json.dumps(
                    {
                        "id": "simple_python_2",
                        "result": "Error during inference: BFCL prompt exceeds max context length.",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (score_model_dir / "BFCL_v4_simple_python_score.json").write_text(
                json.dumps({"accuracy": 0.5})
                + "\n"
                + json.dumps({"id": "simple_python_2", "valid": False})
                + "\n",
                encoding="utf-8",
            )

            stats = write_predictions_jsonl(
                out=out,
                result_dir=result_dir,
                score_dir=score_dir,
                model="dry-model",
            )

            predictions_path = out / "predictions.jsonl"
            rows = [
                json.loads(line)
                for line in predictions_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(stats["prediction_records"], 2)
            self.assertEqual(stats["prediction_scored_records"], 2)
            self.assertEqual(rows[0]["sample_id"], "simple_python_1")
            self.assertEqual(rows[0]["gen_idx"], 0)
            self.assertEqual(rows[0]["prompt"], "prompt text")
            self.assertEqual(rows[0]["generation"], "<think>x</think>")
            self.assertEqual(rows[0]["score"], 1.0)
            self.assertTrue(rows[0]["is_pass"])
            self.assertEqual(rows[0]["parsed"], "<think>x</think>")
            self.assertIsNone(rows[0]["gold"])
            self.assertIsNone(rows[0]["error"])
            self.assertEqual(rows[0]["meta"]["benchmark"], "bfcl")
            self.assertEqual(rows[0]["meta"]["test_category"], "simple_python")
            self.assertFalse(rows[1]["is_pass"])
            self.assertEqual(rows[1]["score"], 0.0)
            self.assertIn("Error during inference", rows[1]["error"])

    def test_bfcl_format_rate_uses_single_turn_subset_expectations(self) -> None:
        with TemporaryDirectory() as tmp:
            result_dir = Path(tmp) / "result"
            model_dir = result_dir / "dry-model" / "non_live"
            model_dir.mkdir(parents=True)
            tool_call_records = [
                {
                    "id": "simple_python_1",
                    "result": (
                        "<think>x</think>\n<tool_call>\n{}\n</tool_call>"
                        "<|im_end|>"
                    ),
                },
                {
                    "id": "simple_python_2",
                    "result": "<think>x</think>\n<response>done</response>",
                },
                {
                    "id": "simple_python_3",
                    "result": (
                        "<think>x</think>\n<tool_call>\n{}\n</tool_call>"
                        "<tool_call>\n{}\n</tool_call>"
                    ),
                },
            ]
            (model_dir / "BFCL_v4_simple_python_result.json").write_text(
                "\n".join(json.dumps(record) for record in tool_call_records) + "\n",
                encoding="utf-8",
            )
            response_records = [
                {
                    "id": "irrelevance_1",
                    "result": (
                        "<think>x</think>\n<response>done</response><|im_end|>"
                    ),
                },
                {
                    "id": "irrelevance_2",
                    "result": "<think>x</think>\n<tool_call>\n{}\n</tool_call>",
                },
            ]
            (model_dir / "BFCL_v4_irrelevance_result.json").write_text(
                "\n".join(json.dumps(record) for record in response_records) + "\n",
                encoding="utf-8",
            )

            rates = compute_format_rates(result_dir, "dry-model")

            self.assertEqual(rates["non_live"], 40.0)

    def test_bfcl_format_rate_uses_multi_turn_ground_truth_and_terminal_step(
        self,
    ) -> None:
        tool_call = "<think>x</think>\n<tool_call>\n{}\n</tool_call>"
        response = "<think>x</think>\n<response>done</response>"
        with TemporaryDirectory() as tmp:
            result_dir = Path(tmp) / "result"
            model_dir = result_dir / "dry-model" / "multi_turn"
            model_dir.mkdir(parents=True)
            records = [
                {
                    "id": "multi_turn_miss_param_1",
                    "result": [[tool_call, response], [response]],
                },
                {
                    "id": "multi_turn_miss_param_2",
                    "result": [[response]],
                },
                {
                    "id": "multi_turn_miss_param_3",
                    "result": [[tool_call, tool_call]],
                },
            ]
            (model_dir / "BFCL_v4_multi_turn_miss_param_result.json").write_text(
                "\n".join(json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )
            ground_truth = {
                "multi_turn_miss_param_1": [["call()"], []],
                "multi_turn_miss_param_2": [["call()"]],
                "multi_turn_miss_param_3": [["call()"]],
            }

            with mock.patch(
                "benchmarks.bfcl.external._load_ground_truth_by_id",
                return_value=ground_truth,
            ):
                rates = compute_format_rates(result_dir, "dry-model")

            self.assertAlmostEqual(rates["multi_turn"], 400.0 / 6.0)

    def test_bfcl_parse_scores_reports_toolrl_columns(self) -> None:
        with TemporaryDirectory() as tmp:
            score_dir = Path(tmp) / "score"
            score_dir.mkdir()
            columns = [
                "Model",
                "Overall Acc",
                "Non-Live AST Acc",
                "Live Acc",
                "Multi Turn Acc",
                "Web Search Acc",
                "Memory Acc",
                "Relevance Detection",
                "Irrelevance Detection",
                "Format Sensitivity Max Delta",
                "Format Sensitivity Standard Deviation",
            ]
            with (score_dir / "data_overall.csv").open(
                "w", encoding="utf-8", newline=""
            ) as f:
                writer = csv.DictWriter(f, fieldnames=columns)
                writer.writeheader()
                writer.writerow(
                    {
                        "Model": "dry-model",
                        "Overall Acc": "38.35%",
                        "Non-Live AST Acc": "56.33%",
                        "Live Acc": "57.31%",
                        "Multi Turn Acc": "0.25%",
                        "Web Search Acc": "20.00%",
                        "Memory Acc": "30.00%",
                        "Relevance Detection": "77.78%",
                        "Irrelevance Detection": "41.84%",
                        "Format Sensitivity Max Delta": "12.5",
                        "Format Sensitivity Standard Deviation": "3.25",
                    }
                )
            with (score_dir / "data_agentic.csv").open(
                "w", encoding="utf-8", newline=""
            ) as f:
                writer = csv.DictWriter(
                    f, fieldnames=["Model", "Agentic Overall Acc"]
                )
                writer.writeheader()
                writer.writerow(
                    {"Model": "dry-model", "Agentic Overall Acc": "25.00%"}
                )

            metrics = parse_scores(score_dir, "dry-model")

            self.assertEqual(metrics["official_overall_acc"], 38.35)
            self.assertEqual(metrics["non_live_acc"], 56.33)
            self.assertEqual(metrics["live_acc"], 57.31)
            self.assertEqual(metrics["multi_turn_acc"], 0.25)
            self.assertEqual(metrics["agentic_acc"], 25.0)
            self.assertEqual(metrics["web_search_acc"], 20.0)
            self.assertEqual(metrics["memory_acc"], 30.0)
            self.assertEqual(metrics["relevance_detection"], 77.78)
            self.assertEqual(metrics["irrelevance_detection"], 41.84)
            self.assertEqual(metrics["format_sensitivity_max_delta"], 12.5)
            self.assertEqual(metrics["format_sensitivity_std"], 3.25)
            self.assertEqual(metrics["avg_acc"], 37.96)

            add_comparison_metrics(
                metrics,
                {"live": 77.44, "non_live": 95.11, "multi_turn": 57.40},
            )
            self.assertEqual(metrics["live_format"], 77.44)
            self.assertEqual(metrics["non_live_format"], 95.11)
            self.assertEqual(metrics["multi_turn_format"], 57.4)
            self.assertEqual(metrics["avg_format"], 76.65)

            self.assertEqual(
                set(metrics),
                {
                    "official_overall_acc",
                    "non_live_acc",
                    "live_acc",
                    "multi_turn_acc",
                    "agentic_acc",
                    "web_search_acc",
                    "memory_acc",
                    "relevance_detection",
                    "irrelevance_detection",
                    "format_sensitivity_max_delta",
                    "format_sensitivity_std",
                    "live_format",
                    "non_live_format",
                    "multi_turn_format",
                    "avg_acc",
                    "avg_format",
                },
            )

    def test_bfcl_comparison_metrics_match_paper_macro_average(self) -> None:
        metrics = {
            "live_acc": 72.73,
            "non_live_acc": 84.75,
            "multi_turn_acc": 12.50,
        }

        add_comparison_metrics(
            metrics,
            {"live": 77.44, "non_live": 95.11, "multi_turn": 57.40},
        )

        self.assertEqual(metrics["avg_acc"], 56.66)
        self.assertEqual(metrics["avg_format"], 76.65)

    def test_bfcl_uses_resolved_generation_config(self) -> None:
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "outputs"
            args = build_parser().parse_args(
                [
                    "--tasks",
                    "bfcl",
                    "--model",
                    "dry-model",
                    "--output-dir",
                    str(out),
                ]
            )
            resolved = resolve_run_arguments(
                args,
                {
                    "runtime": {"backend": "sglang", "dp_size": 3},
                    "generation": {
                        "temperature": 0.25,
                        "max_new_tokens": 1234,
                        "top_p": 0.77,
                        "top_k": 11,
                    },
                    "sglang": {"context_length": 9999},
                },
            )

            spec = build_bfcl_spec(args, resolved, out / "dry-model" / "bfcl")

            self.assertEqual(spec.backend, "sglang")
            self.assertEqual(spec.num_gpus, 3)
            self.assertEqual(spec.dp_size, 3)
            self.assertEqual(spec.tp_size, 1)
            self.assertEqual(spec.router_policy, "cache_aware")
            self.assertEqual(spec.num_threads, 48)
            self.assertEqual(spec.temperature, 0.25)
            self.assertEqual(spec.max_tokens, 1234)
            self.assertEqual(spec.max_context_length, 9999)
            self.assertEqual(spec.top_p, 0.77)
            self.assertEqual(spec.top_k, 11)
            self.assertFalse(spec.verbose)

    def test_bfcl_inference_errors_fail_fast(self) -> None:
        with TemporaryDirectory() as tmp:
            result_dir = Path(tmp) / "result"
            model_dir = result_dir / "dry-model"
            model_dir.mkdir(parents=True)
            (model_dir / "BFCL_v4_simple_python_result.json").write_text(
                json.dumps(
                    {
                        "id": "simple_python_1",
                        "result": "Error during inference: Connection error.",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "simple_python_1"):
                _raise_on_inference_errors(result_dir, "dry-model")

    def test_bfcl_context_overflow_counts_as_zero_score(self) -> None:
        errors = [
            (
                "Error during inference: BFCL prompt exceeds max context length: "
                "input_tokens=95602, max_context_length=32768."
            ),
            (
                "Error during inference: BFCL prompt exceeds max context length: "
                "input_tokens=32817, max_context_length=32768."
            ),
            (
                "Error during inference: Error code: 400 - {'message': "
                "'Input length (32764 tokens) exceeds the maximum allowed length "
                "(32762 tokens). Use a shorter input.'}"
            ),
        ]
        with TemporaryDirectory() as tmp:
            result_dir = Path(tmp) / "result"
            model_dir = result_dir / "dry-model"
            model_dir.mkdir(parents=True)
            (model_dir / "BFCL_v4_multi_turn_long_context_result.json").write_text(
                json.dumps(
                    {
                        "id": "multi_turn_long_context_129",
                        "result": errors[0],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (model_dir / "BFCL_v4_multi_turn_miss_param_result.json").write_text(
                json.dumps(
                    {
                        "id": "multi_turn_miss_param_190",
                        "result": errors[1],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            self.assertTrue(all(_is_allowed_zero_score_error(e) for e in errors))
            _raise_on_inference_errors(result_dir, "dry-model")

    def test_bfcl_print_filter_keeps_errors(self) -> None:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            with _filter_bfcl_prints(enabled=True):
                print("-" * 100)
                print("ID: base_5, Turn: 1, Step: 5")
                print("Empty response from the model. Proceed to next turn.")
                print("❗️❗️ Error occurred during inference.")

        output = buf.getvalue()
        self.assertNotIn("ID: base_5", output)
        self.assertNotIn("Empty response", output)
        self.assertIn("Error occurred", output)

    def test_bfcl_project_root_env_uses_writable_output(self) -> None:
        with TemporaryDirectory() as tmp:
            with mock.patch.dict("os.environ", {}, clear=True):
                _set_bfcl_project_root(Path(tmp) / "bfcl")

                self.assertEqual(
                    os.environ["BFCL_PROJECT_ROOT"],
                    str(Path(tmp) / "bfcl"),
                )

    def test_bfcl_vllm_server_command_receives_context_length(self) -> None:
        vllm_cmd = ["vllm", "serve", "model"]
        spec = ExternalRunSpec(
            model="model",
            output_dir=Path("outputs"),
            backend="vllm",
            max_context_length=65536,
        )

        self.assertEqual(
            _server_command_for_spec(vllm_cmd, spec)[-2:],
            ["--max-model-len", "65536"],
        )


if __name__ == "__main__":
    unittest.main()
