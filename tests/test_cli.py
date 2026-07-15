import contextlib
import csv
import io
import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from aethereval.cli import (
    _build_external_spec,
    _split_native_external_tasks,
    build_parser,
    run_selected_tasks,
)
from aethereval.config import resolve_run_arguments
from benchmarks.bfcl._compat import _set_bfcl_project_root
from benchmarks.bfcl.external import (
    ExternalRunSpec,
    _RouterReadyRequestsProxy,
    _ThreadingDrainProxy,
    _filter_bfcl_prints,
    _gen_args,
    _is_allowed_zero_score_error,
    _raise_on_inference_errors,
    _server_command_for_spec,
    parse_scores,
    write_predictions_jsonl,
)
from benchmarks.bfcl.register import register_rlla_model


class ExternalCliTests(unittest.TestCase):
    def test_bfcl_smg_readiness_waits_for_all_workers(self) -> None:
        class Response:
            def __init__(self, status_code, payload=None):  # noqa: ANN001
                self.status_code = status_code
                self._payload = payload or {}

            def json(self):
                return self._payload

        class Requests:
            class exceptions:
                ConnectionError = ConnectionError

            def get(self, url, *args, **kwargs):  # noqa: ANN001
                del args, kwargs
                if url.endswith("/workers"):
                    return Response(
                        200,
                        {
                            "workers": [
                                {"is_healthy": True},
                                {"is_healthy": True},
                            ]
                        },
                    )
                raise AssertionError("models endpoint must wait for every worker")

        proxy = _RouterReadyRequestsProxy(Requests(), expected_workers=8)

        with mock.patch("benchmarks.bfcl.external.time.sleep"):
            response = proxy.get("http://127.0.0.1:1053/v1/models")

        self.assertEqual(response.status_code, 503)

    def test_bfcl_smg_readiness_accepts_all_healthy_workers(self) -> None:
        response = mock.Mock(status_code=200)
        workers_response = mock.Mock(status_code=200)
        workers_response.json.return_value = {
            "workers": [{"is_healthy": True} for _ in range(8)]
        }
        requests = mock.Mock()
        requests.get.side_effect = [workers_response, response]
        proxy = _RouterReadyRequestsProxy(requests, expected_workers=8)

        actual = proxy.get("http://127.0.0.1:1053/v1/models")

        self.assertIs(actual, response)
        self.assertEqual(actual.status_code, 200)

    def test_bfcl_smg_log_event_drains_until_process_eof(self) -> None:
        threading_module = mock.Mock()
        proxy = _ThreadingDrainProxy(threading_module)

        event = proxy.Event()
        event.set()

        self.assertFalse(event.is_set())

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
                "--skip-evaluation",
            ]
        )

        spec, _run = _build_external_spec(args, task_name="bfcl")

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
        self.assertTrue(spec.verbose)
        self.assertFalse(spec.allow_overwrite)
        self.assertTrue(spec.run_generation)
        self.assertFalse(spec.run_evaluation)

    def test_bfcl_external_spec_supports_unified_phase_flags(self) -> None:
        generate_args = build_parser().parse_args(
            ["--tasks", "bfcl", "--model", "model", "--generate-only"]
        )
        generate_spec, _run = _build_external_spec(generate_args, task_name="bfcl")
        self.assertTrue(generate_spec.run_generation)
        self.assertFalse(generate_spec.run_evaluation)

        eval_args = build_parser().parse_args(
            ["--tasks", "bfcl", "--model", "model", "--eval-only"]
        )
        eval_spec, _run = _build_external_spec(eval_args, task_name="bfcl")
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
                "--output-dir",
                "outputs/bfcl",
                "--dp-size",
                "8",
                "--tp-size",
                "1",
            ]
        )

        spec, _run = _build_external_spec(args, task_name="bfcl")

        self.assertEqual(spec.num_gpus, 8)
        self.assertEqual(spec.dp_size, 8)
        self.assertEqual(spec.tp_size, 1)
        self.assertTrue(spec.use_sglang_router)
        self.assertEqual(spec.router_policy, "cache_aware")
        self.assertEqual(spec.sglang_server_args["log_level"], "warning")
        self.assertEqual(spec.sglang_server_args["router_log_level"], "warn")

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
        args.backend_kwargs = resolved["backend_kwargs"]

        spec, _run = _build_external_spec(args, task_name="bfcl")

        self.assertEqual(resolved["backend_kwargs"]["context_length"], 32768)
        self.assertNotIn(
            "json_model_override_args", resolved["backend_kwargs"]
        )
        self.assertEqual(spec.max_context_length, 131072)
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

    def test_bfcl_legacy_num_gpus_means_sglang_dp(self) -> None:
        args = build_parser().parse_args(
            [
                "--tasks",
                "bfcl",
                "--model",
                "rlla-gdpo",
                "--num-gpus",
                "8",
            ]
        )

        spec, _run = _build_external_spec(args, task_name="bfcl")

        self.assertEqual(spec.num_gpus, 8)
        self.assertEqual(spec.dp_size, 8)
        self.assertEqual(spec.tp_size, 1)

    def test_external_benchmark_flag_is_removed(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                build_parser().parse_args(["--external-benchmark", "bfcl"])

    def test_model_path_flag_is_removed(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                build_parser().parse_args(["--model-path", "/tmp/model"])

    def test_bfcl_can_run_from_tasks_skip_only(self) -> None:
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
                    "--skip-generation",
                    "--skip-evaluation",
                ]
            )
            resolved = resolve_run_arguments(args, {})

            result = run_selected_tasks(args, resolved)

            self.assertEqual(result["selected_tasks"], ["bfcl"])
            self.assertEqual(result["tasks"], ["bfcl"])
            self.assertEqual(
                result["results"]["bfcl"]["primary_metric"], "OverallAcc"
            )
            self.assertEqual(result["results"]["bfcl"]["primary_score"], 0.0)
            self.assertTrue((out / "dry-model" / "bfcl" / "summary.json").exists())
            self.assertTrue(
                (out / "dry-model" / "bfcl" / "predictions.jsonl").exists()
            )
            self.assertTrue((out / "dry-model" / "run_summary.json").exists())

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
                    "--skip-generation",
                    "--skip-evaluation",
                ]
            )

            result = run_selected_tasks(args, resolve_run_arguments(args, {}))

            self.assertEqual(result["model"], model)
            self.assertEqual(result["model_name"], model_name)
            self.assertTrue(
                (
                    out
                    / model_name
                    / "production-1"
                    / "bfcl"
                    / "summary.json"
                ).exists()
            )

            spec, _run = _build_external_spec(args, task_name="bfcl")
            generation_args = _gen_args(spec, out / "raw")
            self.assertEqual(spec.model_name, model_name)
            self.assertEqual(generation_args.model, [model])
            self.assertEqual(generation_args.local_model_path, model)

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
            (model_dir / "BFCL_simple_result.json").write_text(
                json.dumps(
                    {
                        "id": "simple_1",
                        "result": "<think>x</think>",
                        "inference_input_log": {"formatted_prompt": "prompt text"},
                    }
                )
                + "\n"
                + json.dumps(
                    {
                        "id": "simple_2",
                        "result": "Error during inference: BFCL prompt exceeds max context length.",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (score_model_dir / "BFCL_simple_score.json").write_text(
                json.dumps({"accuracy": 0.5}) + "\n"
                + json.dumps({"id": "simple_2", "valid": False}) + "\n",
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
            self.assertEqual(rows[0]["sample_id"], "simple_1")
            self.assertEqual(rows[0]["gen_idx"], 0)
            self.assertEqual(rows[0]["prompt"], "prompt text")
            self.assertEqual(rows[0]["generation"], "<think>x</think>")
            self.assertEqual(rows[0]["score"], 1.0)
            self.assertTrue(rows[0]["is_pass"])
            self.assertEqual(rows[0]["parsed"], "<think>x</think>")
            self.assertIsNone(rows[0]["gold"])
            self.assertIsNone(rows[0]["error"])
            self.assertEqual(rows[0]["meta"]["benchmark"], "bfcl")
            self.assertEqual(rows[0]["meta"]["test_category"], "simple")
            self.assertFalse(rows[1]["is_pass"])
            self.assertEqual(rows[1]["score"], 0.0)
            self.assertIn("Error during inference", rows[1]["error"])

    def test_bfcl_parse_scores_reports_toolrl_columns(self) -> None:
        with TemporaryDirectory() as tmp:
            score_dir = Path(tmp) / "score"
            score_dir.mkdir()
            columns = [
                "Model",
                "Overall Acc",
                "Non-Live AST Acc",
                "Non-Live Exec Acc",
                "Live Acc",
                "Multi Turn Acc",
                "Relevance Detection",
                "Irrelevance Detection",
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
                        "Non-Live Exec Acc": "63.77%",
                        "Live Acc": "57.31%",
                        "Multi Turn Acc": "0.25%",
                        "Relevance Detection": "77.78%",
                        "Irrelevance Detection": "41.84%",
                    }
                )

            metrics = parse_scores(score_dir, "dry-model")

            self.assertEqual(metrics["OverallAcc"], 38.35)
            self.assertEqual(metrics["Non-LiveASTAcc"], 56.33)
            self.assertEqual(metrics["Non-LiveExecAcc"], 63.77)
            self.assertEqual(metrics["LiveAcc"], 57.31)
            self.assertEqual(metrics["MultiTurnAcc"], 0.25)
            self.assertEqual(metrics["RelevanceDetection"], 77.78)
            self.assertEqual(metrics["IrrelevanceDetection"], 41.84)
            self.assertEqual(metrics["avg_acc"], 38.35)
            self.assertEqual(metrics["non_live_ast_acc"], 56.33)

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
                    "--skip-generation",
                    "--skip-evaluation",
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

            result = run_selected_tasks(args, resolved)
            summary = result["results"]["bfcl"]

            self.assertEqual(summary["backend"], "sglang")
            self.assertEqual(summary["num_gpus"], 3)
            self.assertEqual(summary["dp_size"], 3)
            self.assertEqual(summary["tp_size"], 1)
            self.assertTrue(summary["use_sglang_router"])
            self.assertEqual(summary["router_policy"], "cache_aware")
            self.assertEqual(summary["num_threads"], 48)
            self.assertEqual(summary["temperature"], 0.25)
            self.assertEqual(summary["max_tokens"], 1234)
            self.assertEqual(summary["max_context_length"], 9999)
            self.assertEqual(summary["top_p"], 0.77)
            self.assertEqual(summary["top_k"], 11)
            self.assertFalse(summary["verbose"])

    def test_bfcl_inference_errors_fail_fast(self) -> None:
        with TemporaryDirectory() as tmp:
            result_dir = Path(tmp) / "result"
            model_dir = result_dir / "dry-model"
            model_dir.mkdir(parents=True)
            (model_dir / "BFCL_simple_result.json").write_text(
                json.dumps(
                    {
                        "id": "simple_1",
                        "result": "Error during inference: Connection error.",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "simple_1"):
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
            (model_dir / "BFCL_multi_turn_long_context_result.json").write_text(
                json.dumps(
                    {
                        "id": "multi_turn_long_context_129",
                        "result": errors[0],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (model_dir / "BFCL_multi_turn_miss_param_result.json").write_text(
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

    def test_bfcl_server_command_uses_smg_dp_and_inherits_args(self) -> None:
        sglang_cmd = [
            "python",
            "-m",
            "sglang.launch_server",
            "--model-path",
            "model",
            "--tp",
            "8",
            "--mem-fraction-static",
            "0.9",
        ]
        spec = ExternalRunSpec(
            model="model",
            output_dir=Path("outputs"),
            num_gpus=8,
            dp_size=8,
            tp_size=1,
            gpu_memory_utilization=0.86,
            dtype="bfloat16",
            max_context_length=32768,
            sglang_server_args={
                "chunked_prefill_size": 4096,
                "enable_tokenizer_batch_encode": True,
                "generation_batch_size": 64,
            },
        )

        patched = _server_command_for_spec(sglang_cmd, spec)

        self.assertIn("sglang_router.launch_server", patched)
        self.assertNotIn("sglang.launch_server", patched)
        self.assertEqual(patched[patched.index("--dp-size") + 1], "8")
        self.assertEqual(patched[patched.index("--tp-size") + 1], "1")
        self.assertEqual(patched[patched.index("--router-policy") + 1], "cache_aware")
        self.assertEqual(patched[patched.index("--context-length") + 1], "32768")
        self.assertEqual(patched[patched.index("--mem-fraction-static") + 1], "0.86")
        self.assertEqual(patched[patched.index("--chunked-prefill-size") + 1], "4096")
        self.assertIn("--enable-tokenizer-batch-encode", patched)
        self.assertNotIn("--generation-batch-size", patched)

    def test_bfcl_server_command_keeps_single_replica_sglang(self) -> None:
        command = ["python", "-m", "sglang.launch_server", "--model-path", "model"]
        spec = ExternalRunSpec(
            model="model",
            output_dir=Path("outputs"),
            num_gpus=2,
            dp_size=1,
            tp_size=2,
        )

        patched = _server_command_for_spec(command, spec)

        self.assertIn("sglang.launch_server", patched)
        self.assertNotIn("sglang_router.launch_server", patched)
        self.assertEqual(patched[patched.index("--tp-size") + 1], "2")
        self.assertNotIn("--dp-size", patched)

    def test_bfcl_server_command_can_use_native_dp_for_comparison(self) -> None:
        command = ["python", "-m", "sglang.launch_server", "--model-path", "model"]
        spec = ExternalRunSpec(
            model="model",
            output_dir=Path("outputs"),
            num_gpus=8,
            dp_size=8,
            tp_size=1,
            use_sglang_router=False,
        )

        patched = _server_command_for_spec(command, spec)

        self.assertIn("sglang.launch_server", patched)
        self.assertNotIn("sglang_router.launch_server", patched)
        self.assertEqual(patched[patched.index("--dp-size") + 1], "8")

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
