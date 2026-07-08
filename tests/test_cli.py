import contextlib
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
    _filter_bfcl_prints,
    _is_allowed_zero_score_error,
    _raise_on_inference_errors,
    _server_command_with_context,
)


class ExternalCliTests(unittest.TestCase):
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
        self.assertEqual(spec.num_threads, 8)
        self.assertEqual(spec.temperature, 0.2)
        self.assertEqual(spec.max_tokens, 2048)
        self.assertEqual(spec.max_context_length, 8192)
        self.assertEqual(spec.top_p, 0.9)
        self.assertEqual(spec.top_k, 50)
        self.assertTrue(spec.verbose)
        self.assertFalse(spec.allow_overwrite)
        self.assertTrue(spec.run_generation)
        self.assertFalse(spec.run_evaluation)

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

    def test_external_benchmark_flag_is_removed(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                build_parser().parse_args(["--external-benchmark", "bfcl"])

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
            self.assertEqual(result["results"]["bfcl"]["primary_metric"], "avg_acc")
            self.assertEqual(result["results"]["bfcl"]["primary_score"], 0.0)
            self.assertTrue((out / "dry-model" / "bfcl" / "summary.json").exists())
            self.assertTrue((out / "dry-model" / "run_summary.json").exists())

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
            self.assertEqual(summary["num_threads"], 16)
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

    def test_bfcl_server_command_receives_context_length(self) -> None:
        sglang_cmd = [
            "python",
            "-m",
            "sglang.launch_server",
            "--model-path",
            "model",
        ]
        vllm_cmd = ["vllm", "serve", "model"]

        self.assertEqual(
            _server_command_with_context(sglang_cmd, 131072)[-2:],
            ["--context-length", "131072"],
        )
        self.assertEqual(
            _server_command_with_context(vllm_cmd, 65536)[-2:],
            ["--max-model-len", "65536"],
        )


if __name__ == "__main__":
    unittest.main()
