import contextlib
import csv
import io
import json
import os
import types
import unittest
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
from aethereval.core.task_defaults import (
    resolve_phase_num_repeats,
    resolve_task_default_gen,
    resolve_task_num_repeats,
)
from benchmarks.bfcl._compat import (
    _LaxModule,
    _patch_missing_attr,
    _set_bfcl_project_root,
)
from benchmarks.bfcl.cli import build_bfcl_spec
from benchmarks.bfcl.external import (
    ExternalRunSpec,
    _size_bfcl_thread_pool,
    _clear_results,
    _filter_bfcl_prints,
    _evaluation_run_paths,
    _gen_args,
    _is_allowed_zero_score_error,
    _prepare_existing_results,
    _raise_on_inference_errors,
    _run_generation,
    _run_generations,
    _repeat_seed,
    _requested_categories,
    _server_command_for_spec,
    _sort_result_files,
    add_comparison_metrics,
    average_repeat_metrics,
    compute_format_rates,
    parse_scores,
    run as run_bfcl_external,
    write_predictions_jsonl,
)
from benchmarks.bfcl.register import prepare_bfcl_model
from tests._deps import BFCL_MODULES, requires


class ExternalCliTests(unittest.TestCase):
    @requires(*BFCL_MODULES)
    def test_bfcl_v3_thread_pool_size_is_applied_and_restored(self) -> None:
        from bfcl_eval.model_handler.local_inference import base_oss_handler

        original = base_oss_handler.ThreadPoolExecutor
        for size in (17, 512):
            with _size_bfcl_thread_pool(size):
                with base_oss_handler.ThreadPoolExecutor(max_workers=100) as executor:
                    self.assertEqual(executor._max_workers, size)
        self.assertIs(base_oss_handler.ThreadPoolExecutor, original)

    def test_bfcl_resume_repairs_only_an_interrupted_final_jsonl_record(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            result_dir = Path(tmp) / "result"
            model_dir = result_dir / "dry-model" / "non_live"
            model_dir.mkdir(parents=True)
            result_file = model_dir / "BFCL_v3_simple_result.json"
            valid_record = json.dumps({"id": "simple_1", "result": "ok"})
            broken_tail = b'{"id": "simple_2", "result": "unterminated'
            result_file.write_bytes(valid_record.encode() + b"\n" + broken_tail)

            _prepare_existing_results(
                [result_dir],
                "dry-model",
                repair_tail=True,
            )

            self.assertEqual(
                result_file.read_text(encoding="utf-8"),
                valid_record + "\n",
            )
            backups = list(model_dir.glob("*.corrupt-tail"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_bytes(), broken_tail)

    def test_bfcl_resume_rejects_corruption_before_a_later_record(self) -> None:
        with TemporaryDirectory() as tmp:
            result_dir = Path(tmp) / "result"
            model_dir = result_dir / "dry-model" / "non_live"
            model_dir.mkdir(parents=True)
            result_file = model_dir / "BFCL_v3_simple_result.json"
            result_file.write_text(
                '{"id": "broken"\n'
                + json.dumps({"id": "simple_2", "result": "ok"})
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                RuntimeError,
                "Malformed BFCL result JSONL",
            ):
                _prepare_existing_results(
                    [result_dir],
                    "dry-model",
                    repair_tail=True,
                )

            self.assertEqual(list(model_dir.glob("*.corrupt-tail*")), [])

    def test_bfcl_resume_drops_failed_records_but_keeps_context_overflows(
        self,
    ) -> None:
        overflow = (
            "Error during inference: BFCL prompt exceeds max context length: "
            "input_tokens=9000, max_context_length=8192."
        )
        records = [
            {"id": "live_relevance_3-3-0", "result": "ok"},
            {"id": "live_relevance_3-3-0", "result": "Error during inference: 503"},
            {
                "id": "live_relevance_4-4-0",
                "result": [["ok"], ["Error during inference: timeout"]],
            },
            {"id": "live_relevance_5-5-0", "result": overflow},
            {"id": "live_relevance_6-6-0", "result": "ok"},
        ]
        with TemporaryDirectory() as tmp:
            result_dir = Path(tmp) / "result"
            model_dir = result_dir / "dry-model"
            model_dir.mkdir(parents=True)
            result_file = model_dir / "BFCL_v3_live_relevance_result.json"
            original = "".join(json.dumps(record) + "\n" for record in records)
            result_file.write_text(original, encoding="utf-8")

            # Eval-only never drops records; the errors still fail the run.
            _prepare_existing_results([result_dir], "dry-model", repair_tail=False)
            self.assertEqual(result_file.read_text(encoding="utf-8"), original)
            with self.assertRaisesRegex(RuntimeError, "count=2"):
                _raise_on_inference_errors(result_dir, "dry-model")

            # Generation drops every copy of a failed id so upstream regenerates
            # all of them; context-length failures stay as zero-score records.
            _prepare_existing_results([result_dir], "dry-model", repair_tail=True)
            self.assertEqual(
                result_file.read_text(encoding="utf-8"),
                "".join(json.dumps(record) + "\n" for record in records[3:]),
            )
            self.assertEqual(list(model_dir.glob("*.tmp")), [])

    @requires("bfcl_eval")
    def test_bfcl_resume_restores_upstream_result_order(self) -> None:
        with TemporaryDirectory() as tmp:
            result_dir = Path(tmp) / "result"
            model_dir = result_dir / "dry-model"
            model_dir.mkdir(parents=True)
            result_file = model_dir / "BFCL_v3_simple_result.json"
            lines = [
                json.dumps({"id": f"simple_{index}", "result": "ok"}) + "\n"
                for index in (0, 2, 10, 1)
            ]
            result_file.write_text("".join(lines), encoding="utf-8")

            _sort_result_files([result_dir], "dry-model", {"simple"})

            self.assertEqual(
                result_file.read_text(encoding="utf-8"),
                "".join(lines[i] for i in (0, 3, 1, 2)),
            )

    def test_bfcl_overwrite_never_uses_upstream_update_mode(self) -> None:
        spec = ExternalRunSpec(model="m", output_dir=Path("o"), allow_overwrite=True)

        self.assertFalse(_gen_args(spec, Path("o/result")).allow_overwrite)

    def test_bfcl_overwrite_clears_only_selected_category_results(self) -> None:
        with TemporaryDirectory() as tmp:
            result_dir = Path(tmp) / "result"
            model_dir = result_dir / "org_model"
            model_dir.mkdir(parents=True)
            for category in ("simple", "live_simple", "multi_turn_base"):
                (model_dir / f"BFCL_v3_{category}_result.json").write_text("{}\n")

            _clear_results(
                [result_dir], "org/model", {"live_simple", "multi_turn_base"}
            )

            self.assertEqual(
                sorted(path.name for path in model_dir.iterdir()),
                ["BFCL_v3_simple_result.json"],
            )

    @requires("bfcl_eval")
    def test_bfcl_reports_only_requested_categories(self) -> None:
        categories = _requested_categories(["non_live"])
        self.assertIn("simple", categories)
        self.assertNotIn("multi_turn_base", categories)
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "bfcl"
            result_dir = out / "result"
            model_dir = result_dir / "dry-model"
            model_dir.mkdir(parents=True)
            (model_dir / "BFCL_v3_simple_result.json").write_text(
                json.dumps({"id": "simple_0", "result": "<think>x</think>"}) + "\n"
            )
            # A stale category from an earlier run in the same output dir.
            (model_dir / "BFCL_v3_multi_turn_base_result.json").write_text(
                json.dumps(
                    {"id": "multi_turn_base_0", "result": "Error during inference: x"}
                )
                + "\n"
            )

            _raise_on_inference_errors(result_dir, "dry-model", categories)
            rates = compute_format_rates(result_dir, "dry-model", categories)
            stats = write_predictions_jsonl(
                out=out,
                result_dir=result_dir,
                score_dir=out / "score",
                model="dry-model",
                handler="toolrl",
                categories=categories,
            )

        self.assertEqual(set(rates), {"non_live"})
        self.assertEqual(stats["prediction_records"], 1)

    def test_local_judge_automatically_splits_generation_and_evaluation(self) -> None:
        with TemporaryDirectory() as tmp:
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
                    "--output-dir",
                    tmp,
                ]
            )
            resolved = resolve_run_arguments(args, {})
            result = {
                "results": {"healthbench": {"metrics": {"score": 0.5}}},
                "backend": "vllm",
            }

            with mock.patch(
                "aethereval.core.runner._run_phase",
                side_effect=[result, result],
            ) as run_phase:
                actual = run_selected_tasks(args, resolved)

        self.assertEqual(actual["results"], result["results"])
        self.assertEqual(run_phase.call_count, 2)
        generate_call = run_phase.call_args_list[0].kwargs
        evaluate_call = run_phase.call_args_list[1].kwargs
        self.assertTrue(generate_call["generate_only"])
        self.assertFalse(generate_call["eval_only"])
        self.assertTrue(generate_call["overwrite"])
        self.assertFalse(evaluate_call["generate_only"])
        self.assertTrue(evaluate_call["eval_only"])
        self.assertFalse(evaluate_call["overwrite"])
        self.assertFalse(evaluate_call["rescore_existing"])
        self.assertIsNone(evaluate_call["backend"])

    def test_api_judge_automatically_splits_generation_and_evaluation(self) -> None:
        with TemporaryDirectory() as tmp:
            args = build_parser().parse_args(
                [
                    "--model",
                    "candidate/model",
                    "--tasks",
                    "healthbench,llmeval_med",
                    "--overwrite",
                    "--output-dir",
                    tmp,
                ]
            )
            resolved = resolve_run_arguments(args, {})
            result = {
                "results": {
                    "healthbench": {"metrics": {"score": 0.5}},
                    "llmeval_med": {"metrics": {"OP": 30.0}},
                },
                "backend": "vllm",
            }

            with mock.patch(
                "aethereval.core.runner._run_phase",
                side_effect=[result, result],
            ) as run_phase:
                actual = run_selected_tasks(args, resolved)

        self.assertEqual(actual["results"], result["results"])
        self.assertEqual(run_phase.call_count, 2)
        generate_call = run_phase.call_args_list[0].kwargs
        evaluate_call = run_phase.call_args_list[1].kwargs
        self.assertEqual(generate_call["tasks"], "healthbench,llmeval-med")
        self.assertTrue(generate_call["generate_only"])
        self.assertFalse(generate_call["eval_only"])
        self.assertTrue(generate_call["overwrite"])
        self.assertEqual(evaluate_call["tasks"], "healthbench,llmeval-med")
        self.assertFalse(evaluate_call["generate_only"])
        self.assertTrue(evaluate_call["eval_only"])
        self.assertFalse(evaluate_call["overwrite"])
        self.assertFalse(evaluate_call["rescore_existing"])
        self.assertIsNone(evaluate_call["backend"])

    def test_run_summary_lists_every_task_whatever_the_invocation_order(
        self,
    ) -> None:
        ifeval = {"metrics": {"acc": 1.0}, "primary_score": 1.0}
        bfcl = {"metrics": {"overall_acc": 50.0}, "primary_score": 50.0}

        def run_native(**kwargs):  # noqa: ANN003
            run_root = run_output_dir(kwargs["output_dir"], "m", None, None)
            task_dir = run_root / "ifeval"
            task_dir.mkdir(parents=True, exist_ok=True)
            (task_dir / "summary.json").write_text(json.dumps(ifeval))
            return {"results": {"ifeval": ifeval}, "backend": "vllm"}

        def run_bfcl(args, resolved, output_dir):  # noqa: ANN001
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "summary.json").write_text(json.dumps(bfcl))
            return {**bfcl, "task": "bfcl"}

        for order in (("bfcl", "ifeval"), ("ifeval", "bfcl")):
            with (
                self.subTest(order=order),
                TemporaryDirectory() as tmp,
                mock.patch("aethereval.cli.run_evaluation", side_effect=run_native),
                mock.patch("benchmarks.bfcl.cli.run_external", side_effect=run_bfcl),
            ):
                for tasks in order:
                    args = build_parser().parse_args(
                        ["--model", "m", "--tasks", tasks, "--output-dir", tmp]
                    )
                    summary = run_selected_tasks(args, resolve_run_arguments(args, {}))

                self.assertEqual(summary["tasks"], ["bfcl", "ifeval"])
                self.assertEqual(summary["phase"], "generate_and_eval")
                self.assertEqual(summary["primary_score_aggregate"], 25.5)

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
                "--bfcl-handler",
                "official",
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
                "--no-enable-thinking",
                "--num-threads",
                "8",
                "--num-repeats",
                "2",
                "--bfcl-verbose",
                "--no-overwrite",
                "--generate-only",
            ]
        )

        resolved = resolve_run_arguments(args, {})
        spec = build_bfcl_spec(args, resolved, Path(args.output_dir))

        self.assertEqual(spec.handler, "official")
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
        self.assertIs(spec.enable_thinking, False)
        self.assertEqual(spec.num_repeats, 2)
        self.assertTrue(spec.verbose)
        self.assertFalse(spec.allow_overwrite)
        self.assertTrue(spec.run_generation)
        self.assertFalse(spec.run_evaluation)

    def test_bfcl_external_spec_reads_generation_defaults_from_config(self) -> None:
        args = build_parser().parse_args(
            ["--tasks", "bfcl", "--model", "rlla-gdpo", "--backend", "sglang"]
        )
        resolved = resolve_run_arguments(args, {})
        configured = resolve_task_default_gen("bfcl")

        spec = build_bfcl_spec(args, resolved, Path("outputs"))

        self.assertEqual(spec.max_tokens, configured["max_new_tokens"])
        self.assertEqual(spec.handler, configured["handler"])
        self.assertEqual(spec.temperature, configured["temperature"])
        self.assertEqual(spec.top_p, configured["top_p"])
        self.assertEqual(spec.top_k, configured["top_k"])
        self.assertEqual(spec.categories, configured["categories"])
        self.assertEqual(spec.num_repeats, resolve_task_num_repeats("bfcl"))

    def test_bfcl_python_spec_defaults_match_task_config(self) -> None:
        configured = resolve_task_default_gen("bfcl")
        spec = ExternalRunSpec(model="model", output_dir=Path("output"))

        self.assertEqual(spec.max_tokens, configured["max_new_tokens"])
        self.assertEqual(spec.temperature, configured["temperature"])
        self.assertEqual(spec.top_p, configured["top_p"])
        self.assertEqual(spec.top_k, configured["top_k"])
        self.assertEqual(spec.num_repeats, resolve_task_num_repeats("bfcl"))
        self.assertEqual(spec.categories, ["live", "non_live", "multi_turn"])
        self.assertEqual(spec.handler, "toolrl")

    def test_eval_only_reuses_the_saved_num_repeats(self) -> None:
        with TemporaryDirectory() as tmp:
            saved = Path(tmp) / "summary.json"
            saved.write_text(json.dumps({"num_repeats": 3}), encoding="utf-8")

            def resolve(path: Path, override: int | None, eval_only: bool) -> int:
                return resolve_phase_num_repeats(
                    "bfcl", path, runtime_override=override, eval_only=eval_only
                )

            self.assertEqual(resolve(saved, None, True), 3)
            self.assertEqual(resolve(saved, 3, True), 3)
            self.assertEqual(resolve(saved, 2, False), 2)
            self.assertEqual(
                resolve(Path(tmp) / "missing.json", None, True),
                resolve_task_num_repeats("bfcl"),
            )
            with self.assertRaisesRegex(ValueError, "saved run config value 3"):
                resolve(saved, 2, True)

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

    @requires(*BFCL_MODULES)
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
            observed["generate_url"] = os.environ.get("AETHEREVAL_BFCL_GENERATE_URL")

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
                    "AETHEREVAL_BFCL_GENERATE_URL": "http://old/generate",
                },
            ),
        ):
            _run_generation(spec, Path("outputs/result"), generation_main)
            self.assertEqual(os.environ["LOCAL_SERVER_ENDPOINT"], "old-host")
            self.assertEqual(os.environ["LOCAL_SERVER_PORT"], "1234")
            self.assertEqual(
                os.environ["AETHEREVAL_BFCL_GENERATE_URL"],
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

    @requires(*BFCL_MODULES)
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
            observed.append((args.result_dir, os.environ.get("AETHEREVAL_BFCL_SEED")))

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
        self.assertEqual(_repeat_seed(ExternalRunSpec("m", Path("o")), 3), 3)

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
        averaged = average_repeat_metrics(runs)

        self.assertEqual(averaged["live_acc"], 73.0)
        self.assertEqual(averaged["non_live_acc"], 83.0)
        self.assertEqual(averaged["multi_turn_acc"], 13.0)
        self.assertEqual(averaged["overall_acc"], 56.33)
        self.assertEqual(averaged["overall_format"], 80.0)

    @requires(*BFCL_MODULES)
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
            mock.patch("benchmarks.bfcl.external.prepare_bfcl_model"),
            mock.patch("bfcl_eval.eval_checker.eval_runner.main") as evaluation_main,
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
                    num_repeats=4,
                    run_generation=False,
                    run_evaluation=True,
                )
            )
            summary = json.loads((out / "summary.json").read_text())

        self.assertEqual(evaluation_main.call_count, 4)
        self.assertEqual(result.metrics["overall_acc"], 56.33)
        self.assertEqual(result.metrics["overall_format"], 80.0)
        self.assertEqual(result.primary_metric, "overall_acc")
        self.assertEqual(summary["num_repeats"], 4)
        self.assertEqual(summary["repeat_seeds"], [0, 1, 2, 3])
        self.assertEqual(len(summary["repeats"]), 4)
        self.assertEqual(summary["metrics"]["live_acc"], 73.0)
        self.assertEqual(summary["handler"], "toolrl")

    @requires(*BFCL_MODULES)
    def test_bfcl_official_handler_omits_toolrl_format_metrics(self) -> None:
        with (
            TemporaryDirectory() as tmp,
            mock.patch("benchmarks.bfcl.external.prepare_bfcl_model"),
            mock.patch("bfcl_eval.eval_checker.eval_runner.main"),
            mock.patch(
                "benchmarks.bfcl.external.parse_scores",
                return_value={"overall_acc": 42.0},
            ),
            mock.patch(
                "benchmarks.bfcl.external.compute_format_rates"
            ) as compute_format,
        ):
            result = run_bfcl_external(
                ExternalRunSpec(
                    model="google/gemma-3-12b-it",
                    output_dir=Path(tmp),
                    handler="official",
                    num_repeats=1,
                    run_generation=False,
                    run_evaluation=True,
                )
            )

        self.assertEqual(result.metrics, {"overall_acc": 42.0})
        compute_format.assert_not_called()

    def test_bfcl_reuses_global_backend_settings(self) -> None:
        args = build_parser().parse_args(
            [
                "--tasks",
                "ifeval,bfcl",
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
            ]
        )
        resolved = resolve_run_arguments(args, {})
        spec = build_bfcl_spec(args, resolved, Path("outputs"))

        self.assertEqual(resolved["backend_kwargs"]["context_length"], 32768)
        self.assertEqual(spec.max_context_length, 32768)
        self.assertEqual(spec.sglang_server_args, resolved["backend_kwargs"])
        self.assertEqual(spec.sglang_server_args["chunked_prefill_size"], 4096)
        self.assertEqual(spec.sglang_server_args["schedule_conservativeness"], 1.0)

    def test_bfcl_legacy_flags_are_removed(self) -> None:
        parser = build_parser()
        for flag in ("--num-gpus", "--skip-generation", "--skip-evaluation",
                     "--bfcl-context-length", "--bfcl-sglang-arg"):
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

    @requires(*BFCL_MODULES)
    def test_bfcl_registry_handles_slashes_and_underscores(self) -> None:
        model = "/scratch/checkpoints/my_model_v2"
        with TemporaryDirectory() as tmp:
            prepare_bfcl_model(model, project_root=tmp)

        from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING

        # BFCL escapes '/' to '_' for its result folder, then its evaluator turns
        # every '_' back into '/'. Both keys must resolve to the same handler config.
        evaluator_key = model.replace("/", "_").replace("_", "/")
        self.assertIs(MODEL_CONFIG_MAPPING[model], MODEL_CONFIG_MAPPING[evaluator_key])
        self.assertEqual(MODEL_CONFIG_MAPPING[model].model_name, model)

    @requires(*BFCL_MODULES)
    def test_bfcl_official_profile_wraps_registered_prompt_handler(self) -> None:
        from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING

        model = "google/gemma-3-12b-it"
        original = MODEL_CONFIG_MAPPING[model]
        try:
            with TemporaryDirectory() as tmp:
                prepare_bfcl_model(
                    model,
                    handler_profile="official",
                    project_root=tmp,
                )
            adapted = MODEL_CONFIG_MAPPING[model]
            self.assertTrue(adapted.model_handler._aethereval_official_adapter)
            self.assertTrue(issubclass(adapted.model_handler, original.model_handler))
        finally:
            MODEL_CONFIG_MAPPING[model] = original

    @requires(*BFCL_MODULES)
    def test_bfcl_official_profile_rejects_unregistered_checkpoint(self) -> None:
        with TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "exact prompt-mode model ID"):
                prepare_bfcl_model(
                    "/scratch/custom-gemma",
                    handler_profile="official",
                    project_root=tmp,
                )

    def test_bfcl_predictions_jsonl_uses_aethereval_schema(self) -> None:
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "bfcl"
            result_dir = out / "result"
            score_dir = out / "score"
            model_dir = result_dir / "dry-model"
            score_model_dir = score_dir / "dry-model"
            model_dir.mkdir(parents=True)
            score_model_dir.mkdir(parents=True)
            (model_dir / "BFCL_v3_simple_result.json").write_text(
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
            (score_model_dir / "BFCL_v3_simple_score.json").write_text(
                json.dumps({"accuracy": 0.5})
                + "\n"
                + json.dumps({"id": "simple_2", "valid": False})
                + "\n",
                encoding="utf-8",
            )

            stats = write_predictions_jsonl(
                out=out,
                result_dir=result_dir,
                score_dir=score_dir,
                model="dry-model",
                handler="toolrl",
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
            self.assertEqual(rows[0]["meta"]["handler"], "toolrl")
            self.assertEqual(rows[0]["meta"]["test_category"], "simple")
            self.assertEqual(rows[0]["meta"]["evaluation_repeat"], 1)
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
                    "id": "simple_1",
                    "result": (
                        "<think>x</think>\n<tool_call>\n{}\n</tool_call><|im_end|>"
                    ),
                },
                {
                    "id": "simple_2",
                    "result": "<think>x</think>\n<response>done</response>",
                },
                {
                    "id": "simple_3",
                    "result": (
                        "<think>x</think>\n<tool_call>\n{}\n</tool_call>"
                        "<tool_call>\n{}\n</tool_call>"
                    ),
                },
            ]
            (model_dir / "BFCL_v3_simple_result.json").write_text(
                "\n".join(json.dumps(record) for record in tool_call_records) + "\n",
                encoding="utf-8",
            )
            response_records = [
                {
                    "id": "irrelevance_1",
                    "result": ("<think>x</think>\n<response>done</response><|im_end|>"),
                },
                {
                    "id": "irrelevance_2",
                    "result": "<think>x</think>\n<tool_call>\n{}\n</tool_call>",
                },
            ]
            (model_dir / "BFCL_v3_irrelevance_result.json").write_text(
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
            (model_dir / "BFCL_v3_multi_turn_miss_param_result.json").write_text(
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
                    }
                )
            # BFCL also writes one CSV per section; their "Overall" columns are
            # what the published comparison tables report.
            for filename, column, value in (
                ("data_non_live.csv", "Non_Live Overall Acc", "45.24%"),
                ("data_live.csv", "Live Overall Acc", "69.23%"),
                ("data_multi_turn.csv", "Multi Turn Overall Acc", "3.14%"),
            ):
                with (score_dir / filename).open(
                    "w", encoding="utf-8", newline=""
                ) as f:
                    writer = csv.DictWriter(f, fieldnames=["Model", column])
                    writer.writeheader()
                    writer.writerow({"Model": "dry-model", column: value})

            metrics = parse_scores(score_dir)

            self.assertEqual(metrics["overall_acc"], 38.35)
            self.assertEqual(metrics["non_live_acc"], 56.33)
            self.assertEqual(metrics["live_acc"], 57.31)
            self.assertEqual(metrics["multi_turn_acc"], 0.25)

            add_comparison_metrics(
                metrics,
                {"live": 77.44, "non_live": 95.11, "multi_turn": 57.40},
            )
            self.assertEqual(metrics["live_format"], 77.44)
            self.assertEqual(metrics["non_live_format"], 95.11)
            self.assertEqual(metrics["multi_turn_format"], 57.4)
            self.assertEqual(metrics["overall_format"], 76.65)

            # Section "Overall" scores and their unweighted mean (paper Avg Acc).
            self.assertEqual(metrics["non_live_overall_acc"], 45.24)
            self.assertEqual(metrics["live_overall_acc"], 69.23)
            self.assertEqual(metrics["multi_turn_overall_acc"], 3.14)
            self.assertEqual(metrics["avg_acc"], 39.20)

            self.assertEqual(
                set(metrics),
                {
                    "overall_acc",
                    "non_live_acc",
                    "live_acc",
                    "multi_turn_acc",
                    "non_live_overall_acc",
                    "live_overall_acc",
                    "multi_turn_overall_acc",
                    "avg_acc",
                    "live_format",
                    "non_live_format",
                    "multi_turn_format",
                    "overall_format",
                },
            )

    def test_bfcl_parse_scores_requires_official_csv(self) -> None:
        with TemporaryDirectory() as tmp:
            score_dir = Path(tmp) / "score"
            model_dir = score_dir / "dry-model"
            model_dir.mkdir(parents=True)
            (model_dir / "BFCL_v3_simple_score.json").write_text(
                json.dumps({"accuracy": 1.0, "correct_count": 1, "total_count": 1})
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "official report"):
                parse_scores(score_dir)

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

        self.assertEqual(metrics["overall_acc"], 56.66)
        self.assertEqual(metrics["overall_format"], 76.65)

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
            self.assertEqual(spec.num_threads, 192)
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
            (model_dir / "BFCL_v3_simple_result.json").write_text(
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
            (
                "Error during inference: worker failed with HTTP 500: "
                "The input (32889 tokens) is longer than the model's context "
                "length (32768 tokens)."
            ),
        ]
        with TemporaryDirectory() as tmp:
            result_dir = Path(tmp) / "result"
            model_dir = result_dir / "dry-model"
            model_dir.mkdir(parents=True)
            (model_dir / "BFCL_v3_multi_turn_long_context_result.json").write_text(
                json.dumps(
                    {
                        "id": "multi_turn_long_context_129",
                        "result": errors[0],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (model_dir / "BFCL_v3_multi_turn_miss_param_result.json").write_text(
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

    def test_bfcl_compat_stubs_nested_provider_attributes(self) -> None:
        # bfcl_eval's cohere handler evaluates ``list[cohere.types.ToolV2]`` at
        # import time; an absent provider SDK must resolve that without patching
        # the stdlib ``types`` module.
        provider = _LaxModule("aethereval_absent_provider")
        annotation = list[provider.types.ToolV2]

        self.assertIs(annotation.__args__[0], provider.types.ToolV2)
        self.assertFalse(hasattr(types, "ToolV2"))

    def test_bfcl_compat_refuses_to_stub_stdlib_modules(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "standard-library"):
            _patch_missing_attr("types", "ToolV2", AttributeError("ToolV2"))

        self.assertFalse(hasattr(types, "ToolV2"))

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
