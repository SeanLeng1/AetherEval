import asyncio
import hashlib
import importlib.metadata as importlib_metadata
import inspect
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import aethereval.backends.sglang.backend as sglang_backend
import aethereval.backends.sglang.grpc_worker as grpc_worker
import aethereval.backends.sglang.service as sglang_service


class _FakeService:
    def __init__(self) -> None:
        self.calls = []

    def request_many(self, path, payloads, **kwargs):  # noqa: ANN001, ANN003
        self.calls.append((path, list(payloads), dict(kwargs)))
        return [
            {
                "text": f"service:{index}",
                "meta_info": {"completion_tokens": index + 1},
            }
            for index in range(len(payloads))
        ]


class _FakeTokenizer:
    def encode(self, text, add_special_tokens=False):  # noqa: ANN001
        del add_special_tokens
        return text.split()


class _ThinkingTokenizer(_FakeTokenizer):
    def apply_chat_template(
        self,
        messages,  # noqa: ANN001
        tokenize,  # noqa: ANN001
        add_generation_prompt,  # noqa: ANN001
        enable_thinking=None,  # noqa: ANN001
    ):
        del tokenize, add_generation_prompt
        return f"thinking={enable_thinking}:{messages[-1]['content']}"


class SGLangBackendTests(unittest.TestCase):
    def test_bundled_harmony_encoding_has_official_hash(self) -> None:
        with mock.patch.dict(
            sglang_service.os.environ,
            {},
            clear=True,
        ):
            encoding_dir = sglang_service._resolve_harmony_encoding_dir()

        vocab_path = encoding_dir / "o200k_base.tiktoken"
        self.assertTrue(vocab_path.is_file())
        self.assertEqual(
            hashlib.sha256(vocab_path.read_bytes()).hexdigest(),
            sglang_service._HARMONY_ENCODING_SHA256,
        )

    def test_invalid_explicit_harmony_encoding_fails_early(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            with mock.patch.dict(
                sglang_service.os.environ,
                {"TIKTOKEN_ENCODINGS_BASE": temporary_dir},
                clear=True,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "Missing Harmony encoding",
                ):
                    sglang_service._resolve_harmony_encoding_dir()

    def test_normal_shutdown_signals_entire_server_process_group(self) -> None:
        process = mock.Mock()
        process.pid = 12345
        process.poll.return_value = None

        with (
            mock.patch.object(sglang_service.os, "killpg") as killpg,
            mock.patch.object(
                sglang_service,
                "_wait_for_process_group_exit",
                return_value=True,
            ),
        ):
            sglang_service._stop_process(process)

        killpg.assert_called_once_with(12345, sglang_service.signal.SIGTERM)
        process.wait.assert_called_once_with(timeout=20)

    def test_shutdown_cleans_process_group_after_parent_already_died(self) -> None:
        process = mock.Mock()
        process.pid = 12345
        process.poll.return_value = -3

        with (
            mock.patch.object(sglang_service.os, "killpg") as killpg,
            mock.patch.object(
                sglang_service,
                "_wait_for_process_group_exit",
                return_value=True,
            ),
        ):
            sglang_service._stop_process(process)

        killpg.assert_called_once_with(12345, sglang_service.signal.SIGTERM)
        process.wait.assert_not_called()

    def test_server_actor_uses_grpc_worker_and_independent_ports(self) -> None:
        fake_process = mock.Mock()
        fake_ray = SimpleNamespace(
            util=SimpleNamespace(get_node_ip_address=lambda: "10.0.0.1")
        )
        with (
            mock.patch.dict(sys.modules, {"ray": fake_ray}),
            mock.patch.object(
                sglang_service.subprocess,
                "Popen",
                return_value=fake_process,
            ) as popen,
            mock.patch.object(
                sglang_service,
                "_wait_for_port",
            ) as wait_for_port,
            mock.patch.object(
                sglang_service,
                "_free_port",
                side_effect=[55000, 56000],
            ),
            mock.patch.object(sglang_service, "_check_stack_versions") as check_versions,
        ):
            actor = sglang_service._SGLangServerActor(
                "test/model",
                1,
                {},
            )

        command = popen.call_args.args[0]
        env = popen.call_args.kwargs["env"]
        self.assertEqual(
            command[:4],
            [
                sys.executable,
                "-m",
                "aethereval.backends.sglang.process_guard",
                str(sglang_service.os.getpid()),
            ],
        )
        self.assertEqual(
            command[4:8],
            [
                sys.executable,
                "-m",
                "aethereval.backends.sglang.grpc_worker",
                "serve",
            ],
        )
        self.assertEqual(command[command.index("--port") + 1], "55000")
        self.assertEqual(command[command.index("--nccl-port") + 1], "56000")
        self.assertIn("--smg-grpc-mode", command)
        self.assertNotIn("--grpc-mode", command)
        self.assertNotIn("--grpc-http-sidecar-port", command)
        self.assertEqual(command[command.index("--log-level") + 1], "error")
        self.assertNotIn("--log-level-http", command)
        self.assertNotIn("SGLANG_GRPC_PORT", env)
        self.assertNotIn("SGLANG_GRPC_TOKEN_ID_ARRAY", env)
        check_versions.assert_called_once_with()
        self.assertEqual(env["TORCH_CPP_LOG_LEVEL"], "ERROR")
        self.assertEqual(env["TQDM_DISABLE"], "1")
        self.assertNotIn("SGLANG_EXTERNAL_MODEL_PACKAGE", env)
        self.assertEqual(actor.url(), "grpc://10.0.0.1:55000")
        wait_for_port.assert_called_once_with(
            "127.0.0.1",
            55000,
            fake_process,
        )

    def test_reward_adapter_is_selected_by_architecture_not_context_length(self) -> None:
        for architecture in ("GPT2ForSequenceClassification", "GPT2LMHeadModel", "Qwen2ForSequenceClassification"):
            with (
                self.subTest(architecture=architecture),
                mock.patch.dict(sys.modules, {"ray": SimpleNamespace(util=SimpleNamespace(get_node_ip_address=lambda: "127.0.0.1"))}),
                mock.patch.object(sglang_service.subprocess, "Popen") as popen,
                mock.patch.object(sglang_service, "_wait_for_port"),
                mock.patch.object(sglang_service, "_free_port", side_effect=[50000, 51000]),
                mock.patch("transformers.AutoConfig.from_pretrained", return_value=SimpleNamespace(architectures=[architecture])),
                mock.patch.object(sglang_service, "_check_stack_versions"),
            ):
                actor = sglang_service._SGLangServerActor("local-model", 1, {"is_embedding": True, "context_length": 1025})
                env = popen.call_args.kwargs["env"]
                command = popen.call_args.args[0]
                self.assertNotIn("--smg-grpc-mode", command)
                self.assertNotIn("--grpc-mode", command)
                self.assertEqual(actor.url(), "http://127.0.0.1:50000")
                self.assertEqual("SGLANG_EXTERNAL_MODEL_PACKAGE" in env, architecture == "GPT2ForSequenceClassification")

    def test_server_actor_retries_startup_port_collision(self) -> None:
        first_process = mock.Mock()
        first_process.poll.return_value = -3
        second_process = mock.Mock()
        fake_ray = SimpleNamespace(
            util=SimpleNamespace(get_node_ip_address=lambda: "10.0.0.1")
        )
        with (
            mock.patch.dict(sys.modules, {"ray": fake_ray}),
            mock.patch.object(
                sglang_service.subprocess,
                "Popen",
                side_effect=[first_process, second_process],
            ) as popen,
            mock.patch.object(
                sglang_service,
                "_wait_for_port",
                side_effect=[RuntimeError("startup failed"), None],
            ),
            mock.patch.object(
                sglang_service,
                "_free_port",
                side_effect=[45301, 39089, 45302, 39090],
            ),
            mock.patch.object(
                sglang_service,
                "_port_is_available",
                side_effect=[True, False],
            ),
            mock.patch.object(sglang_service, "_stop_process") as stop_process,
            mock.patch.object(sglang_service, "_check_stack_versions"),
        ):
            actor = sglang_service._SGLangServerActor(
                "test/model",
                1,
                {},
            )

        self.assertEqual(popen.call_count, 2)
        stop_process.assert_called_once_with(first_process)
        self.assertEqual(actor.url(), "grpc://10.0.0.1:45302")

    def test_grpc_worker_disables_http_sidecar_hook(self) -> None:
        calls = []

        async def serve_grpc(server_args, model_info=None, on_request_manager_ready=None, **kwargs):  # noqa: ANN001
            calls.append((server_args, model_info, kwargs))
            return "done"

        server = SimpleNamespace(serve_grpc=serve_grpc)
        grpc_worker.disable_smg_http_sidecar(server)

        # SGLang starts the HTTP sidecar only if this hook is in the signature.
        self.assertEqual(
            list(inspect.signature(server.serve_grpc).parameters),
            ["server_args", "model_info"],
        )

        result = asyncio.run(server.serve_grpc("args", "model-info"))

        self.assertEqual(result, "done")
        self.assertEqual(calls, [("args", "model-info", {})])

    def test_gpt2_worker_info_rewrites_limits_and_rejects_changed_layout(self) -> None:
        from aethereval.backends.sglang.models import gpt2_context

        def worker(architecture):  # noqa: ANN001
            config = SimpleNamespace(
                hf_config=SimpleNamespace(architectures=[architecture]),
                is_generation=False,
                context_len=1024,
            )
            runner = SimpleNamespace(
                req_to_token_pool=SimpleNamespace(max_context_len=4096),
                effective_max_total_num_tokens=100000,
            )
            return SimpleNamespace(model_config=config, model_runner=runner, ps=SimpleNamespace(attn_dcp_size=1))

        layout = (1, 2, 3, 4, 1023, 1018, 7, 8, 9, 10, 11, 12)
        get_info = gpt2_context._worker_info(lambda self: layout)
        self.assertEqual(get_info(worker("GPT2ForSequenceClassification")), (1, 2, 3, 4, 1024, 1024, 7, 8, 9, 10, 11, 12))
        self.assertEqual(get_info(worker("Qwen2ForSequenceClassification")), layout)
        for changed in (layout[:11], (*layout[:5], 1023, *layout[6:]), (*layout[:4], "1023", *layout[5:])):
            with self.subTest(changed=changed), self.assertRaisesRegex(RuntimeError, "get_worker_info layout changed"):
                gpt2_context._worker_info(lambda self, info=changed: info)(worker("GPT2ForSequenceClassification"))

    def test_stack_version_floor(self) -> None:
        def versions(installed):  # noqa: ANN001
            def version(package):  # noqa: ANN001
                if package not in installed:
                    raise importlib_metadata.PackageNotFoundError(package)
                return installed[package]

            return mock.patch.object(importlib_metadata, "version", side_effect=version)

        accepted = [
            {"sglang": "0.5.20", "smg-grpc-servicer": "0.9.1"},
            {"sglang": "0.5.18", "smg-grpc-servicer": "0.8.0"},
            {"sglang": "0.5.18rc1", "smg-grpc-servicer": "0.13.0"},
            {"sglang": "0.6.0+cu130", "smg-grpc-servicer": "1.0.0.dev3"},
        ]
        for installed in accepted:
            with self.subTest(installed=installed), versions(installed):
                self.assertEqual(sglang_service._check_stack_versions(), installed)

        rejected = [
            ({"sglang": "0.5.15", "smg-grpc-servicer": "0.9.1"}, ["sglang 0.5.15 is older than 0.5.18"]),
            ({"sglang": "0.5.20", "smg-grpc-servicer": "0.7.0"}, ["smg-grpc-servicer 0.7.0 is older than 0.8.0"]),
            (
                {"sglang": "0.5"},
                ["sglang 0.5 is older than 0.5.18", "smg-grpc-servicer is not installed"],
            ),
            ({"sglang": "main", "smg-grpc-servicer": "0.9.1"}, ["sglang has an unrecognized version 'main'"]),
        ]
        for installed, problems in rejected:
            with self.subTest(installed=installed), versions(installed):
                with self.assertRaises(RuntimeError) as caught:
                    sglang_service._check_stack_versions()
                message = str(caught.exception)
                for problem in problems:
                    self.assertIn(problem, message)
                self.assertIn("sglang>=0.5.18 and smg-grpc-servicer>=0.8.0", message)

    def test_router_uses_requested_policy_and_log_level(self) -> None:
        service = sglang_service.SGLangService.__new__(sglang_service.SGLangService)
        service.router_policy = "round_robin"
        service.router_log_level = "error"
        service.model = "test/model"
        service.model_kwargs = {
            "tokenizer": "test/tokenizer",
            "chat_template": "/tmp/chat-template.jinja",
            "reasoning_parser": "qwen3",
            "tool_call_parser": "qwen",
        }
        service._harmony_encoding_dir = Path("/opt/aethereval/encodings")
        process = mock.Mock()
        with (
            mock.patch.object(
                sglang_service,
                "_free_port",
                side_effect=[18080, 18081],
            ),
            mock.patch.object(
                sglang_service.subprocess,
                "Popen",
                return_value=process,
            ) as popen,
            mock.patch.object(
                sglang_service,
                "_wait_until_ready",
            ) as wait_until_ready,
        ):
            base_url = service._start_router(["grpc://10.0.0.2:9000"])

        command = popen.call_args.args[0]
        env = popen.call_args.kwargs["env"]
        self.assertEqual(base_url, "http://127.0.0.1:18080")
        self.assertEqual(command[command.index("--policy") + 1], "round_robin")
        self.assertEqual(command[command.index("--log-level") + 1], "error")
        self.assertEqual(
            command[command.index("--prometheus-port") + 1],
            "18081",
        )
        self.assertEqual(
            command[command.index("--worker-urls") + 1],
            "grpc://10.0.0.2:9000",
        )
        self.assertEqual(
            command[command.index("--model-path") + 1],
            "test/model",
        )
        self.assertEqual(
            command[command.index("--tokenizer-path") + 1],
            "test/tokenizer",
        )
        self.assertEqual(
            command[command.index("--chat-template") + 1],
            "/tmp/chat-template.jinja",
        )
        self.assertEqual(
            command[command.index("--reasoning-parser") + 1],
            "qwen3",
        )
        self.assertEqual(
            command[command.index("--tool-call-parser") + 1],
            "qwen",
        )
        self.assertEqual(
            env["TIKTOKEN_ENCODINGS_BASE"],
            "/opt/aethereval/encodings",
        )
        wait_until_ready.assert_called_once_with(
            "http://127.0.0.1:18080",
            process,
            endpoint="/readiness",
            tokenizer_model="test/model",
        )

    def test_readiness_waits_for_model_tokenizer_registration(self) -> None:
        response = mock.MagicMock()
        response.__enter__.return_value.read.side_effect = [
            b'{"tokenizers":[{"name":"other/model"}]}',
            b'{"tokenizers":[{"name":"test/model"}]}',
        ]
        with (
            mock.patch.object(sglang_service, "_check_url") as health,
            mock.patch.object(sglang_service._URL_OPENER, "open", return_value=response),
            mock.patch.object(sglang_service.time, "sleep") as sleep,
        ):
            sglang_service._wait_until_ready(
                "http://localhost:18080", None, tokenizer_model="test/model"
            )
        self.assertEqual(health.call_count, 2)
        sleep.assert_called_once_with(1.0)

    def test_router_retries_startup_port_collision(self) -> None:
        service = sglang_service.SGLangService.__new__(sglang_service.SGLangService)
        service.router_policy = "cache_aware"
        service.router_log_level = "warn"
        service.model = "test/model"
        service.model_kwargs = {}
        service._harmony_encoding_dir = Path("/opt/aethereval/encodings")
        first_process = mock.Mock()
        first_process.poll.return_value = 1
        second_process = mock.Mock()
        with (
            mock.patch.object(
                sglang_service,
                "_free_port",
                side_effect=[18080, 18081, 18082, 18083],
            ),
            mock.patch.object(
                sglang_service.subprocess,
                "Popen",
                side_effect=[first_process, second_process],
            ) as popen,
            mock.patch.object(
                sglang_service,
                "_wait_until_ready",
                side_effect=[RuntimeError("startup failed"), None],
            ),
            mock.patch.object(
                sglang_service,
                "_port_is_available",
                return_value=False,
            ),
            mock.patch.object(sglang_service, "_stop_process") as stop_process,
        ):
            base_url = service._start_router(["grpc://10.0.0.2:9000"])

        self.assertEqual(base_url, "http://127.0.0.1:18082")
        self.assertEqual(popen.call_count, 2)
        stop_process.assert_called_once_with(first_process)

    def test_single_replica_uses_managed_service(self) -> None:
        tokenizer = _FakeTokenizer()
        with (
            mock.patch.object(sglang_backend, "SGLangService") as service_cls,
            mock.patch.object(
                sglang_backend,
                "load_chat_tokenizer",
                return_value=tokenizer,
            ),
        ):
            backend = sglang_backend.SGLangBackend(
                model="test/model",
                dp_size=1,
                tensor_parallel_size=2,
                model_kwargs={"dtype": "bfloat16"},
            )

        service_cls.assert_called_once_with(
            model="test/model",
            dp_size=1,
            tensor_parallel_size=2,
            model_kwargs={"dtype": "bfloat16"},
        )
        self.assertIs(backend._tokenizer, tokenizer)

    def test_service_supplies_model_tokenizer_without_overriding_custom_one(self):
        for options, expected in [
            ({}, {"tokenizer_path": "test/model"}),
            ({"tokenizer": "custom"}, {"tokenizer": "custom"}),
            ({"tokenizer_path": "custom"}, {"tokenizer_path": "custom"}),
        ]:
            ray = mock.Mock()
            ray.is_initialized.return_value = True
            ray.get.return_value = ["grpc://127.0.0.1:9000"]
            with (
                mock.patch.dict(sys.modules, {"ray": ray}),
                mock.patch.object(
                    sglang_service.SGLangService,
                    "_start_router",
                    return_value="http://127.0.0.1:9001",
                ),
            ):
                service = sglang_service.SGLangService(
                    model="test/model",
                    dp_size=1,
                    tensor_parallel_size=1,
                    model_kwargs=options,
                )
            self.assertEqual(service.model_kwargs, expected)
            self.assertEqual(
                ray.remote.return_value.return_value.remote.call_args.args[2], expected
            )

    def test_service_cleanup_does_not_mask_dead_actor_error(self) -> None:
        worker = SimpleNamespace(close=SimpleNamespace(remote=lambda: "close-ref"))
        ray = mock.Mock()
        ray.get.side_effect = RuntimeError("actor already died")
        service = sglang_service.SGLangService.__new__(sglang_service.SGLangService)
        service._ray = ray
        service._workers = [worker]
        service._router = None
        service._closed = False

        service.close()

        ray.kill.assert_called_once_with(worker, no_restart=True)
        self.assertEqual(service._workers, [])

    def test_service_adds_model_id_without_overriding_request(self) -> None:
        service = sglang_service.SGLangService.__new__(sglang_service.SGLangService)
        service._closed = False
        service._endpoints = ["http://127.0.0.1:18080"]
        service.model = "test/default-model"
        service.dp_size = 1

        with mock.patch.object(
            sglang_service,
            "_post_json",
            side_effect=lambda url, payload: payload,
        ):
            results = service.request_many(
                "/generate",
                [
                    {"text": "first"},
                    {"model": "test/explicit-model", "text": "second"},
                ],
                show_progress=False,
                progress_desc="test",
                progress_unit="request",
            )

        self.assertEqual(results[0]["model"], "test/default-model")
        self.assertEqual(results[1]["model"], "test/explicit-model")

    def test_service_keeps_only_a_bounded_window_of_futures(self) -> None:
        service = sglang_service.SGLangService.__new__(sglang_service.SGLangService)
        service._closed = False
        service._endpoints = ["http://unused"]
        service.model = "test/model"
        service.dp_size = 1
        real_wait = sglang_service.wait
        windows = []

        def wait_window(futures, **kwargs):
            windows.append(len(futures))
            return real_wait(futures, **kwargs)

        with (
            mock.patch.object(
                sglang_service, "_post_json", side_effect=lambda url, body: body["text"]
            ),
            mock.patch.object(sglang_service, "wait", side_effect=wait_window),
        ):
            results = service.request_many(
                "/generate",
                [{"text": str(i)} for i in range(200)],
                show_progress=False,
                progress_desc="test",
                progress_unit="gen",
            )
        self.assertEqual(results, [str(i) for i in range(200)])
        self.assertLessEqual(max(windows), 64)

    def test_service_in_flight_window_scales_with_replicas(self) -> None:
        for dp_size, inflight, expected in ((8, None, 512), (2, 3, 6)):
            service = sglang_service.SGLangService.__new__(sglang_service.SGLangService)
            service._closed = False
            service._endpoints = ["http://unused"]
            service.model = "test/model"
            service.dp_size = dp_size
            if inflight is not None:
                service.inflight_per_replica = inflight

            with (
                self.subTest(dp_size=dp_size, inflight_per_replica=inflight),
                mock.patch.object(
                    sglang_service,
                    "_post_json",
                    side_effect=lambda url, body: body["text"],
                ),
                mock.patch.object(
                    sglang_service,
                    "ThreadPoolExecutor",
                    wraps=sglang_service.ThreadPoolExecutor,
                ) as executor_cls,
            ):
                results = service.request_many(
                    "/generate",
                    [{"text": str(i)} for i in range(600)],
                    show_progress=False,
                    progress_desc="test",
                    progress_unit="gen",
                )
                self.assertEqual(results, [str(i) for i in range(600)])
                executor_cls.assert_called_once_with(max_workers=expected)

    def test_service_reads_and_validates_inflight_per_replica(self) -> None:
        self.assertEqual(
            sglang_service._server_cli_args({"inflight_per_replica": 128}), []
        )
        with self.assertRaisesRegex(ValueError, "inflight_per_replica must be >= 1"):
            sglang_service.SGLangService(
                model="test/model",
                dp_size=1,
                tensor_parallel_size=1,
                model_kwargs={"inflight_per_replica": 0},
            )
        for options, expected in [({}, 64), ({"inflight_per_replica": 128}, 128)]:
            ray = mock.Mock()
            ray.is_initialized.return_value = True
            ray.get.return_value = ["grpc://127.0.0.1:9000"]
            with (
                mock.patch.dict(sys.modules, {"ray": ray}),
                mock.patch.object(
                    sglang_service.SGLangService,
                    "_start_router",
                    return_value="http://127.0.0.1:9001",
                ),
            ):
                service = sglang_service.SGLangService(
                    model="test/model",
                    dp_size=2,
                    tensor_parallel_size=1,
                    model_kwargs=options,
                )
            self.assertEqual(service.inflight_per_replica, expected)

    def test_service_failure_does_not_wait_for_in_flight_requests(self) -> None:
        service = sglang_service.SGLangService.__new__(sglang_service.SGLangService)
        service._closed = False
        service._endpoints = ["http://unused"]
        service.model = "test/model"
        service.dp_size = 1
        release = threading.Event()

        def post(url, body):  # noqa: ANN001
            if body["text"] == "0":
                raise RuntimeError("SGLang request failed with HTTP 500")
            release.wait(timeout=30)
            return body["text"]

        started = time.monotonic()
        try:
            with (
                mock.patch.object(sglang_service, "_post_json", side_effect=post),
                self.assertRaisesRegex(RuntimeError, "HTTP 500"),
            ):
                service.request_many(
                    "/generate",
                    [{"text": str(i)} for i in range(200)],
                    show_progress=False,
                    progress_desc="test",
                    progress_unit="gen",
                )
            self.assertLess(time.monotonic() - started, 5)
        finally:
            release.set()

    def test_server_prompt_counts_avoid_local_tokenization(self) -> None:
        service = _FakeService()
        service.request_many = mock.Mock(
            return_value=[
                {
                    "text": "a",
                    "meta_info": {
                        "prompt_tokens": 17,
                        "completion_tokens": 1,
                        "finish_reason": {"type": "stop", "matched": 2},
                    },
                },
                [
                    {
                        "text": "b",
                        "meta_info": {
                            "prompt_tokens": 17,
                            "completion_tokens": 1,
                            "finish_reason": "length",
                        },
                    }
                ],
            ]
        )
        tokenizer = _ThinkingTokenizer()
        tokenizer.encode = mock.Mock(
            side_effect=AssertionError("redundant tokenization")
        )
        outputs = sglang_backend._run_service_generation(
            service,
            tokenizer,
            [
                {"idx": 0, "sample_id": "a", "prompt": "hello", "num_generations": 2},
            ],
            {"_show_progress": False},
        )
        self.assertEqual(outputs[0]["meta"]["prompt_token_count"], 17)
        self.assertEqual(outputs[0]["meta"]["finish_reasons"], ["stop", "length"])
        tokenizer.encode.assert_not_called()
        self.assertEqual(
            sglang_backend._parse_generate_response(
                [{"text": "", "meta_info": {"prompt_tokens": 8}}]
            )[1],
            8,
        )

    def test_missing_server_prompt_count_falls_back_once_per_condition(self) -> None:
        tokenizer = _ThinkingTokenizer()
        tokenizer.encode = mock.Mock(return_value=[1, 2, 3])
        outputs = sglang_backend._run_service_generation(
            _FakeService(),
            tokenizer,
            [
                {"idx": 0, "sample_id": "a", "prompt": "hello", "num_generations": 4},
            ],
            {"_show_progress": False},
        )
        self.assertEqual(outputs[0]["meta"]["prompt_token_count"], 3)
        self.assertEqual(outputs[0]["meta"]["finish_reasons"], [None] * 4)
        tokenizer.encode.assert_called_once()

    def test_server_cli_args_preserve_model_options(self) -> None:
        args = sglang_service._server_cli_args(
            {
                "context_length": 32768,
                "trust_remote_code": True,
                "json_model_override_args": {"max_position_embeddings": 32768},
                "dtype": None,
                "log_level": "info",
                "router_log_level": "info",
                "grpc_mode": False,
                "smg_grpc_mode": True,
                "nccl_port": 12345,
            }
        )

        self.assertEqual(
            args,
            [
                "--context-length",
                "32768",
                "--trust-remote-code",
                "--json-model-override-args",
                '{"max_position_embeddings":32768}',
            ],
        )

    def test_server_cli_args_reject_http_sidecar_options(self) -> None:
        for key in (
            "log_level_http",
            "grpc_http_sidecar_port",
            "smg_http_sidecar_port",
        ):
            with (
                self.subTest(key=key),
                self.assertRaisesRegex(
                    ValueError,
                    "do not expose an HTTP sidecar",
                ),
            ):
                sglang_service._server_cli_args({key: "unused"})

    def test_sampling_params_skip_disabled_top_k(self) -> None:
        params = sglang_backend._build_sampling_params(
            {
                "max_new_tokens": 32,
                "temperature": 0.2,
                "top_p": 0.9,
                "top_k": -1,
                "min_p": 0.05,
                "seed": 123,
            }
        )
        self.assertEqual(params["max_new_tokens"], 32)
        self.assertEqual(params["temperature"], 0.2)
        self.assertEqual(params["top_p"], 0.9)
        self.assertEqual(params["min_p"], 0.05)
        self.assertNotIn("seed", params)
        self.assertNotIn("top_k", params)

    def test_sampling_params_include_structured_output_constraints(self) -> None:
        params = sglang_backend._build_sampling_params(
            {
                "regex": "(yes|no)",
                "json_schema": '{"type":"object"}',
                "ebnf": 'root ::= "ok"',
                "structural_tag": "tag",
            }
        )

        self.assertEqual(params["regex"], "(yes|no)")
        self.assertEqual(params["json_schema"], '{"type":"object"}')
        self.assertEqual(params["ebnf"], 'root ::= "ok"')
        self.assertEqual(params["structural_tag"], "tag")

    def test_grpc_single_item_batch_response_is_unwrapped(self) -> None:
        response = [
            {
                "text": "generated",
                "meta_info": {"completion_tokens": 7},
            }
        ]

        self.assertEqual(
            sglang_backend._parse_generate_response(response),
            ("generated", None, 7, None),
        )
        with self.assertRaisesRegex(ValueError, "unexpected batch size"):
            sglang_backend._parse_generate_response([])

    def test_run_generation_supports_thinking_and_no_thinking(self) -> None:
        payloads = [
            {
                "idx": 0,
                "sample_id": "a",
                "prompt": [{"role": "user", "content": "hello"}],
                "num_generations": 1,
            }
        ]
        for enabled in (True, False):
            with self.subTest(enabled=enabled):
                service = _FakeService()
                sglang_backend._run_service_generation(
                    service,
                    _ThinkingTokenizer(),
                    payloads,
                    {
                        "n": 1,
                        "max_new_tokens": 8,
                        "temperature": 0.0,
                        "top_p": 1.0,
                        "enable_thinking": enabled,
                    },
                )
                _, requests, _ = service.calls[0]
                self.assertEqual(requests[0]["text"], f"thinking={enabled}:hello")
                self.assertNotIn("enable_thinking", requests[0]["sampling_params"])

    def test_service_generation_sends_independent_router_requests(self) -> None:
        service = _FakeService()
        payloads = [
            {
                "idx": 0,
                "sample_id": "a",
                "prompt": [{"role": "user", "content": "hello"}],
                "num_generations": 2,
            },
            {
                "idx": 1,
                "sample_id": "b",
                "prompt": [{"role": "user", "content": "world"}],
                "num_generations": 1,
            },
        ]

        outputs = sglang_backend._run_service_generation(
            service,
            _FakeTokenizer(),
            payloads,
            {
                "max_new_tokens": 8,
                "temperature": 0.0,
                "top_p": 1.0,
                "_show_progress": True,
            },
        )

        path, requests, options = service.calls[0]
        self.assertEqual(path, "/generate")
        self.assertEqual(len(requests), 3)
        self.assertEqual(requests[0]["text"], "user: hello")
        self.assertEqual(requests[1]["text"], "user: hello")
        self.assertEqual(requests[2]["text"], "user: world")
        self.assertEqual(requests[0]["sampling_params"]["max_new_tokens"], 8)
        self.assertTrue(options["show_progress"])
        self.assertEqual(outputs[0]["generations"], ["service:0", "service:1"])
        self.assertEqual(outputs[1]["generations"], ["service:2"])


if __name__ == "__main__":
    unittest.main()
