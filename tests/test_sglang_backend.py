import asyncio
import hashlib
import struct
import sys
import tempfile
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
        self.assertIn("--grpc-mode", command)
        self.assertNotIn("--grpc-http-sidecar-port", command)
        self.assertEqual(command[command.index("--log-level") + 1], "error")
        self.assertNotIn("--log-level-http", command)
        self.assertNotIn("SGLANG_GRPC_PORT", env)
        self.assertEqual(env["SGLANG_GRPC_TOKEN_ID_ARRAY"], "1")
        self.assertEqual(env["TORCH_CPP_LOG_LEVEL"], "ERROR")
        self.assertEqual(env["TQDM_DISABLE"], "1")
        self.assertEqual(actor.url(), "grpc://10.0.0.1:55000")
        wait_for_port.assert_called_once_with(
            "127.0.0.1",
            55000,
            fake_process,
        )

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
        ):
            actor = sglang_service._SGLangServerActor(
                "test/model",
                1,
                {},
            )

        self.assertEqual(popen.call_count, 2)
        stop_process.assert_called_once_with(first_process)
        self.assertEqual(actor.url(), "grpc://10.0.0.1:45302")

    def test_grpc_worker_adds_new_optional_request_fields(self) -> None:
        class Request:
            def __init__(
                self,
                *,
                rid=None,  # noqa: ANN001
                input_embeds,  # noqa: ANN001
                token_type_ids,  # noqa: ANN001
            ) -> None:
                self.rid = rid
                self.input_embeds = input_embeds
                self.token_type_ids = token_type_ids

        servicer = SimpleNamespace(TokenizedGenerateReqInput=Request)

        fields = grpc_worker.patch_smg_request_type(servicer)
        request = servicer.TokenizedGenerateReqInput(rid="request-1")

        self.assertEqual(fields, ("input_embeds", "token_type_ids"))
        self.assertEqual(request.rid, "request-1")
        self.assertIsNone(request.input_embeds)
        self.assertIsNone(request.token_type_ids)

    def test_grpc_worker_maps_legacy_embedding_image_field(self) -> None:
        class Request:
            def __init__(self, *, rid=None, mm_inputs) -> None:  # noqa: ANN001
                self.rid = rid
                self.mm_inputs = mm_inputs

        servicer = SimpleNamespace(TokenizedEmbeddingReqInput=Request)

        patched = grpc_worker.patch_smg_embedding_request_type(servicer)
        request = servicer.TokenizedEmbeddingReqInput(
            rid="request-1",
            image_inputs="wrapped-images",
        )

        self.assertTrue(patched)
        self.assertEqual(request.rid, "request-1")
        self.assertEqual(request.mm_inputs, "wrapped-images")

    def test_grpc_worker_wraps_scalar_classifier_output(self) -> None:
        class Manager:
            async def _handle_embedding_output(self, batch_out):  # noqa: ANN001
                self.embeddings = batch_out.embeddings

        request_manager = SimpleNamespace(GrpcRequestManager=Manager)
        grpc_worker.patch_smg_scalar_embedding_output(request_manager)
        manager = Manager()
        batch_out = SimpleNamespace(embeddings=[-1.25, [0.5, 0.75]])

        asyncio.run(manager._handle_embedding_output(batch_out))

        self.assertEqual(manager.embeddings, [[-1.25], [0.5, 0.75]])

    def test_grpc_worker_serializes_router_embedding_schema(self) -> None:
        response = SimpleNamespace(
            embedding=[-1.25],
            prompt_tokens=7,
            embedding_dim=1,
        )

        encoded = grpc_worker._serialize_router_embed_response(response)

        complete = b"\x0a\x04" + struct.pack("<f", -1.25)
        complete += b"\x10\x07\x20\x01"
        self.assertEqual(encoded, b"\x12\x0a" + complete)

    def test_grpc_worker_patches_generated_embedding_serializer(self) -> None:
        calls = {}

        def method_factory(
            behavior,  # noqa: ANN001
            request_deserializer=None,  # noqa: ANN001
            response_serializer=None,  # noqa: ANN001
        ):
            calls["serializer"] = response_serializer
            return behavior

        grpc_module = SimpleNamespace(unary_unary_rpc_method_handler=method_factory)

        class Servicer:
            def Embed(self):  # noqa: N802
                return None

        pb2_grpc = SimpleNamespace(
            sglang__scheduler__pb2=SimpleNamespace(
                EmbedResponse=SimpleNamespace(
                    DESCRIPTOR=SimpleNamespace(fields=[SimpleNamespace(number=4)])
                )
            ),
            grpc=grpc_module,
        )

        def add_servicer(servicer, server):  # noqa: ANN001
            del server
            pb2_grpc.grpc.unary_unary_rpc_method_handler(
                servicer.Embed,
                response_serializer=lambda response: response,
            )

        pb2_grpc.add_SglangSchedulerServicer_to_server = add_servicer

        patched = grpc_worker.patch_smg_embedding_response_wire(pb2_grpc)
        pb2_grpc.add_SglangSchedulerServicer_to_server(Servicer(), object())

        self.assertTrue(patched)
        self.assertIs(
            calls["serializer"],
            grpc_worker._serialize_router_embed_response,
        )
        self.assertIs(
            grpc_module.unary_unary_rpc_method_handler,
            method_factory,
        )

    def test_grpc_worker_disables_http_sidecar_hook(self) -> None:
        calls = []

        async def serve_grpc(server_args, model_info=None, **kwargs):  # noqa: ANN001
            calls.append((server_args, model_info, kwargs))
            return "done"

        server = SimpleNamespace(serve_grpc=serve_grpc)
        grpc_worker.disable_smg_http_sidecar(server)

        result = asyncio.run(server.serve_grpc("args", "model-info"))

        self.assertEqual(result, "done")
        self.assertEqual(calls, [("args", "model-info", {})])

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
        service.base_url = "http://127.0.0.1:18080"
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
        for key in ("log_level_http", "grpc_http_sidecar_port"):
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
        self.assertEqual(params["seed"], 123)
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

        self.assertEqual(sglang_backend._extract_text(response), "generated")
        self.assertEqual(
            sglang_backend._extract_output_token_count(response),
            7,
        )
        with self.assertRaisesRegex(ValueError, "unexpected batch size"):
            sglang_backend._extract_text([])

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
