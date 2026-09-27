"""Canary for the installed SGLang/SMG API surface the worker path relies on.

Runs on CPU in the SGLang runtime image (no GPU, no server) and is skipped where
SGLang is not installed. Run it after every SGLang or smg-grpc-servicer upgrade:
a failure names the patch or flag to review before any GPU launch.
"""

import argparse
import importlib
import inspect
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import aethereval.backends.sglang.grpc_worker as grpc_worker
import aethereval.backends.sglang.service as sglang_service
from aethereval.backends.sglang.models import gpt2_context
from tests._deps import requires


def _worker_argv(model_kwargs):  # noqa: ANN001
    """The `sglang serve` arguments _SGLangServerActor launches for model_kwargs."""
    fake_ray = SimpleNamespace(util=SimpleNamespace(get_node_ip_address=lambda: "127.0.0.1"))
    config = SimpleNamespace(architectures=["Qwen2ForSequenceClassification"])
    with (
        mock.patch.dict(sys.modules, {"ray": fake_ray}),
        mock.patch.object(sglang_service.subprocess, "Popen") as popen,
        mock.patch.object(sglang_service, "_wait_for_port"),
        mock.patch.object(sglang_service, "_free_port", side_effect=[45000, 46000]),
        mock.patch("transformers.AutoConfig.from_pretrained", return_value=config),
    ):
        sglang_service._SGLangServerActor("test/model", 2, model_kwargs)
    command = popen.call_args.args[0]
    return command[command.index("serve") + 1 :]


@requires("sglang", "smg_grpc_servicer")
class InstalledSGLangStackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # The modules grpc_worker.main() imports, plus the SGLang entrypoint that
        # calls the patched serve_grpc.
        cls.server = importlib.import_module("smg_grpc_servicer.sglang.server")
        importlib.import_module("sglang.cli.main")
        cls.grpc_server = importlib.import_module("sglang.srt.entrypoints.grpc_server")
        cls.server_args = importlib.import_module("sglang.srt.server_args").ServerArgs

    def test_installed_versions_meet_the_floor(self) -> None:
        installed = sglang_service._check_stack_versions()
        self.assertEqual(set(installed), set(sglang_service._MIN_STACK_VERSIONS))

    def test_http_sidecar_patch_still_applies(self) -> None:
        source = inspect.getsource(self.grpc_server.serve_grpc)
        self.assertIn("from smg_grpc_servicer.sglang.server import serve_grpc", source)
        self.assertIn("on_request_manager_ready", source)
        self.assertIn("inspect.signature(_serve_grpc)", source)
        parameters = list(inspect.signature(self.server.serve_grpc).parameters)
        self.assertEqual(parameters[:2], ["server_args", "model_info"])
        self.assertIn(
            "on_request_manager_ready",
            parameters,
            "SMG no longer offers the sidecar hook; review disable_smg_http_sidecar",
        )
        with mock.patch.object(self.server, "serve_grpc", self.server.serve_grpc):
            grpc_worker.disable_smg_http_sidecar(self.server)
            self.assertNotIn(
                "on_request_manager_ready",
                inspect.signature(self.server.serve_grpc).parameters,
            )

    def test_worker_commands_parse_with_installed_sglang(self) -> None:
        parser = argparse.ArgumentParser()
        self.server_args.add_cli_args(parser)
        generation = parser.parse_args(
            _worker_argv(
                {
                    "tokenizer_path": "test/model",
                    "context_length": 65536,
                    "mem_fraction_static": 0.85,
                    "weight_loader_prefetch_checkpoints": True,
                    "inflight_per_replica": 64,
                }
            )
        )
        self.assertTrue(generation.smg_grpc_mode)
        self.assertFalse(generation.grpc_mode)
        self.assertEqual((generation.port, generation.nccl_port, generation.tp_size), (45000, 46000, 2))
        self.assertEqual(generation.tokenizer_path, "test/model")
        # The reward-model defaults from benchmark_utils/reward_model.py.
        embedding = parser.parse_args(
            _worker_argv(
                {
                    "cuda_graph_backend_decode": "disabled",
                    "cuda_graph_backend_prefill": "disabled",
                    "is_embedding": True,
                    "trust_remote_code": True,
                    "dtype": "float16",
                    "context_length": 32768,
                }
            )
        )
        self.assertFalse(embedding.smg_grpc_mode or embedding.grpc_mode)
        self.assertTrue(embedding.is_embedding)

    def test_server_arg_guards_name_current_fields(self) -> None:
        fields = set(inspect.signature(self.server_args).parameters)
        self.assertLessEqual({"smg_grpc_mode", "smg_http_sidecar_port", "log_level_http"}, fields)
        self.assertLessEqual({"smg_grpc_mode", "grpc_mode"}, sglang_service._CONTROLLED_SERVER_ARGS)
        self.assertLessEqual({"smg_http_sidecar_port", "log_level_http"}, sglang_service._UNSUPPORTED_SERVER_ARGS)

    def test_gpt2_classification_patch_targets(self) -> None:
        tokenizer_manager = importlib.import_module("sglang.srt.managers.tokenizer_manager")
        tp_worker = importlib.import_module("sglang.srt.managers.tp_worker")
        scheduler = importlib.import_module("sglang.srt.managers.scheduler")
        model_config = importlib.import_module("sglang.srt.configs.model_config")
        self.assertTrue(hasattr(model_config, "ModelConfig"))
        # Building each replacement runs the patch's own anchor checks; nothing is installed.
        gpt2_context._tokenizer_validation(tokenizer_manager.TokenizerManager._validate_one_request)
        gpt2_context._embedding_handler(scheduler.Scheduler.handle_embedding_request)
        self.assertIn(
            "validate_input_length",
            scheduler.Scheduler.handle_embedding_request.__code__.co_names,
        )
        source = inspect.getsource(tp_worker.TpModelWorker.get_worker_info)
        for name in ("max_req_len - 5", "effective_max_total_num_tokens", "attn_dcp_size", "max_context_len"):
            self.assertIn(name, source)
        # Names models/gpt2_classification.py imports (importing it would install the patch).
        for module, names in {
            "sglang.srt.layers.pooler": ("Pooler", "PoolingType", "score_and_pool"),
            "sglang.srt.model_loader.weight_utils": ("default_weight_loader",),
            "sglang.srt.models.gpt2": ("GPT2LMHeadModel", "GPT2Model"),
        }.items():
            loaded = importlib.import_module(module)
            for name in names:
                self.assertTrue(hasattr(loaded, name), f"{module}.{name}")


if __name__ == "__main__":
    unittest.main()
