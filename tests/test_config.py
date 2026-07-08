import argparse
import tempfile
import unittest
from pathlib import Path

from aethereval.config import load_yaml_config, resolve_run_arguments


class ConfigTests(unittest.TestCase):
    def test_load_yaml_and_resolve(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "run.yaml"
            cfg_path.write_text(
                "run:\n"
                "  model: test/model\n"
                "  tasks: [ifeval]\n"
                "runtime:\n"
                "  dp_size: 2\n"
                "  tp_size: 1\n"
                "generation:\n"
                "  max_new_tokens: 123\n"
                "metrics:\n"
                "  bootstrap_resamples: 250\n"
                "  rm_model_path: /models/rm\n"
                "  cm_model_path: /models/cm\n"
                "  rm_batch_size: 2\n"
                "vllm:\n"
                "  max_model_len: 4096\n",
                encoding="utf-8",
            )

            cfg = load_yaml_config(str(cfg_path))
            args = argparse.Namespace(
                model=None,
                backend=None,
                tasks=None,
                inspect=None,
                output_dir=None,
                run_id=None,
                overwrite=None,
                dp_size=None,
                tp_size=None,
                n=None,
                max_new_tokens=None,
                temperature=None,
                top_p=None,
                top_k=None,
                min_p=None,
                seed=None,
                bootstrap_resamples=None,
                bootstrap_seed=None,
                bootstrap_confidence=None,
                gpu_memory_utilization=None,
                max_model_len=None,
                mem_fraction_static=None,
                context_length=None,
                sglang_generation_batch_size=None,
                dtype=None,
                vllm_arg=None,
                sglang_arg=None,
            )
            resolved = resolve_run_arguments(args, cfg)
            self.assertEqual(resolved["model"], "test/model")
            self.assertEqual(resolved["backend"], "vllm")
            self.assertEqual(resolved["tasks"], "ifeval")
            self.assertFalse(resolved["inspect"])
            self.assertEqual(resolved["dp_size"], 2)
            self.assertEqual(resolved["tp_size"], 1)
            self.assertEqual(resolved["gen_overrides"]["max_new_tokens"], 123)
            self.assertEqual(resolved["bootstrap_resamples"], 250)
            self.assertEqual(resolved["metric_options"]["rm_model_path"], "/models/rm")
            self.assertEqual(resolved["metric_options"]["cm_model_path"], "/models/cm")
            self.assertEqual(resolved["metric_options"]["rm_batch_size"], 2)
            self.assertEqual(resolved["model_kwargs"]["max_model_len"], 4096)
            self.assertEqual(resolved["backend_kwargs"]["max_model_len"], 4096)

    def test_cli_overrides_yaml(self) -> None:
        cfg = {
            "run": {"model": "cfg/model", "tasks": ["ifeval"]},
            "runtime": {"dp_size": 2, "tp_size": 1},
            "generation": {"max_new_tokens": 128},
            "metrics": {"bootstrap_seed": 11},
            "vllm": {"extra_model_kwargs": {"trust_remote_code": False}},
        }
        args = argparse.Namespace(
            model="cli/model",
            backend=None,
            tasks="ifeval",
            inspect=True,
            output_dir=None,
            run_id=None,
            overwrite=None,
            dp_size=4,
            tp_size=None,
            n=None,
            max_new_tokens=256,
            temperature=None,
            top_p=None,
            top_k=None,
            min_p=None,
            seed=None,
            bootstrap_resamples=123,
            bootstrap_seed=None,
            bootstrap_confidence=0.9,
            gpu_memory_utilization=None,
            max_model_len=None,
            mem_fraction_static=None,
            context_length=None,
            sglang_generation_batch_size=None,
            dtype=None,
            rm_model_path="/cli/rm",
            cm_model_path=None,
            rm_batch_size=None,
            rm_max_length=None,
            rm_device=None,
            rm_dtype=None,
            rm_trust_remote_code=None,
            vllm_arg=["trust_remote_code=true", "max_num_seqs=64"],
            sglang_arg=None,
        )
        resolved = resolve_run_arguments(args, cfg)
        self.assertEqual(resolved["model"], "cli/model")
        self.assertTrue(resolved["inspect"])
        self.assertEqual(resolved["dp_size"], 4)
        self.assertEqual(resolved["tp_size"], 1)
        self.assertEqual(resolved["gen_overrides"]["max_new_tokens"], 256)
        self.assertEqual(resolved["bootstrap_resamples"], 123)
        self.assertEqual(resolved["bootstrap_confidence"], 0.9)
        self.assertEqual(resolved["metric_options"]["rm_model_path"], "/cli/rm")
        self.assertEqual(resolved["model_kwargs"]["trust_remote_code"], True)
        self.assertEqual(resolved["model_kwargs"]["max_num_seqs"], 64)

    def test_sglang_backend_config(self) -> None:
        cfg = {
            "run": {"model": "cfg/model", "tasks": ["ifeval"]},
            "runtime": {"backend": "sglang", "dp_size": 1, "tp_size": 1},
            "sglang": {
                "mem_fraction_static": 0.75,
                "context_length": 8192,
                "generation_batch_size": 64,
                "dtype": "bfloat16",
                "extra_model_kwargs": {"trust_remote_code": True},
            },
        }
        args = argparse.Namespace(
            model=None,
            backend=None,
            tasks=None,
            inspect=None,
            output_dir=None,
            run_id=None,
            overwrite=None,
            dp_size=None,
            tp_size=None,
            n=None,
            max_new_tokens=None,
            temperature=None,
            top_p=None,
            top_k=None,
            min_p=None,
            seed=None,
            bootstrap_resamples=None,
            bootstrap_seed=None,
            bootstrap_confidence=None,
            gpu_memory_utilization=None,
            max_model_len=None,
            mem_fraction_static=None,
            context_length=None,
            sglang_generation_batch_size=None,
            dtype=None,
            vllm_arg=None,
            sglang_arg=["chunked_prefill_size=4096"],
        )
        resolved = resolve_run_arguments(args, cfg)
        self.assertEqual(resolved["backend"], "sglang")
        self.assertEqual(resolved["backend_kwargs"]["mem_fraction_static"], 0.75)
        self.assertEqual(resolved["backend_kwargs"]["context_length"], 8192)
        self.assertEqual(resolved["backend_kwargs"]["generation_batch_size"], 64)
        self.assertEqual(resolved["backend_kwargs"]["dtype"], "bfloat16")
        self.assertEqual(resolved["backend_kwargs"]["trust_remote_code"], True)
        self.assertEqual(resolved["backend_kwargs"]["chunked_prefill_size"], 4096)


if __name__ == "__main__":
    unittest.main()
