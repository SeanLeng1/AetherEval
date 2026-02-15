from __future__ import annotations

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
                "vllm:\n"
                "  max_model_len: 4096\n",
                encoding="utf-8",
            )

            cfg = load_yaml_config(str(cfg_path))
            args = argparse.Namespace(
                model=None,
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
                dtype=None,
                vllm_arg=None,
            )
            resolved = resolve_run_arguments(args, cfg)
            self.assertEqual(resolved["model"], "test/model")
            self.assertEqual(resolved["tasks"], "ifeval")
            self.assertFalse(resolved["inspect"])
            self.assertEqual(resolved["dp_size"], 2)
            self.assertEqual(resolved["tp_size"], 1)
            self.assertEqual(resolved["gen_overrides"]["max_new_tokens"], 123)
            self.assertEqual(resolved["bootstrap_resamples"], 250)
            self.assertEqual(resolved["model_kwargs"]["max_model_len"], 4096)

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
            dtype=None,
            vllm_arg=["trust_remote_code=true", "max_num_seqs=64"],
        )
        resolved = resolve_run_arguments(args, cfg)
        self.assertEqual(resolved["model"], "cli/model")
        self.assertTrue(resolved["inspect"])
        self.assertEqual(resolved["dp_size"], 4)
        self.assertEqual(resolved["tp_size"], 1)
        self.assertEqual(resolved["gen_overrides"]["max_new_tokens"], 256)
        self.assertEqual(resolved["bootstrap_resamples"], 123)
        self.assertEqual(resolved["bootstrap_confidence"], 0.9)
        self.assertEqual(resolved["model_kwargs"]["trust_remote_code"], True)
        self.assertEqual(resolved["model_kwargs"]["max_num_seqs"], 64)


if __name__ == "__main__":
    unittest.main()
