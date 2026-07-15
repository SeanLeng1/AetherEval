import unittest

import aethereval.backends.sglang.backend as sglang_backend


class _FakeEngine:
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.prompt_batches: list[list[str]] = []
        self.sampling_params: dict | None = None

    def generate(self, prompts, sampling_params):  # noqa: ANN001
        self.prompts = list(prompts)
        self.prompt_batches.append(list(prompts))
        self.sampling_params = dict(sampling_params)
        return [{"text": f"out:{idx}:{prompt}"} for idx, prompt in enumerate(prompts)]


class _ShortEngine(_FakeEngine):
    def generate(self, prompts, sampling_params):  # noqa: ANN001
        del prompts, sampling_params
        return []


class _FakeTokenizer:
    def encode(self, text, add_special_tokens=False):  # noqa: ANN001
        del add_special_tokens
        return text.split()


class SGLangBackendTests(unittest.TestCase):
    def test_engine_args_per_engine_dp1_with_default_memory_saver(self) -> None:
        # Each engine (single or ray worker) runs dp=1; aethereval-level dp shards payloads.
        backend = sglang_backend.SGLangBackend.__new__(sglang_backend.SGLangBackend)
        backend.model = "test/model"
        backend.tensor_parallel_size = 2
        backend.model_kwargs = {"dtype": "bfloat16", "generation_batch_size": 64}

        args = backend._engine_args()

        self.assertEqual(
            args,
            {
                "model_path": "test/model",
                "tp_size": 2,
                "dtype": "bfloat16",
                "enable_memory_saver": True,
            },
        )
        self.assertNotIn("dp_size", args)
        self.assertNotIn("generation_batch_size", args)

    def test_engine_args_memory_saver_overridable(self) -> None:
        backend = sglang_backend.SGLangBackend.__new__(sglang_backend.SGLangBackend)
        backend.model = "test/model"
        backend.tensor_parallel_size = 1
        backend.model_kwargs = {"enable_memory_saver": False}

        args = backend._engine_args()

        self.assertIs(args["enable_memory_saver"], False)

    def test_split_payloads_round_robins_across_workers(self) -> None:
        payloads = [{"idx": idx} for idx in range(5)]

        split = sglang_backend._split_payloads(payloads, num_workers=2)

        self.assertEqual(
            [[item["idx"] for item in worker] for worker in split], [[0, 2, 4], [1, 3]]
        )

    def test_split_contiguous_balances_chunks(self) -> None:
        chunks = sglang_backend._split_contiguous(list(range(5)), 2)
        self.assertEqual(chunks, [[0, 1, 2], [3, 4]])

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

    def test_run_generation_expands_n_and_regroups_outputs(self) -> None:
        engine = _FakeEngine()
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
                "num_generations": 2,
            },
        ]

        outputs = sglang_backend._run_generation(
            engine=engine,
            tokenizer=_FakeTokenizer(),
            payloads=payloads,
            gen_cfg={"n": 2, "max_new_tokens": 8, "temperature": 0.7, "top_p": 1.0},
            batch_size=2,
            show_progress=False,
        )

        self.assertEqual(
            engine.prompt_batches,
            [["user: hello", "user: hello"], ["user: world", "user: world"]],
        )
        self.assertEqual(
            engine.sampling_params,
            {"max_new_tokens": 8, "temperature": 0.7, "top_p": 1.0},
        )
        self.assertEqual(outputs[0]["sample_id"], "a")
        self.assertEqual(len(outputs[0]["generations"]), 2)
        self.assertTrue(outputs[0]["generations"][0].startswith("out:0:"))
        self.assertEqual(outputs[1]["sample_id"], "b")
        self.assertEqual(len(outputs[1]["generations"]), 2)

    def test_run_generation_raises_on_output_count_mismatch(self) -> None:
        payloads = [
            {
                "idx": 0,
                "sample_id": "a",
                "prompt": [{"role": "user", "content": "hello"}],
                "num_generations": 1,
            },
        ]

        with self.assertRaises(RuntimeError):
            sglang_backend._run_generation(
                engine=_ShortEngine(),
                tokenizer=object(),
                payloads=payloads,
                gen_cfg={"n": 1, "max_new_tokens": 8, "temperature": 0.0, "top_p": 1.0},
                batch_size=1,
                show_progress=False,
            )


if __name__ == "__main__":
    unittest.main()
